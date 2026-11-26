"""Time one v0 training step under each candidate speed-up and report tokens per second.

Run it on the machine you train on:
  uv run benchmark_speed.py

Each variant builds a fresh model from the same seed and trains on the same random batch, so the
final losses should agree closely. Nothing here changes decoder_only_transformer.py.
"""

import argparse
import gc
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import median

import torch
from torch.nn import functional as F

from config import load_config
from decoder_only_transformer import VOCAB, Decoder, select_device

WARMUP = 3  # Untimed steps. The first ones include allocation and, for torch.compile, compilation.
BASELINE = "baseline (current code)"
UNSCALED = "autocast float16, no GradScaler"  # Timing only: float16 gradients can underflow.
VARIANTS = {
    BASELINE: {},
    "fused Adam": {"fused": True},
    "autocast bfloat16": {"autocast": torch.bfloat16},
    "autocast float16 + GradScaler": {"autocast": torch.float16, "scaler": True},
    UNSCALED: {"autocast": torch.float16},
    "torch.compile": {"compile": True},
    "float32 matmul precision 'high'": {"tf32": True},  # Only changes NVIDIA GPUs.
}


def measure(config, device, batch, steps, fused=None, autocast=None, scaler=False, compile=False,
            tf32=False):
    """Return (median seconds per step, seconds for the first step, final loss)."""
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.manual_seed(0)
    model = Decoder(config.model).to(device)
    optim = config.optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(optim.beta1, optim.beta2),
                                 eps=optim.epsilon, weight_decay=optim.weight_decay, fused=fused)
    forward = torch.compile(model) if compile else model
    grad_scaler = torch.amp.GradScaler(device.type) if scaler else None
    x, targets = batch
    times = []
    for _ in range(WARMUP + steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=autocast) if autocast else nullcontext():
            logits = forward(x)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                   label_smoothing=config.training.label_smoothing)
        if grad_scaler:
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return median(times[WARMUP:]), times[0], loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("m3-pro.toml"),
                        help="TOML file that sets the model size and batch size (default m3-pro.toml).")
    parser.add_argument("--batch-size", type=int,
                        help="Context windows per step (default: the config's training.batch_size).")
    parser.add_argument("--steps", type=int, default=10,
                        help=f"Timed steps per variant, after {WARMUP} untimed warm-up steps (default 10).")
    args = parser.parse_args()
    config = load_config(args.config, {"training": {"batch_size": args.batch_size}})
    device = select_device(config.runtime.device)
    arch = config.model
    tokens = config.training.batch_size * arch.context_length
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (config.training.batch_size, arch.context_length))
    batch = (x.to(device), x.roll(-1, dims=1).to(device))

    print(f"{device} device, PyTorch {torch.__version__}, {args.config.name}: width {arch.width}, "
          f"{arch.layers} layers, context {arch.context_length}, {tokens:,} tokens per step")
    print(f"{'variant':33s} {'ms/step':>9s} {'tokens/s':>10s} {'speedup':>8s} {'loss':>7s}")
    results = {}

    def run(name, options, note=""):
        try:
            step, first, loss = measure(config, device, batch, args.steps, **options)
        except Exception as exc:  # Report what this device or PyTorch build cannot do, then continue.
            reason = (str(exc).splitlines() or [""])[0][:100]
            print(f"{name:33s} not supported: {type(exc).__name__}: {reason}", flush=True)
            return
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
        results[name] = step
        if options.get("compile"):
            note = f"first step took {first:.1f} s  {note}"
        print(f"{name:33s} {step * 1e3:9.1f} {tokens / step:10,.0f} {results[BASELINE] / step:7.2f}x "
              f"{loss:7.3f}  {note}", flush=True)

    for name, options in VARIANTS.items():
        if "tf32" in options and device.type != "cuda":
            continue
        if "autocast" in options and device.type == "cpu":
            continue  # Most CPUs emulate 16-bit floats, which takes minutes per step.
        run(name, options)
        if BASELINE not in results:
            raise SystemExit("The baseline failed, so there is nothing to compare against.")

    # Combine every variant that beat the baseline by 2% or more, using the faster autocast option.
    faster = {name for name, step in results.items()
              if results[BASELINE] / step >= 1.02 and name != UNSCALED}
    autocasts = {name for name in faster if "autocast" in name}
    if autocasts:
        faster -= autocasts - {min(autocasts, key=results.get)}
    if len(faster) > 1:
        combined = {key: value for name in faster for key, value in VARIANTS[name].items()}
        run("all faster variants combined", combined, note="= " + " + ".join(sorted(faster)))


if __name__ == "__main__":
    main()
