"""Train the reference decoder on streamed FineWeb-Edu and sample from its weights.

Educational copy of decoder_only_transformer.py: the layers and the loss are written by hand
instead of using torch.nn, so the math is visible. It computes the same model, more slowly.
Checkpoints work with either file.
"""

import argparse
import codecs
import math
import os
import sys
import tempfile
import threading
import time
from functools import partial
from itertools import islice
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__:
    from .config import Architecture, Config, load_config
else:
    from config import Architecture, Config, load_config

END_DOCUMENT = 256  # IDs 0 through 255 represent UTF-8 bytes.
VOCAB = END_DOCUMENT + 1
DATASET = "HuggingFaceFW/fineweb-edu"


# Hand-written layers, as in deep_dives/1_attention_is_all_you_need. Each keeps the parameter
# names and shapes of the torch.nn layer it replaced, so existing checkpoints still load.
class Linear(nn.Module):
    def __init__(self, fan_in, fan_out, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fan_out, fan_in))   # (fan_out, fan_in), set by Decoder
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None

    def forward(self, x):                                          # x: (..., fan_in)
        y = x @ self.weight.T                                      # (..., fan_out)
        return y if self.bias is None else y + self.bias


class Embedding(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, width))      # row t is token t's vector

    def forward(self, ids):                                        # ids: (B, n) token ids
        return self.weight[ids]                                    # (B, n, width)


class LayerNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))              # gamma
        self.bias = nn.Parameter(torch.zeros(width))               # beta

    def forward(self, x):                                          # x: (B, n, width)
        mean = x.mean(-1, keepdim=True)                            # (B, n, 1)  mu for each position
        var = ((x - mean) ** 2).mean(-1, keepdim=True)             # (B, n, 1)  sigma^2 for each position
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class ReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))         # x * Phi(x), the exact (not tanh) form


def dropout(x, p, training):
    if not training or p == 0:                                     # p == 0 skips the random draw
        return x
    keep = torch.rand_like(x) > p                                  # same shape as x: True = keep
    return x * keep / (1 - p)


class DecoderBlock(nn.Module):
    """Masked self-attention followed by a position-wise feed-forward layer."""

    def __init__(self, config: Architecture):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.pre_norm = config.pre_norm
        self.q = Linear(width, width, bias=config.attention_bias)
        self.k = Linear(width, width, bias=config.attention_bias)
        self.v = Linear(width, width, bias=config.attention_bias)
        self.out = Linear(width, width, bias=config.attention_bias)
        self.ff = nn.Sequential(
            Linear(width, config.ff_width, bias=config.ff_bias),
            ReLU() if config.activation == "relu" else GELU(),
            Linear(config.ff_width, width, bias=config.ff_bias),
        )
        self.norm1 = LayerNorm(width, eps=config.layer_norm_eps)
        self.norm2 = LayerNorm(width, eps=config.layer_norm_eps)
        self.residual_dropout = config.residual_dropout
        self.attention_dropout = config.attention_dropout

    def attention(self, x, mask):
        batch, length, width = x.shape
        head_width = width // self.heads

        def split_heads(tensor):
            return tensor.reshape(batch, length, self.heads, head_width).transpose(1, 2)

        q, k, v = (split_heads(projection(x)) for projection in (self.q, self.k, self.v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_width)
        weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        weights = dropout(weights, self.attention_dropout, self.training)
        heads = (weights @ v).transpose(1, 2).reshape(batch, length, width)
        return self.out(heads)

    def forward(self, x, mask):
        p, training = self.residual_dropout, self.training
        if self.pre_norm:
            x = x + dropout(self.attention(self.norm1(x), mask), p, training)
            return x + dropout(self.ff(self.norm2(x)), p, training)
        x = self.norm1(x + dropout(self.attention(x, mask), p, training))
        return self.norm2(x + dropout(self.ff(x), p, training))


class Decoder(nn.Module):
    def __init__(self, config: Architecture = Architecture()):
        super().__init__()
        width = config.width
        length = config.context_length
        self.embedding = Embedding(VOCAB, width)
        self.embedding_scale = math.sqrt(width) if config.scale_embeddings else 1.0
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.embedding_dropout = config.embedding_dropout
        self.output = None if config.tie_embeddings else Linear(width, VOCAB, bias=False)
        positions = torch.arange(length)[:, None]
        angles = positions / (config.position_base ** (torch.arange(0, width, 2) / width))
        pe = torch.empty(length, width)
        pe[:, 0::2] = angles.sin()
        pe[:, 1::2] = angles[:, :width // 2].cos()
        self.register_buffer("positions", pe, persistent=False)
        self.register_buffer("mask", torch.ones(length, length, dtype=torch.bool).tril(),
                             persistent=False)

        nn.init.normal_(self.embedding.weight, std=config.init_scale / math.sqrt(width))
        for layer in self.modules():
            if isinstance(layer, Linear):                          # biases already start at 0
                fan_in = layer.weight.shape[1]
                nn.init.normal_(layer.weight, std=config.init_scale / math.sqrt(fan_in))

    def forward(self, ids):
        length = ids.size(1)
        x = self.embedding(ids) * self.embedding_scale + self.positions[:length]
        x = dropout(x, self.embedding_dropout, self.training)
        for block in self.blocks:
            x = block(x, self.mask[:length, :length])
        if self.output is None:
            return x @ self.embedding.weight.T                     # tied: E also scores the next token
        return self.output(x)


def stream_fineweb(data):
    from datasets import load_dataset

    rows = load_dataset(DATASET, name=data.subset, revision=data.revision, split="train",
                        streaming=True, columns=["text"])
    # The timer can stop mid-shard. Keep Parquet's background reads in a worker
    # that can exit independently of the trainer and its CUDA/MPS runtime.
    return DataLoader(rows, batch_size=None, num_workers=data.workers,
                      multiprocessing_context="forkserver",
                      worker_init_fn=partial(exit_with_trainer, os.getpid()))


def exit_with_trainer(trainer_pid, worker_id):
    """Stop this reader when the trainer dies without shutting it down (kill, crash, out of memory)."""
    def watch():
        while True:
            time.sleep(1)
            try:
                os.kill(trainer_pid, 0)
            except OSError:
                os._exit(0)  # A normal exit can wait forever on Parquet's background reads.
    threading.Thread(target=watch, daemon=True).start()


def token_stream(rows):
    for row in rows:
        yield from row["text"].encode("utf-8")
        yield END_DOCUMENT


def batches(rows, batch_size, context_length):
    """Pack adjacent tokens into fixed windows with one shared token between batches."""
    tokens = token_stream(rows)
    previous = next(tokens, None)
    size = batch_size * context_length
    while previous is not None:
        following = list(islice(tokens, size))
        if len(following) < size:
            return
        window = torch.tensor([previous, *following], dtype=torch.long)
        yield window[:-1].view(batch_size, context_length), window[1:].view(batch_size, context_length)
        previous = following[-1]


def select_device(name):
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(),
                 "cpu": True}
    if name == "auto":
        name = next(key for key, enabled in available.items() if enabled)
    if not available[name]:
        raise SystemExit(f"The requested {name} device is unavailable in this PyTorch installation.")
    return torch.device(name)


def save_checkpoint(model, path, steps, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        torch.save({"model": model.state_dict(), "steps": steps,
                    "config": config.model_dump(mode="json")}, staging)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def cross_entropy(logits, targets, label_smoothing):
    # logits: (B, n, VOCAB)    targets: (B, n) token ids
    logp = logits.log_softmax(dim=-1)                              # (B, n, VOCAB)  log p for every token
    nll = -logp.gather(-1, targets[..., None]).squeeze(-1)         # (B, n)  -log p(correct token)
    smooth = -logp.mean(-1)                                        # (B, n)  mean of -log p over all tokens
    return ((1 - label_smoothing) * nll + label_smoothing * smooth).mean()


def train(config: Config, device):
    training = config.training
    optim = config.optimizer
    path = Path(config.runtime.checkpoint)
    if path.is_file():
        print(f"Starting from scratch. Saving this run will replace {path}. "
              "Use a different checkpoint path to keep both runs.", file=sys.stderr, flush=True)
    torch.manual_seed(training.seed)
    model = Decoder(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(optim.beta1, optim.beta2),
                                 eps=optim.epsilon, weight_decay=optim.weight_decay)
    if training.verbose:
        print(f"Opening {DATASET} ({config.data.subset}) on {device}. "
              "Waiting for the first batch before starting the training timer.", flush=True)
    stream = batches(stream_fineweb(config.data), training.batch_size, config.model.context_length)
    batch = next(stream, None)
    if batch is None:
        raise SystemExit("The stream ended before it supplied a full batch.")

    start = time.monotonic()
    deadline = start + training.hours * 3600
    next_log = start
    steps = 0
    while batch is not None:
        steps += 1
        lr = optim.lr_multiplier * config.model.width ** -0.5 * min(
            steps ** -0.5, steps * optim.warmup_steps ** -1.5
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, targets = (tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(x), targets, training.label_smoothing)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

        now = time.monotonic()
        tokens = steps * x.numel()
        if training.verbose and now >= next_log:
            print(f"step {steps}  loss {loss.item():.3f}  {tokens:,} tokens  "
                  f"{tokens / (now - start):,.0f} tokens/s  "
                  f"{max(0, deadline - now):.0f}s remaining", flush=True)
            next_log = now + training.log_interval_seconds
        if now >= deadline:
            break
        try:
            batch = next(stream, None)
        except Exception:
            # A remote read can fail hours into training. Preserve completed updates.
            save_checkpoint(model, path, steps, config)
            print(f"Stream failed at step {steps}. Saved {path}.", file=sys.stderr)
            raise
        if time.monotonic() >= deadline:
            break

    stream.close()
    save_checkpoint(model, path, steps, config)
    print(f"Saved {path} after {steps:,} steps ({tokens:,} tokens) on {device}.", flush=True)


@torch.inference_mode()
def generate(config: Config, prompt: str, device):
    checkpoint = torch.load(config.runtime.checkpoint, map_location="cpu", weights_only=True)
    # Original v0 checkpoints used the fixed default architecture.
    if "config" in checkpoint:
        arch = Architecture.model_validate(checkpoint["config"]["model"])
    else:
        arch = Architecture()
    model = Decoder(arch)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    print(f"Loaded {config.runtime.checkpoint} on {device}.", file=sys.stderr, flush=True)
    generation = config.generation
    if generation.seed is not None:
        torch.manual_seed(generation.seed)

    context = list(prompt.encode("utf-8")[-arch.context_length:]) or [END_DOCUMENT]
    ids = torch.tensor([context], dtype=torch.long, device=device)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def emit(text):
        # Random byte predictions can include terminal control characters.
        sys.stdout.write("".join(c if c.isprintable() or c in "\n\t" else ascii(c)[1:-1]
                                 for c in text))
        sys.stdout.flush()

    emit(prompt)
    for _ in range(generation.max_new_tokens):
        logits = model(ids)[:, -1]
        if generation.temperature == 0:
            sample = logits.argmax(dim=-1, keepdim=True)
        else:
            # Center logits before scaling so small temperatures remain stable.
            centered = logits - logits.amax(dim=-1, keepdim=True)
            temperature = max(generation.temperature, torch.finfo(logits.dtype).tiny)
            probs = (centered / temperature).softmax(dim=-1)
            sample = torch.multinomial(probs, 1)
        token = sample.item()
        if token == END_DOCUMENT:
            break
        emit(decoder.decode(bytes([token])))
        ids = torch.cat((ids, sample), dim=1)[:, -arch.context_length:]
    emit(decoder.decode(b"", final=True) + "\n")


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""First run (3.6 seconds of training once the stream is ready)
  uv run %(prog)s train --config quickstart.toml
  uv run %(prog)s generate --config quickstart.toml --prompt "The Moon "

Training streams public FineWeb-Edu data over the internet. A Hugging Face account is optional.
Use 'train --help' or 'generate --help' for command options.

Linux setup (macOS needs none)
  Install PyTorch once: 'uv sync --group cpu' (no GPU) or 'uv sync --group cuda' (NVIDIA GPU,
  driver 580 or newer). For an older driver, see 'Other PyTorch builds' in README.md.
  A plain 'uv sync' afterward uninstalls PyTorch. Rerun the same --group command to restore it.""",
    )
    config_option = {
        "type": Path, "default": argparse.SUPPRESS, "metavar": "FILE",
        "help": "Optional TOML file. Pass the same file to train and generate to use the same "
                "checkpoint and sampling settings. Omitted keys use built-in defaults. "
                "Explicit command-line values override file settings.",
    }
    parser.add_argument("--config", **config_option)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train", help="Pretrain from scratch on streaming FineWeb-Edu",
                              description="Each run starts from scratch and replaces its checkpoint on save. "
                                          "Use a separate checkpoint path for each experiment. "
                                          "Ctrl-C during training discards this run's unsaved progress.")
    training.add_argument("--hours", type=float,
                          help="Training hours from the first ready batch (0.25 = 15 minutes). "
                               "Required unless the TOML file sets training.hours.")
    training.add_argument("--batch-size", type=int,
                          help=f"Context windows per optimizer step (default {defaults.training.batch_size}).")
    training.add_argument("--verbose", "-v", action=argparse.BooleanOptionalAction, default=None,
                          help="Log loss, tokens per second, and time remaining. Off by default. "
                               "Use --no-verbose to disable logging enabled in the TOML file.")
    inference = sub.add_parser("generate", help="Stream a continuation from a saved checkpoint",
                               description="Load the checkpoint's architecture and stream the continuation.")
    inference.add_argument("--prompt", required=True, help="Required starting text. Quote prompts containing spaces.")
    inference.add_argument("--max-new-tokens", type=int,
                           help=f"Maximum additional byte tokens (default {defaults.generation.max_new_tokens}). "
                                "Generation stops early at a document boundary.")
    for command, action in ((training, "save"), (inference, "load")):
        command.add_argument("--config", **config_option)
        command.add_argument("--checkpoint", metavar="FILE",
                             help=f"Checkpoint file to {action} (default {defaults.runtime.checkpoint} "
                                  "in the working directory).")
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                             help="Optional. By default, auto selects the first available device "
                                  "in order: CUDA GPU (cuda), Apple GPU (mps), processor (cpu).")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    overrides = {"runtime": {"device": args.device, "checkpoint": args.checkpoint}}
    if args.command == "train":
        overrides["training"] = {"hours": args.hours, "batch_size": args.batch_size,
                                 "verbose": args.verbose}
    else:
        overrides["generation"] = {"max_new_tokens": args.max_new_tokens}
    try:
        config = load_config(getattr(args, "config", None), overrides)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.command == "train":
        if config.training.hours is None:
            training.error("set training.hours in the config or pass --hours")
        train(config, select_device(config.runtime.device))
    else:
        path = Path(config.runtime.checkpoint)
        if not path.is_file():
            inference.error(f"checkpoint {path} does not exist; run train first")
        generate(config, args.prompt, select_device(config.runtime.device))


if __name__ == "__main__":
    main()
