"""Train the reference decoder on streamed FineWeb-Edu with MLX and sample from its weights."""

import argparse
import codecs
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from itertools import islice
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

if __package__:
    from .config import Architecture, Config, load_config
else:
    from config import Architecture, Config, load_config

END_DOCUMENT = 256  # IDs 0 through 255 represent UTF-8 bytes.
VOCAB = END_DOCUMENT + 1
DATASET = "HuggingFaceFW/fineweb-edu"


class DecoderBlock(nn.Module):
    """Masked self-attention followed by a position-wise feed-forward layer."""

    def __init__(self, config: Architecture):
        super().__init__()
        width = config.width
        self.heads = config.heads
        self.head_dim = width // config.heads
        self.scale = self.head_dim ** -0.5
        self.pre_norm = config.pre_norm
        self.q = nn.Linear(width, width, bias=config.attention_bias)
        self.k = nn.Linear(width, width, bias=config.attention_bias)
        self.v = nn.Linear(width, width, bias=config.attention_bias)
        self.out = nn.Linear(width, width, bias=config.attention_bias)
        self.ff = nn.Sequential(
            nn.Linear(width, config.ff_width, bias=config.ff_bias),
            nn.ReLU() if config.activation == "relu" else nn.GELU(),
            nn.Linear(config.ff_width, width, bias=config.ff_bias),
        )
        self.norm1 = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.attention_dropout_rate = config.attention_dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def attention(self, x):
        batch, length, width = x.shape

        def split_heads(tensor):
            return tensor.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = (split_heads(projection(x)) for projection in (self.q, self.k, self.v))
        if self.attention_dropout_rate > 0.0 and self.training:
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            mask = nn.MultiHeadAttention.create_additive_causal_mask(length, dtype=scores.dtype)
            weights = self.attention_dropout(mx.softmax(scores + mask, axis=-1, precise=True))
            heads = weights @ v
        else:
            heads = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask="causal"
            )
        return self.out(heads.transpose(0, 2, 1, 3).reshape(batch, length, width))

    def __call__(self, x):
        if self.pre_norm:
            x = x + self.dropout(self.attention(self.norm1(x)))
            return x + self.dropout(self.ff(self.norm2(x)))
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ff(x)))


class Decoder(nn.Module):
    def __init__(self, config: Architecture = Architecture()):
        super().__init__()
        width = config.width
        length = config.context_length
        self.embedding = nn.Embedding(VOCAB, width)
        self.embedding_scale = math.sqrt(width) if config.scale_embeddings else 1.0
        self.blocks = [DecoderBlock(config) for _ in range(config.layers)]
        self.dropout = nn.Dropout(config.embedding_dropout)
        self.output = None if config.tie_embeddings else nn.Linear(width, VOCAB, bias=False)

        positions = mx.arange(length, dtype=mx.float32)[:, None]
        angles = positions / (config.position_base ** (mx.arange(0, width, 2, dtype=mx.float32) / width))
        pe = mx.zeros((length, width), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(angles)
        pe[:, 1::2] = mx.cos(angles[:, : width // 2])
        # Underscore prefix excludes this buffer from parameters(), gradients, and checkpoints.
        self._positions = pe

        self.embedding.weight = mx.random.normal(self.embedding.weight.shape) * (
            config.init_scale / math.sqrt(width)
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                in_features = layer.weight.shape[1]
                layer.weight = mx.random.normal(layer.weight.shape) * (
                    config.init_scale / math.sqrt(in_features)
                )
                if "bias" in layer:
                    layer.bias = mx.zeros_like(layer.bias)

    def __call__(self, ids):
        length = ids.shape[1]
        x = self.embedding(ids)
        scale = mx.array(self.embedding_scale, dtype=x.dtype)
        x = self.dropout(x * scale + self._positions[:length].astype(x.dtype))
        for block in self.blocks:
            x = block(x)
        if self.output is None:
            return self.embedding.as_linear(x)
        return self.output(x)


def exit_with_trainer(trainer_pid):
    """Stop this reader when the trainer dies without shutting it down (kill, crash, out of memory)."""
    def watch():
        while True:
            time.sleep(1)
            try:
                os.kill(trainer_pid, 0)
            except OSError:
                os._exit(0)  # A normal exit can wait forever on Parquet's background reads.

    threading.Thread(target=watch, daemon=True).start()


def _fineweb_worker(subset, revision, trainer_pid, sender):
    exit_with_trainer(trainer_pid)
    try:
        import tqdm

        # The default multiprocessing lock leaks a semaphore when the trainer kills this process.
        tqdm.tqdm.set_lock(threading.RLock())
        from datasets import load_dataset

        rows = load_dataset(
            DATASET,
            name=subset,
            revision=revision,
            split="train",
            streaming=True,
            columns=["text"],
        )
        for row in rows:
            sender.send(("row", row["text"]))
        sender.send(("done", None))
    except BaseException as exc:
        try:
            sender.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass


def wait_for_exit(process, seconds):
    """Return whether the process exited within `seconds`."""
    # is_alive() reaps the process directly. join() also waits for PyTorch's helper process,
    # which `datasets` starts in the worker when PyTorch is installed and which outlives it by ~2 s.
    deadline = time.monotonic() + seconds
    while process.is_alive():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def stream_fineweb(data):
    """Stream FineWeb-Edu texts from an isolated worker process."""
    # Not fork or forkserver: both copy a process that already imported MLX (and its threads),
    # and pyarrow then segfaults while loading in the copy.
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    worker = ctx.Process(
        target=_fineweb_worker,
        args=(data.subset, data.revision, os.getpid(), sender),
        daemon=True,
    )
    worker.start()
    # With the trainer's copy of the send end closed, the worker's death ends the pipe, even mid-message.
    sender.close()
    try:
        while True:
            try:
                kind, payload = receiver.recv()
            except (EOFError, OSError):  # OSError: the pipe ended part-way through a message.
                wait_for_exit(worker, 5)
                raise RuntimeError(f"FineWeb stream worker exited with code {worker.exitcode}") from None
            if kind == "row":
                yield payload
            elif kind == "done":
                break
            else:
                raise RuntimeError(f"FineWeb stream worker failed: {payload}")
    finally:
        worker.terminate()
        if not wait_for_exit(worker, 2):
            worker.kill()
        receiver.close()


def token_stream(rows):
    for text in rows:
        yield from text.encode("utf-8")
        yield END_DOCUMENT


def batches(rows, batch_size, context_length):
    """Pack adjacent tokens into fixed windows with one shared token between batches."""
    try:
        tokens = token_stream(rows)
        previous = next(tokens, None)
        size = batch_size * context_length
        while previous is not None:
            following = list(islice(tokens, size))
            if len(following) < size:
                return
            window = mx.array([previous, *following], dtype=mx.int32)
            yield (
                window[:-1].reshape(batch_size, context_length),
                window[1:].reshape(batch_size, context_length),
            )
            previous = following[-1]
    finally:
        rows.close()


def select_device(name):
    gpu_available = mx.metal.is_available()
    if name == "gpu" and not gpu_available:
        raise SystemExit("Apple GPU (Metal) is unavailable in this MLX installation.")
    use_gpu = name == "gpu" or (name == "auto" and gpu_available)
    mx.set_default_device(mx.gpu if use_gpu else mx.cpu)
    return mx.default_device()


def save_checkpoint(model, path, steps, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".safetensors", delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        weights = dict(tree_flatten(model.parameters()))
        metadata = {
            "steps": str(steps),
            "config": json.dumps(config.model_dump(mode="json")),
        }
        mx.save_safetensors(str(staging), weights, metadata=metadata)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def train(config: Config, device):
    training = config.training
    optim_cfg = config.optimizer
    path = Path(config.runtime.checkpoint)
    if path.is_file():
        print(
            f"Starting from scratch. Saving this run will replace {path}. "
            "Use a different checkpoint path to keep both runs.",
            file=sys.stderr,
            flush=True,
        )

    mx.random.seed(training.seed)
    model = Decoder(config.model)
    model.train()
    optimizer = optim.Adam(
        learning_rate=0.0,
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        eps=optim_cfg.epsilon,
        bias_correction=True,
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.state, optimizer.state)

    mixed_precision = training.mixed_precision and device != mx.cpu
    if training.mixed_precision and not mixed_precision:
        print("training.mixed_precision is ignored on the cpu device.", file=sys.stderr, flush=True)

    def loss_fn(model, x, targets):
        if mixed_precision:
            # Keep float32 master weights and optimizer moments while running forward/backward in bfloat16.
            # Unlike torch.autocast, LayerNorm and the residual stream also run in bfloat16.
            master = model.trainable_parameters()
            model.update(tree_map(lambda p: p.astype(mx.bfloat16), master))
            try:
                logits = model(x)
            finally:
                model.update(master)
        else:
            logits = model(x)
        return nn.losses.cross_entropy(
            logits.astype(mx.float32),
            targets,
            label_smoothing=training.label_smoothing,
            reduction="mean",
        )

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    def step_fn(x, targets):
        loss, grads = loss_and_grad_fn(model, x, targets)
        if optim_cfg.weight_decay > 0.0:
            # Coupled L2 weight decay to match torch.optim.Adam.
            grads = tree_map(
                lambda g, p: g + optim_cfg.weight_decay * p,
                grads,
                model.trainable_parameters(),
            )
        optimizer.update(model, grads)
        return loss

    train_step = mx.compile(step_fn, inputs=state, outputs=state) if training.compile else step_fn

    if training.verbose:
        print(
            f"Opening {DATASET} ({config.data.subset}) on {device}. "
            "Waiting for the first batch before starting the training timer.",
            flush=True,
        )
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
        lr = optim_cfg.lr_multiplier * config.model.width ** -0.5 * min(
            steps ** -0.5, steps * optim_cfg.warmup_steps ** -1.5
        )
        optimizer.learning_rate = lr
        x, targets = batch
        loss = train_step(x, targets)
        mx.eval(state, loss)

        now = time.monotonic()
        tokens = steps * x.size
        if training.verbose and now >= next_log:
            print(
                f"step {steps}  loss {loss.item():.3f}  {tokens:,} tokens  "
                f"{tokens / (now - start):,.0f} tokens/s  "
                f"{max(0, deadline - now):.0f}s remaining",
                flush=True,
            )
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

    save_checkpoint(model, path, steps, config)
    stream.close()
    print(f"Saved {path} after {steps:,} steps ({tokens:,} tokens) on {device}.", flush=True)


def generate(config: Config, prompt: str, device):
    weights, metadata = mx.load(
        str(config.runtime.checkpoint), format="safetensors", return_metadata=True
    )
    arch = Architecture.model_validate(json.loads(metadata["config"])["model"])
    model = Decoder(arch)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    print(f"Loaded {config.runtime.checkpoint} on {device}.", file=sys.stderr, flush=True)

    generation = config.generation
    if generation.seed is not None:
        mx.random.seed(generation.seed)

    context = list(prompt.encode("utf-8")[-arch.context_length :]) or [END_DOCUMENT]
    ids = mx.array([context], dtype=mx.int32)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def emit(text):
        # Random byte predictions can include terminal control characters.
        sys.stdout.write(
            "".join(c if c.isprintable() or c in "\n\t" else ascii(c)[1:-1] for c in text)
        )
        sys.stdout.flush()

    emit(prompt)
    for _ in range(generation.max_new_tokens):
        logits = model(ids)[:, -1]
        if generation.temperature == 0:
            sample = mx.argmax(logits, axis=-1, keepdims=True)
        else:
            logits_f32 = logits.astype(mx.float32)
            centered = logits_f32 - mx.max(logits_f32, axis=-1, keepdims=True)
            # The GPU flushes smaller (subnormal) values to 0, and dividing by 0 gives NaN.
            temperature = max(generation.temperature, mx.finfo(mx.float32).smallest_normal)
            sample = mx.random.categorical(centered / temperature)[..., None]
        token = sample.item()
        if token == END_DOCUMENT:
            break
        emit(decoder.decode(bytes([token])))
        ids = mx.concatenate([ids, sample.astype(mx.int32)], axis=1)[:, -arch.context_length :]
    emit(decoder.decode(b"", final=True) + "\n")


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Quick benchmark (2 minutes of training with configs/19m.toml)
  uv run %(prog)s train --config configs/19m.toml --hours 0.033 --checkpoint checkpoints/19m-benchmark.safetensors
  uv run %(prog)s generate --config configs/19m.toml --checkpoint checkpoints/19m-benchmark.safetensors --prompt "The Moon "

Training streams public FineWeb-Edu data over the internet. A Hugging Face account is optional.
Use 'train --help' or 'generate --help' for command options.""",
    )
    config_option = {
        "type": Path,
        "default": argparse.SUPPRESS,
        "metavar": "FILE",
        "help": "Optional TOML file. Pass the same file to train and generate to use the same "
        "checkpoint and sampling settings. Omitted keys use built-in defaults. "
        "Explicit command-line values override file settings.",
    }
    parser.add_argument("--config", **config_option)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser(
        "train",
        help="Pretrain from scratch on streaming FineWeb-Edu",
        description="Each run starts from scratch and replaces its checkpoint on save. "
        "Use a separate checkpoint path for each experiment. "
        "Ctrl-C during training discards this run's unsaved progress.",
    )
    training.add_argument(
        "--hours",
        type=float,
        help="Training hours from the first ready batch (0.25 = 15 minutes). "
        "Required unless the TOML file sets training.hours.",
    )
    training.add_argument(
        "--batch-size",
        type=int,
        help=f"Context windows per optimizer step (default {defaults.training.batch_size}).",
    )
    training.add_argument(
        "--verbose",
        "-v",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Log loss, tokens per second, and time remaining. Off by default. "
        "Use --no-verbose to disable logging enabled in the TOML file.",
    )
    inference = sub.add_parser(
        "generate",
        help="Stream a continuation from a saved checkpoint",
        description="Load the checkpoint's architecture and stream the continuation.",
    )
    inference.add_argument(
        "--prompt", required=True, help="Required starting text. Quote prompts containing spaces."
    )
    inference.add_argument(
        "--max-new-tokens",
        type=int,
        help=f"Maximum additional byte tokens (default {defaults.generation.max_new_tokens}). "
        "Generation stops early at a document boundary.",
    )
    for command, action in ((training, "save"), (inference, "load")):
        command.add_argument("--config", **config_option)
        command.add_argument(
            "--checkpoint",
            metavar="FILE",
            help=f"Checkpoint file to {action} (default {defaults.runtime.checkpoint} "
            "in the working directory).",
        )
        command.add_argument(
            "--device",
            choices=("auto", "cpu", "gpu"),
            help="Optional. By default, auto selects the Apple GPU (Metal) when available, else cpu.",
        )
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    overrides = {"runtime": {"device": args.device, "checkpoint": args.checkpoint}}
    if args.command == "train":
        overrides["training"] = {
            "hours": args.hours,
            "batch_size": args.batch_size,
            "verbose": args.verbose,
        }
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