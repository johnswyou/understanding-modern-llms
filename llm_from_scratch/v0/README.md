# FineWeb-Edu decoder

From the repository root, enter the v0 project. On Linux, install PyTorch once with the command that matches the machine. Apple Silicon (macOS 14 or newer) skips this step. For other GPU builds, see [Other PyTorch builds](#other-pytorch-builds). [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```sh
cd llm_from_scratch/v0
uv sync --group cpu    # Linux without a GPU
uv sync --group cuda   # Linux with an NVIDIA GPU and driver 580 or newer (CUDA 13.0)
```

To check your driver, run `nvidia-smi`. The header shows the newest CUDA version the driver supports. If it is below 13.0, use [Other PyTorch builds](#other-pytorch-builds) instead. Otherwise PyTorch cannot reach the GPU and training silently runs on the CPU.

Run the script with no arguments to see its commands and examples. Use `train --help` or `generate --help` after the script name for command-specific options.

```sh
uv run decoder_only_transformer.py
```

`decoder_only_transformer_educational.py` takes the same commands and computes the same model, with its layers and loss written by hand so the math is visible. It trains more slowly. Checkpoints work with either file.

## Try it in two commands

The first command streams public FineWeb-Edu text over the internet. You can train without a Hugging Face account, including when the library prints a warning about anonymous requests. The [quickstart config](quickstart.toml) gives the model **3.6 seconds of training**. Allow extra time for setup and the first batch.

```sh
uv run decoder_only_transformer.py train --config quickstart.toml
uv run decoder_only_transformer.py generate --config quickstart.toml --prompt "The surface of the Moon "
```

Keep `--config` the same in both commands so generation uses your experiment's checkpoint path and sampling settings. Both commands automatically select an available device in the order CUDA, Apple MPS, CPU.

## Configure a run

For a full run, copy [config.toml](config.toml). It lists every supported setting. Each call to `train` starts with newly initialized weights. Training saves the weights when it finishes or when the data stream fails partway through. The save overwrites the file `runtime.checkpoint`, if it exists. Pressing Ctrl-C during training discards the current run's unsaved progress. Omitted settings use built-in defaults.

Example config:

```toml
[model]
context_length = 128
width = 96
heads = 6
ff_width = 384
layers = 3

[training]
hours = 0.5

[runtime]
checkpoint = "checkpoints/decoder-96.pt"
```

Save the example as `experiment.toml` and use it for training and generation.

```sh
uv run decoder_only_transformer.py train --config experiment.toml --verbose
uv run decoder_only_transformer.py generate --config experiment.toml --prompt "Sunlight " --max-new-tokens 200
```

| TOML section | What it controls |
| --- | --- |
| `[model]` | Context, width, heads, feed-forward size, depth, dropout, projection biases, activation, normalization, sinusoidal positions, embedding scaling and tying, initialization |
| `[optimizer]` | Adam betas, epsilon, weight decay, warmup steps, and learning-rate multiplier |
| `[training]` | Hours, batch size, label smoothing, random seed, verbose output, log interval, mixed precision, and compilation |
| `[data]` | FineWeb-Edu subset, dataset revision, and reader process count |
| `[runtime]` | Device and checkpoint path, relative to the working directory |
| `[generation]` | Maximum new tokens, sampling temperature, and optional sampling seed |

`training.mixed_precision` and `training.compile` speed up training on a GPU and are off by default. They apply to `decoder_only_transformer.py`. The educational file accepts both settings and ignores them: it always trains uncompiled in float32. Run `uv run benchmark_speed.py` to measure what each gains on your machine.

Explicit CLI flags override TOML values. Omitted fields use their built-in defaults. Supply the duration with either `training.hours` or `--hours`. The timer starts when the first batch is ready and includes subsequent reads and accelerator work. A read or optimizer step in progress can finish past the `--hours` deadline.

Checkpoints store the weights and the resolved training config. Generation **always uses the architecture saved in the checkpoint**. It reads the sampling and runtime settings from the supplied TOML file. For `generate`, `--config` is optional. Without it, pass the checkpoint path directly, and the device and sampling settings use built-in defaults: automatic device selection, 300 new tokens, temperature 1.0, and no sampling seed.

```sh
uv run decoder_only_transformer.py generate --checkpoint checkpoints/decoder-96.pt --prompt "Sunlight "
```

## Other PyTorch builds

For an NVIDIA driver older than 580, or an AMD GPU (ROCm), run:

```sh
uv sync
uv pip install torch --torch-backend=auto
```

`--torch-backend=auto` reads the installed driver and picks the newest PyTorch build that driver can run. `uv.lock` does not pin this build, so its PyTorch version can differ from the locked one. [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)

On Linux, a plain `uv sync` removes PyTorch, because `uv sync` deletes every package the command did not ask for. Repeat the `--group` flag, or the `uv pip install` step, whenever you re-sync. `uv run` only adds missing packages, so it keeps whichever build you installed.
