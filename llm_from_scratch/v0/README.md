# FineWeb-Edu decoder

From the repository root, enter the v0 project and run the script with no arguments to see its commands and examples.

```sh
cd llm_from_scratch/v0
uv run decoder_only_transformer.py
```

Use `train --help` or `generate --help` after the script name for command-specific options. uv creates the project environment on the first invocation. Linux CPU and Apple Silicon (macOS 14 or newer) use the default setup. For a Linux GPU, see [Linux GPU setup](#linux-gpu-setup). [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

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
| `[training]` | Hours, batch size, label smoothing, random seed, verbose output, and log interval |
| `[data]` | FineWeb-Edu subset, dataset revision, and reader process count |
| `[runtime]` | Device and checkpoint path, relative to the working directory |
| `[generation]` | Maximum new tokens, sampling temperature, and optional sampling seed |

Explicit CLI flags override TOML values. Omitted fields use their built-in defaults. Supply the duration with either `training.hours` or `--hours`. The timer starts when the first batch is ready and includes subsequent reads and accelerator work. A read or optimizer step in progress can finish past the `--hours` deadline.

Checkpoints store the weights and the resolved training config. Generation **always uses the architecture saved in the checkpoint**. It reads the sampling and runtime settings from the supplied TOML file. For `generate`, `--config` is optional. Without it, pass the checkpoint path directly, and the device and sampling settings use built-in defaults: automatic device selection, 300 new tokens, temperature 1.0, and no sampling seed.

```sh
uv run decoder_only_transformer.py generate --checkpoint checkpoints/decoder-96.pt --prompt "Sunlight "
```

## Linux GPU setup

On a Linux GPU with a CUDA 13.0-compatible driver, install the CUDA build of PyTorch once in the project environment.

```sh
uv sync --no-group cpu --group cuda
```

For ROCm or another CUDA build, use `uv sync --no-group cpu` followed by `uv pip install torch --torch-backend=auto`. [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/)

After either setup, run the environment's Python directly to keep the selected build. The script detects the GPU automatically.

```sh
.venv/bin/python decoder_only_transformer.py train --config quickstart.toml
.venv/bin/python decoder_only_transformer.py generate --config quickstart.toml --prompt "Gravity "
```
