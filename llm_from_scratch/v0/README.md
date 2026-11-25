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

Look for `step 1`, then `Saved checkpoints/quickstart.pt`. The second command prints `Loaded checkpoints/quickstart.pt` and streams a continuation. Keep `--config` the same in both commands so generation uses your experiment's checkpoint path and sampling settings. Both commands automatically select an available device in the order CUDA, Apple MPS, CPU.

## Configure a run

For a longer run, copy [config.toml](config.toml). It lists every supported setting, with the reference model's values and a 15-minute training budget. Each call to `train` starts with newly initialized weights. A completed run replaces the file at `runtime.checkpoint`. Pressing Ctrl-C during training discards the current run's unsaved progress.

Give each experiment its own checkpoint path to keep both sets of weights. For example, the quickstart uses `checkpoints/quickstart.pt`. The following config trains a wider model for half an hour and saves it to `checkpoints/decoder-96.pt`. Omitted settings use built-in defaults.

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

Explicit CLI flags override TOML values. Omitted fields use their built-in defaults. Supply the duration with either `training.hours` or `--hours`. The timer starts when the first batch is ready and includes subsequent reads and accelerator work. A read or optimizer step in progress can finish past the deadline. Verbose mode prints loss, tokens processed, average tokens per second, and time remaining.

Checkpoints store the weights and the resolved training config. Generation **always uses the architecture saved in the checkpoint**. It reads the sampling and runtime settings from the supplied TOML file. You can also supply just `--checkpoint` and `--prompt`. Original v0 checkpoints load with the reference architecture. Generation echoes the prompt, streams decoded characters to the terminal, and stops at the document boundary token or the requested limit.

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

## Model and data

The [reference decoder](../../deep_dives/1_attention_is_all_you_need/decoder_only_transformer.md) uses a vocabulary built by reading a whole paper. This version uses 256 UTF-8 byte values plus one document boundary token, so it can encode each document as it arrives. It streams the text column of the FineWeb-Edu training split, packs consecutive documents into windows, and predicts the next token at every position. The causal mask, byte vocabulary, and next-token objective are fixed parts of v0. [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

The default decoder has 115,904 trainable parameters, a 64-token context, two blocks, four attention heads, width 64, and a 256-unit ReLU feed-forward layer. It uses sinusoidal positions, post-layer normalization, tied embeddings, and a 200-step warmup followed by inverse-square-root learning-rate decay. Short CPU runs can produce fragmentary text. Invalid UTF-8 is displayed as `�`.
