# FineWeb-Edu decoder (MLX)

This is a port of the [v0 decoder](../v0/README.md) to MLX, Apple's array framework for Apple silicon. It trains the same model on the same streamed FineWeb-Edu bytes and takes the same commands and TOML settings, except for the differences listed below. The v0 README explains how training, configs, and checkpoints work.

## Setup

The project runs only on Apple silicon Macs. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
cd llm_from_scratch/v0_mlx
uv sync
```

`uv sync` also installs PyTorch, which the parity test needs to run v0's code.

## Try it in two commands

```sh
uv run decoder_only_transformer_mlx.py train --config quickstart.toml
uv run decoder_only_transformer_mlx.py generate --config quickstart.toml --prompt "The surface of the Moon "
```

Run the script with no arguments to see every command and a 2-minute benchmark with the 19M preset.

## Differences from v0

| Setting | v0 (PyTorch) | v0_mlx |
| --- | --- | --- |
| `runtime.device` | `auto`, `cpu`, `cuda`, `mps` | `auto`, `cpu`, `gpu` |
| `runtime.checkpoint` | PyTorch `.pt` file | `.safetensors` file. Neither script can read the other's checkpoints. |
| `data.workers` | Number of reader processes | Not accepted. One process reads the stream. |
| `training.mixed_precision` | `torch.autocast`: matrix multiplies in bfloat16; LayerNorm and the residual stream in float32 | Every parameter is cast to bfloat16 for the forward and backward passes, so LayerNorm and the residual stream also run in bfloat16. Losses differ slightly from v0. |
| `training.compile` | `torch.compile` on the model | `mx.compile` on the whole training step |
| `training.label_smoothing` | 0 to 1 | 0 up to but not including 1, because MLX's loss function rejects 1 |

`config.py` and the TOML files are copies of v0's. When you change `[model]`, `[optimizer]`, `[training]`, or `[generation]` in v0, make the same change here. The tests below fail until you do.

## Tests

```sh
uv run python -m unittest
```

- `test_parity.py` compares this port with v0. It checks that the `config.py` defaults and the preset TOML values match v0's. It then gives both ports the same initial weights and the same text, and checks that the forward pass, a short training run (with and without `compile`), and greedy generation match PyTorch. It imports `../v0`, so the two directories must stay side by side.
- `test_config.py` is ported from v0's tests. It covers config validation, a checkpoint round trip, and the data reader, including a reader killed in the middle of sending text.
