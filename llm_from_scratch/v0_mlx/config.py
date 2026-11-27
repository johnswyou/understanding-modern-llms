"""Validated settings for the MLX port of the v0 byte-level decoder.

Copied from ../v0/config.py. Keep [model], [optimizer], [training], and [generation] identical
to v0, except that label_smoothing must be below 1 (MLX's cross_entropy rejects 1).
test_parity.py checks that the defaults and presets still match v0's.
"""

import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


Rate = Annotated[float, Field(ge=0, lt=1)]
Text = Annotated[str, Field(min_length=1)]
Seed = Annotated[int, Field(ge=0, lt=2**64)]


class Architecture(Settings):
    context_length: PositiveInt = 64
    width: PositiveInt = 64
    heads: PositiveInt = 4
    ff_width: PositiveInt = 256
    layers: PositiveInt = 2
    embedding_dropout: Rate = 0.1
    residual_dropout: Rate = 0.1
    attention_dropout: Rate = 0.0
    attention_bias: bool = False
    ff_bias: bool = True
    activation: Literal["relu", "gelu"] = "relu"
    pre_norm: bool = False
    layer_norm_eps: PositiveFloat = 1e-6
    position_base: PositiveFloat = 10000.0
    scale_embeddings: bool = True
    tie_embeddings: bool = True
    init_scale: PositiveFloat = 1.0

    @model_validator(mode="after")
    def check_head_width(self) -> Self:
        if self.width % self.heads:
            raise ValueError("model.width must be divisible by model.heads")
        return self


class Optimizer(Settings):
    beta1: Rate = 0.9
    beta2: Rate = 0.98
    epsilon: PositiveFloat = 1e-9
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_steps: PositiveInt = 200
    lr_multiplier: PositiveFloat = 1.0


class Training(Settings):
    hours: PositiveFloat | None = None
    batch_size: PositiveInt = 32
    label_smoothing: float = Field(default=0.1, ge=0, lt=1)
    seed: Seed = 0
    verbose: bool = False
    log_interval_seconds: PositiveFloat = 5.0
    mixed_precision: bool = False
    compile: bool = False


class Data(Settings):
    subset: Text = "sample-10BT"
    revision: Text = "main"


class Generation(Settings):
    max_new_tokens: int = Field(default=300, ge=0)
    temperature: float = Field(default=1.0, ge=0)
    seed: Seed | None = None


class Runtime(Settings):
    checkpoint: Text = "fineweb-v0.safetensors"
    device: Literal["auto", "cpu", "gpu"] = "auto"


class Config(Settings):
    model: Architecture = Architecture()
    optimizer: Optimizer = Optimizer()
    training: Training = Training()
    data: Data = Data()
    generation: Generation = Generation()
    runtime: Runtime = Runtime()


def load_config(path: Path | None, overrides: dict) -> Config:
    values = {}
    if path is not None:
        with path.open("rb") as file:
            values = tomllib.load(file)
    for section, fields in overrides.items():
        table = values.setdefault(section, {})
        if not isinstance(table, dict):
            raise ValueError(f"[{section}] must be a TOML table")
        table.update({key: value for key, value in fields.items() if value is not None})
    return Config.model_validate(values)
