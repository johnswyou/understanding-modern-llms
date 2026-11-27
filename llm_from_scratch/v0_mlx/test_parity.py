"""Check that the MLX port computes the same model, with the same settings, as the PyTorch port in ../v0."""

import contextlib
import io
import random
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np
import torch

import decoder_only_transformer_mlx as mm
from config import Architecture, Config, Generation, Optimizer, Training

HERE = Path(__file__).resolve().parent
# Importing v0 as a package makes its relative imports load v0/config.py, not this directory's copy.
sys.path.insert(0, str(HERE.parent))
from v0 import config as torch_config
from v0 import decoder_only_transformer as pt

ARCH = dict(context_length=64, width=64, heads=4, ff_width=128, layers=2,
            embedding_dropout=0.0, residual_dropout=0.0, attention_dropout=0.0)


def to_mlx(state_dict):
    # MLX's nn.Sequential stores its children under "layers".
    return [(name.replace(".ff.", ".ff.layers."), mx.array(value.numpy())) for name, value in state_dict.items()]


def to_numpy(weights):
    return {name.replace(".ff.layers.", ".ff."): np.array(value) for name, value in weights.items()}


def texts():
    rng = random.Random(0)
    words = "the moon orbits earth light water stone river city music number field".split()
    return [" ".join(rng.choice(words) for _ in range(rng.randint(5, 30))) for _ in range(40)]


TEXTS = texts()


def settings(checkpoint, model=None, **training):
    return {
        "model": {**ARCH, **(model or {})},
        "optimizer": {"weight_decay": 0.01, "warmup_steps": 4},
        "training": {"hours": 1.0, "batch_size": 4, "label_smoothing": 0.1, **training},
        "generation": {"max_new_tokens": 40, "temperature": 0.0},
        "runtime": {"checkpoint": str(checkpoint)},
    }


def quietly(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        function(*args)
    return out.getvalue()


class SettingsParityTest(unittest.TestCase):
    def test_config_defaults_match_v0(self):
        for ours in (Architecture, Optimizer, Training, Generation):
            with self.subTest(section=ours.__name__):
                self.assertEqual(ours().model_dump(), getattr(torch_config, ours.__name__)().model_dump())

    def test_presets_match_v0(self):
        for name in ("config.toml", "quickstart.toml", "configs/11m.toml", "configs/14m.toml", "configs/19m.toml"):
            ours, theirs = (tomllib.loads((root / name).read_text()) for root in (HERE, HERE.parent / "v0"))
            for section in ("model", "optimizer", "training", "generation"):
                with self.subTest(file=name, section=section):
                    self.assertEqual(ours.get(section), theirs.get(section))


class ModelParityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.device = mm.select_device("auto")

    def train_mlx(self, config):
        with mock.patch.object(mm, "stream_fineweb", lambda data: (t for t in TEXTS)):
            quietly(mm.train, config, self.device)
        return mx.load(str(config.runtime.checkpoint))

    def train_both(self, **mlx_training):
        """Train both ports from the same initial weights on the same text, with dropout off."""
        reference = torch_config.Config.model_validate(settings(self.tmp / "reference.pt"))
        config = Config.model_validate(settings(self.tmp / "mlx.safetensors", **mlx_training))
        torch.manual_seed(reference.training.seed)
        initial = to_mlx(pt.Decoder(reference.model).state_dict())

        class SameInit(mm.Decoder):
            def __init__(self, arch):
                super().__init__(arch)
                self.load_weights(initial)

        with mock.patch.object(pt, "stream_fineweb", lambda data: [{"text": t} for t in TEXTS]):
            quietly(pt.train, reference, torch.device("cpu"))
        with mock.patch.object(mm, "Decoder", SameInit):
            weights = self.train_mlx(config)
        trained = torch.load(reference.runtime.checkpoint, weights_only=True)["model"]
        return reference, config, to_numpy(dict(initial)), trained, weights

    def test_forward_matches_pytorch(self):
        ids = np.random.default_rng(0).integers(0, mm.VOCAB, size=(3, 64))
        for overrides in ({}, {"tie_embeddings": False}, {"pre_norm": True, "activation": "gelu"}):
            with self.subTest(**overrides):
                torch.manual_seed(0)
                reference = pt.Decoder(torch_config.Architecture(**ARCH, **overrides)).eval()
                model = mm.Decoder(Architecture(**ARCH, **overrides))
                model.load_weights(to_mlx(reference.state_dict()))
                model.eval()
                with torch.no_grad():
                    expected = reference(torch.tensor(ids)).numpy()
                np.testing.assert_allclose(np.array(model(mx.array(ids, dtype=mx.int32))), expected, atol=1e-4)

    def test_training_and_generation_match_pytorch(self):
        for compile in (False, True):
            with self.subTest(compile=compile):
                reference, config, initial, trained, weights = self.train_both(compile=compile)
                got = to_numpy(weights)
                for name, value in trained.items():
                    # Compare against how far training moved each tensor, so the bound is scale-free.
                    moved = np.abs(value.numpy() - initial[name]).max()
                    self.assertLess(np.abs(value.numpy() - got[name]).max(), 1e-3 * moved, name)
                self.assertEqual(quietly(mm.generate, config, "the moon ", self.device),
                                 quietly(pt.generate, reference, "the moon ", torch.device("cpu")))

    def test_compile_keeps_dropout_random(self):
        # PyTorch draws different dropout masks, so compare MLX against itself. If mx.compile froze the
        # random state, every step would reuse one mask and the two runs would diverge.
        dropout = {"embedding_dropout": 0.1, "residual_dropout": 0.1, "attention_dropout": 0.1}
        runs = [self.train_mlx(Config.model_validate(settings(self.tmp / f"{c}.safetensors", dropout, compile=c)))
                for c in (False, True)]
        for name, value in runs[0].items():
            np.testing.assert_allclose(np.array(runs[1][name]), np.array(value), atol=1e-4, err_msg=name)

    def test_mixed_precision_saves_finite_float32_weights(self):
        config = Config.model_validate(settings(self.tmp / "bf16.safetensors", compile=True, mixed_precision=True))
        for name, value in self.train_mlx(config).items():
            self.assertEqual(value.dtype, mx.float32, name)
            self.assertTrue(mx.isfinite(value).all().item(), name)


if __name__ == "__main__":
    unittest.main()
