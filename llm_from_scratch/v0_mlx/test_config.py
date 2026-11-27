"""Config, checkpoint, and reader tests, ported from ../v0/test_config.py."""

import io
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import mlx.core as mx
from pydantic import ValidationError

import decoder_only_transformer_mlx as mm
from config import Config, load_config


class ConfigTests(unittest.TestCase):
    def test_example_matches_reference_settings(self):
        config = load_config(Path(__file__).with_name("config.toml"), {})
        expected = Config.model_validate({"training": {"hours": 0.25}})
        self.assertEqual(config, expected)
        for name in type(config).model_fields:
            section = getattr(config, name)
            documented = {key for key, value in section.model_dump().items() if value is not None}
            self.assertEqual(section.model_fields_set, documented)

    def test_overrides_and_rejected_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text("[training]\nhours=2\nbatch_size=4\n")
            config = load_config(path, {"training": {"hours": 0.25, "batch_size": None}})
            self.assertEqual((config.training.hours, config.training.batch_size), (0.25, 4))
        for values in ({"model": {"width": 15, "heads": 4}}, {"model": {"layer": 8}},
                       {"optimizer": {"beta1": 1.2}}, {"training": {"hours": float("nan")}},
                       # Settings that v0 accepts and this port does not.
                       {"training": {"label_smoothing": 1.0}}, {"data": {"workers": 2}},
                       {"runtime": {"device": "mps"}}, {"runtime": {"device": "cuda"}}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Config.model_validate(values)


class CheckpointTests(unittest.TestCase):
    def test_custom_context_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.safetensors"
            inference = {"runtime": {"checkpoint": str(path)},
                         "generation": {"temperature": 0, "max_new_tokens": 2}}
            config = Config.model_validate({
                **inference,
                "model": {"context_length": 7, "width": 15, "ff_width": 18, "layers": 1,
                          "heads": 3, "tie_embeddings": False, "pre_norm": True},
            })
            model = mm.Decoder(config.model)
            x, y = next(mm.batches((t for t in ["abcdefghijklmno"]), 2, config.model.context_length))
            self.assertEqual(x.shape, (2, 7))
            self.assertTrue(mx.array_equal(x.flatten()[1:], y.flatten()[:-1]).item())
            # Make greedy generation deterministic and ensure that it uses the saved output head.
            model.output.weight = mx.zeros_like(model.output.weight)
            mm.save_checkpoint(model, path, 1, config)
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                mm.generate(Config.model_validate(inference), "a prompt longer than seven bytes",
                            mm.select_device("auto"))
            self.assertEqual(out.getvalue(), "a prompt longer than seven bytes\\x00\\x00\n")


def watch_then_sleep(trainer_pid):
    mm.exit_with_trainer(trainer_pid)
    time.sleep(60)


def send_then_die_mid_message(subset, revision, trainer_pid, sender):
    sender.send(("row", "first"))
    threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGKILL)).start()
    sender.send(("row", "x" * 10_000_000))  # Blocks part-way through once the pipe buffer is full.


class ReaderTests(unittest.TestCase):
    def test_reader_exits_when_trainer_dies(self):
        trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(trainer.kill)
        reader = multiprocessing.get_context("spawn").Process(target=watch_then_sleep, args=(trainer.pid,))
        reader.start()
        self.addCleanup(reader.kill)
        trainer.kill()
        trainer.wait()
        reader.join(timeout=15)  # Includes the spawned process's startup.
        self.assertEqual(reader.exitcode, 0)

    def test_reader_killed_mid_message_raises(self):
        result = []

        def read_next():
            try:
                next(rows)
            except RuntimeError as exc:
                result.append(str(exc))

        with mock.patch.object(mm, "_fineweb_worker", send_then_die_mid_message):
            rows = mm.stream_fineweb(Config().data)
            self.assertEqual(next(rows), "first")
            time.sleep(1)  # The worker is now dead, with part of a message in the pipe.
            thread = threading.Thread(target=read_next, daemon=True)
            thread.start()
            thread.join(30)
        self.assertEqual(result, [f"FineWeb stream worker exited with code {-signal.SIGKILL}"])


if __name__ == "__main__":
    unittest.main()
