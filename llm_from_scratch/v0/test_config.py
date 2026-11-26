"""Regression tests for changing the architecture independently of the CLI and weights."""

import io
import multiprocessing
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from pydantic import ValidationError

from config import Config, load_config
from decoder_only_transformer import Decoder, batches, exit_with_trainer, generate, save_checkpoint


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
                       {"optimizer": {"beta1": 1.2}}, {"training": {"hours": float("nan")}}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Config.model_validate(values)

    def test_custom_context_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.pt"
            inference = {"runtime": {"checkpoint": str(path)},
                         "generation": {"temperature": 0, "max_new_tokens": 2}}
            config = Config.model_validate({
                **inference,
                "model": {"context_length": 7, "width": 15, "ff_width": 18, "layers": 1,
                          "heads": 3, "tie_embeddings": False, "pre_norm": True},
            })
            model = Decoder(config.model).eval()
            x, y = next(batches([{"text": "abcdefghijklmno"}], 2, config.model.context_length))
            self.assertEqual(x.shape, (2, 7))
            torch.testing.assert_close(x.flatten()[1:], y.flatten()[:-1])
            # Make greedy generation deterministic and ensure that it uses the saved output head.
            with torch.no_grad():
                model.output.weight.zero_()
            save_checkpoint(model, path, 1, config)
            with redirect_stdout(io.StringIO()) as out:
                generate(Config.model_validate(inference), "a prompt longer than seven bytes", torch.device("cpu"))
            self.assertEqual(out.getvalue(), "a prompt longer than seven bytes\\x00\\x00\n")


def watch_then_sleep(trainer_pid):
    exit_with_trainer(trainer_pid, worker_id=0)
    time.sleep(60)


class ReaderTests(unittest.TestCase):
    def test_reader_exits_when_trainer_dies(self):
        trainer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(trainer.kill)
        reader = multiprocessing.get_context("forkserver").Process(target=watch_then_sleep,
                                                                   args=(trainer.pid,))
        reader.start()
        self.addCleanup(reader.kill)
        trainer.kill()
        trainer.wait()
        reader.join(timeout=5)
        self.assertEqual(reader.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
