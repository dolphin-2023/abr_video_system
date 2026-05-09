import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = PROJECT_ROOT / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


class ConfigUtilsTests(unittest.TestCase):
    def test_load_dump_and_resolve_config(self):
        from config_utils import dump_config, load_config, resolve_path, section

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            payload = {"run": {"name": "unit"}, "empty": None}
            dump_config(payload, path)

            loaded = load_config(path)
            self.assertEqual(loaded["run"]["name"], "unit")
            self.assertEqual(section(loaded, "empty"), {})
            self.assertEqual(resolve_path("child", base_dir=Path(tmp)), Path(tmp) / "child")


class TraceUtilsTests(unittest.TestCase):
    def test_stratified_sampling_keeps_sources_when_possible(self):
        from trace_utils import sample_trace_files

        files = [
            "puffer__a.txt",
            "puffer__b.txt",
            "fcc__a.txt",
            "fcc__b.txt",
            "hsdpa__a.txt",
            "hsdpa__b.txt",
        ]
        sampled = sample_trace_files(files, episodes=3, mode="stratified", seed=7)

        self.assertEqual(len(sampled), 3)
        self.assertEqual({item.split("__", 1)[0] for item in sampled}, {"puffer", "fcc", "hsdpa"})


class ABREnvTests(unittest.TestCase):
    def test_env_reset_and_step_with_temp_trace(self):
        from abr_env import ABREnv

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "puffer__unit.txt"
            trace_path.write_text("\n".join(["1200", "1800", "2400", "3200", "4200"]), encoding="utf-8")
            chunk_sizes = np.full((6, 4), 0.2, dtype=np.float32)

            with patch.dict(os.environ, {}, clear=True):
                env = ABREnv(
                    trace_dir=tmp,
                    random_start=False,
                    chunk_sizes_mb=chunk_sizes,
                    qoe_profile="pensieve",
                )
                state, info = env.reset(seed=1)
                next_state, reward, done, truncated, step_info = env.step(0)

        self.assertEqual(state.shape, (6, 6))
        self.assertEqual(next_state.shape, (6, 6))
        self.assertFalse(truncated)
        self.assertIsInstance(done, bool)
        self.assertIn("qoe", step_info)
        self.assertIsInstance(reward, float)


class PolicyTests(unittest.TestCase):
    def test_mpc_policy_returns_valid_action(self):
        from abr_env import ABREnv
        from evaluate import MpcPolicy

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "puffer__unit.txt"
            trace_path.write_text("\n".join(["3000", "3000", "3000", "3000", "3000"]), encoding="utf-8")
            chunk_sizes = np.full((6, 4), 0.25, dtype=np.float32)

            with patch.dict(os.environ, {}, clear=True):
                env = ABREnv(trace_dir=tmp, random_start=False, chunk_sizes_mb=chunk_sizes)
                state, _ = env.reset(seed=2)
                state[1, -1] = 1.0
                state[2, :] = 3.0
                policy = MpcPolicy()
                policy.reset(env)
                action = policy.act(state, env)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, env.num_bitrates)


class PensieveTests(unittest.TestCase):
    def test_pensieve_forward_shape(self):
        import torch

        from pensieve_torch import PensieveActorCritic

        model = PensieveActorCritic()
        logits, value = model.forward(torch.zeros(6, 6, dtype=torch.float32))

        self.assertEqual(tuple(logits.shape), (1, 6))
        self.assertEqual(tuple(value.shape), (1,))


class PipelineConfigTests(unittest.TestCase):
    def test_default_config_includes_full_pipeline_stages(self):
        from config_utils import load_config

        config = load_config(AGENT_DIR / "configs" / "train_v3.yaml")

        self.assertTrue(config["offline_rl"]["enabled"])
        self.assertTrue(config["pensieve"]["enabled"])
        self.assertIn("netllm-offline-rl", config["eval"]["final"]["policies"])
        self.assertIn("pensieve", config["eval"]["final"]["policies"])


class MetadataTests(unittest.TestCase):
    def test_tracked_training_stats_remains_valid_json(self):
        stats_path = AGENT_DIR / "models" / "training_stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        self.assertIn("target_return", payload)


if __name__ == "__main__":
    unittest.main()
