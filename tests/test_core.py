import json
import os
import random
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


class TracePreparationTests(unittest.TestCase):
    def test_internal_trace_resampling_converts_mbps_to_kbps_and_forward_fills(self):
        from trace_tools.prepare_internal_traces import resample_to_1hz_kbps

        values = resample_to_1hz_kbps([(0.0, 1.0), (0.5, 3.0), (2.0, 5.0)])

        self.assertEqual(values, [2000.0, 2000.0, 5000.0])

    def test_internal_split_builder_writes_deterministic_train_and_test_files(self):
        from trace_tools.prepare_internal_traces import build_splits

        with tempfile.TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "raw"
            output_root = Path(tmp) / "traces"
            source_dir = source_root / "cooked_3gp"
            source_dir.mkdir(parents=True)
            (source_dir / "a").write_text("0 1\n1 2\n2 3\n", encoding="utf-8")
            (source_dir / "b").write_text("0 2\n1 3\n2 4\n", encoding="utf-8")

            manifest = build_splits(
                source_root=source_root,
                output_root=output_root,
                seed=7,
                test_ratio=0.5,
                min_seconds=2,
                max_files_per_source=0,
                overwrite=True,
            )

        self.assertEqual(manifest["summary"]["cooked_3gp"]["train"], 1)
        self.assertEqual(manifest["summary"]["cooked_3gp"]["test"], 1)
        self.assertEqual(len(manifest["splits"]["train"]), 1)
        self.assertEqual(len(manifest["splits"]["test"]), 1)

    def test_data_source_doc_records_selected_puffer_window(self):
        text = (PROJECT_ROOT / "docs" / "data_sources.md").read_text(encoding="utf-8")

        self.assertIn("2026-05-06 11:00:00 UTC", text)
        self.assertIn("2026-05-07 11:00:00 UTC", text)


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


class ExperienceDatasetTests(unittest.TestCase):
    def test_percentile_reward_normalization_records_robust_range(self):
        from experience_dataset import ABRExperienceDataset

        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "expert.npz"
            trajectory = {
                "states": np.zeros((4, 6, 6), dtype=np.float32),
                "actions": np.array([0, 1, 2, 3], dtype=np.int64),
                "rewards": np.array([-100.0, 0.0, 4.0, 10.0], dtype=np.float32),
                "timesteps": np.arange(4, dtype=np.int64),
            }
            np.savez(data_path, trajectories=np.array([trajectory], dtype=object))

            dataset = ABRExperienceDataset(
                data_path=data_path,
                max_length=4,
                return_scale=100.0,
                reward_normalization="percentile",
                reward_percentile_low=25.0,
                reward_percentile_high=75.0,
            )

        self.assertEqual(dataset.stats["reward_normalization"], "percentile")
        self.assertGreater(dataset.stats["min_reward"], dataset.stats["raw_min_reward"])
        self.assertLess(dataset.stats["max_reward"], dataset.stats["raw_max_reward"])
        self.assertEqual(dataset.stats["return_scale"], 100.0)

    def test_sample_weights_are_collated_for_offline_rl(self):
        from experience_dataset import ABRExperienceDataset, collate_abr_samples_with_weights

        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "expert.npz"
            trajectory = {
                "states": np.zeros((4, 6, 6), dtype=np.float32),
                "actions": np.array([0, 1, 2, 3], dtype=np.int64),
                "rewards": np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32),
                "timesteps": np.arange(4, dtype=np.int64),
                "sample_weight": 0.35,
            }
            np.savez(data_path, trajectories=np.array([trajectory], dtype=object))

            dataset = ABRExperienceDataset(data_path=data_path, max_length=4)
            batch = collate_abr_samples_with_weights([dataset[0]])

        self.assertAlmostEqual(dataset.stats["sample_weight_mean"], 0.35, places=6)
        self.assertAlmostEqual(float(batch[-1][0]), 0.35, places=6)


class DataCollectorTests(unittest.TestCase):
    def test_high_mpc_policy_alias_is_normalized(self):
        from data_collector import EXPERT_POLICIES, choose_episode_policy, normalize_expert_policy

        self.assertEqual(normalize_expert_policy("high-mpc"), "high_mpc")
        self.assertIn(choose_episode_policy("high-mpc", random.Random(0), {}), EXPERT_POLICIES)


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
    def test_default_config_can_train_selected_expert_offline_rl_from_scratch(self):
        from config_utils import load_config

        config = load_config(
            AGENT_DIR / "configs" / "train_local_selected_expert_window5_safety.yaml"
        )

        self.assertTrue(config["offline_rl"]["enabled"])
        self.assertEqual(config["data"]["collection_mode"], "selected")
        self.assertEqual(config["sft"]["save_name"], "sft")
        self.assertIn("local_selected", config["sft"]["model_path"])
        self.assertIn("netllm-offline-rl", config["eval"]["final"]["policies"])

    def test_validation_trace_build_fails_clearly_when_data_is_missing(self):
        from train import build_validation_traces

        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "common": {"trace_split": "train", "trace_dir": tmp},
                "validation": {"episodes": 1, "split": "train", "trace_dir": tmp},
            }

            with self.assertRaisesRegex(FileNotFoundError, "Generate traces with trace_tools"):
                build_validation_traces(config)


class MetadataTests(unittest.TestCase):
    def test_chunk_size_input_remains_valid_json(self):
        chunk_size_path = AGENT_DIR / "assets" / "example_chunk_sizes.json"
        payload = json.loads(chunk_size_path.read_text(encoding="utf-8"))
        self.assertIn("sizes_mb", payload)


if __name__ == "__main__":
    unittest.main()
