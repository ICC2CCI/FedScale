import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import ExperimentStore  # noqa: E402
from swanlab_sync import SwanLabSync  # noqa: E402


class FakeRun:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log(self, metrics, step=None):
        self.logs.append((metrics, step))

    def finish(self):
        self.finished = True


class FakeSwanLab:
    def __init__(self):
        self.runs = []
        self.logins = []

    def login(self, **kwargs):
        self.logins.append(kwargs)
        return True

    def init(self, **kwargs):
        run = FakeRun()
        self.runs.append((kwargs, run))
        return run


class SwanLabSyncTests(unittest.TestCase):
    def test_uses_dashboard_project_variable_without_sdk_project_collision(self):
        with mock.patch.dict(
            os.environ,
            {
                "DASHBOARD_SWANLAB_PROJECT": "dashboard-project",
                "SWANLAB_PROJECT": "sdk-reserved-value",
            },
            clear=False,
        ):
            sync = SwanLabSync(ExperimentStore(Path(".")))
            self.assertEqual(sync.project, "dashboard-project")

    def test_publishes_round_and_final_metrics_without_model_data(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"SWANLAB_ENABLED": "true", "SWANLAB_API_KEY": "test-key"},
            clear=False,
        ):
            root = Path(temporary)
            directory = root / "ddp-smoke"
            directory.mkdir()
            (directory / "experiment_config.json").write_text(json.dumps({
                "run_id": 123,
                "model": "openlm-research/open_llama_3b_v2",
                "dataset": "vicgalle/alpaca-gpt4",
                "distributed_strategy": "ddp",
                "run_config": {"train.training-arguments.max-steps": 1},
            }), encoding="utf-8")
            (directory / "experiment_state.json").write_text(json.dumps({
                "status": "completed", "latest_completed_round": 1,
            }), encoding="utf-8")
            (directory / "experiment_summary.json").write_text(json.dumps({
                "status": "completed",
                "aggregated_client_train_metrics": {
                    "1": {"client_training_seconds": 2.5}
                },
            }), encoding="utf-8")
            (directory / "federated_metrics_round_1.json").write_text(json.dumps({
                "server_round": 1, "federated_cycle_s": 8.5,
            }), encoding="utf-8")
            (directory / "evaluation_summary.json").write_text(json.dumps({
                "status": "completed", "metrics": {"ppl": 3.2}
            }), encoding="utf-8")

            sync = SwanLabSync(ExperimentStore(root))
            uploads = []
            sync._swanlab = object()
            sync._run_upload = uploads.append
            sync._last_error = "earlier transient error"
            sync.sync_once()

            self.assertEqual(len(uploads), 1)
            self.assertIsNone(sync.status()["last_error"])
            payload = uploads[0]
            kwargs = payload["run_kwargs"]
            self.assertEqual(kwargs["mode"], "online")
            self.assertEqual(kwargs["id"], "flower-ddp-smoke")
            self.assertEqual(kwargs["name"], "ddp-smoke")
            self.assertIn("log_dir", kwargs)
            self.assertEqual(kwargs["resume"], "allow")
            all_metrics = {
                key: value
                for metrics, _step in payload["events"]
                for key, value in metrics.items()
            }
            self.assertEqual(all_metrics["federated/federated_cycle_s"], 8.5)
            self.assertEqual(all_metrics["client/train/client_training_seconds"], 2.5)
            self.assertEqual(all_metrics["final_evaluation/metrics/ppl"], 3.2)


if __name__ == "__main__":
    unittest.main()
