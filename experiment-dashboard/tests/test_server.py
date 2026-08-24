import json
import base64
import hashlib
import os
import secrets
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import (  # noqa: E402
    ApiError,
    DashboardAuth,
    ExperimentStore,
    KubectlCloudStore,
    compare,
    compare_cloud,
    match_cloud_records,
    normalize_supervisor_control,
    reconcile_experiment_status,
)


def write_experiment(
    root: Path,
    experiment_id: str,
    strategy: str,
    critical_round: float,
    critical_training: float,
    cpu_offload: bool,
) -> None:
    directory = root / experiment_id
    directory.mkdir()
    config = {
        "experiment_id": experiment_id,
        "run_id": 123,
        "status": "completed",
        "started_at": "2026-07-31T09:00:00",
        "model": "openlm-research/open_llama_3b_v2",
        "dataset": "vicgalle/alpaca-gpt4",
        "finetuning_type": "full",
        "distributed_strategy": strategy,
        "ddp_cpu_offload": cpu_offload,
        "rounds_this_attempt": 1,
        "run_config": {
            "model.name": "openlm-research/open_llama_3b_v2",
            "model.finetuning-type": "full",
            "strategy.min-fit-clients": 2,
            "train.seq-length": 512,
            "train.training-arguments.per-device-train-batch-size": 1,
            "train.training-arguments.gradient-accumulation-steps": 1,
            "train.training-arguments.max-steps": 10,
            "train.training-arguments.learning-rate": 1e-6,
            "train.training-arguments.gradient-checkpointing": True,
            "dataset.max-train-samples": 3000,
            "train.full-update-compression": "topk-int8",
            "train.full-update-topk-ratio": 0.001,
            "train.full-local-initialization": True,
            "train.evaluate-after-fit": False,
            "train.distributed-strategy": strategy,
            "train.ddp-cpu-offload": cpu_offload,
        },
    }
    summary = {
        "status": "completed",
        "duration_seconds": critical_round + 100,
        "completed_global_round": 1,
        "aggregated_client_train_metrics": {
            "1": {
                "critical_path_client_round_seconds": critical_round,
                "critical_path_client_training_seconds": critical_training,
                "client_round_seconds": critical_round - 30,
                "client_training_seconds": critical_training - 20,
                "full_update_compression_seconds": 88.0,
                "train_loss": 1.34,
                "distributed_world_size": 3,
                "optimizer_steps": 10,
            }
        },
    }
    (directory / "experiment_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "experiment_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    quality_offset = 0.0 if strategy == "fsdp" else 0.01
    evaluation = {
        "metadata": {"selected_samples": 100},
        "results": [
            {
                "label": experiment_id,
                "metrics": {
                    "assistant_only": {
                        "loss": 1.20 + quality_offset,
                        "ppl": 3.32 + quality_offset,
                        "evaluated_samples": 100,
                    },
                    "rouge_l": {"f1": 0.25 - quality_offset},
                    "bertscore": {"f1": 0.82 - quality_offset},
                    "generation_quality": {
                        "exact_match": 0.01,
                        "empty_prediction_rate": 0.0,
                        "average_generated_tokens": 96,
                    },
                },
            }
        ],
    }
    (directory / "evaluation_summary.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )


class ExperimentStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        write_experiment(self.root, "fsdp-formal", "fsdp", 858.4886, 572.94, False)
        write_experiment(self.root, "ddp-formal", "ddp", 912.7026, 649.34, True)
        self.store = ExperimentStore(self.root)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_lists_and_normalizes_experiments(self):
        experiments = self.store.list()
        self.assertEqual({item["id"] for item in experiments}, {"fsdp-formal", "ddp-formal"})
        fsdp = self.store.read("fsdp-formal")
        self.assertEqual(fsdp["strategy"], "FSDP")
        self.assertEqual(fsdp["run_id"], "123")
        self.assertEqual(fsdp["parameters"]["effective_client_batch"], 3)
        self.assertEqual(fsdp["metrics"]["critical_round_seconds"], 858.4886)
        self.assertEqual(fsdp["quality"]["validation_loss"], 1.2)
        self.assertEqual(fsdp["quality"]["evaluated_samples"], 100)
        self.assertFalse(fsdp["quality_status"]["requested"])
        self.assertTrue(fsdp["quality_status"]["available"])

    def test_builds_chronological_logs_from_persisted_run_artifacts(self):
        directory = self.root / "fsdp-formal"
        (directory / "experiment_state.json").write_text(json.dumps({
            "status": "running", "updated_at": "2026-07-31T09:10:00",
            "latest_completed_round": 5, "total_rounds": 10,
        }), encoding="utf-8")
        (directory / "experiment_attempts.json").write_text(json.dumps([{
            "run_id": 123, "status": "running", "started_at": "2026-07-31T09:00:00",
            "resume_round": 0, "rounds_this_attempt": 2,
        }]), encoding="utf-8")
        (directory / "federated_metrics_round_1.json").write_text(json.dumps({
            "server_round": 1, "timestamp": "2026-07-31T09:09:00", "t_round_total_s": 42.5,
        }), encoding="utf-8")

        logs = self.store.logs("fsdp-formal")

        self.assertEqual(logs["experiment_id"], "fsdp-formal")
        self.assertEqual(logs["status"], "running")
        round_event = next(event for event in logs["events"] if event["level"] == "round")
        self.assertEqual(round_event["details"]["round"], 1)
        self.assertEqual(round_event["details"]["duration_seconds"], 42.5)
        state_event = next(event for event in logs["events"] if event["title"] == "当前状态：运行中")
        self.assertIn("已完成至第 5 轮，共 10 轮", state_event["message"])

    def test_cloud_logs_read_structured_artifacts_from_center_pvc(self):
        cloud_store = KubectlCloudStore()
        record = {
            "config": {
                "run_id": 456,
                "started_at": "2026-08-04T06:29:02",
                "distributed_strategy": "fsdp",
                "rounds_this_attempt": 10,
            },
            "state": {
                "run_id": 456,
                "status": "running",
                "updated_at": "2026-08-04T07:00:00",
                "completed_global_round": 1,
                "total_rounds": 10,
            },
            "summary": {},
            "attempts": [],
            "rounds": [{
                "artifact": "federated_metrics_round_1.json",
                "metrics": {
                    "server_round": 1,
                    "timestamp": "2026-08-04T06:59:00",
                    "t_round_total_s": 1800,
                },
            }],
        }
        with (
            mock.patch.object(
                cloud_store,
                "availability",
                return_value={"available": True, "message": "ok"},
            ),
            mock.patch.object(
                cloud_store, "_kubectl_json", return_value=record
            ) as kubectl_json,
            mock.patch.object(
                cloud_store, "supervisor_snapshot", return_value={}
            ),
        ):
            logs = cloud_store.logs("fsdp-cloud-10r")

        self.assertEqual(logs["status"], "running")
        self.assertEqual(logs["experiment_id"], "fsdp-cloud-10r")
        round_event = next(event for event in logs["events"] if event["level"] == "round")
        self.assertEqual(round_event["details"]["duration_seconds"], 1800)
        target, script = kubectl_json.call_args.args
        self.assertEqual(target, cloud_store.center)
        self.assertIn('experiment_id = "fsdp-cloud-10r"', script)

    def test_cloud_logs_reject_invalid_experiment_id(self):
        with self.assertRaises(ApiError):
            KubectlCloudStore().logs("../outside")

    def test_cloud_load_keeps_center_results_when_one_tke_is_unavailable(self):
        cloud_store = KubectlCloudStore()
        center_records = {
            "fedscale-quick": {
                "config": {
                    "experiment_id": "fedscale-quick",
                    "distributed_strategy": "fedscale",
                    "model": "Qwen/Qwen2.5-7B",
                },
                "summary": {},
                "state": {"status": "running"},
                "evaluation": {},
                "federated_rounds": [],
                "federated_timings": [],
            }
        }
        with (
            mock.patch.object(
                cloud_store,
                "availability",
                return_value={"available": True, "message": "ok"},
            ),
            mock.patch.object(cloud_store, "_flower_run_statuses", return_value={}),
            mock.patch.object(
                cloud_store,
                "_kubectl_json",
                side_effect=[center_records, [], ApiError(HTTPStatus.BAD_GATEWAY, "读取 TKE-B 失败")],
            ),
        ):
            experiments = cloud_store.list(force=True)

        self.assertEqual([item["id"] for item in experiments], ["fedscale-quick"])
        detail = cloud_store.read("fedscale-quick")
        self.assertEqual(detail["cloud"]["read_errors"], {"TKE-B": "读取 TKE-B 失败"})

    def test_cloud_exec_retries_transient_api_error(self):
        cloud_store = KubectlCloudStore()
        cloud_store.exec_retries = 2
        with (
            mock.patch.object(
                cloud_store,
                "_kubectl_json_once",
                side_effect=[ApiError(HTTPStatus.BAD_GATEWAY, "connection reset"), {"ok": True}],
            ) as read_once,
            mock.patch("server.time.sleep") as sleep,
        ):
            self.assertEqual(cloud_store._kubectl_json(cloud_store.clients[1], "print('{}')"), {"ok": True})

        self.assertEqual(read_once.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_validates_supervisor_control_actions_and_bounds(self):
        control = normalize_supervisor_control(
            {"generation": 4, "desired_state": "running", "poll_seconds": 120},
            {
                "action": "pause", "poll_seconds": 30, "stall_seconds": 3600,
                "max_restarts": 2, "strategy": "ddp", "rounds": 20,
                "model": "Qwen/Qwen2.5-7B",
                "dataset": "HuggingFaceH4/ultrachat_200k",
                "finetuning_type": "full",
            },
        )
        self.assertEqual(control["desired_state"], "paused")
        self.assertEqual(control["generation"], 5)
        self.assertEqual(control["poll_seconds"], 30)
        self.assertEqual(control["strategy"], "ddp")
        self.assertEqual(control["rounds"], 20)
        self.assertEqual(control["model"], "Qwen/Qwen2.5-7B")
        self.assertEqual(control["dataset"], "HuggingFaceH4/ultrachat_200k")
        self.assertEqual(control["finetuning_type"], "full")
        with self.assertRaises(ApiError):
            normalize_supervisor_control({}, {"action": "configure", "poll_seconds": 1})
        with self.assertRaises(ApiError):
            normalize_supervisor_control({}, {"action": "configure", "strategy": "ray"})
        with self.assertRaises(ApiError):
            normalize_supervisor_control({}, {"action": "configure", "model": "bad model"})

    def test_supervisor_snapshot_uses_job_state_when_status_file_is_empty(self):
        cloud_store = KubectlCloudStore()
        with (
            mock.patch.object(
                cloud_store,
                "availability",
                return_value={"available": True, "message": "ok"},
            ),
            mock.patch.object(
                cloud_store,
                "_kubectl_json",
                return_value={"control": {"desired_state": "running"}, "status": {}, "events": []},
            ),
            mock.patch.object(
                cloud_store,
                "_latest_supervisor_job",
                return_value={
                    "job_name": "benchmark-matrix-supervisor-abc",
                    "job_phase": "succeeded",
                    "phase": "completed",
                    "message": "监督器 Job 已完成；如需继续调度，请点击“启动监督器”。",
                    "updated_at": "2026-08-21T08:00:00+00:00",
                },
            ),
        ):
            snapshot = cloud_store.supervisor_snapshot()

        self.assertEqual(snapshot["status"]["phase"], "completed")
        self.assertEqual(snapshot["status"]["job_name"], "benchmark-matrix-supervisor-abc")
        self.assertEqual(snapshot["job"]["job_phase"], "succeeded")

    def test_dashboard_basic_auth_accepts_only_matching_pbkdf2_credentials(self):
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 310_000)
        encode = lambda value: base64.urlsafe_b64encode(value).decode().rstrip("=")
        password_hash = f"pbkdf2_sha256$310000${encode(salt)}${encode(digest)}"
        with mock.patch.dict(os.environ, {
            "DASHBOARD_AUTH_USERNAME": "operator",
            "DASHBOARD_AUTH_PASSWORD_HASH": password_hash,
        }, clear=False):
            auth = DashboardAuth()
            good = "Basic " + base64.b64encode(b"operator:secret").decode()
            bad = "Basic " + base64.b64encode(b"operator:nope").decode()
            self.assertTrue(auth.enabled)
            self.assertTrue(auth.accepts(good))
            self.assertFalse(auth.accepts(bad))

    def test_comparison_selects_faster_critical_path(self):
        result = compare(self.store.read("fsdp-formal"), self.store.read("ddp-formal"))
        self.assertEqual(result["comparison_scope"], "formal")
        self.assertEqual(result["primary"]["winner"], "left")
        self.assertAlmostEqual(result["primary"]["delta"], -54.214, places=3)
        self.assertAlmostEqual(result["primary"]["percent"], 5.940, places=2)
        quality_loss = next(
            row
            for row in result["quality_metrics"]
            if row["key"] == "validation_loss"
        )
        self.assertEqual(quality_loss["winner"], "left")

        strategy = next(row for row in result["parameters"] if row["key"] == "strategy")
        self.assertFalse(strategy["same"])
        self.assertTrue(strategy["expected_difference"])

    def test_compares_persisted_partial_results_without_formal_summary(self):
        left = self.store.read("fsdp-formal")
        right = self.store.read("ddp-formal")
        for detail, strategy in ((left, "FSDP"), (right, "DDP")):
            detail["status"] = "running" if strategy == "FSDP" else "failed"
            detail["has_summary"] = False
            detail["raw"]["summary"] = {}
            detail["raw"]["state"] = {
                "status": detail["status"],
                "latest_completed_round": 3,
                "total_rounds": 10,
            }
            detail["metrics"]["completed_round"] = 3
            detail["raw"]["federated_rounds"] = [{
                "server_round": 3,
                "federated_cycle_s": 900.0,
                "server_post_aggregation_s": 40.0,
            }]

        result = compare(left, right)

        self.assertEqual(result["comparison_scope"], "partial")
        self.assertEqual(result["timing"]["left"][0]["round"], 3)
        self.assertEqual(result["timing"]["right"][0]["round"], 3)

    def test_rejects_path_traversal(self):
        with self.assertRaises(ApiError):
            self.store.read("../outside")

    def test_ignores_directories_without_experiment_files(self):
        (self.root / "lost+found").mkdir()
        (self.root / "empty").mkdir()
        self.assertEqual(len(self.store.list()), 2)

    def test_matches_and_compares_cloud_client_metrics(self):
        fsdp = self.store.read("fsdp-formal")
        records = [
            {
                "site": "TKE-A",
                "job_id": "train-round-1-0-100",
                "metrics": {
                    "timestamp": "2026-07-31T09:05:00",
                    "distributed_strategy": "fsdp",
                    "finetuning_type": "full",
                    "gradient_checkpointing": True,
                    "ddp_cpu_offload": False,
                    "optimizer_steps": 10,
                    "world_size": 3,
                    "training_only_s": 572.94,
                    "train_loss": 1.43,
                },
                "detailed": {
                    "training": {
                        "avg_step_time_ms": 45474.81,
                        "avg_forward_ms": 13000,
                        "avg_backward_ms": 27000,
                        "avg_comm_ms": 4000,
                        "avg_optimizer_ms": 474.81,
                        "avg_nccl_comm_ms": 12.5,
                        "total_nccl_bytes": 123456,
                        "throughput_tokens_per_s": 8.94,
                    },
                    "resources": {
                        "gpu_memory_peak_mb": 22100,
                        "gpu_utilization_avg_pct": 86.5,
                        "cpu_memory_peak_mb": 1845,
                        "network_total_bytes": 987654,
                    },
                    "federated": {
                        "state_export_type": "full_state_dict",
                        "full_state_export_s": 33.2,
                        "state_serialization_s": 2.1,
                        "state_bytes": 6853025483,
                        "t_total_round_s": 857.7,
                        "t_full_update_compression_s": 96.17,
                    },
                },
            },
            {
                "site": "TKE-B",
                "job_id": "train-round-1-1-101",
                "metrics": {
                    "timestamp": "2026-07-31T09:06:00",
                    "distributed_strategy": "fsdp",
                    "finetuning_type": "full",
                    "gradient_checkpointing": True,
                    "ddp_cpu_offload": False,
                    "optimizer_steps": 10,
                    "world_size": 3,
                    "training_only_s": 476.95,
                    "train_loss": 1.25,
                },
                    "detailed": {
                        "training": {"avg_step_time_ms": 35668.98, "throughput_tokens_per_s": 10.73},
                        "resources": {"gpu_memory_peak_mb": 22099, "cpu_memory_peak_mb": 2424},
                        "federated": {
                            "t_total_round_s": 689.38,
                            "t_full_update_compression_s": 81.24,
                        },
                    },
            },
        ]
        fsdp["finished_at"] = "2026-07-31T09:10:00"
        fsdp["cloud"] = {"clients": match_cloud_records(fsdp, records)}
        fsdp["raw"]["federated_rounds"] = [{
            "server_round": 1,
            "timestamp": "2026-07-31T09:08:00",
            "federated_cycle_s": 982.49,
            "server_fedavg_aggregation_s": 2.31,
            "server_post_aggregation_s": 38.82,
            "checkpoint_save_s": 5.42,
            "checkpoint_bytes": 6853025483,
        }]
        self.assertEqual(len(fsdp["cloud"]["clients"]), 2)

        ddp = self.store.read("ddp-formal")
        ddp["cloud"] = {
            "clients": [
                {
                    "site": "TKE-A", "round": 1, "training_seconds": 649.34,
                    "client_round_seconds": 800.0,
                    "compression_seconds": 70.0,
                },
                {
                    "site": "TKE-B", "round": 1, "training_seconds": 434.9,
                    "client_round_seconds": 700.0,
                    "compression_seconds": 60.0,
                },
            ]
        }
        ddp["raw"]["federated_rounds"] = [{
            "server_round": 1,
            "federated_cycle_s": 1000.0,
            "server_post_aggregation_s": 40.0,
        }]
        result = compare_cloud(fsdp, ddp)
        self.assertEqual(result["source"], "cloud")
        self.assertEqual(result["primary"]["winner"], "left")
        self.assertAlmostEqual(result["primary"]["delta"], -76.4)
        aggregation = result["aggregation"]
        self.assertEqual(aggregation["left"]["observed_round"], 1)
        self.assertEqual(aggregation["left"]["cluster_count"], 2)
        communication = next(
            row for row in aggregation["performance_metrics"]
            if row["key"] == "average_communication_ms"
        )
        self.assertEqual(communication["left"], 4000)
        gpu_utilization = next(
            row for row in aggregation["resource_metrics"]
            if row["key"] == "gpu_utilization_avg_pct"
        )
        self.assertEqual(gpu_utilization["left"], 86.5)
        self.assertFalse(aggregation["left"]["cross_centre_update"]["implemented"])
        network = next(
            row for row in aggregation["network_metrics"]
            if row["key"] == "network_traffic_bytes"
        )
        self.assertEqual(network["left"], 987654)
        state_export = next(
            row for row in aggregation["state_export_metrics"]
            if row["key"] == "full_state_export_seconds"
        )
        self.assertEqual(state_export["left"], 33.2)
        timing = result["timing"]["left"]
        self.assertEqual(len(timing), 1)
        self.assertEqual(timing[0]["round"], 1)
        self.assertEqual(timing[0]["tke_a_training_seconds"], 572.94)
        self.assertEqual(timing[0]["tke_b_training_seconds"], 476.95)
        self.assertEqual(timing[0]["federated_cycle_seconds"], 982.49)
        self.assertAlmostEqual(timing[0]["server_fedavg_aggregation_seconds"], 2.31)
        self.assertAlmostEqual(timing[0]["checkpoint_save_seconds"], 5.42)
        self.assertAlmostEqual(timing[0]["checkpoint_bytes"], 6853025483)
        self.assertAlmostEqual(timing[0]["checkpoint_interval_seconds"], 1021.31)

    def test_reconciles_authoritative_flower_status(self):
        detail = self.store.read("fsdp-formal")
        detail["status"] = "running"
        resolved = reconcile_experiment_status(
            detail,
            {"status": "finished:stopped", "finished-at": "2026-07-31 09:08:00Z"},
        )
        self.assertEqual(resolved["status"], "stopped")
        self.assertEqual(resolved["status_source"], "flower-run")
        self.assertEqual(resolved["recorded_status"], "running")

    def test_marks_old_running_result_as_stale(self):
        detail = self.store.read("fsdp-formal")
        detail["status"] = "running"
        resolved = reconcile_experiment_status(
            detail, None, now=datetime(2026, 8, 3, tzinfo=timezone.utc)
        )
        self.assertEqual(resolved["status"], "stale")
        self.assertEqual(resolved["status_source"], "stale-timeout")


if __name__ == "__main__":
    unittest.main()
