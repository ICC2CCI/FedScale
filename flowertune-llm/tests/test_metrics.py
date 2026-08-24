import sys
import time
import types
import unittest
from pathlib import Path

# The repository test image does not install the training-only Transformers
# dependency. Stub only TrainerCallback so the pure timing logic is testable.
transformers_stub = types.ModuleType("transformers")
transformers_stub.TrainerCallback = type("TrainerCallback", (), {})
sys.modules.setdefault("transformers", transformers_stub)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowertune_llm.metrics import ResourceMonitor, StepMetricsCallback  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_forward_and_collective_breakdowns_are_aggregated(self):
        callback = StepMetricsCallback(seq_length=4)
        callback._train_start_time = time.perf_counter() - 1.0

        class Args:
            per_device_train_batch_size = 1
            gradient_accumulation_steps = 1

        class State:
            global_step = 1
            log_history = [{"loss": 1.2}]
            learning_rate = 1e-6

        callback.on_step_begin(Args(), State(), None)
        callback.record_forward(12.5)
        callback.record_collective(3.0, 100, "all_gather")
        callback.record_collective(4.0, 200, "reduce_scatter")
        callback._optimizer_time_ms = 2.0
        callback.on_step_end(Args(), State(), None)
        callback.on_step_begin(Args(), State(), None)
        callback.record_forward(10.0)
        callback.record_collective(5.0, 300, "all_gather")
        callback._optimizer_time_ms = 2.0
        callback.on_step_end(Args(), State(), None)

        summary = callback.get_summary()
        self.assertAlmostEqual(summary["avg_forward_ms"], 11.25)
        self.assertAlmostEqual(summary["avg_all_gather_ms"], 4.0)
        self.assertEqual(summary["total_all_gather_bytes"], 400)
        self.assertEqual(summary["reduce_scatter_collective_calls"], 1)
        self.assertGreater(summary["throughput_tokens_per_s"], 0)

    def test_unavailable_gpu_utilization_is_null_not_zero(self):
        monitor = ResourceMonitor()
        monitor._samples = [{
            "gpu_memory_allocated_mb": 100,
            "gpu_utilization_pct": -1,
            "cpu_utilization_pct": 20,
            "cpu_memory_rss_mb": 200,
        }]
        summary = monitor.get_summary()
        self.assertIsNone(summary["gpu_utilization_avg_pct"])


if __name__ == "__main__":
    unittest.main()
