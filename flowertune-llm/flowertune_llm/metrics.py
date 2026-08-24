"""Metrics collection for distributed training performance evaluation."""

import json
import os
import subprocess
import threading
import time
from datetime import datetime

import psutil
import torch
import torch.distributed as dist
from transformers import TrainerCallback


class StepMetricsCallback(TrainerCallback):
    """TrainerCallback that records per-step distributed-training timing.

    Measures forward, backward, DDP All-Reduce communication, and optimizer
    step times using torch.cuda.Event for GPU-accurate profiling.
    """

    def __init__(self, seq_length=512):
        self.step_records = []
        self._step_start = None
        self._forward_end = None
        self._forward_time_ms = 0.0
        self._comm_time_ms = 0.0
        self._optimizer_time_ms = 0.0
        self._optimizer_start = None
        self._comm_bytes = 0
        self._comm_calls = 0
        self._collective_totals_all = {}
        self._original_all_reduce = None
        self._original_collectives = {}
        self._train_start_time = None
        self._total_tokens = 0
        # Use the effective tokenized sequence length. Looking this up through
        # ``model.config.max_position_embeddings`` is incorrect for FSDP: it
        # reports the model capacity (2048 for OpenLLaMA), while this run is
        # actually tokenized to ``train.seq-length`` (512).
        self._seq_length = int(seq_length)

    def _hook_all_reduce(self):
        """Monkey-patch NCCL collective entry points for timing and traffic."""
        collective_names = (
            "all_reduce",
            "all_gather",
            "all_gather_into_tensor",
            "reduce_scatter",
            "reduce_scatter_tensor",
            "all_to_all_single",
        )

        def tensor_bytes(values):
            total = 0
            for value in values:
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
                elif isinstance(value, (list, tuple)):
                    total += tensor_bytes(value)
            return total

        for name in collective_names:
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._original_collectives[name] = original
            if name == "all_reduce":
                self._original_all_reduce = original

            def timed_collective(*args, _name=name, _original=original, **kwargs):
                byte_count = tensor_bytes(args) + tensor_bytes(tuple(kwargs.values()))
                started = time.perf_counter()
                result = _original(*args, **kwargs)
                if hasattr(result, "wait"):
                    result.wait()
                elapsed_ms = (time.perf_counter() - started) * 1000
                self.record_collective(elapsed_ms, byte_count, collective=_name)
                return result

            setattr(dist, name, timed_collective)

    def _unhook_all_reduce(self):
        for name, original in self._original_collectives.items():
            setattr(dist, name, original)
        self._original_collectives.clear()
        self._original_all_reduce = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._train_start_time = time.perf_counter()
        self._hook_all_reduce()

    def on_train_end(self, args, state, control, **kwargs):
        self._unhook_all_reduce()

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()
        self._comm_time_ms = 0.0
        self._optimizer_time_ms = 0.0
        self._optimizer_start = None
        self._forward_time_ms = 0.0
        self._forward_end = None
        self._comm_bytes = 0
        self._comm_calls = 0
        self._collective_totals = {}

    def record_collective(self, elapsed_ms, byte_count, collective="unknown"):
        """Record one NCCL collective observed by a DDP/FSDP hook."""
        elapsed = max(0.0, float(elapsed_ms))
        bytes_count = max(0, int(byte_count))
        self._comm_time_ms += elapsed
        self._comm_bytes += bytes_count
        self._comm_calls += 1
        for totals_by_name in (self._collective_totals, self._collective_totals_all):
            totals = totals_by_name.setdefault(
                collective,
                {"time_ms": 0.0, "bytes": 0, "calls": 0},
            )
            totals["time_ms"] += elapsed
            totals["bytes"] += bytes_count
            totals["calls"] += 1

    def ddp_comm_hook(self, state, bucket):
        """Time DDP gradient all-reduce buckets while preserving averaging."""
        started = time.perf_counter()
        tensor = bucket.buffer()
        byte_count = tensor.numel() * tensor.element_size()
        all_reduce = self._original_all_reduce or dist.all_reduce
        work = all_reduce(tensor, async_op=True)
        future = work.get_future()

        def finish(completed):
            self.record_collective(
                (time.perf_counter() - started) * 1000,
                byte_count,
                collective="all_reduce",
            )
            value = completed.value()
            reduced = value[0] if isinstance(value, (list, tuple)) else value
            return reduced.div_(dist.get_world_size())

        return future.then(finish)

    def record_forward(self, elapsed_ms):
        """Accumulate model forward/loss computation time for this step."""
        self._forward_time_ms += max(0.0, float(elapsed_ms))
        self._forward_end = time.perf_counter()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Start a per-step optimizer timer after gradient synchronization."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._optimizer_start = time.perf_counter()

    def on_optimizer_step(self, args, state, control, **kwargs):
        """Finish the optimizer timer exposed by Transformers' callback loop."""
        if self._optimizer_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._optimizer_time_ms = (time.perf_counter() - self._optimizer_start) * 1000
        self._optimizer_start = None

    def on_pre_backward(self, args, state, control, **kwargs):
        # Older Transformers callback paths expose this hook. Newer TRL paths
        # are timed explicitly around Trainer.compute_loss in distributed_trainer.
        # Keep this as a fallback without double-counting an explicit measurement.
        if self._forward_time_ms <= 0.0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._forward_end = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_end = time.perf_counter()

        total_ms = (step_end - self._step_start) * 1000
        forward_ms = self._forward_time_ms
        if forward_ms <= 0.0 and self._forward_end:
            forward_ms = (self._forward_end - self._step_start) * 1000
        backward_ms = total_ms - forward_ms - self._comm_time_ms - self._optimizer_time_ms
        if backward_ms < 0:
            backward_ms = 0.0

        step_num = state.global_step
        loss = None
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                loss = last_log["loss"]

        lr = None
        if hasattr(state, "learning_rate") and state.learning_rate is not None:
            lr = state.learning_rate
        elif state.log_history:
            last_log = state.log_history[-1]
            if "learning_rate" in last_log:
                lr = last_log["learning_rate"]

        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._total_tokens += batch_size * self._seq_length

        record = {
            "step": step_num,
            "forward_ms": round(forward_ms, 2),
            "backward_ms": round(backward_ms, 2),
            "comm_ms": round(self._comm_time_ms, 2),
            "nccl_bytes": self._comm_bytes,
            "nccl_collective_calls": self._comm_calls,
            "optimizer_ms": round(self._optimizer_time_ms, 2),
            "total_ms": round(total_ms, 2),
            "loss": loss,
            "lr": lr,
        }
        self.step_records.append(record)

    def get_summary(self):
        """Return aggregated training performance summary."""
        total_train_time_s = 0.0
        if self._train_start_time:
            total_train_time_s = time.perf_counter() - self._train_start_time

        steps = self.step_records
        avg_step_ms = sum(s["total_ms"] for s in steps) / len(steps) if steps else 0.0
        avg_forward_ms = sum(s["forward_ms"] for s in steps) / len(steps) if steps else 0.0
        avg_backward_ms = sum(s["backward_ms"] for s in steps) / len(steps) if steps else 0.0
        avg_comm_ms = sum(s["comm_ms"] for s in steps) / len(steps) if steps else 0.0
        avg_optimizer_ms = sum(s["optimizer_ms"] for s in steps) / len(steps) if steps else 0.0
        avg_nccl_bytes = sum(s["nccl_bytes"] for s in steps) / len(steps) if steps else 0.0
        total_nccl_bytes = sum(s["nccl_bytes"] for s in steps)

        throughput = self._total_tokens / total_train_time_s if total_train_time_s > 0 else 0.0

        summary = {
            "steps": steps,
            "total_train_time_s": round(total_train_time_s, 2),
            "avg_step_time_ms": round(avg_step_ms, 2),
            "avg_forward_ms": round(avg_forward_ms, 2),
            "avg_backward_ms": round(avg_backward_ms, 2),
            "avg_comm_ms": round(avg_comm_ms, 2),
            "avg_optimizer_ms": round(avg_optimizer_ms, 2),
            "avg_nccl_comm_ms": round(avg_comm_ms, 2),
            "avg_nccl_bytes": round(avg_nccl_bytes, 2),
            "total_nccl_bytes": total_nccl_bytes,
            "nccl_collective_calls": sum(s["nccl_collective_calls"] for s in steps),
            "throughput_tokens_per_s": round(throughput, 2),
            "total_tokens": self._total_tokens,
        }
        for collective, totals in self._collective_totals_all.items():
            safe_name = collective.replace("-", "_")
            summary[f"avg_{safe_name}_ms"] = round(
                totals["time_ms"] / len(steps), 2
            ) if steps else 0.0
            summary[f"total_{safe_name}_ms"] = round(totals["time_ms"], 2)
            summary[f"total_{safe_name}_bytes"] = totals["bytes"]
            summary[f"{safe_name}_collective_calls"] = totals["calls"]
        return summary


class ResourceMonitor:
    """Background thread that samples GPU and CPU resource usage."""

    def __init__(self, interval=1.0, device_index=0):
        self.interval = interval
        self.device_index = int(device_index)
        self._thread = None
        self._stop_event = threading.Event()
        self._samples = []
        self._gpu_utilization_available = None
        self._network_start = None
        self._network_end = None

    @staticmethod
    def _network_bytes():
        """Read container interface counters; loopback is not training traffic."""
        received = transmitted = 0
        try:
            with open("/proc/net/dev", encoding="utf-8") as handle:
                for line in handle:
                    if ":" not in line:
                        continue
                    interface, values = line.split(":", 1)
                    if interface.strip() == "lo":
                        continue
                    fields = values.split()
                    if len(fields) >= 9:
                        received += int(fields[0])
                        transmitted += int(fields[8])
        except (OSError, ValueError):
            return None
        return {"rx_bytes": received, "tx_bytes": transmitted}

    def _gpu_utilization(self):
        """Read utilization through NVML or nvidia-smi without fake zeroes."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            value = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            pynvml.nvmlShutdown()
            return value
        except (ImportError, AttributeError, RuntimeError, OSError):
            pass
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi", "--id", str(self.device_index),
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            value = float(completed.stdout.strip().splitlines()[0])
            return max(0.0, min(100.0, value))
        except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError, OSError):
            return None

    def _sample(self):
        sample = {}
        if torch.cuda.is_available():
            sample["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            sample["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            utilization = self._gpu_utilization()
            sample["gpu_utilization_pct"] = -1 if utilization is None else utilization
            if utilization is None:
                if self._gpu_utilization_available is not False:
                    print("Resource monitor: GPU utilization unavailable; continuing with null telemetry")
                self._gpu_utilization_available = False
            else:
                self._gpu_utilization_available = True
        sample["cpu_utilization_pct"] = psutil.cpu_percent(interval=None)
        sample["cpu_memory_rss_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        sample["timestamp"] = time.time()
        self._samples.append(sample)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._sample()
            except Exception as exc:
                # Monitoring must never terminate model training. Keep later
                # samples possible in case a transient NVML/psutil error clears.
                print(f"Resource monitor: sample skipped ({type(exc).__name__}: {exc})")
            self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self._samples = []
        self._network_start = self._network_bytes()
        self._network_end = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._network_end = self._network_bytes()
        return self.get_summary()

    def get_summary(self):
        if not self._samples:
            return self._with_network({
                "gpu_memory_peak_mb": 0.0,
                "gpu_utilization_avg_pct": None,
                "cpu_utilization_avg_pct": 0.0,
                "cpu_memory_peak_mb": 0.0,
            })

        gpu_mem_peak = max(s.get("gpu_memory_allocated_mb", 0) for s in self._samples)
        gpu_utils = [s.get("gpu_utilization_pct", 0) for s in self._samples if s.get("gpu_utilization_pct", -1) >= 0]
        gpu_util_avg = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
        cpu_utils = [s.get("cpu_utilization_pct", 0) for s in self._samples]
        cpu_util_avg = sum(cpu_utils) / len(cpu_utils) if cpu_utils else 0.0
        cpu_mem_peak = max(s.get("cpu_memory_rss_mb", 0) for s in self._samples)

        if torch.cuda.is_available():
            gpu_mem_peak = max(gpu_mem_peak, torch.cuda.max_memory_allocated() / (1024 * 1024))

        return self._with_network({
            "gpu_memory_peak_mb": round(gpu_mem_peak, 2),
            "gpu_utilization_avg_pct": (
                round(gpu_util_avg, 2) if gpu_util_avg is not None else None
            ),
            "cpu_utilization_avg_pct": round(cpu_util_avg, 2),
            "cpu_memory_peak_mb": round(cpu_mem_peak, 2),
        })

    def _with_network(self, summary):
        start = self._network_start
        end = self._network_end or self._network_bytes()
        if start and end:
            summary.update({
                "network_rx_bytes": max(0, end["rx_bytes"] - start["rx_bytes"]),
                "network_tx_bytes": max(0, end["tx_bytes"] - start["tx_bytes"]),
            })
            summary["network_total_bytes"] = (
                summary["network_rx_bytes"] + summary["network_tx_bytes"]
            )
        else:
            summary.update({
                "network_rx_bytes": None,
                "network_tx_bytes": None,
                "network_total_bytes": None,
            })
        return summary


def save_metrics_detailed(output_dir, training_summary, resource_summary,
                          validation_metrics=None, downstream_metrics=None,
                          federated_metrics=None):
    """Save all metrics to a single JSON file."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "training": training_summary,
        "resources": resource_summary,
    }
    if validation_metrics:
        metrics["validation"] = validation_metrics
    if downstream_metrics:
        metrics["downstream"] = downstream_metrics
    if federated_metrics:
        metrics["federated"] = federated_metrics

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics_detailed.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved detailed metrics to {path}")
    return path
