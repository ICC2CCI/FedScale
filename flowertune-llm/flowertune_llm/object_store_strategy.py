"""Flower control-plane strategy for full-model object-store federation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from typing import Iterable, Protocol

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from flowertune_llm.model_storage import ModelArtifact, ModelStorage, ObjectStoreError


CONTROL_ARRAY_KEY = "__object_store_control__"
ARTIFACT_RECORD_KEY = "artifact"


class AggregationLauncher(Protocol):
    """Run one aggregation independently of the Flower ServerApp process."""

    def run(
        self, *, experiment_id: str, round_number: int, updates: list[ModelArtifact]
    ) -> ModelArtifact: ...


class KubernetesAggregationLauncher:
    """Create and wait for the one-shot, memory-sized aggregation Job."""

    def __init__(self, storage: ModelStorage) -> None:
        self.storage = storage
        self.namespace = os.environ.get("OBJECT_STORE_AGGREGATION_NAMESPACE", os.environ.get("POD_NAMESPACE", "default"))
        self.image = os.environ.get("OBJECT_STORE_AGGREGATION_IMAGE", "").strip()
        self.secret_name = os.environ.get("OBJECT_STORE_AGGREGATION_SECRET", "object-store-aggregator").strip()
        self.ca_secret_name = os.environ.get("OBJECT_STORE_AGGREGATION_CA_SECRET", "object-store-direct-ca").strip()
        self.timeout = int(os.environ.get("OBJECT_STORE_AGGREGATION_TIMEOUT_SECONDS", "43200"))
        if not self.image:
            raise ObjectStoreError("OBJECT_STORE_AGGREGATION_IMAGE is required in object-store mode")

    @staticmethod
    def _job_name(experiment_id: str, round_number: int) -> str:
        digest = hashlib.sha256(f"{experiment_id}:{round_number}".encode()).hexdigest()[:10]
        return f"object-store-aggregate-r{round_number}-{digest}"

    def run(self, *, experiment_id: str, round_number: int, updates: list[ModelArtifact]) -> ModelArtifact:
        existing = self.storage.get_round_state(experiment_id, round_number)
        if existing and existing.get("status") == "GLOBAL_READY":
            return ModelArtifact(**existing["global"])

        try:
            from kubernetes import client as k8s_client
            from kubernetes import config as k8s_config
        except ImportError as exc:  # pragma: no cover - runtime-only dependency
            raise ObjectStoreError("kubernetes client is required to launch aggregation") from exc
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            k8s_config.load_kube_config()
        batch = k8s_client.BatchV1Api()
        name = self._job_name(experiment_id, round_number)
        self.storage.put_round_state(
            experiment_id,
            round_number,
            {
                "status": "AGGREGATING",
                "experiment_id": experiment_id,
                "round": round_number,
                "clients": {update.role: asdict(update) for update in updates},
            },
        )
        updates_json = json.dumps([asdict(update) for update in updates], separators=(",", ":"))
        container = k8s_client.V1Container(
            name="aggregator",
            image=self.image,
            image_pull_policy="Always",
            command=["python", "-m", "flowertune_llm.aggregation"],
            args=[
                "--experiment-id", experiment_id,
                "--round", str(round_number),
                "--updates-json", updates_json,
                "--workdir", "/scratch",
            ],
            env_from=[k8s_client.V1EnvFromSource(secret_ref=k8s_client.V1SecretEnvSource(name=self.secret_name))],
            resources=k8s_client.V1ResourceRequirements(
                requests={
                    "cpu": os.environ.get("OBJECT_STORE_AGGREGATION_CPU_REQUEST", "4"),
                    "memory": os.environ.get("OBJECT_STORE_AGGREGATION_MEMORY_REQUEST", "8Gi"),
                    "ephemeral-storage": os.environ.get("OBJECT_STORE_AGGREGATION_EPHEMERAL_REQUEST", "20Gi"),
                },
                limits={
                    "cpu": os.environ.get("OBJECT_STORE_AGGREGATION_CPU_LIMIT", "8"),
                    "memory": os.environ.get("OBJECT_STORE_AGGREGATION_MEMORY_LIMIT", "26Gi"),
                    "ephemeral-storage": os.environ.get("OBJECT_STORE_AGGREGATION_EPHEMERAL_LIMIT", "40Gi"),
                },
            ),
            volume_mounts=[
                k8s_client.V1VolumeMount(name="scratch", mount_path="/scratch"),
                k8s_client.V1VolumeMount(
                    name="object-store-ca", mount_path="/var/run/object-store-ca", read_only=True
                ),
            ],
        )
        job = k8s_client.V1Job(
            metadata=k8s_client.V1ObjectMeta(name=name, labels={"app": "object-store-aggregation"}),
            spec=k8s_client.V1JobSpec(
                backoff_limit=0,
                ttl_seconds_after_finished=86400,
                template=k8s_client.V1PodTemplateSpec(
                    metadata=k8s_client.V1ObjectMeta(labels={"app": "object-store-aggregation"}),
                    spec=k8s_client.V1PodSpec(
                        restart_policy="Never",
                        containers=[container],
                        volumes=[
                            k8s_client.V1Volume(
                                name="scratch",
                                empty_dir=k8s_client.V1EmptyDirVolumeSource(size_limit="40Gi"),
                            ),
                            k8s_client.V1Volume(
                                name="object-store-ca",
                                secret=k8s_client.V1SecretVolumeSource(secret_name=self.ca_secret_name),
                            ),
                        ],
                    ),
                ),
            ),
        )
        try:
            batch.create_namespaced_job(self.namespace, job)
        except k8s_client.ApiException as exc:
            if exc.status != 409:
                raise ObjectStoreError(f"could not create aggregation Job {name}: {exc}") from exc
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            status = batch.read_namespaced_job_status(name, self.namespace).status
            if status.succeeded:
                state = self.storage.get_round_state(experiment_id, round_number)
                if not state or state.get("status") != "GLOBAL_READY":
                    raise ObjectStoreError("aggregation Job succeeded without publishing GLOBAL_READY state")
                return ModelArtifact(**state["global"])
            if any(c.type == "Failed" and c.status == "True" for c in (status.conditions or [])):
                # Keep S3 artifacts for diagnosis/retry, but remove the fixed
                # Job name so a resilient ServerApp attempt can create it again.
                batch.delete_namespaced_job(
                    name, self.namespace,
                    propagation_policy="Background",
                )
                raise ObjectStoreError(f"aggregation Job {name} failed; client artifacts were retained")
            time.sleep(10)
        batch.delete_namespaced_job(
            name, self.namespace, propagation_policy="Background"
        )
        raise ObjectStoreError(f"aggregation Job {name} timed out after {self.timeout}s")


class ObjectStoreFedAvg(FedAvg):
    """FedAvg whose model bytes never enter Flower's object transport."""

    def __init__(
        self,
        *args,
        storage: ModelStorage,
        launcher: AggregationLauncher,
        experiment_id: str,
        initial_global: ModelArtifact,
        expected_roles: tuple[str, ...] = ("client-a", "client-b"),
        resume_round: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.storage = storage
        self.launcher = launcher
        self.experiment_id = experiment_id
        self.current_global = initial_global
        self.expected_roles = expected_roles
        self.resume_round = int(resume_round)

    def configure_train(self, server_round, arrays, config, grid):
        config["server-round"] = int(server_round)
        config["full-update-transport"] = "object-store"
        config["global-model-uri"] = self.current_global.uri
        config["global-model-sha256"] = self.current_global.sha256
        config["global-model-size"] = self.current_global.size
        config["global-model-round"] = self.current_global.round
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        valid, failures = self._check_and_log_replies(replies, is_train=True, validate=False)
        if failures or len(valid) != len(self.expected_roles):
            raise ObjectStoreError(
                f"object-store aggregation expected {len(self.expected_roles)} replies, "
                f"received {len(valid)} valid and {len(failures)} failed replies"
            )
        updates: list[ModelArtifact] = []
        metric_contents = []
        for reply in valid:
            content = reply.content
            if ARTIFACT_RECORD_KEY not in content:
                raise ObjectStoreError("client reply omitted object-store artifact metadata")
            artifact_record = content[ARTIFACT_RECORD_KEY]
            if not isinstance(artifact_record, ConfigRecord):
                raise ObjectStoreError("client artifact metadata must be a ConfigRecord")
            if artifact_record.get("status") != "READY":
                raise ObjectStoreError(f"client reported {artifact_record.get('status', 'unknown')}")
            updates.append(ModelArtifact.from_config(dict(artifact_record)))
            metric_contents.append(content)
        roles = tuple(sorted(update.role for update in updates))
        if roles != tuple(sorted(self.expected_roles)):
            raise ObjectStoreError(f"unexpected client artifact roles: {roles}")
        absolute_round = self.resume_round + int(server_round)
        if any(update.round != absolute_round for update in updates):
            raise ObjectStoreError("client artifact round does not match Flower round")
        self.current_global = self.launcher.run(
            experiment_id=self.experiment_id, round_number=absolute_round, updates=updates
        )
        metrics = self.train_metrics_aggr_fn(metric_contents, self.weighted_by_key)
        metrics["object_store_global_bytes"] = self.current_global.size
        return ArrayRecord({CONTROL_ARRAY_KEY: torch.zeros(1, dtype=torch.uint8)}), metrics
