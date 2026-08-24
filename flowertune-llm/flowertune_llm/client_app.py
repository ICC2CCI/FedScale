"""flowertune-llm: A Flower / FlowerTune app with multi-node distributed training support."""

import os
import shutil
import warnings
import time
import json
import torch
from datetime import datetime
from kubernetes import client as k8s_client
from kubernetes import config
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import cosine_annealing, get_model, get_tokenizer
from flowertune_llm.model_state import (
    decode_topk_int8_delta,
    encode_topk_int8_delta,
    finetuning_type,
    get_federated_state_dict,
    is_topk_int8_delta,
)
from flowertune_llm.fedscale_state import (
    apply_fedscale_int8_delta,
    build_canonical_layout,
    encode_fedscale_int8_delta,
    encoded_nbytes,
    is_fedscale_int8_delta,
    mask_hash,
)
from flowertune_llm.model_storage import ModelArtifact, ModelStorage
from flowertune_llm.object_store_strategy import ARTIFACT_RECORD_KEY, CONTROL_ARRAY_KEY

# Paths to training overrides shipped with this app. At train() time we copy
# them to the shared outputs PVC so the train Job can overwrite the versions
# baked into TRAIN_IMAGE. FSDP-QLoRA needs both the launcher and the model
# loader override (for floating-point 4-bit quantization storage), and the
# matching metrics callback used by the launcher.
# This avoids rebuilding the Docker image when the registry is unreachable.
OVERRIDE_TRAINING_FILES = {
    name: os.path.join(os.path.dirname(__file__), name)
    for name in (
        "distributed_trainer.py",
        "train_models.py",
        "train_dataset.py",
        "metrics.py",
        "evaluator.py",
        "model_state.py",
        "cpu_offload_optimizer.py",
        "fedscale_state.py",
    )
}


def _fedscale_config(cfg, config_record):
    """Read and validate the server-signed public FedScale round plan."""
    block_size = int(getattr(cfg.train, "fedscale_block_size", 0) or 0)
    if block_size <= 0:
        raise ValueError("train.fedscale-block-size must be positive")
    layout_hash = str(config_record.get("fedscale-layout-hash", ""))
    mask_hash_value = str(config_record.get("fedscale-mask-hash", ""))
    raw_ids = str(config_record.get("fedscale-block-ids", "")).strip()
    if not layout_hash or not mask_hash_value or not raw_ids:
        raise ValueError("FedScale RoundPlan is missing layout, mask, or block ids")
    try:
        block_ids = tuple(int(value) for value in raw_ids.split(","))
    except ValueError as exc:
        raise ValueError("FedScale RoundPlan has invalid block ids") from exc
    return block_size, layout_hash, mask_hash_value, block_ids

# Configure HuggingFace cache to use PVC mount
# This ensures models are loaded from /app/.cache instead of downloading
os.environ["HF_HOME"] = "/app/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/app/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/.cache/huggingface"

# Use the PVC cache by default. Set TRAIN_HF_HUB_OFFLINE=0 explicitly when
# a controlled cache warm-up/download is intended.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = os.environ.get("TRAIN_HF_HUB_OFFLINE", "1")
os.environ["HF_DATASETS_OFFLINE"] = os.environ.get("TRAIN_HF_DATASETS_OFFLINE", "1")

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Flower ClientApp
app = ClientApp()


def launch_k8s_distributed_job(job_name, num_nodes, gpus_per_node, initial_weights_path, train_config):
    """
    Launch a multi-node distributed training job on Kubernetes.

    Args:
        job_name: Name of the Kubernetes Job
        num_nodes: Number of worker nodes
        gpus_per_node: GPUs per node
        initial_weights_path: Path to initial model weights (saved by Client)
        train_config: Training configuration dict

    Returns:
        bool: True if job completed successfully
    """
    try:
        # Load Kubernetes config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        batch_v1 = k8s_client.BatchV1Api()
        core_v1 = k8s_client.CoreV1Api()
        namespace = os.environ.get('POD_NAMESPACE', 'default')

        # Cluster-specific configuration from environment
        train_image = os.environ.get('TRAIN_IMAGE', 'ccr.ccs.tencentyun.com/flwr_pcl/flwr_client_train:latest')
        cache_pvc = os.environ.get('CACHE_PVC', 'flowertune-cache-pvc')
        output_pvc = os.environ.get('OUTPUT_PVC', 'flowertune-output-pvc')

        print(f"Job configuration:")
        print(f"  Namespace: {namespace}")
        print(f"  Train image: {train_image}")
        print(f"  Cache PVC: {cache_pvc}")
        print(f"  Output PVC: {output_pvc}")

        # Create headless service for worker communication
        service = k8s_client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=k8s_client.V1ObjectMeta(
                name=job_name,
                labels={"app": job_name}
            ),
            spec=k8s_client.V1ServiceSpec(
                cluster_ip="None",
                selector={"app": job_name},
                ports=[k8s_client.V1ServicePort(port=29500, target_port=29500)]
            )
        )

        try:
            core_v1.create_namespaced_service(namespace=namespace, body=service)
            print(f"Created headless service: {job_name}")
        except k8s_client.ApiException as e:
            if e.status != 409:
                raise

        # Create container spec
        override_commands = " && ".join(
            f"cp /app/outputs/{job_name}/{name} /app/flowertune_llm/{name}"
            for name in OVERRIDE_TRAINING_FILES
        )
        # Propagate communication tuning from the SuperExec deployment into
        # each ephemeral training pod.  Unset variables are intentionally not
        # added so distributed_trainer.py can retain safe defaults.
        tuning_env_names = (
            "NCCL_IB_DISABLE",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA",
            "NCCL_CROSS_NIC",
            "NCCL_SOCKET_NTHREADS",
            "NCCL_NSOCKS_PERTHREAD",
            "NCCL_DEBUG",
            "FSDP_SHARDING_STRATEGY",
            "FSDP_BACKWARD_PREFETCH",
            "FSDP_FORWARD_PREFETCH",
            "FSDP_LIMIT_ALL_GATHERS",
        )
        tuning_env = [
            k8s_client.V1EnvVar(name=name, value=os.environ[name])
            for name in tuning_env_names
            if os.environ.get(name) is not None
        ]
        if gpus_per_node > 1:
            trainer_command = (
                "torchrun "
                f"--nnodes={num_nodes} "
                f"--nproc-per-node={gpus_per_node} "
                "--node-rank=${NODE_RANK} "
                f"--master-addr={job_name}-0.{job_name}.{namespace}.svc.cluster.local "
                "--master-port=29500 "
                "-m flowertune_llm.distributed_trainer"
            )
        else:
            trainer_command = "python -m flowertune_llm.distributed_trainer"
        node_rank_env = []
        if gpus_per_node > 1:
            node_rank_env.append(
                k8s_client.V1EnvVar(
                    name="NODE_RANK",
                    value_from=k8s_client.V1EnvVarSource(
                        field_ref=k8s_client.V1ObjectFieldSelector(
                            field_path=(
                                "metadata.annotations['batch.kubernetes.io/"
                                "job-completion-index']"
                            )
                        )
                    ),
                )
            )

        container = k8s_client.V1Container(
            name="trainer",
            image=train_image,
            command=["/bin/sh", "-c"],
            args=[
                f"{override_commands} && {trainer_command}"
            ],
            env=[
                # Use service DNS for master address (resolves to Pod IPs)
                # For Indexed Job with subdomain: {job_name}-{index}.{job_name}.{namespace}.svc.cluster.local
                k8s_client.V1EnvVar(name="MASTER_ADDR", value=f"{job_name}-0.{job_name}.{namespace}.svc.cluster.local"),
                k8s_client.V1EnvVar(name="MASTER_PORT", value="29500"),
                k8s_client.V1EnvVar(name="NAMESPACE", value=namespace),  # Pass namespace to pods
                k8s_client.V1EnvVar(name="WORLD_SIZE", value=str(num_nodes * gpus_per_node)),
                k8s_client.V1EnvVar(name="NUM_NODES", value=str(num_nodes)),
                k8s_client.V1EnvVar(name="GPUS_PER_NODE", value=str(gpus_per_node)),
                k8s_client.V1EnvVar(name="MODEL_WEIGHTS_PATH", value=initial_weights_path),
                k8s_client.V1EnvVar(name="TRAIN_CONFIG", value=json.dumps(train_config)),
                k8s_client.V1EnvVar(name="JOB_NAME", value=job_name),
                # Add pod hostname for proper DNS resolution
                k8s_client.V1EnvVar(
                    name="POD_HOSTNAME",
                    value_from=k8s_client.V1EnvVarSource(
                        field_ref=k8s_client.V1ObjectFieldSelector(
                            field_path="metadata.name"
                        )
                    )
                ),
                # HuggingFace configuration - use the PVC cache by default.
                k8s_client.V1EnvVar(name="HF_HOME", value="/app/.cache/huggingface"),
                k8s_client.V1EnvVar(name="TRANSFORMERS_CACHE", value="/app/.cache/huggingface"),
                k8s_client.V1EnvVar(name="HUGGINGFACE_HUB_CACHE", value="/app/.cache/huggingface"),
                k8s_client.V1EnvVar(name="HF_ENDPOINT", value="https://hf-mirror.com"),
                k8s_client.V1EnvVar(
                    name="HF_HUB_OFFLINE",
                    value=os.environ.get("TRAIN_HF_HUB_OFFLINE", "1"),
                ),
                k8s_client.V1EnvVar(
                    name="HF_DATASETS_OFFLINE",
                    value=os.environ.get("TRAIN_HF_DATASETS_OFFLINE", "1"),
                ),
                k8s_client.V1EnvVar(name="HF_HUB_ENABLE_HF_TRANSFER", value="0"),  # Disable hf_transfer
                k8s_client.V1EnvVar(name="HF_HUB_DISABLE_SYMLINKS_WARNING", value="1"),
            ] + node_rank_env + tuning_env,
            resources=k8s_client.V1ResourceRequirements(
                limits={"nvidia.com/gpu": str(gpus_per_node)},
                requests={"nvidia.com/gpu": str(gpus_per_node)}
            ),
            volume_mounts=[
                k8s_client.V1VolumeMount(name="model-cache", mount_path="/app/.cache"),
                k8s_client.V1VolumeMount(name="outputs", mount_path="/app/outputs"),
                k8s_client.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
            ]
        )

        # Create pod template
        template = k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(
                labels={"app": job_name},
                annotations={
                    "kubectl.kubernetes.io/default-container": "trainer"
                }
            ),
            spec=k8s_client.V1PodSpec(
                containers=[container],
                subdomain=job_name,  # Required for Indexed Job DNS: {job}-{index}.{subdomain}
                volumes=[
                    k8s_client.V1Volume(
                        name="model-cache",
                        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=cache_pvc
                        )
                    ),
                    k8s_client.V1Volume(
                        name="outputs",
                        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=output_pvc
                        )
                    ),
                    k8s_client.V1Volume(
                        name="dshm",
                        # FSDP/NCCL and pinned input batches can briefly use
                        # substantially more shared memory at the first
                        # forward pass.  A too-small tmpfs manifests as a
                        # non-Python SIGBUS (exit code 135).
                        empty_dir=k8s_client.V1EmptyDirVolumeSource(
                            medium="Memory", size_limit="8Gi"
                        )
                    ),
                ],
                restart_policy="Never"
            )
        )

        # Create job spec
        job_spec = k8s_client.V1JobSpec(
            template=template,
            parallelism=num_nodes,
            completions=num_nodes,
            # A distributed rank cannot be retried independently: the other
            # ranks keep the old rendezvous/NCCL process group and then fail
            # with Broken pipe.  Mark the whole Job failed immediately; the
            # ClientApp retries it below with a new name and a fresh group.
            backoff_limit=0,
            completion_mode="Indexed",
            # Give Kubernetes a hard upper bound so a deadlocked NCCL job is
            # terminated instead of leaving all ranks alive indefinitely.
            active_deadline_seconds=int(
                os.environ.get("JOB_COMPLETION_TIMEOUT_SECONDS", "7200")
            ) + 600,
        )

        # Create job
        job = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(name=job_name),
            spec=job_spec
        )

        # Create the job
        api_response = batch_v1.create_namespaced_job(namespace=namespace, body=job)
        print(f"Created distributed training job: {api_response.metadata.name}")
        print(f"  Nodes: {num_nodes}, GPUs per node: {gpus_per_node}")
        print(f"  Total GPUs: {num_nodes * gpus_per_node}")

        # Note: No need to wait here. Training pods will handle DNS resolution themselves.
        # The distributed_trainer.py has built-in retry logic for DNS resolution.
        print(f"\nNote: Training pods will handle DNS resolution internally")

        # Wait for job completion
        return wait_for_job_completion(batch_v1, namespace, job_name)

    except Exception as e:
        print(f"Failed to launch distributed job: {e}")
        import traceback
        traceback.print_exc()
        return False


def wait_for_job_completion(batch_v1, namespace, job_name, timeout=7200):
    """
    Wait for Kubernetes Job to complete.

    Args:
        batch_v1: Kubernetes BatchV1Api client
        namespace: Kubernetes namespace
        job_name: Job name
        timeout: Timeout in seconds (default 2 hours)

    Returns:
        bool: True if job succeeded
    """
    timeout = int(os.environ.get("JOB_COMPLETION_TIMEOUT_SECONDS", str(timeout)))
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace=namespace)

            if job.status.succeeded == job.spec.completions:
                print(f"✓ Job {job_name} completed successfully")
                return True

            # A failed Pod is not terminal for an Indexed Job: Kubernetes can
            # retry that index and still complete all requested replicas.
            conditions = job.status.conditions or []
            is_terminal_failure = any(
                condition.type == "Failed" and condition.status == "True"
                for condition in conditions
            )
            if is_terminal_failure:
                _print_failed_job_logs(namespace, job_name)
                print(f"✗ Job {job_name} reached terminal failure")
                return False

            active = job.status.active or 0
            succeeded = job.status.succeeded or 0
            print(f"⏳ Job {job_name}: {active} active, {succeeded}/{job.spec.completions} completed")
            time.sleep(30)

        except Exception as e:
            print(f"Error checking job status: {e}")
            time.sleep(10)

    print(f"⏰ Timeout waiting for job {job_name}; cleaning up resources")
    cleanup_k8s_job(job_name)
    return False


def _print_failed_job_logs(namespace, job_name):
    """Print failed Job pod termination details before cleanup removes it."""
    try:
        core_v1 = k8s_client.CoreV1Api()
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"app={job_name}",
        ).items
        for pod in pods:
            print(
                f"--- failed Job pod {pod.metadata.name}: "
                f"phase={pod.status.phase} ---"
            )
            for container_status in pod.status.container_statuses or []:
                terminated = getattr(container_status.state, "terminated", None)
                if terminated is not None:
                    print(
                        f"Container {container_status.name} terminated: "
                        f"reason={terminated.reason}, exit_code={terminated.exit_code}, "
                        f"message={terminated.message}"
                    )
                try:
                    logs = core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=namespace,
                        container=container_status.name,
                        tail_lines=200,
                    )
                    print(
                        f"--- failed Job container log "
                        f"{pod.metadata.name}/{container_status.name} ---\n{logs}"
                    )
                except Exception as log_exc:
                    print(
                        f"Unable to read failed Job container log "
                        f"{pod.metadata.name}/{container_status.name}: {log_exc}"
                    )
    except Exception as exc:
        print(f"Unable to inspect failed Job pods: {exc}")


def launch_final_evaluation_job(job_name, eval_config):
    """Run final-global-model evaluation in one GPU Job on this client cluster."""
    try:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        batch_v1 = k8s_client.BatchV1Api()
        namespace = os.environ.get("POD_NAMESPACE", "default")
        train_image = os.environ.get(
            "TRAIN_IMAGE", "ccr.ccs.tencentyun.com/flwr_pcl/flwr_client_train:latest"
        )
        cache_pvc = os.environ.get("CACHE_PVC", "flowertune-cache-pvc")
        output_pvc = os.environ.get("OUTPUT_PVC", "flowertune-output-pvc")
        evaluator_script_path = eval_config["EVALUATOR_SCRIPT_PATH"]

        env = [
            k8s_client.V1EnvVar(name=name, value=str(value))
            for name, value in eval_config.items()
        ]
        env.extend(
            [
                k8s_client.V1EnvVar(name="HF_HOME", value="/app/.cache/huggingface"),
                k8s_client.V1EnvVar(name="TRANSFORMERS_CACHE", value="/app/.cache/huggingface"),
                k8s_client.V1EnvVar(name="HUGGINGFACE_HUB_CACHE", value="/app/.cache/huggingface"),
                k8s_client.V1EnvVar(name="HF_HUB_OFFLINE", value="1"),
                k8s_client.V1EnvVar(name="HF_DATASETS_OFFLINE", value="1"),
                k8s_client.V1EnvVar(name="TOKENIZERS_PARALLELISM", value="false"),
            ]
        )
        container = k8s_client.V1Container(
            name="final-evaluator",
            image=train_image,
            image_pull_policy="Always",
            # Execute the app-bundled runtime override from the shared PVC.
            # This keeps evaluation code independent of the base image tag.
            command=["python", evaluator_script_path],
            env=env,
            resources=k8s_client.V1ResourceRequirements(
                requests={"cpu": "4", "memory": "24Gi", "nvidia.com/gpu": "1"},
                limits={"cpu": "6", "memory": "32Gi", "nvidia.com/gpu": "1"},
            ),
            volume_mounts=[
                k8s_client.V1VolumeMount(
                    name="model-cache", mount_path="/app/.cache"
                ),
                k8s_client.V1VolumeMount(
                    name="outputs", mount_path="/app/outputs"
                ),
                k8s_client.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
            ],
        )
        template = k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(
                labels={"app": job_name, "scope": "final-model-evaluation"}
            ),
            spec=k8s_client.V1PodSpec(
                restart_policy="Never",
                node_selector={
                    "node.tke.cloud.tencent.com/accelerator-type": "gpu"
                },
                containers=[container],
                volumes=[
                    k8s_client.V1Volume(
                        name="model-cache",
                        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=cache_pvc
                        ),
                    ),
                    k8s_client.V1Volume(
                        name="outputs",
                        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                            claim_name=output_pvc
                        ),
                    ),
                    k8s_client.V1Volume(
                        name="dshm",
                        empty_dir=k8s_client.V1EmptyDirVolumeSource(
                            medium="Memory", size_limit="4Gi"
                        ),
                    ),
                ],
            ),
        )
        job = k8s_client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=k8s_client.V1ObjectMeta(name=job_name),
            spec=k8s_client.V1JobSpec(
                template=template,
                backoff_limit=0,
                active_deadline_seconds=int(
                    os.environ.get("FINAL_EVALUATION_TIMEOUT_SECONDS", "7200")
                ),
            ),
        )
        batch_v1.create_namespaced_job(namespace=namespace, body=job)
        print(f"Created final evaluation Job: {job_name}")
        return wait_for_job_completion(
            batch_v1,
            namespace,
            job_name,
            timeout=int(os.environ.get("FINAL_EVALUATION_TIMEOUT_SECONDS", "7200")),
        )
    except Exception as exc:
        print(f"Failed to launch final evaluation Job: {exc}")
        import traceback

        traceback.print_exc()
        return False


def collect_training_results(job_name):
    """
    Collect training results from completed distributed job.

    Args:
        job_name: Kubernetes Job name

    Returns:
        tuple: (model_weights, metrics)
    """
    try:
        output_dir = f"/app/outputs/{job_name}"

        # Load model weights
        weights_path = f"{output_dir}/model_weights.pt"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

        # Keep the large full-model checkpoint file-backed while ArrayRecord
        # chunks it, avoiding a needless second resident copy in ClientApp.
        model_weights = torch.load(
            weights_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        print(f"Loaded model weights from {weights_path}")

        # Load metrics
        metrics_path = f"{output_dir}/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {"train_loss": 0.0, "num_examples": 0}

        return model_weights, metrics

    except Exception as e:
        print(f"Failed to collect results: {e}")
        raise


def collect_training_metrics(job_name):
    """Read metrics without loading a multi-gigabyte full-model checkpoint."""
    output_dir = f"/app/outputs/{job_name}"
    weights_path = f"{output_dir}/model_weights.pt"
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
    metrics_path = f"{output_dir}/metrics.json"
    if not os.path.isfile(metrics_path):
        return weights_path, {"train_loss": 0.0, "num_examples": 0}
    with open(metrics_path, encoding="utf-8") as handle:
        return weights_path, json.load(handle)


def cleanup_k8s_job(job_name):
    """
    Clean up Kubernetes Job and associated Service after training completion.

    Args:
        job_name: Kubernetes Job name to clean up
    """
    try:
        # Load Kubernetes config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        batch_v1 = k8s_client.BatchV1Api()
        core_v1 = k8s_client.CoreV1Api()
        namespace = os.environ.get('POD_NAMESPACE', 'default')

        # Delete the Job
        try:
            batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=namespace,
                propagation_policy='Foreground'
            )
            print(f"✓ Deleted Job: {job_name}")
        except k8s_client.ApiException as e:
            if e.status != 404:
                print(f"Warning: Failed to delete Job {job_name}: {e}")

        # Delete the headless Service
        try:
            core_v1.delete_namespaced_service(
                name=job_name,
                namespace=namespace
            )
            print(f"✓ Deleted Service: {job_name}")
        except k8s_client.ApiException as e:
            if e.status != 404:
                print(f"Warning: Failed to delete Service {job_name}: {e}")

        print(f"Cleanup completed for job: {job_name}")

    except Exception as e:
        print(f"Warning: Cleanup failed for job {job_name}: {e}")


@app.train()
def train(msg: Message, context: Context):
    """
    Train the model using multi-node distributed training on Kubernetes.

    This function:
    1. Saves initial model weights to shared storage
    2. Launches a Kubernetes Job for distributed training
    3. Waits for job completion
    4. Collects and returns the trained model
    """
    # Parse configuration
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    distributed_strategy = str(cfg.train.distributed_strategy).lower()
    full_update_transport = str(
        getattr(cfg.train, "full_update_transport", "flower-rpc")
    ).lower()
    if distributed_strategy not in {"ddp", "fsdp"}:
        raise ValueError(
            "train.distributed-strategy must be either 'ddp' or 'fsdp', got "
            f"{distributed_strategy!r}"
        )
    if full_update_transport not in {"flower-rpc", "object-store"}:
        raise ValueError(
            "train.full-update-transport must be 'flower-rpc' or 'object-store'"
        )
    object_store_transport = full_update_transport == "object-store"
    if object_store_transport and (
        finetuning_type(cfg.model) != "full"
        or str(cfg.train.full_update_compression).lower() != "none"
    ):
        raise ValueError(
            "object-store transport requires full fine-tuning with "
            "train.full-update-compression='none'"
        )

    server_round = int(msg.content["config"]["server-round"])
    resume_round = int(msg.content["config"].get("resume-round", 0) or 0)
    absolute_server_round = resume_round + server_round
    total_rounds = resume_round + num_rounds

    # Get learning rate for the original global round.
    new_lr = cosine_annealing(
        absolute_server_round,
        total_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    # Configuration for distributed training
    job_name = f"train-round-{absolute_server_round}-{partition_id}-{int(time.time())}"
    num_nodes = int(os.environ.get("DIST_NUM_NODES", "2"))
    gpus_per_node = int(os.environ.get("DIST_GPUS_PER_NODE", "4"))

    print(f"\n{'='*60}")
    print(f"Flower Client {partition_id} - Starting Distributed Training")
    print(f"{'='*60}")
    print(f"Round: {server_round}")
    print(f"Resume Round: {resume_round}")
    print(f"Absolute Round: {absolute_server_round}")
    print(f"Distributed Job: {job_name}")
    print(f"Nodes: {num_nodes}, GPUs per node: {gpus_per_node}")
    print(f"Total GPUs: {num_nodes * gpus_per_node}")
    print(f"Distributed Strategy: {distributed_strategy.upper()}")
    print(f"Fine-tuning Type: {cfg.model.finetuning_type.upper()}")
    print(f"Quantization: {cfg.model.quantization}")
    print(f"DDP CPU Offload: {cfg.train.ddp_cpu_offload}")
    print(f"Learning Rate: {new_lr}")
    print(f"{'='*60}\n")

    t_round_start = time.perf_counter()

    try:
        # Step 1: Save initial model weights to shared storage
        print("Step 1: Saving initial model weights...")

        # Load tokenizer first to cache it in PVC (critical for training pods)
        print("\nLoading tokenizer to cache it in PVC...")
        tokenizer = get_tokenizer(cfg.model.name)
        print("✓ Tokenizer loaded and cached in PVC")

        save_dir = f"/app/outputs/{job_name}"
        os.makedirs(save_dir, exist_ok=True)
        initial_weights_path = f"{save_dir}/initial_weights.pt"
        if object_store_transport:
            global_artifact = ModelArtifact(
                uri=str(msg.content["config"]["global-model-uri"]),
                sha256=str(msg.content["config"]["global-model-sha256"]),
                size=int(msg.content["config"]["global-model-size"]),
                experiment_id=str(context.run_config["experiment-id"]),
                round=int(msg.content["config"]["global-model-round"]),
                role="global",
            )
            print(f"Downloading global model from {global_artifact.uri}")
            ModelStorage.from_env().download_model(global_artifact, initial_weights_path)
            print(f"Saved downloaded global model to {initial_weights_path}")
        else:
            # The ServerApp already sends the complete PEFT state dict. Persist it
            # directly instead of loading and quantizing the full base model once
            # more in every ClientApp before the GPU Job can start.
            received_state = msg.content["arrays"].to_torch_state_dict()
            if is_topk_int8_delta(received_state):
                # Multi-round sparse mode sends the current global state as a
                # sparse delta relative to the same preflight-verified base.
                print("Reconstructing dense global state from sparse INT8 delta")
                base_model = get_model(cfg.model)
                try:
                    base_state = get_federated_state_dict(base_model, cfg.model)
                finally:
                    del base_model
                received_state = decode_topk_int8_delta(base_state, received_state)
                del base_state
            elif is_fedscale_int8_delta(received_state):
                print("Reconstructing dense global state from FedScale public blocks")
                block_size, expected_layout_hash, _, _ = _fedscale_config(
                    cfg, msg.content["config"]
                )
                base_model = get_model(cfg.model)
                try:
                    base_state = get_federated_state_dict(base_model, cfg.model)
                    layout = build_canonical_layout(base_state, block_size)
                    if layout.layout_hash != expected_layout_hash:
                        raise ValueError("FedScale RoundPlan layout hash does not match local base")
                    received_state = apply_fedscale_int8_delta(
                        base_state, received_state, layout
                    )
                finally:
                    del base_model
                del base_state
            torch.save(received_state, initial_weights_path)
            print(f"Saved initial weights to {initial_weights_path}")

        # Copy FSDP-aware training files to the shared PVC so every rank starts
        # from exactly the same implementation.
        for name, source_path in OVERRIDE_TRAINING_FILES.items():
            override_dst = f"{save_dir}/{name}"
            shutil.copy2(source_path, override_dst)
            print(f"Saved override {name} to {override_dst}")

        # Step 2: Prepare training configuration
        train_args = cfg.train.training_arguments
        train_config = {
            "partition_id": partition_id,
            "num_partitions": num_partitions,
            "dataset_name": cfg.dataset.name,
            "max_train_samples": int(getattr(cfg.dataset, "max_train_samples", 0) or 0),
            "model_name": cfg.model.name,
            "learning_rate": new_lr,
            "batch_size": train_args.per_device_train_batch_size,
            "seq_length": cfg.train.seq_length,
            "output_dir": "/app/outputs",
            "distributed_strategy": distributed_strategy,
            "finetuning_type": str(cfg.model.finetuning_type).lower(),
            "quantization": int(cfg.model.quantization),
            "gradient_checkpointing": bool(cfg.model.gradient_checkpointing),
            "lora_r": int(cfg.model.lora.peft_lora_r),
            "lora_alpha": int(cfg.model.lora.peft_lora_alpha),
            "ddp_cpu_offload": bool(cfg.train.ddp_cpu_offload),
            "full_local_initialization": bool(
                cfg.train.full_local_initialization
            ),
            "full_update_compression": str(
                cfg.train.full_update_compression
            ).lower(),
            "full_update_topk_ratio": float(cfg.train.full_update_topk_ratio),
            "fedscale_block_size": int(
                getattr(cfg.train, "fedscale_block_size", 0) or 0
            ),
            "run_local_evaluation": cfg.train.evaluate_after_fit,
            "num_eval_samples": cfg.train.num_eval_samples,
            # Training control
            "num_train_epochs": train_args.num_train_epochs,
            "max_steps": train_args.max_steps,
            "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
            "logging_steps": train_args.logging_steps,
            "save_steps": train_args.save_steps,
            "save_total_limit": train_args.save_total_limit,
            "lr_scheduler_type": train_args.lr_scheduler_type,
        }

        # Step 2.5: Wait a bit for PVC to sync (CFS/NFS might have slight delay)
        print("\nStep 2.5: Waiting for PVC sync...")
        time.sleep(5)  # Wait 5 seconds for PVC to sync
        print("✓ PVC sync wait completed")

        # Step 3: Launch distributed training as a gang.  If any rank fails,
        # Kubernetes marks that attempt failed (backoff_limit=0) and we start
        # all ranks again under a new Job/Service name.  Replacing only one
        # rank cannot recover an existing FSDP/NCCL process group.
        max_job_attempts = max(
            1, int(os.environ.get("DISTRIBUTED_JOB_MAX_ATTEMPTS", "3"))
        )
        success = False
        base_job_name = job_name
        base_save_dir = save_dir
        base_initial_weights_path = initial_weights_path

        for attempt in range(1, max_job_attempts + 1):
            attempt_job_name = (
                base_job_name if attempt == 1 else f"{base_job_name}-a{attempt}"
            )
            attempt_save_dir = f"/app/outputs/{attempt_job_name}"
            attempt_initial_weights_path = (
                f"{attempt_save_dir}/initial_weights.pt"
            )

            if attempt > 1:
                os.makedirs(attempt_save_dir, exist_ok=True)
                shutil.copy2(
                    base_initial_weights_path, attempt_initial_weights_path
                )
                for name in OVERRIDE_TRAINING_FILES:
                    shutil.copy2(
                        f"{base_save_dir}/{name}",
                        f"{attempt_save_dir}/{name}",
                    )
                # Give foreground deletion of the failed gang time to release
                # GPUs and let DNS endpoints disappear before the next gang.
                time.sleep(15)

            job_name = attempt_job_name
            initial_weights_path = attempt_initial_weights_path
            print(
                f"\nStep 2: Launching distributed training job "
                f"(gang attempt {attempt}/{max_job_attempts})..."
            )
            success = launch_k8s_distributed_job(
                job_name=job_name,
                num_nodes=num_nodes,
                gpus_per_node=gpus_per_node,
                initial_weights_path=initial_weights_path,
                train_config=train_config,
            )
            if success:
                break

            print(
                f"Distributed gang attempt {attempt}/{max_job_attempts} "
                "failed; cleaning up all ranks."
            )
            cleanup_k8s_job(job_name)

        if not success:
            raise RuntimeError(
                f"Distributed training job failed after "
                f"{max_job_attempts} gang attempts"
            )

        # Step 4: Collect results (with federated timing)
        print("\nStep 3: Collecting training results...")
        t_collect_start = time.perf_counter()
        if object_store_transport:
            weights_path, metrics = collect_training_metrics(job_name)
            print(f"Retained model checkpoint file for object-store upload: {weights_path}")
        else:
            model_weights, metrics = collect_training_results(job_name)
        t_collect_results = time.perf_counter() - t_collect_start

        # Do not return a full 6.85GB FP16 state through the cross-cloud Flower
        # transport.  The distributed Job persisted the local pre-training
        # state; encode the actual optimizer delta only after the Job releases
        # all GPUs.  This preserves full-parameter local training while making
        # the federated update feasible on the measured WAN bandwidth.
        if object_store_transport:
            compression_seconds = 0.0
            encoded_bytes = os.path.getsize(weights_path)
        elif (
            finetuning_type(cfg.model) == "full"
            and str(cfg.train.full_update_compression).lower() == "topk-int8"
        ):
            initial_state_path = (
                f"/app/outputs/{job_name}/initial_model_state.pt"
            )
            if not os.path.isfile(initial_state_path):
                raise FileNotFoundError(
                    "Sparse full-update baseline was not exported by the "
                    f"distributed Job: {initial_state_path}"
                )
            initial_model_state = torch.load(
                initial_state_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            compression_start = time.perf_counter()
            model_weights = encode_topk_int8_delta(
                model_weights,
                initial_model_state,
                float(cfg.train.full_update_topk_ratio),
            )
            compression_seconds = time.perf_counter() - compression_start
            encoded_bytes = sum(
                value.numel() * value.element_size()
                for value in model_weights.values()
            )
            print(
                "Encoded Top-K INT8 full-model delta: "
                f"{encoded_bytes / 1024 / 1024:.2f} MiB, "
                f"ratio={cfg.train.full_update_topk_ratio}, "
                f"encode={compression_seconds:.2f}s"
            )
            del initial_model_state
        elif str(cfg.train.full_update_compression).lower() == "fedscale-int8":
            # Full-model FedScale exports a pre-wrap baseline. LoRA receives
            # the current compact adapter state directly from ServerApp, which
            # is the correct per-round delta baseline.
            initial_state_path = (
                f"/app/outputs/{job_name}/initial_model_state.pt"
                if finetuning_type(cfg.model) == "full"
                else initial_weights_path
            )
            if not os.path.isfile(initial_state_path):
                raise FileNotFoundError(
                    "FedScale baseline was not available for the distributed Job: "
                    f"{initial_state_path}"
                )
            initial_model_state = torch.load(
                initial_state_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
            compression_start = time.perf_counter()
            block_size, expected_layout_hash, expected_mask_hash, block_ids = (
                _fedscale_config(cfg, msg.content["config"])
            )
            layout = build_canonical_layout(initial_model_state, block_size)
            if layout.layout_hash != expected_layout_hash:
                raise ValueError("FedScale local layout hash differs from RoundPlan")
            if mask_hash(layout, block_ids) != expected_mask_hash:
                raise ValueError("FedScale local public mask differs from RoundPlan")
            model_weights = encode_fedscale_int8_delta(
                model_weights, initial_model_state, layout, block_ids
            )
            compression_seconds = time.perf_counter() - compression_start
            encoded_bytes = encoded_nbytes(model_weights)
            print(
                "Encoded FedScale public-block INT8 delta: "
                f"{encoded_bytes / 1024 / 1024:.2f} MiB, "
                f"blocks={len(block_ids)}, encode={compression_seconds:.2f}s"
            )
            del initial_model_state
        else:
            compression_seconds = 0.0
            encoded_bytes = sum(
                value.numel() * value.element_size()
                for value in model_weights.values()
                if hasattr(value, "numel")
            )

        print(f"\nTraining completed successfully!")
        print(f"  Loss: {metrics.get('train_loss', 'N/A')}")
        print(f"  Examples: {metrics.get('num_examples', 'N/A')}")
        print(f"  World Size: {metrics.get('world_size', 'N/A')}")

        # Load detailed metrics if available
        detailed_metrics_path = f"/app/outputs/{job_name}/metrics_detailed.json"
        detailed_metrics = None
        if os.path.exists(detailed_metrics_path):
            with open(detailed_metrics_path, 'r') as f:
                detailed_metrics = json.load(f)
            print(f"  Loaded detailed metrics from {detailed_metrics_path}")

        # Step 5: Clean up Kubernetes resources
        print("\nStep 4: Cleaning up Kubernetes resources...")
        cleanup_k8s_job(job_name)

        artifact = None
        if object_store_transport:
            role = os.environ.get("OBJECT_STORE_CLIENT_ROLE", "").strip()
            if role not in {"client-a", "client-b"}:
                raise ValueError(
                    "OBJECT_STORE_CLIENT_ROLE must be client-a or client-b in object-store mode"
                )
            upload_started = time.perf_counter()
            artifact = ModelStorage.from_env().upload_model(
                weights_path,
                experiment_id=str(context.run_config["experiment-id"]),
                round_number=absolute_server_round,
                role=role,
                num_examples=int(metrics.get("num_examples", 0)),
            )
            encoded_bytes = artifact.size
            print(
                f"Uploaded full model to {artifact.uri} "
                f"({artifact.size} bytes, {time.perf_counter() - upload_started:.2f}s)"
            )

        # Step 6: Measure federated timing
        t_total_round = time.perf_counter() - t_round_start

        # Load model delta export timing (re-measure from detailed metrics if available)
        # t_model_delta_export is measured during collect_training_results (weights loading)
        t_model_delta_export = t_collect_results

        federated_metrics = {
            "t_model_delta_export_s": round(t_model_delta_export, 4),
            "t_collect_results_s": round(t_collect_results, 4),
            "t_full_update_compression_s": round(compression_seconds, 4),
            "t_total_round_s": round(t_total_round, 4),
            "model_delta_bytes": encoded_bytes,
        }
        if detailed_metrics is not None:
            detailed_metrics.setdefault("federated", {}).update(federated_metrics)
            try:
                with open(detailed_metrics_path, "w") as handle:
                    json.dump(detailed_metrics, handle, indent=2)
            except OSError as exc:
                print(f"Warning: failed to enrich detailed metrics: {exc}")
        print(f"\nFederated timing: total_round={t_total_round:.2f}s, collect={t_collect_results:.2f}s")

        # Step 7: Return results to Flower Server. Object-store mode keeps the
        # Flower ArrayRecord to a one-byte control value.
        model_record = (
            ArrayRecord({CONTROL_ARRAY_KEY: torch.zeros(1, dtype=torch.uint8)})
            if object_store_transport
            else ArrayRecord(model_weights)
        )
        metric_record = MetricRecord({
            "train_loss": metrics.get("train_loss", 0.0),
            "num-examples": metrics.get("num_examples", 0),
            # Flower excludes the weighting key (num-examples) from the
            # aggregated MetricRecord. Keep explicit reportable copies of the
            # local workload and distributed execution shape.
            "dataset_train_samples_per_client": metrics.get("num_examples", 0),
            "estimated_sample_presentations_per_client": metrics.get(
                "estimated_sample_presentations", 0
            ),
            "optimizer_steps": metrics.get("optimizer_steps", 0),
            "distributed_world_size": metrics.get(
                "world_size", num_nodes * gpus_per_node
            ),
            "gang_attempt": attempt,
            "client_training_seconds": metrics.get("training_only_s", 0.0),
            "client_evaluation_seconds": metrics.get("evaluation_s", 0.0),
            "client_round_seconds": round(t_total_round, 4),
            "client_non_training_seconds": round(
                max(0.0, t_total_round - metrics.get("training_only_s", 0.0)),
                4,
            ),
            "full_update_compression_seconds": round(compression_seconds, 4),
            "model_delta_bytes": encoded_bytes,
            "model_delta_export_seconds": round(t_model_delta_export, 4),
        })

        content = RecordDict({"arrays": model_record, "metrics": metric_record})
        if artifact is not None:
            content[ARTIFACT_RECORD_KEY] = ConfigRecord(
                {"status": "READY", **artifact.to_config()}
            )
        return Message(content=content, reply_to=msg)

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)

        # Training Jobs are deliberately cleaned up after a failed gang, so
        # their Pod logs are no longer available when the ServerApp receives
        # the metrics-only error reply.  Persist the ClientApp-side exception
        # next to that attempt's shared-PVC inputs so a supervisor/operator can
        # distinguish OOM, rendezvous, and serialization failures.
        try:
            error_path = f"/app/outputs/{job_name}/clientapp_error.json"
            with open(error_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "partition_id": partition_id,
                        "job_name": job_name,
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": error_traceback,
                    },
                    handle,
                    indent=2,
                )
            print(f"Persisted ClientApp failure detail: {error_path}")
        except Exception as artifact_error:
            print(f"Warning: failed to persist ClientApp error artifact: {artifact_error}")

        # Always clean up failed Jobs as well. Otherwise deadlocked ranks and
        # headless Services can contaminate later rounds.
        try:
            print("\nAttempting to clean up resources after failure...")
            cleanup_k8s_job(job_name)
        except Exception as cleanup_error:
            print(f"Warning: Cleanup also failed: {cleanup_error}")

        # Object-store strategy needs an explicit small failure reply rather
        # than treating an absent multi-GB return as a successful completion.
        if "object_store_transport" in locals() and object_store_transport:
            return Message(
                content=RecordDict(
                    {
                        "arrays": ArrayRecord(
                            {CONTROL_ARRAY_KEY: torch.zeros(1, dtype=torch.uint8)}
                        ),
                        "metrics": MetricRecord(
                            {"train_loss": float("inf"), "num-examples": 0}
                        ),
                        ARTIFACT_RECORD_KEY: ConfigRecord({"status": "FAILED"}),
                    }
                ),
                reply_to=msg,
            )

        # Return error metrics
        metric_record = MetricRecord({
            "train_loss": float('inf'),
            "num-examples": 0,
        })
        content = RecordDict({"metrics": metric_record})
        return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the final aggregated global model on this client's held-out data."""
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    server_round = int(msg.content["config"].get("server-round", 0) or 0)
    job_name = f"final-eval-round-{server_round}-{partition_id}-{int(time.time())}"
    output_dir = f"/app/outputs/{job_name}"
    state_path = f"{output_dir}/global_state.pt"
    result_path = f"{output_dir}/evaluation.json"
    evaluator_script_path = f"{output_dir}/final_evaluator.py"

    try:
        os.makedirs(output_dir, exist_ok=True)
        source_evaluator = os.path.join(
            os.path.dirname(__file__), "final_evaluator.py"
        )
        if not os.path.isfile(source_evaluator):
            raise FileNotFoundError(
                f"Runtime final evaluator is missing from the Flower app bundle: "
                f"{source_evaluator}"
            )
        shutil.copy2(source_evaluator, evaluator_script_path)
        received_state = msg.content["arrays"].to_torch_state_dict()
        if is_topk_int8_delta(received_state):
            print("Reconstructing dense global state for final evaluation")
            base_model = get_model(cfg.model)
            try:
                base_state = get_federated_state_dict(base_model, cfg.model)
            finally:
                del base_model
            received_state = decode_topk_int8_delta(base_state, received_state)
            del base_state
        elif is_fedscale_int8_delta(received_state):
            print("Reconstructing dense FedScale global state for final evaluation")
            block_size, expected_layout_hash, _, _ = _fedscale_config(
                cfg, msg.content["config"]
            )
            base_model = get_model(cfg.model)
            try:
                base_state = get_federated_state_dict(base_model, cfg.model)
                layout = build_canonical_layout(base_state, block_size)
                if layout.layout_hash != expected_layout_hash:
                    raise ValueError("FedScale evaluation layout hash does not match local base")
                received_state = apply_fedscale_int8_delta(
                    base_state, received_state, layout
                )
            finally:
                del base_model
            del base_state
        torch.save(received_state, state_path)
        del received_state

        eval_config = {
            "MODEL_NAME": cfg.model.name,
            "FINETUNING_TYPE": str(cfg.model.finetuning_type).lower(),
            "QUANTIZATION": int(cfg.model.quantization),
            "LORA_R": int(cfg.model.lora.peft_lora_r),
            "LORA_ALPHA": int(cfg.model.lora.peft_lora_alpha),
            "DATASET_NAME": cfg.dataset.name,
            "PARTITION_ID": partition_id,
            "NUM_PARTITIONS": num_partitions,
            "MAX_TRAIN_SAMPLES": int(getattr(cfg.dataset, "max_train_samples", 0) or 0),
            "NUM_EVAL_SAMPLES": int(cfg.train.num_eval_samples),
            "EVAL_SPLIT_RATIO": 0.1,
            "MODEL_STATE_PATH": state_path,
            "EVALUATION_OUTPUT_PATH": result_path,
            "EVALUATOR_SCRIPT_PATH": evaluator_script_path,
        }
        succeeded = launch_final_evaluation_job(job_name, eval_config)
        if not succeeded or not os.path.exists(result_path):
            raise RuntimeError(
                f"Final evaluation Job did not publish its result: {job_name}"
            )

        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
        cleanup_k8s_job(job_name)
        validation = result.get("validation", {})
        downstream = result.get("downstream", {})
        evaluated_samples = int(
            downstream.get("num_eval_samples")
            or result.get("evaluated_samples")
            or 0
        )
        metric_record = MetricRecord(
            {
                "num-examples": max(evaluated_samples, 1),
                "evaluation_completed": 1.0,
                "evaluation_seconds": float(result.get("evaluation_seconds", 0.0)),
                "evaluated_samples": evaluated_samples,
                "val_loss": float(validation.get("val_loss", 0.0)),
                "perplexity": float(validation.get("perplexity", 0.0)),
                "accuracy": float(downstream.get("accuracy", 0.0)),
                "macro_f1": float(downstream.get("macro_f1", 0.0)),
                "exact_match": float(downstream.get("exact_match", 0.0)),
                "rouge_l": float(downstream.get("rouge_l", 0.0)),
                "bertscore_f1": float(downstream.get("bertscore_f1", 0.0)),
            }
        )
        print(f"Final evaluation completed: {result}")
        return Message(
            content=RecordDict({"metrics": metric_record}), reply_to=msg
        )
    except Exception as exc:
        print(f"\n✗ Final model evaluation failed: {exc}")
        import traceback

        traceback.print_exc()
        try:
            cleanup_k8s_job(job_name)
        except Exception as cleanup_error:
            print(f"Warning: final evaluation cleanup failed: {cleanup_error}")
        return Message(
            content=RecordDict(
                {
                    "metrics": MetricRecord(
                        {
                            "num-examples": 1,
                            "evaluation_completed": 0.0,
                            "evaluation_seconds": 0.0,
                        }
                    )
                }
            ),
            reply_to=msg,
        )
