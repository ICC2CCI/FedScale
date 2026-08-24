---
name: flower-runtime-overrides
description: Apply runtime code/configuration changes and recover stalled Flower federated experiments in CrossCentreLLMPrivacyTrainingInference through the app bundle, shared output PVC, and Flower run-config overrides. Use for DDP/FSDP experiment changes, smoke tests, metric instrumentation, evaluation jobs, dashboard image publication, registry authentication, cluster updates, and authorized experiment stop/restart workflows.
---

# Flower Runtime Overrides

Use runtime overrides as the default deployment mechanism for the Flower
federated-learning experiment cluster. Keep the base training image unchanged
unless the requested change genuinely requires a missing dependency or base
runtime capability; in that exceptional case, explain the reason and ask before
building or pushing an image.

## Scope and boundaries

- Work in `flower-llm`, especially `flowertune-llm/` and its `ClientApp`.
- Treat the Flower app bundle, the client output PVC, and the run-config as the
  authoritative runtime override paths.
- Use `flower-llm/config-center`, `flower-llm/config-tke-a`, and
  `flower-llm/config-tke-b` only for the Flower federation.
- Never use these paths for `run-admit-svc`, `ccnets-admission`, gVisor, or
  NodePort `30203`; that service has its own kubeconfig and workflow in
  `AGENTS.md`.
- For every Kubernetes command, remove proxy variables only for that command:

  ```bash
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
    -u ALL_PROXY -u all_proxy kubectl ...
  ```

  Do not change global proxy settings or kubeconfigs.

## Standing authorization for experiment recovery

The user has granted standing authorization for this project to stop and
restart a Flower federated experiment when the user requests experiment
recovery (for example, “停止并重启实验”) or when the configured benchmark
supervisor detects a stalled Run. Do not ask for a second confirmation for the
same recovery workflow.

Apply this authorization only to the exact Flower Run and supervisor involved
in the requested experiment:

1. Read the current center-side Run list and the supervisor manifest/status to
   identify the active Run ID. Never stop an unrelated historical or manually
   submitted Run.
2. Stop the active Run through the center Flower control path, then verify its
   terminal state is `finished:stopped` and that no unintended Flower Run
   remains active.
3. Restart the benchmark supervisor with the repository deployment workflow.
   Preserve the results PVC, matrix manifest, experiment artifacts, and usable
   checkpoints. Respect the matrix policy: an errored combination is recorded
   and the next combination is submitted without waiting for another approval.
4. If evidence shows a TKE-A or TKE-B receive-path failure (for example, a
   SuperNode reports heartbeat-online but has no `pull-messages` activity),
   stop the affected Run first, then roll only the affected Flower
   `supernode-*` and `superexec-clientapp-*` Deployments. Wait for readiness,
   verify both required SuperNodes are online and receiving messages, and only
   then resubmit or continue the matrix. Before resubmission, enumerate
   per-round training Jobs in both client namespaces. If a Job is still active
   and is provably owned by the stopped Run, delete that exact Job (and its
   matching headless Service if present) to release its GPUs; never delete a
   Job based only on a similar name or a historical status.

Use exact resource names and namespaces. Do not delete PVCs, checkpoints,
ConfigMaps, unrelated Deployments, the dashboard, or the separate
`run-admit-svc` workflow. A status or diagnosis request without a recovery
request remains read-only.

## Runtime override workflow

1. Inspect the current repository, deployment manifests, image environment,
   PVC mounts, and existing override mechanism before editing.
2. Put code changes in the Flower app source. `flwr run` packages the app for
   the current run; do not rebuild the training image merely to ship Python
   application changes.
3. For files executed inside a Kubernetes training Job, add the file to
   `OVERRIDE_TRAINING_FILES` in
   `flowertune-llm/flowertune_llm/client_app.py` when appropriate. The ClientApp
   must copy the file into `/app/outputs/<job-name>/` before creating the Job,
   and the Job must execute that copied file rather than an unverified copy
   baked into the image.
4. For a new one-off evaluation or diagnostic entry point, copy the script to
   the per-job output directory and invoke it by its explicit PVC path. Keep
   the base image only for its already-installed dependencies.
5. For parameters, use the existing submission interface instead of editing
   the image or deployment:

   ```bash
   ./scripts/run-federated.sh \
     --strategy ddp --rounds 1 --experiment-id <unique-id> \
     --set dataset.max-train-samples=32 \
     --set train.training-arguments.max-steps=1 \
     --set train.seq-length=128
   ```

   Pass every override through `--set key=value` and preserve the exact
   effective run-config in the experiment artifacts.
6. Run syntax/unit checks locally before submission. For cluster changes,
   verify the effective source and configuration in the ClientApp logs and the
   created Job, not just in the local worktree.
7. Use a unique experiment ID for every smoke run. Preserve existing PVC data;
   do not delete Jobs, checkpoints, PVCs, or ConfigMaps as cleanup without
   explicit authorization.

## Registry authentication and dashboard publication

- Never write Docker passwords, API keys, bearer tokens, or the contents of a
  Docker auth file into this Skill, Git, a manifest, or command output. The
  login state is external state owned by the current Docker daemon/terminal.
- Reuse the current Docker login when the user has explicitly authorized an
  image push. Check only non-secret registry names or Docker's sanitized status;
  never print decoded auth values.
- For the experiment dashboard, use the existing Flower registry convention:
  `ccr.ccs.tencentyun.com/flwr_pcl/flower-experiment-dashboard:<release-tag>`.
  Keep the Flower training image separate; do not replace it merely to publish
  dashboard code. Use a new release tag for material UI/backend changes (the
  current release is `control-console-v3`) instead of reusing a mutable tag.
- Cluster nodes may require an image-pull Secret even when the local Docker
  push succeeded. Before creating one, obtain explicit user authorization to
  transfer the current Docker registry credential into the target namespace.
  Name the scoped Secret `ccr-registry-auth` and attach it only to the target
  dashboard Deployment through `imagePullSecrets`; do not attach it to Flower
  training workloads.
- If `dockerhub.kubekey.local` cannot be resolved by cluster nodes, use the
  existing reachable Tencent registry convention above. Do not modify global
  Docker daemon insecure-registry settings or restart the host Docker service
  to work around a pull failure.
- After publication, verify all of the following: image push digest, Deployment
  image reference, Pod `Ready`, `/api/health`, and any application-level
  dependency status. A successful image pull is not proof that an API key or
  metrics sink is authenticated.

### TKE experiment-dashboard release workflow

Use this workflow when the user asks to publish the current dashboard code to
the TKE-hosted console. It applies to the dashboard Deployment only; it does
not restart the active benchmark supervisor or any Flower training Job.

1. Run the dashboard tests and syntax checks before building:

   ```bash
   cd flower-llm
   python3 -m pytest -q experiment-dashboard/tests
   node --check experiment-dashboard/static/app.js
   python3 -m py_compile experiment-dashboard/server.py
   ```

2. Build and push from the `flower-llm` context. Choose a new release tag for
   each material dashboard change and update the TKE manifest before applying:

   ```bash
   docker build -f experiment-dashboard/Dockerfile \
     -t ccr.ccs.tencentyun.com/flwr_pcl/flower-experiment-dashboard:<release-tag> .
   docker push ccr.ccs.tencentyun.com/flwr_pcl/flower-experiment-dashboard:<release-tag>
   ```

3. For TKE-A, use `config-tke-a` and namespace `flower-supernode-a`. The
   `experiment-dashboard-auth`, `experiment-dashboard-kubeconfigs`, and
   `ccr-registry-auth` Secrets must exist in that namespace; a Secret in
   `flower-superlink` cannot be referenced across namespaces. Create or copy
   them only when the user has authorized that credential operation. Never
   print Secret data or Docker auth contents.

4. Apply only the dashboard manifest and wait for the rollout:

   ```bash
   env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
     -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a \
     apply -f flower-llm/configs/experiment-dashboard-tke-a.yaml
   env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
     -u ALL_PROXY -u all_proxy kubectl --kubeconfig flower-llm/config-tke-a \
     -n flower-supernode-a rollout status deployment/experiment-dashboard --timeout=180s
   ```

5. Verify the effective image digest, Pod readiness, unauthenticated probe,
   authenticated page, and cloud experiment API. A temporary port-forward is
   sufficient for the smoke test; prefer an unused local port such as `18080`
   when the user's port `8080` may already be occupied. Do not put a real
   password in this Skill or commit it to the repository.

If rollout fails, inspect Pod events and logs before changing the image or
deleting anything. Do not run `scripts/deploy-benchmark-matrix.sh` as part of a
dashboard release: that script replaces the supervisor Job and may interrupt
an active experiment.

## Verification requirements

Before declaring a runtime override effective, verify all applicable items:

- The submitted Flower app contains the changed file.
- The ClientApp log shows the expected per-job override path and configuration.
- The training/evaluation Job uses the expected command and sees the changed
  file on the shared output PVC.
- The result artifact contains the new metric or behavior.
- A checksum or exact file-content check confirms that the Job executed the
  intended runtime copy when the change is code-related.
- For multi-center runs, inspect TKE-A and TKE-B separately; do not infer one
  center's state from the other.

Use read-only `get`, `describe`, `logs`, and `exec` checks first. Keep cluster
operations scoped to the user-requested experiment. If a runtime override cannot
be applied because the base image lacks a required package, stop before pushing
an image and report the exact missing capability and destination registry.

## Timing and evaluation rules

When adding final-model evaluation:

- Run it after the final global aggregation, preferably as a separate client
  GPU Job using the final global state.
- Write quality results and evaluation elapsed time to a separate evaluation
  artifact such as `evaluation_summary.json`.
- Keep evaluation time out of training-step, client-round, federated-cycle,
  aggregation, checkpoint, and formal efficiency timings.
- Record both the independent evaluation duration and the training/federation
  duration so the end-to-end wall-clock cost remains observable.
- Distinguish local-in-round diagnostic evaluation from evaluation of the final
  aggregated global model; do not label the former as final-model quality.

## Preferred handoff

Report the runtime files changed, the effective override path, the experiment
ID, the verification commands/results, and any remaining limitation. State
explicitly when an image push was avoided and why.
