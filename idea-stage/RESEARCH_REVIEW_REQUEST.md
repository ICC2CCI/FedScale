# RESEARCH_REVIEW_REQUEST — External Review Brief

**Absolute path of this file (the only artifact to hand to the reviewer):**
`C:\大模型安全\CCLPT\pure-github-repo\FedScale\idea-stage\RESEARCH_REVIEW_REQUEST.md`

**Companion artifacts in the same directory (reviewer may read them):**
- `CURRENT_METHOD.md` — the frozen statement of the method under review (evidence-bounded)
- `AI_DESIGN_RELATED_WORK.md` — literature positioning, per-paper relationship typing
- `CURRENT_VS_LITERATURE.md` — the positioning brief this review must adjudicate

---

## 0. Reviewer configuration (override — do not substitute)

```yaml
model: "gpt-5.6-sol"
config: {"model_reasoning_effort": "xhigh"}
```

This project pins `gpt-5.6-sol` at `xhigh` for **every** reviewer call. Do **not** use `gpt-6-astra`.
Do **not** use `ultra` or `max` effort. If `gpt-5.6-sol` is genuinely unavailable to the account,
stop and report `REVIEW_UNAVAILABLE` with the reason — do not silently climb to a different model.
Transport errors (timeout, rate limit, auth, network, parse) are retried at the *same* model and
effort; they are never grounds for downgrading.

---

## 1. What you are reviewing

A **written scheme only** — no code, no new experiments. The scheme is the FedScale
`deploy/central-server-minio` branch's documented method, as captured in these written sources:

- `docs/ai-design/_blockmask.txt` — "Public Block Mask" implementation spec v1
- `docs/ai-design/_secagg.txt` — "Windowed Secure Aggregation" implementation spec v2
- `docs/algorithm/2026-09-04-s3r12v3-block-uniform.md`
- `docs/algorithm/2026-09-10-dual-cluster-s3r12v3-fsdp-current-flow.md`
- `docs/algorithm/2026-09-15-secagg-quantization-precision-issue.md`
- `docs/algorithm/2026-09-17-secagg-quantization-optimization-survey.md`
- `docs/exec-plans/README.md`
- `docs/exec-plans/completed/2026-09-04-s3r12v3-block-uniform.md`
- `docs/exec-plans/completed/2026-09-10-dual-cluster-s3r12v3-fsdp.md`
- `docs/exec-plans/completed/2026-09-11-dual-cluster-to-production.md`
- `docs/exec-plans/active/2026-09-18-secagg-time-efficiency.md`

All are relative to the repository root `C:\大模型安全\CCLPT\pure-github-repo\FedScale`.

### 1.1 Hard scope constraints (violating these invalidates the review)

1. **Do not infer the method from code.** `experiments/*.py`, server/client sources, YAML and tests
   are out of scope. In particular `block_vote`, `block_topk`, Flower/TKE, and the old per-window PUT
   path are **not** part of the method and must not be treated as such.
2. **Do not treat undocumented experiment runs as evidence.** Undocumented run directories
   (especially any `202609181406`) must not be used.
3. **Do not invent novelty.** If the literature already covers a point, say so plainly.
4. **No new experiments, no code changes.** You are adjudicating an existing written artifact set.
5. **Do not re-run the literature search.** `AI_DESIGN_RELATED_WORK.md` already contains the
   verified citations; critique its positioning rather than rebuilding it.

### 1.2 Evidence-boundary caveats you must respect

- Run IDs referenced in the docs (`20260914-final-clean`, `202609162025`, `202609180941`,
  `202609181151`, `202609171809`, `202609171717`, `202609101345`) **exist only as citations inside
  the documents**. There are no timestamped run directories on this machine. They are therefore
  **not independently verifiable**. Cite them as the documents do; do not demand their raw files,
  and do not treat their absence as a blocker.
- The active plan states "20-round algorithm verification in progress" → this is **UNKNOWN**;
  do not assume a result.

---

## 2. The method under review, in four layers

(Full detail and per-item evidence grading: `CURRENT_METHOD.md`.)

**Layer 1 — Block selection ("Hierarchical Public Permutation-and-Rotation Mask").**
A public, deterministic, replayable selection of which canonical blocks (~1 MB flat-view slices) a
client uploads this round. Per-round payload ≈ ρ = slots/H of the block set; H consecutive
*successful* rounds (a "Mask Epoch") form a Mask epoch in which every rotatable block is selected
**exactly once** (permutation positions `p mod H == slot`, without replacement). Failed rounds do
not consume a slot. Mask is derived from `epoch_seed = SHA256(domain ‖ job_id ‖ layout_version ‖
mask_epoch ‖ epoch_anchor_manifest_hash)`, layer seeds `HKDF-SHA256(epoch_seed, salt=layout_hash,
info=... ‖ layer_id)`, and a Fisher–Yates shuffle over a ChaCha20 stream. Clients independently
recompute the mask and check `mask_hash` before entering SecAgg.

**Layer 2 — Security ("FedScale Windowed SecAgg v2").**
X25519 ephemeral DH → pairwise masks (all-to-all, sign by canonical ICC id order) + per-attempt
self mask (`self_master`), all domain-separated by `(session_id, attempt_id, cohort_hash,
mask_hash, window_id, window_layout_hash, peer pair, icc_id)`. The spec additionally defines
t_rec-of-n Shamir sharing of two recovery secrets (pairwise-mask recovery secret and `self_master`),
survivor-set `U*` freezing after deadline, and a "no double reconstruction" invariant
(for any `k ∈ U*`, the server must not obtain shares sufficient for both secrets).

**Layer 3 — Numerics.** FP32 delta (no FP16 round-trip) + block memory (decay 0.9) + quant residual
(decay 1.0) → pad to power of two → sign flip D → FWHT → INT16 stochastic rounding with a single
global scale derived from `global_amax` (1 float per client per round; server takes `max_k` × 1.05)
→ pairwise + self mask mod 2^16 → server unmasks, dequantizes, inverse-FWHT, FedAvg.

**Layer 4 — Transport/execution.** Model updates over MinIO object storage; small control messages
over HTTP; one masked INT16 blob (SAW1, all windows) per client per round; notify handler is O(1)
and does not fetch the object; server finalize unmasks windows in parallel, marks `done` as soon as
`global_delta` is written, and PUTs the full `global_state` asynchronously.

### 2.1 Verified evidence (what the documents actually claim)

| Claim | Run | Numbers |
|---|---|---|
| Block-uniform scheduling, single machine, 2 clients, H=5 ≈ 20% | A-V3 §8 | R20 eval 1.0514 vs full-upload S2 1.0091; upload 18.5–21.5%, range 3.0%; each epoch exactly 100% coverage |
| SecAgg accuracy vs fp16 baseline, 20 rounds | `202609180941` vs `20260914-final-clean` | R20 eval **0.985** vs **0.987**; `train≈38s`, round ≈148s |
| Pre-Hadamard per-window-SecAgg | `202609162025` | R20 eval 1.328 |
| Time optimizations, 5 rounds | `202609181151` | R1–R5 eval **identical** to `202609180941` (R5 = 1.363804); `upload_blocks_MiB` unchanged (R1 = 139.4); round 148s → 110–134s; `wait_agg≈10s` |
| Dual-cluster link, 2 clients, 20 rounds, H=5 | `202609101345` | test-bed link verified |

Setting for all SecAgg runs: **Qwen2.5-0.5B, 2 clients, IID 50/50 medical flashcards, single seed,
`coverage_h=10` (≈10% upload), 20 rounds.**

### 2.2 Not verified — spec-only or unknown

- Shamir threshold sharing, both dropout-recovery paths, survivor-set `U*` freezing, "no double
  reconstruction", `attempt_id` fresh-mask retry semantics, overflow budget check,
  and all five dropout fault-injection tests of the SecAgg spec — **spec text only, zero
  experimental record.**
- All of Layer 1's *publicity/determinism/cross-language-test-vector* properties — spec only.
  Note the spec (ChaCha20 + HKDF + canonical encoding) and the algorithm doc (Python `random` +
  two-level SHA256) **disagree on the PRG and seed derivation**. Whether this was reconciled is
  UNKNOWN.
- Scalability of the time optimizations as window count N grows ("wall clock tracks bandwidth/
  compute, not RTT×N") — listed as an acceptance principle in the active plan, **no measurement
  recorded.** All timing numbers come from ≈269 windows at 0.5B.
- Any generalization beyond one model / two clients / IID data / one seed.

### 2.3 Explicitly not part of the method

`block_vote`, `block_topk`, Flower/TKE, the retired per-window PUT + hex JSON upload path,
`docs/exec-plans/completed/2026-09-04-fedrolex-partial-training.md` and
`docs/exec-plans/completed/2026-09-04-s3r12v2-block-permutation.md` (historical controls only).

---

## 3. The literature positioning to adjudicate

(Full detail: `AI_DESIGN_RELATED_WORK.md`; condensed verdicts: `CURRENT_VS_LITERATURE.md`.)

The position taken by the preceding analysis — **which you are asked to confirm, refute, or
sharpen** — is:

- **Layer 1** is largely covered by the partial-training family. FedRolex (NeurIPS 2022,
  arXiv:2212.01548) already rolls sub-model extraction to cover the whole model evenly; FedNILO's
  freeze-index schedule and Barbieri et al.'s coordinator-constrained layer selection are also
  public and deterministic. The remaining distinctive claims are **block-granular uniform bandwidth**
  and **full-model training with partial upload** (the inverse of FedRolex's tradeoff), plus an
  *unrealized* cross-implementation replayability claim.
- **Layer 2** is a faithful re-derivation of Bonawitz et al., CCS 2017 (pairwise + self masks,
  Shamir sharing of both `sk_i` and `b_i`, and the "either/or, not both" reconstruction constraint).
  No new cryptography. Its all-to-all baseline is on the weak side of the known design space
  (SecAgg+ O(N log N), LightSecAgg one-shot survivor reconstruction).
- **Layer 3** is essentially Bonawitz et al., Asilomar 2019 (arXiv:1912.00131): R = HD rotation,
  uniform quantization, mod-k SecAgg. The error-feedback split comes from the EF literature
  (1901.09847 / 2106.05203). **`global_amax` is argued to be a privacy *regression* relative to the
  2019 autotuning**, which infers bin size from the *aggregate* via a wrapped-normal fit and needs
  no per-client scalar.
- **Layer 4** is systems engineering.
- **SESA (IEEE ISIT 2024)** is flagged as an unaddressed threat: parameter-wise SA over
  heterogeneous submodels can leak the *indices* of updated parameters. The current scheme's public,
  cohort-uniform mask may be a good answer, but this argument is **not made anywhere in the
  documents**.
- Cross-silo federated LLM fine-tuning is actively contested by 2025–2026 LoRA+HE/FE systems
  (SecLoRA, SHE-LoRA, FedShield-LLM, FLAGuard), and DiLoCo-style methods reduce communication
  *frequency* by ~500× — a different axis that must not be conflated with this scheme's per-round
  payload reduction.

---

## 4. What you are asked to deliver

Answer all of the following. Where you disagree with §3, say so explicitly and give the reason.

1. **Strengths relative to prior work.** What, if anything, does this scheme do that existing work
   does not? Be specific about which claim survives.
2. **Weaknesses and evidence gaps.** Rank by the probability a competent reviewer raises them.
   Include the `global_amax` privacy question and the spec-vs-implementation PRG divergence.
3. **Which claims are publishable as stated.** Quote the strongest defensible phrasing for each.
4. **Which claims must be downgraded** to "engineering implementation" or "combination of existing
   methods."
5. **Which claims cannot be made at all** with the current evidence.
6. **The most accurate research position.** Algorithm contribution, systems contribution,
   engineering integration, or a limited-evidence combination? Commit to one and justify it.
7. **Adjudicate the five positioning questions** carried over from `CURRENT_VS_LITERATURE.md` §2:
   - What does the public block mask genuinely add over FedRolex / federated dropout?
   - What remains of Hadamard+INT16 beyond the existing Hadamard-SecAgg work?
   - How strong a SecAgg security claim can 2 clients with no real dropout support?
   - Are object storage / HTTP control plane / single blob / parallel unmask algorithm or systems?
   - What is the correct research position?
8. **Minimum remediation set.** If the position could be strengthened later, name the smallest set
   of *protocol*, *experiment*, or *wording* changes that would do it. **Do not perform any of them**
   — this is a recommendation list only.

### 4.1 Output format

Produce a structured report with these sections, in this order:

```
## Verdict (one paragraph, committed position)
## Strengths
## Weaknesses (ranked)
## Publishable claims (with exact phrasing)
## Claims to downgrade
## Claims that cannot be made
## Research position
## Adjudication of the five positioning questions
## Minimum remediation set (recommendations only)
## Residual uncertainties
```

Tag every substantive judgement as one of: `SUPPORTED` / `PARTIAL` / `SPEC_ONLY` / `UNKNOWN`,
using the evidence boundary defined in §2.1–§2.2. Do not upgrade a `SPEC_ONLY` item to `SUPPORTED`
because the spec is well written.

---

## 5. Language

The reviewer brief is in English. **The final report will be delivered to the user in Chinese**
(简体中文). Write your review in English; a Chinese rendering will be produced downstream. Paper
titles, venue names, arXiv IDs, model names, config keys and code identifiers stay in English.
