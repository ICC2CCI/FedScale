# RESEARCH_REVIEW_ROUND_2 — Scoring Round

**Absolute path (the only artifact to hand the reviewer):**
`C:\大模型安全\CCLPT\pure-github-repo\FedScale\idea-stage\RESEARCH_REVIEW_ROUND_2.md`

This continues threadId `01a0b3e2-54e9-7b03-9024-821824c2a7e4`. Round 1 produced the full
positioning review (saved verbatim at
`.aris/traces/research-review/2026-09-18_run01/round1_response.md`); the original brief is at
`idea-stage/RESEARCH_REVIEW_REQUEST.md`. **Do not re-litigate round 1.** This round only converts
that review into numbers.

---

## 1. What is being asked

Produce a **mock peer review with scores** of the artifact set you already reviewed, written as if
it had been submitted to a venue. The scheme is a systems-flavoured federated-learning artifact, so
score it as a **systems / MLSys-style submission whose contribution type is engineering integration**
— which is your own round-1 verdict. Do not score it as if it claimed an algorithmic contribution;
round 1 already established it does not have one, and scoring it against an algorithm-paper bar
would be double-counting the same finding.

Deliver **three things**:

### 1.1 A mock review in standard venue format

```
## Summary            (2–4 sentences, neutral)
## Strengths          (bulleted, each tied to a specific verified artifact)
## Weaknesses         (bulleted, ranked; each must be actionable or fatal)
## Questions for Authors   (the questions you would actually ask in rebuttal)
## Score
## Confidence
## What Would Move Toward Accept
```

### 1.2 A score, under two framings

Give **both**, clearly labelled:

- **Score-as-is** — the artifact set exactly as it stands today.
- **Score-after-remediation** — the score it would earn if the §6 "minimum remediation set" from
  `AI_DESIGN_REVIEW.md` were fully executed (R1–R11), with **no other changes**.

Use a numeric scale you state explicitly (e.g. NeurIPS 1–10, or ACL/ARR-style, or your own — but
name it and give the accept/reject boundary and the borderline band). For each framing give:

| field | meaning |
|---|---|
| `rating` | the number, on your declared scale |
| `confidence` | how confident you are in that rating (state the scale, e.g. 1–5) |
| `decision` | accept / borderline / reject, against the boundary you declared |

Then give a **per-axis breakdown** so the single number is decomposable. Score each axis on a
declared sub-scale and weight it or not, as you see fit — but say what you did:

1. **Methodological novelty** — round 1 found essentially none in layers 1–3.
2. **Technical soundness / correctness** — this is where the `N_max·Q_max < q/2` finding (valid only
   for N=2) and the PRG divergences live. Weight this heavily.
3. **Evidence quality** — one model, two clients, IID, one seed, one ratio, one run; 5-round timing.
4. **Systems / engineering contribution** — blob, O(1) notify, parallel unmask, async persistence,
   and the *unmeasured* scaling axis.
5. **Significance / impact** — for cross-silo federated LLM fine-tuning.
6. **Clarity / specification quality** — the written specs are unusually complete, but the
   spec-vs-execution boundary is blurred.
7. **Reproducibility** — run IDs are document-level citations only, not independently verifiable.

### 1.3 A results-to-claims sensitivity

State how the score moves under each of these hypothetical outcomes. This is a scoring exercise,
not a request to run them:

| # | Hypothetical outcome | Your predicted rating | Why |
|---|---|---|---|
| H1 | Cross-language test vectors pass on one authoritative mask construction | ? | |
| H2 | Arithmetic contract fixed (e.g. 2^32 modulus) + N=8 non-IID heterogeneous-weight 20-round run reproduces 0.985-vs-0.987-class alignment | ? | |
| H3 | Window-count scaling measured, showing wall time does not follow `RTT × N` | ? | |
| H4 | Full SecAgg conformance campaign passes (two dropout positions, threshold failure, survivor freezing, no-double-reconstruction, fresh-attempt retry) at a cohort with a non-trivial threshold | ? | |
| H5 | Two more models (one ≥1B) + multi-seed, no improvement over the current single-configuration result | ? | |

---

## 2. Scoring discipline (binding)

1. **Score the artifact, not the ambition.** Score what is documented and evidenced, per the round-1
   evidence boundary. Do not credit intended-but-unimplemented protocol.
2. **Do not re-open round-1 conclusions.** If you believe a round-1 finding was wrong, say so in one
   line and score accordingly — but do not re-derive the whole positioning.
3. **Do not invent novelty to justify a higher score,** and do not manufacture weaknesses to justify
   a lower one. If the honest number is a clear reject, say so plainly.
4. **Every weakness you list must be one you would actually write in a real review** — i.e. something
   the authors could act on, or something that is fatal. No filler.
5. **State your scale and your accept boundary explicitly** before giving any number. A rating whose
   scale is unstated is not a rating.
6. Do not propose new experiments beyond the hypotheticals in §1.3. Round 1 already produced the
   remediation list; this round only prices it.
7. Respect the standing scope: no code review, no inference of the method from code leftovers
   (`block_vote`, `block_topk`, Flower/TKE, retired per-window PUT), no use of undocumented run
   directories.

---

## 3. Output

English. Append a `## Residual uncertainties` section stating anything that would have changed your
number had it been resolvable. Keep the whole response to a length a real reviewer would actually
submit — scores first, prose second.
