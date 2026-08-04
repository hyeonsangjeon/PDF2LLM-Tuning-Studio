# RL Experiment Plan (programmatic-verifier RL) — **status: `planned`**

> **No RL has been run in this repository, and no RL result is implied.** This
> document, the reward functions in
> [`pdf_qa_extraction/evaluation/rewards.py`](../pdf_qa_extraction/evaluation/rewards.py),
> and their tests exist to design the **reward/eval interface first**. Any
> GRPO/PPO stage is gated on the feasibility checks below and is **future work**.

If a reinforcement-learning stage is ever added, it would be a
**programmatic-verifier RL** — rewards come from deterministic rule/verifier
checks, **not** from human-preference models. We deliberately do **not** frame
this as RLHF/PPO/DPO experience, and we do not expand the claim beyond what is
implemented.

## Why interface-first

RL is easy to add prematurely and hard to trust. So we pin down the reward
surface and its failure modes before any policy optimisation:

- The rewards are **deterministic** pure functions of a Q&A-with-evidence record
  (schema: `pdf_qa/schemas/qa_with_evidence.schema.json`).
- They are **unit-tested** (`evaluation/tests/test_rewards.py`) on public/
  synthetic records only (the committed `public_finance_demo` gold fixture +
  small hand-built cases).
- Each reward ships a **RewardCard** (definition, range, failure modes) in
  `rewards.py::REWARD_CARDS`.

## Reward candidates (implemented as verifiers, not yet used for RL)

| Component | What it verifies |
|---|---|
| `evidence_validity` | citation integrity: verbatim quote + `quote_sha256` + page + parser element id (full P0-8 verifier when source docs are supplied) |
| `numeric_consistency` | the answer's numbers/dates/currency are grounded in the cited quotes |
| `schema_compliance` | the record validates against the Q&A-with-evidence JSON schema |
| `calculation` | rule-based calc correctness vs a declared `computation`/gold value — **never fabricates** a ground truth |
| `abstention` | abstains iff the question is unanswerable/policy-violating; penalises both fabrication and over-refusal |
| `pii_nonexposure` | no real PII in the answer or quotes (`pdf_qa.pii` baseline; see [`TRUST_AND_DATA.md`](TRUST_AND_DATA.md)) |
| `version_recency` | cites the latest document version; penalises stale/revoked sources |
| `length_penalty` (guard) | penalises padding beyond the evidence and verbatim copy-through (a reward-hacking guard) |

`score_record` aggregates these with explicit weights and applies the length
guard multiplicatively, returning per-component scores so any total is auditable.

## Feasibility gates (all must pass before adding GRPO/PPO)

1. Is there a **clear, remaining quality gap** after the SFT baseline?
2. Is each reward **deterministic** or **correlated with human judgement**?
3. Is there a concrete reason **GRPO is needed over SFT/DPO**?
4. Can we measure **KL drift, reward hacking, length bias, and general-capability
   regression**?
5. Were rewards and thresholds designed on a **dev set only**, without opening the
   final test?

Only after these pass would a candidate config
`workflows/pdf_native_post_training/configs/rl-feasibility.yaml` be added. It is
**intentionally absent** today (the gates are unmet), and RL orchestration would
live as an optional future stage of the PDF-native workflow — **never** inside
the `quantization/` track.

## Guardrails (non-negotiable)

- **Public/synthetic data only** for reward development and tests.
- **Reward functions + unit tests exist first** (this repo already satisfies this).
- **No GRPO/PPO without a baseline and pre-declared success criteria.**
- RL results are shown as **`planned`** until actually executed.
- **Final-test examples and their failures must not enter reward development.**
  Rewards/thresholds are designed on dev data; the final holdout stays sealed.
- If no improvement beyond the pre-declared bar is found, we publish the honest
  conclusion **"RL was not needed"** rather than manufacturing a result.

## Deliverables — status

| Item | Status |
|---|---|
| Reward functions + RewardCards | ✅ implemented (`evaluation/rewards.py`) |
| Reward unit tests (public/synthetic) | ✅ implemented (`evaluation/tests/test_rewards.py`) |
| This plan | ✅ |
| Human-label ↔ reward correlation study | ⏳ planned |
| Per-component ablation | ⏳ planned |
| SFT vs RL on the same P1-5 final set (3 seeds, raw trajectories) | ⏳ planned |
| `rl-feasibility.yaml` + any GRPO run | ⛔ gated (feasibility gates unmet) |

Related: [`SECURITY.md`](../SECURITY.md) · [`TRUST_AND_DATA.md`](TRUST_AND_DATA.md) ·
[`DATA_AND_LICENSES.md`](DATA_AND_LICENSES.md).
