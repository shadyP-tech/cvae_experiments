# HARP v21: normalized correction masses and frozen proposer calibration

The public entrypoint is `fit_source_router`. This independent successor keeps
the pairwise donor ranker and exact B/U/D01_ONLY/D10_ONLY/BOTH menu. It learns
baseline-anchored class-normalized correction masses from full 3840-dimensional
features and computes the expected gain of each actual composite. A separate
baseline-anchored posterior supplies proper-loss estimates.

| Responsibility | Modules |
|---|---|
| Immutable menus, full-feature arrays and branch contracts | `menu_contracts.py`, `composite_contract.py`, `composition.py` |
| Evidence design, class-normalized targets, fitting and models | `evidence/`, `patch_evidence.py` |
| Batched actual-action effects and proper-loss estimates | `outcome_model.py`, `outcome_targets.py` |
| Pairwise donor numerics, features, ranking and proposals | `ranker_numerics.py`, `ranker_features.py`, `pairwise_ranker.py`, `proposal_model.py` |
| Positive-gain and proper-loss screening | `risk_selection.py`, `candidate_prediction.py` |
| Disjoint fit/calibration partition and unchanged proposer | `calibration_split.py`, `learning.py`, `winner_gate.py` |
| Nested evidence-variant and abstention selection | `crossfit.py`, `frontier.py`, `frontier_joins.py` |
| Source-only scoring, admission and sealed policy | `truth.py`, `aligned_metrics.py`, `admission.py`, `policy.py` |

Class mass factors into a nonnegative total and a softmax allocation anchored
at B. For signed hard-prediction changes delta, the estimated gain is
`0.5 * sum(delta * (mass_1 - mass_0))`. These conditional mean estimates do not
supply case-harm probabilities. A small binary gate learns harm of the complete
selected winner on disjoint calibration cases. Its proposer stays frozen.
A gate veto always returns exact B without trying a runner-up.

Five outer and four inner source folds compare baseline, calibrated-baseline
and embedding-residual evidence, plus nine abstention thresholds. Unchanged
policy-level coverage and risk requirements may still reject every route.
No target labels or predecessor scientific state enter selection.

The executor is `runtime/harp_v21_execution`; independent preparation, source
closure, amendment and lease handling live in
`diagnostics/fixed_bank_harp_router_v21`. The workstation profile keeps two
persistent GPU workers and four classifier workers. Truth-bearing science fits
stay in one process with one BLAS thread. Immutable float32 transport, bounded
scope-aware caches, batched effects and reused threshold predictions avoid
repeating physical work for the evidence variants.

See [the architecture and verification note](../../../../../docs/wiki/03-experiments/midogpp-harp-v21-correction-mass-router.md).
This is a planned terminal consumed-test diagnostic. Code, tests and synthetic
benchmarks do not issue authority or establish real routing utility.
