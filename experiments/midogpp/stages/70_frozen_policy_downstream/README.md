# Stage 70: Frozen-Policy Downstream Evaluation

This is the thesis-facing downstream utility stage. It evaluates a routing or
composition policy only after expert selection, generation settings,
classifier settings, thresholds, budgets, and seeds have been frozen.

Held-out target labels are scoring-only. Reports must include matched
baselines, candidate eligibility, budget matching, leakage and identity audits,
BACC, macro-F1, and stability where available.

Status: the reservation, final-authorization gate, and descriptive evaluator
are `active`. Independent protocol, architecture, and generative-model review
authorizes exactly one hash-bound consumed-test execution; no Stage-70 result
exists until that run completes and its bundle validates. The test split's
9,928 eligible rows were already consumed for representation adoption. They may
therefore support only a descriptive comparison of the already frozen policies,
never fresh confirmation, policy promotion, recipe selection, or a deployment
claim.

The execution order is closed and irreversible:

1. The active reservation consumes
   `midogpp_uniform_b_test_consumption_ledger_v1`, validates the frozen
   reference/bank/generation/policy state, and materializes only opaque target
   identities. It cannot generate, predict, open labels, or score metrics.
2. The label-blind Virchow2 test cache is a catalog-only derived feature cache,
   not a registry experiment output. Its strict config is bound to reservation
   `reservation_b541f28d223a315edbec2b630c915fc2e7cd47cbfcc064dabecda860aa14c9e3`
   and extractor protocol
   `8bae5653e53b087f1662216308dcf2352975f5efa6fdda5fae4e72ff04d3790b`.
3. The active final-authorization gate consumes the narrow
   `midogpp_frozen_policy_test_scoring_manifest_v1` alias, hashes but does not
   parse it, binds the validated cache content, and emits a prediction-only
   token.
4. The evaluator config is now bound to validated cache content
   `df0bdbf64881ee00...d187db`, row order `bd1a85b954962035...33c389`, and
   final authorization token `a344cd66fc88daae`. Its hardened seal and
   independent closed-world bundle validator passed final review, so execution
   is authorized. All 243 prediction cells must be durably sealed before test
   labels are opened for scoring.

All arms retain matched candidate eligibility, synthetic budgets,
GenerationLock streams, seed replicates, classifier settings, and target rows.
The previously reported `0.7701` BACC is Stage-20 seven-source source-inner
evidence. It is context only, not an expectation, reference value, or promotion
threshold for Stage 70.
