# Direct Support-NELBO Experiments

This root owns the direct support-NELBO experiment surface:

- `configs/experiments/`: thesis-facing support-estimated utility configs.
- `scripts/run/`: launchers for support-NELBO sweeps and calibration runs.
- `scripts/preflight/`: dataset/domain feasibility checks.
- `scripts/reports/`: consolidation, decision-table, and verification artifact builders.
- `artifacts/comparison_tables/`: tracked support-NELBO result tables and reports.
- `tests/`: focused regression tests for support-NELBO artifacts and calibration.

The shared CVAE training, routing, and evaluator implementation remains in
`../cvae_testing/src`. Scripts in this root add `../cvae_testing` to
`sys.path` and write direct support-NELBO outputs back into this root under
`artifacts/`.

Protocol boundary:

`Query -> Compatibility Estimation -> Routing Decision -> Expert Selection -> Utility (NELBO)`

Support NELBO may estimate target-local compatibility only from a disjoint
unlabeled support split. Held-out target evaluation NELBO remains evaluation-only
and must not be used by routing or alpha/model selection.
