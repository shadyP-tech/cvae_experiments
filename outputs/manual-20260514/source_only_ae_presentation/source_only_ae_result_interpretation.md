RESULT INTERPRETATION:

### Evidence Source

- Aggregate decision table: `cvae_testing/results/comparison_tables/ae_first_routing_decision_table.csv`.
- Cross-seed summary: `cvae_testing/results/summaries/ae_first_routing_decision_table.md`.
- Per-seed method summaries: `cvae_testing/outputs/{breakhis,camelyon17}/learned_utility_ae_first_routing_v1/*/reports/learned_utility_method_summary.csv`.

### Thesis Question

Can source-trained Autoencoder confidence route target/query samples without target support, and does that proxy recover held-out NELBO utility better than metadata routing?

### Primary Metrics

- BreakHis: metadata top-1 0.342, gap 35.7; AE argmin top-1 0.697, gap 6.6; margin-gated AE top-1 0.598, gap 11.8, coverage 0.789, harmful-vs-metadata 0.160.
- Camelyon17: metadata top-1 0.329, gap 11.0; AE argmin top-1 0.426, gap 8.9; margin-gated AE top-1 0.394, gap 9.3, coverage 0.710, harmful-vs-metadata 0.267.

### Baseline Comparison

The source-only AE proxy beats metadata routing on top-1 oracle hit and normalized oracle gap in both datasets, with a stronger effect on BreakHis. The gated policy reduces risk relative to unconstrained AE argmin, but it still has non-trivial harmful routing decisions.

### Claim Classification

`DIAGNOSTIC ONLY`: the AE proxy contains useful distributional signal, but the margin-gated policy fails the domain-level non-degradation requirement and therefore should not be presented as an adoption-ready router.

### Thesis Text

Source-only AE confidence can partially recover expert compatibility without target support, improving over metadata routing on oracle-hit and oracle-gap metrics. However, because the same proxy still produces harmful overrides and fails the domain-level safety criterion, it supports the thesis pivot from source-only proxy routing toward target-local NELBO utility estimation.

### Caveats

- The method estimates reconstruction fit in embedding space, not CVAE utility directly.
- The cross-dataset verdict remains `DIAGNOSTIC ONLY`, so do not claim this as the final routing method.
- The effect is dataset-dependent: BreakHis shows a clearer gain than Camelyon17.

### Next Evidence Needed

For the main thesis claim, place this table next to direct support-NELBO routing results to show that target-local utility estimation is more reliable than source-only proxy confidence.
