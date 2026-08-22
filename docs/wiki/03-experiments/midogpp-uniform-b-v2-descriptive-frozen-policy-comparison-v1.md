# MIDOG++ Uniform-B V2 Descriptive Frozen-Policy Comparison V1

## Decision

The canonical Stage-70 run is `COMPLETE` and validates `PASS` with decision
`DESCRIPTIVE_COMPARISON_COMPLETE`.

```text
artifacts/midogpp/70_frozen_policy_downstream/
uniform_b_v2_descriptive_frozen_policy_comparison/v1/
```

Its claim scope is
`descriptive_frozen_policy_comparison_on_previously_consumed_test`.
`fresh_confirmatory_status` is `BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT`, and
`routing_policy_promoted=false`.

## Frozen-arm results

| Policy | Equal-center/equal-seed mean BACC | Mean macro-F1 | Cells |
| --- | ---: | ---: | ---: |
| `equal_union_control` | `0.7749677917` | `0.7726084368` | 81 |
| `metadata_max_tie_union` | `0.7450994314` | `0.7399571571` | 81 |
| `utility_regret_frozen_policy` | `0.7749677917` | `0.7726084368` | 81 |

Metadata minus equal-union BACC is `-0.0298683603`. The descriptive paired
bootstrap interval is `[-0.0504064265, -0.0087054046]`, based on 2,000 valid
replicates that resample centers and cases within center while retaining the
full crossed seed grid. This interval is
`descriptive_resampling_uncertainty_only`; it is not fresh confirmatory
inference.

The utility/regret policy is exactly equivalent to equal-union in all 81
cells. Its predictions, probabilities, and metrics are identical because the
Stage-60 uncertainty gate froze equal-union fallback for all nine outer folds.
It is therefore a control-equivalence result, not a second independent routing
hypothesis.

## Protocol evidence

- 243 policy/target/seed prediction cells were materialized and sealed before
  target labels opened.
- Target labels were used for scoring only and never for fitting, selection,
  routing, or prediction.
- No target support was used.
- Routing was not recomputed at Stage 70.
- No Stage-50 or Stage-90 artifact was an input.
- Identity-overlap and leakage reports validate `PASS`.
- The artifact records a clean repository revision (`380a0a99...`).

The primary evidence files are:

```text
reports/run_state.json
reports/validation_report.json
reports/leakage_report.json
reports/publication_decision.json
reports/utility_control_equivalence.json
tables/arm_summaries.csv
tables/bootstrap_summary.csv
tables/paired_deltas.csv
tables/target_metrics.csv
```

## Interpretation

The tested exact-match metadata proxy is related to domain structure but does
not transport reliably into downstream utility. On this descriptive surface,
its max-tie policy is worse than the dense equal-union composition. The
source-inner utility/regret policy behaves safely by abstaining, but it does
not demonstrate adaptive routing.

Allowed thesis use:

- a protocol-clean descriptive negative result for this metadata policy;
- evidence that dense equal-union CVAE composition is a strong baseline;
- evidence that the conservative utility/regret policy preserved its frozen
  exact fallback contract.

Forbidden use:

- fresh or confirmatory routing superiority;
- policy promotion or deployment authorization;
- external-dataset or new-center generalization;
- selection or tuning of a successor on these labels;
- feeding Stage-90 post-hoc diagnostics back into Stage 60 or Stage 70.

## Next evidence

A scientific routing-success claim now requires a separately predeclared,
whole-case/patient/slide-disjoint and genuinely unconsumed evaluation surface,
or an external/new-center confirmation. Repeated router development on this
consumed test set may diagnose mechanisms but cannot convert the Stage-70
result into fresh evidence.
