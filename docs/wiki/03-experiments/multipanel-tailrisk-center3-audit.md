# Multipanel Tail-Risk And Center3 Audit

## Purpose

Document the `virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1`
experiment and its focused Center3 failure audit.

This page records the result as source-only dense stochastic generative
composition evidence. It is not a compatibility-router result and does not
claim random mass-bagging discovers target-compatible experts.

## Key Claims

- The multipanel tail-risk method reaches high mean utility, with center-equal
  mean BACC 0.9087 on the 14-cell intersection set.
- The locked method still fails because center3/min-center remains 0.7897,
  center3 regresses by -0.0136 versus prior tailrisk, and tail-risk transfer is
  flagged.
- The useful result is not routing discovery. It is evidence that probability
  pooling can improve mean, bottom-tail, and seed-stability metrics while still
  missing a weak-center gate.
- The Center3 audit assigns the primary failed cell to near class collapse,
  pooling suppression of rare useful seed-level signal, and confident wrong
  predictions.
- Component coverage is not the main explanation for the primary failure cell.

## Evidence / Source Artifacts

Verified artifact root:

```text
cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/
```

Primary files:

- `reports/decision_summary.md`
- `reports/leakage_report.json`
- `manifests/protocol_manifest.json`
- `tables/multipanel_tailrisk_summary.csv`
- `tables/multipanel_tailrisk_paired_deltas.csv`
- `tables/multipanel_tailrisk_failure_decomposition.csv`
- `tables/multipanel_tailrisk_panel_disagreement.csv`
- `tables/multipanel_tailrisk_probability_invariants.csv`
- `center3_failure_audit/center3_failure_conclusion.md`
- `center3_failure_audit/center3_failure_cell_summary.csv`
- `center3_failure_audit/center3_failure_pooling_path.csv`
- `center3_failure_audit/center3_failure_source_weight_comparison.csv`
- `center3_failure_audit/center3_failure_component_coverage_comparison.csv`

Protocol status:

```text
leakage report: PASS
target support used: false
selection used target labels: false
target eval labels: scoring/audit only
panel seeds are not evaluation replicates
```

## Multipanel Result

Primary method:

```text
component_union_tailrisk_multipanel_shrink050_random_mass_bag_blend050
```

Primary verdict:

```text
MULTIPANEL_TAILRISK_STABILIZATION_FAIL
```

Diagnostic flags:

```text
TAIL_RISK_TRANSFER |
MIN_CENTER_NOT_IMPROVED |
CENTER3_NOT_IMPROVED_AND_BELOW_0P82
```

Key metrics:

| Metric | Value |
| --- | ---: |
| Intersection cells | 14 |
| Center-equal mean BACC | 0.9087 |
| Min-center BACC | 0.7897 |
| Center3 BACC | 0.7897 |
| Frozen bottom20 BACC | 0.7077 |
| Seed std BACC | 0.0431 |
| Delta vs prior tailrisk | +0.0130 |
| Delta vs canonical random mass-bag | +0.0103 |
| Bottom20 delta vs prior tailrisk | +0.0408 |
| Seed std delta vs prior tailrisk | -0.0079 |
| Worst per-center regression vs prior tailrisk | -0.0136 |
| Worst seed-center BACC | 0.4975 |

Interpretation:

The method passes the mean-BACC and bottom-tail direction of travel, but it
does not pass the thesis-facing weak-center rule. The result shows that 0.90
mean BACC is achievable for a source-only full-matrix CVAE/component-union
variant, but mean BACC is not enough.

## Center3 Failure Audit

Audit scope:

```text
diagnostic-only; target labels used only after fixed predictions for scoring
and failure analysis
```

Primary failed cell:

```text
experiment_seed=42 x heldout_center=3
```

Assigned failure mode:

```text
near_class_collapse | probability_pooling_suppresses_best_seed |
confident_wrong_predictions
```

Key audit details:

| Quantity | Value |
| --- | ---: |
| Final v2 BACC | 0.4975 |
| Class0 / class1 support | 198 / 2 |
| Final class0 / class1 predicted count | 199 / 1 |
| Final class0 recall | 0.9949 |
| Final class1 recall | 0.0000 |
| Final mean confidence | 0.9795 |
| Final mean margin | 0.9590 |
| Pooled anchor BACC | 0.4975 |
| Pooled random mass-bag BACC | 0.4975 |
| Canonical random mass-bag BACC | 0.5000 |
| Seed 101 anchor BACC | 0.9949 |
| Seed 101 blend BACC | 0.7475 |
| Seed 127 blend BACC | 0.7323 |
| Panel JS divergence | 0.0019 |
| Panel hard-label disagreement | 0.0033 |

The failed final bundle is not uncertain in the ordinary sense. It is highly
confident and almost entirely predicts the majority class. The rare seed-level
signal that sees at least one positive sample is diluted by panel and all-seed
probability pooling.

The `44 x center3` control reaches 0.9823 BACC with class counts class0 = 87
and class1 = 113. This means center3 is not globally impossible; the failure is
specific to the rare-positive `42 x center3` regime.

The `43 x center4` tail-repair control reaches 0.7923 BACC versus 0.5923 for
pooled anchor and 0.6923 for pooled random mass-bag. This shows the same
method can repair some weak-tail cells, but not the primary rare-positive
Center3 collapse.

## Source And Component Audit

The source-weight comparison does not show a simple bad-source dominance story
for `42 x center3`. The tailrisk blend has effective source count around 3.98
and low L1 distance from uniform around 0.077 in the primary failed cell.

The component coverage comparison also does not explain the failure:

```text
primary failed cell component mass coverage: 1.0
primary failed cell unsampled active components: 0
```

This points away from component undersampling as the main bottleneck for the
primary failed cell.

## Interpretation

The result changes the diagnosis:

```text
mean utility can be high enough
bottom-tail can improve
seed std can decrease
but rare-positive weak-center collapse can survive probability pooling
```

The overlooked bottleneck is not just stochastic prior/sample variance. In the
primary failure cell, panels have low disagreement and mostly agree on the
wrong high-confidence majority-class decision. More panels alone are unlikely
to repair this unless the method changes how source-only calibration, class
priors, or pooling handle rare useful minority-class evidence.

## Implication

The generated-embedding thesis story should not use this method as a final
PASS. It should use it as strong bottleneck evidence:

```text
source-only dense stochastic generative composition can cross 0.90 mean BACC,
but weak-center utility is still limited by minority-class decision collapse.
```

Any follow-up method must be predeclared and source-only. The audit-only target
labels cannot be used to choose seeds, thresholds, calibration, routing, or
policy switches.

## Limitations

- The Center3 audit is target-label-informed after fixed predictions, so it is
  diagnostic-only.
- The audit focuses on four predeclared cells and should not be generalized to
  every center without additional evidence.
- The good seed-level rows are not deployable seed-selection evidence because
  they were identified after target-eval scoring.
- The result does not validate random mass-bag as compatibility estimation.

## Next Checks

- Predeclare any Center3 follow-up before evaluation.
- Add source-only calibration or class-prior diagnostics if they are used,
  keeping target-eval labels scoring-only.
- Report threshold-free diagnostics such as AUROC or ranking separation only as
  audit evidence unless the thresholding rule is source-only and predeclared.
- Keep center3, min-center, frozen bottom20, worst seed-center, and seed std as
  primary gates for generated-embedding robustness claims.
