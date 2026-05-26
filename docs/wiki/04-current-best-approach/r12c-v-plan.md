# R1.2c-V Lineage

## Purpose

Record the R1.2c-V diagnostic design that was extracted into SAIL.

## Problem

R1.2b suggests Virchow2 real-feature transfer is strong on average, but exact top-1 config selection is brittle.

## Hypothesis

Top-k dense aggregation over source-selected Virchow2 configs can reduce top-1 selection regret and improve weak-center/seed stability. This is now implemented through SAIL.

## Method

Primary row:

```text
row_role = source_inner_lodo_selected_dense_virchow2
```

R1.2c-V is the historical name for the diagnostic that became SAIL. It is real-feature config aggregation, not CVAE routing, and it does not evaluate generated embeddings.

## Candidate Pool

```text
backbone = virchow2 only
representation in {raw, PCA64, PCA128, PCA256}
classifier hyperparameters: C, class_weight
standardization: source-fit only
```

## Source-Only Robust Score

```text
robust_score =
    mean_inner_bacc
    - 0.25 * std_inner_bacc
    - 0.50 * max(0, 0.85 - min_inner_bacc)
```

The score penalizes weak-center collapse instead of selecting by mean BACC alone.

## Top-K Config Aggregation

1. Select top-k Virchow2 configs.
2. Train one classifier per selected config.
3. Predict target sample probabilities.
4. Align by canonical `sample_id`.
5. Aggregate probabilities.
6. Produce final prediction.

## Geometric Vs Arithmetic Pooling

Geometric:

```text
score_c = mean_i log(max(p_i(c), eps))
```

Arithmetic:

```text
score_c = mean_i p_i(c)
```

## Calibration Handling

Primary rows use no calibration. Source-temperature calibration is audit-only and cannot satisfy rebuild readiness.

## K=1 Sanity Baseline

`k=1` is a sanity/audit baseline that should reproduce R1.2b Virchow2 top-1 behavior when the selected config is identical.

## Audit-Only Cross-Backbone Extension

Cross-backbone aggregation across Phikon, UNI, and Virchow2 is audit-only. It estimates ensemble ceiling but cannot justify CVAE rebuild readiness.

## Leakage Guardrails

Target evaluation labels may be used only for scoring. They must not choose backbone, representation, PCA dimension, classifier hyperparameters, class weight, k, aggregation rule, calibration rule, or decision threshold.

## Success Gate

Virchow2-only dense primary rows must satisfy:

```text
mean BACC >= 0.92
worst center >= 0.85
seed mean-BACC std <= 0.03
no seed has worst-center BACC < 0.75
```

Optional: clear positive delta versus R1.2b Virchow2 top-1 or clear weak-center/stability improvement.

## Failure Interpretations

- If SAIL fails but a future cross-backbone audit passes, do not rebuild CVAEs yet; inspect selector calibration, weak-center instability, and ensemble diversity.
- If SAIL and cross-backbone audit both fail, pathology embedding transfer may still be unstable.

## Next Decision After Pass/Fail

If SAIL passes, compare it against the already-run Virchow2 CVAE
generated-embedding preservation/composition artifacts before adding new CVAE
complexity. If it fails, do not use cross-backbone audit success as rebuild
readiness.

The D-series generated-embedding artifacts now provide CVAE preservation and
dense aggregation evidence, but they do not replace the SAIL real-feature gate
and do not prove sparse generated-embedding routing.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../sail/configs/sail_virchow2.yaml`
- `../../../sail/src/sail/`
- `../../../sail/tests/test_smoke.py`
- `../../../cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/README.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`

## Interpretation

The active SAIL tests cover imports, config loading, CLI help, synthetic evaluation, and no target-eval labels used for selection. Legacy R1.2c tests are archived only for provenance.

## Implication For Thesis

SAIL is the immediate real-feature bridge. D-series artifacts are the current
generated-embedding preservation, prior-composition, and dense aggregation
record.

## Limitations

No local SAIL output artifact directory is present. TODO: verify against artifact.

## Next Checks

- Generate or sync SAIL outputs.
- Validate `reports/leakage_report.json`.
- Update this page with verified pass/fail status.
- Keep D-series generated-embedding conclusions in the D-series and generated
  CVAE synthesis pages.
