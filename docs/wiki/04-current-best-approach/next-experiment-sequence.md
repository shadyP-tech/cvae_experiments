# Next Experiment Sequence

## Purpose

Record the current ordered next steps after the SAIL extraction and the
Virchow2 CVAE D-series decentralized composition runs.

## Key Claims

1. SAIL remains the real-feature source-only aggregation gate.
2. The paired dense all4 reliability confirmation provides a full PASS for
   dense generated-embedding aggregation, not target-conditioned routing.
3. Component-union/random mass-bag follow-ups show high mean BACC but do not
   validate source-mass allocation because controls are competitive.
4. Multipanel tail-risk mass-bagging shows that 0.90 mean BACC is reachable,
   but still fails center3/min-center; its audit points to confident
   minority-class collapse.
5. Before any new sparse generated-embedding selector, reuse paired
   generation/prediction invariants so identical source sets are evaluated with
   identical generated bundles.
6. The cleanest generated-embedding evidence is heldout-excluded
   reliability-weighted dense all4 aggregation under paired invariants.
7. Support-NELBO, source-inner transfer, support8 calibration, and point
   source-mass reliability should remain diagnostic/negative until they beat
   matched controls.
8. The active generated-embedding bottleneck is weak-center/tail robustness and
   rare-positive confidence collapse, not mean BACC alone.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../sail/configs/sail_virchow2.yaml`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1/tables/decentralized_support8_top3_tau05_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_conclusion.md`
- `../../../cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml`

## Current Ordered Sequence

1. Run or sync SAIL:

   ```text
   Virchow2 dense source-selected config aggregation
   ```

   This remains the real-feature gate.

2. Verify SAIL leakage and decision artifacts:

   ```text
   target-eval labels scoring-only
   source-inner selection source-only
   weak-center and seed-stability gates
   ```

3. For generated embeddings, do not extend D1.5 source-inner transfer directly.

   D1.5 failed:

   ```text
   mean BACC: 0.8354
   delta vs equal all4: -0.0165
   source-inner Spearman: -0.1102
   shuffled-score control gap: -0.0326
   ```

4. Treat the paired dense all4 reliability confirmation as the current
   generated-embedding dense aggregation result:

   ```text
   full reliability-weighted center-equal BACC: 0.8506
   delta vs equal all4: +0.0271
   delta vs strongest negative control: +0.0416
   pairing invariant audit: 420/420 PASS
   ```

5. Treat component-union/random mass-bag as the current high-mean diagnostic
   surface:

   ```text
   mass-bagged component union mean BACC: 0.8903
   retention vs source-union K16: 0.9960
   delta vs random mass-bag control: -0.0016
   ```

   This means there is little mean-BACC headroom left unless a method beats
   random/shuffled controls or improves weak-tail behavior.

6. Treat multipanel tail-risk mass-bagging as the latest high-mean negative
   stabilization result:

   ```text
   center-equal mean BACC: 0.9087
   min-center / center3 BACC: 0.7897
   bottom20 delta vs prior tailrisk: +0.0408
   seed std delta vs prior tailrisk: -0.0079
   verdict: MULTIPANEL_TAILRISK_STABILIZATION_FAIL
   ```

   The Center3 audit assigned `42 x center3` to near class collapse,
   probability-pooling suppression of the best seed, and confident wrong
   predictions.

7. Reuse paired generation-cache invariants before any further generated
   selector:

   ```text
   same source set + same budgets + same replicate seed
   -> same generated features and prediction bundles
   regardless of method label
   ```

8. Treat support-NELBO as diagnostic unless a future locked run beats:

   ```text
   matched reliability-only baseline
   matched equal dense baseline
   shuffled-support control
   ```

9. Treat source-inner transfer as negative unless a future repaired/invariant
   run reverses:

   ```text
   negative Spearman vs target subset utility
   underperformance vs equal all4
   underperformance vs shuffled-score control
   ```

10. Current generated-embedding next step:

   ```text
   predeclared Center3/weak-tail follow-up focused on source-only calibration,
   minority-class decision stability, or pooling preservation of rare useful
   seed-level evidence
   ```

   The multipanel audit is target-label-informed after fixed predictions, so it
   cannot choose thresholds, calibration, seeds, or policy. Harmful-source
   suppression should be validated only if final artifacts are later synced.

## Interpretation

The immediate generated-embedding bottleneck is not another mean-BACC allocator
or another sparse top-k selector. Component-union/random mass-bagging has
already reached the source-union K16 region in mean BACC, but controls are too
competitive. The remaining bottleneck is weak-center/tail robustness and
whether source-only calibration, pooling, or source-interaction diagnostics can
prevent rare-positive confidence collapse before target evaluation.

## Implication For Thesis

The thesis can still make a strong, disciplined argument:

```text
source-union K16 diagnoses that latent prior sampling can recover utility
paired dense all4 reliability weighting gives a protocol-clean dense aggregation win
component-union/random mass-bagging exposes high generated-embedding capacity
multipanel tail-risk bagging shows mean BACC can exceed 0.90 but still fail weak-center robustness
D1.3/D1.5/support8 show that obvious compatibility signals are not sufficient
```

This is weaker than a final deployable router, but stronger scientifically than
overclaiming support-NELBO or source-inner transfer.

## Limitations

- SAIL artifacts are still TODO locally.
- The paired dense all4 result is not sparse routing.
- Random mass-bag/component-union success is not a compatibility proof.
- The earlier harmful-source suppression run is not yet final evidence.
- The multipanel Center3 audit is diagnostic-only and cannot be used for method
  selection.
- D1.5 has an identified paired-sampling audit issue, now addressed only for
  dense all4 comparisons.
- No current generated-embedding experiment supports formal privacy.

## Next Checks

- Sync or run SAIL artifacts.
- Reuse paired generated-bundle tests before any D1.6-style selector.
- Repair D1.5 `dropped_source_target_utility_rank` before using drop-rank
  tables in the thesis.
- Predeclare any Center3 follow-up before evaluation.
- Sync and validate harmful-source suppression final artifacts only if that
  earlier run is resumed.
- Decide whether the final thesis contribution is framed as partial
  data-minimizing composition evidence plus bottleneck analysis, or whether a
  final robustness confirmation is worth the compute.
