# D-Series Decentralized Virchow2 CVAE Composition

## Purpose

Summarize the D-series experiments that tested whether Virchow2 CVAE generated
embeddings can preserve source-union GMM utility under decentralized,
raw-data-free summary exchange and expert aggregation.

## Key Claims

- The source-union K16 GMM prior is the strongest generated-embedding diagnostic
  upper bound, but it is centralized and not deployable as decentralized
  routing.
- Strict decentralized K16 preserved high utility when it was eligible, but the
  fixed K4-per-source-class requirement made the primary method ineligible.
- Adaptive source-local summaries restored eligibility but reduced utility and
  stability.
- Paired dense all4 source-local reliability weighting is the strongest
  protocol-clean generated-embedding dense aggregation result.
- Component-union and random mass-bag follow-ups reach higher mean BACC, but
  are not adopted because matched controls, random mass, and weak-tail failures
  remain unresolved.
- Multipanel probability-level tail-risk bagging reaches 0.9087 mean BACC, but
  fails because center3/min-center stays 0.7897 and the Center3 audit shows
  confident minority-class collapse.
- Support-NELBO and source-inner transfer are not validated as final
  compatibility selectors because matched controls remain competitive or better.
- The paired dense all4 confirmation is not sparse routing: all non-target
  source experts are included, and reliability only changes weighting, pooling,
  or synthetic budget allocation.

## Evidence / Source Artifacts

Verified synced artifacts under:

```text
/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/
```

Primary files:

- `virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `virchow2_cvae_source_union_k24_gmm_prior_v1/tables/source_union_k24_gmm_summary.csv`
- `virchow2_cvae_decentralized_k16_gmm_prior_v1/tables/decentralized_k16_summary.csv`
- `virchow2_cvae_decentralized_adaptive_gmm_prior_v1/tables/decentralized_adaptive_summary.csv`
- `virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1/tables/decentralized_support_nelbo_reliability_summary.csv`
- `virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1/tables/decentralized_support8_top3_tau05_summary.csv`
- `virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1/tables/decentralized_reliability_top3_summary.csv`
- `virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `virchow2_cvae_decentralized_component_union_reliability_shrink025_v2/reports/decision_summary.md`
- `virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1/reports/decision_summary.md`
- `virchow2_cvae_support8_calibrated_component_union_prior_v1/reports/decision_summary.md`
- `virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
- `virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/reports/decision_summary.md`
- `virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_conclusion.md`

Verified local paired confirmation artifacts:

- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/leakage_report.json`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/manifests/protocol_manifest.json`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_gap_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_delta_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_generation_invariant_audit.csv`

All listed D-series leakage reports inspected in the current update reported
`PASS`.

## Result Matrix

| Experiment | Primary method | Verdict | Mean BACC | Min center | Seed std | Claim status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| D1 strict K16 | `decentralized_exported_k4x4_cc_diag_gmm_k16_late_geom` | `INELIGIBLE` | 0.8806 | 0.8704 | 0.0384 | diagnostic only; fixed K4 failed eligibility |
| D1.1 adaptive K | `decentralized_exported_adaptive_k_cc_diag_gmm_late_geom` | `D1_1_PARTIAL_EVIDENCE` | 0.8143 | 0.7380 | 0.0681 | partial preservation evidence |
| D1.2 reliability weighted | `decentralized_exported_adaptive_k_source_reliability_weighted_geom` | `D1_2_PARTIAL_EVIDENCE` | 0.8493 | 0.8013 | 0.0382 | strongest decentralized partial evidence |
| D1.3 support-NELBO x reliability | `decentralized_exported_adaptive_k_support_nelbo_x_reliability_weighted_geom` | `D1_3_PARTIAL_EVIDENCE` | 0.8495 | 0.7262 | 0.0297 | target-conditioned diagnostic; not validated |
| D1.3.1 support8 top3 tau0.5 | `decentralized_support8_top3_tau05_support_nelbo_x_reliability_geom` | `D1_3_1_WEAK_PASS` | 0.8310 | 0.7804 | 0.0450 | weak pass label, but controls block strong claim |
| D1.4 reliability top3 | `decentralized_reliability_top3_geom_confirmation` | `D1_4_DIAGNOSTIC_ONLY` | 0.8212 | 0.7529 | 0.0498 | sparse reliability selection not adopted |
| D1.5 source-inner transfer top3 | `decentralized_source_inner_transfer_top3_geom_confirmation` | `D1_5_FAIL` | 0.8354 | 0.7092 | 0.0015 | negative evidence for source-inner drop-one selection |
| Paired dense all4 reliability confirmation | `paired_reliability_all4_weighted_geom` | `PAIRED_DENSE_ALL4_RELIABILITY_PASS` | 0.8506 | 0.8173 | 0.0308 | dense aggregation PASS; not sparse routing |
| Component-union shrink025 v2 | `decentralized_component_union_reliability_shrink025` | `COMPONENT_UNION_FAIL` | 0.8892 | 0.8168 | 0.0501 | high mean, but matched null/control/source-ablation failures |
| Mass-bagged component union | `decentralized_component_union_mass_uncertainty_bagged_v1` | `MASS_BAGGED_COMPONENT_UNION_FAIL` | 0.8903 | 0.7931 | 0.0568 | high mean, but random mass-bag control competitive |
| Support8 calibrated component union | `support8_calibrated_component_union_softmax_shrink050` | `SUPPORT_CALIBRATED_COMPONENT_UNION_FAIL` | 0.8727 | 0.7886 | 0.0369 | target-support calibration fails matched support/random controls |
| Dense reliability tail shield | `dense_reliability_tailshield_random_mass_bag_blend25_75` | `DENSE_TAILSHIELD_RANDOM_MASS_BAG_FAIL` | 0.8988 | 0.7896 | 0.0403 | high mean, but center3/worst-cell failure remains |
| Multipanel tail-risk mass-bag stabilization | `component_union_tailrisk_multipanel_shrink050_random_mass_bag_blend050` | `MULTIPANEL_TAILRISK_STABILIZATION_FAIL` | 0.9087 | 0.7897 | 0.0431 | high mean and bottom20 gain, but center3/min-center failure and tail-risk transfer remain |

## Source-Union Diagnostic Context

The best centralized diagnostic row is:

```text
source_union_cc_diag_gmm_k16_prior_sample_diagnostic
```

Verified mean BACC from `gmm_prior_gap_summary.csv`:

```text
0.8924
```

The K24 follow-up did not improve over K16:

```text
source_union_cc_diag_gmm_k24_prior_sample: 0.8751
delta vs vanilla K16: -0.0122
primary verdict: K24_COMPONENT_UNDERSAMPLED
```

Interpretation:

K16 is useful as a CVAE prior-preservation diagnostic. It should not be framed
as a deployable decentralized MoErging method.

## Method Lessons

D1 strict K16:

- Strong result when eligible.
- Retention vs source-union K16: 0.9925.
- Retention vs center-balanced K16: 1.0252.
- Primary status: ineligible due fixed local K4 component-fit failure.

D1.1 adaptive K:

- Adaptive K restored eligibility.
- Utility dropped to 0.8143 mean BACC.
- Did not beat single-source oracle adaptive K.
- Retention vs source-union K16 fell to 0.9178.

D1.2 reliability weighting:

- Improved over D1.1 by +0.0211 BACC.
- Reached 0.8493 mean BACC and 0.8013 min-center BACC.
- Retention vs source-union K16: 0.9572.
- Source reliability vs target single-source Spearman: 0.0461.

Interpretation:

Reliability is useful as a source-quality prior, but it is not target-specific
compatibility evidence.

Paired dense all4 reliability confirmation:

- Decision report primary method: `paired_reliability_all4_shrink050_geom`.
- Best reliability method: `paired_reliability_all4_weighted_geom`.
- Leakage report: `PASS`.
- Pairing invariant audit: 420/420 rows `PASS`.
- Equal all4 center-equal BACC: 0.8235.
- Full reliability-weighted center-equal BACC: 0.8506.
- Delta vs equal all4: +0.0271 center-equal BACC.
- Delta vs strongest negative control: +0.0416.
- Positive paired cells: 9/14.
- Centers improved vs equal all4: 4/5.
- Min center BACC: 0.8173.
- Seed std BACC: 0.0308.
- Gap vs real-feature dense reference: -0.0570 BACC.

Interpretation:

Source-only heldout-excluded reliability is useful for dense generated-embedding
aggregation when generated features and per-source predictions are paired across
methods. Most of the gain comes from synthetic budget allocation, with full
reliability weighting outperforming pool-only and budget-only variants. This
does not prove sparse source selection, because dense all4 includes every
non-target source.

D1.3 and D1.3.1 support-NELBO:

- D1.3 support-NELBO Spearman: 0.2228.
- D1.3.1 support8 top3 Spearman: 0.3143.
- D1.3.1 top3 oracle containment: 0.8571.
- D1.3.1 shuffled-support control gap: -0.0096.

Interpretation:

Support-NELBO shows some ranking signal, but controls are too competitive to
claim a validated target-conditioned compatibility router.

D1.4 reliability top3:

- Mean BACC: 0.8212.
- Delta vs equal all4: -0.0062.
- Random source-drop and shuffled-reliability controls are competitive.

Interpretation:

Sparse reliability-only source dropping is not a thesis-facing win.

D1.5 source-inner off-diagonal transfer:

- Mean BACC: 0.8354.
- Equal all4 reference: 0.8519.
- Shuffled-score control: 0.8680.
- Source-inner score vs target subset utility Spearman: -0.1102.

Interpretation:

Source-inner off-diagonal transfer, as implemented, does not identify the
correct drop-one source subset for heldout target utility.

Component-union and mass-allocation follow-ups:

- Component-union shrink025 v2 reached 0.8892 mean BACC and 0.9948 retention
  vs source-union K16, but failed because matched shuffled reliability,
  negative controls, and source-ablation dominance remained problematic.
- Mass-uncertainty bagged component union reached 0.8903 mean BACC and 0.9960
  retention vs source-union K16, but its delta vs random mass-bag control was
  -0.0016.
- Reliability shrink050 confirmation reached 0.8800 mean BACC but had verdict
  `ANCHOR_MISMATCH` and failed matched-null/random-mass separation.
- Support8 calibrated component union reached 0.8727 mean BACC, but was below
  shrink050 and random mass-bag and did not clear the shuffled-support null.
- Dense reliability tail shield reached 0.8988 mean BACC and improved bottom20
  vs random mass-bag by +0.0244, but failed because center3 stayed at 0.7896
  and worst seed-center BACC remained 0.4971.
- Multipanel tail-risk stabilization reached 0.9087 mean BACC, improved
  bottom20 vs prior tailrisk by +0.0408, and reduced seed std by -0.0079, but
  failed because center3/min-center stayed at 0.7897, center3 regressed by
  -0.0136 versus prior tailrisk, tail-risk transfer was flagged, and the worst
  seed-center BACC was 0.4975.

Interpretation:

```text
component-level composition has high generated-embedding capacity,
but source-mass allocation remains underidentified and weak-tail robustness is
not solved by reliability, support-NELBO, fixed probability blending, or
more predeclared random mass-bag panels.
```

Center3 failure audit:

```text
42 x center3 final v2 BACC: 0.4975
class counts: class0 = 198, class1 = 2
final predicted counts: class0 = 199, class1 = 1
final class1 recall: 0.0000
mean confidence: 0.9795
seed101 anchor BACC: 0.9949
seed101 blend BACC: 0.7475
seed127 blend BACC: 0.7323
component mass coverage: 1.0
unsampled active components: 0
```

Interpretation:

The primary center3 failure is not simply missing component coverage or
insufficient panel diversity. It is a rare-positive, high-confidence
majority-class collapse where probability pooling suppresses rare useful
seed-level signal.

## Scientific Audit Notes

D1.5 surfaced two implementation/scientific audit issues:

1. `top4` diagnostic and equal all4 use the same source set and budgets, but
   produce different generated feature hashes and different BACC. Future
   selector comparisons need method-invariant generation/prediction bundles for
   identical source sets.
2. `source_drop_frequency_summary.csv` has
   `dropped_source_target_utility_rank = nan` for selected rows. Repair before
   using that field in thesis prose.

These issues do not create a false D1.5 failure. They motivated the paired
dense all4 confirmation, which enforced method-invariant generated/prediction
bundles for identical source sets and budgets.

## Implication

The thesis should not claim that the D-series has solved target-conditioned
compatibility routing or sparse expert selection. The best current
generated-embedding claim is narrower:

```text
Heldout-excluded source-local reliability improves dense all-source
generated-embedding aggregation over equal all-source aggregation under paired
generation and prediction invariants.
```

The component-union follow-ups add a second, diagnostic claim:

```text
source-local component union can approach centralized source-union K16 mean
utility, but current mass-allocation signals are not clean enough to be adopted
as compatibility estimators.
```

The multipanel tail-risk follow-up adds a third, more specific diagnostic
claim:

```text
source-only dense stochastic composition can cross 0.90 mean BACC, but
rare-positive weak-center collapse can survive probability-level pooling.
```

## Limitations

- D1.2 is partial evidence; the paired dense all4 confirmation is the stronger
  dense aggregation result.
- D1.3/D1.3.1 are not validated support-NELBO routing wins.
- D1.5 remains negative evidence for source-inner sparse selection.
- Component-union/random mass-bag rows are high-mean diagnostic surfaces, not
  final routing methods.
- The multipanel Center3 audit is diagnostic-only and cannot be used to select
  seeds, thresholds, calibration, or pooling policy.
- Support8 component-union calibration used target support and still failed
  matched shuffled-support controls.
- The paired dense all4 confirmation has 14 eligible seed-center cells because
  seed 43 / heldout center 2 was excluded as mono-class target evaluation.
- Source-union K16 remains centralized and diagnostic.
- None of these experiments make a formal differential privacy claim.

## Next Checks

- Reuse the paired generation-cache invariant audit before any new sparse
  selector confirmation.
- Keep equal all4 and paired reliability-weighted dense aggregation as fixed
  generated-embedding baselines.
- Treat support-NELBO and source-inner transfer as diagnostic/negative until
  they beat matched controls.
- Treat random mass-bag and multipanel bagging as strong controls/surfaces with
  little mean-BACC headroom.
- Predeclare any Center3 follow-up; target source-only calibration,
  minority-class decision stability, or pooling preservation of rare useful
  seed-level evidence.
- Validate harmful-source suppression only if final artifacts are synced.
- Repair D1.5 drop-rank fields if the source-drop tables are reused.
