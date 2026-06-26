# Current Experimental State

Last updated: 2026-06-22

## Purpose

This file records the changing experimental synthesis for the thesis
repository. It should be updated when new runs, synced artifacts, decision
reports, or implementation plans change the current interpretation.

Stable framing belongs in:

```text
docs/context/thesis_project_context.md
```

## Evidence Status

Labels used below:

- Verified artifact: read from a local report, table, config, source file, or
  test in this repository, including synced workstation artifacts under
  `cvae_rebuild/artifacts/`.
- Existing synthesis note: read from a local narrative note, but not necessarily
  independently revalidated against raw artifacts in this update.
- Provided synthesis: supplied in the current user prompt; verify against
  artifact if available.
- TODO: missing or unverified evidence.

## Current Readout

There are now two separate current-state surfaces:

```text
real-feature source-only compatibility / aggregation
generated-embedding CVAE preservation / decentralized composition
```

Real-feature surface:

```text
R1.2b Virchow2 source-selected evidence
-> SAIL Virchow2 dense source-selected config aggregation
-> SAIL artifacts still TODO locally
```

Generated-embedding CVAE surface:

```text
Virchow2 CVAE repair showed decoder/source-pool capacity can preserve utility
-> source-union K16 GMM is the strongest centralized prior diagnostic
-> decentralized D-series summary-exchange variants gave partial evidence
-> paired dense-all4 reliability confirmation passes as dense aggregation
-> component-union and random mass-bag audits reach high mean BACC
-> multipanel tail-risk mass-bagging reaches >0.90 mean BACC but fails
   weak-center gates
-> mass allocation, weak-center robustness, and minority-class confidence
   collapse remain the current bottlenecks
```

Current best generated-embedding interpretation:

```text
Best diagnostic upper bound:
  centralized source-union class-conditional diagonal GMM K16

Best clean source-quality / dense aggregation evidence:
  paired dense-all4 heldout-excluded reliability confirmation

Best high-mean component-composition surface:
  component-union / random mass-bag / multipanel tail-risk family,
  diagnostic unless controls are beaten and weak-center failures are repaired

Not currently supported as final thesis-facing winners:
  support-NELBO weighting
  reliability-only sparse top-3
  source-inner off-diagonal drop-one selection
  point source-mass reliability shrinkage
  random mass-bag as meaningful compatibility by itself
```

No D-series experiment currently supports a full PASS claim for deployable
decentralized compatibility routing or sparse expert selection. The strongest
safe generated-embedding claims are narrower:

```text
1. heldout-excluded source-local reliability improves dense all-source
   generated-embedding aggregation over equal all-source aggregation under
   paired generation and prediction invariants

2. component-union/random mass-bag composition can reach the source-union K16
   utility region, but current mass-allocation signals are underidentified
   because random/shuffled controls remain competitive

3. weak-center and bottom-tail robustness, not mean BACC alone, is now the
   limiting generated-embedding bottleneck

4. the latest multipanel tail-risk run shows that source-only probability
   pooling can clear 0.90 mean BACC, but this is not thesis-facing success
   when center3/min-center remains below gate and tail-risk transfer appears
```

## Synced Artifact Root

The latest CVAE rebuild and D-series artifacts inspected in this update are
synced under:

```text
/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/
```

These artifacts are now present under this working repo's
`cvae_rebuild/artifacts/` directory.

The paired dense-all4 reliability confirmation artifact was synced into this
working repository under:

```text
cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/
```

## Verified Source Artifacts

Primary real-feature artifacts:

- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `sail/configs/sail_virchow2.yaml`
- `sail/src/sail/`
- `sail/tests/test_smoke.py`
- `PROTOCOL_STATUS.md`

Primary generated-embedding artifacts:

- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_preservation_repair_v1/reports/decision_summary.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_k24_gmm_prior_v1/tables/source_union_k24_gmm_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_k16_gmm_prior_v1/tables/decentralized_k16_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_adaptive_gmm_prior_v1/tables/decentralized_adaptive_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1/tables/decentralized_support_nelbo_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1/tables/decentralized_support8_top3_tau05_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1/tables/decentralized_reliability_top3_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_gap_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_delta_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_generation_invariant_audit.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_support8_calibrated_component_union_prior_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/tables/multipanel_tailrisk_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/tables/multipanel_tailrisk_paired_deltas.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/tables/multipanel_tailrisk_panel_disagreement.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/tables/multipanel_tailrisk_probability_invariants.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_conclusion.md`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_cell_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_pooling_path.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_source_weight_comparison.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_component_coverage_comparison.csv`
- `cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml`
- `cvae_rebuild/src/source_inner_harmful_source_suppression.py`

Earlier in-progress generated-embedding artifact:

- `cvae_rebuild/artifacts/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1/`

Status: local synced result reports/tables/manifests are not verified yet.
Treat harmful-source suppression as earlier implementation/run status, not
latest final evidence, unless the final rerun artifacts are later synced.

Historical and contextual artifacts:

- `cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `cvae_downstream_evaluation/artifacts/tables/r12_backbone_ranking.csv`
- `cvae_downstream_evaluation/artifacts/tables/r12_center_summary.csv`
- `cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `cvae_testing/thesis_outline.txt`

## R1.2b Source-Only Compatibility Selector Audit

Verified artifact:
`cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`

Verified summary:

- Best posthoc target-eval mean BACC: 0.9481.
- Best source-inner-LODO selected mean BACC: 0.9155.
- Mean delta vs Z1.1 DINOv2/PCA64: 0.0931.
- Top source-selected backbone: `virchow2` with mean BACC 0.9155.
- Top posthoc audit backbone: `virchow2` with mean BACC 0.9474.
- Mean selector oracle gap: 0.0326.
- Median selected rank under target utility: 47.0.

Verified selector diagnostics from `r12b_selector_oracle_gap.csv`, primary
robust-penalty rows:

- Rows inspected: 15.
- Top-1 config match: 0/15.
- Top-3 contains oracle: 15/15.
- Mean selected target BACC: 0.9155.
- Mean oracle gap: 0.0326.
- Median selected rank under target utility: 47.0.
- Mean Spearman source-score vs target BACC: 0.4589.
- Seed mean BACC values: seed 42 = 0.8956, seed 43 = 0.8644, seed 44 = 0.9865.
- Seed mean-BACC sample std: 0.0635.

Interpretation:

The selector appears to identify a useful neighborhood of high-utility configs,
because the oracle is always in the top 3 in the inspected primary rows. Exact
top-1 selection is unstable, because top-1 never matches the oracle and some
seed/center rows collapse below the SAIL rebuild-stability floor.

This remains the real-feature motivation for dense top-k aggregation.

## SAIL Status

Verified artifacts:

- `sail/configs/sail_virchow2.yaml`
- `sail/src/sail/`
- `sail/tests/test_smoke.py`
- `cvae_downstream_evaluation/legacy/superseded_by_sail/r12c/README.md`

Current status:

```text
SAIL is the active real-feature implementation.
R1.2c-V code/config/tests are archived as provenance.
SAIL output artifacts are still missing locally.
```

Missing expected artifact root:

```text
sail/artifacts/virchow2_dense_source_selected
```

TODO: run or sync SAIL artifacts and verify:

- `tables/source_lodo_selection_matrix.csv`
- `tables/source_k_selection_matrix.csv`
- `tables/dense_aggregation_matrix.csv`
- `tables/member_manifest.csv`
- `tables/center_summary.csv`
- `manifests/protocol_manifest.json`
- `reports/leakage_report.json`
- `reports/decision_report.md`

## Virchow2 CVAE Preservation And Prior Diagnostics

Verified synced artifacts under:
`/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/`

Summary:

| Experiment | Status | Key result | Interpretation |
| --- | --- | ---: | --- |
| `virchow2_cvae_preservation_diagnosis_v1` | leakage PASS | mean CVAE prior BACC 0.5425 | vanilla prior is a bottleneck |
| `virchow2_cvae_preservation_repair_v1` | `REPAIR_PARTIAL` | decode(mu) mean BACC 0.8572 | decoder/source-pool capacity exists, but this is not deployable prior sampling |
| `virchow2_cvae_latent_prior_calibration_v1` | `PRIOR_CALIBRATION_PARTIAL` | calibrated prior mean BACC 0.7436 | calibration helps but does not solve preservation |
| `virchow2_cvae_pca64_sampling_continuation_v1` | insufficient decision rows | prior mean BACC 0.7011; posterior/decode around 0.85 | sampling/prior remains bottleneck |
| `virchow2_cvae_covariance_prior_confirmation_v1` | `COVARIANCE_PRIOR_PARTIAL` | covariance prior mean BACC 0.7908 | improves standard/diag references, but unstable |
| `virchow2_cvae_covariance_shrinkage_stability_v1` | `SHRINKAGE075_PARTIAL` | center-equal mean BACC 0.8077 | shrinkage helps but tail failures remain |

Interpretation:

Virchow2 embeddings were not the only bottleneck. The CVAE decoder can preserve
utility under favorable latent inputs, but prior sampling is the limiting
factor. This motivated source-union and decentralized GMM prior experiments.

## Source-Union GMM Prior Diagnostics

Verified synced artifacts:

- `virchow2_cvae_source_union_gmm_prior_v1`
- `virchow2_cvae_source_union_center_balanced_gmm_prior_v1`
- `virchow2_cvae_source_union_k24_gmm_prior_v1`

Key results:

| Row / experiment | Mean BACC | Status / caveat |
| --- | ---: | --- |
| `source_union_cc_diag_gmm_k16_prior_sample_diagnostic` | 0.8924 | strongest centralized source-union diagnostic row |
| `source_union_cc_diag_gmm_k24_prior_sample` | 0.8751 | K24 underperforms K16 and has component-undersampling caveat |
| `source_union_center_balanced_cc_diag_gmm_k16_prior_sample` | 0.8589 | center-balanced K16 underperforms vanilla K16 |
| source-union GMM primary K8 summary | 0.8241 | primary verdict `GMM_FIT_INELIGIBLE` |

Interpretation:

The source-union K16 GMM prior is the strongest generated-embedding utility
diagnostic so far. It is not the thesis-facing deployable method because it
uses centralized source-union fitting and does not implement decentralized
expert routing or summary exchange.

Safe claim:

```text
The K16 source-union GMM diagnostic shows that improved latent prior sampling
can recover much of Virchow2 generated-embedding downstream utility.
```

Unsafe claim:

```text
Centralized source-union K16 is a deployable MoErging/routing method.
```

## D-Series Decentralized Composition Results

Verified synced artifacts under:
`/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/`

| Experiment | Primary method | Verdict | Mean BACC | Min center | Seed std | Main conclusion |
| --- | --- | --- | ---: | ---: | ---: | --- |
| D1 strict K16 | `decentralized_exported_k4x4_cc_diag_gmm_k16_late_geom` | `INELIGIBLE` | 0.8806 | 0.8704 | 0.0384 | High retention but fixed K4 fit ineligible |
| D1.1 adaptive K | `decentralized_exported_adaptive_k_cc_diag_gmm_late_geom` | `D1_1_PARTIAL_EVIDENCE` | 0.8143 | 0.7380 | 0.0681 | Restores eligibility but loses centralized retention/stability |
| D1.2 reliability weighted | `decentralized_exported_adaptive_k_source_reliability_weighted_geom` | `D1_2_PARTIAL_EVIDENCE` | 0.8493 | 0.8013 | 0.0382 | Best clean decentralized preservation/reliability evidence so far |
| D1.3 support-NELBO x reliability | `decentralized_exported_adaptive_k_support_nelbo_x_reliability_weighted_geom` | `D1_3_PARTIAL_EVIDENCE` | 0.8495 | 0.7262 | 0.0297 | Utility improves on support-eval subset, but alignment/stability weak |
| D1.3.1 support8 top3 tau0.5 | `decentralized_support8_top3_tau05_support_nelbo_x_reliability_geom` | `D1_3_1_WEAK_PASS` | 0.8310 | 0.7804 | 0.0450 | Support-NELBO alignment improves, but shuffled-support control is competitive |
| D1.4 reliability top3 | `decentralized_reliability_top3_geom_confirmation` | `D1_4_DIAGNOSTIC_ONLY` | 0.8212 | 0.7529 | 0.0498 | Sparse reliability top3 does not beat equal all4 and controls |
| D1.5 source-inner drop-one | `decentralized_source_inner_transfer_top3_geom_confirmation` | `D1_5_FAIL` | 0.8354 | 0.7092 | 0.0015 | Source-inner transfer does not predict target subset utility and loses to equal all4 |
| Paired dense all4 reliability confirmation | `paired_reliability_all4_weighted_geom` | `PAIRED_DENSE_ALL4_RELIABILITY_PASS` | 0.8506 | 0.8173 | 0.0308 | Heldout-excluded reliability improves dense all-source aggregation under paired invariants |

Important D-series diagnostics:

- D1 strict K16 retention vs source-union K16: 0.9925, but primary is
  ineligible due component-fit failure.
- D1.2 delta vs D1.1 equal adaptive geom: +0.0211.
- D1.2 source-reliability vs target single-source Spearman: 0.0461, so
  reliability behaves as a source-quality prior, not target compatibility.
- D1.3 support-NELBO vs downstream Spearman: 0.2228, but top-2 oracle
  containment is 0.6429 and Spearman stability is weak.
- D1.3.1 support-NELBO Spearman: 0.3143 and top-3 oracle containment: 0.8571,
  but shuffled-support control beats the primary by 0.0096 BACC.
- D1.4 random-source-drop and shuffled-reliability controls are competitive.
- D1.5 source-inner score vs target subset utility Spearman: -0.1102.
- D1.5 shuffled-score control beats the primary by 0.0326 BACC.
- Paired dense all4 reliability confirmation:
  - decision report primary method: `paired_reliability_all4_shrink050_geom`.
  - best reliability method: `paired_reliability_all4_weighted_geom`.
  - leakage report: `PASS`.
  - paired generation invariant audit: 420/420 rows `PASS`.
  - equal all4 center-equal BACC: 0.8235.
  - full reliability-weighted center-equal BACC: 0.8506.
  - delta vs equal all4: +0.0271 center-equal BACC.
  - delta vs strongest negative control: +0.0416.
  - positive paired cells: 9/14.
  - centers improved vs equal all4: 4/5.
  - paired bootstrap CI for mean delta: [-0.0098, 0.0764].
  - gap vs real-feature dense reference remains -0.0570 BACC.

Interpretation:

The paired dense all4 confirmation supersedes D1.2 as the best protocol-clean
generated-embedding dense aggregation result. It supports source-local
reliability as a useful dense aggregation compatibility proxy under paired
generation and prediction invariants. It does not support sparse expert
selection or target-specific compatibility routing.

D1.3 and D1.3.1 do not validate support-NELBO as the final target-conditioned
compatibility signal because shuffled-support controls remain too competitive.

D1.5 is negative evidence against source-inner off-diagonal transfer as
implemented here.

## D1.5 Scientific Audit Notes

Verified artifact:
`/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/`

Protocol status is clean:

- leakage report: `PASS`
- target expert excluded
- no target support features or labels
- target eval labels scoring-only
- source-inner transfer matrix has no self-transfer rows
- heldout target rows are not used for source-inner scoring

Empirical status:

- primary mean BACC: 0.8354
- equal all4 reference: 0.8519
- reliability top3 reference: 0.8355
- shuffled-score control: 0.8680
- source-inner Spearman vs target subset utility: -0.1102

Scientific/audit issues to fix before any D1.5-style rerun:

1. `top4` diagnostic is structurally equivalent to equal all4, but does not
   reproduce equal all4 because method-specific synthetic sampling seeds change
   generated features. Future subset-selection confirmations need paired,
   method-invariant generated prediction bundles for identical source sets.
2. `source_drop_frequency_summary.csv` has `dropped_source_target_utility_rank`
   as `nan` for selected rows. The rank field should be repaired if the table
   is used for thesis diagnostics.
3. Seed 43 is fully ineligible in D1.5 due mono-class source-inner/target eval
   cells, leaving only 10 eligible seed-center cells.

These audit issues do not overturn the fail verdict; they make the negative
interpretation more conservative.

## Component-Union And Source-Mass Follow-Up Audits

Verified local artifacts:

- `cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_support8_calibrated_component_union_prior_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1/reports/decision_summary.md`
- `cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`

Summary table:

| Artifact | Primary method | Verdict | Mean BACC | Min center | Seed std | Key interpretation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| component-union shrink025 v2 | `decentralized_component_union_reliability_shrink025` | `COMPONENT_UNION_FAIL` | 0.8892 | 0.8168 | 0.0501 | high utility, but matched shuffled reliability and source-ablation dominance block adoption |
| source-inner dense/component hybrid | `source_inner_validated_dense_component_binary_gate` | `HYBRID_FAIL` | 0.8103 | 0.7192 | 0.0424 | source-inner gate fails to identify when component union is safe |
| mass-uncertainty bagged component union | `decentralized_component_union_mass_uncertainty_bagged_v1` | `MASS_BAGGED_COMPONENT_UNION_FAIL` | 0.8903 | 0.7931 | 0.0568 | high mean and source-union retention, but random mass-bag and negative controls are competitive |
| reliability shrink050 confirmation | `decentralized_component_union_reliability_shrink050` | `ANCHOR_MISMATCH` | 0.8800 | 0.8000 | 0.0527 | weak reliability prior is not separated from matched shuffled/random controls |
| support8 calibrated component union | `support8_calibrated_component_union_softmax_shrink050` | `SUPPORT_CALIBRATED_COMPONENT_UNION_FAIL` | 0.8727 | 0.7886 | 0.0369 | unlabeled target support does not beat shrink050/random mass-bag or shuffled-support null |
| shrink050/random mass tail-risk blend | `component_union_tailrisk_anchored_shrink050_random_mass_bag_blend050` | `TAILRISK_ANCHORED_COMPONENT_UNION_USEFUL_THESIS_SUCCESS` | 0.8957 | 0.8032 | 0.0510 | useful robustness evidence, but anchor mismatch and bottom-tail limits remain |
| dense reliability tail shield | `dense_reliability_tailshield_random_mass_bag_blend25_75` | `DENSE_TAILSHIELD_RANDOM_MASS_BAG_FAIL` | 0.8988 | 0.7896 | 0.0403 | high mean and bottom20 gain, but center3/worst-cell failure remains |
| multipanel tail-risk mass-bag stabilization | `component_union_tailrisk_multipanel_shrink050_random_mass_bag_blend050` | `MULTIPANEL_TAILRISK_STABILIZATION_FAIL` | 0.9087 | 0.7897 | 0.0431 | clears 0.90 mean and improves bottom20/seed std, but center3/min-center fail and tail-risk transfer appears |

Important verified numbers:

- Mass-bagged component union retention vs source-union K16: 0.9960.
- Mass-bagged oracle gap vs source-union K16: 0.0035.
- Mass-bagged delta vs random mass-bag control: -0.0016.
- Support8 calibrated primary minus shuffled-support null mean: -0.0050.
- Dense tail shield bottom20 delta vs random mass-bag: +0.0244, but center3
  delta vs random mass-bag: 0.0000 and worst seed-center BACC: 0.4971.
- Multipanel tail-risk stabilization delta vs prior tailrisk: +0.0130, delta
  vs canonical random mass-bag: +0.0103, bottom20 delta vs prior tailrisk:
  +0.0408, seed std delta vs prior tailrisk: -0.0079, but center3/min-center
  delta vs prior tailrisk: -0.0136 and worst seed-center BACC: 0.4975.
- Multipanel leakage report: `PASS`; protocol manifest states no target
  support, no target-label selection, source-inner calibration primary, and
  target evaluation labels scoring/audit only.

Interpretation:

The component-union family shifts the bottleneck. Mean BACC is no longer the
main limitation: several rows reach roughly the source-union K16 region. The
problem is that source-mass allocation is underidentified. Random or shuffled
mass controls often match the proposed primary method, and weak-center/tail
failures remain.

The multipanel tail-risk run sharpens this conclusion. It is the first
source-only full-matrix CVAE/component-union variant in this record to clear
0.90 center-equal mean BACC, but it still fails the locked stabilization claim
because the weak center is not repaired. The result supports a high-capacity
composition claim, not method adoption.

Safe claim:

```text
Component-level source-local summaries expose a high-utility generated
embedding composition surface, but current source-only or support-calibrated
mass-allocation signals do not yet provide a clean deployable compatibility
rule.
```

Unsafe claim:

```text
Random mass-bag success proves random source weights are meaningful
compatibility estimates.
```

## Center3 Multipanel Failure Audit

Verified audit artifacts:

- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_conclusion.md`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_cell_summary.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_pooling_path.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_source_weight_comparison.csv`
- `cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_component_coverage_comparison.csv`

Protocol label:

```text
diagnostic / audit-only; target labels are post-prediction scoring and failure
analysis only
```

Assigned failure mode for `experiment_seed=42 x heldout_center=3`:

```text
near_class_collapse | probability_pooling_suppresses_best_seed |
confident_wrong_predictions
```

Verified audit details:

- Final v2 BACC in `42 x center3`: 0.4975.
- Target eval class counts: class0 = 198, class1 = 2.
- Final predicted counts: class0 = 199, class1 = 1.
- Final class1 recall: 0.0000, mean confidence: 0.9795, mean margin: 0.9590.
- Pooled anchor and pooled random mass-bag also score 0.4975 BACC.
- Canonical random mass-bag scores 0.5000 BACC by predicting class0 for all
  200 samples.
- Seed 101 anchor BACC: 0.9949; seed 101 blend BACC: 0.7475.
- Seed 127 blend BACC: 0.7323.
- All panel/final pools suppress the rare useful seed-level minority-class
  signal.
- Panel disagreement at `42 x center3` is low, with mean pairwise JS divergence
  0.0019 and hard-label disagreement 0.0033, so the panels mostly agree on the
  wrong majority-class decision.
- Component coverage does not explain the failure: the primary failed cell has
  full component mass coverage and no unsampled active components.
- The `44 x center3` control reaches 0.9823 BACC with class counts class0 = 87
  and class1 = 113, so center3 is not globally impossible; the observed failure
  is a specific rare-positive/seed-regime collapse.
- The `43 x center4` tail-repair control shows the same machinery can improve
  a weak-tail cell, reaching 0.7923 BACC versus 0.5923 for pooled anchor and
  0.6923 for pooled random mass-bag.

Interpretation:

The failed center3 cell is not mainly explained by stochastic panel diversity
or component undersampling. The evidence points to a systematic minority-class
decision-boundary or calibration failure under extreme class imbalance. More
random panels alone are unlikely to solve this unless the fixed, source-only
method also changes how rare useful seed-level minority-class evidence is
calibrated or pooled.

## Earlier Harmful-Source Suppression Run Status

Implementation artifacts:

- `cvae_rebuild/src/source_inner_harmful_source_suppression.py`
- `cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml`

User-provided run status, not final artifact evidence:

- Workstation run wrote 12,602 generated caches and 12,602 prediction caches
  before `systemd-oomd` killed the tmux unit.
- No final reports/tables/manifests were available at that point.
- The run is being retried with BLAS/thread limits and unbuffered logging.

Earlier purpose:

```text
test whether source-inner leave-one-source harmfulness can suppress sources
that poison target-like regimes before heldout target evaluation
```

Evidence label:

```text
implemented/running at prior update; final result TODO: verify against artifact
if this line is resumed
```

## Current Best Approach

Current best real-feature approach:

```text
SAIL Virchow2 dense source-selected config aggregation
```

Status: implemented, but output artifacts still TODO locally.

Current best generated-embedding diagnostic:

```text
centralized source-union K16 GMM prior
```

Status: strong diagnostic upper bound, not deployable.

Current best clean decentralized generated-embedding dense aggregation evidence:

```text
paired dense all4 heldout-excluded source-local reliability weighting
```

Status: PASS for dense generated-embedding aggregation only; not a sparse
routing or target-conditioned compatibility-router PASS.

Current best generated-embedding mean-utility surface:

```text
component-union / random mass-bag / multipanel tail-risk probability ensembles
```

Status: high-mean diagnostic surface, not adopted as a compatibility method
because matched controls are competitive and weak-center/tail failures remain.

Current generated-embedding bottleneck:

```text
source-mass allocation is underidentified;
weak-center/tail robustness is unresolved;
minority-class confidence collapse can survive probability pooling.
```

Current rejected or downgraded directions:

- strict D1 K16 as primary: ineligible fixed K4 component fits
- D1.3/D1.3.1 support-NELBO as final selector: controls too competitive
- D1.4 reliability-only sparse top3: diagnostic-only, controls competitive
- D1.5 source-inner transfer drop-one: fail
- component-union shrink025/shrink050 as source-reliability proof: controls
  remain competitive
- source-inner dense/component binary gate: fails and underuses component union
- support8 calibrated component-union prior: shuffled-support/random mass
  controls are competitive
- dense reliability tail shield: high mean but fails center3/worst-cell repair
- multipanel tail-risk mass-bag stabilization: mean BACC exceeds 0.90, but
  center3/min-center fail, tail-risk transfer is flagged, and audit shows
  confident rare-class collapse in `42 x center3`
- K24 source-union GMM: weaker than K16 and component undersampled
- metadata-only routing: baseline/proxy, not current winner

## Claim Boundaries

Supported by current evidence:

```text
Virchow2 is a strong pathology feature space for source-selected real-feature
transfer diagnostics.
```

```text
Virchow2 CVAE decoder/source-pool capacity can preserve substantial utility
when latent inputs are favorable.
```

```text
Latent prior sampling is a central bottleneck for Virchow2 generated-embedding
utility.
```

```text
Source-local adaptive latent summaries plus source-local reliability weighting
improve dense generated-embedding aggregation over equal all-source aggregation
under paired generation and prediction invariants.
```

```text
Component-union/random mass-bag generated-embedding composition can approach
or exceed the centralized K16 diagnostic region in mean BACC, but this is
evidence of composition capacity and source-mass underidentification, not a
validated compatibility estimator.
```

```text
Fixed source-only robustness aggregation can improve mean, bottom-tail, and
seed-stability metrics, but the latest multipanel artifact did not solve
center3/min-center failure.
```

```text
Audit-only evidence suggests that the worst center3 failure is a confident
minority-class collapse under extreme class imbalance, not merely component
undersampling or insufficient panel diversity.
```

Not supported by current evidence:

```text
Support-NELBO is a validated target-conditioned compatibility router.
```

```text
Source-inner off-diagonal transfer is a reliable drop-one source selector.
```

```text
Centralized source-union K16 is deployable.
```

```text
The thesis has a full PASS decentralized generated-embedding routing method.
```

```text
Formal privacy preservation.
```

```text
Random mass-bag control performance proves a meaningful routing signal.
```

```text
Unlabeled support-NELBO calibration is currently a validated target-support
mass allocator for component union.
```

## Current Next Sequence

1. Do not extend D1.5 source-inner transfer directly as a top-k/drop-one
   selector.

2. The paired dense all4 reliability confirmation has implemented the required
   paired generation/prediction invariant audit:

   ```text
   same source set + same budgets + same replicate seed
   -> same generated/prediction bundle regardless of method label
   ```

   This prevents method-specific random seeds from confounding dense
   aggregation comparisons. Reuse the same invariant design for any future
   sparse selector confirmation.

3. Repair the D1.5 drop-rank audit field if source-drop tables are reused.

4. Use equal all4, D1.2 reliability-weighted, D1.3 support-eval reference, and
   source-union K16 as fixed baselines. Do not compare against historical
   full-target rows when support/eval subsets differ.

5. Treat the paired dense all4 reliability result as the clean
   generated-embedding dense aggregation success. Treat component-union/random
   mass-bag rows as high-mean diagnostic surfaces unless they beat matched
   random/shuffled controls and repair weak-center/tail failures.

6. Keep centralized source-union K16 as a CVAE prior-preservation diagnostic
   upper bound only.

7. Do not treat more random panels as the default next step. The latest
   multipanel audit suggests the next generated-embedding work should target
   source-only calibration, minority-class decision stability, and pooling
   rules that preserve rare useful seed-level evidence without target-label
   seed selection.

8. Any Center3 follow-up method must be predeclared separately. The current
   audit is target-label-informed after predictions and cannot be used to
   choose seeds, thresholds, calibration, routing, or method policy.

## Missing Artifacts / TODOs

- TODO: run or sync SAIL output artifacts under
  `sail/artifacts/virchow2_dense_source_selected`.
- TODO: verify whether SAIL primary rows pass the rebuild gate.
- TODO: reuse paired generation-cache/invariant checks if any further D-series
  sparse selector confirmation is implemented.
- TODO: repair D1.5 `dropped_source_target_utility_rank` if the drop-frequency
  table is used in thesis writing.
- TODO: verify C6.3 numerical synthesis against raw/synced decision artifacts
  before using it as a final thesis result table.
- TODO: sync and validate final
  `virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1`
  reports/tables/manifests if that earlier run is resumed.
- TODO: if harmful-source suppression fails, document whether the failure is
  source-inner signal non-transfer, insufficient harmfulness precision, or a
  need for target-regime information.
- TODO: predeclare any Center3 follow-up before evaluation; do not use the
  audit-only row-level target labels to tune method choices.
