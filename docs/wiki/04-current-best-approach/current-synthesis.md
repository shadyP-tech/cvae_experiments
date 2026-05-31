# Current Synthesis

## Purpose

State the current result synthesis across the real-feature SAIL surface and the
generated-embedding Virchow2 CVAE D-series surface.

## Key Claims

- 0.90 BACC was probably not supported by the older DINOv2/PCA64 setup. Provided synthesis; verify against artifact if available.
- 0.90 appears supported by pathology embeddings, especially Virchow2.
- It is not yet robustly deployable because weak-center and seed instability remain.
- The immediate bottleneck is not another backbone screen.
- The real-feature bottleneck is source-selected stability and later CVAE preservation.
- The generated-embedding bottleneck is latent prior/composition rather than
  Virchow2 feature quality alone.
- The active current method is SAIL: Source-only Aggregation via Inner-domain Leaveout.
- The strongest generated-embedding diagnostic is centralized source-union K16.
- The cleanest decentralized generated-embedding dense aggregation result is
  the paired dense all4 heldout-excluded reliability confirmation.
- Component-union/random mass-bag rows now provide the strongest high-mean
  generated-embedding surface, but not an adopted compatibility method.
- Support-NELBO, source-inner transfer, and point source-mass reliability
  calibration are not validated final selectors.
- The current generated-embedding bottleneck is weak-center/tail robustness and
  harmful source interaction.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_delta_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_generation_invariant_audit.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_support8_calibrated_component_union_prior_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`

## Interpretation

### Real-Feature Layer

R1.2b supports a Virchow2 real-feature path: source-inner-LODO selected Virchow2 reaches mean BACC 0.9155, with 4/5 centers at or above 0.85. It does not pass the stability spirit needed for immediate CVAE rebuild because the inspected primary rows include seed/center failures below 0.75 and seed mean-BACC sample std 0.0635.

Selector insight:

| Diagnostic | R1.2b primary rows |
| --- | ---: |
| Top-1 oracle match | 0/15 |
| Top-3 contains oracle | 15/15 |
| Median selected rank | 47.0 |
| Mean Spearman | 0.4589 |

The selector knows a useful neighborhood but has poor exact top-1 precision.

SAIL is the current implementation of that diagnostic direction. It tests
whether top-k source-selected config aggregation can convert useful neighborhood
information into stable held-out utility without using target-eval labels for
selection.

### Generated-Embedding Layer

The Virchow2 CVAE rebuild sequence shows:

```text
vanilla prior failed
-> decoder/source-pool capacity partially rescued utility
-> source-union K16 GMM diagnosed the prior bottleneck
-> decentralized reliability-weighted summaries partially preserved utility
-> paired dense all4 reliability weighting passed as dense aggregation
-> component-union/random mass-bagging reached high mean utility but exposed
   source-mass underidentification and weak-tail failures
```

Key verified generated-embedding numbers:

| Evidence | Mean BACC | Claim status |
| --- | ---: | --- |
| source-union K16 diagnostic | 0.8924 | centralized diagnostic upper bound |
| D1 strict decentralized K16 | 0.8806 | ineligible fixed K4 primary |
| D1.2 reliability weighted | 0.8493 | prior decentralized partial evidence |
| D1.3 support-NELBO x reliability | 0.8495 | partial; controls/stability limit claim |
| D1.5 source-inner transfer | 0.8354 | fail; shuffled-score control beats primary |
| paired dense all4 reliability weighted | 0.8506 | dense aggregation PASS; not sparse routing |
| component-union shrink025 v2 | 0.8892 | high utility but fail; matched null/control/source-ablation issues |
| mass-uncertainty bagged component union | 0.8903 | high utility but fail; random mass-bag control competitive |
| support8 calibrated component union | 0.8727 | fail; shuffled-support and random mass-bag controls competitive |
| dense reliability tail shield | 0.8988 | fail; high mean but center3/worst-cell failure remains |

## Implication For Thesis

The thesis should present a two-layer story:

```text
SAIL / R1.2b:
  real-feature source-only aggregation and representation-transfer evidence

D-series:
  generated-embedding CVAE preservation and decentralized composition evidence
```

The current generated-embedding evidence supports a protocol-clean dense
source-local reliability aggregation claim and a high-capacity component-union
diagnostic surface. It does not yet support a final target-conditioned
compatibility router, sparse expert selector, or clean source-mass allocator.

## Limitations

SAIL output artifacts are not present locally yet. TODO: verify against artifact.

The D-series artifacts are synced under this working repo's
`cvae_rebuild/artifacts/` directory. Keep artifact paths explicit when citing
results.

D1.5 found a paired-sampling audit issue: identical source sets can get
different generated feature hashes across method labels. The paired dense all4
confirmation fixed this for dense all4 comparisons; future selector claims need
the same method-invariant generation/prediction bundle discipline.

The latest harmful-source suppression run is implemented but not yet a final
result. User-provided run status indicates the first workstation attempt was
OOM-killed after cache creation and is being rerun with memory constraints.
TODO: verify against final artifact.

## Next Checks

- Run or sync SAIL with `sail/configs/sail_virchow2.yaml`.
- Verify mean BACC, worst center, seed mean-BACC std, and no-seed worst-center floor.
- Do not extend D1.5 directly.
- Reuse paired generation-cache invariants before any further
  generated-embedding selector confirmation.
- Treat component-union/random mass-bag as the high-mean diagnostic surface,
  not as a final method unless controls are beaten and weak-tail failures are
  repaired.
- Validate source-inner harmful-source suppression once final reports are
  synced.
