# Virchow2-Only Rationale

## Purpose

Explain why the active SAIL instantiation is Virchow2-only rather than cross-backbone.

## Key Claims

- R1.2b identifies Virchow2 as the top source-inner-LODO selected backbone in local artifacts.
- Virchow2-only rows are the only rows that can justify a Virchow2 CVAE preservation test.
- Cross-backbone aggregation may improve performance through ensemble diversity but cannot prove one generative feature space is sufficient.
- The later Virchow2 CVAE artifacts confirm that generated-embedding utility is
  mainly bottlenecked by latent prior/composition rather than by abandoning
  Virchow2 as the feature space.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `../../../sail/configs/sail_virchow2.yaml`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_gap_summary.csv`

## Interpretation

Virchow2-only aggregation tests the feature space that a future Virchow2 CVAE would need to model. Cross-backbone aggregation tests ensemble diversity across incompatible feature spaces.

The D-series generated-embedding artifacts should be read as Virchow2
composition/preservation diagnostics. Source-union K16 demonstrates that
improved latent priors can recover utility in this feature space, while the
paired dense-all4 reliability confirmation shows that heldout-excluded
source-local reliability can improve dense generated-embedding aggregation.
Both results still leave the sparse routing and real-feature-reference gaps
separate from the Virchow2-only SAIL gate.

## Implication For Thesis

Rebuild readiness must come from Virchow2-only primary rows, not from mixed-backbone audit rows.

## Limitations

Virchow2 real-feature success does not prove Virchow2 CVAE utility preservation.

Source-union K16 success also does not prove decentralized expert routing.

Paired dense-all4 reliability success also does not prove sparse expert
selection, because every non-target source remains included.

## Next Checks

- Verify the SAIL Virchow2-only gate after output artifacts exist.
- If Virchow2-only fails, inspect whether failure is weak-center, seed, or aggregation-rule specific.
- Keep cross-backbone and source-union rows out of deployable CVAE routing
  claims.
