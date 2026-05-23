# Virchow2-Only Rationale

## Purpose

Explain why the active SAIL instantiation is Virchow2-only rather than cross-backbone.

## Key Claims

- R1.2b identifies Virchow2 as the top source-inner-LODO selected backbone in local artifacts.
- Virchow2-only rows are the only rows that can justify a Virchow2 CVAE preservation test.
- Cross-backbone aggregation may improve performance through ensemble diversity but cannot prove one generative feature space is sufficient.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `../../../sail/configs/sail_virchow2.yaml`

## Interpretation

Virchow2-only aggregation tests the feature space that a future Virchow2 CVAE would need to model. Cross-backbone aggregation tests ensemble diversity across incompatible feature spaces.

## Implication For Thesis

Rebuild readiness must come from Virchow2-only primary rows, not from mixed-backbone audit rows.

## Limitations

Virchow2 real-feature success does not prove Virchow2 CVAE utility preservation.

## Next Checks

- Verify the SAIL Virchow2-only gate after output artifacts exist.
- If Virchow2-only fails, inspect whether failure is weak-center, seed, or aggregation-rule specific.
