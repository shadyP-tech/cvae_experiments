# R-Series Real-Feature Routing

## Purpose

Summarize R-series pathology embedding and source-selected real-feature diagnostics.

## Key Claims

- R1.2 showed pathology embeddings improved the real-feature direction but weak-center behavior persisted.
- R1.2b made Virchow2 the strongest source-inner-LODO selected backbone in the local artifacts.
- R1.2b source-selected Virchow2 mean BACC is 0.9155 with worst center 0.8070 and 4/5 centers at or above 0.85.
- Exact top-1 source-selected config precision is poor, motivating dense top-k aggregation.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`

## Interpretation

The source selector has useful neighborhood information: in inspected primary rows, top-3 contains oracle is 15/15, while top-1 oracle match is 0/15. The selector is useful but too brittle for sparse deployment.

## Implication For Thesis

SAIL should test whether Virchow2-only top-k aggregation repairs top-1 brittleness enough to justify a CVAE preservation test.

## Limitations

R1.2b is real-feature diagnostic evidence. It does not show Virchow2 CVAE generation preserves utility.

## Next Checks

- Run or sync SAIL artifacts.
- Verify rebuild gate metrics from primary Virchow2-only dense rows.
