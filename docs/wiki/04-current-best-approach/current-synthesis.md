# Current Synthesis

## Purpose

State the current result synthesis and the active SAIL approach.

## Key Claims

- 0.90 BACC was probably not supported by the older DINOv2/PCA64 setup. Provided synthesis; verify against artifact if available.
- 0.90 appears supported by pathology embeddings, especially Virchow2.
- It is not yet robustly deployable because weak-center and seed instability remain.
- The immediate bottleneck is not another backbone screen.
- The immediate bottleneck is source-selected stability and later CVAE preservation.
- The active current method is SAIL: Source-only Aggregation via Inner-domain Leaveout.
- R1.3a Vanilla Virchow2 CVAE Rebuild should only follow if SAIL passes the rebuild gate.

## Evidence / Source Artifacts

- `../../context/current_experimental_state.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_backbone_ranking.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_leakage_report.json`

## Interpretation

R1.2b supports a Virchow2 real-feature path: source-inner-LODO selected Virchow2 reaches mean BACC 0.9155, with 4/5 centers at or above 0.85. It does not pass the stability spirit needed for immediate CVAE rebuild because the inspected primary rows include seed/center failures below 0.75 and seed mean-BACC sample std 0.0635.

Selector insight:

| Diagnostic | R1.2b primary rows |
| --- | ---: |
| Top-1 oracle match | 0/15 |
| Top-3 contains oracle | 15/15 |
| Median selected rank | 47.0 |
| Mean Spearman | 0.4589 |

The selector knows a useful neighborhood but has poor exact top-1 precision.

## Implication For Thesis

SAIL is the current implementation of that diagnostic direction. It tests whether top-k source-selected config aggregation can convert useful neighborhood information into stable held-out utility without using target-eval labels for selection.

## Limitations

SAIL output artifacts are not present locally yet. TODO: verify against artifact.

## Next Checks

- Run or sync SAIL with `sail/configs/sail_virchow2.yaml`.
- Verify mean BACC, worst center, seed mean-BACC std, and no-seed worst-center floor.
- If SAIL fails, do not rebuild CVAEs based on archived cross-backbone audit ideas.
