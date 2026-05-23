# Empirical Pivot

## Purpose

Explain the pivot from metadata-first routing to compatibility-driven routing and aggregation.

## Key Claims

- Earlier framing treated metadata as the likely primary routing signal.
- Current evidence supports treating metadata as a baseline, proxy, and interpretability signal.
- The current bottleneck has shifted from representation ceiling alone to source-selected stability and later CVAE preservation.
- SAIL is a diagnostic gate for Virchow2 real-feature transfer stability, not a final CVAE method.

## Evidence / Source Artifacts

- `../../context/pivot_statement.md`
- `../../context/current_experimental_state.md`
- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`

## Interpretation

Pathology foundation embeddings raised the real-feature transfer ceiling. The remaining failure is exact source-selected top-1 instability and the unproven question of whether CVAE generation preserves real-feature utility.

## Implication For Thesis

The next thesis-facing diagnostic is SAIL. A CVAE rebuild should follow only if the Virchow2 dense rows pass the rebuild gate.

## Limitations

Provided synthesis says older DINOv2/PCA64 likely did not support 0.90 BACC. Where not verified against an artifact, label this: `Provided synthesis; verify against artifact if available.`

## Next Checks

- Verify SAIL output artifacts once generated or synced.
- Create/verify R1.3a config only after SAIL passes.
