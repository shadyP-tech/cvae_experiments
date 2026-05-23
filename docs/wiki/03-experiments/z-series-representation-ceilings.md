# Z-Series Representation Ceilings

## Purpose

Document representation-ceiling and synthetic-preservation audit context.

## Key Claims

- Z1.1 local report states real-feature ceiling rows were not available.
- The current user-provided synthesis says older DINOv2/PCA64 likely did not support 0.90 BACC.
- This older ceiling concern motivated pathology embedding screens.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/reports/z11_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/tables/z11_real_feature_ceiling_matrix.csv`
- `../../../cvae_downstream_evaluation/artifacts/tables/z11_synthetic_preservation_gap.csv`
- Provided synthesis; verify against artifact if available.

## Interpretation

Z-series evidence should be used conservatively. The local Z1.1 report marks real-feature evidence as missing, so the older DINOv2/PCA64 ceiling statement should be labeled as provided synthesis unless a verified artifact is added.

## Implication For Thesis

The thesis can say the empirical path moved beyond older generic/PCA representations because pathology embedding screens produced stronger verified evidence.

## Limitations

Do not quote older DINOv2/PCA64 numerical conclusions without verifying the relevant table.

## Next Checks

- Verify whether `z11_real_feature_ceiling_matrix.csv` contains usable rows.
- Add exact Z-series claim status after artifact validation.
