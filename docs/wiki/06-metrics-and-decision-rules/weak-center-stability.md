# Weak-Center Stability

## Purpose

Define weak-center and seed-stability checks.

## Key Claims

- Mean BACC can hide center-specific collapse.
- SAIL explicitly penalizes weak-center collapse through robust source-only scoring.
- CVAE rebuild readiness requires both mean utility and stability.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../sail/configs/sail_virchow2.yaml`

## Interpretation

R1.2b's worst center is 0.8070 despite mean BACC 0.9155. Inspected primary rows include seed/center BACC values below 0.75. This is why SAIL requires weak-center and seed checks.

## Implication For Thesis

The thesis should report worst-case performance and seed stability alongside mean performance.

## Limitations

Weak-center labels can be affected by class-balance caveats. Cite eval warnings when present.

## Next Checks

- Verify SAIL `center_summary.csv` when available.
- Track low-minority/single-class eval warnings.
