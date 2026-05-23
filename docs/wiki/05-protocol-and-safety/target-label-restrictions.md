# Target-Label Restrictions

## Purpose

Record the target-label firewall used in documentation and experiment interpretation.

## Key Claims

Target evaluation labels may only be used for final scoring.

Target evaluation labels must not choose:

- backbone
- representation
- PCA dimension
- classifier hyperparameters
- class weight
- CVAE checkpoint
- source expert
- routing method
- k
- aggregation rule
- calibration rule
- decision threshold

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `../../../sail/configs/sail_virchow2.yaml`

## Interpretation

Rows that use target labels to choose settings are posthoc or audit-only, even if their final performance is high.

## Implication For Thesis

The thesis must separate feasibility ceilings from deployable/source-only methods.

## Limitations

This page states the rule. Each experiment still needs its own leakage evidence.

## Next Checks

- Require leakage reports in final result tables.
- Add target-label restriction checks to every current-approach page as artifacts appear.
