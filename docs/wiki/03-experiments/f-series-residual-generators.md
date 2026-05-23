# F-Series Residual Or Empirical Transfer Diagnostics

## Purpose

Record residual or empirical transfer generator diagnostics if present.

## Key Claims

- Residual routing decision tables exist in `cvae_testing/results/comparison_tables/`.
- Current project context treats residual/empirical transfer generators as relevant baselines or diagnostics, not the current best path.
- Adoption eligibility must be checked before any residual result is thesis-facing.

## Evidence / Source Artifacts

- `../../../cvae_testing/results/comparison_tables/residual_routing_decision_table.csv`
- `../../../cvae_testing/results/comparison_tables/residual_safe_v2_decision_table.csv`
- `../../../PROTOCOL_STATUS.md`

## Interpretation

Residual tables may provide useful negative or diagnostic evidence about routing variants, but the current empirical path is SAIL with the Virchow2 instantiation, followed by CVAE preservation testing if gates pass.

## Implication For Thesis

Use residual results as baselines or failure analyses only after validating protocol version, adoption eligibility, and claim boundaries.

## Limitations

The docs have not verified whether these residual artifacts correspond to an explicit F-series naming convention. TODO: verify against artifact.

## Next Checks

- Validate residual tables with the thesis artifact validator before final thesis use.
- Add exact conclusion once artifact status is confirmed.
