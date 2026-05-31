# Weak-Center Stability

## Purpose

Define weak-center and seed-stability checks.

## Key Claims

- Mean BACC can hide center-specific collapse.
- SAIL explicitly penalizes weak-center collapse through robust source-only scoring.
- CVAE rebuild readiness requires both mean utility and stability.
- The latest component-union audits make weak-center/tail metrics central:
  several methods reach high mean BACC while center3, bottom20, or worst
  seed-center cells remain unacceptable.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../sail/configs/sail_virchow2.yaml`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`

## Interpretation

R1.2b's worst center is 0.8070 despite mean BACC 0.9155. Inspected primary rows include seed/center BACC values below 0.75. This is why SAIL requires weak-center and seed checks.

The same rule now applies to generated embeddings. Dense tail-shield reached
0.8988 center-equal mean BACC, but failed because center3 stayed at 0.7896 and
the worst seed-center BACC stayed at 0.4971. Mass-bagged component union reached
0.8903 mean BACC, but had min-center 0.7931 and seed std 0.0568. Mean BACC
alone is therefore not thesis-facing evidence.

## Implication For Thesis

The thesis should report worst-case performance and seed stability alongside mean performance.

## Limitations

Weak-center labels can be affected by class-balance caveats. Cite eval warnings when present.

## Next Checks

- Verify SAIL `center_summary.csv` when available.
- Track low-minority/single-class eval warnings.
- For component-union methods, report center3, bottom20, worst seed-center,
  min-center, and seed std alongside mean BACC.
