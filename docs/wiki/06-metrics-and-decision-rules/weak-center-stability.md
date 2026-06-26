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
- The multipanel tail-risk run makes this rule sharper: 0.9087 mean BACC still
  fails if center3/min-center is 0.7897 and the worst seed-center is 0.4975.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_center_summary.csv`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../sail/configs/sail_virchow2.yaml`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1/center3_failure_audit/center3_failure_conclusion.md`

## Interpretation

R1.2b's worst center is 0.8070 despite mean BACC 0.9155. Inspected primary rows include seed/center BACC values below 0.75. This is why SAIL requires weak-center and seed checks.

The same rule now applies to generated embeddings. Dense tail-shield reached
0.8988 center-equal mean BACC, but failed because center3 stayed at 0.7896 and
the worst seed-center BACC stayed at 0.4971. Mass-bagged component union reached
0.8903 mean BACC, but had min-center 0.7931 and seed std 0.0568. Mean BACC
alone is therefore not thesis-facing evidence.

The multipanel tail-risk run reached 0.9087 center-equal mean BACC and improved
bottom20/seed-std metrics, but still failed because center3/min-center stayed
0.7897 and the worst seed-center was 0.4975. Its Center3 audit shows why
weak-center reporting needs class-specific diagnostics: the failed `42 x
center3` cell had class counts class0 = 198 and class1 = 2, final class1 recall
0.0000, and mean confidence 0.9795.

## Implication For Thesis

The thesis should report worst-case performance and seed stability alongside mean performance.

## Limitations

Weak-center labels can be affected by class-balance caveats. Cite eval warnings when present.

## Next Checks

- Verify SAIL `center_summary.csv` when available.
- Track low-minority/single-class eval warnings.
- For component-union methods, report center3, bottom20, worst seed-center,
  min-center, and seed std alongside mean BACC.
- For any Center3 follow-up, report per-class recall, predicted class counts,
  confidence/margin distributions, and whether audit-only target labels were
  used only after predictions were fixed.
