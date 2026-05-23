# Audit Vs Deployable Claims

## Purpose

Define evidence labels and claim boundaries.

## Key Claims

| Evidence type | Meaning |
| --- | --- |
| Deployable evidence | Chosen without target evaluation labels. |
| Deployable diagnostic | Protocol-clean diagnostic evidence, not necessarily a complete deployable method. |
| Audit-only evidence | Diagnostic; cannot justify deployable claim. |
| Posthoc evidence | Uses target outcomes; feasibility only. |
| Oracle evidence | Upper bound using target outcomes. |
| Negative result | Valid result showing no useful gain. |
| Assumption | Plausible but not yet verified. |
| TODO | Missing or unverified evidence. |

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../PROTOCOL_STATUS.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`

## Interpretation

Posthoc and oracle rows are valuable because they show headroom and diagnose regret. They cannot be used to choose a deployable method.

## Implication For Thesis

Every result section should state the strongest allowable claim and the forbidden overclaim.

## Limitations

Evidence labels depend on artifact fields. If labels are missing, classify the claim as incomplete until verified.

## Next Checks

- Add evidence labels to new wiki result pages as SAIL artifacts arrive.
