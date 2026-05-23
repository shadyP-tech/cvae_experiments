# Thesis Wiki Home

## Purpose

This wiki gives a navigable research record for the thesis repository: project framing, experiment history, protocol rules, current results, negative findings, and the current best next approach.

## Main Sections

| Section | Use |
| --- | --- |
| [01 Thesis Frame](01-thesis-frame/README.md) | Problem, conceptual pipeline, compatibility definition, empirical pivot. |
| [02 Literature And Background](02-literature-and-background/README.md) | MoErging, CVAE experts, foundation embeddings, domain shift, metadata as proxy. |
| [03 Experiments](03-experiments/README.md) | Z/R/C/F experiment families and negative results. |
| [04 Current Best Approach](04-current-best-approach/README.md) | SAIL method, current synthesis, Virchow2 rationale, rebuild gate. |
| [05 Protocol And Safety](05-protocol-and-safety/README.md) | Leakage rules, source-only selection, target-label restrictions, claim classes. |
| [06 Metrics And Decision Rules](06-metrics-and-decision-rules/README.md) | BACC, NELBO, oracle gap, top-k containment, Spearman, weak-center stability. |
| [07 Glossary](07-glossary/README.md) | Terms used across the thesis. |

## Current Thesis Direction

The thesis investigates compatibility-driven routing and aggregation of generative experts for privacy-preserving domain adaptation in multi-domain medical imaging.

The current diagnostic path is:

```text
R1.2b Virchow2 source-selected evidence
-> SAIL Virchow2 dense source-selected config aggregation
-> R1.3a vanilla Virchow2 CVAE preservation test if gates pass
```

## Evidence / Source Artifacts

- `docs/context/thesis_project_context.md`
- `docs/context/current_experimental_state.md`
- `PROTOCOL_STATUS.md`
- `cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `sail/configs/sail_virchow2.yaml`

## Next Checks

- Run or sync SAIL artifacts.
- Verify whether Virchow2-only dense rows pass the rebuild gate.
- Keep cross-backbone aggregation audit-only.
