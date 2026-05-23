# Metric Definitions

## Purpose

Define commonly used metrics.

## Key Claims

| Metric | Meaning | Use |
| --- | --- | --- |
| NELBO | Negative evidence lower bound; lower is better unless converted to `-NELBO`. | CVAE compatibility/utility. |
| BACC | Balanced accuracy. | Current real-feature and downstream classifier utility. |
| Macro F1 | Class-balanced F1 summary. | Secondary downstream utility. |
| AUROC | Ranking quality for binary tasks when valid. | Secondary classifier metric. |
| Oracle gap | Regret to best candidate/oracle. | Selector quality. |
| Spearman | Rank correlation between predicted and actual utility. | Proxy compatibility quality. |
| Top-k containment | Whether oracle appears within top-k selected candidates. | Neighborhood quality. |

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/support_nelbo_consolidation_report.md`

## Interpretation

Metrics are not interchangeable. NELBO utility supports CVAE compatibility claims; downstream BACC supports held-out classifier utility claims.

## Implication For Thesis

Each result page must state which metric supports which claim.

## Limitations

Some artifacts use audit metrics or diagnostic fidelity metrics. Do not treat fidelity as downstream utility.

## Next Checks

- Add exact metric formulas where required by the thesis methods chapter.
