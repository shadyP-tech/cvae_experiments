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
| Center-equal mean BACC | Mean of heldout-center mean BACC values. | Primary cross-center summary when centers should contribute equally. |
| Seed-equal mean BACC | Mean of experiment-seed mean BACC values. | Seed-stability summary. |
| Replicate-row mean BACC | Mean over all eligible seed/center/replicate rows. | Row-level diagnostic; not a substitute for center-equal reporting. |
| Negative-control gap | Primary BACC minus strongest matched negative-control BACC. | Tests whether a signal beats shuffled/control alternatives. |
| Paired delta | Variant BACC minus equal-baseline BACC within the same seed/center cell. | Main effect estimate when generated bundles are paired. |
| Pairing invariant audit | Hash audit confirming identical generated features and per-source predictions for matched source sets and budgets. | Guards against method-label sampling confounds. |

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/support_nelbo_consolidation_report.md`

## Interpretation

Metrics are not interchangeable. NELBO utility supports CVAE compatibility claims; downstream BACC supports held-out classifier utility claims.

For D-series generated-embedding artifacts, prefer center-equal mean BACC and
min-center BACC over row-weighted means. A method that wins by row mean but
loses weak-center stability should not be treated as stronger.

For paired generated-embedding confirmations, prioritize paired delta summaries
over unpaired mean differences. A small delta is interpretable only if the
pairing invariant audit passes.

For support-NELBO and source-inner transfer, Spearman/top-k containment are
alignment diagnostics. They become thesis-facing only when accompanied by
matched downstream BACC improvement and negative-control gaps.

## Implication For Thesis

Each result page must state which metric supports which claim.

## Limitations

Some artifacts use audit metrics or diagnostic fidelity metrics. Do not treat fidelity as downstream utility.

## Next Checks

- Add exact metric formulas where required by the thesis methods chapter.
