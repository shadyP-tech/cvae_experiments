# Compatibility Definition

## Purpose

Define compatibility so experiment claims do not confuse similarity with utility.

## Key Claims

True compatibility is expected utility:

```text
C_true(q,e) = -NELBO(q,e)
```

Proxy compatibility may come from:

- metadata similarity
- latent similarity
- learned predictors
- source-inner validation
- support-set statistics
- real-feature transfer diagnostics
- source-selected downstream performance

Core rule:

```text
Compatibility = expected utility, not similarity.
```

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/support_nelbo_consolidation_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`

## Interpretation

Metadata, embedding distance, and support statistics are candidate signals. They become useful only if they predict held-out NELBO or downstream utility under the locked protocol.

## Implication For Thesis

The thesis should report whether a proxy predicts utility through oracle gap, Spearman rank correlation, top-k containment, and downstream performance.

## Limitations

`-NELBO` is the direct compatibility target for CVAE utility, while downstream BACC is a second-stage held-out utility target. The two should not be collapsed into one metric.

## Next Checks

- Keep metric pages explicit about which utility target is being measured.
- Add TODOs where artifacts report similarity without utility validation.
