# Metadata As Proxy

## Purpose

Define the updated role of metadata after the empirical pivot.

## Key Claims

- Metadata is not the assumed primary routing solution.
- Metadata remains a serious baseline, candidate compatibility proxy, interpretability signal, and low-data fallback.
- Metadata similarity does not prove compatibility.

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/support_nelbo_consolidation_report.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/camelyon17_support_estimated_utility_routing_v2.md`
- `../../../cvae_testing/results/compact_interpretation_summary.md`

## Interpretation

Metadata can be useful when it predicts held-out utility. In support-NELBO reports, metadata routing is a baseline that direct support-NELBO often improves upon. In historical BreakHis CVAE routing, metadata selection was technically correct but did not beat the global CVAE.

## Implication For Thesis

Metadata should be retained as a baseline and explanation layer, not discarded. It should not be promoted into a deployable claim unless it wins under protocol-clean utility evaluation.

## Limitations

Metadata-rich domain structure varies by dataset. A negative metadata result in one setting does not prove metadata is useless in general.

## Next Checks

- Keep metadata baselines in final result tables.
- Report metadata failures as negative results, not as universal impossibility claims.
