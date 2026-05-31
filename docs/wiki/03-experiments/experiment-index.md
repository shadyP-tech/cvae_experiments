# Experiment Index

## Purpose

Summarize major experiment families and their relation to the current thesis direction.

## Key Claims

| Family | Purpose | Current conclusion | Relation to current best approach | Open TODOs |
| --- | --- | --- | --- | --- |
| Z-series | Representation ceiling and synthetic preservation audits. | Z1.1 local report says real-feature ceiling rows were missing and synthetic evidence was missing. | Baseline context for why R1.2/R1.2b were needed. | Verify any newer Z-series artifacts before quoting numbers. |
| R-series | Real-feature pathology embedding routing/source-selected diagnostics. | R1.2b makes Virchow2 the current strongest source-selected backbone but top-1 selection is brittle. | Directly motivates SAIL. | Run or sync SAIL outputs. |
| C-series | CVAE generator and aggregation experiments. | C6.3 synthesis supports dense late aggregation over frozen CVAE components, but not Virchow2 CVAE preservation. | Provides aggregation precedent and the CVAE preservation question. | Verify C6.3 numbers against raw/synced artifacts. |
| D-series | Virchow2 CVAE preservation, source-union GMM priors, decentralized source-local summary composition, reliability/support-NELBO/source-inner selectors. | Source-union K16 is the strongest centralized diagnostic; paired dense all4 reliability is the cleanest dense aggregation result; component-union/random mass-bag rows show high mean utility but fail adoption gates because controls and weak-tail behavior remain problematic. | Current generated-embedding evidence base. | Validate harmful-source suppression when final artifacts exist. |
| F-series | Residual or empirical transfer generator diagnostics, if present. | Residual tables exist, but current docs treat them as diagnostic unless protocol-clean adoption is established. | Baseline/diagnostic context, not current best path. | Verify exact F-series naming and claim status. |

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/reports/z11_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_decentralized_component_union_mass_bagged_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_support8_calibrated_component_union_prior_v1/reports/decision_summary.md`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1/reports/decision_summary.md`
- `../../../cvae_testing/results/comparison_tables/residual_routing_decision_table.csv`
- `../../../cvae_testing/results/comparison_tables/residual_safe_v2_decision_table.csv`

## Interpretation

The experiment history supports a pivot from metadata-first or sparse top-1
routing toward utility-driven compatibility and dense risk reduction. The
D-series adds a second lesson: generated-embedding CVAE utility is recoverable
with better latent priors and component-level composition, but source-mass
allocation is underidentified. Heldout-excluded reliability can improve clean
dense all-source aggregation, while random/component mass-bag controls reveal
that high mean BACC alone is not a deployable compatibility claim.

## Implication For Thesis

The current thesis narrative should keep two layers separate:

```text
SAIL = real-feature source-only aggregation gate
D-series = generated-embedding CVAE preservation/composition evidence
```

SAIL decides whether Virchow2 real-feature transfer is stable. D-series shows
which generated-embedding composition ideas preserve or fail to preserve that
utility after CVAE generation. The current generated-embedding bottleneck is
weak-center/tail robustness and harmful source interaction, not another
mean-BACC-only allocator.

## Limitations

The index is a synthesis. It does not replace individual decision reports or leakage reports.

## Next Checks

- Add SAIL result rows when available.
- Keep the paired dense all4 result scoped to clean dense aggregation, not
  final sparse routing.
- Document random mass-bag/component-union results as high-mean diagnostic
  surfaces unless matched controls are beaten.
- Validate source-inner harmful-source suppression once final reports are
  synced.
- Decide whether F-series should be renamed once exact artifact provenance is verified.
