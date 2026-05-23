# Top-K Ranking And Spearman

## Purpose

Document ranking diagnostics used to evaluate proxy compatibility.

## Key Claims

- Top-1 match measures exact selection.
- Top-k containment measures whether the selector identifies a useful candidate neighborhood.
- Spearman measures utility-ranking correlation.
- A selector can have useful top-k information while failing exact top-1 selection.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../cvae_support_routing/artifacts/comparison_tables/support_nelbo_consolidation_report.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/camelyon17_support_estimated_utility_routing_v2.md`

## Interpretation

R1.2b shows the key pattern: top-1 oracle match is 0/15, top-3 contains oracle is 15/15, and mean Spearman is 0.4589. That supports dense top-k aggregation as the next diagnostic.

## Implication For Thesis

When top-k containment is high but top-1 is brittle, aggregation can be a principled risk-reduction move.

## Limitations

Top-k containment alone does not prove aggregation will improve final utility. It only motivates the test.

## Next Checks

- Add SAIL top-k selected-member diagnostics when output exists.
