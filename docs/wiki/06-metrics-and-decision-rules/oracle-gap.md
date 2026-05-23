# Oracle Gap

## Purpose

Explain oracle gap as a regret metric.

## Key Claims

- Oracle gap measures how far the selected candidate is from the best available candidate under held-out utility.
- Candidate oracle rows are diagnostic upper bounds.
- A small oracle gap can indicate useful compatibility estimation even when top-1 exact match is imperfect.

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/tables/r12b_selector_oracle_gap.csv`
- `../../../cvae_support_routing/artifacts/comparison_tables/support_nelbo_consolidation_report.md`
- `../../../cvae_support_routing/artifacts/comparison_tables/breakhis_support_estimated_utility_routing_v1.md`

## Interpretation

In R1.2b primary rows, mean selector oracle gap is 0.0326, but median selected rank is 47.0 and top-1 oracle match is 0/15. This supports the neighborhood-useful but exact-top-1-brittle interpretation.

## Implication For Thesis

Use oracle gap to distinguish catastrophic selection failure from benign rank mismatch.

## Limitations

Oracle gap uses target utility and is diagnostic. It cannot be used to select a deployable method.

## Next Checks

- Report SAIL gap to R1.2b posthoc once artifacts exist.
