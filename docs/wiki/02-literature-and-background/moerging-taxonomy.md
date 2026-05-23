# MoErging Taxonomy

## Purpose

Clarify how routing, sparse selection, dense output aggregation, and parameter aggregation relate to the thesis.

## Key Claims

- Sparse selection chooses one expert/config.
- Dense aggregation combines multiple selected experts/configs or their outputs.
- Output aggregation and parameter aggregation are different claims.
- SAIL is dense real-feature classifier aggregation, not CVAE expert routing.

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`

## Interpretation

Dense aggregation can reduce regret when top-1 selection is brittle. The repository already shows this principle in C6.3 synthesis and tests it again in SAIL for real-feature Virchow2 configs.

## Implication For Thesis

The thesis can use dense aggregation as a risk-reduction concept, but must identify whether aggregation happens over classifiers, generated embeddings, CVAE experts, outputs, or parameters.

## Limitations

This page does not independently review external MoErging literature. It summarizes how the repository uses the taxonomy.

## Next Checks

- Add paper citations in final thesis bibliography.
- Keep cross-backbone aggregation audit-only.
