# Problem Statement

## Purpose

State the thesis problem in a form that connects privacy, generative experts, routing, and downstream utility.

## Key Claims

- The thesis targets privacy-preserving domain adaptation in multi-domain medical imaging.
- The project replaces raw clinical data sharing with independently trained source-domain generative experts or generated embedding-level data.
- The central method question is how to route or aggregate experts/configs for a target query without using target evaluation labels.

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_testing/thesis_outline.txt`
- `../../../cvae_downstream_evaluation/README.md`

## Interpretation

The problem is not just model generation quality. The routing decision must select or combine useful source information for an unseen target while preserving protocol boundaries.

## Implication For Thesis

The thesis contribution should be framed around compatibility estimation and routing/aggregation under privacy constraints, not simply around metadata matching or CVAE reconstruction.

## Limitations

The current documentation does not prove privacy guarantees beyond the repository's no-raw-data-sharing experimental setup.

## Next Checks

- Link final thesis chapters to this problem framing.
- Add privacy-threat-model detail if the thesis requires a formal privacy section.
