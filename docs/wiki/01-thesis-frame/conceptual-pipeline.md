# Conceptual Pipeline

## Purpose

Define the pipeline used to interpret routing, aggregation, and downstream utility experiments.

## Key Claims

The repository should be interpreted through:

```text
Query / target domain or sample
-> Compatibility Estimation
-> Routing Decision
-> Expert Selection or Aggregation
-> Utility
```

Dense config aggregation, CVAE expert aggregation, metadata routing, and support-NELBO routing are different instantiations of this same pipeline.

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_support_routing/README.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `../../../PROTOCOL_STATUS.md`

## Interpretation

The key evaluation target is whether the selected or aggregated source information improves utility on held-out target data. Similarity and fidelity metrics are useful only if they predict utility or explain failures.

## Implication For Thesis

Every experiment page should identify the query, candidate experts/configs, selection signal, utility metric, and allowed target information.

## Limitations

Some historical artifacts predate the current pipeline vocabulary and must be interpreted through protocol status and quarantine labels.

## Next Checks

- Add fold-level diagrams once final thesis figures are chosen.
- Keep SAIL documented as real-feature config aggregation, not CVAE routing.
