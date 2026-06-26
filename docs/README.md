# Thesis Documentation

## Purpose

This documentation folder is the thesis wiki for the CVAE metadata-routing repository. It separates stable project framing from current experimental state and links synthesis pages back to repository artifacts.

## Navigation

- [Project context](context/thesis_project_context.md)
- [Current experimental state](context/current_experimental_state.md)
- [Pivot statement](context/pivot_statement.md)
- [Wiki home](wiki/00-home.md)
- [Thesis frame](wiki/01-thesis-frame/README.md)
- [Literature and background](wiki/02-literature-and-background/README.md)
- [Experiments](wiki/03-experiments/README.md)
- [Current best approach](wiki/04-current-best-approach/README.md)
- [Protocol and safety](wiki/05-protocol-and-safety/README.md)
- [Metrics and decision rules](wiki/06-metrics-and-decision-rules/README.md)
- [Glossary](wiki/07-glossary/README.md)

## Evidence Rules

Use `context/thesis_project_context.md` for stable framing and `context/current_experimental_state.md` for the current result synthesis. Do not treat audit-only, posthoc, oracle, or target-label-informed evidence as deployable evidence.

When an expected artifact is absent, write `TODO: verify against artifact.`

## Current Artifact Note

The latest Virchow2 CVAE rebuild and D-series decentralized composition
artifacts were inspected from the synced artifact root:

```text
/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/
```

They are documented in:

- [Current experimental state](context/current_experimental_state.md)
- [D-series decentralized Virchow2 CVAE composition](wiki/03-experiments/d-series-decentralized-cvae.md)
- [Component-union mass allocation audits](wiki/03-experiments/component-union-mass-allocation.md)
- [Multipanel tail-risk and Center3 audit](wiki/03-experiments/multipanel-tailrisk-center3-audit.md)
- [Generated-embedding CVAE current synthesis](wiki/04-current-best-approach/generative-cvae-current-synthesis.md)

The paired dense-all4 reliability confirmation was synced into this working
repository at:

```text
cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/
```

It is the cleanest protocol-clean generated-embedding dense aggregation result,
but it is not sparse routing or target-conditioned expert selection.

Newer component-union and random mass-bag artifacts reach higher mean BACC, but
they are documented as diagnostic/high-capacity surfaces unless they beat
matched controls and repair weak-center/tail failures.

The multipanel tail-risk mass-bag artifact reaches 0.9087 mean BACC but remains
a negative stabilization result because center3/min-center fail. Its Center3
audit is diagnostic-only and points to confident rare-class collapse rather
than simple component undersampling.
