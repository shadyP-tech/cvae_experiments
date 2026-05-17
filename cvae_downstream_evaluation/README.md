# CVAE Downstream Evaluation

This root package is the skeleton for the second-stage evaluation after direct support-NELBO routing.

It keeps the thesis pipeline explicit:

```text
target support set
-> direct support-NELBO compatibility estimation
-> selected source expert
-> synthetic embedding generation
-> downstream classifier / distributional evaluation
-> utility on held-out target evaluation data
```

The package is intentionally separate from `cvae_support_routing` and `cvae_testing`:

- `cvae_support_routing`: established support-NELBO selection and protocol artifacts.
- `cvae_testing`: existing training, routing, and evaluation runtime.
- `cvae_downstream_evaluation`: next-stage synthetic-embedding utility experiments.

## Thesis Placement

This package supports the thesis outline sections on:

- independently trained CVAE-style generative experts
- post-hoc routing without expert retraining or parameter sharing
- domain-specific and unseen-domain generalization
- generation quality and downstream classification performance
- comparison against metadata routing, random experts, naive ensembles, and oracle diagnostics

The primary question is:

```text
Does the expert selected by direct support-NELBO generate synthetic embeddings
that are useful for the held-out target domain?
```

NELBO remains the selection signal. Downstream performance is the second-stage utility.

## Structure

```text
cvae_downstream_evaluation/
  configs/       Versioned experiment templates.
  docs/          Protocol, claim boundaries, thesis alignment, implementation order.
  scripts/       Thin CLI entrypoints for future runs and reports.
  src/           Implementation modules with explicit ownership boundaries.
  tests/         Protocol and artifact contract tests.
  artifacts/     Ignored generated manifests, reports, tables, and plots.
```

## First Experiment

Start with:

```text
direct_support_nelbo_selected_synthetic_downstream_v1
```

Locked v1 scope:

- Camelyon17 only.
- Synthetic-only downstream utility.
- Class-stratified reference-posterior resampling through frozen source experts.
- Single-expert downstream oracle diagnostic.
- Late probability ensemble baseline, reported outside the single-expert oracle.
- Lightweight distributional fidelity diagnostics.

Defer source-only augmentation, few-shot target augmentation, and corruption robustness until the primary downstream transfer question has a stable answer.

## Non-Negotiable Protocol

- Candidate expert pool excludes the held-out target/query expert under LOQDO/LODO.
- Target support samples are disjoint from target evaluation samples.
- Target support labels are not used for routing in the primary protocol.
- Target evaluation labels, target evaluation NELBO, downstream oracle expert labels, and target test metrics are forbidden before routing and generation decisions are locked.
- Source expert checkpoints are frozen; routing and downstream evaluation must not update them.
- Synthetic budget, class prior, classifier architecture, classifier hyperparameters, and metric set must be fixed before inspecting target evaluation results.
- Single-expert synthetic samples live in the selected expert's projected CVAE feature frame; target evaluation embeddings must be projected through the same expert head.
- The current CVAE is not treated as class-conditional. Labels enter the primary generation mode through labeled source reference pools.

## Expected Outputs

Each completed run should produce:

- protocol manifest
- split manifest
- expert provenance table
- support-NELBO selection table
- all-expert downstream utility matrix
- selected-vs-oracle downstream gap table
- baseline comparison table
- support-size stratified downstream summary
- generation seed stability table
- classifier seed stability table
- fidelity diagnostics table
- leakage/provenance report

Generated outputs belong under `artifacts/` during local development or under the existing repo-wide `outputs/` convention for full experiment runs.
