# Thesis Project Context

Last updated: 2026-08-05

This file is the stable context anchor for the thesis project. It should define
the durable protocol vocabulary, evidence categories, and claim boundaries.
Fast-changing result summaries, current best metrics, and run-by-run status
belong in `docs/context/current_experimental_state.md` or experiment-specific
notes under `docs/wiki/`.

## Project Frame

The project studies source-only and privacy-preserving transfer for pathology
foundation-model embeddings. The core thesis question is whether independently
trained source-domain components can be selected, composed, or aggregated to
support an unseen target domain without using target evaluation labels during
selection.

There are two related but distinct evidence surfaces:

- real-feature reference experiments, which use cached foundation-model
  embeddings directly and measure what source-only transfer can achieve before
  introducing generated embeddings;
- CVAE downstream experiments, which test whether frozen source-domain
  generative experts, routing signals, and post-hoc composition preserve or
  improve downstream utility on held-out target domains.

Real-feature results can justify that a dataset/backbone has nontrivial
source-only signal and can provide a reference surface. They do not, by
themselves, prove CVAE preservation, routing quality, synthetic utility, or
generative quality.

## Canonical Repository Boundary

Active implementation is owned by the single package
`src/midogpp_thesis/`. Dataset state belongs under `datasets/midogpp/`, staged
experiment definitions under `experiments/midogpp/`, MIDOG++ evidence under
`artifacts/midogpp/`, and interpretation under `docs/`. Retired capability
package roots are not active imports, runners, or artifact fallbacks.

The active evidence stages are encoded in
`experiments/midogpp/registry.yaml`. Stages 10 and 20 contain the real-feature
and CVAE-preservation implementations. Stage 30 now contains a validated bank
of independently trained source-domain CVAE experts that is authorized as an
input to future routing experiments. That authorization establishes expert-bank
construction and provenance only; it does not establish routing quality.
Stage 40 contains a validated, target-data-free GenerationLock, and Stage 60
contains validated fixed equal-union, metadata-compatibility/max-tie, and
source-inner utility/regret policy locks. The utility surface is a one-time,
non-selecting source-inner policy-training input; its per-query best candidate
is an oracle reference only for regret calculation. The policy may select one
source only under its predeclared uncertainty gate and otherwise reuses the
exact frozen equal-union control. These are generation, proxy, and
policy-contract results only; they establish no routing-quality or downstream-
utility result. Stage 70 remains planned pending separately authorized fresh
target evaluation and matched scoring. Stage 90 contains non-deployable
diagnostics and rejected historical MIDOG++ lineages.

BreakHis, Camelyon17, and generic historical material is isolated under
`artifacts/cross_dataset_archive/` and is outside the active MIDOG++ registry.
Historical path strings may remain inside immutable provenance records, but
they are not current operational paths.

## Evaluation Regime

The intended downstream regime is LOQDO/LODO-style evaluation:

```text
held-out target domain H
source domains = all domains except H
unlabeled target support S_H = routing or compatibility estimation only
labeled target evaluation T_H = final downstream utility only
```

The thesis-facing protocol preserves this sequence:

```text
Query -> Compatibility Estimation -> Routing Decision -> Expert Selection
      -> Synthetic Generation or Source-Only Aggregation -> Downstream Utility
```

For CVAE routing work, compatibility means expected downstream utility under
allowed pre-evaluation information. It is not the same as metadata similarity,
distributional similarity, support-set NELBO alone, or fidelity alone.

## Protocol Guardrails

Before routing, generation settings, classifier settings, thresholds,
calibration, or aggregation rules are frozen, the implementation must not use:

- target evaluation labels;
- target evaluation metrics;
- downstream oracle expert labels;
- target evaluation NELBO or target evaluation distributional metrics;
- direct target identity fields as deployable predictive features;
- target-evaluation-tuned generation budgets, temperatures, classifiers,
  hyperparameters, thresholds, or calibration settings.

Allowed pre-selection information includes source expert checkpoints, source
expert provenance, source-domain metadata, source-trained artifacts, unlabeled
target support embeddings, query metadata available at routing time,
target-support-only diagnostics, and source-inner validation statistics.

Candidate CVAE experts must be independently trained source-domain experts. In
each held-out target fold, the target/query expert is excluded from deployable
candidate pools. It may appear only as an explicitly marked diagnostic
reference. Routing and composition must not update source expert checkpoints.

A fixed all-eligible-source control need not consume target support or compute
compatibility when it performs no ranking, selection, or learned weighting.
For such a control, target-center identity may define the held-out fold,
target-expert exclusion, and a predeclared label-blind shuffle namespace; it
must not become a predictive feature, score, rank, or source weight.

For real-feature source-only experiments, classifier selection and aggregation
recipes must be selected inside source-inner pseudo-target folds, then frozen
before held-out target-center scoring.

For CVAE recipe selection, fully nested source-inner pseudo-target evidence may
freeze a `RecipeLock` for later source-expert training. Real held-out outer
preservation metrics are evaluation-only: they may never revise the objective,
sampler family, expert recipe, generation policy, router, or composition rule.
Stage-20 recipe selection and pooled preservation do not replace the later
post-expert-bank Stage-40 generation boundary. The active Stage-40
GenerationLock freezes source-only generation, frame, seed, budget, and
classifier settings without target data; its health probes are not routing or
utility evidence.

## Evidence Labels

Use these labels consistently in docs and reports:

- `PASS`: protocol-clean evidence directly supports the named thesis claim.
- `WEAK PASS`: protocol-clean evidence supports a narrow claim, but important
  stability, baseline, or scope limitations remain.
- `CONDITIONAL PASS`: acceptable only with named guardrails, tests, artifact
  checks, or TODOs carried into implementation.
- `DIAGNOSTIC ONLY`: useful for debugging, planning, or context, but not a
  thesis-facing adoption claim.
- `NEGATIVE_RESULT`: protocol-clean evidence against the tested method or
  hypothesis.
- `AUDIT_ONLY`: artifact is intended for provenance, leakage, schema, or
  reproducibility review.
- `REJECTED`: artifact or claim is unusable for the named purpose because of
  protocol, leakage, schema, or scope problems.
- `TODO_VERIFY_ARTIFACT`: interpretation depends on a local artifact check that
  has not yet been completed.

Prefer the stricter label when an artifact supports one narrow claim but could
be misread as supporting a broader one.

## Baseline Taxonomy

Important baseline classes:

- metadata routing: a serious deployable baseline when metadata is available at
  routing time, but direct target identity is not a deployable feature;
- random expert or random candidate selection: a sanity baseline for routing
  and selection;
- source-global or uniform source aggregation: a simple source-only reference;
- SAIL or other real-feature source-only aggregation: a real-feature reference
  surface, not CVAE evidence;
- dense CVAE aggregation or late ensembling: post-hoc composition over frozen
  CVAE expert or mode classifiers, not learned compatibility unless selected
  by a validated compatibility estimator;
- downstream oracle: diagnostic upper bound computed after all candidate
  downstream scores exist, never a deployable selection method.

A source-inner per-query best candidate used to normalize regret is likewise a
non-deployable oracle reference. It may define the predeclared source-inner
policy objective, but it is neither target evidence nor a deployed selection
rule.

Budget matching matters. Claimed comparisons should use matched synthetic
budgets, classifier seeds, candidate eligibility, and feature frames unless the
report explicitly marks a row as diagnostic-only.

## Claim Boundaries

Allowed claims must name the evidence surface precisely:

- Real-feature gates and source-only classifier references can support claims
  about source-only transfer signal, headroom, and candidate-surface
  feasibility.
- CVAE downstream artifacts can support synthetic-embedding utility claims only
  when they use generated embeddings and evaluate on held-out target data after
  selection has been frozen.
- Source-inner tuning can support model-selection claims only when all
  hyperparameters, thresholds, weights, and policies are selected without the
  real held-out target labels.
- Dense late aggregation can support routing-risk-reduction or post-hoc
  composition claims when it uses frozen components and protocol-clean
  selection rules.

Forbidden or overbroad claims:

- lower support NELBO proves better downstream utility;
- fidelity metrics alone prove downstream utility;
- real-feature transfer proves CVAE preservation;
- downstream oracle performance is deployable routing performance;
- target-center BACC, macro-F1, AUROC, or PR-AUC can be used to choose
  hyperparameters, thresholds, weights, or policies for that same held-out
  target;
- source-discriminative generated geometry is a utility improvement unless
  held-out downstream utility also improves.

## Metric Priorities

For routing-to-downstream alignment, prioritize:

- top-1 downstream oracle hit;
- support-NELBO-to-downstream rank correlation when applicable;
- downstream oracle gap in BACC and macro-F1;
- pairwise preference accuracy or top-k recall when available;
- seed, fold, and worst-domain stability.

For downstream utility, prioritize:

- balanced accuracy;
- macro-F1;
- AUROC when binary or otherwise well-defined;
- AUPRC when class imbalance makes it informative;
- worst-domain and minimum-center performance.

For fidelity diagnostics, report metrics such as MMD, energy distance, Frechet
embedding distance, mean/covariance distance, and kNN
precision/recall/density/coverage separately from downstream utility. Fidelity
can be interpreted as secondary evidence or diagnostic context, not as a
replacement for utility.

## Documentation Map

Use these files as the main context sources:

- `docs/context/current_experimental_state.md`: current result synthesis,
  latest artifact status, and fast-changing TODOs.
- `docs/context/midogpp_experiment_file_workflow.md`: current MIDOG++ input,
  runner, artifact, validation, and sync-path lookup for future experiments.
- `docs/wiki/03-experiments/`: experiment-specific interpretation notes.
- `docs/context/protocol_status.md`: concise active, planned, and quarantined
  implementation status.
- `README.md`: canonical checkout layout and package entrypoint.
- `datasets/midogpp/README.md`: raw data, frozen contract, patch-payload, and
  feature-cache ownership.
- `experiments/midogpp/README.md`: registry and stage lifecycle.
- `artifacts/midogpp/README.md`: canonical evidence layout and run-bundle
  requirements.
- `artifacts/cross_dataset_archive/README.md`: scope and restrictions for
  retired non-MIDOG++ evidence.

When updating docs, keep stable definitions here and put run-specific numbers,
artifact roots, validation outputs, and current recommendations in the current
state or experiment-specific files.
