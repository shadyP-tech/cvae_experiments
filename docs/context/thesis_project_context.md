# Thesis Project Context

## Purpose

This file defines the stable conceptual, methodological, and protocol context for
this thesis repository.

Use it when creating or updating:

- wiki documentation
- experiment summaries
- implementation-plan reviews
- result syntheses
- thesis notes
- decision reports
- reproducibility notes

This file is not a raw result table. Numerical results and fast-changing
experiment status belong in:

```text
docs/context/current_experimental_state.md
```

## Repository Map

The repository is organized around three related experiment surfaces:

- `cvae_testing/`: shared CVAE training, routing, evaluation, historical
  metadata-routing work, quarantined legacy paths, and tracked comparison
  tables.
- `cvae_support_routing/`: direct support-NELBO routing experiments, report
  builders, comparison artifacts, and tests that preserve target-support /
  target-evaluation separation.
- `cvae_downstream_evaluation/`: downstream synthetic-embedding utility
  experiments, pathology foundation embedding screens, real-feature transfer
  diagnostics, dense aggregation audits, reports, and artifact contracts.

Protocol status and quarantined paths are summarized in `PROTOCOL_STATUS.md`.
Do not use quarantined artifacts for thesis-facing claims.

## Thesis Topic

The thesis investigates compatibility-driven routing and aggregation of
generative experts for privacy-preserving domain adaptation in multi-domain
medical imaging.

The original project framing emphasized metadata-driven MoErging of generative
models. The current empirical direction keeps the MoErging and
privacy-preserving generative-expert framing, but it no longer assumes metadata
is the primary method.

Current stable framing:

```text
independently trained source-domain generative experts
+ pathology foundation-model embeddings
+ source-only compatibility estimation
+ sparse or dense expert/config aggregation
-> improved downstream utility on unseen medical-imaging domains
without sharing raw clinical data
```

Metadata remains important, but its role is:

```text
metadata = compatibility proxy / baseline / interpretability signal
```

not:

```text
metadata = assumed primary routing solution
```

## Pivot Statement

The thesis has pivoted from metadata-first routing to compatibility-driven
routing and aggregation. Metadata remains a baseline and candidate proxy, but
the current empirical path tests source-only utility selection in pathology
foundation feature spaces before rebuilding or extending CVAE experts.

This pivot changes how compatibility is estimated. It does not change the
privacy-preserving goal of using local experts or generated embedding-level data
instead of exchanging raw clinical data.

## Original Project Vision

The project aims to replace the exchange of sensitive, high-volume clinical
datasets with lightweight models or generated embedding-level data that preserve
useful local data distributions.

The original thesis outline in `cvae_testing/thesis_outline.txt` frames the
repository around independently trained models, post-hoc composition, and
unseen-domain evaluation. That motivation remains valid.

## Core Conceptual Pipeline

All routing, compatibility, and downstream utility documentation should follow:

```text
Query / target domain or sample
-> Compatibility Estimation
-> Routing Decision
-> Expert Selection or Aggregation
-> Utility
```

Definitions:

```text
Query q:
  A target domain, target support set, target sample, or generation/adaptation
  request.

Expert e:
  A source-trained generative model, source-domain model, downstream classifier
  config, feature-space model, or diagnostic candidate.

Compatibility C(q,e):
  The expected utility of using expert e for query q.

True compatibility:
  C_true(q,e) = -NELBO(q,e)

Proxy compatibility:
  metadata similarity
  latent similarity
  learned predictors
  source-inner validation
  support-set statistics
  real-feature transfer diagnostics
  source-selected downstream performance
```

Core rule:

```text
Compatibility = expected utility, not similarity.
```

Similarity is allowed only as a proxy. It must be judged by whether it predicts
utility.

## What Counts as Utility

Utility is the evidence that a representation, expert, config, or routing
decision actually helps.

Depending on the experiment, utility may be measured by:

- negative ELBO / NELBO
- downstream balanced accuracy
- macro F1
- AUROC
- oracle gap
- weak-center performance
- weak-domain performance
- seed stability
- rank correlation between predicted and actual utility
- top-k containment of oracle candidates

For real-feature pathology embedding screens, the dominant utility signal is
source-selected downstream balanced accuracy under leave-one-domain/center-out
evaluation.

For CVAE phases, the key question is:

```text
How much of the real-feature utility is preserved after CVAE generation?
```

## Role of Metadata

Metadata remains relevant, but the thesis should not assume metadata similarity
is itself compatibility.

Updated metadata roles:

- Baseline: metadata routing remains a serious, interpretable baseline.
- Proxy signal: metadata can estimate compatibility if it predicts utility.
- Interpretability layer: metadata can help explain why source domains or
  experts transfer.
- Stress-test structure: metadata-rich datasets make scanner, lab, stain,
  tissue, tumor-type, center, and magnification shifts analyzable.
- Low-data fallback: metadata may remain deployable when source-inner utility
  diagnostics are unavailable.

Unsafe claim:

```text
metadata similarity = compatibility
```

Safe claim:

```text
metadata similarity is a candidate compatibility proxy whose value must be
measured against downstream utility, NELBO, ranking quality, and oracle gap
```

## Role of Pathology Foundation Embeddings

Feature embeddings are central to the current empirical path. The repository
contains pathology foundation embedding screens under
`cvae_downstream_evaluation/`.

Relevant backbones include:

- DINOv2
- Phikon
- UNI
- Virchow2

The stable representation lesson is:

```text
Generic visual embeddings plus aggressive PCA may not preserve enough
pathology-specific utility. Pathology-specific embeddings can raise the
real-feature transfer ceiling, but real-feature success does not prove CVAE
generation will preserve that utility.
```

Therefore:

```text
real-feature success -> justifies CVAE preservation tests
real-feature success -/> proves CVAE success
```

## CVAE Expert Framing

CVAEs remain the intended generative expert family.

A CVAE expert should be judged by utility preservation, not only by
reconstruction quality.

Key CVAE questions:

```text
Can a source-trained CVAE model the chosen embedding space?
Does generation preserve class geometry?
Does generation preserve downstream transfer utility?
Does routing or aggregation among CVAE experts improve unseen-domain performance?
```

Do not jump from real-feature classifier success to a deployable generative
claim. Treat CVAE rebuilds as preservation tests unless their generated
embeddings are evaluated under a protocol-clean downstream utility workflow.

## Routing and Aggregation Vocabulary

Distinguish these surfaces:

- expert routing
- config selection
- dense classifier aggregation
- CVAE expert aggregation
- output aggregation
- parameter aggregation

Dense aggregation is aligned with the MoErging design space because multiple
selected experts, configs, or outputs may be combined rather than selecting one
winner. The claim depends on the surface being aggregated. Dense real-feature
classifier aggregation is not the same as CVAE routing.

## Protocol Rules

Target evaluation labels may be used only for final scoring.

Target evaluation labels must not choose:

- backbone
- representation
- PCA dimension
- classifier hyperparameters
- class weight
- CVAE checkpoint
- source expert
- routing method
- k
- aggregation rule
- calibration rule
- decision threshold

Evidence labels:

| Evidence type | Meaning |
| --- | --- |
| Deployable evidence | Chosen without target evaluation labels |
| Deployable diagnostic | Protocol-clean diagnostic evidence, not necessarily a complete deployable method |
| Audit-only evidence | Diagnostic; cannot justify a deployable claim |
| Posthoc evidence | Uses target outcomes; feasibility only |
| Oracle evidence | Upper bound using target outcomes |
| Negative result | Valid result showing no useful gain |
| Assumption | Plausible but not yet verified |
| TODO | Missing or unverified evidence |

Do not use audit-only, posthoc, or oracle evidence to support deployable claims.

## Baselines

Relevant baselines include:

- metadata routing
- source-only top-1 selection
- source-only top-k aggregation
- real-feature classifier ceiling
- oracle config/expert selection
- posthoc best config
- naive ensemble
- cross-backbone audit ensemble
- CVAE single expert
- CVAE dense ensemble
- residual/empirical transfer generators
- decoder weight averaging / FedAvg-style baselines

Metadata should remain a serious baseline. Do not assume learned, latent, or
foundation-embedding methods beat metadata unless they do so consistently and
protocol-cleanly.

## Evaluation Priorities

Prioritize:

1. protocol validity
2. source-only deployability
3. balanced accuracy
4. weak-center / weak-domain performance
5. seed stability
6. oracle gap
7. Spearman rank correlation
8. top-1 selection accuracy
9. top-k oracle containment
10. comparison to metadata and source-only baselines

A method is stronger only if improvement is:

- consistent
- stable across seeds
- robust across centers/domains
- better than the relevant baseline
- not dependent on target-label selection
- not merely posthoc

## Claim Discipline

Safe claims:

```text
Pathology foundation embeddings can raise the real-feature source-transfer
ceiling relative to earlier generic embedding setups when verified by
protocol-clean source-selected rows.
```

```text
The current bottleneck has shifted from representation ceiling alone toward
source-selected stability and CVAE preservation.
```

```text
A passing real-feature dense aggregation result can justify a CVAE preservation
test in the same feature space.
```

Unsafe claims:

```text
Metadata similarity proves compatibility.
```

```text
Embedding similarity proves compatibility.
```

```text
Real-feature classifier success proves CVAE generation will work.
```

```text
Cross-backbone dense aggregation proves one clean generative feature space is
sufficient.
```

```text
Posthoc best target performance is deployable.
```

## Documentation Rules

Documentation should be:

- evidence-backed
- protocol-aware
- clear about what is proven and not proven
- useful for thesis writing
- linked to artifacts where possible
- honest about missing evidence

Numerical claims should come from:

- artifact tables
- decision reports
- leakage reports
- experiment logs
- explicitly labeled user-provided synthesis

If a number is provided by synthesis but not verified against artifacts, label
it:

```text
Provided synthesis; verify against artifact if available.
```

If an expected artifact is missing, write:

```text
TODO: verify against artifact.
```

## Relationship to Current Experimental State

This file defines the stable project frame after the empirical pivot.

Fast-changing synthesis lives in:

```text
docs/context/current_experimental_state.md
```

That file should contain:

- latest result numbers
- current experiment status
- verified artifact paths
- current pass/fail decisions
- next planned run
- unresolved TODOs

Update this file only when the thesis framing or protocol vocabulary changes.
