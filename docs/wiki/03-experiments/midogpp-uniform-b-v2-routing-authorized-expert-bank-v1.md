# MIDOG++ Uniform-B V2 Routing-Authorized Expert Bank

## Purpose And Status

This page records the completed Stage-30 promotion experiment:

```text
midogpp.expert_bank.uniform_b_v2_routing_promotion.v1
```

Canonical artifact:

```text
artifacts/midogpp/30_expert_bank/uniform_b_v2_routing_authorized_expert_bank_v1/
```

Status: run `COMPLETE`; validator `PASS`; decision
`PROMOTED_AS_ROUTING_AUTHORIZED_EXPERT_BANK`; publication state
`ROUTING_AUTHORIZED`. The allowed claim scope is
`expert_bank_construction_only`.

## Evidence Sequence

```text
canonical Uniform-B cache
  -> independently train 9 source centers x 3 training seeds
       -> Stage-20 aggregate-prior union v2 source-inner study
            -> reviewed Stage-30 whole-bank promotion
                 -> validated Stage-40 GenerationLock
                      -> validated Stage-60 equal-union control policy lock
                           -> future matched routing comparisons
```

The source experiment is
`midogpp.cvae.uniform_b_geco_aggregate_prior_union_source_inner.v2`. It is
`COMPLETE`, validates `PASS`, and reports
`TARGET_METRIC_REACHED_REQUIRES_SEPARATE_PROMOTION`. It could not feed routing
directly. The Stage-30 experiment is the explicit authorization firewall.

## Promotion Gates

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| PS mean BACC | `>= 0.70` | `0.770112` | pass |
| worst training-seed PS mean | `>= 0.75` | `0.764562` | pass |
| PS minus P0 | `>= 0.005` | `+0.012764` | pass |
| posterior ceiling minus PS | `<= 0.01` | `0.001459` | pass |
| checkpoint records | `27` | `27` | pass |
| source identity overlap failures | `0` | `0` | pass |
| classifier convergence failures | `0` | `0` | pass |
| aggregate-prior fallback | none | none | pass |

The PS training-seed mean BACCs are `0.772334` for seed `17`, `0.773440` for
seed `42`, and `0.764562` for seed `101`. The matched source-study arm means
are `0.757348` for P0, `0.770112` for PS, `0.771571` for posterior samples,
`0.771145` for posterior means, and `0.766401` for the PCA-only reference.

## Promoted Bank

The bank retains all 27 independently trained experts: eligible source centers
`0,1,2,3,5,6,7,8,9` crossed with training seeds `17,42,101`. It also carries
nine source-local decoding frames and source-only class-conditional
full-shrinkage aggregate-prior states. No held-out metric selected an expert
or checkpoint seed.

All promoted checkpoint hashes, frame hashes, sampler states, source-row
hashes, and input artifact hashes are validated fail-closed. Checkpoints were
materialized as same-filesystem hard links, so promotion avoids a second
physical checkpoint copy while content validation protects the logical
artifact.

Key identities:

- source protocol hash: `70d54442a031a43e`
- promotion protocol hash: `5b3087f3aa41c388`
- bank lock: `9972a41dcd4814cd`
- equal-union control lock: `cddbcc3b3343fe38`

## Canonical Routing Control

`uniform_b_v2_equal_union_ps` is the mandatory direct control for future
routing experiments. For every held-out target `H`, it:

- excludes expert `H`;
- includes all eight non-target source centers;
- generates 1,024 samples per class, allocated as 128 per source;
- crosses training and generation seeds `17,42,101`;
- reports each replicate and the predeclared mean without seed selection;
- uses no target-conditioned source weights.

A comparison router and this control must share candidate eligibility, total
synthetic budget, generation RNG, shuffles, classifier settings, evaluation
rows, and reporting aggregation. The now-frozen control still requires fresh
matched Stage-70 scoring: the promotion evidence used seven-source source-
inner tasks, whereas a real held-out target fold has eight eligible source
experts.

## Claim And Test-Consumption Boundary

The promotion preserves the thesis requirement of independently trained CVAE
experts. It authorizes the frozen bank as input to routing or expert-selection
experiments but does not claim that any router succeeds. It also does not
establish Stage-40 outer generation quality or Stage-70 downstream utility.

The source-inner evaluation labels are recorded as
`CONSUMED_FOR_WHOLE_BANK_ADOPTION`. They may be reused for locked-control
scoring but may not be reused as fresh bank-selection evidence. In particular,
they cannot select a source center, one of the three checkpoint seeds, router
features, router hyperparameters, or a routing policy.

Future routing operates over source-center experts. The three training seeds
remain aligned replications; they are not 27 independently selectable routing
candidates.

## Reproducibility Caveat

The promoted contents and inputs are hash-locked, but the provenance snapshot
records `repository_dirty=true` at revision
`40221038ca714bf33fd21582857d21fa1db4e6f3`, with repository-status hash
`ccbf122866d4495dc6c19fd93f907c6945d1d48a65ab1777277fb19b93d4fc58`.
This does not invalidate the locked artifact, but it weakens reconstruction
from the commit alone. Preserve the exact diff or regenerate and revalidate
from a clean committed revision before thesis archival.

## Next Evidence

The Stage-40 GenerationLock and Stage-60 equal-union policy lock are now
complete and independently validated. Stage 60 consumes only the authorized
bank and GenerationLock and freezes 81 replicates and 648 target-excluded
source assignments under policy lock `4b9ea514308b084f`; it computes no BACC,
macro-F1, routing quality, or downstream synthetic utility.

The first protocol-clean comparison is now frozen: a non-selecting exact-match
metadata compatibility lock (`4b46b3d157b07781`) and an all-maximum-ties
policy lock (`27f16953b32c46cd`). The policy retains all 81 seed replicates and
153 assignments without target samples, labels, NELBO, or utility evidence.

The substantive source-inner candidate utility/regret surface and policy are
now frozen; see
`docs/wiki/03-experiments/midogpp-uniform-b-v2-source-inner-utility-regret-policy-v1.md`.
No fold met its single-source uncertainty gate, so every outer fold reuses the
exact equal-union fallback. A separate target-evaluation artifact must now be
authorized before matched Stage-70 scoring against equal-union and metadata
baselines. Routing quality and downstream utility remain unknown until that
fresh target evaluation is run.
