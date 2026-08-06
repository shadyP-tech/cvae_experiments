# Stage 30: Expert Bank

This stage builds and validates independently trained source-domain CVAE
experts. It owns experiment composition and provenance, not the model classes.

Each held-out-target fold must exclude the target expert from the deployable
pool. Checkpoints must record dataset contract, cache, source split, CVAE
variant, seeds, code revision, and content hashes. Imported or byte-identical
cross-dataset checkpoints fail this stage.

Status: `ACTIVE; ROUTING-AUTHORIZED BANK AVAILABLE`. The completed experiment
`midogpp.expert_bank.uniform_b_v2_routing_promotion.v1` validates and promotes
the independently trained Uniform-B v2 bank. Its canonical artifact is:

```text
artifacts/midogpp/30_expert_bank/uniform_b_v2_routing_authorized_expert_bank_v1/
```

The bundle is `COMPLETE`, validation is `PASS`, and publication state is
`ROUTING_AUTHORIZED`. It contains all 27 experts: nine source centers crossed
with training seeds `17,42,101`. No expert or seed was selected by held-out
performance. The bank lock is `9972a41dcd4814cd`.

The promotion also freezes `uniform_b_v2_equal_union_ps` as the canonical
control for every future routing experiment. For target `H`, it excludes
expert `H`, uses all eight remaining source centers, generates 1,024 samples
per class as 128 per source, and reports every training/generation replicate
plus the predeclared mean without seed selection. The control lock is
`cddbcc3b3343fe38`.

This authorization is not routing evidence. It permits the Stage-30 output to
feed deployable-selection experiments, but it does not select a routing policy
or claim routing quality. The v2 source-inner labels are consumed for whole-bank
adoption and may not be reused as fresh evidence to select experts, seeds, or a
router. Future routing comparisons must freshly score the eight-source control
and candidate router under paired budgets, RNG, shuffles, and candidate pools.

The separate planned `midogpp.expert_bank.provenance_clean.v1` entry consumes
the validated Stage-20 training-seed
consensus artifact from
`midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1`. The
fail-closed loader requires `reports/publication_state.json` status
`PUBLISHED`, validates the complete stability bundle, requires every consensus
lock bundle-wide to be valid and export-ready, and only then returns consensus
lock `H` for Stage-30 fold `H`. `PENDING`, `FAILED`, missing, invalid, or
non-exportable state blocks consumption. These locks may freeze a source-only
CVAE objective and prior-sampler recipe for independent expert training. They
do not activate that alternative runner and do not themselves prove expert
quality. The
scalar seed-42 source-inner bundle is not the registered Stage-30 input.

Outer held-out-center preservation metrics from
`midogpp.cvae.prior_recovery_outer.v1` must never feed Stage-30 model or recipe
selection. The outer bundle is scoring-only preservation evidence.
