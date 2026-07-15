# Stage 30: Expert Bank

This stage builds and validates independently trained source-domain CVAE
experts. It owns experiment composition and provenance, not the model classes.

Each held-out-target fold must exclude the target expert from the deployable
pool. Checkpoints must record dataset contract, cache, source split, CVAE
variant, seeds, code revision, and content hashes. Imported or byte-identical
cross-dataset checkpoints fail this stage.

Status: `PLANNED`. No active expert-bank runner was carried forward from the
rejected legacy stack. A new implementation must live under
`src/midogpp_thesis/cvae/` and pass protocol review before activation.

The registered planning entry consumes the validated Stage-20 training-seed
consensus artifact from
`midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1`. The
fail-closed loader requires `reports/publication_state.json` status
`PUBLISHED`, validates the complete stability bundle, requires every consensus
lock bundle-wide to be valid and export-ready, and only then returns consensus
lock `H` for Stage-30 fold `H`. `PENDING`, `FAILED`, missing, invalid, or
non-exportable state blocks consumption. These locks may freeze a source-only
CVAE objective and prior-sampler recipe for independent expert training. They
do not make Stage 30 active and do not themselves prove expert quality. The
scalar seed-42 source-inner bundle is not the registered Stage-30 input.

Outer held-out-center preservation metrics from
`midogpp.cvae.prior_recovery_outer.v1` must never feed Stage-30 model or recipe
selection. The outer bundle is scoring-only preservation evidence.
