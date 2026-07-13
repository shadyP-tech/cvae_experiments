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
