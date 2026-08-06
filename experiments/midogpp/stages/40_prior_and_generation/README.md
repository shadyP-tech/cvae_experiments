# Stage 40: Prior and Generation

This stage evaluates frozen expert sampling and generation behavior, including
prior recovery and generation diagnostics. Generation budgets, temperatures,
and sampling settings must be frozen without held-out target-evaluation
metrics.

The prerequisite now exists as the validated, routing-authorized Uniform-B v2
Stage-30 bank. Stage 40's job is to validate generation from those frozen
independently trained experts without modifying checkpoints or selecting a
training seed from held-out performance.
The Stage-20 source-inner prior-recovery implementation selects a possible
training/sampler recipe and conditionally evaluates preservation on pooled
source models; it does not constitute Stage-40 generation validation and does
not activate this stage.

Fidelity, latent distance, or support NELBO remain proxies. They do not prove
routing quality or downstream utility without the relevant later-stage test.

Status: `COMPLETE; VALIDATED; SETTINGS LOCK ONLY`. The registered
`midogpp.prior_and_generation.uniform_b_v2_generation_lock.v1` runner consumes
only the routing-authorized v2 bank and predeclares semantic lock
`34e551425710362e`. It freezes 81 target- and policy-independent source streams
(nine centers x three training seeds x three generation seeds), 81 held-out
target control replicates, the 1,024-per-class budget, 128-per-source prefixes,
source-local inverse frames, aggregate-prior states, deterministic shuffle
seeds, and the downstream classifier contract.

The Stage-40 run performs only one-sample-per-class shape, finiteness, inverse-
frame, and repeatability probes. Its distinct
`generation_settings_and_frame_lock` output may be consumed as frozen settings
after validation; ordinary `generation_diagnostics_only` outputs remain
non-selectable. No target data, routing score, NELBO, classifier fit, BACC, or
downstream utility is produced here. The workstation artifact passed its
independent validator rerun with 27 experts, 81 source streams, 81 target
replicates, and 162/162 health records passing. The Stage-20 source-inner PS
score remains mechanism/adoption evidence and is not a fresh Stage-40
eight-source result.

Reproducibility caveat: the production run records revision `40221038...` with
`repository_dirty=true` and repository-status hash `9fddbbc8828994dd4...`.
The artifact and all input bytes validate, but thesis archival still requires
committing/preserving the implementation diff or regenerating from a clean
revision.
