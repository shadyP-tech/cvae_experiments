# Stage 40: Prior and Generation

This stage evaluates frozen expert sampling and generation behavior, including
prior recovery and generation diagnostics. Generation budgets, temperatures,
and sampling settings must be frozen without held-out target-evaluation
metrics.

Stage 40 begins only after a provenance-clean Stage-30 expert bank exists. Its
job is to validate generation from those frozen independently trained experts.
The Stage-20 source-inner prior-recovery implementation selects a possible
training/sampler recipe and conditionally evaluates preservation on pooled
source models; it does not constitute Stage-40 generation validation and does
not activate this stage.

Fidelity, latent distance, or support NELBO remain proxies. They do not prove
routing quality or downstream utility without the relevant later-stage test.

Status: `PLANNED`. Prior and generation code from rejected or cross-dataset
lineages was not promoted into the active package.
