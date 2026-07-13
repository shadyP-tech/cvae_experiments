# Stage 40: Prior and Generation

This stage evaluates frozen expert sampling and generation behavior, including
prior recovery and generation diagnostics. Generation budgets, temperatures,
and sampling settings must be frozen without held-out target-evaluation
metrics.

Fidelity, latent distance, or support NELBO remain proxies. They do not prove
routing quality or downstream utility without the relevant later-stage test.

Status: `PLANNED`. Prior and generation code from rejected or cross-dataset
lineages was not promoted into the active package.
