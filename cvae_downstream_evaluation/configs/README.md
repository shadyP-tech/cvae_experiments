# Configs

Config files in this directory are templates until every `TODO_LOCK_BEFORE_RUN` value is replaced with a concrete value and the protocol manifest is generated.

Config rules:

- Keep routing, generation, downstream training, fidelity, and reporting sections separate.
- Do not tune generation or classifier settings with target evaluation labels or target evaluation metrics.
- Keep dataset-specific settings under `datasets`.
- Keep implementation paths relative to the repository root unless an existing runtime requires otherwise.
- Prefer additive version names such as `direct_support_nelbo_selected_synthetic_downstream_v1`.

