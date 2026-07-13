# MIDOG++ Experiment Workspace

This directory owns staged experiment configs, the capability registry, the
artifact catalog, and shared protocol defaults. Reusable implementation lives
only under `src/midogpp_thesis/`; generated evidence lives only under
`artifacts/midogpp/`.

The stage order makes the claim boundary explicit:

1. `10_real_feature_reference`
2. `20_cvae_preservation`
3. `30_expert_bank`
4. `40_prior_and_generation`
5. `50_all_candidate_utility_matrix`
6. `60_routing_and_composition`
7. `70_frozen_policy_downstream`
8. `90_oracles_and_diagnostics`

Stages 50 and 90 are diagnostic and can never select a deployable policy.
Stages 30 through 70 remain planned until their new MIDOG++ implementations
and protocol tests are reviewed; removed legacy runners are not compatibility
fallbacks.

Core workspace files:

- `registry.yaml`: experiment stage, status, dependencies, runners, and claim scope.
- `artifact_catalog.yaml`: logical artifact IDs, canonical paths, hashes, and reuse policy.
- `shared/workspace.yaml`: canonical repository roots.
- `shared/protocol_defaults.yaml`: held-out-center and leakage guardrails.

Run from the repository root:

```bash
conda run -n thesis python -m midogpp_thesis workspace validate
conda run -n thesis python -m midogpp_thesis workspace list
conda run -n thesis python -m midogpp_thesis workspace command \
  midogpp.cvae.tuned_classifier_preservation.v1
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.tuned_classifier_preservation.v1
```

`prepare` resolves `artifact://` and `output://` references, verifies cataloged
hashes, and writes a resolved config plus input-provenance manifest. Missing,
rejected, claim-incompatible, or hash-mismatched inputs fail closed.
