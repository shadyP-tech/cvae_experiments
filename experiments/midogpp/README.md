# MIDOG++ Experiment Workspace

This directory owns staged experiment configs, the capability registry, the
artifact catalog, and shared protocol defaults. Reusable implementation lives
only under `src/midogpp_thesis/`; generated evidence lives only under
`artifacts/midogpp/`.

Generated experiment evidence and canonical run outputs are distinct from
reusable data inputs. Frozen dataset contracts remain under
`datasets/midogpp/contract/`, and derived feature caches remain under
`datasets/midogpp/derived/features/`; both are consumed through cataloged
`artifact://` references.

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
Stage 30 now contains the validated Uniform-B v2 routing-authorized expert
bank. Stage 40 contains the validated Uniform-B v2 GenerationLock, and Stage 60
contains independently validated equal-union, metadata-compatibility, metadata
max-tie, source-inner utility, and utility/regret policy locks. The substantive
utility/regret gate selected no single source, so every outer fold is frozen to
the exact equal-union fallback. A separate target-evaluation artifact must now
be authorized for matched Stage-70 evaluation; no Stage-60 lock establishes
routing quality or downstream utility, and removed legacy runners are not
compatibility fallbacks.

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
