# Stage 20: CVAE Preservation

This stage tests whether decoded or sampled CVAE representations preserve a
frozen real-feature classifier surface. A passing result supports only the
declared preservation claim.

Every run must identify the real comparator, CVAE checkpoint lineage, feature
frame, fit-only transformations, scoring-only label use, preservation ratio,
negative controls, and identity-overlap audit. Preservation is not routing,
prior quality, controllable generation, or downstream synthetic utility.

Reusable implementation lives in
`src/midogpp_thesis/cvae/preservation/`.

Canonical configs are:

- `configs/preservation_sanity_v1.yaml`
- `configs/preservation_gate_pca128_v1.yaml`
- `configs/preservation_condition_audit_v1.yaml`
- `configs/tuned_classifier_preservation_v1.yaml`
- `configs/prior_recovery_source_inner_v1.yaml`
- `configs/prior_recovery_outer_v1.yaml`

They use logical artifact IDs for the dataset contract, corrected `xyxy`
cache, and their declared real-feature inputs, and write only to stage-20
canonical artifact roots. The tuned-classifier result is the only currently
validated thesis-facing artifact. The two prior-recovery configs are active
implementation surfaces but have no result yet; the other three are diagnostic
surfaces.

## Prior-Recovery Selection And Evaluation Firewall

Prior recovery is split into two independent bundles so held-out outer metrics
cannot change the recipe.

1. `midogpp.cvae.prior_recovery_source_inner.v1` removes each outer center
   completely and uses each remaining center as an inner pseudo-target.
   `C=0.01` is an evidence-informed, predeclared design constant inherited
   from the earlier Stage-10 source-inner result; it is not a runtime input or
   a Stage-20 sweep. Within each `H/I` fold, only
   `class_weight in {none, balanced}` is selected on still-deeper centers.
   Every classifier fit, fixed PCA128 fit, CVAE/Task-Fisher fit, and sampler fit
   excludes both `H` and `I`. It produces per-outer `RecipeLock` files with
   `claim_scope=cvae_recipe_lock_only`.
2. `midogpp.cvae.prior_recovery_outer.v1` may run only when all nine source-inner
   locks are valid, all select a conditional arm (`C` or `D`), the leakage
   report passes, and `reports/gate_decision.json` records
   `factorial_triggered=true`. The outer run imports the eligible-only Stage-10
   matched reference v2 and evaluates the frozen 2x2 factorial.

The arms are:

| Arm | CVAE reconstruction objective | Prior sampler |
| --- | --- | --- |
| `A` | stochastic isotropic beta objective | standard normal |
| `B` | stochastic Task-Fisher beta objective | standard normal |
| `C` | stochastic isotropic beta objective | source-inner-selected class-conditional ex-post posterior sampler |
| `D` | stochastic Task-Fisher beta objective | the same locked conditional sampler family |

Task-Fisher changes the reconstruction geometry using a source-fitted,
trace-normalized logistic-probe Fisher metric. It is not an auxiliary
classification loss and the implementation deliberately makes no ELBO/NELBO
claim for this normalized beta objective.

The source-inner conditional-sampler gate requires the predeclared preservation
ratio improvement and inner-center wins, viable requested sampler families for
both classes, and complete evidence. Task-Fisher is eligible only after that
gate and must add value over `C` without violating decode/posterior safety.
Failure to pass the conditional gate is a completed negative source-inner
result and blocks the outer factorial; it is not permission to inspect outer
metrics and revise the recipe.

PCA128 is one fixed, source-fit preprocessing step per fold, not another
hyperparameter sweep. Exact protocol keys cache each fitted frame. CVAE
checkpoints are persisted incrementally under exact training-key sidecars, so
rerunning the same command resumes completed fits. Changed rows, cache hashes,
folds, classifier specifications, recipe/protocol hashes, code versions, or
library identities do not count as a cache hit. A matching corrupt entry fails
closed. Timing and cache-hit reports are diagnostic-only and cannot influence
the gate or any `RecipeLock`.

## Exact Run Sequence

The Stage-10 reference and source-inner selection bundle can be run before the
outer bundle. From an installed checkout:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.eligible_tuned_predict_reference.v2
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.eligible_tuned_predict_reference.v2

conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_source_inner.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_source_inner.v1
```

Inspect the source-inner gate at:

```text
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/reports/gate_decision.json
```

Only when it validates with `factorial_triggered=true` run:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_outer.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_outer.v1
```

Canonical outputs are:

```text
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/
artifacts/midogpp/20_cvae_preservation/prior_recovery_outer_v1/seeds17_42_101/
```

The source-inner bundle contains the resolved config and input provenance,
protocol/selection/checkpoint/Task-Fisher/feature-frame manifests, nine hashed
`RecipeLock` files, gate, leakage, runtime-summary, and run-state reports,
nested classifier-tuning evidence, nested real references, source-inner
metrics, sampler realizations, runtime timings, identity audits, persisted
Task-Fisher states, and content-addressed checkpoints. The outer bundle
contains the resolved config and input provenance,
protocol/coverage/checkpoint/Task-Fisher/feature-frame manifests, decision,
leakage, runtime-summary, and run-state reports, preservation metrics, runtime
timings, sampler realizations, paired deltas, aggregation summary,
checkpoint-reuse and identity audits, persisted Task-Fisher states, and
content-addressed checkpoints. The outer
validator requires `tables/sampler_realizations.csv` and cross-checks its
sampler identities against the preservation metric rows.

## Claim Boundaries

- Validated source-inner `RecipeLock` files may feed the planned Stage-30
  expert-bank recipe. They are not preservation results, routing evidence, or
  deployable selections.
- Outer target labels are scoring-only. Outer preservation metrics, arm wins,
  decision reports, and checkpoints may never select or revise a model recipe,
  sampler, expert, generation policy, router, or composition rule.
- Complete valid outer factorials retain `claim_scope=cvae_preservation_only`.
  They report `POSITIVE_PRESERVATION` when the positive gate passes and
  `NEGATIVE_PRESERVATION` when it does not. `diagnostic_only` is reserved for
  incomplete or invalid executions. Neither positive nor negative preservation
  evidence can support expert-bank, routing, compatibility, or
  downstream-utility claims.
- Stage 40 remains the post-expert-bank validation of frozen expert generation.
  Stage-20 aggregate-posterior prior recovery does not activate or replace it.

Status: `IMPLEMENTED, NO VALIDATED RESULT`. Both new output artifacts remain
`TODO_VERIFY_ARTIFACT`. A terminated pre-resume source-inner partial root is
non-evidence and its old checkpoints are incompatible with
`prior_recovery_v2_resume`; only checkpoints written with exact v2 sidecars are
resumable. No prior-recovery or Task-Fisher result exists until the required
bundles are completed and validated.
