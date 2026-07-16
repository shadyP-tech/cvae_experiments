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
- `configs/prior_recovery_source_inner_training_seed_stability_v1.yaml`
- `configs/learned_conditional_prior_source_inner_v2.yaml`
- `configs/task_fisher_shrinkage_source_inner_v2.yaml`
- `configs/prior_recovery_outer_v1.yaml`

They use logical artifact IDs for the dataset contract, corrected `xyxy`
cache, and their declared real-feature inputs, and write only to stage-20
canonical artifact roots. The scalar source-inner prior-recovery result is
complete and validated on `xai-master`; its outer factorial is blocked by the
predeclared gate. The training-seed stability surface is implemented and
registered; its validated workstation status is documented separately in the
current-state pages. Two additive v2 source-inner mechanism studies are now
implemented and registered but have not been production-run. They cannot
publish or replace a `RecipeLock`. The other preservation configs retain their
documented thesis-facing or diagnostic status.

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
2. `midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1`
   repeats the exact source-inner computation over training seeds
   `17,42,101`, fully crossed with generation seeds `17,42,101`. It writes 27
   wrapped seed locks and nine predeclared consensus locks. Those consensus
   locks, not the scalar seed-42 locks, are the registered Stage-30 recipe
   input. Deterministic preparation and Task-Fisher state are shared per
   outer/inner fold `(H,I)`; training RNG identities remain distinct by
   training seed, while posterior/prior noise is paired by generation seed and
   recorded in `tables/rng_pairing_audit.csv`.
3. `midogpp.cvae.prior_recovery_outer.v1` remains unchanged and may run only
   when all nine scalar source-inner
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

## Separate V2 Mechanism Studies

The v2 studies answer two new source-inner questions without reopening the v1
recipe decision. Both use the corrected `xyxy` Virchow2 cache, source-fit
PCA128, all eligible outer/inner folds, and the fully crossed training and
generation seeds `17,42,101`.

| Experiment | Fixed axis | Compared axis | Purpose |
| --- | --- | --- | --- |
| `midogpp.cvae.learned_conditional_prior_source_inner.v2` | stochastic isotropic objective | `A` standard normal, `C-diag` ex-post conditional diagonal, `E` jointly learned class-conditional diagonal Gaussian | test whether learning `p(z|y)` through the KL reduces the training/sampling mismatch |
| `midogpp.cvae.task_fisher_shrinkage_source_inner.v2` | standard-normal prior | `alpha in {0,0.05,0.10,0.25}` | test whether shrinking the source-only rank-one Task-Fisher metric improves preservation stability |

Arm `E` uses
`p(z|y)=Normal(mu_y, diag(sigma_y^2))`, with two learned class rows for
`mu_y` and `rho_y`, bounded log variance
`6*tanh(rho_y/6)`, an analytic latent-dimension-normalized KL, zero prior
weight decay, and separate gradient clipping. Its base encoder/decoder
initialization and stochastic training stream are paired with arm `A` for each
`(H,I,training seed)`. `C-diag` reuses the corresponding `A` checkpoint and
fits its total-moment diagonal sampler from source-fit posterior state only.

The Fisher study fits one raw source-only Fisher state per `(H,I)` and derives
all nonzero metrics from that same state:

```text
M_alpha = (I + alpha * F_tilde) / (1 + alpha)
```

`alpha=0` is the literal isotropic `metric=None` training path. All alpha arms
share initialization and stochastic streams within a training seed. Evaluation
epsilon is paired across arms and training seeds by `(H,I,generation seed,
class,row,stream)` in both studies, and generation budgets reproduce the source
`y_fit` class counts.

The studies write v2-owned checkpoints, state indexes, exact grid/coverage
manifests, RNG and initialization audits, paired deltas, child decisions, and
per-outer consensus decisions. Invalid A/E state invalidates the learned-prior
decision; an unavailable `C-diag` disables only the secondary E-vs-C
comparison; a finite but mechanism-ineligible E remains valid negative
evidence. Invalid raw Fisher state preserves the literal alpha-zero baseline
but makes the complete Fisher decision invalid rather than silently inventing
a nonzero metric.

Both outputs have `claim_scope=cvae_source_inner_study_only` and hard-code
`may_feed_model_recipe=false` and `may_feed_deployable_selection=false`. They
emit no publication state and are forbidden as Stage-30, Stage-40, routing,
compatibility, generation, or downstream-utility inputs. Running either study
does not alter the scalar v1 gate, the published training-seed consensus locks,
outer v1, or the existing Stage-30 input edge.

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

Run the bounded stability panel once with:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
```

The canonical workstation stability artifact is complete and published as
recorded in the current-state documentation. A reproduction must validate all
27 seed locks, all nine consensus locks, the seed-free preparation, shared
Task-Fisher states, distinct training identities, paired generation-seed
policy, and leakage reports. It publishes
`reports/publication_state.json` only after validation; `PENDING` and `FAILED`
are not consumable by Stage 30.

Run the two independent, non-adoptive v2 studies only through their registered
workspace entries:

```bash
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.learned_conditional_prior_source_inner.v2

conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.task_fisher_shrinkage_source_inner.v2
```

Their canonical destinations are:

```text
artifacts/midogpp/20_cvae_preservation/learned_conditional_prior_source_inner_v2/seeds17_42_101/
artifacts/midogpp/20_cvae_preservation/task_fisher_shrinkage_source_inner_v2/seeds17_42_101/
```

Status: `IMPLEMENTED AND REGISTERED, NOT YET PRODUCTION-RUN`. Do not infer a
mechanism result until a complete canonical bundle passes its own validator.
Neither run is a prerequisite for proceeding with the already frozen Stage-30
consensus locks.

Independently of the stability run, only when the scalar source-inner bundle
validates with `factorial_triggered=true` run:

```bash
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_outer.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_outer.v1
```

Canonical outputs are:

```text
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_training_seed_stability_v1/seeds17_42_101/
artifacts/midogpp/20_cvae_preservation/prior_recovery_outer_v1/seeds17_42_101/
```

The source-inner bundle contains the resolved config and input provenance,
protocol/selection/checkpoint/Task-Fisher/feature-frame manifests, nine hashed
`RecipeLock` files, gate, leakage, runtime-summary, and run-state reports,
nested classifier-tuning evidence, nested real references, source-inner
metrics, sampler realizations, runtime timings, identity audits, persisted
Task-Fisher states, and content-addressed checkpoints. The stability bundle
adds per-training-seed and consensus locks,
`reports/stability_decision.json`, `reports/publication_state.json`, and
`tables/rng_pairing_audit.csv`. The outer bundle
contains the resolved config and input provenance,
protocol/coverage/checkpoint/Task-Fisher/feature-frame manifests, decision,
leakage, runtime-summary, and run-state reports, preservation metrics, runtime
timings, sampler realizations, paired deltas, aggregation summary,
checkpoint-reuse and identity audits, persisted Task-Fisher states, and
content-addressed checkpoints. The outer
validator requires `tables/sampler_realizations.csv` and cross-checks its
sampler identities against the preservation metric rows.

## Claim Boundaries

- Validated scalar source-inner `RecipeLock` files are evidence for one
  training seed. Only validated fold-level consensus locks from the bounded
  stability bundle may feed the registered planned Stage-30 expert-bank
  recipe. Stage 30 accepts them only when publication state is `PUBLISHED` and
  every consensus lock in the bundle is valid/export-ready, then loads fold
  `H`'s lock for fold `H`. Neither artifact is a preservation result, routing
  evidence, or a deployable selection.
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

Status: scalar source-inner `COMPLETE/PASS` on `xai-master`; stability
`COMPLETE/PUBLISHED`; the two additive v2 mechanism studies `IMPLEMENTED AND
REGISTERED, NOT YET PRODUCTION-RUN`; outer `BLOCKED BY VALID SOURCE-INNER
GATE`. The new v2 destinations remain `TODO_VERIFY_ARTIFACT`. A terminated
pre-resume source-inner partial root is non-evidence and its old checkpoints
are incompatible with `prior_recovery_v2_resume`; only checkpoints written
with exact sidecars are resumable.
