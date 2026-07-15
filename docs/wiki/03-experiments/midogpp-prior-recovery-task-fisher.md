# MIDOG++ Prior Recovery And Task-Fisher Preservation

## Purpose And Status

This page documents the implemented prior-recovery protocol and the completed
Stage-10 matched-reference and Stage-20 source-inner results. It does not report
an outer-preservation result.

Current status: the matched Stage-10 v2 bundle validates `PASS`; the Stage-20
source-inner bundle is `COMPLETE`, passes its full validator, and reports
`NEGATIVE_GATE_COMPLETE`. Seven of nine locks are conditional, while centers
`5` and `9` retain the standard-normal fallback. Because
`factorial_triggered=false`, the registered outer v1 is blocked by design.
These workstation bundles are not yet synced locally, and their catalog
destinations still carry `TODO_VERIFY_ARTIFACT` lifecycle labels. Earlier
invalid or terminated partial roots remain non-evidence.

The separate training-seed stability experiment is implemented and registered
but has not been production-run. It has no result on this page.

## Evidence Sequence

```text
eligible-only Stage-10 matched reference v2
  -> fully nested Stage-20 source-inner RecipeLocks
       -> bounded training-seed consensus RecipeLocks
            -> planned Stage-30 independent expert bank
                 -> planned Stage-40 frozen-expert generation validation
       -> conditional Stage-20 outer A/B/C/D preservation (scoring only)
```

The sequence contains two firewalls. Only validated fold-level consensus locks
from the bounded stability bundle may freeze a recipe for Stage 30. Scalar
seed-42 locks remain one-seed source-inner evidence. Outer preservation metrics
are scoring-only and may never flow backward into model or routing selection.
Stage 40 remains a later post-expert-bank generation-validation stage.

## Stage-10 Matched Reference v2

Experiment and config:

```text
midogpp.real_feature.eligible_tuned_predict_reference.v2
experiments/midogpp/stages/10_real_feature_reference/configs/eligible_tuned_real_reference_v2.yaml
```

The reference uses the corrected full 2560-dimensional Virchow2 cache, the
nine eligible centers `0,1,2,3,5,6,7,8,9`, classifier seed `23`, experiment
seed `42`, and the frozen 10-spec grid identified by
`5abd0897d02bdcaa`. The grid retains `C in {0.01,0.1,1,10,100}` and
`class_weight in {none,balanced}` while fixing `max_iter=5000` so numerical
under-budget candidates cannot invalidate an otherwise complete bundle.
Center `4` is excluded. For each outer target, classifier
selection uses only source-inner folds and the target center is absent from
fit/selection. Final target labels are scoring-only. The threshold policy is
sklearn `predict`, so this is a separate matched denominator rather than a new
interpretation of the validated fixed-0.5 v1 artifact.

Canonical output:

```text
artifacts/midogpp/10_real_feature_reference/eligible_tuned_real_reference_v2/seed42/
```

Expected outputs:

```text
config.resolved.yaml
provenance/input_artifacts.json
manifests/protocol_manifest.json
reports/leakage_provenance_report.json
tables/source_inner_classifier_tuning.csv
tables/classifier_tuned_source_results.csv
tables/classifier_tuned_predictions.csv
```

For a complete run, this is a content- and provenance-bound denominator rather
than just a directory of tables. `reference_bundle_hash` binds the tuning,
held-out-result, and prediction table contents; `protocol_hash` binds that
content identity to the eligible-center set, exact classifier grid, predict
policy, manifest hash, and feature-cache hash. The resolved config and
`provenance/input_artifacts.json` must independently reproduce the registered
dataset/cache identities and SHA-256 values, and the passing leakage report
must carry the same protocol hash. The outer Stage-20 protocol additionally
records this reference protocol/bundle identity, each center's classifier-spec
hash and real BACC, and the SHA-256 of the reference protocol manifest.

Claim boundary: `real_feature_transfer_only`. This artifact can be the matched
real denominator for Stage 20; it cannot establish CVAE preservation, prior
quality, expert quality, routing, generation, or downstream utility.

### Validated result

The full bundle validator passes on `xai-master`. Across the nine eligible
held-out centers, mean BACC is `0.740312` and mean macro-F1 is `0.737205`.
Center `1` is worst at `0.679245` BACC and center `6` is best at `0.792350`.
The protocol hash is `786589b799d61b14`, and the content-bound reference
bundle hash is `995aa193c82ee7ec`.

## Fully Nested Source-Inner Gate

Experiment and config:

```text
midogpp.cvae.prior_recovery_source_inner.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_source_inner_v1.yaml
```

For every outer center `H`, the runner removes all rows from `H`. Each remaining
eligible center becomes an inner pseudo-target `I`. The fixed `C=0.01` is an
evidence-informed, predeclared design constant inherited from the earlier
Stage-10 source-inner result; that artifact is not a runtime dependency and
Stage 20 does not resweep `C`. Its distinct two-spec grid, hash
`59b9fa2a008dedc5`, varies only
`class_weight in {none, balanced}`. That choice is selected on the deeper
centers excluding `H` and `I`. Every final real-classifier fit, fixed source-fit
PCA128 frame, CVAE training, Task-Fisher probe, and sampler fit also excludes
both `H` and `I`. Inner labels may select the sampler/objective recipe. The
real outer rows and labels are never passed to this run.

PCA128 is one locked preprocessing dimension, not a PCA sweep. Each fold frame
is fitted once and stored under a key binding the fold/rows, dataset and feature
cache, PCA policy, runtime protocol, code version, and numerical-library
versions. Each CVAE checkpoint is written incrementally with an exact
training-key sidecar. Reissuing the same workspace command resumes exact hits;
identity drift is a miss and matching corruption is a hard protocol error. The
terminated pre-v2 partial checkpoints lack this identity and are not reusable
under `prior_recovery_v2_resume`.

The initial sampler gate uses the exact configured families
`standard_normal`, `class_conditional_diagonal_total_moment`, and
`class_conditional_shrinkage_full_total_moment` with an isotropic CVAE
objective:

- arm `A`: standard-normal prior sampling;
- arm `C`: class-conditional ex-post aggregate-posterior sampling, testing
  diagonal total-moment and shrinkage-full total-moment families.

The requested family must be realized for both classes. The predeclared sampler
gate requires at least `0.05` mean preservation-ratio improvement over `A` and
at least six inner-center wins. If viable candidates are within `0.01`, the
diagonal family wins the tie. If no conditional sampler passes, the valid lock
remains `A`, `factorial_triggered=false`, and the outer experiment is blocked.

Only after `C` passes does the runner evaluate the Task-Fisher cells:

- arm `B`: Task-Fisher objective with standard-normal sampling;
- arm `D`: Task-Fisher objective with the selected conditional sampler.

`D` is locked only if it meets the same improvement/win gate against `A`, adds
more than `0.01` preservation ratio over `C`, and keeps mean decode and
posterior BACC regressions within `0.01`. Otherwise the conditional lock remains
`C`. Incomplete Task-Fisher cells invalidate a lock when the conditional
factorial was required.

Task-Fisher fits a source-only logistic probe in the source-fit PCA frame,
builds a trace-normalized positive-semidefinite input-Fisher metric, and uses it
in the CVAE reconstruction loss with locked `alpha=1.0`. It changes
reconstruction geometry; it is not an auxiliary classifier loss. The code
names the training criterion a normalized beta objective and intentionally
does not call it an ELBO or NELBO.

Canonical output:

```text
artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/
```

The bundle must include resolved config/input provenance, protocol and
selection-evidence manifests, checkpoint, Task-Fisher, and feature-frame
indexes, nine hashed `manifests/recipe_locks/<center>.json` files,
gate/leakage/runtime-summary/run-state reports, source-inner metrics, nested
classifier-tuning rows, nested real references, sampler realizations, runtime
timings, identity audits, persisted Task-Fisher states, and content-addressed
checkpoints.
`tables/checkpoint_reuse_audit.csv` is required: it enforces `A/C` checkpoint
identity in every source-inner fold and, when the Task-Fisher cells are
triggered, `B/D` checkpoint identity plus paired `A/B` initialization and
stochastic streams. The source-inner validator also requires equal per-class
generation counts across the compared arms for each outer/inner fold, training
seed, generation seed, and representation role.

`tables/runtime_timings.csv` and `reports/runtime_summary.json` report phase
costs and cache hits with `claim_scope=diagnostic_only` and
`used_for_selection=false`. They are deliberately excluded from the
selection-evidence hash and cannot alter a lock.

The nested real denominator and every generated arm are evaluated with the
same frozen classifier specification and sklearn `predict` policy for that
fold. Here, "frozen classifier" means frozen hyperparameters/policy and bound
specification hash, not reuse of one fitted weight vector across different
representations. Preservation is chance-corrected BACC:

```text
preservation_ratio = (BACC_generated - 0.5) / (BACC_real - 0.5)
```

The denominator must meet the predeclared real-BACC floor `0.55`; otherwise the
ratio is invalid rather than evidence for or against a sampler.

Claim boundary: `cvae_recipe_lock_only`. This scalar lock bundle supplies the
per-seed computation but is not the registered Stage-30 input; validated
cross-seed consensus locks may feed the planned Stage-30 model recipe. It is
not outer preservation, routing, compatibility, expert-selection, or
downstream-utility evidence.

### Validated source-inner result

The seed-42 source-inner bundle is `COMPLETE` and passes the full validator.
All nine recipe locks validate:

| Outer center | Locked arm | Objective | Sampler |
| --- | --- | --- | --- |
| `0,6,7,8` | `D` | Task-Fisher | full conditional |
| `1,2` | `C` | isotropic | full conditional |
| `3` | `C` | isotropic | diagonal conditional |
| `5,9` | `A` | isotropic | standard normal |

The gate records `n_locks=9`, `n_valid_locks=9`,
`n_conditional_locks=7`, `factorial_triggered=false`, and
`status=NEGATIVE_GATE_COMPLETE`. The selection-bundle hash is
`1e929d05ff987ad9`; the protocol hash is `dd7ca955d79fade4`.

The negative gate is substantive rather than numerical or procedural. Center
`5` is borderline and generation-seed-sensitive: its best diagonal sampler
improves mean preservation ratio over `A` by `+0.109849`, but wins only `5/8`
strict inner comparisons. Center `9` is inconsistent: its best full sampler
improves the mean by `+0.087841`, but wins only `4/8`. Both therefore miss the
predeclared six-win requirement despite clearing the mean-delta threshold. The
full bundle has valid locks, sampler realizations, checkpoint audits, identity
audits, and zero recorded overlap. Training-seed stability has not yet been
measured because this run fixed the training seed at `42`.

Implication: the result supports seven conditional source-inner recipe locks
and two standard-normal fallbacks, not a universal conditional-prior or
Task-Fisher claim. It also does not authorize changing the gate post hoc.

## Conditional Outer 2x2 Factorial

Experiment and config:

```text
midogpp.cvae.prior_recovery_outer.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_outer_v1.yaml
```

The runner imports the matched Stage-10 v2 reference and validates/recomputes
the source-inner lock bundle. It runs only if all nine locks are valid and each
selects conditional arm `C` or `D` with `factorial_triggered=true`.

Current status: `BLOCKED BY VALID SOURCE-INNER GATE`. Centers `5` and `9` lock
arm `A`, and the source-inner report records `factorial_triggered=false`.
Running the registered outer v1 against this bundle must fail closed; no outer
preservation decision exists.

| Arm | Objective | Sampler |
| --- | --- | --- |
| `A` | isotropic | standard normal |
| `B` | Task-Fisher | standard normal |
| `C` | isotropic | locked class-conditional ex-post sampler |
| `D` | Task-Fisher | locked class-conditional ex-post sampler |

The factorial uses training seeds `17,42,101` and generation seeds
`17,42,101`. Checkpoint identity is audited so `A/C` share the isotropic model,
`B/D` share the Task-Fisher model, and objective comparisons use paired
initialization/stochastic streams. Decode, posterior, and prior rows are all
required for complete coverage. For each outer center, training seed,
generation seed, and representation role, the validator also requires the same
two-class generation-count vector across all four arms. This prevents an arm
from receiving a larger source-generation budget.

The outer run is cryptographically bound to the source decision boundary. All
nine locks must validate against one source-inner protocol and one
`selection_bundle_hash`; the outer protocol records the source protocol and
selection-evidence file SHA-256 values, embeds each lock payload and lock hash,
and binds every metric row to its lock. A changed, mixed, stale, or incomplete
source bundle therefore blocks the outer run instead of silently changing the
factorial recipe.

The predeclared positive-preservation decision requires complete valid
coverage, valid locks, mean locked-policy preservation ratio at least `0.80`, a
positive delta over `A` for every training seed, at least seven center wins,
the paired worst-center BACC guard, and decode/posterior safety. Failing the
positive gate after a complete valid factorial produces
`status=NEGATIVE_PRESERVATION` while retaining
`claim_scope=cvae_preservation_only`. `status=INCOMPLETE_OR_INVALID_DIAGNOSTIC`
with `claim_scope=diagnostic_only` is reserved for incomplete or invalid runs.
A complete valid negative is therefore protocol-clean negative evidence for
the locked preservation hypothesis, not a failed or diagnostic-only artifact.
It does not support a prior-recovery benefit and cannot be used to retune the
recipe.

Canonical output:

```text
artifacts/midogpp/20_cvae_preservation/prior_recovery_outer_v1/seeds17_42_101/
```

Expected tables and reports include `tables/preservation_metrics.csv`,
`tables/sampler_realizations.csv`, `tables/paired_deltas.csv`,
`tables/aggregation_summary.csv`, `tables/checkpoint_reuse_audit.csv`,
`tables/identity_overlap_audit.csv`, `tables/runtime_timings.csv`,
protocol/coverage/checkpoint/Task-Fisher/feature-frame manifests, and
decision/leakage/runtime-summary/run-state reports, plus persisted states and
checkpoints.
The outer validator requires the sampler-realization table and binds its
requested/realized families, fallback metadata, and state hashes to the metric
rows.

Claim boundary: every complete valid positive or negative factorial supports
only `cvae_preservation_only`. Outer target labels are scoring-only, and all
outer metrics have `may_feed_model_recipe=false` and
`may_feed_deployable_selection=false`. They may never choose or revise a CVAE
objective, prior sampler, expert, generation setting, router, or composition
policy.

## Existing Workspace Commands

Run from the repository root after installing the checkout into the `thesis`
environment. The first commands reproduce or resume the registered seed-42
reference and source-inner experiments. The final pair runs the separate
bounded training-seed stability experiment.

```bash
conda run -n thesis python -m midogpp_thesis workspace validate

conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.real_feature.eligible_tuned_predict_reference.v2
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.real_feature.eligible_tuned_predict_reference.v2

conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_source_inner.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_source_inner.v1

conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1
```

Do not run `midogpp.cvae.prior_recovery_outer.v1` with the current lock bundle.

## Registered Training-Seed Stability Check

Run one bounded source-inner stability check with training seeds `17,42,101`
and the existing generation seeds `17,42,101`, then stop Stage-20 optimization
and proceed to the planned Stage-30 provenance-clean expert bank.

Status: `IMPLEMENTED AND REGISTERED, NOT YET PRODUCTION-RUN`. The experiment
`midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1` fully
crosses training and generation seeds `17,42,101`. It shares deterministic
nested-classifier, real-reference, identity-audit, and PCA preparation once per
outer/inner fold `(H,I)`. When Task-Fisher is needed, one fitted Task-Fisher
state is shared across training seeds for that `(H,I)`. Each training seed has
distinct initialization and stochastic-stream identities; posterior and prior
generation noise is paired by generation seed across training arms. The
recomputable contract is persisted in `tables/rng_pairing_audit.csv`. Source
class-count budgets remain fixed. The bundle writes 27 training-seed wrappers
and nine consensus locks.

The frozen consensus rule exports `D` only for unanimous `D` with one sampler
family; exports `C` when all seeds share one conditional family and any seed
selects `C`; and conservatively exports standard-normal `A` when arm or sampler
family choices disagree. Any invalid child lock disables export. Structural
bundle validity and Stage-30 readiness remain separate fields.
`reports/publication_state.json` starts as `PENDING`, becomes `PUBLISHED` only
after the complete bundle validates, and becomes `FAILED` if publication
validation fails. Only `PUBLISHED` is consumable. The Stage-30 loader then
requires every consensus lock in the bundle to be valid and export-ready before
returning consensus lock `H` for expert-bank fold `H`.

This stability input changes neither neighboring evidence boundary: outer v1
continues to consume the scalar source-inner locks and remains blocked by its
original gate, while scalar seed-42 evidence remains a one-training-seed result.
No scientific stability result exists until the canonical production artifact
is run, validated, and published.

## Validation Before Interpretation

The metrics above are limited to the two workstation bundles that passed their
full validators. Before catalog promotion or thesis-facing reuse, preserve the
resolved config, input provenance, protocol manifest, accepted leakage report,
complete tables, identity checks, and tamper-evident checkpoint/Task-Fisher
indexes during local sync. Any future outer artifact additionally needs
`manifests/coverage_manifest.json` with `status=PASS` and a decision report
that recomputes from the metric rows, plus a validated
`tables/sampler_realizations.csv` bound to those rows.
