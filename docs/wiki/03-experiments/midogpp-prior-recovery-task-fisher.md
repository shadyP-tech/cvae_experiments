# MIDOG++ Prior Recovery And Task-Fisher Preservation

## Purpose And Status

This page documents the implemented, unrun prior-recovery protocol. It does not
report a result.

Current status: `IMPLEMENTED, NOT RUN`. The three canonical output roots are
absent and their catalog labels remain `TODO_VERIFY_ARTIFACT`. No preservation
metric, gate outcome, Task-Fisher benefit, or thesis-facing decision exists
until the bundles are produced and validated.

## Evidence Sequence

```text
eligible-only Stage-10 matched reference v2
  -> fully nested Stage-20 source-inner RecipeLocks
  -> conditional Stage-20 outer A/B/C/D preservation
  -> planned Stage-30 independent expert bank
  -> planned Stage-40 frozen-expert generation validation
```

The sequence contains two firewalls. Source-inner `RecipeLock` files may freeze
a recipe for Stage 30. Outer preservation metrics are scoring-only and may
never flow backward into model or routing selection. Stage 40 remains a later
post-expert-bank generation-validation stage.

## Stage-10 Matched Reference v2

Experiment and config:

```text
midogpp.real_feature.eligible_tuned_predict_reference.v2
experiments/midogpp/stages/10_real_feature_reference/configs/eligible_tuned_real_reference_v2.yaml
```

The reference uses the corrected full 2560-dimensional Virchow2 cache, the
nine eligible centers `0,1,2,3,5,6,7,8,9`, classifier seed `23`, experiment
seed `42`, and the frozen 20-spec grid identified by
`16a7a1183ea3f65b`. Center `4` is excluded. For each outer target, classifier
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

## Fully Nested Source-Inner Gate

Experiment and config:

```text
midogpp.cvae.prior_recovery_source_inner.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_source_inner_v1.yaml
```

For every outer center `H`, the runner removes all rows from `H`. Each remaining
eligible center becomes an inner pseudo-target `I`; the real classifier,
source-fit PCA128 frame, CVAE training, Task-Fisher probe, and sampler fitting
use only the deeper source centers excluding both `H` and `I`. Inner labels may
select the recipe. The real outer rows and labels are never passed to this run.

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
selection-evidence manifests, checkpoint and Task-Fisher indexes, nine hashed
`manifests/recipe_locks/<center>.json` files, gate/leakage reports,
source-inner metrics, nested real references, sampler realizations, identity
audits, persisted Task-Fisher states, and content-addressed checkpoints.
`tables/checkpoint_reuse_audit.csv` is required: it enforces `A/C` checkpoint
identity in every source-inner fold and, when the Task-Fisher cells are
triggered, `B/D` checkpoint identity plus paired `A/B` initialization and
stochastic streams. The source-inner validator also requires equal per-class
generation counts across the compared arms for each outer/inner fold, training
seed, generation seed, and representation role.

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

Claim boundary: `cvae_recipe_lock_only`. A validated lock bundle may feed the
planned Stage-30 model recipe. It is not outer preservation, routing,
compatibility, expert-selection, or downstream-utility evidence.

## Conditional Outer 2x2 Factorial

Experiment and config:

```text
midogpp.cvae.prior_recovery_outer.v1
experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_outer_v1.yaml
```

The runner imports the matched Stage-10 v2 reference and validates/recomputes
the source-inner lock bundle. It runs only if all nine locks are valid and each
selects conditional arm `C` or `D` with `factorial_triggered=true`.

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
`tables/identity_overlap_audit.csv`, protocol/coverage/checkpoint/Task-Fisher
manifests, and decision/leakage reports, plus persisted states and checkpoints.
The outer validator requires the sampler-realization table and binds its
requested/realized families, fallback metadata, and state hashes to the metric
rows.

Claim boundary: every complete valid positive or negative factorial supports
only `cvae_preservation_only`. Outer target labels are scoring-only, and all
outer metrics have `may_feed_model_recipe=false` and
`may_feed_deployable_selection=false`. They may never choose or revise a CVAE
objective, prior sampler, expert, generation setting, router, or composition
policy.

## Exact Workspace Commands

Run from the repository root after installing the checkout into the `thesis`
environment. The outer pair is conditional on the validated source-inner gate.

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
  midogpp.cvae.prior_recovery_outer.v1
conda run -n thesis python -m midogpp_thesis workspace run \
  midogpp.cvae.prior_recovery_outer.v1
```

## Validation Before Interpretation

Do not add metrics to this page until each claimed bundle has its required
resolved config, input provenance, protocol manifest, accepted leakage report,
complete tables, identity checks, and tamper-evident checkpoint/Task-Fisher
indexes. The outer artifact additionally needs
`manifests/coverage_manifest.json` with `status=PASS` and a decision report
that recomputes from the metric rows, plus a validated
`tables/sampler_realizations.csv` bound to those rows.
