# Protocol Contract

Protocol status: `LOCKED_V1_NEEDS_SYNCED_ARTIFACTS` until the implementation writes audited manifests and run reports from synced workstation artifacts.

## Regime

The intended regime is LOQDO/LODO-style second-stage evaluation:

```text
held-out target domain H
source domains = all domains except H
unlabeled target support S_H = routing only
labeled target evaluation T_H = final downstream utility only
```

The protocol preserves:

```text
Query -> Compatibility Estimation -> Routing Decision -> Expert Selection -> Downstream Utility
```

For the v1 baseline, the compatibility estimator is direct support-set NELBO.
For the selection/aggregation extension, the deployable estimator is a learned
downstream utility predictor trained only on source-inner pseudo-target folds.
The downstream utility is balanced accuracy and macro-F1 on held-out target
evaluation data after synthetic embedding generation. Distributional fidelity is
secondary evidence only.

## Allowed Query Information

Before expert selection, the router may use:

- source expert checkpoints
- source expert provenance
- source-domain metadata
- source-trained artifacts
- unlabeled target support embeddings
- query metadata available at routing time
- source-inner validation utility and calibration statistics
- target-support-only distributional or preservation diagnostics

The primary protocol forbids target support labels.

## Forbidden Pre-Selection Information

Before routing, generation settings, and classifier settings are locked, the implementation must not use:

- target evaluation labels
- target evaluation NELBO
- downstream oracle expert labels
- target test metrics
- target evaluation distributional metrics
- direct target identity fields as predictive features
- target evaluation-tuned generation temperature or synthetic budget
- target evaluation-tuned classifier architecture or hyperparameters

Direct target identity fields include `target_domain`, `target_domain_id`,
direct fold identity, excluded-expert identity, oracle labels, and equivalent
aliases. They may appear in lineage, grouping, manifests, and final reports, but
not in deployable feature vectors, estimator inputs, normalization/calibration,
routing decisions, or aggregation weights.

## Expert Pool

Candidate experts must be independently trained source-domain CVAE experts. Under each held-out target fold, the target/query expert is excluded from the candidate pool except as an explicitly marked diagnostic reference that is not selectable.

Routing must not update source expert checkpoints.

## Required First-Stage Carryover

The implementation should reuse existing support-NELBO artifacts where possible:

- support/evaluation split logic
- candidate expert exclusion checks
- support-NELBO score matrices
- metadata routing baselines
- random expert diagnostics
- static embedding baselines when comparable

## New Second-Stage Requirements

For every target fold, support seed, generation seed, and classifier seed:

1. Compute or load support-NELBO scores for every candidate expert.
2. Select the deployment expert with `argmin(mean_support_nelbo)`.
3. Generate class-balanced synthetic embeddings from every candidate expert, not only the selected expert.
4. Keep each single-expert classifier in that expert's projected CVAE feature frame.
5. Train the same locked downstream head for every expert-generated synthetic dataset.
6. Evaluate each head on held-out target evaluation embeddings projected through the matching expert head.
7. Build the downstream oracle from the all-expert downstream matrix.
8. Compare selected expert utility to metadata, random, source-global, ensemble, and downstream oracle baselines.

## Selection/Aggregation Extension

The extended selection-first path adds the following frozen stages:

1. Build an atomic candidate manifest. A candidate is the tuple of expert
   checkpoint, source domain, checkpoint seed, generation mode, latent sampling
   setting, class prior rule, synthetic budget, generation seed, aggregation
   recipe when applicable, and classifier seed.
2. Tag every candidate as `selection_eligible` or `diagnostic_only`.
3. Freeze and hash candidate pool, generation settings, classifier settings,
   feature definitions, routing choices, metrics, and aggregation parameters
   before target evaluation metrics are written.
4. Write all-candidate downstream utility for the selection/aggregation path
   only to `diagnostic_downstream_utility.*`. Legacy direct-support reports may
   still read `all_expert_downstream_matrix.csv`, but deployable learned
   selectors must consume only allowed feature tables.
5. Train learned downstream utility estimators on source-inner pseudo-target
   folds only.
6. Apply top1, top-k uniform, or soft aggregation rules to allowed feature
   tables only.

The learned target is:

```text
E[U_downstream(q_inner_eval, e) | allowed_features(q_inner_support, e)]
```

Real held-out target BACC, macro-F1, downstream oracle ranks, and target-eval
fidelity metrics are never training labels or deployable feature columns.

## Primary Metrics

Routing-to-downstream alignment:

- `top1_downstream_oracle_hit`
- `spearman_neg_nelbo_vs_bacc`
- `downstream_oracle_gap_bacc`
- `downstream_oracle_gap_macro_f1`
- `top1_downstream_oracle_hit`
- `pairwise_preference_accuracy`
- `top_k_recall`

Downstream performance:

- `bacc`
- `macro_f1`
- `auroc` when binary or otherwise well-defined
- `auprc` when class imbalance makes it informative

Stability:

- support seed standard deviation
- generation seed standard deviation
- classifier seed standard deviation
- worst-domain performance
- bootstrap or fold/seed intervals computed over target/fold or seed-target rows

## Secondary Diagnostics

Distributional fidelity is secondary evidence:

- MMD
- energy distance
- Frechet embedding distance
- mean/covariance distance
- kNN precision, recall, density, coverage

Report correlations between fidelity metrics and downstream utility. Do not treat fidelity as downstream utility.

## Required Implementation Gates

- split overlap test by patient/slide/group/sample
- target expert exclusion test for deployable candidates and dense baselines
- feature whitelist and metadata identity tests
- estimator training rejection for real held-out target labels
- diagnostic matrix import/read firewall
- config freeze/hash before target metrics are written
- schema tests for matrix, feature, selection, candidate, and report artifacts
- artifact lineage check from every selection row to exactly one feature row and candidate row
- aggregation budget matching for claimed comparisons
- source-inner fold parity with the real held-out target protocol

## Claim Boundary

Allowed:

```text
Direct support-NELBO is a direct target-local compatibility estimate. This
experiment evaluates whether it transfers to useful synthetic embeddings under
held-out downstream target evaluation.
```

Forbidden:

```text
Lower support NELBO proves better generative quality or downstream utility.
The current CVAE is a true class-conditional generator.
Support-size-stratified results can rescue or overturn the predefined primary decision.
```
