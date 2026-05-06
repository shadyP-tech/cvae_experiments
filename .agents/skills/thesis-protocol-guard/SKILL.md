---
name: thesis-protocol-guard
description: Blocking protocol-integrity review for CVAE metadata-routing thesis experiments. Use before implementing or reviewing routing, compatibility estimation, LOQDO/LODO folds, expert selection, support-set calibration, split construction, utility evaluation, or experiment configs. Checks for target leakage, split leakage, expert data sharing, target-support misuse, and confusion between proxy similarity and true NELBO utility.
---

# Thesis Protocol Guard

Treat this as a blocking thesis-integrity review. Reject or stop any change that violates the protocol, even if it improves metrics.

## Required Protocol

Every routing experiment must preserve this chain:

`Query -> Compatibility Estimation -> Routing Decision -> Expert Selection -> Utility (NELBO)`

Do not collapse later stages into earlier ones. Compatibility/proxy scores may choose an expert; only held-out true utility may evaluate whether that choice was good.

Definitions:

- True compatibility: `C_true(q, e) = -NELBO(q, e)`
- Proxy compatibility: metadata similarity, latent similarity, learned predictor score, support-set estimate, or other routing score
- Compatibility means expected utility. Similarity is only a proxy.

## Allowed Target-Support Use

A small support set from the held-out target/query domain may be used only to estimate target-local compatibility for that target fold, for example by evaluating candidate source experts on support NELBO.

It must:

- be disjoint from target evaluation samples
- not update source experts
- not train shared/global models
- not tune decisions using target evaluation utility
- not leak the target oracle expert from the evaluation set

## Hard Stops

Block the change if any item is true:

1. Target expert leakage:
   The target/query domain expert, target labels, target test NELBO, oracle expert, or target-domain identity is used to train, tune, calibrate, normalize, or select a router for that same held-out target.

2. LOQDO/LODO is broken:
   The declared held-out query/target domain group is not fully excluded from router fitting, learned compatibility fitting, threshold selection, hyperparameter tuning, or normalization, unless the protocol explicitly defines a disjoint target-support calibration step.

3. Support/evaluation separation is broken:
   Target support samples used for compatibility estimation overlap with target evaluation samples.

4. Target-local support rule is broken:
   Target support data updates source experts, trains a shared model, tunes source parameters, or uses target evaluation utility.

5. Expert isolation is broken:
   Source-domain generative experts are not independently trained on their own source-domain train split only, or routing modifies expert checkpoints post hoc.

6. Routing is evaluated against similarity/proxy instead of true utility:
   Selection quality is judged by metadata/latent similarity alone rather than held-out NELBO-derived utility, oracle gap, top-1 true-utility hit, rank, or Spearman with true utility.

7. Split leakage appears:
   Patient/slide/group IDs, sample paths, manifests, or caches overlap across train/validation/test/support/evaluation in a way that violates the declared protocol.

## Metric Priority

Evaluate routing methods in this order:

1. Top-1 oracle agreement
2. Spearman rank correlation with true utility
3. Normalized oracle gap
4. Seed/fold stability

A method is better only if improvement is consistent, stable, and beats the metadata baseline.

## Claim Discipline

- Do not claim utility alignment from similarity alignment alone.
- Do not claim learned methods are better unless they consistently beat metadata on the primary metrics.
- Negative results are valid findings.
- If oracle gap is low but top-1/Spearman are weak, state that utility loss is small but expert identification remains unreliable.

## Repo Inspection Map

Check the files touched by the change, plus these protocol surfaces when relevant:

Splits and leakage reports:

- `cvae_testing/src/data/shared_split.py`
- `cvae_testing/src/data/datasets/breakhis.py`
- `cvae_testing/src/data/datasets/camelyon17.py`
- `outputs/**/manifests/samples.csv`
- `outputs/**/reports/leakage_report.json`

Expert training and checkpoint provenance:

- `cvae_testing/src/train/train_experts.py`
- `cvae_testing/src/train/checkpoint_provenance.py`
- `cvae_testing/src/train/checkpoint_utils.py`

Routing/proxy scores:

- `cvae_testing/src/routing/router.py`
- `cvae_testing/src/routing/strategies.py`
- `cvae_testing/src/routing/registry.py`

Experiment orchestration:

- `cvae_testing/src/experiments/*.py`
- `cvae_testing/src/run_experiment.py`
- `cvae_testing/configs/experiments/**/*.yaml`

True utility evaluation:

- `cvae_testing/src/eval/evaluators/routing.py`
- `cvae_testing/src/eval/evaluators/learned_utility.py`
- `cvae_testing/src/eval/evaluators/latent_compatibility.py`
- `cvae_testing/scripts/**`

## Review Procedure

1. State the intended query domain, source expert domains, support split, and evaluation split.
2. Trace data flow from manifests/caches into training, compatibility estimation, routing, expert scoring, and reports.
3. Verify expert checkpoints are produced before post-hoc routing and are not modified by routing methods.
4. For LOQDO/LODO, verify the held-out query domain is excluded from router/utility-model fitting and all expected folds are reported.
5. Verify normalization, thresholds, alpha selection, hyperparameters, and model selection use only allowed training/support folds.
6. Verify final claims compare selected experts against true held-out NELBO utility.
7. Verify proxy-vs-utility diagnostics are included when metadata, latent, learned, or support-set proxy scores are used.

## Output Format

Lead with exactly one of:

`BLOCKED`: protocol violation found.

- Violation:
- Evidence:
- Minimal compliant fix:

`NEEDS EVIDENCE`: protocol may be valid but required split/provenance/utility evidence is missing.

- Missing evidence:
- Risk:
- Required check:

`PASS`: no protocol violation found in the inspected surface.

- Inspected surface:
- Residual risk:
- Untested artifact:

## Implementation Requests

Before coding:

1. Identify the protocol regime.
2. State the allowed data available to the router.
3. Implement only if the data flow is compliant.
4. If the compliant data flow is ambiguous, stop and request the missing protocol detail.
