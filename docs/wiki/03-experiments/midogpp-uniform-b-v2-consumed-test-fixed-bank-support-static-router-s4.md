# MIDOG++ Uniform-B v2 consumed-test fixed-bank support-static router S4

## Status and claim boundary

Experiment:
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4.v1`

Status: registered and runnable, but not yet executed. Its maximum evidence role
is `POST_HOC_CONSUMED_TEST_SENSITIVITY`; the immutable terminal decision is
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`.

This study asks whether using four whole support folds can identify a useful
single static A1 source action. It does not establish fresh routing utility,
confirm a policy, or authorize an action. Any result is terminal descriptive
evidence from already-consumed test bytes.

## Inputs and isolation

The experiment has exactly six ordered inputs:

1. promoted Uniform-B v2 expert bank;
2. immutable GenerationLock;
3. dedicated label-free consumed-test cache alias;
4. dedicated capability-gated manifest alias;
5. byte-exact alias of the immutable original test-consumption ledger; and
6. the direct, single-consumer S4 ledger amendment.

It consumes no earlier Stage-90 result or amendment, no prior prediction
surface, and no prior scratch or checkpoint. The cache's Stage-70-derived
feature lineage is preserved as label-free cache provenance only; Stage-70
predictions, scoring, policy outputs, and downstream results are not inputs.

## Frozen method

The global probability surface contains 810 cells: nine targets by nine
train/generation seed pairs by ten physical actions. The ten actions are B, U,
and eight non-target A1 source actions. Exact-nine probabilities are averaged
before selection or scoring. U is an internal control and is never an S4
candidate.

For each target `H` and held fold `f`:

- support is the other four whole-case folds of `H`;
- evaluation is fold `f`;
- all eight A1 candidates and B are scored by exact pooled support BACC;
- the highest strictly positive A1 gain over B is selected;
- ties within `1e-12` use numeric source-center order; and
- no strictly positive gain, or support lacking either class, returns B.

For each candidate source `e`, `G_static` uses the equal-center mean exact gain
over donor query centers `q` outside `{H,e}`. It never reads same-target support
labels, never admits `e` as its own donor query, and falls back to B unless its
best gain is strictly positive. The implementation contains no case features,
donor model, target-local calibration, shared fit, threshold search, or
hyperparameter search. Terminal `O_static` and `O_case` are bounds, not
pre-evaluation methods.

## Label capabilities and null

B, U, and every A1 probability seal globally before any label opens. Label
capabilities are route-scoped, but evaluation has a global durable barrier:
all 45 S4 decisions and all 45 null plans must be recorded, persisted, and
bound back to their aggregate seals before the first evaluation-role label can
open. A case may act as support in a different fold rotation; that does not
grant it evaluation authority for the current route.

The fixed null retains B and permutes no labels. For each support case it orders
the eight complete A1 sufficient-statistic contribution blocks by SHA-256 over
the seed, fold, case, and action identity. Each of 10,000 null indices applies a
counter-SplitMix64 nonzero cyclic shift in `1..7`, preserving the candidate
multiset, class denominators, TP, and TN. The same strict S4 selection rule is
then recomputed. Only descriptive exceedance counts and fractions are reported.

## Evaluation and execution

The terminal endpoint is center-pooled exact BACC over whole-case OOF
predictions. Descriptive contrasts are `S4-B`, `S4-U`, `S4-G_static`,
`O_static-S4`, and `O_case-O_static`. Nine target centers are the outer units;
technical seed cells are not independent units. Intervals are descriptive
two-sided t8 intervals. There is no confirmatory p-value, lower-bound gate, or
PASS outcome.

The workstation uses two persistent A5000 source workers followed by a
phase-disjoint CUDA-free `4 x 3` CPU pool. Source/probability storage is
float32, scientific reductions are float64, the isolated scratch root is
`/data/local/fixed_bank_support_static_router_s4_v1`, and two fresh-process
replay validations are mandatory.

Automatic resume is not supported. An interrupted run restarts
deterministically from admission and non-repairingly validates any replayed
phase products. The terminal checkpoint is only an atomic publication boundary;
it is not a direct recovery surface.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4.v1
```

Regardless of metric values, this experiment cannot authorize routing success,
action or geometry selection, model/expert changes, policy changes, promotion,
deployment, recipe selection, another Stage-90 diagnostic, any numbered stage,
or another experiment.
