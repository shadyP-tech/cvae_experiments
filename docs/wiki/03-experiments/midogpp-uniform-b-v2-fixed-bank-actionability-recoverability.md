# MIDOG++ Uniform-B v2 fixed-bank actionability/recoverability

## Status and claim boundary

The experiment is implemented and independently registered as
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability.v1`.
It may reuse the complete 218-case MIDOG++ test split only as a terminal
post-hoc mechanism diagnostic. The output is always
`EXPLORATORY_CONSUMED_DATA_ONLY` / `DO_NOT_PROMOTE`; it cannot authorize a
router, action, geometry, recipe, model or expert update, promotion,
deployment, numbered-stage feed, or another experiment.

The six-input fence contains only the promoted fixed bank, GenerationLock,
dedicated single-consumer test-cache and manifest aliases, an exact alias of
the original consumption ledger, and this diagnostic's direct-parent
amendment. No previous Stage-90 result, prediction surface, amendment, scratch,
or checkpoint is consumed.

## What it isolates

The previous signed gate selected zero residual scale in all 45 folds. This
experiment moves one level upstream:

- **Actionability:** do any fixed source actions improve over a budget-matched
  uniform action, and is there additional case-conditional oracle headroom?
- **Recoverability:** can a strict donor-trained label-blind model or a
  same-target evaluation-disjoint support fold recover that action ordering?

It does not test a deployable policy. Oracle rows are constructed only after
terminal labels open.

## Frozen action library

For target center `H`, the target expert is absent and the candidates are the
other eight fixed experts.

| Action | Rows per source/class | Fit weights | Role |
|---|---|---|---|
| `B` | 128 for each source | 1 | immutable baseline |
| `U` | 144 for each source | 1 | shared 1152-row control |
| `A0(e)` | 256 for `e`, 128 for each other source | 1 | stronger physical allocation |
| `A1(e)` | exactly the `A0(e)` rows | `23/16` for `e`, `7/8` otherwise | budget-neutral reweighting |

Both A0 and A1 have effective mass 1152 per class. A1 weights affect only the
logistic-regression fit; the scaler remains unweighted. There is no action
strength sweep, source pair, class-specific variant, seed selection, or
cross-geometry selector.

## Methods and exclusions

The nine training/generation seed pairs are averaged before case features are
built. For each geometry:

- `G` learns a case-independent candidate prior;
- `R` learns from label-blind case/action probability, disagreement, margin,
  entropy, and seed-stability summaries;
- `P` cyclically deranges complete candidate feature blocks and refits the same
  ridge capacity;
- `S_y` selects one static action from `U` plus eight source actions using only
  the other four whole-case folds of the same target;
- `O_static` and `O_case` are terminal static and case-conditional oracle
  bounds.

`G/R/P` predict class-balanced proper-log-loss gain relative to `U`. Each final
fit excludes outer target `H` and candidate source `e`; each nested diagnostic
also excludes held query `q`. A nonpositive prediction falls back to `U`.
Target support cannot tune features, ridge alpha, action strength, or geometry.

## Endpoints

The primary actionability contrasts are `O_static-U` and
`O_case-O_static`. Recoverability contrasts are `R-U`, `R-G`, `R-P`, and
`S_y-U`. The bundle also records `U-B`, hard-decision complementarity,
support/evaluation action-rank stability, normalized U-to-static-oracle gaps,
center-pooled exact BACC with an equal-center aggregate, and paired whole-case
cluster uncertainty. Single-class cases are retained through additive
confusion sufficient statistics; per-case BACC and raw labels are never
persisted.

## Workstation execution

The W-2265 profile uses two persistent one-worker A5000 pools for frozen source
generation. The parent then removes CUDA visibility and runs four spawned CPU
workers with three threads each. Source and probability arrays are float32;
scientific reductions are float64. Prediction tasks are grouped by
target/training/generation seed so all 18 physical actions reuse one source
load. Hash-valid source and prediction tasks may resume, and local scratch is
removed only after closed-world validation passes.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability.v1
```

The independent validator reloads source and prediction arrays, reconstructs
the exact-nine probability and prelabel feature surfaces, rebuilds LOCO utility
targets from scoped labels, refits every `G/R/P` model, recomputes all 495
decisions, and recomputes terminal sufficient statistics and uncertainty before
accepting the closed-world bundle.
