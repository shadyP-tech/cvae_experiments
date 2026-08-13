# MIDOG++ Uniform-B v2 consumed-test fixed-bank case-directional correctness abstention router

## Status

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_case_directional_correctness_abstention_router.v1`
- Stage: `90_oracles_and_diagnostics`
- Evidence: reused MIDOG++ test labels, terminal diagnostic only
- Publication decision: `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`
- Fresh evidence: `false`

## Question

DCSE improved BACC, but most routes still behaved like a static selector and
rarely identified when the held case should use a source-specific directional
flip or abstain to B. This experiment asks whether a frozen, low-capacity
correctness model can use only label-free held-case flip geometry plus legal
same-`H`, held-`c`-excluded support labels to improve that decision diagnostically.

## Frozen method

Each route is `(H,c,e,d)`, where `e` is one of eight non-target sources and
`d` is `zero_to_one` or `one_to_zero`. The held-case feature vector contains:

1. directional flip rate;
2. baseline absolute margin on directional flips;
3. candidate absolute margin on directional flips;
4. directional probability shift on flips;
5. seed directional flip robustness;
6. candidate seed disagreement on directional flips.

These features are label-free. The route model is a NumPy ridge-binomial
logistic IRLS fit on whole cases in `H` excluding `c` only: alpha `1`, unpenalized
intercept, zero initialization, at most 50 iterations, tolerance `1e-12`, eta
clip `[-30,30]`, and probability clip `[1e-12,1-1e-12]`. No-trial or
nonconvergent fits fail closed to a zero case proxy.

The final source score is exactly one half the held-case correctness proxy plus
one half the donor prior `G_d(H,e)`. The donor prior uses only query centers
`q` outside `{H,e}`. OFF has score zero and is first in ties within `1e-12`;
source ties use numeric order. The selected sources compose only the applicable
B-defined direction and the final probability is thresholded once at `0.5`.

Canonical methods are B, U, `CDCA_LOO`, `G_directional_matched`, and
`CDCA_case_proxy_only`. `O_directional_static` and `O_case_directional` are
terminal oracles. `CDCA_feature_block_permutation_descriptive` permutes whole
six-feature candidate blocks within each `(H,c,d)`, refits the identical
pipeline, and is never a gate.

## Label and lineage firewall

- exactly six successor-fenced original inputs;
- no previous Stage-90 output, fitted model, amendment, prediction, scratch, or checkpoint;
- B, U, and all eight A1 probability cells seal before labels;
- all 72 `q not in {H,e}` donor grants complete before target-local support opens;
- route `c` is absent from its scaler, fit, denominators, and decision;
- all 218 route decisions and predictions plus the aggregate seal precede terminal labels;
- raw labels, paths, and per-case BACC are not persisted.

## Workstation execution

Generation uses two persistent spawned RTX A5000 workers. CUDA is then hidden
before a separate CPU phase of four spawned workers with three BLAS threads
each. Float32 stores source/probability surfaces, int64 stores confusion counts,
and float64 performs scientific reductions. Scratch is dedicated to this run
and is not a recovery input. Completion requires a 43-member closed-world
bundle and two independent CUDA-free process replays.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_case_directional_correctness_abstention_router.v1
```

## Interpretation boundary

The MIDOG++ test split has already been consumed repeatedly. Positive terminal
contrasts can describe whether this frozen diagnostic routes better on the
same data, but cannot establish fresh routing quality, predict held-case exact
BACC, update the bank or policy, or authorize Stage 50/60/70, promotion, or
deployment. A fresh claim would require an independently reserved whole-
patient/slide evaluation after freezing this complete method.
