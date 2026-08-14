# MIDOG++ Uniform-B v2 consumed-test opportunity-gated dual-endpoint router

## Status

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_opportunity_gated_dual_endpoint_router.v1`
- Stage: `90_oracles_and_diagnostics`
- Evidence: repeatedly reused MIDOG++ test cases, terminal diagnostic only
- Publication label: `POST_HOC_CONSUMED_TEST_SENSITIVITY`
- Fresh evidence: `false`
- Method development and weights selected on this surface: `true`

## Question

The previous correctness-abstention router was dominated by donor-scale drift,
zero-opportunity source choices, and thousands of sparse route-local fits. The
robust nine-arm endpoint was more stable but did not identify held-case routing
opportunities. This successor tests whether a strict opportunity gate and
scale-calibrated identification endpoint can complement a separately
reconstructed robust probability endpoint.

## Fixed identification endpoint I

For each target center `H`, held whole case `c`, non-target source `e`, and
B-defined direction `d`, the ridge-binomial correctness model and its scaler
use only whole cases in `H` excluding `c`. Its output is named a
**support-calibrated expected-BACC proxy**. It is not NELBO, held-case utility,
or a generative compatibility estimate.

A source is eligible only when its held-case directional flip count is
positive, its fit is valid, and its case proxy is strictly positive. Across the
exact eight candidates, case proxies and legal donor priors are separately
scaled by their mean absolute magnitudes. The frozen score is

```text
I_score = 4/5 * case_proxy / mean_abs_case
        + 1/5 * donor_G    / mean_abs_G
```

No epsilon is used. A zero case scale admits no source; a zero donor scale makes
the donor term exactly zero; any nonfinite route fails closed to OFF. The
winner must be strictly positive. OFF is first in ties within `1e-12`, then
sources use numeric order.

## Fixed robust endpoint R and portfolio

R independently recomputes the nine directional arms

```text
K in {4,5,6} x w in {1/2,3/5,7/10}.
```

Each arm ranks all eight candidates by legal `G(H,e)` from query centers
outside `{H,e}`, selects from its top `K`, and scores `w*S + (1-w)*G` against
OFF. All nine arm identities remain in the probability average even when two
arms select the same action.

I and R consume the same row-aligned B/U/A1 probability contract but make
independent endpoint decisions. Their only portfolio is

```text
P(OGDE_PORTFOLIO) = 3/5 * P(I_OPPORTUNITY_GATED)
                  + 2/5 * P(R_NINE_ARM_ROBUST)
```

The probability is thresholded once at `0.5`, with equality positive. This is
a prediction-level score ensemble, not a CVAE mixture or generative
composition.

## Attribution and identification diagnostics

The artifact must report B, U, I, R, and the portfolio alongside:

- `3/5 B + 2/5 R` calibration-only attribution;
- candidate-feature-block permutations for I and the portfolio;
- gate-only and source-only controls;
- matched donor-G routing;
- full-pipeline delete-one-center reconstruction;
- OFF precision, recall, and balanced accuracy;
- overall exact-action and active-source top-1 agreement;
- macro Spearman and normalized oracle gap;
- Brier score, log loss, calibration slope/intercept, and threshold crossings.

All controls, intervals, and delete-center results are descriptive. The
implementation freezes `incremental_vs_R_is_inconclusive=true`,
`source_identification_is_established=false`, and both nominal coverage and
significance claims to false.

## Label and lineage firewall

- exactly six ordered, successor-fenced original inputs;
- one direct amendment of the immutable original consumed-test ledger;
- no prior Stage-90 output, amendment, prediction, scratch, or checkpoint;
- common physical probabilities and label-free features seal before labels;
- donor `G` excludes `H` and `e`;
- every route fit, scaler, response, denominator, gate, and decision excludes
  its held whole case `c`;
- all 218 I, R, portfolio, and attribution decisions seal before terminal
  labels;
- raw labels, image paths, sample paths, and per-case BACC are not persisted.

## Workstation execution

The generation phase uses one persistent spawned source worker on each RTX
A5000. CUDA is then hidden before a disjoint CPU routing phase with four
spawned workers and three BLAS threads per worker. Float32 stores generated
streams and probability surfaces, int64 stores confusion counts, and float64
performs scientific reductions. Scratch is dedicated to
`/data/local/fixed_bank_loo_opportunity_gated_dual_endpoint_router_v1`.

No recovery strategy exists. Owned-task replay, foreign checkpoint reuse,
cross-run recovery, and terminal recovery are forbidden. A completed artifact
must pass two independent CUDA-free fresh-process reconstruction validations.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_opportunity_gated_dual_endpoint_router.v1
```

## Interpretation boundary

The test split and the portfolio weights were repeatedly inspected during
method development. A positive result may support only a post-hoc diagnostic
statement about abstention-calibrated directional probability composition on
this same MIDOG++ surface. It cannot establish fresh routing quality, reliable
active-source identification, superiority to R, downstream utility, policy
selection, promotion, or deployment. Any confirmatory claim requires a newly
reserved whole-patient/slide evaluation after freezing the complete method.
