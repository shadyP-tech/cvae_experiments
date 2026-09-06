# HARP v21: normalized correction evidence

SCOPE LIMITED — v21 is an independent, planned terminal diagnostic. The
implementation does not issue an amendment, claim a lease, open scientific
labels, or establish real routing success. V17–v20 remain exhausted. Their
outputs, caches, models, thresholds, capabilities and authority are not inputs.

V21 addresses the identification failures exposed by v19 and v20. It retains
the directional action menu and donor ranker, replaces the independent action
outcome heads with normalized correction masses, and calibrates an unchanged
proposer on disjoint source cases. Baseline, calibrated-baseline and embedding
residual evidence are compared only inside nested source folds.

## Estimation

For each source training case, let N_ik be its class-k patch count and let
v_ck=n_c/S_ck be its center's inverse supporting-case fraction. The supervised
mass target for patch j is T_ijk=v_ck*1[Y_ij=k]/N_ik, zero for a missing class.
Center identity enters this training response and case weighting, never a
predictive feature. The model estimates its conditional mean using only
label-free own-case inputs. Inference receives no target class counts.

Each class mass factors into a nonnegative total mass and a softmax allocation
over patches anchored at B's class probabilities. Allocation training includes
the additional v_ck weight required by this factorization. Total mass is
estimated by a six-feature case-level ridge model and clipped below at zero;
it may exceed one and is not a probability. The patch posterior is a separate
logistic residual around logit(B). The residual variant uses all canonical
3840 features, standardized inside its exact fitting scope. Fixed ridge is
0.1; L-BFGS uses at most 128 iterations and eight history pairs. No width-based
feature dilution or shared learned transform crosses fitting scopes.

For an actual candidate a, delta_j=1[a_j>=0.5]-1[B_j>=0.5]. Its estimated gain
is exactly 0.5*sum_j delta_j*(m_j1-m_j0). B and hard-prediction no-ops have
zero estimated BACC gain. Expected Brier and log-loss differences use every
probability change and the separate patch posterior. These are model estimates
for the frozen source-normalization population, not exact expected utility on
an unknown finite target cohort. Actual held-out scoring recomputes the
registered scope-specific normalization after predictions are sealed.

## Selection and calibration

Candidate eligibility requires positive estimated gain and predicted Brier
and log-loss deltas at most 0.002 and 0.005. The highest-gain eligible action
wins, with deterministic ties. A binary case-harm gate is then applied to
that complete selection procedure. Patch marginals are not treated as a joint
label distribution, and no independent-patch simulation supplies confidence.

Every complete fit partitions its source scope deterministically into roughly
two-thirds proposer-fitting and one-third calibration cases, stratified by
center. The ranker and correction evidence fit only the former. All selected
calibration predictions are sealed before outcome scoring; the small gate
fits on these selected outcomes. The proposer is not refitted afterward.
Full-population equal-center case weights are calculated before eligibility
filtering and retained under a single global renormalization.

Five outer folds evaluate this algorithm. Four inner folds compare the three
evidence variants and nine abstention thresholds. The outer held cases cannot
shape any model, normalization, calibration, variant choice, or threshold.
Every variant's inner diagnostic frontier and joins are retained. Final
selection repeats source-inner selection and creates the same frozen
fit/calibration pair; it does not select a variant from outer results.

The menu remains B, exact U, D01_ONLY, D10_ONLY and BOTH, with K in {1,2,4}
and lambda in {0.25,0.5,0.75,1}: at most 38 configurations. Duplicate/no-op
rules are label-free; unselected branches copy exact B probabilities. A gate
veto returns exact B without trying a runner-up.

Admission retains the matched requirements: positive equal-center gain,
harm at most 0.25, Brier delta at most 0.002, log-loss delta at most 0.005,
at least 18 routed cases, and six centers with at least two routes each.
Zero source routing stops before bootstrap or target actions. Approximate
within-center case bootstrap bounds do not include full refitting uncertainty,
are not conformal guarantees, and do not establish unseen-center performance.

## Modules and workstation execution

The independent packages are:

- `routing/correction_mass_router_v21`: evidence design, targets, fitting,
  diagnostics and immutable models in `evidence/`; donor numerics/features/
  comparisons/proposals in separate modules; stable fit/calibration,
  cross-fitting, policy, admission and artifact verification modules.
- `runtime/harp_v21_execution`: physical work, immutable full-feature transport,
  stores, source fitting, independent selector/gate reconstruction and terminal
  evaluation. Paths above are relative to `src/midogpp_thesis/cvae/`.
- `diagnostics/fixed_bank_harp_router_v21`: preparation, independent source
  closure, amendment/lease lifecycle, phase ordering and durable reports.

The workstation profile remains two persistent GPU workers and four classifier
workers with three BLAS threads each. The source science parent uses one BLAS
thread. The same 81 classifier tasks and 810 physical fits supply all evidence
variants; nested selection adds no generation or physical classifier fits.
Full feature matrices use immutable float32 byte-backed arrays, avoiding Python
float-tuple expansion. Stores use fast compression and authenticated single
decompression. Exact action effects are batched and threshold sweeps reuse
predictions. Learned caches include exact source scope, capability, menu and
evidence variant; only variant-independent ranker work is shared across modes.

Reports persist held posterior/mass predictions, fit and scoring normalization
hashes, mass errors, calibration on the union of D01/D10 flip regions, signed
effect errors, candidate/winner joins, and every rejection frontier. Raw sample
labels are not serialized. These diagnostics cannot promote a failed policy.

## Inspection and verification

The mutation-free architecture check is:

```sh
PYTHONPATH=src python -m midogpp_thesis cvae-diagnostics \
  fixed-bank-harp-router-v21 \
  --config experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v21.yaml \
  --inspect-plan
```

`--dry-run` with the planned config is also path-free. A scientific run requires
new prepared v21 inputs, reviewed source closure, amendment, and single-use
lease. This page does not authorize preparation or launch.

`tools/benchmark_harp_v21.py` fabricates source cases and probabilities in
memory. Its default geometry is 216 cases, 48 patches and 3840 features, with
the full five-by-four selection and three evidence variants. It reads no data
labels and measures CPU construction only. It cannot establish MIDOG++
identification or routing performance. Validation results are recorded in
`docs/validation/harp_v21_implementation_2026-09-06.json`.

On 2026-09-06 the Xeon W-2265 workstation completed this full synthetic
benchmark in 619.74 seconds, with 2.46 GiB peak resident memory and one BLAS
thread. It selected embedding-residual evidence and admitted 216 synthetic
routes. The fabricated signal is deliberately learnable; this outcome is an
execution check, not an estimate of real MIDOG++ routing accuracy. Physical
generation and the 81 classifier tasks were outside this benchmark.

The integrated v21 suite passed 185 tests; shared CLI, workspace, authority
and predecessor checks passed another 120. After final protocol metadata
corrections, the 44 affected contract tests passed again. Workspace validation,
mutation-free inspection/dry-run, and the 241-member independent scientific
source closure passed. The benchmark router/runtime/script bytes match the
final implementation; config-only metadata changes are recorded separately.

To repeat only this synthetic benchmark from a checkout containing v21 on the
workstation:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' \
  /home/stud/spark/.venvs/cvae-breakhis/bin/python tools/benchmark_harp_v21.py \
  --output /tmp/harp-v21-synthetic-cpu-benchmark.json
```

The registered scope remains MIDOG++ Virchow2_3840, nine fixed centers,
216 source training cases/9,648 patches and 218 terminal test cases/9,928
patches. Source q and target H exclude their own expert, using C-minus-q and
C-minus-H. All source and target menus and bank proofs seal before source
label capabilities open; target evaluation labels remain closed until all
actions, predictions and independent reconstructions seal. The original bank
and generation locks remain canonical inputs. Any eventual target result is
POST_HOC_CONSUMED_TEST_SENSITIVITY, TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE,
and fresh_evidence=false.
