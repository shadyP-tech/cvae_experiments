# HARP v20: patch evidence and risk-aligned action selection

V20 implements a new source-trained router after v17–v19 failed source
admission. It changes the information available to action selection and the
objective used to choose an action. It does not relax the admission gate or
claim that real routing will succeed. Status: **implemented, planned,
execution not authorized**.

## Why the architecture changes

The v19 source audit found actual safe alternatives but poor discrimination of
harmful winners. Recalibrating the same signals or lowering an abstention
threshold cannot manufacture missing class evidence. A harm veto applied only
after a gain-oriented winner has been chosen also cannot recover another
candidate that should have won.

V20 retains exact directional recipes and the pairwise donor ranker. It adds:

1. A fixed, label-free sketch of the actual Virchow2 patches. A small,
   regularized class-evidence model is fitted independently inside every
   training scope. Actual changed patches now contribute class evidence to
   each candidate descriptor.
2. A signed-gain objective penalized by predicted harm, Brier excess and log
   loss excess **before** winner selection. Proper-loss forecasts can change
   which candidate wins.
3. Full nested selection of a small, fixed risk-penalty grid and the winner
   gate threshold. Inner selection must meet the same minimum case/center
   coverage as final admission, rejecting sparse accidental policies.
4. Persisted source OOF direct-classifier controls, complete candidate
   frontiers and selected-winner diagnostics. The auxiliary control is
   diagnostic only; it is not another routing arm or a policy-selection score.

## Fixed protocol and claim boundary

The artifact and claim dataset are MIDOG++, with canonical Virchow2_3840
features and scanner/center as domain axis. Source development uses 9,648
training patches in 216 cases over centers `{0,1,2,3,5,6,7,8,9}`. Eventual target
evaluation uses all 9,928 consumed test patches in 218 cases from those known
centers. This is a **known-center train-to-test terminal diagnostic**, not an
unseen-center experiment. Source case folds, rather than center holdouts,
are the validation units.

For a source context `q`, the physical expert bank is `C minus q`; for a target
context `H`, it is `C minus H`. The context's own expert is excluded. B and U
use the same context-excluded source pool. The baseline classification
threshold is 0.5. The 38 action configurations are:

- exact B and exact U_FULL;
- D01_ONLY, D10_ONLY and BOTH for K in `{1,2,4}` and lambda in
  `{0.25,0.5,0.75,1}`.

Ineligible and byte-identical duplicate actions are diagnosed, not fitted as
separate evidence. D01_ONLY preserves the B probabilities on the D10 branch;
D10_ONLY preserves the D01 branch. B remains byte-identical on fallback.

All physical source/target menus, patch features and bank attestations seal
before source labels open. The center-sharded source capability permits only
scoped v20 router training, including the auxiliary patch-evidence model.
Experts, physical classifier fits, generation, and the canonical feature
frame stay frozen. Target labels cannot fit, select or calibrate any model.
Source admission precedes target action construction; global target prediction
seals and fresh reconstruction checks precede terminal evaluation labels.

The result remains `POST_HOC_CONSUMED_TEST_SENSITIVITY`,
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`, `fresh_evidence=false`. It cannot feed
another experiment, Stage 60/70, deployment or a confirmatory improvement
claim. Models and approximate OOF bounds do not guarantee final-refit safety.

## Patch evidence and actual-action formulas

The fixed sketch partitions a seeded permutation of 3,840 coordinates into
64 groups. For each group it sums fixed signed coordinates and divides by the
square root of the group size. Seed is 20020. It is row-local: adding or
changing target rows cannot change a source sketch. No PCA or projection is
fitted on source or target observations.

Within a fitting scope, standardize the sketch using equal-center,
equal-case, equal-patch-within-case weights. Fit a logistic model with weighted
mean log loss plus `0.01/2 * ||beta_without_intercept||²`. This source-trained
probability estimate `p_i` is predictive evidence, not calibrated truth or an
action safety certificate.

For baseline probabilities `b_i` and actual candidate probabilities `a_i`, the
new descriptor includes mean and lower-decile `p_i` on actual 0-to-1 flips,
mean and lower-decile `1-p_i` on actual 1-to-0 flips, baseline disagreement,
and these conditional expected proper-loss changes:

```
E[delta Brier_i | p_i] = (a_i-b_i) * (a_i+b_i-2*p_i)
E[delta logloss_i | p_i] = -p_i*log(a_i/b_i)
                         -(1-p_i)*log((1-a_i)/(1-b_i))
```

Log-loss probabilities are clipped to `[1e-6,1-1e-6]`. Proper-loss terms include
**every** probability change, including changes that do not cross 0.5.
Only hard-changing nonbaseline candidates participate in routing-head fitting;
probability-only actions remain visible in diagnostics.

The action model predicts signed aligned BACC contribution, Brier change,
log-loss change and coherent safe/harm/other probabilities. Conditional
positive/negative magnitude heads remain auxiliary diagnostics. The signed
gain head, rather than safe-event probability times magnitude, drives selection.

Let `g_hat`, `h_hat`, `b_hat`, and `l_hat` be the predicted action gain, harm
probability, Brier delta and log-loss delta. The objective is

```
J(a) = g_hat(a) - rho * [0.05*(h_hat(a)-0.25)
                       + 1.00*(b_hat(a)-0.002)
                       + 0.25*(l_hat(a)-0.005)]
```

`rho` is selected from `{0.5,1,2}` in inner source CV. These coefficients are
fixed design choices, not calibrated statistical bounds. Negative risk excess
is credit in this Lagrangian-style objective; individual predicted gain or
loss constraints are not separate hard vetoes. Actual policy-level gain and
risk checks remain necessary. Exact B retains score zero and is excluded from
nonbaseline winner fitting.

Select the unique nonbaseline hard-changing action with maximal `J`, breaking
ties by arm ID. The complete winner procedure has its own held-out harm gate.
An enabled policy routes only if `J > 0` and `1-P(winner harm) >= tau`.
Otherwise it returns exact B. Gate vetoes never select a second candidate.

## Nesting and admission

`FitProposer(S)` uses four center-stratified case folds. For every held block,
fit the pairwise ranker and patch model on `S minus held`; seal the resulting
candidate composites and patch evidence before scoring their source outcomes.
Fit the action heads from this held-out evidence, then refit the ranker and
patch model on S for predictions outside S.

`FitComplete(S)` uses three winner folds. Each fold independently builds
`FitProposer(S minus held)`, predicts and seals its unthresholded winners,
then joins the excluded outcomes to train the winner gate. Harmful and
negative-score winners remain in gate training. Empty menus stay in the
normalization inventory.

Five outer folds evaluate the complete selection procedure. Inside each
outer training scope, four inner folds separately evaluate every predeclared
rho and tau combination. Choose an enabled setting with largest actual
inner OOF gain; ties favor stronger rho and then larger tau. If no setting
passes, disable that fold's policy. The final full-source refit uses settings
selected by a separate full-source inner procedure. Outer outcomes do not
select its settings.

For center c with `n_c` cases and `n_cy` cases supporting class y, each case's
aligned contribution is

```
g_ck = n_c/2 * sum_y (I(case k supports y) * recall_delta_cky / n_cy)
```

Its case mean is that center's BACC gain; averaging centers equally gives the
registered estimand. Denominators come from the complete owning scoring scope,
including fallback and probability-only cases. This avoids treating single-class
cases as an arbitrary half-weight BACC observation.

Inner selection requires positive signed policy gain, routed harm <=0.25,
routed Brier delta <=0.002, routed log-loss delta <=0.005, at least 18 routed
cases, and at least six centers with two routed cases each. Final outer OOF
admission applies those coverage requirements and 1,024 approximate
studentized max-stat bootstrap replicates (seed 20020). It resamples cases
within fixed centers and recomputes class-support denominators; it does not
claim uncertainty for unseen centers or refit the learning algorithm inside
bootstrap replicates. Zero coverage fails before bootstrap and target actions.

## Implementation and workstation execution

Science, physical execution, and lifecycle/authority are separate packages:

- `src/midogpp_thesis/cvae/routing/risk_aligned_router_v20/`
- `src/midogpp_thesis/cvae/runtime/harp_v20_execution/`
- `src/midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v20/`

The science README maps individual modules to responsibilities. V20 owns its
new cache, source capability, terminal release, output, scratch, source seal,
amendment and single-use lease. It imports none of the exhausted v17–v19
implementations. It uses the immutable expert-bank lock `9972a41dcd4814cd` and
generation lock `34e551425710362e`; the other input bytes and repository source
closure must be bound by the separate preparation/activation workflow. Planned
input hashes remain null until that workflow. V20 does not consume previous
HARP learned models, caches, source outcomes or selected thresholds.

Physical execution retains 27 generation stream jobs, two persistent GPU
workers, and 81 classifier tasks/810 fits with four classifier processes and
three BLAS threads each. The patch model adds small CPU fits, **zero physical
classifier tasks and zero GPU fits**. It uses authenticated existing canonical
frame rows and stores float32 sketches in the sealed compact menu NPZ.

Truth-bearing nested fits stay in the parent with one BLAS thread. The
execution-local fit cache is bounded at 256 entries and keys exact training
menus, source capability and fitting configuration. Risk scales reuse
ranker/proposer fits because those fits do not depend on the decision penalty;
complete winner-gate fits include the scale in their cache key. Raw composite
features use an 8,192-entry bounded cache; learned patch probabilities never
enter that cache. Thresholds replay already sealed predictions.

Source reports persist before admission, including every fold/action/threshold
and penalty setting, route counts, center counts, utility/risk moments and
rejection reasons. Aggregate inner policy rows carry the actual 18/6/2
selection constraints. Raw candidate rows and per-fold slices are diagnostics
and do not claim to have passed whole-policy admission. Source OOF direct
patch-classifier scores are reported separately from routed policy outcomes;
no target direct-classifier comparison is implemented in this version.

## Validation and commands

From the repository root, inspect without resolving scientific input paths:

```bash
PYTHONPATH=src python -m midogpp_thesis workspace validate
PYTHONPATH=src python -m midogpp_thesis cvae-diagnostics fixed-bank-harp-router-v20 \
  --config experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v20.yaml \
  --inspect-plan
```

A scientific run requires the v20 preparation authority, sealed inputs and
repository closure, separate amendment, and new single-use lease. The planned
CLI fails closed without them. Do not reactivate old identities or use force.

Run the synthetic performance check on delli2 without reading scientific data:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/stud/spark/.venvs/cvae-breakhis/bin/python benchmarks/harp_v20_synthetic.py \
  --output-dir "$(mktemp -d /tmp/harp-v20-synthetic.XXXXXX)"
```

This fixture contains an intentionally easy, perfectly repeated relationship
between fabricated patch features and fabricated labels. Successful routing
there proves that the construction can activate actions; it says nothing about
MIDOG++ action learnability. Validation on 2026-09-05: 159 local v20 tests passed; 36 shared-dispatch and
workspace checks passed (overlapping some v20 tests). On delli2, 91 numerical
and runtime tests passed, followed by 32 overlapping final-guard checks.
Workspace validation, planned inspection/dry-run and diff checks passed.
Inspection resolved no scientific paths, changed no files and issued no authority.

The complete synthetic source-policy benchmark took **1,358.29 seconds
(22.64 minutes)** with **3.60 GiB peak RSS**, one BLAS thread, no CUDA and no
scientific data. All 216 synthetic OOF cases routed and admission passed.
It performed 600 actual ranker fits, 120 candidate-model fits and 78 complete
winner-gate fits, with 192 fit-cache hits and 929,164 raw-feature-cache hits.
There were 33,696 frontier rows. These timings exclude physical generation
and classifier fitting.

The full-size benchmark used an implementation snapshot before the final
menu-binding, empty-target fallback, prelabel-seal and activation-metadata
guards. Those changes preserve the numerical learner and were subsequently
tested locally and on delli2. Snapshot hashes, changed modules, complete
benchmark output and test counts are recorded in
`docs/validation/harp_v20_implementation_2026-09-05.json`.
