# Stage 90: Oracles and Diagnostics

This stage contains post-hoc oracle upper bounds, fidelity analyses, rejected
lineage audits, and other non-deployable diagnostics.

Every oracle row must be marked non-deployable. Diagnostic results may explain
a failure or quantify headroom, but they must not tune or select a deployable
policy and must remain separate from held-out utility claims.

The rejected subtree also preserves evidence from the old limited-domain
MIDOG++ scanner support-routing surface. That surface did not cover the current
full dataset contract, so its top-1, rank, and oracle-gap values are not
thesis-facing and must be regenerated before any routing claim.

## Uniform-B v3 Retrospective Replay v1

`uniform_b_v3_retrospective_replay_v1` replays one fixed representation,
`annotation_jpeg_fixed_center_b_v3`, across all nine eligible outer centers.
It imports the already-frozen per-fold classifier locks from the completed v3
pilot, refits canonical A and B without the held-out center, and checks exact A
replay plus row-level agreement with the source v3 tables. The loader opens
only the A and B shards; physical-multiscale C is not an experiment input.

The independently validated result is:

| Quantity | Canonical A | Uniform B | B minus A |
| --- | ---: | ---: | ---: |
| equal-center mean BACC | `0.740312` | `0.792087` | `+0.051775` |

B wins on eight of nine centers and has a worst-center delta of `-0.002890`.
The conditional paired case bootstrap has 2,000 valid replicates and a 95%
percentile interval of `[+0.038962, +0.063599]`. It conditions on the observed
centers, fixed fits, and imported classifier locks; it does not cover the
uncertainty induced by choosing B after observing these outer-center results or
the uncertainty of transferring to a new center.

This is a `POSTHOC_DISCOVERY`, not independent confirmation. It validates
deterministic reproducibility and gives a consolidated fixed-B estimate for the
same nine centers, but it is non-adoptive and cannot change the Stage-10
reference, feed Stage 20 through 70, or support deployment, routing, CVAE,
generation, calibration, or new-center claims.

Run the registered replay with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v3_retrospective_replay.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v3_retrospective_replay_v1/seed42/
```

## Uniform-B v3 Prospective Test Confirmation v1

Phase B locks B, the nine source-only classifier locks, the primary endpoint,
and a four-part gate before extracting or scoring B on the case-disjoint test
split. The 9,928 eligible test rows have zero sample and case overlap with the
9,648 train rows used for the retrospective discovery; validation rows are not
used. The B-only test cache is independently validated and fully hash-promoted.

The predeclared gate requires mean BACC delta at least `+0.02`, at least six
strict center wins, worst-center delta at least `-0.01`, and a positive 95%
conditional-bootstrap lower bound. The independently reconstructed result
passes all four checks:

| Quantity | Canonical A | Uniform B | B minus A |
| --- | ---: | ---: | ---: |
| equal-center test mean BACC | `0.735733` | `0.799159` | `+0.063426` |

B wins on all nine centers, its worst delta is `+0.010638`, and the 2,000-
replicate conditional paired case interval is `[+0.050709, +0.073650]`. The
decision is `CONFIRMED_WITHIN_CENTER`.

This is prospective confirmation for new cases from the same observed centers,
not external-dataset or new-center confirmation. It remains Stage-90
`DIAGNOSTIC ONLY`, cannot automatically replace the canonical Stage-10
reference, and cannot feed any Stage-20-through-70 choice.

## Canonical Uniform-B Nyström Nonlinear Probe v1

The bounded nonlinear probe keeps canonical B fixed and changes only its
decision function to `StandardScaler -> Nyström RBF -> L2 logistic`. Exact
nested source-inner LODO selects among 36 predeclared candidates while
inheriting each outer fold's canonical-B class weight.

The independently reconstructed diagnostic passes its full progression gate:

| Quantity | Linear B | Nonlinear B | Delta |
| --- | ---: | ---: | ---: |
| equal-center train-surface BACC | `0.792087` | `0.815278` | `+0.023192` |

It wins on all nine centers and has a worst-center delta of `+0.002008`. Its
supportive case-within-center bootstrap interval is
`[+0.011774, +0.032390]`. It rescues 751 linear errors while introducing 550,
and resolves 613 of 1,025 baseline errors whose source-only centroid geometry
already favored the true class.

This supports a nonlinear-boundary limitation in B; it does not prove B is
sufficient. The result is diagnostic-only, uses an already inspected train
surface, and does not authorize automatic canonical migration. The validation
split remains unfeaturized and unscored; its 44 cases provide only 3–7 cases
per center, below the existing confirmation minimum.

Run the registered probe with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_nystroem_nonlinear_probe.v1
```

## Uniform-B Robust Interaction Probe v1

The paired audit finds 550 standard-Nyström regressions, of which 246 are
near the decision boundary and only two are confidently wrong. Center 2
accounts for 257 false-positive regressions; center 9 contributes 24
false-negative regressions.

A bounded nested comparison then tests equal center×class weighting, two
group-DRO rates, and bilinear ranks `{4,8,16}`. Robust Nyström retains a
`+0.020437` equal-center BACC gain over linear B and fixes much of center 2's
specificity loss, but center-9 recall falls by `−0.127168`. The bilinear
family is `−0.006267` below linear B on average.

Decision: `NO_FAMILY_PASSES_ROBUST_BPLUS_GATE`. No classifier is frozen and
validation/test remain untouched. The result indicates that generic group
reweighting moves the worst-group failure rather than resolving it.

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_robust_interaction_probe_v1/seed42/
```

## Uniform-B Sensitivity/Specificity-Constrained Nyström Probe v1

This fixed-threshold successor tests four inherited Nyström objectives along a
native-logit capacity path while enforcing per-inner-center recall,
specificity, and BACC deltas versus exact linear-B refits.

Decision: `NO_CONSTRAINED_BPLUS_CANDIDATE_PASSES`. Only centers 6 and 9 admit
nonlinear capacity at `alpha=0.25`; the other seven centers fail closed to
exact linear B. Equal-center mean BACC improves only `+0.00282`, with a
`−0.03468` worst recall delta at center 9.

The final CPU-only run took `275.92` seconds using four workers × three
threads. Independent reconstruction validation passes. Validation and test
remain untouched.

Canonical output:
`artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_sens_spec_constrained_nystroem_probe_v1/seed42/`.

## Uniform-B v2 Consumed-Validation Dense Residual Router v1

`midogpp.oracle.uniform_b_v2_consumed_validation_dense_residual_router.v1`
is a Stage-90 mechanism prototype over validation bytes already consumed by
the frozen Stage-60 source-inner policy family. It reads the original frozen
expert bank and GenerationLock, but accesses the validation cache and manifest
only through these experiment-fenced aliases:

- `midogpp_stage90_dense_residual_router_validation_cache_v1`
- `midogpp_stage90_dense_residual_router_validation_manifest_v1`

The aliases retain the exact physical paths and required SHA-256 hashes of the
current Stage-60 inputs. They declare
`CONSUMED_FOR_STAGE90_DIAGNOSTIC_ROUTER_PROTOTYPING`, create no fresh evidence,
and authorize only this experiment for `oracle_and_diagnostic_evidence`.

For each outer target `H`, two cases per remaining query center are assigned to
label-free compatibility support with seed `20260806`. The fixed compatibility
proxy is the class-marginalized PS variational energy with prior `[0.5, 0.5]`
in the common 3,840-dimensional frame, own-source median/MAD calibration, and
an arithmetic mean across all three expert replicas. It is explicitly not an
exact NELBO claim. The router evaluates `rho` in `{0, 0.25, 0.5}` with
temperature `1`, maximum source weight `0.25`, minimum effective source count
`6`, and a minimum integer allocation of one sample per legal source. Nested
seven-source development uses exactly 1,008 samples per class (144 per source
at `rho=0`); eight-source target scoring uses the canonical 1,024 per class
(128 per source at `rho=0`) for every `(training_seed, generation_seed)` cell
in `{17,42,101} x {17,42,101}`.

Before any validation label is opened, the runner materializes and durably
seals all 324 target prediction cells (three candidate actions plus the
separate exact-control alias for every target and seed cell). Development
label access requires that global seal and rehashes both target prediction
files, closing the circular case where a center is target `H` in one fold and
pseudo-target `q` in another. The control alias reuses the already fitted
`rho=0` model, so the frozen runtime budget is 1,944 development fits plus 243
unique target fits (2,187 total), with at most nine generated source blocks
resident at once.

Selection minimizes mean regret plus `0.5` times upper-quartile CVaR regret
plus `0.01` times mean squared L2 distance from uniform, equally weighting all
`q != H` queries and all nine paired seed cells. A nonuniform action must also
have a strictly positive mean paired BACC delta versus `rho=0`; otherwise the
diagnostic falls back to `rho=0`. Ties prefer smaller `rho`, then `action_id`.
The classifier is fixed synthetic-only L2 logistic regression with `C=0.01`,
`lbfgs`, `max_iter=3000`, and `random_state=23`.

Run the registered diagnostic with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_dense_residual_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_dense_residual_router/v1/
```

The publication label is `EXPLORATORY_CONSUMED_DATA_ONLY`. The output is
`diagnostic_only`; it claims neither routing quality nor fresh confirmation and
cannot reopen or feed Stage 60, Stage 70, recipe selection, deployable
selection, promotion, or deployment.

## Uniform-B v2 Local Marginal-Utility Router v1

`midogpp.oracle.uniform_b_v2_consumed_validation_local_marginal_utility_router.v1`
is the next Stage-90 diagnostic after the dense residual router. It asks the
utility-aligned question that compatibility rank alone could not answer: near
equal union, does adding mass to source `e` improve downstream BACC on query
center `q`?

For every outer target `H`, query center `q != H`, and legal source
`e != H,q`, the experiment pairs an exact equal-union control with the fixed
one-sided perturbation

```text
w(+e) = (1 - epsilon) u + epsilon one_hot(e),  epsilon = 1/8
Y(H,q,e,s,g) = [BACC(w(+e)) - BACC(u)] / epsilon
```

The seven-source development control allocates 144 samples per source and
class. A perturbation allocates 252 to the boosted source and 126 to each of
the other six sources, preserving exactly 1,008 generated samples per class,
a maximum source weight of `0.25`, and an effective source count of `6.4`.
All three training seeds and all three generation seeds are retained and
reported; no seed is selected.

The global prediction phase materializes 5,184 classifier cells and seals all
prediction bytes and evaluation-row identities before any validation label is
opened. Only then are 4,536 paired marginal-utility rows scored. Support cases
are label-free and case-disjoint from evaluation. For a given outer fold,
`H` is excluded both from development queries and from every development
expert pool.

Learnability is evaluated with nested leave-one-domain-out ridge fits. In every
outer and inner fold, the held-out domain is excluded from both query-center
and source-center roles. The feature interface is label-free: calibrated
compatibility energy, within-query centered/scaled energy, rank and gap
geometry, plus source identity indicators. Alpha is selected by equal-query
cluster MSE wholly inside the strict inner folds. Learnability is reported in
the thesis-relevant order—top-1 utility agreement, utility Spearman,
normalized oracle gap, fold stability, and secondary RMSE. A robust constrained
optimizer may emit target-center weight plans from label-free target features,
but those plans are deliberately unscored: target-`H` labels are never opened
to evaluate or choose the plan for `H`. The plan also crosses from seven-source
development responses (1,008/class) to an eight-source target geometry
(1,024/class), so it is explicitly recorded as extrapolative and cannot be
treated as deployable or as evidence that equal union was beaten.

Run the registered diagnostic with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_local_marginal_utility_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_local_marginal_utility_router/v1/
```

This uses newly experiment-fenced aliases of the same already-consumed,
hash-pinned validation manifest and label-free cache bytes. Its publication
label is `EXPLORATORY_CONSUMED_DATA_ONLY`: the surface can diagnose whether
local utility is learnable, but it is not fresh routing evidence, does not
establish target BACC, and cannot feed Stage 60, Stage 70, recipe selection,
deployable selection, promotion, or deployment.

## Uniform-B v2 Conditional-Contrast MMD Router v1

`midogpp.oracle.uniform_b_v2_consumed_validation_conditional_contrast_mmd_router.v1`
is a separately fenced Stage-90 diagnostic that addresses the pooled-MMD
router's main identifiability failure: class-0 and class-1 kernel means can
cancel after marginal pooling even when an expert has the wrong class
geometry. For each target, one target-excluded, equal-source/class-balanced
synthetic pool fits the scaler, source-only soft-class reference, and shared
Nyström map. The target support cases remain unlabeled and case-disjoint from
evaluation.

The fixed proxy is

```text
sum_c alpha_c ||sum_e w_e mu[e,c] - mu[H,c]||^2
  + kappa ||sum_e w_e (mu[e,1]-mu[e,0]) - (mu[H,1]-mu[H,0])||^2
  + lambda ||w-u||^2
```

with `alpha=(0.5,0.5)`, `kappa=1`, and `lambda=0.1`. It is solved through an
exact lifted feature representation under the existing source cap and
effective-source constraint. The runner abstains to exact equal union when a
support case has insufficient soft-class mass/effective rows, any class or
contrast component worsens, leave-one-support-case/seed/prior checks fail, the
raw solution exceeds `L1(w,u)=0.25`, the action duplicates the recomputed raw
pooled-MMD numerical direction or energy direction, or integer allocation
equals the control. The pooled reference uses the original `lambda=0.05`
base solve before its policy-level stability gates; no prior Stage-90 output is
consumed.

The workstation profile is frozen for the Xeon W-2265 and two RTX A5000s:
one persistent worker per GPU for generation and target routing, followed by
four classifier workers with three BLAS threads each. The 81 source/seed
prefix blocks are generated once into a hash-bound float32 memmap; source,
route, and prediction-cell checkpoints support phase resume. All 162 route and
control prediction cells are globally sealed before evaluation labels open.
For backward compatibility with the shared scorer, routed table rows retain
the generic `mmd_kmm` arm role; the authoritative method identity is
`protocol_manifest.proxy.family=class_conditional_contrast_mmd_kmm`.

Run the registered diagnostic with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_conditional_contrast_mmd_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_conditional_contrast_mmd_router/v1/
```

The proxy is compatibility-only, not downstream utility. Because the target
validation labels are already consumed, any score is exploratory and cannot
feed Stage 60, Stage 70, recipe selection, deployable selection, promotion, or
a routing-quality claim.
