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

## Uniform-B v2 Antisymmetric Residual-MMD Router v1

`midogpp.oracle.uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router.v1`
is the terminal consumed-validation diagnostic for a solver-constrained,
class-specific residual route. For eight eligible non-target sources, it uses

```text
w[class 0] = u + d
w[class 1] = u - d
sum_e d[e] = 0
```

where `u[e]=1/8`. This preserves each source's total exposure across the two
balanced generated classes. The optimizer minimizes the predeclared robust
class-conditional/contrast MMD proxy over support-case, training-seed,
generation-seed, and source-only prior variants. The source cap, effective
source floor, and `L1(d)<=0.25` trust region are constraints of the numerical
problem itself. Solver failure, weak soft-class support, nonpositive robust
improvement, or any variant worsening produces the byte-equivalent equal-union
fallback. Continuous residuals are realized as paired integer counts
`n0[e]=128+k[e]`, `n1[e]=128-k[e]`, with `sum_e k[e]=0`.

The estimand is explicitly cohort-adaptive and transductive. The deterministic
two support cases per center remain calibration-only and are never scored. For
each of the 26 remaining cases, the route may use those fixed cases plus the
unlabeled embeddings of every other evaluation case in that center; all rows
of the case being predicted are excluded from its own route. The 468 case-arm
prediction cells are globally sealed before any validation label is opened,
then case slices are reassembled before one target-level BACC and macro-F1 is
computed per seed cell. This is not a static online policy or held-out-target
evaluation.

The workstation runner fits source generation in 27 resumable jobs with one
persistent process on each RTX A5000. It then fits one target-excluded scaler,
source-only class prior, and shared Nyström map per target in nine two-GPU
jobs and reuses each target workspace across its case folds. Downstream work
uses 81 target/seed tasks, four CPU workers, and three BLAS threads per worker;
the equal-union classifier is fit once per target/seed cell and routed fits are
reused by composition hash. The worst-case classifier-fit count is 315.
Before source jobs start, a no-context `nvidia-smi` preflight verifies the two
visible RTX A5000 devices, configured VRAM reserve, 12-CPU affinity, 100 GiB
RAM, 8 GiB artifact-disk reserve, deterministic thread environment, spawn
support, dependencies, and atomic rename behavior. It also rejects a CUDA
context initialized in the parent process.

Run the registered diagnostic with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router/v1/
```

The MMD objective is a label-free compatibility proxy, not NELBO or downstream
utility. These validation labels are already consumed, so the artifact cannot
feed Stage 60, Stage 70, recipe selection, deployable selection, promotion, or
a routing-quality claim. A positive descriptive delta would require a frozen
rerun on separately authorized fresh case-disjoint evidence before promotion.

## Uniform-B v2 Residual Top-up Router v1

`midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_router.v1`
tests a deliberately conservative composition architecture: retain an
immutable equal-union backbone and route only a small additive suffix. The
target geometry uses eight non-target sources, the canonical 128 rows per
source and class as the 1,024-row backbone, and a fixed 128-row top-up. Its
matched control adds the same budget uniformly; the original 1,024-row
equal-union arm remains a separate budget reference.

The only routed action is a parameter-free, label-free ordering of calibrated
variational energy. Lower-energy sources receive larger linear rank priority,
ties are broken by canonical source ID, and Hamilton allocation realizes the
fixed top-up budget. There is no temperature, action-strength, source, expert,
seed, or budget search. Development folds exclude both outer target `H` and
query `q`; their seven-source geometry uses a 1,008-row equal backbone plus a
126-row matched top-up, preserving the same `1/8` top-up ratio.

All development and target predictions for all three training seeds and all
three generation seeds are materialized and globally sealed before any label
is opened. Only after that seal, the fixed action gate computes paired BACC
gains over the eight `q != H` query centers. It selects the routed top-up for
`H` only when the one-sided 95% query-center Student-t lower bound is strictly
positive; otherwise it uses the exact uniform top-up. Labels from `H` never
select the action for `H`.

The workstation schedule is frozen for the Xeon W-2265 and two RTX A5000s.
Twenty-seven source/training-seed jobs are split across one spawned worker per
GPU and materialize 81 hash-validated float32 blocks. Downstream prediction
uses four spawned CPU workers with three BLAS threads each. Source and
prediction checkpoints are hash-validated, global products are independently
revalidated on resume, and source-cache publication uses fsync plus atomic
replacement. GPU and CPU pools are phase-disjoint.

Run the registered diagnostic with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_residual_topup_router/v1/
```

This is a terminal `EXPLORATORY_CONSUMED_DATA_ONLY` Stage-90 diagnostic. Its
energy score is a proxy, not NELBO or downstream utility, and the resulting
artifact cannot feed Stage 60, Stage 70, recipe selection, deployable
selection, promotion, or a routing-quality claim.

## Uniform-B v2 Residual Top-up B/U/G/S Case-OOF v1

`midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof.v1`
is the quarantined same-dataset decomposition of the positive residual-top-up
mechanism. It uses the already-consumed MIDOG++ validation surface, so it is a
terminal Stage-90 diagnostic rather than a fresh policy or downstream
confirmation.

For each target center `H`, two whole cases are frozen as label-free support
`S_H`. The remaining 26 evaluation cases form whole-case OOF scoring folds,
and no evaluation embedding participates in its own route. `G` aggregates
true normalized-midrank ballots over fixed support cases from `q != H` while
excluding both `H` and `q`; `S` uses only fixed `S_H`. All three training
replicas are averaged before each case ballot and Hamilton allocation receives
the explicit Borda priority `1-b`.

The action library contains the original equal-union base `B`, matched uniform
top-up `U`, global-rank top-up `G`, support-rank top-up `S`, one fixed
source-identity permutation control, and all eight `H x e` single-source-tail
diagnostics. Every target/action/training-seed/generation-seed prediction is
globally sealed before the label-bearing manifest can open. No utility
selector, fallback gate, temperature, budget, source, expert, or seed search
exists.

The predeclared primary endpoint is all-nine-seed probability-ensemble BACC.
The primary center-level contrasts are `S-U` and `S-G`; `G-U`, `U-B`, `S-B`,
and `S-P` are secondary or diagnostic. The 81 seed cells are technical repeats;
inference uses the nine target centers. The sealed `H x e` matrix reports
headroom and rank diagnostics only and cannot update an action.

Run the registered diagnostic on the workstation with:

```bash
cd /home/stud/spark/cvae_experiments && env PYTHONPATH=.:src \
  /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof/v1/
```

Regardless of outcome, this artifact cannot claim fresh evidence, successful
target-specific routing, NELBO or utility prediction, promotion, or deployment,
and it cannot feed Stage 60 or Stage 70.

## Uniform-B v2 Consumed-Test Fixed-Bank Label-Aware Case-OOF Ceiling v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling.v1`
is a terminal information-ceiling diagnostic. It rematerializes `B` and all
legal direct `Hxe` actions for the frozen known bank, globally seals all 729
seed-cell probabilities, and uses five deterministic whole-case folds per
target. Every one of the 218 consumed test cases is evaluated exactly once.
The implementation and configuration are registered, but no canonical
workstation result exists until the complete closed-world bundle validates.

The label-derived LOCO prior `G_H` uses only other-center labels, excludes all
labels from `H`, is not shared across targets, and is sealed before `H` support
labels open. A fold-local posterior uses exact candidate-minus-`G_H` BACC from
only the other four folds of the same center; other centers are equally
weighted in `G_H`. All 45 decisions are sealed before evaluation-role labels open.
Exact-nine probability-ensemble BACC is the only decision/gate utility; smooth
metrics are descriptive and dependency-disconnected. The 10,000 fixed null
draws derange only the eight candidate-source labels inside each target/fold/
support-case block; `B` is fixed, the candidate multiset is preserved, and no
evaluation case supplies a donor. All null actions are durably sealed before
evaluation-role labels open.

The experiment has its own user-authorized, hash-chained consumed-test ledger
amendment. Its result is never fresh evidence and cannot update a model,
authorize a route/action/policy, feed Stage 50/60/70 or another Stage-90 run,
select a recipe, promote a method, or support deployment.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling.v1
```

The workstation completed and sealed the full v1 probability surface, then
stopped before any fold decision was created: its per-case BACC implementation
incorrectly required both classes inside each individual case. MIDOG++ has 213
mixed cases, four negative-only cases, and one positive-only case. The v1 root,
partial capability history, scratch, and checkpoints are therefore quarantined;
v1 must not be resumed or used as input.

## Uniform-B v2 Consumed-Test Fixed-Bank Pooled-BACC Case-OOF Ceiling v2

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling.v2`
is the separately authorized terminal replacement diagnostic. It recomputes
`B` and all legal direct `Hxe` probabilities from the original six fenced
inputs under a new contract hash, scratch namespace, output root, and
hash-chained amendment. It consumes no v1 output, prediction store, prior,
decision, label-capability state, scratch, or checkpoint.

The dedicated test-cache alias retains its explicit Stage-70-derived label-free
feature-cache lineage. It is not a Stage-70 prediction, scoring, or policy
result; none of those outputs is consumed.

V2 retains all 218 cases. For each case/action it stores only `n+`, `TP`, `n-`,
and `TN`; a case may lack one class. Exact BACC is computed only after those
counts are pooled over the full legal support, donor, or evaluation scope,
which must contain both classes. Whole cases are paired uncertainty clusters,
not equal-weight utility observations. Thus the utility is row-pooled while
the uncertainty respects within-case dependence.

For target `H` and source `e`, each LOCO effect uses the seven legal donor
centers `H'` outside `{H,e}`. `G_H` takes the candidate with maximum prior mean
only when its 95% lower bound versus `B` is strictly positive; otherwise it is
`B`. Candidate-versus-selected-`G_H` pairwise priors use the same shared legal
donors—seven when `G_H=B`, six when it is a source—and all `G_H` and pairwise
prior seals precede any `H` support-label access. Fold support updates use the
predeclared paired whole-case influence variance and normal-normal posterior.
The router takes the candidate with maximum posterior lower bound only when it
is strictly positive; otherwise it abstains to `G_H`.

All 45 observed actions and all `10,000 x 45 = 450,000` null actions are sealed
before evaluation labels open. For each support case, the null orders the eight
candidates by SHA-256 over seed/fold/case/action and applies an independent
counter-SplitMix64 nonzero cyclic shift in `{1,...,7}` for each null index. It
therefore deranges complete candidate sufficient-statistic blocks, fixes `B`,
preserves the candidate multiset, and recomputes the same pooled-BACC cluster
posterior. This is the predeclared restricted cyclic-shift family, not a uniform
sample from all eight-action derangements. Evaluation reports pooled exact BACC
per center and uses the nine target centers as equal-weight inference units.
The permutation primary statistic is equal-center `R-G_H`; its upper-tail field
`one_sided_p_value` is `(1 + #null >= observed)/(K + 1)`, its
`lower_tail_p_value` is `(1 + #null <= observed)/(K + 1)`, and
`two_sided_p_value` is `min(1, 2*min(upper, lower))`.

Normalized regret is fixed to `0.0` when oracle headroom over `B` is at most
`1e-12`: that fold has no routing opportunity, and the predeclared convention
avoids an undefined numerical-zero denominator.

The workstation schedule remains two persistent A5000 workers with a CUDA-free
parent, followed by four CPU workers with three BLAS threads each. V2 scratch is
`/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2`.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling.v2
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling/v2/
```

This remains `EXPLORATORY_CONSUMED_DATA_ONLY` and `DO_NOT_PROMOTE` regardless
of outcome. It cannot authorize a route, action, policy, model or expert update,
routing-quality claim, recipe, promotion, deployment, or feed to Stage 50/60/70
or any later Stage-90 experiment.

## Uniform-B v2 consumed-test utility-aligned target-static endpoint router v1

`midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router.v1`
is a separately authorized terminal diagnostic over the already-consumed
MIDOG++ test surface. It has exactly six inputs: the promoted expert bank,
GenerationLock, a dedicated byte-exact label-free cache alias, one capability-
gated manifest alias containing both `manifest.csv` and `domain_mapping.json`,
an alias of the immutable original consumption ledger, and a direct single-
consumer amendment. It consumes no prior Stage-50/60/70 result, Stage-90
output or amendment, prediction surface, scratch directory, or checkpoint.

Membership is label-free and seed-independent. After canonical case-ID sort,
the first eight whole cases in every center form support (72 cases and 2,902
rows); the other 146 cases and 7,026 rows form evaluation. All 218 cases and
9,928 rows participate. Support labels never open. Class coverage is checked
only after membership is frozen and cannot change the partition.

The development surface contains exactly 504 `H/q/e` exact-nine probability-
ensemble BACC deltas. Each fit excludes `H`, `q`, and candidate `e` from their
respective roles. Development probabilities seal before other-center `q`
labels open. The one global predictor is derived locally from the hash-pinned
three-axis domain mapping in the experiment manifest alias; center/domain IDs
and labels are not model features, and no seventh metadata artifact is used.
M1 adds one ensemble-first, label-free support action-probability-shift scalar;
the permutation control is a deterministic same-capacity refit.

The feature runtime also records posterior-mean reconstruction MSE,
latent-dimension-normalized analytic PS KL, and linear-kernel squared mean
discrepancy as descriptive CVAE diagnostics. None enters M0/M1/P, none is an
exact NELBO or downstream utility, and the unsigned probability-shift predictor
is classifier sensitivity rather than generative compatibility. Every Hxe (and
any G/R/P alias that selects it) remains an equal-union B action plus a
single-source tail, not standalone expert utility.

Each target receives one static plan, never a per-case route. The plan is built
with the neutral `evaluate_ensemble_cardinality_transfer` and
`build_ensemble_utility_policy` APIs using exactly 32 whole-case support
bootstrap replicates under seed `90703`. `R` is selected only when the frozen
source-inner transfer/capacity gates and its selected-gain lower bound pass;
otherwise the exact `B` action is retained. `G` and `P` are diagnostic
selections and `U` is a terminal matched control. This is not a simultaneous
prelabel lower-bound gate against `U/G/P`.

All nine plans and all target predictions are globally sealed before the
same-outer-`H` evaluation labels open. Those labels never build plan `H`;
other-center evaluation labels may already have opened as development `q`
labels after the development seal. Terminal reports contain exact-nine BACC
for `B/U/G/R/P` and all eight `Hxe` candidates, the predeclared `R-B`, `R-U`,
`R-G`, and `R-P` contrasts, and descriptive Hxe top-1/rank/normalized-oracle-
gap diagnostics. No terminal label can update a plan or policy.

The workstation schedule uses one spawned generation worker on each RTX A5000,
then a CUDA-free four-worker CPU phase with three BLAS threads per worker.
Arrays are stored as float32; scientific reductions are float64.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router/v1/
```

Regardless of outcome, the bundle remains
`EXPLORATORY_CONSUMED_DATA_ONLY` and `DO_NOT_PROMOTE`. It cannot establish
fresh evidence or routing success, authorize an action/policy/model/expert
update, feed any numbered stage or another experiment, select a recipe, or
support promotion or deployment.

## Uniform-B v2 Consumed-Test Fixed-Bank Hierarchical Residual Stacker v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker.v1`
is the terminal mechanism diagnostic motivated by the pooled-BACC ceiling's
retrospective headroom and unstable hard actions. It does not consume that
ceiling. A dedicated six-input fence and a new direct-to-original-ledger
single-consumer amendment require a full probability recomputation under a new
root and scratch namespace.

The architecture remains anchored to `B`. Whole-case residual-logit features
and a probability-only source descriptor feed separate positive- and
negative-class ridge effect models under strict `H/q/e` exclusion. `B_cal`
isolates support-only threshold calibration, `G` is a case-independent stack,
`R` is case-conditional, and `P` permutes whole local-feature blocks before both
fit and inference and refits the same capacity. A soft `B_cal` class gate avoids
hard pseudo-class reversal. Only two positive-score sources per class may enter,
lambda is at most `0.25`, and lambda zero is an exact `B_cal` fallback.

Five deterministic whole-case folds evaluate every case once. Support selects
only `b` and one common lambda by fixed class-balanced log loss; exact pooled
BACC supplies the paired whole-case lower-bound gate and terminal endpoint. The
primary equal-center contrasts are `R-B_cal`, `R-G`, and `R-P`. All method
actions are sealed before evaluation labels open.

The workstation uses one two-A5000 probability phase followed by four spawned
CPU workers with three threads over shared float32 memmaps. Config contract hash
is `cb7050fcdaac86ac`; the amendment SHA-256 is
`e915134fc15901f1d5c43fb5fb974f1693282ca4622a2ade169eaa7487566b1b`.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker.v1
```

No result exists until the 39-file bundle validates. Regardless of outcome,
the artifact remains terminal `EXPLORATORY_CONSUMED_DATA_ONLY` evidence and may
not feed another experiment, numbered stage, action, policy, recipe, promotion,
or deployment.

## Uniform-B v2 consumed-test fixed-bank actionability/recoverability v1

This registered diagnostic asks two questions left unresolved by the failed
signed correction gate: whether the fixed bank contains useful complementary
actions, and whether their utility ordering is recoverable. It does not consume
that gate or any other Stage-90 output. A direct-to-original-ledger amendment
authorizes only this one reused-test terminal analysis.

`B` and shared `U` are recomputed. Each geometry has eight source actions:
`A0` uses a 256/128 selected/other row allocation, while `A1` reuses exactly
those rows with fixed selected/other weights `23/16` and `7/8`. The two
geometries are reported independently; no strength, class, source-pair, or
geometry search is legal. Per-geometry `G/R/P` models use strict `H/q/e`
exclusions, and same-target support may choose only the static `S_y` action
within that already-frozen geometry. `O_static` and `O_case` are terminal
label-informed bounds, never routing methods.

The canonical workstation schedule uses two persistent A5000 workers for the
frozen source streams, then clears CUDA visibility and uses four spawned CPU
workers with three threads each for 1,458 classifier cells, the small ridge
models, and paired whole-case inference.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_actionability_recoverability.v1
```

The result remains `EXPLORATORY_CONSUMED_DATA_ONLY` and `DO_NOT_PROMOTE` even
if an oracle or learned method improves BACC. It cannot select an action or
geometry, authorize routing, update a model or expert, feed another experiment,
or establish fresh evidence.

## Uniform-B v2 consumed-test labeled-support case-conditional flip router v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_labeled_support_case_conditional_flip_router.v1`
tests the remaining labeled-support, case-conditional quadrant identified by
the actionability/recoverability audit. This is a new, isolated Stage-90
diagnostic. It recomputes its own surfaces from exactly six fenced inputs and
does not consume an earlier Stage-90 result, amendment, prediction, scratch, or
checkpoint.

The action library is frozen to `B`, shared `U`, and the eight A1 source
actions. All 810 target/action/seed probability cells and every label-free
case-action feature are sealed before labels open. Within each target center,
five deterministic whole-case folds rotate three roles: fold `f` is evaluated,
fold `f+1 mod 5` calibrates only the target-local directional flip rule, and the
other three folds select among the eight static A1 challengers. `B` and `U` are
controls, not static-selection candidates; a nonpositive or unauthorized
challenger falls back to exactly `B`.

`F_S` is the primary heuristic uncertainty-gated case router. It may flip the
`S_static` hard prediction only where the frozen case features, fixed
ridge-alpha `1.0` model, fixed directional calibration prior, and predeclared
descriptive score bound admit the change. That per-case bound is not calibrated
confidence or a safety guarantee. `G_static` selects its source by an additive
query/source fixed-effect fit to exact per-`(q,e)` pooled gains under sum-to-zero
effects, avoiding unequal query-mixture comparisons. `S_static`, `F_G`, and the same-capacity refit
permutation `F_P` are controls. `O_static` and `O_case` open only during
terminal scoring. The primary contrasts are `F_S-B`, `F_S-U`, `F_S-F_G`,
`F_S-F_P`, and `F_S-S_static`; top-1 oracle agreement, Spearman rank,
normalized oracle gap, and stability remain identification diagnostics.

The workstation schedule uses one persistent spawned generation worker per RTX
A5000, followed by a disjoint CUDA-free pool of four CPU workers with three
threads each. Float32 memmaps and hash-validated checkpoints live preferentially
under `/data/local/fixed_bank_labeled_support_case_conditional_flip_router_v1`;
scientific reductions and 10,000 whole-case bootstrap replicates use float64.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_labeled_support_case_conditional_flip_router.v1
```

The result is terminal `EXPLORATORY_CONSUMED_DATA_ONLY` evidence regardless of
its metric values. It cannot establish fresh routing success, authorize an
action or policy, update a model or expert, select a recipe, feed another
experiment or numbered stage, or support promotion or deployment.

## Uniform-B v2 consumed-test multi-challenger hierarchical flip router v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_multi_challenger_hierarchical_flip_router.v1`
is the isolated follow-up to the negative single-challenger result. It keeps
the same immutable A1 action geometry but recomputes all 810 exact-nine
probability cells from the original six fenced inputs. It never consumes the
earlier Stage-90 result, prediction surface, amendment, scratch, or checkpoint.

Within each five-fold whole-case rotation, three selection folds rank all eight
A1 candidates against fixed B and seal B plus the top three. The best
positive support action is the static anchor (otherwise B). A disjoint
calibration fold fits only two target-local Gaussian-prior direction offsets;
the strict-H/q/e shared models remain frozen. Sparse directions retain the
donor prediction and its prior or Laplace uncertainty rather than aborting or
receiving zero variance.

The shared directional models use pooled beneficial-flip binomial counts with
penalized query and candidate-source effects. `R_multi` adds the fixed eleven
label-free case/action features, `G_multi` omits those features, and `P_multi`
refits the same capacity after complete-case feature-block permutation. The
router compares all menu actions jointly and leaves the support-static anchor
only when the winner-versus-runner-up expected-gain margin has a positive
predeclared 1.96 bound. The bound includes shared-model and calibration
covariance but no irreducible outcome variance, and is therefore a descriptive
asymptotic action margin rather than calibrated confidence or safety.

The workstation schedule uses one persistent spawned source worker per A5000,
then a disjoint CUDA-free CPU phase with four workers and three threads each.
Probability lookups are pre-indexed by case/action, generated streams use
float32 memmaps, reductions use float64, and fitted values are replayed with a
field-level allow-listed tolerance while provenance, hyperparameters, menus,
ranks, actions, reasons, counts, and terminal confusion products remain exact.
Finalization and recovery accept a validation report only after two independent
fresh Python processes reconstruct the same full-bundle checks exactly.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_multi_challenger_hierarchical_flip_router.v1
```

`R_multi` must beat B, U, `F_single`, `G_multi`, `P_multi`, and `S_static` by a
strictly positive one-sided cross-center LCB for its diagnostic gate to pass.
Regardless of that result, the artifact remains
`EXPLORATORY_CONSUMED_DATA_ONLY` and `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`;
it cannot establish fresh routing or deployment evidence or feed another
experiment.
