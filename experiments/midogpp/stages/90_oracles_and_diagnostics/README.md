# Stage 90: Oracles and Diagnostics

This stage contains post-hoc oracle upper bounds, fidelity analyses, rejected
lineage audits, and other non-deployable diagnostics.

Every oracle row must be marked non-deployable. Diagnostic results may explain
a failure or quantify headroom, but they must not tune or select a deployable
policy and must remain separate from held-out utility claims.

## PCSI-PARC Boundary-Projected Whole-Policy Regret Router v1

This Stage-90 experiment is planned but not runnable. It is designed as one final
`POST_HOC_CONSUMED_TEST_SENSITIVITY` over all 9,928 eligible MIDOG++ test rows.
It would evaluate each of the 218 whole cases exactly once with only `H minus c`
available as route-local support and recompute 810 physical cells from exactly
six original inputs; no predecessor Stage-90 output, amendment, probability
surface, decision, checkpoint, or scratch state is an input.

PCSI-PARC projects only hard P-to-B/I/R crossings to the nearest binary32 side
of `0.5`, collapses byte-identical projected vectors, and keeps off-crossing P
probabilities byte-identical. Target influence would select an action. Intended
authorization combines joint BACC/Brier/log-loss ridge responses, true H/J
double exclusion, support-conditioned endpoint-reconstructed transport, and
144 full-policy pseudo-target replays. `RAW_FULL_ACTION_PARC`, fresh legacy
dual-veto, projected-no-PARC, and blocked-fingerprint controls remain
hash-isolated.

NEEDS EVIDENCE — `execution_authorized=false` and
`transport_protocol_status=BLOCKED_IDENTITY_LEVEL_ROUTE_FEEDBACK`. Transport
uses upstream source-prior and route-local endpoint-target-minus-held-case
support capabilities. Although held-case, pseudo-evaluation, and terminal
capabilities are not opened directly, a center-wide aggregate of OOF endpoint
states can carry one case's label identity through other case fits back into
that case's authorization. Identity-level route noninterference is therefore
unproved, and canonical execution and persistence are blocked pending a
route-scoped transport redesign plus poison validation.

The intended workstation schedule uses two persistent RTX A5000 generation workers,
then disjoint CUDA-hidden CPU phases: four endpoint processes with three BLAS
threads and four posterior/utility/replay processes with one BLAS thread. The
frozen counts are 3,488 endpoint fits, 436 posterior fits, 1,395 utility fits,
and 144 whole-policy replays.

The registration remains `status: planned`; `workspace run` refuses before the
runner, and the runner independently rejects the invalid transport lineage.

Reserved canonical output (no canonical bundle may be persisted while blocked):

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router/v1/
```

Any future corrected run remains terminal-only: `fresh_evidence=false` and
`may_feed_another_experiment=false`. It cannot establish routing quality,
NELBO compatibility, downstream utility, nominal coverage or significance,
promotion, or deployment. Until route-scoped transport is validated, retain P;
after a compliant diagnostic, move the unchanged rule
to a fresh whole-case/patient/slide-disjoint reservation.

## Uniform-B v2 Consumed-Test Opportunity-Gated Dual-Endpoint Router v1

This terminal diagnostic independently recomputes the common B/U/eight-A1
probability surface and combines two frozen endpoints. I admits only positive
held-case flip opportunities with a strictly positive `H minus c`
support-calibrated expected-BACC proxy, separately mean-absolute normalizes the
eight case and donor-G scores, and ranks them with fixed weights `4/5` and
`1/5`. R recomputes the exact nine `K in {4,5,6}` by
`w in {1/2,3/5,7/10}` robust arms. The portfolio is exactly
`3/5 P(I) + 2/5 P(R)` before the sole threshold `0.5`.

The experiment uses six successor-fenced original inputs and a direct
original-ledger amendment; no earlier Stage-90 result or probability surface
is an input. It has no recovery strategy. The reused test surface and both
portfolio weights are post-hoc, so all BACC, calibration, identification,
permutation, and delete-center summaries are descriptive only. Active-source
identification, incremental superiority to R, nominal significance, fresh
routing, downstream utility, promotion, and deployment remain unestablished.

See
[`midogpp-uniform-b-v2-consumed-test-fixed-bank-loo-opportunity-gated-dual-endpoint-router.md`](../../../../docs/wiki/03-experiments/midogpp-uniform-b-v2-consumed-test-fixed-bank-loo-opportunity-gated-dual-endpoint-router.md)
for the full protocol and workstation schedule.

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

## Uniform-B v2 consumed-test fixed-bank support-static router S4 v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4.v1`
is a quarantined support-size sensitivity diagnostic over the already-consumed
MIDOG++ test split. It recomputes from exactly six ordered inputs: the promoted
bank, GenerationLock, dedicated cache and manifest aliases, a byte-exact alias
of the immutable original consumption ledger, and this run's direct amendment.
No earlier Stage-90 output, amendment, prediction, scratch, or checkpoint is an
input.

The fixed action surface is B, internal control U, and eight A1 source actions.
Across five deterministic whole-case folds, the other four folds are support
for S4 and the held fold is evaluation. S4 compares all eight A1 actions with B
by exact pooled support BACC, selects the highest gain only when it is strictly
positive, uses a `1e-12` tolerance plus numeric-source tie break, and otherwise
returns exactly B. A support aggregate lacking either class also returns B.
For each candidate source `e`, `G_static` is the equal-center mean exact gain
over query centers `q` outside `{H,e}`; it also requires a strictly positive
best gain or falls back to B. There are no case features, donor fits,
target-local calibration, thresholds, or learned hyperparameters. `O_static`
and `O_case` exist only after terminal label open.

B, U, and all A1 probabilities seal globally before labels open. Label access
then follows per-`(H,f)` role capabilities: each S4 route and its 10,000-draw
null plan seal before the evaluation capability for that same target/fold can
open. The null holds B fixed and shifts complete A1 case contribution blocks
using the predeclared SHA-256/counter-SplitMix64 nonzero cyclic-shift family;
labels are never permuted.

The workstation schedule uses one persistent spawned source worker per RTX
A5000, then a CUDA-free pool of four CPU workers with three threads each.
Float32 memmaps hold source and probability products, float64 is used for
scientific reductions, and scratch is isolated at
`/data/local/fixed_bank_support_static_router_s4_v1`. Two fresh-process replay
validations are required.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_support_static_router_s4.v1
```

The registered run is unexecuted. Any future output remains
`POST_HOC_CONSUMED_TEST_SENSITIVITY` and
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`. Two-sided t8 intervals and null
exceedance counts/fractions are descriptive only; there is no confirmatory
p-value or pass/fail gate, and the result may not feed another experiment,
numbered stage, action, policy, recipe, promotion, or deployment.

## Uniform-B v2 consumed-test fixed-bank LOO directional-shrinkage ensemble v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_directional_shrinkage_ensemble.v1`
is a new isolated Stage-90 diagnostic over the already-consumed MIDOG++ test
split. Its six inputs are the promoted bank, GenerationLock, dedicated cache
and manifest aliases, a byte-exact original-ledger alias, and this experiment's
direct amendment. It consumes no previous Stage-90 artifact, amendment,
prediction surface, scratch directory, or checkpoint.

B, U, and eight frozen A1 source actions are recomputed as 810 physical
target/action/seed cells and globally sealed before labels open. Each of the
218 whole cases or groups `c` is then held out in turn; support for that route
is all other cases in the same target `H`. Directional support gains are pooled
additive confusion-count gains over B on the `zero_to_one` and `one_to_zero`
B-defined hard branches. The donor prior for source `e` equal-center averages
only query centers `q` outside `{H,e}`.

For each direction, sources are ranked by G. The complete executable arm grid
is `K={4,5,6}` crossed with exact rational weights
`w={1/2,3/5,7/10}`. Each arm selects from OFF plus its top-K sources using
`w*S + (1-w)*G`; OFF has score zero and contributes B. Comparisons remain
rational until a final `1e-12` tie check, ordered OFF then numeric source. All
nine arm identities remain present even when they select duplicate endpoints.
DCSE averages the nine selected endpoint probabilities separately on the two B
branches before the sole `0.5` threshold, with equality mapped positive. The
matched G method uses the identical pipeline with `S:=G`.

Every endpoint plan and aggregate method decision for all 218 held cases must
seal before terminal labels open. Controls include raw directional LOO, the
nested delete-one-support frequency committee, hard vote, unique-action mean,
uniform A1, direction decomposition, leave-one-arm ablations, and whole-pipeline
delete-one-center recomputations. The 10,000-replicate candidate-identity null
uses seed `20260813` and is descriptive only; it carries no exchangeability or
p-value claim.

The workstation schedule uses two persistent spawned RTX A5000 generation
workers, followed by a disjoint CUDA-free CPU phase with four workers and three
BLAS threads each. Float32 stores sources and probabilities, int64 stores
confusion counts, and float64 performs scientific reductions. Dedicated
`/data/local/fixed_bank_loo_directional_shrinkage_ensemble_v1` scratch is
throughput-only. Intra-launch atomic task checkpoints are cleaned after their
validated global seal; owned-task replay, terminal recovery, and cross-run
recovery are forbidden.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_directional_shrinkage_ensemble.v1
```

The run is registered but unexecuted. The contract hash is
`500dc61f9f8d3bd0`, the protocol hash is `d3dfdfb4d612a97b`, and the direct
amendment SHA-256 is
`05f800f1bd053528477abd1e67163612c01d44f56418f98961bcdf64677bdc52`.
Its bounded interpretation remains **no stable incremental target-support
advantage versus G**. Even if every descriptive stability check is positive,
the result remains `fresh_evidence=false`, terminal diagnostic-only evidence;
it cannot establish routing or downstream utility or authorize another stage,
experiment, action, policy, recipe, promotion, deployment, or generic reuse.

## Uniform-B v2 consumed-test fixed-bank case-directional correctness abstention router v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_case_directional_correctness_abstention_router.v1`
is an isolated successor diagnostic for the remaining DCSE bottleneck: deciding
whether a source's actual held-case directional flips are likely to be helpful
or should abstain to B. It recomputes the frozen B, U, and eight A1 actions from
the original six experiment-fenced inputs. It consumes no earlier Stage-90
artifact, fitted model, amendment, prediction surface, scratch directory, or
checkpoint.

For every held whole case `c` in target center `H`, six label-free directional
features are computed for each source and each B-defined flip direction. A
separate fixed ridge-binomial logistic model is fitted on labeled cases in
`H` excluding `c` only. Its expected directional correctness is converted to a support-
denominator utility proxy and averaged, with exact weight `1/2`, with the donor
prior computed only from query centers `q` outside `{H,e}`. OFF is score zero,
is first in `1e-12` ties, and leaves B unchanged. The model family, alpha,
iterations, convergence tolerance, clips, six features, score weights, and
candidate order are frozen; there is no fitted threshold or parameter search.

The label capability boundary is global. All 72 donor grants complete before
any target-local route support is opened. The held route's labels never enter
its feature row, scaler, fit, denominators, or decision. All 218 route decisions
and predictions bind to one aggregate seal before terminal labels can open.
A deterministic candidate-feature-block permutation is persisted as a
descriptive specificity control only. Terminal BACC contrasts and directional
oracles are descriptive and cannot become a success gate.

The workstation runs two persistent spawned RTX A5000 generation workers,
then clears CUDA and runs four spawned CPU route workers with three BLAS
threads each. Source/probability storage is float32, confusion counts are
int64, and scientific reductions are float64. Dedicated
`/data/local/fixed_bank_case_directional_correctness_abstention_router_v1`
scratch is throughput-only; cross-run and terminal recovery are forbidden.
Two independent CUDA-free interpreter replays must reconstruct the same
closed-world bundle before completion.

The first workstation execution reached validation finalization but exposed a
numerical-topology defect: route workers fitted under three-thread BLAS while
the serial validator replayed under one thread, changing only last-bit model
coefficients and their derived hashes. The repair shares one exact
three-thread route-numerics scope between production and reconstruction and
retains exact comparison with no tolerance. A registered validation-only
finalization strategy accepts only that exact failed state and inventory; it
cannot rewrite indexed science or terminal products. Ordinary cross-run and
terminal recovery remain forbidden.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_case_directional_correctness_abstention_router.v1
```

The registered run has sealed terminal products but is not complete evidence
until the repaired parent and two fresh-process replays pass. The config
contract hash is `a41dce9dfd086f4a`, the protocol hash is
`336c64c87e48a37b7437c9cd0b6cf44ddf155fc10f93f6e701debd1aaf571429`,
and the direct amendment SHA-256 is
`edbd969666bdd1c5752e2d9904505e07026e6b8307430cfd8fa804010a06e3be`.
Every possible result remains `POST_HOC_CONSUMED_TEST_SENSITIVITY` and
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`: it cannot establish fresh routing,
held-case exact BACC prediction, downstream utility, or deployment evidence,
and it may not feed another experiment or stage.

## Uniform-B v2 consumed-test nested donor endpoint-regret router v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_nested_donor_endpoint_regret_router.v1`
is the isolated successor diagnostic for testing whether case-level endpoint
choice can be stabilized across domains. It independently regenerates B, U,
and all eight frozen A1 actions from exactly six experiment-fenced inputs and
seals 810 exact-nine probability cells. It consumes no earlier Stage-90
result, amendment, prediction surface, scratch directory, or checkpoint.

For each target case `c` in center `H`, endpoint nomination uses whole-case
support `H-c`. Every support voter `s` is evaluated by one unordered state
fitted on `H-{c,s}`; each unordered state is reused for its two ordered voter
directions. The canonical topology is 218 outer routes, 2,660 unordered pair
states, 5,320 ordered voters, and 46,048 endpoint-model fits. All route plans
are sealed before labels open.

The stability mechanism trains on every case in each donor center `J != H`,
including explicit zero-regret protected-P rows when no alternative exists.
Paired BACC regret and paired log-loss delta are separate response surfaces.
The fixed partial-pool Ridge gives each donor center equal total weight, uses
`alpha=1` for feature slopes and `alpha=8` for donor-center effects, and refits
both preprocessing and models for every donor-center deletion. Crucially, a
donor descriptor for outer target `H` computes its external prior only from
query centers outside `H`, donor `J`, and candidate source `e`. It rebinds that
prior over an already-fitted IRLS basis, so outer-H labels cannot leak through
another donor's descriptor and no extra endpoint fit is required.

The primary route switches away from `P_PROTECTED` only when the full model
predicts positive BACC regret, at least seven of eight delete-donor models
remain positive, nested support gain exceeds one half of its voter dispersion,
the full log-loss delta is nonpositive, and at least seven of eight deleted
fits are also proper-loss safe. Every finite gate failure falls back to
`P_PROTECTED`; malformed scientific topology aborts the run. The loss checks
are point-estimate no-worse gates, not noninferiority tests. Both the route and
fallback depend on legal consumed-test support labels, so neither is described
as label-blind.

The eight donor-center blocks are retained only as a descriptive feasibility
screen. Because their cross-fitted fits are dependent, the reported
independent-binomial tails are optimistic power references rather than valid
LTT p-values; statistical authorization is disabled and the feasibility method
returns protected P for every target. The exact 512 center-sign-flip control
reselects the frozen policy identity but holds features, fits, and case
decisions fixed, so it is not a full-pipeline or full-selection null replay.
Terminal output separately reports case-weighted and equal-center endpoint
agreement, rank, normalized oracle gap, center contrasts, switch attribution,
and donor-regret rank association without a nominal significance claim.

The workstation schedule uses two persistent spawned RTX A5000 generation
workers, then clears CUDA and runs four LPT-balanced CPU route workers with
three BLAS threads each. Float32 stores sources and probability cells, int64
stores confusion counts, and float64 performs scientific reductions. Scratch
prefers `/data/local/fixed_bank_loo_nested_donor_endpoint_regret_router_v1`
and otherwise uses an artifact-parent directory. Two sequential CUDA-free
fresh interpreters must reconstruct the full science exactly before the
closed-world bundle can complete.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_nested_donor_endpoint_regret_router.v1
```

The experiment is registered but unexecuted. Its config contract hash is
`1f60b4352a67c60f`, protocol hash is
`474ef49cf7b2fd6ce60ac10d473d5ffdb49abf028737b1aa5ee1d644f782884b`,
and direct amendment SHA-256 is
`20af29472bc6d8e1dc81f6167f65b038e4b57dbb8c0e93dd79a8e84e8b6439dc`.
Every outcome remains `POST_HOC_CONSUMED_TEST_SENSITIVITY` and
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`. It cannot establish fresh routing,
generalization, downstream utility, or deployment evidence, and no artifact
or decision from this run may feed another experiment or stage.

## Uniform-B v2 consumed-test directional signed-utility router v1

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_directional_signed_utility_router.v1`
is the isolated successor to the crossing-helpfulness and nested-regret
diagnostics. It recomputes the same 810 exact-nine probability cells from six
original fenced inputs and retains the lean 218-case, 3,488-endpoint-fit
topology. No earlier diagnostic artifact, amendment, probability surface, or
scratch state is an input.

The scientific unit is the complete case by `{B,I,R}` by
`{zero_to_one,one_to_zero}` rectangle. Structural no-crossing cells remain as
exact-zero rows. One center-balanced multivariate ridge design, with six
unpenalized action-direction intercepts, predicts the direct equal-center
contributions to BACC, Brier score, and log loss. It is refit for all eight
donor deletions; predictions are bias-corrected using only the corresponding
held-donor residuals. These 162 light fits are stability checks,
not independent confidence replicates.

`PDSUR_STABLE` selects at most one endpoint per direction, requires seven of
eight donor-deletion signs to agree, applies the proper-loss safety gates, and
falls back to `P_PROTECTED` on every tie or failed gate. Selected endpoint
probabilities receive a fixed `0.25` shrinkage toward `0.5`; this preserves the
selected hard class while reducing overconfidence. BACC-only, full-only, and
case-blocked-feature controls are persisted separately.

On the workstation:

```bash
env PYTHONPATH=src \
  /home/stud/spark/.venvs/cvae-breakhis/bin/python \
  -m midogpp_thesis workspace validate

env PYTHONPATH=src \
  /home/stud/spark/.venvs/cvae-breakhis/bin/python \
  -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_directional_signed_utility_router.v1
```

The config contract hash is `50915569869e8b8d`, the protocol hash is
`1094b4d5487bfa1f5ec9b76a5f55fc8a23637184777c1bbd9e9044285f39c14b`,
and the direct amendment SHA-256 is
`e17a88f15b1f4ec7537ae61aaa78369a4714869ce9d4abcea53e39d4037b34b8`.
MIDOG++ test reuse is terminal diagnostic analysis only: every outcome remains
`POST_HOC_CONSUMED_TEST_SENSITIVITY` and
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE` and cannot feed another experiment.

## Route-scoped boundary-projected PCSI case-regret diagnostic v2 repair

`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router.v2`
is the fresh, executable sibling of the blocked center-aggregated PCSI-PARC
design. The blocked predecessor remains registered as planned evidence of the
identity-feedback failure and is not an input. This successor independently
recomputes all 810 physical probability cells from six direct-original inputs
and reuses all 9,928 eligible MIDOG++ test rows through 218 leave-one-whole-case
routes.

The first execution identity, `.v1`, failed preterminally on the workstation
when Python 3.12 attempted to transfer a `mappingproxy`-containing utility-model
result from a spawned worker. Its output, scratch, checkpoints, arrays, seals,
and partial label-capability history are quarantined. The `.v2` amendment
authorizes only the mechanical spawn-serialization repair in commit
`1fbd792430dcdb2bfe16f78ead4095fd91c52f0e`; the workstation pickle and real
spawn-process regression gate passed `2 passed, 5 deselected`. V2 cannot read
any v1 state and starts from new output and scratch roots.

Every scientific object that can affect a decision carries case identity. A
target candidate and screen use only `(H,c)` state `H-c`. A pseudo candidate
uses only `(H,J,d)` state `J-d`. Target reference blocks use H/K-excluded donor
states, while pseudo reference blocks use H/J/K-excluded states. Equal center
then equal case weighting, median/MAD normalization, and the leave-one-reference
center maximum threshold are sealed before any pseudo-evaluation capability
opens. Pseudo transport is audit-only and never vetoes or authorizes a target
route.

For geometry `g`, outer center `H`, donor center `J`, pseudo case `d`, and
favorable endpoint coordinate `k`, the diagnostic residual is
`r[g,H,J,d,k] = Ghat[g,H,J,d,k] - G[g,H,J,d,k]`. The primary margin is
`q[g,H,k] = max(0, max_J max_d r[g,H,J,d,k])`. Every donor case is retained;
missing, nonfinite, or incomplete scope invalidates the outer geometry. A target
case changes only if its transport screen passes and all three coordinates of
`Ghat[g,H,c]-q[g,H]` are strictly positive. Equality and every failed gate emit
byte-exact `P_PROTECTED`.

The frozen workload is 3,488 endpoint fits, 436 route-local posterior fits,
1,314 ridge fits, 15,914 role-bound transport descriptors representing 14,170
numeric leaves, 576 reference blocks, 1,962 route screens, and 3,488 case-local
pseudo replays. The final surfaces are P, projected observed-max, raw
observed-max, and projected q=0. The former 81-fit legacy control is removed.

The v2 config contract hash is `aa1fd1d4b63b2404`, the unchanged protocol hash is
`e9da22f3909cd68d8e2bc1cfda727de5167ea93e6ca7aa2e6d466dc9e7f2b85a`,
and the fresh direct-original-ledger v2 amendment SHA-256 is
`5836f034b7f90d46741560f005ebaa1cbbe141e16c5731b41cf9ed112553be87`.
The observed donor-case maximum is neither conformal nor a confidence bound and
has no coverage guarantee. Every output remains post-hoc consumed-test
sensitivity only, cannot establish routing success or target performance, and
may not feed another experiment, stage, model, expert, policy, or deployment.
