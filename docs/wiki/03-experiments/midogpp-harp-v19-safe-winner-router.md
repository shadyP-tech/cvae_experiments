# HARP v19: safe-benefit routing with a complete winner gate

V19 implements a dedicated, modular successor to HARP v18. Its checked-in
state is **planned and execution unauthorized**. No MIDOG++ source dataset
was run for v19, and no v19 source-label capability, execution amendment, or
single-use lease has been issued. Synthetic construction and performance
checks do not change that state.

V17 and v18 remain exhausted. Their learned models, prediction caches,
outputs, labels, thresholds, authority, and seals are not successor inputs.
Their terminal diagnostics motivated the design: v18's directional menu
contained useful oracle actions, but its selected actions had substantially
more observed harm than predicted. V19 therefore changes the learning
population, executed-action features, selection objective, and calibration of
the complete winner. It does not lower the policy's scientific risk limits.

## Candidate construction and learning population

The maximum menu remains 38 recipes: exact B, exact U, and
`D01_ONLY(K, lambda)`, `D10_ONLY(K, lambda)`, `BOTH(K, lambda)` for K in
`{1,2,4}` and lambda in `{0.25,0.5,0.75,1}`. The pairwise donor ranker remains
source-trained. Every directional composite uses the actual ranked donors,
top-K averaging, and lambda mixing; untouched branches copy the exact B bytes.
Exact U retains its physical probability vector.

The prediction-changing candidate set A_i contains structurally eligible,
unique, nonbaseline candidates with at least one actual hard-prediction
change at the fixed classifier threshold 0.5. Exact B is an analytic fallback.
Probability-only changes have exactly zero BACC gain and are excluded from
the candidate routing model's training population. They remain in the source
frontier and proper-loss diagnostics. Negative and zero-gain examples among
hard-changing candidates are retained. Deduplication uses exact probability
vectors, not merely identical hard classes.

Features describe the executed mixture rather than an average of primitive
margin statistics. They include sample counts, separate D01/D10 flip counts
and fractions, margins and shifts on the actual flip sets, selected-donor
disagreement, family, K, and lambda. Whole-expert compatibility remains
explicitly named context. Opposite, unexecuted directional changes do not
enter executed-action descriptors.

Exact per-seed composite predictions are unavailable on this surface. V19
therefore does **not** claim or substitute exact composite seed variance.
A mean of donor seed dispersions would not be the composite's variance.

Candidate fitting assigns one equal-center, equal-case weight to each
participating case, divided across that case's remaining candidates. It
normalizes those weights after filtering. Cases with empty A_i remain in the
full scoring-scope normalizer and in policy evaluation.

## Outcome model and exact decision rule

Let g_ia be the aligned whole-case BACC contribution of action a. Define the
mutually exclusive categories

\[
S=\{g_{ia}>0,\;d_{B,ia}\le0,\;d_{L,ia}\le0\},\qquad
H=\{g_{ia}<0\},\qquad O=(S\cup H)^c.
\]

A shared regularized softmax predicts p_S, p_H, p_O. Separate conditional
magnitude heads predict nonnegative m_S=E[g|S,z] and m_H=E[-g|H,z]. The candidate
score is

\[
s_{ia}=\widehat p_S\widehat m_S-\widehat p_H\widehat m_H.
\]

This estimates safe benefit minus BACC damage. It is **not a confidence lower
bound**. Additional heads expose signed aligned gain, classwise recall
changes, Brier delta, and log-loss delta. The proper-loss forecasts are
available to the winner gate; they do not replace the measured policy-level
constraints.

Choose exactly one nonbaseline winner a_i* by maximum s_ia, breaking ties by
lexical arm ID. Save that winner even if its score is negative. The winner
gate fits its own coherent S/H/O model on honestly held-out winners, including
harmful and negative-score winners. It receives candidate forecasts, exact
flip-local descriptors, family, candidate count, and the winner–runner-up gap.
Its gate score is t_i=1-q_H(x_i,a_i*).

For an admitted and enabled policy, the complete rule is

\[
\pi_\tau(x_i)=
\begin{cases}
a_i^*, & A_i\ne\varnothing,\quad s_{ia_i^*}>0,\quad t_i\ge\tau,\\
B, & \text{otherwise}.
\end{cases}
\]

The threshold comparison is inclusive; exact B wins the zero-benefit boundary.
A gate rejection never tries the second-ranked candidate. There is no
individual predicted-harm, predicted-Brier, or predicted-log-loss veto.
Admission still constrains the actual complete policy.

Candidate and gate ridge penalties are fixed at 0.01 on the normalized
mean-loss scale; the donor ranker retains 1.0. Only tau is tuned, from the nine
fixed values `{0,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95}`. This implementation does
not shop across regularization choices or promote a diagnostic ablation to
the primary procedure.

## Complete nesting and pre-truth evidence

`FitProposer(S)` builds four ranker folds within S. Each fitted ranker proposes
immutable composites for its held cases. Once every such composite in S is
sealed, the full-S class-support normalizer supplies outcomes to the shared
candidate model. The final donor ranker is fitted on S.

`FitComplete(S)` adds three winner folds. For each held block W, it calls
`FitProposer(S minus W)` and records the unthresholded winners for W. Thus W
is absent from every upstream ranker, candidate-outcome fit, and fitted
feature transformation. All winners are sealed before their outcome join.
The gate fits on those records with one center-balanced weight per
participating case. A final proposer is then fitted on S.

Five outer folds evaluate this entire learner. Within each outer training set,
four inner folds rebuild `FitComplete(T minus V)` before predicting V and
selecting tau. Deleting V from a globally precomputed winner table is not used.
All outer selections are sealed before the full-source OOF normalizer and
scores are joined. Final threshold selection repeats full-source inner CV,
then refits the complete learner. Nested OOF assesses that learning procedure;
it does not certify the eventual full-source refit.

Each complete model binds its proposer, candidate model, winner gate, exact
training keys, and nested fit receipts. Pre-truth decisions separately bind
the signed winner score and nonnegative gate score. Source OOF seals also bind
the inner-selected policy-enabled flag and exact fallback reason, so a disabled
policy cannot be confused with a gate rejection or an enabled abstention. A winner transcript
contains its identity, coherent S/H/O probabilities, raw ordered gate feature
names and values, gate model hash, and prediction hash. Fresh validation can
replay the authenticated gate transform and softmax, then check the complete
rule. Admitted routes must implement both directions of the rule: a passing
winner must route, and a failing winner must use exact B with the matching
reason.

## Estimand and admission

This is known-center source-train development to terminal full-test routing,
conditional on the already frozen expert bank. Source menus use C minus q;
target menus use C minus H. The intended inventory is 216 training cases
(9,648 rows) and 218 test cases (9,928 rows), across centers
`{0,1,2,3,5,6,7,8,9}`. Source and target cases are disjoint. It does not estimate
unseen-center generalization or an end-to-end refit of the expert bank.

For center c, N_c is its number of cases and M_ck its number of cases supporting
class k in the declared scoring scope. The source contribution is

\[
g_{ia}=\frac{N_c}{2}\left(\frac{\Delta r_{ia0}}{M_{c0}}+
                              \frac{\Delta r_{ia1}}{M_{c1}}\right).
\]

An absent class contributes zero for that case. Every fitting/scoring center
must have class support in its complete scope. Individual contributions can
exceed [-1,1]; their center average is the aligned BACC difference. Class
counts and these normalizers are outcome-side quantities, never target
features. Hard-prediction harm is the whole-case event g<0, not patch error.

With all-case equal-center weights and route indicator r, admission retains

\[
G=E_w[rg],\quad M_H=E_w[r(1[g<0]-.25)],\quad
M_B=E_w[r(d_B-.002)],\quad M_L=E_w[r(d_L-.005)].
\]

Positive gain and the unchanged risk constraints are necessary. Coverage needs
at least 18 routed OOF cases and at least six centers with two or more routes
each. Incidental one-route centers still enter the risk accounting. B fallback
cases must not dilute conditional routed risk; dividing the route-weighted
moments by E_w[r] gives the corresponding routed risk.

The existing approximate studentized max-statistic bootstrap uses 1,024 draws,
alpha 0.05, and seed 19019. It resamples whole cases **within the fixed centers**
and recomputes supporting-class denominators. Missing class support in a draw
receives conservative center gain and is reported. It does not resample
technical seeds as independent cases or make a new-center claim. These
approximate bounds exclude final-refit uncertainty and are not conformal or
finite-sample deployment certificates.

Zero source OOF coverage fails closed before target actions and before
bootstrap bounds. Other nonadmission follows the declared exact-B behavior.
An admitted policy with zero nonbaseline target actions stops before test
truth. Tiny nested training scopes or missing center/class support fail
explicitly; held-out normalizers are never borrowed to repair them. A correct
implementation may still find no scientifically admissible routing.

## Dedicated module responsibilities

The scientific folder is
`src/midogpp_thesis/cvae/routing/safe_winner_router_v19/`.

| Modules | Responsibility |
|---|---|
| `contracts.py`, `composition.py` | Typed menus, directional recipes, exact-B branches and aliases. |
| `features.py`, `estimators.py`, `outcome_model.py` | Executed features, coherent categories, mean-loss fitting and safe-benefit estimates. |
| `modeling.py`, `proposer.py` | Pairwise donor ranker and ranker-stacked candidate learning. |
| `winner_records.py`, `winner_gate.py`, `learning.py` | Unthresholded winner seals, selected-population gate and `FitComplete`. |
| `candidate_prediction.py`, `decision_evidence.py`, `records.py` | Deterministic winner, abstention rule and replayable decision evidence. |
| `splitting.py`, `crossfit.py` | Label-independent case folds and complete nested threshold selection. |
| `truth.py`, `aligned_metrics.py`, `admission.py` | Nonserializable capabilities, whole-scope aligned outcomes and policy admission. |
| `frontier.py`, `frontier_joins.py` | Threshold-matched summaries and detailed candidate/winner outcome joins. |
| `fit_cache.py`, `model_integrity.py` | Execution-scoped memoization and fresh model hash verification. |
| `policy.py`, `stacked_fitting.py` | Public policy API and small compatibility facade. |

`runtime/harp_v19_execution/` owns physical generation/classification, the
source/target adapter, exact mechanism descriptors, durable probability
stores, gate replay, and independent reconstruction. The lifecycle resides in
`diagnostics/fixed_bank_harp_router_v19/`; `diagnostics/harp_v19_cli.py` registers
its dedicated commands. These paths are relative to `src/midogpp_thesis/cvae/`.

The source reports are `reports/candidate_frontier.json`,
`reports/source_headroom_diagnostics.json`, and
`reports/source_candidate_winner_joins.json`. Joins retain actual-composite
features, reconstruction recipes, probability hashes, estimates, outcomes,
selection identity, gate transcripts, and threshold decisions. Detailed joins
are saved once per aggregate held-out scope, avoiding duplicate fold and
aggregate copies. Enabled diagnostic threshold sweeps are explicitly marked;
they are not actual nested admitted routes. Oracle opportunities never select
or authorize a policy.

## Workstation topology and measured construction cost

The verified workstation profile is Xeon W-2265 (12 cores/24 threads), about
125.5 GiB RAM, and two 24 GB RTX A5000 GPUs. Two persistent GPU workers are
followed by four classifier workers with three BLAS threads each. Queues are
bounded, multiprocessing uses spawn, CPU workers hide CUDA, and competing
CPU pools do not overlap. The 81 classifier tasks still perform 810 physical
fits. Candidate and winner nesting adds **zero physical generation or
classifier fits**.

All 18 source/target physical menus and bank attestations are sealed before
source truth is opened. The nested truth-bearing router fit stays in the
parent process, under an active one-thread BLAS limit. The configured science
worker capacity is not permission to serialize truth into four workers.
Physical probabilities use float32; fitting and scientific reductions use
float64. Bootstrap calculations use bounded batches. Two fresh processes
reconstruct and validate globally frozen routes before any test labels open.

An execution-local fit cache stores up to 96 models, keyed by exact menus,
training scope, scoped capability identity, and configuration. A separate
8,192-entry cache reuses only raw label-free executed features and restores
its context at the end of the fit. Learned transforms, normalizers, outcomes,
and models cannot cross training scopes through that raw cache. Neither cache
is an exhausted predecessor input or a persisted cross-run learned cache.

The full synthetic benchmark on 5 September 2026 took **522.16 seconds** on
**macOS arm64 with Python 3.11.14**, one BLAS thread, and no CUDA. It used the
216-case/9-center geometry, 48 fabricated samples per case, all 38 recipes,
5 outer/4 inner/4 ranker/3 winner folds, and the nine thresholds. It made 600
ranker fits, 120 candidate fits, and 30 winner-gate fits; peak RSS was about
1.56 GiB. Raw-feature cache reuse was 97.1% (524,821 hits, 15,611 misses).
The exact-scope fit cache recorded **zero hits** in that benchmark; it must not
be credited with a measured speedup.

Profiling identified repeated validation of the same immutable probability
tuples. A bounded identity cache now avoids that repeated scan, while exact
first-use type checks prevent equal-valued invalid objects from borrowing a
valid cache entry. A separate cold complete-fit comparison observed
23.50 to 22.59 seconds with the identical model hash; 100,000 cached validation
calls took 0.1742 versus 0.0147 seconds. This is a single synthetic timing
comparison, not a general speedup guarantee.

The easy fabricated fixture routed 216/216 OOF cases and returned `ADMITTED`.
That demonstrates that the construction can activate actions. It is neither
workstation timing nor evidence of reliable MIDOG++ routing. It does not
benchmark GPU generation/classifier work. The observed result is saved at
`/tmp/harp_v19_synthetic_full_20260905_a/benchmark.json`; timings precede the
final attestation and probability-cache changes and are not a scientific run.

On the actual `xai-master` workstation (`delli2`, Python 3.12.3), the isolated
`--complete-only` synthetic benchmark took **41.86 seconds total** and
**39.55 seconds fitting**, with **566 MiB peak RSS**. It fitted 20 rankers,
four candidate models, and one winner gate on 216 fabricated cases. This mode
does not run outer CV or calculate admission. Floating-point fitted model
hashes differ across the macOS and Linux numeric environments; byte identity
was checked within each environment, not asserted across them.

The checked validation record is
[`docs/validation/harp_v19_2026-09-05.json`](../../validation/harp_v19_2026-09-05.json).
It records test results, construction fingerprints, benchmark scope, and the
closed execution state. Workstation validation used code-only snapshots in
fresh `/tmp` directories; the workstation experiment checkout was not changed.

## Mutation-free inspection and synthetic benchmark

From this repository in the configured Python environment, these commands
were checked against the CLI and executed locally without mutation:

```sh
PYTHONPATH=src python -m midogpp_thesis cvae-diagnostics \
  fixed-bank-harp-router-v19 \
  --config experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v19.yaml \
  --inspect-plan

PYTHONPATH=src python -m midogpp_thesis cvae-diagnostics \
  fixed-bank-harp-router-v19 \
  --config experiments/midogpp/stages/90_oracles_and_diagnostics/configs/uniform_b_v2_consumed_test_fixed_bank_harp_router_v19.yaml \
  --dry-run
```

With the planned config, inspection returns
`PLANNED_NEEDS_SEPARATE_EXECUTION_AMENDMENT`; dry-run returns
`NEEDS_SEPARATE_EXECUTION_AMENDMENT`. Both report `paths_resolved=false`,
`filesystem_mutations=0`, and closed source/test labels. This planned dry-run
is not a validation of prepared workstation inputs or launch authority.

The reusable CPU-only synthetic benchmark accepts a fresh subdirectory of the
system temporary directory and refuses to overwrite its result:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python benchmarks/harp_v19_synthetic.py \
  --output-dir /tmp/harp_v19_synthetic_benchmark_new
```

Add `--complete-only` for the shorter full-source learner benchmark. On the
workstation, use `/home/stud/spark/.venvs/cvae-breakhis/bin/python` in place of
`python`. This option explicitly reports `admission_computed=false` and does
not stand in for a complete nested OOF experiment.

The scientific experiment identity is
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_harp_router.v19`.
Its output namespace ends in
`uniform_b_v2_consumed_test_fixed_bank_harp_router/v19`; scratch is
`/data/local/fixed_bank_harp_router_v19`. V19 has its own catalog aliases for
the source/target cache, source-label capability, evaluation release, parent
ledger binding, amendment, and output, plus dedicated source-snapshot and
lease paths. Preparation, authorization, and execution are
separate lifecycle transitions; no scientific launch command is granted here.
Never force, reopen, or reactivate the exhausted v17/v18 identities.

Any eventual result remains `POST_HOC_CONSUMED_TEST_SENSITIVITY`,
`TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`, and `fresh_evidence=false`. It cannot
establish fresh confirmatory improvement, deployment safety, significance, or
unseen-center generalization, and cannot feed Stage 60/70 or another
experiment. Reliable real routing is an empirical admission outcome, not a
consequence of the new formula or the synthetic fixture.
