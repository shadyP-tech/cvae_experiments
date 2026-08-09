# MIDOG++ fixed-bank pooled-BACC case-OOF ceiling v2

Experiment:
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling.v2`

Status: implemented additional terminal consumed-test diagnostic. No canonical
result exists until its new closed-world workstation bundle completes and
validates. Regardless of the result, the decision is `DO_NOT_PROMOTE`.

## Why v2 exists

The first label-aware ceiling sealed all 729 probability cells, then failed
before decisions because it tried to define balanced accuracy independently
inside every case. That estimator is undefined for MIDOG++'s four
negative-only and one positive-only cases. Dropping those cases or assigning a
conventional per-case value would change the estimand after labels were seen.

V2 instead implements the explicitly authorized pooled exact-BACC estimand and
quarantines the entire v1 state. It does not resume v1 and does not read its
output, scratch, predictions, partial priors, label-access state, or
checkpoints. The original six inputs are rematerialized under new identities,
a new contract hash, a new scratch path, and a new hash-chained amendment.

## Frozen utility and uncertainty

For every `(H, case, action)`, v2 retains only four additive counts:
`n_positive`, `true_positive`, `n_negative`, and `true_negative`. One class
count may be zero; both may not be zero. For a legal set of cases `S`,

```text
U_S(a) = 0.5 * (sum TP / sum n_positive + sum TN / sum n_negative).
```

Both pooled denominators must be positive. This reproduces exact BACC computed
from all raw hard predictions in the scope while retaining all 218 whole cases:
213 mixed, four negative-only, and one positive-only. No per-case BACC is
stored or used.

The utility remains pooled over rows. Whole cases define the paired uncertainty
clusters. For a candidate-minus-reference contrast, the fixed influence for
case `c` is

```text
psi_c = 0.5 * [
  n_c+ / N+ * (case positive-accuracy difference - pooled positive difference)
  + n_c- / N- * (case negative-accuracy difference - pooled negative difference)
].
```

Only the term for an absent case class is omitted. With `m` support cases,
`Vf = max(m/(m-1) * sum psi_c^2, 1e-6)`.

## LOCO prior and fold decision

For target `H`, candidate source `e`, and legal donor center `H'` outside
`{H,e}`, the LOCO effect is `U_H'(e)-U_H'(B)`. The seven donor centers are
equally weighted. `G_H` is the candidate with maximum prior mean, with
lexicographic ties, only if its `1.96` lower bound versus `B` is strictly
positive; otherwise `G_H=B`.

Before any `H` support label opens, v2 also seals every alternative
candidate-versus-selected-`G_H` pairwise prior. It uses seven shared donors
when `G_H=B` and six when `G_H` is another source. For donor effects `d_j`,
`mu0=mean(d_j)` and `V0=max(sample_variance(d_j)/J, 1e-6)`.

The same-center, nonheld-fold support contrast is
`D=U_support(e)-U_support(G_H)`. The normal-normal update is

```text
Vpost = (1/V0 + 1/Vf)^-1
mupost = Vpost * (mu0/V0 + D/Vf)
LCB = mupost - 1.96 * sqrt(Vpost).
```

The router chooses the candidate with maximum LCB only when that LCB is
strictly positive; otherwise it abstains to `G_H`. Ties are lexicographic and
all thresholds are predeclared.

## Capability order and null

1. Materialize and seal all `B` plus eight legal `Hxe` probabilities, averaging
   the nine seed pairs before thresholding.
2. Open only other-center LOCO labels and seal all nine `G_H` choices and all
   pairwise priors before any same-`H` support access.
3. For each of 45 folds, open only nonheld same-center support labels, form the
   sufficient-statistic posterior, and compute the observed action plus 10,000
   fixed-null actions.
4. Seal all 45 observed and 450,000 null actions.
5. Only then open evaluation labels and compute pooled center BACC with
   equal-center inference.

Inside each `(H, fold, support case)` block, the null first orders the eight
candidates by SHA-256 over seed/fold/case/action. Each case and null index then
receives an independent counter-SplitMix64 cyclic shift drawn from the fixed
inclusive range `{1,...,7}`. Zero shift is forbidden, so complete candidate
sufficient-statistic blocks are deranged; `B` remains fixed, the eight-candidate
multiset is preserved, evaluation cases are never donors, and the exact same
pooled-BACC cluster posterior is recomputed. This is a restricted cyclic-shift
family and is explicitly not uniform over all possible derangements.

The permutation primary statistic is the equal-center mean `R-G_H`. With `K`
sealed null draws, `one_sided_p_value` is the upper tail
`(1 + #null >= observed)/(K + 1)`, `lower_tail_p_value` is
`(1 + #null <= observed)/(K + 1)`, and `two_sided_p_value` is
`min(1, 2*min(upper, lower))`. These formulas and the restricted null family
are fixed before any evaluation label is opened.

Normalized regret uses the predeclared no-opportunity convention: when oracle
headroom over `B` is at most `1e-12`, normalized regret is exactly `0.0`. Such
a fold has no available routing improvement to miss; the convention prevents
division by numerical zero and is fixed before evaluation labels are opened.

## Consumed-test and claim boundary

The new amendment is chained directly to the immutable original parent SHA-256
`8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16`
and whitelists only v2. Its six inputs are the frozen bank, GenerationLock,
dedicated cache and manifest aliases, dedicated parent-ledger alias, and the
new amendment. The cache alias explicitly retains the lineage of the
Stage-70-derived label-free feature cache. No Stage-50/60 result, Stage-70
prediction/scoring/policy output, prior Stage-90 output, or v1 state is
admissible.

This is a post-hoc information ceiling over already-consumed data, not a fresh
router. It cannot establish routing quality or target performance, update an
expert or shared model, authorize an action or policy, feed another stage or
experiment, select a recipe, support promotion, or support deployment.

## Workstation execution

The run uses one persistent worker on each RTX A5000 while the parent remains
CUDA-free, followed by four spawned CPU workers with three BLAS threads each.
GPU and CPU pools are phase-disjoint, arrays are float32 memmaps, reductions
are float64, and v2 checkpoints use
`/data/local/fixed_bank_pooled_bacc_case_oof_ceiling_v2`.

```bash
python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling.v2
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_fixed_bank_pooled_bacc_case_oof_ceiling/v2/
```
