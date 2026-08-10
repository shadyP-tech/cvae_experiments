# MIDOG++ Uniform-B v2 consumed-test fixed-bank hierarchical residual stacker

## Status

Experiment ID:
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker.v1`

Implementation status: registered and locally contract-tested. No scientific
result exists until the workstation produces a complete closed-world bundle
whose independent validation report is `PASS`.

Evidence status: `EXPLORATORY_CONSUMED_DATA_ONLY`, `DO_NOT_PROMOTE`.

Claim role:
`known_fixed_bank_label_aware_case_oof_stacking_mechanism_diagnostic`.

This is a post-hoc terminal mechanism diagnostic over the already-consumed
MIDOG++ test split. It can neither establish fresh routing quality nor authorize
an action, policy, model update, later experiment, recipe, promotion, or
deployment.

## Why this architecture is being tested

The completed pooled-BACC case-OOF ceiling found retrospective fixed-bank
headroom but almost no reliable hard routing. `B` and the global prior both
scored `0.800896`; the fold router scored `0.801230` and routed only four of
218 cases. Candidate probabilities were nearly identical and their useful
near-threshold corrections were case- and class-direction dependent.

The new diagnostic therefore changes the decision geometry, not the expert
bank. It keeps `B` as an exact anchor and tests whether a small, case-conditional
convex residual composition can exploit sparse corrections while exposing
calibration-only, case-independent, and feature-alignment controls.

## Six-input firewall

The only inputs are:

1. the routing-authorized fixed expert bank;
2. the immutable GenerationLock;
3. a dedicated label-free test-cache alias;
4. a dedicated capability-gated test-manifest alias;
5. a byte-exact alias of the original test-consumption ledger; and
6. a direct-to-original-ledger, hash-chained, single-consumer amendment.

No metadata artifact is an input. No Stage-50/60 result, Stage-70 prediction,
scoring, or policy result, prior Stage-90 output, prior prediction array,
scratch file, checkpoint, or label-capability state may enter the run. The
amendment SHA-256 is
`e915134fc15901f1d5c43fb5fb974f1693282ca4622a2ade169eaa7487566b1b`.

## Probability and feature surface

The run recomputes all 729 exact-nine `B/Hxe` probability cells and seals them
before any label access. With `eps=1e-4`, the row residual for candidate `e` is

```text
r_e = logit(clip(p_Hxe, eps, 1-eps))
      - logit(clip(p_B, eps, 1-eps)).
```

For each whole case and candidate, the four local features are signed residual
mean, absolute residual mean, population standard deviation, and hard
disagreement rate against `B`. A probability-only global source descriptor is
the equal-legal-query mean of equal-case mean absolute residual logit. It never
uses labels or metadata.

The ten-column model vector is the intercept, four local features, the global
source descriptor, and its four interactions with the local features. All
non-intercept columns are standardized on legal donor training cases only;
zero-variance columns map to zero.

## Strict H/q/e fit

For deployed pair `(H,e)`, a final training row `(query=t, candidate=s)` is
legal only when both `t` and `s` are outside `{H,e}` and `t != s`. In nested
held-query validation for `q`, both roles must additionally exclude `q`.
The source descriptor for a training source `s` uses probability donors outside
`{H,e,s}`, and additionally outside `q` in the nested fold.

Separate positive- and negative-class ridge models predict smooth case effects.
The positive response is the case mean difference of
`sigmoid((p-0.5)/0.05)`; the negative response uses
`sigmoid((0.5-p)/0.05)`. A missing case class omits only that response. Ridge
alpha is selected from `[0.1, 1.0, 10.0]` by nested legal-query,
class-count-weighted response squared error, with larger alpha winning ties.
Target-center labels never select alpha, features, or rank. V1 freezes one
probability-derived source descriptor and includes no learned source identity
factor or rank-two challenger.

## Stacker and controls

The diagnostic methods are `B`, `B_cal`, `G`, `R`, and `P`.

- `B_cal` adds a support-only intercept selected by fixed class-balanced log
  loss.
- `G` is a case-independent stack using only intercept and the probability-only
  source descriptor.
- `R` uses the complete ten-column case-conditional model.
- `P` deterministically permutes complete four-feature case blocks before both
  donor fitting and target inference, preserves `g`, residuals, responses, and
  labels, and refits a separate model with the same capacity.

For each class direction, only the two largest strictly positive source scores
are admitted and receive a softmax with temperature `0.01`. The class branches
are combined softly:

```text
delta = (1-p_Bcal) * sum_e alpha_0,e r_e
      + p_Bcal     * sum_e alpha_1,e r_e

p_M = sigmoid(logit(p_B) + b_H + lambda_H * delta).
```

The soft gate avoids a hard pseudo-class sign reversal close to threshold.
`lambda=0` returns exactly to `B_cal`.

## Whole-case support and evaluation

Each target center has five deterministic whole-case folds. Four folds are
support and the fifth is evaluation; every one of the 218 cases is evaluated
once. Support first selects `b` from
`[-0.1,-0.05,0,0.05,0.1]`, then an `R` lambda from
`[0,0.05,0.1,0.2,0.25]`, using fixed class-balanced log loss. The same `b` and
lambda are applied to `G`, `R`, and `P` for matched controls.

The selected nonzero lambda is admitted only when the paired whole-case
cluster lower confidence bound for support exact pooled-BACC `R-B_cal` is
strictly positive. Otherwise lambda is zero. Exact BACC is never defined per
case; the four single-class-negative cases and one single-class-positive case
contribute their available confusion counts to the legal pooled scope.

The terminal primary contrasts are `R-B_cal`, `R-G`, and `R-P`, aggregated
equally over the nine centers. A 10,000-replicate case-cluster bootstrap
resamples whole cases within center and is explicitly conditional on the nine
observed centers. Passing a descriptive screening bound would not override the
terminal consumed-data claim boundary.

## Workstation execution

The frozen profile targets the Xeon W-2265 workstation with 125 GB RAM and two
RTX A5000 24 GB GPUs. Two persistent GPU workers materialize the probability
surface once. GPU and CPU pools are phase-disjoint. Four spawned CPU workers
with three threads each perform vectorized float64 model reductions and the
case-cluster bootstrap over shared float32 memmaps. Parent resident memory is
capped at 48 GiB, and resume accepts only hash-validated atomic checkpoints in
the new scratch namespace.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_fixed_bank_hierarchical_residual_stacker/v1/
```

Frozen config contract hash: `cb7050fcdaac86ac`.
