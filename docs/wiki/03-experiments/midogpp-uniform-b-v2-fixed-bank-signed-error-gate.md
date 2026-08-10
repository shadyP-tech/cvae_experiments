# MIDOG++ Uniform-B v2 fixed-bank signed sample-level error gate

## Status and scope

The scientific core is implemented in
`src/midogpp_thesis/cvae/diagnostics/fixed_bank_signed_error_gate/`.

Its canonical mode is `EXPLORATORY_CONSUMED_DATA_ONLY`. Reusing the MIDOG++
test set can provide a post-hoc mechanism analysis only. It cannot authorize
routing, promotion, a policy update, another experiment, or deployment. A
fresh claim requires a new predeclared whole-case/patient/slide-disjoint
support/evaluation reservation with predictions sealed before label access.

This is an unrun scientific core. It has no registry activation, consumption-
ledger amendment, runnable config, or canonical artifact bundle, and it cannot
inherit the hierarchical residual stacker's consumed-test authorization.

## Why the class branch was removed

The completed hierarchical residual stacker learned the donor smooth-response
surface, but all 45 support optimizers selected `lambda=0`. Oracle-only probes
showed that the `B_cal`-weighted positive/negative branch behaved like
confirmation feedback. The successor therefore predicts one signed correction;
it never uses the baseline predicted class to select a direction.

## Label-blind sample features

For every target sample, the fixed non-target bank is reduced to:

- absolute baseline logit margin;
- signed, absolute, and standard-deviation summaries of candidate-minus-`B`
  residual logits;
- positive and negative residual mass;
- hard disagreement rate and candidate-probability dispersion;
- residual/disagreement interactions with a fixed near-threshold envelope.

There is no candidate selection, target expert, metadata feature, label feature,
or pseudo-class branch. `G` is an intercept-only case-independent control. `P`
deterministically deranges complete sample-feature blocks within each center and
is fitted separately with the same capacity.

## Strict OOF signed target

For donor sample `i`, the response is the class-balanced negative log-loss
gradient with respect to the baseline logit:

```text
g_i = w(y_i) * (y_i - p_B,i)
```

For outer target `H`, all `H` labels are absent from fitting,
standardization, alpha selection, and uncertainty estimation. Ridge alpha is
selected on `[0.1, 1, 10]` by nested held-query-center prediction error; each
nested model excludes both `H` and that query center.

The final correction is clipped to `+-2` logits. Nested donor-center fits give
a directional uncertainty estimate. `R_raw` retains the direct correction;
`R_safe` is zero unless all nested fits agree on direction and
`abs(delta)-1.96*SE(delta)>0`. The two surfaces receive different content
hashes before same-target support access.

## Baseline anchoring and support decision

Support selects `b` from `[-0.1,-0.05,0,0.05,0.1]` and lambda from
`[0,0.05,0.1,0.2,0.25]` by fixed class-balanced log loss. Composition is:

```text
p = sigmoid(logit(p_B) + b + lambda * m(p_Bcal) * delta)
m(p) = exp(-(abs(logit(p))/1.0)^2)
```

Thus lambda zero is bit-exact `B_cal`, and corrections concentrate near the
decision threshold. Every lambda row records support loss, deterioration versus
zero, and threshold-crossing count. The existing paired whole-case exact-BACC
LCB must still be strictly positive; otherwise the decision falls back to zero
with an explicit reason. The LCB is not relaxed.

## Workstation profile

The implementation keeps the existing two-A5000 exact-nine probability phase
and reuses that sealed in-run surface. GPU and CPU phases remain disjoint. The
small signed models run as four spawned CPU workers with three BLAS threads each
on the Xeon W-2265; sealed probability storage is float32 and scientific
reductions are float64. Context features are streamed and hash-revalidated per
target instead of retaining all 81 contexts; the process-local probability
input is bounded to four concurrent worker copies. The parent process must not
retain a CUDA context during CPU fitting.

The core is split into protocol, contracts, features, gradients, model,
composition, calibration, execution, sealing, terminal, and evaluation modules.
The terminal adapter verifies the partition, per-method prediction and decision
seals, label-capability report, and runtime bounds before emitting a provenance-
bound diagnostic envelope. The same mathematics can later be bound to a fresh-
data runner without changing the consumed-test claim boundary.
