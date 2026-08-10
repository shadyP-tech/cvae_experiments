# MIDOG++ Uniform-B v2 fixed-bank signed sample-level error gate

## Status and scope

The scientific core is implemented in
`src/midogpp_thesis/cvae/diagnostics/fixed_bank_signed_error_gate/`.

Its canonical mode is `EXPLORATORY_CONSUMED_DATA_ONLY`. Reusing the MIDOG++
test set can provide a post-hoc mechanism analysis only. It cannot authorize
routing, promotion, a policy update, another experiment, or deployment. A
fresh claim requires a new predeclared whole-case/patient/slide-disjoint
support/evaluation reservation with predictions sealed before label access.

The diagnostic is independently registered and has completed as
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_signed_error_gate.v1`.
It has its own direct-parent consumption-ledger amendment, signed-only cache and
manifest aliases, config contract, runner, and closed-world bundle validator;
it does not inherit or consume the hierarchical residual stacker's amendment or
output. The canonical workstation bundle is `COMPLETE` and independently
validates `PASS`.

All 45 folds selected final `lambda=0`. Twenty-four support proper-loss
proposals were already zero, and the 21 nonzero proposals failed the unchanged
exact-BACC lower-confidence gate. `B_cal`, `G`, `R_raw`, `R_safe`, and `P` are
therefore terminally identical at equal-center BACC `0.800935`; `B` is
`0.800896`. The `B_cal-B` contrast is `+0.0000397` with 95% interval
`[-0.003022,+0.003102]`. Nested MSE was `0.148202` for `R`, `0.147442` for
`P`, and `0.147427` for `G`; `R` was worse than `P` in all nine outer centers.
The result motivates an upstream actionability/recoverability audit, not more
gate capacity or a relaxed safety threshold.

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

The implementation uses the two A5000s only to generate the frozen source
streams. After the parent drops CUDA visibility, four spawned CPU workers with
three BLAS threads each materialize the 729 classifier probability cells and
fit the small signed models; GPU and CPU phases remain disjoint. The models run
on the Xeon W-2265; the shared runtime stores sealed probabilities as compressed
float32 NPZ and scientific reductions are float64. Each worker rebuilds and
hash-revalidates only its current target family's float64 outer/nested contexts;
there is no cross-target context cache, and probability input is bounded to four
concurrent process-local copies. The parent process must not retain a CUDA
context during CPU fitting.

The core is split into protocol, contracts, features, gradients, model,
composition, calibration, execution, sealing, terminal, and evaluation modules.
The terminal adapter verifies the partition, per-method prediction and decision
seals, label-capability report, and runtime bounds before emitting a provenance-
bound diagnostic envelope. A future fresh-data study may reuse the mathematics
only under a separate predeclared authorization; it cannot consume this output.

## Running the registered diagnostic

From the repository root on the canonical workstation and in the thesis Conda
environment:

```bash
python -m midogpp_thesis workspace run midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_signed_error_gate.v1
```

The workspace launcher resolves and records the exact six inputs before the
runner starts. The runner performs the two-GPU frozen-source phase, enters a
CUDA-free four-worker-by-three-thread probability/model phase, durably seals all nine LOCO
model families and 270 fold-method decisions, and only then opens terminal
labels. Its validator reloads the arrays, recomputes the exact-nine surface and
feature contexts, refits the signed models from legal donor labels, recomputes
all support decisions, reopens terminal labels only after resealing, recomputes
terminal metrics, and rejects any unexpected bundle member. Interrupted source
and prediction tasks resume only from hash-valid checkpoints; later phases are
deterministically replayed against immutable artifacts. A committed terminal
phase resumes through validation only, without reopening the scientific phases.
