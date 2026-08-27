# MIDOG++ Uniform-B v2 OE-PPUR v3 source-supervised successor

## Status and claim boundary

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router.v3`
- Stage: `90_oracles_and_diagnostics`
- Registration: planned and non-runnable
- Publication label: `POST_HOC_CONSUMED_TEST_SENSITIVITY`
- Terminal decision: `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`
- Fresh evidence: `false`
- Scientific router implementation: present
- Source-supervision bundle: not materialized
- Execution amendment: absent and not issued

OE-PPUR v3 is a separate successor. It neither edits nor imports OE-PPUR v2,
and no v2 output, scratch directory, lease, probability surface, report, or
capability history is an input. Implementation does not authorize or launch a
consumed-test run.

## Exact seven-input contract

The future execution identity accepts exactly these ordered inputs:

1. the immutable Uniform-B v2 source expert bank;
2. the immutable GenerationLock;
3. one v3-fenced source-only action-supervision bundle;
4. one v3-fenced label-free MIDOG++ test-cache alias;
5. one v3-fenced canonical test-manifest alias;
6. one v3-fenced byte-exact original parent-ledger alias;
7. one future v3-only single-consumer execution amendment.

The third input is the scientific addition. It must contain the immutable
source row inventory, source outcomes, compiled action probabilities, exact
held-center candidate-pool lineage, representation and row-order lineage, and
a validation receipt. It must attest `target_rows_present=false` and
`target_labels_used=false`. It cannot resolve through a previous Stage-90
router or through the previously policy-fenced Stage-60 source-inner artifact.

The checked-in artifact identity for input seven is only a placeholder. No
amendment file or hash is present.

## Pool-indexed source supervision

For an outer target center `H` and a held source center `q`, every supervised
source probability surface is compiled from exactly `C ∖ {H,q}`. The final
label-free target surface is compiled from exactly `C ∖ {H}`. A final-H
pool is never substituted for a held-source pool.

"Pool invariant" refers to the compiler rule and its provenance, not to
identical numerical outputs. The rule is deterministic, label-free, and
permutation-equivariant over eligible experts:

- protected `P` is a fixed blend of equal-union `B` and pooled-union `U`;
- `B` uses the equal-union surface;
- `I` uses the row-wise eligible-expert maximum for `zero_to_one` and minimum
  for `one_to_zero`;
- `R` uses the row-wise median of the pooled and eligible-expert surfaces;
- every directional candidate changes `P` only across its declared 0.5
  boundary; all other rows remain exactly `P`.

The resulting columns are the fixed little-endian float32 inventory
`P_PROTECTED`, `B/I/R::zero_to_one`, and `B/I/R::one_to_zero`. Structural
no-ops are excluded from learning, and any incomplete, non-finite, misbound,
or unsupported state selects exact `P`.

## Scientific fit and terminal firewall

For each `H`, the source-only fit uses explicit, distinct `H/J/K/L/d` roles.
The held case `d` belongs to query center `J`; `H`, `J`, `K`, `L`, and `d` are
excluded from estimator fitting according to their roles. Complete rotating
`K` folds choose the fixed low-capacity ridge setting. Rotating held-`L`
out-of-fold residuals calibrate action- and contrast-specific uncertainty.
The final source posterior is refit on all legal `C ∖ {H}` source rows only.

Target labels stay closed while all 218 decisions are computed and sealed.
Only an exact nominal terminal evaluator may receive a one-shot capability
after a complete typed preterminal attestation. It returns aggregate BACC,
Brier, and log-loss diagnostics and cannot persist row labels or case labels.
These metrics do not establish CVAE or NELBO compatibility.

## Workstation execution design

The implementation fixes two persistent GPU workers (`cuda:0`, `cuda:1`) and
four CUDA-hidden `spawn` CPU workers with one BLAS thread each. Probability
shards are little-endian float32 and read-only after parsing; scientific
reductions use float64. Worker payloads contain only primitive pickle-safe
DTOs. Nested process pools and cross-run recovery are forbidden.

There is deliberately no valid run command yet. The planned runner and CLI
inspection path reject before resolving inputs, creating output or scratch,
claiming a lease, opening labels, or starting workers. A later request must
materialize and hash-validate the source-supervision bundle and separately
issue the v3 amendment before an executable resolved config can exist.

## Interpretation

Any later authorized result would still reuse the complete previously consumed
MIDOG++ test split. It can support only a terminal diagnostic of this frozen
router. It cannot support fresh routing, target-domain generalization,
significance, thesis-confirmatory evidence, Stage-60/70 promotion, deployment,
or another experiment.
