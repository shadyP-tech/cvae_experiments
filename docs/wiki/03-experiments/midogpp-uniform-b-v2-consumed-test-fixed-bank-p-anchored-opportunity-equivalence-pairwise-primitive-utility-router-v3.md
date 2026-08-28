# MIDOG++ Uniform-B v2 OE-PPUR v3 source-supervised successor

## Status and claim boundary

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router.v3`
- Stage: `90_oracles_and_diagnostics`
- Generic workspace registration: deliberately planned and non-runnable
- Publication label: `POST_HOC_CONSUMED_TEST_SENSITIVITY`
- Terminal decision: `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`
- Fresh evidence: `false`
- Scientific router implementation: present
- Source-supervision producer implementation: present
- Dedicated source/amendment/resolved-config executable: implemented
- Workstation source-supervision bundle: materialized only by the dedicated gate
- Execution amendment: issued only by the explicit single-use authorization gate

OE-PPUR v3 is a separate successor. It neither edits nor imports OE-PPUR v2,
and no v2 output, scratch directory, lease, probability surface, report, or
capability history is an input. The checked-in generic workspace registration
cannot prepare or launch it; the separately authorized executable owns the
three guarded workstation phases described below.

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

The two upstream byte pins name the direct inputs themselves: the Stage-30
bank content index is
`6b74fe794bd30cf6c1e42190427e506d1ff50ecd9280b9dcfee2a7592ec6a318`
and the Stage-40 GenerationLock content index is
`086eb106a11fd52df5fc1f692d17a33edccaf6f707b2b9dd0fba15895d891d86`.
The earlier v2 values pointed one producer stage upstream and are deliberately
not inherited by v3. The semantic bank and generation locks remain
`9972a41dcd4814cd` and `34e551425710362e`.

Input seven is never checked into the planned config. The dedicated publisher
creates its deterministic bytes once, after binding them to the parsed source
receipt, invariant scientific protocol hash, and a live content seal over the
dedicated executable plus every preparation module. It names the immutable parent
ledger, not the v3 resolution alias, authorizes exactly this one consumer and
one run, and records every prior Stage-90 reuse field as false.

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

Input three now has a dedicated, source-only producer. It validates the
canonical source cache and the immutable bank/GenerationLock, materializes
source streams with two persistent GPU workers (`cuda:0`, `cuda:1`), and runs
the exact 324 held-pair tasks with four CUDA-hidden `spawn` CPU workers and one
BLAS thread per worker. The producer computes 2,916 classifier fits, reduces
the complete 3-by-3 training/generation seed grid in float64, writes the 72
oriented held-center blocks as little-endian float32, and atomically publishes
and reparses the exact six-member source-supervision bundle. Source outcomes
remain capability-closed until every label-free probability block is sealed.

The terminal router uses the same two-GPU/four-CPU topology for its label-free
physical surface. Probability shards are little-endian float32 and read-only
after parsing; scientific reductions use float64. Worker payloads contain only
primitive pickle-safe DTOs. Nested process pools and terminal-run cross-run
recovery are forbidden. The pre-authorization source producer alone may resume
its own exact live-seal-bound checkpoints after interruption; it deletes that
work root after successful six-member publication. Before the irreversible
single-use lease, a read-only capacity gate
requires two sufficiently free A5000 GPUs, 32 GiB available RAM, and separately
budgeted artifact/scratch storage (or their combined budget on one filesystem).
The resolved config and lease are pinned to the sole catalog-canonical v3 output
root; no relocated config can mint another lease. Direct-input artifact scopes
must be disjoint from both output and scratch in either ancestor direction.

All 218 decisions and the parsed 9,928-by-7 probability matrix are persisted
together with no-follow exclusive writes and file, directory, and artifact-root
durability barriers. Two fresh processes independently revalidate those bytes
before the manifest-backed label reader can return aggregate-only diagnostics.
Aggregate receipt and evaluator construction is manager-owned and token-gated;
arbitrary reader subclasses or caller-constructed metrics cannot enter the
terminal lifecycle. A second two-process attestation validates the final
aggregate bytes. Completion then rehashes the terminal metrics, reconstructs
the typed receipts and lineage, checks the validation-index hashes and exact
catalog inventory, and binds the result to the monotone `COMPLETE` state and
the sibling single-use lease outcome.

The generic `workspace prepare/run` path remains blocked because it would
render the path-free planned config. The executable successor instead performs
the required lifecycle directly and in order:

```bash
cd /home/stud/spark/cvae_experiments
export PYTHONPATH="$PWD/src"
OE_PPUR_PY=/home/stud/spark/.venvs/cvae-breakhis/bin/python

"$OE_PPUR_PY" -m midogpp_thesis.oe_ppur_v3 materialize-source
"$OE_PPUR_PY" -m midogpp_thesis.oe_ppur_v3 authorize \
  --repository-root "$PWD"
"$OE_PPUR_PY" -m midogpp_thesis.oe_ppur_v3 run \
  --repository-root "$PWD"
```

The first command opens source-train outcomes only after all 72 label-free
blocks seal, atomically publishes six members, and reconstructively parses
them. The second command refuses any prior output, terminal scratch, lease, or
amendment state. Before issuance it mutation-freely rehashes the six existing
inputs, reconstructs the prospective authorized config, validates the live
lifecycle seal, parses and validates the exact prospective path-bearing
seven-input envelope, and projects receipt-derived `materialized=true` and
`authorized=true` facts for inputs three and seven. Only then does it issue
input seven once and publish, with a kernel-enforced no-replace directory
commit, only
`config.resolved.yaml` plus `provenance/input_artifacts.json`. It immediately
reparses both and performs the complete read-only seven-input admission. The
third command revalidates source, bank, GenerationLock, label-free test cache,
manifest and parent aliases, amendment, source seal, workstation topology, and
capacity before claiming the irreversible lease. Target labels remain closed
until the durable preterminal decision artifact has two independent fresh
attestations.

If the amendment was published but envelope rendering failed before the lease
was claimed, retry only the bounded preparation edge below. It validates the
existing amendment and any exact already-rendered envelope byte for byte; it
does not rewrite input seven, consume the lease, recover a terminal run, or open
labels.

```bash
"$OE_PPUR_PY" -m midogpp_thesis.oe_ppur_v3 render-existing \
  --repository-root "$PWD"
```

## Interpretation

Any later authorized result would still reuse the complete previously consumed
MIDOG++ test split. It can support only a terminal diagnostic of this frozen
router. It cannot support fresh routing, target-domain generalization,
significance, thesis-confirmatory evidence, Stage-60/70 promotion, deployment,
or another experiment.
