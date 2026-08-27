# MIDOG++ Uniform-B v2 OE-PPUR executable-successor mechanics

## Status

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router.v2`
- Stage: `90_oracles_and_diagnostics`
- Registration: planned and non-runnable
- Publication label: `POST_HOC_CONSUMED_TEST_SENSITIVITY`
- Terminal decision: `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`
- Fresh evidence: `false`
- Canonical scientific service implemented: `false`
- Canonical terminal evaluator implemented: `false`

No amendment has been issued and no run has been authorized or launched. The
implementation supplies the successor's fail-closed execution mechanics; it
does not yet supply a scientific routing result.

## Exact six-input admission

An authorized successor must receive exactly these ordered, immutable inputs:

1. promoted Uniform-B v2 source expert bank;
2. frozen GenerationLock;
3. v2-fenced label-free MIDOG++ test cache;
4. v2-fenced canonical test manifest;
5. byte-exact original consumed-test parent ledger;
6. one v2-only single-consumer authorization amendment.

Admission validates complete bank and GenerationLock content indexes, pins the
index-file bytes, verifies the cache content and row-order identities, and
binds the exact ordered roles, artifact IDs, canonical paths, kinds, content
hashes, output root, and scratch root. Predecessor Stage-90 state and path
overlap are rejected. The read-only admission precedes an external durable
single-use lease; any later failure is permanently `FAILED_EXHAUSTED`.

## Parsed probability surface

The only accepted surface is an exact `9,928 x 7` little-endian float32,
C-order matrix with columns:

```text
P_PROTECTED
B::zero_to_one  B::one_to_zero
I::zero_to_one  I::one_to_zero
R::zero_to_one  R::one_to_zero
```

The parser derives extents from the actual bytes, requires complete ordered row
coverage, finite values in `[0,1]`, canonical cache/manifest row lineage, and
exact GPU batch, worker, result-file, and surface hashes. Shards must be strict
descendants of the admitted scratch root and are opened through no-follow,
directory-relative descriptors with identity revalidation. Matrix, shard, and
per-column hashes are derived from parsed bytes rather than worker declarations.

## Preterminal and terminal boundaries

The guarded preterminal ledger contains exactly 218 case decisions and binds
each action to its matrix column, outer-center result, candidate pool, pairwise
model, calibration, opportunity receipt, and ranking policy. Two independent
artifact-only fresh processes must validate the frozen artifact. Only then may
a one-shot aggregate-only terminal capability exist; it requires all 218 cases
and cannot persist raw, row-level, or case-level labels.

The neutral pairwise primitive-utility core is implemented, but the source-only
workspace adapter that materializes expert probabilities and runs the full
`H/J/K/L/d` orchestration is not. A separate post-attestation manifest reader
and aggregate-only terminal evaluator are also still absent. The nominal
factory therefore rejects before the authorization lease. No workstation run
command is valid in the current registration.

## Interpretation boundary

Even a future separately authorized completion would reuse the already
consumed MIDOG++ test set. It could support only terminal post-hoc diagnostic
analysis of this fixed method. It cannot establish fresh routing, downstream
utility, CVAE or NELBO compatibility, significance, promotion, deployment, or
transferable thesis evidence.
