# MIDOG++ Uniform-B v2 OE-PPUR v4 workspace-sealed successor

## Status and claim boundary

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router.v4`
- Stage: `90_oracles_and_diagnostics`
- Workspace registration: planned and deliberately non-runnable
- Publication label: `POST_HOC_CONSUMED_TEST_SENSITIVITY`
- Terminal decision: `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`
- Fresh evidence: `false`
- Execution amendment: fresh v4 preflight and publication required
- Launch authority: absent
- Execution launched by this successor authorization: no
- FUSE required or activated: no
- Current implementation scope: preparation only
- Scientific execution adapter: not implemented or runnable

OE-PPUR v4 is a lifecycle successor that closes the workspace-binding and NFS
publication gaps found after the v3 amendment was issued. It does not revise or
consume OE-PPUR v3 amendment #7. That amendment remains byte-exact at SHA-256
`56269322ead01ef683c985d8f295b0369fb35ddef04d12115704f1df18a0c425`
and its preserved state is `issued/unrendered/unclaimed/no-run`. It is recorded
only as a preservation witness; it cannot authorize v4 and no v3 output,
envelope, lease, scratch, run state, report, probability surface, or label
capability is inherited.

The v4 implementation and any later result remain a post-hoc diagnostic over
the already consumed complete MIDOG++ test split. Neither a successful
preflight nor a sealed amendment is routing evidence. A later separately
authorized terminal run cannot establish fresh routing, unseen-center
generalization, downstream CVAE utility, NELBO compatibility, statistical
significance, deployment fitness, promotion, or thesis-confirmatory evidence.

## Exact seven-input contract

The ordered direct inputs are:

1. the immutable Uniform-B v2 source expert bank;
2. the immutable GenerationLock;
3. the v4-fenced immutable source-only action-supervision alias;
4. the v4-fenced label-free MIDOG++ test-cache alias;
5. the v4-fenced canonical test-manifest alias;
6. the v4-fenced byte-exact original parent-ledger alias;
7. one newly issued v4 workspace-sealed amendment.

Input three points directly to the existing immutable source-only bundle at
`artifacts/midogpp/90_oracles_and_diagnostics/oe_ppur_source_training_action_supervision/v3`.
This is the explicit, user-authorized v4-only hash-exact source-content reuse
exception. It acknowledges rather than rewrites the v3 artifact's no-feed
fence, is limited to this registered v4 consumer, and is content reuse rather
than authority reuse. Admission binds all six members:

| Member | SHA-256 |
|---|---|
| `manifests/source_training_surface.json` | `2313db90779d1b509db620faa5425ddad2a2e0824c1d709a3489ce7f7f99294b` |
| `manifests/source_pool_lineage.json` | `c3599f8f56c89382494a19c019432dee5a8dc12d45c638a5f8388875c658edf5` |
| `tables/source_rows.csv` | `a324215960961074d924d5b67198263b5afdc906b6800eb96835b448d5d45a31` |
| `arrays/source_action_probabilities.npy` | `979d7575ef933bb4b208ce58ca469a88d8861d23fb9bcb682cbe7a6b7f4fb649` |
| `manifests/content_index.json` | `1cb9c1a2b548b7b31250b57b5be4a9870ef97ce299877a54c8de6780898f4d5f` |
| `reports/validation_report.json` | `881377105eb62cd09c2a17aa27cdeb1ab59e01e57b4a2af44672b54fab44b71a` |

It also binds the source receipt
`e9b6a05a6f4a4c982ce51eff7606b4ca303fd4c9df7dc466a6a3cc88fd93fe66`,
surface
`51084af5dfdf9ab7a34ccac2524b664c3df0860bdf02589b55c6d94310f968dd`,
row order
`73bd8ade9944cbcf2e2dd9c4a8f4f247190ca68e880819a1f810a78ac64ae9bb`,
producer seal
`74bf5b5c01d50190a6a0639533f298cea7ece8d381fdbbd578fa82537c48ab91`,
and independent recomputation receipt
`e6b641a605ecd85774ef7a6ad06c1f47c0abe68ecf6448f1aa3f0b4eee353241`.
The source bundle contains no target-test rows or labels.

The planned path-free config contract is
`8739d9f044a8eaaba960ea6d0cc750126de20d2a21cd9061a5f3601c714aa8bb`,
the scientific protocol contract is
`2bb395c7d851c6b4169cbada998bc8928979d4c28b452fe554db9e5494f57ac8`,
and the planned seven-input contract is
`2e0d7011ba3d9b1e2821d657aa6f9bc2cae7c5210434f2bcf51cce6b96093d5c`.

## Workspace-sealed authorization

V4 uses a two-level, non-circular commitment:

1. A mutation-free preflight seals the Git `HEAD`, `HEAD^{tree}`, Git index
   bytes, exact status/dirty-file bytes, allowlisted implementation and
   registration files, registry, catalog, planned config, helper, existing six
   inputs, resolved canonical paths, scientific and lifecycle seals, and the
   prospective NFS-safe publication topology. The prospective config and input
   manifest use an amendment-digest placeholder whose meaning is fixed by the
   plan.
2. The v4 amendment binds that pre-amendment plan. The final authorization
   envelope then embeds the complete plan payload and binds both the plan and
   the actual amendment digest. The durable preflight JSON also embeds that
   complete plan payload, so dirty allowlisted bytes remain independently
   auditable rather than becoming a hash-only commitment. The plan is replayed
   immediately before publication, after publication, and again before any
   future input opening or lease claim.

Any Git, index, dirty-content, path, input, source, config-template,
manifest-template, helper, topology, or lifecycle drift fails closed. A fresh
preflight must start from absent v4 amendment, output, lease, and scratch
surfaces. Preparation may publish only the v4 amendment and its validation
receipt. It must not create the run root, acquire the single-use lease, start
workers, open target labels, or interpret successful construction as launch
authority.

## NFS-safe workstation publication

V4 replaces the directory `RENAME_NOREPLACE` assumption with
`NFS_SAFE_IN_PLACE_COMMIT`:

1. acquire the absent canonical final root exclusively;
2. create every member with no-follow `O_EXCL` semantics;
3. fsync member files and their parent directories;
4. validate the exact envelope bytes in place;
5. write and fsync `COMMITTED` last.

An output without the final marker is incomplete and inadmissible. Existing
roots are never recovered or overwritten. The run scratch remains host-local,
cross-run recovery is forbidden, and FUSE is not part of the v4 topology.

The workstation compute topology remains two persistent RTX A5000 prediction
workers plus four CUDA-hidden `spawn` CPU workers, one BLAS thread per worker,
primitive pickle-safe process DTOs, little-endian float32 prediction transport,
float64 scientific reductions, and no nested pools.

## No-launch boundary

The implementation can be inspected without mutation:

```bash
cd /home/stud/spark/cvae_experiments
export PYTHONPATH="$PWD/src"
OE_PPUR_PY=/home/stud/spark/.venvs/cvae-breakhis/bin/python

"$OE_PPUR_PY" -m midogpp_thesis.oe_ppur_v4 inspect
```

Preflight and amendment publication are preparation-only operations. They do
not authorize the `run` subcommand. The scientific execution adapter is not yet
implemented or runnable, so sealing the fresh amendment alone still cannot
create a run edge. Any future run edge requires a separately reviewed adapter,
fresh replay validation, and explicit launch authorization after the preflight
and amendment have both sealed. No launch command is documented under the
current authorization.
