# Protocol Status

Last updated: 2026-07-23

## Active MIDOG++ surfaces

- Dataset contract and runtime validation:
  `src/midogpp_thesis/data/contract/`
- Real-feature controls and tuned classifier reference:
  `src/midogpp_thesis/real_features/`
- CVAE preservation surfaces:
  `src/midogpp_thesis/cvae/preservation/`
- Registry, artifact resolution, and fail-closed launch preparation:
  `src/midogpp_thesis/workspace/`

The active regime is MIDOG++ with Virchow2 embeddings, corrected `xyxy` patch
geometry, case-disjoint splits, and center held-out evaluation over eligible
centers `0,1,2,3,5,6,7,8,9`. Target-evaluation labels are scoring-only.

The working tree contains active implementations for:

- the eligible-only, predict-policy Stage-10 matched reference v2;
- the audit-blocked v1/v2 physical-multiscale histories and the pre-activation,
  non-adoptive v3 clipped-bbox annotation-local Stage-10 pilot;
- fully nested Stage-20 source-inner ex-post-prior and Task-Fisher
  `RecipeLock` selection;
- the bounded Stage-20 training-seed stability panel and fold-level consensus
  `RecipeLock` export;
- the separate non-adoptive learned class-conditional diagonal-prior
  source-inner v2 study;
- the separate non-adoptive Task-Fisher shrinkage source-inner v2 study;
- a conditionally unlocked Stage-20 outer A/B/C/D preservation factorial.

The first three runs are complete on `xai-master` and pass their full
validators. The Stage-10 v2 result is a matched
`real_feature_transfer_only` denominator. The scalar Stage-20 source-inner
result is a valid `cvae_recipe_lock_only` bundle with seven conditional locks
and standard-normal fallbacks for centers `5` and `9`. Its gate is
`NEGATIVE_GATE_COMPLETE` with `factorial_triggered=false`, so the registered
outer factorial is blocked and provides no outer-preservation result. The
bounded training-seed stability bundle is `COMPLETE` and `PUBLISHED`, with
`27/27` valid child locks, `9/9` export-ready consensus locks, and
`stage30_recipe_ready=true`. These workstation bundles have not yet been
synced locally, and their catalog output entries still carry stale
`TODO_VERIFY_ARTIFACT` lifecycle labels pending catalog promotion.

The two v2 mechanism studies are active implementations but remain unrun. They
fully cross training and generation seeds `17,42,101` over the nested nine-by-
eight center structure. Their only allowed claim scope is
`cvae_source_inner_study_only`: study decisions have recipe and deployable
consumption fixed false. They cannot change the scalar v1 gate, the published
stability consensus, outer v1, or the registered Stage-30 input.

## Current thesis-facing evidence

The tuned-classifier preservation result supports only
`cvae_preservation_only`: decoded means and posterior samples preserve most of
the frozen PCA128 real-feature classifier surface. It does not support routing,
expert selection, NELBO compatibility, prior generation, controllable
generation, or downstream synthetic utility. Prior sampling remains materially
weaker and is a separate bottleneck.

The training-seed panel is thesis-facing only for the narrow
`cvae_recipe_lock_only` stability claim. Exactly unanimous recipe selection is
limited to centers `6` and `7`; centers `0,1,2,3,5,8,9` are unstable under the
predeclared consensus rule. This is a `NEGATIVE_RESULT` for broad recipe
stability and an operational `PASS` for publishing a conservative complete
recipe bank. It is not outer-preservation, routing, generation, or downstream
utility evidence.

## Planned surfaces

Physical-multiscale v1 and v2 remain `planned` only as non-runnable
failed-audit histories. V3 is registered as `diagnostic`; its label-blind
`xai-master` source audit passes all 9,648 eligible rows and 216 TIFFs with 84
clipped bboxes, no row exclusion, and no synthesized pixels. Its immutable
contract and atomic B/C feature bundle are independently validated and have
complete required-file SHA-256 coverage. The Stage-10 run is authorized but
has not executed. All three lineages remain forbidden from Stage 20 through
Stage 70.

Stages 30 through 70 are not active implementations:

- provenance-clean expert bank
- prior and generation validation
- all-candidate utility matrix for a clean bank
- support-based routing or composition
- frozen-policy downstream synthetic utility

Legacy source trees were retired rather than exposed as fallbacks. Any new
implementation must use only the canonical dataset/cache lineage and must pass
target-leakage, expert-isolation, candidate-exclusion, and proxy-versus-utility
checks before activation.

The published training-seed consensus `RecipeLock` files are now eligible to
feed the fold-matched Stage-30 expert recipe. Consensus is `A` for centers
`0,1,2,3,5,9`, `D` with the full conditional sampler for centers `6,7`, and
`C` with the full conditional sampler for center `8`. The bundle's protocol
hash is `bbde3e5c5a1e3374`, and its selection-bundle hash is
`79cb9b614779c23b`. Full bundle, leakage, identity-overlap, and RNG validation
passes, and the Stage-30 loader accepts the publication.

Scalar seed-42 locks remain evidence for the completed source-inner result but
are no longer the registered Stage-30 input. Stage-30 consumption remains
fail-closed: the stability bundle must be `PUBLISHED`, and every consensus lock
must be valid and export-ready before the loader returns lock `H` for fold `H`.
The Stage-30 registry entry is still planned and has no runnable expert-bank
runner, so the eligible input does not activate Stage 30. Outer v1 still
consumes the scalar source-inner bundle and retains its original
all-conditional/factorial gate. Outer preservation metrics and decisions may
never feed model, sampler, expert, routing, or composition selection. Stage 40
remains a post-expert-bank generation-validation stage.

The v2 mechanism studies may run independently while Stage-30 implementation
proceeds; they are not prerequisites for it. A fully executed v2 bundle may
still report `INVALID_INCOMPLETE`, for example when a raw Fisher state is
invalid. In that case `run_state=COMPLETE` means execution and mechanical
bundle validation finished, not that a scientific mechanism decision passed.

## Quarantine and historical evidence

- `artifacts/midogpp/90_oracles_and_diagnostics/` contains rejected MIDOG++
  lineages and non-deployable diagnostics.
- `artifacts/cross_dataset_archive/` contains BreakHis, Camelyon17, and generic
  historical outputs outside the MIDOG++ registry.
- Historical paths embedded in immutable manifests remain provenance strings;
  they are not active resolution fallbacks.

The complete 22,569-patch contract, corrected feature cache, and two tuned
evidence bundles are present locally and on the workstation at their canonical
paths. Dataset validation passes, cache split counts align exactly, and
cataloged authoritative hashes match. The approximately 65 GB raw source tree
is workstation-only at `datasets/midogpp/raw/MIDOGpp/` by design.

The repository-migration audit is `AUDIT_ONLY` at
`artifacts/midogpp/90_oracles_and_diagnostics/repository_migration/2026-07-12_xai_master/`.
Its pre/post manifests match, and all retired workstation source artifact/data
locations are absent.

After installing this repository with
`conda run -n thesis python -m pip install -e .`, the canonical protocol check
is:

```bash
conda run -n thesis python -m midogpp_thesis workspace validate
```
