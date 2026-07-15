# Protocol Status

Last updated: 2026-07-15

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
- fully nested Stage-20 source-inner ex-post-prior and Task-Fisher
  `RecipeLock` selection;
- the bounded Stage-20 training-seed stability panel and fold-level consensus
  `RecipeLock` export;
- a conditionally unlocked Stage-20 outer A/B/C/D preservation factorial.

The first two runs are complete on `xai-master` and pass their full validators.
The Stage-10 v2 result is a matched `real_feature_transfer_only` denominator.
The Stage-20 source-inner result is a valid `cvae_recipe_lock_only` bundle with
seven conditional locks and standard-normal fallbacks for centers `5` and `9`.
Its gate is `NEGATIVE_GATE_COMPLETE` with `factorial_triggered=false`, so the
registered outer factorial is blocked and provides no outer-preservation
result. The new bundles have not yet been synced locally, and their catalog
output entries still carry `TODO_VERIFY_ARTIFACT` lifecycle labels.

## Current thesis-facing evidence

The tuned-classifier preservation result supports only
`cvae_preservation_only`: decoded means and posterior samples preserve most of
the frozen PCA128 real-feature classifier surface. It does not support routing,
expert selection, NELBO compatibility, prior generation, controllable
generation, or downstream synthetic utility. Prior sampling remains materially
weaker and is a separate bottleneck.

## Planned surfaces

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

Validated training-seed consensus `RecipeLock` files may later feed the
fold-matched Stage-30 expert recipe. Scalar seed-42 locks remain evidence for
the completed source-inner result but are no longer the registered Stage-30
input. Stage-30 consumption is fail-closed: the stability bundle must have
`reports/publication_state.json` status `PUBLISHED`, and every consensus lock
must be valid and export-ready before the loader returns lock `H` for fold `H`.
Outer v1 still consumes the scalar source-inner bundle and retains its original
all-conditional/factorial gate. Outer preservation metrics and decisions may
never feed model, sampler, expert, routing, or composition selection. Stage 40
remains a post-expert-bank generation-validation stage; the new Stage-20
prior-recovery surface does not activate it.

Before Stage 30, run at most one bounded source-inner training-seed stability
check over training seeds `17,42,101` and generation seeds `17,42,101`. This
check is implemented and registered as
`midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1`, but it
has not yet produced a production artifact. Its fully crossed training and
generation seeds, seed-free `(H,I)` preparation, shared Task-Fisher state,
distinct training RNG identities, generation-seed-paired posterior/prior noise,
and cross-seed rule are frozen in the config and validator. The RNG identities
are persisted in `tables/rng_pairing_audit.csv`; the catalog destination
remains `TODO_VERIFY_ARTIFACT` until the full bundle is run, validated, and
published.

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
