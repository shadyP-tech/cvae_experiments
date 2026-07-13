# Protocol Status

Last updated: 2026-07-13

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

The working tree now contains active but unrun implementations for:

- the eligible-only, predict-policy Stage-10 matched reference v2;
- fully nested Stage-20 source-inner ex-post-prior and Task-Fisher
  `RecipeLock` selection;
- a conditionally unlocked Stage-20 outer A/B/C/D preservation factorial.

Their catalog entries remain `TODO_VERIFY_ARTIFACT`, and their canonical output
roots are absent. Therefore they provide no new metric, decision, or thesis
result yet.

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

Validated source-inner `RecipeLock` files may later feed the Stage-30 expert
recipe. Outer preservation metrics and decisions may never feed model, sampler,
expert, routing, or composition selection. Stage 40 remains a post-expert-bank
generation-validation stage; the new Stage-20 prior-recovery surface does not
activate it.

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
