# MIDOG++ Thesis Experiments

This checkout has one active Python package and one canonical MIDOG++
experiment workspace. Historical capability-package roots have been retired so
dataset ownership, experiment configuration, implementation, and evidence no
longer overlap.

```text
src/midogpp_thesis/        reusable dataset, real-feature, CVAE, and workspace code
tests/                     package and protocol tests
datasets/midogpp/          raw-data contract, frozen contract, and derived features
experiments/midogpp/       staged configs, registry, catalog, and protocol defaults
artifacts/midogpp/         canonical MIDOG++ evidence and run destinations
artifacts/cross_dataset_archive/
                           non-MIDOG++ historical evidence, outside the live registry
docs/                      thesis context and evidence-backed interpretation
```

The evidence firewall is ordered as real-feature reference, CVAE preservation,
expert bank, prior/generation, all-candidate diagnostics, routing/composition,
and frozen-policy downstream utility. Passing an earlier stage never implies a
later-stage claim.

Install the package into the thesis environment once:

```bash
conda run -n thesis python -m pip install -e '.[cache,dataset-full]'
```

Then run the workspace from the repository root:

```bash
conda run -n thesis python -m midogpp_thesis workspace validate
conda run -n thesis python -m midogpp_thesis workspace list
conda run -n thesis python -m midogpp_thesis workspace prepare \
  midogpp.cvae.tuned_classifier_preservation.v1
```

The complete 22,569-patch contract, corrected `xyxy` cache, tuned real-feature
reference, and tuned CVAE preservation bundle are present locally and on the
workstation at their canonical paths. Contract validation passes, cache split
counts align exactly, and cataloged authoritative hashes match. The historical
train-only cache is never substituted for the corrected cache.

The approximately 65 GB raw MIDOG++ source tree is intentionally workstation
only at `datasets/midogpp/raw/MIDOGpp/`; it is not required for reruns that
consume the frozen contract and existing feature cache. The repository
migration audit is under
`artifacts/midogpp/90_oracles_and_diagnostics/repository_migration/2026-07-12_xai_master/`.

Run the local verification suite with:

```bash
conda run -n thesis python -m pytest -q
```
