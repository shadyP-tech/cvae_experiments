# Artifacts

Generated artifacts for local downstream evaluation development belong here.

Tracked placeholders keep the intended layout visible:

- `manifests/`: protocol, split, generated embedding, and provenance manifests
- `reports/`: leakage reports and decision summaries
- `tables/`: downstream matrices, routing alignment, baseline comparison, support-size stratification, and stability tables
- `plots/`: optional diagnostic plots

Large or run-specific outputs are ignored by default.

Before full v1 runs, sync these consumer-only manifests into `manifests/`:

- `expert_checkpoints.csv`
- `embedding_cache_manifest.csv`
- `expert_provenance.csv`

The downstream pipeline must fail clearly if these are missing. It must not
retrain experts or regenerate source artifacts.
