# Canonical MIDOG++ Artifacts

New MIDOG++ experiment outputs belong here. Relevant local historical bundles
were relocated into the appropriate stage or stage-90 quarantine with
pre/post content-fingerprint checks. Their embedded original paths and run
manifests remain unchanged as provenance. The artifact catalog is the
authority for current physical locations and expected hashes.

Use the layout:

```text
<stage>/<experiment_id>/<run_id>/
├── config.resolved.yaml
├── manifests/protocol_manifest.json
├── provenance/input_artifacts.json  # semantic identities + verified file hashes
├── reports/leakage_report.json
├── tables/
├── predictions/       # when applicable
└── checkpoints/       # when applicable
```

The stage names match `experiments/midogpp/registry.yaml`. A run's manifest and
reports, not its directory name, are the authority for claim eligibility.

Generated artifacts are intentionally ignored by Git. Sync selected run
bundles from the workstation without broad checkpoint or embedding trees.

The validated tuned real-feature and CVAE preservation bundles are present
locally and on the workstation at their cataloged canonical paths. Their
authoritative file hashes are pinned in the artifact catalog. The stage-90
repository-migration audit preserves matching pre/post manifests for those
bundles, the corrected cache, the complete contract, and critical raw-data
metadata.
