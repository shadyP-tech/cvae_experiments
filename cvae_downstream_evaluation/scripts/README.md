# Scripts

Scripts in this directory should stay thin. Put reusable logic under `src/cvae_downstream_evaluation/`.

Planned entrypoints:

- `run_direct_support_nelbo_downstream.py`: run the locked second-stage experiment.
- `build_downstream_decision_report.py`: consolidate artifacts into thesis-facing tables.
- `run_z11_ceiling_audit.py`: run the Z1.1 current-setup real-feature/PCA/C6.3 ceiling audit.

Scripts must write manifests before running expensive generation or classifier work.
