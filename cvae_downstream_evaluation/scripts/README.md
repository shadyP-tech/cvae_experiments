# Scripts

Scripts in this directory should stay thin. Put reusable logic under `src/cvae_downstream_evaluation/`.

Planned entrypoints:

- `run_direct_support_nelbo_downstream.py`: run the locked second-stage experiment.
- `run_family_e1_direct_embedding_sampler_downstream.py`: run the non-CVAE Family E1 direct embedding sampler diagnostic.
- `build_downstream_decision_report.py`: consolidate artifacts into thesis-facing tables.

Scripts must write manifests before running expensive generation or classifier work.
