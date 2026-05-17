# Thesis Alignment

This skeleton maps the next implementation work to the thesis outline.

## Objective 1: Rigorous Evaluation Framework

Folder responsibility:

- `configs/`: locked experiment settings
- `docs/protocol.md`: split, leakage, and metric contract
- `tests/`: protocol invariants
- `artifacts/manifests/`: split, provenance, and run manifests

Expected evidence:

- support/evaluation disjointness
- target expert exclusion
- seed/fold coverage
- reproducible config hash or manifest

## Objective 2: Conditioned Generative Models In Feature Space

Folder responsibility:

- `src/cvae_downstream_evaluation/generation.py`

Expected evidence:

- generated object is a foundation-model embedding
- class prior is declared before evaluation
- decoder checkpoints are frozen
- synthetic budget and sampling temperature are locked

## Objective 3: Routing Mechanism

Folder responsibility:

- `src/cvae_downstream_evaluation/routing.py`

Expected evidence:

- support-NELBO is labeled as a direct utility estimate
- metadata routing remains a serious baseline
- random and ensemble baselines are retained
- downstream oracle is diagnostic only

## Objective 4: MoErging Generator

Folder responsibility:

- `src/cvae_downstream_evaluation/generation.py`
- `src/cvae_downstream_evaluation/schemas.py`

Expected evidence:

- source experts are independently trained
- composition/routing is post hoc
- no parameter sharing or retraining occurs at routing time
- selected expert generation and all-expert diagnostics are separated

## Objective 5: Domain-Specific And Generalization Performance

Folder responsibility:

- `src/cvae_downstream_evaluation/downstream.py`
- `src/cvae_downstream_evaluation/fidelity.py`
- `src/cvae_downstream_evaluation/reporting.py`

Expected evidence:

- downstream balanced accuracy and macro-F1 on held-out target evaluation data
- oracle gap and ranking alignment for downstream utility
- fidelity diagnostics as secondary evidence
- domain breakdown and seed stability

## Thesis Narrative Placement

Primary placement:

- Chapter 6: utility-aligned compatibility and downstream transfer

Supporting placement:

- Chapter 3: protocol and evaluation design
- Chapter 5: per-domain heterogeneity, if downstream failures concentrate by domain
- Chapter 8: implications and limitations

Do not use this package to revise the existing negative result unless downstream evidence directly changes the thesis claim boundary.

