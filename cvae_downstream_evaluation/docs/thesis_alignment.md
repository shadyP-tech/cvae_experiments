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
- MIDOG++ real-feature gate evidence from
  `midogpp_real_feature_gate/artifacts/midogpp_real_feature_gate_v1/`:
  leakage/provenance `PASS`, 9/9 valid eligible source-only held-out-center
  folds, mean source-only BACC `0.668`, mean AUROC `0.728`, and pooled
  diagnostic ceiling mean BACC `0.902`.

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
- `src/cvae_downstream_evaluation/compatibility/`
- `src/cvae_downstream_evaluation/features/`
- `src/cvae_downstream_evaluation/utility_matrix/`

Expected evidence:

- support-NELBO is labeled as a direct utility estimate
- metadata routing remains a serious baseline
- random and ensemble baselines are retained
- downstream oracle is diagnostic only
- learned downstream utility estimators are trained on source-inner folds only
- direct target identity is not a deployable metadata feature

## Objective 4: MoErging Generator

Folder responsibility:

- `src/cvae_downstream_evaluation/generation.py`
- `src/cvae_downstream_evaluation/schemas/`
- `src/cvae_downstream_evaluation/artifacts/`

Expected evidence:

- source experts are independently trained
- composition/routing is post hoc
- no parameter sharing or retraining occurs at routing time
- selected expert generation and all-expert diagnostics are separated
- `selection_eligible` and `diagnostic_only` candidate rows are schema-enforced

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
- For MIDOG++ candidate-surface work, cite the real-feature gate as the
  transfer/headroom baseline: source-only real features are above chance across
  eligible centers, with weakest eligible center `2` at BACC `0.587`; pooled
  diagnostic rows are non-adoption-eligible and only quantify headroom.

## Thesis Narrative Placement

Primary placement:

- Chapter 6: utility-aligned compatibility and downstream transfer

Supporting placement:

- Chapter 3: protocol and evaluation design
- Chapter 5: per-domain heterogeneity, if downstream failures concentrate by domain
- Chapter 8: implications and limitations

Do not use this package to revise the existing negative result unless downstream evidence directly changes the thesis claim boundary.

## MIDOG++ Real-Feature Gate Placement

The MIDOG++ real-feature gate is Chapter 3/6 support evidence for the evaluation
framework and unseen-domain transfer setup. It can motivate exploratory CVAE
candidate-surface experiments because the source-only real-feature baseline is
nontrivial and the pooled diagnostic ceiling leaves substantial headroom.

It cannot support claims about CVAE preservation, compatibility routing,
synthetic utility, or generative quality. Those require separate generated
embedding and downstream utility artifacts.

## Current Downstream Synthesis

For the current CVAE downstream branch, use
[`c63_conceptual_synthesis.md`](c63_conceptual_synthesis.md) as the working synthesis.

Key takeaway:

- C6.3 is the strongest current CVAE setup because it reframes the failure from sparse top-1 compatibility selection to dense post-hoc routing-risk reduction.
- C6.3 should be described as dense late aggregation over frozen CVAE expert/mode classifiers, not as learned compatibility estimation.
- C7.1a source-probe CE is negative diagnostic evidence: improving source-discriminative generated geometry did not improve held-out downstream utility.
