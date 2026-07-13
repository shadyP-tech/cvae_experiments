# Implementation Order

This is an historical legacy document from the archived CVAE-downstream package
tree. It is preserved for provenance context and is **not** an active runbook
for this repository's current `src/midogpp_thesis` workflow.

Use this order to keep the next code changes small and auditable.

## 0. MIDOG++ Real-Feature Gate Baseline

Current synced gate:

- artifact root: `midogpp_real_feature_gate/artifacts/midogpp_real_feature_gate_v1/`
- decision labels: `GO_REAL_FEATURE_GATE_PASSED`,
  `CLAIM_SCOPE_REAL_FEATURE_TRANSFER_ONLY`
- valid source-only held-out-center folds: `9/9`
- mean source-only BACC/AUROC: `0.668` / `0.728`
- worst eligible center: center `2`, BACC `0.587`, AUROC `0.629`
- pooled diagnostic ceiling mean BACC/AUROC: `0.902` / `0.964`

Implication:

- CVAE candidate-surface work is justified as exploratory because real features
  carry nontrivial held-out-center signal and pooled diagnostics show large
  headroom.
- Do not use this gate to claim CVAE preservation, NELBO compatibility, routing
  quality, synthetic utility, or generative quality.
- Before making stronger thesis-facing claims, add or sync negative-control and
  uncertainty/seed-stability artifacts for the gate.

Completed real-feature classifier-reference follow-up:

- artifact root:
  `cvae_downstream_evaluation/artifacts/midogpp/real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42/`
- workstation leakage/protocol inspection: `PASS`
- leakage/provenance report: `PASS`
- evidence label: `WEAK_PASS_REAL_FEATURE_TRANSFER_ONLY`
- source-inner-tuned fixed-0.5 mean BACC/macro-F1: `0.740312` / `0.737205`
- untuned default fixed-0.5 mean BACC/macro-F1: `0.665812` / `0.661730`
- tuned-minus-default mean BACC delta at fixed `0.5`: `+0.074500`
- tuned wins over untuned default on `9/9` eligible held-out centers
- source-inner threshold selection chose `0.5` for every held-out center and
  added `+0.000000` mean BACC over fixed `0.5`

Implication:

- Treat this as the current strongest synced MIDOG++ real-feature source-only
  classifier reference for later candidate-surface comparisons.
- Treat threshold tuning as protocol-clean but empirically inert for this
  corrected `xyxy` cache; do not prioritize another threshold run unless the
  threshold rule changes for a predeclared reason.
- Do not use it to claim CVAE preservation, NELBO compatibility, routing
  quality, synthetic utility, or generative quality.
- Before promoting the claim beyond weak pass, add classifier-seed stability
  and a formal paired comparison against the active SAIL real-feature
  aggregation outputs.

## 1. Contracts And Manifests

Implement:

- config loader
- protocol manifest writer
- split manifest writer
- expert provenance loader
- strict Camelyon17 v1 config validation
- candidate manifest schema
- frozen protocol snapshot hash
- feature/selection/report schema contracts

Done when:

- target support and target evaluation rows are disjoint
- held-out target expert is excluded from candidate experts
- forbidden target evaluation fields are absent from routing inputs
- stale TODOs, conditional-generation wording, and non-Camelyon17 v1 scope are rejected
- direct target identity is allowed only in lineage/reporting fields

## 2. Selection Bridge

Implement:

- support-NELBO score loading from existing support-routing artifacts
- direct support-NELBO argmin selection
- metadata, random, and ensemble baseline selection descriptors
- deterministic random baseline construction when absent from support artifacts
- source-inner learned downstream utility estimator contracts
- top1, top-k uniform, and soft aggregation rules
- candidate eligibility filtering for every deployable method
- allowed feature table builder from candidate/support/source-inner/metadata CSVs
- learned-utility selection report builder
- source-inner utility estimator trainer and prediction builder
- leakage report builder

Done when:

- selections can be reproduced from the manifest alone
- routing decisions are made before synthetic generation
- downstream candidate scores are computed once per candidate expert and not duplicated by support seed/size
- real held-out target BACC/macro-F1 cannot enter estimator training
- diagnostic matrices cannot be imported or read by deployable selection code
- `allowed_pre_eval_features.csv`, `adoption_eligible_predictions.csv`, and `learned_utility_alignment.csv` can be built from frozen artifacts
- leakage report flags are generated from artifact checks, not manually filled

## 3. Synthetic Embedding Generation

Implement:

- class-balanced label schedule
- decoder sampling wrapper
- generation seed handling
- generated embedding manifest
- matching projection-frame contract for expert-specific heads

Done when:

- every candidate expert can generate the same locked synthetic budget
- selected-expert generation and all-expert oracle diagnostics share the same sampling contract
- naive all-expert ensemble uses late probability averaging rather than mixed-frame concatenation

## 4. Synthetic-Only Downstream Utility

Implement:

- small downstream classifier
- fixed classifier hyperparameters
- all-expert downstream matrix
- diagnostic all-candidate downstream matrix quarantine
- target evaluation metrics

Done when:

- every candidate expert has a comparable downstream score
- downstream oracle is computed only after all candidate scores exist
- frozen config hashes exist before metric files are written

## 5. Fidelity Diagnostics

Implement:

- MMD
- energy distance
- Frechet embedding distance
- mean/covariance distance
- kNN precision/recall/density/coverage if practical

Done when:

- fidelity metrics are reported separately from downstream utility
- correlation with downstream utility is computed as diagnostic evidence

## 6. Reporting

Implement:

- routing-to-downstream alignment table
- downstream performance table
- fidelity diagnostics table
- support-size stratified summary table
- stability table
- leakage/provenance report
- allowed pre-evaluation feature table
- adoption-eligible prediction/selection table
- decision summary

Done when:

- the report can classify the result as PASS, WEAK PASS, DIAGNOSTIC ONLY, or FAIL
- the summary states allowed and forbidden thesis claims
- PASS gates use only the primary generation mode and budget 128; support-size stratification is descriptive only
- improvement claims use budget-matched dense aggregation, with full-budget dense reported separately
