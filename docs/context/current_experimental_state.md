# Current Experimental State

Last updated: 2026-08-08

This page records verified evidence and canonical availability after the
completed MIDOG++ repository migration. The canonical contract, feature
caches, source-inner evidence, and the completed Uniform-B v2 Stage-30 bank,
Stage-40 GenerationLock, and Stage-60 equal-union, metadata, validation-cache,
source-inner utility, and utility/regret policy artifacts are registered in
this checkout. Their canonical production artifacts are validated on the
workstation. These boundaries authorize a bank, generation contract, direct
control, compatibility proxy, and pre-evaluation policies; they do not
establish routing quality or frozen-policy downstream utility.

## Local Readiness Snapshot

- `conda run -n thesis python -m midogpp_thesis workspace validate` passes once
  the package is installed from this repository.
- The complete 22,569-patch contract is present locally and on the workstation
  under `datasets/midogpp/contract/annotation_patch_v1/`; `dataset-validate`
  passes with nine eligible domains.
- The corrected `xyxy` Virchow2 cache is present locally and on the workstation
  under
  `datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/seed42/`.
  Train, validation, and test counts align exactly with the contract, and the
  train tensor hash is
  `f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`.
- The tuned real-feature reference and tuned CVAE preservation bundle are
  present locally and on the workstation at their canonical stage-10 and
  stage-20 paths and validate `PASS`.
- The approximately 65 GB raw source tree is workstation-only at
  `datasets/midogpp/raw/MIDOGpp/` and intentionally not synced to the Mac.
- Physical-multiscale v1 and v2 are immutable failed-audit lineages. The clean
  v3 clipped-bbox annotation-local implementation is registered as
  `diagnostic`; its 9,648-row/216-TIFF `xai-master` source audit passes with 84
  deterministic bbox clips, no row exclusion, and no synthesized pixels. Its
  immutable contract and atomic B/C cache are workstation-built, independently
  validated, and fully hash-promoted. The Stage-10 diagnostic and Stage-90
  fixed-B retrospective replay are complete, independently validated, and
  fully hash-promoted. Phase-B case-disjoint test confirmation is also
  complete, validates, and passes its predeclared within-center gate.
- Stage 30 has an active, validated Uniform-B v2 expert bank with publication
  state `ROUTING_AUTHORIZED`. Stage 40 has an independently validated,
  target-data-free GenerationLock. Stage 60 has independently validated
  equal-union, metadata, label-blind validation-cache, non-selecting
  source-inner utility, and utility/regret policy artifacts. All comparison
  policies have publication state `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`;
  the utility/regret policy falls back exactly to equal-union for every outer
  fold.
- Stage 70 has no result. The next gate is a separately authorized fresh
  target-evaluation artifact, followed by matched scoring of the frozen
  equal-union, metadata max-tie, and utility/regret arms; target labels remain
  scoring-only after predictions.
- The Stage-10 matched-reference v2, Stage-10 conditional-logit alignment
  diagnostic, Stage-20 scalar source-inner prior-recovery/Task-Fisher run, and
  bounded Stage-20 training-seed stability run are complete on `xai-master`
  and protocol-clean. These four workstation bundles have not yet been synced
  into this local checkout.
- The stability bundle remains a valid, published input to the separate
  consensus-recipe `midogpp.expert_bank.provenance_clean.v1` plan, whose runner
  is still unimplemented. The active Uniform-B v2 bank instead comes from its
  separately reviewed Stage-20 aggregate-prior union study and Stage-30
  promotion. The registered Stage-20 outer run remains blocked by the scalar
  source-inner gate.

## MIDOG++ Real-Feature Gate

Canonical target:

```text
artifacts/midogpp/10_real_feature_reference/midogpp_real_feature_gate_v1/
```

Availability: verified historical evidence, not currently present in the local
canonical target. The original verified source was the retired path
`midogpp_real_feature_gate/artifacts/midogpp_real_feature_gate_v1/`; that string
is provenance only and is not an active path.

Validation status:

- schema: `midogpp_real_feature_transfer_ceiling_v1`
- leakage/provenance report: `PASS`
- decision labels: `GO_REAL_FEATURE_GATE_PASSED` and
  `CLAIM_SCOPE_REAL_FEATURE_TRANSFER_ONLY`
- valid source-only held-out centers: 9/9 eligible centers
- center `4`: quarantine-only diagnostic row

Result summary:

- source-only mean BACC `0.668`, macro-F1 `0.662`, AUROC `0.728`, and PR-AUC
  `0.737`
- pooled diagnostic ceiling mean BACC `0.902`, macro-F1 `0.902`, AUROC `0.964`,
  and PR-AUC `0.969`

Interpretation: `USABLE WITH CAVEATS` for real-feature source-only signal and
headroom. It is not CVAE preservation, compatibility, routing, generation, or
synthetic downstream-utility evidence.

## MIDOG++ Source-Inner Reliability Dense Ensemble

Canonical local artifact:

```text
artifacts/midogpp/10_real_feature_reference/midogpp_source_inner_reliability_v1/
```

Validation status:

- protocol boundary: `PASS`
- evidence labels: `NEGATIVE_RESULT`, `DIAGNOSTIC_ONLY`, and
  `TODO_VERIFY_ARTIFACT` for missing bundle elements
- target expert excluded; target labels scoring-only
- source-inner normalization used target center: `false`
- target labels used for weights: `false`
- uniform dense baseline included
- no copied config snapshot or decision report is present in the retained
  bundle

Result summary:

- source-inner weighted mean BACC `0.689295`, macro-F1 `0.686507`, AUROC
  `0.777831`
- uniform dense mean BACC `0.689704`, macro-F1 `0.685938`, AUROC `0.786136`
- weighted selection wins on 5/9 centers but is slightly worse on mean and
  worst-center BACC

Interpretation: protocol-clean negative evidence for this exact source-inner
softmax dense weighting rule. Uniform dense remains the stronger baseline for
this run. This real-feature result does not establish CVAE or routing claims.

## MIDOG++ Tuned Real-Feature Classifier Reference

Canonical target:

```text
artifacts/midogpp/10_real_feature_reference/real_feature_threshold_both_annotation_patch_xyxy_virchow2_seed42/
```

Availability: `local_and_workstation`, migrated with byte-identical pre/post
manifests. The retired workstation source path is absent; it remains only in
the migration audit as provenance.

Validation status:

- workstation leakage/protocol inspection: `PASS`
- schema: `midogpp_real_feature_source_only_classifier_reference_v1`
- evidence label: `WEAK_PASS_REAL_FEATURE_TRANSFER_ONLY`
- claim scope: `real_feature_transfer_only`
- generated embeddings and CVAE checkpoint used: `false`
- router: `false`
- target labels used for selection: `false`
- target labels used for final scoring only: `true`
- manifest hash:
  `db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869`
- feature-cache hash:
  `f6608e513fb2d06671e3ec117b093a85d58530b77b1fae44a3be1680d9feabd2`

Result summary:

- tuned fixed-0.5 mean BACC `0.740312`, macro-F1 `0.737205`
- untuned fixed-0.5 mean BACC `0.665812`, macro-F1 `0.661730`
- tuned-minus-default mean BACC `+0.074500`
- tuned classifier wins on all 9 eligible held-out centers
- worst tuned center: center `1`, BACC `0.679245`
- best tuned center: center `6`, BACC `0.792350`

Threshold conclusion: the source-inner one-SE rule selected `0.5` for every
fold, so threshold tuning changed no prediction and added `+0.000000` mean
BACC. The diagnostic raw argmax thresholds had mixed target effects and are not
an adoption rule.

Interpretation: `WEAK PASS` for source-inner real-feature classifier tuning.
This is the frozen comparator for stage-20 preservation; it does not establish
CVAE preservation, routing, or generated-embedding utility.

Missing evidence:

- classifier-seed stability beyond seed `23`
- formal paired comparison against another active real-feature aggregation
  baseline on hash-matched inputs

## MIDOG++ Conditional-Logit Alignment Diagnostic

Canonical target:

```text
artifacts/midogpp/10_real_feature_reference/conditional_logit_alignment_v1/seed42/
```

Availability: `workstation_only`; the complete 16-file bundle was inspected
directly at the canonical path on `xai-master` and has not been synced locally.
The registry status remains `diagnostic`. The artifact catalog still carries
`TODO_VERIFY_ARTIFACT`; updating that lifecycle metadata after sync is an
explicit TODO, not a reason to weaken the directly verified result.

Validation status:

- runtime status: `COMPLETE`
- leakage/provenance report: `PASS`
- evidence labels: `NEGATIVE_RESULT` and `DIAGNOSTIC_ONLY`
- claim scope: `real_feature_transfer_only`
- target evaluation labels used for fit or gamma selection: `false`
- target evaluation labels used for scoring only: `true`
- generated embeddings, CVAE checkpoint, expert bank, NELBO, and router used:
  `false`
- protocol hash: `3806cca63f914a09`
- table-bundle hash: `4b19d9052da239f6`

Primary result against the matched `gamma=0` classifier:

| Quantity | Selected CLA | `gamma=0` | Selected minus `gamma=0` |
| --- | ---: | ---: | ---: |
| mean BACC | `0.7434042753` | `0.7434898099` | `-0.0000855346` |
| mean macro-F1 | `0.7414722419` | `0.7414225940` | `+0.0000496479` |
| worst-center BACC | `0.6913746631` | `0.6913746631` | `0.0000000000` |

Source-inner selection chose a positive gamma in every outer fold (`gamma=10`
for seven centers and `gamma=1` for two), but outer BACC produced only two
wins, four ties, and three losses. Only `22/9,648` predictions changed: nine
changes corrected the `gamma=0` prediction and thirteen introduced an error.
The post-run source-inner-versus-outer delta Spearman correlation was `0.052`.
These descriptive audits cannot override the predeclared BACC decision.

Interpretation: the regularizer was mechanism-active but utility-inert. The
outer conditional operator occupies a rank-14 subspace of the 2,560-dimensional
feature frame; the artifact-backed causal audit attributes the weak response to
low overlap with the fitted discriminative direction and to decision-threshold
plateaus. This audit interpretation comes from the supplied artifact-backed
synthesis; it is not a claim of causal domain-shift removal. No post-hoc
probability metric is adoption evidence.

Stop recommendation: do not adopt CLA and do not expand the gamma sweep. Any
future CLA work should be optional, predeclared, and restricted to a mechanism
diagnostic. This result cannot replace the Stage-10 matched denominator or
support CVAE preservation, routing, composition, generation, or downstream
utility claims. See
`docs/wiki/03-experiments/midogpp-conditional-logit-alignment-diagnostic.md`.

## MIDOG++ Physical Multiscale Clipped-Bbox Annotation-Local Pilot v3

Canonical target:

```text
artifacts/midogpp/10_real_feature_reference/physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3/seed42/
```

Availability: `COMPLETE; VALIDATED`. The code, configs, registry/catalog
entries, dataset ownership split, command surfaces, independent validators,
and centralized tests are implemented. The workstation source audit passes
for 9,648 rows and 216 slides. It repairs 84 partially outside bboxes using
their clipped continuous centroid; the retained-area fraction ranges from
`0.40` to `0.98`, the maximum anchor correction is `15` pixels, and the two
formerly invalid centers become `(11.5, 2416)` and `(4211, 10)`. No row is
excluded and no pixel is padded or synthesized. This audit remains Stage-90
provenance only.

The workstation-only immutable v3 contract and atomic 3,840-dimensional B /
11,520-dimensional C cache bundle now exist and independently validate. The
contract covers 9,648 rows, 216 TIFFs, and 28,944 scale-level pooling records;
its contract hash is `4d6c3088e8b72073`. The numeric canonical-A bridge passes
with minimum cosine `0.9999994039535522` and maximum relative L2
`2.322336695215199e-05`. The frozen task bridge agrees on 9,647/9,648
predictions. All 37 required files across the contract, atomic parent, B child,
and C child have cataloged SHA-256 values. Registry status is `diagnostic`.

The predeclared selector compares canonical A, v3 canonical-JPEG fixed-center
B, and the complete v3 clipped-bbox-anchor, shifted-in-bounds, multiscale
annotation-local C under the same ten-spec classifier grid.
Each outer center is absent from 240 source-inner cells, all nine decisions are
locked before any outer shard is opened, and the gate requires mean BACC delta
at least `+0.02`, six of eight strict wins, and worst delta at least `-0.01`.
Failure falls back exactly to canonical A.

Claim boundary: at most the diagnostic performance of the complete nested
adaptive pipeline. The boundary handling changes the anchor for 84 rows, so no
future result may isolate pooling, scale, crop-shift, or anchor effects. No
individual component effect is identified. The locked policy selected B for
six centers, C for two, and A for one. Its mean BACC is `0.784403` versus
`0.740312` for A, a paired delta of `+0.044091`; the conditional bootstrap 95%
percentile interval is `[+0.031537, +0.054889]`. The pilot cannot establish CVAE preservation,
calibration, probabilistic or NELBO validity, routing, deployment, new-center
generalization, or any Stage-20 through Stage-70 decision. V1 and v2 remain
non-runnable failed-audit lineages and are not fallbacks.

## MIDOG++ Uniform-B v3 Retrospective Replay v1

Canonical target:

```text
artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_v3_retrospective_replay_v1/seed42/
```

Availability: `COMPLETE; VALIDATED; DIAGNOSTIC ONLY`. The replay imports all
nine source-v3 classifier locks, fixes B globally, refits A and B without each
held-out center, and exactly replays the canonical A and source-v3 result rows.
The B-only store never loads C. The validator passes with 9 source locks, 18
outer result rows, 19,296 prediction rows, and 9 paired center comparisons.

Result summary:

- uniform-B equal-center mean BACC: `0.792087`;
- canonical-A equal-center mean BACC: `0.740312`;
- paired mean delta: `+0.051775`;
- strict wins: `8/9` centers;
- worst-center delta: `-0.002890` at center `9`;
- conditional paired case-bootstrap 95% percentile interval:
  `[+0.038962, +0.063599]` from 2,000 valid replicates.

Interpretation: this confirms exact reproducibility of the fixed-B result on
the same nine centers and indicates that the adaptive selector left useful B
performance on the table. It does not confirm B prospectively: the choice to
replay B uniformly was informed by the already-observed target scores. The
bootstrap conditions on this choice, the observed centers, fixed fits, and
imported classifier locks and does not cover representation-choice or
new-center uncertainty. The result is non-adoptive and cannot replace the
canonical Stage-10 representation or feed any downstream stage.

## MIDOG++ Uniform-B v3 Prospective Test Confirmation v1

Canonical target:

```text
artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_v3_prospective_test_confirmation_v1/seed42/
```

Availability: `COMPLETE; VALIDATED; DIAGNOSTIC ONLY`. Before test-B extraction,
the protocol froze B, the source-only per-center classifier locks, the
equal-center test BACC estimand, and four confirmation checks. The evaluation
uses 9,928 eligible test rows with zero sample or case overlap against the
9,648 discovery/train rows. The validation split is unused and target-test
labels are scoring-only.

Result summary:

- decision: `CONFIRMED_WITHIN_CENTER`;
- uniform-B equal-center test mean BACC: `0.799159`;
- canonical-A equal-center test mean BACC: `0.735733`;
- paired mean delta: `+0.063426`;
- strict wins: `9/9` centers;
- worst-center delta: `+0.010638` at center `7`;
- conditional paired case-bootstrap 95% percentile interval:
  `[+0.050709, +0.073650]` from 2,000 valid replicates.

All predeclared checks pass: mean delta at least `+0.02`, at least six wins,
worst delta at least `-0.01`, and a positive bootstrap lower bound. This is
independent confirmation for new cases within the same nine centers. It is not
external-dataset evidence and does not quantify new-center uncertainty. The
Stage-90 placement preserves the existing firewall: the result cannot
automatically replace the canonical reference or feed Stage 20 through 70.

## MIDOG++ Tuned-Classifier CVAE Preservation

Canonical target:

```text
artifacts/midogpp/20_cvae_preservation/virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42/
```

Availability: `local_and_workstation`, migrated with byte-identical pre/post
manifests. The retired workstation source path is absent; it remains only in
the migration audit as provenance.

Validation status:

- verdict: `THESIS-FACING` for `claim_scope=cvae_preservation_only`
- leakage report and identity-overlap audit: `PASS`
- zero overlap for `sample_id`, `case_id`, `image_path`, and
  `feature_row_index`
- generated embeddings and CVAE checkpoint used: `true`
- target labels used for selection: `false`
- target center used during fit: `false`
- target evaluation labels used for scoring only: `true`
- eligible held-out centers: `0,1,2,3,5,6,7,8,9`
- imported real-reference hash: `78c1d254019a2cc0`

Result summary:

| Representation | Mean BACC | Macro-F1 | Preservation ratio |
| --- | ---: | ---: | ---: |
| imported tuned real-feature reference | `0.740312` | `0.737205` | — |
| `decode_mu_fit_to_real_eval` | `0.719681` | `0.717766` | `0.919368` |
| `posterior_sample_fit_to_real_eval` | `0.716630` | `0.714110` | `0.910740` |
| `prior_sample_fit_to_real_eval` | `0.637563` | `0.630151` | `0.571675` |
| `real_pca128_reference` | `0.720533` | `0.718135` | `0.922785` |

Interpretation: the single `pca128_beta001` run supports the narrow claim that
decode and posterior representations preserve almost all of the PCA128
real-feature classifier surface over nine centers. Prior sampling is materially
weaker and does not support an unconditional-generation utility claim.

This artifact is explicitly forbidden as routing, expert-selection,
NELBO-compatibility, controllable-generation, GMM-composition, or downstream
synthetic-utility evidence. It has
`may_feed_deployable_selection=false`.

Next preservation-specific evidence:

- seed or variant stability for the same predeclared preservation protocol
- separate expert-bank, generation, routing, and held-out downstream stages
  before making any broader thesis claim

## Prior-Recovery And Task-Fisher Result

Status: the Stage-10 matched reference, scalar Stage-20 source-inner bundle,
and bounded Stage-20 training-seed stability bundle are complete and directly
validated on `xai-master`. The catalog still carries the stale lifecycle label
`TODO_VERIFY_ARTIFACT` for these canonical output destinations; catalog
promotion and a local artifact sync remain separate follow-up work.

| Stage | Experiment | Canonical output | Current evidence |
| --- | --- | --- | --- |
| 10 | `midogpp.real_feature.eligible_tuned_predict_reference.v2` | `artifacts/midogpp/10_real_feature_reference/eligible_tuned_real_reference_v2/seed42/` | validator `PASS` |
| 20 | `midogpp.cvae.prior_recovery_source_inner.v1` | `artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_v1/seed42/` | `COMPLETE`; full validator `PASS`; `NEGATIVE_GATE_COMPLETE` |
| 20 | `midogpp.cvae.prior_recovery_source_inner_training_seed_stability.v1` | `artifacts/midogpp/20_cvae_preservation/prior_recovery_source_inner_training_seed_stability_v1/seeds17_42_101/` | `COMPLETE`; `PUBLISHED`; full validator `PASS`; narrow training-seed-stability `NEGATIVE_RESULT` |
| 20 | `midogpp.cvae.learned_conditional_prior_source_inner.v2` | `artifacts/midogpp/20_cvae_preservation/learned_conditional_prior_source_inner_v2/seeds17_42_101/` | `IMPLEMENTED AND REGISTERED`; not production-run; non-adoptive source-inner study only |
| 20 | `midogpp.cvae.task_fisher_shrinkage_source_inner.v2` | `artifacts/midogpp/20_cvae_preservation/task_fisher_shrinkage_source_inner_v2/seeds17_42_101/` | `IMPLEMENTED AND REGISTERED`; not production-run; non-adoptive source-inner study only |
| 20 | `midogpp.cvae.prior_recovery_outer.v1` | `artifacts/midogpp/20_cvae_preservation/prior_recovery_outer_v1/seeds17_42_101/` | blocked by the source-inner gate; no outer result |

The matched Stage-10 v2 reference uses the eligible nine centers, full
Virchow2 features, and sklearn `predict`. Its mean BACC is `0.740312` and mean
macro-F1 is `0.737205`; center `1` is worst at `0.679245` BACC and center `6`
is best at `0.792350`. Its protocol hash is `786589b799d61b14` and its bound
reference-bundle hash is `995aa193c82ee7ec`. This confirms the matched
denominator only; it remains a `real_feature_transfer_only` result.

The source-inner Stage-20 bundle contains nine valid `RecipeLock` files. Seven
locks select a conditional sampler: centers `0,6,7,8` select Task-Fisher plus
the full conditional sampler (`D`), centers `1,2` select isotropic plus full
conditional sampling (`C`), and center `3` selects isotropic plus diagonal
conditional sampling (`C`). Centers `5` and `9` retain isotropic standard-normal
sampling (`A`). The selection bundle hash is `1e929d05ff987ad9`; the protocol
hash is `dd7ca955d79fade4`.

The gate outcome is `NEGATIVE_GATE_COMPLETE` with
`factorial_triggered=false`. This is a protocol-complete negative gate, not a
numerical, leakage, identity, checkpoint, or sampler-realization failure.
Center `5` is borderline and generation-seed-sensitive: its best diagonal
sampler has mean
preservation-ratio delta `+0.109849` over `A` but wins only `5/8` strict inner
comparisons. Center `9` is less consistent: its best full sampler has mean
delta `+0.087841` but wins only `4/8`. Both miss the predeclared six-win gate.

The registered outer v1 requires all nine locks to select `C` or `D` and
requires `factorial_triggered=true`; it therefore must not be run against this
bundle. No outer preservation, routing, compatibility, or downstream-utility
claim follows from the source-inner result.

The bounded stability experiment fully crosses training and generation seeds
`17,42,101`. Its canonical artifact is `COMPLETE` and `PUBLISHED`; the full
bundle, leakage, identity-overlap, and RNG validators pass. All `27/27` child
locks validate, all `9/9` consensus locks are export-ready, and
`stage30_recipe_ready=true`. The Stage-30 loader accepts the bundle. Its
protocol hash is `bbde3e5c5a1e3374`, and its selection-bundle hash is
`79cb9b614779c23b`.

The predeclared cross-seed consensus result is:

| Outer center | Consensus recipe | Cross-seed interpretation |
| --- | --- | --- |
| `0,1,2,3,5,9` | `A`: isotropic objective, standard-normal sampler | cross-seed arm or conditional-family disagreement; conservative fallback |
| `6,7` | `D`: Task-Fisher objective, full conditional sampler | exactly unanimous across training seeds |
| `8` | `C`: isotropic objective, full conditional sampler | sampler family stable, objective unstable |

Only centers `6` and `7` are exactly unanimous. Centers
`0,1,2,3,5,8,9` are unstable under the declared rule. This is a
protocol-valid, thesis-facing `NEGATIVE_RESULT` for broad training-seed
stability of source-inner recipe selection, together with an operational
`PASS` for the conservative consensus publication gate. Export-ready means the
predeclared fallback produced a valid fold recipe; it does not mean conditional
recipe selection was stable for every fold.

The result is limited to `claim_scope=cvae_recipe_lock_only`. It establishes no
outer preservation, routing, compatibility, generation-quality, or downstream
utility claim. The scalar seed-42 source-inner result and blocked outer-v1 gate
remain unchanged. Its consensus locks still feed only the separate planned
`midogpp.expert_bank.provenance_clean.v1` path. They were not used to construct
or select the now-authorized Uniform-B v2 bank described below, and the
existence of that bank does not convert this older recipe-lock result into
routing evidence.

Two separately registered v2 source-inner mechanism studies are an explicit
non-adoptive exception to that stop rule. The learned-prior study compares
`A`, `C-diag`, and a jointly learned class-conditional diagonal Gaussian `E`
under the fixed isotropic objective. The objective study fixes the
standard-normal prior and compares Fisher strengths
`alpha in {0,0.05,0.10,0.25}`. Both use the full outer/inner structure and the
crossed training/generation seeds `17,42,101` from the outset.

Status: implementation and registration only; no canonical result exists yet.
Their claim scope is `cvae_source_inner_study_only`. They cannot emit a
`RecipeLock`, cannot publish a Stage-30 recipe, and cannot replace or revise the
current scalar or consensus locks. Stage 30 may proceed with the currently
published consensus bundle only along its planned alternative path. A future v2
result may support only its named source-inner mechanism/stability question;
it is not outer-preservation, generation, routing, or downstream-utility
evidence.

## Uniform-B V2 Routing-Authorized Expert Bank

Canonical Stage-20 source evidence:

```text
artifacts/midogpp/20_cvae_preservation/uniform_b_geco_aggregate_prior_union_source_inner_v2/seeds17_42_101/
```

Canonical Stage-30 promoted bank:

```text
artifacts/midogpp/30_expert_bank/uniform_b_v2_routing_authorized_expert_bank_v1/
```

The source study is `COMPLETE` and validates `PASS`. Its full-shrinkage
aggregate-posterior sampler (`PS`) reaches mean source-inner BACC `0.770112`,
compared with `0.757348` for the standard-normal prior (`P0`) and `0.771571`
for the posterior-sample ceiling (`Q`). The PS gain over P0 is `+0.012764`,
the remaining Q-minus-PS gap is only `0.001459`, and the PS training-seed
means are `0.772334`, `0.773440`, and `0.764562` for seeds `17`, `42`, and
`101`. This passed the predeclared source-study gate but remained
non-consumable until separate Stage-30 review.

`midogpp.expert_bank.uniform_b_v2_routing_promotion.v1` completed that review.
The run state is `COMPLETE`; promotion and independent validation both report
`PASS`; the decision is `PROMOTED_AS_ROUTING_AUTHORIZED_EXPERT_BANK`; and the
publication state is `ROUTING_AUTHORIZED`. The bundle re-audits and retains
all 27 independently trained checkpoints (nine source centers by three
training seeds), all source-local frames, and all source-only full-shrinkage
sampler states. No expert or seed was selected by held-out performance, and
all identity-overlap, sampler-fallback, and classifier-convergence counts are
zero. The bank lock is `9972a41dcd4814cd`, the equal-union control lock is
`cddbcc3b3343fe38`, and the promotion protocol hash is `5b3087f3aa41c388`.

The frozen direct control for future routing is
`uniform_b_v2_equal_union_ps`. For each held-out target it excludes the
target expert, includes all eight remaining source centers, generates 1,024
samples per class as 128 per source, crosses training and generation seeds
`17,42,101`, and forbids target-conditioned weights and seed selection.

Claim boundary: this is `expert_bank_construction_only`. It authorizes the
Stage-30 artifact to feed future deployable-selection experiments, but it
does not claim that a router works. The source-inner evaluation labels are
consumed once for whole-bank adoption and cannot be reused as fresh evidence
to select an expert, checkpoint seed, or routing policy. The reported
`0.770112` is based on seven-source inner tasks; the future eight-source
equal-union control must be freshly scored under paired RNG, shuffles,
classifier budgets, candidate pools, and evaluation rows.

Reproducibility caveat: artifact inputs and promoted contents are hash-locked,
but the recorded repository revision `40221038ca714bf33fd21582857d21fa1db4e6f3`
had `repository_dirty=true`. Before thesis archival, preserve the exact diff
or regenerate and revalidate the promotion from a clean committed revision.
The Stage-40 GenerationLock run inherited the same caveat and records
repository-status hash
`9fddbbc8828994dd484e26cb3cf16a97262e01e6208a75474731604628a92254`.

## Uniform-B V2 Generation And Stage-60 Policy Locks

Canonical Stage-40 GenerationLock:

```text
artifacts/midogpp/40_prior_and_generation/uniform_b_v2_generation_lock/v1/
```

Canonical Stage-60 equal-union policy lock:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_equal_union_policy_lock/v1/
```

Canonical Stage-60 metadata compatibility and comparison-policy locks:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_exact_match_compatibility/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_tie_union_policy_lock/v1/
```

Canonical completed utility/regret chain:

```text
datasets/midogpp/derived/features/virchow2/uniform_b_v2_routing_validation_cache_v1/seed42/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_source_inner_candidate_utility/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_utility_regret_policy_lock/v1/
```

Both workstation artifacts are `COMPLETE`; their production validation and
independent validator reruns report `PASS`. Stage 40 freezes source-only prior,
frame, budget, seed, shuffle, and classifier settings under GenerationLock
`34e551425710362e`. It contains 81 source streams, 81 target-replicate
contracts, and 162 passing health records. These are integrity and readiness
checks, not held-out performance.

Stage 60 freezes `uniform_b_v2_equal_union_ps` as the canonical direct
control. Its decision is
`FROZEN_AS_CANONICAL_EQUAL_UNION_ROUTING_CONTROL`, publication state is
`POLICY_FROZEN_FOR_STAGE70_EVALUATION`, and policy lock is
`4b9ea514308b084f`. The bundle enumerates 81 target/training-seed/generation-
seed replicates and 648 assignments: all eight non-target sources in each
replicate, 128 generated samples per class and source, and no expert, seed,
rank, or target-conditioned weight selection. Target-center identity is used
only for the held-out fold, target-expert exclusion, and a predeclared label-
blind within-class shuffle namespace.

This is a policy-contract result only. It uses no target samples, support rows,
or labels and computes no BACC, macro-F1, routing advantage, or downstream
utility.

The first comparison is also complete and independently validates `PASS`.
Compatibility lock `4b46b3d157b07781` records 72 ordered target-excluded
componentwise exact-match scores over the routing-time metadata axes tumor
type, lab or origin, and scanner model. It is non-selecting and is explicitly a
proxy rather than NELBO, expected utility, oracle agreement, or routing
quality. The separate max-tie policy retains all maximum-score ties, splits
1,024 rows per class equally among them, and binds policy lock
`27f16953b32c46cd`. It contains nine target selections, all 81 training-by-
generation seed replicates, and 153 assignments. The exact selected sets are
`0->{5}`, `1->{2}`, `2->{1}`, `3->{1,2}`, `5->{6,7}`, `6->{5,7}`,
`7->{5,6,8,9}`, `8->{7,9}`, and `9->{7,8}`.

The remaining substantive Stage-60 chain is now also `COMPLETE` and validates
`PASS`. Its prerequisite is a label-blind validation cache with 2,615 rows from
44 cases, 3,840 dimensions, and the nine eligible centers. The cache persists
no labels; its content hash is `e1d281d44e47c7b2` and its frozen build-protocol
hash is `f57aad1bf7f7efed`.

The non-selecting source-inner utility artifact fits 81 source-only synthetic
classifiers, materializes predictions for every validation row before labels
are opened, then emits 648 `q != e` utility rows and 3,168 case-confusion rows.
Its utility lock is `a787b24b8e62e203`. Those rows are policy-training evidence
only: `selection_source` is source-inner validation case-confusion utility,
`prior_method` is the promoted aggregate prior `PS`, and `claim_role` is never
target evidence. The per-query best source is a non-deployable source-inner
oracle reference used only to compute regret; it neither selects a deployed
source nor represents an outer-target or Stage-70 result.

The consuming policy removes both `q = H` and `e = H` before forming 4,536
regret cells and 72 candidate summaries, then freezes nine outer-fold
selections. All nine uncertainty gates fail: best-source win probabilities
range from `0.392` to `0.786` (below `0.80`) and every paired-regret margin
lower bound is negative. The policy therefore reuses the exact canonical
equal-union assignments, streams, budgets, order, and shuffle seeds for every
`H`; it does not re-estimate a fallback. Its decision is
`FROZEN_AS_SOURCE_INNER_UTILITY_REGRET_POLICY_WITH_EXACT_EQUAL_UNION_FALLBACK`,
publication state is `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`, policy
lock is `d504ea0a07302acd`, plan hash is `cefe176313b1ea23`, and assignment hash
is `bd004f2bbb49228b`.

This is a conservative source-inner policy result, not routing-quality or
downstream-utility evidence. The next evidence is separately authorized,
fresh, matched Stage-70 target scoring of the frozen equal-union, metadata
max-tie, and utility/regret policy arms. The Stage-20 value `0.770112` must not
be presented as an expected Stage-70 result because it came from seven-source
source-inner tasks.

The equal-union artifact pins all 11 materialized files; each metadata artifact
pins 12 files; and the new cache, utility, and policy artifacts are catalogued
with their own required-file hashes. The metadata runs record revision
`40221038ca714bf33fd21582857d21fa1db4e6f3` with `repository_dirty=true` and
repository-status hash
`c92e4edfd89571b62af36c19c14f99d32664d25e87e783978f6739391d108c5b`.
The completed utility/regret policy records the same dirty revision and
repository-status hash
`b9581b6f4e31a7ed63791e9bcf9591c73223a103d54ca15b8de17b49e45f7d2d`.
Preserve the exact working-tree diff or regenerate and revalidate all
thesis-facing locks from a clean committed revision before archival.

## Quarantine And Planned Work

Repository migration audit:

```text
artifacts/midogpp/90_oracles_and_diagnostics/repository_migration/2026-07-12_xai_master/
```

The audit is `AUDIT_ONLY` and present locally and on the workstation. Its
catalog-pinned pre/post manifests match for the complete contract, corrected
cache, tuned real-feature reference, and tuned preservation bundle. Raw-data
verification preserves stable relative-path/size metadata and hashes for six
critical files. All retired workstation source artifact/data locations are
absent.

- The stale `coco_xywh` config and cache lineage are `REJECTED` and live under
  `datasets/midogpp/configs/quarantine/` and
  `datasets/midogpp/derived/features/quarantine/`.
- Rejected legacy expert-bank, prior, and routing artifacts live under
  `artifacts/midogpp/90_oracles_and_diagnostics/rejected/`; they are audit-only
  and cannot seed new stages 30 through 70.
- The retained stage-50 phase-1 bundle is a historical post-hoc diagnostic. Its
  target utility and oracle rows cannot train or select a deployable router.
- BreakHis, Camelyon17, and generic historical material is outside the active
  registry under `artifacts/cross_dataset_archive/`.
- The consensus-recipe `midogpp.expert_bank.provenance_clean.v1` path remains a
  planned alternative; its published locks do not supersede the active v2
  bank. The Stage-40 GenerationLock and all Stage-60 control/comparison locks,
  including the utility/regret policy with exact equal-union fallback, are
  complete and independently validated. A separately authorized target-
  evaluation artifact and matched Stage-70 evaluation follow. Routing quality
  and downstream utility remain unknown until that fresh evaluation is complete.

- The fixed residual-top-up B/U/G/S decomposition is now implemented and
  registered as two `planned` experiments:
  `midogpp.routing_and_composition.uniform_b_v2_residual_topup_b_u_g_s_policy_lock.v1`
  and
  `midogpp.frozen_policy_downstream.uniform_b_v2_residual_topup_b_u_g_s_fresh.v1`.
  The Stage-60 code freezes B/U/G/S, a fixed source-identity permutation
  control, and the complete `H x e` single-source-tail diagnostic menu. The
  Stage-70 code predeclares all-nine-seed probability-ensemble BACC and center-
  level `S-U`, `S-G`, `G-U`, `U-B`, and `S-B` contrasts, with all predictions
  sealed before labels. Both entries remain fail-closed and non-runnable: the
  checkout has no independently reserved, unconsumed, case-disjoint
  pseudoquery/support/evaluation surface. Consumed Stage-70 and Stage-90 bytes
  cannot satisfy the new artifact identities.

- A utility-aligned residual-router successor is implemented as a separate,
  immutable experiment family. One Stage-60 producer learns the exact response
  of the deployed additive action under strict `H/q/e` exclusion; a second,
  independently reserved producer constructs only label-free target-support
  point and whole-case-bootstrap features; and a CPU-only third job validates
  both surfaces and the distinct support/Stage-70 reservations before freezing
  the policy. The policy uses at least eight unlabeled support cases and 32
  typed whole-case bootstrap resamples per target and falls back exactly to
  `B` when uncertain. Fresh
  Stage 70 seals `B/U/G_delta/R/P` plus all `H x e` actions and requires
  positive center-level lower bounds for `R-B`, `R-G_delta`, `R-U`, and `R-P`.
  This family remains `planned` and non-runnable until its entirely new
  development, target-support, and target-evaluation artifacts exist. Its
  six-to-seven-source diagnostic is an eligibility check, not evidence that
  seven-to-eight-source target routing will improve.

- A separate same-dataset fallback is implemented and registered as
  `midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof.v1`.
  It is a runnable terminal Stage-90 diagnostic over the already-consumed
  MIDOG++ validation surface, not the planned fresh confirmation. Two fixed
  label-free support cases per center define `S_H`; the other 26 cases are
  scored exactly once as whole-case OOF folds. `G` excludes both `H` and `q`,
  `S` uses only `S_H`, and B/U/G/S/P plus every `H x e` action are sealed for
  all nine seed cells before labels open. Primary inference is over the nine
  center-level probability-ensemble `S-U` and `S-G` contrasts. The artifact is
  post-hoc, consumed-data, non-promotional, and forbidden from feeding Stage 60,
  Stage 70, recipe selection, or deployment regardless of its result.

- A sibling utility-aligned fallback is registered as
  `midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_exact_tail_router.v1`.
  It replaces proxy ranking with outer-center cross-fitted exact-tail utility
  responses under strict `H/q/e` exclusion. The all-center development-label
  phase begins only after the complete inner prediction seal, and each
  `plan_H` excludes all rows with `q=H`. Its target proposal `R2` sees exactly
  two fixed unlabeled support cases and is permanently marked
  `INSUFFICIENT_SUPPORT_FOR_POLICY`. Target actions are globally predicted
  before terminal scoring. This runnable Stage-90 experiment cannot create
  fresh evidence or feed a policy, Stage 60, Stage 70, promotion, or
  deployment path.

## Latest Uniform-B Nonlinear-Boundary Diagnostic

Canonical target:

```text
artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_nystroem_nonlinear_probe_v1/seed42/
```

Availability: `COMPLETE; VALIDATED; HASH-PROMOTED; DIAGNOSTIC ONLY`. This
bounded experiment retains canonical B and tests only a nonlinear
`StandardScaler -> Nyström RBF -> L2 logistic` boundary under exact nested
source-inner LODO. The validator reconstructs all 2,592 selector cells, 324
primary kernel transforms, baseline identity, outer metrics, bootstrap,
progression decision, and split isolation.

Result summary:

- decision: `NONLINEAR_B_DIAGNOSTIC_GATE_PASS`;
- linear-B equal-center BACC: `0.792087`;
- nonlinear-B equal-center BACC: `0.815278`;
- paired mean delta: `+0.023192`;
- strict wins: `9/9`;
- worst-center delta: `+0.002008`;
- equal-center positive-recall delta: `+0.018308`;
- equal-center specificity delta: `+0.028075`;
- supportive bootstrap interval: `[+0.011774, +0.032390]`.

The nonlinear boundary rescues 751 linear errors and introduces 550, for a
net 201-row rescue. It resolves 613 of 1,025 baseline errors for which the
source-only class centroid was already closer to the true class. This supports
a nonlinear-separability limitation rather than an immediate need to replace
B with C or B-spatial.

The result remains post-hoc diagnostic evidence on an already inspected train
surface. It does not prove that B is sufficient or authorize automatic B+
adoption. Center 2 loses `0.123972` specificity while gaining substantial
positive recall, and 1,318 rows remain wrong under both boundaries. These are
the priority follow-up analyses.

The validation split remains unfeaturized and unscored. Its 2,615 eligible
rows comprise only 44 cases and 3–7 cases per center, below the frozen
ten-case-per-center confirmation minimum. Formal confirmation requires a
larger untouched, external, or genuinely new-center surface.

## Latest Uniform-B Robust Interaction Diagnostic

Canonical target:

```text
artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_robust_interaction_probe_v1/seed42/
```

Availability: `COMPLETE; VALIDATED; HASH-PROMOTED; DIAGNOSTIC ONLY`. The
paired audit covers 751 rescues, 550 regressions, and 1,318 shared hard-core
errors. Of the regressions, 246 are near-boundary and only two are confidently
wrong. Center 2 contains 257 false-positive regressions.

Robust Nyström retains `+0.020437` equal-center BACC over linear B and loses
only `−0.002755` versus standard Nyström. It improves the center-2
specificity exchange, but center-9 recall is `−0.127168` versus linear B,
violating the frozen `−0.05` class-direction floor. The minimal bilinear model
is `−0.006267` below linear B and wins only two centers.

Decision: `NO_FAMILY_PASSES_ROBUST_BPLUS_GATE`. No final B+ protocol is
frozen. Generic group weighting moves the limiting error from center 2
specificity to center 9 recall, while the low-rank bilinear interaction does
not explain the Nyström gain.

Validation and test remain untouched. The next justified work is a bounded
source-inner constrained-sensitivity/specificity objective or deeper review
of the 1,318-case hard core—not B-spatial by default.

## Latest constrained nonlinear-B diagnostic

The source-inner sensitivity/specificity-constrained Nyström successor is
complete and independently validated at:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_sens_spec_constrained_nystroem_probe_v1/seed42/
```

Decision: `NO_CONSTRAINED_BPLUS_CANDIDATE_PASSES`. Seven of nine outer recipes
had no feasible nonlinear source-inner candidate and fell back to exact linear
B. Only centers 6 and 9 selected `alpha=0.25`. The equal-center mean BACC gain
is `+0.00282`, with only `2/9` strict wins and a `−0.03468` worst recall delta.

This means the original nonlinear gain cannot presently be retained under the
frozen uniform class-direction constraints. No final B+ protocol is promoted.
The result is post-hoc and diagnostic; validation and test remain untouched.
The next work should examine why feasibility collapses across seven centers,
rather than relaxing the constraint or changing the global threshold.
