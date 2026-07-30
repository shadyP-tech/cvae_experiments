# Current Experimental State

Last updated: 2026-07-27

This page records verified evidence and canonical availability after the
completed MIDOG++ repository migration. Active inputs and the two tuned
evidence bundles are present locally and on the workstation at canonical paths
with cataloged hash verification. The matched Stage-10 reference, Stage-10
conditional-logit alignment diagnostic, scalar Stage-20 source-inner result,
and bounded training-seed stability bundle are validated on `xai-master` but
have not yet been synced locally.

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
  validated, and fully hash-promoted. Stage 10 is ready but has not run.
- Stages 30 through 70 have no active expert-bank, generation, routing, or
  frozen-policy downstream implementation. Their directories and protocol
  contracts are planning scaffolds, not experimental evidence.
- The Stage-10 matched-reference v2, Stage-10 conditional-logit alignment
  diagnostic, Stage-20 scalar source-inner prior-recovery/Task-Fisher run, and
  bounded Stage-20 training-seed stability run are complete on `xai-master`
  and protocol-clean. These four workstation bundles have not yet been synced
  into this local checkout.
- The Stage-90 Uniform-B low-noise paired audit is implemented, registered, and
  protocol-reviewed locally, but its code is not yet synced to `xai-master`
  and neither registered output exists. It is not result evidence.
- The stability bundle is a valid, published Stage-30 recipe input. Stage 30
  remains planned and has no runnable expert-bank implementation. The
  registered Stage-20 outer run remains blocked by the scalar source-inner
  gate.

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

Availability: `READY TO RUN; NOT RUN`. The code, configs, registry/catalog
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
Stage-10 result exists yet. The pilot cannot establish CVAE preservation,
calibration, probabilistic or NELBO validity, routing, deployment, new-center
generalization, or any Stage-20 through Stage-70 decision. V1 and v2 remain
non-runnable failed-audit lineages and are not fallbacks.

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
| 20 | `midogpp.cvae.aggregate_posterior_mixture_geco_source_inner.v3` | `artifacts/midogpp/20_cvae_preservation/aggregate_posterior_mixture_geco_source_inner_v3/seeds17_42_101/` | `IMPLEMENTED AND REGISTERED`; not production-run; independently trained source-local four-arm study; non-consumable until a separate promotion |
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
remain unchanged. Stage 30 now has an eligible input, but its registry entry is
still a planned placeholder without a runnable expert-bank implementation.
Stop open-ended Stage-20 tuning. The bounded v3 independent-source
prior-mismatch study is the explicit exception: its fixed four-arm contract
tests aggregate-posterior coordinate matching plus GECO and cannot change a
recipe without a separate promotion artifact.

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
published consensus bundle whether or not either v2 study is run. A future v2
result may support only its named source-inner mechanism/stability question;
it is not outer-preservation, generation, routing, or downstream-utility
evidence.

## Workstation-only Variant-B diagnostics

Two completed Stage-90 diagnostics examine a separate canonical-B CVAE
representation on deterministic train-case holdouts. Both are
`diagnostic_only`; neither exports a `RecipeLock` nor may feed the expert bank,
generation, routing, or downstream utility. They are not Stage-20 preservation
evidence and do not alter the published Stage-30 consensus-lock input.

The artifacts remain workstation-only under
`/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/`:

| Remote artifact root | Remote experiment namespace | Status and decision | Claim boundary |
| --- | --- | --- | --- |
| `uniform_b_source_expert_adaptation_pilot_v2/` | `midogpp.oracle.uniform_b_source_expert_adaptation_pilot.v2` | `COMPLETE`, validation `PASS`, `B_ADAPTATION_NOT_FEASIBLE` | `diagnostic_only`; `may_feed_expert_bank=false` |
| `uniform_b_block_tail_average_stability_probe_v1/` | `midogpp.oracle.uniform_b_block_tail_average_stability_probe.v1` | `COMPLETE` (`12/12`, zero failures), validation `PASS`, `TAIL_AVERAGING_INSUFFICIENT` | `diagnostic_only`; `may_feed_expert_bank=false` |

The pilot's decision and validation reports are
`uniform_b_source_expert_adaptation_pilot_v2/reports/pilot_decision.json` and
`uniform_b_source_expert_adaptation_pilot_v2/reports/validation_report.json`.
It evaluates the `a_global_pca128`, `b_joint_pca128`, and
`b_block_pca96_32` readouts across centers `2,5,6,9` and training/generation
seeds `17,42,101`. The decoded B-block mean BACC is `0.745109`, versus
`0.728308` for B-joint and `0.645872` for A-global. This representation signal
is not sufficient for adoption: the pilot's B-block seed mean-preservation
range is `0.069490` and its conservative prior is not viable. The conditional
prior improved B-block prior-sample mean BACC from `0.698459` (standard normal)
to `0.801297`, but center-9 conditional-prior recall averaged `0.531467` and
reached `0.468750`, below its `0.55` viability floor.

The tail-average decision and validation reports are
`uniform_b_block_tail_average_stability_probe_v1/reports/stability_decision.json`
and
`uniform_b_block_tail_average_stability_probe_v1/reports/validation_report.json`.
It is a fresh exact v2 endpoint replay of only B-block jobs, averaging uniform
FP32 post-update parameter states from steps `751` through `1000`. Validation
passes lineage hashes, terminal-control replay, tail-average derivation,
prediction/metric reconciliation, decision recomputation, and the claim
firewall. Tail averaging preserved mean decoded behavior but slightly reduced
mean BACC from `0.745109` to `0.741535` (`-0.003574`). It passed `20/22`
predeclared gates, but failed the seed mean-preservation range (`0.082269` vs
maximum `0.05`) and maximum center/class-direction seed range (`0.192771` vs
maximum `0.15`). The latter is driven by center-9 specificity; center-5 recall
also remains variable (`0.166667`).

Interpretation: tail averaging is not a viable stabilization or promotion
intervention for this B-block recipe. Do not run the conditionally named
prior-only replay, promote its checkpoints or priors, or use either diagnostic
as a Stage-30 input. The active eligible Stage-30 input remains the separately
published bounded training-seed consensus `RecipeLock` bundle described above.

The next bounded Variant-B study is now implemented and registered as a
two-step Stage-90 workflow:

| Experiment | Canonical output | Current status |
| --- | --- | --- |
| `midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1` | `artifacts/midogpp/90_oracles_and_diagnostics/inputs/uniform_b_paired_reparameterization_snapshot_v1/` | implemented; not run |
| `midogpp.oracle.uniform_b_paired_reparameterization_audit.v1` | `artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_paired_reparameterization_audit/v1/` | implemented; blocked on snapshot; not run |

It compares an exact legacy replay, a fold-fixed one-epsilon control, and a
fold-fixed antithetic estimator over the same four centers and three
initialization seeds. The controlled arms share schedules and epsilon traces;
initializations remain distinct across seeds. The audit stays
`diagnostic_only`, and its outputs cannot revise the active consensus locks or
feed the expert bank, prior/generation, routing, or downstream stages. No
result should be documented until both emitted bundles validate `PASS`.

Reproducibility note: a component-hash path defect was corrected before these
completed artifacts were produced. Component hashing had expected
`cvae/models.py`, although the model is packaged under `cvae/models/`. The
passing frozen manifests bind `dependency/models/__init__.py` and
`dependency/models/cvae.py`; the tail-average validator's `lineage_hashes`
check is `PASS`. This repairs provenance binding only; it does not change the
diagnostic-only claim boundary or make an earlier failed run promotable.

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
- Adoptive Stage-20 tuning stops with the completed bounded stability panel.
  The two v2 mechanism studies remain separate and non-adoptive. The immediate
  deployable-pipeline implementation step is the provenance-clean independently
  trained MIDOG++ Stage-30 source-expert bank using the published consensus
  locks. Routing remains premature until that bank and a fresh protocol-clean
  utility surface exist.
