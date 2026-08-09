# Protocol Status

Last updated: 2026-08-09

## Active MIDOG++ surfaces

- Dataset contract and runtime validation:
  `src/midogpp_thesis/data/contract/`
- Real-feature controls and tuned classifier reference:
  `src/midogpp_thesis/real_features/`
- CVAE preservation surfaces:
  `src/midogpp_thesis/cvae/preservation/`
- Stage-30 expert-bank promotion and fail-closed loading:
  `src/midogpp_thesis/cvae/expert_bank/`
- Stage-40 generation locking and validation:
  `src/midogpp_thesis/cvae/generation/`
- Stage-60 policy locking and validation:
  `src/midogpp_thesis/cvae/routing/`
- Registry, artifact resolution, and fail-closed launch preparation:
  `src/midogpp_thesis/workspace/`

The active regime is MIDOG++ with Virchow2 embeddings, corrected `xyxy` patch
geometry, case-disjoint splits, and center held-out evaluation over eligible
centers `0,1,2,3,5,6,7,8,9`. Target-evaluation labels are scoring-only.

The working tree contains active implementations for:

- the eligible-only, predict-policy Stage-10 matched reference v2;
- the audit-blocked v1/v2 physical-multiscale histories and the pre-activation,
  non-adoptive v3 clipped-bbox annotation-local Stage-10 pilot;
- the Stage-90 fixed-B v3 retrospective replay, isolated from downstream
  consumption;
- the Stage-90 prospective fixed-B confirmation on case-disjoint test rows,
  also isolated from downstream consumption;
- fully nested Stage-20 source-inner ex-post-prior and Task-Fisher
  `RecipeLock` selection;
- the bounded Stage-20 training-seed stability panel and fold-level consensus
  `RecipeLock` export;
- the separate non-adoptive learned class-conditional diagonal-prior
  source-inner v2 study;
- the separate non-adoptive Task-Fisher shrinkage source-inner v2 study;
- the completed Uniform-B GECO aggregate-prior union v2 source-inner study;
- the completed Uniform-B v2 Stage-30 promotion, validation, and
  routing-authorized expert loader;
- the completed Uniform-B v2 Stage-40 GenerationLock;
- the completed Uniform-B v2 Stage-60 equal-union, metadata, source-inner
  utility, and uncertainty-gated utility/regret policy locks;
- the runnable consumed-data Stage-90 utility-aligned exact-tail diagnostic,
  whose fixed-two-case `R2` arm is always
  `INSUFFICIENT_SUPPORT_FOR_POLICY` and cannot feed later stages;
- the endpoint-aligned successor implementation, whose primary Stage-60
  response is one all-nine probability-ensemble BACC delta per `(H,q,e)`, not
  nine seed-cell BACC deltas, and whose only target-local predictor is a
  label-free exact-action support probability shift;
- a conditionally unlocked Stage-20 outer A/B/C/D preservation factorial.

The first three runs are complete on `xai-master` and pass their full
validators. The Stage-10 v2 result is a matched
`real_feature_transfer_only` denominator. The scalar Stage-20 source-inner
result is a valid `cvae_recipe_lock_only` bundle with seven conditional locks
and standard-normal fallbacks for centers `5` and `9`. Its gate is
`NEGATIVE_GATE_COMPLETE` with `factorial_triggered=false`, so the registered
outer factorial is blocked and provides no outer-preservation result. The
bounded training-seed stability bundle is `COMPLETE` and `PUBLISHED`, with
`27/27` valid child locks, `9/9` export-ready consensus locks, and
`stage30_recipe_ready=true`. These workstation bundles have not yet been
synced locally, and their catalog output entries still carry stale
`TODO_VERIFY_ARTIFACT` lifecycle labels pending catalog promotion.

The two v2 mechanism studies are active implementations but remain unrun. They
fully cross training and generation seeds `17,42,101` over the nested nine-by-
eight center structure. Their only allowed claim scope is
`cvae_source_inner_study_only`: study decisions have recipe and deployable
consumption fixed false. They cannot change the scalar v1 gate, the published
stability consensus, outer v1, or the registered Stage-30 input.

The completed Uniform-B aggregate-prior union v2 study is a separate source-
inner lineage. Its PS arm reaches mean BACC `0.770112`, improves over P0 by
`+0.012764`, and lies `0.001459` below the posterior-sample ceiling. The
minimum training-seed PS mean is `0.764562`. Its direct publication remains
non-consumable, but the separate reviewed Stage-30 promotion passed every
gate and consumed that evidence for whole-bank adoption.

The promoted artifact is `COMPLETE`, validation is `PASS`, and publication
state is `ROUTING_AUTHORIZED`. It retains all 27 independently trained
source/seed experts without individual selection. The bank may feed future
deployable-selection experiments; the Stage-20 source artifact may not.

The Stage-40 GenerationLock and the Stage-60 equal-union, metadata, and
utility/regret locks are
`COMPLETE`; their production runs and independent validator reruns report
`PASS`. The equal-union control freezes 81 target/seed replicates and 648
target-excluded assignments under policy lock `4b9ea514308b084f`. The
non-selecting metadata compatibility artifact freezes 72 target-excluded
exact-match proxy scores under lock `4b46b3d157b07781`. The max-tie policy then
retains every maximum-score tie and freezes nine selections, 81 replicates,
and 153 assignments under lock `27f16953b32c46cd`. The completed source-inner
utility surface is label-blind until its 81 predictions are materialized, then
uses one-time validation labels only to produce 648 `q != e` utility and 3,168
case-confusion rows under lock `a787b24b8e62e203`. Its sole consumer creates
4,536 regret cells and 72 candidate summaries, then freezes all nine folds as
the exact equal-union fallback: no best-source win probability reaches `0.80`
and all margin lower bounds are negative. The policy lock is
`d504ea0a07302acd`; publication state is
`POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`. No Stage-60 artifact supports
a routing-quality or target downstream-utility claim.

## Current thesis-facing evidence

The tuned-classifier preservation result supports only
`cvae_preservation_only`: decoded means and posterior samples preserve most of
the frozen PCA128 real-feature classifier surface. It does not support routing,
expert selection, NELBO compatibility, prior generation, controllable
generation, or downstream synthetic utility. Prior sampling remains materially
weaker and is a separate bottleneck.

The training-seed panel is thesis-facing only for the narrow
`cvae_recipe_lock_only` stability claim. Exactly unanimous recipe selection is
limited to centers `6` and `7`; centers `0,1,2,3,5,8,9` are unstable under the
predeclared consensus rule. This is a `NEGATIVE_RESULT` for broad recipe
stability and an operational `PASS` for publishing a conservative complete
recipe bank. It is not outer-preservation, routing, generation, or downstream
utility evidence.

The Stage-30 promotion supports the narrow
`expert_bank_construction_only` claim. It establishes a validated bank of
independently trained experts and freezes equal-union PS as the future direct
control. It does not establish routing, compatibility estimation, generation
quality on outer targets, or downstream synthetic utility. Source-inner
labels used for whole-bank adoption may not be reused to select an expert,
seed, or routing policy.

Stage 40 supports only the frozen generation-contract claim. Stage 60 supports
only that the all-eligible-source equal-union control, metadata compatibility
proxy/max-tie comparison, and predeclared source-inner utility/regret policy
are fully specified, target-excluded, and frozen before evaluation. The utility
rows are policy-training only, and their per-query best source is a
non-deployable regret oracle, not target evidence. Neither stage establishes a
held-out BACC, a routing advantage, or downstream utility.

## Planned surfaces

Physical-multiscale v1 and v2 remain `planned` only as non-runnable
failed-audit histories. V3 is registered as `diagnostic`; its label-blind
`xai-master` source audit passes all 9,648 eligible rows and 216 TIFFs with 84
clipped bboxes, no row exclusion, and no synthesized pixels. Its immutable
contract and atomic B/C feature bundle are independently validated and have
complete required-file SHA-256 coverage. The Stage-10 run is complete and
validates: the nested adaptive policy selected B for six centers, C for two,
and A for one and improved mean BACC by `+0.044091` over A. All three lineages
remain forbidden from Stage 20 through Stage 70.

The Stage-90 uniform-B replay is also complete and validates. It imports the
nine frozen source-v3 classifier locks and applies B to every previously scored
outer center, producing mean BACC `0.792087` and a paired `+0.051775` over A,
with eight of nine strict wins. Because the decision to test B uniformly was
informed by these same outer results, its status is `POSTHOC_DISCOVERY`, its
catalog label is `DIAGNOSTIC ONLY`, and `independent_confirmation=false`.
Neither its conditional bootstrap nor its deterministic replay covers
representation-choice or new-center uncertainty.

Phase B freezes the hypothesis before test-B extraction and evaluates 9,928
case-disjoint test rows from the same nine centers. The result is
`CONFIRMED_WITHIN_CENTER`: B reaches mean BACC `0.799159` versus `0.735733` for
A (`+0.063426`), wins all nine centers, and has a positive worst-center delta
of `+0.010638`. The 95% conditional paired case interval is
`[+0.050709, +0.073650]`; all four predeclared confirmation checks pass.

This resolves representation-choice uncertainty for new cases sampled from
the observed centers more convincingly than the retrospective replay. It does
not resolve center-sampling uncertainty, external-dataset transfer, or
deployment utility. Phase B therefore remains `DIAGNOSTIC ONLY` in Stage 90
and is forbidden from revising the matched denominator or feeding later stages.

Stages 30 and 40 plus the equal-union control, metadata comparison, and
utility/regret policy in Stage 60 are active and validated. The remaining
planned surfaces are:

- the alternative consensus-recipe provenance-clean v1 bank
- all-candidate utility matrix for a clean bank
- a separately authorized target-evaluation artifact
- matched Stage-70 control-versus-metadata-versus-utility-router downstream utility
- the planned fixed residual-top-up B/U/G/S Stage-60 lock and fresh Stage-70
  decomposition, blocked until a new unconsumed case-disjoint
  pseudoquery/support/evaluation surface exists
- the planned utility-aligned exact-tail and label-free target-support Stage-60
  producers, CPU-only policy lock, and fresh Stage-70 `B/U/G_delta/R/P`
  confirmation, blocked until their separate fresh development, support-parent,
  and target-evaluation MIDOG++ artifacts exist; their aliases must be newly
  hash-bound and whole-case disjoint even though the dataset family remains
  MIDOG++. The exact-tail producer seals a 504-row candidate endpoint and keeps
  4,536 seed rows descriptive. The target-support producer runs 81 resumable
  label-free action-probe tasks (729 fits) and builds 32 whole-case bootstrap
  surfaces. Failed transfer/uncertainty gates are valid and freeze exact `B`
  rather than authorizing a routed action

The separately registered
`midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof.v1`
is runnable only as a terminal Stage-90 fallback on the existing MIDOG++
validation bytes. It freezes two label-free support cases per center, excludes
each of the other 26 whole cases from its own route, seals B/U/G/S/P and all
`H x e` predictions before labels, and performs center-level `S-U`/`S-G`
inference without a selector or fallback. It may diagnose whether the observed
mechanism is global or target-conditioned, but it cannot supply fresh evidence
or authorize routing, promotion, Stage 60, or Stage 70.

The independent
`midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_proxy_information_audit.v1`
is implemented but has no canonical result yet. It cannot read the earlier
Stage-90 router bundle. Its own experiment-fenced aliases regenerate and seal
the development probability surface before labels, after which 504 exact-nine
candidate responses screen fixed compact proxy families using 120-row folds
with `H/q/e` excluded from outer, query, and source roles. Two support cases remain diagnostic-only, seed cells are not
inference units, no target actions or labels exist, and even a positive screen
cannot authorize policy construction or downstream confirmation.

Legacy source trees were retired rather than exposed as fallbacks. Any new
implementation must use only the canonical dataset/cache lineage and must pass
target-leakage, expert-isolation, candidate-exclusion, and proxy-versus-utility
checks before activation.

For the separate planned `midogpp.expert_bank.provenance_clean.v1` path, the
published training-seed consensus `RecipeLock` files remain eligible to
feed the fold-matched Stage-30 expert recipe. Consensus is `A` for centers
`0,1,2,3,5,9`, `D` with the full conditional sampler for centers `6,7`, and
`C` with the full conditional sampler for center `8`. The bundle's protocol
hash is `bbde3e5c5a1e3374`, and its selection-bundle hash is
`79cb9b614779c23b`. Full bundle, leakage, identity-overlap, and RNG validation
passes, and the Stage-30 loader accepts the publication.

Scalar seed-42 locks remain evidence for the completed source-inner result but
are no longer the input to that planned path. Its consumption remains
fail-closed: the stability bundle must be `PUBLISHED`, and every consensus lock
must be valid and export-ready before the loader returns lock `H` for fold `H`.
That alternative runner remains unimplemented and is distinct from the active
Uniform-B v2 promotion. Outer v1 still consumes the scalar source-inner bundle
and retains its original all-conditional/factorial gate. Outer preservation
metrics and decisions may never feed model, sampler, expert, routing, or
composition selection. Stage 40 now has a complete, independently validated,
target-data-free GenerationLock artifact. It freezes settings and records 162
passing integrity health checks only; it does not establish an eight-source
BACC, routing result, or downstream utility claim. Stage 60 now has complete,
independently validated equal-union, metadata-compatibility/max-tie, and
source-inner utility/regret locks. The latter uses source-inner case-confusion
evidence only and falls back exactly to equal-union for every outer fold. These
artifacts contain no target evaluation or performance result.

The older v2 mechanism studies may run independently of the active Uniform-B
bank and the planned consensus-recipe bank; they are prerequisites for
neither. A fully executed v2 bundle may
still report `INVALID_INCOMPLETE`, for example when a raw Fisher state is
invalid. In that case `run_state=COMPLETE` means execution and mechanical
bundle validation finished, not that a scientific mechanism decision passed.

## Quarantine and historical evidence

- `artifacts/midogpp/90_oracles_and_diagnostics/` contains rejected MIDOG++
  lineages and non-deployable diagnostics.
- `artifacts/cross_dataset_archive/` contains BreakHis, Camelyon17, and generic
  historical outputs outside the MIDOG++ registry.
- Historical paths embedded in immutable manifests remain provenance strings;
  they are not active resolution fallbacks.

The complete 22,569-patch contract, corrected feature cache, and two tuned
evidence bundles are present locally and on the workstation at their canonical
paths. Dataset validation passes, cache split counts align exactly, and
cataloged authoritative hashes match. The approximately 65 GB raw source tree
is workstation-only at `datasets/midogpp/raw/MIDOGpp/` by design.

The repository-migration audit is `AUDIT_ONLY` at
`artifacts/midogpp/90_oracles_and_diagnostics/repository_migration/2026-07-12_xai_master/`.
Its pre/post manifests match, and all retired workstation source artifact/data
locations are absent.

After installing this repository with
`conda run -n thesis python -m pip install -e .`, the canonical protocol check
is:

```bash
conda run -n thesis python -m midogpp_thesis workspace validate
```

## Constrained Uniform-B classifier status

`uniform_b_sens_spec_constrained_nystroem_probe_v1` is complete, validated,
and diagnostic only. It keeps the global threshold fixed at `0.5`, selects
capacity using source-inner folds only, and records that the study design was
informed by previously inspected outer outcomes.

The decision is `NO_CONSTRAINED_BPLUS_CANDIDATE_PASSES`; it does not authorize
canonical-reference replacement, validation scoring, test scoring, recipe
selection, or deployable selection. Seven centers use the exact-linear
fallback and only two retain nonlinear capacity.
