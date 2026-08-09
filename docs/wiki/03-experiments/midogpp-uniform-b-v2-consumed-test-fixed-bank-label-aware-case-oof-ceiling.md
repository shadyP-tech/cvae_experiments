# MIDOG++ fixed-bank label-aware case-OOF ceiling

Experiment:
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling.v1`

Status: quarantined failed terminal Stage-90 diagnostic. The workstation sealed
the full probability surface but stopped before any fold decision because v1
required both binary classes inside every individual case. MIDOG++ contains
four negative-only and one positive-only case. No canonical closed-world v1
result exists, and its output, partial label-capability history, scratch, and
checkpoints must not be resumed or reused.

## Question

Earlier routing diagnostics asked label-free proxies to predict a small and
noisy utility difference. This ceiling instead asks whether the known frozen
expert bank becomes routable when a strictly local amount of target-label
information is available. It separates a missing-information bottleneck from
a classifier or composition-capacity bottleneck; it is not a deployable
label-free router.

## Frozen direct-target design

- The experiment independently materializes `B` plus all eight legal direct
  `Hxe` actions for each target center `H`. The target expert is excluded.
- All 81 target/action identities and all 729 training-seed/generation-seed
  probability cells are sealed globally before any label capability opens.
- Each target center is partitioned deterministically into five whole-case
  folds under seed `90902026`. Every one of the 218 cases is evaluation in
  exactly one fold and is absent from that fold's support and decision fit.
- A separate `label_derived_LOCO_global_prior`, `G_H`, uses labels only from
  centers other than `H`. There is no shared all-center prior: `G_H` excludes
  every `H` label, has fixed predeclared hyperparameters, and is sealed before
  same-center support labels can be read.
- For fold `k`, only the other four folds of the same center may update the
  local posterior, using exact per-case candidate-minus-`G_H` BACC. Other
  centers contribute equally—not in proportion to case count—to `G_H`. No
  support label updates an expert, generator, classifier, shared router, or a
  different center's local posterior.
- The fixed posterior uses prior strength `8`, variance floor `1e-6`, a `1.96`
  confidence multiplier, zero minimum gain,
  and lexical action-ID tie breaks. `R` abstains through `G_H` to exact `B`
  when evidence is insufficient.
- All 45 `(H, fold)` decisions are sealed before the evaluation-label
  capability opens. The complete dependency and capability seals are
  persisted so a resume cannot weaken this ordering.

## Endpoint and controls

The primary utility is exact BACC after averaging the nine probability vectors
and thresholding once at `0.5`. The report compares `R-G_H`, `R-B`, and
`G_H-B`, along with regret, top-1/tie-aware agreement, coverage, and source
selection concentration. Inference keeps target center—not a technical seed
or row—as the outer unit. Ten thousand deterministic candidate-label
permutations derange the eight sources separately inside each
`(H, fold, support case)` block. `B` stays fixed, the eight-`Hxe` multiset is
preserved, and no evaluation case is a donor. Pre-evaluation null decisions
use the same deterministic lexicographic action-ID tie break as the observed
decision rule; they never inspect exact evaluation utility. Every null action
is durably sealed before evaluation labels open. This avoids the degenerate
null obtained by merely moving complete action vectors between cases.

Smooth metrics may be written only as post-seal descriptive tables. They have
no dependency path into a prior, posterior, candidate choice, threshold,
inferential gate, or publication decision.

## Reused test-set boundary

The user explicitly authorized this one additional use of the already-consumed
MIDOG++ test split. A dedicated hash-chained amendment names only this
experiment. Its six-input fence contains the frozen bank, GenerationLock,
dedicated test-cache and manifest aliases, the immutable parent ledger alias,
and that amendment. It reads no metadata artifact, Stage-50/60/70 output, or
previous Stage-90 result.

The result is always `EXPLORATORY_CONSUMED_DATA_ONLY`. Even a positive result
would show only that local labels contain exploitable routing information for
the known bank on these consumed centers. It cannot establish fresh routing
quality, authorize an action or policy, update any model, feed another stage or
experiment, select a recipe, promote a method, or support deployment.

## Workstation execution

The run uses one persistent generation worker on each A5000, keeps the parent
CUDA-free, disables AMP and TF32, then runs four spawned classifier workers
with three BLAS threads each. GPU and CPU pools are phase-disjoint. Float32
memmaps and hash-validated atomic checkpoints are staged under
`/data/local/fixed_bank_label_aware_case_oof_ceiling_v1` before canonical
persistence.

Run through the workspace registry:

```bash
python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_fixed_bank_label_aware_case_oof_ceiling/v1/
```
