# Stage 60: Routing and Composition

This stage selects or composes source experts using only allowed
pre-evaluation information. Unlabeled target support must be disjoint from
target evaluation, the target expert must be excluded, and routing must not
modify source checkpoints.

The complete routing policy is frozen before stage 70. Stage 50 utilities and
stage 90 oracle identities are forbidden inputs.

Status: `COMPLETE; VALIDATED; CONTROL AND BOTH COMPARISON POLICIES FROZEN FOR
MATCHED STAGE70 EVALUATION`. The authorized v2 bank is the expert input, and
`midogpp_output_uniform_b_v2_generation_lock_v1` is the only generation-settings
artifact authorized for deployable selection. `generation_diagnostics_only`,
the Stage-20 source study, Stage-50 utilities, and Stage-90 oracle identities
remain forbidden as direct routing inputs.

The registered
`midogpp.routing_and_composition.uniform_b_v2_equal_union_policy_lock.v1`
experiment freezes `uniform_b_v2_equal_union_ps` as the direct control: all
eight non-target source centers, 1,024 samples per class, exactly 128 per
source, training and generation seeds `17,42,101`, and no target-conditioned
weighting or seed selection. It consumes the validated bank and GenerationLock
and preserves the locked source-stream RNG, shuffles, classifier budget, and
replicate identities.

Target-center identity is structural only: it defines the held-out fold,
excludes that center's expert, and seeds the predeclared label-blind within-class
shuffle. It is never a predictive feature, compatibility score, rank, or source
weight.

This Stage-60 artifact is a policy contract, not a performance result. It uses
no target samples, support rows, or labels, and it reports no BACC, macro-F1,
routing quality, or downstream synthetic utility. The eight-source control
must be generated and scored afresh in Stage 70 because it is not the seven-
source source-inner task used for bank promotion. The first comparison policy
is now frozen, but Stage-70 scoring must still wait until a separate target-
evaluation artifact is authorized so that eligibility, budgets, seeds, and
evaluation rows remain matched.

The canonical workstation artifact is:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_equal_union_policy_lock/v1/
```

The production run and an independent validator rerun both report `PASS`. The
decision is `FROZEN_AS_CANONICAL_EQUAL_UNION_ROUTING_CONTROL`, publication
state is `POLICY_FROZEN_FOR_STAGE70_EVALUATION`, and the bundle contains 81
target/seed replicates and 648 target-excluded source assignments. Its policy
lock is `4b9ea514308b084f`, policy-plan hash is `9ec24122d7d0cdf1`, and
assignment-table hash is `c85415c1b953c04e`. The artifact catalog pins the
SHA-256 of every one of its 11 files.

The first comparison is a two-artifact, fail-closed metadata baseline. The
non-selecting compatibility artifact parses only the hash-pinned routing-time
fields `tumor_type`, `lab_or_origin`, and `scanner_model`, then records the
unweighted componentwise exact-match count for all 72 ordered target-excluded
pairs. Center or domain IDs are not passed to the scorer. The artifact emits no
rank, selected source, weight, NELBO, or utility:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_exact_match_compatibility/v1/
```

Its run and independent validation are `PASS`; decision is
`FROZEN_AS_METADATA_PROXY_COMPATIBILITY_INPUT`; compatibility lock is
`4b46b3d157b07781`; and all 12 files are SHA-256-pinned.

The consuming policy retains every source tied at the maximum exact-match
score, with canonical center order used only for deterministic ordering. Its
selected source sets are `0->{5}`, `1->{2}`, `2->{1}`, `3->{1,2}`,
`5->{6,7}`, `6->{5,7}`, `7->{5,6,8,9}`, `8->{7,9}`, and `9->{7,8}`. The fixed
1,024-per-class budget is divided equally within each tie: 1,024, 512, or 256
rows per selected source. It reuses the exact Stage-40 source-stream prefixes
and class-shuffle seeds and retains every training/generation seed replicate:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_tie_union_policy_lock/v1/
```

Its run and independent validation are `PASS`; decision is
`FROZEN_AS_METADATA_EXACT_MATCH_TIE_UNION_COMPARISON_POLICY`; publication state
is `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`; policy lock is
`27f16953b32c46cd`; and the bundle contains nine selections, 81 replicates, and
153 assignments. All 12 files are SHA-256-pinned.

This comparison is a metadata-similarity proxy policy, not evidence that the
selected sources are useful. It consumes no target samples, support rows,
target labels, Stage-50 utility, or Stage-90 oracle data and computes no BACC,
macro-F1, routing advantage, or downstream utility. Its value must be measured
only by a fresh, paired Stage-70 evaluation against the equal-union control.

The substantive source-inner comparison is now also complete. Its direct
label-blind prerequisite is the immutable 3,840-dimensional validation cache:

```text
datasets/midogpp/derived/features/virchow2/uniform_b_v2_routing_validation_cache_v1/seed42/
```

The cache contains 2,615 rows from 44 cases across all nine eligible centers,
persists no labels, and independently validates `PASS`. Validation labels were
reserved once for the predeclared policy family and were opened only after all
81 source-only classifiers had materialized their predictions. The resulting
non-selecting utility artifact contains all 648 `q != e` training/generation-
seed cells and 3,168 paired case-confusion rows:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_source_inner_candidate_utility/v1/
```

Its utility lock is `a787b24b8e62e203`. These rows are source-inner policy-
training evidence only. Their per-query best source is a non-deployable oracle
reference used solely to define regret; it is not an outer-target oracle or a
Stage-70 result. The selection source is the frozen source-inner validation
case-confusion utility, the generation prior is the promoted aggregate prior
PS, and macro-F1 is descriptive only.

The consuming policy removes both `q = H` and `e = H` before constructing
4,536 regret cells and 72 outer candidate summaries. It then applies the
predeclared pseudo-target/case/paired-seed bootstrap with 2,000 valid
replicates per outer center:

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_utility_regret_policy_lock/v1/
```

No single source passed the uncertainty gate. Best-source win probabilities
ranged from `0.392` to `0.786`, below the required `0.80`, and every 2.5%
paired-regret margin bound was negative. Consequently all nine outer folds
reuse the exact frozen equal-union assignments, streams, budgets, order, and
shuffle seeds. The decision is
`FROZEN_AS_SOURCE_INNER_UTILITY_REGRET_POLICY_WITH_EXACT_EQUAL_UNION_FALLBACK`,
publication state is `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`, policy
lock is `d504ea0a07302acd`, plan hash is `cefe176313b1ea23`, and assignment-
table hash is `bd004f2bbb49228b`.

This is a valid conservative routing result: the evidence did not justify
source-specific routing. It does not claim that equal-union is good, that the
metadata policy is better or worse, or that routing improves target utility.
The next target is a separately authorized, fresh, matched Stage-70 evaluation
of the frozen equal-union, metadata max-tie, and utility/regret policy arms.

The runs record revision `40221038ca714bf33fd21582857d21fa1db4e6f3` with
`repository_dirty=true`; the final policy provenance records repository-status
hash `b9581b6f4e31a7ed63791e9bcf9591c73223a103d54ca15b8de17b49e45f7d2d`.
The artifact bytes are hash-locked, but the exact working-tree diff must be
preserved or the runs regenerated from a clean revision before thesis archival.
