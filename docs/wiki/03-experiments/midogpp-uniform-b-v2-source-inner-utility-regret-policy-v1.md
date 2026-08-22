# MIDOG++ Uniform-B V2 Source-Inner Utility/Regret Policy V1

## Result

The completed Stage-60 source-inner utility/regret chain validates `PASS` and
freezes a conservative policy for future matched Stage-70 scoring.

- Decision: `FROZEN_AS_SOURCE_INNER_UTILITY_REGRET_POLICY_WITH_EXACT_EQUAL_UNION_FALLBACK`
- Publication: `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`
- Utility lock: `a787b24b8e62e203`
- Policy lock: `d504ea0a07302acd`
- Policy plan hash: `cefe176313b1ea23`
- Assignment-table hash: `bd004f2bbb49228b`

All nine outer folds fail the predeclared unique-winner gate. Best-source win
probabilities range from `0.392` to `0.786`, below the required `0.80`, and
every 2.5% paired-regret margin lower bound is negative. The frozen action for
every held-out target `H` is therefore the **exact** canonical equal-union
fallback: the existing assignments, source streams, total budget, deterministic
order, and shuffle seeds are reused without re-estimation.

This does not claim routing quality, a preferred source, or downstream target
utility. It says only that the permitted source-inner evidence was insufficient
to authorize single-source routing under the frozen uncertainty rule.

## Artifacts and dependencies

The chain is MIDOG++ / Virchow2 / 3,840-dimensional and uses the nine eligible
centers `0,1,2,3,5,6,7,8,9`; center `4` remains excluded.

```text
Stage-30 bank
artifacts/midogpp/30_expert_bank/uniform_b_v2_routing_authorized_expert_bank_v1/

Stage-40 GenerationLock
artifacts/midogpp/40_prior_and_generation/uniform_b_v2_generation_lock/v1/

Frozen direct-control fallback
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_equal_union_policy_lock/v1/

Label-blind validation cache
datasets/midogpp/derived/features/virchow2/uniform_b_v2_routing_validation_cache_v1/seed42/

Non-selecting source-inner utility
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_source_inner_candidate_utility/v1/

Frozen utility/regret policy
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_utility_regret_policy_lock/v1/
```

The cache is `PASS`, has 2,615 validation rows from 44 cases, persists no
labels, and has content hash `e1d281d44e47c7b2` plus build-protocol hash
`f57aad1bf7f7efed`. The source-inner utility uses the promoted aggregate prior
`PS`, 81 source-only synthetic classifier fits, all nine paired training by
generation seed cells, 648 `q != e` utility rows, and 3,168 case-confusion
rows.

## Dependency commands

Run from the repository root on the workstation. The cache is a direct routing
surface; the subsequent two steps are registered workspace experiments and must
remain at their catalogued canonical outputs.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis cvae-routing \
  uniform-b-v2-routing-validation-cache \
  --config experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_routing_validation_cache_v1.yaml

/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.routing_and_composition.uniform_b_v2_source_inner_candidate_utility.v1

/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.routing_and_composition.uniform_b_v2_utility_regret_policy_lock.v1
```

For cache-only verification, use the same direct command with `--validate-only`.
The utility and policy configs are respectively:

```text
experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_source_inner_candidate_utility_v1.yaml
experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_utility_regret_policy_lock_v1.yaml
```

## Protocol and claim boundary

The utility surface is non-selecting source-inner **policy-training** evidence.
All 81 predictions are materialized before validation labels are joined from
the hash-pinned manifest. The labels are then used only for the single
predeclared policy-consumption family; `selection_source` is source-inner
validation case-confusion utility. No train or test rows, outer target,
target support, Stage-20 metric, Stage-50 utility, Stage-90 diagnostic, target
metadata, alternative router, or seed selection is available to this step.

For each pseudo-target query `q`, the best source is a non-deployable
source-inner oracle reference used **only** to calculate paired BACC regret.
It is not a source selection, does not become target evidence, and cannot be
reported as an outer-target oracle or Stage-70 result. `prior_method` is the
promoted aggregate prior `PS`; `claim_role` is policy-training only.

The policy removes both `q = H` and candidate `e = H` before any regret
normalization or bootstrap. It forms 4,536 regret cells and 72 outer candidate
summaries, then applies the frozen three-level pseudo-target / case / paired-
seed bootstrap. Macro-F1 is descriptive only; no individual seed is selected.

## Completed descriptive Stage-70 evidence and archival caveat

The separate descriptive Stage-70 comparison is complete. This utility/regret
arm is exactly equivalent to equal-union in all 81 cells: both reach mean BACC
`0.7749677917` and macro-F1 `0.7726084368`, and the prediction/probability
hashes match exactly. This confirms that the frozen fallback contract was
honored; it is not evidence for adaptive routing.

The comparison validates `PASS`, seals all predictions before label opening,
and uses target labels for scoring only. Its claim scope is nevertheless
`descriptive_frozen_policy_comparison_on_previously_consumed_test`, with
`fresh_confirmatory_status=BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT` and no policy
promotion. See
`docs/wiki/03-experiments/midogpp-uniform-b-v2-descriptive-frozen-policy-comparison-v1.md`.

A routing-success claim would require one frozen hypothesis and a genuinely
unconsumed whole-case/patient/slide-disjoint or external/new-center evaluation
surface. Stage-90 post-hoc diagnostics cannot repair this evidence gap or feed
back into the Stage-60 policy.

The artifact bytes are hash-locked, but the runs record dirty revision
`40221038ca714bf33fd21582857d21fa1db4e6f3` (`repository_dirty=true`). The
final policy's repository-status hash is
`b9581b6f4e31a7ed63791e9bcf9591c73223a103d54ca15b8de17b49e45f7d2d`.
Preserve the exact working-tree diff, or regenerate and revalidate from a clean
committed revision, before thesis archival.
