# MIDOG++ Uniform-B V2 Metadata Routing Baseline V1

## Result

The first Stage-60 comparison against the canonical equal-union control is
frozen and independently validated.

- Compatibility decision: `FROZEN_AS_METADATA_PROXY_COMPATIBILITY_INPUT`
- Compatibility lock: `4b46b3d157b07781`
- Policy decision: `FROZEN_AS_METADATA_EXACT_MATCH_TIE_UNION_COMPARISON_POLICY`
- Policy publication state: `POLICY_FROZEN_FOR_MATCHED_STAGE70_EVALUATION`
- Policy lock: `27f16953b32c46cd`
- Production and independent validation: `PASS`

This is a lock-only result. It establishes neither routing quality nor
downstream utility.

## Compatibility contract

The non-selecting artifact consumes only the SHA-256-pinned MIDOG++ domain
mapping and exposes three routing-time values: tumor type, lab or origin, and
scanner model. Its scorer receives profile values, not center IDs, and records
the unweighted componentwise exact-match count for all 72 ordered
target-excluded pairs.

It uses no target samples, support rows, labels, Stage-20 scores, Stage-50
utility, Stage-90 oracle evidence, NELBO, or target metrics. It emits no source
rank, selection, or weight.

## Frozen policy

Every source tied at the maximum metadata score is retained:

| Target | Selected sources | Rows per selected source and class |
|---:|---|---:|
| 0 | 5 | 1,024 |
| 1 | 2 | 1,024 |
| 2 | 1 | 1,024 |
| 3 | 1, 2 | 512 |
| 5 | 6, 7 | 512 |
| 6 | 5, 7 | 512 |
| 7 | 5, 6, 8, 9 | 256 |
| 8 | 7, 9 | 512 |
| 9 | 7, 8 | 512 |

The total remains 1,024 generated rows per class. Canonical source order is
used only for deterministic ordering, never to break a tie. All 81 training-
seed by generation-seed replicates are retained, producing 153 assignments
with no seed selection. Source streams, prefixes, and within-class shuffle
seeds are inherited exactly from the Stage-40 GenerationLock and paired
equal-union control.

## Canonical artifacts

```text
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_exact_match_compatibility/v1/
artifacts/midogpp/60_routing_and_composition/uniform_b_v2_metadata_tie_union_policy_lock/v1/
```

Each artifact has 12 required files, all SHA-256-pinned in the MIDOG++ artifact
catalog. Both record revision `40221038ca714bf33fd21582857d21fa1db4e6f3`
with `repository_dirty=true`; preserve the exact diff or regenerate from a
clean revision before final archival.

## Next evidence

The substantive source-inner utility/regret policy is now frozen; see
`docs/wiki/03-experiments/midogpp-uniform-b-v2-source-inner-utility-regret-policy-v1.md`.
Its uncertainty gate rejected single-source routing for every outer fold and
reused the exact equal-union fallback. Its source-inner labels remain
policy-training evidence only; consumed Stage-20 evidence and Stage-50/90
artifacts still cannot feed the policy.

Authorize one immutable target-evaluation input and implement matched Stage-70
scoring for equal-union, metadata max-tie, and utility/regret policies.
Candidate eligibility, generated sample budget, GenerationLock streams,
classifier settings, seed replicates, and evaluation rows must be identical.
Only that fresh evaluation may support a routing-quality or downstream-utility
claim.
