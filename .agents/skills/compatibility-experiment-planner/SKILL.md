---
name: compatibility-experiment-planner
description: Convert fuzzy CVAE metadata-routing experiment ideas into implementation-safe plans. Use when asked to plan compatibility, oracle-gap, support-set NELBO calibration, learned utility, latent/metadata proxy, expert routing, LOQDO/LODO, or baseline comparison experiments. Enforces that compatibility is expected utility (-NELBO), while similarity is only a proxy.
---

# Compatibility Experiment Planner

Use this skill to turn an experiment idea into a protocol-safe implementation plan for this repo.

## Non-Negotiable Definitions

- Compatibility means expected utility: `C(q, e) = -NELBO(q, e)`.
- Support-set NELBO is a direct utility estimate when computed on a disjoint target support set.
- Similarity is a proxy only: metadata distance, latent distance, learned score, or other non-evaluation score.
- Any non-oracle routing signal is valid only if it is checked against held-out true utility.
- Reject plans that only show "better similarity" and do not measure top-1 oracle hit, Spearman with true utility, and oracle gap.

Before implementation, also apply `$thesis-protocol-guard`.

## Planning Checks

1. Identify the query/target domain, source expert domains, split regime, support split, and evaluation split.
2. State exactly what information the router sees before expert selection.
3. Keep target support and target evaluation samples disjoint.
4. Keep target/query experts out of the source expert pool for LOQDO/LODO unless they are explicitly marked diagnostic-only oracle references.
5. Train CVAE experts independently on their own source-domain train split; routing must not update expert checkpoints.
6. Evaluate selected experts with held-out NELBO and report true utility alignment, not proxy alignment alone.

## Required Output

For a compliant plan, output exactly these headings in this order:

### Hypothesis

One falsifiable claim about whether a routing signal improves expected utility over baselines.

### Compatibility signal

Name the proposed routing score and label it as one of:

- oracle true utility: `-NELBO` on held-out evaluation data, used only for evaluation/oracle diagnostics
- direct utility estimate: `-NELBO` estimated on a disjoint target support set
- proxy: metadata similarity, latent similarity, learned predictor, or other routing score

If the score is not held-out true utility, state how it will be checked against held-out `-NELBO`.

### Allowed query information

List only pre-selection information available to the router: query metadata, source expert metadata, source-trained artifacts, and optionally a disjoint target support set. Exclude target evaluation NELBO, oracle expert labels, and target test labels.

Target-domain identity may be used only if it is part of the declared query metadata available at routing time and does not directly identify the excluded oracle expert or evaluation fold in a way that collapses routing into lookup.

### Expert pool

List candidate CVAE experts and checkpoint provenance. For LOQDO/LODO, use source-domain experts only and exclude the held-out target/query expert from routing candidates.

### Routing rule

Specify the exact decision rule, for example argmax proxy compatibility, argmin predicted NELBO, support-set top-k uniform, softmax weighting, or hard metadata routing. Include tie-breaking and whether the rule is per-query, per-domain, or aggregate.

### Evaluation protocol

Specify folds, seeds, support/evaluation split separation, expert scoring matrix construction, and how selected experts are scored on held-out evaluation NELBO. Use existing surfaces when relevant:

- routing strategies: `cvae_testing/src/routing/`
- learned utility: `cvae_testing/src/eval/evaluators/learned_utility.py`
- latent compatibility: `cvae_testing/src/eval/evaluators/latent_compatibility.py`
- support-set calibration: `cvae_testing/src/eval/evaluators/support_set_calibration.py`
- domain oracle gap: `cvae_testing/src/eval/evaluators/domain_query_oracle_gap.py`

### Primary metrics

Must include all three:

- `top1_oracle_hit`
- Spearman with true utility (`spearman`, `spearman_with_oracle`, or `spearman_with_true_utility`)
- oracle gap (`mean_oracle_gap`, `mean_oracle_gap_pct`, or normalized oracle gap)

Add mean rank, pairwise AUC, NELBO delta, and seed/fold stability when useful.

### Baselines

Must include metadata routing plus comparable baselines where applicable:

- hard/soft metadata routing
- random expert or uniform expert sampling
- naive/equal-weight ensemble or equal-weight scoring
- weight averaging, pooled/global CVAE, or hybrid pooled baseline when implemented and comparable for the same artifact type
- oracle expert or per-query oracle as diagnostic-only upper bound

If a baseline is not implemented or not comparable, mark it unavailable and do not make claims against it.

### Leakage risks

Name concrete risks: target expert leakage, target evaluation NELBO in router fitting, support/evaluation overlap, fold leakage, normalization/tuning on held-out targets, or proxy-vs-utility claim inflation. State the required guardrail for each.

### Expected artifact outputs

List exact files/tables/plots expected from the run, such as:

- config YAML under `cvae_testing/configs/experiments/`
- run reports under `outputs/**/reports/`
- per-query selections CSV
- expert NELBO or utility matrix
- domain breakdown CSV
- proxy diagnostics CSV
- oracle-gap summary table
- baseline comparison table
- leakage/provenance report

### Decision rule

Define the pass/fail rule and classify the expected outcome as one of:

`PASS`:
The method improves top-1 oracle hit, Spearman with true utility, and oracle gap versus metadata routing and required comparable baselines across declared seeds/folds.

`WEAK PASS`:
The method improves at least one primary utility metric without materially degrading the others, but evidence is insufficient to claim superiority.

`DIAGNOSTIC ONLY`:
The method reveals compatibility structure, routing heterogeneity, or proxy/utility mismatch, but does not beat metadata routing.

`FAIL`:
The method is protocol-compliant but performs worse than metadata routing or required comparable baselines.

`REJECTED`:
The method improves proxy similarity but expected utility is not validated, or required utility metrics are missing.

### Claim boundary

Do not claim a method is better than metadata unless it satisfies `PASS`. Do not claim compatibility recovery from proxy improvement alone. If the method is `WEAK PASS` or `DIAGNOSTIC ONLY`, describe it as evidence about signal structure, not as a routing improvement. Negative results must still report what compatibility signal failed and under which split regime.

## Rejection Format

If the requested idea cannot be made compliant without changing its claim, output:

`REJECTED: <one sentence>`

- Missing utility validation:
- Leakage or protocol risk:
- Minimal compliant revision:
