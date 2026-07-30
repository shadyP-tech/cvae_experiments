# MIDOG++ Uniform-B v3 Retrospective Replay v1

## Question and status

The completed Stage-10 v3 pilot used source-inner selection separately for
each outer center and chose B for six centers, C for two, and A for one. The
uniform-B replay asks a narrower follow-up question: what result is obtained if
the already-defined 3,840-dimensional canonical-JPEG fixed-center B
representation is applied to every one of those nine outer centers while the
per-fold classifier locks remain frozen?

The run is complete, independently validated, and cataloged as `DIAGNOSTIC
ONLY`. Its study-design status is `POSTHOC_DISCOVERY` because the choice to
evaluate B uniformly was made after these same outer-center outcomes were
known. It is not an independent validation cohort.

## Frozen replay protocol

The experiment imports the nine immutable v3 outer decision locks and extracts
the classifier specification already selected for A and B in each fold. For
every held-out center it then:

1. opens only the other eight centers' embedded A and B feature shards;
2. fits preprocessing and the imported classifier independently for A and B;
3. opens the held-out A/B shard only after the global uniform-B lock exists;
4. scores both representations on identical rows;
5. requires exact canonical-A prediction replay and exact agreement with the
   corresponding source-v3 A/B rows.

C is not resolved or loaded. There is no representation selection, classifier
selection, or target-label feedback in the replay itself. The fixed global
representation lock records that the choice was informed by prior target
scores and is not adoption-eligible.

## Validated result

| Held-out center | A BACC | B BACC | B minus A |
| --- | ---: | ---: | ---: |
| 0 | `0.722835` | `0.795247` | `+0.072412` |
| 1 | `0.679245` | `0.762803` | `+0.083558` |
| 2 | `0.696827` | `0.740306` | `+0.043478` |
| 3 | `0.756545` | `0.824607` | `+0.068063` |
| 5 | `0.765176` | `0.792332` | `+0.027157` |
| 6 | `0.792350` | `0.833333` | `+0.040984` |
| 7 | `0.779116` | `0.833333` | `+0.054217` |
| 8 | `0.762621` | `0.841615` | `+0.078995` |
| 9 | `0.708092` | `0.705202` | `-0.002890` |
| equal-center mean | `0.740312` | `0.792087` | `+0.051775` |

B wins strictly on eight of nine centers. The conditional paired case
bootstrap produced 2,000 valid replicates, mean delta `+0.051525`, and a 95%
percentile interval of `[+0.038962, +0.063599]`. No p-value or significance
decision is computed.

The independent validator passes with 9 imported source locks, 18 outer-result
rows, 19,296 row-level predictions, 9 center comparisons, and exact replay of
all 9 source folds.

## Interpretation boundary

The result shows that B is a strong, stable candidate on these already-observed
centers and that the per-center adaptive policy did not maximize the eventual
outer mean. It also establishes a reproducible fixed-B estimate under the
frozen classifier locks. Those are useful mechanism and planning signals.

It does not establish prospective uniform superiority. The observed uplift is
conditional on selecting B after inspecting the same family of outer results;
the bootstrap does not account for that selection event, training/lock
selection uncertainty, or sampling a new center. Therefore the replay cannot
adopt B, replace canonical A, revise the Stage-10 denominator, or feed any
Stage-20-through-70 choice.

Prospective within-center confirmation is now complete under
`midogpp-uniform-b-v3-prospective-test-confirmation-v1.md`. It froze B, its
preprocessing, classifier locks, inclusion rules, primary metric, and failure
criteria before B was extracted or scored on case-disjoint test rows. It does
not replace the still-open requirement for genuinely new-center or external-
dataset validation.

## Reproduction

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v3_retrospective_replay.v1
```

Canonical artifact:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v3_retrospective_replay_v1/seed42/
```
