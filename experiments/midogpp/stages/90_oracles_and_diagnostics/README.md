# Stage 90: Oracles and Diagnostics

This stage contains post-hoc oracle upper bounds, fidelity analyses, rejected
lineage audits, and other non-deployable diagnostics.

Every oracle row must be marked non-deployable. Diagnostic results may explain
a failure or quantify headroom, but they must not tune or select a deployable
policy and must remain separate from held-out utility claims.

The rejected subtree also preserves evidence from the old limited-domain
MIDOG++ scanner support-routing surface. That surface did not cover the current
full dataset contract, so its top-1, rank, and oracle-gap values are not
thesis-facing and must be regenerated before any routing claim.

## Uniform-B v3 Retrospective Replay v1

`uniform_b_v3_retrospective_replay_v1` replays one fixed representation,
`annotation_jpeg_fixed_center_b_v3`, across all nine eligible outer centers.
It imports the already-frozen per-fold classifier locks from the completed v3
pilot, refits canonical A and B without the held-out center, and checks exact A
replay plus row-level agreement with the source v3 tables. The loader opens
only the A and B shards; physical-multiscale C is not an experiment input.

The independently validated result is:

| Quantity | Canonical A | Uniform B | B minus A |
| --- | ---: | ---: | ---: |
| equal-center mean BACC | `0.740312` | `0.792087` | `+0.051775` |

B wins on eight of nine centers and has a worst-center delta of `-0.002890`.
The conditional paired case bootstrap has 2,000 valid replicates and a 95%
percentile interval of `[+0.038962, +0.063599]`. It conditions on the observed
centers, fixed fits, and imported classifier locks; it does not cover the
uncertainty induced by choosing B after observing these outer-center results or
the uncertainty of transferring to a new center.

This is a `POSTHOC_DISCOVERY`, not independent confirmation. It validates
deterministic reproducibility and gives a consolidated fixed-B estimate for the
same nine centers, but it is non-adoptive and cannot change the Stage-10
reference, feed Stage 20 through 70, or support deployment, routing, CVAE,
generation, calibration, or new-center claims.

Run the registered replay with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v3_retrospective_replay.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v3_retrospective_replay_v1/seed42/
```

## Uniform-B v3 Prospective Test Confirmation v1

Phase B locks B, the nine source-only classifier locks, the primary endpoint,
and a four-part gate before extracting or scoring B on the case-disjoint test
split. The 9,928 eligible test rows have zero sample and case overlap with the
9,648 train rows used for the retrospective discovery; validation rows are not
used. The B-only test cache is independently validated and fully hash-promoted.

The predeclared gate requires mean BACC delta at least `+0.02`, at least six
strict center wins, worst-center delta at least `-0.01`, and a positive 95%
conditional-bootstrap lower bound. The independently reconstructed result
passes all four checks:

| Quantity | Canonical A | Uniform B | B minus A |
| --- | ---: | ---: | ---: |
| equal-center test mean BACC | `0.735733` | `0.799159` | `+0.063426` |

B wins on all nine centers, its worst delta is `+0.010638`, and the 2,000-
replicate conditional paired case interval is `[+0.050709, +0.073650]`. The
decision is `CONFIRMED_WITHIN_CENTER`.

This is prospective confirmation for new cases from the same observed centers,
not external-dataset or new-center confirmation. It remains Stage-90
`DIAGNOSTIC ONLY`, cannot automatically replace the canonical Stage-10
reference, and cannot feed any Stage-20-through-70 choice.

## Canonical Uniform-B Nyström Nonlinear Probe v1

The bounded nonlinear probe keeps canonical B fixed and changes only its
decision function to `StandardScaler -> Nyström RBF -> L2 logistic`. Exact
nested source-inner LODO selects among 36 predeclared candidates while
inheriting each outer fold's canonical-B class weight.

The independently reconstructed diagnostic passes its full progression gate:

| Quantity | Linear B | Nonlinear B | Delta |
| --- | ---: | ---: | ---: |
| equal-center train-surface BACC | `0.792087` | `0.815278` | `+0.023192` |

It wins on all nine centers and has a worst-center delta of `+0.002008`. Its
supportive case-within-center bootstrap interval is
`[+0.011774, +0.032390]`. It rescues 751 linear errors while introducing 550,
and resolves 613 of 1,025 baseline errors whose source-only centroid geometry
already favored the true class.

This supports a nonlinear-boundary limitation in B; it does not prove B is
sufficient. The result is diagnostic-only, uses an already inspected train
surface, and does not authorize automatic canonical migration. The validation
split remains unfeaturized and unscored; its 44 cases provide only 3–7 cases
per center, below the existing confirmation minimum.

Run the registered probe with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_nystroem_nonlinear_probe.v1
```

## Uniform-B Robust Interaction Probe v1

The paired audit finds 550 standard-Nyström regressions, of which 246 are
near the decision boundary and only two are confidently wrong. Center 2
accounts for 257 false-positive regressions; center 9 contributes 24
false-negative regressions.

A bounded nested comparison then tests equal center×class weighting, two
group-DRO rates, and bilinear ranks `{4,8,16}`. Robust Nyström retains a
`+0.020437` equal-center BACC gain over linear B and fixes much of center 2's
specificity loss, but center-9 recall falls by `−0.127168`. The bilinear
family is `−0.006267` below linear B on average.

Decision: `NO_FAMILY_PASSES_ROBUST_BPLUS_GATE`. No classifier is frozen and
validation/test remain untouched. The result indicates that generic group
reweighting moves the worst-group failure rather than resolving it.

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_robust_interaction_probe_v1/seed42/
```

## Uniform-B Sensitivity/Specificity-Constrained Nyström Probe v1

This fixed-threshold successor tests four inherited Nyström objectives along a
native-logit capacity path while enforcing per-inner-center recall,
specificity, and BACC deltas versus exact linear-B refits.

Decision: `NO_CONSTRAINED_BPLUS_CANDIDATE_PASSES`. Only centers 6 and 9 admit
nonlinear capacity at `alpha=0.25`; the other seven centers fail closed to
exact linear B. Equal-center mean BACC improves only `+0.00282`, with a
`−0.03468` worst recall delta at center 9.

The final CPU-only run took `275.92` seconds using four workers × three
threads. Independent reconstruction validation passes. Validation and test
remain untouched.

Canonical output:
`artifacts/midogpp/90_oracles_and_diagnostics/uniform_b_sens_spec_constrained_nystroem_probe_v1/seed42/`.
