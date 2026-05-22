# C6.3 Conceptual Synthesis

## Purpose

C6.3 is the strongest current downstream CVAE result because it changes the system response to an observed failure mode:

```text
sparse top-1 routing is unstable,
but the frozen CVAE expert/mode bank contains useful components.
```

The method should be presented as dense post-hoc aggregation over independently trained CVAE-style experts, not as a new compatibility estimator.

```text
Frozen CVAE expert/mode components
-> independently trained downstream classifiers
-> predeclared geometric/log-probability aggregation
-> held-out target downstream BACC
```

This fits the thesis vision of replacing raw data sharing with lightweight local generative experts that can be composed after training. It also fits the MoErging/composition objective: experts are not retrained together, parameters are not shared, and composition happens post hoc.

## Claim Boundary

C6.3 supports this claim:

```text
When sparse support-based or learned top-1 routing is unreliable, dense post-hoc output aggregation over frozen CVAE expert/mode classifiers can recover stronger held-out downstream utility without target labels, expert retraining, or parameter sharing.
```

C6.3 does not support this claim:

```text
C6.3 learns a better target-expert compatibility estimator.
```

The distinction is important. Compatibility remains expected utility. C6.3 reduces routing risk by using many plausible components instead of trusting one selected expert/mode.

## Thesis Placement

C6.3 belongs mainly in these thesis sections:

- Objective 2: conditioned generative models in foundation-model feature space.
- Objective 4: post-hoc MoErging/composition of independently trained CVAE experts.
- Objective 5: unseen-domain downstream evaluation.
- Chapter 6: utility-aligned downstream transfer and composition.
- Chapter 8: implications and limitations.

It should not be described as the main evidence for metadata routing. It is better framed as the strongest downstream composition result after metadata/support/ranking signals proved too brittle for sparse selection.

## Conceptual Evolution Toward C6.3

### C4.1: learned decoder variance

C4.1 asked whether the CVAE generator itself was under-expressive because the decoder used an implicit fixed-variance/MSE likelihood.

The change was:

```text
MSE decoder mean
-> diagonal Gaussian decoder likelihood in PCA64 space
```

This was a clean generator-distribution experiment. Routing, support protocol, PCA dimension, latent dimension, and beta were fixed. The result was diagnostically useful, but learned heteroscedastic variance did not reliably convert CVAE generation into selected downstream utility.

Interpretation:

```text
Output uncertainty alone was not the main limiting factor.
```

### C4.2: source-class latent GMM prior

C4.2 asked whether the standard normal latent prior was sampling poor latent regions. It reused C4.1 artifacts and changed post-hoc latent sampling through source-class aggregated posterior GMMs.

The claim boundary was:

```text
fixed decoder and projection
fixed locked routing decision
only latent sampling prior changes
```

This tested whether the generator bank had better utility under a source-calibrated latent prior. It helped diagnose prior mismatch, underdispersion, and overdispersion, but it still did not solve selected downstream utility.

Interpretation:

```text
The bank may contain useful modes, but better latent sampling alone does not solve the system-level selection problem.
```

### C5.1: support-distance mode-aware routing

C5.1 moved from generator modeling to selection. It asked whether unlabeled target-support distribution statistics could choose an expert/mode candidate.

The proxy was:

```text
support-vs-synthetic distance in shared DINO space
```

The downstream utility was still held-out target BACC after synthetic generation. This experiment clarified a core thesis point:

```text
distributional similarity is only a proxy for utility.
```

Support-distance selection was protocol-valid but too weak and unstable as a routing signal.

Interpretation:

```text
The problem is not simply matching the support distribution.
```

### C5.2: source-LOCO utility ranking

C5.2 moved closer to the thesis definition of compatibility by learning from source-side utility evidence. It trained a source-LOCO ranker to predict candidate utility for a held-out domain without using that domain's utility before selection.

This was more aligned with the thesis principle:

```text
compatibility = expected downstream utility,
not metadata similarity,
not support-distance similarity,
not latent similarity.
```

However, top-1 selection remained brittle. The useful signal was not strong enough to reliably select the best expert/mode.

Interpretation:

```text
The generator bank has oracle potential, but sparse selection has high regret.
```

### C6.1: pooled robust multi-source mixtures

C6.1 changed the strategy from sparse routing to routing-risk reduction. Instead of selecting one expert/mode, it pooled synthetic samples from multiple non-heldout source experts and safe modes.

The key protocol rule was:

```text
source-specific PCA64 synthetic sample
-> inverse scaler
-> inverse PCA
-> original DINO embedding
-> pool across components
-> train one classifier
```

This was a direct test of coverage:

```text
Can the fixed CVAE bank produce a transferable synthetic training distribution if top-1 routing is avoided?
```

Pooling helped clarify the aggregation hypothesis, but a single classifier over a heterogeneous synthetic pool can suffer from decision-boundary interference.

Interpretation:

```text
Coverage helps, but pooled synthetic geometries can conflict.
```

### C6.2: late probability ensemble

C6.2 preserved each component's local decision boundary by training a separate classifier per expert/mode/generation-seed member and averaging probabilities.

The change was:

```text
pooled synthetic embeddings -> one classifier
```

to:

```text
component-specific synthetic embeddings -> component-specific classifiers -> probability aggregation
```

This is a stronger dense aggregation design because it does not force heterogeneous generated distributions into one shared classifier training space.

Interpretation:

```text
Late aggregation better matches the frozen expert-bank setting than pooled synthetic training.
```

### C6.3: geometric late ensemble

C6.3 is a narrow aggregation-rule ablation over C6.2. It keeps the same frozen member bank, generated datasets, classifiers, budgets, class order, and evaluation cells. Only the aggregation operator changes:

```text
arithmetic probability averaging
-> log-probability / geometric probability pooling
```

The rule is:

```text
score_c = sum_i w_i * log(max(p_i(c), eps))
prediction = argmax_c score_c
```

This makes the dense ensemble more consensus-sensitive. It is useful when arithmetic averaging is too tolerant of overconfident or poorly calibrated members.

Interpretation:

```text
C6.3 asks whether dense aggregation should behave more like a mixture or more like a consensus/product-of-experts rule.
```

## Why C6.3 Works

C6.3 works because it removes two fragile assumptions that earlier experiments relied on:

1. It does not require selecting exactly one expert/mode.
2. It does not require pooling heterogeneous synthetic samples into one classifier.

Instead, it keeps local component classifiers separate and combines only their outputs.

This is well matched to the empirical trajectory:

```text
C4.x: generator tweaks alone did not solve selected utility.
C5.x: top-1 expert/mode routing remained too unstable.
C6.1: pooled aggregation reduced routing risk but introduced pooling interference.
C6.2/C6.3: late aggregation reduced routing risk while preserving component-specific decision boundaries.
```

The best current evidence should be stated conservatively. From the synced C7.1a contextual fields, the C6.3 full-context result is approximately:

```text
mean BACC: 0.814
center 0: 0.886
center 1: 0.771
center 2: 0.833
center 3: 0.742
center 4: 0.840
```

This crosses the 0.80 mean-BACC target, but weak centers remain. It is evidence for robust dense aggregation, not evidence that the generator bank reaches uniformly high utility across all domains.

## Why C7.1a Does Not Supersede C6.3

C7.1a tested a source-probe CE objective on decoder means. It was a generator-objective diagnostic:

```text
same heteroscedastic class-conditioned CVAE
same hetero_mean generation
same fixed late aggregation
only difference = frozen source-probe CE on decoder means
```

The result was negative:

```text
C7.1_base mean BACC:             0.674
C7.1_source_probe_ce mean BACC:  0.578
delta:                          -0.095
```

The source-probe CE increased generated source-probe separability, but held-out downstream BACC dropped. This supports the failure label:

```text
SOURCE_GEOMETRY_NOT_TARGET_UTILITY
```

This strengthens the C6.3 interpretation. The current best path is not to make source-local synthetic samples more source-discriminative. The stronger evidence is that dense composition of existing components is more reliable than source-supervised generator sharpening.

## Thesis-Facing Result

Use C6.3 as a thesis-facing result with this wording:

```text
Dense late aggregation over frozen CVAE expert/mode classifiers achieved the strongest downstream transfer in the current CVAE setup. This suggests that the main practical bottleneck after C4/C5 was not the absence of useful generated components, but the instability of sparse expert/mode routing under domain shift.
```

Avoid this wording:

```text
C6.3 learns target compatibility.
```

Use this instead:

```text
C6.3 reduces routing regret by hedging across independently trained CVAE components.
```

## Realistic Ceiling

The current CVAE setup supports a realistic thesis-facing target around:

```text
mean BACC: 0.80-0.82
```

Reaching 0.90 with this exact setup is not realistic without changing the problem boundary. A 0.90 target would likely require one of:

- a substantially stronger generator objective validated against target-like utility,
- target adaptation or pseudo-labeling,
- more powerful foundation-model embeddings,
- additional real source data or richer domain coverage,
- a supervised or semi-supervised target calibration mechanism.

Several of those would weaken the current thesis boundary if they use target labels, target pseudo-labels, or target-evaluation-tuned decisions.

## Recommended Future Experiment Heuristics

Future plans should follow these rules:

1. Treat C6.3 as the current best CVAE downstream baseline.
2. Do not treat source-discriminative geometry gains as utility gains unless held-out BACC improves.
3. Do not return to top-1 routing unless the proposed signal is utility-supervised and beats dense aggregation.
4. Do not pool multi-expert PCA64 coordinates; inverse-project to shared DINO space before pooling.
5. Prefer dense or risk-reducing composition when oracle potential is high but selector stability is weak.
6. Treat metadata/support-NELBO weighting as proxy compatibility; only downstream utility validates it.
7. Use weak-center analysis for centers 1 and 3, but do not tune on their target-eval labels.

## Writing Placement

Suggested thesis placement:

- Chapter 3: describe the locked downstream evaluation protocol and artifact provenance.
- Chapter 4 or 5: summarize negative generator/routing diagnostics as motivation.
- Chapter 6: present C6.3 as the strongest downstream MoErging composition result.
- Chapter 8: discuss limits: weak centers, proxy-vs-utility mismatch, and why 0.90 likely requires a broader setup change.

