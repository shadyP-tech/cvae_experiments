# Current thesis progress synthesis for supervisor meeting

Date: 22 August 2026  
Project: Metadata-guided MoErging of generative models for privacy-preserving domain adaptation in medical image analysis  
Current active benchmark: MIDOG++ with Virchow2 embeddings

## Executive synthesis

The thesis has progressed from an early proof that expert-target compatibility could be estimated to a much more rigorous answer about where the approach succeeds and where it still fails.

The positive progress is substantial. A reproducible multi-domain MIDOG++ evaluation framework is in place; the real Virchow2 representation contains useful cross-domain task signal; a CVAE preserves most of the classifier-relevant structure when decoding posterior information; a better aggregate prior closes most of the source-inner gap to posterior sampling; 27 independently trained source experts have been promoted into a routing-authorized bank; and the generation and comparison policies are frozen and reproducible.

The unresolved thesis result is routing. On the completed, protocol-clean but previously consumed Stage-70 test comparison, the dense equal-union CVAE control achieves mean BACC `0.774968`, while the metadata max-tie policy achieves `0.745099`. The metadata policy is worse by `-0.029868` BACC, with a descriptive paired bootstrap interval of `[-0.050406, -0.008705]`. The source-inner utility/regret policy is exactly identical to equal-union because all nine uncertainty gates abstained. This is useful negative evidence: the system can generate a competitive dense CVAE ensemble, but neither simple metadata similarity nor the current learned utility proxy can safely improve on it.

The latest completed Stage-90 diagnostics localize the remaining bottleneck more precisely. There is action-level headroom, but support-derived or transferred utility estimates do not rank realized actions reliably. Conservative uncertainty envelopes then collapse to zero routing. A newer center-balanced posterior-utility prefix experiment initially failed closed during workstation preflight and then progressed through its full physical and posterior workload after repair. The rerun ultimately terminated `FAILED` during fresh-process content validation with `CBPUPR persisted posterior prediction/model lineage drifted.` It therefore contributes operational and debugging evidence but no scientific result to this synthesis.

The supervisor-facing conclusion should therefore be:

> The thesis now contains a defensible end-to-end negative routing result, not merely an unfinished positive idea. It shows that independently trained CVAE experts and dense composition are viable, while proxy-to-utility transport under domain shift is the central unsolved problem. A fresh confirmatory routing claim remains blocked by the absence of an unconsumed, case-disjoint evaluation surface.

## 1. Connection to the previous presentation

The previous meeting deck, [Thesis_experiment_v3.pdf](Thesis_experiment_v3.pdf), ended with four findings on the earlier BreakHis/DINOv2 experimental surface:

1. Support-NELBO was the strongest leakage-safe compatibility signal, but its top-1 downstream BACC (`0.5252`) was only slightly above random (`0.5185`). Top-2, top-3, and all-four geometric composition improved progressively to `0.5463`, `0.5758`, and `0.5887`; even the single-expert oracle was only `0.6769`.
2. The real PCA64 reconstruction reference reached approximately `0.862`, indicating that representation capacity was not the main bottleneck.
3. CVAE generation was a real bottleneck. The unconditioned selected expert reached `0.4761`, while class conditioning raised the selected result to `0.7107` and the oracle to `0.7687`.
4. Dense composition reduced sparse-routing risk. A diagnostic GMM reached `0.8073` selected and `0.8442` oracle; dense CVAE probability averaging reached `0.7883`; geometric pooling reached `0.8144`, an average gain of `+0.0261` over arithmetic averaging with improvement on four of five centers.

The current work does not simply append another routing method to that deck. It rebuilds the same scientific decomposition under a stronger benchmark and a stricter evidence protocol:

| Previous question | Current MIDOG++ answer |
| --- | --- |
| Is the representation informative? | Yes. The tuned real-feature reference is `0.740312` BACC, and a separately frozen multiscale representation B reaches `0.799159` on new cases within the same centers. |
| Does the CVAE preserve useful structure? | Mostly for reconstruction/posterior decoding, but not for naive prior sampling. Decode and posterior BACC are `0.719681` and `0.716630`; prior sampling is `0.637563`. |
| Can the sampling prior be improved? | Yes, source-inner. Aggregate-posterior sampling reaches `0.770112`, versus `0.757348` for the standard-normal prior and `0.771571` for the posterior-sample ceiling. |
| Can independently trained experts be composed? | Yes operationally. A 27-checkpoint expert bank, generation lock, equal-union control, metadata policy, and utility/regret policy are frozen and validated. |
| Does sparse or metadata routing beat dense composition? | No on the completed descriptive comparison. Metadata max-tie is `0.745099` versus equal-union `0.774968`; the utility/regret router abstains to equal-union in all folds. |
| Is the routing result confirmatory? | No. The evaluation split was previously consumed, so the result is descriptive and cannot promote a policy. |

The qualitative connection is unusually consistent across the two experimental generations: dense composition is robust, while sparse proxy-based selection is brittle. The numeric values should not be compared directly because the dataset, backbone, feature dimensionality, domains, sample definitions, and protocol differ.

## 2. What changed in the experimental setup

### Benchmark and representation

The active experiment moved from BreakHis/DINOv2 to MIDOG++/Virchow2. MIDOG++ is scientifically appropriate because its domains jointly vary by tumor type, laboratory, scanner, staining/acquisition context, and species. The dataset paper explicitly reports deterioration under domain shift and improved generalization from multi-domain leave-one-domain-out training ([Aubreville et al., 2023](https://www.nature.com/articles/s41597-023-02327-4); Zotero item `5XAK2NBQ`).

Virchow2 is a pathology-specific foundation model trained on 3.1 million whole-slide images from diverse tissues, institutions, and stains, and evaluated across tile-level tasks and out-of-domain settings ([Zimmermann et al., 2024](https://arxiv.org/html/2408.00738v2); Zotero item `XBTGS3C2`). This supports its use as a strong feature extractor, but it does not remove the need for the thesis-specific real-feature reference. The Stage-10 experiments provide that necessary task- and split-specific denominator.

### Evaluation unit and protocol

The active protocol uses nine eligible centers: `0,1,2,3,5,6,7,8,9`; center `4` remains quarantined. The held-out target center is excluded from source training and candidate pools. Source-inner selection occurs without target evaluation labels. Target predictions are sealed before evaluation labels open. Balanced accuracy is primary, with macro-F1 and proper losses as secondary diagnostics. Cases, not patches or seed cells, are the independence-preserving unit inside the relevant folds; centers are the outer comparison unit.

This is a major thesis contribution in itself: the work now distinguishes bank construction, generation readiness, policy freezing, descriptive evaluation, terminal diagnostic development, and fresh confirmatory evidence. A method passing software validation is not automatically treated as a scientific routing success.

## 3. Evidence ladder and current results

### Stage 10: real-feature reference

The tuned Virchow2 real-feature classifier reaches mean BACC `0.740312` and macro-F1 `0.737205`, compared with `0.665812` BACC for the untuned reference. The fixed source-inner threshold remains `0.5`; threshold tuning itself changes no predictions. This establishes a usable source-only real-feature denominator, not a CVAE or routing result.

A separate representation-B prospective test confirmation uses 9,928 eligible test rows with no sample or case overlap against the 9,648 discovery rows. It reaches BACC `0.799159`, compared with `0.735733` for canonical representation A, for a paired gain of `+0.063426`, nine wins in nine centers, and a conditional paired case-bootstrap interval of `[+0.050709, +0.073650]`. This confirms representation B on new cases within the same centers. Its Stage-90 placement prevents post-hoc adoption into other stages without a separate authorization.

Interpretation: the current bottleneck is not absence of task signal in the pathology representation.

### Stage 20: CVAE preservation and prior recovery

The tuned-classifier preservation experiment reports:

| Representation | Mean BACC | Macro-F1 | Preservation ratio |
| --- | ---: | ---: | ---: |
| Tuned real-feature reference | `0.740312` | `0.737205` | - |
| CVAE decode mean | `0.719681` | `0.717766` | `0.919368` |
| CVAE posterior sample | `0.716630` | `0.714110` | `0.910740` |
| CVAE prior sample | `0.637563` | `0.630151` | `0.571675` |
| Real PCA128 reference | `0.720533` | `0.718135` | `0.922785` |

The CVAE architecture can preserve most classifier-relevant geometry when the latent state is informed by real data, but naive prior sampling loses much more utility. This turns the generative problem from “the decoder fails” into “the deployable prior must recover a useful latent distribution.”

The Uniform-B aggregate-posterior prior study improves the source-inner standard-normal prior from `0.757348` to `0.770112`, leaving only `0.001459` to the posterior-sample ceiling `0.771571`. This is the strongest positive mechanism result in the current CVAE chain. It is source-inner evidence; it does not independently prove held-out target utility.

The scientific basis is aligned with feature-space CVAE work showing that conditional generative models can model foundation-model embedding distributions for privacy-aware synthetic data generation ([Di Salvo et al., 2024](https://bmvc2024.org/proceedings/145/); Zotero item `3XP2EMY6`). Later DP-CVAE work demonstrates that embedding generation can be combined with formal differential privacy and downstream training in a federated setting ([Di Salvo, Nguyen, and Ledig, MICCAI 2025 proceedings](https://papers.miccai.org/miccai-2025/paper/0970_paper.pdf); Zotero item `VPVWCUAC`). The current MIDOG++ pipeline does not yet contain differential-privacy training, privacy accounting, or an empirical privacy evaluation; therefore no formal privacy claim is currently supported.

### Stage 30: independently trained expert bank

The bank contains 27 independently trained checkpoints: nine source centers by three training seeds. Promotion and independent validation pass; no held-out target performance was used to select an expert or seed. The bank publication state is `ROUTING_AUTHORIZED`.

The frozen equal-union control excludes the held-out target expert, uses all eight eligible source experts, and allocates 128 generated samples per source and class, for 1,024 per class. This is an expert-bank construction and provenance result, not evidence that a router works.

### Stages 40 and 60: generation and policy locks

The Stage-40 GenerationLock validates 81 source streams and 162 health records. It freezes prior, frame, sample budget, seed, shuffle, and classifier settings without target data.

Stage 60 freezes three relevant arms:

1. `equal_union_control`: all eight non-target sources, equal budget.
2. `metadata_max_tie_union`: sources with the maximum exact-match score on tumor type, laboratory/origin, and scanner; all maximum ties are retained.
3. `utility_regret_frozen_policy`: a source-inner learned policy with a conservative uncertainty gate and exact equal-union fallback.

The source-inner utility/regret gate rejects single-source routing for all nine targets. Best-source win probabilities range from `0.392` to `0.786`, below the frozen `0.80` threshold, and all paired-regret lower bounds are negative. The result is not a failed run; it is a valid fail-closed policy decision.

### Stage 70: completed descriptive frozen-policy comparison

The live workstation contains a complete and validated Stage-70 comparison at:

```text
/home/stud/spark/cvae_experiments/artifacts/midogpp/70_frozen_policy_downstream/
uniform_b_v2_descriptive_frozen_policy_comparison/v1/
```

The artifact was produced from a clean repository revision (`380a0a99...`), seals 243 prediction cells before label opening, uses target labels for scoring only, and does not consume Stage-50 or Stage-90 outputs.

| Frozen policy | Mean BACC | Mean macro-F1 | Interpretation |
| --- | ---: | ---: | --- |
| Equal-union control | `0.774968` | `0.772608` | Strongest deployable-style descriptive arm |
| Metadata max-tie union | `0.745099` | `0.739957` | Worse than equal-union on average |
| Utility/regret policy | `0.774968` | `0.772608` | Exact equal-union fallback equivalence |

The metadata-minus-equal-union BACC delta is `-0.029868`; the descriptive paired bootstrap interval is `[-0.050406, -0.008705]`. Averaged across the nine seed cells, metadata routing improves centers `2` and `3`, but harms the other seven. The largest visible weaknesses are centers `0`, `1`, `5`, `8`, and `9`. This pattern explains why a global metadata rule is not reliable even when metadata are genuinely related to domain structure.

Claim boundary: the run is `DESCRIPTIVE_COMPARISON_COMPLETE`, with `fresh_confirmatory_status=BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT`. It does not promote a routing policy, establish external or new-center generalization, or authorize deployment.

### Stage 90: what the latest diagnostics add

The terminal diagnostics should be presented as bottleneck analysis, not as a sequence of candidate thesis wins.

- The consumed-validation exact-tail diagnostic reports BACC `0.770276` for the dense base `B` and `0.762182` for routed arm `R2`; `R2-B=-0.008093`, with interval `[-0.034655, 0.018468]`. A terminal single-source oracle reaches `0.791928`, but the router identifies the best source only `1/9` times, has mean Spearman `-0.000860`, and leaves normalized oracle gap `0.513228`. There is headroom, but the proxy does not rank action utility.
- The simultaneous-shift calibrated posterior-utility router returns zero routes, exactly reproduces protected portfolio `P` at BACC `0.807317`, and fails its information gate. Its own report names the bottleneck `POSTERIOR_UTILITY_DOES_NOT_RANK_REALIZED_ACTIONS`.
- PCSI-RACR v2 completes and validates the full workload: 810 physical probability cells, 218 case routes, 3,488 endpoint fits, 436 target posteriors, 1,314 utility fits, 3,488 policy replays, and two independent fresh-process reconstructions. The primary observed-donor maximum envelope authorizes zero routes and leaves BACC unchanged at `0.807317`. The no-envelope sensitivity authorizes 11 target case policies. This suggests that uncertainty transport, rather than complete absence of candidate action signal, is the immediate choke point. Because the test set is consumed and the method is post-hoc, this remains terminal sensitivity only. The canonical artifact records a dirty repository revision (`d0754475...`), so it is not archival-ready despite validation passing.
- The newest center-balanced posterior-utility prefix router first failed closed during `WORKSTATION_PREFLIGHT` with `Label-free workstation topology drifted.` A repaired run then passed admission and completed 81 source streams, 810 physical classifier cells, 3,488 outer endpoint fits, and 436 target posterior fits. It ultimately terminated `FAILED` during `CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION` with `CBPUPR persisted posterior prediction/model lineage drifted.` The scoped leakage report passes own-route noninterference, but final validation failure invalidates all provisional diagnostic and publication metrics. The root is nonrecoverable and provides no scientific result.

The collective diagnosis is stronger than any one router result: the current dense CVAE composition is useful, but target-local support information and transferred utility models are not sufficiently stable to authorize beneficial sparse action changes under center shift.

## 4. Alignment with the five thesis objectives

| Thesis objective | Current status | Evidence | Remaining gap |
| --- | --- | --- | --- |
| 1. Define a rigorous evaluation framework | Strongly achieved | MIDOG++ domain structure; nine-center held-out protocol; case and identity firewalls; source-inner selection; matched BACC/macro-F1; staged artifact and claim contracts | External dataset or genuinely new-center confirmation would strengthen generalization |
| 2. Develop conditioned generative models | Partially achieved | Class-conditioned CVAEs; preservation study; aggregate-posterior prior recovery; multi-seed bank | Formal DP training/accounting and privacy evaluation are absent; prior-to-target utility remains imperfect |
| 3. Design a metadata-based routing mechanism | Implemented, negative result | Exact metadata compatibility and max-tie policy are frozen; descriptive Stage-70 evaluation is complete | Metadata policy underperforms equal-union by `0.0299` BACC; richer routing signals have not transported utility reliably |
| 4. Construct the MoErging generator | Operationally achieved | Independently trained 27-expert bank; target-expert exclusion; GenerationLock; equal-union and metadata compositions without expert retraining | The adaptive personalized route is not validated; current safe policy is dense equal-union |
| 5. Evaluate domain-specific and generalization performance | Descriptively achieved, confirmatory gap | Stage-10 references, Stage-20 preservation, Stage-70 three-arm comparison, worst-center and case-level analyses | No unconsumed eligible split remains for fresh confirmation; no external/new-center or deployment claim; no privacy-utility curve |

## 5. Scientific interpretation

### What the thesis can claim now

1. A pathology-specific foundation-model embedding surface contains nontrivial source-only signal across MIDOG++ domains.
2. A class-conditioned CVAE can preserve most PCA128 classifier-relevant structure for reconstruction/posterior decoding, while naive prior sampling is a major loss point.
3. Aggregate-posterior prior recovery substantially improves source-inner prior sampling and supports a provenance-clean bank of independently trained experts.
4. Dense equal-union CVAE composition is a strong and stable descriptive baseline.
5. Simple metadata matching and the current source-inner utility router do not improve that dense baseline on the consumed test comparison.
6. Later terminal diagnostics show action headroom but fail to convert support/proxy information into stable realized-utility rankings.

This fits the MoErging literature's separation between experts, routing, and downstream use: building high-quality independent experts does not itself validate the router ([Yadav et al., 2025](https://arxiv.org/html/2408.07057v2); Zotero item `4Y8FZ77X`). It also agrees with broader synthetic-data methodology that intended-task utility must be measured directly rather than inferred from distributional fidelity alone ([Achterberg et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12546680/)).

### What the thesis cannot claim yet

- Metadata-guided routing improves held-out downstream utility.
- The current router generalizes to a new center or external dataset.
- The Stage-90 protected portfolio or any zero-route sensitivity is a deployable policy.
- Passing validation or reconstructing an artifact proves routing success.
- The current CVAE pipeline provides differential privacy or a quantified privacy guarantee.
- Support-NELBO, metadata similarity, posterior utility, proper-loss safety, or fidelity is a substitute for blind downstream evaluation.

### The central research insight

The research question has narrowed from “Can metadata select a compatible expert?” to the more precise and defensible question:

> Under severe, multi-axis pathology domain shift, can pre-evaluation information estimate the signed downstream utility of changing a dense CVAE expert portfolio well enough to outperform a strong equal-union fallback?

The current answer is “not with the tested metadata, support, and transferred-utility mechanisms.” That negative result is scientifically meaningful because the representation, generator, bank, budgets, policy locks, and evaluation boundary have all been separately audited.

## 6. Recommended supervisor presentation storyline

The next presentation should preserve the structure of the June deck but replace the old “three possible bottlenecks” slide with an evidence ladder.

### Slide 1 - One-sentence progress update

“The CVAE expert system is now operational and preserves useful signal; the remaining bottleneck is safe routing under domain shift.”

### Slide 2 - Bridge from the previous meeting

Show the June conclusion in three lines:

- representation had headroom;
- generation and prior sampling lost utility;
- dense composition beat sparse top-1 routing.

Then state that the current work retests this story on MIDOG++/Virchow2 under a stronger protocol.

### Slide 3 - Thesis objective and protocol

Show the chain:

```text
Real feature reference -> CVAE preservation -> expert bank -> generation lock
-> frozen policies -> sealed predictions -> label-only scoring
```

Emphasize the target-expert exclusion and the distinction between fresh, descriptive, and terminal diagnostic evidence.

### Slide 4 - Representation and preservation

Use one compact ladder:

```text
Real reference       0.740
CVAE decode          0.720
CVAE posterior       0.717
CVAE naive prior     0.638
```

Message: the decoder is not the main failure; prior sampling is.

### Slide 5 - Prior recovery and expert-bank readiness

Show:

```text
P0 standard prior    0.757
PS aggregate prior   0.770
Q posterior ceiling  0.772
```

Then show “27 experts; 9 centers x 3 seeds; routing-authorized; no held-out selection.”

### Slide 6 - The decisive three-arm comparison

Show BACC bars:

```text
equal union          0.775
metadata max-tie     0.745
utility/regret       0.775 (exact fallback)
```

Message: metadata matching is not a reliable utility router; the conservative learned policy correctly abstains.

### Slide 7 - Why routing failed

Use a small diagnostic table:

| Diagnostic | Headroom | Identification |
| --- | ---: | --- |
| Exact-tail consumed validation | Oracle `0.792` vs base `0.770` | Top-1 `1/9`; rho approximately `0` |
| PSSCUR | Candidate signal exists | Primary routes `0`; information gate fails |
| PCSI-RACR v2 | No-envelope authorizes 11 cases | Observed-max envelope routes `0` |

Message: the problem is not the absence of candidate actions; it is reliable utility ranking and uncertainty transport.

### Slide 8 - Claim boundary

Use two columns:

Supported: representation signal, CVAE preservation, prior recovery, bank readiness, dense composition, negative metadata-routing result.  
Not supported: fresh routing superiority, new-center generalization, deployment, formal privacy.

### Slide 9 - Thesis objective scorecard

Use the five-objective table from Section 4, shortened to one line per objective.

### Slide 10 - Decision requested from the supervisor

Ask for a strategic choice among:

1. Treat the negative routing result as the thesis conclusion and focus on clean archival, writing, and a bounded external/fresh confirmation if feasible.
2. Reserve a genuinely fresh domain/split and run one final predeclared matched confirmation of a single frozen routing hypothesis.
3. Add a formal privacy experiment, because privacy is central to the thesis title but not yet empirically demonstrated in the current pipeline.

The scientifically safest recommendation is option 1 plus a small, explicitly bounded privacy or external-validation component if time permits. Continuing to design routers on the consumed MIDOG++ test set can improve mechanistic understanding but cannot produce fresh thesis-confirmatory routing evidence.

## 7. Immediate next work

1. Keep the now-synchronized Stage-70 state page and focused evidence note aligned with the clean canonical artifact.
2. Preserve the clean Stage-70 artifact as the primary downstream evidence and create a concise figure from `arm_summaries.csv`, `bootstrap_summary.csv`, and center-wise means.
3. Decide whether the thesis needs one final fresh/external routing study or whether the negative routing result is the terminal conclusion.
4. Diagnose the prefix router's persisted posterior prediction/model lineage drift using the failed root only as debugging evidence. Do not repair or resume the root in place, quote its provisional metrics, or consume it downstream. Any rerun requires a new experiment/output identity and must remain terminal diagnostic only.
5. Add an explicit privacy-status subsection to the thesis. Either implement and evaluate a bounded DP-CVAE/privacy-accounting experiment, or narrow the wording to “privacy-motivated feature sharing” and state that formal privacy is future work.
6. Archive reproducibility evidence. The clean Stage-70 run is suitable; PCSI-RACR v2 records a dirty revision, and the failed prefix root is not archival scientific evidence.

## 8. Evidence and literature record

### Live workstation artifacts inspected

```text
/home/stud/spark/cvae_experiments/artifacts/midogpp/70_frozen_policy_downstream/
uniform_b_v2_descriptive_frozen_policy_comparison/v1/

/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/
uniform_b_v2_consumed_test_fixed_bank_p_anchored_simultaneous_shift_calibrated_utility_router/v1/

/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/
uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_boundary_projected_pcsi_policy_regret_router/v2/

/home/stud/spark/cvae_experiments/artifacts/midogpp/90_oracles_and_diagnostics/
uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router/v1/
```

Live workstation checkout at inspection: branch `main`, HEAD `72e02217f8fab9414e12280f2a0834cdbbe88f9a`, with modified and untracked files. Artifact-specific provenance, rather than current checkout state, is used for scientific interpretation.

Final prefix-router status at inspection: `FAILED`, phase `CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION`, updated `2026-08-22T14:48:28Z`, with error `CBPUPR persisted posterior prediction/model lineage drifted.` Its completed runtime workload is reported only as operational context, not as a thesis result.

### Literature search

Exa was used to review 95 returned sources across three search workstreams: MIDOG++/pathology foundation models and domain shift; CVAE synthetic embeddings, privacy, and downstream utility; and MoErging/routing/ensemble behavior. Duplicate URLs and secondary summaries were filtered, and the synthesis above relies primarily on original dataset papers, conference papers, peer-reviewed articles, and original model papers.

### Zotero library anchors

| Topic | Zotero item key | BibTeX key |
| --- | --- | --- |
| MIDOG++ dataset | `5XAK2NBQ` | `aubrevilleComprehensiveMultidomainDataset2023` |
| Virchow2 | `XBTGS3C2` | `zimmermannVirchow2ScalingSelfSupervised2024` |
| Feature-space CVAE data sharing | `3XP2EMY6` | `disalvoPrivacypreservingDatasetsCapturing2024` |
| DP-CVAE federated embedding sharing | `VPVWCUAC` | `disalvoEmbeddingBasedFederatedData2026` |
| Model MoErging survey | `4Y8FZ77X` | `yadavSurveyModelMoErging2025` |
| Open medical OOD benchmarking | `W54ZTLFA` | `gutbrodOpenMIBOODOpenMedical2025` |

## Final supervisor-facing verdict

The thesis has passed the “can the system be built and evaluated rigorously?” milestone. It has also produced positive evidence for the representation, CVAE preservation, prior recovery, expert-bank construction, and dense composition components. The main hypothesis that metadata- or support-guided routing improves downstream performance is not supported by the current evidence. The strongest defensible thesis contribution is therefore a protocol-rigorous decomposition of why: useful expert diversity exists, but proxy-to-utility transport is unstable, and conservative uncertainty control correctly collapses to dense fallback. Fresh routing confirmation, new-center generalization, and formal privacy remain open.
