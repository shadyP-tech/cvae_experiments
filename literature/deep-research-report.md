# Routers in Image Classification With a Focus on Mixture-of-Experts Routing

## Foundations and motivations

Routers (also called *gates* or *routing networks*) are the mechanism that enables **conditional computation**: instead of activating all parameters for every image, the model activates only a subset (e.g., a small set of “experts”) conditioned on the input. The classical deep-learning Mixture-of-Experts (MoE) formulation can be written as a weighted mixture of expert functions, where a learned routing function produces the mixture weights. citeturn5view0turn1search2

The original motivation—still the dominant one in modern vision MoEs—is that **model capacity and compute can be decoupled**. In sparse MoE layers, the router produces a sparse assignment (typically “top-*k*”), so the *number of experts* (and therefore total parameters) can grow substantially while the *per-token compute* stays proportional to *k* rather than to the total expert count. This conditional-computation promise was a core claim of early sparse MoE work, which argued that it can massively increase capacity with relatively small compute losses when implemented well. citeturn1search2turn23academia43

In vision, the strongest empirical demonstration of this motivation arrived with **V-MoE**, which replaces portions of a Vision Transformer’s MLP/FFN blocks with sparse MoE blocks and routes **image patches (tokens)** to experts. The V-MoE paper explicitly frames the problem as replacing dense computation with conditional computation for scalability, reporting competitive image-recognition performance while reducing inference compute, and demonstrating very large parameter counts (up to ~15B) for vision transformers. citeturn5view0turn4view0

A second motivation is **specialization**. In image classification, specialization can happen at multiple granularities: patch-level (certain experts take responsibility for certain visual patterns), region-level (foreground vs background), class-superclass structure (coarse-to-fine semantics), or distributional domains. For example, Mobile V-MoEs explicitly propose **per-image routing** and a **super-class-guided router loss** intended to stabilize routing and encourage experts to specialize by semantic clusters. citeturn14view0turn6view6

Finally, routers are not just an algorithmic object; they define a **systems workload**. In distributed MoE training and inference, tokens are dispatched to experts (often on different devices) and then combined back. Systems work like **Tutel** emphasizes that routing induces dynamic and potentially imbalanced workloads and that efficient implementations must optimize all-to-all communication and related overheads. citeturn25view1turn25view0

![Schematic of common MoE router patterns](sandbox:/mnt/data/router_schematics.png)

## Research landscape and key papers

Recent work in vision MoE routing has converged around a small number of router “families” (token-choice top-*k*, expert-choice, OT/Sinkhorn variants, and soft/slot-based routing), and a set of recurring questions: how to stabilize training (avoid collapse and dead experts), how to trade accuracy against compute, and how to make MoEs practical on real hardware. The table below highlights seminal and recent papers most directly relevant to *image classification* (with some router-centric papers included even when their primary experiments are outside vision, because they introduced router designs widely reused in vision MoEs).

| Work | Vision backbone & where MoE is inserted | Router design focus | Classification evidence (datasets/metrics) | Code / reproducibility signals |
|---|---|---|---|---|
| **Shazeer et al. (2017)** “Sparsely-Gated MoE” | Not vision-first; establishes sparse MoE layer and noisy/top-*k* gating | Sparse routing as conditional computation | Foundational method paper; establishes sparse routing + scaling claims citeturn1search2turn23academia43 | Widely reimplemented; canonical reference citeturn1search6 |
| **Riquelme et al. (2021)** “Scaling Vision with Sparse MoE (V‑MoE)” | ViT; replaces some MLP/FFN blocks with MoE MLP experts routed per **patch token** | Top-*k* token routing; capacity buffers; **Batch Prioritized Routing** to skip low-importance tokens | Reports 90.35% ImageNet accuracy for a very large model and describes inference compute reduction and training FLOP savings via prioritized routing citeturn5view0turn6view3 | Official repo + pretrained models released citeturn12search2turn0search3 |
| **Zhou et al. (2022)** “Expert Choice Routing” | Router algorithm paper (not vision-specific), but impacts vision router design | **Expert Choice**: experts select tokens; load-balancing “by design” | Motivates expert-choice via token dropping / over-capacity ratios in token-choice routing citeturn15view0 | NeurIPS paper; many downstream uses citeturn1search9 |
| **Hwang et al. (2023)** “Tutel” (system) + **SwinV2‑MoE** | Swin Transformer V2 MoE variant (vision) | Efficient routing execution: flexible A2A, adaptive parallelism/pipelining | Reports end-to-end speedups for SwinV2‑MoE training/inference and positions MoE as a dynamic workload with dispatch/combine A2A citeturn25view0turn25view1 | Tutel open-source; Swin-MoE integration in Swin repo citeturn24search0turn3search11 |
| **Daxberger et al. (2023)** “Mobile V‑MoEs” | Small ViTs; MoE in MLP blocks but **single per-image router** | Per-image routing to reduce overhead; super-class guided routing loss | ImageNet-1K: reports consistent gains, e.g., +3.39% for ViT‑Tiny; +4.66% for a 54M-FLOP model citeturn6view6turn14view0 | Paper does not prominently advertise an official training repo (as of the paper snapshot) citeturn14view1 |
| **Puigcerver et al. (ICLR 2024)** “Soft MoE” | Transformer (incl. vision); replaces FFN w/ slot-based expert computation | **Soft (fully differentiable) routing** via token→slot mixing; aims to fix instability/dropping | Provides detailed accuracy/compute comparisons for vision; Soft MoE scales to far more params with small inference-time increase citeturn4view2turn11view1 | Paper via ICLR; multiple community PyTorch ports exist (not official) citeturn12search4turn12search11 |
| **Liu et al. (TMLR 2024)** “Routers in Vision MoE” | ViT-based vision MoE; controlled router comparisons | Unified view via dispatch/combine routing tensors; compares **token-choice vs expert-choice vs SoftMoE vs OT/Sinkhorn** | ImageNet-1K 10-shot: SoftMoE and expert-choice variants outperform token-choice; provides per-router accuracies citeturn19view1turn19view2 | TMLR paper (OpenReview PDF available) citeturn15view3turn12search8 |
| **Videau et al. (2024)** “MoE in Image Classification: What’s the Sweet Spot?” | **ConvNeXt** + ViT; inserts MoE into both CNN-style and transformer backbones | Empirical “design-space” study: #experts, MoE layer placement, top‑1 vs top‑2 | ImageNet-1K/21K: shows small but real gains for moderate activated params; suggests gains diminish when scaling parameters-per-sample too high citeturn20view0turn21view4turn21view0 | arXiv HTML includes extensive tables for replication-style studies citeturn20view0 |
| **Zhu et al. (NeurIPS 2024)** “MoE Jetpack” | Converts dense checkpoints into MoE (ViT + ConvNeXt in experiments) | Fine-tuning dense checkpoints into MoE; introduces **SpheroMoE** + dual-path routing structure | ImageNet-1K table: MoE Jetpack improves accuracy (e.g., 79.9 vs 77.1 Soft MoE baseline in their setting) and reports broad dataset results citeturn26view2turn16view0 | Official code repo provided citeturn3search10turn16view0 |
| **Chowdhury et al. (ICML 2023)** “Patch-level routing theory for CNNs” | CNNs; patch-level routing MoE analyzed | Theory: discriminative routing + sample efficiency for patch routing | Claims provable sample-complexity improvements for patch-level routing in CNN classifiers under study assumptions citeturn15view1 | PMLR PDF available citeturn15view1 |

## Router architecture design space

A useful way to compare routers is to separate two decisions: **(a) how token–expert affinity is computed**, and **(b) how assignments are produced under capacity/load constraints**. The “Routers in Vision MoE” study emphasizes this split by introducing a unified view in terms of **routing tensors** (dispatch and combine) and then varying both the affinity parameterization (e.g., Softmax vs Sinkhorn/OT-derived matrices) and the allocation rule (token-choice vs expert-choice vs soft routing). citeturn15view3turn19view2

### Token-choice top-k routing

This is the most common baseline family: for each token (patch embedding), compute router logits and take the top-*k* experts. V-MoE describes its routing function in this style and explicitly notes that it routes **patch representations**, not entire images; it uses top-*k* after the Softmax, with optional noise. citeturn5view0

Token-choice routers must contend with **expert capacity** and **load imbalance**. V-MoE highlights a fixed per-expert buffer capacity computed from batch size, token count, number of experts, and a capacity ratio; tokens beyond capacity may not be processed by the assigned expert, though residual paths preserve information. citeturn5view0

This design yields a clear benefit: compute scales with *k*, but it can introduce major training and serving pathologies (token dropping, expert collapse, step latency dominated by overloaded experts). These issues are explicitly discussed in router-design papers like Expert Choice Routing, which reports that over-capacity ratios can become very large early in training and that many tokens routed to overloaded experts can be dropped. citeturn15view0

### Expert-choice routing

Expert-choice flips the selection direction: instead of tokens choosing experts, **each expert chooses the top tokens up to its capacity**, guaranteeing balanced loads by construction. The Expert Choice Routing paper describes this as achieving “perfect load balancing by design.” citeturn15view0

In vision, expert-choice and its variants show up as strong competitors in controlled studies. The “Routers in Vision MoE” paper reports that expert-choice routers generally outperform token-choice routers in ImageNet few-shot transfer, and that SoftMoE typically performs best in their comparisons. citeturn19view1turn19view2

### Optimal-transport and Sinkhorn-type routers

A different line of work casts routing as a constrained assignment problem, often framed via **entropy-regularized optimal transport** solved approximately by Sinkhorn iterations. The vision-focused router study lists OT/Sinkhorn parameterizations as a major family (with language-origin routers adapted to vision), and it reports measurable differences between Softmax-based and Sinkhorn-derived affinity matrices, particularly for token-choice routing. citeturn15view3turn19view2

The main tradeoff is that OT/Sinkhorn routing can reduce reliance on auxiliary load-balancing losses and improve load control, but it tends to add **routing overhead** (iterative normalization; sometimes expensive projections), which can become a bottleneck compared to the expert FFNs—especially for smaller backbones. The vision router study explicitly notes higher cost for sparsity-constrained variants because of sorting and projection steps. citeturn19view2

### Soft (fully differentiable) routing

**Soft MoE** replaces “hard” sparse selection (top-*k*) with a **slot-based soft assignment**: experts process a fixed number of “slots,” where each slot is a weighted combination of tokens, and outputs are mixed back into tokens with a matching mixing matrix. The Soft MoE paper argues this addresses several failure modes of sparse MoEs while retaining efficiency, and provides large-scale vision comparisons against token-choice and expert-choice. citeturn4view2turn11view1

MoE Jetpack adopts Soft MoE as a baseline and explains the token→slot mixing formulation in detail (with explicit matrices and reconstruction), aligning with Soft MoE’s description while using it in a dense-checkpoint-to-MoE conversion pipeline. citeturn16view0turn26view2

### Routing granularity in vision

“Routing granularity” often matters as much as router math:

* **Patch/token-level routing** (V-MoE, many ViT/Swin MoEs): maximum flexibility, but can load many experts per image not at once but across all patches—creating memory and systems overhead. V-MoE routes patch representations, and Mobile V-MoEs explicitly argue that per-patch routing can make inference inefficient because many (or even most) experts may be needed for one image. citeturn5view0turn14view0  
* **Per-image routing** (Mobile V-MoEs): a single router selects experts for the whole example, trading flexibility for efficiency and reducing the number of experts needed per image. citeturn14view0  
* **Adaptive compute routing** (V-MoE Batch Prioritized Routing): uses routing plus reduced capacity to **discard less useful patches** and provide a test-time compute–accuracy knob. citeturn4view0turn6view3  
* **Backbone-aware routing** (ConvNeXt vs ViT): design-space studies show that optimal MoE placement and the best top-*k* differ across architectures; “What’s the Sweet Spot?” reports systematic differences in which MoE placement strategy is robust for ConvNeXt vs ViT. citeturn21view0turn21view4  

## Integration patterns with CNNs, ViTs, and other backbones

Most modern image-classification MoEs reuse a simple integration recipe: **replace the feed-forward sub-layer** (MLP/FFN) inside a block with a bank of expert MLPs and a router. V-MoE follows this pattern for ViT, replacing a subset of ViT MLPs with MoE MLP experts. citeturn5view0

### Vision Transformers

For ViT-like backbones, MoE layers nearly always replace the FFN/MLP inside transformer blocks (rather than attention) because FFNs dominate parameter count and are straightforward to shard as independent experts. V-MoE describes experts as MLPs and introduces routing and capacity handling at the token level. citeturn5view0

Soft MoE keeps the “replace the FFN” convention but changes the router/assignment mechanism. Its reported results show substantial improvements in compute–accuracy tradeoffs over dense ViTs and over classical sparse MoEs. citeturn11view1turn4view2

### Hierarchical vision transformers

Swin-style hierarchical transformers (with windowed attention and multi-stage pyramids) can be made MoE by replacing their FFN/MLP parts with MoE FFNs. While the Tutel paper is primarily a systems paper, it explicitly uses **SwinV2-MoE** as a real vision MoE workload and describes the dispatch/combine communication pattern induced by routing (all-to-all in both directions), implying the same core MoE layer structure. citeturn25view1turn25view0

From a practitioner standpoint, Swin-MoE is notable because it is one of the few widely referenced vision MoEs with an “official-ish” implementation pathway: the Swin Transformer repository documents adding **Swin-MoE** implemented using Tutel. citeturn3search11turn13search9

### CNNs and ConvNeXt-style backbones

Classic CNN MoEs can be implemented by routing images (or feature-map regions) to different convolutional branches, routing channels, or using mixtures of convolution kernels. Even though some early conditional-convolution approaches predate the last five years, the **modern “vision MoE” revival** has also extended to CNN-like backbones through ConvNeXt because ConvNeXt contains transformer-inspired MLP-style substructures that are natural MoE insertion points.

“What’s the Sweet Spot?” performs an explicit MoE integration into **ConvNeXt** and **ViT** for ImageNet-1K/21K, reporting that which MoE placement schedule works best depends on architecture (e.g., a “Last 2” insertion strategy is presented as robust in their ConvNeXt setup, while different patterns can be preferred for ViT variants). citeturn21view0turn21view4

MoE Jetpack further strengthens the CNN/ConvNeXt story by evaluating both ViT and ConvNeXt in its dense-checkpoint-to-MoE pipeline and reporting multi-dataset classification results under a roughly fixed-FLOP comparison design. citeturn26view2turn16view0

### Routing and sample efficiency in CNN classifiers

The ICML 2023 patch-routing theory paper analyzes patch-level routing MoE behavior for CNN classifiers and claims that patch-level routing can reduce sample complexity under their assumptions by filtering label-irrelevant patches and routing discriminative patches to the same expert. This matters for image classification because it provides a *theoretical* lens on why patch/token routing might help beyond pure parameter-scaling arguments. citeturn15view1

## Empirical performance and scalability evidence

This section emphasizes **what is directly reported** in primary sources, because MoE results can be sensitive to dataset scale, training recipe, and systems setup.

### Accuracy–compute tradeoffs in Soft MoE vs dense ViT

Soft MoE provides unusually detailed reporting of training budgets, inference compute, and ImageNet outcomes. In the paper’s Table 1, dense ViTs and Soft MoE variants are shown with parameters, inference GFLOPs/image, and ImageNet-1K fine-tuning accuracy. For example, Soft MoE B/16 is reported with **32.0 GFLOPs/image** and **88.5** ImageNet fine-tune, while dense ViT B/16 has **35.1 GFLOPs/image** and **86.6** fine-tune in the same table, illustrating a strong compute-quality advantage at that operating point. citeturn11view1

![Soft MoE vs dense ViT tradeoff](sandbox:/mnt/data/softmoe_vs_vit_tradeoff.png)

In the same table, Soft MoE L/16 reaches **89.2** ImageNet fine-tune with **111.1 GFLOPs/image**, compared with dense ViT L/16 at **88.5** with **122.9 GFLOPs/image**, again suggesting a better accuracy–compute point for the MoE model under the reported recipe. citeturn11view1

Soft MoE also argues that it scales to much larger parameter counts with small inference-time increases; the paper highlights that a huge Soft MoE configuration can have orders-of-magnitude more parameters than a dense baseline with only a small inference-time increase (under their setup). citeturn4view2turn6view5

### Small-model regime: Mobile V-MoEs

A consistent theme in Mobile V-MoEs is that naive token-level MoE routing can be inefficient for small devices because many experts may need to be loaded across the patches of a single image. The paper responds by routing at the **image level** and guiding routing with super-class labels to reduce instability. citeturn14view0turn4view1

Quantitatively, the paper reports ImageNet-1K gains across a range of tiny ViT scales. For instance, for a 12-layer, 192-dim ViT (“ViT-Tiny” in their notation), dense accuracy **59.51** vs MoE accuracy **62.90** is shown (+3.39). For a very small 54M-FLOP model, dense **36.64** vs MoE **41.30** is reported (+4.66). citeturn6view6

![Mobile V-MoE tradeoff in the small-model regime](sandbox:/mnt/data/mobile_vmoe_tradeoff.png)

### Router ablations in controlled vision MoE studies

The vision router study (TMLR 2024) is especially valuable for thesis planning because it compares routers **head-to-head** under a controlled setup and reports downstream *few-shot* ImageNet-1K transfer metrics. In its extracted tables for the B/16 architecture and ImageNet 10-shot, SoftMoE is listed at **72.84%** (for one configuration), exceeding token-choice and expert-choice variants in that table. citeturn19view1

![Router ablation (B16, ImageNet 10-shot)](sandbox:/mnt/data/router_ablation_b16_10shot.png)

The same table segment also shows a typical pattern: token-choice (Softmax or Sinkhorn) is worst; expert-choice variants are stronger; SoftMoE is best (in the reported setting). citeturn19view1

### Practical ImageNet-1K/21K results on open datasets

A recurring criticism of early large-scale vision MoEs is that the most dramatic results often require massive private datasets (e.g., JFT-scale). “What’s the Sweet Spot?” explicitly addresses this by running systematic experiments on **ImageNet-1K and ImageNet-21K** and by inserting MoE layers into both ViT and ConvNeXt.

Their reported ImageNet base-results table shows, for example, ViT-S (no MoE) at **79.8** top-1 accuracy, and ViT-S with 8 experts under certain MoE placement strategies reaching around **80.5–80.7** in their experiments (with additional model/activation details provided in adjacent tables). citeturn21view0turn21view4

Crucially for thesis framing, the same paper argues (in its abstract and discussion) that the “best” results appear when the number of activated parameters per sample is moderate, and that benefits can diminish as parameters-per-sample increase. citeturn20view0turn21view0

### Dense-checkpoint-to-MoE conversion results

MoE Jetpack targets a practical bottleneck: while dense vision checkpoints are abundant, MoE checkpoints are scarce, and training MoEs from scratch is expensive. The paper reports multi-dataset classification improvements when converting dense checkpoints into MoE models, including an ImageNet-1K comparison table where **MoE Jetpack reaches 79.9** vs **77.1** for a Soft MoE baseline (and 75.6 for a dense “21k” baseline in that table), explicitly annotating a **+2.8** gain on ImageNet-1K in their setup. citeturn26view2turn16view0

MoE Jetpack also includes a targeted ablation (its Table 4) where a dual-path SpheroMoE structure reaches **79.9** ImageNet accuracy at **1.1G FLOPs**, compared with a Soft MoE baseline at **78.4** and **1.2G** FLOPs in the same mini-table, reporting a better accuracy–compute point for their modified routing/architecture. citeturn26view2turn26view0

### Systems evidence: routing overhead and scalability

From a scalability standpoint, the Tutel paper offers a primary-source view of MoE routing as a systems problem: an MoE layer uses a gating function to decide token destinations, followed by **all-to-all dispatch**, local expert FFN compute, and **all-to-all combine** back to token origins. citeturn25view1

It reports large speedups for MoE-layer execution and end-to-end training/inference acceleration for a vision MoE workload (SwinV2‑MoE), including up to **1.55× training** and **2.11× inference** speedups over a baseline framework in their experiments. citeturn25view0turn25view1

## Open-source implementations and practical notes

A key feasibility question for a master’s thesis is whether you can reproduce and extend results without proprietary infrastructure. The landscape is mixed: some of the most influential vision-MoE papers provide code, while others provide only a paper. Below are the most relevant repositories and what they concretely enable.

**V-MoE (JAX/Flax, official):** Google Research provides an official *vmoe* repository for training and fine-tuning sparse MoE models for vision and references reproducing results from the V-MoE paper. This is one of the most complete “official” vision MoE codebases, though large-scale results still depend on large compute/data. citeturn12search2turn0search3turn5view0

**Swin-MoE pathway (PyTorch, semi-official):** The official Swin Transformer repository notes inclusion of **Swin-MoE** implemented using **Tutel**, giving a practical entry point if your thesis needs a hierarchical ViT-like backbone with MoE layers. citeturn3search11turn13search9

**Tutel (PyTorch, official):** Tutel is open-sourced by Microsoft and is explicitly designed as a high-performance MoE implementation for dynamic routing workloads. If your thesis includes *systems-style profiling* (routing overhead, scaling, communication), Tutel is among the most relevant codebases. citeturn24search0turn25view1

**AdaMV-MoE code (official research repo):** The ICCV 2023 AdaMV-MoE paper page states that code is available in the Google Research repository under a multi-task MoE project. While multi-task, it explicitly benchmarks ImageNet classification and provides routing visualizations, making it relevant if you want to study router behavior in “classification + something else” regimes. citeturn13search11turn3search32turn0search9

**MoE Jetpack (official):** MoE Jetpack provides an explicit GitHub repository and focuses on a pragmatically important workflow: converting dense checkpoints into MoE models for vision tasks. This is attractive for thesis work because it can reduce reliance on massive pretraining. citeturn3search10turn16view0turn26view2

**GMoE for domain generalization (official):** While not strictly “standard ImageNet classification,” GMoE is a vision MoE backbone explicitly aimed at robustness to distribution shift and includes a public repository. This is relevant if your thesis topic is “routing for robustness” rather than pure in-distribution accuracy. citeturn22search10turn22search6

**Community implementations:** For Soft MoE, multiple third-party PyTorch repositories implement the paper’s slot-based router. These can be useful for rapid experimentation when you don’t need “official” code, but you should treat them as reproducibility aids rather than canonical references. citeturn12search4turn12search11turn12search0

**Practical implementation implications for experiments:**
* If you care about **single-node feasibility** (common for a thesis), the most tractable route is usually: start from a standard ViT/ConvNeXt in PyTorch, insert MoE in FFNs, and test router variants on ImageNet-1K-scale training or smaller proxies (ImageNet-100, CIFAR-100), while measuring routing overhead carefully. MoE Jetpack’s reported experiments include GPU runtimes on an RTX 4090, suggesting some of their pipelines are designed with tractable compute in mind. citeturn26view2  
* If you care about **true MoE scaling** (many experts, expert parallelism), frameworks like Tutel (and broader MoE systems such as DeepSpeed-MoE) become relevant, because naïve implementations can be dominated by dispatch/combine overheads. citeturn25view1turn24search24turn24search1  

## Challenges, open research questions, and thesis topics

### Key technical challenges and limitations

**Routing overhead and communication cost.** A router is not free: it adds compute (scoring), memory bandwidth, and—at scale—communication. Tutel frames MoE as requiring routing-driven all-to-all dispatch and combine, and it reports that dynamic and imbalanced routing workloads create inefficiencies unless the system adapts execution. citeturn25view1turn25view0

**Load imbalance, token dropping, and latency tail.** Token-choice routers commonly need capacity factors and auxiliary objectives to avoid collapse. Expert Choice Routing emphasizes that load imbalance can cause token dropping (over-capacity tokens not processed) and that step latency can be dominated by the most loaded expert, harming inference time predictability. citeturn15view0

**Training stability and collapse modes.** Both vision papers and router-centric papers repeatedly report instabilities: expert collapse (few experts dominate), unreliable routing, and sensitivity to MoE placement/configuration. ViMoE’s abstract notes sensitivity to MoE layer configuration and attributes performance degradation to unreliable routing that prevents experts from learning useful knowledge. citeturn15view4turn12search9  
Beyond vision, representation collapse has been analyzed as a phenomenon where routing encourages token clustering around expert centroids, motivating alternative routing-score parameterizations (e.g., on hyperspheres). citeturn22search1turn22search29

**Benefits can be scale-dependent.** “What’s the Sweet Spot?” argues that MoE gains are most visible in smaller/moderate regimes and can diminish as activated parameters per sample increase, which complicates “MoE will always win” narratives for ImageNet-style classification. citeturn20view0turn21view0

**Interpretability: what do experts learn?** Vision MoE papers often visualize routing patterns and suggest experts focus on different image parts or semantically related classes. MoE Jetpack includes analyses of expert attention patterns and notes that different experts focus on different parts of an input image, interpreting this as specialization. citeturn17view2turn26view2  
However, interpreting routers is still hard: specialization may be brittle, may shift during fine-tuning, and may not align with human semantic categories unless explicitly encouraged (e.g., Mobile V-MoE’s super-class guidance). citeturn14view0turn26view2

### Emerging trends and open research questions

Soft/slot-based routing is a clear emerging direction in vision MoEs because it aims to remove discrete routing barriers (non-differentiability, token dropping) while retaining sparse-like compute advantages. Soft MoE is the flagship example, and controlled router studies show SoftMoE often wins in accuracy–compute tradeoffs under their settings. citeturn4view2turn19view1

Another trend is **making MoEs practical without massive pretraining**. MoE Jetpack is directly motivated by the scarcity of MoE checkpoints and proposes dense-checkpoint conversion as a way to bypass full MoE pretraining. citeturn16view0turn26view2

A third trend is **router evaluation as a first-class object**. The “Routers in Vision MoE” paper is explicitly motivated by the lack of head-to-head router comparisons in vision and provides a template for doing so (few-shot transfer + fine-tuning, multiple routers, upstream pretraining). citeturn15view3turn19view2

Finally, **granularity and deployment constraints** are becoming central. Mobile V-MoEs argues that per-patch routing may be ill-suited for resource-constrained deployment because it can require loading many experts for a single image, motivating per-image routing and guided router training. citeturn14view0turn6view6

### Curated master’s thesis topics with suggested setups and evaluation criteria

The topics below are chosen to align with gaps repeatedly implied by the literature: router stability vs specialization tradeoffs, router cost vs benefit accounting, and making vision MoEs usable on realistic compute.

**Router robustness under distribution shift for image classification.**  
Use a ViT-S or ConvNeXt-T backbone with MoE inserted into the last few FFNs (matching common practice), then compare routers (token-choice top‑1/top‑2, expert-choice, SoftMoE, Sinkhorn variants where feasible). Evaluate on ImageNet-1K in-domain, plus shift benchmarks explicitly used in “What’s the Sweet Spot?” such as ImageNet-A/R/Sketch and ImageNet-V2. Primary metrics: top‑1 accuracy (ID/OOD), calibration error, and routing entropy/utilization statistics. citeturn21view0turn21view4turn19view1

**Compute-accounting thesis: when does routing overhead erase MoE gains in vision?**  
Replicate a subset of the “Sweet Spot” and Soft MoE comparisons but add *profiling-first* measurement: router forward cost, dispatch/combine cost (if distributed), and expert compute. Use Tutel for scalability experiments and a single-GPU implementation for controlled microbenchmarks. Evaluation criteria: accuracy vs *measured latency* (not just FLOPs), memory footprint, and utilization (load balance). citeturn25view1turn25view0turn21view0

**Per-image vs per-token routing in small vision models.**  
Extend Mobile V-MoEs by implementing both per-image routing and per-token routing on the same tiny ViT family and measure (a) accuracy, (b) router/expert loading overhead, (c) stability (collapse frequency), and (d) expert specialization. Mobile V-MoEs provides reported baselines and a strong hypothesis that per-token routing is inefficient in this regime. citeturn14view0turn6view6

**Designing “hybrid” routers: combine soft routing with explicit load constraints.**  
Starting from SoftMoE’s slot mechanism, introduce explicit capacity-like constraints or regularizers (e.g., encourage diversity across experts, penalize redundant expert usage) and compare against SoftMoE and expert-choice baselines. Use ImageNet-1K plus few-shot variants (10-shot) as in the vision router study. Metrics: accuracy, stability (variance across seeds), and expert diversity measures. citeturn4view2turn19view1

**Dense-checkpoint-to-MoE conversion for standard vision backbones beyond ViT.**  
Reproduce MoE Jetpack’s core idea on backbones emphasized in timm-style practice (ConvNeXt, Swin) and test whether conversion gains persist when (a) training data is smaller, (b) augmentation is stronger, or (c) fine-tuning is multi-stage. Evaluate on ImageNet-1K and at least two fine-grained datasets (e.g., Flowers, Pets) as Jetpack does. Criteria: convergence speed (epochs to target accuracy), final accuracy, and sensitivity to router choice. citeturn26view2turn16view0

**Interpretable expert specialization for image classification.**  
Use routing visualizations as a central thesis artifact: quantify whether experts align with semantic clusters (super-classes, textures, foreground objects). Mobile V-MoEs provides an explicit super-class construction recipe (confusion-matrix clustering) to guide routers; use that to test interpretability gains. Metrics: mutual information between expert assignment and semantic labels; qualitative saliency/attention maps; stability of specialization after fine-tuning. citeturn14view0turn26view2

**MoE layer placement as an algorithmic design problem.**  
“What’s the Sweet Spot?” finds that MoE placement strategies (“Last 2” vs “Every 2” vs stage-wise variants) matter and differ across ConvNeXt and ViT. A thesis can formalize placement selection as a search/optimization problem (e.g., small-scale proxy tasks + transfer). Evaluate: best-found placement vs published heuristics, compute cost of search, and generalization across datasets/backbones. citeturn21view0turn21view4

**Router choice under fixed compute: token-choice vs expert-choice vs SoftMoE on open data.**  
The controlled vision router study provides strong few-shot evidence on ImageNet for router rankings. A thesis can extend this to *full supervised training on open datasets* (ImageNet-1K from scratch, ImageNet-21K pretrain where possible) and measure whether the same ranking holds. Include router-cost measurement (Sinkhorn overhead vs Softmax). Evaluation: accuracy (ID/OOD), token dropping rate, and runtime. citeturn19view1turn21view0

**Theory-meets-practice project: patch routing in CNNs with modern backbones.**  
The ICML 2023 theory paper suggests patch-level routing provides sample-efficiency benefits due to discriminative routing. A thesis could implement patch-routing MoE in a modern CNN-like architecture (ConvNeXt-T) and test whether empirical gains align with the theory’s qualitative claims on real datasets. Evaluation: accuracy vs data size (learning curves), routing discriminativeness metrics, and compute overhead. citeturn15view1turn21view3

**Multi-objective router design: accuracy, latency tail, and load balance.**  
Expert-choice is attractive for load balance; token-choice is simple; SoftMoE is often best in accuracy–compute tradeoffs but may introduce dense mixing costs. A thesis can define a *multi-objective* router score and explore Pareto fronts across (a) mean latency, (b) p95/p99 latency, (c) accuracy, and (d) utilization. Tutel’s emphasis on dynamic workload and performance scaling makes it a good platform for this. citeturn25view0turn15view0turn19view2

If you treat these topics as proposal candidates, the most “standard master’s thesis” path (high signal-to-noise, feasible compute) is typically: **(1) pick one backbone (ViT-S or ConvNeXt-T), (2) implement 3–4 router types (token-choice top‑k, expert-choice, SoftMoE, optionally Sinkhorn), (3) evaluate on ImageNet-1K + 1–2 robustness/shift sets, and (4) include explicit profiling and router-behavior analysis**—because recent papers repeatedly show that routing behavior (collapse, imbalance, overhead) can dominate outcomes. citeturn21view0turn25view1turn19view2turn15view0