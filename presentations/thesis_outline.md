Metadata-Guided  MoErging  of  Generative  Models  for
Privacy-Preserving  Domain  Adaptation  in  Medical  Image
Analysis

Project  vision:  The  vision  of  this  project  is  to  reshape  healthcare  data-sharing  by
replacing  the  exchange  of  sensitive,  high-volume  datasets  with  privacy-preserving,
efﬁcient,  and  personalized  generative  models.  Rather  than  sharing  large-scale  raw
clinical  images  and  associated  personal  health  information  (PHI),  institutions  will
exchange  lightweight  models  capturing  local  data  distributions  while  offering  formal
privacy  guarantees.  As  illustrated  in  Figure  1,  the  proposed  approach  allows
sampling  (i.e.,  generating)  synthetic  data  tailored  to  new  and  previously  unseen
domains while rigorously protecting patient privacy. Furthermore, its modular designs
support the addition or removal of institutions with minimal overhead, as no end-to-
end training across sites is required.

yi

xi

i=1

Nh

  and  label

  .  The  raw  image

  deﬁnes  the  number  of  samples  and

Figure  1:  Overview  of  SynGenacy,  a  modular  generative  framework  enabling
privacy-preserving, domain-adapted, and efﬁcient synthetic data generation
 independently
across multiple institutions. (1) Each institution
h ∈ {1,…, H}
,  thereby  providing
trains  a  differentially  private  (DP)  generative  model
𝓖DP
h
𝓓h = {(xi , yi, mi)}Nh
formal  privacy  guarantees,  on  its  private  dataset
  ,
  encodes  structured
where
mi
metadata  (e.g.,  scanner  type,  resolution,  patient  information)  associated
  is  ﬁrst  processed  through  a
with  image
large pre-trained feature extractor (F.E.) to obtain a compact and informative
latent representation  . (2) A target institution provides a support set
𝓓target
from  which  metadata  statistics  are  used  to  compute  similarity  scores  with
respect  to  the  source  domains.  These  scores  deﬁne  soft  routing  weights
over the pretrained generators, enabling the construction of a personalized
generator
.  Notably,  this  aggregation  is  performed  post-hoc.
Therefore,  during  training,  each  model  remains  fully  local.  (3)  The
personalized  generator
  is  used  to  synthesize  a  diverse  synthetic
𝓖target
~
target-speciﬁc  dataset
  via  an  augmented  sampling.  (4)  Lastly,  the
𝓓target
synthetic  personalized  dataset  supports  downstream  classiﬁer  training
without access to raw images or personal health information (PHI), ensuring
data privacy.

𝓖target

xi

fi

Scope  of  the  thesis:
  In  multi-institutional  medical  imaging,  models  trained
independently  on  local  datasets  often  capture  complementary  but  domain-speciﬁc
variations  (eg,  scanner  type,  staining  protocols,  patient  population,  and  more).
Integrating  these  models  into  a  uniﬁed  system  without  data  sharing  is  crucial  for
scalability and privacy preservation.

This  thesis  explores  a  novel  metadata-driven  model  merging  approach  inspired  by
recent  work  on  model  MoErging  [1],  a  paradigm  combining  ideas  from  model
merging  [2]  and  Mixture-of-Experts  (MoE)  [3,4].  Unlike  traditional  model  fusion  (eg,
weight  interpolation  or  knowledge  distillation),  MoErging  enables  post-hoc,
decentralized,  and  privacy-aware  model  integration  by  leveraging  structured
metadata  to  guide  model  selection  at  inference  time.  To  facilitate  comparison  and
clarify assumptions, we adopt a recent taxonomy proposed in [1], which categorizes
MoErging  design  choices  along  three  core  dimensions:  (i)  the  experts  ,  which  are
independently  trained  generative  models  shared  by  contributors  in  a  decentralized
setting;  (ii)  the  routing  strategy  ,  which  determines  how  experts  are  selected  and
potentially aggregated based on task or domain metadata; and (iii) the downstream
application  ,  which  speciﬁes  how  the  composed  model  is  used,  eg,  image
generation,  augmentation,  or  diagnosis  support.  This  framework  enables  the
systematic exploration of personalized model composition without data centralization
or expert retraining.

Therefore,  given  a  set  of  pre-training  datasets  (from  different  institutions)  and  a
target dataset, the aim is to independently train generative experts on each source
domain  and  later  compose  a  personalized  generator  by  selecting  and  aggregating
the  most  relevant  experts  based  on  their  afﬁnity  to  the  target  domain,  measured
through structured metadata or via dataset's numerical statistics.

Research Objectives

1. Deﬁne a rigorous evaluation framework based on available datasets

• Explore the available datasets listed in the next section and deﬁne a “priority”

list, indicating the most suitable ones for the current evaluation.

o The  list  is  surely  not  exhaustive  and  novel  datasets  come  out  quite
frequently. In case we ﬁnd further nice datasets, we can consider them.
• Based  on  the  shortlisted  datasets,  deﬁne  a  rigorous  evaluation  framework,
including  number  of  experts  (based  on  metadata),  respective  distribution
shifts, downstream evaluation metrics, and so on.

2. Develop conditioned Generative Models

• Develop conditioned generative models for privacy-preserving data sampling

by adopting foundation models for feature extraction as in [5,6].

• These  models  will  be  conditioned  on  demographic  and  acquisition-related

attributes to enable controlled and fair sampling across subgroups.

3. Design a Metadata-Based Routing Mechanism

• Deﬁne  domain  similarity  metrics  using  structured  metadata  (e.g.,  patient
cohort  statistics,  scanner  vendor,  resolution,  staining  protocol)  or  using  raw-
data statistics.

• Develop  a  routing  mechanism  that  dynamically  selects  a  relevant  subset  of

experts for each generation task.

4. Construct the MoErging Generator

• Use  independently  trained  generative  models  (e.g.,  Conditional  VAEs  [5,6])

for each source domain.

• Develop  an  inference-time  composition  method  that  routes  generation
requests  through  selected  experts  without  retraining  or  parameter  sharing. A
possible baseline is given by FedAvg on experts’ decoders [6].

5. Evaluate domain-speciﬁc and generalization performance

• Assess generation quality and downstream classiﬁcation performance.
• Benchmark across unseen test domains with varying metadata overlap.
• Compare against baselines such as naive ensembling and weight averaging

Positioned  at  the  intersection  of  generative  models,  model  aggregation,  mixture-of-
experts , and privacy , this project contributes to multiple active research areas and
holds strong potential for  a publishable outcome!

Datsets

• MIDOG++    [7],  distributed  under  MIT  License. This  dataset  provides  explicit
metadata  on  scanner,  laboratory,  and  staining  protocols.  A  clear  split
separation is further deﬁned in OpenMIBOOD [8].

• Camelyon17  [9,10]  shared  under  Creative  Commons  Zero  (CC0)  License.
This  is  an  established  dataset  including  slides  acquired  from  ﬁve  different
hospitals.

• BCNB

  [11],  custom  license,  available  for  academic  research  use.  It

comprises patient- and study-level clinical metadata.

• BreakHis

  [12],  distributed  under  the  Creative  Commons  Attribution  4.0
International License (CC BY-NC 4.0). It includes slides acquired with different
magniﬁcation factors (40×, 100×, 200×, 400×).
(optional) PathMNIST-C [13], distributed under the Creative Commons (CC)
License.  It  includes  real-world  corruption  types  utilized  for  benchmarking
downstream model robustness against distribution shifts.

•

o Optional - Another rigorous evaluation and stress test protocol for our
routing  can  be  via  controlled  data  with  synthetic  shifts,  where
distribution shifts are induced by image-based realistic corruptions (Di
Salvo,  Doerrich,  &  Ledig,  2024)  and  the  corresponding  augmentation
parameters  are  encoded  as  metadata.  This  will  enable  controlled
robustness stress tests and a comprehensive validation of our routing
and  aggregation  strategies  in  a  setting  where  the  metadata-shift
relationship is fully known.

References

[1] Yadav, P., Raffel, C., Muqeeth, M., Caccia, L., Liu, H., Chen, T., ... & Sordoni, A.
(2024).  A  survey  on  model  moerging:  Recycling  and  routing  among  specialized
experts for collaborative learning. arXiv preprint arXiv:2408.07057.

[2] Xu, Z., Yuan, K., Wang, H., Wang, Y., Song, M., & Song, J. (2024). Training-free
pretrained  model  merging.  In  Proceedings  of  the  IEEE/CVF  Conference  on
Computer Vision and Pattern Recognition (pp. 5915-5925).

[3] Jacobs, RA, Jordan, MI, Nowlan, SJ, & Hinton, GE (1991). Adaptive mixtures of
local experts. Neural computation.

[4] Jiang, AQ, Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., ... &
Sayed, WE (2024). Mixtral of experts. arXiv preprint arXiv:2401.04088.

[5]  Di  Salvo,  F.,  Taﬂer,  D.,  Doerrich,  S.,  &  Ledig,  C.  (2024).  Privacy-preserving
datasets  by  capturing  feature  distributions  with  Conditional  VAEs.  The  35th  British
Machine Vision Conference (BMVC).

[6] Di Salvo, F., Nguyen, H. H. M., & Ledig, C. (2025, September). Embedding-based
federated  data  sharing  via  differentially  private  conditional  VAEs.  In  International
Conference  on  Medical  Image  Computing  and  Computer-Assisted  Intervention  (pp.
138-147). Cham: Springer Nature Switzerland.

[7] Aubreville, M., Wilm, F., Stathonikos, N., Breininger, K., Donovan, T. A., Jabari, S.,
... & Bertram, C. A. (2023). A comprehensive multi-domain dataset for mitotic ﬁgure
detection. Scientiﬁc data, 10(1), 484.

[8]  Gutbrod,  M.,  Rauber,  D.,  Nunes,  D.  W.,  &  Palm,  C.  (2025).  Openmibood:  Open
medical imaging benchmarks for out-of-distribution detection. In Proceedings of the
IEEE/CVF  Conference  on  Computer  Vision  and  Pattern  Recognition  (pp.
25874-25886).

[9] Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., ...
&  Liang,  P.  (2021,  July).  Wilds:  A  benchmark  of  in-the-wild  distribution  shifts.  In
International conference on machine learning (pp. 5637-5664). PMLR.

[10] Bandi, P., Geessink, O., Manson, Q., Van Dijk, M., Balkenhol, M., Hermsen, M., .
.  .  Zhong,  A.  (2018).  From  detection  of  individual  metastases  to  classiﬁcation  of
lymph  node  status  at  the  patient  level:  the  CAMELYON17  challenge.  IEEE
Transactions on Medical Imaging.

[11]  Xu,  F.,  Zhu,  C.,  Tang,  W.,  Wang,  Y.,  Zhang,  Y.,  Li,  J.,  ...  &  Jin,  M.  (2021).
Predicting axillary lymph node metastasis in early breast cancer using deep learning
on primary tumor biopsy slides. Frontiers in oncology.

[12]  Spanhol,  F. A.,  Oliveira,  L.  S.,  Petitjean,  C.,  &  Heutte,  L.  (2015). A  dataset  for
breast  cancer  histopathological  image  classiﬁcation.  IEEE  transactions  on
biomedical engineering, 63(7), 1455-1462.

[13]  Di  Salvo,  F.,  Doerrich,  S.,  &  Ledig,  C.  (2024).  MedMNIST-C:  Comprehensive
benchmark  and  improved  classiﬁer  robustness  by  simulating  realistic  image
corruptions.  Workshop  on  Advancing  Data  Solutions  in  Medical  Imaging  AI  at
MICCAI.


