# Terms

## Purpose

Define recurring terms used in the thesis wiki.

## Key Claims

| Term | Definition |
| --- | --- |
| Compatibility | Expected utility of using expert/config `e` for query `q`. |
| True compatibility | `C_true(q,e) = -NELBO(q,e)`. |
| Proxy compatibility | Similarity or score used to estimate utility, such as metadata, latent distance, source-inner validation, or support statistics. |
| Query | Target domain, support set, target sample, or generation/adaptation request. |
| Expert | Source-trained generator, source-domain model, downstream classifier config, or candidate component. |
| Deployable evidence | Evidence selected without target evaluation labels. |
| Audit-only evidence | Diagnostic evidence that cannot justify a deployable claim. |
| Posthoc evidence | Evidence using target outcomes; feasibility only. |
| Oracle evidence | Upper bound using target outcomes. |
| Source-only selection | Selection using source/allowed-support information, not target evaluation labels. |
| SAIL | Source-only Aggregation via Inner-domain Leaveout. |
| R1.2c-V | Historical name for the Virchow2 dense source-selected config aggregation diagnostic extracted into SAIL. |
| R1.2c-X | Historical cross-backbone dense audit extension; audit-only. |
| CVAE preservation | Whether generated embeddings preserve real-feature downstream utility. |
| Source-union GMM prior | Centralized diagnostic prior fitted over pooled non-target source latent codes; useful as an upper bound, not decentralized deployment evidence. |
| Source-local latent summary | Target-agnostic GMM-style summary exported by a source expert without sharing raw source embeddings. |
| Raw-data-free summary exchange | Data-minimizing protocol that exchanges summaries or scores rather than raw images/embeddings; not formal differential privacy. |
| Source-local reliability | Target-agnostic score measuring whether a source-generated classifier preserves utility on that source's own real rows. |
| Heldout-excluded reliability | Source-local reliability recomputed for each heldout center using only non-heldout centers, so target-center outcomes do not influence weights, pooling, or synthetic budgets. |
| Dense all4 aggregation | Camelyon17 LOQDO aggregation that includes all four non-target source experts; it can test weighting, pooling, or budget allocation but not sparse expert selection. |
| Paired generation invariant | Audit rule requiring compared methods with the same source set, budget policy, heldout center, experiment seed, and replicate seed to share generated-feature and prediction hashes. |
| Paired dense-all4 reliability confirmation | D-series confirmation experiment showing that heldout-excluded source-local reliability improved dense all-source generated-embedding aggregation under paired generation and prediction invariants. |
| Support-NELBO weighting | Target-conditioned diagnostic that weights experts using unlabeled target support NELBO; not validated unless it beats matched controls. |
| Source-inner transfer | Source-only off-diagonal transfer score using non-target source eval centers as pseudo-targets. |
| Drop-one source selection | In Camelyon17 LOQDO with four candidate sources, top-3 selection is equivalent to excluding one source. |

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../context/current_experimental_state.md`

## Interpretation

The most important distinction is utility versus similarity. Similarity can be a proxy, but compatibility claims require utility evidence.

## Implication For Thesis

Use these terms consistently across thesis chapters and documentation.

## Limitations

Add terms as the final thesis vocabulary stabilizes.

## Next Checks

- Add exact definitions for any final named experiment variants.
