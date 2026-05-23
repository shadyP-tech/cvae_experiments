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
