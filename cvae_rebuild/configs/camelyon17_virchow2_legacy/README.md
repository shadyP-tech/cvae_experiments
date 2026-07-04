# Camelyon17 Virchow2 Legacy Configs

This folder is a quarantine for CVAE rebuild configs that use the Camelyon17
center-domain regime with Virchow2 embeddings.

Identification criteria for this set:

- `heldout_centers` is the Camelyon17 five-center set `0,1,2,3,4`
- the feature cache root points to the legacy `pathology_embeddings/virchow2`
  cache family
- no `domain_regime: midogpp_annotation_patch_v1` contract is present

Do not use these configs as MIDOG++ Virchow2 configs. MIDOG++ configs remain in
`cvae_rebuild/configs/` with explicit `midogpp` names or artifact roots under
`cvae_rebuild/artifacts/midogpp/`.
