from __future__ import annotations

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.action_compiler import (
    BasePredictionSurface,
    canonical_compiler_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.candidate_pools import (
    build_held_center_candidate_pool,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_supervision import (
    SourceSupervisionContractReceipt,
    compile_verified_source_block,
    reconstruct_compiler_recomputation_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.held_actions import (
    canonical_held_action_library,
)


def test_recomputation_receipt_is_derived_from_all_actual_label_free_blocks() -> None:
    compiler = canonical_compiler_receipt()
    library = canonical_held_action_library()
    producer_seal = "a" * 64
    contract = SourceSupervisionContractReceipt(
        compiler_receipt_hash=compiler.receipt_hash,
        producer_source_seal_sha256=producer_seal,
        held_action_library_sha256=library.library_hash,
        held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
    )
    inventory = tuple((f"expert-{center}", center) for center in CENTERS)
    bases = []
    compiled = []
    for h in CENTERS:
        for q in CENTERS:
            if q == h:
                continue
            pool = build_held_center_candidate_pool(
                outer_target_center=h,
                held_center=q,
                all_center_ids=CENTERS,
                expert_inventory=inventory,
                bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
                source_supervision_contract_hash=contract.contract_hash,
                compiler=compiler,
            )
            row_ids = (f"{h}-{q}-0", f"{h}-{q}-1")
            base = BasePredictionSurface(
                outer_target_center=h,
                evaluated_center=q,
                row_ids=row_ids,
                equal_union_probabilities=(0.30, 0.70),
                union_probabilities=(0.40, 0.60),
                expert_probabilities=tuple(
                    (center, (0.20 + index * 0.01, 0.80 - index * 0.01))
                    for index, center in enumerate(pool.candidate_center_ids)
                ),
                candidate_pool_receipt_hash=pool.receipt_hash,
            )
            lineage, result = compile_verified_source_block(
                base,
                candidate_pool=pool,
                compiler=compiler,
                producer_source_seal_sha256=producer_seal,
            )
            bases.append(lineage)
            compiled.append(result.receipt)
    receipt = reconstruct_compiler_recomputation_receipt(
        producer_source_seal_sha256=producer_seal,
        compiler=compiler,
        base_probability_lineage_receipts=bases,
        compiled_surface_receipts=compiled,
    )
    assert len(receipt.block_pairs) == 72
    assert len(receipt.receipt_hash) == 64
    assert contract.contract_hash not in {receipt.receipt_hash, producer_seal}
    assert contract.manifest_payload(
        compiler_recomputation_receipt_sha256=receipt.receipt_hash
    )["producer_compiler_recomputation_receipt_sha256"] == receipt.receipt_hash
