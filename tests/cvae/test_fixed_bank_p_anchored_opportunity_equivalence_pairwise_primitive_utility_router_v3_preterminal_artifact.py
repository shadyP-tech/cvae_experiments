from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.candidate_pools import (
    ALL_ACTION_IDS,
    P_ACTION_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.preterminal_artifact import (
    MANIFEST_MEMBER,
    MATRIX_MEMBER,
    PersistedPreterminalArtifact,
    _array_sha256,
    _validate_preterminal_files,
    attest_preterminal_artifact_twice,
    attest_terminal_aggregate_twice,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.science.target_inventory import (
    CANONICAL_TARGET_CASE_INVENTORY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal.contracts import (
    ALLOWED_AGGREGATE_METRICS,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
)


def _persist_synthetic_preterminal(root: Path) -> PersistedPreterminalArtifact:
    matrix_path = root / MATRIX_MEMBER
    manifest_path = root / MANIFEST_MEMBER
    matrix_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    values = np.full(EXPECTED_PROBABILITY_MATRIX_SHAPE, 0.5, dtype="<f4")
    np.save(matrix_path, values, allow_pickle=False)

    offsets: dict[str, tuple[int, int]] = {}
    row_ids: list[str] = []
    bindings: list[tuple[str, str, str]] = []
    decision_rows = []
    decision_hashes = []
    outer_hashes = tuple(canonical_hash(("outer", center)) for center in CENTERS)
    surface_hashes = tuple(canonical_hash(("surface", center)) for center in CENTERS)
    cursor = 0
    counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    for center_index, center in enumerate(CENTERS):
        center_cases = tuple(
            case
            for candidate_center, case in CANONICAL_TARGET_CASE_INVENTORY
            if candidate_center == center
        )
        start = cursor
        quotient, remainder = divmod(counts[center], len(center_cases))
        local_cursor = 0
        for case_index, case in enumerate(center_cases):
            case_count = quotient + (case_index < remainder)
            local_indices = tuple(range(local_cursor, local_cursor + case_count))
            global_indices = tuple(start + value for value in local_indices)
            case_rows = tuple(f"row-{value:05d}" for value in global_indices)
            row_ids.extend(case_rows)
            bindings.extend((row_id, center, case) for row_id in case_rows)
            row_manifest_hash = canonical_hash(case_rows)
            scores = tuple((action, None) for action in ALL_ACTION_IDS)
            body = {
                "schema": "oe_ppur_v3_preterminal_target_case_decision_v1",
                "center_id": center,
                "case_id": case,
                "selected_action_id": P_ACTION_ID,
                "reason": "synthetic_exact_p_fallback",
                "row_indices": local_indices,
                "row_manifest_hash": row_manifest_hash,
                "outer_result_hash": outer_hashes[center_index],
                "predicted_action_scores": scores,
                "rank_available": False,
                "admission_decision_receipt_hash": None,
                "selection_decision_hash": None,
                "exact_P_fallback": True,
                "target_labels_used": False,
            }
            decision_hash = canonical_hash(body)
            decision_hashes.append(decision_hash)
            decision_rows.append(
                {
                    "center_id": center,
                    "case_id": case,
                    "selected_action_id": P_ACTION_ID,
                    "reason": "synthetic_exact_p_fallback",
                    "row_indices": list(local_indices),
                    "row_manifest_hash": row_manifest_hash,
                    "outer_result_hash": outer_hashes[center_index],
                    "predicted_action_scores": [list(value) for value in scores],
                    "rank_available": False,
                    "admission_decision_receipt_hash": None,
                    "selection_decision_hash": None,
                    "decision_hash": decision_hash,
                }
            )
            local_cursor += case_count
        cursor += counts[center]
        offsets[center] = (start, cursor)

    row_ids_tuple = tuple(row_ids)
    surfaces = tuple(zip(CENTERS, surface_hashes, strict=True))
    matrix_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_compiled_probability_matrix_v1",
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "row_ids_sha256": canonical_hash(row_ids_tuple),
            "center_offsets": offsets,
            "action_ids": ALL_ACTION_IDS,
            "matrix_f4_sha256": _array_sha256(values),
            "surface_hashes": surfaces,
            "labels_present": False,
        }
    )
    ledger_hash = canonical_hash(
        {
            "schema": "oe_ppur_v3_exact_218_case_preterminal_ledger_v1",
            "case_inventory": CANONICAL_TARGET_CASE_INVENTORY,
            "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            "decision_hashes": tuple(decision_hashes),
            "exact_P_count": EXPECTED_CASE_COUNT,
            "rank_unavailable_count": EXPECTED_CASE_COUNT,
            "rank_diagnostic_policy": "AVAILABLE_CASES_ONLY_NO_IMPUTATION",
            "terminal_labels_opened": False,
        }
    )
    request_hash = "1" * 64
    factory_hash = "2" * 64
    seven_hash = "3" * 64
    seal_hash = "4" * 64
    source_hash = "5" * 64
    pool_hashes = tuple(canonical_hash(("pool", center)) for center in CENTERS)
    result_hash = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_complete_preterminal_result_v1",
            "request_hash": request_hash,
            "service_factory_identity_hash": factory_hash,
            "seven_input_contract_hash": seven_hash,
            "source_seal_hash": seal_hash,
            "source_training_surface_receipt_hash": source_hash,
            "final_pool_receipt_hashes": pool_hashes,
            "outer_science_result_hashes": outer_hashes,
            "final_surface_hashes": surface_hashes,
            "probability_matrix_hash": matrix_hash,
            "decision_ledger_hash": ledger_hash,
            "case_count": EXPECTED_CASE_COUNT,
            "exact_P_count": EXPECTED_CASE_COUNT,
            "target_labels_opened": False,
        }
    )
    payload = {
        "schema_version": "oe_ppur_v3_persisted_preterminal_result_v1",
        "result_hash": result_hash,
        "request_hash": request_hash,
        "service_factory_identity_hash": factory_hash,
        "seven_input_contract_hash": seven_hash,
        "source_seal_hash": seal_hash,
        "source_training_surface_receipt_hash": source_hash,
        "final_pool_receipt_hashes": list(pool_hashes),
        "outer_science_result_hashes": list(outer_hashes),
        "final_surface_hashes": list(surface_hashes),
        "probability_matrix_hash": matrix_hash,
        "matrix_shape": list(values.shape),
        "matrix_dtype": values.dtype.str,
        "matrix_f4_sha256": _array_sha256(values),
        "matrix_row_ids": list(row_ids_tuple),
        "matrix_center_offsets": {
            center: list(offsets[center]) for center in CENTERS
        },
        "matrix_action_ids": list(ALL_ACTION_IDS),
        "matrix_surface_hashes": [list(value) for value in surfaces],
        "row_bindings": [list(value) for value in bindings],
        "case_inventory": [list(value) for value in CANONICAL_TARGET_CASE_INVENTORY],
        "case_inventory_sha256": EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        "decisions": decision_rows,
        "decision_ledger_hash": ledger_hash,
        "exact_p_count": EXPECTED_CASE_COUNT,
        "rank_unavailable_count": EXPECTED_CASE_COUNT,
        "target_labels_opened": False,
    }
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    validated = _validate_preterminal_files(
        manifest_path,
        matrix_path,
        expected_ledger_hash=ledger_hash,
        expected_result_hash=result_hash,
    )
    return PersistedPreterminalArtifact(
        root=root,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        artifact_file_sha256=str(validated["artifact_file_sha256"]),
        artifact_file_identity_sha256=str(
            validated["artifact_file_identity_sha256"]
        ),
        decision_ledger_hash=ledger_hash,
        result_hash=result_hash,
    )


def test_preterminal_artifact_is_revalidated_by_two_fresh_processes(
    tmp_path: Path,
) -> None:
    artifact = _persist_synthetic_preterminal(tmp_path / "artifact")
    attestations = attest_preterminal_artifact_twice(artifact, timeout_seconds=60)

    assert len({row.process_pid for row in attestations}) == 2
    assert {row.artifact_file_sha256 for row in attestations} == {
        artifact.artifact_file_sha256
    }
    assert all(
        row.sealed_ledger_receipt_hash == artifact.decision_ledger_hash
        for row in attestations
    )


def test_final_aggregate_is_revalidated_by_two_fresh_processes(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "oe_ppur_v3_aggregate_only_terminal_receipt_v1",
        "boundary_receipt_hash": "a" * 64,
        "decision_ledger_receipt_hash": "b" * 64,
        "evaluated_case_count": EXPECTED_CASE_COUNT,
        "routed_case_count": 0,
        "exact_p_fallback_count": EXPECTED_CASE_COUNT,
        "aggregate_metrics": {
            name: 0.0 for name in ALLOWED_AGGREGATE_METRICS
        },
        "raw_paths_present": False,
        "raw_labels_present": False,
        "per_row_values_present": False,
        "per_case_values_present": False,
    }
    payload["receipt_hash"] = canonical_hash(payload)
    receipt = _reconstruct_persisted_aggregate_only_terminal_receipt(payload)
    path = tmp_path / "terminal_metrics.json"
    raw = json.dumps(
        receipt.to_payload(), sort_keys=True, separators=(",", ":")
    ) + "\n"
    path.write_text(raw, encoding="utf-8")
    attestation = attest_terminal_aggregate_twice(
        path,
        receipt,
        timeout_seconds=60,
    )

    assert len(set(attestation.validator_process_pids)) == 2
    assert attestation.terminal_receipt_hash == receipt.receipt_hash
    assert attestation.terminal_file_sha256 == hashlib.sha256(raw.encode()).hexdigest()
