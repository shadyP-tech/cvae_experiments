"""Durable prelabel and scored bundle persistence for fresh HARP."""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...runtime.harp_probability_menu.hashing import canonical_sha256, raw_array_sha256
from ...runtime.harp_probability_menu import EXACT_NINE_SEED_PAIRS
from .config import HarpFreshStage70Config
from .policy import FrozenHarpPolicy
from .scoring import HarpFreshDescriptiveResult
from .sealing import HarpFreshPrelabelSeal
from .target_loading import HarpFreshLoadedTarget
from .workspace_binding import HarpFreshWorkspaceBinding


_CONTENT_EXCLUSIONS = {
    "manifests/content_index.json",
    "reports/validation_report.json",
    "reports/run_state.json",
}


def _fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError("Fresh HARP metric table cannot be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(rows[0])
    if any(tuple(row) != columns for row in rows):
        raise ProtocolError("Fresh HARP metric rows have inconsistent columns.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def harp_prelabel_durable_hash(
    config: HarpFreshStage70Config,
    policy: FrozenHarpPolicy,
    target: HarpFreshLoadedTarget,
    menu_seal_hash: str,
    routed_vector_hashes: Sequence[str],
    physical_ablation_routed_vector_hashes: Sequence[str],
    physical_ablation_reference_preserving_sha256: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "schema_version": "midogpp_harp_fresh_durable_prelabel_plan_v2",
            "config_contract_hash": config.contract_hash,
            "policy_lock_hash": policy.metadata.policy_lock_hash,
            "policy_receipt_hash": policy.policy_receipt_hash,
            "reservation_hash": target.cache.reservation.reservation_hash,
            "target_cache_hash": target.cache.cache_hash,
            "target_cache_content_hash": target.cache_content_hash,
            "prediction_menu_seal_hash": menu_seal_hash,
            "routed_vector_hashes": list(routed_vector_hashes),
            "physical_ablation_routed_vector_hashes": list(
                physical_ablation_routed_vector_hashes
            ),
            "physical_ablation_reference_preserving_sha256": list(
                physical_ablation_reference_preserving_sha256
            ),
            "physical_ablation_action_universe": "Hxe_lambda_one_only",
            "physical_ablation_reference_preserving_semantics": (
                "eligible_Hxe_lambda_one_else_exact_U"
            ),
            "physical_ablation_selection_labels_used": False,
            "all_routes_and_vectors_complete": True,
            "labels_opened": False,
        }
    )


def write_harp_fresh_prelabel_bundle(
    root: Path,
    *,
    config: HarpFreshStage70Config,
    binding: HarpFreshWorkspaceBinding,
    policy: FrozenHarpPolicy,
    target: HarpFreshLoadedTarget,
    seal: HarpFreshPrelabelSeal,
) -> str:
    """Fsync the complete menu/routes/vectors before label authorization."""

    root.mkdir(parents=True, exist_ok=True)
    expected_durable = harp_prelabel_durable_hash(
        config,
        policy,
        target,
        seal.menu.seal_hash,
        [vector.routed_vector_seal_hash for vector in seal.routed_vectors],
        [
            vector.routed_vector_seal_hash
            for vector in seal.physical_ablation_vectors
        ],
        seal.physical_ablation_reference_preserving_sha256,
    )
    if seal.durable_bundle_hash != expected_durable:
        raise ProtocolError("Fresh HARP durable prelabel identity drifted.")
    _atomic_bytes(root / "config.resolved.yaml", config.source_path.read_bytes())
    provenance = {
        "schema_version": "midogpp_harp_fresh_input_provenance_v1",
        "experiment_id": config.experiment_id,
        "output_artifact_id": config.output_artifact_id,
        "input_artifact_ids": list(config.input_artifact_ids),
        "workspace_binding_hash": binding.binding_hash,
        "policy_lock_hash": policy.metadata.policy_lock_hash,
        "policy_receipt_hash": policy.policy_receipt_hash,
        "reservation_hash": target.cache.reservation.reservation_hash,
        "target_cache_hash": target.cache.cache_hash,
        "target_cache_content_hash": target.cache_content_hash,
        "scoring_manifest_sha256": target.scoring_manifest_sha256,
        "selection_used_target_labels": False,
        "physical_ablation_selection_used_target_labels": False,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
    }
    provenance["provenance_hash"] = canonical_sha256(provenance)
    _atomic_json(root / "provenance/input_artifacts.json", provenance)
    menu_payload = {
        "schema_version": "midogpp_harp_fresh_prediction_menu_manifest_v1",
        "status": seal.menu.status,
        "prediction_menu_seal_hash": seal.menu.seal_hash,
        "action_menu_hash": seal.menu.action_menu_hash,
        "prediction_store_hash": seal.menu.prediction_store_hash,
        "action_count": len(seal.menu.actions),
        "prediction_cell_count": len(seal.menu.cells),
        "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
        "workstation": seal.menu.workstation.to_payload(),
        "workstation_hash": seal.menu.workstation.runtime_hash,
        "prediction_checkpoint_root": "checkpoints/predictions",
        "actions": [action.to_payload() for action in seal.menu.actions],
        "prediction_cells": [cell.to_payload() for cell in seal.menu.cells],
        "labels_consumed": False,
    }
    menu_payload["manifest_hash"] = canonical_sha256(menu_payload)
    _atomic_json(root / "manifests/prediction_menu_seal.json", menu_payload)
    route_payload = {
        "schema_version": "midogpp_harp_fresh_route_set_manifest_v2",
        "status": "DURABLE_ALL_ROUTES_SEALED_BEFORE_LABELS",
        "prelabel_seal_hash": seal.seal_hash,
        "route_set_hash": seal.route_set_hash,
        "prediction_menu_seal_hash": seal.menu.seal_hash,
        "policy_lock_hash": seal.policy_hash,
        "reservation_hash": seal.reservation_hash,
        "target_cache_hash": seal.target_cache_hash,
        "durable_bundle_hash": seal.durable_bundle_hash,
        "independent_validation_hashes": list(seal.independent_validation_hashes),
        "routed_vector_hashes": [
            vector.routed_vector_seal_hash for vector in seal.routed_vectors
        ],
        "physical_ablation_routed_vector_hashes": [
            vector.routed_vector_seal_hash
            for vector in seal.physical_ablation_vectors
        ],
        "physical_ablation_reference_preserving_sha256": list(
            seal.physical_ablation_reference_preserving_sha256
        ),
        "physical_ablation_action_universe": "Hxe_lambda_one_only",
        "physical_ablation_reference_preserving_semantics": (
            "eligible_Hxe_lambda_one_else_exact_U"
        ),
        "decisions": [
            decision.to_payload()
            for vector in seal.routed_vectors
            for decision in vector.decisions
        ],
        "physical_ablation_decisions": [
            decision.to_payload()
            for vector in seal.physical_ablation_vectors
            for decision in vector.decisions
        ],
        "physical_ablation_selection_labels_used": False,
        "labels_opened": False,
    }
    route_payload["manifest_hash"] = canonical_sha256(route_payload)
    _atomic_json(root / "manifests/route_set_seal.json", route_payload)
    arrays: dict[str, np.ndarray] = {}
    for center, vector in zip(CENTERS, seal.routed_vectors, strict=True):
        arrays[f"center_{center}_baseline"] = vector.baseline_probabilities
        arrays[f"center_{center}_reference"] = vector.reference_probabilities
        arrays[f"center_{center}_selected"] = vector.selected_action_probabilities
        arrays[f"center_{center}_routed"] = vector.routed_probabilities
    for center, vector, reference_preserving in zip(
        CENTERS,
        seal.physical_ablation_vectors,
        seal.physical_ablation_reference_preserving_vectors,
        strict=True,
    ):
        prefix = f"center_{center}_physical_ablation"
        arrays[f"{prefix}_baseline"] = vector.baseline_probabilities
        arrays[f"{prefix}_reference"] = vector.reference_probabilities
        arrays[f"{prefix}_selected"] = vector.selected_action_probabilities
        arrays[f"{prefix}_routed"] = vector.routed_probabilities
        arrays[f"{prefix}_reference_preserving"] = reference_preserving
    array_path = root / "arrays/routed_probabilities.npz"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(f".{array_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, array_path)
    _fsync_directory(array_path.parent)
    # Re-open the durable archive and verify exact float64 bytes before labels.
    with np.load(array_path, allow_pickle=False) as stored:
        for name, values in arrays.items():
            observed = np.asarray(stored[name])
            if observed.dtype != np.float64 or raw_array_sha256(observed) != raw_array_sha256(values):
                raise ProtocolError("Fresh HARP routed-vector archive changed bytes.")
    members = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {
            "manifests/prelabel_content_index.json",
            "reports/run_state.json",
        }
    )
    records = [{"path": member, "sha256": _sha256_file(root / member)} for member in members]
    index = {
        "schema_version": "midogpp_harp_fresh_prelabel_content_index_v2",
        "status": "COMPLETE_BEFORE_LABEL_ACCESS",
        "durable_bundle_hash": seal.durable_bundle_hash,
        "prelabel_seal_hash": seal.seal_hash,
        "file_count": len(records),
        "files": records,
        "labels_opened": False,
    }
    index["content_hash"] = canonical_sha256(index)
    _atomic_json(root / "manifests/prelabel_content_index.json", index)
    return str(index["content_hash"])


def write_harp_fresh_scored_bundle(
    root: Path,
    *,
    seal: HarpFreshPrelabelSeal,
    result: HarpFreshDescriptiveResult,
    prelabel_content_hash: str,
) -> None:
    if result.prediction_seal_hash != seal.seal_hash:
        raise ProtocolError("Fresh HARP scored result escaped its prelabel seal.")
    _write_csv(
        root / "tables/case_metrics.csv",
        [asdict(row) for row in result.case_metrics],
    )
    _write_csv(
        root / "tables/center_metrics.csv",
        [asdict(row) for row in result.center_metrics],
    )
    _write_csv(
        root / "tables/action_matrix_metrics.csv",
        [asdict(row) for row in result.oracle_diagnostics.action_matrix],
    )
    inference = {
        "schema_version": "midogpp_harp_fresh_center_inference_v1",
        "prediction_seal_hash": seal.seal_hash,
        "result_hash": result.result_hash,
        "oracle_diagnostics_result_hash": result.oracle_diagnostics.result_hash,
        "inference_unit": "target_center",
        "inference_unit_count": len(CENTERS),
        "seed_cells_are_inference_units": False,
        "summaries": [asdict(row) for row in result.center_inference],
        "fresh_claim_requires_completed_eligible_reservation_and_bundle": True,
    }
    inference["inference_hash"] = canonical_sha256(inference)
    _atomic_json(root / "reports/center_inference.json", inference)
    oracle = {
        "schema_version": "midogpp_harp_fresh_action_oracle_report_v2",
        "prelabel_seal_hash": seal.seal_hash,
        "oracle_result_hash": result.oracle_diagnostics.result_hash,
        "action_matrix_row_count": len(result.oracle_diagnostics.action_matrix),
        "center_diagnostics": [
            asdict(row) for row in result.oracle_diagnostics.center_diagnostics
        ],
        "diagnostic_only": True,
        "labels_used_after_route_seal_only": True,
        "physical_ablation_scored_after_prelabel_seal_only": True,
        "labels_available_to_policy": False,
        "policy_or_threshold_update_emitted": False,
    }
    oracle["oracle_report_hash"] = canonical_sha256(oracle)
    _atomic_json(root / "reports/action_oracle_diagnostics.json", oracle)
    leakage = {
        "schema_version": "midogpp_harp_fresh_leakage_report_v1",
        "status": "PASS",
        "prelabel_seal_hash": seal.seal_hash,
        "prelabel_content_hash": prelabel_content_hash,
        "complete_action_menu_before_routing": True,
        "complete_routes_and_vectors_before_labels": True,
        "complete_physical_ablation_routes_and_vectors_before_labels": True,
        "labels_used_for_scoring_only": True,
        "labels_available_to_generation_prediction_or_routing": False,
        "full_action_matrix_scored_after_route_seal_only": True,
        "oracle_diagnostics_available_to_policy": False,
        "physical_ablation_selection_labels_used": False,
        "oracle_diagnostics_may_update_policy_or_thresholds": False,
        "support_evaluation_cases_globally_disjoint": True,
        "target_expert_excluded": True,
        "consumed_test_used": False,
        "consumed_test_rows_used": False,
        "consumed_validation_used": False,
        "consumed_validation_rows_used": False,
        "consumed_stage90_used": False,
        "stage50_or_stage90_artifacts_used": False,
        "policy_update_emitted": False,
    }
    leakage["leakage_hash"] = canonical_sha256(leakage)
    _atomic_json(root / "reports/leakage_report.json", leakage)


def write_harp_fresh_content_index(root: Path) -> str:
    members = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in _CONTENT_EXCLUSIONS
    )
    records = [{"path": member, "sha256": _sha256_file(root / member)} for member in members]
    payload = {
        "schema_version": "midogpp_harp_fresh_content_index_v1",
        "status": "COMPLETE",
        "file_count": len(records),
        "files": records,
        "excluded_members": sorted(_CONTENT_EXCLUSIONS),
        "scratch_authoritative": False,
    }
    payload["content_hash"] = canonical_sha256(payload)
    _atomic_json(root / "manifests/content_index.json", payload)
    return str(payload["content_hash"])


__all__ = (
    "harp_prelabel_durable_hash",
    "write_harp_fresh_content_index",
    "write_harp_fresh_prelabel_bundle",
    "write_harp_fresh_scored_bundle",
)
