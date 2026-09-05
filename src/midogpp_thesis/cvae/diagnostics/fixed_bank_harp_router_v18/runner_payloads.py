"""Deterministic durable manifests for the HARP v18 phase coordinator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import sha256_file
from .identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION


def seal(payload: Mapping[str, object], *, field: str = "seal_hash") -> dict[str, object]:
    body = dict(payload)
    return {**body, field: canonical_hash(body)}


def build_surface_seal_indexes(
    seal_sets: Sequence[object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = tuple(sorted(seal_sets, key=_center_id))
    if (
        tuple(_center_id(row) for row in rows) != CENTERS
        or len({_center_id(row) for row in rows}) != len(CENTERS)
    ):
        raise ProtocolError("HARP v18 seal indexes require exact center coverage.")
    common = {
        "experiment_id": EXPERIMENT_ID,
        "ordered_center_ids": [_center_id(row) for row in rows],
        "source_train_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    source_train = seal(
        {
            "schema_version": "midogpp_harp_v18_source_train_menu_seal_index_v1",
            **common,
            "role": "source_train",
            "members": [
                {
                    "center_id": _center_id(row),
                    "path": str(row.source_train_menu_seal_path),
                    "sha256": row.source_train_menu_seal_sha256,
                    "seal_hash": row.source_train_menu_seal_hash,
                    "store_receipt_hash": row.physical_store_receipt_hash,
                }
                for row in rows
            ],
        },
        field="index_hash",
    )
    target = seal(
        {
            "schema_version": "midogpp_harp_v18_target_evaluation_menu_seal_index_v1",
            **common,
            "role": "target",
            "members": [
                {
                    "center_id": _center_id(row),
                    "path": str(row.target_evaluation_menu_seal_path),
                    "sha256": row.target_evaluation_menu_seal_sha256,
                    "seal_hash": row.target_evaluation_menu_seal_hash,
                    "store_receipt_hash": row.physical_store_receipt_hash,
                }
                for row in rows
            ],
        },
        field="index_hash",
    )
    attestations = seal(
        {
            "schema_version": "midogpp_harp_v18_bank_independence_attestation_index_v1",
            **common,
            "members": [
                {
                    "center_id": _center_id(row),
                    "path": str(row.bank_independence_attestation_path),
                    "sha256": row.bank_independence_attestation_sha256,
                }
                for row in rows
            ],
            "own_center_expert_unrepresentable_per_context": True,
        },
        field="index_hash",
    )
    return source_train, target, attestations


def _center_id(value: object) -> str:
    return str(getattr(value, "center_id", ""))


def build_model_lock(
    *,
    config_hash: str,
    centers: Sequence[str],
    source_train_surface: object,
    source_train_receipt: object,
    fitted: object,
    model_receipt: object,
    compatibility: object,
    compatibility_receipt: object,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v18_model_lock_v1",
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "source_train_surface_hash": source_train_surface.manifest["surface_hash"],
            "source_train_store_manifest_sha256": source_train_receipt.manifest_sha256,
            "source_train_store_npz_sha256": source_train_receipt.npz_sha256,
            "compatibility_feature_hash": compatibility.manifest[
                "compatibility_feature_hash"
            ],
            "compatibility_store_manifest_sha256": compatibility_receipt.manifest_sha256,
            "compatibility_store_npz_sha256": compatibility_receipt.npz_sha256,
            "model_hash": fitted.manifest["model_hash"],
            "policy_hash": fitted.manifest["policy_hash"],
            "model_store_manifest_sha256": model_receipt.manifest_sha256,
            "model_store_npz_sha256": model_receipt.npz_sha256,
            "scientific_manifest": dict(fitted.manifest),
            "source_train_only": True,
            "pooled_known_center_policy_count": 1,
            "nested_center_stratified_outer_folds": 5,
            "nested_center_stratified_inner_folds": 4,
            "target_evaluation_features_used_for_fit": False,
            "evaluation_labels_used": False,
        },
        field="model_lock_hash",
    )


def build_policy_admission_seal(
    *, config_hash: str, fitted: object
) -> dict[str, object]:
    policy = fitted.state.policy
    admission = policy.admission.public_payload()
    return seal(
        {
            "schema_version": "midogpp_harp_v18_source_policy_admission_seal_v1",
            "config_hash": config_hash,
            "model_hash": fitted.manifest["model_hash"],
            "policy_hash": fitted.manifest["policy_hash"],
            "selected_arm_id": policy.selected_arm_id,
            "route_threshold": policy.route_threshold,
            "source_only_admission": admission,
            "no_nonzero_safe_oof_coverage_aborts_before_target_actions": True,
            "other_nonadmission_uses_exact_b_fallback": True,
            "pooled_known_center_policy_count": 1,
            "whole_policy_source_oof_replayed": True,
            "approximate_source_oof_bounds_only": True,
            "evaluation_labels_used": False,
        }
    )


def build_target_action_seal(
    *,
    config_hash: str,
    centers: Sequence[str],
    target_actions: object,
    target_receipt: object,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v18_target_action_seal_v1",
            "status": "COMPLETE_BEFORE_EVALUATION_LABELS",
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "model_hash": target_actions.manifest["model_hash"],
            "policy_hash": target_actions.manifest["policy_hash"],
            "target_action_hash": target_actions.manifest["target_action_hash"],
            "target_case_count": target_actions.manifest["target_case_count"],
            "target_store_manifest_sha256": target_receipt.manifest_sha256,
            "target_store_npz_sha256": target_receipt.npz_sha256,
            "pooled_selected_policy_recipe": True,
            "soft_topk_k_values": [1, 2, 4],
            "soft_topk_lambda_values": [0.25, 0.5, 0.75, 1.0],
            "exact_b_fallback_byte_identical": True,
            "evaluation_labels_opened": False,
        }
    )


def build_prelabel_bundle(
    *,
    config_hash: str,
    centers: Sequence[str],
    source_train_surface_seal_hash: str,
    model_lock: Mapping[str, object],
    policy_admission_seal: Mapping[str, object],
    target_seal: Mapping[str, object],
    routes: object,
    route_receipt: object,
    route_summary: Mapping[str, object],
    rejection_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v18_prelabel_route_bundle_v1",
            "status": "DURABLE_CASE_ROUTES_BEFORE_EVALUATION_LABELS",
            "source_train_surface_seal_hash": source_train_surface_seal_hash,
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "model_lock_hash": model_lock["model_lock_hash"],
            "policy_admission_seal_hash": policy_admission_seal["seal_hash"],
            "target_action_seal_hash": target_seal["seal_hash"],
            "route_hash": routes.route_hash,
            "ordered_case_identity_hash": routes.ordered_case_identity_hash,
            "ordered_sample_identity_hash": routes.ordered_sample_identity_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "target_action_hash": routes.target_action_hash,
            "route_store_manifest_sha256": route_receipt.manifest_sha256,
            "route_store_npz_sha256": route_receipt.npz_sha256,
            "route_summary": dict(route_summary),
            "prelabel_rejection_diagnostic_hash": rejection_diagnostics[
                "diagnostic_hash"
            ],
            "case_consistent": True,
            "exact_b_fallback_byte_identity": route_summary[
                "exact_b_fallback_byte_identity"
            ],
            "evaluation_labels_opened": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        },
        field="bundle_hash",
    )


def build_validation_bundle(
    *,
    config_hash: str,
    centers: Sequence[str],
    validations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v18_fresh_validation_bundle_v1",
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "validations": [dict(value) for value in validations],
            "distinct_process_ids": len(
                {value["process_id"] for value in validations}
            )
            == 2,
            "evaluation_labels_opened": False,
        },
        field="bundle_hash",
    )


def build_frozen_route_seal(
    *,
    config_hash: str,
    centers: Sequence[str],
    prelabel: Mapping[str, object],
    routes: object,
    validation_bundle: Mapping[str, object],
    validations: Sequence[Mapping[str, object]],
    route_summary: Mapping[str, object],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v18_frozen_route_seal_v1",
            "status": "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS",
            "prelabel_bundle_hash": prelabel["bundle_hash"],
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "route_hash": routes.route_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "target_action_hash": routes.target_action_hash,
            "validation_bundle_hash": validation_bundle["bundle_hash"],
            "independent_validation_hashes": [
                value["validation_hash"] for value in validations
            ],
            "case_count": len(routes.cases),
            "ordered_case_identity_hash": routes.ordered_case_identity_hash,
            "ordered_sample_identity_hash": routes.ordered_sample_identity_hash,
            "exact_b_fallback_byte_identity": route_summary[
                "exact_b_fallback_byte_identity"
            ],
            "evaluation_labels_opened": False,
        }
    )


def build_run_state(
    *,
    root: Path,
    ledger: object,
    lease: object,
    finalization_path: Path,
    content_index_path: Path,
    terminal_paths: Sequence[Path],
    frozen: Mapping[str, object],
    sealed_routes: object,
    terminal_metrics: Mapping[str, object],
    scratch: Path,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_harp_v18_run_state_v1",
        "status": "COMPLETE_EXHAUSTED",
        "phase": "TERMINAL_DIAGNOSTIC_COMPLETE",
        "phase_order": list(ledger.observed),
        "authorization_lease_hash": lease.lease_hash,
        "authorization_finalization_sha256": sha256_file(finalization_path),
        "content_index_sha256": sha256_file(content_index_path),
        "terminal_member_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in terminal_paths
        },
        "frozen_route_seal_hash": frozen["seal_hash"],
        "evaluated_reconstructed_route_hash": sealed_routes.route_hash,
        "terminal_result_hash": terminal_metrics["result_hash"],
        "scratch_root_used": str(scratch),
        "completion_commit_protocol": (
            "fsync_files_then_atomic_marker_then_fsync_directories"
        ),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "final_commit": True,
    }


__all__ = (
    "build_frozen_route_seal",
    "build_model_lock",
    "build_policy_admission_seal",
    "build_prelabel_bundle",
    "build_run_state",
    "build_surface_seal_indexes",
    "build_target_action_seal",
    "build_validation_bundle",
    "seal",
)
