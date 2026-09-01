"""Pure manifest builders for the HARP v6 execution lifecycle.

The production runner owns phase ordering and capability transitions.  This
module owns only deterministic payload construction so the orchestration path
does not also become the schema implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import sha256_file
from .identity import PUBLICATION_STATUS, TERMINAL_DECISION


def seal(payload: Mapping[str, object], *, field: str = "seal_hash") -> dict[str, object]:
    """Return a copied payload with its canonical identity appended."""

    body = dict(payload)
    return {**body, field: canonical_hash(body)}


def build_development_surface_seal(
    *,
    config: Any,
    menus: Sequence[Any],
    menu_receipts: Sequence[Any],
    compatibility_hash: str,
    compatibility_receipt: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v6_development_surface_seal_v1",
            "status": "DURABLE_COMPLETE_LABEL_FREE_B_U_HXE_MENU",
            "outer_menu_hashes": {
                menu.outer_target_id: menu.menu_hash for menu in menus
            },
            "outer_menu_manifest_sha256": {
                receipt.root.name.removeprefix("outer_"): receipt.manifest_sha256
                for receipt in menu_receipts
            },
            "compatibility_hash": compatibility_hash,
            "compatibility_store_manifest_sha256": compatibility_receipt.manifest_sha256,
            "compatibility_store_npz_sha256": compatibility_receipt.npz_sha256,
            "compatibility_proxy_is_exact_nelbo": False,
            "compatibility_proxy_is_true_utility": False,
            "target_support_labels_consumed": False,
            "strict_outer_center_exclusion": True,
            "physical_expert_lambda_grid": list(
                config.protocol["physical_expert_lambda_grid"]
            ),
            "probability_transport_dtype": "float32",
            "all_action_cells_present_before_label_access": True,
            "labels_consumed": False,
        }
    )


def build_model_lock(
    *,
    config_hash: str,
    centers: Sequence[str],
    development: Any,
    fitted: Any,
    model_receipt: Any,
    development_receipt: Any,
    compatibility_hash: str,
    compatibility_receipt: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v6_model_lock_v1",
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "development_surface_hash": development.manifest["surface_hash"],
            "model_hash": fitted.manifest["model_hash"],
            "model_store_manifest_sha256": model_receipt.manifest_sha256,
            "model_store_npz_sha256": model_receipt.npz_sha256,
            "scientific_manifest": dict(fitted.manifest),
            "source_development_store_manifest_sha256": development_receipt.manifest_sha256,
            "compatibility_hash": compatibility_hash,
            "compatibility_store_manifest_sha256": compatibility_receipt.manifest_sha256,
            "evaluation_labels_used": False,
        },
        field="model_lock_hash",
    )


def build_learnability_seal(
    *,
    config_hash: str,
    fitted: Any,
    learnability: Any,
    admission_receipt: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v6_learnability_admission_seal_v1",
            "config_hash": config_hash,
            "model_hash": fitted.manifest["model_hash"],
            "admission_hash": learnability.manifest["admission_hash"],
            "router_admitted": learnability.manifest["router_admitted"],
            "admission_store_manifest_sha256": admission_receipt.manifest_sha256,
            "admission_store_npz_sha256": admission_receipt.npz_sha256,
            "evaluation_labels_used": False,
        }
    )


def build_target_action_seal(
    *,
    config: Any,
    centers: Sequence[str],
    fitted: Any,
    compatibility_hash: str,
    learnability: Any,
    target_actions: Any,
    target_receipt: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v6_target_action_seal_v1",
            "status": "COMPLETE_B_U_HXE_BEFORE_EVALUATION_LABELS",
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "model_hash": fitted.manifest["model_hash"],
            "compatibility_hash": compatibility_hash,
            "learnability_admission_hash": learnability.manifest["admission_hash"],
            "router_admitted": learnability.manifest["router_admitted"],
            "outer_menu_hashes": target_actions.manifest["outer_menu_hashes"],
            "target_store_manifest_sha256": target_receipt.manifest_sha256,
            "target_store_npz_sha256": target_receipt.npz_sha256,
            "target_action_hash": target_actions.manifest["target_action_hash"],
            "directional_soft_top_k": int(config.model["soft_top_k"]),
            "evaluation_labels_opened": False,
        }
    )


def build_prelabel_bundle(
    *,
    config_hash: str,
    centers: Sequence[str],
    development_seal: Mapping[str, object],
    model_lock: Mapping[str, object],
    learnability_seal: Mapping[str, object],
    target_seal: Mapping[str, object],
    learnability: Any,
    routes: Any,
    route_receipt: Any,
    route_summary: Mapping[str, object],
    rejection_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v6_prelabel_route_bundle_v1",
            "status": "DURABLE_CASE_ROUTES_BEFORE_EVALUATION_LABELS",
            "development_surface_seal_hash": development_seal["seal_hash"],
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "model_lock_hash": model_lock["model_lock_hash"],
            "learnability_admission_seal_hash": learnability_seal["seal_hash"],
            "router_admitted": learnability.manifest["router_admitted"],
            "target_action_seal_hash": target_seal["seal_hash"],
            "route_hash": routes.route_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "learnability_admission_hash": learnability.manifest["admission_hash"],
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
            "schema_version": "midogpp_harp_v6_fresh_validation_bundle_v1",
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
    routes: Any,
    learnability: Any,
    validation_bundle: Mapping[str, object],
    validations: Sequence[Mapping[str, object]],
    route_summary: Mapping[str, object],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v6_frozen_route_seal_v1",
            "status": "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS",
            "prelabel_bundle_hash": prelabel["bundle_hash"],
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "route_hash": routes.route_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "learnability_admission_hash": learnability.manifest["admission_hash"],
            "target_action_hash": routes.target_action_hash,
            "validation_bundle_hash": validation_bundle["bundle_hash"],
            "independent_validation_hashes": [
                value["validation_hash"] for value in validations
            ],
            "case_count": len(routes.cases),
            "exact_b_fallback_byte_identity": route_summary[
                "exact_b_fallback_byte_identity"
            ],
            "evaluation_labels_opened": False,
        }
    )


def build_run_state(
    *,
    root: Path,
    ledger: Any,
    lease: Any,
    finalization_path: Path,
    content_index_path: Path,
    terminal_paths: Sequence[Path],
    frozen: Mapping[str, object],
    sealed_routes: Any,
    terminal_metrics: Mapping[str, object],
    scratch: Path,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_harp_v6_run_state_v1",
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
        "final_commit": True,
    }


__all__ = (
    "build_development_surface_seal",
    "build_frozen_route_seal",
    "build_learnability_seal",
    "build_model_lock",
    "build_prelabel_bundle",
    "build_run_state",
    "build_target_action_seal",
    "build_validation_bundle",
    "seal",
)
