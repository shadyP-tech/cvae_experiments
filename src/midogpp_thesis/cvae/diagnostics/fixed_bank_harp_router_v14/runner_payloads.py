"""Pure manifest builders for the HARP v14 execution lifecycle.

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
    effective_menu: Any,
    effective_menu_receipt: Any,
    source_crossfit: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v14_development_surface_seal_v1",
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
            "effective_menu_hash": effective_menu.manifest["effective_menu_hash"],
            "effective_menu_store_manifest_sha256": (
                effective_menu_receipt.manifest_sha256
            ),
            "effective_menu_store_npz_sha256": effective_menu_receipt.npz_sha256,
            "source_crossfit_surface_hash": source_crossfit.physical_surface.surface_hash,
            "source_crossfit_surface_receipt_hash": (
                source_crossfit.surface_receipt.receipt_hash
            ),
            "source_crossfit_inventory_hash": (
                source_crossfit.surface_receipt.inventory_hash
            ),
            "source_crossfit_manifest_sha256": (
                source_crossfit.surface_receipt.manifest_sha256
            ),
            "source_crossfit_probabilities_sha256": (
                source_crossfit.surface_receipt.probabilities_sha256
            ),
            "source_crossfit_dispersion_sha256": (
                source_crossfit.surface_receipt.dispersion_sha256
            ),
            "source_crossfit_compatibility_sha256": (
                source_crossfit.surface_receipt.compatibility_sha256
            ),
            "source_crossfit_effective_adapter_hash": (
                source_crossfit.effective_surface.adapter_hash
            ),
            "shared_source_target_effective_menu_sealed": True,
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
    source_crossfit: Any,
    fold_seal_set: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v14_model_lock_v1",
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
            "source_crossfit_surface_hash": source_crossfit.physical_surface.surface_hash,
            "source_crossfit_surface_receipt_hash": (
                source_crossfit.surface_receipt.receipt_hash
            ),
            "source_crossfit_effective_adapter_hash": (
                source_crossfit.effective_surface.adapter_hash
            ),
            "source_prelabel_fold_seal_set_hash": fold_seal_set.seal_set_hash,
            "source_fold_menu_binding_certificate_hash": (
                fold_seal_set.fold_menu_binding_certificate_hash
            ),
            "source_fold_menu_binding_certificate_receipt_hash": (
                fold_seal_set.fold_menu_binding_certificate_receipt_hash
            ),
            "exact_fold_outcome_universe_set_hash": fitted.manifest[
                "exact_fold_outcome_universe_set_hash"
            ],
            "exact_fold_outcome_universe_hashes": fitted.manifest[
                "exact_fold_outcome_universe_hashes"
            ],
            "source_prelabel_fold_set_manifest_sha256": (
                fold_seal_set.manifest_sha256
            ),
            "legacy_fit_source_lodo_used": False,
            "presealed_fold_assembly_only": True,
            "evaluation_labels_used": False,
        },
        field="model_lock_hash",
    )


def build_policy_admission_seal(
    *,
    config_hash: str,
    fitted: Any,
    policy: Any,
    admission_receipt: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v14_source_policy_admission_seal_v1",
            "config_hash": config_hash,
            "model_hash": fitted.manifest["model_hash"],
            "admission_hash": policy.manifest["admission_hash"],
            "admitted_outer_count": policy.manifest["admitted_outer_count"],
            "outer_policy_count": len(policy.manifest["outer_policies"]),
            "per_outer_local_admission": True,
            "global_kill_switch_used": False,
            "whole_policy_oof_replayed": True,
            "exact_nested_outcome_universe_hashes": policy.manifest[
                "exact_nested_outcome_universe_hashes"
            ],
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
    policy: Any,
    target_actions: Any,
    target_receipt: Any,
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v14_target_action_seal_v1",
            "status": "COMPLETE_B_U_HXE_BEFORE_EVALUATION_LABELS",
            "config_hash": config.config_hash,
            "expected_center_ids": list(centers),
            "model_hash": fitted.manifest["model_hash"],
            "compatibility_hash": compatibility_hash,
            "source_policy_admission_hash": policy.manifest["admission_hash"],
            "admitted_outer_count": policy.manifest["admitted_outer_count"],
            "outer_menu_hashes": target_actions.manifest["outer_menu_hashes"],
            "target_store_manifest_sha256": target_receipt.manifest_sha256,
            "target_store_npz_sha256": target_receipt.npz_sha256,
            "target_action_hash": target_actions.manifest["target_action_hash"],
            "target_case_count": target_actions.manifest["target_case_count"],
            "ordered_case_identity_hash": target_actions.manifest[
                "ordered_case_identity_hash"
            ],
            "ordered_sample_identity_hash": target_actions.manifest[
                "ordered_sample_identity_hash"
            ],
            "exact_top1_physical_action_only": True,
            "unevaluated_action_mixtures_used": False,
            "evaluation_labels_opened": False,
        }
    )


def build_prelabel_bundle(
    *,
    config_hash: str,
    centers: Sequence[str],
    development_seal: Mapping[str, object],
    model_lock: Mapping[str, object],
    policy_admission_seal: Mapping[str, object],
    target_seal: Mapping[str, object],
    policy: Any,
    routes: Any,
    route_receipt: Any,
    route_summary: Mapping[str, object],
    rejection_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v14_prelabel_route_bundle_v1",
            "status": "DURABLE_CASE_ROUTES_BEFORE_EVALUATION_LABELS",
            "development_surface_seal_hash": development_seal["seal_hash"],
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "model_lock_hash": model_lock["model_lock_hash"],
            "source_policy_admission_seal_hash": policy_admission_seal["seal_hash"],
            "admitted_outer_count": policy.manifest["admitted_outer_count"],
            "target_action_seal_hash": target_seal["seal_hash"],
            "route_hash": routes.route_hash,
            "ordered_case_identity_hash": routes.ordered_case_identity_hash,
            "ordered_sample_identity_hash": routes.ordered_sample_identity_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "source_policy_admission_hash": policy.manifest["admission_hash"],
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
            "schema_version": "midogpp_harp_v14_fresh_validation_bundle_v1",
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
    policy: Any,
    validation_bundle: Mapping[str, object],
    validations: Sequence[Mapping[str, object]],
    route_summary: Mapping[str, object],
) -> dict[str, object]:
    return seal(
        {
            "schema_version": "midogpp_harp_v14_frozen_route_seal_v1",
            "status": "FROZEN_AFTER_TWO_FRESH_RECONSTRUCTIONS",
            "prelabel_bundle_hash": prelabel["bundle_hash"],
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "route_hash": routes.route_hash,
            "policy_hash": routes.policy_hash,
            "model_hash": routes.model_hash,
            "source_policy_admission_hash": policy.manifest["admission_hash"],
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
        "schema_version": "midogpp_harp_v14_run_state_v1",
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
    "build_policy_admission_seal",
    "build_model_lock",
    "build_prelabel_bundle",
    "build_run_state",
    "build_target_action_seal",
    "build_validation_bundle",
    "seal",
)
