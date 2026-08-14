"""Content-first, read-only reconstruction of the complete OGDE bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .bundle import assert_closed_world, validate_content_index
from .label_capabilities import DualEndpointLabelFirewall
from .protocol import build_frozen_science_protocol
from .runner_services import read_scoped_manifest_labels
from .split_plans import build_whole_case_loo_plans, seal_whole_case_loo_plans
from .terminal import evaluate_terminal
from .validation_endpoint import reconstruct_endpoint_products
from .validation_prelabel import (
    reconstruct_admission_and_prelabel,
    validate_path_free_json_members,
)
from .validation_route import reconstruct_route_products
from .validation_terminal import (
    validate_final_attestation,
    validate_terminal_products,
    validate_terminal_reports,
)


VALIDATION_SCHEMA = "fixed_bank_ogde_validation_v1"


def validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    """Rebuild every scientific surface without mutating persisted evidence."""

    path = Path(root)
    if path.resolve() != Path(getattr(config, "artifact_root")).resolve():
        raise ProtocolError("Dual-endpoint validation root/config binding drifted.")
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending_validation,
    )
    protocol = build_frozen_science_protocol()
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.protocol_hash,
    )
    # Only after the content seal is accepted may any scientific member be read.
    validate_path_free_json_members(path)
    admission = reconstruct_admission_and_prelabel(
        path, config=config, protocol=protocol
    )
    frame = admission["frame"]
    prelabel = admission["prelabel"]
    surface = prelabel["probability_surface"]
    plans = build_whole_case_loo_plans(
        frame.rows, probability_surface_hash=str(surface.surface_hash)
    )
    science_plan_seal = seal_whole_case_loo_plans(
        plans, probability_surface_hash=str(surface.surface_hash)
    )
    firewall = DualEndpointLabelFirewall(
        science_plan_seal,
        lambda allowed: read_scoped_manifest_labels(
            config, frame, allowed_keys=allowed
        ),
    )
    route = reconstruct_route_products(
        path,
        config=config,
        frame=frame,
        surface=surface,
        physical_prelabel_seal_hash=str(prelabel["physical_prelabel_seal_hash"]),
        label_firewall=firewall,
    )
    endpoint = reconstruct_endpoint_products(
        path,
        surface=surface,
        route_products=route,
        label_firewall=firewall,
    )
    terminal = evaluate_terminal(
        surface=surface,
        plans=route["plans"],
        directional_support_gains=route["directional_support_gains"],
        identification_decisions=route["identification_decisions"],
        robust_arm_decisions=route["robust_arm_decisions"],
        method_predictions=endpoint["method_predictions"],
        terminal_labels=firewall.open_terminal_labels(),
        aggregate_seal_hash=str(endpoint["seals"]["aggregate"]["seal_hash"]),
        config=config,
    )
    terminal_checks = validate_terminal_products(path, reconstructed=terminal)
    capability = firewall.report_payload()
    validate_terminal_reports(
        path,
        config=config,
        preflight=admission["preflight"],
        prelabel=prelabel,
        feature_seal=route["feature_seal"],
        aggregate_seal=endpoint["seals"]["aggregate"],
        capability_report=capability,
        terminal=terminal,
        allow_pending_validation=allow_pending_validation,
    )
    checks = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.protocol_hash,
        "input_artifact_count": len(admission["provenance"]),
        "workspace_binding": admission["workspace"],
        "pre_gpu_firewall_status": admission["pre_gpu_firewall"]["status"],
        "workstation_preflight_status": admission["preflight"]["status"],
        "physical_cell_count": len(prelabel["prediction"].store.cells),
        "exact_nine_probability_index_count": prelabel["probability_index_count"],
        "probability_surface_hash": prelabel["probability_surface_hash"],
        "loo_plan_count": len(route["plans"]),
        "identification_decision_count": len(route["identification_decisions"]),
        "robust_arm_decision_count": len(route["robust_arm_decisions"]),
        "preterminal_prediction_count": len(endpoint["method_predictions"]),
        **dict(terminal_checks),
        "all_science_reconstructed_exactly": True,
        "content_index_validated_before_scientific_members": True,
        "fitted_route_blas_threads": 3,
        "two_fresh_cuda_free_process_replays_required": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if allow_pending_validation:
        return checks
    return validate_final_attestation(path, checks=checks)


__all__ = ("VALIDATION_SCHEMA", "validate_fixed_bank_loo_opportunity_gated_dual_endpoint_router_bundle")
