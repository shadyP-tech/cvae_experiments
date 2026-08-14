"""Content-first, exact scientific reconstruction of a terminal CDCA bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world, validate_content_index
from .config_payloads import (
    CLASSIFIER,
    canonical_action_library_payload,
    canonical_case_correctness_router_payload,
    canonical_claim_boundary_payload,
    canonical_controls_payload,
    canonical_evaluation_payload,
    canonical_protocol_payload,
    canonical_runtime_payload,
)
from .execution_adapter import load_validated_workstation_preflight
from .fresh_process_validation import verify_attested_validation_checks
from .hashing import canonical_hash
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .protocol import build_frozen_science_protocol
from .reports import protocol_manifest_payload
from .recovery_provenance import audit_for_validation
from .route_numerics import ROUTE_BLAS_THREADS
from .validation_prelabel import reconstruct_prelabel, validate_action_products
from .validation_science import (
    reconstruct_plan_and_feature_products,
    reconstruct_route_products,
    reconstruct_terminal_products,
)


def validate_fixed_bank_case_directional_correctness_abstention_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
    finalization_recovery_audit: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Rebuild every scientific product without writing or repairing evidence."""

    path = Path(root)
    protocol = build_frozen_science_protocol()

    # Content is admitted before any scientific member can influence replay.
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending_validation,
    )
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.protocol_hash,
    )
    finalization_audit = audit_for_validation(
        path,
        allow_pending_validation=allow_pending_validation,
        explicit_audit=finalization_recovery_audit,
    )
    _reject_forbidden_persistence(path)

    _validate_config_contract(config)
    assert_input_fence(config)
    workspace = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
    firewall["workspace_binding"] = workspace
    expected_protocol_manifest = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in getattr(config, "input_artifact_ids")
        },
        cache_binding_hash=str(frame.cache_binding_hash),
        firewall=firewall,
    )
    if (
        read_json(path / "manifests/protocol_manifest.json")
        != expected_protocol_manifest
    ):
        raise ProtocolError(
            "Case-directional protocol manifest is not reconstructive."
        )
    action_checks = validate_action_products(path)
    preflight = load_validated_workstation_preflight(
        path, runtime=getattr(config, "runtime")
    )
    prelabel = reconstruct_prelabel(
        path,
        config=config,
        frame=frame,
        generation_lock_hash=locks.generation.generation_lock_hash,
    )
    plan_products = reconstruct_plan_and_feature_products(
        path,
        frame=frame,
        probability_surface=prelabel["probability_surface"],
        probability_surface_hash=str(prelabel["probability_surface_hash"]),
        physical_prelabel_seal_hash=str(
            prelabel["physical_prelabel_seal_hash"]
        ),
    )
    route_products = reconstruct_route_products(
        path,
        probability_surface=prelabel["probability_surface"],
        plans=plan_products["plans"],
        science_plan_seal=plan_products["science_plan_seal"],
        features=plan_products["features"],
        persisted_plan_seal_hash=str(
            plan_products["persisted_plan_seal"]["seal_hash"]
        ),
        feature_seal_hash=str(plan_products["feature_seal"]["seal_hash"]),
        label_loader=lambda allowed: _read_scoped_manifest_labels(
            config, frame, allowed_keys=allowed
        ),
    )
    terminal_checks = reconstruct_terminal_products(
        path,
        probability_surface=prelabel["probability_surface"],
        method_predictions=route_products["method_predictions"],
        descriptive_predictions=route_products["descriptive_predictions"],
        decisions=route_products["decisions"],
        aggregate_seal=route_products["aggregate_seal"],
        firewall=route_products["firewall"],
        prediction=prelabel["prediction"],
        source=prelabel["source"],
        preflight=preflight,
        runtime=getattr(config, "runtime"),
        physical_prelabel_seal_hash=str(
            prelabel["physical_prelabel_seal_hash"]
        ),
        feature_seal_hash=str(plan_products["feature_seal"]["seal_hash"]),
    )
    checks = {
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.protocol_hash,
        "finalization_recovery": dict(finalization_audit),
        "workspace_binding": workspace,
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": firewall["status"],
        "workstation_preflight_status": preflight["status"],
        **dict(action_checks),
        "source_stream_count": len(prelabel["source"].records),
        "physical_probability_cell_count": len(
            prelabel["prediction"].store.cells
        ),
        "probability_index_count": prelabel["probability_index_count"],
        "probability_surface_hash": prelabel["probability_surface_hash"],
        "physical_prelabel_seal_hash": prelabel[
            "physical_prelabel_seal_hash"
        ],
        "held_case_plan_count": plan_products["held_case_plan_count"],
        "held_case_feature_count": plan_products["held_case_feature_count"],
        "held_case_plan_seal_hash": plan_products["persisted_plan_seal"][
            "seal_hash"
        ],
        "held_case_feature_seal_hash": plan_products["feature_seal"][
            "seal_hash"
        ],
        "support_response_count": route_products["support_response_count"],
        "donor_prior_count": route_products["donor_prior_count"],
        "route_model_fit_count": route_products["route_model_fit_count"],
        "route_candidate_score_count": route_products[
            "route_candidate_score_count"
        ],
        "route_decision_count": route_products["route_decision_count"],
        "preterminal_prediction_count": route_products[
            "preterminal_prediction_count"
        ],
        "descriptive_prediction_count": route_products[
            "descriptive_prediction_count"
        ],
        "donor_prior_seal_hash": route_products["prior_seal"]["seal_hash"],
        "route_model_seal_hash": route_products["model_seal"]["seal_hash"],
        "route_decision_seal_hash": route_products["decision_seal"][
            "seal_hash"
        ],
        "aggregate_plan_decision_seal_hash": route_products[
            "aggregate_seal"
        ]["seal_hash"],
        **dict(terminal_checks),
        "all_17_persisted_tables_reconstructed_exactly": True,
        "all_experiment_seals_reconstructed_exactly": True,
        "all_72_donor_grants_before_route_support": True,
        "every_route_fit_is_H_minus_c": True,
        "canonical_and_permuted_routes_reconstructed": True,
        "held_case_excluded_from_every_fit": True,
        "donor_H_and_e_excluded": True,
        "label_free_feature_schema_exact": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "predicted_held_case_exact_bacc_claimed": False,
        "content_index_validated_before_scientific_members": True,
        "two_fresh_cuda_free_process_replays_required": True,
        "route_fit_and_replay_blas_threads": ROUTE_BLAS_THREADS,
        "fitted_numeric_tolerance_used": False,
        "nonrepairing_validation": True,
        "closed_world": True,
        "terminal_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if allow_pending_validation:
        return checks

    return _validate_attested_report(path, checks)


def _validate_attested_report(
    path: Path, checks: Mapping[str, object]
) -> Mapping[str, object]:
    report = read_json(path / "reports/validation_report.json")
    if report.get("schema_version") != "fixed_bank_cdca_validation_report_v1":
        raise ProtocolError("Case-directional validation report header drifted.")
    persisted_checks = {
        key: value for key, value in report.items() if key != "schema_version"
    }
    attested = verify_attested_validation_checks(
        persisted_checks,
        expected_reconstructed_checks=checks,
    )
    if report != {
        "schema_version": "fixed_bank_cdca_validation_report_v1",
        **attested,
    }:
        raise ProtocolError(
            "Case-directional validation report is not reconstructive."
        )
    return attested


def _validate_config_contract(config: object) -> None:
    expected = {
        "protocol": canonical_protocol_payload(),
        "action_library": canonical_action_library_payload(),
        "case_correctness_router": canonical_case_correctness_router_payload(),
        "controls": canonical_controls_payload(),
        "evaluation": canonical_evaluation_payload(),
        "runtime": canonical_runtime_payload(),
        "claim_boundary": canonical_claim_boundary_payload(),
    }
    try:
        drifted = any(
            dict(getattr(config, key)) != value
            for key, value in expected.items()
        ) or getattr(config, "classifier") != CLASSIFIER
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "Case-directional persisted config contract is malformed."
        ) from exc
    if drifted:
        raise ProtocolError("Case-directional persisted config contract drifted.")


def _read_scoped_manifest_labels(
    config: object,
    frame: object,
    *,
    allowed_keys: frozenset[tuple[str, str, str]],
) -> Sequence[object]:
    """Read only the identities granted by the fresh validation firewall."""

    from .products import BinaryLabel

    universe = {(row.center, row.case_id, row.sample_id): row for row in frame.rows}
    if not allowed_keys or not set(allowed_keys) <= set(universe):
        raise ProtocolError("Case-directional validation label grant escaped.")
    ordered_keys = tuple(
        (row.center, row.case_id, row.sample_id)
        for row in frame.rows
        if (row.center, row.case_id, row.sample_id) in allowed_keys
    )
    if len(ordered_keys) != len(allowed_keys):
        raise ProtocolError(
            "Case-directional validation label grant order drifted."
        )
    requested = {key: universe[key] for key in ordered_keys}
    found: dict[tuple[str, str, str], object] = {}
    manifest = Path(getattr(config, "test_manifest_path"))
    manifest_hash = str(getattr(config, "expected_manifest_sha256"))
    try:
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            for ordinal, raw in enumerate(csv.DictReader(handle)):
                key = (
                    str(raw.get("center", "")),
                    str(raw.get("case_id", "")),
                    evaluation_row_id(manifest_hash, ordinal),
                )
                if key not in requested:
                    continue
                if requested[key].manifest_row_index != ordinal or key in found:
                    raise ProtocolError(
                        "Case-directional validation manifest order drifted."
                    )
                found[key] = BinaryLabel(
                    *key, int(raw["label"]), "validator_loader"
                )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(
            "Cannot load scoped case-directional validation labels."
        ) from exc
    if set(found) != set(requested):
        raise ProtocolError(
            "Case-directional validation label coverage drifted."
        )
    return tuple(found[key] for key in ordered_keys)


def _reject_forbidden_persistence(root: Path) -> None:
    forbidden = {
        "label",
        "labels",
        "ground_truth",
        "true_label",
        "image_path",
        "sample_path",
        "manifest_path",
    }
    excluded = {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/content_index.json",
    }
    for path in root.rglob("*.json"):
        if path.relative_to(root).as_posix() in excluded:
            continue
        value = _json(path)
        if _contains_key(value, forbidden):
            raise ProtocolError(
                "Case-directional persisted a forbidden raw label/path field."
            )
    for path in root.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                fields = csv.DictReader(handle).fieldnames
        except OSError as exc:
            raise ProtocolError(
                f"Cannot read case-directional table header: {path}."
            ) from exc
        if fields is None or forbidden & {field.casefold() for field in fields}:
            raise ProtocolError(
                "Case-directional persisted a forbidden raw CSV field."
            )


def _contains_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in forbidden
            or str(key).casefold().endswith("_path")
            or _contains_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(
            f"Case-directional JSON is unreadable: {path}."
        ) from exc


__all__ = (
    "validate_fixed_bank_case_directional_correctness_abstention_router_bundle",
)
