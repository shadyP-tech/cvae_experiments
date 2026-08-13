"""Content-first, non-repairing validation of the closed-world S4 bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world, validate_content_index
from .execution_adapter import (
    load_frozen_source_streams,
    load_validated_workstation_preflight,
)
from .hashing import canonical_hash
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_pre_gpu_firewall,
)
from .protocol import canonical_consumed_test_protocol
from .reports import (
    leakage_report_payload,
    protocol_manifest_payload,
    publication_decision_payload,
)
from .validation_science import validate_scientific_surfaces
from .workspace_inputs import (
    validate_active_diagnostic_workspace_binding,
    validate_workspace_provenance,
)


def validate_fixed_bank_support_static_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
    skip_fresh_process_report: bool = False,
) -> Mapping[str, object]:
    """Reconstruct exact categorical science without trusting generated reports."""

    path = Path(root)
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending_validation,
    )
    protocol = canonical_consumed_test_protocol()
    # The content index is always the first member opened after inventory.
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    _reject_raw_label_persistence(path)

    assert_input_fence(config)
    workspace = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
    firewall["workspace_binding"] = workspace
    expected_protocol = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in getattr(config, "input_artifact_ids")
        },
        cache_binding_hash=frame.cache_binding_hash,
        firewall=firewall,
    )
    if read_json(path / "manifests/protocol_manifest.json") != expected_protocol:
        raise ProtocolError("S4 protocol manifest is not reconstructive.")

    preflight = load_validated_workstation_preflight(
        path, runtime=getattr(config, "runtime")
    )
    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=locks.generation.generation_lock_hash,
    )
    science = dict(
        validate_scientific_surfaces(path, config=config, frame=frame)
    )
    _validate_reports(
        path,
        science=science,
        preflight=preflight,
        source=source,
    )
    checks = {
        "schema_version": "fixed_bank_support_static_router_validation_v1",
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "source_stream_lock_hash": source.lock_hash,
        "workspace_binding": workspace,
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": firewall["status"],
        "workstation_preflight_status": preflight["status"],
        **{key: value for key, value in science.items() if key != "label_capability_report"},
        "content_index_validated_before_scientific_members": True,
        "scientific_factories_replayed": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted_or_used": False,
        "own_evaluation_noninterference_enforced": True,
        "other_fold_support_label_reuse_allowed": True,
        "action_identity_null_descriptive_only": True,
        "exchangeability_claimed": False,
        "confirmatory_p_value": False,
        "null_summary_in_pass_gate": False,
        "two_fresh_process_replays_required": True,
        "terminal_consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if not skip_fresh_process_report:
        _validate_fresh_process_report(path, checks)
    validation_path = path / "reports/validation_report.json"
    if validation_path.is_file() and read_json(validation_path) != checks:
        raise ProtocolError("S4 validation report is not reconstructive.")
    if not validation_path.is_file() and not allow_pending_validation:
        raise ProtocolError("S4 validation report is absent.")
    _validate_run_state(path)
    return checks


def assert_completed_bundle_binding(
    root: str | Path,
    *,
    config: object,
    expected_checks: Mapping[str, object],
) -> None:
    path = Path(root)
    observed = validate_fixed_bank_support_static_router_bundle(path, config=config)
    if dict(observed) != dict(expected_checks):
        raise ProtocolError("S4 COMPLETE state differs from its reconstructive replay.")
    state = read_json(path / "reports/run_state.json")
    if state.get("status") != "COMPLETE" or state.get("phase") != "COMPLETE":
        raise ProtocolError("S4 bundle is validated but not durably COMPLETE.")


def _validate_reports(
    root: Path,
    *,
    science: Mapping[str, object],
    preflight: Mapping[str, object],
    source: object,
) -> None:
    capability = science["label_capability_report"]
    null_seal = read_json(root / "manifests/action_identity_null_seal.json")
    expected_leakage = leakage_report_payload(
        prediction_seal_hash=str(science["global_prediction_seal_hash"]),
        probability_surface_hash=str(science["probability_surface_hash"]),
        capability_report=capability,
        global_static_seal_hash=str(science["global_static_seal_hash"]),
        decision_seal_hash=str(science["route_decision_seal_hash"]),
        null_seal_hash=str(null_seal["null_seal_hash"]),
    )
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    expected_publication = publication_decision_payload(
        str(science["sealed_result_hash"])
    )
    runtime = read_json(root / "reports/runtime_summary.json")
    null_summary = read_json(root / "reports/action_identity_null_summary.json")
    exact_capability = {
        "fold_plan_count": 45,
        "g_static_candidate_donor_grant_count": 72,
        "g_static_selection_seal_count": 9,
        "support_grant_count": 45,
        "route_decision_seal_count": 45,
        "null_selection_seal_count": 45,
        "route_evaluation_grant_count": 45,
        "pre_evaluation_aggregate_decision_seal_count": 1,
        "pre_evaluation_aggregate_null_plan_seal_count": 1,
    }
    if (
        leakage != expected_leakage
        or publication != expected_publication
        or any(capability.get(key) != value for key, value in exact_capability.items())
        or runtime.get("status") != "PASS"
        or runtime.get("source_stream_lock_hash") != getattr(source, "lock_hash")
        or runtime.get("global_prediction_seal_hash")
        != science["global_prediction_seal_hash"]
        or runtime.get("workstation_preflight") != dict(preflight)
        or runtime.get("classifier_cell_count") != 810
        or runtime.get("unique_classifier_fit_count") != 810
        or runtime.get("persistent_a5000_gpu_worker_count") != 2
        or runtime.get("cpu_classifier_worker_count") != 4
        or runtime.get("blas_threads_per_classifier_worker") != 3
        or runtime.get("donor_model_fit_count") != 0
        or runtime.get("target_calibration_fit_count") != 0
        or runtime.get("replayed_phase_checkpoints_hash_validated") is not True
        or runtime.get("terminal_checkpoint_recovery_supported") is not False
        or runtime.get("terminal_checkpoint_is_atomicity_boundary_only") is not True
        or runtime.get("previous_stage90_output_prediction_or_scratch_reused")
        is not False
        or null_summary.get("exchangeability_claimed") is not False
        or null_summary.get("confirmatory_p_value") is not False
        or null_summary.get("pass_gate_used") is not False
        or null_summary.get("null_replicate_count") != 10_000
        or null_summary.get("null_seal_hash") != null_seal.get("null_seal_hash")
    ):
        raise ProtocolError("S4 terminal reports drifted.")


def _validate_fresh_process_report(
    root: Path, checks: Mapping[str, object]
) -> None:
    path = root / "reports/fresh_process_validation.json"
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("S4 two-process validation report is absent or unsafe.")
    payload = read_json(path)
    expected_hash = canonical_hash(dict(checks))
    if (
        payload.get("status") != "PASS"
        or payload.get("replay_count") != 2
        or payload.get("replay_hashes") != [expected_hash, expected_hash]
        or payload.get("byte_identical_replay_results") is not True
        or payload.get("independent_processes") is not True
        or payload.get("cuda_visible_devices") != ""
        or payload.get("blas_threads_per_fresh_process") != 1
        or payload.get("python_hash_seed") != 0
        or payload.get("validation_result") != dict(checks)
        or payload.get("fresh_evidence") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ProtocolError("S4 two fresh-process replays drifted.")


def _validate_run_state(root: Path) -> None:
    payload = read_json(root / "reports/run_state.json")
    if (
        payload.get("schema_version")
        != "fixed_bank_support_static_router_run_state_v1"
        or payload.get("status") not in {"RUNNING", "COMPLETE"}
        or payload.get("phase") not in {"FINALIZATION", "COMPLETE"}
        or payload.get("terminal_consumed_test_diagnostic_only") is not True
        or payload.get("automatic_resume_supported") is not False
        or payload.get(
            "deterministic_restart_from_admission_requires_hash_validation"
        )
        is not True
        or payload.get("terminal_checkpoint_recovery_supported") is not False
        or payload.get("terminal_checkpoint_is_atomicity_boundary_only") is not True
    ):
        raise ProtocolError("S4 terminal run state drifted.")


def _reject_raw_label_persistence(root: Path) -> None:
    forbidden = {
        "label",
        "labels",
        "ground_truth",
        "true_label",
        "image_path",
        "sample_path",
    }
    for path in root.rglob("*.json"):
        if path.name == "input_artifacts.json":
            continue
        value = _json(path)
        if _contains_forbidden_key(value, forbidden):
            raise ProtocolError(f"S4 persisted a forbidden raw field: {path}.")
    for path in root.rglob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames
        if fields is None or forbidden & set(fields):
            raise ProtocolError(f"S4 persisted a forbidden raw CSV field: {path}.")


def _contains_forbidden_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in forbidden or _contains_forbidden_key(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"S4 JSON is unreadable: {path}.") from exc


__all__ = (
    "assert_completed_bundle_binding",
    "validate_fixed_bank_support_static_router_bundle",
)
