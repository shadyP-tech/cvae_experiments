"""Content-first, closed-world validation for the terminal diagnostic."""

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
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .protocol import canonical_consumed_test_protocol
from .reports import (
    leakage_report_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .validation_science import validate_scientific_surfaces


def validate_fixed_bank_labeled_support_case_conditional_flip_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    """Reconstruct bytes and science without trusting generated reports."""

    path = Path(root)
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=allow_pending_validation,
    )
    protocol = canonical_consumed_test_protocol()
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    _reject_raw_label_persistence(path)

    # Re-resolve every external byte after the content index has passed.
    assert_input_fence(config)
    workspace = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = validate_pre_gpu_firewall(config, frame, locks)
    firewall = {**dict(firewall), "workspace_binding": workspace}
    expected_protocol_manifest = protocol_manifest_payload(
        config,
        protocol=protocol,
        input_artifact_hashes={
            artifact_id: canonical_hash(provenance[artifact_id])
            for artifact_id in getattr(config, "input_artifact_ids")
        },
        cache_binding_hash=frame.cache_binding_hash,
        firewall=firewall,
    )
    if read_json(path / "manifests/protocol_manifest.json") != expected_protocol_manifest:
        raise ProtocolError("Flip-router protocol manifest is not reconstructive.")
    preflight = load_validated_workstation_preflight(
        path, runtime=getattr(config, "runtime")
    )
    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=locks.generation.generation_lock_hash,
    )
    science = validate_scientific_surfaces(path, config=config, frame=frame)
    _validate_reports(
        path,
        science=science,
        preflight=preflight,
        source=source,
        allow_pending_validation=allow_pending_validation,
    )
    checks = {
        "schema_version": "fixed_bank_labeled_support_flip_validation_v1",
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "source_stream_lock_hash": source.lock_hash,
        "workspace_binding": workspace,
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": firewall["status"],
        "workstation_preflight_status": preflight["status"],
        **dict(science),
        "content_index_validated_before_scientific_members": True,
        "scientific_factories_replayed": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted_or_used": False,
        "terminal_consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if not allow_pending_validation:
        report = read_json(path / "reports/validation_report.json")
        if report != checks:
            raise ProtocolError("Flip-router validation report is not reconstructive.")
    return checks


def assert_completed_bundle_binding(
    root: str | Path,
    *,
    config: object,
    expected_checks: Mapping[str, object],
) -> None:
    """Bind COMPLETE to the one already-completed reconstructive validation.

    This is deliberately not a second scientific validator.  The caller must
    pass the in-memory result returned by the mandatory pending full replay;
    this check then proves that the closed-world bytes, persisted report, and
    terminal run state still describe that exact replay result.
    """

    path = Path(root)
    assert_closed_world(path, allow_incomplete=False)
    protocol = canonical_consumed_test_protocol()
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    checks = dict(expected_checks)
    if (
        checks.get("schema_version")
        != "fixed_bank_labeled_support_flip_validation_v1"
        or checks.get("status") != "PASS"
        or checks.get("scientific_factories_replayed") is not True
        or checks.get("nonrepairing_validation") is not True
        or checks.get("closed_world") is not True
        or checks.get("content_hash") != content.get("content_hash")
        or checks.get("config_contract_hash") != str(getattr(config, "contract_hash"))
        or checks.get("protocol_contract_hash") != protocol.contract_hash
        or read_json(path / "reports/validation_report.json") != checks
        or read_json(path / "reports/run_state.json")
        != run_state_payload("COMPLETE", "COMPLETE")
    ):
        raise ProtocolError(
            "Flip-router COMPLETE state is not bound to its full validation."
        )


def _validate_reports(
    root: Path,
    *,
    science: Mapping[str, object],
    preflight: Mapping[str, object],
    source: object,
    allow_pending_validation: bool,
) -> None:
    capability = read_json(root / "reports/label_capability_report.json")
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    runtime = read_json(root / "reports/runtime_summary.json")
    terminal = read_json(root / "manifests/sealed_terminal_evaluation.json")
    decision = read_json(root / "manifests/all_method_decisions_seal.json")
    feature = read_json(root / "manifests/prelabel_feature_seal.json")
    prediction = read_json(root / "manifests/fixed_bank_a1_prediction_seal.json")
    run_state = read_json(root / "reports/run_state.json")
    expected_leakage = leakage_report_payload(
        prediction_seal_hash=str(prediction["global_prediction_seal_hash"]),
        feature_seal_hash=str(feature["feature_surface_hash"]),
        capability_report=capability,
        donor_model_seal_count=9,
        fold_decision_seal_count=45,
    )
    sealed_hash = terminal.get("sealed_result_hash")
    if not isinstance(sealed_hash, str) or len(sealed_hash) != 64:
        raise ProtocolError("Flip-router terminal result hash is absent.")
    expected_publication = publication_decision_payload(sealed_hash)
    local = runtime.get("local_source_staging")
    if not isinstance(local, Mapping):
        raise ProtocolError("Flip-router runtime staging report is absent.")
    # Runtime is rebuilt from the persisted prediction seal by the science
    # validator; compare all stable topology fields here, including preflight.
    if (
        leakage != expected_leakage
        or publication != expected_publication
        or runtime.get("status") != "PASS"
        or runtime.get("source_stream_lock_hash") != getattr(source, "lock_hash")
        or runtime.get("global_prediction_seal_hash")
        != prediction.get("global_prediction_seal_hash")
        or runtime.get("workstation_preflight") != dict(preflight)
        or runtime.get("classifier_cell_count") != 810
        or runtime.get("unique_classifier_fit_count") != 810
        or runtime.get("scratch_root")
        != "/data/local/fixed_bank_labeled_support_case_conditional_flip_router_v1"
        or runtime.get("previous_stage90_output_prediction_or_scratch_reused") is not False
        or decision.get("decision_count") != science.get("method_decision_count")
        or terminal.get("decision_bundle_hash") != decision.get("decision_bundle_hash")
        or terminal.get("terminal_scoring_after_all_45_decision_seals") is not True
        or terminal.get("raw_labels_persisted") is not False
        or run_state
        != (
            run_state_payload("RUNNING", "FINALIZATION")
            if allow_pending_validation
            else run_state_payload("COMPLETE", "COMPLETE")
        )
    ):
        raise ProtocolError("Flip-router terminal reports drifted.")


def _reject_raw_label_persistence(root: Path) -> None:
    forbidden = {"label", "labels", "ground_truth", "true_label", "image_path", "sample_path"}
    for path in root.rglob("*.json"):
        if path.name == "input_artifacts.json":
            continue
        value = _json(path)
        if _contains_forbidden_key(value, forbidden):
            raise ProtocolError(f"Flip-router persisted a forbidden raw field: {path}.")
    for path in root.rglob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames
        if fields is None or forbidden & set(fields):
            raise ProtocolError(f"Flip-router persisted a forbidden raw CSV field: {path}.")


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
        raise ProtocolError(f"Flip-router JSON is unreadable: {path}.") from exc


__all__ = (
    "assert_completed_bundle_binding",
    "validate_fixed_bank_labeled_support_case_conditional_flip_router_bundle",
)
