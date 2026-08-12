"""Closed-world, content-first validation for the multi-challenger diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world, validate_content_index
from .constants import SCRATCH_ROOT
from .execution_adapter import (
    load_frozen_source_streams,
    load_validated_workstation_preflight,
)
from .fresh_process_validation import (
    ATTESTATION_KEY,
    verify_attested_validation_checks,
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
from .recovery_provenance import (
    current_repair_repository_state,
    original_repository_state_from_provenance,
    sealed_recovery_input_hashes,
    validate_recovery_audit_payload,
)
from .reports import (
    leakage_report_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .validation_science import validate_scientific_surfaces


VALIDATION_SCHEMA = "fixed_bank_multi_challenger_validation_v1"


def validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    """Replay external admission and all science without repairing the bundle."""

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
        raise ProtocolError("Multi-challenger protocol manifest is not reconstructive.")
    preflight = load_validated_workstation_preflight(
        path, runtime=getattr(config, "runtime")
    )
    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=locks.generation.generation_lock_hash,
    )
    science = validate_scientific_surfaces(path, config=config, frame=frame)
    recovery_audit = _validate_reports(
        path,
        science=science,
        preflight=preflight,
        source=source,
        allow_pending_validation=allow_pending_validation,
    )
    checks = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "source_stream_lock_hash": source.lock_hash,
        "workspace_binding": workspace,
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": firewall["status"],
        "workstation_preflight_status": preflight["status"],
        "mappingproxy_recovery_used": recovery_audit["recovery_used"],
        **dict(science),
        "content_index_validated_before_scientific_members": True,
        "scientific_factories_replayed": True,
        "derived_fitted_numerics_self_fingerprinted_and_tolerance_aware": True,
        "fitted_numeric_tolerance_is_explicit_path_allowlist": True,
        "unallowlisted_numeric_fields_compared_exactly": True,
        "provenance_topology_menus_ranks_actions_reasons_confusions_exact": True,
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
        attested = verify_attested_validation_checks(
            report,
            expected_reconstructed_checks=checks,
        )
        if report != attested:
            raise ProtocolError(
                "Multi-challenger validation report is not reconstructive."
            )
        return attested
    return checks


def assert_completed_bundle_binding(
    root: str | Path,
    *,
    config: object,
    expected_checks: Mapping[str, object],
) -> None:
    path = Path(root)
    assert_closed_world(path, allow_incomplete=False)
    protocol = canonical_consumed_test_protocol()
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    checks = dict(verify_attested_validation_checks(expected_checks))
    if (
        checks.get("schema_version") != VALIDATION_SCHEMA
        or checks.get("status") != "PASS"
        or checks.get("scientific_factories_replayed") is not True
        or checks.get("fit_fingerprint_semantics_validated") is not True
        or checks.get("nonrepairing_validation") is not True
        or checks.get("closed_world") is not True
        or not isinstance(checks.get(ATTESTATION_KEY), Mapping)
        or checks[ATTESTATION_KEY].get("status") != "PASS"
        or checks[ATTESTATION_KEY].get("fresh_python_process_count") != 2
        or checks.get("content_hash") != content.get("content_hash")
        or checks.get("config_contract_hash")
        != str(getattr(config, "contract_hash"))
        or checks.get("protocol_contract_hash") != protocol.contract_hash
        or read_json(path / "reports/validation_report.json") != checks
        or read_json(path / "reports/run_state.json")
        != run_state_payload("COMPLETE", "COMPLETE")
    ):
        raise ProtocolError(
            "Multi-challenger COMPLETE state is not bound to full validation."
        )


def _validate_reports(
    root: Path,
    *,
    science: Mapping[str, object],
    preflight: Mapping[str, object],
    source: object,
    allow_pending_validation: bool,
) -> Mapping[str, object]:
    capability = read_json(root / "reports/label_capability_report.json")
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    runtime = read_json(root / "reports/runtime_summary.json")
    terminal = read_json(root / "manifests/sealed_terminal_evaluation.json")
    decision = read_json(root / "manifests/all_method_decisions_seal.json")
    feature = read_json(root / "manifests/prelabel_feature_seal.json")
    prediction = read_json(root / "manifests/fixed_bank_a1_prediction_seal.json")
    run_state = read_json(root / "reports/run_state.json")
    recovery_audit = _validate_recovery_lineage(root, runtime=runtime)
    expected_leakage = leakage_report_payload(
        prediction_seal_hash=str(prediction["global_prediction_seal_hash"]),
        feature_seal_hash=str(feature["feature_surface_hash"]),
        capability_report=capability,
    )
    sealed_hash = terminal.get("sealed_result_hash")
    gate = terminal.get("diagnostic_routing_gate")
    if (
        not isinstance(sealed_hash, str)
        or len(sealed_hash) != 64
        or not isinstance(gate, Mapping)
    ):
        raise ProtocolError("Multi-challenger terminal seal is incomplete.")
    expected_publication = publication_decision_payload(
        sealed_hash, diagnostic_gate=gate
    )
    local = runtime.get("local_source_staging")
    if not isinstance(local, Mapping):
        raise ProtocolError("Multi-challenger runtime staging report is absent.")
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
        or runtime.get("scratch_root") != SCRATCH_ROOT
        or runtime.get("previous_stage90_output_prediction_or_scratch_reused")
        is not False
        or runtime.get("recomputed_from_original_six_inputs") is not True
        or decision.get("decision_count") != science.get("method_decision_count")
        or terminal.get("decision_bundle_hash")
        != decision.get("decision_bundle_hash")
        or terminal.get("terminal_scoring_after_all_45_decision_seals") is not True
        or terminal.get("terminal_oracles_used_for_decisions") is not False
        or terminal.get("raw_labels_persisted") is not False
        or run_state
        != (
            run_state_payload("RUNNING", "FINALIZATION")
            if allow_pending_validation
            else run_state_payload("COMPLETE", "COMPLETE")
        )
    ):
        raise ProtocolError("Multi-challenger terminal reports drifted.")
    return recovery_audit


def _validate_recovery_lineage(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    """Bind the persisted audit to live files and both repository identities."""

    return validate_recovery_audit_payload(
        runtime.get("mappingproxy_recovery"),
        original_repository_state=original_repository_state_from_provenance(root),
        current_repository_state=current_repair_repository_state(),
        **sealed_recovery_input_hashes(root),
    )


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
            raise ProtocolError(
                f"Multi-challenger persisted a forbidden raw field: {path}."
            )
    for path in root.rglob("*.csv"):
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames
        if fields is None or forbidden & set(fields):
            raise ProtocolError(
                f"Multi-challenger persisted a forbidden raw CSV field: {path}."
            )


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
        raise ProtocolError(f"Multi-challenger JSON is unreadable: {path}.") from exc


__all__ = (
    "VALIDATION_SCHEMA",
    "assert_completed_bundle_binding",
    "validate_fixed_bank_multi_challenger_hierarchical_flip_router_bundle",
)
