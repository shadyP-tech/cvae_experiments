"""Artifact-only, no-refit reconstruction of SCALE-BP v2 bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .authorization_lease import validate_persisted_authorization_lease
from .artifacts.content import (
    validate_final_content_index,
    validate_preterminal_bundle as validate_preterminal_artifacts,
)
from .artifacts.hashing import canonical_hash
from .artifacts.io import member_path, read_json_object
from .identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    P_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .protocol import GovernanceError
from .run_state import PHASE_ORDER, read_run_state
from .terminal import validate_persisted_terminal_aggregate


def validate_preterminal_bundle(
    root: str | Path,
    *,
    expected_decision_seal_hash: str | None = None,
    no_refit: bool = True,
) -> dict[str, object]:
    """Rehash the frozen preterminal surface without scientific fitting."""

    if no_refit is not True:
        raise GovernanceError("SCALE-BP v2 validation is artifact-only and no-refit.")
    path = Path(root).resolve()
    checks = validate_preterminal_artifacts(
        path,
        allow_post_preterminal_members=True,
        expected_decision_seal_hash=expected_decision_seal_hash,
    )
    state = read_run_state(path)
    if PHASE_ORDER.index(str(state["phase"])) < PHASE_ORDER.index("PRETERMINAL_SEALED"):
        raise GovernanceError("SCALE-BP v2 preterminal state is not durably sealed.")
    lease = validate_persisted_authorization_lease(
        path,
        expected_claim_hash=str(state["authorization_lease_claim_hash"]),
    )
    protocol_manifest = read_json_object(
        member_path(path, "manifests/protocol_manifest.json")
    )
    if (
        lease.get("run_identity_hash") != state.get("run_identity_hash")
        or lease.get("config_contract_hash") != state.get("config_hash")
        or lease.get("protocol_hash") != state.get("protocol_hash")
        or protocol_manifest.get("authorization_lease_claim_hash")
        != state.get("authorization_lease_claim_hash")
    ):
        raise GovernanceError("SCALE-BP v2 lease/run provenance binding drifted.")
    payload = {
        "schema_version": "scale_bp_v2_preterminal_validation_checks_v1",
        "status": "PASS",
        **checks,
        "run_authorization_consumed": True,
        "run_authorization_exhausted": True,
        "authorization_lease_claim_hash": state[
            "authorization_lease_claim_hash"
        ],
        "authorization_lease_external_and_durable": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "complete_route_count": EXPECTED_CASE_COUNT,
        "complete_center_count": len(CENTERS),
        "raw_labels_persisted": False,
        "artifact_only_reconstruction": True,
        "scientific_refit_performed": False,
    }
    return {**payload, "validation_hash": canonical_hash(payload)}


def validate_final_bundle(
    root: str | Path,
    *,
    no_refit: bool = True,
    require_fresh_attestation: bool = False,
) -> dict[str, object]:
    """Validate final aggregate science and claim boundaries from bytes only."""

    if no_refit is not True:
        raise GovernanceError("SCALE-BP v2 final validation cannot refit science.")
    path = Path(root).resolve()
    content = validate_final_content_index(path)
    terminal = validate_persisted_terminal_aggregate(
        path, expected_decision_seal_hash=str(content["decision_seal_hash"])
    )
    if (
        terminal.get("terminal_seal_hash") != content.get("terminal_seal_hash")
        or terminal.get("terminal_metrics_hash") != content.get("terminal_metrics_hash")
    ):
        raise GovernanceError("SCALE-BP v2 final index/terminal metric binding drifted.")
    methods = terminal.get("methods")
    if (
        not isinstance(methods, list)
        or not methods
        or methods[0].get("method_id") != P_METHOD_ID
        or any(
            not isinstance(row, Mapping)
            or row.get("row_count") != EXPECTED_TEST_ROW_COUNT
            or not isinstance(row.get("centers"), list)
            or tuple(item.get("target_center") for item in row["centers"])
            != CENTERS
            or sum(int(item.get("row_count", -1)) for item in row["centers"])
            != EXPECTED_TEST_ROW_COUNT
            or row.get("raw_labels_persisted") is not False
            for row in methods
        )
    ):
        raise GovernanceError("SCALE-BP v2 final terminal topology drifted.")
    state = read_run_state(path)
    if PHASE_ORDER.index(str(state["phase"])) < PHASE_ORDER.index("FINAL_INDEX_SEALED"):
        raise GovernanceError("SCALE-BP v2 final run state is premature.")
    lease = validate_persisted_authorization_lease(
        path,
        expected_claim_hash=str(state["authorization_lease_claim_hash"]),
    )
    protocol_manifest = read_json_object(
        member_path(path, "manifests/protocol_manifest.json")
    )
    if (
        lease.get("run_identity_hash") != state.get("run_identity_hash")
        or lease.get("config_contract_hash") != state.get("config_hash")
        or lease.get("protocol_hash") != state.get("protocol_hash")
        or protocol_manifest.get("authorization_lease_claim_hash")
        != state.get("authorization_lease_claim_hash")
    ):
        raise GovernanceError("SCALE-BP v2 final lease/run binding drifted.")
    publication = _optional_json(path, "reports/publication_decision.json")
    if publication is not None and (
        publication.get("status") != PUBLICATION_STATUS
        or publication.get("terminal_decision") != TERMINAL_DECISION
        or publication.get("fresh_evidence") is not False
        or publication.get("promotion_eligible") is not False
        or publication.get("may_feed_another_experiment") is not False
    ):
        raise GovernanceError("SCALE-BP v2 publication firewall drifted.")
    checks = {
        "schema_version": "scale_bp_v2_final_validation_checks_v1",
        "status": "PASS",
        **content,
        "method_count": len(methods),
        "row_count_per_method": EXPECTED_TEST_ROW_COUNT,
        "center_count_per_method": len(CENTERS),
        "protected_p_first": True,
        "raw_labels_persisted": False,
        "artifact_only_reconstruction": True,
        "scientific_refit_performed": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "authorization_lease_claim_hash": state[
            "authorization_lease_claim_hash"
        ],
        "authorization_lease_external_and_durable": True,
    }
    if require_fresh_attestation:
        from .fresh_process_validation import validate_fresh_process_attestation

        reconstructed = {**checks, "validation_hash": canonical_hash(checks)}
        attestation = validate_fresh_process_attestation(
            path, phase="final", expected_checks=reconstructed
        )
        checks["fresh_process_attestation_hash"] = attestation["attestation_hash"]
    return {**checks, "validation_hash": canonical_hash(checks)}


def _optional_json(root: Path, member: str) -> dict[str, object] | None:
    path = member_path(root, member)
    if not path.exists():
        return None
    return read_json_object(path)


__all__ = ("validate_final_bundle", "validate_preterminal_bundle")
