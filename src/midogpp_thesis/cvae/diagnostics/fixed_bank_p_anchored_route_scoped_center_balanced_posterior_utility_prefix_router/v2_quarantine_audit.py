"""Read-only phase, lineage, journal, and scratch audit for v2 quarantine."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path
import stat
from typing import Iterator

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .bundle import (
    PRETERMINAL_ATTESTED_FILES,
    PRETERMINAL_SCIENTIFIC_MEMBERS,
    validate_preterminal_content_index,
)
from .config import (
    load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config,
)
from .config_payloads import canonical_claim_boundary_payload
from .constants import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    EXPECTED_OUTER_PLAN_COUNT,
    EXPECTED_PSEUDO_ROUTE_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    SCRATCH_ROOT,
    STAGE_ID,
)
from .hashing import canonical_hash
from .preterminal_gate import validate_preterminal_gate_artifacts
from .protocol import FROZEN_PROTOCOL_HASH, frozen_protocol_payload
from .terminal_access_journal import (
    TERMINAL_ACCESS_INTENT_MEMBER,
    TERMINAL_ACCESS_OPENED_MEMBER,
    validate_terminal_label_access_intent,
    validate_terminal_label_access_journal,
)
from .v2_quarantine_contracts import (
    AUDIT_SCHEMA,
    CAPABILITY_KEYS,
    ELIGIBLE_NEXT_ACTION,
    EXPECTED_PRETERMINAL_CAPABILITY_EVENT_COUNT,
    PROTOCOL_MANIFEST_BASE_KEYS,
    RUN_STATE_KEYS,
    SCRATCH_TO_ARTIFACT_SOURCE_MEMBERS,
    SHA256_PATTERN,
    V2_FINAL_PERSISTENCE_ORDER,
    V2_FINAL_PHASE,
    V2_TERMINAL_FAILURE_ARTIFACT_DIRECTORIES,
    V2_TERMINAL_FAILURE_SCRATCH_DIRECTORIES,
    V2_TERMINAL_FAILURE_SCRATCH_FILES,
    V2_TERMINAL_PERSISTENCE_ORDER,
    V2_TERMINAL_PHASE,
)


def audit_failed_v2_terminal_or_final_for_quarantine(
    root: Path,
) -> Mapping[str, object]:
    logical_root = logical_source_root(root)
    logical_scratch = v2_scratch_root()
    if not logical_root.is_dir():
        raise ProtocolError("CBPUPR v2 terminal/final root is absent or unsafe.")
    observed_scratch = logical_scratch if is_present(logical_scratch) else None
    with exclusive_existing_v2_terminal_run_lock(logical_root):
        return audit_locked_failed_v2_terminal_or_final(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=logical_scratch,
            observed_scratch=observed_scratch,
        )


def audit_locked_failed_v2_terminal_or_final(
    *,
    logical_root: Path,
    observed_root: Path,
    logical_scratch: Path,
    observed_scratch: Path | None,
) -> Mapping[str, object]:
    state = _validate_failed_state(observed_root)
    phase = str(state["phase"])
    terminal_prefix, final_prefix = _validate_artifact_inventory(
        observed_root, phase=phase
    )
    config = (
        load_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_config(
            observed_root / "config.resolved.yaml"
        )
    )
    if (
        str(config.experiment_id) != EXPERIMENT_ID
        or str(config.output_artifact_id) != OUTPUT_ARTIFACT_ID
        or Path(config.artifact_root).resolve() != logical_root
        or dict(config.protocol) != frozen_protocol_payload()
        or dict(config.claim_boundary) != canonical_claim_boundary_payload()
    ):
        raise ProtocolError("CBPUPR v2 terminal/final config identity drifted.")
    _validate_no_reuse_claim_boundary(config.claim_boundary)
    protocol = _validate_protocol_manifest(observed_root, config=config)
    content = validate_preterminal_content_index(observed_root)
    checks = _validate_durable_preterminal_gate(
        observed_root, config=config, protocol=protocol, content=content
    )
    _validate_preterminal_capability(observed_root)
    terminal_access_status = _validate_terminal_journal_prefix(
        observed_root, terminal_prefix=terminal_prefix, expected_checks=checks
    )

    allow_cleanup_state = (
        phase == V2_FINAL_PHASE and final_prefix == V2_FINAL_PERSISTENCE_ORDER
    )
    scratch_state, scratch_directories, scratch_members = _audit_scratch(
        observed_scratch,
        artifact_root=observed_root,
        allow_cleanup_state=allow_cleanup_state,
    )
    artifact_members = [
        member_payload(observed_root, relative)
        for relative in (
            ".run.lock",
            *sorted(
                path.relative_to(observed_root).as_posix()
                for path in observed_root.rglob("*")
                if path.is_file() and path.name != ".run.lock"
            ),
        )
    ]
    firewall = protocol["pre_gpu_firewall"]
    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS",
        "source_root": str(logical_root),
        "source_scratch_root": str(logical_scratch),
        "source_experiment_id": EXPERIMENT_ID,
        "source_output_artifact_id": OUTPUT_ARTIFACT_ID,
        "source_run_status": "FAILED",
        "source_run_phase": phase,
        "source_error_class": state["error_class"],
        "source_error": state["error"],
        "publication_status": PUBLICATION_STATUS,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "config_contract_hash": config.contract_hash,
        "protocol_contract_hash": FROZEN_PROTOCOL_HASH,
        "protocol_manifest_hash": protocol["protocol_manifest_hash"],
        "preterminal_content_index_hash": content["content_index_hash"],
        "preterminal_validation_checks_hash": checks["validation_checks_hash"],
        "preterminal_hash": checks["preterminal_hash"],
        "repair_source_manifest_sha256": firewall[
            "repair_source_manifest_sha256"
        ],
        "repair_source_tree_sha256": firewall["repair_source_tree_sha256"],
        "repair_source_member_count": firewall["repair_source_member_count"],
        "test_manifest_sha256": config.expected_manifest_sha256,
        "test_consumption_ledger_sha256": (
            config.expected_test_consumption_ledger_sha256
        ),
        "ledger_amendment_sha256": config.expected_ledger_amendment_sha256,
        "durable_preterminal_gate_revalidated": True,
        "terminal_persistence_prefix": list(terminal_prefix),
        "terminal_persistence_prefix_length": len(terminal_prefix),
        "final_persistence_prefix": list(final_prefix),
        "final_persistence_prefix_length": len(final_prefix),
        "terminal_access_journal_status": terminal_access_status,
        "terminal_capability_report_persisted": (
            "reports/label_capability_report.json" in terminal_prefix
        ),
        "scratch_state": scratch_state,
        "scratch_directories": scratch_directories,
        "scratch_members": scratch_members,
        "artifact_members": artifact_members,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "v2_rerun_authorized": False,
        "v1_output_scratch_or_capability_history_may_be_used": False,
        "quarantined_v2_results_may_be_promoted": False,
        "eligible_next_action": ELIGIBLE_NEXT_ACTION,
    }
    return {**payload, "quarantine_audit_hash": canonical_hash(payload)}


def _validate_failed_state(root: Path) -> Mapping[str, object]:
    state = read_json(root / "reports/run_state.json")
    try:
        timestamp = datetime.fromisoformat(str(state.get("updated_at_utc")))
    except ValueError as exc:
        raise ProtocolError(
            "CBPUPR v2 terminal/final failed-state timestamp drifted."
        ) from exc
    if (
        set(state) != RUN_STATE_KEYS
        or state.get("schema_version") != "fixed_bank_cbpupr_run_state_v1"
        or state.get("status") != "FAILED"
        or state.get("phase") not in {V2_TERMINAL_PHASE, V2_FINAL_PHASE}
        or not isinstance(state.get("error"), str)
        or not isinstance(state.get("error_class"), str)
        or not state.get("error_class")
        or timestamp.tzinfo is None
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR v2 terminal/final failed state drifted.")
    return state


def _validate_artifact_inventory(
    root: Path, *, phase: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    assert_regular_tree(root, role="v2 terminal/final artifact")
    members = tuple(root.rglob("*"))
    directories = frozenset(
        path.relative_to(root).as_posix() for path in members if path.is_dir()
    )
    files = frozenset(
        path.relative_to(root).as_posix() for path in members if path.is_file()
    )
    if (
        directories != V2_TERMINAL_FAILURE_ARTIFACT_DIRECTORIES
        or ".run.lock" not in files
    ):
        raise ProtocolError(
            "CBPUPR v2 terminal/final artifact directory inventory drifted."
        )
    observed = files - {".run.lock"}
    baseline = frozenset(PRETERMINAL_ATTESTED_FILES)
    if phase == V2_TERMINAL_PHASE:
        for length in range(len(V2_TERMINAL_PERSISTENCE_ORDER) + 1):
            prefix = V2_TERMINAL_PERSISTENCE_ORDER[:length]
            if observed == baseline | frozenset(prefix):
                return prefix, ()
    if phase == V2_FINAL_PHASE:
        terminal = frozenset(V2_TERMINAL_PERSISTENCE_ORDER)
        for length in range(len(V2_FINAL_PERSISTENCE_ORDER) + 1):
            prefix = V2_FINAL_PERSISTENCE_ORDER[:length]
            if observed == baseline | terminal | frozenset(prefix):
                return V2_TERMINAL_PERSISTENCE_ORDER, prefix
    raise ProtocolError(
        "CBPUPR v2 terminal/final phase-aware artifact inventory drifted."
    )


def _validate_protocol_manifest(root: Path, *, config: object) -> Mapping[str, object]:
    protocol = read_json(root / "manifests/protocol_manifest.json")
    unhashed = {
        key: value for key, value in protocol.items() if key != "protocol_manifest_hash"
    }
    input_hashes = protocol.get("input_artifact_hashes")
    firewall = protocol.get("pre_gpu_firewall")
    config_protocol = dict(config.protocol)
    if (
        set(protocol) != PROTOCOL_MANIFEST_BASE_KEYS
        or protocol.get("schema_version")
        != "fixed_bank_cbpupr_protocol_manifest_v1"
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or protocol.get("config_contract_hash") != config.contract_hash
        or protocol.get("protocol_contract_hash") != FROZEN_PROTOCOL_HASH
        or protocol.get("stage") != STAGE_ID
        or protocol.get("claim_scope") != CLAIM_SCOPE
        or protocol.get("claim_role") != CLAIM_ROLE
        or not isinstance(input_hashes, Mapping)
        or set(input_hashes) != set(config.input_artifact_ids)
        or any(not is_sha256(value) for value in input_hashes.values())
        or not is_sha256(protocol.get("cache_binding_hash"))
        or not isinstance(firewall, Mapping)
        or firewall.get("repair_source_manifest_validated") is not True
        or firewall.get("repair_source_manifest_sha256")
        != config_protocol.get("repair_source_manifest_sha256")
        or firewall.get("repair_source_tree_sha256")
        != config_protocol.get("repair_source_tree_sha256")
        or firewall.get("repair_source_member_count")
        != config_protocol.get("repair_source_member_count")
        or protocol.get("exact_six_original_inputs") is not True
        or protocol.get("previous_stage90_output_or_checkpoint_used") is not False
        or protocol.get("test_split_previously_consumed") is not True
        or protocol.get("fresh_evidence") is not False
        or protocol.get("publication_status") != PUBLICATION_STATUS
        or protocol.get("protocol_manifest_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR v2 terminal/final protocol identity drifted.")
    return protocol


def _validate_no_reuse_claim_boundary(boundary: Mapping[str, object]) -> None:
    false_fields = (
        "fresh_evidence",
        "quarantined_v1_output_used",
        "quarantined_v1_scratch_or_checkpoint_used",
        "quarantined_v1_terminal_outputs_used",
        "prior_v1_label_capability_history_used",
        "prior_v1_amendment_used",
        "routing_success_claimed",
        "routing_quality_claimed",
        "downstream_utility_claimed",
        "expert_selection_claimed",
        "deployment_claimed",
        "promotion_eligible",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
        "previous_stage90_outputs_used",
        "previous_stage90_amendments_used",
        "previous_probability_surface_used",
        "previous_stage90_scratch_or_checkpoint_used",
    )
    if (
        boundary.get("schema_version") != "fixed_bank_cbpupr_claim_boundary_v2"
        or boundary.get("publication_status") != PUBLICATION_STATUS
        or boundary.get("claim_role") != CLAIM_ROLE
        or boundary.get("consumed_test_data") is not True
        or boundary.get("terminal_stage90_diagnostic") is not True
        or any(boundary.get(field) is not False for field in false_fields)
    ):
        raise ProtocolError("CBPUPR v2 terminal/final claim boundary drifted.")


def _validate_durable_preterminal_gate(
    root: Path,
    *,
    config: object,
    protocol: Mapping[str, object],
    content: Mapping[str, object],
) -> Mapping[str, object]:
    report = read_json(root / "reports/preterminal_validation_report.json")
    checks = report.get("checks")
    expected = {
        "config_contract_hash": config.contract_hash,
        "protocol_contract_hash": FROZEN_PROTOCOL_HASH,
        "content_index_hash": content.get("content_index_hash"),
        "preterminal_scientific_member_count": len(PRETERMINAL_SCIENTIFIC_MEMBERS),
        "outer_route_count": EXPECTED_OUTER_PLAN_COUNT,
        "target_posterior_model_fit_count": EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
        "pseudo_posterior_model_fit_count": 0,
        "pseudo_posterior_reference_count": 2 * EXPECTED_PSEUDO_ROUTE_COUNT,
        "terminal_opened": False,
        "terminal_product_count": 0,
        "terminal_only_consumed_test": True,
        "formal_claim_authorized": False,
    }
    if (
        not isinstance(checks, Mapping)
        or any(checks.get(key) != value for key, value in expected.items())
        or not is_sha256(checks.get("preterminal_hash"))
        or protocol.get("config_contract_hash") != checks.get("config_contract_hash")
        or protocol.get("protocol_contract_hash")
        != checks.get("protocol_contract_hash")
    ):
        raise ProtocolError("CBPUPR v2 durable preterminal checks drifted.")
    validate_preterminal_gate_artifacts(root, expected_checks=checks)
    return checks


def _validate_preterminal_capability(root: Path) -> None:
    capability = read_json(root / "reports/preterminal_label_capability_report.json")
    events = capability.get("events")
    seal = read_json(root / "manifests/outer_plan_seal.json")
    if (
        set(capability) != CAPABILITY_KEYS
        or capability.get("schema_version")
        != "fixed_bank_cbpupr_label_access_audit_v1"
        or not isinstance(events, list)
        or len(events) != EXPECTED_PRETERMINAL_CAPABILITY_EVENT_COUNT
        or capability.get("event_count") != len(events)
        or capability.get("audit_hash") != canonical_hash(events)
        or any(
            not isinstance(event, Mapping)
            or event.get("raw_labels_persisted") is not False
            or event.get("role") == "target_terminal_after_aggregate_seal"
            for event in events
        )
        or capability.get("plan_seal_hash") != seal.get("seal_hash")
        or capability.get("target_candidate_seal_complete") is not True
        or capability.get("pre_evaluation_seal_complete") is not True
        or capability.get("pseudo_evaluation_route_count")
        != EXPECTED_PSEUDO_ROUTE_COUNT
        or capability.get("calibration_seal_complete") is not True
        or capability.get("decision_count") != 4 * EXPECTED_OUTER_PLAN_COUNT
        or capability.get("aggregate_seal_complete") is not True
        or capability.get("terminal_opened") is not False
        or capability.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("CBPUPR v2 durable preterminal capability state drifted.")


def _validate_terminal_journal_prefix(
    root: Path,
    *,
    terminal_prefix: tuple[str, ...],
    expected_checks: Mapping[str, object],
) -> str:
    if TERMINAL_ACCESS_INTENT_MEMBER not in terminal_prefix:
        return "NOT_OPENED"
    intent = read_json(root / TERMINAL_ACCESS_INTENT_MEMBER)
    validate_terminal_label_access_intent(intent)
    if (
        intent.get("preterminal_content_index_hash")
        != expected_checks.get("content_index_hash")
        or intent.get("preterminal_validation_checks_hash")
        != expected_checks.get("validation_checks_hash")
        or intent.get("preterminal_hash") != expected_checks.get("preterminal_hash")
    ):
        raise ProtocolError("CBPUPR terminal access/preterminal binding drifted.")
    if TERMINAL_ACCESS_OPENED_MEMBER not in terminal_prefix:
        return "UNKNOWN_CONSERVATIVELY_CONSUMED"
    validate_terminal_label_access_journal(root, expected_checks=expected_checks)
    return "OPENED"


def _audit_scratch(
    observed_scratch: Path | None,
    *,
    artifact_root: Path,
    allow_cleanup_state: bool,
) -> tuple[str, list[str], list[dict[str, object]]]:
    if observed_scratch is None:
        if not allow_cleanup_state:
            raise ProtocolError(
                "CBPUPR v2 terminal/final scratch is absent before cleanup edge."
            )
        return "ABSENT_AFTER_FINAL_REPORT", [], []
    assert_regular_tree(observed_scratch, role="v2 terminal/final scratch")
    members = tuple(observed_scratch.rglob("*"))
    directories = frozenset(
        path.relative_to(observed_scratch).as_posix()
        for path in members
        if path.is_dir()
    )
    files = frozenset(
        path.relative_to(observed_scratch).as_posix()
        for path in members
        if path.is_file()
    )
    exact = (
        directories == V2_TERMINAL_FAILURE_SCRATCH_DIRECTORIES
        and files == V2_TERMINAL_FAILURE_SCRATCH_FILES
    )
    cleanup_subset = (
        allow_cleanup_state
        and directories <= V2_TERMINAL_FAILURE_SCRATCH_DIRECTORIES
        and files <= V2_TERMINAL_FAILURE_SCRATCH_FILES
    )
    if not exact and not cleanup_subset:
        raise ProtocolError("CBPUPR v2 terminal/final scratch inventory drifted.")
    for scratch_relative, artifact_relative in SCRATCH_TO_ARTIFACT_SOURCE_MEMBERS.items():
        if scratch_relative in files and (
            (observed_scratch / scratch_relative).stat().st_size
            != (artifact_root / artifact_relative).stat().st_size
            or sha256_file(observed_scratch / scratch_relative)
            != sha256_file(artifact_root / artifact_relative)
        ):
            raise ProtocolError(
                "CBPUPR v2 terminal/final scratch/source artifact bytes differ."
            )
    state = "FULL" if exact else "PARTIAL_AFTER_FINAL_REPORT_CLEANUP"
    return (
        state,
        sorted(directories),
        [member_payload(observed_scratch, relative) for relative in sorted(files)],
    )


def logical_source_root(root: Path) -> Path:
    unresolved = Path(root)
    if unresolved.is_symlink():
        raise ProtocolError("CBPUPR v2 terminal/final root is a symlink.")
    parent = unresolved.absolute().parent
    if parent.is_symlink() or not parent.is_dir():
        raise ProtocolError("CBPUPR v2 terminal/final root parent is unsafe.")
    return parent.resolve() / unresolved.name


def v2_scratch_root() -> Path:
    scratch = Path(SCRATCH_ROOT)
    if not scratch.is_absolute() or str(scratch) != SCRATCH_ROOT:
        raise ProtocolError("CBPUPR v2 terminal/final scratch root drifted.")
    if scratch.is_symlink():
        raise ProtocolError("CBPUPR v2 terminal/final scratch root is a symlink.")
    if scratch.parent.is_symlink() or not scratch.parent.is_dir():
        raise ProtocolError("CBPUPR v2 terminal/final scratch parent is unsafe.")
    return scratch.parent.resolve() / scratch.name


def assert_regular_tree(root: Path, *, role: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError(f"CBPUPR {role} root is absent or unsafe.")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ProtocolError(f"CBPUPR {role} tree contains an unsafe member.")


def member_payload(root: Path, relative: object) -> dict[str, object]:
    path = root / str(relative)
    return {
        "path": str(relative),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


@contextmanager
def exclusive_existing_v2_terminal_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("CBPUPR v2 terminal/final run lock is absent or unsafe.")
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ProtocolError("CBPUPR v2 terminal/final run lock is not regular.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError as exc:
            raise ProtocolError(
                "CBPUPR v2 terminal diagnostic is active; quarantine is forbidden."
            ) from exc
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = (
    "assert_regular_tree",
    "audit_failed_v2_terminal_or_final_for_quarantine",
    "audit_locked_failed_v2_terminal_or_final",
    "exclusive_existing_v2_terminal_run_lock",
    "is_present",
    "logical_source_root",
    "member_payload",
    "v2_scratch_root",
)
