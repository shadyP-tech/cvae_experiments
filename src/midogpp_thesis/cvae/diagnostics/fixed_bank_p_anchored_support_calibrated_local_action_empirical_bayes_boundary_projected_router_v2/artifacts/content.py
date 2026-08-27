"""Closed-world preterminal and final content indexes and aggregate seals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import os

from ..identity import CENTERS, EXPECTED_CASE_COUNT
from ..protocol import GovernanceError
from .chunks import CenterManifestRef, validate_center_manifest
from .hashing import canonical_hash, require_sha256
from .io import atomic_json, indexed_file_row, member_path, read_json_object, safe_member
from .journal import (
    persist_label_capability_journal,
    validate_persisted_label_capability_journal,
)


PRETERMINAL_INDEX_MEMBER = "manifests/preterminal_content_index.json"
PRETERMINAL_SEAL_MEMBER = "manifests/preterminal_aggregate_seal.json"
FINAL_INDEX_MEMBER = "manifests/final_content_index.json"
FINAL_SEAL_MEMBER = "manifests/final_aggregate_seal.json"
MUTABLE_LIFECYCLE_MEMBERS = (
    ".run.lock",
    ".state.lock",
    "reports/run_state.json",
)
PRETERMINAL_POST_SEAL_MEMBERS = (
    "reports/preterminal_fresh_process_attestation.json",
    "reports/label_capability_journal_final.json",
    "reports/terminal_metrics.json",
    "reports/runtime.json",
    "reports/leakage.json",
    "reports/publication_decision.json",
    "reports/validation.json",
    "reports/final_fresh_process_attestation.json",
    FINAL_INDEX_MEMBER,
    FINAL_SEAL_MEMBER,
)
FINAL_POST_INDEX_MEMBERS = (
    "reports/final_fresh_process_attestation.json",
    "reports/validation.json",
)


def persist_preterminal_bundle(
    root: str | Path,
    *,
    decision_seal_hash: str,
    route_count: int,
    center_manifests: Sequence[CenterManifestRef],
    label_capability_journal: Mapping[str, object],
    required_members: Sequence[str] = (),
) -> dict[str, object]:
    """Freeze every current scientific byte before terminal labels can open."""

    path = Path(root)
    seal = require_sha256(decision_seal_hash, "preterminal decision seal")
    manifests = _validate_center_inventory(path, center_manifests)
    if type(route_count) is not int or route_count != EXPECTED_CASE_COUNT:
        raise GovernanceError("SCALE-BP v2 preterminal route count drifted.")
    journal = persist_label_capability_journal(
        path, label_capability_journal, phase="preterminal"
    )
    if journal.get("decision_seal_hash") != seal:
        raise GovernanceError("SCALE-BP v2 journal/decision seal binding drifted.")
    for member in required_members:
        indexed_file_row(path, safe_member(member))
    excluded = {
        *MUTABLE_LIFECYCLE_MEMBERS,
        PRETERMINAL_INDEX_MEMBER,
        PRETERMINAL_SEAL_MEMBER,
    }
    members = _current_members(path, excluded=excluded)
    index = _content_index_payload(
        path,
        members,
        phase="preterminal",
        mutable_exclusions=MUTABLE_LIFECYCLE_MEMBERS,
        post_index_allowlist=PRETERMINAL_POST_SEAL_MEMBERS,
    )
    atomic_json(member_path(path, PRETERMINAL_INDEX_MEMBER), index)
    body = {
        "schema_version": "scale_bp_v2_preterminal_aggregate_seal_v1",
        "decision_seal_hash": seal,
        "route_count": route_count,
        "content_index_hash": index["content_index_hash"],
        "label_capability_journal_hash": journal["audit_hash"],
        "center_manifest_hashes": {
            row.target_center: row.manifest_hash for row in manifests
        },
        "center_result_hashes": {
            row.target_center: row.result_hash for row in manifests
        },
        "all_case_actions_and_emitted_probabilities_durably_sealed": True,
        "terminal_labels_opened": False,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
    }
    aggregate = {**body, "aggregate_seal_hash": canonical_hash(body)}
    atomic_json(member_path(path, PRETERMINAL_SEAL_MEMBER), aggregate)
    validate_preterminal_bundle(path, allow_post_preterminal_members=False)
    return aggregate


def validate_preterminal_bundle(
    root: str | Path,
    *,
    allow_post_preterminal_members: bool = True,
    expected_decision_seal_hash: str | None = None,
) -> dict[str, object]:
    path = Path(root)
    index = _validate_content_index(
        path, PRETERMINAL_INDEX_MEMBER, phase="preterminal"
    )
    aggregate = read_json_object(member_path(path, PRETERMINAL_SEAL_MEMBER))
    body = {key: value for key, value in aggregate.items() if key != "aggregate_seal_hash"}
    journal = validate_persisted_label_capability_journal(path, phase="preterminal")
    center_hashes = aggregate.get("center_manifest_hashes")
    center_result_hashes = aggregate.get("center_result_hashes")
    if (
        aggregate.get("schema_version") != "scale_bp_v2_preterminal_aggregate_seal_v1"
        or aggregate.get("route_count") != EXPECTED_CASE_COUNT
        or aggregate.get("content_index_hash") != index.get("content_index_hash")
        or aggregate.get("label_capability_journal_hash") != journal.get("audit_hash")
        or aggregate.get("decision_seal_hash") != journal.get("decision_seal_hash")
        or aggregate.get("aggregate_seal_hash") != canonical_hash(body)
        or aggregate.get("all_case_actions_and_emitted_probabilities_durably_sealed") is not True
        or aggregate.get("terminal_labels_opened") is not False
        or aggregate.get("raw_labels_persisted") is not False
        or aggregate.get("fresh_evidence") is not False
        or not isinstance(center_hashes, Mapping)
        or tuple(center_hashes) != CENTERS
        or not isinstance(center_result_hashes, Mapping)
        or tuple(center_result_hashes) != CENTERS
        or (
            expected_decision_seal_hash is not None
            and aggregate.get("decision_seal_hash")
            != require_sha256(expected_decision_seal_hash, "expected decision seal")
        )
    ):
        raise GovernanceError("SCALE-BP v2 preterminal aggregate seal drifted.")
    for center in CENTERS:
        member = f"manifests/centers/center_{center}.json"
        manifest = read_json_object(member_path(path, member))
        indexed = indexed_file_row(path, member)
        reference = CenterManifestRef(
            target_center=center,
            member=member,
            size_bytes=int(indexed["size_bytes"]),
            sha256=str(indexed["sha256"]),
            task_hash=str(manifest.get("task_hash")),
            result_hash=str(manifest.get("result_hash")),
            manifest_hash=str(manifest.get("manifest_hash")),
        )
        validated_manifest = validate_center_manifest(path, reference)
        if (
            validated_manifest.get("outer_result_persisted") is not True
            or center_hashes.get(center) != reference.manifest_hash
            or center_result_hashes.get(center) != reference.result_hash
        ):
            raise GovernanceError("SCALE-BP v2 aggregate/center manifest binding drifted.")
    _assert_closed_world(
        path,
        indexed={str(row["member"]) for row in index["members"]},  # type: ignore[index]
        structural={PRETERMINAL_INDEX_MEMBER, PRETERMINAL_SEAL_MEMBER},
        allowed_post=(PRETERMINAL_POST_SEAL_MEMBERS if allow_post_preterminal_members else ()),
    )
    return {
        "status": "PASS",
        "phase": "preterminal",
        "content_index_hash": index["content_index_hash"],
        "aggregate_seal_hash": aggregate["aggregate_seal_hash"],
        "decision_seal_hash": aggregate["decision_seal_hash"],
        "label_capability_journal_hash": journal["audit_hash"],
        "artifact_only_reconstruction": True,
        "scientific_refit_performed": False,
    }


def persist_final_content_index(
    root: str | Path,
    *,
    terminal_seal_hash: str,
    terminal_metrics_hash: str,
    label_capability_journal: Mapping[str, object],
    required_members: Sequence[str] = (),
) -> dict[str, object]:
    """Freeze all immutable terminal products; mutable lifecycle files are named."""

    path = Path(root)
    terminal_seal = require_sha256(terminal_seal_hash, "terminal aggregate seal")
    metrics_hash = require_sha256(terminal_metrics_hash, "terminal metrics hash")
    preterminal = validate_preterminal_bundle(
        path, allow_post_preterminal_members=True
    )
    terminal = read_json_object(member_path(path, "reports/terminal_metrics.json"))
    if (
        terminal.get("terminal_seal_hash") != terminal_seal
        or terminal.get("terminal_metrics_hash") != metrics_hash
        or terminal.get("decision_seal_hash") != preterminal.get("decision_seal_hash")
        or terminal.get("raw_labels_persisted") is not False
        or terminal.get("row_level_labels_persisted") is not False
    ):
        raise GovernanceError("SCALE-BP v2 terminal/final index binding drifted.")
    journal = persist_label_capability_journal(path, label_capability_journal, phase="final")
    if journal.get("decision_seal_hash") != preterminal.get("decision_seal_hash"):
        raise GovernanceError("SCALE-BP v2 final journal changed the decision seal.")
    for member in required_members:
        indexed_file_row(path, safe_member(member))
    excluded = {
        *MUTABLE_LIFECYCLE_MEMBERS,
        FINAL_INDEX_MEMBER,
        FINAL_SEAL_MEMBER,
        *FINAL_POST_INDEX_MEMBERS,
    }
    members = _current_members(path, excluded=excluded)
    index = _content_index_payload(
        path,
        members,
        phase="final",
        mutable_exclusions=MUTABLE_LIFECYCLE_MEMBERS,
        post_index_allowlist=FINAL_POST_INDEX_MEMBERS,
    )
    atomic_json(member_path(path, FINAL_INDEX_MEMBER), index)
    body = {
        "schema_version": "scale_bp_v2_final_aggregate_seal_v1",
        "preterminal_aggregate_seal_hash": preterminal["aggregate_seal_hash"],
        "decision_seal_hash": preterminal["decision_seal_hash"],
        "terminal_seal_hash": terminal_seal,
        "terminal_metrics_hash": metrics_hash,
        "final_label_capability_journal_hash": journal["audit_hash"],
        "content_index_hash": index["content_index_hash"],
        "raw_labels_persisted": False,
        "fresh_evidence": False,
        "terminal_diagnostic_only": True,
    }
    aggregate = {**body, "aggregate_seal_hash": canonical_hash(body)}
    atomic_json(member_path(path, FINAL_SEAL_MEMBER), aggregate)
    return aggregate


def validate_final_content_index(root: str | Path) -> dict[str, object]:
    path = Path(root)
    preterminal = validate_preterminal_bundle(
        path, allow_post_preterminal_members=True
    )
    index = _validate_content_index(path, FINAL_INDEX_MEMBER, phase="final")
    aggregate = read_json_object(member_path(path, FINAL_SEAL_MEMBER))
    journal = validate_persisted_label_capability_journal(path, phase="final")
    body = {key: value for key, value in aggregate.items() if key != "aggregate_seal_hash"}
    if (
        aggregate.get("schema_version") != "scale_bp_v2_final_aggregate_seal_v1"
        or aggregate.get("preterminal_aggregate_seal_hash")
        != preterminal.get("aggregate_seal_hash")
        or aggregate.get("decision_seal_hash") != preterminal.get("decision_seal_hash")
        or aggregate.get("final_label_capability_journal_hash") != journal.get("audit_hash")
        or aggregate.get("content_index_hash") != index.get("content_index_hash")
        or aggregate.get("aggregate_seal_hash") != canonical_hash(body)
        or aggregate.get("raw_labels_persisted") is not False
        or aggregate.get("fresh_evidence") is not False
        or aggregate.get("terminal_diagnostic_only") is not True
    ):
        raise GovernanceError("SCALE-BP v2 final aggregate seal drifted.")
    for role in ("terminal_seal_hash", "terminal_metrics_hash"):
        require_sha256(aggregate.get(role), role)
    _assert_closed_world(
        path,
        indexed={str(row["member"]) for row in index["members"]},  # type: ignore[index]
        structural={FINAL_INDEX_MEMBER, FINAL_SEAL_MEMBER},
        allowed_post=FINAL_POST_INDEX_MEMBERS,
    )
    return {
        "status": "PASS",
        "phase": "final",
        "preterminal_aggregate_seal_hash": preterminal["aggregate_seal_hash"],
        "decision_seal_hash": aggregate["decision_seal_hash"],
        "final_content_index_hash": index["content_index_hash"],
        "final_aggregate_seal_hash": aggregate["aggregate_seal_hash"],
        "terminal_seal_hash": aggregate["terminal_seal_hash"],
        "terminal_metrics_hash": aggregate["terminal_metrics_hash"],
        "final_label_capability_journal_hash": journal["audit_hash"],
        "artifact_only_reconstruction": True,
        "scientific_refit_performed": False,
    }


def _validate_center_inventory(
    root: Path, rows: Sequence[CenterManifestRef]
) -> tuple[CenterManifestRef, ...]:
    manifests = tuple(rows)
    if tuple(row.target_center for row in manifests) != CENTERS:
        raise GovernanceError("SCALE-BP v2 center-manifest inventory drifted.")
    for row in manifests:
        payload = validate_center_manifest(root, row)
        if payload.get("outer_result_persisted") is not True:
            raise GovernanceError(
                "SCALE-BP v2 preterminal center manifest lacks its result payload."
            )
    return manifests


def _content_index_payload(
    root: Path,
    members: Sequence[str],
    *,
    phase: str,
    mutable_exclusions: Sequence[str],
    post_index_allowlist: Sequence[str],
) -> dict[str, object]:
    rows = [indexed_file_row(root, member) for member in members]
    body = {
        "schema_version": f"scale_bp_v2_{phase}_content_index_v1",
        "phase": phase,
        "members": rows,
        "member_count": len(rows),
        "mutable_lifecycle_exclusions": list(mutable_exclusions),
        "post_index_member_allowlist": list(post_index_allowlist),
        "complete_closed_world_inventory": True,
        "raw_labels_persisted": False,
    }
    return {**body, "content_index_hash": canonical_hash(body)}


def _validate_content_index(
    root: Path, member: str, *, phase: str
) -> dict[str, object]:
    payload = read_json_object(member_path(root, member))
    rows = payload.get("members")
    body = {key: value for key, value in payload.items() if key != "content_index_hash"}
    if (
        payload.get("schema_version") != f"scale_bp_v2_{phase}_content_index_v1"
        or payload.get("phase") != phase
        or not isinstance(rows, list)
        or payload.get("member_count") != len(rows)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != sorted(row.get("member") for row in rows if isinstance(row, Mapping))
        or payload.get("complete_closed_world_inventory") is not True
        or payload.get("raw_labels_persisted") is not False
        or payload.get("content_index_hash") != canonical_hash(body)
    ):
        raise GovernanceError("SCALE-BP v2 content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"member", "size_bytes", "sha256"}:
            raise GovernanceError("SCALE-BP v2 content-index row is malformed.")
        observed = indexed_file_row(root, str(row["member"]))
        if observed != dict(row):
            raise GovernanceError("SCALE-BP v2 indexed member bytes drifted.")
    return payload


def _current_members(root: Path, *, excluded: set[str]) -> tuple[str, ...]:
    if root.is_symlink() or not root.is_dir():
        raise GovernanceError("SCALE-BP v2 bundle root is absent or unsafe.")
    rows: list[str] = []
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            if (base / name).is_symlink():
                raise GovernanceError("SCALE-BP v2 bundle contains a symlink.")
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            if relative.endswith(".tmp") or "/." in f"/{relative}":
                raise GovernanceError("SCALE-BP v2 bundle contains a temporary member.")
            rows.append(relative)
    return tuple(sorted(rows))


def _assert_closed_world(
    root: Path,
    *,
    indexed: set[str],
    structural: set[str],
    allowed_post: Sequence[str],
) -> None:
    actual = set(_current_members(root, excluded=set(MUTABLE_LIFECYCLE_MEMBERS)))
    allowed = indexed | structural | set(allowed_post)
    foreign = sorted(actual - allowed)
    missing = sorted(indexed - actual)
    if foreign or missing:
        raise GovernanceError(
            f"SCALE-BP v2 closed-world bundle drifted; foreign={foreign}, missing={missing}."
        )


__all__ = (
    "FINAL_INDEX_MEMBER",
    "FINAL_SEAL_MEMBER",
    "MUTABLE_LIFECYCLE_MEMBERS",
    "PRETERMINAL_INDEX_MEMBER",
    "PRETERMINAL_SEAL_MEMBER",
    "persist_final_content_index",
    "persist_preterminal_bundle",
    "validate_final_content_index",
    "validate_preterminal_bundle",
)
