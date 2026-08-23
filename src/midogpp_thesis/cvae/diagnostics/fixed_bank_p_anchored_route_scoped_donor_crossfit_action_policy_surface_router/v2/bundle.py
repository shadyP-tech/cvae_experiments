"""Closed-world controlled-subtree indexes for P-DCAPS v2 bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json, sha256_file
from ..identity import canonical_hash


PRETERMINAL_INDEX_MEMBER = "manifests/preterminal_content_index.json"
FINAL_INDEX_MEMBER = "manifests/final_content_index.json"
CONTROLLED_SUBTREES = (
    "arrays",
    "tables",
    "reports",
    "manifests",
    "provenance",
)
UNINDEXED_LIFECYCLE_MEMBERS = frozenset(
    {
        "reports/run_state.json",
        "reports/final_fresh_process_attestation.json",
        "reports/validation_report.json",
    }
)
PRETERMINAL_POST_INDEX_MEMBERS = frozenset(
    {"reports/preterminal_fresh_process_attestation.json"}
)


def build_closed_world_index(
    root: Path,
    *,
    required_members: Sequence[str],
    phase: str,
) -> dict[str, object]:
    """Index exact science members and reject extras in controlled subtrees."""

    path = Path(root)
    phase_id = str(phase)
    index_member = (
        PRETERMINAL_INDEX_MEMBER if phase_id == "preterminal" else FINAL_INDEX_MEMBER
    )
    required = tuple(sorted(str(value) for value in required_members))
    if (
        phase_id not in {"preterminal", "final"}
        or not required
        or len(required) != len(set(required))
        or index_member in required
        or any(
            value.startswith("/")
            or ".." in Path(value).parts
            or (
                value != "config.resolved.yaml"
                and Path(value).parts[0]
                not in {"arrays", "tables", "reports", "manifests", "provenance"}
            )
            for value in required
        )
    ):
        raise ProtocolError("P-DCAPS v2 content-index inventory drifted.")
    rows = []
    for member in required:
        target = path / member
        if not target.is_file() or target.is_symlink():
            raise ProtocolError(f"P-DCAPS v2 required member is absent: {member}.")
        rows.append(
            {
                "member": member,
                "sha256": sha256_file(target),
                "size": target.stat().st_size,
            }
        )
    controlled_subtrees = CONTROLLED_SUBTREES
    observed_controlled = _observed_controlled_members(
        path, index_member=index_member
    )
    expected_controlled = tuple(
        value
        for value in required
        if Path(value).parts[0] in set(controlled_subtrees)
    )
    if observed_controlled != expected_controlled:
        raise ProtocolError(
            "P-DCAPS v2 controlled bundle subtree is not closed-world."
        )
    observed_root_files = tuple(
        sorted(
            target.name
            for target in path.iterdir()
            if (target.is_file() or target.is_symlink())
            and target.name != ".run.lock"
        )
    )
    expected_root_files = tuple(
        value for value in required if len(Path(value).parts) == 1
    )
    observed_root_directories = tuple(
        sorted(target.name for target in path.iterdir() if target.is_dir())
    )
    if (
        observed_root_files != expected_root_files
        or observed_root_directories != tuple(sorted(controlled_subtrees))
    ):
        raise ProtocolError("P-DCAPS v2 bundle root is not closed-world.")
    base = {
        "schema_version": "pdcaps_v2_closed_world_content_index_v1",
        "phase": phase_id,
        "members": rows,
        "member_count": len(rows),
        "controlled_subtrees": list(controlled_subtrees),
        "observed_controlled_members": list(observed_controlled),
        "unexpected_controlled_member_count": 0,
    }
    payload = {**base, "content_index_hash": canonical_hash(base)}
    atomic_json(path / index_member, payload)
    return payload


def verify_closed_world_index(
    root: Path,
    *,
    phase: str,
    expected_members: Sequence[str] | None = None,
) -> dict[str, object]:
    path = Path(root)
    index_member = (
        PRETERMINAL_INDEX_MEMBER if phase == "preterminal" else FINAL_INDEX_MEMBER
    )
    payload = read_json(path / index_member)
    verify_index_payload_members(
        path,
        payload,
        phase=phase,
        expected_members=expected_members,
    )
    observed_controlled = _observed_controlled_members(
        path, index_member=index_member
    )
    if observed_controlled != tuple(payload.get("observed_controlled_members", ())):
        raise ProtocolError("P-DCAPS v2 unexpected controlled member appeared.")
    return payload


def verify_index_payload_members(
    root: Path,
    payload: dict[str, object],
    *,
    phase: str,
    expected_members: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Verify an index and its bytes, optionally against a frozen inventory."""

    path = Path(root)
    rows = payload.get("members")
    if (
        payload.get("schema_version")
        != "pdcaps_v2_closed_world_content_index_v1"
        or payload.get("phase") != phase
        or payload.get("controlled_subtrees") != list(CONTROLLED_SUBTREES)
        or not isinstance(rows, list)
        or not all(isinstance(row, dict) for row in rows)
        or payload.get("member_count") != len(rows)
        or payload.get("unexpected_controlled_member_count") != 0
    ):
        raise ProtocolError("P-DCAPS v2 content index is malformed.")
    unhashed = {
        key: value for key, value in payload.items() if key != "content_index_hash"
    }
    if payload.get("content_index_hash") != canonical_hash(unhashed):
        raise ProtocolError("P-DCAPS v2 content index hash drifted.")
    members = tuple(str(row.get("member")) for row in rows)
    frozen = (
        members
        if expected_members is None
        else tuple(sorted(str(value) for value in expected_members))
    )
    if (
        members != tuple(sorted(members))
        or len(members) != len(set(members))
        or members != frozen
    ):
        raise ProtocolError("P-DCAPS v2 indexed member inventory drifted.")
    for row in rows:
        target = path / str(row["member"])
        if (
            not target.is_file()
            or target.is_symlink()
            or row.get("sha256") != sha256_file(target)
            or row.get("size") != target.stat().st_size
        ):
            raise ProtocolError("P-DCAPS v2 indexed member bytes drifted.")
    return members


def _observed_controlled_members(
    root: Path, *, index_member: str
) -> tuple[str, ...]:
    path = Path(root)
    excluded = set(UNINDEXED_LIFECYCLE_MEMBERS)
    if index_member == PRETERMINAL_INDEX_MEMBER:
        # The durable attestation is written only after two validators have
        # verified the preterminal index.  It must remain outside that index so
        # the immediate pre-label barrier can verify the same frozen bytes.
        excluded.update(PRETERMINAL_POST_INDEX_MEMBERS)
    return tuple(
        sorted(
            relative
            for subtree in CONTROLLED_SUBTREES
            if (path / subtree).is_dir()
            for target in (path / subtree).rglob("*")
            if target.is_file() or target.is_symlink()
            if (relative := target.relative_to(path).as_posix()) != index_member
            and relative not in excluded
        )
    )


__all__ = (
    "CONTROLLED_SUBTREES",
    "FINAL_INDEX_MEMBER",
    "PRETERMINAL_INDEX_MEMBER",
    "build_closed_world_index",
    "verify_index_payload_members",
    "verify_closed_world_index",
)
