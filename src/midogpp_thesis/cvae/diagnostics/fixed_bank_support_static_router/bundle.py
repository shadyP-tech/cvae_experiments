"""Closed-world, content-indexed output inventory for the S4 diagnostic."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .hashing import canonical_hash


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/fixed_bank_a1_action_probabilities.npz",
    "arrays/action_identity_null_selections.npz",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/five_fold_partition.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/fixed_bank_a1_prediction_index.json",
    "manifests/fixed_bank_a1_prediction_seal.json",
    "manifests/sealed_probability_surface.json",
    "manifests/route_fold_plan_seals.json",
    "manifests/global_static_selection_seal.json",
    "manifests/all_route_decisions_seal.json",
    "manifests/action_identity_null_selection_plan_seal.json",
    "manifests/action_identity_null_seal.json",
    "manifests/sealed_terminal_evaluation.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/five_fold_partitions.csv",
    "tables/global_static_action_scores.csv",
    "tables/global_static_selections.csv",
    "tables/fold_support_action_scores.csv",
    "tables/route_decisions.csv",
    "tables/method_decisions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "tables/null_route_selection_counts.csv",
    "reports/workstation_preflight.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/action_identity_null_summary.json",
    "reports/run_state.json",
    "reports/fresh_process_validation.json",
    "reports/validation_report.json",
)

_INDEX_EXCLUDED = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/fresh_process_validation.json",
        "reports/validation_report.json",
    }
)
CONTENT_INDEX_MEMBERS = tuple(
    member for member in REQUIRED_FILES if member not in _INDEX_EXCLUDED
)


def relative_files(root: Path) -> tuple[str, ...]:
    _reject_symlinks(root)
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".run.lock"
        )
    )


def assert_closed_world(
    root: Path,
    *,
    allow_incomplete: bool,
    allow_pending_validation: bool = False,
) -> None:
    observed = set(relative_files(root))
    if allow_incomplete:
        observed = {member for member in observed if not _owned_checkpoint(member)}
    required = set(REQUIRED_FILES)
    extras = sorted(observed - required)
    pending = (
        {
            "reports/fresh_process_validation.json",
            "reports/validation_report.json",
        }
        if allow_pending_validation
        else set()
    )
    missing = sorted(required - observed - (required if allow_incomplete else pending))
    if extras or missing:
        raise ProtocolError(
            f"S4 closed-world inventory drifted: extras={extras}, missing={missing}."
        )


def cleanup_owned_atomic_temps(root: Path) -> None:
    if not root.exists():
        return
    _reject_symlinks(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = re.fullmatch(r"(?P<member>.+)\.[1-9][0-9]*\.tmp", relative)
        if match and (
            match.group("member") in REQUIRED_FILES
            or _owned_checkpoint(match.group("member"))
        ):
            path.unlink()


def write_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    rows = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file() or path.is_symlink():
            raise ProtocolError(f"S4 content member is absent or unsafe: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": "fixed_bank_support_static_router_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "previous_stage90_output_prediction_or_scratch_consumed": False,
    }
    payload = {**unhashed, "content_hash": canonical_hash(unhashed)}
    _persist_or_validate(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    rows = payload.get("members")
    if (
        payload.get("content_hash") != canonical_hash(unhashed)
        or payload.get("schema_version")
        != "fixed_bank_support_static_router_content_index_v1"
        or payload.get("config_contract_hash") != config_contract_hash
        or payload.get("protocol_contract_hash") != protocol_contract_hash
        or payload.get("member_count") != len(CONTENT_INDEX_MEMBERS)
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
        or payload.get("closed_world") is not True
        or payload.get("raw_labels_persisted") is not False
        or payload.get("per_case_bacc_persisted") is not False
        or payload.get("consumed_test_diagnostic_only") is not True
        or payload.get("fresh_evidence") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("may_feed_another_experiment") is not False
        or payload.get(
            "previous_stage90_output_prediction_or_scratch_consumed"
        )
        is not False
    ):
        raise ProtocolError("S4 content index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("S4 content index row is malformed.")
        path = root / str(row["member"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("S4 content-index member drifted.")
    return payload


def _persist_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ProtocolError("S4 JSON output cannot be a symlink.")
    if path.is_file():
        if read_json(path) != dict(payload):
            raise ProtocolError(f"S4 refuses to repair changed JSON: {path}.")
    else:
        atomic_json(path, payload)


def _owned_checkpoint(member: str) -> bool:
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    patterns = (
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
        r"checkpoints/fixed_bank_a1_action_predictions/(?:target_scratch\.json|target_embeddings\.npy)",
        rf"checkpoints/fixed_bank_a1_action_predictions/tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
        r"checkpoints/terminal/sealed_result\.json",
    )
    return any(re.fullmatch(pattern, member) is not None for pattern in patterns)


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("S4 bundle root cannot be a symlink.")
    if not root.exists():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            path = base / name
            if path.is_symlink():
                raise ProtocolError(f"S4 closed world contains a symlink: {path}.")


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "cleanup_owned_atomic_temps",
    "relative_files",
    "validate_content_index",
    "write_content_index",
)
