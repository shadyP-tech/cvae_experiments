"""Closed-world, nonrepairing output inventory for the case-directional correctness diagnostic."""

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
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/fixed_bank_a1_prediction_index.json",
    "manifests/fixed_bank_a1_prediction_seal.json",
    "manifests/physical_prelabel_seal.json",
    "manifests/held_case_plan_seal.json",
    "manifests/held_case_feature_seal.json",
    "manifests/donor_prior_seal.json",
    "manifests/route_model_seal.json",
    "manifests/route_decision_seal.json",
    "manifests/aggregate_plan_decision_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/exact_nine_probability_index.csv",
    "tables/held_case_plans.csv",
    "tables/held_case_features.csv",
    "tables/support_response_counts.csv",
    "tables/donor_priors.csv",
    "tables/route_model_fits.csv",
    "tables/route_candidate_scores.csv",
    "tables/route_decisions.csv",
    "tables/method_predictions.csv",
    "tables/descriptive_method_predictions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_method_metrics.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "tables/router_identification_metrics.csv",
    "tables/feature_permutation_summary.csv",
    "reports/workstation_preflight.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

_INDEX_EXCLUDED = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
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
    _reject_foreign_directories(root)
    observed = set(relative_files(root))
    if allow_incomplete:
        observed = {member for member in observed if not _owned_task_checkpoint(member)}
    required = set(REQUIRED_FILES)
    extras = sorted(observed - required)
    allowed_missing = (
        required
        if allow_incomplete
        else {"reports/validation_report.json"}
        if allow_pending_validation
        else set()
    )
    missing = sorted(required - observed - allowed_missing)
    if extras or missing:
        raise ProtocolError(
            "Case-directional closed-world inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def cleanup_owned_atomic_temps(root: Path) -> None:
    """Delete only recognized atomic-write remnants, never scientific products."""

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
            or _owned_task_checkpoint(match.group("member"))
        ):
            path.unlink()


def write_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(f"Case-directional content member absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": "fixed_bank_cdca_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "previous_stage90_artifact_checkpoint_or_scratch_consumed": False,
        "publication_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
    }
    payload = {**unhashed, "content_hash": canonical_hash(unhashed)}
    _persist_or_validate(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    raw_rows = payload.get("members")
    exact_header = {
        "schema_version": "fixed_bank_cdca_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "member_count": len(CONTENT_INDEX_MEMBERS),
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "previous_stage90_artifact_checkpoint_or_scratch_consumed": False,
        "publication_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
    }
    if (
        payload.get("content_hash") != canonical_hash(unhashed)
        or set(payload) != set(exact_header) | {"members", "content_hash"}
        or any(payload.get(key) != value for key, value in exact_header.items())
        or not isinstance(raw_rows, list)
        or len(raw_rows) != len(CONTENT_INDEX_MEMBERS)
        or [row.get("member") for row in raw_rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Case-directional content-index header drifted.")
    for row in raw_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("Case-directional content-index row malformed.")
        path = root / str(row["member"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("Case-directional indexed member drifted.")
    return payload


def _persist_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ProtocolError("Case-directional content index is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(payload):
            raise ProtocolError("Case-directional refuses content-index repair.")
        return
    atomic_json(path, payload)


def _owned_task_checkpoint(member: str) -> bool:
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    patterns = (
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
        r"checkpoints/fixed_bank_a1_action_predictions/(?:target_scratch\.json|target_embeddings\.npy)",
        rf"checkpoints/fixed_bank_a1_action_predictions/tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
    )
    return any(re.fullmatch(pattern, member) is not None for pattern in patterns)


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("Case-directional bundle root cannot be a symlink.")
    if not root.exists():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            if (base / name).is_symlink():
                raise ProtocolError("Case-directional bundle contains a symlink.")


def _reject_foreign_directories(root: Path) -> None:
    if not root.exists():
        return
    allowed = {"arrays", "manifests", "provenance", "reports", "tables"}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    if observed - allowed:
        raise ProtocolError(
            "Case-directional bundle contains a foreign directory."
        )


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "cleanup_owned_atomic_temps",
    "relative_files",
    "validate_content_index",
    "write_content_index",
)

