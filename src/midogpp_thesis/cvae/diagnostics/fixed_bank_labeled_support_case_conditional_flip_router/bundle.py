"""Closed-world output inventory for the terminal flip-router diagnostic."""

from __future__ import annotations

from pathlib import Path
import os
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
    "manifests/three_role_partition.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/fixed_bank_a1_prediction_index.json",
    "manifests/fixed_bank_a1_prediction_seal.json",
    "manifests/sealed_probability_surface.json",
    "manifests/prelabel_feature_seal.json",
    "manifests/fold_plan_seals.json",
    "manifests/donor_model_seals.json",
    "manifests/static_selection_seals.json",
    "manifests/calibration_seals.json",
    "manifests/all_method_decisions_seal.json",
    "manifests/permutation_provenance_seal.json",
    "manifests/sealed_terminal_evaluation.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/three_role_partitions.csv",
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/case_action_features.csv",
    "tables/donor_contribution_targets.csv",
    "tables/model_fits.csv",
    "tables/static_source_selections.csv",
    "tables/directional_calibrations.csv",
    "tables/method_decisions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "tables/router_identification_metrics.csv",
    "tables/permutation_metrics.csv",
    "reports/workstation_preflight.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)
_INDEX_EXCLUDED = {
    "manifests/content_index.json", "reports/run_state.json", "reports/validation_report.json",
}
CONTENT_INDEX_MEMBERS = tuple(member for member in REQUIRED_FILES if member not in _INDEX_EXCLUDED)


def relative_files(root: Path) -> tuple[str, ...]:
    _reject_symlinks(root)
    return tuple(sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name != ".run.lock"
    ))


def assert_closed_world(root: Path, *, allow_incomplete: bool, allow_pending_validation: bool = False) -> None:
    observed = set(relative_files(root))
    if allow_incomplete:
        observed = {member for member in observed if not _owned_checkpoint(member)}
    required = set(REQUIRED_FILES)
    extras = sorted(observed - required)
    permitted = required if allow_incomplete else {"reports/validation_report.json"} if allow_pending_validation else set()
    missing = sorted(required - observed - permitted)
    if extras or missing:
        raise ProtocolError(f"Flip-router closed-world inventory drifted: extras={extras}, missing={missing}.")


def cleanup_owned_atomic_temps(root: Path) -> None:
    if not root.exists():
        return
    _reject_symlinks(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = re.fullmatch(r"(?P<member>.+)\.[1-9][0-9]*\.tmp", relative)
        if match and (match.group("member") in REQUIRED_FILES or _owned_checkpoint(match.group("member"))):
            path.unlink()


def write_content_index(root: Path, *, config_contract_hash: str, protocol_contract_hash: str) -> Mapping[str, object]:
    rows = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Flip-router content member absent: {member}.")
        rows.append({"member": member, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    unhashed = {
        "schema_version": "fixed_bank_labeled_support_flip_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
        "may_feed_another_experiment": False,
        "previous_stage90_output_prediction_or_scratch_consumed": False,
    }
    payload = {**unhashed, "content_hash": canonical_hash(unhashed)}
    _persist_or_validate(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(root: Path, *, config_contract_hash: str, protocol_contract_hash: str) -> Mapping[str, object]:
    payload = read_json(root / "manifests/content_index.json")
    if (
        payload.get("content_hash") != canonical_hash({key: value for key, value in payload.items() if key != "content_hash"})
        or payload.get("config_contract_hash") != config_contract_hash
        or payload.get("protocol_contract_hash") != protocol_contract_hash
        or payload.get("member_count") != len(CONTENT_INDEX_MEMBERS)
        or [row.get("member") for row in payload.get("members", [])] != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Flip-router content index header drifted.")
    for row in payload["members"]:
        path = root / str(row["member"])
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ProtocolError("Flip-router content index member drifted.")
    return payload


def _persist_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_file():
        if read_json(path) != dict(payload):
            raise ProtocolError(f"Flip-router refuses to repair changed JSON: {path}.")
    else:
        atomic_json(path, payload)


def _owned_checkpoint(member: str) -> bool:
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    patterns = (
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
        r"checkpoints/fixed_bank_a1_action_predictions/(?:target_scratch\.json|target_embeddings\.npy)",
        rf"checkpoints/fixed_bank_a1_action_predictions/tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
        r"checkpoints/terminal_evaluation/sealed_result\.json",
    )
    return any(re.fullmatch(pattern, member) is not None for pattern in patterns)


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ProtocolError("Flip-router bundle root cannot be a symlink.")
    if not root.exists():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            path = base / name
            if path.is_symlink():
                raise ProtocolError(
                    f"Flip-router closed world contains a symlink: {path}."
                )


__all__ = (
    "CONTENT_INDEX_MEMBERS", "REQUIRED_FILES", "assert_closed_world", "cleanup_owned_atomic_temps",
    "relative_files", "validate_content_index", "write_content_index",
)
