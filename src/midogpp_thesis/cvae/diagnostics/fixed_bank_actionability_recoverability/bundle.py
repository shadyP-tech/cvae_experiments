"""Closed-world artifact inventory for the terminal diagnostic."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from .artifact_io import (
    persist_or_validate_json,
    read_json,
    relative_files,
    sha256_file,
)
from .constants import GEOMETRY_IDS
from .hashing import canonical_hash


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/actionability_action_probabilities.npz",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/case_oof_partition.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/actionability_prediction_index.json",
    "manifests/actionability_prediction_seal.json",
    "manifests/sealed_probability_surface.json",
    "manifests/prelabel_feature_seal.json",
    "manifests/loco_utility_seals.json",
    "manifests/model_seals.json",
    "manifests/pre_support_decisions_seal.json",
    "manifests/all_method_decisions_seal.json",
    "manifests/permutation_provenance_seal.json",
    "manifests/sealed_terminal_evaluation.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/case_oof_partitions.csv",
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/case_action_features.csv",
    "tables/loco_utility_targets.csv",
    "tables/model_fits.csv",
    "tables/model_predictions.csv",
    "tables/method_decisions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_method_summary.csv",
    "tables/terminal_contrasts.csv",
    "tables/oracle_rank_metrics.csv",
    "tables/complementarity.csv",
    "tables/rank_stability.csv",
    "tables/permutation_metrics.csv",
    "reports/workstation_preflight.json",
    "reports/phase_01_prelabel_seal_complete.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

_INDEX_EXCLUDED = {
    "manifests/content_index.json",
    "reports/run_state.json",
    "reports/validation_report.json",
}
CONTENT_INDEX_MEMBERS = tuple(
    member for member in REQUIRED_FILES if member not in _INDEX_EXCLUDED
)


def cleanup_owned_atomic_temps(root: Path) -> None:
    """Remove interrupted package-owned writes while holding the run lock."""

    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = re.fullmatch(r"(?P<member>.+)\.[1-9][0-9]*\.tmp", relative)
        if match and (
            match.group("member") in REQUIRED_FILES
            or _is_owned_resume_checkpoint(match.group("member"))
        ):
            path.unlink()


def assert_closed_world(
    root: Path,
    *,
    allow_incomplete: bool,
    allow_pending_validation: bool = False,
) -> None:
    observed = set(relative_files(root))
    if allow_incomplete:
        observed = {
            member for member in observed if not _is_owned_resume_checkpoint(member)
        }
    required = set(REQUIRED_FILES)
    extras = sorted(observed - required)
    permitted_missing = (
        required
        if allow_incomplete
        else {"reports/validation_report.json"}
        if allow_pending_validation
        else set()
    )
    missing = sorted(required - observed - permitted_missing)
    if extras or missing:
        raise ProtocolError(
            "Actionability/recoverability closed-world inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def assert_terminal_phase_complete(root: Path) -> None:
    expected = set(REQUIRED_FILES) - {
        "manifests/content_index.json",
        "reports/validation_report.json",
    }
    observed = set(relative_files(root))
    extras = sorted(observed - expected)
    missing = sorted(expected - observed)
    if extras or missing:
        raise ProtocolError(
            "Terminal recovery inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def write_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Content member is absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": (
            "midogpp_fixed_bank_actionability_recoverability_content_index_v1"
        ),
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "geometry_ids": list(GEOMETRY_IDS),
        "global_method_ids": ["B"],
        "per_geometry_method_ids": [
            "U",
            "G",
            "R",
            "P",
            "S_y",
            "O_static",
            "O_case",
        ],
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
        "deployable_policy_or_action_capability_present": False,
        "may_feed_another_experiment": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
    }
    payload = {**unhashed, "content_hash": canonical_hash(unhashed)}
    persist_or_validate_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    observed = read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    expected_header = {
        "schema_version": (
            "midogpp_fixed_bank_actionability_recoverability_content_index_v1"
        ),
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "member_count": len(CONTENT_INDEX_MEMBERS),
        "closed_world": True,
        "geometry_ids": list(GEOMETRY_IDS),
        "global_method_ids": ["B"],
        "per_geometry_method_ids": [
            "U",
            "G",
            "R",
            "P",
            "S_y",
            "O_static",
            "O_case",
        ],
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "consumed_test_diagnostic_only": True,
        "deployable_policy_or_action_capability_present": False,
        "may_feed_another_experiment": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
    }
    if (
        observed.get("content_hash") != canonical_hash(unhashed)
        or any(observed.get(key) != value for key, value in expected_header.items())
        or not isinstance(observed.get("members"), list)
        or [row.get("member") for row in observed["members"]]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Content-index header drifted.")
    for row in observed["members"]:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("Content-index row is malformed.")
        path = root / str(row["member"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ProtocolError("Content-index member drifted.")
    return observed


def _is_owned_resume_checkpoint(member: str) -> bool:
    centers = "(?:0|1|2|3|5|6|7|8|9)"
    seeds = "(?:17|42|101)"
    patterns = (
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
        r"checkpoints/actionability_action_predictions/(?:target_embeddings\.npy|target_scratch\.json)",
        rf"checkpoints/actionability_action_predictions/tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
    )
    return any(re.fullmatch(pattern, member) is not None for pattern in patterns)


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "assert_terminal_phase_complete",
    "cleanup_owned_atomic_temps",
    "validate_content_index",
    "write_content_index",
)
