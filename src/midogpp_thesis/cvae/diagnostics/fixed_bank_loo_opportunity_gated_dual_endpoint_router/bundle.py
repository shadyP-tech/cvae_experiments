"""Closed-world output inventory and content-first integrity index."""

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
    "manifests/loo_plan_seal.json",
    "manifests/label_free_feature_seal.json",
    "manifests/donor_prior_seal.json",
    "manifests/identification_endpoint_seal.json",
    "manifests/robust_endpoint_seal.json",
    "manifests/portfolio_prediction_seal.json",
    "manifests/aggregate_plan_decision_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/exact_nine_probability_index.csv",
    "tables/whole_case_loo_plans.csv",
    "tables/label_free_candidate_features.csv",
    "tables/case_action_confusions.csv",
    "tables/route_correctness_observations.csv",
    "tables/route_model_fits.csv",
    "tables/directional_support_gains.csv",
    "tables/donor_priors.csv",
    "tables/endpoint_arms.csv",
    "tables/identification_decisions.csv",
    "tables/robust_arm_decisions.csv",
    "tables/method_predictions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_method_metrics.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "tables/router_identification_metrics.csv",
    "tables/calibration_metrics.csv",
    "tables/whole_pipeline_delete_one_center.csv",
    "tables/attribution_controls.csv",
    "reports/workstation_preflight.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/fresh_process_attestation.json",
    "reports/validation_report.json",
)

INDEX_EXCLUDED = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/fresh_process_attestation.json",
        "reports/validation_report.json",
    }
)
CONTENT_INDEX_MEMBERS = tuple(
    member for member in REQUIRED_FILES if member not in INDEX_EXCLUDED
)


def relative_files(root: Path) -> tuple[str, ...]:
    _reject_symlinks(root)
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path != root / ".run.lock"
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
    if allow_incomplete:
        allowed_missing = required
    elif allow_pending_validation:
        allowed_missing = {
            "reports/fresh_process_attestation.json",
            "reports/validation_report.json",
        }
    else:
        allowed_missing = set()
    missing = sorted(required - observed - allowed_missing)
    if extras or missing:
        raise ProtocolError(
            f"Dual-endpoint closed-world drifted: extras={extras}, missing={missing}."
        )


def write_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    rows = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(f"Dual-endpoint content member absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": "fixed_bank_dual_endpoint_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "routing_success_claimed": False,
        "weights_selected_on_same_evaluation_surface": True,
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
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
    exact_header = {
        "schema_version": "fixed_bank_dual_endpoint_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "member_count": len(CONTENT_INDEX_MEMBERS),
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_checkpoint_persisted": False,
        "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "routing_success_claimed": False,
        "weights_selected_on_same_evaluation_surface": True,
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
    }
    if (
        set(payload) != set(exact_header) | {"members", "content_hash"}
        or any(payload.get(key) != value for key, value in exact_header.items())
        or payload.get("content_hash") != canonical_hash(unhashed)
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Dual-endpoint content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        } or (
            type(row.get("size_bytes")) is not int
            or int(row["size_bytes"]) < 0
            or not isinstance(row.get("sha256"), str)
            or len(str(row["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in str(row["sha256"]))
        ):
            raise ProtocolError("Dual-endpoint content-index row malformed.")
        path = root / str(row["member"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("Dual-endpoint indexed member drifted.")
    return payload


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
            or _owned_task_checkpoint(match.group("member"))
        ):
            path.unlink()


def _persist_or_validate(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ProtocolError("Dual-endpoint content index is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(payload):
            raise ProtocolError("Dual-endpoint refuses content-index repair.")
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
        raise ProtocolError("Dual-endpoint bundle root cannot be a symlink.")
    if not root.exists():
        return
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            if (base / name).is_symlink():
                raise ProtocolError("Dual-endpoint bundle contains a symlink.")


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
        raise ProtocolError("Dual-endpoint bundle contains a foreign directory.")


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "INDEX_EXCLUDED",
    "REQUIRED_FILES",
    "assert_closed_world",
    "cleanup_owned_atomic_temps",
    "relative_files",
    "validate_content_index",
    "write_content_index",
)
