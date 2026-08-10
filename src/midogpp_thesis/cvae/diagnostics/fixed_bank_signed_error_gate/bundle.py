"""Closed-world inventory and byte-first index for the signed diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.artifact_io import (
    persist_or_validate_json,
    read_json,
    relative_files,
    sha256_file,
)


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/label_free_action_probabilities.npz",
    "manifests/protocol_manifest.json",
    "manifests/case_oof_partition.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/label_free_prediction_index.json",
    "manifests/label_free_prediction_seal.json",
    "manifests/sealed_probability_surface.json",
    "manifests/signed_prelabel_feature_seal.json",
    "manifests/signed_loco_model_seals.json",
    "manifests/signed_correction_surface_seals.json",
    "manifests/signed_fold_products.json",
    "manifests/all_fold_method_decisions_seal.json",
    "manifests/permutation_provenance_seal.json",
    "manifests/sealed_terminal_evaluation.json",
    "manifests/content_index.json",
    "tables/case_oof_partitions.csv",
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/signed_loco_models.csv",
    "tables/signed_alpha_path.csv",
    "tables/signed_corrections.csv",
    "tables/fold_decisions.csv",
    "tables/lambda_path.csv",
    "tables/oof_predictions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "reports/workstation_preflight.json",
    "reports/phase_01_prediction_and_feature_seal_complete.json",
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
    """Remove only interrupted atomic-write temps owned by this bundle."""

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = re.fullmatch(r"(?P<member>.+)\.[1-9][0-9]*\.tmp", relative)
        if match is None:
            continue
        member = match.group("member")
        if member in REQUIRED_FILES or _is_owned_resume_checkpoint(member):
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
            "Signed-error closed-world inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def assert_terminal_phase_complete(root: Path) -> None:
    """Require the exact pre-index terminal bundle for validation-only recovery."""

    observed = set(relative_files(root))
    expected = set(REQUIRED_FILES) - {
        "manifests/content_index.json",
        "reports/validation_report.json",
    }
    extras = sorted(observed - expected)
    missing = sorted(expected - observed)
    if extras or missing:
        raise ProtocolError(
            "Signed-error terminal recovery inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def write_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Signed-error content member is absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": "midogpp_fixed_bank_signed_error_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "method_ids": ["B", "B_cal", "G", "R_raw", "R_safe", "P"],
        "support_selection_objective": "fixed_class_balanced_negative_log_loss_only",
        "terminal_metric": "center_pooled_exact_bacc_equal_center_aggregate",
        "uncertainty_unit": "paired_whole_case_cluster",
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "deployable_policy_or_action_capability_present": False,
        "generic_consumer_authorized": False,
        "may_feed_another_experiment": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
    }
    payload = {**unhashed, "content_hash": _sha256(unhashed)}
    persist_or_validate_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path, *, config_contract_hash: str, protocol_contract_hash: str
) -> Mapping[str, object]:
    observed = read_json(root / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    if (
        observed.get("schema_version")
        != "midogpp_fixed_bank_signed_error_content_index_v1"
        or observed.get("content_hash") != _sha256(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or observed.get("protocol_contract_hash") != protocol_contract_hash
        or observed.get("member_count") != len(CONTENT_INDEX_MEMBERS)
        or observed.get("closed_world") is not True
        or observed.get("method_ids")
        != ["B", "B_cal", "G", "R_raw", "R_safe", "P"]
        or observed.get("support_selection_objective")
        != "fixed_class_balanced_negative_log_loss_only"
        or observed.get("terminal_metric")
        != "center_pooled_exact_bacc_equal_center_aggregate"
        or observed.get("uncertainty_unit") != "paired_whole_case_cluster"
        or observed.get("raw_labels_persisted") is not False
        or observed.get("per_case_bacc_persisted") is not False
        or observed.get("terminal_consumed_test_diagnostic_only") is not True
        or observed.get("deployable_policy_or_action_capability_present") is not False
        or observed.get("generic_consumer_authorized") is not False
        or observed.get("may_feed_another_experiment") is not False
        or observed.get("prior_stage90_artifact_or_scratch_consumed") is not False
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Signed-error content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("Signed-error content-index row is malformed.")
        path = root / str(row["member"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ProtocolError("Signed-error content-index member drifted.")
    return observed


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _is_owned_resume_checkpoint(member: str) -> bool:
    centers = "(?:0|1|2|3|5|6|7|8|9)"
    seeds = "(?:17|42|101)"
    patterns = (
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
        r"checkpoints/label_free_action_predictions/(?:target_embeddings\.npy|target_scratch\.json)",
        rf"checkpoints/label_free_action_predictions/tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
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
