"""Closed-world inventory and byte-first index for the residual stacker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from .artifact_io import persist_or_validate_json, read_json, relative_files, sha256_file


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
    "manifests/label_free_case_feature_surface.json",
    "manifests/label_free_source_control_surface.json",
    "manifests/loco_hierarchical_model_seals.json",
    "manifests/fold_calibrations_and_method_decisions.json",
    "manifests/all_fold_method_decisions_seal.json",
    "manifests/permutation_provenance_seal.json",
    "manifests/terminal_pooled_bacc_evaluation.json",
    "manifests/content_index.json",
    "tables/case_oof_partitions.csv",
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/label_free_case_features.csv",
    "tables/label_free_source_controls.csv",
    "tables/loco_donor_responses.csv",
    "tables/loco_model_components.csv",
    "tables/fold_calibrations.csv",
    "tables/fold_method_decisions.csv",
    "tables/oof_case_confusion_sufficient_statistics.csv",
    "tables/oof_pooled_exact_bacc.csv",
    "tables/paired_whole_case_cluster_contrasts.csv",
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


def assert_closed_world(
    root: Path,
    *,
    allow_incomplete: bool,
    allow_pending_validation: bool = False,
) -> None:
    observed = set(relative_files(root))
    if allow_incomplete:
        observed = {member for member in observed if not _is_owned_resume_checkpoint(member)}
    required = set(REQUIRED_FILES)
    permitted_missing = required if allow_incomplete else (
        {"reports/validation_report.json"} if allow_pending_validation else set()
    )
    extras = sorted(observed - required)
    missing = sorted(required - observed - permitted_missing)
    if extras or missing:
        raise ProtocolError(
            f"Residual-stacker closed-world inventory drifted: extras={extras}, missing={missing}."
        )


def write_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Residual-stacker content member is absent: {member}.")
        rows.append(
            {"member": member, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    unhashed = {
        "schema_version": "midogpp_hierarchical_residual_stacker_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "support_selection_objective": "fixed_class_balanced_log_loss_only",
        "terminal_metric": "pooled_exact_bacc",
        "uncertainty_unit": "paired_whole_case_cluster",
        "per_case_bacc_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "diagnostic_method_predictions_present": True,
        "deployable_policy_or_action_capability_present": False,
        "generic_consumer_authorized": False,
        "may_feed_another_stage90": False,
        "prior_stage90_artifact_or_scratch_consumed": False,
    }
    payload = {**unhashed, "content_hash": _sha256(unhashed)}
    persist_or_validate_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    observed = read_json(root / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    expected_keys = {
        "schema_version", "config_contract_hash", "members", "member_count",
        "closed_world", "support_selection_objective", "terminal_metric",
        "uncertainty_unit", "per_case_bacc_persisted",
        "terminal_consumed_test_diagnostic_only", "diagnostic_method_predictions_present",
        "deployable_policy_or_action_capability_present", "generic_consumer_authorized",
        "may_feed_another_stage90", "prior_stage90_artifact_or_scratch_consumed",
        "content_hash",
    }
    if (
        set(observed) != expected_keys
        or observed.get("schema_version")
        != "midogpp_hierarchical_residual_stacker_content_index_v1"
        or observed.get("content_hash") != _sha256(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or observed.get("member_count") != len(CONTENT_INDEX_MEMBERS)
        or observed.get("closed_world") is not True
        or observed.get("support_selection_objective")
        != "fixed_class_balanced_log_loss_only"
        or observed.get("terminal_metric") != "pooled_exact_bacc"
        or observed.get("uncertainty_unit") != "paired_whole_case_cluster"
        or observed.get("per_case_bacc_persisted") is not False
        or observed.get("terminal_consumed_test_diagnostic_only") is not True
        or observed.get("diagnostic_method_predictions_present") is not True
        or observed.get("deployable_policy_or_action_capability_present") is not False
        or observed.get("generic_consumer_authorized") is not False
        or observed.get("may_feed_another_stage90") is not False
        or observed.get("prior_stage90_artifact_or_scratch_consumed") is not False
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Residual-stacker content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"member", "size_bytes", "sha256"}:
            raise ProtocolError("Residual-stacker content-index row is malformed.")
        path = root / str(row.get("member"))
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("Residual-stacker content-index member drifted.")
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
    "validate_content_index",
    "write_content_index",
)
