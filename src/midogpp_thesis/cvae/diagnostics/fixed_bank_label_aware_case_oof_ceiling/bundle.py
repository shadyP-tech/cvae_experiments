"""Closed-world inventory and content-first validation for the ceiling bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from .artifact_io import persist_or_validate_json, read_json, relative_files, sha256_file


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/label_free_action_probabilities.npz",
    "arrays/permutation_null_actions.npy",
    "manifests/protocol_manifest.json",
    "manifests/case_oof_partition.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/label_free_prediction_index.json",
    "manifests/label_free_prediction_seal.json",
    "manifests/sealed_probability_surface.json",
    "manifests/loco_global_prior_seals.json",
    "manifests/fold_posterior_seals.json",
    "manifests/fold_decisions.json",
    "manifests/all_fold_decisions_seal.json",
    "manifests/permutation_null_decision_seal.json",
    "manifests/ceiling_evaluation.json",
    "manifests/content_index.json",
    "tables/case_oof_partitions.csv",
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/loco_global_priors.csv",
    "tables/fold_posteriors.csv",
    "tables/fold_decisions.csv",
    "tables/oof_case_metrics.csv",
    "tables/oof_center_metrics.csv",
    "tables/action_selection_metrics.csv",
    "tables/permutation_null_summary.csv",
    "reports/workstation_preflight.json",
    "reports/phase_01_global_prediction_seal_complete.json",
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
CONTENT_INDEX_MEMBERS = tuple(member for member in REQUIRED_FILES if member not in _INDEX_EXCLUDED)


def assert_closed_world(
    root: Path,
    *,
    allow_incomplete: bool,
    allow_pending_validation: bool = False,
) -> None:
    observed = set(relative_files(root))
    if allow_incomplete:
        resumable_prefixes = (
            "checkpoints/frozen_source_streams/",
            "checkpoints/label_free_action_predictions/",
        )
        observed = {
            member
            for member in observed
            if not any(member.startswith(prefix) for prefix in resumable_prefixes)
        }
    required = set(REQUIRED_FILES)
    permitted_missing = required if allow_incomplete else (
        {"reports/validation_report.json"} if allow_pending_validation else set()
    )
    extras = sorted(observed - required)
    missing = sorted(required - observed - permitted_missing)
    if extras or missing:
        raise ProtocolError(
            f"Label-aware closed-world inventory drifted: extras={extras}, missing={missing}."
        )


def write_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Label-aware content member is absent: {member}.")
        rows.append({"member": member, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    unhashed = {
        "schema_version": "midogpp_label_aware_case_oof_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "terminal_consumed_test_diagnostic_only": True,
        "support_labels_used": True,
        "policy_or_action_capability_present": False,
        "generic_consumer_authorized": False,
        "may_feed_another_stage90": False,
    }
    payload = {**unhashed, "content_hash": _sha256(unhashed)}
    persist_or_validate_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    """Validate byte inventory before reading any scientific result member."""

    observed = read_json(root / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    if (
        observed.get("schema_version") != "midogpp_label_aware_case_oof_content_index_v1"
        or observed.get("content_hash") != _sha256(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or observed.get("closed_world") is not True
        or observed.get("terminal_consumed_test_diagnostic_only") is not True
        or observed.get("support_labels_used") is not True
        or observed.get("policy_or_action_capability_present") is not False
        or observed.get("generic_consumer_authorized") is not False
        or observed.get("may_feed_another_stage90") is not False
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)] != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Label-aware content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Label-aware content-index row is malformed.")
        path = root / str(row.get("member"))
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("Label-aware content-index member drifted.")
    return observed


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "validate_content_index",
    "write_content_index",
)
