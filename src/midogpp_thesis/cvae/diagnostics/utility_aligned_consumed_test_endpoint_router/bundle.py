"""Closed-world inventory for the target-static endpoint-router bundle."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import (
    persist_or_validate_json,
    read_json,
    relative_files,
    sha256_file,
)
from .experiment_contracts import ACTION_IDS, PRIMARY_CONTRASTS


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/development_probabilities.npz",
    "arrays/target_action_probabilities.npz",
    "manifests/protocol_manifest.json",
    "manifests/support_partition_lock.json",
    "manifests/action_library.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/development_prediction_index.json",
    "manifests/development_prediction_seal.json",
    "manifests/development_endpoint_response_seal.json",
    "manifests/feature_surface_set.json",
    "manifests/model_index.json",
    "manifests/cardinality_transfer_seal.json",
    "manifests/target_policy_plans.json",
    "manifests/frozen_actions.json",
    "manifests/global_prelabel_seal.json",
    "manifests/target_prediction_index.json",
    "manifests/target_prediction_seal.json",
    "manifests/sealed_terminal_evaluation.json",
    "manifests/content_index.json",
    "tables/support_partitions.csv",
    "tables/development_endpoint_responses.csv",
    "tables/source_inner_feature_rows.csv",
    "tables/model_index.csv",
    "tables/target_feature_rows.csv",
    "tables/target_policy_plans.csv",
    "tables/frozen_actions.csv",
    "tables/terminal_endpoint_scores.csv",
    "tables/center_contrasts.csv",
    "tables/aggregate_contrasts.csv",
    "tables/oracle_rank_diagnostics.csv",
    "reports/workstation_preflight.json",
    "reports/development_label_access_report.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/runtime_summary.json",
    "reports/publication_decision.json",
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
    root: str | Path,
    *,
    allow_incomplete: bool,
    allow_pending_validation: bool = False,
) -> None:
    base = Path(root)
    observed = set(relative_files(base))
    # The runner owns exactly one transient root lock while validating phase
    # checkpoints. A nested lock is unexpected and remains a closed-world extra.
    observed.discard(".run.lock")
    if allow_incomplete:
        observed = {
            member
            for member in observed
            if not _owned_resume_checkpoint(member)
            and not _owned_atomic_temporary(member)
        }
    required = set(REQUIRED_FILES)
    permitted_missing = (
        required
        if allow_incomplete
        else {"reports/validation_report.json"}
        if allow_pending_validation
        else set()
    )
    extras = sorted(observed - required)
    missing = sorted(required - observed - permitted_missing)
    if extras or missing:
        raise ProtocolError(
            "Consumed-test endpoint-router closed-world inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def write_content_index(
    root: str | Path,
    *,
    config_contract_hash: str,
    protocol_contract_hash: str,
) -> Mapping[str, object]:
    base = Path(root)
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = base / member
        if not path.is_file():
            raise ProtocolError(f"Endpoint-router content member is absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": (
            "midogpp_utility_aligned_consumed_test_endpoint_router_content_index_v1"
        ),
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "method_ids": list(ACTION_IDS),
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "support_labels_used": False,
        "same_outer_H_evaluation_labels_used_for_plan_H": False,
        "terminal_consumed_test_diagnostic_only": True,
        "deployable_policy_or_action_capability_present": False,
        "generic_consumer_authorized": False,
        "may_feed_another_experiment": False,
        "prior_stage90_output_or_amendment_consumed": False,
    }
    payload = {**unhashed, "content_hash": canonical_sha256(unhashed)}
    persist_or_validate_json(base / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: str | Path,
    *,
    config_contract_hash: str,
    protocol_contract_hash: str,
) -> Mapping[str, object]:
    base = Path(root)
    observed = read_json(base / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    if (
        observed.get("schema_version")
        != "midogpp_utility_aligned_consumed_test_endpoint_router_content_index_v1"
        or observed.get("content_hash") != canonical_sha256(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or observed.get("protocol_contract_hash") != protocol_contract_hash
        or observed.get("member_count") != len(CONTENT_INDEX_MEMBERS)
        or observed.get("closed_world") is not True
        or observed.get("method_ids") != list(ACTION_IDS)
        or observed.get("primary_contrasts") != list(PRIMARY_CONTRASTS)
        or observed.get("support_labels_used") is not False
        or observed.get("same_outer_H_evaluation_labels_used_for_plan_H") is not False
        or observed.get("terminal_consumed_test_diagnostic_only") is not True
        or observed.get("deployable_policy_or_action_capability_present") is not False
        or observed.get("generic_consumer_authorized") is not False
        or observed.get("may_feed_another_experiment") is not False
        or observed.get("prior_stage90_output_or_amendment_consumed") is not False
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Endpoint-router content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "member",
            "size_bytes",
            "sha256",
        }:
            raise ProtocolError("Endpoint-router content-index row is malformed.")
        path = base / str(row["member"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ProtocolError("Endpoint-router content-index member drifted.")
    return observed


def _owned_atomic_temporary(member: str) -> bool:
    path = Path(member)
    name = path.name
    if not (name.startswith(".") and name.endswith(".tmp")):
        return False
    for destination in REQUIRED_FILES:
        expected = Path(destination)
        if path.parent == expected.parent and name.startswith(f".{expected.name}."):
            return True
    return bool(
        _CHECKPOINT_TEMP_RE.fullmatch(member)
    )


def _owned_resume_checkpoint(member: str) -> bool:
    return bool(_CHECKPOINT_MEMBER_RE.fullmatch(member))


_CHECKPOINT_STEM = (
    r"(?:development|target)_H[0-9]+_q[0-9]+_"
    r"train(?:17|42|101)_gen(?:17|42|101)"
)
_CHECKPOINT_MEMBER_RE = re.compile(
    rf"checkpoints/(?:development_predictions|target_predictions)/"
    rf"{_CHECKPOINT_STEM}\.(?:json|npz)"
    r"|checkpoints/frozen_source_streams/[a-zA-Z0-9_.-]+\.(?:json|npy|npz)"
    r"|checkpoints/feature_runtime/feature_input_seal\.json"
    r"|checkpoints/feature_runtime/support_q(?:0|1|2|3|5|6|7|8|9)\.npy"
    r"|checkpoints/feature_runtime/feature_e(?:0|1|2|3|5|6|7|8|9)_"
    r"train(?:17|42|101)\.(?:json|npz)"
)
_CHECKPOINT_TEMP_RE = re.compile(
    rf"checkpoints/(?:development_predictions|target_predictions)/\."
    rf"{_CHECKPOINT_STEM}\.(?:json|npz)\.[a-zA-Z0-9_-]+\.tmp"
    r"|checkpoints/frozen_source_streams/\.[a-zA-Z0-9_.-]+\."
    r"[a-zA-Z0-9_-]+\.tmp"
    r"|checkpoints/feature_runtime/\.(?:feature_input_seal\.json|"
    r"support_q(?:0|1|2|3|5|6|7|8|9)\.npy|"
    r"feature_e(?:0|1|2|3|5|6|7|8|9)_train(?:17|42|101)\.(?:json|npz))\."
    r"[a-zA-Z0-9_-]+\.tmp"
)


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "validate_content_index",
    "write_content_index",
)
