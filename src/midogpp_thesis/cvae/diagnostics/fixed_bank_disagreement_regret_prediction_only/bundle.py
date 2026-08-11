"""Closed-world inventory for the label-free prediction-only bundle."""

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
from .hashing import canonical_hash
from .constants import SOURCE_CHECKPOINT_DIRECTORY, TEST_CHECKPOINT_DIRECTORY
from .development_prediction_contracts import DEVELOPMENT_CHECKPOINT_DIRECTORY


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/action_classifier_scaler_mean.npy",
    "arrays/action_classifier_scaler_scale.npy",
    "arrays/action_classifier_coefficients.npy",
    "arrays/action_classifier_intercepts.npy",
    "arrays/source_oof_classifier_scaler_mean.npy",
    "arrays/source_oof_classifier_scaler_scale.npy",
    "arrays/source_oof_classifier_coefficients.npy",
    "arrays/source_oof_classifier_intercepts.npy",
    "arrays/source_oof_action_probabilities.npz",
    "arrays/test_action_probabilities.npz",
    "arrays/model_bank.npz",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/source_oof_action_library.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/action_classifier_bank_index.json",
    "manifests/action_classifier_bank_seal.json",
    "manifests/source_oof_classifier_bank_index.json",
    "manifests/source_oof_classifier_bank_seal.json",
    "manifests/source_oof_prediction_index.json",
    "manifests/source_oof_prediction_seal.json",
    "manifests/prelabel_prediction_composite_seal.json",
    "manifests/test_prediction_index.json",
    "manifests/test_prediction_seal.json",
    "manifests/prelabel_feature_seal.json",
    "manifests/source_label_capability_report.json",
    "manifests/model_bank_index.json",
    "manifests/model_bank_seal.json",
    "manifests/frozen_test_prediction_seal.json",
    "manifests/content_index.json",
    "tables/source_case_features.csv",
    "tables/source_regret_responses.csv",
    "tables/model_index.csv",
    "tables/test_case_features.csv",
    "tables/test_candidate_contrasts.csv",
    "tables/test_selection_diagnostics.csv",
    "tables/test_prediction_summary.csv",
    "reports/workstation_preflight.json",
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
_OWNED_CHECKPOINT_PREFIXES = (
    f"{SOURCE_CHECKPOINT_DIRECTORY}/",
    f"{TEST_CHECKPOINT_DIRECTORY}/",
    f"{DEVELOPMENT_CHECKPOINT_DIRECTORY}/",
)


def cleanup_owned_atomic_temps(root: Path) -> None:
    """Remove only interrupted writes owned by this package while locked."""

    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        match = re.fullmatch(r"(?P<member>.+)\.[1-9][0-9]*\.tmp", relative)
        if match and (
            match.group("member") in REQUIRED_FILES
            or match.group("member").startswith(_OWNED_CHECKPOINT_PREFIXES)
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
            member
            for member in observed
            if not member.startswith(_OWNED_CHECKPOINT_PREFIXES)
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
            "Prediction-only closed-world inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def write_content_index(
    root: Path,
    *,
    config_contract_hash: str,
    protocol_contract_hash: str,
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
        "schema_version": "midogpp_disagreement_regret_prediction_only_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
        "whole_test_row_count": 9_928,
        "test_labels_opened": False,
        "test_metrics_present": False,
        "raw_source_labels_persisted": False,
        "prediction_only": True,
        "consumed_test_data": True,
        "fresh_evidence": False,
        "may_authorize_routing": False,
        "may_feed_another_experiment": False,
        "deployable_policy_present": False,
        "prior_stage90_output_consumed": False,
    }
    payload = {**unhashed, "content_hash": canonical_hash(unhashed)}
    persist_or_validate_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path,
    *,
    config_contract_hash: str,
    protocol_contract_hash: str,
) -> Mapping[str, object]:
    observed = read_json(root / "manifests/content_index.json")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    expected_header = {
        "schema_version": "midogpp_disagreement_regret_prediction_only_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "protocol_contract_hash": protocol_contract_hash,
        "member_count": len(CONTENT_INDEX_MEMBERS),
        "closed_world": True,
        "whole_test_row_count": 9_928,
        "test_labels_opened": False,
        "test_metrics_present": False,
        "raw_source_labels_persisted": False,
        "prediction_only": True,
        "consumed_test_data": True,
        "fresh_evidence": False,
        "may_authorize_routing": False,
        "may_feed_another_experiment": False,
        "deployable_policy_present": False,
        "prior_stage90_output_consumed": False,
    }
    members = observed.get("members")
    if (
        observed.get("content_hash") != canonical_hash(unhashed)
        or any(observed.get(key) != value for key, value in expected_header.items())
        or not isinstance(members, list)
        or [row.get("member") for row in members if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Prediction-only content-index header drifted.")
    for row in members:
        if not isinstance(row, Mapping):
            raise ProtocolError("Prediction-only content-index row is malformed.")
        path = root / str(row.get("member"))
        if (
            not path.is_file()
            or row.get("size_bytes") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise ProtocolError("Prediction-only content-index bytes drifted.")
    return observed


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "cleanup_owned_atomic_temps",
    "validate_content_index",
    "write_content_index",
)
