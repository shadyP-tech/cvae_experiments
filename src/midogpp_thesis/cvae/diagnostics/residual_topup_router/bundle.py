"""Closed-world bundle inventory and content-index construction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json, sha256_file


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/support_partition_lock.json",
    "manifests/source_cache_lock.json",
    "manifests/router_plan_lock.json",
    "manifests/global_all_action_prediction_seal.json",
    "manifests/calibration_lock.json",
    "manifests/content_index.json",
    "arrays/source_prefix_blocks.npy",
    "arrays/all_action_predictions.npz",
    "tables/support_partitions.csv",
    "tables/source_block_index.csv",
    "tables/compatibility_case_energy.csv",
    "tables/action_plans.csv",
    "tables/action_assignments.csv",
    "tables/prediction_index.csv",
    "tables/all_action_metrics.csv",
    "tables/development_paired_gains.csv",
    "tables/query_cluster_gains.csv",
    "tables/diagnostic_selections.csv",
    "tables/target_paired_deltas.csv",
    "tables/probability_ensemble_metrics.csv",
    "reports/phase_01_source_cache_complete.json",
    "reports/phase_02_all_actions_sealed.json",
    "reports/phase_03_calibration_complete.json",
    "reports/phase_04_scoring_complete.json",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

CONTENT_INDEX_MEMBERS = tuple(
    member
    for member in REQUIRED_FILES
    if member
    not in {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)


def write_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    rows = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Residual top-up content member is absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed = {
        "schema_version": "midogpp_residual_topup_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "members": rows,
        "member_count": len(rows),
    }
    payload = {**unhashed, "content_hash": stable_hash(unhashed)}
    atomic_write_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    from .artifact_io import read_json

    observed = read_json(root / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    if (
        observed.get("content_hash") != stable_hash(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or not isinstance(rows, list)
        or [row.get("member") for row in rows] != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Residual top-up content index header drifted.")
    for row in rows:
        path = root / str(row["member"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ProtocolError("Residual top-up content member hash drifted.")
    return observed


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "validate_content_index",
    "write_content_index",
)
