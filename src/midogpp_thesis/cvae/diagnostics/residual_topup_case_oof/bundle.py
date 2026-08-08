"""Closed-world inventory and content lock for the case-OOF artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_write_json, read_json, sha256_file


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/support_partition_lock.json",
    "manifests/crossfit_fold_lock.json",
    "manifests/source_cache_lock.json",
    "manifests/router_plan_lock.json",
    "manifests/global_all_action_prediction_seal.json",
    "manifests/content_index.json",
    "arrays/source_prefix_blocks.npy",
    "arrays/all_action_predictions.npz",
    "tables/support_partitions.csv",
    "tables/crossfit_folds.csv",
    "tables/source_block_index.csv",
    "tables/compatibility_case_energy.csv",
    "tables/proxy_case_ballots.csv",
    "tables/proxy_rank_actions.csv",
    "tables/action_plans.csv",
    "tables/action_assignments.csv",
    "tables/prediction_index.csv",
    "tables/center_seed_metrics.csv",
    "tables/center_ensemble_metrics.csv",
    "tables/primary_contrasts.csv",
    "tables/contrast_inference.csv",
    "tables/oracle_hxe_diagnostics.csv",
    "reports/phase_01_source_cache_complete.json",
    "reports/phase_02_all_predictions_sealed.json",
    "reports/phase_03_terminal_scoring_complete.json",
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


def write_content_index(
    root: Path, *, config_contract_hash: str
) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Case-OOF content member is absent: {member}.")
        rows.append(
            {
                "member": member,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    unhashed: dict[str, object] = {
        "schema_version": "midogpp_residual_topup_case_oof_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
    }
    payload = {**unhashed, "content_hash": stable_hash(unhashed)}
    atomic_write_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(
    root: Path, *, config_contract_hash: str
) -> Mapping[str, object]:
    observed = read_json(root / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {
        key: value for key, value in observed.items() if key != "content_hash"
    }
    if (
        observed.get("content_hash") != stable_hash(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or observed.get("closed_world") is not True
        or not isinstance(rows, list)
        or [row.get("member") for row in rows] != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Case-OOF content index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Case-OOF content row is malformed.")
        path = root / str(row["member"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ProtocolError("Case-OOF content member hash drifted.")
    return observed


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "validate_content_index",
    "write_content_index",
)
