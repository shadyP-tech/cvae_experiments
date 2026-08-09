"""Closed-world inventory for the versioned ensemble-endpoint diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import atomic_json, read_json, relative_files, sha256_file


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/utility_aligned_source_prefixes.npy",
    "arrays/utility_aligned_support_components.npy",
    "arrays/ensemble_endpoint_development_predictions.npz",
    "arrays/ensemble_endpoint_target_predictions.npz",
    "manifests/protocol_manifest.json",
    "manifests/support_partition_lock.json",
    "manifests/case_fold_lock.json",
    "manifests/utility_aligned_source_cache_lock.json",
    "manifests/feature_surface_set.json",
    "manifests/source_inner_support_action_shift_lock.json",
    "manifests/ensemble_endpoint_development_prediction_index.json",
    "manifests/ensemble_endpoint_global_development_prediction_seal.json",
    "manifests/model_set.json",
    "manifests/diagnostic_plan_set.json",
    "manifests/action_library.json",
    "manifests/target_support_action_shift_lock.json",
    "manifests/ensemble_endpoint_target_probe_seal.json",
    "manifests/ensemble_endpoint_target_prediction_cache.json",
    "manifests/ensemble_endpoint_global_target_prediction_seal.json",
    "manifests/content_index.json",
    "tables/support_partitions.csv",
    "tables/case_folds.csv",
    "tables/utility_aligned_source_streams.csv",
    "tables/utility_aligned_support_components.csv",
    "tables/inner_ensemble_features.csv",
    "tables/target_ensemble_features.csv",
    "tables/source_inner_support_action_shifts.csv",
    "tables/target_support_action_shifts.csv",
    "tables/source_inner_ensemble_endpoints.csv",
    "tables/source_inner_seed_diagnostics.csv",
    "tables/model_summary.csv",
    "tables/diagnostic_plans.csv",
    "tables/target_actions.csv",
    "tables/target_prediction_index.csv",
    "tables/target_seed_diagnostics.csv",
    "tables/target_ensemble_metrics.csv",
    "tables/center_contrasts.csv",
    "tables/contrast_inference.csv",
    "tables/oracle_hxe_diagnostics.csv",
    "reports/phase_01_source_cache_and_features_complete.json",
    "reports/phase_02_development_scoring_and_action_lock_complete.json",
    "reports/phase_03_global_target_prediction_seal_complete.json",
    "reports/phase_04_terminal_scoring_complete.json",
    "reports/development_label_access_report.json",
    "reports/target_label_access_report.json",
    "reports/leakage_report.json",
    "reports/scoring_summary.json",
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
    if not allow_incomplete and not allow_pending_validation:
        checkpoint_root = root / "checkpoints"
        if checkpoint_root.is_dir():
            observed.update(
                path.relative_to(root).as_posix()
                for path in checkpoint_root.rglob("*")
                if path.is_file()
            )
    required = set(REQUIRED_FILES)
    permitted_missing = (
        required
        if allow_incomplete
        else ({"reports/validation_report.json"} if allow_pending_validation else set())
    )
    extras = sorted(observed - required)
    missing = sorted(required - observed - permitted_missing)
    if extras or missing:
        raise ProtocolError(
            "Ensemble-endpoint closed-world inventory drifted: "
            f"extras={extras}, missing={missing}."
        )


def write_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    rows: list[dict[str, object]] = []
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        if not path.is_file():
            raise ProtocolError(f"Ensemble-endpoint content member is absent: {member}.")
        rows.append(
            {"member": member, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    unhashed = {
        "schema_version": "midogpp_stage90_ensemble_endpoint_content_index_v1",
        "config_contract_hash": config_contract_hash,
        "members": rows,
        "member_count": len(rows),
        "closed_world": True,
    }
    payload = {**unhashed, "content_hash": stable_hash(unhashed)}
    atomic_json(root / "manifests/content_index.json", payload)
    return payload


def validate_content_index(root: Path, *, config_contract_hash: str) -> Mapping[str, object]:
    observed = read_json(root / "manifests/content_index.json")
    rows = observed.get("members")
    unhashed = {key: value for key, value in observed.items() if key != "content_hash"}
    if (
        observed.get("schema_version") != "midogpp_stage90_ensemble_endpoint_content_index_v1"
        or observed.get("content_hash") != stable_hash(unhashed)
        or observed.get("config_contract_hash") != config_contract_hash
        or observed.get("closed_world") is not True
        or not isinstance(rows, list)
        or [row.get("member") for row in rows if isinstance(row, Mapping)]
        != list(CONTENT_INDEX_MEMBERS)
    ):
        raise ProtocolError("Ensemble-endpoint content-index header drifted.")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("Ensemble-endpoint content-index row is malformed.")
        path = root / str(row.get("member"))
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("size_bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise ProtocolError("Ensemble-endpoint content-index member drifted.")
    return observed


__all__ = (
    "CONTENT_INDEX_MEMBERS",
    "REQUIRED_FILES",
    "assert_closed_world",
    "validate_content_index",
    "write_content_index",
)
