"""Pure row normalization, hash-chain construction, and CLA bundle writing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import prepare_artifact_dirs, stable_hash, write_csv_rows, write_json
from ..protocol import ProtocolError
from .reporting import build_decision_summary, render_decision_report
from .schema import (
    AlignmentArtifactTables,
    CLA_CLAIM_ROLE,
    CLA_CLAIM_SCOPE,
    CLA_CODE_VERSION,
    CLA_EXPERIMENT_ID,
    CLA_EXPERIMENT_NAME,
    CLA_METHOD,
    CLA_PRIOR_METHOD,
    CLA_SCHEMA_VERSION,
    CLA_SELECTION_SOURCE,
    CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION,
    CONTENT_INDEX_SCHEMA_VERSION,
    DECISION_SUMMARY_SCHEMA_VERSION,
    FAIL_CLOSED_CLAIM_VALUES,
    FROZEN_PROTOCOL_SCHEMA_VERSION,
    LEAKAGE_REPORT_SCHEMA_VERSION,
    OUTER_COMPARISON_SCHEMA_VERSION,
    OUTER_PREDICTION_SCHEMA_VERSION,
    OUTER_RESULT_SCHEMA_VERSION,
    PRIMARY_CONTRAST,
    PROTOCOL_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SUMMARY_SCHEMA_VERSION,
    SOLVER_AUDIT_SCHEMA_VERSION,
    SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION,
    SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION,
    TABLE_COLUMNS,
    TABLE_PATHS,
    claim_fields,
    table_bundle_hash,
    table_hashes,
)


def design_payload(config: object) -> dict[str, object]:
    """Convert the frozen config dataclass/mapping to a stable JSON payload."""

    raw: object
    if hasattr(config, "design_payload"):
        candidate = getattr(config, "design_payload")
        raw = candidate() if callable(candidate) else candidate
    elif is_dataclass(config):
        raw = asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = dict(vars(config))
    normalized = _json_safe(raw)
    if not isinstance(normalized, dict):
        raise ProtocolError("CLA config did not normalize to a mapping.")
    # Physical output/input paths are bound by the protocol manifest and workspace
    # provenance; the declarative design identity must remain portable.
    for path_field in (
        "artifact_root",
        "manifest_path",
        "feature_cache_path",
        "config_source_path",
    ):
        normalized.pop(path_field, None)
    return normalized


def frozen_protocol_payload(
    config: object,
    *,
    workspace_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the first, pre-evaluation link of the hash chain."""

    design = design_payload(config)
    return {
        "schema_version": FROZEN_PROTOCOL_SCHEMA_VERSION,
        "experiment_id": CLA_EXPERIMENT_ID,
        "experiment_name": CLA_EXPERIMENT_NAME,
        "mode": CLA_METHOD,
        "code_version": CLA_CODE_VERSION,
        "design": design,
        "design_hash": stable_hash(design),
        "claim_scope": CLA_CLAIM_SCOPE,
        "diagnostic_only": True,
        "non_adoptive": True,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "workspace_binding": (
            None if workspace_binding is None else _json_safe(workspace_binding)
        ),
    }


def write_frozen_protocol_snapshot(
    root: str | Path,
    config: object,
    *,
    workspace_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Persist the immutable design before any outer target is scored."""

    artifact_root = prepare_artifact_dirs(root)
    payload = frozen_protocol_payload(config, workspace_binding=workspace_binding)
    write_json(artifact_root / "manifests/frozen_protocol_snapshot.json", payload)
    return payload


def bind_protocol_hash(
    tables: AlignmentArtifactTables,
    protocol_hash: str,
) -> AlignmentArtifactTables:
    """Return table rows bound to a protocol without changing table identities."""

    return AlignmentArtifactTables.from_mapping(
        {
            name: tuple(dict(row) | {"protocol_hash": str(protocol_hash)} for row in rows)
            for name, rows in tables.as_mapping().items()
        }
    )


def build_protocol_manifest(
    *,
    frozen: Mapping[str, object],
    tables: AlignmentArtifactTables,
    frame: object,
    heldout_centers: Sequence[str],
    gamma_grid: Sequence[float],
    classifier_config_hash: str,
    expected_counts: Mapping[str, int],
    coverage_mode: str,
    experiment_seed: int,
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    """Bind frozen design, current inputs, and the seven table contents."""

    hashes = table_hashes(tables)
    payload: dict[str, object] = {
        "schema_version": PROTOCOL_MANIFEST_SCHEMA_VERSION,
        "experiment_id": CLA_EXPERIMENT_ID,
        "experiment_name": CLA_EXPERIMENT_NAME,
        "mode": CLA_METHOD,
        "code_version": CLA_CODE_VERSION,
        "method": CLA_METHOD,
        "experiment_seed": int(experiment_seed),
        "heldout_centers": [str(value) for value in heldout_centers],
        "eligible_centers": [str(value) for value in getattr(frame, "eligible_centers")],
        "excluded_centers": ["4"],
        "gamma_grid": [float(value) for value in gamma_grid],
        "classifier_config_hash": str(classifier_config_hash),
        "coverage_mode": str(coverage_mode),
        "expected_counts": {key: int(value) for key, value in expected_counts.items()},
        "manifest_path": str(Path(getattr(frame, "manifest_path")).resolve()),
        "feature_cache_path": str(Path(getattr(frame, "feature_cache_path")).resolve()),
        "manifest_hash": str(getattr(frame, "manifest_hash")),
        "feature_cache_hash": str(getattr(frame, "feature_cache_hash")),
        "expected_feature_dim": int(getattr(frame, "expected_feature_dim")),
        "runtime_environment": _json_safe(runtime_environment),
        "design_hash": str(frozen["design_hash"]),
        "workspace_binding": _json_safe(frozen.get("workspace_binding")),
        "table_hashes": hashes,
        "table_bundle_hash": table_bundle_hash(tables),
        "outer_evaluation_roles": ["selected", "gamma0"],
        "outer_all_gamma_scoring": False,
        "outer_oracle_gamma_computed": False,
        "factor_representation": "rectangular_contrast_factor",
        "operator_normalization": "unit_trace",
        "dense_matrix_materialized": False,
        "selection_source": CLA_SELECTION_SOURCE,
        "prior_method": CLA_PRIOR_METHOD,
        "claim_scope": CLA_CLAIM_SCOPE,
        "claim_role": CLA_CLAIM_ROLE,
        "diagnostic_only": True,
        "non_adoptive": True,
        "adoption_eligible": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_fit": False,
        "target_eval_labels_used_for_selection": False,
        "source_inner_labels_used": True,
        "support_labels_used": False,
        "oracle_eligible": False,
        "uses_generated_embeddings": False,
        "uses_cvae_checkpoint": False,
        "uses_encoder_posterior": False,
        "uses_decoder_likelihood": False,
        "uses_prior": False,
        "uses_nelbo": False,
        "uses_latent_representation": False,
        "models_embedding_distribution": False,
        "uses_expert_bank": False,
        "uses_router": False,
        "performs_expert_selection": False,
        "performs_expert_weighting": False,
        "performs_aggregation": False,
        "compatibility_signal": "none",
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def build_leakage_report(
    protocol: Mapping[str, object],
    frame_audits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Create a complete frame-level exclusion/overlap report."""

    overlap_fields = (
        "fit_eval_sample_overlap_count",
        "fit_eval_case_overlap_count",
        "fit_eval_image_path_overlap_count",
        "fit_eval_row_index_overlap_count",
    )
    failing = []
    for row in frame_audits:
        if (
            not _as_bool(row.get("heldout_center_excluded"))
            or (
                row.get("fold_scope") == "source_inner"
                and not _as_bool(row.get("inner_center_excluded"))
            )
            or any(int(row.get(field, 0)) != 0 for field in overlap_fields)
            or any(
                _as_bool(row.get(field))
                for field in (
                    "target_rows_used_for_scaler",
                    "target_rows_used_for_operator",
                    "target_rows_used_for_fit",
                )
            )
        ):
            failing.append(str(row.get("conditional_frame_identity", "")))
    return {
        "schema_version": LEAKAGE_REPORT_SCHEMA_VERSION,
        "status": "PASS" if not failing else "REJECTED",
        "protocol_hash": str(protocol["protocol_hash"]),
        "design_hash": str(protocol["design_hash"]),
        "table_bundle_hash": str(protocol["table_bundle_hash"]),
        "manifest_hash": str(protocol["manifest_hash"]),
        "feature_cache_hash": str(protocol["feature_cache_hash"]),
        "frame_audit_count": len(frame_audits),
        "failing_frame_identities": failing,
        "target_eval_labels_used_for_scoring_only": True,
        "target_eval_labels_used_for_fit": False,
        "target_eval_labels_used_for_selection": False,
        "source_inner_labels_used_for_selection": True,
        "support_labels_used": False,
        "target_rows_used_for_scaler": False,
        "target_rows_used_for_operator": False,
        "target_rows_used_for_classifier_fit": False,
        "sample_overlap_count": sum(
            int(row.get("fit_eval_sample_overlap_count", 0)) for row in frame_audits
        ),
        "case_overlap_count": sum(
            int(row.get("fit_eval_case_overlap_count", 0)) for row in frame_audits
        ),
        "image_path_overlap_count": sum(
            int(row.get("fit_eval_image_path_overlap_count", 0))
            for row in frame_audits
        ),
        "row_index_overlap_count": sum(
            int(row.get("fit_eval_row_index_overlap_count", 0))
            for row in frame_audits
        ),
        "quarantined_center_excluded": True,
        "claim_scope": CLA_CLAIM_SCOPE,
        "diagnostic_only": True,
        "non_adoptive": True,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }


def build_runtime_summary(
    protocol: Mapping[str, object],
    tables: AlignmentArtifactTables,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build the mechanical execution summary."""

    if not math.isfinite(float(elapsed_seconds)) or float(elapsed_seconds) < 0.0:
        raise ProtocolError("CLA elapsed time must be finite and nonnegative.")
    return {
        "schema_version": RUNTIME_SUMMARY_SCHEMA_VERSION,
        "status": "COMPLETE",
        "protocol_hash": str(protocol["protocol_hash"]),
        "design_hash": str(protocol["design_hash"]),
        "table_bundle_hash": str(protocol["table_bundle_hash"]),
        "elapsed_seconds": float(elapsed_seconds),
        "table_row_counts": {
            name: len(rows) for name, rows in tables.as_mapping().items()
        },
        "unique_solver_fit_count": len(tables.solver_audit),
        "runtime_environment": protocol["runtime_environment"],
        "used_for_selection": False,
        "claim_scope": CLA_CLAIM_SCOPE,
        "diagnostic_only": True,
        "non_adoptive": True,
    }


def write_completed_bundle(
    root: str | Path,
    *,
    frozen: Mapping[str, object],
    unbound_tables: AlignmentArtifactTables,
    frame: object,
    heldout_centers: Sequence[str],
    gamma_grid: Sequence[float],
    classifier_config_hash: str,
    expected_counts: Mapping[str, int],
    coverage_mode: str,
    experiment_seed: int,
    elapsed_seconds: float,
    runtime_environment: Mapping[str, object],
    decision_numerical_epsilon: float = 1e-12,
    pass_min_nonnegative_center_deltas: int = 5,
) -> tuple[AlignmentArtifactTables, dict[str, object]]:
    """Write tables and reports in the acyclic design->table->decision chain."""

    artifact_root = prepare_artifact_dirs(root)
    protocol = build_protocol_manifest(
        frozen=frozen,
        tables=unbound_tables,
        frame=frame,
        heldout_centers=heldout_centers,
        gamma_grid=gamma_grid,
        classifier_config_hash=classifier_config_hash,
        expected_counts=expected_counts,
        coverage_mode=coverage_mode,
        experiment_seed=experiment_seed,
        runtime_environment=runtime_environment,
    )
    tables = bind_protocol_hash(unbound_tables, str(protocol["protocol_hash"]))
    for name, rows in tables.as_mapping().items():
        write_csv_rows(
            artifact_root / TABLE_PATHS[name],
            rows,
            TABLE_COLUMNS[name],
        )
    write_json(artifact_root / "manifests/protocol_manifest.json", protocol)
    leakage = build_leakage_report(protocol, tables.conditional_frame_audit)
    write_json(
        artifact_root / "reports/leakage_provenance_report.json",
        leakage,
    )
    decision = build_decision_summary(
        tables.outer_results,
        tables.outer_comparison,
        tables.source_inner_gamma_summary,
        design_hash=str(protocol["design_hash"]),
        table_bundle_hash=str(protocol["table_bundle_hash"]),
        protocol_hash=str(protocol["protocol_hash"]),
        numerical_epsilon=decision_numerical_epsilon,
        pass_min_nonnegative_center_deltas=pass_min_nonnegative_center_deltas,
    )
    if decision.get("schema_version") != DECISION_SUMMARY_SCHEMA_VERSION:
        raise ProtocolError("CLA decision schema drifted during construction.")
    write_json(artifact_root / "reports/decision_summary.json", decision)
    (artifact_root / "reports/decision_report.md").write_text(
        render_decision_report(decision), encoding="utf-8"
    )
    runtime = build_runtime_summary(
        protocol,
        tables,
        elapsed_seconds=elapsed_seconds,
    )
    write_json(artifact_root / "reports/runtime_summary.json", runtime)
    write_content_index(artifact_root)
    return tables, protocol


def write_content_index(root: str | Path) -> dict[str, object]:
    """Write the final file-level hash index, explicitly excluding itself."""

    artifact_root = Path(root)
    index_relative = "manifests/content_index.json"
    entries = []
    for path in sorted(item for item in artifact_root.rglob("*") if item.is_file()):
        relative = path.relative_to(artifact_root).as_posix()
        if relative == index_relative:
            continue
        entries.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": CONTENT_INDEX_SCHEMA_VERSION,
        "hash_algorithm": "sha256",
        "self_excluded": True,
        "indexed_file_count": len(entries),
        "entries": entries,
    }
    payload["content_index_hash"] = stable_hash(payload)
    write_json(artifact_root / index_relative, payload)
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_payload"):
        return _json_safe(value.to_payload())
    return str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def json_cell(value: object) -> str:
    """Encode nested values consistently for canonical CSV cells."""

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))


__all__ = [
    "bind_protocol_hash",
    "build_leakage_report",
    "build_protocol_manifest",
    "build_runtime_summary",
    "design_payload",
    "file_sha256",
    "frozen_protocol_payload",
    "json_cell",
    "write_completed_bundle",
    "write_content_index",
    "write_frozen_protocol_snapshot",
]
