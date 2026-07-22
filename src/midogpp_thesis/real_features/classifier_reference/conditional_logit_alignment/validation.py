"""Independent, fail-closed validation of a complete CLA artifact bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..classifiers import standardize_fit_eval
from ..downstream import balanced_accuracy, macro_f1
from ..protocol import ProtocolError
from ..real_feature_frame import RealFeatureFrame, load_midogpp_real_feature_frame
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from .artifacts import (
    build_leakage_report,
    build_runtime_summary,
    file_sha256,
)
from .config import EXPECTED_VERSIONS, GAMMA_GRID, canonical_classifier_spec
from .penalty import build_conditional_penalty
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
    CLA_COMPLETE_REQUIRED_OUTPUTS,
    CLA_RUNNER_REQUIRED_OUTPUTS,
    CLA_SELECTION_SOURCE,
    CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION,
    CONTENT_INDEX_SCHEMA_VERSION,
    DECISION_SUMMARY_SCHEMA_VERSION,
    FAIL_CLOSED_CLAIM_VALUES,
    FROZEN_PROTOCOL_SCHEMA_VERSION,
    LEAKAGE_REPORT_SCHEMA_VERSION,
    OUTER_COMPARISON_SCHEMA_VERSION,
    OUTER_EVALUATION_ROLES,
    OUTER_PREDICTION_SCHEMA_VERSION,
    OUTER_RESULT_SCHEMA_VERSION,
    PRIMARY_CONTRAST,
    PRODUCTION_FRAME_AUDIT_COUNT,
    PRODUCTION_GAMMA_COUNT,
    PRODUCTION_GAMMA_SUMMARY_COUNT,
    PRODUCTION_INNER_SCORE_COUNT,
    PRODUCTION_OUTER_COMPARISON_COUNT,
    PRODUCTION_OUTER_FOLD_COUNT,
    PRODUCTION_OUTER_RESULT_COUNT,
    PROTOCOL_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SUMMARY_SCHEMA_VERSION,
    SOLVER_AUDIT_SCHEMA_VERSION,
    SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION,
    SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION,
    TABLE_COLUMNS,
    TABLE_PATHS,
    table_bundle_hash,
    table_hashes,
)
from .table_rows import conditional_frame_identity
from .workspace_binding import validate_persisted_workspace_binding


@dataclass(frozen=True)
class _ReconstructedFrame:
    fold_scope: str
    heldout_center: str
    inner_center: str
    fit_centers: tuple[str, ...]
    fit_rows: tuple[object, ...]
    eval_rows: tuple[object, ...]
    n_fit: int
    n_eval: int
    fit_row_hash: str
    eval_row_hash: str
    fit_case_hash: str
    eval_case_hash: str
    fit_image_path_hash: str
    eval_image_path_hash: str
    fit_row_index_hash: str
    eval_row_index_hash: str
    training_frame_hash: str
    scaler_state_hash: str
    penalty_operator_hash: str
    operator_rank: int
    maximum_operator_rank: int
    operator_trace: float
    conditional_frame_identity: str


def assert_conditional_logit_alignment_artifacts(
    root: Path,
    already_loaded_frame: RealFeatureFrame | None = None,
) -> None:
    """Validate hashes, rows, nested selection, outer roles, and claims."""

    artifact_root = Path(root)
    missing = [
        relative
        for relative in CLA_RUNNER_REQUIRED_OUTPUTS
        if not (artifact_root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(f"CLA artifact missing required outputs: {missing}")

    frozen = _read_json(artifact_root / "manifests/frozen_protocol_snapshot.json")
    protocol = _read_json(artifact_root / "manifests/protocol_manifest.json")
    leakage = _read_json(artifact_root / "reports/leakage_provenance_report.json")
    decision = _read_json(artifact_root / "reports/decision_summary.json")
    runtime = _read_json(artifact_root / "reports/runtime_summary.json")
    content_index = _read_json(artifact_root / "manifests/content_index.json")
    tables = AlignmentArtifactTables.from_mapping(
        {
            name: _read_csv(artifact_root / relative, TABLE_COLUMNS[name])
            for name, relative in TABLE_PATHS.items()
        }
    )

    coverage_mode = str(protocol.get("coverage_mode", ""))
    _validate_required_file_set(artifact_root, coverage_mode)
    _validate_hash_chain(frozen, protocol, decision, tables)
    _validate_protocol(protocol, frozen)
    if coverage_mode == "complete":
        validate_persisted_workspace_binding(
            artifact_root,
            frozen.get("workspace_binding"),
            protocol.get("workspace_binding"),
        )
    frame = already_loaded_frame or _load_bound_frame(protocol)
    _validate_bound_frame(frame, protocol)
    reconstructed = _reconstruct_frames(frame, protocol)
    selected_gammas = _validate_inner_tables(tables, protocol, reconstructed)
    _validate_frame_audits(tables.conditional_frame_audit, reconstructed, protocol)
    expected_solver_ids = _validate_outer_tables(
        tables,
        protocol,
        reconstructed,
        selected_gammas,
    )
    _validate_solver_audits(
        tables.solver_audit,
        reconstructed,
        protocol,
        selected_gammas,
        expected_solver_ids,
    )
    _validate_cardinalities(tables, protocol, reconstructed, selected_gammas)
    _validate_reports(
        artifact_root,
        leakage,
        decision,
        runtime,
        protocol,
        tables,
    )
    _validate_content_index(artifact_root, content_index, coverage_mode)


def _validate_hash_chain(
    frozen: Mapping[str, object],
    protocol: Mapping[str, object],
    decision: Mapping[str, object],
    tables: AlignmentArtifactTables,
) -> None:
    design = frozen.get("design")
    if not isinstance(design, Mapping):
        raise ProtocolError("CLA frozen snapshot lacks a design mapping.")
    if frozen.get("design_hash") != stable_hash(design):
        raise ProtocolError("CLA frozen design hash mismatch.")
    if protocol.get("design_hash") != frozen.get("design_hash"):
        raise ProtocolError("CLA protocol is not bound to the frozen design.")
    expected_table_hashes = table_hashes(tables)
    expected_bundle_hash = table_bundle_hash(tables)
    if (
        protocol.get("table_hashes") != expected_table_hashes
        or protocol.get("table_bundle_hash") != expected_bundle_hash
    ):
        raise ProtocolError("CLA canonical table bundle hash mismatch.")
    protocol_without_hash = dict(protocol)
    persisted_protocol_hash = protocol_without_hash.pop("protocol_hash", None)
    expected_protocol_hash = stable_hash(protocol_without_hash)
    if persisted_protocol_hash != expected_protocol_hash:
        raise ProtocolError("CLA protocol manifest hash mismatch.")
    for name, rows in tables.as_mapping().items():
        if any(row.get("protocol_hash") != expected_protocol_hash for row in rows):
            raise ProtocolError(f"CLA {name} row is not bound to protocol_hash.")
    if (
        decision.get("design_hash") != frozen.get("design_hash")
        or decision.get("table_bundle_hash") != expected_bundle_hash
        or decision.get("protocol_hash") != expected_protocol_hash
    ):
        raise ProtocolError("CLA decision is not bound to design/tables/protocol.")
    decision_without_hash = dict(decision)
    persisted_decision_hash = decision_without_hash.pop("decision_hash", None)
    if persisted_decision_hash != stable_hash(decision_without_hash):
        raise ProtocolError("CLA decision hash mismatch.")


def _validate_protocol(
    protocol: Mapping[str, object], frozen: Mapping[str, object]
) -> None:
    expected = {
        "schema_version": PROTOCOL_MANIFEST_SCHEMA_VERSION,
        "experiment_id": CLA_EXPERIMENT_ID,
        "experiment_name": CLA_EXPERIMENT_NAME,
        "mode": CLA_METHOD,
        "code_version": CLA_CODE_VERSION,
        "method": CLA_METHOD,
        "outer_evaluation_roles": list(OUTER_EVALUATION_ROLES),
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
    for field, value in expected.items():
        if protocol.get(field) != value:
            raise ProtocolError(f"CLA protocol field {field} drifted.")
    if (
        protocol.get("classifier_config_hash")
        != canonical_classifier_spec().config_hash
        or protocol.get("experiment_seed") != 42
        or protocol.get("excluded_centers") != ["4"]
        or protocol.get("coverage_mode") not in {"complete", "partial_test"}
    ):
        raise ProtocolError("CLA canonical classifier/seed/coverage locks drifted.")
    coverage_mode = str(protocol.get("coverage_mode"))
    if coverage_mode == "complete":
        if not isinstance(frozen.get("workspace_binding"), Mapping) or not isinstance(
            protocol.get("workspace_binding"), Mapping
        ):
            raise ProtocolError("Complete CLA protocol lacks its workspace binding.")
    elif coverage_mode == "partial_test":
        if frozen.get("workspace_binding") is not None or protocol.get(
            "workspace_binding"
        ) is not None:
            raise ProtocolError("Partial-test CLA bundle must not claim a workspace binding.")
    else:
        raise ProtocolError("CLA coverage mode is invalid.")
    if frozen.get("schema_version") != FROZEN_PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError("CLA frozen snapshot schema drifted.")
    for field, value in {
        "claim_scope": CLA_CLAIM_SCOPE,
        "diagnostic_only": True,
        "non_adoptive": True,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
    }.items():
        if frozen.get(field) != value:
            raise ProtocolError(f"CLA frozen claim boundary {field} drifted.")
    design = _mapping(frozen.get("design"), "frozen design")
    classifier = _mapping(design.get("classifier_spec"), "frozen classifier_spec")
    optimizer = _mapping(design.get("optimizer"), "frozen optimizer")
    if (
        design.get("name") != CLA_EXPERIMENT_NAME
        or design.get("mode") != CLA_METHOD
        or design.get("code_version") != CLA_CODE_VERSION
        or int(design.get("experiment_seed", -1)) != 42
        or tuple(float(value) for value in _list(design.get("gamma_grid"), "frozen gamma_grid"))
        != GAMMA_GRID
        or float(classifier.get("C", math.nan)) != 0.01
        or classifier.get("penalty") != "l2"
        or classifier.get("solver") != "lbfgs"
        or int(classifier.get("max_iter", -1)) != 5000
        or classifier.get("class_weight") is not None
        or int(classifier.get("random_state", -1)) != 23
        or classifier.get("threshold_policy") != "predict"
        or optimizer.get("require_single_thread") is not True
        or int(optimizer.get("max_iter", -1)) != 5000
        or dict(_mapping(design.get("expected_versions"), "frozen expected_versions"))
        != dict(EXPECTED_VERSIONS)
    ):
        raise ProtocolError("CLA frozen scientific/numerical design locks drifted.")
    gammas = tuple(float(value) for value in _list(protocol.get("gamma_grid"), "gamma_grid"))
    if gammas != GAMMA_GRID:
        raise ProtocolError("CLA protocol gamma grid drifted.")
    runtime = _mapping(protocol.get("runtime_environment"), "runtime_environment")
    expected_versions = _mapping(runtime.get("expected_versions"), "expected_versions")
    observed_versions = _mapping(runtime.get("observed_versions"), "observed_versions")
    threads = _mapping(runtime.get("thread_environment"), "thread_environment")
    if (
        dict(expected_versions) != dict(EXPECTED_VERSIONS)
        or dict(observed_versions) != dict(EXPECTED_VERSIONS)
        or runtime.get("versions_match") is not True
        or runtime.get("require_single_thread") is not True
        or runtime.get("single_thread_match") is not True
        or set(threads)
        != {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        }
        or any(value != "1" for value in threads.values())
    ):
        raise ProtocolError("CLA recorded runtime/version/thread audit is invalid.")


def _load_bound_frame(protocol: Mapping[str, object]) -> RealFeatureFrame:
    manifest = Path(str(protocol.get("manifest_path", "")))
    cache = Path(str(protocol.get("feature_cache_path", "")))
    if not manifest.is_file() or not cache.is_file():
        raise ProtocolError("CLA bound manifest/feature cache is unavailable for validation.")
    return load_midogpp_real_feature_frame(
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_dim=int(protocol["expected_feature_dim"]),
    )


def _validate_bound_frame(
    frame: RealFeatureFrame, protocol: Mapping[str, object]
) -> None:
    if (
        frame.manifest_hash != protocol.get("manifest_hash")
        or frame.feature_cache_hash != protocol.get("feature_cache_hash")
        or int(frame.expected_feature_dim) != int(protocol.get("expected_feature_dim", -1))
        or list(frame.eligible_centers) != protocol.get("eligible_centers")
    ):
        raise ProtocolError("CLA current feature frame differs from bound protocol inputs.")
    heldouts = tuple(str(value) for value in _list(protocol.get("heldout_centers"), "heldout_centers"))
    if not heldouts or len(set(heldouts)) != len(heldouts):
        raise ProtocolError("CLA protocol held-out centers are empty or duplicate.")
    if any(value not in frame.eligible_centers for value in heldouts):
        raise ProtocolError("CLA protocol held-out center is absent from bound inputs.")
    if protocol.get("coverage_mode") == "complete" and (
        heldouts != MIDOGPP_ELIGIBLE_CENTERS
        or frame.eligible_centers != MIDOGPP_ELIGIBLE_CENTERS
        or int(protocol.get("expected_feature_dim", -1)) != 2560
    ):
        raise ProtocolError("Production CLA coverage is not the exact nine-center set.")


def _reconstruct_frames(
    frame: RealFeatureFrame,
    protocol: Mapping[str, object],
) -> dict[tuple[str, str, str], _ReconstructedFrame]:
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])  # type: ignore[arg-type]
    centers = tuple(str(value) for value in frame.eligible_centers)
    reconstructed: dict[tuple[str, str, str], _ReconstructedFrame] = {}
    for heldout in heldouts:
        for inner in tuple(value for value in centers if value != heldout):
            key = ("source_inner", heldout, inner)
            reconstructed[key] = _reconstruct_frame(
                frame,
                fold_scope="source_inner",
                heldout=heldout,
                inner=inner,
            )
        key = ("outer", heldout, "")
        reconstructed[key] = _reconstruct_frame(
            frame,
            fold_scope="outer",
            heldout=heldout,
            inner="",
        )
    return reconstructed


def _reconstruct_frame(
    frame: RealFeatureFrame,
    *,
    fold_scope: str,
    heldout: str,
    inner: str,
) -> _ReconstructedFrame:
    import numpy as np

    excluded = {heldout}
    eval_center = heldout
    if fold_scope == "source_inner":
        excluded.add(inner)
        eval_center = inner
    fit_centers = tuple(center for center in frame.eligible_centers if center not in excluded)
    fit_indices = tuple(
        index for index, row in enumerate(frame.rows) if row.center in set(fit_centers)
    )
    eval_indices = tuple(
        index for index, row in enumerate(frame.rows) if row.center == eval_center
    )
    fit_rows = tuple(frame.rows[index] for index in fit_indices)
    eval_rows = tuple(frame.rows[index] for index in eval_indices)
    _assert_no_identity_overlap(fit_rows, eval_rows)
    embeddings = frame.embeddings
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    array = np.asarray(embeddings, dtype=float)
    standardized = standardize_fit_eval(array[list(fit_indices)], array[list(eval_indices)])
    operator = build_conditional_penalty(
        standardized.fit_embeddings,
        tuple(int(row.label) for row in fit_rows),
        tuple(str(row.center) for row in fit_rows),
    )
    fit_row_hash = _identity_hash(row.sample_id for row in fit_rows)
    eval_row_hash = _identity_hash(row.sample_id for row in eval_rows)
    identities = [
        {
            "row_index": int(row.row_index),
            "sample_id": str(row.sample_id),
            "case_id": str(row.case_id),
            "image_path": str(getattr(row, "image_path", "")),
            "center": str(row.center),
            "label": int(row.label),
        }
        for row in fit_rows
    ]
    training_frame_hash = stable_hash(
        {
            "outer_target_center": heldout,
            "inner_pseudo_target_center": inner or None,
            "eval_center": eval_center,
            "fit_centers": list(fit_centers),
            "fit_row_hash": fit_row_hash,
            "fit_identities": identities,
        }
    )
    frame_identity = stable_hash(
        {
            "training_frame_hash": training_frame_hash,
            "fit_row_hash": fit_row_hash,
            "eval_row_hash": eval_row_hash,
            "scaler_state_hash": standardized.scaler_state_hash,
            "penalty_operator_hash": operator.factor_hash,
        }
    )
    return _ReconstructedFrame(
        fold_scope=fold_scope,
        heldout_center=heldout,
        inner_center=inner,
        fit_centers=fit_centers,
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        n_fit=len(fit_rows),
        n_eval=len(eval_rows),
        fit_row_hash=fit_row_hash,
        eval_row_hash=eval_row_hash,
        fit_case_hash=_identity_hash(row.case_id for row in fit_rows),
        eval_case_hash=_identity_hash(row.case_id for row in eval_rows),
        fit_image_path_hash=_identity_hash(getattr(row, "image_path", "") for row in fit_rows),
        eval_image_path_hash=_identity_hash(getattr(row, "image_path", "") for row in eval_rows),
        fit_row_index_hash=_identity_hash(row.row_index for row in fit_rows),
        eval_row_index_hash=_identity_hash(row.row_index for row in eval_rows),
        training_frame_hash=training_frame_hash,
        scaler_state_hash=standardized.scaler_state_hash,
        penalty_operator_hash=operator.factor_hash,
        operator_rank=int(operator.rank),
        maximum_operator_rank=int(operator.maximum_rank),
        operator_trace=float(operator.trace),
        conditional_frame_identity=frame_identity,
    )


def _validate_inner_tables(
    tables: AlignmentArtifactTables,
    protocol: Mapping[str, object],
    reconstructed: Mapping[tuple[str, str, str], _ReconstructedFrame],
) -> dict[str, float]:
    gammas = tuple(float(value) for value in protocol["gamma_grid"])  # type: ignore[arg-type]
    expected_keys = {
        (heldout, inner, gamma)
        for scope, heldout, inner in reconstructed
        if scope == "source_inner"
        for gamma in gammas
    }
    by_key = _unique_rows(
        tables.source_inner_fold_scores,
        lambda row: (
            str(row.get("heldout_center", "")),
            str(row.get("inner_center", "")),
            _float(row.get("gamma"), "gamma"),
        ),
        "source-inner fold score",
    )
    if set(by_key) != expected_keys:
        raise ProtocolError("CLA source-inner fold-score matrix is incomplete or expanded.")

    classifier_hash = str(protocol["classifier_config_hash"])
    for (heldout, inner, gamma), row in by_key.items():
        expected_frame = reconstructed[("source_inner", heldout, inner)]
        _validate_common_fit_fields(row, expected_frame, classifier_hash, gamma)
        _validate_claim_row(row, "source_inner_fold_score")
        expected = {
            "schema_version": SOURCE_INNER_FOLD_SCORE_SCHEMA_VERSION,
            "method": CLA_METHOD,
            "heldout_center": heldout,
            "inner_center": inner,
            "fit_centers": _json_cell(expected_frame.fit_centers),
            "n_fit": str(expected_frame.n_fit),
            "n_eval": str(expected_frame.n_eval),
            "eval_row_hash": expected_frame.eval_row_hash,
            "training_frame_hash": expected_frame.training_frame_hash,
            "converged": "true",
            "status": "ok",
        }
        _assert_fields(row, expected, "source-inner score")
        for field in ("inner_bacc", "inner_macro_f1"):
            metric = _float(row.get(field), field)
            if not 0.0 <= metric <= 1.0:
                raise ProtocolError(f"CLA {field} is outside [0, 1].")
        _validate_n_iter(row.get("n_iter"), gamma=gamma)

    summary_by_key = _unique_rows(
        tables.source_inner_gamma_summary,
        lambda row: (
            str(row.get("heldout_center", "")),
            _float(row.get("gamma"), "gamma"),
        ),
        "source-inner gamma summary",
    )
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])  # type: ignore[arg-type]
    expected_summary_keys = {(heldout, gamma) for heldout in heldouts for gamma in gammas}
    if set(summary_by_key) != expected_summary_keys:
        raise ProtocolError("CLA source-inner gamma summaries are incomplete or expanded.")

    selected_gammas: dict[str, float] = {}
    for heldout in heldouts:
        inner_centers = tuple(
            inner
            for scope, outer, inner in reconstructed
            if scope == "source_inner" and outer == heldout
        )
        means: dict[float, float] = {}
        macro_means: dict[float, float] = {}
        minima: dict[float, float] = {}
        for gamma in gammas:
            score_rows = [by_key[(heldout, inner, gamma)] for inner in inner_centers]
            baccs = [_float(row["inner_bacc"], "inner_bacc") for row in score_rows]
            f1s = [_float(row["inner_macro_f1"], "inner_macro_f1") for row in score_rows]
            means[gamma] = sum(baccs) / float(len(baccs))
            macro_means[gamma] = sum(f1s) / float(len(f1s))
            minima[gamma] = min(baccs)
        best = max(means.values())
        tied = [gamma for gamma in gammas if abs(means[gamma] - best) <= 1e-12]
        selected = min(tied)
        selected_gammas[heldout] = selected
        ranks = _summary_ranks(means)
        selected_count = 0
        for gamma in gammas:
            row = summary_by_key[(heldout, gamma)]
            is_selected = row.get("selected") == "true"
            selected_count += int(is_selected)
            _validate_claim_row(row, "source_inner_gamma_summary")
            expected = {
                "schema_version": SOURCE_INNER_GAMMA_SUMMARY_SCHEMA_VERSION,
                "method": CLA_METHOD,
                "heldout_center": heldout,
                "n_inner_centers": str(len(inner_centers)),
                "selected": "true" if gamma == selected else "false",
                "selection_rank": str(ranks[gamma]),
                "tie_atol": "1e-12",
                "tie_rtol": "0.0",
                "status": "PASS",
            }
            _assert_fields(row, expected, "gamma summary")
            _assert_close(row.get("mean_inner_bacc"), means[gamma], "mean_inner_bacc")
            _assert_close(
                row.get("mean_inner_macro_f1"),
                macro_means[gamma],
                "mean_inner_macro_f1",
            )
            _assert_close(
                row.get("minimum_inner_bacc"), minima[gamma], "minimum_inner_bacc"
            )
        if selected_count != 1:
            raise ProtocolError("CLA must select exactly one source-inner gamma per H.")
    return selected_gammas


def _validate_frame_audits(
    rows: Sequence[Mapping[str, object]],
    reconstructed: Mapping[tuple[str, str, str], _ReconstructedFrame],
    protocol: Mapping[str, object],
) -> None:
    by_key = _unique_rows(
        rows,
        lambda row: (
            str(row.get("fold_scope", "")),
            str(row.get("heldout_center", "")),
            str(row.get("inner_center", "")),
        ),
        "conditional frame audit",
    )
    if set(by_key) != set(reconstructed):
        raise ProtocolError("CLA conditional-frame audit coverage is incomplete.")
    for key, expected_frame in reconstructed.items():
        row = by_key[key]
        _validate_claim_row(row, "conditional_frame_audit")
        expected = {
            "schema_version": CONDITIONAL_FRAME_AUDIT_SCHEMA_VERSION,
            "method": CLA_METHOD,
            "fold_scope": expected_frame.fold_scope,
            "heldout_center": expected_frame.heldout_center,
            "inner_center": expected_frame.inner_center,
            "conditional_frame_identity": expected_frame.conditional_frame_identity,
            "fit_centers": _json_cell(expected_frame.fit_centers),
            "n_fit": str(expected_frame.n_fit),
            "n_domains": str(len(expected_frame.fit_centers)),
            "fit_row_hash": expected_frame.fit_row_hash,
            "eval_row_hash": expected_frame.eval_row_hash,
            "fit_case_hash": expected_frame.fit_case_hash,
            "eval_case_hash": expected_frame.eval_case_hash,
            "fit_image_path_hash": expected_frame.fit_image_path_hash,
            "eval_image_path_hash": expected_frame.eval_image_path_hash,
            "fit_row_index_hash": expected_frame.fit_row_index_hash,
            "eval_row_index_hash": expected_frame.eval_row_index_hash,
            "training_frame_hash": expected_frame.training_frame_hash,
            "scaler_state_hash": expected_frame.scaler_state_hash,
            "penalty_operator_hash": expected_frame.penalty_operator_hash,
            "operator_rank": str(expected_frame.operator_rank),
            "maximum_operator_rank": str(expected_frame.maximum_operator_rank),
            "required_cell_count": str(2 * len(expected_frame.fit_centers)),
            "observed_cell_count": str(2 * len(expected_frame.fit_centers)),
            "missing_cell_count": "0",
            "factor_representation": "rectangular_contrast_factor",
            "normalization": "unit_trace",
            "dense_matrix_materialized": "false",
            "heldout_center_excluded": "true",
            "inner_center_excluded": (
                "true" if expected_frame.fold_scope == "source_inner" else "not_applicable"
            ),
            "fit_eval_sample_overlap_count": "0",
            "fit_eval_case_overlap_count": "0",
            "fit_eval_image_path_overlap_count": "0",
            "fit_eval_row_index_overlap_count": "0",
            "target_rows_used_for_scaler": "false",
            "target_rows_used_for_operator": "false",
            "target_rows_used_for_fit": "false",
            "status": "PASS",
        }
        _assert_fields(row, expected, "conditional-frame audit")
        _assert_close(row.get("operator_trace"), expected_frame.operator_trace, "operator_trace")
        if not (
            0 < int(row["operator_rank"])
            <= int(row["maximum_operator_rank"])
            <= min(2560 if protocol.get("coverage_mode") == "complete" else int(protocol["expected_feature_dim"]), 2 * (len(expected_frame.fit_centers) - 1))
        ):
            raise ProtocolError("CLA conditional operator rank bound is invalid.")
        if not math.isclose(
            _float(row.get("operator_trace"), "operator_trace"),
            1.0,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise ProtocolError("CLA conditional operator is not unit trace.")


def _validate_outer_tables(
    tables: AlignmentArtifactTables,
    protocol: Mapping[str, object],
    reconstructed: Mapping[tuple[str, str, str], _ReconstructedFrame],
    selected_gammas: Mapping[str, float],
) -> set[str]:
    results = _unique_rows(
        tables.outer_results,
        lambda row: (
            str(row.get("heldout_center", "")),
            str(row.get("evaluation_role", "")),
        ),
        "outer result",
    )
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])  # type: ignore[arg-type]
    expected_result_keys = {
        (heldout, role) for heldout in heldouts for role in OUTER_EVALUATION_ROLES
    }
    if set(results) != expected_result_keys:
        raise ProtocolError(
            "CLA outer results must contain selected/gamma0 roles only; all-gamma rows are forbidden."
        )
    predictions_by_key: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in tables.outer_predictions:
        key = (str(row.get("heldout_center", "")), str(row.get("evaluation_role", "")))
        if key not in expected_result_keys:
            raise ProtocolError("CLA outer predictions contain an all-gamma/unknown role.")
        predictions_by_key.setdefault(key, []).append(row)
    if set(predictions_by_key) != expected_result_keys:
        raise ProtocolError("CLA outer prediction role coverage is incomplete.")
    comparisons = _unique_rows(
        tables.outer_comparison,
        lambda row: str(row.get("heldout_center", "")),
        "outer comparison",
    )
    if set(comparisons) != set(heldouts):
        raise ProtocolError("CLA outer paired comparison coverage is incomplete.")

    expected_solver_ids: set[str] = set()
    classifier_hash = str(protocol["classifier_config_hash"])
    for heldout in heldouts:
        expected_frame = reconstructed[("outer", heldout, "")]
        selected_gamma = float(selected_gammas[heldout])
        role_metrics: dict[str, tuple[float, float]] = {}
        role_fit_ids: dict[str, str] = {}
        role_prediction_payloads: dict[str, list[tuple[str, int, float]]] = {}
        for role in OUTER_EVALUATION_ROLES:
            gamma = selected_gamma if role == "selected" else 0.0
            row = results[(heldout, role)]
            _validate_common_fit_fields(row, expected_frame, classifier_hash, gamma)
            _validate_claim_row(row, "outer_diagnostic_result")
            shared = selected_gamma == 0.0
            expected = {
                "schema_version": OUTER_RESULT_SCHEMA_VERSION,
                "method": CLA_METHOD,
                "heldout_center": heldout,
                "evaluation_role": role,
                "selected_gamma": str(selected_gamma),
                "shared_fit": "true" if shared else "false",
                "fit_centers": _json_cell(expected_frame.fit_centers),
                "n_fit": str(expected_frame.n_fit),
                "n_eval": str(expected_frame.n_eval),
                "eval_row_hash": expected_frame.eval_row_hash,
                "training_frame_hash": expected_frame.training_frame_hash,
                "converged": "true",
                "status": "ok",
                "manifest_hash": str(protocol["manifest_hash"]),
                "feature_cache_hash": str(protocol["feature_cache_hash"]),
            }
            _assert_fields(row, expected, "outer result")
            _validate_n_iter(row.get("n_iter"), gamma=gamma)
            prediction_rows = predictions_by_key[(heldout, role)]
            if len(prediction_rows) != expected_frame.n_eval:
                raise ProtocolError("CLA outer prediction count differs from target rows.")
            truths: list[int] = []
            predictions: list[int] = []
            semantic_payload: list[tuple[str, int, float]] = []
            for persisted, expected_eval in zip(prediction_rows, expected_frame.eval_rows):
                _validate_claim_row(persisted, "outer_diagnostic_prediction")
                _validate_common_fit_fields(
                    persisted, expected_frame, classifier_hash, gamma
                )
                _assert_fields(
                    persisted,
                    {
                        "schema_version": OUTER_PREDICTION_SCHEMA_VERSION,
                        "method": CLA_METHOD,
                        "heldout_center": heldout,
                        "evaluation_role": role,
                        "selected_gamma": str(selected_gamma),
                        "shared_fit": "true" if shared else "false",
                        "sample_id": str(expected_eval.sample_id),
                        "case_id": str(expected_eval.case_id),
                        "center": heldout,
                        "y_true": str(int(expected_eval.label)),
                        "eval_row_hash": expected_frame.eval_row_hash,
                        "training_frame_hash": expected_frame.training_frame_hash,
                    },
                    "outer prediction",
                )
                truth = int(str(persisted.get("y_true")))
                predicted = int(str(persisted.get("y_pred")))
                probability = _float(persisted.get("prob_pos"), "prob_pos")
                if truth not in {0, 1} or predicted not in {0, 1} or not 0.0 <= probability <= 1.0:
                    raise ProtocolError("CLA outer prediction values are invalid.")
                if predicted != int(probability > 0.5):
                    raise ProtocolError("CLA outer prediction disagrees with predict policy.")
                truths.append(truth)
                predictions.append(predicted)
                semantic_payload.append((str(expected_eval.sample_id), predicted, probability))
            bacc = float(balanced_accuracy(truths, predictions))
            f1 = float(macro_f1(truths, predictions))
            _assert_close(row.get("heldout_bacc"), bacc, "heldout_bacc")
            _assert_close(row.get("heldout_macro_f1"), f1, "heldout_macro_f1")
            role_metrics[role] = (bacc, f1)
            role_fit_ids[role] = str(row["fit_identity"])
            role_prediction_payloads[role] = semantic_payload
            expected_solver_ids.add(str(row["fit_identity"]))

        shared = selected_gamma == 0.0
        if shared:
            if (
                role_fit_ids["selected"] != role_fit_ids["gamma0"]
                or role_prediction_payloads["selected"]
                != role_prediction_payloads["gamma0"]
            ):
                raise ProtocolError("CLA selected=0 semantic roles do not share one physical fit.")
        elif role_fit_ids["selected"] == role_fit_ids["gamma0"]:
            raise ProtocolError("CLA nonzero selected gamma shares gamma-zero fit identity.")

        comparison = comparisons[heldout]
        _validate_claim_row(comparison, "outer_paired_comparison")
        selected_metrics = role_metrics["selected"]
        gamma0_metrics = role_metrics["gamma0"]
        _assert_fields(
            comparison,
            {
                "schema_version": OUTER_COMPARISON_SCHEMA_VERSION,
                "method": CLA_METHOD,
                "heldout_center": heldout,
                "contrast_id": PRIMARY_CONTRAST,
                "selected_gamma": str(selected_gamma),
                "selected_fit_identity": role_fit_ids["selected"],
                "gamma0_fit_identity": role_fit_ids["gamma0"],
                "shared_fit": "true" if shared else "false",
                "eval_row_hash": expected_frame.eval_row_hash,
                "status": "PASS",
            },
            "outer comparison",
        )
        metrics = {
            "selected_bacc": selected_metrics[0],
            "gamma0_bacc": gamma0_metrics[0],
            "delta_bacc": selected_metrics[0] - gamma0_metrics[0],
            "selected_macro_f1": selected_metrics[1],
            "gamma0_macro_f1": gamma0_metrics[1],
            "delta_macro_f1": selected_metrics[1] - gamma0_metrics[1],
        }
        for field, value in metrics.items():
            _assert_close(comparison.get(field), value, field)
        if shared and any(
            abs(_float(comparison.get(field), field)) > 1e-15
            for field in ("delta_bacc", "delta_macro_f1")
        ):
            raise ProtocolError("CLA selected=0 shared fit has a nonzero paired delta.")

    for scope, heldout, inner in reconstructed:
        if scope != "source_inner":
            continue
        expected_frame = reconstructed[(scope, heldout, inner)]
        for gamma in protocol["gamma_grid"]:  # type: ignore[assignment]
            expected_solver_ids.add(
                _fit_identity(expected_frame, classifier_hash, float(gamma))
            )
    return expected_solver_ids


def _validate_solver_audits(
    rows: Sequence[Mapping[str, object]],
    reconstructed: Mapping[tuple[str, str, str], _ReconstructedFrame],
    protocol: Mapping[str, object],
    selected_gammas: Mapping[str, float],
    expected_fit_ids: set[str],
) -> None:
    by_fit_id = _unique_rows(
        rows,
        lambda row: str(row.get("fit_identity", "")),
        "solver audit",
    )
    if set(by_fit_id) != expected_fit_ids:
        raise ProtocolError("CLA solver audit is not one row per unique physical fit.")
    classifier_hash = str(protocol["classifier_config_hash"])
    for fit_identity, row in by_fit_id.items():
        scope = str(row.get("fold_scope", ""))
        heldout = str(row.get("heldout_center", ""))
        inner = str(row.get("inner_center", ""))
        frame_key = (scope, heldout, inner)
        expected_frame = reconstructed.get(frame_key)
        if expected_frame is None:
            raise ProtocolError("CLA solver audit references an unknown fit frame.")
        gamma = _float(row.get("gamma"), "gamma")
        if gamma not in GAMMA_GRID:
            raise ProtocolError("CLA solver audit gamma is outside the frozen grid.")
        if scope == "outer" and gamma not in {0.0, selected_gammas[heldout]}:
            raise ProtocolError("CLA solver audit exposes forbidden outer all-gamma fitting.")
        _validate_common_fit_fields(row, expected_frame, classifier_hash, gamma)
        _validate_claim_row(row, "solver_fit_audit")
        expected_backend = "sklearn_lbfgs" if gamma == 0.0 else "scipy_lbfgsb"
        expected = {
            "schema_version": SOLVER_AUDIT_SCHEMA_VERSION,
            "method": CLA_METHOD,
            "fold_scope": scope,
            "heldout_center": heldout,
            "inner_center": inner,
            "fit_identity": fit_identity,
            "backend": expected_backend,
            "warm_start": (
                "not_applicable_shared_sklearn"
                if gamma == 0.0
                else "pooled_gamma0_solution"
            ),
            "converged": "true",
            "l2_normalization": "1/(2*C*N_fit)",
            "intercept_penalized": "false",
            "gamma_zero_shared_sklearn_path": "true" if gamma == 0.0 else "false",
            "status": "PASS",
        }
        _assert_fields(row, expected, "solver audit")
        objective = _float(row.get("objective_value"), "objective_value")
        gradient = _float(row.get("gradient_inf_norm"), "gradient_inf_norm")
        if objective < 0.0 or gradient < 0.0:
            raise ProtocolError("CLA solver objective/gradient audit is invalid.")
        if gamma > 0.0 and gradient > 1.0e-4:
            raise ProtocolError("CLA positive-gamma solver gradient exceeds frozen tolerance.")
        _validate_n_iter(row.get("n_iter"), gamma=gamma)

def _validate_cardinalities(
    tables: AlignmentArtifactTables,
    protocol: Mapping[str, object],
    reconstructed: Mapping[tuple[str, str, str], _ReconstructedFrame],
    selected_gammas: Mapping[str, float],
) -> None:
    n_outer = len(protocol["heldout_centers"])  # type: ignore[arg-type]
    n_inner = sum(scope == "source_inner" for scope, _, _ in reconstructed)
    n_gamma = len(protocol["gamma_grid"])  # type: ignore[arg-type]
    expected_solver_count = n_inner * n_gamma + sum(
        1 if float(gamma) == 0.0 else 2 for gamma in selected_gammas.values()
    )
    expected_predictions = 2 * sum(
        frame.n_eval
        for key, frame in reconstructed.items()
        if key[0] == "outer"
    )
    expected = {
        "source_inner_fold_scores": n_inner * n_gamma,
        "source_inner_gamma_summary": n_outer * n_gamma,
        "outer_results": n_outer * 2,
        "outer_predictions": expected_predictions,
        "conditional_frame_audit": n_inner + n_outer,
        "solver_audit": expected_solver_count,
        "outer_comparison": n_outer,
    }
    actual = {name: len(rows) for name, rows in tables.as_mapping().items()}
    if actual != expected:
        raise ProtocolError(
            f"CLA logical table cardinalities are invalid: expected={expected}, actual={actual}."
        )
    persisted = _mapping(protocol.get("expected_counts"), "expected_counts")
    if dict(persisted) != expected:
        raise ProtocolError("CLA protocol expected-count contract does not reconstruct.")
    if protocol.get("coverage_mode") == "complete":
        production = {
            "source_inner_fold_scores": PRODUCTION_INNER_SCORE_COUNT,
            "source_inner_gamma_summary": PRODUCTION_GAMMA_SUMMARY_COUNT,
            "outer_results": PRODUCTION_OUTER_RESULT_COUNT,
            "conditional_frame_audit": PRODUCTION_FRAME_AUDIT_COUNT,
            "outer_comparison": PRODUCTION_OUTER_COMPARISON_COUNT,
        }
        if (
            n_outer != PRODUCTION_OUTER_FOLD_COUNT
            or n_gamma != PRODUCTION_GAMMA_COUNT
            or any(expected[key] != value for key, value in production.items())
        ):
            raise ProtocolError("CLA production cardinality locks are not satisfied.")


def _validate_reports(
    root: Path,
    leakage: Mapping[str, object],
    decision: Mapping[str, object],
    runtime: Mapping[str, object],
    protocol: Mapping[str, object],
    tables: AlignmentArtifactTables,
) -> None:
    expected_leakage = build_leakage_report(
        protocol,
        tables.conditional_frame_audit,
    )
    if dict(leakage) != expected_leakage:
        raise ProtocolError("CLA leakage/provenance report does not reconstruct.")
    if leakage.get("schema_version") != LEAKAGE_REPORT_SCHEMA_VERSION or leakage.get(
        "status"
    ) != "PASS":
        raise ProtocolError("CLA leakage report is not PASS.")

    expected_decision = build_decision_summary(
        tables.outer_results,
        tables.outer_comparison,
        tables.source_inner_gamma_summary,
        design_hash=str(protocol["design_hash"]),
        table_bundle_hash=str(protocol["table_bundle_hash"]),
        protocol_hash=str(protocol["protocol_hash"]),
        numerical_epsilon=1e-12,
        pass_min_nonnegative_center_deltas=5,
    )
    if dict(decision) != expected_decision:
        raise ProtocolError("CLA decision summary does not independently reconstruct.")
    if decision.get("schema_version") != DECISION_SUMMARY_SCHEMA_VERSION:
        raise ProtocolError("CLA decision summary schema drifted.")
    for field, value in {
        "claim_scope": CLA_CLAIM_SCOPE,
        "diagnostic_only": True,
        "non_adoptive": True,
        "adoption_enabled": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "uses_generated_embeddings": False,
        "uses_cvae_checkpoint": False,
        "uses_router": False,
        "uses_expert_bank": False,
        "uses_nelbo": False,
    }.items():
        if decision.get(field) != value:
            raise ProtocolError(f"CLA decision claim flag {field} drifted.")
    report = (root / "reports/decision_report.md").read_text(encoding="utf-8")
    if report != render_decision_report(decision):
        raise ProtocolError("CLA Markdown decision report does not match decision JSON.")

    elapsed = _float(runtime.get("elapsed_seconds"), "elapsed_seconds")
    expected_runtime = build_runtime_summary(
        protocol,
        tables,
        elapsed_seconds=elapsed,
    )
    if dict(runtime) != expected_runtime:
        raise ProtocolError("CLA runtime summary does not reconstruct.")
    if runtime.get("schema_version") != RUNTIME_SUMMARY_SCHEMA_VERSION:
        raise ProtocolError("CLA runtime summary schema drifted.")


def _validate_required_file_set(root: Path, coverage_mode: str) -> None:
    if coverage_mode == "complete":
        expected = set(CLA_COMPLETE_REQUIRED_OUTPUTS)
    elif coverage_mode == "partial_test":
        expected = set(CLA_RUNNER_REQUIRED_OUTPUTS)
    else:
        raise ProtocolError("CLA coverage mode is invalid for required-file validation.")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    missing = sorted(expected.difference(actual))
    stale = sorted(actual.difference(expected))
    if missing or stale:
        raise ProtocolError(
            "CLA required file set is not exact: "
            f"missing={missing}, stale extra={stale}."
        )


def _validate_content_index(
    root: Path,
    content_index: Mapping[str, object],
    coverage_mode: str,
) -> None:
    if (
        content_index.get("schema_version") != CONTENT_INDEX_SCHEMA_VERSION
        or content_index.get("hash_algorithm") != "sha256"
        or content_index.get("self_excluded") is not True
    ):
        raise ProtocolError("CLA content index contract drifted.")
    payload_without_hash = dict(content_index)
    persisted_hash = payload_without_hash.pop("content_index_hash", None)
    if persisted_hash != stable_hash(payload_without_hash):
        raise ProtocolError("CLA content index hash mismatch.")
    entries = content_index.get("entries")
    if not isinstance(entries, list):
        raise ProtocolError("CLA content index entries are malformed.")
    expected_entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "manifests/content_index.json":
            continue
        expected_entries.append(
            {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    actual_paths = {str(entry["path"]) for entry in expected_entries}
    required = (
        CLA_COMPLETE_REQUIRED_OUTPUTS
        if coverage_mode == "complete"
        else CLA_RUNNER_REQUIRED_OUTPUTS
    )
    expected_paths = set(required).difference({"manifests/content_index.json"})
    if actual_paths != expected_paths:
        raise ProtocolError("CLA content index contains missing or stale extra paths.")
    if (
        entries != expected_entries
        or content_index.get("indexed_file_count") != len(expected_entries)
        or any(
            entry.get("path") == "manifests/content_index.json"
            for entry in entries
            if isinstance(entry, Mapping)
        )
    ):
        raise ProtocolError("CLA content index does not match files or includes itself.")


def _validate_common_fit_fields(
    row: Mapping[str, object],
    frame: _ReconstructedFrame,
    classifier_hash: str,
    gamma: float,
) -> None:
    expected_fit_identity = _fit_identity(frame, classifier_hash, gamma)
    expected = {
        "gamma": str(float(gamma)),
        "fit_identity": expected_fit_identity,
        "conditional_frame_identity": frame.conditional_frame_identity,
        "fit_row_hash": frame.fit_row_hash,
        "scaler_state_hash": frame.scaler_state_hash,
        "penalty_operator_hash": frame.penalty_operator_hash,
        "classifier_config_hash": classifier_hash,
    }
    _assert_fields(row, expected, "fit identity")


def _validate_claim_row(row: Mapping[str, object], expected_row_role: str) -> None:
    for field, value in FAIL_CLOSED_CLAIM_VALUES.items():
        if row.get(field) != value:
            raise ProtocolError(f"CLA row claim field {field} drifted.")
    if row.get("row_role") != expected_row_role:
        raise ProtocolError("CLA row_role drifted or crossed table semantics.")


def _fit_identity(
    frame: _ReconstructedFrame,
    classifier_hash: str,
    gamma: float,
) -> str:
    return stable_hash(
        {
            "method": CLA_METHOD,
            "training_frame_hash": frame.training_frame_hash,
            "fit_row_hash": frame.fit_row_hash,
            "scaler_state_hash": frame.scaler_state_hash,
            "factor_hash": frame.penalty_operator_hash,
            "classifier_config_hash": classifier_hash,
            "gamma": float(gamma),
        }
    )


def _assert_no_identity_overlap(
    fit_rows: Sequence[object], eval_rows: Sequence[object]
) -> None:
    fields = ("sample_id", "case_id", "image_path", "row_index")
    for field in fields:
        fit = {getattr(row, field, "") for row in fit_rows}
        eval_ = {getattr(row, field, "") for row in eval_rows}
        if field in {"case_id", "image_path"}:
            fit.discard("")
            eval_.discard("")
        if fit.intersection(eval_):
            raise ProtocolError(f"CLA reconstructed fit/eval {field} overlap detected.")


def _summary_ranks(means: Mapping[float, float]) -> dict[float, int]:
    best = max(means.values())
    ordered = sorted(
        means,
        key=lambda gamma: (
            0 if abs(means[gamma] - best) <= 1e-12 else 1,
            gamma if abs(means[gamma] - best) <= 1e-12 else -means[gamma],
            gamma,
        ),
    )
    return {gamma: index + 1 for index, gamma in enumerate(ordered)}


def _validate_n_iter(value: object, *, gamma: float) -> None:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("CLA n_iter payload is malformed.") from exc
    if (
        not isinstance(payload, list)
        or not payload
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item >= 5000
            for item in payload
        )
        or (gamma == 0.0 and any(item <= 0 for item in payload))
    ):
        raise ProtocolError("CLA n_iter payload violates solver locks.")


def _unique_rows(
    rows: Sequence[Mapping[str, object]],
    key_fn: object,
    label: str,
) -> dict[object, Mapping[str, object]]:
    output: dict[object, Mapping[str, object]] = {}
    for row in rows:
        key = key_fn(row)  # type: ignore[operator]
        if key in output or key in {"", ("",)}:
            raise ProtocolError(f"CLA {label} has a duplicate or empty primary key.")
        output[key] = row
    return output


def _assert_fields(
    row: Mapping[str, object],
    expected: Mapping[str, object],
    label: str,
) -> None:
    for field, value in expected.items():
        if row.get(field) != value:
            raise ProtocolError(
                f"CLA {label} field {field} mismatch: expected={value!r}, "
                f"observed={row.get(field)!r}."
            )


def _assert_close(value: object, expected: float, label: str) -> None:
    observed = _float(value, label)
    if not math.isclose(observed, float(expected), abs_tol=1e-12, rel_tol=0.0):
        raise ProtocolError(f"CLA {label} does not recompute.")


def _float(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"CLA {label} is not numeric.") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"CLA {label} is not finite.")
    return number


def _identity_hash(values: object) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")  # type: ignore[arg-type]
    ).hexdigest()


def _json_cell(values: object) -> str:
    return json.dumps(list(values), sort_keys=True, separators=(",", ":"))  # type: ignore[arg-type]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"CLA {label} must be a mapping.")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ProtocolError(f"CLA {label} must be a list.")
    return value


def _read_csv(path: Path, columns: Sequence[str]) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ProtocolError(f"CLA table columns drifted: {path}.")
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise ProtocolError(f"CLA table is empty: {path}.")
    return rows


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed CLA JSON artifact: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"CLA JSON artifact must be an object: {path}.")
    return payload


__all__ = ["assert_conditional_logit_alignment_artifacts"]
