"""Seed-invariant source-inner fold preparation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from time import perf_counter
from typing import Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.classifiers import ClassifierSpec
from ...real_features.classifier_reference.matched_reference import (
    PredictOnlySelection,
    select_nested_predict_spec_source_only,
)
from ...real_features.classifier_reference.midogpp_real_feature_classifier import (
    RealFeatureFrame,
)
from ...real_features.classifier_reference.protocol import ProtocolError
from ..feature_frame import ExpertFeatureFrame
from .prior_recovery_runtime_cache import FeatureFrameCache
from .prior_recovery_schema import NESTED_REAL_REFERENCE_SCHEMA
from .prior_recovery_timing import RuntimeTimingRecorder
from .scoring import score_representation
from .splits import (
    assert_identity_overlap_pass,
    frame_arrays,
    identity_overlap_audit,
    indices_for_centers,
    inner_split,
    row_hash,
)


@dataclass(frozen=True)
class PreparedSourceInnerFold:
    outer: str
    inner: str
    fit_centers: tuple[str, ...]
    spec: ClassifierSpec
    selection: PredictOnlySelection
    frame: ExpertFeatureFrame
    x_fit: object
    y_fit: tuple[int, ...]
    source_ids: tuple[str, ...]
    x_eval: object
    y_eval: tuple[int, ...]
    eval_ids: tuple[str, ...]
    real_bacc: float
    real_reference_protocol_hash: str
    manifest_hash: str
    feature_cache_hash: str


def prepare_source_inner_fold(
    *,
    pca_dim: int,
    frame: RealFeatureFrame,
    outer: str,
    inner: str,
    candidate_specs: Sequence[ClassifierSpec],
    preparation_protocol_hash: str,
    preparation_code_version: str,
    frame_cache: FeatureFrameCache,
    timings: RuntimeTimingRecorder,
) -> tuple[PreparedSourceInnerFold, dict[str, object], dict[str, object]]:
    """Prepare classifier, denominator, identity audit, and PCA once per H/I."""

    split = inner_split(outer, inner, centers=frame.eligible_centers)
    started = perf_counter()
    selection = select_nested_predict_spec_source_only(
        frame,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        candidate_specs=candidate_specs,
    )
    timings.record(
        phase="nested_classifier_selection",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
    )
    fit_idx = indices_for_centers(frame, split.fit_centers)
    eval_idx = indices_for_centers(frame, (inner,))
    audit = identity_overlap_audit(
        frame,
        fit_indices=fit_idx,
        eval_indices=eval_idx,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
    )
    assert_identity_overlap_pass(audit)
    x_fit_full, y_fit, source_ids = frame_arrays(frame, fit_idx)
    x_eval_full, y_eval, eval_ids = frame_arrays(frame, eval_idx)
    real_score = score_representation(
        x_fit_full,
        y_fit,
        x_eval_full,
        y_eval,
        spec=selection.selected_spec,
    )
    if not real_score.converged:
        raise ProtocolError(
            f"Nested real reference did not converge for H={outer}, I={inner}."
        )
    started = perf_counter()
    feature_frame, frame_cache_hit = frame_cache.fit_or_load(
        expert_id=f"source_inner_H{outer}_I{inner}",
        source_train_embeddings=x_fit_full,
        fit_centers=split.fit_centers,
        fit_row_hash=row_hash(source_ids),
        requested_dim=int(pca_dim),
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
        protocol_hash=preparation_protocol_hash,
        code_version=preparation_code_version,
    )
    timings.record(
        phase="pca_frame",
        elapsed_seconds=perf_counter() - started,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        cache_status="hit" if frame_cache_hit else "miss",
    )
    real_reference_hash = stable_hash(
        {
            "outer": outer,
            "inner": inner,
            "fit_row_hash": row_hash(source_ids),
            "eval_row_hash": row_hash(eval_ids),
            "classifier_spec_hash": selection.selected_spec.config_hash,
            "grid_hash": selection.grid_hash,
        }
    )
    prepared = PreparedSourceInnerFold(
        outer=outer,
        inner=inner,
        fit_centers=split.fit_centers,
        spec=selection.selected_spec,
        selection=selection,
        frame=feature_frame,
        x_fit=feature_frame.transform(x_fit_full),
        y_fit=y_fit,
        source_ids=source_ids,
        x_eval=feature_frame.transform(x_eval_full),
        y_eval=y_eval,
        eval_ids=eval_ids,
        real_bacc=real_score.bacc,
        real_reference_protocol_hash=real_reference_hash,
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
    )
    nested_row = {
        "schema_version": NESTED_REAL_REFERENCE_SCHEMA,
        "outer_target_center": outer,
        "inner_pseudo_target_center": inner,
        "deeper_validation_centers": json.dumps(list(selection.center_scores)),
        "fit_centers": json.dumps(list(split.fit_centers)),
        "fit_row_hash": row_hash(source_ids),
        "eval_row_hash": row_hash(eval_ids),
        "classifier_grid_hash": selection.grid_hash,
        "selected_classifier_spec": json.dumps(
            selection.selected_spec.to_payload(),
            sort_keys=True,
        ),
        "selected_classifier_spec_hash": selection.selected_spec.config_hash,
        "real_reference_protocol_hash": real_reference_hash,
        "n_fit": len(fit_idx),
        "n_eval": len(eval_idx),
        "bacc": real_score.bacc,
        "macro_f1": real_score.macro_f1,
        "converged": True,
        "status": "ok",
        "target_eval_labels_used_for_scoring_only": False,
        "selection_used_outer_or_inner_labels": False,
    }
    return prepared, nested_row, audit
