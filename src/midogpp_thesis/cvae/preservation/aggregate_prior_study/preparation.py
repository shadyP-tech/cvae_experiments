"""Source-isolated data and feature-frame preparation for v3."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ....real_features.classifier_reference.matched_reference import (
    PredictOnlySelection,
    select_nested_predict_spec_source_only,
)
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    load_midogpp_real_feature_frame,
)
from ...feature_frame import ExpertFeatureFrame
from ..prior_recovery_classifier import source_inner_classifier_specs
from ..prior_recovery_runtime_cache import FeatureFrameCache
from ..scoring import RepresentationScore, score_representation
from ..splits import frame_arrays, indices_for_centers, row_hash, source_only_frame
from .config import AggregatePriorStudyConfig


@dataclass(frozen=True)
class PreparedSourceExpert:
    source_center: str
    frame: ExpertFeatureFrame
    x_full: np.ndarray
    x_projected: np.ndarray
    labels: tuple[int, ...]
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    source_row_hash: str
    source_case_hash: str
    source_image_hash: str
    manifest_hash: str
    feature_cache_hash: str


@dataclass(frozen=True)
class PreparedEvaluation:
    outer_target_center: str
    inner_pseudo_target_center: str
    source_center: str
    selection: PredictOnlySelection
    x_eval_full: np.ndarray
    y_eval: tuple[int, ...]
    eval_sample_ids: tuple[str, ...]
    eval_case_ids: tuple[str, ...]
    eval_image_ids: tuple[str, ...]
    eval_row_hash: str
    real_source_score: RepresentationScore


def load_frame(config: AggregatePriorStudyConfig) -> RealFeatureFrame:
    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(config.manifest_path),
        feature_cache_path=Path(config.feature_cache_path),
        expected_feature_dim=config.expected_feature_dim,
    )
    if frame.eligible_centers != config.heldout_centers:
        raise ProtocolError(
            "Aggregate-prior v3 requires exact ordered eligible-center coverage."
        )
    return frame


def prepare_source_expert(
    config: AggregatePriorStudyConfig,
    *,
    frame: RealFeatureFrame,
    source_center: str,
    frame_cache: FeatureFrameCache,
    protocol_hash: str,
) -> PreparedSourceExpert:
    source = str(source_center)
    indices = indices_for_centers(frame, (source,))
    if not indices:
        raise ProtocolError(f"Source center {source} has no rows.")
    x_full, labels, sample_ids = frame_arrays(frame, indices)
    if set(labels) != {0, 1}:
        raise ProtocolError(f"Source center {source} does not contain both classes.")
    case_ids = tuple(str(frame.rows[index].case_id) for index in indices)
    image_ids = tuple(str(frame.rows[index].image_path) for index in indices)
    feature_frame, _ = frame_cache.fit_or_load(
        expert_id=f"independent_source_center_{source}",
        source_train_embeddings=x_full,
        fit_centers=(source,),
        fit_row_hash=row_hash(sample_ids),
        requested_dim=config.pca_dim,
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
        protocol_hash=protocol_hash,
        code_version=config.code_version,
    )
    if feature_frame.fit_scope != "per_expert_source_train":
        raise ProtocolError("Source expert frame is not source-local.")
    projected = np.asarray(feature_frame.transform(x_full), dtype=np.float32)
    if projected.shape != (len(indices), feature_frame.effective_dim):
        raise ProtocolError("Source-local PCA output has an invalid shape.")
    return PreparedSourceExpert(
        source_center=source,
        frame=feature_frame,
        x_full=np.asarray(x_full, dtype=np.float32),
        x_projected=projected,
        labels=tuple(labels),
        sample_ids=tuple(sample_ids),
        case_ids=case_ids,
        image_ids=image_ids,
        source_row_hash=row_hash(sample_ids),
        source_case_hash=row_hash(sorted(set(case_ids))),
        source_image_hash=row_hash(sorted(set(image_ids))),
        manifest_hash=frame.manifest_hash,
        feature_cache_hash=frame.feature_cache_hash,
    )


def prepare_evaluation(
    *,
    frame: RealFeatureFrame,
    source: PreparedSourceExpert,
    outer_target_center: str,
    inner_pseudo_target_center: str,
) -> PreparedEvaluation:
    outer = str(outer_target_center)
    inner = str(inner_pseudo_target_center)
    if len({outer, inner, source.source_center}) != 3:
        raise ProtocolError("Evaluation requires distinct H, I, and source.")
    selection_frame = source_only_frame(
        frame,
        outer_target_center=outer,
    )
    selection = select_nested_predict_spec_source_only(
        selection_frame,
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        candidate_specs=source_inner_classifier_specs(classifier_seed=23),
    )
    eval_indices = indices_for_centers(frame, (inner,))
    x_eval, y_eval, eval_ids = frame_arrays(frame, eval_indices)
    eval_cases = tuple(str(frame.rows[index].case_id) for index in eval_indices)
    eval_images = tuple(str(frame.rows[index].image_path) for index in eval_indices)
    _assert_disjoint(
        source,
        eval_sample_ids=eval_ids,
        eval_case_ids=eval_cases,
        eval_image_ids=eval_images,
    )
    real_score = score_representation(
        source.x_full,
        source.labels,
        x_eval,
        y_eval,
        spec=selection.selected_spec,
    )
    if not real_score.converged:
        raise ProtocolError(
            f"Source-local real comparator failed for H={outer}, I={inner}, "
            f"E={source.source_center}."
        )
    return PreparedEvaluation(
        outer_target_center=outer,
        inner_pseudo_target_center=inner,
        source_center=source.source_center,
        selection=selection,
        x_eval_full=np.asarray(x_eval, dtype=np.float32),
        y_eval=tuple(y_eval),
        eval_sample_ids=tuple(eval_ids),
        eval_case_ids=eval_cases,
        eval_image_ids=eval_images,
        eval_row_hash=row_hash(eval_ids),
        real_source_score=real_score,
    )


def isolation_audit(
    *,
    source: PreparedSourceExpert,
    evaluation: PreparedEvaluation,
) -> Mapping[str, object]:
    source_samples = set(source.sample_ids)
    source_cases = set(source.case_ids)
    source_images = set(source.image_ids)
    eval_samples = set(evaluation.eval_sample_ids)
    eval_cases = set(evaluation.eval_case_ids)
    eval_images = set(evaluation.eval_image_ids)
    sample_overlap = source_samples.intersection(eval_samples)
    case_overlap = source_cases.intersection(eval_cases)
    image_overlap = {
        value for value in source_images.intersection(eval_images) if value
    }
    passed = not sample_overlap and not case_overlap and not image_overlap
    return {
        "schema_version": "midogpp_independent_source_isolation_audit_v3",
        "outer_target_center": evaluation.outer_target_center,
        "inner_pseudo_target_center": evaluation.inner_pseudo_target_center,
        "source_center": source.source_center,
        "fit_centers": json.dumps([source.source_center]),
        "outer_center_absent_from_fit": (
            evaluation.outer_target_center != source.source_center
        ),
        "inner_center_absent_from_fit": (
            evaluation.inner_pseudo_target_center != source.source_center
        ),
        "sample_overlap_count": len(sample_overlap),
        "case_overlap_count": len(case_overlap),
        "image_overlap_count": len(image_overlap),
        "source_row_hash": source.source_row_hash,
        "inner_eval_row_hash": evaluation.eval_row_hash,
        "outer_rows_used": False,
        "inner_rows_used_for_fit": False,
        "inner_labels_used_for_scoring_only": True,
        "status": "PASS" if passed else "FAIL",
    }


def _assert_disjoint(
    source: PreparedSourceExpert,
    *,
    eval_sample_ids: tuple[str, ...],
    eval_case_ids: tuple[str, ...],
    eval_image_ids: tuple[str, ...],
) -> None:
    sample_overlap = set(source.sample_ids).intersection(eval_sample_ids)
    case_overlap = set(source.case_ids).intersection(eval_case_ids)
    image_overlap = {
        value for value in set(source.image_ids).intersection(eval_image_ids) if value
    }
    if sample_overlap or case_overlap or image_overlap:
        raise ProtocolError(
            "Source expert and inner evaluation identities overlap."
        )
