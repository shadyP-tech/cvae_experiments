"""Label-free target contrast scoring for fitted pairwise regret models."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CandidateContrastRow,
    DisagreementFeatureSurface,
    PairwiseRegretModel,
)
from .design import DesignEncoder
from .hashing import canonical_sha256
from .inference_contracts import (
    LabelFreeInferenceContext,
    assert_label_free_inference_context,
)
from .model_bank import PairwiseRegretModelBank
from .provenance import DevelopmentContext, assert_development_context
from .probability_contracts import LABEL_FREE_INFERENCE_SURFACE_ROLE


def _encoder_from_model(model: PairwiseRegretModel) -> DesignEncoder:
    return DesignEncoder(
        action_ids=model.action_ids,
        feature_names=model.feature_names,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
    )


def _contrast(
    model: PairwiseRegretModel,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float]:
    difference = left - right
    mean = float(difference @ model.coefficients)
    variance = float(difference @ model.coefficient_covariance @ difference)
    if variance < -1.0e-10:
        raise ProtocolError("Candidate contrast covariance produced negative variance.")
    return mean, math.sqrt(max(0.0, variance))


def _score_contrasts(
    fitted: tuple[PairwiseRegretModel, ...],
    features: DisagreementFeatureSurface,
    *,
    target_query_id: str,
) -> tuple[CandidateContrastRow, ...]:
    model_by_action = {model.candidate_action_id: model for model in fitted}
    if len(model_by_action) != len(fitted):
        raise ProtocolError("Candidate scoring received duplicate action models.")
    if set(model_by_action) != set(features.candidate_source_by_action):
        raise ProtocolError("Candidate scoring requires the complete candidate model bank.")
    target_rows = tuple(row for row in features.rows if row.query_id == target_query_id)
    rows_by_case: dict[str, dict[str, object]] = defaultdict(dict)
    for row in target_rows:
        rows_by_case[row.case_id][row.action_id] = row
    if not rows_by_case:
        raise ProtocolError("No label-free rows exist for the outer target query.")

    output: list[CandidateContrastRow] = []
    for case_id, row_by_action in sorted(rows_by_case.items()):
        try:
            baseline = row_by_action[features.baseline_action_id]
        except KeyError as exc:
            raise ProtocolError("Target case lacks immutable B features.") from exc
        for action_id, model in sorted(model_by_action.items()):
            try:
                candidate = row_by_action[action_id]
            except KeyError as exc:
                raise ProtocolError("Target case lacks a modeled candidate action.") from exc
            encoder = _encoder_from_model(model)
            candidate_design = encoder.encode(candidate)
            baseline_design = encoder.encode(baseline)
            control_design = encoder.encode_control()
            gain_u, se_u = _contrast(model, candidate_design, control_design)
            gain_b, se_b = _contrast(model, candidate_design, baseline_design)
            output.append(
                CandidateContrastRow(
                    family=model.family,
                    target_query_id=target_query_id,
                    case_id=case_id,
                    candidate_action_id=action_id,
                    candidate_source_id=model.candidate_source_id,
                    predicted_preference_margin_vs_control=gain_u,
                    standard_error_vs_control=se_u,
                    predicted_preference_margin_vs_baseline=gain_b,
                    standard_error_vs_baseline=se_b,
                    model_hash=model.model_hash,
                )
            )
    return tuple(sorted(output, key=lambda row: row.row_key))


def score_target_candidate_contrasts(
    models: Sequence[PairwiseRegretModel],
    features: DisagreementFeatureSurface,
    *,
    context: DevelopmentContext,
) -> tuple[CandidateContrastRow, ...]:
    """Score the same sealed development surface used during model fitting."""

    assert_development_context(context)
    fitted = tuple(models)
    if not fitted or any(not isinstance(model, PairwiseRegretModel) for model in fitted):
        raise ProtocolError("Candidate scoring requires typed fitted models.")
    if any(model.heldout_query_id is not None for model in fitted):
        raise ProtocolError("Nested donor-q models cannot score an outer target surface.")
    families = {model.family for model in fitted}
    targets = {model.outer_target_id for model in fitted}
    if len(families) != 1 or targets != {context.outer_target_id}:
        raise ProtocolError("Candidate models cannot mix families or outer targets.")
    context_hash = canonical_sha256(context.to_payload())
    expected_mapping = tuple(sorted(features.candidate_source_by_action.items()))
    if any(
        model.feature_surface_hash != features.surface_hash
        or model.prediction_seal_hash != features.prediction_seal_hash
        or model.development_context_hash != context_hash
        or model.baseline_action_id != features.baseline_action_id
        or model.control_action_id != features.control_action_id
        or model.candidate_source_by_action != expected_mapping
        or model.family != features.family
        for model in fitted
    ):
        raise ProtocolError("Candidate model/scoring feature lineage drifted.")
    return _score_contrasts(
        fitted,
        features,
        target_query_id=context.outer_target_id,
    )


def score_label_free_inference_candidate_contrasts(
    model_bank: PairwiseRegretModelBank,
    features: DisagreementFeatureSurface,
    *,
    context: LabelFreeInferenceContext,
) -> tuple[CandidateContrastRow, ...]:
    """Apply a frozen train-only bank to a separately sealed target surface."""

    assert_label_free_inference_context(context)
    if not isinstance(model_bank, PairwiseRegretModelBank):
        raise ProtocolError("Inference scoring requires a typed frozen model bank.")
    if not isinstance(features, DisagreementFeatureSurface):
        raise ProtocolError("Inference scoring requires a typed feature surface.")
    if model_bank.model_bank_hash != context.model_bank_hash:
        raise ProtocolError("Inference model-bank seal drifted from the context.")
    if model_bank.action_schema != context.action_schema:
        raise ProtocolError("Inference model bank drifted from the frozen action schema.")
    schema = context.action_schema
    if (
        model_bank.outer_target_id != context.outer_target_id
        or features.outer_target_id != context.outer_target_id
        or features.dataset_family != context.dataset_family
        or features.query_ids != (context.outer_target_id,)
        or features.surface_role != LABEL_FREE_INFERENCE_SURFACE_ROLE
        or features.development_context_hash != context.context_hash
        or features.prediction_seal_hash != context.prediction_seal_hash
        or features.family != schema.family
        or features.baseline_action_id != schema.baseline_action_id
        or features.control_action_id != schema.control_action_id
        or tuple(sorted(features.candidate_source_by_action.items()))
        != schema.candidate_source_by_action
    ):
        raise ProtocolError("Inference feature/cache/action lineage drifted.")
    return _score_contrasts(
        model_bank.models,
        features,
        target_query_id=context.outer_target_id,
    )


__all__ = (
    "score_label_free_inference_candidate_contrasts",
    "score_target_candidate_contrasts",
)
