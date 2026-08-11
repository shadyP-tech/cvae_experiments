"""Known-bank H/q/e orchestration for pairwise regret fitting."""

from __future__ import annotations

from ...protocol import ProtocolError
from ._solver import (
    GRADIENT_TOLERANCE,
    MAX_NEWTON_ITERATIONS,
    solve_pairwise_logit,
)
from .contracts import (
    FEATURE_NAMES,
    DisagreementFeatureSurface,
    ExactRegretSurface,
    PairwiseRegretModel,
)
from .controls import assert_control_surface_matches_parent
from .design import (
    ACTION_L2_PENALTY,
    SHARED_L2_PENALTY,
    build_pairwise_training_design,
)
from .hashing import canonical_sha256
from .provenance import DevelopmentContext, assert_development_context
from .probability_contracts import LABEL_FREE_INFERENCE_SURFACE_ROLE


def _fit_one_candidate(
    features: DisagreementFeatureSurface,
    responses: ExactRegretSurface,
    *,
    context: DevelopmentContext,
    family: str,
    candidate_action_id: str,
    heldout_query_id: str | None,
) -> PairwiseRegretModel:
    source_id = features.candidate_source_by_action[candidate_action_id]
    response_queries = tuple(sorted({row.query_id for row in responses.rows}))
    if source_id not in response_queries:
        raise ProtocolError(
            "Known-bank H/q/e exclusion requires candidate source identity as a donor query."
        )
    excluded = {context.outer_target_id, source_id}
    if heldout_query_id is not None:
        heldout = str(heldout_query_id)
        if heldout == context.outer_target_id:
            raise ProtocolError("Nested heldout q cannot equal the outer target H.")
        if heldout not in response_queries:
            raise ProtocolError("Nested heldout q must be a source-OOF donor query.")
        excluded.add(heldout)
    legal_queries = tuple(query for query in response_queries if query not in excluded)
    forbidden_source_ids = {context.outer_target_id}
    if heldout_query_id is not None:
        forbidden_source_ids.add(str(heldout_query_id))
    legal_feature_rows = tuple(
        row for row in features.rows if row.source_id not in forbidden_source_ids
    )
    legal_response_rows = tuple(
        row for row in responses.rows if row.source_id not in forbidden_source_ids
    )
    action_ids = tuple(
        sorted(
            (
                features.baseline_action_id,
                *(
                    action_id
                    for action_id, action_source in features.candidate_source_by_action.items()
                    if action_source not in forbidden_source_ids
                ),
            )
        )
    )
    design = build_pairwise_training_design(
        legal_feature_rows,
        legal_response_rows,
        legal_query_ids=legal_queries,
        action_ids=action_ids,
        control_action_id=features.control_action_id,
    )
    coefficients, covariance, iteration_count = solve_pairwise_logit(design)
    legal_query_set = set(legal_queries)
    training_feature_hash = canonical_sha256(
        [
            row.feature_hash
            for row in legal_feature_rows
            if row.query_id in legal_query_set
        ]
    )
    training_response_hash = canonical_sha256(
        [
            row.response_hash
            for row in legal_response_rows
            if row.query_id in legal_query_set
        ]
    )
    return PairwiseRegretModel(
        family=family,
        outer_target_id=context.outer_target_id,
        candidate_action_id=candidate_action_id,
        candidate_source_id=source_id,
        heldout_query_id=heldout_query_id,
        action_ids=design.encoder.action_ids,
        feature_names=FEATURE_NAMES,
        feature_mean=design.encoder.feature_mean,
        feature_scale=design.encoder.feature_scale,
        coefficients=coefficients,
        coefficient_covariance=covariance,
        training_query_ids=legal_queries,
        excluded_query_ids=tuple(sorted(excluded)),
        observation_count=len(design.values),
        converged=True,
        iteration_count=iteration_count,
        feature_surface_hash=features.surface_hash,
        response_surface_hash=responses.surface_hash,
        prediction_seal_hash=features.prediction_seal_hash,
        development_context_hash=features.development_context_hash,
        baseline_action_id=features.baseline_action_id,
        control_action_id=features.control_action_id,
        candidate_source_by_action=tuple(features.candidate_source_by_action.items()),
        training_feature_hash=training_feature_hash,
        training_response_hash=training_response_hash,
        shared_l2_penalty=SHARED_L2_PENALTY,
        action_l2_penalty=ACTION_L2_PENALTY,
        max_newton_iterations=MAX_NEWTON_ITERATIONS,
        gradient_tolerance=GRADIENT_TOLERANCE,
        training_scope=context.scope.value,
        training_surface_role=features.surface_role,
    )


def fit_known_bank_pairwise_models(
    features: DisagreementFeatureSurface,
    responses: ExactRegretSurface,
    *,
    context: DevelopmentContext,
    family: str,
    heldout_query_id: str | None = None,
    aligned_parent_features: DisagreementFeatureSurface | None = None,
) -> tuple[PairwiseRegretModel, ...]:
    """Fit one strict H/q/e model per known-bank candidate.

    Hyperparameters are fixed constants.  This function exposes no alpha,
    feature, lambda, threshold, or model-capacity search surface.
    """

    assert_development_context(context)
    if family not in ("G", "R", "P"):
        raise ProtocolError("Model family must be G, R, or P.")
    if not isinstance(features, DisagreementFeatureSurface) or not isinstance(
        responses, ExactRegretSurface
    ):
        raise ProtocolError("Pairwise fitting requires typed feature/response surfaces.")
    if features.surface_role == LABEL_FREE_INFERENCE_SURFACE_ROLE:
        raise ProtocolError("Label-free inference surfaces cannot enter model fitting.")
    if responses.response_semantics != "source_oof_exact_bacc_gain_vs_control":
        raise ProtocolError("Only exact-BACC source-OOF response semantics are admissible.")
    context_hash = canonical_sha256(context.to_payload())
    if (
        features.family != family
        or features.development_context_hash != context_hash
        or features.dataset_family != context.dataset_family
        or features.outer_target_id != context.outer_target_id
        or responses.development_context_hash != context_hash
        or responses.prediction_seal_hash != features.prediction_seal_hash
    ):
        raise ProtocolError("Pairwise feature/response/context lineage drifted.")
    if family == "R":
        if (
            features.parent_surface_hash is not None
            or responses.feature_surface_hash != features.surface_hash
            or aligned_parent_features is not None
        ):
            raise ProtocolError("Aligned R features drifted from exact response lineage.")
    else:
        if not isinstance(aligned_parent_features, DisagreementFeatureSurface):
            raise ProtocolError("G/P fitting requires the typed aligned R parent surface.")
        assert_control_surface_matches_parent(features, aligned_parent_features)
        if aligned_parent_features.surface_hash != responses.feature_surface_hash:
            raise ProtocolError("G/P response lineage drifted from aligned R.")
    feature_keys = {
        row.row_key
        for row in features.rows
        if row.query_id in {response.query_id for response in responses.rows}
    }
    response_by_key = {row.row_key: row for row in responses.rows}
    if feature_keys != set(response_by_key):
        raise ProtocolError("Control features do not align to the exact response rows.")
    feature_by_key = {
        row.row_key: row
        for row in features.rows
        if row.query_id in {response.query_id for response in responses.rows}
    }
    if any(
        response.source_id != feature_by_key[key].source_id
        for key, response in response_by_key.items()
    ):
        raise ProtocolError("Exact response source identity drifted from feature lineage.")
    return tuple(
        _fit_one_candidate(
            features,
            responses,
            context=context,
            family=family,
            candidate_action_id=action_id,
            heldout_query_id=heldout_query_id,
        )
        for action_id in sorted(features.candidate_source_by_action)
        if features.candidate_source_by_action[action_id] != context.outer_target_id
        and (
            heldout_query_id is None
            or features.candidate_source_by_action[action_id] != str(heldout_query_id)
        )
    )


__all__ = ("fit_known_bank_pairwise_models",)
