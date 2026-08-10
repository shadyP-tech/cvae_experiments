"""Strict H/q/e-excluded, rank-one source-inner ridge transfer model."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .case_features import (
    compute_source_control,
    context_permute_training_features,
    feature_surface_hash,
)
from .contracts import (
    CandidateClassModel,
    CaseClassWeights,
    CaseFeatureRow,
    DonorResponseRow,
    HierarchicalResidualModel,
    SourceControlRow,
    Standardization,
)
from .donor_responses import response_surface_hash
from .scientific_constants import (
    DESIGN_TERMS,
    MIDOGPP_CENTERS,
    RIDGE_GRID,
    SOFTMAX_TEMPERATURE,
    SPARSE_SOURCE_BUDGET,
    candidate_sources,
)


def strict_transfer_training_rows(
    responses: Sequence[DonorResponseRow],
    *,
    target_center: str,
    heldout_query_center: str | None,
    heldout_source_id: str,
    class_side: int,
) -> tuple[DonorResponseRow, ...]:
    """Filter model-fitting rows with the exact H/q/e exclusions.

    ``e`` is excluded both as a donor center and as a response source.  This
    forces transfer through the scalar probability-only descriptor instead of
    allowing source identity or source-specific labels to enter the fit.
    """

    target = str(target_center)
    source = str(heldout_source_id)
    query = None if heldout_query_center is None else str(heldout_query_center)
    if target not in MIDOGPP_CENTERS or source not in candidate_sources(target):
        raise ProtocolError("Strict transfer filter received an illegal H/e pair.")
    if query is not None and query in (target, source):
        raise ProtocolError("Nested held-query center must differ from H and e.")
    if class_side not in (0, 1) or isinstance(class_side, bool):
        raise ProtocolError("class_side must be integer zero or one.")
    forbidden_donors = {target, source}
    if query is not None:
        forbidden_donors.add(query)
    return tuple(
        sorted(
            row
            for row in responses
            if row.class_side == class_side
            and row.donor_center not in forbidden_donors
            and row.source_id not in forbidden_donors
        )
    )


def fit_standardization(raw_rows: Sequence[Sequence[float]]) -> Standardization:
    array = np.asarray(tuple(tuple(float(value) for value in row) for row in raw_rows), dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 5:
        raise ProtocolError("Donor-inner standardization requires a non-empty n-by-5 matrix.")
    if not np.isfinite(array).all():
        raise ProtocolError("Donor-inner standardization received non-finite data.")
    means = array.mean(axis=0, dtype=np.float64)
    scales = array.std(axis=0, ddof=0, dtype=np.float64)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    return Standardization(tuple(float(value) for value in means), tuple(float(value) for value in scales))


def interaction_design(
    phi: Sequence[float],
    global_source_control: float,
    standardization: Standardization,
    *,
    use_local_features: bool = True,
) -> tuple[float, ...]:
    """Build ``[1, phi_4, g, g*phi_4]`` with no source-ID coordinate."""

    if len(tuple(phi)) != 4:
        raise ProtocolError("Interaction design needs the exact four-component phi.")
    raw = np.asarray((*tuple(float(value) for value in phi), float(global_source_control)), dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ProtocolError("Interaction design received non-finite data.")
    means = np.asarray(standardization.means, dtype=np.float64)
    scales = np.asarray(standardization.scales, dtype=np.float64)
    standardized = (raw - means) / scales
    z_phi = standardized[:4] if use_local_features else np.zeros(4, dtype=np.float64)
    z_g = float(standardized[4])
    design = (1.0, *tuple(float(value) for value in z_phi), z_g, *tuple(float(z_g * value) for value in z_phi))
    if len(design) != len(DESIGN_TERMS):
        raise ProtocolError("Interaction design dimensionality drifted.")
    return design


def fit_loco_hierarchical_model(
    features: Sequence[CaseFeatureRow],
    responses: Sequence[DonorResponseRow],
    *,
    target_center: str,
    source_control_features: Sequence[CaseFeatureRow] | None = None,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    model_family: str = "R",
) -> HierarchicalResidualModel:
    """Fit separate class responses and leave each target candidate source out.

    Hyperparameters are selected only on other-center donor responses.  For a
    validation pair ``(q,e)``, fitting rows exclude donor centers ``H,q,e`` and
    response source ``e``.  The final fit excludes ``H,e`` and response source
    ``e``.  ``source_control_features`` lets the P control permute phi while
    retaining the canonical unpermuted ``g``.
    """

    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Unknown target center for hierarchical model.")
    family = str(model_family)
    if family not in ("G", "R", "P"):
        raise ProtocolError("Hierarchical model family must be G, R, or P.")
    if family == "P" and source_control_features is None:
        raise ProtocolError("P must receive unpermuted source-control features explicitly.")
    grid = tuple(float(value) for value in ridge_grid)
    if grid != RIDGE_GRID:
        raise ProtocolError("Ridge tuning left the frozen [0.1, 1, 10] grid.")
    feature_rows = tuple(features)
    response_rows = tuple(responses)
    control_rows = feature_rows if source_control_features is None else tuple(source_control_features)
    feature_lookup = _feature_lookup(feature_rows)
    context_feature_cache: dict[
        tuple[str, str | None, bool], Mapping[tuple[str, str, str], CaseFeatureRow]
    ] = {}
    source_control_cache: dict[tuple[str, str | None, str], SourceControlRow] = {}
    models: list[CandidateClassModel] = []
    for source in candidate_sources(target):
        # One alpha is selected for the H/e/model-family pair and shared by
        # both class-specific fits.  Validation errors are weighted by the
        # response class counts, never by a terminal exact endpoint.
        validation_sse = {alpha: 0.0 for alpha in grid}
        validation_weight = {alpha: 0 for alpha in grid}
        query_centers = tuple(
            center
            for center in MIDOGPP_CENTERS
            if center not in (target, source)
            and any(row.donor_center == center and row.source_id == source for row in response_rows)
        )
        if not query_centers:
            raise ProtocolError("Nested donor validation has no legal q/e response rows.")
        for query in query_centers:
            for side in (0, 1):
                training = strict_transfer_training_rows(
                    response_rows,
                    target_center=target,
                    heldout_query_center=query,
                    heldout_source_id=source,
                    class_side=side,
                )
                validation = tuple(
                    row
                    for row in response_rows
                    if row.donor_center == query
                    and row.source_id == source
                    and row.class_side == side
                )
                if not training or not validation:
                    continue
                train_raw, train_y = _raw_design_rows(
                    training,
                    feature_lookup,
                    control_rows,
                    target_center=target,
                    excluded_query_center=query,
                    heldout_source_id=source,
                    model_family=family,
                    context_feature_cache=context_feature_cache,
                    source_control_cache=source_control_cache,
                )
                standardization = _fit_family_standardization(train_raw, family)
                train_design = np.asarray(
                    [
                        interaction_design(
                            row[:4], row[4], standardization, use_local_features=family != "G"
                        )
                        for row in train_raw
                    ],
                    dtype=np.float64,
                )
                validation_raw, validation_y = _raw_design_rows(
                    validation,
                    feature_lookup,
                    control_rows,
                    target_center=target,
                    excluded_query_center=query,
                    heldout_source_id=source,
                    model_family=family,
                    context_feature_cache=context_feature_cache,
                    source_control_cache=source_control_cache,
                )
                validation_design = np.asarray(
                    [
                        interaction_design(
                            row[:4], row[4], standardization, use_local_features=family != "G"
                        )
                        for row in validation_raw
                    ],
                    dtype=np.float64,
                )
                validation_counts = np.asarray(
                    [row.sample_count for row in validation], dtype=np.float64
                )
                for alpha in grid:
                    coefficients = _ridge_coefficients(train_design, train_y, alpha)
                    residual = validation_design @ coefficients - validation_y
                    validation_sse[alpha] += float(
                        np.sum(validation_counts * residual * residual, dtype=np.float64)
                    )
                    validation_weight[alpha] += int(np.sum(validation_counts, dtype=np.float64))
        if any(validation_weight[alpha] <= 0 for alpha in grid):
            raise ProtocolError("Nested H/q/e validation did not score every frozen ridge value.")
        mean_losses = tuple(
            (alpha, validation_sse[alpha] / validation_weight[alpha]) for alpha in grid
        )
        # Canonical tie-break is the larger alpha.
        selected_alpha = min(mean_losses, key=lambda item: (item[1], -item[0]))[0]
        for side in (0, 1):
            final_training = strict_transfer_training_rows(
                response_rows,
                target_center=target,
                heldout_query_center=None,
                heldout_source_id=source,
                class_side=side,
            )
            final_raw, final_y = _raw_design_rows(
                final_training,
                feature_lookup,
                control_rows,
                target_center=target,
                excluded_query_center=None,
                heldout_source_id=source,
                model_family=family,
                context_feature_cache=context_feature_cache,
                source_control_cache=source_control_cache,
            )
            standardization = _fit_family_standardization(final_raw, family)
            design = np.asarray(
                [
                    interaction_design(
                        row[:4], row[4], standardization, use_local_features=family != "G"
                    )
                    for row in final_raw
                ],
                dtype=np.float64,
            )
            coefficients = _ridge_coefficients(design, final_y, selected_alpha)
            models.append(
                CandidateClassModel(
                    target_center=target,
                    heldout_source_id=source,
                    class_side=side,
                    ridge_alpha=selected_alpha,
                    coefficients=tuple(float(value) for value in coefficients),
                    standardization=standardization,
                    training_row_count=len(final_training),
                    donor_centers=tuple(sorted({row.donor_center for row in final_training})),
                    nested_validation_mse=mean_losses,
                    model_family=family,
                )
            )
    return HierarchicalResidualModel(
        target_center=target,
        candidate_models=tuple(sorted(models, key=lambda row: (row.heldout_source_id, row.class_side))),
        feature_surface_hash=feature_surface_hash(feature_rows),
        response_surface_hash=response_surface_hash(response_rows),
        model_family=family,
    )


def predict_candidate_gain(
    model: CandidateClassModel,
    feature: CaseFeatureRow,
    source_control: SourceControlRow,
) -> float:
    if (
        feature.source_id != model.heldout_source_id
        or source_control.source_id != model.heldout_source_id
        or feature.target_center != model.target_center
        or source_control.target_center != model.target_center
        or source_control.excluded_query_center is not None
        or source_control.context_excluded_centers
    ):
        raise ProtocolError("Candidate inference inputs do not match one final H/e model.")
    design = interaction_design(
        feature.phi,
        source_control.global_source_control,
        model.standardization,
        use_local_features=model.model_family != "G",
    )
    return float(np.dot(np.asarray(design), np.asarray(model.coefficients)))


def top2_sparse_simplex(
    scores: Mapping[str, float],
    *,
    temperature: float = SOFTMAX_TEMPERATURE,
    top_k: int = SPARSE_SOURCE_BUDGET,
) -> tuple[tuple[str, float], ...]:
    if abs(float(temperature) - SOFTMAX_TEMPERATURE) > 1.0e-15 or top_k != 2:
        raise ProtocolError("Sparse softmax left the frozen tau=0.01/top-2 contract.")
    if len(scores) < top_k:
        raise ProtocolError("Top-2 sparse simplex needs at least two candidate scores.")
    values = {str(source): float(score) for source, score in scores.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ProtocolError("Sparse softmax scores must be finite.")
    admitted = tuple(source for source, value in values.items() if value > 0.0)
    if not admitted:
        return ()
    selected = sorted(admitted, key=lambda source: (-values[source], source))[:top_k]
    maximum = max(values[source] for source in selected)
    exponentials = {
        source: math.exp(max((values[source] - maximum) / temperature, -700.0))
        for source in selected
    }
    normalizer = math.fsum(exponentials.values())
    return tuple(sorted((source, exponentials[source] / normalizer) for source in selected))


def predict_case_weights(
    model: HierarchicalResidualModel,
    features: Sequence[CaseFeatureRow],
    source_controls: Sequence[SourceControlRow] | None = None,
) -> tuple[CaseClassWeights, ...]:
    target = model.target_center
    if model.model_family == "P" and source_controls is None:
        raise ProtocolError("P inference must receive the preserved unpermuted source controls.")
    target_features = tuple(row for row in features if row.target_center == target)
    if not target_features:
        raise ProtocolError("No target-case features were supplied for model inference.")
    controls = (
        tuple(source_controls)
        if source_controls is not None
        else tuple(
            compute_source_control(features, target_center=target, source_id=source)
            for source in candidate_sources(target)
        )
    )
    control_lookup = {(row.target_center, row.source_id): row for row in controls}
    feature_by_case: dict[str, dict[str, CaseFeatureRow]] = defaultdict(dict)
    for row in target_features:
        if row.source_id in feature_by_case[row.case_id]:
            raise ProtocolError("Duplicate case/source feature at target inference.")
        feature_by_case[row.case_id][row.source_id] = row
    output: list[CaseClassWeights] = []
    for case, source_features in sorted(feature_by_case.items()):
        if set(source_features) != set(candidate_sources(target)):
            raise ProtocolError("Each target case needs all eight source features.")
        for side in (0, 1):
            gains = {
                source: predict_candidate_gain(
                    model.candidate(source, side),
                    source_features[source],
                    control_lookup[(target, source)],
                )
                for source in candidate_sources(target)
            }
            output.append(
                CaseClassWeights(
                    target_center=target,
                    case_id=case,
                    class_side=side,
                    weights=top2_sparse_simplex(gains),
                    predicted_gains=tuple(sorted(gains.items())),
                )
            )
    return tuple(sorted(output))


def _feature_lookup(
    features: Sequence[CaseFeatureRow],
) -> dict[tuple[str, str, str], CaseFeatureRow]:
    lookup: dict[tuple[str, str, str], CaseFeatureRow] = {}
    for row in features:
        key = (row.target_center, row.case_id, row.source_id)
        if key in lookup:
            raise ProtocolError("Feature surface contains duplicate query/case/source rows.")
        lookup[key] = row
    return lookup


def _raw_design_rows(
    responses: Sequence[DonorResponseRow],
    feature_lookup: Mapping[tuple[str, str, str], CaseFeatureRow],
    source_control_features: Sequence[CaseFeatureRow],
    *,
    target_center: str,
    excluded_query_center: str | None,
    heldout_source_id: str,
    model_family: str,
    context_feature_cache: dict[
        tuple[str, str | None, bool], Mapping[tuple[str, str, str], CaseFeatureRow]
    ],
    source_control_cache: dict[tuple[str, str | None, str], SourceControlRow],
) -> tuple[tuple[tuple[float, ...], ...], np.ndarray]:
    active_feature_lookup = feature_lookup
    if model_family == "P":
        include_destination = any(row.source_id == heldout_source_id for row in responses)
        context_key = (heldout_source_id, excluded_query_center, include_destination)
        active_feature_lookup = context_feature_cache.get(context_key, {})
        if not active_feature_lookup:
            contextual = context_permute_training_features(
                source_control_features,
                target_center=target_center,
                heldout_source_id=heldout_source_id,
                excluded_query_center=excluded_query_center,
                include_heldout_destination=include_destination,
            )
            active_feature_lookup = _feature_lookup(contextual)
            context_feature_cache[context_key] = active_feature_lookup
    raw: list[tuple[float, ...]] = []
    outcome: list[float] = []
    for row in responses:
        feature = active_feature_lookup.get((row.donor_center, row.case_id, row.source_id))
        if feature is None:
            raise ProtocolError("Donor response lacks its aligned case/source feature.")
        forbidden_origins = {target_center, heldout_source_id}
        if excluded_query_center is not None:
            forbidden_origins.add(excluded_query_center)
        is_validation_destination = row.source_id == heldout_source_id
        if feature.feature_origin_source_id in forbidden_origins and (
            model_family == "P" or not is_validation_destination
        ):
            raise ProtocolError("Model design admitted a held-role local-feature origin.")
        control_key = (heldout_source_id, excluded_query_center, row.source_id)
        if control_key not in source_control_cache:
            additional_exclusions = (
                () if row.source_id == heldout_source_id else (heldout_source_id,)
            )
            source_control_cache[control_key] = compute_source_control(
                source_control_features,
                target_center=target_center,
                source_id=row.source_id,
                excluded_query_center=excluded_query_center,
                additional_excluded_centers=additional_exclusions,
            )
        raw.append((*feature.phi, source_control_cache[control_key].global_source_control))
        outcome.append(row.smooth_response)
    if not raw:
        raise ProtocolError("No legal H/q/e donor rows remain for model fitting.")
    return tuple(raw), np.asarray(outcome, dtype=np.float64)


def _ridge_coefficients(design: np.ndarray, outcome: np.ndarray, alpha: float) -> np.ndarray:
    if design.ndim != 2 or design.shape[1] != len(DESIGN_TERMS) or outcome.shape != (design.shape[0],):
        raise ProtocolError("Ridge design/outcome shapes are invalid.")
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ outcome
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise ProtocolError("Ridge fit produced non-finite coefficients.")
    return coefficients


def _fit_family_standardization(
    raw_rows: Sequence[Sequence[float]], model_family: str
) -> Standardization:
    if model_family != "G":
        return fit_standardization(raw_rows)
    # G has exactly [intercept, g].  Local coordinates are fixed to neutral
    # standardization constants so neither their values nor their hashes can
    # influence this case-independent control.
    raw = np.asarray(raw_rows, dtype=np.float64)
    g = raw[:, 4]
    mean = float(g.mean(dtype=np.float64))
    scale = float(g.std(ddof=0, dtype=np.float64))
    if scale <= 1.0e-12:
        scale = 1.0
    return Standardization((0.0, 0.0, 0.0, 0.0, mean), (1.0, 1.0, 1.0, 1.0, scale))


__all__ = (
    "fit_loco_hierarchical_model",
    "fit_standardization",
    "interaction_design",
    "predict_candidate_gain",
    "predict_case_weights",
    "strict_transfer_training_rows",
    "top2_sparse_simplex",
)
