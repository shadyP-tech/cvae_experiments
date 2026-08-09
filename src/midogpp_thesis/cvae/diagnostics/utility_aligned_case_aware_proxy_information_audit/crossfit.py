"""Fixed-alpha cross-fitting with strict all-role H/q/e exclusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.local_marginal_utility.ridge import fit_cluster_weighted_ridge
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    CENTERS,
    FAMILY_IDS,
    RESPONSE_NAMES,
    RIDGE_ALPHA,
    CaseAwareCrossfitResult,
    CaseAwareFeatureSurface,
    CaseAwareResponseRow,
    CaseAwareResponseSurface,
    CrossfitFoldAudit,
    CrossfitPredictionRow,
    expected_strict_training_row_count,
)
from .family_designs import build_family_designs
from .response_surfaces import build_response_surface


_FOLD_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "family_id",
        "response_name",
        "predicted_row_key",
        "excluded_domain_ids",
        "training_row_keys",
        "training_row_count",
        "ridge_alpha",
        "ridge_cluster_unit",
        "strict_H_q_e_exclusion_from_all_training_roles",
        "scaling_fit_on_training_fold_only",
        "hyperparameter_selection",
        "feature_mean",
        "feature_scale",
        "intercept",
        "coefficients",
        "family_design_hash",
        "fold_hash",
    }
)


def crossfit_fold_audit_from_payload(
    payload: Mapping[str, object],
) -> CrossfitFoldAudit:
    """Reconstruct and hash-verify one persisted fitted fold audit."""

    if not isinstance(payload, Mapping) or set(payload) != _FOLD_PAYLOAD_KEYS:
        raise ProtocolError("Crossfit fold payload does not match the exact schema.")
    if (
        payload.get("schema_version")
        != "midogpp_stage90_case_aware_crossfit_fold_v1"
        or payload.get("ridge_cluster_unit") != "outer_target_query"
        or payload.get("strict_H_q_e_exclusion_from_all_training_roles") is not True
        or payload.get("scaling_fit_on_training_fold_only") is not True
        or payload.get("hyperparameter_selection")
        != "none_fixed_predeclared"
    ):
        raise ProtocolError("Crossfit fold payload semantics drifted.")
    supplied_hash = payload.get("fold_hash")
    unhashed = {key: payload[key] for key in payload if key != "fold_hash"}
    if supplied_hash != canonical_sha256(unhashed):
        raise ProtocolError("Crossfit fold fitted provenance hash drifted.")
    return CrossfitFoldAudit(
        family_id=payload["family_id"],  # type: ignore[arg-type]
        response_name=payload["response_name"],  # type: ignore[arg-type]
        predicted_row_key=tuple(payload["predicted_row_key"]),  # type: ignore[arg-type]
        excluded_domain_ids=tuple(payload["excluded_domain_ids"]),  # type: ignore[arg-type]
        training_row_keys=tuple(
            tuple(key) for key in payload["training_row_keys"]  # type: ignore[union-attr]
        ),
        training_row_count=payload["training_row_count"],  # type: ignore[arg-type]
        ridge_alpha=payload["ridge_alpha"],  # type: ignore[arg-type]
        feature_mean=tuple(payload["feature_mean"]),  # type: ignore[arg-type]
        feature_scale=tuple(payload["feature_scale"]),  # type: ignore[arg-type]
        intercept=payload["intercept"],  # type: ignore[arg-type]
        coefficients=tuple(payload["coefficients"]),  # type: ignore[arg-type]
        family_design_hash=payload["family_design_hash"],  # type: ignore[arg-type]
        fold_hash=supplied_hash,  # type: ignore[arg-type]
    )


def crossfit_proxy_families(
    feature_surface: CaseAwareFeatureSurface,
    responses: CaseAwareResponseSurface
    | Sequence[CaseAwareResponseRow | Mapping[str, object]],
    *,
    family_ids: Sequence[str] = FAMILY_IDS,
    response_names: Sequence[str] = RESPONSE_NAMES,
) -> CaseAwareCrossfitResult:
    """Cross-fit every requested family separately for both responses.

    The default call fits all seven predeclared families independently on the
    exact and smooth response surfaces.  Alpha is fixed at one; there is no
    tuning or model selection.  Rows containing a predicted row's H, q, or e
    in *any* domain role are unavailable to that prediction.
    """

    if not isinstance(feature_surface, CaseAwareFeatureSurface):
        raise ProtocolError("Crossfit requires a typed feature surface.")
    response_surface = (
        responses
        if isinstance(responses, CaseAwareResponseSurface)
        else build_response_surface(feature_surface, responses)
    )
    if feature_surface.row_keys != response_surface.row_keys:
        raise ProtocolError("Feature and response H/q/e surfaces do not align.")
    selected_families = _selected(family_ids, FAMILY_IDS, "family")
    selected_responses = _selected(response_names, RESPONSE_NAMES, "response")
    designs = build_family_designs(feature_surface)
    rows = feature_surface.rows
    expected_training_count = expected_strict_training_row_count(
        tuple({value for row in rows for value in row.row_key})
    )
    # The six role permutations of one domain triple share the same excluded
    # set and training fold.  Reusing the fit is an exact optimization, not a
    # change in the per-prediction fold contract.
    prediction_indices_by_excluded: dict[frozenset[str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        prediction_indices_by_excluded[frozenset(row.row_key)].append(index)

    prediction_rows: list[CrossfitPredictionRow] = []
    audits: list[CrossfitFoldAudit] = []
    for family_id in selected_families:
        design = designs[family_id]
        for response_name in selected_responses:
            observed = np.asarray(
                [row.response_value(response_name) for row in response_surface.rows],
                dtype=np.float64,
            )
            for excluded_set, prediction_indices in sorted(
                prediction_indices_by_excluded.items(),
                key=lambda item: tuple(
                    center for center in CENTERS if center in item[0]
                ),
            ):
                training_indices = tuple(
                    index
                    for index, row in enumerate(rows)
                    if excluded_set.isdisjoint(row.row_key)
                )
                if len(training_indices) != expected_training_count:
                    raise ProtocolError(
                        "Strict H/q/e training-row geometry drifted."
                    )
                training_rows = tuple(rows[index] for index in training_indices)
                if any(
                    not excluded_set.isdisjoint(training.row_key)
                    for training in training_rows
                ):
                    raise ProtocolError("Strict H/q/e all-role exclusion failed.")
                training_index_array = np.asarray(training_indices, dtype=np.int64)
                training_clusters = tuple(
                    f"{row.outer_target_id}::{row.query_id}"
                    for row in training_rows
                )
                model = fit_cluster_weighted_ridge(
                    design.values[training_index_array],
                    observed[training_index_array],
                    training_clusters,
                    alpha=RIDGE_ALPHA,
                    feature_names=design.spec.predictor_names,
                )
                prediction_index_array = np.asarray(
                    sorted(prediction_indices), dtype=np.int64
                )
                predicted_values = model.predict(design.values[prediction_index_array])
                for prediction_index, predicted in zip(
                    prediction_index_array.tolist(), predicted_values, strict=True
                ):
                    predicted_row = rows[prediction_index]
                    excluded = tuple(
                        center for center in CENTERS if center in excluded_set
                    )
                    fold_unhashed = {
                        "schema_version": "midogpp_stage90_case_aware_crossfit_fold_v1",
                        "family_id": family_id,
                        "response_name": response_name,
                        "predicted_row_key": list(predicted_row.row_key),
                        "excluded_domain_ids": list(excluded),
                        "training_row_keys": [
                            list(row.row_key) for row in training_rows
                        ],
                        "training_row_count": len(training_rows),
                        "ridge_alpha": RIDGE_ALPHA,
                        "ridge_cluster_unit": "outer_target_query",
                        "strict_H_q_e_exclusion_from_all_training_roles": True,
                        "scaling_fit_on_training_fold_only": True,
                        "hyperparameter_selection": "none_fixed_predeclared",
                        "feature_mean": model.feature_mean.tolist(),
                        "feature_scale": model.feature_scale.tolist(),
                        "intercept": model.intercept,
                        "coefficients": model.coefficients.tolist(),
                        "family_design_hash": design.design_hash,
                    }
                    fold_hash = canonical_sha256(fold_unhashed)
                    audit = CrossfitFoldAudit(
                        family_id=family_id,
                        response_name=response_name,
                        predicted_row_key=predicted_row.row_key,
                        excluded_domain_ids=excluded,
                        training_row_keys=tuple(
                            row.row_key for row in training_rows
                        ),
                        training_row_count=len(training_rows),
                        ridge_alpha=RIDGE_ALPHA,
                        feature_mean=tuple(
                            float(value) for value in model.feature_mean
                        ),
                        feature_scale=tuple(
                            float(value) for value in model.feature_scale
                        ),
                        intercept=model.intercept,
                        coefficients=tuple(
                            float(value) for value in model.coefficients
                        ),
                        family_design_hash=design.design_hash,
                        fold_hash=fold_hash,
                    )
                    row_unhashed = {
                        "schema_version": (
                            "midogpp_stage90_case_aware_crossfit_prediction_v1"
                        ),
                        "family_id": family_id,
                        "response_name": response_name,
                        "outer_target_id": predicted_row.outer_target_id,
                        "query_id": predicted_row.query_id,
                        "candidate_source": predicted_row.candidate_source,
                        "predicted_delta": float(predicted),
                        "observed_delta": float(observed[prediction_index]),
                        "predictor_count": design.spec.predictor_count,
                        "training_row_count": len(training_rows),
                        "fold_hash": fold_hash,
                        "response_is_primary": response_name == RESPONSE_NAMES[0],
                        "smooth_response_is_diagnostic_only": (
                            response_name != RESPONSE_NAMES[0]
                        ),
                        "technical_seed_rows_are_independent_observations": False,
                    }
                    prediction_rows.append(
                        CrossfitPredictionRow(
                            family_id=family_id,
                            response_name=response_name,
                            outer_target_id=predicted_row.outer_target_id,
                            query_id=predicted_row.query_id,
                            candidate_source=predicted_row.candidate_source,
                            predicted_delta=float(predicted),
                            observed_delta=float(observed[prediction_index]),
                            predictor_count=design.spec.predictor_count,
                            training_row_count=len(training_rows),
                            fold_hash=fold_hash,
                            row_hash=canonical_sha256(row_unhashed),
                        )
                    )
                    audits.append(audit)

    # Fits were grouped by excluded set, so restore a stable public order.
    family_order = {value: index for index, value in enumerate(selected_families)}
    response_order = {value: index for index, value in enumerate(selected_responses)}
    row_order = {key: index for index, key in enumerate(feature_surface.row_keys)}
    sort_key = lambda value: (
        family_order[value.family_id],
        response_order[value.response_name],
        row_order[value.predicted_row_key if isinstance(value, CrossfitFoldAudit) else value.row_key],
    )
    ordered_predictions = tuple(sorted(prediction_rows, key=sort_key))
    ordered_audits = tuple(sorted(audits, key=sort_key))
    expected_count = (
        len(selected_families)
        * len(selected_responses)
        * len(feature_surface.rows)
    )
    if len(ordered_predictions) != expected_count or len(ordered_audits) != expected_count:
        raise ProtocolError("Crossfit family/response/fold coverage drifted.")
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_crossfit_result_v1",
        "family_ids": list(selected_families),
        "response_names": list(selected_responses),
        "feature_surface_hash": feature_surface.surface_hash,
        "response_surface_hash": response_surface.surface_hash,
        "ordered_prediction_row_hashes": [
            row.row_hash for row in ordered_predictions
        ],
        "ordered_fold_hashes": [row.fold_hash for row in ordered_audits],
        "prediction_row_count": len(ordered_predictions),
        "ridge_alpha": RIDGE_ALPHA,
        "hyperparameter_selection": "none_fixed_predeclared",
        "strict_H_q_e_exclusion_from_all_training_roles": True,
        "expected_training_row_count_from_geometry": expected_training_count,
        "exact_response_is_primary": True,
        "smooth_response_is_diagnostic_only": True,
    }
    return CaseAwareCrossfitResult(
        predictions=ordered_predictions,
        fold_audits=ordered_audits,
        family_ids=selected_families,
        response_names=selected_responses,
        feature_surface_hash=feature_surface.surface_hash,
        response_surface_hash=response_surface.surface_hash,
        result_hash=canonical_sha256(unhashed),
    )


def _selected(
    requested: Sequence[str], allowed: Sequence[str], role: str
) -> tuple[str, ...]:
    values = tuple(requested)
    if not values or len(set(values)) != len(values) or any(
        value not in allowed for value in values
    ):
        raise ProtocolError(f"Crossfit {role} selection drifted from predeclaration.")
    return values


__all__ = ("crossfit_fold_audit_from_payload", "crossfit_proxy_families")
