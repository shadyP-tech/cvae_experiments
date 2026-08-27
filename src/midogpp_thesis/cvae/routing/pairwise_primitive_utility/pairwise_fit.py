"""Nested delete-center model selection and final pairwise ridge refit."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ActionUtilityObservation,
    BaccRankingPolicy,
    CandidatePoolReceipt,
    OpportunityCaseReceipt,
    PairwiseRankerModel,
    SourceScopeReceipt,
    canonical_sha256,
)
from .pairwise_contrasts import action_schema, build_contrasts, canonical_observations
from .pairwise_features import contrast_matrix, design_names, normalization
from .row_posterior_features import assert_label_free_feature_names


PAIRWISE_ALPHA_GRID = (0.1, 1.0, 10.0)


def _fit_coefficients(
    matrix: np.ndarray, response: np.ndarray, weights: np.ndarray, *, alpha: float
) -> np.ndarray:
    normal = matrix.T @ (weights[:, None] * matrix)
    normal.flat[:: normal.shape[0] + 1] += float(alpha)
    rhs = matrix.T @ (weights * response)
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise ProtocolError("Pairwise ridge coefficients are non-finite.")
    return coefficients


def _validate_surface(
    rows: tuple[ActionUtilityObservation, ...],
    scopes: tuple[SourceScopeReceipt, ...],
    candidate_pool: CandidatePoolReceipt,
    opportunity_rows: tuple[OpportunityCaseReceipt, ...],
) -> tuple[str, tuple[str, ...]]:
    receipt_by_case = {(row.center_id, row.case_id): row for row in opportunity_rows}
    if len(receipt_by_case) != len(opportunity_rows):
        raise ProtocolError("Pairwise opportunity-case receipts are duplicated.")
    if (
        len(scopes) < 2
        or any(not isinstance(scope, SourceScopeReceipt) for scope in scopes)
        or len({scope.hyperparameter_center for scope in scopes}) != len(scopes)
    ):
        raise ProtocolError("Pairwise alpha selection requires at least two unique held-K scopes.")
    outer_targets = {scope.outer_target_center for scope in scopes}
    if len(outer_targets) != 1:
        raise ProtocolError("Nested pairwise scopes must share one outer target H.")
    outer_target = next(iter(outer_targets))
    if outer_target != candidate_pool.outer_target_center:
        raise ProtocolError("Pairwise H drifted from the candidate-pool receipt.")
    if tuple(sorted(scope.hyperparameter_center for scope in scopes)) != candidate_pool.candidate_center_ids:
        raise ProtocolError("Nested pairwise K folds do not cover the exact C-minus-H pool.")
    if any(row.center_id == outer_target for row in rows):
        raise ProtocolError("Pairwise source surface contains forbidden outer-target H rows.")
    known_centers = {row.center_id for row in rows}
    if any(
        not {
            scope.query_center,
            scope.hyperparameter_center,
            scope.calibration_center,
            *scope.training_center_ids,
        }.issubset(known_centers)
        for scope in scopes
    ):
        raise ProtocolError("Nested pairwise scope references an absent source center.")
    if (
        {row.source_scope_receipt_hash for row in rows}
        != {candidate_pool.source_surface_receipt_hash}
        or {row.candidate_pool_receipt_hash for row in rows} != {candidate_pool.receipt_hash}
    ):
        raise ProtocolError("Pairwise source utility surface lineage is not singular.")
    final_centers = tuple(sorted(known_centers))
    final_case_keys = tuple(sorted({(row.center_id, row.case_id) for row in rows}))
    if len(final_centers) < 4 or len(final_case_keys) < 3:
        raise ProtocolError("Pairwise ranking needs at least four source centers and three cases.")
    if final_centers != candidate_pool.candidate_center_ids:
        raise ProtocolError("Pairwise source surface is not the exact C-minus-H candidate pool.")
    if set(receipt_by_case) != set(final_case_keys):
        raise ProtocolError("Pairwise opportunity receipts do not exactly cover source cases.")
    frozen_inventories = {receipt.candidate_action_ids for receipt in opportunity_rows}
    if len(frozen_inventories) != 1:
        raise ProtocolError("Pairwise opportunity receipts drifted from one frozen action inventory.")
    for key, receipt in receipt_by_case.items():
        case_rows = tuple(row for row in rows if (row.center_id, row.case_id) == key)
        if (
            {row.action_id for row in case_rows} != set(receipt.active_representative_ids)
            or {row.opportunity_case_receipt_hash for row in case_rows} != {receipt.receipt_hash}
            or any(
                row.response.action_id != row.action_id
                or row.response.baseline_probability_hash
                != receipt.opportunity.baseline_hash
                or row.response.candidate_probability_hash
                != receipt.opportunity.member(row.action_id).probability_hash
                for row in case_rows
            )
        ):
            raise ProtocolError(
                "Pairwise rows include no-op, duplicate, or cross-surface action utilities."
            )
    response_groups: dict[tuple[str, str], list[ActionUtilityObservation]] = defaultdict(list)
    for row in rows:
        response_groups[(row.center_id, row.case_id)].append(row)
    for group in response_groups.values():
        lineage = {
            (
                row.response.denominator_scope_id,
                row.response.denominator_eta_hash,
                row.response.row_manifest_hash,
                row.response.posterior_model_hash,
                row.response.posterior_scope_receipt_hash,
            )
            for row in group
        }
        if len(lineage) != 1:
            raise ProtocolError("Pairwise action responses have mismatched denominator lineage.")
    return outer_target, next(iter(frozen_inventories))


def fit_pairwise_ranker(
    observations: Sequence[ActionUtilityObservation],
    *,
    delete_center_scopes: Sequence[SourceScopeReceipt],
    candidate_pool: CandidatePoolReceipt,
    opportunity_receipts: Sequence[OpportunityCaseReceipt],
    ranking_policy: BaccRankingPolicy,
) -> PairwiseRankerModel:
    """Select alpha by nested worst-center K validation, then refit C-minus-H."""

    rows = canonical_observations(observations)
    if not isinstance(candidate_pool, CandidatePoolReceipt):
        raise ProtocolError("Pairwise fitting requires a typed candidate-pool receipt.")
    if not isinstance(ranking_policy, BaccRankingPolicy):
        raise ProtocolError("Pairwise fitting requires the frozen BACC ranking policy.")
    scopes = tuple(delete_center_scopes)
    opportunity_rows = tuple(opportunity_receipts)
    outer_target, candidate_action_ids = _validate_surface(
        rows, scopes, candidate_pool, opportunity_rows
    )

    feature_names = assert_label_free_feature_names(rows[0].feature_names)
    if any(row.feature_names != feature_names for row in rows):
        raise ProtocolError("Pairwise feature schema drifted across source rows.")
    schema = action_schema(rows)
    all_action_ids = {row.action_id for row in rows}
    names = design_names(feature_names, schema)

    fold_losses: list[tuple[float, str, float]] = []
    for scope in sorted(scopes, key=lambda value: value.hyperparameter_center):
        training_centers = set(scope.training_center_ids)
        held_d = (scope.heldout_case_center, scope.heldout_case_id)
        fold_rows = tuple(
            row
            for row in rows
            if row.center_id in training_centers and (row.center_id, row.case_id) != held_d
        )
        held_rows = tuple(
            row
            for row in rows
            if row.center_id == scope.hyperparameter_center
            and (row.center_id, row.case_id) != held_d
        )
        if (
            tuple(sorted({row.center_id for row in fold_rows})) != scope.training_center_ids
            or tuple(sorted({(row.center_id, row.case_id) for row in fold_rows}))
            != scope.training_case_keys
            or not held_rows
            or {row.action_id for row in fold_rows} != all_action_ids
            or {row.action_id for row in held_rows} != all_action_ids
        ):
            raise ProtocolError("Nested pairwise K fold is incomplete or drifted from its receipt.")
        fold_mean, fold_scale = normalization(fold_rows)
        held_contrasts = build_contrasts(held_rows)
        fold_matrix, fold_response, fold_weights = contrast_matrix(
            build_contrasts(fold_rows),
            feature_names=feature_names,
            mean=fold_mean,
            scale=fold_scale,
            action_schema=schema,
            design_names=names,
        )
        held_matrix, held_response, held_weights = contrast_matrix(
            held_contrasts,
            feature_names=feature_names,
            mean=fold_mean,
            scale=fold_scale,
            action_schema=schema,
            design_names=names,
        )
        for alpha in PAIRWISE_ALPHA_GRID:
            coefficients = _fit_coefficients(
                fold_matrix, fold_response, fold_weights, alpha=alpha
            )
            residual = held_response - held_matrix @ coefficients
            loss = float(
                np.dot(held_weights, residual * residual)
                / np.sum(held_weights, dtype=np.float64)
            )
            fold_losses.append((alpha, scope.hyperparameter_center, loss))

    summaries: list[tuple[float, float, float]] = []
    for alpha in PAIRWISE_ALPHA_GRID:
        values = tuple(loss for candidate, _, loss in fold_losses if candidate == alpha)
        if len(values) != len(scopes):
            raise ProtocolError("Nested pairwise alpha surface is incomplete.")
        summaries.append((alpha, max(values), float(np.mean(values, dtype=np.float64))))
    selected_alpha = min(summaries, key=lambda row: (row[1], row[2], row[0]))[0]

    mean, scale = normalization(rows)
    training_contrasts = build_contrasts(rows)
    matrix, response, weights = contrast_matrix(
        training_contrasts,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        action_schema=schema,
        design_names=names,
    )
    coefficients = _fit_coefficients(matrix, response, weights, alpha=selected_alpha)
    combined_receipt_hash = canonical_sha256(
        {
            "schema": "pairwise_nested_delete_center_receipt_v2",
            "outer_target_H": outer_target,
            "source_surface_hash": candidate_pool.source_surface_receipt_hash,
            "fold_receipt_hashes": tuple(sorted(scope.receipt_hash for scope in scopes)),
            "fixed_alpha_grid": PAIRWISE_ALPHA_GRID,
            "selection_rule": "min_worst_then_center_mean_then_alpha",
            "final_refit_centers": tuple(sorted({row.center_id for row in rows})),
            "candidate_action_ids": candidate_action_ids,
            "target_labels_used": False,
            "opportunity_receipt_hashes": tuple(
                sorted(receipt.receipt_hash for receipt in opportunity_rows)
            ),
            "bacc_ranking_policy_hash": ranking_policy.policy_hash,
        }
    )
    return PairwiseRankerModel(
        feature_names=feature_names,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        action_schema=schema,
        candidate_action_ids=candidate_action_ids,
        design_names=names,
        coefficients=tuple(float(value) for value in coefficients),
        selected_alpha=selected_alpha,
        alpha_grid=PAIRWISE_ALPHA_GRID,
        delete_center_losses=tuple(sorted(fold_losses)),
        alpha_selection_summary=tuple(summaries),
        training_center_ids=tuple(sorted({row.center_id for row in rows})),
        training_case_count=len({(row.center_id, row.case_id) for row in rows}),
        training_contrast_count=len(training_contrasts),
        source_scope_receipt_hash=combined_receipt_hash,
        candidate_pool_receipt_hash=candidate_pool.receipt_hash,
        opportunity_surface_receipt_hash=canonical_sha256(
            tuple(sorted(receipt.receipt_hash for receipt in opportunity_rows))
        ),
        bacc_ranking_policy_hash=ranking_policy.policy_hash,
    )


__all__ = ("PAIRWISE_ALPHA_GRID", "fit_pairwise_ranker")
