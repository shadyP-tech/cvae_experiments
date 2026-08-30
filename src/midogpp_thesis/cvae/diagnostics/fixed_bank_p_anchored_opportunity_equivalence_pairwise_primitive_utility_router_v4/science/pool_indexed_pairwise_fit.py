"""Pool-indexed nested pairwise fitting for source-held OE-PPUR surfaces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ....routing.pairwise_primitive_utility import (
    ActionUtilityObservation,
    ActionQuery,
    BaccRankingPolicy,
    OpportunityCaseReceipt,
    PairwiseRankerModel,
    SourceScopeReceipt,
    assert_label_free_feature_names,
    canonical_sha256,
)
from ....routing.pairwise_primitive_utility.pairwise_contrasts import (
    action_schema,
    build_contrasts,
    canonical_observations,
)
from ....routing.pairwise_primitive_utility.pairwise_features import (
    contrast_matrix,
    design_names,
    feature_vector,
    normalization,
)
from ..candidate_pools import (
    CANDIDATE_ACTION_IDS,
    FinalOuterCandidatePoolReceipt,
    HeldCenterCandidatePoolReceipt,
    PoolInvariantActionCompilerReceipt,
    validate_complete_pool_lineage,
)


PAIRWISE_ALPHA_GRID = (0.1, 1.0, 10.0)
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class HeldLActionQuery:
    """One label-free action query whose source center is genuinely held L."""

    center_id: str
    case_id: str
    query: ActionQuery

    def __post_init__(self) -> None:
        if not self.center_id or not self.case_id or not isinstance(self.query, ActionQuery):
            raise ProtocolError("OE-PPUR v4 held-L action query is untyped.")


@dataclass(frozen=True, slots=True)
class HeldLActionPrediction:
    center_id: str
    case_id: str
    action_id: str
    predicted_score: float
    source_scope_receipt_hash: str
    model_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        score = float(self.predicted_score)
        scope_hash = str(self.source_scope_receipt_hash).strip().lower()
        model_hash = str(self.model_hash).strip().lower()
        if (
            not self.center_id
            or not self.case_id
            or self.action_id not in CANDIDATE_ACTION_IDS
            or not np.isfinite(score)
            or _SHA256.fullmatch(scope_hash) is None
            or _SHA256.fullmatch(model_hash) is None
        ):
            raise ProtocolError("OE-PPUR v4 held-L action prediction drifted.")
        object.__setattr__(self, "predicted_score", score)
        object.__setattr__(self, "source_scope_receipt_hash", scope_hash)
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(self, "prediction_hash", canonical_sha256({
            "schema": "oe_ppur_v4_genuine_held_L_action_prediction_v1",
            "center_id": self.center_id,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "predicted_score": score,
            "source_scope_receipt_hash": scope_hash,
            "model_hash": model_hash,
        }))


def _fit_coefficients(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    normal = matrix.T @ (weights[:, None] * matrix)
    normal.flat[:: normal.shape[0] + 1] += float(alpha)
    rhs = matrix.T @ (weights * response)
    try:
        result = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        result = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    if not np.isfinite(result).all():
        raise ProtocolError("OE-PPUR v4 pairwise ridge coefficients are non-finite.")
    return result


def _validate_pool_indexed_surface(
    rows: tuple[ActionUtilityObservation, ...],
    scopes: tuple[SourceScopeReceipt, ...],
    held_pools: tuple[HeldCenterCandidatePoolReceipt, ...],
    final_pool: FinalOuterCandidatePoolReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
    opportunities: tuple[OpportunityCaseReceipt, ...],
    source_surface_lineage_hash: str,
) -> tuple[str, tuple[str, ...], str]:
    validate_complete_pool_lineage(
        held_pools, final_pool=final_pool, compiler=compiler
    )
    h = final_pool.outer_target_center
    source_centers = final_pool.candidate_center_ids
    pool_by_q = {pool.held_center: pool for pool in held_pools}
    receipt_by_case = {
        (receipt.center_id, receipt.case_id): receipt for receipt in opportunities
    }
    if (
        len(receipt_by_case) != len(opportunities)
        or len(scopes) != len(source_centers)
        or any(not isinstance(scope, SourceScopeReceipt) for scope in scopes)
        or tuple(sorted(scope.hyperparameter_center for scope in scopes))
        != source_centers
        or any(scope.outer_target_center != h for scope in scopes)
        or tuple(sorted({row.center_id for row in rows})) != source_centers
        or any(row.center_id == h for row in rows)
        or {row.action_id for row in rows} != set(CANDIDATE_ACTION_IDS)
        or {row.source_scope_receipt_hash for row in rows}
        != {source_surface_lineage_hash}
    ):
        raise ProtocolError("OE-PPUR v4 pool-indexed source surface topology drifted.")
    for row in rows:
        receipt = receipt_by_case.get((row.center_id, row.case_id))
        pool = pool_by_q.get(row.center_id)
        if (
            receipt is None
            or pool is None
            or row.candidate_pool_receipt_hash != pool.receipt_hash
            or row.opportunity_case_receipt_hash != receipt.receipt_hash
            or row.action_id not in receipt.active_representative_ids
            or row.response.action_id != row.action_id
            or row.response.baseline_probability_hash
            != receipt.opportunity.baseline_hash
            or row.response.candidate_probability_hash
            != receipt.opportunity.member(row.action_id).probability_hash
        ):
            raise ProtocolError("OE-PPUR v4 row drifted from held-q pool/opportunity lineage.")
    inventories = {receipt.candidate_action_ids for receipt in opportunities}
    if inventories != {CANDIDATE_ACTION_IDS}:
        raise ProtocolError("OE-PPUR v4 opportunity inventory is not the frozen six actions.")
    grouped: dict[tuple[str, str], list[ActionUtilityObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.center_id, row.case_id)].append(row)
    for group in grouped.values():
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
            raise ProtocolError("OE-PPUR v4 action utilities mixed denominator lineage.")
    active_by_case = {
        (receipt.center_id, receipt.case_id): receipt.active_representative_ids
        for receipt in opportunities
    }
    if (
        set(grouped) != {key for key, active in active_by_case.items() if active}
        or any(
            tuple(sorted(row.action_id for row in grouped[key])) != active
            for key, active in active_by_case.items()
            if active
        )
    ):
        raise ProtocolError("OE-PPUR v4 active opportunity pruning inventory drifted.")
    pruning_hash = canonical_sha256({
        "schema": "oe_ppur_v4_pool_indexed_opportunity_pruning_v1",
        "cases": tuple(
            (center, case, active_by_case[(center, case)])
            for center, case in sorted(active_by_case)
        ),
        "structural_noops_only": True,
        "labels_used": False,
    })
    return h, CANDIDATE_ACTION_IDS, pruning_hash


def fit_pool_indexed_pairwise_ranker(
    observations: Sequence[ActionUtilityObservation],
    *,
    delete_center_scopes: Sequence[SourceScopeReceipt],
    held_pool_receipts: Sequence[HeldCenterCandidatePoolReceipt],
    final_pool_receipt: FinalOuterCandidatePoolReceipt,
    compiler: PoolInvariantActionCompilerReceipt,
    opportunity_receipts: Sequence[OpportunityCaseReceipt],
    ranking_policy: BaccRankingPolicy,
    source_surface_lineage_hash: object,
) -> PairwiseRankerModel:
    """Select alpha across K while respecting each observation's ``C\\{H,q}``."""

    rows = canonical_observations(observations)
    scopes = tuple(delete_center_scopes)
    held_pools = tuple(held_pool_receipts)
    opportunities = tuple(opportunity_receipts)
    lineage_hash = str(source_surface_lineage_hash).strip()
    if (
        not isinstance(final_pool_receipt, FinalOuterCandidatePoolReceipt)
        or not isinstance(compiler, PoolInvariantActionCompilerReceipt)
        or not isinstance(ranking_policy, BaccRankingPolicy)
        or _SHA256.fullmatch(lineage_hash) is None
    ):
        raise ProtocolError("OE-PPUR v4 pairwise fitting requires typed frozen lineage.")
    h, candidate_action_ids, opportunity_pruning_hash = _validate_pool_indexed_surface(
        rows,
        scopes,
        held_pools,
        final_pool_receipt,
        compiler,
        opportunities,
        lineage_hash,
    )
    feature_names = assert_label_free_feature_names(rows[0].feature_names)
    if any(row.feature_names != feature_names for row in rows):
        raise ProtocolError("OE-PPUR v4 pairwise feature schema drifted.")
    schema = action_schema(rows)
    names = design_names(feature_names, schema)
    all_actions = set(CANDIDATE_ACTION_IDS)

    fold_losses: list[tuple[float, str, float]] = []
    for scope in sorted(scopes, key=lambda row: row.hyperparameter_center):
        held_d = (scope.heldout_case_center, scope.heldout_case_id)
        fold_rows = tuple(
            row for row in rows
            if row.center_id in set(scope.training_center_ids)
            and (row.center_id, row.case_id) != held_d
        )
        held_rows = tuple(
            row for row in rows
            if row.center_id == scope.hyperparameter_center
            and (row.center_id, row.case_id) != held_d
        )
        observed_training_cases = {
            (row.center_id, row.case_id) for row in fold_rows
        }
        expected_training_cases = {
            (receipt.center_id, receipt.case_id)
            for receipt in opportunities
            if receipt.active_representative_ids
            and receipt.center_id in set(scope.training_center_ids)
            and (receipt.center_id, receipt.case_id) != held_d
        }
        expected_held_cases = {
            (receipt.center_id, receipt.case_id)
            for receipt in opportunities
            if receipt.active_representative_ids
            and receipt.center_id == scope.hyperparameter_center
            and (receipt.center_id, receipt.case_id) != held_d
        }
        if (
            tuple(sorted({row.center_id for row in fold_rows}))
            != scope.training_center_ids
            or observed_training_cases != expected_training_cases
            or not observed_training_cases.issubset(set(scope.training_case_keys))
            or {(row.center_id, row.case_id) for row in held_rows} != expected_held_cases
            or not held_rows
            or {row.action_id for row in fold_rows} != all_actions
            or {row.action_id for row in held_rows} != all_actions
        ):
            raise ProtocolError("OE-PPUR v4 nested K fold is incomplete; seal H to P.")
        fold_mean, fold_scale = normalization(fold_rows)
        fold_matrix, fold_response, fold_weights = contrast_matrix(
            build_contrasts(fold_rows),
            feature_names=feature_names,
            mean=fold_mean,
            scale=fold_scale,
            action_schema=schema,
            design_names=names,
        )
        held_matrix, held_response, held_weights = contrast_matrix(
            build_contrasts(held_rows),
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

    summaries = []
    for alpha in PAIRWISE_ALPHA_GRID:
        values = tuple(loss for candidate, _center, loss in fold_losses if candidate == alpha)
        if len(values) != len(scopes):
            raise ProtocolError("OE-PPUR v4 nested alpha surface is incomplete.")
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
    coefficients = _fit_coefficients(
        matrix, response, weights, alpha=selected_alpha
    )
    combined_scope_hash = canonical_sha256(
        {
            "schema": "oe_ppur_v4_pool_indexed_pairwise_fit_v1",
            "H": h,
            "source_surface_lineage_hash": lineage_hash,
            "held_pool_receipt_hashes": tuple(
                pool.receipt_hash for pool in held_pools
            ),
            "final_pool_receipt_hash": final_pool_receipt.receipt_hash,
            "compiler_receipt_hash": compiler.receipt_hash,
            "fold_receipt_hashes": tuple(sorted(scope.receipt_hash for scope in scopes)),
            "alpha_grid": PAIRWISE_ALPHA_GRID,
            "selection_rule": "min_worst_then_mean_then_alpha",
            "candidate_action_ids": candidate_action_ids,
            "opportunity_pruning_receipt_hash": opportunity_pruning_hash,
            "target_labels_used": False,
        }
    )
    neutral_final = final_pool_receipt.to_neutral()
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
        source_scope_receipt_hash=combined_scope_hash,
        candidate_pool_receipt_hash=neutral_final.receipt_hash,
        opportunity_surface_receipt_hash=canonical_sha256(
            tuple(sorted(receipt.receipt_hash for receipt in opportunities))
        ),
        bacc_ranking_policy_hash=ranking_policy.policy_hash,
    )


def fit_genuine_held_l_action_predictions(
    observations: Sequence[ActionUtilityObservation],
    queries: Sequence[HeldLActionQuery],
    *,
    calibration_scopes: Sequence[SourceScopeReceipt],
    selected_alpha: float,
) -> tuple[HeldLActionPrediction, ...]:
    """Fit one fixed-alpha source model without each L and predict all L actions."""

    rows = canonical_observations(observations)
    query_rows = tuple(sorted(tuple(queries), key=lambda row: (row.center_id, row.case_id, row.query.action_id)))
    scopes = tuple(calibration_scopes)
    if selected_alpha not in PAIRWISE_ALPHA_GRID or not query_rows or len(scopes) < 4:
        raise ProtocolError("OE-PPUR v4 held-L prediction surface is incomplete.")
    query_keys = tuple((row.center_id, row.case_id, row.query.action_id) for row in query_rows)
    if len(set(query_keys)) != len(query_keys):
        raise ProtocolError("OE-PPUR v4 held-L queries are duplicated.")
    feature_names = assert_label_free_feature_names(rows[0].feature_names)
    schema = action_schema(rows)
    if {action for action, _family, _direction in schema} != set(CANDIDATE_ACTION_IDS):
        raise ProtocolError("OE-PPUR v4 held-L fitting lacks the frozen action schema.")
    names = design_names(feature_names, schema)
    output: list[HeldLActionPrediction] = []
    seen_l: set[str] = set()
    for scope in sorted(scopes, key=lambda row: row.calibration_center):
        ell = scope.calibration_center
        if ell in seen_l:
            raise ProtocolError("OE-PPUR v4 held-L scopes duplicate calibration centers.")
        seen_l.add(ell)
        training = tuple(row for row in rows if row.center_id in set(scope.training_center_ids))
        held_queries = tuple(row for row in query_rows if row.center_id == ell)
        expected_training_cases = {
            (row.center_id, row.case_id)
            for row in rows
            if row.center_id in set(scope.training_center_ids)
        }
        if (
            not training
            or not held_queries
            or {(row.center_id, row.case_id) for row in training} != expected_training_cases
            or not expected_training_cases.issubset(set(scope.training_case_keys))
            or {row.action_id for row in training} != set(CANDIDATE_ACTION_IDS)
        ):
            raise ProtocolError("OE-PPUR v4 genuine held-L fit scope is incomplete.")
        mean, scale = normalization(training)
        matrix, response, weights = contrast_matrix(
            build_contrasts(training),
            feature_names=feature_names,
            mean=mean,
            scale=scale,
            action_schema=schema,
            design_names=names,
        )
        coefficients = _fit_coefficients(matrix, response, weights, alpha=selected_alpha)
        model_hash = canonical_sha256({
            "schema": "oe_ppur_v4_genuine_held_L_fixed_alpha_model_v1",
            "scope_receipt_hash": scope.receipt_hash,
            "selected_alpha": selected_alpha,
            "feature_names": feature_names,
            "feature_mean": tuple(float(value) for value in mean),
            "feature_scale": tuple(float(value) for value in scale),
            "action_schema": schema,
            "design_names": names,
            "coefficients": tuple(float(value) for value in coefficients),
            "held_L": ell,
        })
        for held in held_queries:
            vector = feature_vector(
                held.query,
                feature_names=feature_names,
                mean=mean,
                scale=scale,
                action_schema=schema,
                design_names=names,
            )
            output.append(HeldLActionPrediction(
                center_id=held.center_id,
                case_id=held.case_id,
                action_id=held.query.action_id,
                predicted_score=float(vector @ coefficients),
                source_scope_receipt_hash=scope.receipt_hash,
                model_hash=model_hash,
            ))
    expected_centers = {scope.calibration_center for scope in scopes}
    if seen_l != expected_centers or {row.center_id for row in output} != expected_centers:
        raise ProtocolError("OE-PPUR v4 held-L prediction coverage drifted.")
    return tuple(sorted(output, key=lambda row: (row.center_id, row.case_id, row.action_id)))


__all__ = (
    "PAIRWISE_ALPHA_GRID",
    "HeldLActionPrediction",
    "HeldLActionQuery",
    "fit_genuine_held_l_action_predictions",
    "fit_pool_indexed_pairwise_ranker",
)
