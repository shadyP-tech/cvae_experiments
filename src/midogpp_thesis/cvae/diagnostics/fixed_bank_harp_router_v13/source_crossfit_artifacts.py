"""Deterministic artifact payloads for the HARP v13 source crossfit.

This module owns construction of the label-free, prelabel-prediction,
development, and fitted-model artifacts.  It also owns the durable
physical-surface round trip that must complete before an effective adapter is
built.  Label access and fold orchestration remain outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.harp_v13_execution.contracts import ArtifactValue
from ...runtime.harp_v13_execution.crossfit_contracts import (
    FoldConditionedSourceSurface,
)
from ...runtime.harp_v13_execution.crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
    persist_source_crossfit_surface,
    reconstruct_source_crossfit_surface,
)
from ...runtime.harp_v13_execution.model_adapter import (
    RouterFitState,
    model_manifest,
)
from ...runtime.harp_v13_execution.source_development import SourceDevelopmentState
from .source_crossfit_fold_store import SourceCrossfitFoldSealSet
from .source_label_capability import AggregateSourceLabelCapability

if TYPE_CHECKING:
    from .source_crossfit_orchestration import LabelFreeSourceCrossfitBundle
    from .fold_outcome_universes import ExactFoldOutcomeUniverseSet


def persist_and_reconstruct_source_crossfit_surface(
    durable_root: Path,
    physical: FoldConditionedSourceSurface,
) -> tuple[FoldConditionedSourceSurface, SourceCrossfitSurfaceReceipt]:
    """Persist and freshly reconstruct one complete label-free surface."""

    written_receipt = persist_source_crossfit_surface(Path(durable_root), physical)
    reconstructed, receipt = reconstruct_source_crossfit_surface(
        Path(durable_root), expected_surface_hash=physical.surface_hash
    )
    if (
        receipt.receipt_hash != written_receipt.receipt_hash
        or reconstructed.surface_hash != physical.surface_hash
    ):
        raise ProtocolError(
            "HARP v13 durable source-crossfit reconstruction changed identity."
        )
    return reconstructed, receipt


def build_source_crossfit_effective_artifact(
    bundle: LabelFreeSourceCrossfitBundle,
) -> ArtifactValue:
    """Project fold-effective membership without duplicating physical bytes."""

    rows: list[dict[str, object]] = []
    features: list[tuple[float, ...]] = []
    offsets = [0]
    for wrapper in bundle.effective_surface.menus:
        menu = wrapper.menu
        for action in menu.actions:
            features.append(action.feature_values)
        offsets.append(len(features))
        rows.append(
            {
                "outer_target_id": wrapper.outer_target_id,
                "heldout_center_id": wrapper.heldout_center_id,
                "current_query_center_id": wrapper.current_query_center_id,
                "case_id": menu.case_id,
                "candidate_source_ids": list(wrapper.candidate_source_ids),
                "fold_menu_hash": wrapper.fold_menu_hash,
                "effective_menu_hash": menu.menu_hash,
                "physical_block_hashes": list(wrapper.physical_block_hashes),
                "compatibility_receipt_hashes": list(
                    wrapper.compatibility_receipt_hashes
                ),
                "action_ids": [action.action_id for action in menu.actions],
                "action_hashes": [action.action_hash for action in menu.actions],
                "prediction_fold": wrapper.prediction_fold,
            }
        )
    body = {
        "schema_version": "midogpp_harp_v13_source_crossfit_effective_menu_store_v1",
        "source_surface_hash": bundle.physical_surface.surface_hash,
        "source_surface_receipt_hash": bundle.surface_receipt.receipt_hash,
        "effective_adapter_hash": bundle.effective_surface.adapter_hash,
        "rows": rows,
        "fold_menu_count": len(rows),
        "effective_action_count": len(features),
        "physical_probability_bytes_duplicated": False,
        "labels_consumed": False,
    }
    width = len(features[0]) if features else 0
    return ArtifactValue(
        state=bundle.effective_surface,
        manifest={**body, "effective_menu_hash": canonical_hash(body)},
        arrays={
            "effective_action_features": np.asarray(features, dtype=np.float64).reshape(
                (-1, width)
            ),
            "effective_action_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def build_source_prelabel_prediction_artifact(
    fold_seal_set: SourceCrossfitFoldSealSet,
) -> ArtifactValue:
    """Compact all durable heldout-q predictions into one catalogued store."""

    predictions = tuple(
        prediction
        for seal in fold_seal_set.fold_seals
        for prediction in seal.nested_fold.heldout_predictions
    )
    score_rows = [
        (
            score.pairwise_score,
            score.predicted_budget_gain,
            score.predicted_allocation_gain,
            score.predicted_total_gain,
            score.predicted_harm_probability,
            score.predicted_brier_delta,
            score.predicted_log_delta,
            score.acceptance_probability,
            float(score.model_available),
        )
        for prediction in predictions
        for score in prediction.action_scores
    ]
    offsets = [0]
    for prediction in predictions:
        offsets.append(offsets[-1] + len(prediction.action_scores))
    body = {
        "schema_version": "midogpp_harp_v13_source_prelabel_q_prediction_store_v1",
        "source_surface_receipt_hash": fold_seal_set.source_surface_receipt_hash,
        "source_surface_hash": fold_seal_set.source_surface_hash,
        "effective_adapter_hash": fold_seal_set.effective_adapter_hash,
        "fold_menu_binding_certificate_hash": (
            fold_seal_set.fold_menu_binding_certificate_hash
        ),
        "fold_menu_binding_certificate_receipt_hash": (
            fold_seal_set.fold_menu_binding_certificate_receipt_hash
        ),
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "fold_seal_hashes": [row.seal_hash for row in fold_seal_set.fold_seals],
        "label_free_case_inventory_hashes": [
            row.label_free_case_inventory_hash for row in fold_seal_set.fold_seals
        ],
        "exact_b_control_count_across_fold_views": sum(
            row.exact_b_control_count for row in fold_seal_set.fold_seals
        ),
        "active_menu_count_across_fold_views": sum(
            row.active_menu_count for row in fold_seal_set.fold_seals
        ),
        "prediction_rows": [row.public_payload() for row in predictions],
        "prediction_count": len(predictions),
        "heldout_q_outcomes_consumed_by_own_fold": False,
        "label_free_case_inventories_sealed_before_q_outcomes": True,
        "aggregate_source_labels_opened": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=fold_seal_set,
        manifest={**body, "prediction_store_hash": canonical_hash(body)},
        arrays={
            "prediction_values": np.asarray(
                [
                    (
                        row.acceptance_probability,
                        row.rank_margin,
                        float(len(row.action_scores)),
                        float(row.top_action_id != "B"),
                    )
                    for row in predictions
                ],
                dtype=np.float64,
            ).reshape((-1, 4)),
            "action_score_values": np.asarray(score_rows, dtype=np.float64).reshape(
                (-1, 9)
            ),
            "action_score_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def source_fold_capability_seal_payload(
    fold_seal_set: SourceCrossfitFoldSealSet,
) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v13_source_fold_label_capability_seals_v1",
        "source_surface_receipt_hash": fold_seal_set.source_surface_receipt_hash,
        "source_surface_hash": fold_seal_set.source_surface_hash,
        "effective_adapter_hash": fold_seal_set.effective_adapter_hash,
        "fold_menu_binding_certificate_hash": (
            fold_seal_set.fold_menu_binding_certificate_hash
        ),
        "fold_menu_binding_certificate_receipt_hash": (
            fold_seal_set.fold_menu_binding_certificate_receipt_hash
        ),
        "folds": [
            {
                "outer_target_id": seal.outer_target_id,
                "heldout_center_id": seal.heldout_center_id,
                "allowed_center_ids": [
                    center
                    for center in CENTERS
                    if center not in {seal.outer_target_id, seal.heldout_center_id}
                ],
                "excluded_center_ids": [
                    seal.outer_target_id,
                    seal.heldout_center_id,
                ],
                "label_capability_hash": seal.label_capability_hash,
                "prediction_surface_hash": seal.prediction_surface_hash,
                "fitting_surface_hash": seal.fitting_surface_hash,
                "isolation_receipt_hash": seal.isolation_receipt_hash,
                "fold_menu_binding_hash": seal.fold_menu_binding_hash,
                "label_free_case_inventory_hash": (
                    seal.label_free_case_inventory_hash
                ),
                "exact_b_control_count": seal.exact_b_control_count,
                "active_menu_count": seal.active_menu_count,
            }
            for seal in fold_seal_set.fold_seals
        ],
        "fold_count": len(fold_seal_set.fold_seals),
        "label_free_case_inventory_count": len(fold_seal_set.fold_seals),
        "heldout_q_label_shard_unauthorized_and_not_opened_by_typed_loader_in_own_H_q_worker": True,
        "global_source_label_open_order_claimed": False,
        "evaluation_labels_authorized": False,
    }
    return {**body, "seal_hash": canonical_hash(body)}


def source_prelabel_q_prediction_seal_payload(
    fold_seal_set: SourceCrossfitFoldSealSet,
    prediction_artifact: ArtifactValue,
    *,
    store_manifest_sha256: str,
    store_npz_sha256: str,
) -> dict[str, object]:
    body = {
        "schema_version": "midogpp_harp_v13_source_prelabel_q_prediction_seal_v1",
        "source_surface_receipt_hash": fold_seal_set.source_surface_receipt_hash,
        "source_surface_hash": fold_seal_set.source_surface_hash,
        "effective_adapter_hash": fold_seal_set.effective_adapter_hash,
        "fold_menu_binding_certificate_hash": (
            fold_seal_set.fold_menu_binding_certificate_hash
        ),
        "fold_menu_binding_certificate_receipt_hash": (
            fold_seal_set.fold_menu_binding_certificate_receipt_hash
        ),
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "prediction_store_hash": prediction_artifact.manifest[
            "prediction_store_hash"
        ],
        "prediction_store_manifest_sha256": store_manifest_sha256,
        "prediction_store_npz_sha256": store_npz_sha256,
        "fold_seal_hashes": [row.seal_hash for row in fold_seal_set.fold_seals],
        "label_free_case_inventory_hashes": [
            row.label_free_case_inventory_hash for row in fold_seal_set.fold_seals
        ],
        "label_free_case_inventories_durable_before_source_outcome_join": True,
        "pseudo_target_q_predictions_sealed_before_q_outcomes_joined_to_same_fold": True,
        "aggregate_source_labels_opened": False,
        "evaluation_labels_opened": False,
    }
    return {**body, "seal_hash": canonical_hash(body)}


def build_development_artifact(
    state: SourceDevelopmentState,
    *,
    config_hash: str,
    bundle: LabelFreeSourceCrossfitBundle,
    fold_seal_set: SourceCrossfitFoldSealSet,
    aggregate_capability: AggregateSourceLabelCapability,
    exact_fold_outcome_universes: ExactFoldOutcomeUniverseSet,
) -> ArtifactValue:
    outcomes = state.outcomes
    body = {
        "schema_version": "midogpp_harp_v13_source_train_crossfit_development_surface_v1",
        "config_hash": config_hash,
        "outer_targets": list(CENTERS),
        "expected_center_ids": list(CENTERS),
        "source_surface_hash": bundle.physical_surface.surface_hash,
        "source_surface_receipt_hash": bundle.surface_receipt.receipt_hash,
        "effective_adapter_hash": bundle.effective_surface.adapter_hash,
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "fold_menu_binding_certificate_hash": (
            fold_seal_set.fold_menu_binding_certificate_hash
        ),
        "fold_menu_binding_certificate_receipt_hash": (
            fold_seal_set.fold_menu_binding_certificate_receipt_hash
        ),
        "aggregate_source_label_capability_hash": aggregate_capability.capability_hash,
        "exact_fold_outcome_universe_set_hash": (
            exact_fold_outcome_universes.set_hash
        ),
        "exact_fold_outcome_universe_hashes": [
            [
                row.outer_target_id,
                row.heldout_center_id,
                row.universe.universe_hash,
                row.binding_hash,
            ]
            for row in exact_fold_outcome_universes.folds
        ],
        "observation_count": len(outcomes),
        "effective_menu_count": len(state.effective_menus),
        "source_response_hashes": [row.outcome_hash for row in outcomes],
        "effective_menu_hashes": [row.menu_hash for row in state.effective_menus],
        "strict_outer_H_and_fold_local_q_exclusion": True,
        "q_predictions_presealed_before_same_q_outcomes_joined": True,
        "exact_H_q_r_outcomes_created_only_after_full_source_labels_opened": True,
        "posthoc_H_r_r_outcome_projection_used": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=state,
        manifest={**body, "surface_hash": canonical_hash(body)},
        arrays={
            "feature_values": np.asarray(
                [row.action.feature_values for row in outcomes], dtype=np.float64
            ),
            "endpoint_effects": np.asarray(
                [(row.bacc_gain, row.brier_delta, row.log_delta) for row in outcomes],
                dtype=np.float64,
            ),
        },
    )


def build_model_artifact(
    state: RouterFitState,
    *,
    development: ArtifactValue,
    config_hash: str,
    target_compatibility_hash: str,
    bundle: LabelFreeSourceCrossfitBundle,
    fold_seal_set: SourceCrossfitFoldSealSet,
    aggregate_capability: AggregateSourceLabelCapability,
    exact_fold_outcome_universes: ExactFoldOutcomeUniverseSet,
) -> ArtifactValue:
    body = {
        **model_manifest(state),
        "development_surface_hash": development.manifest["surface_hash"],
        "compatibility_hash": target_compatibility_hash,
        "config_hash": config_hash,
        "expected_center_ids": list(CENTERS),
        "source_surface_hash": bundle.physical_surface.surface_hash,
        "source_surface_receipt_hash": bundle.surface_receipt.receipt_hash,
        "effective_adapter_hash": bundle.effective_surface.adapter_hash,
        "fold_seal_set_hash": fold_seal_set.seal_set_hash,
        "aggregate_source_label_capability_hash": aggregate_capability.capability_hash,
        "fold_menu_binding_certificate_hash": (
            fold_seal_set.fold_menu_binding_certificate_hash
        ),
        "exact_fold_outcome_universe_set_hash": (
            exact_fold_outcome_universes.set_hash
        ),
        "exact_fold_outcome_universe_hashes": [
            [
                row.outer_target_id,
                row.heldout_center_id,
                row.universe.universe_hash,
                row.binding_hash,
            ]
            for row in exact_fold_outcome_universes.folds
        ],
        "legacy_fit_source_lodo_used": False,
        "presealed_fold_assembly_only": True,
        "all_preprocessing_fit_inside_source_lodo": True,
        "nested_policy_calibration_uses_exact_H_q_r_outcome_universes": True,
        "posthoc_H_r_r_outcome_projection_used": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=state,
        manifest={**body, "model_hash": canonical_hash(body)},
        arrays=numeric_oof_arrays(state),
    )


def numeric_oof_arrays(state: RouterFitState) -> Mapping[str, np.ndarray]:
    case_rows: list[tuple[float, ...]] = []
    score_rows: list[tuple[float, ...]] = []
    offsets = [0]
    for bundle in state.bundles:
        for prediction in bundle.lodo.oof_predictions:
            case_rows.append(
                (
                    prediction.acceptance_probability,
                    prediction.rank_margin,
                    float(len(prediction.action_scores)),
                    float(prediction.top_action_id != "B"),
                )
            )
            score_rows.extend(
                (
                    row.pairwise_score,
                    row.predicted_budget_gain,
                    row.predicted_allocation_gain,
                    row.predicted_total_gain,
                    row.predicted_harm_probability,
                    row.predicted_brier_delta,
                    row.predicted_log_delta,
                    row.acceptance_probability,
                    float(row.model_available),
                )
                for row in prediction.action_scores
            )
            offsets.append(len(score_rows))
    return MappingProxyType(
        {
            "oof_case_values": np.asarray(case_rows, dtype=np.float64).reshape((-1, 4)),
            "oof_action_scores": np.asarray(score_rows, dtype=np.float64).reshape((-1, 9)),
            "oof_action_score_offsets": np.asarray(offsets, dtype=np.int64),
        }
    )


__all__ = (
    "build_development_artifact",
    "build_model_artifact",
    "build_source_crossfit_effective_artifact",
    "build_source_prelabel_prediction_artifact",
    "numeric_oof_arrays",
    "persist_and_reconstruct_source_crossfit_surface",
    "source_fold_capability_seal_payload",
    "source_prelabel_q_prediction_seal_payload",
)
