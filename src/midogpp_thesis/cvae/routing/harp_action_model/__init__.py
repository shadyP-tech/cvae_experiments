"""HARP action-response modeling."""

from .contracts import DIRECTIONS, LAMBDA_GRID, OUTCOMES, HarpActionScore, HarpSupportCell, HarpTargetAction, HarpTrainingObservation
from .fitting import DEFAULT_ALPHAS, HarpActionModelBank, HarpLodoFoldAudit, HarpOutcomeModel, fit_harp_action_model_bank, score_harp_actions
from .ridge import HarpRidgeModel, fit_partial_pool_ridge
from .surface_adapter import training_observation_surface_payload, training_observations_from_surfaces
from .serialization import (
    deserialize_model_bank_collection,
    model_bank_collection_from_payload,
    model_bank_collection_payload,
    model_bank_from_payload,
    model_bank_payload,
    serialize_model_bank_collection,
)

__all__ = (
    "DEFAULT_ALPHAS", "DIRECTIONS", "LAMBDA_GRID", "OUTCOMES", "HarpActionModelBank",
    "HarpActionScore", "HarpLodoFoldAudit", "HarpOutcomeModel", "HarpRidgeModel",
    "HarpSupportCell", "HarpTargetAction", "HarpTrainingObservation",
    "fit_harp_action_model_bank", "fit_partial_pool_ridge", "score_harp_actions",
    "training_observations_from_surfaces",
    "training_observation_surface_payload",
    "deserialize_model_bank_collection", "model_bank_collection_from_payload",
    "model_bank_collection_payload", "model_bank_from_payload", "model_bank_payload",
    "serialize_model_bank_collection",
)
