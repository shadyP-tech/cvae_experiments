"""Matched B/B_cal/G/R/P model and prediction assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError
from .case_features import compute_source_controls, permute_case_features
from .composition import (
    baseline_predictions,
    calibrated_baseline_predictions,
    compose_probabilities,
)
from .contracts import (
    CaseClassWeights,
    CaseFeatureRow,
    DonorResponseRow,
    HierarchicalResidualModel,
    PredictionRow,
    SampleActionProbability,
)
from .core_hashing import canonical_hash
from .hierarchical_model import fit_loco_hierarchical_model, predict_case_weights
from .scientific_constants import METHOD_IDS


@dataclass(frozen=True)
class ModelFamilyBundle:
    target_center: str
    global_model: HierarchicalResidualModel
    residual_model: HierarchicalResidualModel
    permuted_model: HierarchicalResidualModel
    permuted_features: tuple[CaseFeatureRow, ...]
    bundle_hash: str = field(init=False)

    def __post_init__(self) -> None:
        models = (self.global_model, self.residual_model, self.permuted_model)
        if tuple(model.model_family for model in models) != ("G", "R", "P"):
            raise ProtocolError("Model bundle must contain separately fit G, R, and P families.")
        if any(model.target_center != self.target_center for model in models):
            raise ProtocolError("Model bundle target centers drifted.")
        if self.residual_model.model_hash == self.permuted_model.model_hash:
            raise ProtocolError("P must not reuse R coefficients/model identity.")
        object.__setattr__(self, "permuted_features", tuple(self.permuted_features))
        object.__setattr__(self, "bundle_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_hierarchical_residual_stacker_model_families_v1",
            "target_center": self.target_center,
            "G_model_hash": self.global_model.model_hash,
            "R_model_hash": self.residual_model.model_hash,
            "P_model_hash": self.permuted_model.model_hash,
            "P_feature_hashes": [row.feature_hash for row in self.permuted_features],
            "separate_fits": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "bundle_hash": self.bundle_hash}


def fit_model_families(
    features: Sequence[CaseFeatureRow],
    responses: Sequence[DonorResponseRow],
    *,
    target_center: str,
) -> ModelFamilyBundle:
    feature_rows = tuple(features)
    permuted = permute_case_features(feature_rows)
    global_model = fit_loco_hierarchical_model(
        feature_rows, responses, target_center=target_center, model_family="G"
    )
    residual_model = fit_loco_hierarchical_model(
        feature_rows, responses, target_center=target_center, model_family="R"
    )
    permuted_model = fit_loco_hierarchical_model(
        feature_rows,
        responses,
        target_center=target_center,
        source_control_features=feature_rows,
        model_family="P",
    )
    return ModelFamilyBundle(
        target_center=str(target_center),
        global_model=global_model,
        residual_model=residual_model,
        permuted_model=permuted_model,
        permuted_features=permuted,
    )


def predict_family_weights(
    bundle: ModelFamilyBundle,
    original_features: Sequence[CaseFeatureRow],
) -> Mapping[str, tuple[CaseClassWeights, ...]]:
    controls = compute_source_controls(original_features, target_center=bundle.target_center)
    return {
        "G": predict_case_weights(bundle.global_model, original_features, controls),
        "R": predict_case_weights(bundle.residual_model, original_features, controls),
        "P": predict_case_weights(bundle.permuted_model, bundle.permuted_features, controls),
    }


def build_method_predictions(
    probabilities: Sequence[SampleActionProbability],
    *,
    intercept: float,
    residual_scale: float,
    global_weights: Sequence[CaseClassWeights],
    residual_weights: Sequence[CaseClassWeights],
    permuted_weights: Sequence[CaseClassWeights],
) -> Mapping[str, tuple[PredictionRow, ...]]:
    predictions = {
        "B": baseline_predictions(probabilities, method_id="B"),
        "B_cal": calibrated_baseline_predictions(
            probabilities, intercept=intercept, method_id="B_cal"
        ),
        "G": compose_probabilities(
            probabilities,
            global_weights,
            intercept=intercept,
            residual_scale=residual_scale,
            method_id="G",
        ),
        "R": compose_probabilities(
            probabilities,
            residual_weights,
            intercept=intercept,
            residual_scale=residual_scale,
            method_id="R",
        ),
        "P": compose_probabilities(
            probabilities,
            permuted_weights,
            intercept=intercept,
            residual_scale=residual_scale,
            method_id="P",
        ),
    }
    if tuple(predictions) != METHOD_IDS:
        raise ProtocolError("Diagnostic method IDs drifted from B/B_cal/G/R/P.")
    if residual_scale == 0.0:
        baseline_values = tuple(row.probability for row in predictions["B_cal"])
        for method in ("G", "R", "P"):
            if tuple(row.probability for row in predictions[method]) != baseline_values:
                raise ProtocolError("Lambda-zero matched controls are not bit-exact B_cal.")
    return predictions


__all__ = (
    "ModelFamilyBundle",
    "build_method_predictions",
    "fit_model_families",
    "predict_family_weights",
)
