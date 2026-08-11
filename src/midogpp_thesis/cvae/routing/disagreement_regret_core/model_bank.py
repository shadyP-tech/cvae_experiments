"""Canonical freezing and serialization for pairwise regret model banks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .hashing import canonical_sha256, is_sha256
from .inference_contracts import InferenceActionSchema
from .model_contracts import PairwiseRegretModel
from .probability_contracts import SOURCE_OOF_TRAINING_SURFACE_ROLE


MODEL_BANK_SCHEMA_VERSION = "midogpp_disagreement_regret_model_bank_v1"


@dataclass(frozen=True)
class PairwiseRegretModelBank:
    """One complete, frozen family/target bank for label-free inference."""

    models: tuple[PairwiseRegretModel, ...]
    action_schema: InferenceActionSchema = field(init=False)
    family: str = field(init=False)
    outer_target_id: str = field(init=False)
    model_bank_hash: str = field(init=False)

    def __post_init__(self) -> None:
        models = tuple(self.models)
        if not models or any(not isinstance(model, PairwiseRegretModel) for model in models):
            raise ProtocolError("A model bank requires typed pairwise regret models.")
        if tuple(sorted(models, key=lambda model: model.candidate_action_id)) != models:
            raise ProtocolError("Model bank entries must use canonical candidate ordering.")
        if len({model.candidate_action_id for model in models}) != len(models):
            raise ProtocolError("Model bank contains duplicate candidate actions.")
        if any(model.heldout_query_id is not None for model in models):
            raise ProtocolError("Nested donor-q models cannot enter an inference model bank.")
        families = {model.family for model in models}
        targets = {model.outer_target_id for model in models}
        if len(families) != 1 or len(targets) != 1:
            raise ProtocolError("A model bank cannot mix families or outer targets.")
        reference = models[0]
        fixed_lineage = (
            "feature_surface_hash",
            "response_surface_hash",
            "prediction_seal_hash",
            "development_context_hash",
            "baseline_action_id",
            "control_action_id",
            "candidate_source_by_action",
            "feature_names",
            "action_ids",
            "shared_l2_penalty",
            "action_l2_penalty",
            "max_newton_iterations",
            "gradient_tolerance",
            "source_history_mode",
            "training_scope",
            "training_surface_role",
        )
        if any(
            any(getattr(model, name) != getattr(reference, name) for name in fixed_lineage)
            for model in models[1:]
        ):
            raise ProtocolError("Model bank training lineage or schema drifted.")
        mapping = tuple(reference.candidate_source_by_action)
        if {model.candidate_action_id for model in models} != {
            action_id for action_id, _source_id in mapping
        }:
            raise ProtocolError("Model bank must contain every candidate action exactly once.")
        if any(
            dict(mapping).get(model.candidate_action_id) != model.candidate_source_id
            for model in models
        ):
            raise ProtocolError("Model bank candidate/source identity drifted.")
        if (
            reference.training_scope
            in ("AUTHORIZED_SOURCE_OOF", "AUTHORIZED_POSTHOC_SOURCE_OOF")
            and reference.training_surface_role != SOURCE_OOF_TRAINING_SURFACE_ROLE
        ):
            raise ProtocolError(
                "A real-data inference bank must be frozen before target admission."
            )
        schema = InferenceActionSchema(
            family=reference.family,
            baseline_action_id=reference.baseline_action_id,
            control_action_id=reference.control_action_id,
            candidate_source_by_action=mapping,
            feature_names=reference.feature_names,
        )
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "action_schema", schema)
        object.__setattr__(self, "family", reference.family)
        object.__setattr__(self, "outer_target_id", reference.outer_target_id)
        object.__setattr__(self, "model_bank_hash", canonical_sha256(self._unhashed_payload()))

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": MODEL_BANK_SCHEMA_VERSION,
            "family": self.family,
            "outer_target_id": self.outer_target_id,
            "action_schema": self.action_schema.to_payload(),
            "action_schema_hash": self.action_schema.schema_hash,
            "model_hashes": [model.model_hash for model in self.models],
            "models": [model.to_payload() for model in self.models],
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._unhashed_payload()
        payload["model_bank_hash"] = self.model_bank_hash
        return payload


def freeze_pairwise_model_bank(
    models: Sequence[PairwiseRegretModel],
) -> PairwiseRegretModelBank:
    """Freeze a complete outer-target model sequence into one typed identity."""

    return PairwiseRegretModelBank(tuple(models))


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def serialize_pairwise_model_bank(bank: PairwiseRegretModelBank) -> str:
    """Serialize a bank to deterministic canonical JSON."""

    if not isinstance(bank, PairwiseRegretModelBank):
        raise ProtocolError("Model-bank serialization requires a typed frozen bank.")
    return _canonical_json(bank.to_payload())


def _reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError("Serialized model bank contains duplicate object keys.")
        output[key] = value
    return output


def _require_exact_keys(
    payload: Mapping[str, object], expected: set[str], *, name: str
) -> None:
    if set(payload) != expected:
        raise ProtocolError(f"Serialized {name} keys drifted from the canonical schema.")


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ProtocolError(f"Serialized {name} must be an object with string keys.")
    return value


def _require_list(value: object, *, name: str) -> list[object]:
    if type(value) is not list:
        raise ProtocolError(f"Serialized {name} must be a JSON array.")
    return value


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ProtocolError(f"Serialized {name} must be a string.")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    rows = _require_list(value, name=name)
    if any(type(row) is not str for row in rows):
        raise ProtocolError(f"Serialized {name} must contain only strings.")
    return tuple(rows)


def _pair_tuple(value: object, *, name: str) -> tuple[tuple[str, str], ...]:
    rows = _require_list(value, name=name)
    output: list[tuple[str, str]] = []
    for row in rows:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
        ):
            raise ProtocolError(f"Serialized {name} must contain string pairs.")
        output.append((row[0], row[1]))
    return tuple(output)


def _array(value: object, *, name: str, dimensions: int) -> np.ndarray:
    rows = _require_list(value, name=name)
    try:
        array = np.asarray(rows, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Serialized {name} is not a numeric array.") from exc
    if array.ndim != dimensions:
        raise ProtocolError(f"Serialized {name} has the wrong rank.")
    return array


def _number(value: object, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ProtocolError(f"Serialized {name} must be a finite number.")
    return float(value)


def _model_from_payload(value: object) -> PairwiseRegretModel:
    payload = _require_mapping(value, name="model")
    expected = {
        "schema_version",
        "family",
        "source_history_mode",
        "training_scope",
        "training_surface_role",
        "outer_target_id",
        "candidate_action_id",
        "candidate_source_id",
        "heldout_query_id",
        "action_ids",
        "feature_names",
        "feature_mean",
        "feature_scale",
        "coefficients",
        "coefficient_covariance",
        "training_query_ids",
        "excluded_query_ids",
        "observation_count",
        "converged",
        "iteration_count",
        "feature_surface_hash",
        "response_surface_hash",
        "prediction_seal_hash",
        "development_context_hash",
        "baseline_action_id",
        "control_action_id",
        "candidate_source_by_action",
        "training_feature_hash",
        "training_response_hash",
        "hyperparameters",
        "model_hash",
    }
    _require_exact_keys(payload, expected, name="model")
    if payload["schema_version"] != "midogpp_pairwise_disagreement_regret_model_v1":
        raise ProtocolError("Serialized model schema version drifted.")
    heldout = payload["heldout_query_id"]
    if heldout is not None and type(heldout) is not str:
        raise ProtocolError("Serialized heldout_query_id must be null or a string.")
    if type(payload["observation_count"]) is not int:
        raise ProtocolError("Serialized observation_count must be an integer.")
    if type(payload["iteration_count"]) is not int:
        raise ProtocolError("Serialized iteration_count must be an integer.")
    if payload["converged"] is not True:
        raise ProtocolError("Serialized model must attest exact convergence.")
    hyperparameters = _require_mapping(payload["hyperparameters"], name="hyperparameters")
    _require_exact_keys(
        hyperparameters,
        {
            "shared_l2_penalty",
            "action_l2_penalty",
            "max_newton_iterations",
            "gradient_tolerance",
        },
        name="hyperparameters",
    )
    if type(hyperparameters["max_newton_iterations"]) is not int:
        raise ProtocolError("Serialized Newton cap must be an integer.")
    model = PairwiseRegretModel(
        family=_require_string(payload["family"], name="family"),
        source_history_mode=_require_string(
            payload["source_history_mode"], name="source_history_mode"
        ),
        training_scope=_require_string(payload["training_scope"], name="training_scope"),
        training_surface_role=_require_string(
            payload["training_surface_role"], name="training_surface_role"
        ),
        outer_target_id=_require_string(payload["outer_target_id"], name="outer_target_id"),
        candidate_action_id=_require_string(
            payload["candidate_action_id"], name="candidate_action_id"
        ),
        candidate_source_id=_require_string(
            payload["candidate_source_id"], name="candidate_source_id"
        ),
        heldout_query_id=heldout,
        action_ids=_string_tuple(payload["action_ids"], name="action_ids"),
        feature_names=_string_tuple(payload["feature_names"], name="feature_names"),
        feature_mean=_array(payload["feature_mean"], name="feature_mean", dimensions=1),
        feature_scale=_array(payload["feature_scale"], name="feature_scale", dimensions=1),
        coefficients=_array(payload["coefficients"], name="coefficients", dimensions=1),
        coefficient_covariance=_array(
            payload["coefficient_covariance"],
            name="coefficient_covariance",
            dimensions=2,
        ),
        training_query_ids=_string_tuple(
            payload["training_query_ids"], name="training_query_ids"
        ),
        excluded_query_ids=_string_tuple(
            payload["excluded_query_ids"], name="excluded_query_ids"
        ),
        observation_count=payload["observation_count"],
        converged=True,
        iteration_count=payload["iteration_count"],
        feature_surface_hash=_require_string(
            payload["feature_surface_hash"], name="feature_surface_hash"
        ),
        response_surface_hash=_require_string(
            payload["response_surface_hash"], name="response_surface_hash"
        ),
        prediction_seal_hash=_require_string(
            payload["prediction_seal_hash"], name="prediction_seal_hash"
        ),
        development_context_hash=_require_string(
            payload["development_context_hash"], name="development_context_hash"
        ),
        baseline_action_id=_require_string(
            payload["baseline_action_id"], name="baseline_action_id"
        ),
        control_action_id=_require_string(
            payload["control_action_id"], name="control_action_id"
        ),
        candidate_source_by_action=_pair_tuple(
            payload["candidate_source_by_action"], name="candidate_source_by_action"
        ),
        training_feature_hash=_require_string(
            payload["training_feature_hash"], name="training_feature_hash"
        ),
        training_response_hash=_require_string(
            payload["training_response_hash"], name="training_response_hash"
        ),
        shared_l2_penalty=_number(
            hyperparameters["shared_l2_penalty"], name="shared_l2_penalty"
        ),
        action_l2_penalty=_number(
            hyperparameters["action_l2_penalty"], name="action_l2_penalty"
        ),
        max_newton_iterations=hyperparameters["max_newton_iterations"],
        gradient_tolerance=_number(
            hyperparameters["gradient_tolerance"], name="gradient_tolerance"
        ),
    )
    supplied_hash = _require_string(payload["model_hash"], name="model_hash")
    if not is_sha256(supplied_hash) or supplied_hash != model.model_hash:
        raise ProtocolError("Serialized model hash failed canonical replay.")
    return model


def deserialize_pairwise_model_bank(serialized: str) -> PairwiseRegretModelBank:
    """Replay and validate a canonical model-bank JSON document."""

    if type(serialized) is not str or not serialized:
        raise ProtocolError("Serialized model bank must be a nonempty JSON string.")
    try:
        value = json.loads(serialized, object_pairs_hook=_reject_duplicate_object_keys)
    except ProtocolError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError("Serialized model bank is not valid JSON.") from exc
    payload = _require_mapping(value, name="model bank")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "family",
            "outer_target_id",
            "action_schema",
            "action_schema_hash",
            "model_hashes",
            "models",
            "model_bank_hash",
        },
        name="model bank",
    )
    if payload["schema_version"] != MODEL_BANK_SCHEMA_VERSION:
        raise ProtocolError("Serialized model-bank schema version drifted.")
    if serialized != _canonical_json(payload):
        raise ProtocolError("Serialized model bank is not canonical JSON.")
    model_values = _require_list(payload["models"], name="models")
    bank = PairwiseRegretModelBank(tuple(_model_from_payload(row) for row in model_values))
    supplied_hash = _require_string(payload["model_bank_hash"], name="model_bank_hash")
    if not is_sha256(supplied_hash) or supplied_hash != bank.model_bank_hash:
        raise ProtocolError("Serialized model-bank hash failed canonical replay.")
    if payload["family"] != bank.family or payload["outer_target_id"] != bank.outer_target_id:
        raise ProtocolError("Serialized model-bank target/family summary drifted.")
    if payload["action_schema"] != bank.action_schema.to_payload() or (
        payload["action_schema_hash"] != bank.action_schema.schema_hash
    ):
        raise ProtocolError("Serialized model-bank action schema drifted.")
    if payload["model_hashes"] != [model.model_hash for model in bank.models]:
        raise ProtocolError("Serialized model-bank model hash list drifted.")
    if serialize_pairwise_model_bank(bank) != serialized:
        raise ProtocolError("Serialized model bank failed byte-exact canonical replay.")
    return bank


__all__ = (
    "MODEL_BANK_SCHEMA_VERSION",
    "PairwiseRegretModelBank",
    "deserialize_pairwise_model_bank",
    "freeze_pairwise_model_bank",
    "serialize_pairwise_model_bank",
)
