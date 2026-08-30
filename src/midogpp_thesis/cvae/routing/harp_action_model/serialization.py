"""Canonical serialization of complete HARP model-bank state."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..harp_protocol.hashing import canonical_hash
from .contracts import HarpSupportCell
from .fitting import HarpActionModelBank, HarpLodoFoldAudit, HarpOutcomeModel
from .ridge import HarpRidgeModel


def _ridge_payload(model: HarpRidgeModel) -> dict[str, object]:
    return {
        "feature_names": list(model.feature_names),
        "candidate_levels": list(model.candidate_levels),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "coefficients": model.coefficients.tolist(),
        "normal_inverse": model.normal_inverse.tolist(),
        "alpha": model.alpha,
        "training_query_ids": list(model.training_query_ids),
        "training_source_ids": list(model.training_source_ids),
        "training_case_ids": list(model.training_case_ids),
        "excluded_donor_ids": list(model.excluded_donor_ids),
    }


def _ridge_from_payload(raw: object) -> HarpRidgeModel:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Serialized HARP ridge model must be an object.")
    required = {"feature_names", "candidate_levels", "feature_mean", "feature_scale", "coefficients", "normal_inverse", "alpha", "training_query_ids", "training_source_ids", "training_case_ids", "excluded_donor_ids"}
    if set(raw) != required:
        raise ProtocolError("Serialized HARP ridge keys drifted.")
    try:
        return HarpRidgeModel(
            tuple(str(value) for value in raw["feature_names"]),
            tuple(str(value) for value in raw["candidate_levels"]),
            np.asarray(raw["feature_mean"], dtype=np.float64),
            np.asarray(raw["feature_scale"], dtype=np.float64),
            np.asarray(raw["coefficients"], dtype=np.float64),
            np.asarray(raw["normal_inverse"], dtype=np.float64),
            float(raw["alpha"]),
            tuple(str(value) for value in raw["training_query_ids"]),
            tuple(str(value) for value in raw["training_source_ids"]),
            tuple(str(value) for value in raw["training_case_ids"]),
            tuple(str(value) for value in raw["excluded_donor_ids"]),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Serialized HARP ridge values are malformed.") from exc


def model_bank_payload(bank: HarpActionModelBank) -> dict[str, object]:
    if not isinstance(bank, HarpActionModelBank):
        raise ProtocolError("HARP serialization requires a typed model bank.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_harp_action_model_bank_v1",
        "outer_target_id": bank.outer_target_id,
        "feature_names": list(bank.feature_names),
        "prediction_seal_hashes": list(bank.prediction_seal_hashes),
        "response_receipt_hashes": list(bank.response_receipt_hashes),
        "models": [
            {
                "outcome": model.outcome,
                "direction": model.direction,
                "full_model": _ridge_payload(model.full_model),
                "delete_donor_models": [
                    {"donor_id": donor, "model": _ridge_payload(deleted)}
                    for donor, deleted in model.delete_donor_models
                ],
                "nested_lodo_audit": [
                    {
                        "heldout_donor_id": fold.heldout_donor_id,
                        "training_query_ids": list(fold.training_query_ids),
                        "training_source_ids": list(fold.training_source_ids),
                        "selected_alpha": fold.selected_alpha,
                        "validation_mse": fold.validation_mse,
                    }
                    for fold in model.nested_lodo_audit
                ],
            }
            for model in bank.models
        ],
        "support_cells": [
            {
                "candidate_source_id": cell.candidate_source_id,
                "lambda_value": cell.lambda_value,
                "direction": cell.direction,
                "donor_count": cell.donor_count,
                "paired_case_count": cell.paired_case_count,
                "truth_classes": list(cell.truth_classes),
            }
            for cell in bank.support_cells
        ],
    }
    payload["model_bank_hash"] = canonical_hash(payload)
    return payload


def model_bank_from_payload(raw: object) -> HarpActionModelBank:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Serialized HARP model bank must be an object.")
    required = {"schema_version", "outer_target_id", "feature_names", "prediction_seal_hashes", "response_receipt_hashes", "models", "support_cells", "model_bank_hash"}
    if set(raw) != required or raw.get("schema_version") != "midogpp_harp_action_model_bank_v1" or raw.get("model_bank_hash") != canonical_hash({key: value for key, value in raw.items() if key != "model_bank_hash"}):
        raise ProtocolError("Serialized HARP model-bank schema or hash drifted.")
    models: list[HarpOutcomeModel] = []
    try:
        for item in raw["models"]:
            if not isinstance(item, Mapping) or set(item) != {"outcome", "direction", "full_model", "delete_donor_models", "nested_lodo_audit"}:
                raise ProtocolError("Serialized HARP outcome model keys drifted.")
            deleted = tuple((str(value["donor_id"]), _ridge_from_payload(value["model"])) for value in item["delete_donor_models"])
            audits = tuple(HarpLodoFoldAudit(str(value["heldout_donor_id"]), tuple(str(v) for v in value["training_query_ids"]), tuple(str(v) for v in value["training_source_ids"]), float(value["selected_alpha"]), float(value["validation_mse"])) for value in item["nested_lodo_audit"])
            models.append(HarpOutcomeModel(str(item["outcome"]), str(item["direction"]), _ridge_from_payload(item["full_model"]), deleted, audits))
        supports = tuple(HarpSupportCell(str(value["candidate_source_id"]), float(value["lambda_value"]), str(value["direction"]), int(value["donor_count"]), int(value["paired_case_count"]), tuple(int(v) for v in value["truth_classes"])) for value in raw["support_cells"])
        bank = HarpActionModelBank(str(raw["outer_target_id"]), tuple(str(v) for v in raw["feature_names"]), tuple(str(v) for v in raw["prediction_seal_hashes"]), tuple(str(v) for v in raw["response_receipt_hashes"]), tuple(models), supports)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Serialized HARP model-bank values are malformed.") from exc
    if model_bank_payload(bank)["model_bank_hash"] != raw["model_bank_hash"]:
        raise ProtocolError("Reconstructed HARP model bank changed identity.")
    return bank


def model_bank_collection_payload(banks: Sequence[HarpActionModelBank]) -> dict[str, object]:
    values = tuple(sorted(banks, key=lambda bank: bank.outer_target_id))
    if not values or len({bank.outer_target_id for bank in values}) != len(values):
        raise ProtocolError("HARP model-bank collection requires unique outer targets.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_harp_action_model_bank_collection_v1",
        "outer_target_ids": [bank.outer_target_id for bank in values],
        "banks": [model_bank_payload(bank) for bank in values],
    }
    payload["collection_hash"] = canonical_hash(payload)
    return payload


def model_bank_collection_from_payload(raw: object) -> tuple[HarpActionModelBank, ...]:
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "outer_target_ids", "banks", "collection_hash"} or raw.get("schema_version") != "midogpp_harp_action_model_bank_collection_v1" or raw.get("collection_hash") != canonical_hash({key: value for key, value in raw.items() if key != "collection_hash"}):
        raise ProtocolError("Serialized HARP model-bank collection drifted.")
    banks = tuple(model_bank_from_payload(value) for value in raw["banks"])
    if [bank.outer_target_id for bank in banks] != raw["outer_target_ids"]:
        raise ProtocolError("HARP model-bank collection target order drifted.")
    return banks


def serialize_model_bank_collection(banks: Sequence[HarpActionModelBank]) -> str:
    return json.dumps(model_bank_collection_payload(banks), sort_keys=True, separators=(",", ":"))


def deserialize_model_bank_collection(text: str) -> tuple[HarpActionModelBank, ...]:
    try:
        raw = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Serialized HARP model-bank collection is invalid JSON.") from exc
    return model_bank_collection_from_payload(raw)


__all__ = (
    "deserialize_model_bank_collection", "model_bank_collection_from_payload",
    "model_bank_collection_payload", "model_bank_from_payload", "model_bank_payload",
    "serialize_model_bank_collection",
)
