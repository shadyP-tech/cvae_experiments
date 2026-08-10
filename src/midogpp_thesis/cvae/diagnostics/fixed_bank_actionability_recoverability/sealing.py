"""Durable label-capability transitions for the diagnostic runner.

This module deliberately contains no scientific fitting logic.  It translates
already-persisted product seals into the small, stateful label capability API.
Keeping that transition here makes it hard for a runner refactor to open labels
before the corresponding bytes are durable on disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...protocol import ProtocolError
from .constants import GEOMETRY_IDS, MIDOGPP_CENTERS
from .label_capabilities import (
    EXPECTED_ALL_DECISION_COUNT,
    EXPECTED_PRE_SUPPORT_DECISION_COUNT,
    ActionabilityLabelCapabilityManager,
)


@dataclass(frozen=True, order=True)
class ModelSealRecord:
    target_center: str
    geometry_id: str
    family: str
    seal_hash: str


@dataclass(frozen=True, order=True)
class DecisionSealRecord:
    target_center: str
    fold_ordinal: int
    method_id: str
    geometry_id: str | None
    decision_hash: str


def record_durable_model_seals(
    capability: ActionabilityLabelCapabilityManager,
    products: object,
) -> None:
    """Record all 54 geometry/family seals after their manifest is durable."""

    records = _model_records(products)
    expected = {
        (target, geometry, family)
        for target in MIDOGPP_CENTERS
        for geometry in GEOMETRY_IDS
        for family in ("G", "R", "P")
    }
    if {(row.target_center, row.geometry_id, row.family) for row in records} != expected:
        raise ProtocolError("Durable model-seal topology is incomplete.")
    for target in MIDOGPP_CENTERS:
        hashes = {
            f"{row.geometry_id}:{row.family}": row.seal_hash
            for row in records
            if row.target_center == target
        }
        capability.record_loco_model_seals(target, hashes)


def record_durable_pre_support_seals(
    capability: ActionabilityLabelCapabilityManager,
    products: object,
) -> None:
    """Record B/U/G/R/P fold seals before any same-H support is opened."""

    records = _decision_records(products, "pre_support_seal_records")
    if len(records) != EXPECTED_PRE_SUPPORT_DECISION_COUNT:
        raise ProtocolError("Pre-support decision-seal count drifted.")
    for row in records:
        capability.record_pre_support_decision(
            row.target_center,
            row.fold_ordinal,
            row.method_id,
            row.decision_hash,
            geometry_id=row.geometry_id,
        )
    capability.record_pre_support_seal(
        _hash_attribute(products, "pre_support_seal_hash"),
        decision_count=len(records),
    )


def record_durable_preevaluation_seals(
    capability: ActionabilityLabelCapabilityManager,
    products: object,
) -> None:
    """Record S_y and the aggregate/permutation seals before terminal labels."""

    support = _decision_records(products, "support_seal_records")
    for row in support:
        if row.method_id != "S_y" or row.geometry_id not in GEOMETRY_IDS:
            raise ProtocolError("Support seal must be a geometry-local S_y decision.")
        capability.record_support_decision(
            row.target_center,
            row.fold_ordinal,
            row.geometry_id,
            row.decision_hash,
        )
    pre_support = _decision_records(products, "pre_support_seal_records")
    if len(pre_support) + len(support) != EXPECTED_ALL_DECISION_COUNT:
        raise ProtocolError("All-method decision-seal count drifted.")
    capability.record_preevaluation_seals(
        _hash_attribute(products, "all_decisions_seal_hash"),
        _hash_attribute(products, "permutation_provenance_hash"),
        decision_count=len(pre_support) + len(support),
    )


def _model_records(products: object) -> tuple[ModelSealRecord, ...]:
    raw = getattr(
        products,
        "model_seal_records",
        getattr(products, "model_seals_by_target", products),
    )
    if isinstance(raw, Mapping):
        expanded: list[ModelSealRecord] = []
        for target, values in raw.items():
            if not isinstance(values, Mapping):
                raise ProtocolError("Model-seal mapping is malformed.")
            for key, seal_hash in values.items():
                try:
                    geometry, family = str(key).split(":", 1)
                except ValueError as exc:
                    raise ProtocolError("Model-seal key is malformed.") from exc
                expanded.append(
                    ModelSealRecord(str(target), geometry, family, str(seal_hash))
                )
        return tuple(sorted(expanded))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ProtocolError("Model products do not expose seal records.")
    return tuple(sorted(_as_model_record(value) for value in raw))


def _decision_records(products: object, attribute: str) -> tuple[DecisionSealRecord, ...]:
    raw = getattr(products, attribute, None)
    if raw is None and attribute == "pre_support_seal_records":
        raw = getattr(products, "fold_seals", None)
    if raw is None and attribute in {
        "pre_support_seal_records",
        "support_seal_records",
    }:
        mapping_name = (
            "pre_support_decision_hashes"
            if attribute == "pre_support_seal_records"
            else "all_decision_hashes"
        )
        mapping = getattr(products, mapping_name, None)
        if isinstance(mapping, Mapping):
            if attribute == "support_seal_records":
                mapping = {
                    key: value
                    for key, value in mapping.items()
                    if isinstance(key, tuple) and len(key) == 4 and key[2] == "S_y"
                }
            raw = [
                {
                    "target_center": key[0],
                    "fold_ordinal": key[1],
                    "method_id": key[2],
                    "geometry_id": key[3],
                    "decision_hash": value,
                }
                for key, value in mapping.items()
            ]
    if raw is None:
        raise ProtocolError(f"Decision products do not expose {attribute}.")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ProtocolError(f"{attribute} must be a sequence.")
    records = tuple(sorted(_as_decision_record(value) for value in raw))
    identities = {
        (row.target_center, row.fold_ordinal, row.method_id, row.geometry_id)
        for row in records
    }
    if len(identities) != len(records):
        raise ProtocolError(f"{attribute} contains duplicate identities.")
    return records


def _as_model_record(value: object) -> ModelSealRecord:
    if isinstance(value, ModelSealRecord):
        return value
    raw = value if isinstance(value, Mapping) else getattr(value, "__dict__", None)
    if not isinstance(raw, Mapping):
        raise ProtocolError("Model seal record is malformed.")
    try:
        return ModelSealRecord(
            str(raw["target_center"]),
            str(raw["geometry_id"]),
            str(raw["family"]),
            str(raw.get("seal_hash", raw.get("model_hash"))),
        )
    except KeyError as exc:
        raise ProtocolError("Model seal record is incomplete.") from exc


def _as_decision_record(value: object) -> DecisionSealRecord:
    if isinstance(value, DecisionSealRecord):
        return value
    raw = value if isinstance(value, Mapping) else getattr(value, "__dict__", None)
    if not isinstance(raw, Mapping):
        raise ProtocolError("Decision seal record is malformed.")
    try:
        geometry = raw.get("geometry_id")
        return DecisionSealRecord(
            str(raw["target_center"]),
            int(raw["fold_ordinal"]),
            str(raw["method_id"]),
            None if geometry is None else str(geometry),
            str(raw.get("decision_hash", raw.get("seal_hash"))),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Decision seal record is incomplete.") from exc


def _hash_attribute(products: object, name: str) -> str:
    value = getattr(products, name, None)
    if not isinstance(value, str):
        raise ProtocolError(f"Decision products do not expose {name}.")
    return value


__all__ = (
    "DecisionSealRecord",
    "ModelSealRecord",
    "record_durable_model_seals",
    "record_durable_preevaluation_seals",
    "record_durable_pre_support_seals",
)
