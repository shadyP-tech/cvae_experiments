"""Small strict parsing helpers for the endpoint-router config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError


def mapping_section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Consumed-test endpoint-router section {key!r} is absent.")
    return value


def require_exact(observed: object, expected: object, role: str) -> None:
    if observed != expected:
        raise ProtocolError(f"Consumed-test endpoint-router config {role} drifted.")


def require_text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Consumed-test endpoint-router {role} must be text.")
    return value


def require_artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    observed = require_text(value, artifact_id)
    if observed.startswith("artifact://"):
        valid = observed == expected
    else:
        # ``workspace prepare/run`` replaces the exact registered artifact URI
        # with its resolved absolute member path. Relative or opaque strings
        # must never bypass that workspace-owned resolution boundary.
        valid = Path(observed).is_absolute()
    if not valid:
        raise ProtocolError(f"Consumed-test endpoint-router artifact URI drifted: {artifact_id}.")


def resolve_path(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def input_path(source: Path, inputs: Mapping[str, object], key: str) -> Path:
    return resolve_path(source.parent, require_text(inputs[key], key))


def parse_classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=str(raw["family"]), C=float(raw["C"]),
            penalty=str(raw["penalty"]), solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None if raw["class_weight"] is None else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Consumed-test endpoint-router classifier is malformed.") from exc


def reject_pending(raw: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            reject_pending(value, (*trail, str(key)))
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            reject_pending(value, (*trail, str(index)))
    elif isinstance(raw, str) and ("pending://" in raw or "PENDING" in raw):
        raise ProtocolError(
            "Consumed-test endpoint-router config contains pending value at "
            f"{'.'.join(trail)}."
        )


__all__ = (
    "input_path", "mapping_section", "parse_classifier", "reject_pending",
    "require_artifact_uri", "require_exact", "require_text", "resolve_path",
)
