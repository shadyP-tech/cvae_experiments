"""Strict leaf helpers for the residual-stacker configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError


CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "protocol",
        "probability_surface",
        "features",
        "hierarchical_model",
        "target_support",
        "stacker",
        "controls",
        "classifier",
        "evaluation",
        "runtime",
        "claim_boundary",
    }
)


def mapping_section(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Residual-stacker config section {key!r} is absent.")
    return value


def require_exact(observed: object, expected: object, role: str) -> None:
    if observed != expected:
        raise ProtocolError(f"Residual-stacker config {role} drifted.")


def require_text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Residual-stacker config {role} must be text.")
    return value


def require_artifact_uri(value: object, *, artifact_id: str, member: str) -> None:
    expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
    observed = require_text(value, artifact_id)
    if observed.startswith("artifact://") and observed != expected:
        raise ProtocolError(f"Residual-stacker artifact URI drifted: {artifact_id}.")


def resolve_config_path(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def parse_classifier(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(
                None if raw["class_weight"] is None else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(
                None if raw["l1_ratio"] is None else float(raw["l1_ratio"])
            ),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Residual-stacker classifier is malformed.") from exc


def reject_pending(raw: object, trail: tuple[str, ...] = ()) -> None:
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            reject_pending(value, (*trail, str(key)))
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            reject_pending(value, (*trail, str(index)))
    elif isinstance(raw, str) and ("pending://" in raw or "PENDING" in raw):
        raise ProtocolError(
            f"Residual-stacker config contains pending value at {'.'.join(trail)}."
        )


__all__ = (
    "CONFIG_TOP_LEVEL",
    "mapping_section",
    "parse_classifier",
    "reject_pending",
    "require_artifact_uri",
    "require_exact",
    "require_text",
    "resolve_config_path",
)
