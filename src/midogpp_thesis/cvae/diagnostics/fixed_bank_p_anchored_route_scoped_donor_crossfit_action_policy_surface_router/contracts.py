"""Small cross-layer immutable contracts for P-DCAPS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .identity import DIRECT_INPUT_ROLES, canonical_hash, require_sha256


@dataclass(frozen=True)
class SixInputBinding:
    artifact_ids_by_role: tuple[tuple[str, str], ...]
    content_hashes_by_role: tuple[tuple[str, str], ...]
    protocol_hash: str
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        ids = tuple((str(role), str(value)) for role, value in self.artifact_ids_by_role)
        hashes = tuple((str(role), str(value)) for role, value in self.content_hashes_by_role)
        if (
            tuple(role for role, _ in ids) != DIRECT_INPUT_ROLES
            or tuple(role for role, _ in hashes) != DIRECT_INPUT_ROLES
            or len({value for _, value in ids}) != len(DIRECT_INPUT_ROLES)
        ):
            raise ProtocolError("P-DCAPS requires exactly six ordered direct inputs.")
        for role, digest in hashes:
            require_sha256(digest, f"{role} content hash")
        require_sha256(self.protocol_hash, "protocol hash")
        object.__setattr__(self, "artifact_ids_by_role", ids)
        object.__setattr__(self, "content_hashes_by_role", hashes)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_six_input_binding_v1",
                    "artifact_ids_by_role": ids,
                    "content_hashes_by_role": hashes,
                    "protocol_hash": self.protocol_hash,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_six_input_binding_v1",
            "artifact_ids_by_role": [list(row) for row in self.artifact_ids_by_role],
            "content_hashes_by_role": [list(row) for row in self.content_hashes_by_role],
            "protocol_hash": self.protocol_hash,
            "binding_hash": self.binding_hash,
        }


@dataclass(frozen=True, order=True)
class RouteKey:
    surface_role: str
    outer_center: str
    route_center: str
    held_case_id: str
    excluded_outer_center: str
    excluded_scored_center: str | None
    fit_scope_hash: str
    exclusion_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        role = str(self.surface_role)
        outer = str(self.outer_center)
        center = str(self.route_center)
        held = str(self.held_case_id)
        excluded_scored = (
            None if self.excluded_scored_center is None else str(self.excluded_scored_center)
        )
        if (
            role not in {"target", "pseudo"}
            or outer not in CENTERS
            or center not in CENTERS
            or not held
            or str(self.excluded_outer_center) != outer
            or (role == "target" and (center != outer or excluded_scored is not None))
            or (role == "pseudo" and (center == outer or excluded_scored != center))
        ):
            raise ProtocolError("P-DCAPS route role or exclusion identity drifted.")
        require_sha256(self.fit_scope_hash, "route fit-scope hash")
        exclusion_hash = canonical_hash(
            {
                "surface_role": role,
                "outer_center": outer,
                "route_center": center,
                "held_case_id": held,
                "excluded_outer_center": outer,
                "excluded_scored_center": excluded_scored,
                "fit_scope_hash": self.fit_scope_hash,
            }
        )
        object.__setattr__(self, "surface_role", role)
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "route_center", center)
        object.__setattr__(self, "held_case_id", held)
        object.__setattr__(self, "excluded_outer_center", outer)
        object.__setattr__(self, "excluded_scored_center", excluded_scored)
        object.__setattr__(self, "exclusion_hash", exclusion_hash)

    def to_payload(self) -> dict[str, object]:
        return {
            "surface_role": self.surface_role,
            "outer_center": self.outer_center,
            "route_center": self.route_center,
            "held_case_id": self.held_case_id,
            "excluded_outer_center": self.excluded_outer_center,
            "excluded_scored_center": self.excluded_scored_center,
            "fit_scope_hash": self.fit_scope_hash,
            "exclusion_hash": self.exclusion_hash,
        }


@dataclass(frozen=True)
class FavorableUtility:
    bacc_gain: float
    brier_gain: float
    log_gain: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_tuple(), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolError("P-DCAPS favorable utility is nonfinite.")

    @classmethod
    def zeros(cls) -> "FavorableUtility":
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_array(cls, value: object) -> "FavorableUtility":
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,):
            raise ProtocolError("P-DCAPS favorable utility must have three coordinates.")
        return cls(*(float(item) for item in array))

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc_gain, self.brier_gain, self.log_gain

    def to_payload(self) -> dict[str, float]:
        return {
            "bacc_gain": self.bacc_gain,
            "brier_gain": self.brier_gain,
            "log_gain": self.log_gain,
        }

    def __add__(self, other: "FavorableUtility") -> "FavorableUtility":
        return FavorableUtility.from_array(
            np.asarray(self.as_tuple()) + np.asarray(other.as_tuple())
        )

    def __sub__(self, other: "FavorableUtility") -> "FavorableUtility":
        return FavorableUtility.from_array(
            np.asarray(self.as_tuple()) - np.asarray(other.as_tuple())
        )


@dataclass(frozen=True)
class BankViability:
    row_preserving: bool
    class_domain_support_preserved: bool
    per_class_effective_sample_size: tuple[tuple[str, float], ...]
    minimum_effective_sample_size: float
    provenance_hash: str

    def __post_init__(self) -> None:
        rows = tuple((str(label), float(value)) for label, value in self.per_class_effective_sample_size)
        if (
            tuple(label for label, _ in rows) != ("0", "1")
            or self.minimum_effective_sample_size <= 0.0
            or any(not np.isfinite(value) or value < 0.0 for _, value in rows)
        ):
            raise ProtocolError("P-DCAPS bank viability contract drifted.")
        require_sha256(self.provenance_hash, "bank viability provenance")
        object.__setattr__(self, "per_class_effective_sample_size", rows)

    @property
    def passed(self) -> bool:
        return (
            self.class_domain_support_preserved
            and all(
                value >= self.minimum_effective_sample_size
                for _, value in self.per_class_effective_sample_size
            )
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "row_preserving": self.row_preserving,
            "class_domain_support_preserved": self.class_domain_support_preserved,
            "per_class_effective_sample_size": [list(row) for row in self.per_class_effective_sample_size],
            "minimum_effective_sample_size": self.minimum_effective_sample_size,
            "provenance_hash": self.provenance_hash,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SealManifest:
    phase: str
    input_hashes: tuple[str, ...]
    row_order_hash: str
    protocol_hash: str
    model_hashes: tuple[str, ...]
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        inputs = tuple(str(value) for value in self.input_hashes)
        models = tuple(str(value) for value in self.model_hashes)
        for role, digest in (
            *(("input", value) for value in inputs),
            (("row order", self.row_order_hash)),
            (("protocol", self.protocol_hash)),
            *(("model", value) for value in models),
        ):
            require_sha256(digest, role)
        payload = {
            "schema_version": "pdcaps_seal_manifest_v1",
            "phase": str(self.phase),
            "input_hashes": inputs,
            "row_order_hash": self.row_order_hash,
            "protocol_hash": self.protocol_hash,
            "model_hashes": models,
        }
        object.__setattr__(self, "input_hashes", inputs)
        object.__setattr__(self, "model_hashes", models)
        object.__setattr__(self, "seal_hash", canonical_hash(payload))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_seal_manifest_v1",
            "phase": self.phase,
            "input_hashes": list(self.input_hashes),
            "row_order_hash": self.row_order_hash,
            "protocol_hash": self.protocol_hash,
            "model_hashes": list(self.model_hashes),
            "seal_hash": self.seal_hash,
        }


def binding_from_mappings(
    artifact_ids: Mapping[str, str],
    content_hashes: Mapping[str, str],
    *,
    protocol_hash: str,
) -> SixInputBinding:
    return SixInputBinding(
        tuple((role, artifact_ids[role]) for role in DIRECT_INPUT_ROLES),
        tuple((role, content_hashes[role]) for role in DIRECT_INPUT_ROLES),
        protocol_hash,
    )


__all__ = (
    "BankViability",
    "FavorableUtility",
    "RouteKey",
    "SealManifest",
    "SixInputBinding",
    "binding_from_mappings",
)
