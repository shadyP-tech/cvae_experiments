"""Sealed B/I/R endpoint assembly from the exact 810-cell physical bank."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np

from ..action_geometry import (
    BoundaryProjection,
    build_boundary_projection,
    canonical_probabilities,
    probability_hash,
)
from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_FAMILIES, DIRECTIONS, GENERATION_SEEDS, TRAINING_SEEDS
from ..protocol import ProtocolError
from .library import (
    A1_PREFIX,
    B_ACTION_ID,
    U_ACTION_ID,
    PhysicalCellIdentity,
    action_ids_for_target,
)


DERIVATION_RULES = {
    "B": "EXACT_NINE_MEAN_B",
    "I": "DIRECTIONAL_ROW_EXTREME_OF_EIGHT_TARGET_EXCLUDED_A1_MEANS",
    "R": "ROW_MEDIAN_OF_U_AND_EIGHT_TARGET_EXCLUDED_A1_MEANS",
}


_PHYSICAL_SURFACE_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class PhysicalCellSurface:
    """One case slice from one sealed physical-cell memmap."""

    identity: PhysicalCellIdentity
    case_id: str
    cache_content_hash: str
    row_order_hash: str
    probabilities: tuple[float, ...]
    physical_bank_receipt_hash: str
    memmap_reference_hash: str
    memmap_slice_sha256: str
    memmap_row_index_hash: str
    _factory_token: InitVar[object] = None
    probability_hash: str = field(init=False)
    surface_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _PHYSICAL_SURFACE_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP physical surface bypassed the read-only memmap loader."
            )
        case = str(self.case_id)
        cache_hash = require_sha256(self.cache_content_hash, "physical cache hash")
        row_hash = require_sha256(self.row_order_hash, "physical row-order hash")
        bank_hash = require_sha256(
            self.physical_bank_receipt_hash, "physical-bank receipt hash"
        )
        reference_hash = require_sha256(
            self.memmap_reference_hash, "physical memmap-reference hash"
        )
        slice_hash = require_sha256(
            self.memmap_slice_sha256, "physical memmap-slice hash"
        )
        index_hash = require_sha256(
            self.memmap_row_index_hash, "physical memmap row-index hash"
        )
        values = canonical_probabilities(self.probabilities)
        if not isinstance(self.identity, PhysicalCellIdentity) or not case:
            raise ProtocolError("SCALE-BP physical cell surface identity drifted.")
        vector_hash = probability_hash(values)
        values_tuple = tuple(float(value) for value in values)
        payload = {
            "schema_version": "scale_bp_physical_cell_surface_v2",
            "cell_hash": self.identity.cell_hash,
            "case_id": case,
            "cache_content_hash": cache_hash,
            "row_order_hash": row_hash,
            "row_count": len(values),
            "probability_hash": vector_hash,
            "physical_bank_receipt_hash": bank_hash,
            "memmap_reference_hash": reference_hash,
            "memmap_slice_sha256": slice_hash,
            "memmap_row_index_hash": index_hash,
            "stored_dtype": "float32",
            "read_only_memmap_loaded": True,
            "labels_used": False,
        }
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "cache_content_hash", cache_hash)
        object.__setattr__(self, "row_order_hash", row_hash)
        object.__setattr__(self, "physical_bank_receipt_hash", bank_hash)
        object.__setattr__(self, "memmap_reference_hash", reference_hash)
        object.__setattr__(self, "memmap_slice_sha256", slice_hash)
        object.__setattr__(self, "memmap_row_index_hash", index_hash)
        object.__setattr__(self, "probabilities", values_tuple)
        object.__setattr__(self, "probability_hash", vector_hash)
        object.__setattr__(self, "surface_hash", canonical_hash(payload))


def _issue_physical_cell_surface(**kwargs: object) -> PhysicalCellSurface:
    return PhysicalCellSurface(
        **kwargs,
        _factory_token=_PHYSICAL_SURFACE_FACTORY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class EndpointSurfaceReceipt:
    """Endpoint re-derived from a complete physical surface on construction."""

    target_center: str
    case_id: str
    family: str
    direction: str
    physical_surfaces: tuple[PhysicalCellSurface, ...]
    cache_content_hash: str = field(init=False)
    row_order_hash: str = field(init=False)
    physical_bank_receipt_hash: str = field(init=False)
    physical_cell_identity_hashes: tuple[str, ...] = field(init=False)
    physical_cell_surface_hashes: tuple[str, ...] = field(init=False)
    derivation_rule: str = field(init=False)
    endpoint_probabilities: tuple[float, ...] = field(init=False)
    endpoint_probability_hash: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        case = str(self.case_id)
        family = str(self.family)
        direction = str(self.direction)
        ordered_rows, values = _derive_endpoint(
            tuple(self.physical_surfaces),
            target_center=target,
            case_id=case,
            family=family,
            direction=direction,
        )
        cache_hash = ordered_rows[0].cache_content_hash
        row_hash = ordered_rows[0].row_order_hash
        bank_hash = ordered_rows[0].physical_bank_receipt_hash
        identity_hashes = tuple(row.identity.cell_hash for row in ordered_rows)
        surface_hashes = tuple(row.surface_hash for row in ordered_rows)
        endpoint_hash = probability_hash(values)
        values_tuple = tuple(float(value) for value in values)
        payload = {
            "schema_version": "scale_bp_endpoint_surface_receipt_v2",
            "target_center": target,
            "case_id": case,
            "family": family,
            "direction": direction,
            "cache_content_hash": cache_hash,
            "row_order_hash": row_hash,
            "physical_bank_receipt_hash": bank_hash,
            "physical_cell_identity_hashes": identity_hashes,
            "physical_cell_surface_hashes": surface_hashes,
            "derivation_rule": DERIVATION_RULES[family],
            "endpoint_probability_hash": endpoint_hash,
            "row_count": len(values),
            "physical_cell_count": 90,
            "target_expert_excluded": True,
            "labels_used": False,
        }
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "physical_surfaces", ordered_rows)
        object.__setattr__(self, "cache_content_hash", cache_hash)
        object.__setattr__(self, "row_order_hash", row_hash)
        object.__setattr__(self, "physical_bank_receipt_hash", bank_hash)
        object.__setattr__(self, "physical_cell_identity_hashes", identity_hashes)
        object.__setattr__(self, "physical_cell_surface_hashes", surface_hashes)
        object.__setattr__(self, "derivation_rule", DERIVATION_RULES[family])
        object.__setattr__(self, "endpoint_probabilities", values_tuple)
        object.__setattr__(self, "endpoint_probability_hash", endpoint_hash)
        object.__setattr__(self, "receipt_hash", canonical_hash(payload))

    @property
    def action_id(self) -> str:
        return f"{self.family}::{self.direction}"

    def endpoint_array(self) -> np.ndarray:
        return canonical_probabilities(self.endpoint_probabilities)


@dataclass(frozen=True, slots=True)
class EndpointProjectionReceipt:
    endpoint: EndpointSurfaceReceipt
    projection: BoundaryProjection
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.endpoint.action_id != self.projection.action_id
            or self.endpoint.endpoint_probability_hash
            != self.projection.source_endpoint_hash
            or len(self.endpoint.endpoint_probabilities) != self.projection.row_count
        ):
            raise ProtocolError("SCALE-BP endpoint/projection lineage drifted.")
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_endpoint_projection_receipt_v1",
                    "endpoint_receipt_hash": self.endpoint.receipt_hash,
                    "geometry_hash": self.projection.geometry_hash,
                }
            ),
        )


def _derive_endpoint(
    rows: tuple[PhysicalCellSurface, ...],
    *,
    target_center: str,
    case_id: str,
    family: str,
    direction: str,
) -> tuple[tuple[PhysicalCellSurface, ...], np.ndarray]:
    """Validate the exact rectangle and recompute its deterministic endpoint."""

    expected = tuple(
        (action, training_seed, generation_seed)
        for action in action_ids_for_target(target_center)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    by_key = {
        (
            row.identity.action_id,
            row.identity.training_seed,
            row.identity.generation_seed,
        ): row
        for row in rows
        if isinstance(row, PhysicalCellSurface)
    }
    if (
        len(rows) != 90
        or len(by_key) != 90
        or set(by_key) != set(expected)
        or any(row.identity.target_center != target_center for row in rows)
        or {row.case_id for row in rows} != {str(case_id)}
        or len({row.cache_content_hash for row in rows}) != 1
        or len({row.row_order_hash for row in rows}) != 1
        or len({row.physical_bank_receipt_hash for row in rows}) != 1
        or len({len(row.probabilities) for row in rows}) != 1
        or family not in ACTION_FAMILIES
        or direction not in DIRECTIONS
    ):
        raise ProtocolError("SCALE-BP endpoint physical rectangle drifted.")

    action_means: dict[str, np.ndarray] = {}
    for action in action_ids_for_target(target_center):
        values = np.asarray(
            [by_key[(action, training, generation)].probabilities for training in TRAINING_SEEDS for generation in GENERATION_SEEDS],
            dtype=np.float64,
        )
        action_means[action] = np.mean(values, axis=0, dtype=np.float64)
    if family == "B":
        endpoint = action_means[B_ACTION_ID]
    else:
        a1 = np.asarray(
            [
                action_means[action]
                for action in action_ids_for_target(target_center)
                if action.startswith(A1_PREFIX)
            ],
            dtype=np.float64,
        )
        if family == "I":
            endpoint = (
                np.max(a1, axis=0)
                if direction == "zero_to_one"
                else np.min(a1, axis=0)
            )
        else:
            robust = np.vstack((action_means[U_ACTION_ID][None, :], a1))
            endpoint = np.median(robust, axis=0)
    endpoint32 = np.ascontiguousarray(endpoint, dtype=np.float32)
    ordered_rows = tuple(by_key[key] for key in expected)
    return ordered_rows, endpoint32


def assemble_endpoint_surface(
    surfaces: object,
    *,
    target_center: str,
    case_id: str,
    family: str,
    direction: str,
) -> EndpointSurfaceReceipt:
    """Derive one endpoint from exactly 10 actions x 9 frozen seed pairs."""

    return EndpointSurfaceReceipt(
        target_center=str(target_center),
        case_id=str(case_id),
        family=str(family),
        direction=str(direction),
        physical_surfaces=tuple(surfaces),  # type: ignore[arg-type]
    )


def build_projection_from_endpoint(
    portfolio: object, endpoint: EndpointSurfaceReceipt
) -> EndpointProjectionReceipt:
    projection = build_boundary_projection(
        portfolio,
        endpoint.endpoint_probabilities,
        family=endpoint.family,
        direction=endpoint.direction,
    )
    return EndpointProjectionReceipt(endpoint, projection)


__all__ = (
    "DERIVATION_RULES",
    "EndpointProjectionReceipt",
    "EndpointSurfaceReceipt",
    "PhysicalCellSurface",
    "assemble_endpoint_surface",
    "build_projection_from_endpoint",
)
