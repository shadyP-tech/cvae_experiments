"""Closed-world complete pseudo-case/action universe and semantic seal."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ..hashing import canonical_hash, require_sha256
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, MetricVector
from ..routing.admission import AdmissionObservation
from .scope import PseudoRouteKey, PseudoRouteScope, build_pseudo_route_scopes


REQUIRED_COMPONENT_ROLES = (
    "physical_store",
    "protected_p_plans",
    "action_rectangles",
    "donor_models",
    "local_models",
    "preargmax_bounds",
    "route_decisions",
)


@dataclass(frozen=True, slots=True)
class PseudoActionRecord:
    scope: PseudoRouteScope
    action_id: str
    predicted: MetricVector
    realized: MetricVector
    descriptor_hash: str
    estimate_hash: str
    value_hash: str
    selected: bool
    structural_noop: bool
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        descriptor_hash = require_sha256(self.descriptor_hash, "descriptor hash")
        estimate_hash = require_sha256(self.estimate_hash, "estimate hash")
        value_hash = require_sha256(self.value_hash, "action-value hash")
        if (
            self.action_id not in ACTION_IDS
            or (self.structural_noop and self.realized != MetricVector.zeros())
            or (self.structural_noop and self.selected)
        ):
            raise GovernanceError("SCALE-BP v2 pseudo action record drifted.")
        object.__setattr__(self, "descriptor_hash", descriptor_hash)
        object.__setattr__(self, "estimate_hash", estimate_hash)
        object.__setattr__(self, "value_hash", value_hash)
        object.__setattr__(
            self,
            "record_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_pseudo_action_record_v1",
                    "scope_hash": self.scope.scope_hash,
                    "action_id": self.action_id,
                    "predicted": self.predicted.to_payload(),
                    "realized": self.realized.to_payload(),
                    "descriptor_hash": descriptor_hash,
                    "estimate_hash": estimate_hash,
                    "value_hash": value_hash,
                    "selected": self.selected,
                    "structural_noop": self.structural_noop,
                    "raw_labels_persisted": False,
                }
            ),
        )

    @property
    def route_key(self) -> PseudoRouteKey:
        return self.scope.key


@dataclass(frozen=True, slots=True)
class PseudoUniverseSeal:
    scopes: tuple[PseudoRouteScope, ...]
    records: tuple[PseudoActionRecord, ...]
    component_hashes: Mapping[str, str]
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        scopes = tuple(self.scopes)
        records = tuple(self.records)
        components = {
            str(role): require_sha256(value, f"{role} component hash")
            for role, value in self.component_hashes.items()
        }
        expected = tuple(
            (scope.key, action_id) for scope in scopes for action_id in ACTION_IDS
        )
        actual = tuple((record.route_key, record.action_id) for record in records)
        scopes_by_hash = {scope.scope_hash: scope for scope in scopes}
        if (
            not scopes
            or len({scope.key for scope in scopes}) != len(scopes)
            or len(scopes_by_hash) != len(scopes)
            or actual != expected
            or len({record.record_hash for record in records}) != len(records)
            or any(record.scope.scope_hash not in scopes_by_hash for record in records)
            or any(
                sum(record.selected for record in records if record.route_key == scope.key) > 1
                for scope in scopes
            )
            or tuple(components) != REQUIRED_COMPONENT_ROLES
            or any(not value for value in components.values())
        ):
            raise GovernanceError("SCALE-BP v2 pseudo universe is incomplete or drifted.")
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "component_hashes", MappingProxyType(components))
        object.__setattr__(
            self,
            "seal_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_pseudo_universe_seal_v1",
                    "scope_count": len(scopes),
                    "record_count": len(records),
                    "scope_hashes": tuple(scope.scope_hash for scope in scopes),
                    "record_hashes": tuple(record.record_hash for record in records),
                    "component_hashes": components,
                    "complete_six_action_rectangle_per_route": True,
                    "all_structural_noops_retained": True,
                    "H_J_d_exclusions_validated": True,
                    "labels_persisted": False,
                }
            ),
        )

    def admission_observations(self) -> tuple[AdmissionObservation, ...]:
        grouped: dict[PseudoRouteKey, list[PseudoActionRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.route_key, []).append(record)
        output: list[AdmissionObservation] = []
        for scope in self.scopes:
            rows = tuple(grouped[scope.key])
            selected = tuple(row.action_id for row in rows if row.selected)
            output.append(
                AdmissionObservation(
                    scope.key.donor_center,
                    f"H={scope.key.outer_center}::J={scope.key.donor_center}::d={scope.key.case_id}",
                    {row.action_id: row.predicted for row in rows},
                    {row.action_id: row.realized for row in rows},
                    selected[0] if selected else None,
                )
            )
        return tuple(output)


def build_pseudo_universe(
    records: Sequence[PseudoActionRecord],
    *,
    case_ids_by_center: Mapping[str, tuple[str, ...]],
    component_hashes: Mapping[str, str],
) -> PseudoUniverseSeal:
    scopes = build_pseudo_route_scopes(case_ids_by_center)
    by_key = {(record.route_key, record.action_id): record for record in records}
    expected = tuple(
        (scope.key, action_id) for scope in scopes for action_id in ACTION_IDS
    )
    if len(by_key) != len(tuple(records)) or set(by_key) != set(expected):
        raise GovernanceError("SCALE-BP v2 pseudo record universe is not closed-world.")
    ordered = tuple(by_key[key] for key in expected)
    return PseudoUniverseSeal(scopes, ordered, component_hashes)


__all__ = (
    "PseudoActionRecord",
    "PseudoUniverseSeal",
    "REQUIRED_COMPONENT_ROLES",
    "build_pseudo_universe",
)
