"""Manifest-derived whole-case route identity receipts for SCALE-BP."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Sequence

from .case_inventory import DatasetCaseInventory
from .hashing import canonical_hash, require_sha256
from .identity import CENTERS, EXPECTED_CASE_COUNT
from .protocol import ProtocolError


_IDENTITY_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True, order=True)
class SampleIdentity:
    """One label-free manifest row identity."""

    center: str
    case_id: str
    group_id: str
    patient_id: str
    slide_id: str
    sample_id: str

    def __post_init__(self) -> None:
        values = tuple(
            str(value)
            for value in (
                self.center,
                self.case_id,
                self.group_id,
                self.patient_id,
                self.slide_id,
                self.sample_id,
            )
        )
        if values[0] not in CENTERS or any(not value for value in values[1:]):
            raise ProtocolError("SCALE-BP sample identity drifted.")
        for name, value in zip(
            ("center", "case_id", "group_id", "patient_id", "slide_id", "sample_id"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.center, self.case_id, self.sample_id


@dataclass(frozen=True, slots=True)
class RouteCaseBinding:
    """Exact label-free sample-key receipt for one whole case."""

    center: str
    case_id: str
    group_id: str
    patient_id: str
    slide_id: str
    row_count: int
    sample_key_hash: str
    _factory_token: InitVar[object] = None
    binding_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _IDENTITY_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP route-case binding bypassed manifest derivation."
            )
        values = tuple(
            str(value)
            for value in (
                self.center,
                self.case_id,
                self.group_id,
                self.patient_id,
                self.slide_id,
            )
        )
        count = int(self.row_count)
        key_hash = require_sha256(self.sample_key_hash, "case sample-key hash")
        if values[0] not in CENTERS or any(not value for value in values[1:]) or count <= 0:
            raise ProtocolError("SCALE-BP route-case binding drifted.")
        for name, value in zip(
            ("center", "case_id", "group_id", "patient_id", "slide_id"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "row_count", count)
        object.__setattr__(self, "sample_key_hash", key_hash)
        object.__setattr__(
            self,
            "binding_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_route_case_binding_v1",
                    "center": values[0],
                    "case_id": values[1],
                    "group_id": values[2],
                    "patient_id": values[3],
                    "slide_id": values[4],
                    "row_count": count,
                    "sample_key_hash": key_hash,
                    "labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RouteIdentityInventory:
    """Compact exact-key identity inventory derived once from the manifest."""

    case_inventory: DatasetCaseInventory
    case_bindings: tuple[RouteCaseBinding, ...]
    population_key_hash: str
    _factory_token: InitVar[object] = None
    inventory_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _IDENTITY_FACTORY_TOKEN:
            raise ProtocolError(
                "SCALE-BP route identity inventory bypassed manifest derivation."
            )
        if not isinstance(self.case_inventory, DatasetCaseInventory):
            raise ProtocolError("SCALE-BP route identity case inventory drifted.")
        bindings = tuple(self.case_bindings)
        population_hash = require_sha256(
            self.population_key_hash,
            "route identity population-key hash",
        )
        expected_cases = tuple(
            (center, case)
            for center in CENTERS
            for case in self.case_inventory.cases(center)
        )
        if (
            len(bindings) != EXPECTED_CASE_COUNT
            or any(not isinstance(row, RouteCaseBinding) for row in bindings)
            or tuple((row.center, row.case_id) for row in bindings) != expected_cases
            or len({row.binding_hash for row in bindings}) != len(bindings)
        ):
            raise ProtocolError("SCALE-BP route identity inventory drifted.")
        object.__setattr__(self, "case_bindings", bindings)
        object.__setattr__(self, "population_key_hash", population_hash)
        object.__setattr__(
            self,
            "inventory_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_route_identity_inventory_v1",
                    "case_inventory_hash": self.case_inventory.inventory_hash,
                    "cache_content_hash": self.case_inventory.cache_content_hash,
                    "row_order_hash": self.case_inventory.row_order_hash,
                    "manifest_hash": self.case_inventory.manifest_hash,
                    "population_key_hash": population_hash,
                    "case_binding_hashes": tuple(row.binding_hash for row in bindings),
                    "case_count": EXPECTED_CASE_COUNT,
                    "labels_used": False,
                }
            ),
        )

    def binding(self, center: str, case_id: str) -> RouteCaseBinding:
        matches = tuple(
            row
            for row in self.case_bindings
            if row.center == center and row.case_id == case_id
        )
        if len(matches) != 1:
            raise ProtocolError("SCALE-BP route case binding lookup drifted.")
        return matches[0]


@dataclass(frozen=True, slots=True)
class RouteScopeWitness:
    """Exact evaluation/support scope for one H/c route."""

    target_center: str
    held_case_id: str
    identity_inventory: RouteIdentityInventory
    witness_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        held = str(self.held_case_id)
        if (
            target not in CENTERS
            or not held
            or not isinstance(self.identity_inventory, RouteIdentityInventory)
            or held not in self.identity_inventory.case_inventory.cases(target)
        ):
            raise ProtocolError("SCALE-BP route scope witness drifted.")
        evaluation = self.identity_inventory.binding(target, held)
        support = self.support_bindings
        if (
            tuple(row.case_id for row in support)
            != tuple(
                case
                for case in self.identity_inventory.case_inventory.cases(target)
                if case != held
            )
            or evaluation.group_id in {row.group_id for row in support}
            or evaluation.patient_id in {row.patient_id for row in support}
            or evaluation.slide_id in {row.slide_id for row in support}
        ):
            raise ProtocolError("SCALE-BP route support/evaluation witness drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "held_case_id", held)
        object.__setattr__(
            self,
            "witness_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_route_scope_witness_v2",
                    "target_center": target,
                    "held_case_id": held,
                    "route_identity_inventory_hash": self.identity_inventory.inventory_hash,
                    "evaluation_binding_hash": evaluation.binding_hash,
                    "evaluation_sample_key_hash": evaluation.sample_key_hash,
                    "support_binding_hashes": tuple(row.binding_hash for row in support),
                    "support_sample_key_hash": self.support_sample_key_hash,
                    "whole_case_patient_slide_group_disjoint": True,
                    "labels_used": False,
                }
            ),
        )

    @property
    def evaluation_binding(self) -> RouteCaseBinding:
        return self.identity_inventory.binding(self.target_center, self.held_case_id)

    @property
    def support_bindings(self) -> tuple[RouteCaseBinding, ...]:
        return tuple(
            row
            for row in self.identity_inventory.case_bindings
            if row.center == self.target_center and row.case_id != self.held_case_id
        )

    @property
    def support_case_ids(self) -> tuple[str, ...]:
        return tuple(row.case_id for row in self.support_bindings)

    @property
    def support_sample_key_hash(self) -> str:
        return canonical_hash(
            {
                "schema_version": "scale_bp_route_support_sample_keys_v1",
                "case_sample_key_hashes": tuple(
                    (row.case_id, row.row_count, row.sample_key_hash)
                    for row in self.support_bindings
                ),
            }
        )

    @property
    def held_group_id(self) -> str:
        return self.evaluation_binding.group_id

    @property
    def held_patient_id(self) -> str:
        return self.evaluation_binding.patient_id

    @property
    def held_slide_id(self) -> str:
        return self.evaluation_binding.slide_id


def build_route_identity_inventory(
    identities: Sequence[SampleIdentity],
    *,
    case_inventory: DatasetCaseInventory,
) -> RouteIdentityInventory:
    """Derive the exact compact identity receipt from all manifest rows."""

    rows = tuple(identities)
    if (
        not rows
        or any(not isinstance(row, SampleIdentity) for row in rows)
        or len({row.key for row in rows}) != len(rows)
        or not isinstance(case_inventory, DatasetCaseInventory)
    ):
        raise ProtocolError("SCALE-BP route identity population drifted.")
    ordered = tuple(sorted(rows, key=lambda row: row.key))
    expected_cases = {
        (center, case)
        for center in CENTERS
        for case in case_inventory.cases(center)
    }
    if {(row.center, row.case_id) for row in ordered} != expected_cases:
        raise ProtocolError("SCALE-BP route identity case population is incomplete.")
    bindings: list[RouteCaseBinding] = []
    group_owner: dict[str, tuple[str, str]] = {}
    patient_owner: dict[str, tuple[str, str]] = {}
    slide_owner: dict[str, tuple[str, str]] = {}
    for center in CENTERS:
        for case in case_inventory.cases(center):
            case_rows = tuple(
                row for row in ordered if row.center == center and row.case_id == case
            )
            groups = {row.group_id for row in case_rows}
            patients = {row.patient_id for row in case_rows}
            slides = {row.slide_id for row in case_rows}
            if len(groups) != 1 or len(patients) != 1 or len(slides) != 1:
                raise ProtocolError(
                    "SCALE-BP one case spans multiple group/patient/slide identities."
                )
            owner = (center, case)
            for identity, owners in (
                (next(iter(groups)), group_owner),
                (next(iter(patients)), patient_owner),
                (next(iter(slides)), slide_owner),
            ):
                prior = owners.setdefault(identity, owner)
                if prior != owner:
                    raise ProtocolError(
                        "SCALE-BP group/patient/slide identity spans multiple cases."
                    )
            keys = tuple(row.key for row in case_rows)
            bindings.append(
                RouteCaseBinding(
                    center=center,
                    case_id=case,
                    group_id=next(iter(groups)),
                    patient_id=next(iter(patients)),
                    slide_id=next(iter(slides)),
                    row_count=len(case_rows),
                    sample_key_hash=canonical_hash(
                        {
                            "schema_version": "scale_bp_case_sample_keys_v1",
                            "keys": keys,
                        }
                    ),
                    _factory_token=_IDENTITY_FACTORY_TOKEN,
                )
            )
    return RouteIdentityInventory(
        case_inventory=case_inventory,
        case_bindings=tuple(bindings),
        population_key_hash=canonical_hash(
            {
                "schema_version": "scale_bp_route_identity_population_keys_v1",
                "keys": tuple(row.key for row in ordered),
            }
        ),
        _factory_token=_IDENTITY_FACTORY_TOKEN,
    )


def build_route_scope_witness(
    identities: Sequence[SampleIdentity],
    *,
    target_center: str,
    held_case_id: str,
    case_inventory: DatasetCaseInventory,
) -> RouteScopeWitness:
    inventory = build_route_identity_inventory(
        identities,
        case_inventory=case_inventory,
    )
    return RouteScopeWitness(str(target_center), str(held_case_id), inventory)


__all__ = (
    "RouteCaseBinding",
    "RouteIdentityInventory",
    "RouteScopeWitness",
    "SampleIdentity",
    "build_route_identity_inventory",
    "build_route_scope_witness",
)
