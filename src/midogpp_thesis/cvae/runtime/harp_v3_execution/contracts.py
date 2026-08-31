"""Typed protocol boundary between HARP v3 orchestration and numerics.

Only label-free physical probability bytes cross the materialization boundary.
Development and evaluation labels are passed to distinct methods after the
runner has durably crossed the corresponding access barriers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash


class ActionKind(str, Enum):
    B = "B"
    U = "U"
    HXE = "Hxe"


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ProtocolError(f"HARP v3 {name} is not a canonical identity.")
    return value


def _float32_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.dtype != np.dtype("float32")
        or raw.ndim != 1
        or not raw.size
        or not np.isfinite(raw).all()
        or np.any((raw < 0.0) | (raw > 1.0))
    ):
        raise ProtocolError(f"HARP v3 {name} must be a finite float32 probability vector.")
    # A bytes-backed view cannot later be made writeable by a caller.
    return np.frombuffer(np.ascontiguousarray(raw).tobytes(order="C"), dtype=np.float32)


def array_bytes_sha256(values: np.ndarray) -> str:
    raw = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(raw.dtype).encode("ascii"))
    digest.update(str(tuple(int(value) for value in raw.shape)).encode("ascii"))
    digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LabelFreeActionBlock:
    """One physical lambda=1 probability vector in an outer-H menu."""

    surface_role: str
    outer_target_id: str
    query_center_id: str
    action_kind: ActionKind
    selected_source_id: str | None
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]
    probabilities: np.ndarray
    block_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.surface_role not in {"development", "target"}:
            raise ProtocolError("HARP v3 action block has an unknown surface role.")
        outer = _identifier(self.outer_target_id, name="outer target H")
        query = _identifier(self.query_center_id, name="query center q")
        try:
            kind = ActionKind(self.action_kind)
        except ValueError as exc:
            raise ProtocolError("HARP v3 action kind is outside B/U/Hxe.") from exc
        source = self.selected_source_id
        if kind is ActionKind.HXE:
            source = _identifier(source, name="selected expert e")
            if source in {outer, query}:
                raise ProtocolError("HARP v3 expert action violated H/q exclusion.")
        elif source is not None:
            raise ProtocolError("HARP v3 B/U actions cannot carry an expert source.")
        if self.surface_role == "development" and query == outer:
            raise ProtocolError("HARP v3 development query leaked outer target H.")
        if self.surface_role == "target" and query != outer:
            raise ProtocolError("HARP v3 target action is not query H.")
        samples = tuple(_identifier(value, name="sample id") for value in self.sample_ids)
        cases = tuple(_identifier(value, name="case id") for value in self.case_ids)
        values = _float32_vector(self.probabilities, name="transport")
        if len(samples) != len(cases) or len(samples) != len(values):
            raise ProtocolError("HARP v3 action identities and probabilities are misaligned.")
        if len(samples) != len(set(samples)):
            raise ProtocolError("HARP v3 action block duplicates a sample identity.")
        base = {
            "schema_version": "midogpp_harp_v3_label_free_action_block_v1",
            "surface_role": self.surface_role,
            "outer_target_id": outer,
            "query_center_id": query,
            "action_kind": kind.value,
            "selected_source_id": source,
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "probability_bytes_sha256": array_bytes_sha256(values),
            "probability_dtype": "float32",
            "physical_expert_weight": 1.0 if kind is ActionKind.HXE else None,
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_center_id", query)
        object.__setattr__(self, "action_kind", kind)
        object.__setattr__(self, "selected_source_id", source)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "probabilities", values)
        object.__setattr__(self, "block_hash", canonical_hash(base))

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.surface_role,
            self.outer_target_id,
            self.query_center_id,
            self.action_kind.value,
            self.selected_source_id or "",
        )


@dataclass(frozen=True, slots=True)
class LabelFreeOuterMenu:
    """Complete physical B/U/Hxe menu for one excluded target H."""

    outer_target_id: str
    blocks: tuple[LabelFreeActionBlock, ...]
    lineage: Mapping[str, object]
    menu_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _identifier(self.outer_target_id, name="outer target H")
        blocks = tuple(self.blocks)
        if not blocks or any(
            not isinstance(block, LabelFreeActionBlock)
            or block.outer_target_id != outer
            for block in blocks
        ):
            raise ProtocolError("HARP v3 outer menu has invalid block membership.")
        ordered = tuple(sorted(blocks, key=lambda block: block.key))
        if blocks != ordered or len({block.key for block in blocks}) != len(blocks):
            raise ProtocolError("HARP v3 outer menu order or uniqueness drifted.")
        by_context: dict[tuple[str, str], list[LabelFreeActionBlock]] = {}
        for block in blocks:
            by_context.setdefault((block.surface_role, block.query_center_id), []).append(block)
        for scoped in by_context.values():
            first = scoped[0]
            if any(
                row.sample_ids != first.sample_ids or row.case_ids != first.case_ids
                for row in scoped[1:]
            ):
                raise ProtocolError("HARP v3 B/U/Hxe action identities are misaligned.")
            kinds = [row.action_kind for row in scoped]
            if kinds.count(ActionKind.B) != 1 or kinds.count(ActionKind.U) != 1:
                raise ProtocolError("HARP v3 context lacks exactly one physical B and U.")
            expected_sources = {
                row.selected_source_id for row in scoped if row.action_kind is ActionKind.HXE
            }
            if None in expected_sources or outer in expected_sources or first.query_center_id in expected_sources:
                raise ProtocolError("HARP v3 context contains an illegal Hxe source.")
        if not any(role == "development" for role, _ in by_context) or (
            "target", outer
        ) not in by_context:
            raise ProtocolError("HARP v3 outer menu lacks development or target actions.")
        lineage = MappingProxyType(dict(self.lineage))
        base = {
            "schema_version": "midogpp_harp_v3_label_free_outer_menu_v1",
            "outer_target_id": outer,
            "block_hashes": [block.block_hash for block in blocks],
            "lineage": dict(lineage),
            "complete_physical_B_U_Hxe": True,
            "physical_expert_weight": 1.0,
            "strict_outer_target_exclusion": True,
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "menu_hash", canonical_hash(base))

    def target_block(self, kind: ActionKind, source: str | None = None) -> LabelFreeActionBlock:
        matched = tuple(
            block
            for block in self.blocks
            if block.surface_role == "target"
            and block.action_kind is kind
            and block.selected_source_id == source
        )
        if len(matched) != 1:
            raise ProtocolError("HARP v3 target action lookup is incomplete or ambiguous.")
        return matched[0]


@dataclass(frozen=True, slots=True)
class ArtifactValue:
    """Opaque in-memory scientific value plus deterministic durable projection."""

    state: object
    manifest: Mapping[str, object]
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        manifest = MappingProxyType(dict(self.manifest))
        arrays: dict[str, np.ndarray] = {}
        for key, value in self.arrays.items():
            name = _identifier(key, name="artifact array name")
            raw = np.asarray(value)
            if raw.dtype.hasobject or not np.isfinite(raw).all():
                raise ProtocolError("HARP v3 artifact arrays must be finite and pickle-free.")
            arrays[name] = np.ascontiguousarray(raw)
            arrays[name].setflags(write=False)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "arrays", MappingProxyType(arrays))


@dataclass(frozen=True, slots=True)
class RoutedCase:
    """One case-consistent B/U/Hxe decision and its physical probability bytes."""

    outer_target_id: str
    case_id: str
    sample_ids: tuple[str, ...]
    selected_kind: ActionKind
    selected_source_id: str | None
    reason: str
    baseline_probabilities: np.ndarray
    uniform_probabilities: np.ndarray
    selected_probabilities: np.ndarray
    routed_probabilities: np.ndarray
    decision_payload: Mapping[str, object] = field(default_factory=dict)
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _identifier(self.outer_target_id, name="route outer target")
        case = _identifier(self.case_id, name="route case")
        samples = tuple(_identifier(value, name="route sample") for value in self.sample_ids)
        if not samples or len(samples) != len(set(samples)):
            raise ProtocolError("HARP v3 routed case has invalid sample membership.")
        try:
            kind = ActionKind(self.selected_kind)
        except ValueError as exc:
            raise ProtocolError("HARP v3 routed case has an unknown action.") from exc
        source = self.selected_source_id
        if kind is ActionKind.HXE:
            source = _identifier(source, name="routed expert source")
            if source == outer:
                raise ProtocolError("HARP v3 routed expert equals held-out target H.")
        elif source is not None:
            raise ProtocolError("HARP v3 B/U route cannot carry an expert source.")
        arrays = tuple(
            _float32_vector(value, name=name)
            for value, name in (
                (self.baseline_probabilities, "case baseline"),
                (self.uniform_probabilities, "case uniform"),
                (self.selected_probabilities, "case selected"),
                (self.routed_probabilities, "case routed"),
            )
        )
        if any(len(value) != len(samples) for value in arrays):
            raise ProtocolError("HARP v3 routed case arrays are not sample aligned.")
        baseline, uniform, selected, routed = arrays
        expected = baseline if kind is ActionKind.B else uniform if kind is ActionKind.U else selected
        if routed.tobytes(order="C") != expected.tobytes(order="C"):
            raise ProtocolError("HARP v3 routed bytes do not equal the selected physical action.")
        payload = MappingProxyType(dict(self.decision_payload))
        base = {
            "schema_version": "midogpp_harp_v3_case_routing_decision_v1",
            "outer_target_id": outer,
            "case_id": case,
            "sample_ids": list(samples),
            "selected_kind": kind.value,
            "selected_source_id": source,
            "reason": _identifier(self.reason, name="route reason"),
            "baseline_bytes_sha256": array_bytes_sha256(baseline),
            "uniform_bytes_sha256": array_bytes_sha256(uniform),
            "selected_bytes_sha256": array_bytes_sha256(selected),
            "routed_bytes_sha256": array_bytes_sha256(routed),
            "decision_payload": dict(payload),
            "case_consistent": True,
            "physical_expert_weight": 1.0 if kind is ActionKind.HXE else None,
            "labels_consumed": False,
        }
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "selected_kind", kind)
        object.__setattr__(self, "selected_source_id", source)
        object.__setattr__(self, "baseline_probabilities", baseline)
        object.__setattr__(self, "uniform_probabilities", uniform)
        object.__setattr__(self, "selected_probabilities", selected)
        object.__setattr__(self, "routed_probabilities", routed)
        object.__setattr__(self, "decision_payload", payload)
        object.__setattr__(self, "decision_hash", canonical_hash(base))


@dataclass(frozen=True, slots=True)
class PrelabelRouteSet:
    cases: tuple[RoutedCase, ...]
    policy_hash: str
    model_hash: str
    target_action_hash: str
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, RoutedCase) for case in cases):
            raise ProtocolError("HARP v3 prelabel route set is empty or untyped.")
        order = tuple((row.outer_target_id, row.case_id) for row in cases)
        if order != tuple(sorted(order)) or len(set(order)) != len(order):
            raise ProtocolError("HARP v3 routed-case order or uniqueness drifted.")
        for name in ("policy_hash", "model_hash", "target_action_hash"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ProtocolError(f"HARP v3 {name} must be SHA-256.")
        base = {
            "schema_version": "midogpp_harp_v3_prelabel_route_set_v1",
            "decision_hashes": [case.decision_hash for case in cases],
            "policy_hash": self.policy_hash,
            "model_hash": self.model_hash,
            "target_action_hash": self.target_action_hash,
            "case_consistent": True,
            "physical_actions_only": True,
            "evaluation_labels_opened": False,
        }
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "route_hash", canonical_hash(base))


@dataclass(frozen=True, slots=True)
class TerminalEvaluation:
    metrics: Mapping[str, object]
    oracle_diagnostic: Mapping[str, object]
    route_reasons: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self, "oracle_diagnostic", MappingProxyType(dict(self.oracle_diagnostic))
        )
        object.__setattr__(self, "route_reasons", MappingProxyType(dict(self.route_reasons)))


class HarpV3Pipeline(Protocol):
    """Production/synthetic numerical service boundary used by the runner."""

    def preflight(self, config: Any, cache: Any) -> Mapping[str, object]: ...

    def materialize_label_free_outer_menus(
        self,
        config: Any,
        cache: Any,
        *,
        outer_targets: Sequence[str],
        scratch_root: Any,
    ) -> Sequence[LabelFreeOuterMenu]: ...

    def build_development_case_surface(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        development_labels: object,
        *,
        config: Any,
    ) -> ArtifactValue: ...

    def fit_source_only_model(
        self, development: ArtifactValue, *, config: Any
    ) -> ArtifactValue: ...

    def build_complete_target_case_actions(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        fit: ArtifactValue,
        *,
        config: Any,
    ) -> ArtifactValue: ...

    def route_case_actions(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        target_actions: ArtifactValue,
        fit: ArtifactValue,
        *,
        config: Any,
    ) -> PrelabelRouteSet: ...

    def evaluate_terminal(
        self,
        routes: PrelabelRouteSet,
        evaluation_truth: object,
        *,
        config: Any,
    ) -> TerminalEvaluation: ...


__all__ = (
    "ActionKind",
    "ArtifactValue",
    "HarpV3Pipeline",
    "LabelFreeActionBlock",
    "LabelFreeOuterMenu",
    "PrelabelRouteSet",
    "RoutedCase",
    "TerminalEvaluation",
    "array_bytes_sha256",
)
