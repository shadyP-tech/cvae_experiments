"""Typed protocol boundary between HARP v19 orchestration and numerics.

Only label-free physical probability bytes cross the materialization boundary.
Development and evaluation labels are passed to distinct methods after the
runner has durably crossed the corresponding access barriers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from .physical_contracts import PhysicalInputReceipt

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash


class ActionKind(str, Enum):
    B = "B"
    U = "U"
    HXE = "Hxe"
    SOFT_TOPK_PROBABILITY_BLEND = "SOFT_TOPK_PROBABILITY_BLEND"


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ProtocolError(f"HARP v19 {name} is not a canonical identity.")
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
        raise ProtocolError(f"HARP v19 {name} must be a finite float32 probability vector.")
    # A bytes-backed view cannot later be made writeable by a caller.
    return np.frombuffer(np.ascontiguousarray(raw).tobytes(order="C"), dtype=np.float32)


def _float32_dispersion_vector(value: object, *, size: int) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.dtype != np.dtype("float32")
        or raw.ndim != 1
        or len(raw) != size
        or not np.isfinite(raw).all()
        or np.any(raw < 0.0)
    ):
        raise ProtocolError(
            "HARP v19 seed dispersion must be a finite nonnegative float32 vector."
        )
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
    seed_dispersion: np.ndarray | None = None
    block_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.surface_role not in {"source_train", "target"}:
            raise ProtocolError("HARP v19 action block has an unknown surface role.")
        outer = _identifier(self.outer_target_id, name="outer target H")
        query = _identifier(self.query_center_id, name="query center q")
        try:
            kind = ActionKind(self.action_kind)
        except ValueError as exc:
            raise ProtocolError("HARP v19 action kind is outside B/U/Hxe.") from exc
        if kind is ActionKind.SOFT_TOPK_PROBABILITY_BLEND:
            raise ProtocolError(
                "HARP v19 composite routes cannot masquerade as physical action blocks."
            )
        source = self.selected_source_id
        if kind is ActionKind.HXE:
            source = _identifier(source, name="selected expert e")
            if source in {outer, query}:
                raise ProtocolError("HARP v19 expert action violated H/q exclusion.")
        elif source is not None:
            raise ProtocolError("HARP v19 B/U actions cannot carry an expert source.")
        if query != outer:
            raise ProtocolError("HARP v19 support/target action is not query H.")
        samples = tuple(_identifier(value, name="sample id") for value in self.sample_ids)
        cases = tuple(_identifier(value, name="case id") for value in self.case_ids)
        values = _float32_vector(self.probabilities, name="transport")
        dispersion = _float32_dispersion_vector(
            (
                np.zeros(len(values), dtype=np.float32)
                if self.seed_dispersion is None
                else self.seed_dispersion
            ),
            size=len(values),
        )
        if len(samples) != len(cases) or len(samples) != len(values):
            raise ProtocolError("HARP v19 action identities and probabilities are misaligned.")
        if len(samples) != len(set(samples)):
            raise ProtocolError("HARP v19 action block duplicates a sample identity.")
        base = {
            "schema_version": "midogpp_harp_v19_label_free_action_block_v2",
            "surface_role": self.surface_role,
            "outer_target_id": outer,
            "query_center_id": query,
            "action_kind": kind.value,
            "selected_source_id": source,
            "sample_ids": list(samples),
            "case_ids": list(cases),
            "probability_bytes_sha256": array_bytes_sha256(values),
            "seed_dispersion_bytes_sha256": array_bytes_sha256(dispersion),
            "probability_dtype": "float32",
            "seed_dispersion_dtype": "float32",
            "seed_dispersion_replications": 9,
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
        object.__setattr__(self, "seed_dispersion", dispersion)
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
        if outer not in CENTERS:
            raise ProtocolError("HARP v19 outer menu center is outside canonical C.")
        blocks = tuple(self.blocks)
        if not blocks or any(
            not isinstance(block, LabelFreeActionBlock)
            or block.outer_target_id != outer
            for block in blocks
        ):
            raise ProtocolError("HARP v19 outer menu has invalid block membership.")
        ordered = tuple(sorted(blocks, key=lambda block: block.key))
        if blocks != ordered or len({block.key for block in blocks}) != len(blocks):
            raise ProtocolError("HARP v19 outer menu order or uniqueness drifted.")
        by_context: dict[tuple[str, str], list[LabelFreeActionBlock]] = {}
        for block in blocks:
            by_context.setdefault((block.surface_role, block.query_center_id), []).append(block)
        expected_contexts = {("source_train", outer), ("target", outer)}
        if set(by_context) != expected_contexts:
            raise ProtocolError(
                "HARP v19 outer menu must contain exactly source-train and target contexts."
            )
        legal_sources = frozenset(center for center in CENTERS if center != outer)
        for scoped in by_context.values():
            first = scoped[0]
            if any(
                row.sample_ids != first.sample_ids or row.case_ids != first.case_ids
                for row in scoped[1:]
            ):
                raise ProtocolError("HARP v19 B/U/Hxe action identities are misaligned.")
            kinds = [row.action_kind for row in scoped]
            if kinds.count(ActionKind.B) != 1 or kinds.count(ActionKind.U) != 1:
                raise ProtocolError("HARP v19 context lacks exactly one physical B and U.")
            observed_sources = {
                row.selected_source_id for row in scoped if row.action_kind is ActionKind.HXE
            }
            hxe_count = sum(row.action_kind is ActionKind.HXE for row in scoped)
            if observed_sources != legal_sources or hxe_count != len(legal_sources):
                raise ProtocolError(
                    "HARP v19 context must contain exactly the eight legal C-minus-context Hxe donors."
                )
        support = by_context[("source_train", outer)]
        target = by_context[("target", outer)]
        support_ids = tuple(
            (row.action_kind.value, row.selected_source_id or "") for row in support
        )
        target_ids = tuple(
            (row.action_kind.value, row.selected_source_id or "") for row in target
        )
        if support_ids != target_ids:
            raise ProtocolError("HARP v19 support/target action inventories differ.")
        lineage = MappingProxyType(dict(self.lineage))
        base = {
            "schema_version": "midogpp_harp_v19_label_free_outer_menu_v1",
            "outer_target_id": outer,
            "block_hashes": [block.block_hash for block in blocks],
            "lineage": dict(lineage),
            "complete_physical_B_U_Hxe": True,
            "canonical_center_ids": list(CENTERS),
            "candidate_pool_semantics": "C_MINUS_CONTEXT_CENTER",
            "legal_hxe_source_ids": [center for center in CENTERS if center != outer],
            "exact_hxe_donor_count_per_context": len(legal_sources),
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
            raise ProtocolError("HARP v19 target action lookup is incomplete or ambiguous.")
        return matched[0]

    def support_block(
        self, kind: ActionKind, source: str | None = None
    ) -> LabelFreeActionBlock:
        """Compatibility spelling for the source-train development block."""

        return self.source_train_block(kind, source)

    def source_train_block(
        self, kind: ActionKind, source: str | None = None
    ) -> LabelFreeActionBlock:
        matched = tuple(
            block
            for block in self.blocks
            if block.surface_role == "source_train"
            and block.action_kind is kind
            and block.selected_source_id == source
        )
        if len(matched) != 1:
            raise ProtocolError(
                "HARP v19 source-train action lookup is incomplete or ambiguous."
            )
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
                raise ProtocolError("HARP v19 artifact arrays must be finite and pickle-free.")
            arrays[name] = np.ascontiguousarray(raw)
            arrays[name].setflags(write=False)
        object.__setattr__(self, "manifest", manifest)
        object.__setattr__(self, "arrays", MappingProxyType(arrays))


def reconstruct_selected_probability_blend(
    component_probabilities: Sequence[np.ndarray],
    component_weights: Sequence[float],
    *,
    baseline_probabilities: np.ndarray | None = None,
    component_action_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Reconstruct one selected top-K surface with the frozen reduction order.

    Physical probabilities remain float32.  Convex composition is accumulated
    in float64, in declared component order, and cast exactly once at the
    transport boundary.  The function intentionally accepts one component so
    K=1 follows the same contract as every other selected recipe.
    """

    components = tuple(
        _float32_vector(value, name="route component probability")
        for value in component_probabilities
    )
    weights = tuple(float(value) for value in component_weights)
    if (
        not components
        or len(components) != len(weights)
        or len({len(value) for value in components}) != 1
        or any(not np.isfinite(value) or value <= 0.0 for value in weights)
    ):
        raise ProtocolError("HARP v19 soft top-K component recipe is malformed.")
    stacked = np.stack(components).astype(np.float64, copy=False)
    ids = (
        tuple(_identifier(value, name="route component action") for value in component_action_ids)
        if component_action_ids is not None
        else ()
    )
    if ids and len(ids) != len(components):
        raise ProtocolError("HARP v19 component identities are misaligned.")
    baseline = (
        None
        if baseline_probabilities is None
        else _float32_vector(baseline_probabilities, name="case baseline")
    )
    if baseline is not None and baseline.shape != components[0].shape:
        raise ProtocolError("HARP v19 blend baseline geometry is malformed.")

    groups: tuple[tuple[int, ...], ...]
    if ids:
        d01 = tuple(index for index, value in enumerate(ids) if value.endswith(":D01"))
        d10 = tuple(index for index, value in enumerate(ids) if value.endswith(":D10"))
        full = tuple(index for index, value in enumerate(ids) if value == "U:FULL")
        if full:
            if len(full) != 1 or len(ids) != 1:
                raise ProtocolError("HARP v19 FULL-U recipe is not singular.")
            groups = (full,)
        elif d01 and d10 and len(d01) == len(d10) and len(d01) + len(d10) == len(ids):
            groups = (d01, d10)
        elif (d01 or d10) and not (d01 and d10) and len(d01) + len(d10) == len(ids):
            groups = (d01 or d10,)
        else:
            raise ProtocolError("HARP v19 component directions are malformed.")
    else:
        groups = (tuple(range(len(components))),)
    for group in groups:
        if not math.isclose(
            sum(weights[index] for index in group),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ProtocolError("HARP v19 branch component weights do not sum to one.")

    if len(groups) == 2:
        if baseline is None:
            raise ProtocolError("HARP v19 directional blend requires exact B.")
        selected64 = baseline.astype(np.float64, copy=True)
        baseline_positive = baseline >= np.float32(0.5)
        for group, branch_mask in zip(
            groups, (~baseline_positive, baseline_positive), strict=True
        ):
            branch = np.zeros(stacked.shape[1], dtype=np.float64)
            for index in group:
                branch += weights[index] * stacked[index]
            selected64[branch_mask] = branch[branch_mask]
        selected = np.ascontiguousarray(selected64, dtype="<f4")
    else:
        group = groups[0]
        selected64 = np.zeros(stacked.shape[1], dtype=np.float64)
        for index in group:
            selected64 += weights[index] * stacked[index]
        selected = np.ascontiguousarray(selected64, dtype="<f4")
    if baseline is not None:
        component_bytes = stacked.astype("<f4", copy=False).view(np.uint32)
        baseline_bytes = np.asarray(baseline, dtype="<f4").view(np.uint32)
        active = np.any(component_bytes != baseline_bytes[None, :], axis=0)
        selected[~active] = baseline[~active]
    return selected


def reconstruct_shrunk_probability_blend(
    baseline_probabilities: np.ndarray,
    selected_probabilities: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Reconstruct ``(1-lambda) B + lambda selected`` byte-exactly."""

    baseline = _float32_vector(baseline_probabilities, name="case baseline")
    selected = _float32_vector(selected_probabilities, name="case selected")
    lam = float(shrinkage)
    if (
        baseline.shape != selected.shape
        or not np.isfinite(lam)
        or lam <= 0.0
        or lam > 1.0
    ):
        raise ProtocolError("HARP v19 soft top-K shrinkage recipe is malformed.")
    if lam == 1.0:
        routed = np.ascontiguousarray(selected, dtype="<f4").copy()
    else:
        routed = np.ascontiguousarray(
            (1.0 - lam) * baseline.astype(np.float64)
            + lam * selected.astype(np.float64),
            dtype="<f4",
        )
    unchanged = selected.view(np.uint32) == baseline.view(np.uint32)
    routed[unchanged] = baseline[unchanged]
    return routed


@dataclass(frozen=True, slots=True)
class RoutedCase:
    """One case-consistent soft top-K recipe or protected exact-B fallback."""

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
    direction: str | None = None
    shrinkage: float = 0.0
    component_action_ids: tuple[str, ...] = ()
    component_weights: tuple[float, ...] = ()
    component_probabilities: tuple[np.ndarray, ...] = ()
    decision_payload: Mapping[str, object] = field(default_factory=dict)
    recipe_kind: str | None = None
    recipe_hash: str = field(init=False)
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _identifier(self.outer_target_id, name="route outer target")
        case = _identifier(self.case_id, name="route case")
        samples = tuple(_identifier(value, name="route sample") for value in self.sample_ids)
        if not samples or len(samples) != len(set(samples)):
            raise ProtocolError("HARP v19 routed case has invalid sample membership.")
        try:
            kind = ActionKind(self.selected_kind)
        except ValueError as exc:
            raise ProtocolError("HARP v19 routed case has an unknown action.") from exc
        source = self.selected_source_id
        if kind is ActionKind.HXE:
            source = _identifier(source, name="routed expert source")
            if source == outer:
                raise ProtocolError("HARP v19 routed expert equals held-out target H.")
        elif source is not None:
            raise ProtocolError("HARP v19 B/U/composite route cannot carry an expert source.")
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
            raise ProtocolError("HARP v19 routed case arrays are not sample aligned.")
        baseline, uniform, selected, routed = arrays
        direction = self.direction
        shrinkage = float(self.shrinkage)
        action_ids = tuple(
            _identifier(value, name="route component action")
            for value in self.component_action_ids
        )
        weights = tuple(float(value) for value in self.component_weights)
        components = tuple(
            _float32_vector(value, name="route component probability")
            for value in self.component_probabilities
        )
        if any(len(value) != len(samples) for value in components):
            raise ProtocolError("HARP v19 route components are not sample aligned.")
        recipe_kind = (
            "EXACT_B"
            if kind is ActionKind.B
            else "EXACT_U_FULL"
            if kind is ActionKind.U
            else "SOFT_TOPK_PROBABILITY_BLEND"
        )
        if self.recipe_kind is not None and self.recipe_kind != recipe_kind:
            raise ProtocolError("HARP v19 route recipe kind drifted.")
        if kind is ActionKind.B:
            if (
                direction is not None
                or shrinkage != 0.0
                or action_ids
                or weights
                or components
                or selected.tobytes(order="C") != baseline.tobytes(order="C")
                or routed.tobytes(order="C") != baseline.tobytes(order="C")
            ):
                raise ProtocolError("HARP v19 OFF route is not exact B.")
        else:
            if (
                kind not in {
                    ActionKind.U,
                    ActionKind.HXE,
                    ActionKind.SOFT_TOPK_PROBABILITY_BLEND,
                }
                or direction not in {"D01", "D10", "FULL", "MIXED"}
                or not (0.0 < shrinkage <= 1.0)
                or not action_ids
                or len(action_ids) != len(set(action_ids))
                or len(action_ids) != len(weights)
                or len(action_ids) != len(components)
            ):
                raise ProtocolError("HARP v19 soft top-K action contract drifted.")
            if kind is ActionKind.U:
                if action_ids != ("U:FULL",) or direction != "FULL":
                    raise ProtocolError("HARP v19 U route has an invalid FULL action.")
            elif kind is ActionKind.HXE:
                if (
                    source is None
                    or len(action_ids) != 1
                    or action_ids[0] != f"HXE:{source}:{direction}"
                    or direction not in {"D01", "D10"}
                ):
                    raise ProtocolError("HARP v19 Hxe route has an invalid action.")
            elif kind is ActionKind.SOFT_TOPK_PROBABILITY_BLEND:
                from .branch_recipe import validate_branch_recipe
                validate_branch_recipe(
                    direction=direction, component_ids=action_ids, components=components,
                    baseline=baseline, routed=routed, payload=self.decision_payload,
                )
            expected_selected = reconstruct_selected_probability_blend(
                components,
                weights,
                baseline_probabilities=baseline,
                component_action_ids=action_ids,
            )
            expected_routed = reconstruct_shrunk_probability_blend(
                baseline, expected_selected, shrinkage
            )
            if (
                selected.tobytes(order="C")
                != expected_selected.tobytes(order="C")
                or routed.tobytes(order="C")
                != expected_routed.tobytes(order="C")
            ):
                raise ProtocolError(
                    "HARP v19 routed bytes do not reconstruct from the selected recipe."
                )
        payload = MappingProxyType(dict(self.decision_payload))
        recipe_body = {
            "schema_version": "midogpp_harp_v19_probability_recipe_v1",
            "recipe_kind": recipe_kind,
            "component_action_ids": list(action_ids),
            "component_weights": list(weights),
            "component_probability_sha256": [
                array_bytes_sha256(value) for value in components
            ],
            "shrinkage": shrinkage,
            "baseline_bytes_sha256": array_bytes_sha256(baseline),
            "selected_bytes_sha256": array_bytes_sha256(selected),
            "routed_bytes_sha256": array_bytes_sha256(routed),
            "float32_transport_float64_composition": True,
        }
        recipe_hash = canonical_hash(recipe_body)
        base = {
            "schema_version": "midogpp_harp_v19_case_routing_decision_v4",
            "outer_target_id": outer,
            "case_id": case,
            "sample_ids": list(samples),
            "selected_kind": kind.value,
            "selected_source_id": source,
            "direction": direction,
            "shrinkage": shrinkage,
            "component_action_ids": list(action_ids),
            "component_weights": list(weights),
            "component_probability_sha256": [
                array_bytes_sha256(value) for value in components
            ],
            "recipe_kind": recipe_kind,
            "recipe_hash": recipe_hash,
            "reason": _identifier(self.reason, name="route reason"),
            "baseline_bytes_sha256": array_bytes_sha256(baseline),
            "uniform_bytes_sha256": array_bytes_sha256(uniform),
            "selected_bytes_sha256": array_bytes_sha256(selected),
            "routed_bytes_sha256": array_bytes_sha256(routed),
            "decision_payload": dict(payload),
            "case_consistent": True,
            "soft_topk_probability_blend": kind
            in {ActionKind.HXE, ActionKind.SOFT_TOPK_PROBABILITY_BLEND},
            "all_k_lambda_probability_matrices_persisted": False,
            "exact_b_fallback": kind is ActionKind.B,
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
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "shrinkage", shrinkage)
        object.__setattr__(self, "component_action_ids", action_ids)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(self, "component_probabilities", components)
        object.__setattr__(self, "decision_payload", payload)
        object.__setattr__(self, "recipe_kind", recipe_kind)
        object.__setattr__(self, "recipe_hash", recipe_hash)
        object.__setattr__(self, "decision_hash", canonical_hash(base))


@dataclass(frozen=True, slots=True)
class PrelabelRouteSet:
    cases: tuple[RoutedCase, ...]
    policy_hash: str
    model_hash: str
    target_action_hash: str
    ordered_case_identity_hash: str = field(init=False)
    ordered_sample_identity_hash: str = field(init=False)
    route_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        if not cases or any(not isinstance(case, RoutedCase) for case in cases):
            raise ProtocolError("HARP v19 prelabel route set is empty or untyped.")
        order = tuple((row.outer_target_id, row.case_id) for row in cases)
        if order != tuple(sorted(order)) or len(set(order)) != len(order):
            raise ProtocolError("HARP v19 routed-case order or uniqueness drifted.")
        for name in ("policy_hash", "model_hash", "target_action_hash"):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ProtocolError(f"HARP v19 {name} must be SHA-256.")
        case_identity_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v19_ordered_target_case_identity_v1",
                "ordered_cases": [list(value) for value in order],
            }
        )
        sample_identity_hash = canonical_hash(
            {
                "schema_version": "midogpp_harp_v19_ordered_target_sample_identity_v1",
                "ordered_case_samples": [
                    {
                        "outer_target_id": case.outer_target_id,
                        "case_id": case.case_id,
                        "sample_ids": list(case.sample_ids),
                    }
                    for case in cases
                ],
            }
        )
        base = {
            "schema_version": "midogpp_harp_v19_prelabel_route_set_v3",
            "decision_hashes": [case.decision_hash for case in cases],
            "ordered_case_identity_hash": case_identity_hash,
            "ordered_sample_identity_hash": sample_identity_hash,
            "policy_hash": self.policy_hash,
            "model_hash": self.model_hash,
            "target_action_hash": self.target_action_hash,
            "case_consistent": True,
            "physical_components_only": True,
            "soft_topk_probability_blends_allowed": True,
            "all_k_lambda_probability_matrices_persisted": False,
            "exact_b_byte_identical_fallback": True,
            "selection_status": "FROZEN_SOURCE_TRAIN_POLICY",
            "probability_status": "RECONSTRUCTED_FROM_SEALED_COMPONENTS",
            "prediction_status": "SEALED_BEFORE_EVALUATION_LABELS",
            "utility_status": "NOT_OPENED",
            "evaluation_labels_opened": False,
        }
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "ordered_case_identity_hash", case_identity_hash)
        object.__setattr__(self, "ordered_sample_identity_hash", sample_identity_hash)
        object.__setattr__(self, "route_hash", canonical_hash(base))


@dataclass(frozen=True, slots=True)
class FrozenRouteReceipt:
    """Typed capability binding terminal evaluation to the durable route seal.

    The receipt is created only after the route store has been reconstructed
    from disk and the two fresh validation identities have been authenticated.
    It carries no labels and therefore may cross the evaluation-label barrier.
    """

    seal_hash: str
    config_hash: str
    route_hash: str
    policy_hash: str
    model_hash: str
    target_action_hash: str
    validation_bundle_hash: str
    independent_validation_hashes: tuple[str, str]
    expected_center_ids: tuple[str, ...]
    case_count: int
    ordered_case_identity_hash: str
    ordered_sample_identity_hash: str

    def __post_init__(self) -> None:
        for name in (
            "seal_hash",
            "config_hash",
            "route_hash",
            "policy_hash",
            "model_hash",
            "target_action_hash",
            "validation_bundle_hash",
            "ordered_case_identity_hash",
            "ordered_sample_identity_hash",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ProtocolError(f"HARP v19 frozen-route {name} must be SHA-256.")
        validations = tuple(self.independent_validation_hashes)
        if (
            len(validations) != 2
            or len(set(validations)) != 2
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in validations
            )
        ):
            raise ProtocolError(
                "HARP v19 frozen-route receipt requires two distinct validations."
            )
        centers = tuple(
            _identifier(value, name="frozen-route center")
            for value in self.expected_center_ids
        )
        if (
            not centers
            or centers != tuple(sorted(centers))
            or len(set(centers)) != len(centers)
            or type(self.case_count) is not int
            or self.case_count < 1
        ):
            raise ProtocolError("HARP v19 frozen-route receipt inventory is malformed.")
        object.__setattr__(self, "independent_validation_hashes", validations)
        object.__setattr__(self, "expected_center_ids", centers)


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


class HarpV19Pipeline(Protocol):
    """Production/synthetic numerical service boundary used by the runner."""

    @property
    def physical_input_receipt(self) -> "PhysicalInputReceipt": ...

    def preflight(self, config: Any, cache: Any) -> Mapping[str, object]: ...

    def materialize_label_free_outer_menus(
        self,
        config: Any,
        cache: Any,
        *,
        outer_targets: Sequence[str],
        scratch_root: Any,
    ) -> Sequence[LabelFreeOuterMenu]: ...

    def compile_label_free_support_target_menus(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        *,
        scratch_root: Any,
    ) -> tuple[Sequence[object], ArtifactValue]: ...

    def compile_label_free_source_target_menus(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        *,
        scratch_root: Any,
    ) -> tuple[Sequence[object], ArtifactValue]: ...

    def build_support_case_surface(
        self,
        bundles: Sequence[object],
        support_labels_by_outer: Mapping[str, Sequence[object]],
    ) -> ArtifactValue: ...

    def build_source_train_case_surface(
        self,
        bundles: Sequence[object],
        source_labels_by_center: Mapping[str, Sequence[object]],
    ) -> ArtifactValue: ...

    def fit_pooled_source_router(
        self, support: ArtifactValue, *, config: Any
    ) -> ArtifactValue: ...

    def build_complete_target_case_actions(
        self,
        bundles: Sequence[object],
        fitted: ArtifactValue,
        *,
        config: Any,
    ) -> ArtifactValue: ...

    def route_case_actions(
        self,
        bundles: Sequence[object],
        fitted: ArtifactValue,
        target_actions: ArtifactValue,
    ) -> PrelabelRouteSet: ...

    def evaluate_terminal(
        self,
        routes: PrelabelRouteSet,
        evaluation_truth: object,
        *,
        frozen_receipt: FrozenRouteReceipt,
        artifact_root: Any,
        config: Any,
        menus: Sequence[LabelFreeOuterMenu],
    ) -> TerminalEvaluation: ...


__all__ = (
    "ActionKind",
    "ArtifactValue",
    "FrozenRouteReceipt",
    "HarpV19Pipeline",
    "LabelFreeActionBlock",
    "LabelFreeOuterMenu",
    "PrelabelRouteSet",
    "RoutedCase",
    "TerminalEvaluation",
    "array_bytes_sha256",
    "reconstruct_selected_probability_blend",
    "reconstruct_shrunk_probability_blend",
)
