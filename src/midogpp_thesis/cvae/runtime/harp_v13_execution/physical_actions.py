"""V13-owned leakage-safe exact-B, matched-budget U, and Hxe action algebra.

This is an identity-neutral physical composition primitive.  It intentionally
does not import the legacy probability-menu package root, whose public surface
also exposes predecessor routing-policy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ...routing.residual_topup.actions import (
    build_single_source_tail_action,
    build_uniform_topup_action,
)
from ...routing.residual_topup.composition import (
    EqualUnionBaseComposition,
    ResidualTopupComposition,
    compose_equal_union_base_blocks,
    compose_residual_topup_blocks,
)
from ...routing.residual_topup.contracts import (
    ResidualTopupAction,
    TopupGeometry,
    inner_topup_geometry,
    target_topup_geometry,
)
from ...routing.harp_protocol import canonical_hash


BASE_ACTION_ID = "B"
UNIFORM_ACTION_ID = "U"
H_X_E_ACTION_PREFIX = "Hxe::"
DEVELOPMENT_SURFACE = "development"
TARGET_SURFACE = "target"
SURFACE_KINDS = (DEVELOPMENT_SURFACE, TARGET_SURFACE)
HARP_COMPOSITION_SHUFFLE_NAMESPACE = "midogpp_harp_hq_composition_shuffle_v1"
EXACT_NINE_SEED_PAIRS = tuple(
    (training_seed, generation_seed)
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
)


def _center(value: object, *, name: str) -> str:
    if type(value) is not str or value not in CENTERS:
        raise ProtocolError(f"HARP {name} is outside the frozen MIDOG++ centers.")
    return value


@dataclass(frozen=True, kw_only=True)
class HarpActionSpec:
    """One immutable, label-free probability action.

    Development actions have four roles: held-out outer target ``H``,
    pseudo-query ``q``, candidate source ``e`` (for Hxe), and the
    remaining source bank.  Target actions set ``q == H`` and exclude that
    center from the source bank.  The source expert for ``H`` is therefore not
    representable in either menu.
    """

    surface_kind: str
    outer_target_id: str
    query_center_id: str
    selected_source_id: str | None = None
    # ``action_id`` is accepted at construction solely so a serialized control
    # arm can be reconstructed without conflating B and U.  Candidate actions
    # are still derived canonically from ``selected_source_id``.
    action_id: str | None = None
    source_order: tuple[str, ...] = field(init=False)
    geometry: TopupGeometry = field(init=False, repr=False)
    residual_action: ResidualTopupAction | None = field(init=False, repr=False)
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _center(self.outer_target_id, name="outer target H")
        query = _center(self.query_center_id, name="query q")
        if self.surface_kind not in SURFACE_KINDS:
            raise ProtocolError("HARP action surface kind is unknown.")
        if self.surface_kind == DEVELOPMENT_SURFACE:
            if query == outer:
                raise ProtocolError("HARP development query q must exclude outer H.")
            sources = tuple(center for center in CENTERS if center not in {outer, query})
            geometry = inner_topup_geometry(sources)
        else:
            if query != outer:
                raise ProtocolError("HARP target actions require q == H.")
            sources = tuple(center for center in CENTERS if center != outer)
            geometry = target_topup_geometry(sources)

        selected = self.selected_source_id
        requested_action_id = self.action_id
        if selected is not None:
            selected = _center(selected, name="candidate source e")
            if selected not in sources:
                raise ProtocolError("HARP candidate source e must exclude H and q.")
            residual = build_single_source_tail_action(selected, geometry=geometry)
            action_id = f"{H_X_E_ACTION_PREFIX}{selected}"
            if requested_action_id not in (None, action_id):
                raise ProtocolError("HARP candidate action identity drifted.")
        else:
            if requested_action_id in (None, BASE_ACTION_ID):
                residual = None
                action_id = BASE_ACTION_ID
            elif requested_action_id == UNIFORM_ACTION_ID:
                residual = build_uniform_topup_action(geometry)
                action_id = UNIFORM_ACTION_ID
            else:
                raise ProtocolError("HARP control action identity is unknown.")

        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_center_id", query)
        object.__setattr__(self, "selected_source_id", selected)
        object.__setattr__(self, "source_order", sources)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "residual_action", residual)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(
            self,
            "action_hash",
            canonical_hash(self._payload_without_hash()),
        )

    @property
    def is_exact_b(self) -> bool:
        return self.action_id == BASE_ACTION_ID

    @property
    def is_uniform_topup(self) -> bool:
        return self.action_id == UNIFORM_ACTION_ID

    @property
    def labels_consumed(self) -> bool:
        return False

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        return (
            self.surface_kind,
            self.outer_target_id,
            self.query_center_id,
            0 if self.is_exact_b else 1 if self.is_uniform_topup else 2,
            "" if self.selected_source_id is None else self.selected_source_id,
        )

    def _payload_without_hash(self) -> dict[str, object]:
        counts = {
            str(label): {
                source: (
                    self.geometry.base_per_source
                    if self.residual_action is None
                    else self.residual_action.final_counts_by_class[label][source]
                )
                for source in self.source_order
            }
            for label in self.geometry.class_labels
        }
        return {
            "schema_version": "midogpp_harp_action_spec_v2",
            "surface_kind": self.surface_kind,
            "outer_target_id": self.outer_target_id,
            "query_center_id": self.query_center_id,
            "action_id": self.action_id,
            "selected_source_id": self.selected_source_id,
            "source_order": list(self.source_order),
            "geometry": self.geometry.to_payload(),
            "counts_by_class": counts,
            "residual_action_hash": (
                None if self.residual_action is None else self.residual_action.action_hash
            ),
            "composition_kind": (
                "exact_equal_union_base"
                if self.is_exact_b
                else (
                    "matched_budget_uniform_residual_topup"
                    if self.is_uniform_topup
                    else "matched_budget_single_source_residual_topup"
                )
            ),
            "predictive_reference_action_id": UNIFORM_ACTION_ID,
            "operational_fallback_action_id": BASE_ACTION_ID,
            "matched_budget_with_candidate_actions": not self.is_exact_b,
            "outer_target_excluded": True,
            "pseudo_query_excluded_from_sources": (
                self.surface_kind == DEVELOPMENT_SURFACE
            ),
            "candidate_source_excluded_from_query": (
                self.selected_source_id is None
                or self.selected_source_id != self.query_center_id
            ),
            "labels_consumed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload_without_hash(), "action_hash": self.action_hash}


def build_development_action_menu(
    outer_target_id: object, query_center_id: object
) -> tuple[HarpActionSpec, ...]:
    outer = _center(outer_target_id, name="outer target H")
    query = _center(query_center_id, name="pseudo-query q")
    if outer == query:
        raise ProtocolError("HARP development menu requires distinct H and q.")
    sources = tuple(center for center in CENTERS if center not in {outer, query})
    controls = (
        HarpActionSpec(
            surface_kind=DEVELOPMENT_SURFACE,
            outer_target_id=outer,
            query_center_id=query,
            action_id=BASE_ACTION_ID,
        ),
        HarpActionSpec(
            surface_kind=DEVELOPMENT_SURFACE,
            outer_target_id=outer,
            query_center_id=query,
            action_id=UNIFORM_ACTION_ID,
        ),
    )
    candidates = tuple(
        HarpActionSpec(
            surface_kind=DEVELOPMENT_SURFACE,
            outer_target_id=outer,
            query_center_id=query,
            selected_source_id=selected,
        )
        for selected in sources
    )
    return (*controls, *candidates)


def build_target_action_menu(outer_target_id: object) -> tuple[HarpActionSpec, ...]:
    outer = _center(outer_target_id, name="outer target H")
    sources = tuple(center for center in CENTERS if center != outer)
    controls = (
        HarpActionSpec(
            surface_kind=TARGET_SURFACE,
            outer_target_id=outer,
            query_center_id=outer,
            action_id=BASE_ACTION_ID,
        ),
        HarpActionSpec(
            surface_kind=TARGET_SURFACE,
            outer_target_id=outer,
            query_center_id=outer,
            action_id=UNIFORM_ACTION_ID,
        ),
    )
    candidates = tuple(
        HarpActionSpec(
            surface_kind=TARGET_SURFACE,
            outer_target_id=outer,
            query_center_id=outer,
            selected_source_id=selected,
        )
        for selected in sources
    )
    return (*controls, *candidates)


def build_all_development_actions() -> tuple[HarpActionSpec, ...]:
    return tuple(
        action
        for outer in CENTERS
        for query in CENTERS
        if query != outer
        for action in build_development_action_menu(outer, query)
    )


def build_all_target_actions() -> tuple[HarpActionSpec, ...]:
    return tuple(
        action for outer in CENTERS for action in build_target_action_menu(outer)
    )


def compose_harp_action(
    source_blocks: Mapping[object, object],
    action: HarpActionSpec,
    *,
    shuffle_seed_by_class: Mapping[object, int],
) -> EqualUnionBaseComposition | ResidualTopupComposition:
    """Realize B, U, or Hxe using shared immutable composition primitives."""

    if not isinstance(action, HarpActionSpec):
        raise ProtocolError("HARP composition requires a typed action spec.")
    if tuple(sorted(str(value) for value in source_blocks)) != action.source_order:
        raise ProtocolError("HARP source blocks do not match the action source fence.")
    if action.residual_action is None:
        return compose_equal_union_base_blocks(
            source_blocks,
            action.geometry,
            shuffle_seed_by_class=shuffle_seed_by_class,
        )
    return compose_residual_topup_blocks(
        source_blocks,
        action.residual_action,
        shuffle_seed_by_class=shuffle_seed_by_class,
    )


def harp_composition_seed(
    *,
    generation_lock_hash: str,
    outer_target_id: str,
    query_center_id: str,
    training_seed: int,
    generation_seed: int,
    class_label: int,
) -> int:
    """Derive the one frozen HARP classwise composition seed.

    The target replay uses ``query_center_id == outer_target_id`` while the
    development surface uses a distinct pseudo-query.  Keeping both identities
    in this neutral primitive prevents the two stages from silently using
    different shuffles for otherwise identical actions.
    """

    outer = _center(outer_target_id, name="outer target H")
    query = _center(query_center_id, name="query q")
    if isinstance(training_seed, bool) or int(training_seed) not in (17, 42, 101):
        raise ProtocolError("HARP composition training seed is outside the frozen grid.")
    if isinstance(generation_seed, bool) or int(generation_seed) not in (17, 42, 101):
        raise ProtocolError("HARP composition generation seed is outside the frozen grid.")
    if isinstance(class_label, bool) or int(class_label) not in (0, 1):
        raise ProtocolError("HARP composition class label must be binary.")
    digest = str(generation_lock_hash)
    if not digest:
        raise ProtocolError("HARP composition generation lock is absent.")
    payload = {
        "namespace": HARP_COMPOSITION_SHUFFLE_NAMESPACE,
        "generation_lock_hash": digest,
        "outer_target": outer,
        "query_center": query,
        "training_seed": int(training_seed),
        "generation_seed": int(generation_seed),
        "class_label": int(class_label),
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:8],
        "big",
        signed=False,
    )


def validate_action_menu(actions: Sequence[HarpActionSpec]) -> tuple[HarpActionSpec, ...]:
    menu = tuple(actions)
    if not menu or any(not isinstance(action, HarpActionSpec) for action in menu):
        raise ProtocolError("HARP action menu must be nonempty and typed.")
    if tuple(action.key for action in menu) != tuple(sorted(action.key for action in menu)):
        raise ProtocolError("HARP action menu order is noncanonical.")
    if len({action.key for action in menu}) != len(menu):
        raise ProtocolError("HARP action menu contains duplicate actions.")

    by_query: dict[tuple[str, str, str], list[HarpActionSpec]] = {}
    for action in menu:
        by_query.setdefault(
            (action.surface_kind, action.outer_target_id, action.query_center_id), []
        ).append(action)
    for (surface, outer, query), rows in by_query.items():
        expected = (
            build_development_action_menu(outer, query)
            if surface == DEVELOPMENT_SURFACE
            else build_target_action_menu(outer)
        )
        if tuple(action.action_hash for action in rows) != tuple(
            action.action_hash for action in expected
        ):
            raise ProtocolError("HARP action menu is incomplete for a query surface.")
    return menu


__all__ = (
    "BASE_ACTION_ID",
    "DEVELOPMENT_SURFACE",
    "EXACT_NINE_SEED_PAIRS",
    "H_X_E_ACTION_PREFIX",
    "HARP_COMPOSITION_SHUFFLE_NAMESPACE",
    "HarpActionSpec",
    "SURFACE_KINDS",
    "TARGET_SURFACE",
    "UNIFORM_ACTION_ID",
    "build_all_development_actions",
    "build_all_target_actions",
    "build_development_action_menu",
    "build_target_action_menu",
    "compose_harp_action",
    "harp_composition_seed",
    "validate_action_menu",
)
