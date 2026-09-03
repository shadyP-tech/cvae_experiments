"""Fold-conditioned physical action algebra for HARP v12 source cross-fitting.

The terminal menu has only one exclusion identity (the target ``H``), while a
source cross-fit prediction has two: the outer target ``H`` and the held-out
pseudo-target ``q``.  For a current source query ``r`` this module constructs
the physical classifier from ``C - {H, q, r}``; when ``r == q`` the set reduces
to ``C - {H, q}``.  Keeping all three identities in the action hash prevents a
probability vector composed with the wrong expert bank from being reused by a
nested fold.

The six-source calibration geometry deliberately keeps the same 1008-row base
and 126-row top-up budget as the seven-source prediction geometry.  A pure
single-source top-up therefore has maximum weight ``7/27`` and effective
source count ``243/43``.  Those diagnostic-only bounds are explicit; this
module does not claim the generic residual-top-up ``<= .25``/``>= 6`` bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
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
    SourceClassWindows,
    TopupGeometry,
    build_topup_geometry,
    inner_topup_geometry,
)
from ...routing.residual_topup.hashing import canonical_sha256
from .physical_actions import (
    BASE_ACTION_ID,
    EXACT_NINE_SEED_PAIRS,
    H_X_E_ACTION_PREFIX,
    UNIFORM_ACTION_ID,
)


CROSSFIT_SURFACE = "source_crossfit"
CROSSFIT_ACTION_SCHEMA = "midogpp_harp_v12_fold_conditioned_action_spec_v1"
CROSSFIT_COMPOSITION_SHUFFLE_NAMESPACE = (
    "midogpp_harp_v12_hqr_crossfit_composition_shuffle_v1"
)
SIX_SOURCE_BASE_PER_SOURCE = 168
SIX_SOURCE_BASE_TOTAL_PER_CLASS = 1008
SIX_SOURCE_TOPUP_TOTAL_PER_CLASS = 126
SIX_SOURCE_FINAL_TOTAL_PER_CLASS = 1134
SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT = 7.0 / 27.0
SIX_SOURCE_PURE_TOPUP_EFFECTIVE_SOURCES = 243.0 / 43.0
SIX_SOURCE_DENSITY_SEMANTICS = (
    "diagnostic_only_six_source_1008_plus_126_no_projection_"
    "pure_topup_max_weight_7_over_27_effective_sources_243_over_43"
)


def _center(value: object, *, name: str) -> str:
    if type(value) is not str or value not in CENTERS:
        raise ProtocolError(f"HARP v12 {name} is outside the frozen centers.")
    return value


def six_source_crossfit_geometry(
    candidate_sources: Sequence[object],
) -> TopupGeometry:
    """Return the audited six-source geometry with the unchanged total budget."""

    sources = tuple(sorted(str(value) for value in candidate_sources))
    if len(sources) != 6 or len(set(sources)) != 6:
        raise ProtocolError("HARP v12 crossfit geometry requires six sources.")
    geometry = build_topup_geometry(
        sources,
        base_per_source=SIX_SOURCE_BASE_PER_SOURCE,
        topup_total_per_class=SIX_SOURCE_TOPUP_TOTAL_PER_CLASS,
    )
    if (
        geometry.base_total_per_class != SIX_SOURCE_BASE_TOTAL_PER_CLASS
        or geometry.final_total_per_class != SIX_SOURCE_FINAL_TOTAL_PER_CLASS
    ):
        raise ProtocolError("HARP v12 six-source budget geometry drifted.")
    return geometry


def six_source_geometry_audit() -> Mapping[str, object]:
    body = {
        "schema_version": "midogpp_harp_v12_six_source_geometry_audit_v1",
        "source_count": 6,
        "base_per_source": SIX_SOURCE_BASE_PER_SOURCE,
        "base_total_per_class": SIX_SOURCE_BASE_TOTAL_PER_CLASS,
        "topup_total_per_class": SIX_SOURCE_TOPUP_TOTAL_PER_CLASS,
        "final_total_per_class": SIX_SOURCE_FINAL_TOTAL_PER_CLASS,
        "pure_selected_topup_maximum_source_weight": (
            SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT
        ),
        "pure_selected_topup_effective_source_count": (
            SIX_SOURCE_PURE_TOPUP_EFFECTIVE_SOURCES
        ),
        "generic_max_weight_quarter_claimed": False,
        "generic_min_effective_sources_six_claimed": False,
        "density_semantics": SIX_SOURCE_DENSITY_SEMANTICS,
        "publication_scope": "diagnostic_only",
    }
    return MappingProxyType({**body, "geometry_audit_hash": canonical_hash(body)})


def _six_source_single_tail_action(
    selected_source: str, *, geometry: TopupGeometry
) -> ResidualTopupAction:
    """Construct the explicit diagnostic tail action without generic bounds."""

    if geometry.source_count != 6 or selected_source not in geometry.source_order:
        raise ProtocolError("HARP v12 six-source selected tail is outside its pool.")
    topup_counts = {
        source: SIX_SOURCE_TOPUP_TOTAL_PER_CLASS
        if source == selected_source
        else 0
        for source in geometry.source_order
    }
    final_counts: dict[int, Mapping[str, int]] = {}
    final_weights: dict[int, Mapping[str, float]] = {}
    windows: dict[int, Mapping[str, SourceClassWindows]] = {}
    effective: dict[int, float] = {}
    for label in geometry.class_labels:
        counts = {
            source: geometry.base_per_source + topup_counts[source]
            for source in geometry.source_order
        }
        weights = {
            source: counts[source] / float(geometry.final_total_per_class)
            for source in geometry.source_order
        }
        eff = 1.0 / sum(value * value for value in weights.values())
        if (
            abs(max(weights.values()) - SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT) > 1e-15
            or abs(eff - SIX_SOURCE_PURE_TOPUP_EFFECTIVE_SOURCES) > 1e-12
        ):
            raise ProtocolError("HARP v12 six-source density audit drifted.")
        final_counts[label] = MappingProxyType(counts)
        final_weights[label] = MappingProxyType(weights)
        effective[label] = eff
        windows[label] = MappingProxyType(
            {
                source: SourceClassWindows(
                    base_start=0,
                    base_stop=geometry.base_per_source,
                    topup_start=geometry.base_per_source,
                    topup_stop=geometry.base_per_source + topup_counts[source],
                )
                for source in geometry.source_order
            }
        )
    allocation_payload = {
        "schema_version": "midogpp_harp_v12_six_source_tail_allocation_v1",
        "source_order": list(geometry.source_order),
        "selected_source_id": selected_source,
        "topup_counts": topup_counts,
        "topup_total_per_class": geometry.topup_total_per_class,
        "density_semantics": SIX_SOURCE_DENSITY_SEMANTICS,
    }
    window_payload = {
        "schema_version": "midogpp_harp_v12_six_source_tail_windows_v1",
        "source_order": list(geometry.source_order),
        "windows_by_class": {
            str(label): {
                source: windows[label][source].to_payload()
                for source in geometry.source_order
            }
            for label in geometry.class_labels
        },
    }
    allocation_hash = canonical_sha256(allocation_payload)
    window_hash = canonical_sha256(window_payload)
    action_payload = {
        "schema_version": "midogpp_harp_v12_six_source_tail_action_v1",
        "geometry": geometry.to_payload(),
        "selected_source_id": selected_source,
        "topup_counts": topup_counts,
        "allocation_hash": allocation_hash,
        "window_hash": window_hash,
        "density_semantics": SIX_SOURCE_DENSITY_SEMANTICS,
    }
    return ResidualTopupAction(
        geometry=geometry,
        action_kind="harp_v12_six_source_single_source_tail_diagnostic",
        direction_semantics="all_topup_weight_to_selected_source",
        temperature=None,
        calibrated_energy_by_source=MappingProxyType({}),
        direction_weights=MappingProxyType(
            {
                source: float(source == selected_source)
                for source in geometry.source_order
            }
        ),
        topup_counts=MappingProxyType(topup_counts),
        final_counts_by_class=MappingProxyType(final_counts),
        final_weights_by_class=MappingProxyType(final_weights),
        windows_by_class=MappingProxyType(windows),
        effective_source_count_by_class=MappingProxyType(effective),
        maximum_source_weight=SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT,
        allocation_hash=allocation_hash,
        window_hash=window_hash,
        action_hash=canonical_sha256(action_payload),
        density_constraint_semantics=SIX_SOURCE_DENSITY_SEMANTICS,
    )


@dataclass(frozen=True, kw_only=True)
class FoldConditionedActionSpec:
    """One action keyed by ``(outer H, heldout q, current query r)``."""

    outer_target_id: str
    heldout_center_id: str
    current_query_center_id: str
    selected_source_id: str | None = None
    action_id: str | None = None
    surface_kind: str = field(init=False, default=CROSSFIT_SURFACE)
    source_order: tuple[str, ...] = field(init=False)
    geometry: TopupGeometry = field(init=False, repr=False)
    residual_action: ResidualTopupAction | None = field(init=False, repr=False)
    action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = _center(self.outer_target_id, name="outer target H")
        heldout = _center(self.heldout_center_id, name="heldout pseudo-target q")
        query = _center(self.current_query_center_id, name="current query r")
        if outer == heldout or outer == query:
            raise ProtocolError("HARP v12 crossfit requires H distinct from q and r.")
        sources = tuple(
            center for center in CENTERS if center not in {outer, heldout, query}
        )
        if query == heldout:
            geometry = inner_topup_geometry(sources)
        else:
            geometry = six_source_crossfit_geometry(sources)

        selected = self.selected_source_id
        requested = self.action_id
        if selected is not None:
            selected = _center(selected, name="crossfit candidate source e")
            if selected not in sources:
                raise ProtocolError("HARP v12 candidate escaped C-{H,q,r}.")
            residual = (
                build_single_source_tail_action(selected, geometry=geometry)
                if len(sources) == 7
                else _six_source_single_tail_action(selected, geometry=geometry)
            )
            action_id = f"{H_X_E_ACTION_PREFIX}{selected}"
            if requested not in (None, action_id):
                raise ProtocolError("HARP v12 crossfit candidate identity drifted.")
        elif requested in (None, BASE_ACTION_ID):
            residual = None
            action_id = BASE_ACTION_ID
        elif requested == UNIFORM_ACTION_ID:
            residual = build_uniform_topup_action(geometry)
            action_id = UNIFORM_ACTION_ID
        else:
            raise ProtocolError("HARP v12 crossfit control identity is unknown.")

        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "heldout_center_id", heldout)
        object.__setattr__(self, "current_query_center_id", query)
        object.__setattr__(self, "selected_source_id", selected)
        object.__setattr__(self, "source_order", sources)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "residual_action", residual)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_hash", canonical_hash(self._body()))

    @property
    def query_center_id(self) -> str:
        """Compatibility alias for the evaluated current query ``r``."""

        return self.current_query_center_id

    @property
    def is_exact_b(self) -> bool:
        return self.action_id == BASE_ACTION_ID

    @property
    def is_uniform_topup(self) -> bool:
        return self.action_id == UNIFORM_ACTION_ID

    @property
    def prediction_fold(self) -> bool:
        return self.current_query_center_id == self.heldout_center_id

    @property
    def key(self) -> tuple[str, str, str, int, str]:
        return (
            self.outer_target_id,
            self.heldout_center_id,
            self.current_query_center_id,
            0 if self.is_exact_b else 1 if self.is_uniform_topup else 2,
            self.selected_source_id or "",
        )

    def _body(self) -> dict[str, object]:
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
            "schema_version": CROSSFIT_ACTION_SCHEMA,
            "surface_kind": CROSSFIT_SURFACE,
            "outer_target_id": self.outer_target_id,
            "heldout_center_id": self.heldout_center_id,
            "current_query_center_id": self.current_query_center_id,
            "action_id": self.action_id,
            "selected_source_id": self.selected_source_id,
            "source_order": list(self.source_order),
            "geometry": self.geometry.to_payload(),
            "geometry_audit": (
                dict(six_source_geometry_audit())
                if len(self.source_order) == 6
                else {
                    "source_count": 7,
                    "generic_density_bounds_apply": True,
                }
            ),
            "counts_by_class": counts,
            "residual_action_hash": (
                None
                if self.residual_action is None
                else self.residual_action.action_hash
            ),
            "candidate_pool_semantics": "C_MINUS_H_MINUS_Q_MINUS_R",
            "heldout_q_physically_excluded": True,
            "current_query_r_physically_excluded": True,
            "all_nine_seed_cells_retained": True,
            "labels_consumed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._body(), "action_hash": self.action_hash}


def build_fold_conditioned_action_menu(
    outer_target_id: object,
    heldout_center_id: object,
    current_query_center_id: object,
) -> tuple[FoldConditionedActionSpec, ...]:
    probe = FoldConditionedActionSpec(
        outer_target_id=str(outer_target_id),
        heldout_center_id=str(heldout_center_id),
        current_query_center_id=str(current_query_center_id),
        action_id=BASE_ACTION_ID,
    )
    controls = (
        probe,
        FoldConditionedActionSpec(
            outer_target_id=probe.outer_target_id,
            heldout_center_id=probe.heldout_center_id,
            current_query_center_id=probe.current_query_center_id,
            action_id=UNIFORM_ACTION_ID,
        ),
    )
    candidates = tuple(
        FoldConditionedActionSpec(
            outer_target_id=probe.outer_target_id,
            heldout_center_id=probe.heldout_center_id,
            current_query_center_id=probe.current_query_center_id,
            selected_source_id=source,
        )
        for source in probe.source_order
    )
    return (*controls, *candidates)


def build_all_fold_conditioned_actions(
    outer_targets: Sequence[str] = CENTERS,
) -> tuple[FoldConditionedActionSpec, ...]:
    requested = tuple(str(value) for value in outer_targets)
    if tuple(center for center in CENTERS if center in set(requested)) != requested:
        raise ProtocolError("HARP v12 crossfit outer-target order is noncanonical.")
    return tuple(
        action
        for outer in requested
        for heldout in CENTERS
        if heldout != outer
        for query in CENTERS
        if query != outer
        for action in build_fold_conditioned_action_menu(outer, heldout, query)
    )


def fold_conditioned_action_from_payload(raw: object) -> FoldConditionedActionSpec:
    if not isinstance(raw, Mapping) or raw.get("schema_version") != CROSSFIT_ACTION_SCHEMA:
        raise ProtocolError("HARP v12 fold-conditioned action payload is malformed.")
    action = FoldConditionedActionSpec(
        outer_target_id=str(raw.get("outer_target_id")),
        heldout_center_id=str(raw.get("heldout_center_id")),
        current_query_center_id=str(raw.get("current_query_center_id")),
        selected_source_id=(
            None
            if raw.get("selected_source_id") is None
            else str(raw.get("selected_source_id"))
        ),
        action_id=str(raw.get("action_id")),
    )
    if action.action_hash != raw.get("action_hash"):
        raise ProtocolError("HARP v12 fold-conditioned action hash drifted.")
    return action


def compose_fold_conditioned_action(
    source_blocks: Mapping[object, object],
    action: FoldConditionedActionSpec,
    *,
    shuffle_seed_by_class: Mapping[object, int],
) -> EqualUnionBaseComposition | ResidualTopupComposition:
    if not isinstance(action, FoldConditionedActionSpec):
        raise ProtocolError("HARP v12 crossfit composition requires a typed action.")
    if tuple(sorted(str(value) for value in source_blocks)) != action.source_order:
        raise ProtocolError("HARP v12 crossfit source blocks escaped C-{H,q,r}.")
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


def fold_conditioned_composition_seed(
    *,
    generation_lock_hash: str,
    outer_target_id: str,
    heldout_center_id: str,
    current_query_center_id: str,
    training_seed: int,
    generation_seed: int,
    class_label: int,
) -> int:
    outer = _center(outer_target_id, name="composition outer H")
    heldout = _center(heldout_center_id, name="composition heldout q")
    query = _center(current_query_center_id, name="composition query r")
    if outer == heldout or outer == query:
        raise ProtocolError("HARP v12 crossfit composition exclusions drifted.")
    if (training_seed, generation_seed) not in EXACT_NINE_SEED_PAIRS:
        raise ProtocolError("HARP v12 crossfit composition seed cell drifted.")
    if class_label not in (0, 1):
        raise ProtocolError("HARP v12 crossfit composition class is not binary.")
    body = {
        "namespace": CROSSFIT_COMPOSITION_SHUFFLE_NAMESPACE,
        "generation_lock_hash": str(generation_lock_hash),
        "outer_target_id": outer,
        "heldout_center_id": heldout,
        "current_query_center_id": query,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "class_label": class_label,
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:8],
        "big",
        signed=False,
    )


__all__ = (
    "CROSSFIT_ACTION_SCHEMA",
    "CROSSFIT_SURFACE",
    "FoldConditionedActionSpec",
    "SIX_SOURCE_DENSITY_SEMANTICS",
    "SIX_SOURCE_PURE_TOPUP_EFFECTIVE_SOURCES",
    "SIX_SOURCE_PURE_TOPUP_MAX_WEIGHT",
    "build_all_fold_conditioned_actions",
    "build_fold_conditioned_action_menu",
    "compose_fold_conditioned_action",
    "fold_conditioned_action_from_payload",
    "fold_conditioned_composition_seed",
    "six_source_crossfit_geometry",
    "six_source_geometry_audit",
)
