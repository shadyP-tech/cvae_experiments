"""Hash-locked exact additive-tail actions for development and target scoring."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup import (
    ResidualTopupAction,
    TopupGeometry,
    build_single_source_tail_action,
    build_uniform_topup_action,
    inner_topup_geometry,
    target_topup_geometry,
)
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_FROZEN_TARGET_ACTION_COUNT,
    EXPECTED_TARGET_ACTION_COUNT,
    GLOBAL_DELTA_ACTION_ID,
    PERMUTATION_ACTION_ID,
    PRIMARY_DIAGNOSTIC_ACTION_IDS,
    R2_ACTION_ID,
    ROUTING_STATUS,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
    h_x_e_source,
    inner_candidate_sources,
)
from .r2_policy import FrozenR2DiagnosticPlan, Stage90R2PlanSet


INNER_ACTION_ROLE = "source_inner_exact_tail_development"
TARGET_ACTION_ROLE = "consumed_target_exact_tail_diagnostic"


@dataclass(frozen=True)
class FrozenExactTailAction:
    """One reconstructively validated B, U, or exact single-source tail."""

    outer_target_id: str
    query_id: str
    action_id: str
    action_role: str
    geometry: TopupGeometry
    selected_source: str | None
    core_action: ResidualTopupAction | None
    router_plan_hash: str | None
    diagnostic_control: bool
    action_hash: str
    diagnostic_only: bool = True
    evaluation_embeddings_used_to_build: bool = False
    seed_selection_performed: bool = False
    policy_authorized: bool = False
    fallback_authorized: bool = False
    promotion_authorized: bool = False
    deployment_authorized: bool = False

    def __post_init__(self) -> None:
        outer = str(self.outer_target_id)
        query = str(self.query_id)
        action_id = str(self.action_id)
        selected = None if self.selected_source is None else str(self.selected_source)
        if self.action_role == INNER_ACTION_ROLE:
            sources = inner_candidate_sources(outer, query)
            expected_geometry = inner_topup_geometry(sources)
            allowed = (BASE_ACTION_ID, *(h_x_e_action_id(source) for source in sources))
            if self.router_plan_hash is not None:
                raise ProtocolError("Source-inner actions cannot carry a target router plan.")
        elif self.action_role == TARGET_ACTION_ROLE:
            if outer != query:
                raise ProtocolError("Target action requires query == outer target H.")
            sources = candidate_sources(outer)
            expected_geometry = target_topup_geometry(sources)
            allowed = expected_target_action_ids(outer)
            if not _is_hash(self.router_plan_hash):
                raise ProtocolError("Target actions require a frozen router-plan hash.")
        else:
            raise ProtocolError("Exact-tail action role is invalid.")
        if (
            self.geometry != expected_geometry
            or action_id not in allowed
            or self.diagnostic_only is not True
            or self.evaluation_embeddings_used_to_build is not False
            or self.seed_selection_performed is not False
            or self.policy_authorized is not False
            or self.fallback_authorized is not False
            or self.promotion_authorized is not False
            or self.deployment_authorized is not False
            or type(self.diagnostic_control) is not bool
        ):
            raise ProtocolError("Exact-tail action boundary drifted.")
        expected_core = _expected_core(
            action_id=action_id,
            geometry=expected_geometry,
            selected_source=selected,
        )
        if expected_core is None:
            if self.core_action is not None or selected is not None:
                raise ProtocolError("Exact base B cannot carry a top-up route.")
        elif (
            not isinstance(self.core_action, ResidualTopupAction)
            or self.core_action.action_hash != expected_core.action_hash
            or self.core_action.geometry != expected_geometry
        ):
            raise ProtocolError("Exact-tail action does not reconstruct from its primitive.")
        if action_id == UNIFORM_ACTION_ID and selected is not None:
            raise ProtocolError("Uniform U cannot select one source.")
        if action_id not in {BASE_ACTION_ID, UNIFORM_ACTION_ID} and selected not in sources:
            raise ProtocolError("Exact-tail action has no legal selected source.")
        embedded_source = h_x_e_source(action_id)
        if embedded_source is not None and embedded_source != selected:
            raise ProtocolError("Hxe action identity and selected source drifted.")
        expected_hash = canonical_sha256(
            self._unhashed_payload(
                outer=outer,
                query=query,
                action_id=action_id,
                selected=selected,
            )
        )
        if self.action_hash != expected_hash:
            raise ProtocolError("Frozen exact-tail action hash drifted.")
        object.__setattr__(self, "outer_target_id", outer)
        object.__setattr__(self, "query_id", query)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "selected_source", selected)

    @property
    def source_order(self) -> tuple[str, ...]:
        return self.geometry.source_order

    @property
    def base_per_source_per_class(self) -> int:
        return self.geometry.base_per_source

    @property
    def topup_total_per_class(self) -> int:
        return 0 if self.core_action is None else self.geometry.topup_total_per_class

    @property
    def final_total_per_class(self) -> int:
        return self.geometry.base_total_per_class + self.topup_total_per_class

    @property
    def topup_counts_by_source(self) -> Mapping[str, int]:
        if self.core_action is None:
            return MappingProxyType({source: 0 for source in self.source_order})
        return self.core_action.topup_counts

    @property
    def final_counts_by_class(self) -> Mapping[int, Mapping[str, int]]:
        if self.core_action is not None:
            return self.core_action.final_counts_by_class
        return MappingProxyType(
            {
                label: MappingProxyType(
                    {
                        source: self.geometry.base_per_source
                        for source in self.source_order
                    }
                )
                for label in self.geometry.class_labels
            }
        )

    @property
    def required_source_capacity_per_class(self) -> int:
        return max(self.final_counts_by_class[0].values())

    def _unhashed_payload(
        self,
        *,
        outer: str | None = None,
        query: str | None = None,
        action_id: str | None = None,
        selected: str | None = None,
    ) -> dict[str, object]:
        resolved_selected = self.selected_source if selected is None else selected
        # For B/U, ``None`` is meaningful and the branch above is equivalent.
        return {
            "schema_version": "midogpp_utility_aligned_stage90_exact_tail_action_v1",
            "outer_target_id": outer or self.outer_target_id,
            "query_id": query or self.query_id,
            "action_id": action_id or self.action_id,
            "action_role": self.action_role,
            "geometry": self.geometry.to_payload(),
            "selected_source": resolved_selected,
            "realized_topup_total_per_class": self.topup_total_per_class,
            "realized_final_total_per_class": self.final_total_per_class,
            "topup_counts_by_source": dict(self.topup_counts_by_source),
            "final_counts_by_class": {
                str(label): dict(self.final_counts_by_class[label])
                for label in self.geometry.class_labels
            },
            "core_action_hash": (
                None if self.core_action is None else self.core_action.action_hash
            ),
            "router_plan_hash": self.router_plan_hash,
            "diagnostic_control": self.diagnostic_control,
            "diagnostic_only": True,
            **_action_label_provenance(self.action_role, action_id or self.action_id),
            "evaluation_embeddings_used_to_build": False,
            "seed_selection_performed": False,
            "policy_authorized": False,
            "fallback_authorized": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "action_hash": self.action_hash}


@dataclass(frozen=True)
class FrozenInnerExactTailActionLibrary:
    actions_by_outer_and_query: Mapping[
        str, Mapping[str, tuple[FrozenExactTailAction, ...]]
    ]
    action_library_hash: str

    def __post_init__(self) -> None:
        outer_mapping = {
            str(outer): {str(query): tuple(actions) for query, actions in queries.items()}
            for outer, queries in self.actions_by_outer_and_query.items()
        }
        if tuple(outer_mapping) != CENTERS:
            raise ProtocolError("Inner exact-tail action library outer coverage drifted.")
        for outer in CENTERS:
            queries = candidate_sources(outer)
            if tuple(outer_mapping[outer]) != queries:
                raise ProtocolError("Inner exact-tail action library query coverage drifted.")
            for query in queries:
                sources = inner_candidate_sources(outer, query)
                actions = outer_mapping[outer][query]
                if (
                    tuple(action.action_id for action in actions)
                    != (BASE_ACTION_ID, *(h_x_e_action_id(source) for source in sources))
                    or any(
                        action.outer_target_id != outer
                        or action.query_id != query
                        or action.action_role != INNER_ACTION_ROLE
                        for action in actions
                    )
                ):
                    raise ProtocolError("Inner exact-tail action menu drifted.")
        expected = canonical_sha256(_inner_library_payload(outer_mapping))
        if self.action_library_hash != expected:
            raise ProtocolError("Inner exact-tail action-library hash drifted.")
        object.__setattr__(
            self,
            "actions_by_outer_and_query",
            MappingProxyType(
                {
                    outer: MappingProxyType(queries)
                    for outer, queries in outer_mapping.items()
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            **_inner_library_payload(self.actions_by_outer_and_query),
            "action_library_hash": self.action_library_hash,
        }


@dataclass(frozen=True)
class FrozenExactTailActionLibrary:
    actions_by_target: Mapping[str, tuple[FrozenExactTailAction, ...]]
    plan_hashes_by_target: Mapping[str, str]
    plan_set_hash: str
    action_library_hash: str

    def __post_init__(self) -> None:
        actions = {
            str(target): tuple(value) for target, value in self.actions_by_target.items()
        }
        plan_hashes = {
            str(target): str(value) for target, value in self.plan_hashes_by_target.items()
        }
        if tuple(actions) != CENTERS or tuple(plan_hashes) != CENTERS:
            raise ProtocolError("Target exact-tail action library coverage drifted.")
        observed: set[str] = set()
        for target in CENTERS:
            target_actions = actions[target]
            if (
                len(target_actions) != EXPECTED_TARGET_ACTION_COUNT
                or tuple(action.action_id for action in target_actions)
                != expected_target_action_ids(target)
                or any(
                    action.outer_target_id != target
                    or action.query_id != target
                    or action.action_role != TARGET_ACTION_ROLE
                    or action.router_plan_hash != plan_hashes[target]
                    for action in target_actions
                )
            ):
                raise ProtocolError("Target exact-tail action menu drifted.")
            for action in target_actions:
                if action.action_hash in observed:
                    raise ProtocolError("Frozen target action hashes must be unique.")
                observed.add(action.action_hash)
        if len(observed) != EXPECTED_FROZEN_TARGET_ACTION_COUNT:
            raise ProtocolError("Frozen target exact-tail action count drifted.")
        expected = canonical_sha256(
            _target_library_payload(
                actions,
                plan_hashes=plan_hashes,
                plan_set_hash=self.plan_set_hash,
            )
        )
        if self.action_library_hash != expected:
            raise ProtocolError("Target exact-tail action-library hash drifted.")
        object.__setattr__(self, "actions_by_target", MappingProxyType(actions))
        object.__setattr__(self, "plan_hashes_by_target", MappingProxyType(plan_hashes))

    @property
    def action_count(self) -> int:
        return sum(len(actions) for actions in self.actions_by_target.values())

    def action(self, target_center: object, action_id: object) -> FrozenExactTailAction:
        target = str(target_center)
        identifier = str(action_id)
        if target not in self.actions_by_target:
            raise ProtocolError("Target exact-tail action lookup target is unknown.")
        for action in self.actions_by_target[target]:
            if action.action_id == identifier:
                return action
        raise ProtocolError("Target exact-tail action lookup identifier is unknown.")

    def to_payload(self) -> dict[str, object]:
        return {
            **_target_library_payload(
                self.actions_by_target,
                plan_hashes=self.plan_hashes_by_target,
                plan_set_hash=self.plan_set_hash,
            ),
            "action_library_hash": self.action_library_hash,
        }


def build_inner_exact_tail_actions(
    outer_target: object,
    query_center: object,
) -> tuple[FrozenExactTailAction, ...]:
    outer = str(outer_target)
    query = str(query_center)
    sources = inner_candidate_sources(outer, query)
    geometry = inner_topup_geometry(sources)
    return (
        _freeze_action(
            outer=outer,
            query=query,
            action_id=BASE_ACTION_ID,
            action_role=INNER_ACTION_ROLE,
            geometry=geometry,
            selected_source=None,
            core=None,
            router_plan_hash=None,
            diagnostic_control=False,
        ),
        *(
            _freeze_action(
                outer=outer,
                query=query,
                action_id=h_x_e_action_id(source),
                action_role=INNER_ACTION_ROLE,
                geometry=geometry,
                selected_source=source,
                core=build_single_source_tail_action(source, geometry=geometry),
                router_plan_hash=None,
                diagnostic_control=True,
            )
            for source in sources
        ),
    )


def build_inner_exact_tail_action_library() -> FrozenInnerExactTailActionLibrary:
    actions = {
        outer: {
            query: build_inner_exact_tail_actions(outer, query)
            for query in candidate_sources(outer)
        }
        for outer in CENTERS
    }
    payload = _inner_library_payload(actions)
    return FrozenInnerExactTailActionLibrary(
        actions_by_outer_and_query=actions,
        action_library_hash=canonical_sha256(payload),
    )


def build_exact_tail_action_library(
    plans: Stage90R2PlanSet,
) -> FrozenExactTailActionLibrary:
    """Freeze B/U/G_delta/R2/P/Hxe for all targets before target scoring."""

    if not isinstance(plans, Stage90R2PlanSet):
        raise ProtocolError("Target action construction requires a typed R2 plan set.")
    actions_by_target: dict[str, tuple[FrozenExactTailAction, ...]] = {}
    plan_hashes: dict[str, str] = {}
    for target in CENTERS:
        plan = plans.by_target[target]
        actions_by_target[target] = build_target_actions(plan)
        plan_hashes[target] = plan.plan_hash
    payload = _target_library_payload(
        actions_by_target,
        plan_hashes=plan_hashes,
        plan_set_hash=plans.plan_set_hash,
    )
    return FrozenExactTailActionLibrary(
        actions_by_target=actions_by_target,
        plan_hashes_by_target=plan_hashes,
        plan_set_hash=plans.plan_set_hash,
        action_library_hash=canonical_sha256(payload),
    )


build_target_action_library = build_exact_tail_action_library


def build_target_actions(
    plan: FrozenR2DiagnosticPlan,
) -> tuple[FrozenExactTailAction, ...]:
    """Build one target menu for bounded target-parallel execution."""

    if not isinstance(plan, FrozenR2DiagnosticPlan) or plan.routing_status != ROUTING_STATUS:
        raise ProtocolError("Target actions require an insufficient-support diagnostic plan.")
    target = plan.target_id
    sources = candidate_sources(target)
    geometry = target_topup_geometry(sources)
    result = [
        _freeze_action(
            outer=target,
            query=target,
            action_id=BASE_ACTION_ID,
            action_role=TARGET_ACTION_ROLE,
            geometry=geometry,
            selected_source=None,
            core=None,
            router_plan_hash=plan.plan_hash,
            diagnostic_control=False,
        ),
        _freeze_action(
            outer=target,
            query=target,
            action_id=UNIFORM_ACTION_ID,
            action_role=TARGET_ACTION_ROLE,
            geometry=geometry,
            selected_source=None,
            core=build_uniform_topup_action(geometry),
            router_plan_hash=plan.plan_hash,
            diagnostic_control=False,
        ),
    ]
    for router in (GLOBAL_DELTA_ACTION_ID, R2_ACTION_ID, PERMUTATION_ACTION_ID):
        source = plan.proposed_source_by_router[router]
        result.append(
            _freeze_action(
                outer=target,
                query=target,
                action_id=router,
                action_role=TARGET_ACTION_ROLE,
                geometry=geometry,
                selected_source=source,
                core=build_single_source_tail_action(source, geometry=geometry),
                router_plan_hash=plan.plan_hash,
                diagnostic_control=router == PERMUTATION_ACTION_ID,
            )
        )
    result.extend(
        _freeze_action(
            outer=target,
            query=target,
            action_id=h_x_e_action_id(source),
            action_role=TARGET_ACTION_ROLE,
            geometry=geometry,
            selected_source=source,
            core=build_single_source_tail_action(source, geometry=geometry),
            router_plan_hash=plan.plan_hash,
            diagnostic_control=True,
        )
        for source in sources
    )
    if tuple(action.action_id for action in result) != expected_target_action_ids(target):
        raise ProtocolError("Target exact-tail action construction order drifted.")
    return tuple(result)


def _freeze_action(
    *,
    outer: str,
    query: str,
    action_id: str,
    action_role: str,
    geometry: TopupGeometry,
    selected_source: str | None,
    core: ResidualTopupAction | None,
    router_plan_hash: str | None,
    diagnostic_control: bool,
) -> FrozenExactTailAction:
    topup = (
        {source: 0 for source in geometry.source_order}
        if core is None
        else dict(core.topup_counts)
    )
    final = (
        {
            label: {source: geometry.base_per_source for source in geometry.source_order}
            for label in geometry.class_labels
        }
        if core is None
        else {
            label: dict(core.final_counts_by_class[label])
            for label in geometry.class_labels
        }
    )
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_exact_tail_action_v1",
        "outer_target_id": outer,
        "query_id": query,
        "action_id": action_id,
        "action_role": action_role,
        "geometry": geometry.to_payload(),
        "selected_source": selected_source,
        "realized_topup_total_per_class": (
            0 if core is None else geometry.topup_total_per_class
        ),
        "realized_final_total_per_class": (
            geometry.base_total_per_class
            if core is None
            else geometry.final_total_per_class
        ),
        "topup_counts_by_source": topup,
        "final_counts_by_class": {
            str(label): final[label] for label in geometry.class_labels
        },
        "core_action_hash": None if core is None else core.action_hash,
        "router_plan_hash": router_plan_hash,
        "diagnostic_control": diagnostic_control,
        "diagnostic_only": True,
        **_action_label_provenance(action_role, action_id),
        "evaluation_embeddings_used_to_build": False,
        "seed_selection_performed": False,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }
    return FrozenExactTailAction(
        outer_target_id=outer,
        query_id=query,
        action_id=action_id,
        action_role=action_role,
        geometry=geometry,
        selected_source=selected_source,
        core_action=core,
        router_plan_hash=router_plan_hash,
        diagnostic_control=diagnostic_control,
        action_hash=canonical_sha256(unhashed),
    )


def _expected_core(
    *,
    action_id: str,
    geometry: TopupGeometry,
    selected_source: str | None,
) -> ResidualTopupAction | None:
    if action_id == BASE_ACTION_ID:
        return None
    if action_id == UNIFORM_ACTION_ID:
        return build_uniform_topup_action(geometry)
    if action_id in {GLOBAL_DELTA_ACTION_ID, R2_ACTION_ID, PERMUTATION_ACTION_ID} or h_x_e_source(action_id) is not None:
        if selected_source is None:
            raise ProtocolError("Exact single-source tail has no selected source.")
        return build_single_source_tail_action(selected_source, geometry=geometry)
    raise ProtocolError("Exact-tail action identifier is unsupported.")


def _inner_library_payload(
    actions: Mapping[str, Mapping[str, tuple[FrozenExactTailAction, ...]]],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_inner_action_library_v1",
        "centers": list(CENTERS),
        "actions_by_outer_and_query": {
            outer: {
                query: [action.to_payload() for action in actions[outer][query]]
                for query in candidate_sources(outer)
            }
            for outer in CENTERS
        },
        "exact_inner_geometry": "7x144_per_class_base_plus_126_one_source_tail",
        "inner_actions_label_free": True,
        "development_crossfit_utility_labels_used": False,
        "policy_authorized": False,
    }


def _target_library_payload(
    actions: Mapping[str, tuple[FrozenExactTailAction, ...]],
    *,
    plan_hashes: Mapping[str, str],
    plan_set_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_target_action_library_v1",
        "centers": list(CENTERS),
        "action_count": sum(len(actions[target]) for target in CENTERS),
        "actions_by_target": {
            target: [action.to_payload() for action in actions[target]]
            for target in CENTERS
        },
        "plan_hashes_by_target": dict(plan_hashes),
        "plan_set_hash": plan_set_hash,
        "routing_status": ROUTING_STATUS,
        "actions_frozen_after_crossfit_development_scoring": True,
        "outer_H_development_rows_excluded_from_action_plan_H": True,
        "actions_frozen_before_terminal_target_scoring": True,
        "crossfit_development_utility_labels_used_for_G_delta_R2_P": True,
        "fixed_B_U_Hxe_controls_label_independent": True,
        "target_support_labels_used_for_actions": False,
        "terminal_target_labels_used_for_actions": False,
        "diagnostic_only": True,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }


def _action_label_provenance(action_role: str, action_id: str) -> dict[str, bool]:
    routed = action_role == TARGET_ACTION_ROLE and action_id in {
        GLOBAL_DELTA_ACTION_ID,
        R2_ACTION_ID,
        PERMUTATION_ACTION_ID,
    }
    return {
        "action_geometry_label_free": True,
        "crossfit_development_utility_labels_used_for_route": routed,
        "outer_H_development_rows_used_for_route": False,
        "target_support_labels_used_for_route": False,
        "terminal_target_labels_used_for_route": False,
    }


def _is_hash(value: object) -> bool:
    text = "" if value is None else str(value)
    return len(text) in {16, 64} and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = (
    "INNER_ACTION_ROLE",
    "TARGET_ACTION_ROLE",
    "FrozenExactTailAction",
    "FrozenExactTailActionLibrary",
    "FrozenInnerExactTailActionLibrary",
    "build_exact_tail_action_library",
    "build_inner_exact_tail_action_library",
    "build_inner_exact_tail_actions",
    "build_target_action_library",
    "build_target_actions",
)
