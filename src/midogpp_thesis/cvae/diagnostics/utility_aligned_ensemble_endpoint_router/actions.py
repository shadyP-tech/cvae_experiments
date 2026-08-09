"""Hash-locked additive-tail actions for the ensemble-endpoint diagnostic."""

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
    PERMUTATION_ACTION_ID,
    PRIMARY_DIAGNOSTIC_ACTION_IDS,
    ROUTER_DIAGNOSTIC_IDS,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
    h_x_e_source,
    inner_candidate_sources,
)
from .diagnostic_plan import (
    FrozenEnsembleEndpointDiagnosticPlan,
    Stage90EnsembleDiagnosticPlanSet,
)


INNER_ACTION_ROLE = "source_inner_ensemble_endpoint_development"
TARGET_ACTION_ROLE = "consumed_target_ensemble_endpoint_diagnostic"


@dataclass(frozen=True)
class FrozenEnsembleEndpointAction:
    outer_target_id: str
    query_id: str
    action_id: str
    action_role: str
    geometry: TopupGeometry
    selected_source: str | None
    core_action: ResidualTopupAction | None
    router_plan_hash: str | None
    inner_support_shift_lock_hash: str | None
    target_support_shift_lock_hash: str | None
    target_probe_seal_hash: str | None
    diagnostic_control: bool
    action_hash: str
    diagnostic_only: bool = True
    evaluation_embeddings_used_to_build: bool = False
    seed_selection_performed: bool = False
    may_update_policy: bool = False
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
            if self.router_plan_hash is not None or any(
                value is not None
                for value in (
                    self.inner_support_shift_lock_hash,
                    self.target_support_shift_lock_hash,
                    self.target_probe_seal_hash,
                )
            ):
                raise ProtocolError("Source-inner ensemble actions cannot carry a target plan.")
        elif self.action_role == TARGET_ACTION_ROLE:
            if outer != query:
                raise ProtocolError("Target ensemble actions require q == H.")
            sources = candidate_sources(outer)
            expected_geometry = target_topup_geometry(sources)
            allowed = expected_target_action_ids(outer)
            if not _is_hash(self.router_plan_hash):
                raise ProtocolError("Target ensemble actions require a frozen plan hash.")
            if any(
                not _is_hash(value)
                for value in (
                    self.inner_support_shift_lock_hash,
                    self.target_support_shift_lock_hash,
                    self.target_probe_seal_hash,
                )
            ):
                raise ProtocolError("Target ensemble actions require sealed shift/probe locks.")
        else:
            raise ProtocolError("Ensemble endpoint action role is invalid.")
        if (
            self.geometry != expected_geometry
            or action_id not in allowed
            or self.diagnostic_only is not True
            or self.evaluation_embeddings_used_to_build is not False
            or self.seed_selection_performed is not False
            or self.may_update_policy is not False
            or self.policy_authorized is not False
            or self.fallback_authorized is not False
            or self.promotion_authorized is not False
            or self.deployment_authorized is not False
            or type(self.diagnostic_control) is not bool
        ):
            raise ProtocolError("Ensemble endpoint action claim boundary drifted.")
        expected_core = _expected_core(
            action_id=action_id,
            geometry=expected_geometry,
            selected_source=selected,
        )
        if expected_core is None:
            if self.core_action is not None or selected is not None:
                raise ProtocolError("Base action B cannot carry a top-up route.")
        elif (
            not isinstance(self.core_action, ResidualTopupAction)
            or self.core_action.action_hash != expected_core.action_hash
            or self.core_action.geometry != expected_geometry
        ):
            raise ProtocolError("Ensemble endpoint action does not reconstruct.")
        if action_id == UNIFORM_ACTION_ID and selected is not None:
            raise ProtocolError("Uniform action U cannot select one source.")
        if action_id not in {BASE_ACTION_ID, UNIFORM_ACTION_ID} and selected not in sources:
            raise ProtocolError("Ensemble endpoint tail action has no legal source.")
        embedded = h_x_e_source(action_id)
        if embedded is not None and embedded != selected:
            raise ProtocolError("Hxe action identity and selected source drifted.")
        expected_hash = canonical_sha256(
            _action_payload(
                outer=outer,
                query=query,
                action_id=action_id,
                action_role=self.action_role,
                geometry=self.geometry,
                selected_source=selected,
                core_action=self.core_action,
                router_plan_hash=self.router_plan_hash,
                inner_support_shift_lock_hash=self.inner_support_shift_lock_hash,
                target_support_shift_lock_hash=self.target_support_shift_lock_hash,
                target_probe_seal_hash=self.target_probe_seal_hash,
                diagnostic_control=self.diagnostic_control,
            )
        )
        if self.action_hash != expected_hash:
            raise ProtocolError("Frozen ensemble endpoint action hash drifted.")
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
                    {source: self.geometry.base_per_source for source in self.source_order}
                )
                for label in self.geometry.class_labels
            }
        )

    @property
    def required_source_capacity_per_class(self) -> int:
        return max(self.final_counts_by_class[0].values())

    def to_payload(self) -> dict[str, object]:
        return {
            **_action_payload(
                outer=self.outer_target_id,
                query=self.query_id,
                action_id=self.action_id,
                action_role=self.action_role,
                geometry=self.geometry,
                selected_source=self.selected_source,
                core_action=self.core_action,
                router_plan_hash=self.router_plan_hash,
                inner_support_shift_lock_hash=self.inner_support_shift_lock_hash,
                target_support_shift_lock_hash=self.target_support_shift_lock_hash,
                target_probe_seal_hash=self.target_probe_seal_hash,
                diagnostic_control=self.diagnostic_control,
            ),
            "action_hash": self.action_hash,
        }


@dataclass(frozen=True)
class FrozenInnerEnsembleEndpointActionLibrary:
    actions_by_outer_and_query: Mapping[
        str, Mapping[str, tuple[FrozenEnsembleEndpointAction, ...]]
    ]
    action_library_hash: str

    def __post_init__(self) -> None:
        values = {
            str(outer): {str(query): tuple(actions) for query, actions in queries.items()}
            for outer, queries in self.actions_by_outer_and_query.items()
        }
        if tuple(values) != CENTERS:
            raise ProtocolError("Inner ensemble action library H coverage drifted.")
        for outer in CENTERS:
            if tuple(values[outer]) != candidate_sources(outer):
                raise ProtocolError("Inner ensemble action library q coverage drifted.")
            for query, actions in values[outer].items():
                expected = (
                    BASE_ACTION_ID,
                    *(h_x_e_action_id(source) for source in inner_candidate_sources(outer, query)),
                )
                if tuple(action.action_id for action in actions) != expected:
                    raise ProtocolError("Inner ensemble action menu drifted.")
        expected_hash = canonical_sha256(_inner_library_payload(values))
        if self.action_library_hash != expected_hash:
            raise ProtocolError("Inner ensemble action-library hash drifted.")
        object.__setattr__(
            self,
            "actions_by_outer_and_query",
            MappingProxyType(
                {outer: MappingProxyType(queries) for outer, queries in values.items()}
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {**_inner_library_payload(self.actions_by_outer_and_query), "action_library_hash": self.action_library_hash}


@dataclass(frozen=True)
class FrozenEnsembleEndpointActionLibrary:
    actions_by_target: Mapping[str, tuple[FrozenEnsembleEndpointAction, ...]]
    plan_hashes_by_target: Mapping[str, str]
    plan_set_hash: str
    action_library_hash: str

    def __post_init__(self) -> None:
        actions = {str(target): tuple(value) for target, value in self.actions_by_target.items()}
        plan_hashes = {str(target): str(value) for target, value in self.plan_hashes_by_target.items()}
        if tuple(actions) != CENTERS or tuple(plan_hashes) != CENTERS:
            raise ProtocolError("Target ensemble action library coverage drifted.")
        hashes: set[str] = set()
        for target in CENTERS:
            values = actions[target]
            if (
                len(values) != EXPECTED_TARGET_ACTION_COUNT
                or tuple(action.action_id for action in values) != expected_target_action_ids(target)
                or any(
                    action.outer_target_id != target
                    or action.query_id != target
                    or action.action_role != TARGET_ACTION_ROLE
                    or action.router_plan_hash != plan_hashes[target]
                    for action in values
                )
            ):
                raise ProtocolError("Target ensemble action menu drifted.")
            hashes.update(action.action_hash for action in values)
        if len(hashes) != EXPECTED_FROZEN_TARGET_ACTION_COUNT:
            raise ProtocolError("Target ensemble action hashes are not unique.")
        expected = canonical_sha256(
            _target_library_payload(
                actions,
                plan_hashes=plan_hashes,
                plan_set_hash=self.plan_set_hash,
            )
        )
        if self.action_library_hash != expected:
            raise ProtocolError("Target ensemble action-library hash drifted.")
        object.__setattr__(self, "actions_by_target", MappingProxyType(actions))
        object.__setattr__(self, "plan_hashes_by_target", MappingProxyType(plan_hashes))

    @property
    def action_count(self) -> int:
        return sum(len(values) for values in self.actions_by_target.values())

    def action(self, target_center: object, action_id: object) -> FrozenEnsembleEndpointAction:
        target = str(target_center)
        identifier = str(action_id)
        for action in self.actions_by_target.get(target, ()):
            if action.action_id == identifier:
                return action
        raise ProtocolError("Target ensemble action lookup is unknown.")

    def to_payload(self) -> dict[str, object]:
        return {
            **_target_library_payload(
                self.actions_by_target,
                plan_hashes=self.plan_hashes_by_target,
                plan_set_hash=self.plan_set_hash,
            ),
            "action_library_hash": self.action_library_hash,
        }


def build_inner_ensemble_endpoint_actions(
    outer_target: object,
    query_center: object,
) -> tuple[FrozenEnsembleEndpointAction, ...]:
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
            router_plan_hash=None,
            inner_support_shift_lock_hash=None,
            target_support_shift_lock_hash=None,
            target_probe_seal_hash=None,
            diagnostic_control=True,
        ),
        *(
            _freeze_action(
                outer=outer,
                query=query,
                action_id=h_x_e_action_id(source),
                action_role=INNER_ACTION_ROLE,
                geometry=geometry,
                selected_source=source,
                router_plan_hash=None,
                inner_support_shift_lock_hash=None,
                target_support_shift_lock_hash=None,
                target_probe_seal_hash=None,
                diagnostic_control=False,
            )
            for source in sources
        ),
    )


def inner_action_library_for(
    outer_target: object,
    query_center: object,
) -> tuple[FrozenEnsembleEndpointAction, ...]:
    """Execution-facing base-plus-seven-tail menu for one ``(H,q)``."""

    return build_inner_ensemble_endpoint_actions(outer_target, query_center)


def build_inner_ensemble_endpoint_action_library() -> FrozenInnerEnsembleEndpointActionLibrary:
    values = {
        outer: {
            query: build_inner_ensemble_endpoint_actions(outer, query)
            for query in candidate_sources(outer)
        }
        for outer in CENTERS
    }
    payload = _inner_library_payload(values)
    return FrozenInnerEnsembleEndpointActionLibrary(
        actions_by_outer_and_query=values,
        action_library_hash=canonical_sha256(payload),
    )


def build_target_ensemble_endpoint_actions(
    plan: FrozenEnsembleEndpointDiagnosticPlan,
) -> tuple[FrozenEnsembleEndpointAction, ...]:
    if not isinstance(plan, FrozenEnsembleEndpointDiagnosticPlan):
        raise ProtocolError("Target ensemble actions require a typed plan.")
    target = plan.target_id
    geometry = target_topup_geometry(plan.candidate_sources)
    selected_by_action = {
        BASE_ACTION_ID: None,
        UNIFORM_ACTION_ID: None,
        **dict(plan.proposed_source_by_router),
        **{h_x_e_action_id(source): source for source in plan.candidate_sources},
    }
    return tuple(
        _freeze_action(
            outer=target,
            query=target,
            action_id=action_id,
            action_role=TARGET_ACTION_ROLE,
            geometry=geometry,
            selected_source=selected_by_action[action_id],
            router_plan_hash=plan.plan_hash,
            inner_support_shift_lock_hash=plan.inner_support_shift_lock_hash,
            target_support_shift_lock_hash=plan.target_support_shift_lock_hash,
            target_probe_seal_hash=plan.target_probe_seal_hash,
            diagnostic_control=(
                action_id in {BASE_ACTION_ID, UNIFORM_ACTION_ID, PERMUTATION_ACTION_ID}
                or h_x_e_source(action_id) is not None
            ),
        )
        for action_id in expected_target_action_ids(target)
    )


def build_ensemble_endpoint_action_library(
    plans: Stage90EnsembleDiagnosticPlanSet,
) -> FrozenEnsembleEndpointActionLibrary:
    if not isinstance(plans, Stage90EnsembleDiagnosticPlanSet):
        raise ProtocolError("Target ensemble action library requires a typed plan set.")
    values = {
        target: build_target_ensemble_endpoint_actions(plans.by_target[target])
        for target in CENTERS
    }
    plan_hashes = {target: plans.by_target[target].plan_hash for target in CENTERS}
    payload = _target_library_payload(
        values,
        plan_hashes=plan_hashes,
        plan_set_hash=plans.plan_set_hash,
    )
    return FrozenEnsembleEndpointActionLibrary(
        actions_by_target=values,
        plan_hashes_by_target=plan_hashes,
        plan_set_hash=plans.plan_set_hash,
        action_library_hash=canonical_sha256(payload),
    )


def _freeze_action(
    *,
    outer: str,
    query: str,
    action_id: str,
    action_role: str,
    geometry: TopupGeometry,
    selected_source: str | None,
    router_plan_hash: str | None,
    inner_support_shift_lock_hash: str | None,
    target_support_shift_lock_hash: str | None,
    target_probe_seal_hash: str | None,
    diagnostic_control: bool,
) -> FrozenEnsembleEndpointAction:
    core = _expected_core(
        action_id=action_id,
        geometry=geometry,
        selected_source=selected_source,
    )
    payload = _action_payload(
        outer=outer,
        query=query,
        action_id=action_id,
        action_role=action_role,
        geometry=geometry,
        selected_source=selected_source,
        core_action=core,
        router_plan_hash=router_plan_hash,
        inner_support_shift_lock_hash=inner_support_shift_lock_hash,
        target_support_shift_lock_hash=target_support_shift_lock_hash,
        target_probe_seal_hash=target_probe_seal_hash,
        diagnostic_control=diagnostic_control,
    )
    return FrozenEnsembleEndpointAction(
        outer_target_id=outer,
        query_id=query,
        action_id=action_id,
        action_role=action_role,
        geometry=geometry,
        selected_source=selected_source,
        core_action=core,
        router_plan_hash=router_plan_hash,
        inner_support_shift_lock_hash=inner_support_shift_lock_hash,
        target_support_shift_lock_hash=target_support_shift_lock_hash,
        target_probe_seal_hash=target_probe_seal_hash,
        diagnostic_control=diagnostic_control,
        action_hash=canonical_sha256(payload),
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
    if selected_source is None:
        raise ProtocolError("A non-control ensemble action requires a selected source.")
    return build_single_source_tail_action(selected_source, geometry=geometry)


def _action_payload(
    *,
    outer: str,
    query: str,
    action_id: str,
    action_role: str,
    geometry: TopupGeometry,
    selected_source: str | None,
    core_action: ResidualTopupAction | None,
    router_plan_hash: str | None,
    inner_support_shift_lock_hash: str | None,
    target_support_shift_lock_hash: str | None,
    target_probe_seal_hash: str | None,
    diagnostic_control: bool,
) -> dict[str, object]:
    topup_counts = (
        {source: 0 for source in geometry.source_order}
        if core_action is None
        else dict(core_action.topup_counts)
    )
    final_counts = (
        {
            str(label): {source: geometry.base_per_source for source in geometry.source_order}
            for label in geometry.class_labels
        }
        if core_action is None
        else {
            str(label): dict(core_action.final_counts_by_class[label])
            for label in geometry.class_labels
        }
    )
    uses_development_response = (
        action_role == TARGET_ACTION_ROLE and action_id in ROUTER_DIAGNOSTIC_IDS
    )
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_action_v1",
        "outer_target_id": outer,
        "query_id": query,
        "action_id": action_id,
        "action_role": action_role,
        "geometry": geometry.to_payload(),
        "selected_source": selected_source,
        "realized_topup_total_per_class": 0 if core_action is None else geometry.topup_total_per_class,
        "realized_final_total_per_class": geometry.base_total_per_class + (0 if core_action is None else geometry.topup_total_per_class),
        "topup_counts_by_source": topup_counts,
        "final_counts_by_class": final_counts,
        "core_action_hash": None if core_action is None else core_action.action_hash,
        "router_plan_hash": router_plan_hash,
        "inner_support_shift_lock_hash": inner_support_shift_lock_hash,
        "target_support_shift_lock_hash": target_support_shift_lock_hash,
        "target_probe_seal_hash": target_probe_seal_hash,
        "diagnostic_control": diagnostic_control,
        "diagnostic_only": True,
        "development_candidate_ensemble_responses_used_to_build": uses_development_response,
        "development_per_seed_utility_rows_used_to_build": False,
        "target_support_labels_used_to_build": False,
        "terminal_target_labels_used_to_build": False,
        "evaluation_embeddings_used_to_build": False,
        "seed_selection_performed": False,
        "may_update_policy": False,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }


def _inner_library_payload(
    values: Mapping[str, Mapping[str, tuple[FrozenEnsembleEndpointAction, ...]]],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_inner_action_library_v1",
        "centers": list(CENTERS),
        "action_hashes_by_outer_and_query": {
            outer: {
                query: [action.action_hash for action in values[outer][query]]
                for query in candidate_sources(outer)
            }
            for outer in CENTERS
        },
        "development_candidate_ensemble_response_count": 504,
        "development_per_seed_utility_rows_may_feed_model": False,
    }


def _target_library_payload(
    values: Mapping[str, tuple[FrozenEnsembleEndpointAction, ...]],
    *,
    plan_hashes: Mapping[str, str],
    plan_set_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_utility_aligned_stage90_ensemble_action_library_v1",
        "centers": list(CENTERS),
        "action_ids_by_target": {
            target: [action.action_id for action in values[target]] for target in CENTERS
        },
        "action_hashes_by_target": {
            target: [action.action_hash for action in values[target]] for target in CENTERS
        },
        "plan_hashes_by_target": dict(plan_hashes),
        "plan_set_hash": plan_set_hash,
        "action_count": sum(len(actions) for actions in values.values()),
        "routing_status": "INSUFFICIENT_SUPPORT_FOR_POLICY",
        "diagnostic_only": True,
        "may_update_policy": False,
        "policy_authorized": False,
        "fallback_authorized": False,
        "promotion_authorized": False,
        "deployment_authorized": False,
    }


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 16 and value.strip() == value


__all__ = (
    "INNER_ACTION_ROLE",
    "TARGET_ACTION_ROLE",
    "FrozenEnsembleEndpointAction",
    "FrozenEnsembleEndpointActionLibrary",
    "FrozenInnerEnsembleEndpointActionLibrary",
    "build_ensemble_endpoint_action_library",
    "build_inner_ensemble_endpoint_action_library",
    "build_inner_ensemble_endpoint_actions",
    "build_target_ensemble_endpoint_actions",
    "inner_action_library_for",
)
