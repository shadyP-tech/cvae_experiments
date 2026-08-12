"""Frozen physical B/U/G/R/P/Hxe action identities and geometries."""

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
    EXPECTED_TARGET_ACTION_COUNT,
    GLOBAL_ACTION_ID,
    H_X_E_ACTION_PREFIX,
    ORACLE_ACTION_ROLE,
    PERMUTATION_ACTION_ID,
    ROUTED_ACTION_ID,
    SOURCE_INNER_ACTION_ROLE,
    TARGET_ACTION_ROLE,
    UNIFORM_ACTION_ID,
    candidate_sources,
    expected_development_action_ids,
    expected_target_action_ids,
    h_x_e_action_id,
    h_x_e_source,
    inner_candidate_sources,
)
from .policy import FrozenTargetPolicy, FrozenTargetPolicySet


@dataclass(frozen=True)
class FrozenEndpointAction:
    """One immutable physical action; ``core_action=None`` means exact B."""

    outer_target_id: str
    query_id: str
    action_id: str
    action_role: str
    geometry: TopupGeometry
    selected_source: str | None
    effective_action_id: str
    core_action: ResidualTopupAction | None
    policy_hash: str | None
    diagnostic_control: bool
    action_hash: str

    def __post_init__(self) -> None:
        outer = str(self.outer_target_id)
        query = str(self.query_id)
        if outer not in CENTERS or query not in CENTERS:
            raise ProtocolError("Frozen endpoint action center is invalid.")
        if self.action_role == SOURCE_INNER_ACTION_ROLE:
            allowed_sources = inner_candidate_sources(outer, query)
            allowed_ids = expected_development_action_ids(outer, query)
            if self.policy_hash is not None:
                raise ProtocolError("Development actions cannot bind a target policy.")
        elif self.action_role in {TARGET_ACTION_ROLE, ORACLE_ACTION_ROLE}:
            if query != outer:
                raise ProtocolError("Target-static actions require q == H.")
            allowed_sources = candidate_sources(outer)
            allowed_ids = expected_target_action_ids(outer)
            if not _text(self.policy_hash):
                raise ProtocolError("Target actions require a frozen target policy hash.")
            if (self.action_role == ORACLE_ACTION_ROLE) != (
                h_x_e_source(self.action_id) is not None
            ):
                raise ProtocolError("Hxe oracle action role drifted.")
        else:
            raise ProtocolError("Frozen endpoint action role is invalid.")
        if (
            self.action_id not in allowed_ids
            or self.geometry.source_order != tuple(sorted(allowed_sources))
            or self.selected_source is not None
            and self.selected_source not in allowed_sources
            or self.effective_action_id not in {
                BASE_ACTION_ID,
                UNIFORM_ACTION_ID,
                GLOBAL_ACTION_ID,
                ROUTED_ACTION_ID,
                PERMUTATION_ACTION_ID,
                self.action_id,
            }
        ):
            raise ProtocolError("Frozen endpoint action identity/geometry drifted.")
        is_exact_base = self.effective_action_id == BASE_ACTION_ID
        if is_exact_base != (self.core_action is None):
            raise ProtocolError("Exact-B realization and core action disagree.")
        if self.core_action is not None:
            if self.core_action.geometry != self.geometry:
                raise ProtocolError("Frozen endpoint core geometry drifted.")
            if self.action_id == UNIFORM_ACTION_ID:
                if self.selected_source is not None:
                    raise ProtocolError("Uniform control cannot select a source.")
            elif self.selected_source is None:
                raise ProtocolError("A realized single-source action lacks its source.")
        if self.action_hash != canonical_sha256(self._unhashed_payload()):
            raise ProtocolError("Frozen endpoint action hash drifted.")

    @property
    def topup_counts_by_source(self) -> Mapping[str, int]:
        if self.core_action is None:
            return MappingProxyType(
                {source: 0 for source in self.geometry.source_order}
            )
        return self.core_action.topup_counts

    @property
    def realized_total_per_class(self) -> int:
        return (
            self.geometry.base_total_per_class
            if self.core_action is None
            else self.geometry.final_total_per_class
        )

    def _unhashed_payload(self) -> dict[str, object]:
        return _action_payload(
            outer=self.outer_target_id,
            query=self.query_id,
            action_id=self.action_id,
            action_role=self.action_role,
            geometry=self.geometry,
            selected_source=self.selected_source,
            effective_action_id=self.effective_action_id,
            core_action=self.core_action,
            policy_hash=self.policy_hash,
            diagnostic_control=self.diagnostic_control,
        )

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "action_hash": self.action_hash}


@dataclass(frozen=True)
class FrozenTargetActionSet:
    target_id: str
    actions: tuple[FrozenEndpointAction, ...]
    policy_hash: str
    action_set_hash: str

    def __post_init__(self) -> None:
        target = str(self.target_id)
        expected_ids = expected_target_action_ids(target)
        if (
            len(self.actions) != EXPECTED_TARGET_ACTION_COUNT
            or tuple(action.action_id for action in self.actions) != expected_ids
            or any(
                action.outer_target_id != target
                or action.query_id != target
                or action.policy_hash != self.policy_hash
                for action in self.actions
            )
        ):
            raise ProtocolError("Frozen target action set is incomplete.")
        if self.action_set_hash != canonical_sha256(self._unhashed_payload(target)):
            raise ProtocolError("Frozen target action-set hash drifted.")
        object.__setattr__(self, "target_id", target)

    @property
    def by_action_id(self) -> Mapping[str, FrozenEndpointAction]:
        return MappingProxyType({action.action_id: action for action in self.actions})

    def _unhashed_payload(self, target: str | None = None) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_frozen_target_action_set_v1",
            "target_id": target or self.target_id,
            "action_ids": [action.action_id for action in self.actions],
            "action_hashes": [action.action_hash for action in self.actions],
            "policy_hash": self.policy_hash,
            "one_static_action_per_target_geometry": True,
            "terminal_scores_may_update_actions": False,
            "diagnostic_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "action_set_hash": self.action_set_hash}


@dataclass(frozen=True)
class FrozenTargetActionLibrary:
    by_target: Mapping[str, FrozenTargetActionSet]
    policy_set_hash: str
    action_library_hash: str

    def __post_init__(self) -> None:
        values = {str(key): value for key, value in self.by_target.items()}
        if (
            tuple(values) != CENTERS
            or any(value.target_id != target for target, value in values.items())
        ):
            raise ProtocolError("Frozen target action library is incomplete.")
        payload = _action_library_payload(values, self.policy_set_hash)
        if self.action_library_hash != canonical_sha256(payload):
            raise ProtocolError("Frozen target action-library hash drifted.")
        object.__setattr__(self, "by_target", MappingProxyType(values))

    def to_payload(self) -> dict[str, object]:
        return {
            **_action_library_payload(self.by_target, self.policy_set_hash),
            "action_library_hash": self.action_library_hash,
        }


def build_development_actions(
    *, outer_target_id: object, query_id: object
) -> tuple[FrozenEndpointAction, ...]:
    """Build B and seven Hxe actions used to create one q response list."""

    outer = str(outer_target_id)
    query = str(query_id)
    sources = inner_candidate_sources(outer, query)
    geometry = inner_topup_geometry(sources)
    return tuple(
        _build_action(
            outer=outer,
            query=query,
            action_id=action_id,
            action_role=SOURCE_INNER_ACTION_ROLE,
            geometry=geometry,
            selected_source=h_x_e_source(action_id),
            effective_action_id=(
                BASE_ACTION_ID if action_id == BASE_ACTION_ID else action_id
            ),
            policy_hash=None,
            diagnostic_control=action_id == BASE_ACTION_ID,
        )
        for action_id in expected_development_action_ids(outer, query)
    )


def build_frozen_target_actions(policy: FrozenTargetPolicy) -> FrozenTargetActionSet:
    """Realize B/U/G/R/P/Hxe after the target policy is sealed."""

    if not isinstance(policy, FrozenTargetPolicy):
        raise ProtocolError("Target actions require a typed frozen policy.")
    target = policy.target_id
    geometry = target_topup_geometry(candidate_sources(target))
    core = policy.core_policy
    selections = {
        GLOBAL_ACTION_ID: (
            core.role_selected_source[GLOBAL_ACTION_ID],
            core.role_selected_action[GLOBAL_ACTION_ID],
        ),
        ROUTED_ACTION_ID: (
            core.role_selected_source[ROUTED_ACTION_ID],
            core.role_selected_action[ROUTED_ACTION_ID],
        ),
        PERMUTATION_ACTION_ID: (
            core.role_selected_source[PERMUTATION_ACTION_ID],
            core.role_selected_action[PERMUTATION_ACTION_ID],
        ),
    }
    actions: list[FrozenEndpointAction] = []
    for action_id in expected_target_action_ids(target):
        oracle_source = h_x_e_source(action_id)
        if action_id == BASE_ACTION_ID:
            selected, effective = None, BASE_ACTION_ID
        elif action_id == UNIFORM_ACTION_ID:
            selected, effective = None, UNIFORM_ACTION_ID
        elif oracle_source is not None:
            selected, effective = oracle_source, action_id
        else:
            selected, effective = selections[action_id]
        actions.append(
            _build_action(
                outer=target,
                query=target,
                action_id=action_id,
                action_role=(
                    ORACLE_ACTION_ROLE
                    if oracle_source is not None
                    else TARGET_ACTION_ROLE
                ),
                geometry=geometry,
                selected_source=selected,
                effective_action_id=effective,
                policy_hash=policy.policy_hash,
                diagnostic_control=(
                    action_id in {
                        BASE_ACTION_ID,
                        UNIFORM_ACTION_ID,
                        GLOBAL_ACTION_ID,
                        PERMUTATION_ACTION_ID,
                    }
                    or oracle_source is not None
                ),
            )
        )
    payload = {
        "schema_version": "midogpp_consumed_test_frozen_target_action_set_v1",
        "target_id": target,
        "action_ids": [action.action_id for action in actions],
        "action_hashes": [action.action_hash for action in actions],
        "policy_hash": policy.policy_hash,
        "one_static_action_per_target_geometry": True,
        "terminal_scores_may_update_actions": False,
        "diagnostic_only": True,
    }
    return FrozenTargetActionSet(
        target_id=target,
        actions=tuple(actions),
        policy_hash=policy.policy_hash,
        action_set_hash=canonical_sha256(payload),
    )


def build_frozen_target_action_library(
    policies: FrozenTargetPolicySet,
) -> FrozenTargetActionLibrary:
    if not isinstance(policies, FrozenTargetPolicySet):
        raise ProtocolError("Target action library requires the typed policy set.")
    values = {
        target: build_frozen_target_actions(policies.by_target[target])
        for target in CENTERS
    }
    payload = _action_library_payload(values, policies.policy_set_hash)
    return FrozenTargetActionLibrary(
        by_target=values,
        policy_set_hash=policies.policy_set_hash,
        action_library_hash=canonical_sha256(payload),
    )


def _build_action(
    *,
    outer: str,
    query: str,
    action_id: str,
    action_role: str,
    geometry: TopupGeometry,
    selected_source: str | None,
    effective_action_id: str,
    policy_hash: str | None,
    diagnostic_control: bool,
) -> FrozenEndpointAction:
    if effective_action_id == BASE_ACTION_ID:
        core_action = None
    elif action_id == UNIFORM_ACTION_ID:
        core_action = build_uniform_topup_action(geometry)
    else:
        if selected_source is None:
            raise ProtocolError("Realized endpoint action requires a selected source.")
        core_action = build_single_source_tail_action(
            selected_source, geometry=geometry
        )
    payload = _action_payload(
        outer=outer,
        query=query,
        action_id=action_id,
        action_role=action_role,
        geometry=geometry,
        selected_source=selected_source,
        effective_action_id=effective_action_id,
        core_action=core_action,
        policy_hash=policy_hash,
        diagnostic_control=diagnostic_control,
    )
    return FrozenEndpointAction(
        outer_target_id=outer,
        query_id=query,
        action_id=action_id,
        action_role=action_role,
        geometry=geometry,
        selected_source=selected_source,
        effective_action_id=effective_action_id,
        core_action=core_action,
        policy_hash=policy_hash,
        diagnostic_control=diagnostic_control,
        action_hash=canonical_sha256(payload),
    )


def _action_payload(
    *,
    outer: str,
    query: str,
    action_id: str,
    action_role: str,
    geometry: TopupGeometry,
    selected_source: str | None,
    effective_action_id: str,
    core_action: ResidualTopupAction | None,
    policy_hash: str | None,
    diagnostic_control: bool,
) -> dict[str, object]:
    topup_counts = (
        {source: 0 for source in geometry.source_order}
        if core_action is None
        else dict(core_action.topup_counts)
    )
    return {
        "schema_version": "midogpp_consumed_test_frozen_endpoint_action_v1",
        "outer_target_id": outer,
        "query_id": query,
        "action_id": action_id,
        "action_role": action_role,
        "effective_action_id": effective_action_id,
        "selected_source": selected_source,
        "geometry": geometry.to_payload(),
        "topup_counts_by_source": topup_counts,
        "realized_total_per_class": (
            geometry.base_total_per_class
            if core_action is None
            else geometry.final_total_per_class
        ),
        "core_action_hash": None if core_action is None else core_action.action_hash,
        "policy_hash": policy_hash,
        "diagnostic_control": diagnostic_control,
        "target_static": action_role != SOURCE_INNER_ACTION_ROLE,
        "case_router_used": False,
        "labels_used_to_build": False,
        "terminal_scores_used_to_build": False,
        "diagnostic_only": True,
    }


def _action_library_payload(
    values: Mapping[str, FrozenTargetActionSet], policy_set_hash: str
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_frozen_target_action_library_v1",
        "centers": list(CENTERS),
        "action_set_hashes_by_target": {
            target: values[target].action_set_hash for target in CENTERS
        },
        "policy_set_hash": policy_set_hash,
        "target_count": len(CENTERS),
        "reported_action_count": sum(
            len(values[target].actions) for target in CENTERS
        ),
        "physical_action_count": len(CENTERS) * (2 + len(CENTERS) - 1),
        "terminal_scores_may_update_actions": False,
        "diagnostic_only": True,
    }


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


__all__ = (
    "FrozenEndpointAction",
    "FrozenTargetActionLibrary",
    "FrozenTargetActionSet",
    "build_development_actions",
    "build_frozen_target_action_library",
    "build_frozen_target_actions",
)
