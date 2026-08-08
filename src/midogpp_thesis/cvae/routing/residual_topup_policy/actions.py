"""Frozen B/U/G/S/P and single-source-tail action-library construction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ..residual_topup import (
    ResidualTopupAction,
    build_borda_directed_topup_action,
    build_single_source_tail_action,
    build_uniform_topup_action,
    target_topup_geometry,
)
from ..residual_topup.hashing import canonical_sha256
from .config import (
    BASE_ACTION_ID,
    BASE_PER_SOURCE_PER_CLASS,
    BASE_TOTAL_PER_CLASS,
    GLOBAL_ACTION_ID,
    MATCHED_TOTAL_PER_CLASS,
    PERMUTATION_ACTION_ID,
    SINGLE_SOURCE_ACTION_NAMESPACE,
    SUPPORT_ACTION_ID,
    TOPUP_TOTAL_PER_CLASS,
    UNIFORM_ACTION_ID,
)
from .contracts import TargetProxyPolicySummary, canonical_source_ids


BASE_ACTION_KIND = "fixed_equal_union_base_only"
UNIFORM_POLICY_ACTION_KIND = "fixed_uniform_residual_topup_control"
GLOBAL_POLICY_ACTION_KIND = (
    "fixed_leave_target_out_global_proxy_midrank_residual_topup"
)
SUPPORT_POLICY_ACTION_KIND = "fixed_target_support_proxy_midrank_residual_topup"
PERMUTATION_POLICY_ACTION_KIND = (
    "fixed_target_support_proxy_midrank_source_identity_permutation_control"
)
SINGLE_SOURCE_POLICY_ACTION_KIND = "fixed_single_source_tail_residual_topup_diagnostic"

BASE_ACTION_SEMANTICS = "equal_union_base_128_per_source_per_class_no_topup"
UNIFORM_ACTION_SEMANTICS = "equal_union_base_plus_exact_uniform_128_per_class_tail"
GLOBAL_ACTION_SEMANTICS = (
    "equal_union_base_plus_leave_target_and_query_out_global_proxy_borda_tail"
)
SUPPORT_ACTION_SEMANTICS = (
    "equal_union_base_plus_unlabeled_target_support_proxy_borda_tail"
)
PERMUTATION_ACTION_SEMANTICS = (
    "support_proxy_ranks_reassigned_by_frozen_source_identity_permutation_before_"
    "borda_tail"
)
SINGLE_SOURCE_ACTION_SEMANTICS = (
    "equal_union_base_plus_all_128_tail_rows_from_one_predeclared_source"
)


@dataclass(frozen=True)
class FrozenPolicyAction:
    """One fully enumerated action frozen before Stage-70 extraction."""

    outer_target: str
    action_id: str
    policy_id: str
    action_kind: str
    action_semantics: str
    source_order: tuple[str, ...]
    base_per_source_per_class: int
    topup_total_per_class: int
    final_total_per_class: int
    mean_normalized_midrank_by_source: Mapping[str, float]
    source_identity_permutation: Mapping[str, str]
    selected_source: str | None
    direction_weights_by_source: Mapping[str, float]
    topup_counts_by_source: Mapping[str, int]
    final_counts_by_class: Mapping[int, Mapping[str, int]]
    core_action_kind: str | None
    core_action_hash: str | None
    diagnostic_control: bool
    action_hash: str

    def __post_init__(self) -> None:
        sources = canonical_source_ids(self.source_order)
        if sources != self.source_order or self.outer_target in sources:
            raise ProtocolError("Frozen policy action source geometry is invalid.")
        rank_values = _float_mapping(
            self.mean_normalized_midrank_by_source,
            allowed_sources=sources,
            allow_empty=True,
        )
        if rank_values and set(rank_values) != set(sources):
            raise ProtocolError("Frozen policy action rank grid is incomplete.")
        permutation = {
            str(source): str(value)
            for source, value in self.source_identity_permutation.items()
        }
        if permutation and (
            tuple(permutation) != sources or set(permutation.values()) != set(sources)
        ):
            raise ProtocolError("Frozen policy action permutation grid is invalid.")
        direction = _float_mapping(
            self.direction_weights_by_source,
            allowed_sources=sources,
            allow_empty=True,
        )
        topup = _int_mapping(self.topup_counts_by_source, sources=sources)
        final = _nested_counts(self.final_counts_by_class, sources=sources)
        if (
            self.base_per_source_per_class != BASE_PER_SOURCE_PER_CLASS
            or self.topup_total_per_class not in {0, TOPUP_TOTAL_PER_CLASS}
            or self.final_total_per_class
            != BASE_TOTAL_PER_CLASS + self.topup_total_per_class
            or sum(topup.values()) != self.topup_total_per_class
            or any(
                sum(class_counts.values()) != self.final_total_per_class
                for class_counts in final.values()
            )
            or any(
                class_counts[source]
                != BASE_PER_SOURCE_PER_CLASS + topup[source]
                for class_counts in final.values()
                for source in sources
            )
        ):
            raise ProtocolError("Frozen policy action count geometry drifted.")
        if self.topup_total_per_class == 0:
            if direction or self.core_action_kind is not None or self.core_action_hash is not None:
                raise ProtocolError("Base-only action must not carry a top-up action.")
        elif (
            set(direction) != set(sources)
            or abs(sum(direction.values()) - 1.0) > 1.0e-12
            or not self.core_action_kind
            or not self.core_action_hash
        ):
            raise ProtocolError("Frozen top-up action payload is incomplete.")
        payload = self._payload_without_hash(
            ranks=rank_values,
            permutation=permutation,
            direction=direction,
            topup=topup,
            final=final,
        )
        if self.action_hash != canonical_sha256(payload):
            raise ProtocolError("Frozen policy action hash is invalid.")
        object.__setattr__(
            self, "mean_normalized_midrank_by_source", MappingProxyType(rank_values)
        )
        object.__setattr__(
            self, "source_identity_permutation", MappingProxyType(permutation)
        )
        object.__setattr__(
            self, "direction_weights_by_source", MappingProxyType(direction)
        )
        object.__setattr__(self, "topup_counts_by_source", MappingProxyType(topup))
        object.__setattr__(
            self,
            "final_counts_by_class",
            MappingProxyType(
                {label: MappingProxyType(values) for label, values in final.items()}
            ),
        )

    def _payload_without_hash(
        self,
        *,
        ranks: Mapping[str, float],
        permutation: Mapping[str, str],
        direction: Mapping[str, float],
        topup: Mapping[str, int],
        final: Mapping[int, Mapping[str, int]],
    ) -> dict[str, object]:
        return {
            "schema_version": "midogpp_residual_topup_frozen_policy_action_v1",
            "outer_target": self.outer_target,
            "action_id": self.action_id,
            "policy_id": self.policy_id,
            "action_kind": self.action_kind,
            "action_semantics": self.action_semantics,
            "source_order": list(self.source_order),
            "base_per_source_per_class": self.base_per_source_per_class,
            "topup_total_per_class": self.topup_total_per_class,
            "final_total_per_class": self.final_total_per_class,
            "mean_normalized_midrank_by_source": dict(ranks),
            "source_identity_permutation": dict(permutation),
            "selected_source": self.selected_source,
            "direction_weights_by_source": dict(direction),
            "topup_counts_by_source": dict(topup),
            "final_counts_by_class": {
                str(label): dict(final[label]) for label in (0, 1)
            },
            "core_action_kind": self.core_action_kind,
            "core_action_hash": self.core_action_hash,
            "diagnostic_control": self.diagnostic_control,
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._payload_without_hash(
            ranks=self.mean_normalized_midrank_by_source,
            permutation=self.source_identity_permutation,
            direction=self.direction_weights_by_source,
            topup=self.topup_counts_by_source,
            final=self.final_counts_by_class,
        )
        payload["action_hash"] = self.action_hash
        return payload


@dataclass(frozen=True)
class FrozenActionLibrary:
    centers: tuple[str, ...]
    actions_by_target: Mapping[str, tuple[FrozenPolicyAction, ...]]
    action_count: int
    action_library_hash: str

    def __post_init__(self) -> None:
        centers = canonical_source_ids(self.centers)
        if centers != self.centers or tuple(self.actions_by_target) != centers:
            raise ProtocolError("Frozen action-library target order is invalid.")
        copied: dict[str, tuple[FrozenPolicyAction, ...]] = {}
        hashes: set[str] = set()
        for target in centers:
            actions = tuple(self.actions_by_target[target])
            expected_sources = tuple(center for center in centers if center != target)
            expected_ids = (
                BASE_ACTION_ID,
                UNIFORM_ACTION_ID,
                GLOBAL_ACTION_ID,
                SUPPORT_ACTION_ID,
                PERMUTATION_ACTION_ID,
                *(
                    f"{SINGLE_SOURCE_ACTION_NAMESPACE}::{source}"
                    for source in expected_sources
                ),
            )
            if (
                tuple(action.action_id for action in actions) != expected_ids
                or any(action.outer_target != target for action in actions)
                or any(action.source_order != expected_sources for action in actions)
            ):
                raise ProtocolError("Frozen action-library target payload drifted.")
            for action in actions:
                if action.action_hash in hashes:
                    raise ProtocolError("Frozen action identities must be globally distinct.")
                hashes.add(action.action_hash)
            copied[target] = actions
        if self.action_count != sum(len(actions) for actions in copied.values()):
            raise ProtocolError("Frozen action-library count drifted.")
        payload = {
            "schema_version": "midogpp_residual_topup_frozen_action_library_v1",
            "centers": list(centers),
            "action_count": self.action_count,
            "actions_by_target": {
                target: [action.to_payload() for action in copied[target]]
                for target in centers
            },
            "policy_frozen_before_stage70": True,
        }
        if self.action_library_hash != canonical_sha256(payload):
            raise ProtocolError("Frozen action-library hash is invalid.")
        object.__setattr__(self, "actions_by_target", MappingProxyType(copied))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "midogpp_residual_topup_frozen_action_library_v1",
            "centers": list(self.centers),
            "action_count": self.action_count,
            "actions_by_target": {
                target: [action.to_payload() for action in self.actions_by_target[target]]
                for target in self.centers
            },
            "policy_frozen_before_stage70": True,
        }
        payload["action_library_hash"] = self.action_library_hash
        return payload


def build_frozen_action_library(
    summaries_by_target: Mapping[str, TargetProxyPolicySummary],
) -> FrozenActionLibrary:
    """Build every predeclared action without reading downstream outcomes."""

    if not isinstance(summaries_by_target, Mapping):
        raise ProtocolError("Proxy-policy summaries must be a target mapping.")
    centers = canonical_source_ids(summaries_by_target)
    actions_by_target: dict[str, tuple[FrozenPolicyAction, ...]] = {}
    for target in centers:
        summary = summaries_by_target[target]
        if not isinstance(summary, TargetProxyPolicySummary):
            raise ProtocolError("Proxy-policy target summary is invalid.")
        sources = summary.candidate_sources
        if sources != tuple(center for center in centers if center != target):
            raise ProtocolError("Proxy-policy target candidate universe drifted.")
        geometry = target_topup_geometry(sources)
        uniform = build_uniform_topup_action(geometry)
        global_action = build_borda_directed_topup_action(
            summary.global_summary.mean_normalized_midrank_by_source,
            geometry=geometry,
        )
        support_action = build_borda_directed_topup_action(
            summary.support_summary.mean_normalized_midrank_by_source,
            geometry=geometry,
        )
        permuted_ranks = {
            source: summary.support_summary.mean_normalized_midrank_by_source[
                summary.source_identity_permutation[source]
            ]
            for source in sources
        }
        permutation_action = build_borda_directed_topup_action(
            permuted_ranks,
            geometry=geometry,
        )
        target_actions = [
            _base_action(target=target, sources=sources),
            _wrapped_action(
                target=target,
                action_id=UNIFORM_ACTION_ID,
                policy_id="U",
                action_kind=UNIFORM_POLICY_ACTION_KIND,
                action_semantics=UNIFORM_ACTION_SEMANTICS,
                core=uniform,
            ),
            _wrapped_action(
                target=target,
                action_id=GLOBAL_ACTION_ID,
                policy_id="G",
                action_kind=GLOBAL_POLICY_ACTION_KIND,
                action_semantics=GLOBAL_ACTION_SEMANTICS,
                core=global_action,
                ranks=summary.global_summary.mean_normalized_midrank_by_source,
            ),
            _wrapped_action(
                target=target,
                action_id=SUPPORT_ACTION_ID,
                policy_id="S",
                action_kind=SUPPORT_POLICY_ACTION_KIND,
                action_semantics=SUPPORT_ACTION_SEMANTICS,
                core=support_action,
                ranks=summary.support_summary.mean_normalized_midrank_by_source,
            ),
            _wrapped_action(
                target=target,
                action_id=PERMUTATION_ACTION_ID,
                policy_id="P",
                action_kind=PERMUTATION_POLICY_ACTION_KIND,
                action_semantics=PERMUTATION_ACTION_SEMANTICS,
                core=permutation_action,
                ranks=permuted_ranks,
                permutation=summary.source_identity_permutation,
                diagnostic_control=True,
            ),
        ]
        target_actions.extend(
            _wrapped_action(
                target=target,
                action_id=f"{SINGLE_SOURCE_ACTION_NAMESPACE}::{source}",
                policy_id=f"Hxe::{source}",
                action_kind=SINGLE_SOURCE_POLICY_ACTION_KIND,
                action_semantics=SINGLE_SOURCE_ACTION_SEMANTICS,
                core=build_single_source_tail_action(source, geometry=geometry),
                selected_source=source,
                diagnostic_control=True,
            )
            for source in sources
        )
        actions_by_target[target] = tuple(target_actions)
    payload = {
        "schema_version": "midogpp_residual_topup_frozen_action_library_v1",
        "centers": list(centers),
        "action_count": sum(len(actions) for actions in actions_by_target.values()),
        "actions_by_target": {
            target: [action.to_payload() for action in actions_by_target[target]]
            for target in centers
        },
        "policy_frozen_before_stage70": True,
    }
    return FrozenActionLibrary(
        centers=centers,
        actions_by_target=actions_by_target,
        action_count=int(payload["action_count"]),
        action_library_hash=canonical_sha256(payload),
    )


def _base_action(*, target: str, sources: tuple[str, ...]) -> FrozenPolicyAction:
    topup = {source: 0 for source in sources}
    final = {
        label: {source: BASE_PER_SOURCE_PER_CLASS for source in sources}
        for label in (0, 1)
    }
    payload = _action_payload(
        target=target,
        action_id=BASE_ACTION_ID,
        policy_id="B",
        action_kind=BASE_ACTION_KIND,
        action_semantics=BASE_ACTION_SEMANTICS,
        sources=sources,
        topup_total=0,
        ranks={},
        permutation={},
        selected_source=None,
        direction={},
        topup=topup,
        final=final,
        core_action_kind=None,
        core_action_hash=None,
        diagnostic_control=False,
    )
    return FrozenPolicyAction(**payload, action_hash=_action_hash(payload))


def _wrapped_action(
    *,
    target: str,
    action_id: str,
    policy_id: str,
    action_kind: str,
    action_semantics: str,
    core: ResidualTopupAction,
    ranks: Mapping[str, float] | None = None,
    permutation: Mapping[str, str] | None = None,
    selected_source: str | None = None,
    diagnostic_control: bool = False,
) -> FrozenPolicyAction:
    payload = _action_payload(
        target=target,
        action_id=action_id,
        policy_id=policy_id,
        action_kind=action_kind,
        action_semantics=action_semantics,
        sources=core.geometry.source_order,
        topup_total=core.geometry.topup_total_per_class,
        ranks=dict(ranks or {}),
        permutation=dict(permutation or {}),
        selected_source=selected_source,
        direction=dict(core.direction_weights),
        topup=dict(core.topup_counts),
        final={label: dict(core.final_counts_by_class[label]) for label in (0, 1)},
        core_action_kind=core.action_kind,
        core_action_hash=core.action_hash,
        diagnostic_control=diagnostic_control,
    )
    return FrozenPolicyAction(**payload, action_hash=_action_hash(payload))


def _action_payload(
    *,
    target: str,
    action_id: str,
    policy_id: str,
    action_kind: str,
    action_semantics: str,
    sources: tuple[str, ...],
    topup_total: int,
    ranks: Mapping[str, float],
    permutation: Mapping[str, str],
    selected_source: str | None,
    direction: Mapping[str, float],
    topup: Mapping[str, int],
    final: Mapping[int, Mapping[str, int]],
    core_action_kind: str | None,
    core_action_hash: str | None,
    diagnostic_control: bool,
) -> dict[str, object]:
    return {
        "outer_target": target,
        "action_id": action_id,
        "policy_id": policy_id,
        "action_kind": action_kind,
        "action_semantics": action_semantics,
        "source_order": sources,
        "base_per_source_per_class": BASE_PER_SOURCE_PER_CLASS,
        "topup_total_per_class": topup_total,
        "final_total_per_class": BASE_TOTAL_PER_CLASS + topup_total,
        "mean_normalized_midrank_by_source": dict(ranks),
        "source_identity_permutation": dict(permutation),
        "selected_source": selected_source,
        "direction_weights_by_source": dict(direction),
        "topup_counts_by_source": dict(topup),
        "final_counts_by_class": {label: dict(final[label]) for label in (0, 1)},
        "core_action_kind": core_action_kind,
        "core_action_hash": core_action_hash,
        "diagnostic_control": diagnostic_control,
    }


def _action_hash(payload: Mapping[str, object]) -> str:
    final = payload["final_counts_by_class"]
    if not isinstance(final, Mapping):
        raise ProtocolError("Frozen policy action class counts are invalid.")
    serialized = {
        "schema_version": "midogpp_residual_topup_frozen_policy_action_v1",
        **dict(payload),
        "source_order": list(payload["source_order"]),  # type: ignore[arg-type]
        "final_counts_by_class": {
            str(label): dict(final[label]) for label in (0, 1)  # type: ignore[arg-type]
        },
    }
    return canonical_sha256(serialized)


def _float_mapping(
    values: Mapping[str, float],
    *,
    allowed_sources: tuple[str, ...],
    allow_empty: bool,
) -> dict[str, float]:
    result = {str(source): float(value) for source, value in values.items()}
    if (not result and not allow_empty) or not set(result).issubset(allowed_sources):
        raise ProtocolError("Frozen policy action floating mapping is invalid.")
    if any(value < 0.0 or value > 1.0 for value in result.values()):
        raise ProtocolError("Frozen policy action floating values are invalid.")
    return {source: result[source] for source in allowed_sources if source in result}


def _int_mapping(values: Mapping[str, int], *, sources: tuple[str, ...]) -> dict[str, int]:
    result = {str(source): int(value) for source, value in values.items()}
    if set(result) != set(sources) or any(value < 0 for value in result.values()):
        raise ProtocolError("Frozen policy action count mapping is invalid.")
    return {source: result[source] for source in sources}


def _nested_counts(
    values: Mapping[int, Mapping[str, int]],
    *,
    sources: tuple[str, ...],
) -> dict[int, dict[str, int]]:
    if set(values) != {0, 1}:
        raise ProtocolError("Frozen policy action class-count grid is invalid.")
    return {
        label: _int_mapping(values[label], sources=sources) for label in (0, 1)
    }


__all__ = (
    "BASE_ACTION_KIND",
    "GLOBAL_POLICY_ACTION_KIND",
    "PERMUTATION_POLICY_ACTION_KIND",
    "SINGLE_SOURCE_POLICY_ACTION_KIND",
    "SUPPORT_POLICY_ACTION_KIND",
    "UNIFORM_POLICY_ACTION_KIND",
    "FrozenActionLibrary",
    "FrozenPolicyAction",
    "build_frozen_action_library",
)
