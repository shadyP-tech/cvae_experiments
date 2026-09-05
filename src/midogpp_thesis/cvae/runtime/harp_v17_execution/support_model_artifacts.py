"""Pooled source-development outcome and policy artifacts for HARP v17.

All nine source-q menu bundles are flattened into one fit. Raw source labels
remain inside one non-serializable ``SupportTruthCapability`` per case; only
aggregate primitive endpoints and the fitted pooled policy are projected into
durable artifacts. The fitted policy is then applied unchanged to all nine
target-H menus before evaluation truth can open.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.pooled_pairwise_selected_policy_router_v17.contracts import (
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
)
from ...routing.pooled_pairwise_selected_policy_router_v17.hashing import (
    canonical_hash,
    require_sha256,
)
from ...routing.pooled_pairwise_selected_policy_router_v17.truth import (
    SupportTruthCapability,
)
from .contracts import ArtifactValue, PrelabelRouteSet
from .science_pool import science_pool_plan
from .support_target_adapter import (
    SupportTargetMenuBundle,
    attach_support_outcome_inventory,
    build_support_prelabel_route_set,
)


_SOURCE_ENDPOINT_ARRAY_COLUMNS = (
    "bacc_gain",
    "harm_indicator",
    "brier_delta",
    "log_loss_delta",
)

_MODEL_CONFIG_ALIASES = {
    "nested_outer_folds": "outer_folds",
    "nested_inner_folds": "inner_folds",
    "minimum_routed_oof_cases_per_counted_center": (
        "minimum_routed_oof_cases_per_center"
    ),
    "source_oof_bootstrap_replicates": "bootstrap_replicates",
    "source_oof_bootstrap_alpha": "bootstrap_alpha",
    "source_oof_bootstrap_seed": "bootstrap_seed",
}
_TUPLE_CONFIG_FIELDS = {
    "opportunity_ridge_alphas",
    "ranker_ridge_alphas",
    "k_values",
    "lambda_values",
    "route_thresholds",
}
_MODEL_SCHEMA = "midogpp_harp_stage90_pooled_pairwise_selected_policy_router_v17"


class _PooledPolicy(Protocol):
    policy_hash: str

    def public_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class SupportOutcomeSurface:
    """Aggregate source endpoints plus memory-only case truth capabilities."""

    bundles: tuple[SupportTargetMenuBundle, ...] = field(repr=False, compare=False)
    outcomes_by_outer: tuple[tuple[str, tuple[SupportActionOutcome, ...]], ...]
    case_profiles_by_outer: tuple[
        tuple[str, tuple[SupportCaseClassProfile, ...]], ...
    ]
    truth_capabilities: tuple[SupportTruthCapability, ...] = field(
        repr=False, compare=False
    )
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        bundles = tuple(sorted(self.bundles, key=lambda row: row.center_id))
        outcomes = tuple(sorted(self.outcomes_by_outer, key=lambda row: row[0]))
        profiles = tuple(sorted(self.case_profiles_by_outer, key=lambda row: row[0]))
        capabilities = tuple(self.truth_capabilities)
        center_ids = tuple(row.center_id for row in bundles)
        expected_case_keys = tuple(
            (bundle.center_id, menu.case_id)
            for bundle in bundles
            for menu in bundle.source_menus
        )
        capability_case_keys = tuple(
            key for capability in capabilities for key in capability.case_keys
        )
        if (
            not bundles
            or len(center_ids) != len(set(center_ids))
            or tuple(center for center, _rows in outcomes) != center_ids
            or tuple(center for center, _rows in profiles) != center_ids
            or len(capabilities) != len(expected_case_keys)
            or any(
                not isinstance(capability, SupportTruthCapability)
                or len(capability.case_keys) != 1
                for capability in capabilities
            )
            or capability_case_keys != expected_case_keys
        ):
            raise ProtocolError("HARP v17 pooled source outcome surface is incomplete.")
        by_center = {row.center_id: row for row in bundles}
        for center, rows in outcomes:
            if any(
                not isinstance(row, SupportActionOutcome)
                or row.action.center_id != center
                for row in rows
            ):
                raise ProtocolError("HARP v17 source action outcomes crossed centers.")
        for center, rows in profiles:
            expected_cases = tuple(menu.case_id for menu in by_center[center].source_menus)
            if (
                tuple(row.case_id for row in rows) != expected_cases
                or any(
                    not isinstance(row, SupportCaseClassProfile)
                    or row.center_id != center
                    for row in rows
                )
            ):
                raise ProtocolError("HARP v17 source case profiles crossed centers.")
        object.__setattr__(self, "bundles", bundles)
        object.__setattr__(self, "outcomes_by_outer", outcomes)
        object.__setattr__(self, "case_profiles_by_outer", profiles)
        object.__setattr__(self, "truth_capabilities", capabilities)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v17_pooled_source_outcome_surface_v1",
                    "bundle_hashes": tuple(
                        (row.center_id, row.bundle_hash) for row in bundles
                    ),
                    "outcome_hashes": tuple(
                        (center, tuple(row.outcome_hash for row in rows))
                        for center, rows in outcomes
                    ),
                    "case_profile_hashes": tuple(
                        (center, tuple(row.profile_hash for row in rows))
                        for center, rows in profiles
                    ),
                    "truth_capability_hashes": tuple(
                        row.capability_hash for row in capabilities
                    ),
                    "one_memory_only_capability_per_source_case": True,
                    "raw_source_labels_persisted": False,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def source_menus(self) -> tuple[LabelFreeCaseMenu, ...]:
        return tuple(menu for bundle in self.bundles for menu in bundle.source_menus)

    def bundle_for(self, center_id: str) -> SupportTargetMenuBundle:
        for row in self.bundles:
            if row.center_id == center_id:
                return row
        raise ProtocolError("HARP v17 source surface lacks a requested center.")

    def outcomes_for(self, center_id: str) -> tuple[SupportActionOutcome, ...]:
        for center, rows in self.outcomes_by_outer:
            if center == center_id:
                return rows
        raise ProtocolError("HARP v17 source outcomes lack a requested center.")

    def case_profiles_for(
        self, center_id: str
    ) -> tuple[SupportCaseClassProfile, ...]:
        for center, rows in self.case_profiles_by_outer:
            if center == center_id:
                return rows
        raise ProtocolError("HARP v17 source profiles lack a requested center.")

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v17_pooled_source_outcome_surface_v1",
            "surface_hash": self.surface_hash,
            "source_center_ids": [row.center_id for row in self.bundles],
            "source_case_count": len(self.source_menus),
            "bundles": [row.report() for row in self.bundles],
            "outcomes_by_source_center": {
                center: [row.public_payload() for row in rows]
                for center, rows in self.outcomes_by_outer
            },
            "case_profiles_by_source_center": {
                center: [row.public_payload() for row in rows]
                for center, rows in self.case_profiles_by_outer
            },
            "truth_capabilities": [
                row.public_payload() for row in self.truth_capabilities
            ],
            "one_memory_only_capability_per_source_case": True,
            "raw_source_labels_persisted": False,
            "target_evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportRouterFitState:
    """The sole pooled policy fitted from all source-q development cases."""

    policy: _PooledPolicy = field(repr=False, compare=False)
    support_surface_hash: str
    state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        policy_hash = require_sha256(
            getattr(self.policy, "policy_hash", None), name="pooled policy hash"
        )
        require_sha256(self.support_surface_hash, name="source surface hash")
        if not callable(getattr(self.policy, "public_payload", None)):
            raise ProtocolError("HARP v17 pooled policy lacks a public projection.")
        object.__setattr__(
            self,
            "state_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v17_pooled_policy_fit_state_v1",
                    "support_surface_hash": self.support_surface_hash,
                    "policy_hash": policy_hash,
                    "pooled_policy_count": 1,
                    "source_labels_consumed": True,
                    "target_evaluation_labels_consumed": False,
                }
            ),
        )

    @property
    def routers(self) -> tuple[_PooledPolicy, ...]:
        """Compatibility view for runner code during the v17 transition."""

        return (self.policy,)

    def for_outer(self, _outer_target_id: str) -> _PooledPolicy:
        """Every target H receives the exact same pooled policy."""

        return self.policy

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v17_pooled_policy_fit_state_v1",
            "support_surface_hash": self.support_surface_hash,
            "policy": dict(self.policy.public_payload()),
            "policy_hash": self.policy.policy_hash,
            "pooled_policy_count": 1,
            "state_hash": self.state_hash,
            "source_labels_consumed": True,
            "target_evaluation_labels_consumed": False,
        }


def _as_router_config(value: object | None) -> RouterFitConfig:
    if value is None:
        return RouterFitConfig()
    if isinstance(value, RouterFitConfig):
        return value
    has_model_envelope = hasattr(value, "model")
    raw = getattr(value, "model", value)
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v17 router configuration is not a mapping.")
    names = tuple(RouterFitConfig.__dataclass_fields__)
    aliases_by_field = {
        field: source for source, field in _MODEL_CONFIG_ALIASES.items()
    }
    is_model_contract = has_model_envelope or "schema_version" in raw
    if is_model_contract and raw.get("schema_version") != _MODEL_SCHEMA:
        raise ProtocolError("HARP v17 router model schema drifted.")
    selected: dict[str, object] = {}
    missing: list[str] = []
    for name in names:
        alias = aliases_by_field.get(name)
        direct_present = name in raw
        alias_present = alias is not None and alias in raw
        if direct_present and alias_present:
            direct = raw[name]
            aliased = raw[alias]
            left = tuple(direct) if isinstance(direct, (tuple, list)) else direct
            right = tuple(aliased) if isinstance(aliased, (tuple, list)) else aliased
            if left != right:
                raise ProtocolError(
                    f"HARP v17 router configuration disagrees for {name}."
                )
        if direct_present:
            selected[name] = raw[name]
        elif alias_present:
            selected[name] = raw[alias]  # type: ignore[index]
        elif is_model_contract:
            missing.append(alias or name)
    if missing:
        raise ProtocolError(
            "HARP v17 router model omits science settings: " + ", ".join(missing)
        )
    if not is_model_contract:
        recognized = set(names) | set(_MODEL_CONFIG_ALIASES)
        unknown = sorted(str(key) for key in raw if key not in recognized)
        if unknown:
            raise ProtocolError(
                "HARP v17 router configuration has unknown settings: "
                + ", ".join(unknown)
            )
    for name in _TUPLE_CONFIG_FIELDS & set(selected):
        value_for_field = selected[name]
        if not isinstance(value_for_field, (tuple, list)):
            raise ProtocolError(f"HARP v17 router grid {name} is malformed.")
        selected[name] = tuple(value_for_field)
    try:
        return RouterFitConfig(**selected)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v17 router configuration is malformed.") from exc


def build_support_outcome_artifact(
    bundles: Sequence[SupportTargetMenuBundle],
    support_labels_by_outer: Mapping[str, Sequence[object]],
) -> ArtifactValue:
    """Attach source labels to sealed q cases without retaining raw labels."""

    bundle_rows = tuple(sorted(bundles, key=lambda row: row.center_id))
    expected = tuple(row.center_id for row in bundle_rows)
    if (
        not bundle_rows
        or len(expected) != len(set(expected))
        or tuple(sorted(str(key) for key in support_labels_by_outer)) != expected
    ):
        raise ProtocolError("HARP v17 source label shards do not match menu centers.")
    attachments = tuple(
        attach_support_outcome_inventory(
            bundle,
            support_labels_by_outer[bundle.center_id],
        )
        for bundle in bundle_rows
    )
    surface = SupportOutcomeSurface(
        bundles=bundle_rows,
        outcomes_by_outer=tuple(
            (row.center_id, row.action_outcomes) for row in attachments
        ),
        case_profiles_by_outer=tuple(
            (row.center_id, row.case_profiles) for row in attachments
        ),
        truth_capabilities=tuple(
            capability
            for row in attachments
            for capability in row.truth_capabilities
        ),
    )
    numeric = np.asarray(
        [
            (
                row.bacc_gain,
                float(row.harmed),
                row.brier_delta,
                row.log_loss_delta,
            )
            for _center, rows in surface.outcomes_by_outer
            for row in rows
        ],
        dtype=np.float64,
    ).reshape((-1, len(_SOURCE_ENDPOINT_ARRAY_COLUMNS)))
    offsets = [0]
    for _center, rows in surface.outcomes_by_outer:
        offsets.append(offsets[-1] + len(rows))
    body = {
        **surface.public_payload(),
        "raw_source_labels_persisted": False,
        "endpoint_array_columns": list(_SOURCE_ENDPOINT_ARRAY_COLUMNS),
    }
    return ArtifactValue(
        state=surface,
        manifest={**body, "artifact_hash": canonical_hash(body)},
        arrays={
            "source_endpoint_values": numeric,
            "source_center_outcome_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def _require_support_surface(value: ArtifactValue) -> SupportOutcomeSurface:
    if not isinstance(value, ArtifactValue) or not isinstance(
        value.state, SupportOutcomeSurface
    ):
        raise ProtocolError("HARP v17 pooled fitting requires a typed source artifact.")
    if value.manifest.get("surface_hash") != value.state.surface_hash:
        raise ProtocolError("HARP v17 source artifact hash drifted.")
    return value.state


def build_support_router_artifact(
    support: ArtifactValue,
    *,
    config: RouterFitConfig | Mapping[str, object] | object | None = None,
) -> ArtifactValue:
    """Fit exactly one pooled policy over all nine source-q menu bundles."""

    surface = _require_support_surface(support)
    if tuple(row.center_id for row in surface.bundles) != tuple(CENTERS):
        raise ProtocolError("HARP v17 pooled fitting requires all nine source centers.")
    selected_config = _as_router_config(config)
    try:
        from ...routing.pooled_pairwise_selected_policy_router_v17 import (
            fit_source_router,
        )
    except ImportError as exc:  # pragma: no cover - incomplete source checkout
        raise ProtocolError("HARP v17 pooled science router is unavailable.") from exc
    policy = fit_source_router(
        surface.source_menus,
        surface.truth_capabilities,
        config=selected_config,
        case_profiles=tuple(
            row for _center, rows in surface.case_profiles_by_outer for row in rows
        ),
        action_outcomes=tuple(
            row for _center, rows in surface.outcomes_by_outer for row in rows
        ),
    )
    state = SupportRouterFitState(policy, surface.surface_hash)
    runtime = getattr(config, "runtime", None)
    if isinstance(runtime, Mapping):
        configured_pool_capacity: Mapping[str, object] | None = dict(
            science_pool_plan(runtime)
        )
    else:
        configured_pool_capacity = None
    fit_execution = {
        "schema_version": "midogpp_harp_v17_pooled_fit_execution_v1",
        "execution_mode": "parent_process_nonserializable_truth_capability",
        "worker_count": 0,
        "blas_threads": 1,
        "cuda_used": False,
        "truth_capability_cross_process_transport": False,
        "phase_disjoint_from_gpu_and_classifier_pools": True,
    }
    model_hash = require_sha256(
        getattr(policy, "model_hash", policy.policy_hash), name="pooled model hash"
    )
    body = {
        **state.public_payload(),
        "model_hash": model_hash,
        "policy_hash": policy.policy_hash,
        "router_config": selected_config.public_payload(),
        "pooled_source_center_count": len(surface.bundles),
        "pooled_source_case_count": len(surface.source_menus),
        "truth_capability_count": len(surface.truth_capabilities),
        "one_pooled_policy_fit": True,
        "target_evaluation_features_used_for_fit": False,
        "target_evaluation_labels_used": False,
        "raw_source_labels_persisted": False,
        "science_pool_topology": fit_execution,
        "configured_science_pool_capacity": (
            None
            if configured_pool_capacity is None
            else dict(configured_pool_capacity)
        ),
        "configured_science_pool_used_for_truth_bearing_fit": False,
        "truth_capabilities_nonserializable": True,
        "aggregate_source_surface_reused_without_rederivation": True,
    }
    return ArtifactValue(
        state=state,
        manifest={**body, "artifact_hash": canonical_hash(body)},
        arrays={},
    )


def build_support_target_routes(
    bundles: Sequence[SupportTargetMenuBundle],
    fitted: ArtifactValue,
    *,
    target_action_hash: str | None = None,
) -> PrelabelRouteSet:
    """Apply the sole pooled policy to every target-H menu before truth opens."""

    if not isinstance(fitted, ArtifactValue) or not isinstance(
        fitted.state, SupportRouterFitState
    ):
        raise ProtocolError("HARP v17 target routing requires a pooled policy.")
    if fitted.manifest.get("state_hash") != fitted.state.state_hash:
        raise ProtocolError("HARP v17 pooled policy artifact drifted.")
    return build_support_prelabel_route_set(
        bundles,
        fitted.state.policy,
        target_action_hash=target_action_hash,
    )


def report_support_router_artifact(fitted: ArtifactValue) -> Mapping[str, object]:
    """Expose the pooled policy/admission report without raw source labels."""

    if not isinstance(fitted, ArtifactValue) or not isinstance(
        fitted.state, SupportRouterFitState
    ):
        raise ProtocolError("HARP v17 policy report requires a fitted artifact.")
    return {
        "schema_version": "midogpp_harp_v17_pooled_policy_report_v1",
        "support_surface_hash": fitted.state.support_surface_hash,
        "state_hash": fitted.state.state_hash,
        "model_hash": fitted.manifest.get("model_hash"),
        "policy_hash": fitted.state.policy.policy_hash,
        "pooled_policy_count": 1,
        "policy": dict(fitted.state.policy.public_payload()),
        "source_labels_consumed": True,
        "raw_source_labels_persisted": False,
        "target_evaluation_labels_consumed": False,
    }


__all__ = (
    "SupportOutcomeSurface",
    "SupportRouterFitState",
    "build_support_outcome_artifact",
    "build_support_router_artifact",
    "build_support_target_routes",
    "report_support_router_artifact",
)
