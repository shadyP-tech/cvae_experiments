"""H-local support outcome and router artifacts for HARP v16."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ...protocol import ProtocolError
from ...routing.hierarchical_support_action_risk_router_v16 import (
    FittedSupportRouter,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
    fit_support_router,
)
from ...routing.hierarchical_support_action_risk_router_v16.hashing import (
    canonical_hash,
)
from .contracts import ArtifactValue, PrelabelRouteSet
from .science_pool import execute_science_jobs, science_pool_plan
from .support_target_adapter import (
    SupportTargetMenuBundle,
    attach_support_outcome_inventory,
    build_support_prelabel_route_set,
)


_SUPPORT_OOF_ARRAY_COLUMNS = (
    "predicted_gain",
    "predicted_harm_probability",
    "predicted_brier_delta",
    "predicted_log_loss_delta",
    "observed_gain",
    "observed_harm",
    "observed_brier_delta",
    "observed_log_loss_delta",
)


@dataclass(frozen=True, slots=True)
class SupportOutcomeSurface:
    bundles: tuple[SupportTargetMenuBundle, ...] = field(repr=False, compare=False)
    outcomes_by_outer: tuple[tuple[str, tuple[SupportActionOutcome, ...]], ...]
    case_profiles_by_outer: tuple[
        tuple[str, tuple[SupportCaseClassProfile, ...]], ...
    ]
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        bundles = tuple(sorted(self.bundles, key=lambda row: row.outer_target_id))
        outcomes = tuple(sorted(self.outcomes_by_outer, key=lambda row: row[0]))
        profiles = tuple(sorted(self.case_profiles_by_outer, key=lambda row: row[0]))
        bundle_ids = tuple(row.outer_target_id for row in bundles)
        if (
            not bundles
            or len(set(bundle_ids)) != len(bundle_ids)
            or tuple(outer for outer, _ in outcomes) != bundle_ids
            or tuple(outer for outer, _ in profiles) != bundle_ids
            or any(
                any(
                    not isinstance(row, SupportActionOutcome)
                    or row.action.outer_target_id != outer
                    or not row.has_class_local_components
                    or row.normalization_case_count is not None
                    for row in rows
                )
                for outer, rows in outcomes
            )
            or any(
                tuple(row.case_id for row in profile_rows)
                != tuple(
                    menu.case_id
                    for menu in next(
                        bundle for bundle in bundles if bundle.outer_target_id == outer
                    ).support_menus
                )
                or any(
                    not isinstance(row, SupportCaseClassProfile)
                    or row.outer_target_id != outer
                    for row in profile_rows
                )
                for outer, profile_rows in profiles
            )
        ):
            raise ProtocolError("HARP v16 support outcome surface is incomplete.")
        object.__setattr__(self, "bundles", bundles)
        object.__setattr__(self, "outcomes_by_outer", outcomes)
        object.__setattr__(self, "case_profiles_by_outer", profiles)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v16_support_outcome_surface_v1",
                    "bundle_hashes": tuple(
                        (row.outer_target_id, row.bundle_hash) for row in bundles
                    ),
                    "outcome_hashes": tuple(
                        (
                            outer,
                            tuple(row.outcome_hash for row in rows),
                        )
                        for outer, rows in outcomes
                    ),
                    "case_profile_hashes": tuple(
                        (outer, tuple(row.profile_hash for row in rows))
                        for outer, rows in profiles
                    ),
                    "support_labels_consumed": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def bundle_for(self, outer_target_id: str) -> SupportTargetMenuBundle:
        for row in self.bundles:
            if row.outer_target_id == outer_target_id:
                return row
        raise ProtocolError("HARP v16 support surface lacks target H.")

    def outcomes_for(self, outer_target_id: str) -> tuple[SupportActionOutcome, ...]:
        for outer, rows in self.outcomes_by_outer:
            if outer == outer_target_id:
                return rows
        raise ProtocolError("HARP v16 support outcomes lack target H.")

    def case_profiles_for(
        self, outer_target_id: str
    ) -> tuple[SupportCaseClassProfile, ...]:
        for outer, rows in self.case_profiles_by_outer:
            if outer == outer_target_id:
                return rows
        raise ProtocolError("HARP v16 support class profiles lack target H.")

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v16_support_outcome_surface_v1",
            "surface_hash": self.surface_hash,
            "outer_targets": [row.outer_target_id for row in self.bundles],
            "bundles": [row.report() for row in self.bundles],
            "outcomes_by_outer": {
                outer: [row.public_payload() for row in rows]
                for outer, rows in self.outcomes_by_outer
            },
            "case_profiles_by_outer": {
                outer: [row.public_payload() for row in rows]
                for outer, rows in self.case_profiles_by_outer
            },
            "support_labels_consumed": True,
            "evaluation_labels_consumed": False,
        }


@dataclass(frozen=True, slots=True)
class SupportRouterFitState:
    routers: tuple[FittedSupportRouter, ...]
    support_surface_hash: str
    state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        routers = tuple(sorted(self.routers, key=lambda row: row.outer_target_id))
        if (
            not routers
            or len({row.outer_target_id for row in routers}) != len(routers)
            or len(self.support_surface_hash) != 64
        ):
            raise ProtocolError("HARP v16 support router state is malformed.")
        object.__setattr__(self, "routers", routers)
        object.__setattr__(
            self,
            "state_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_v16_support_router_fit_state_v1",
                    "support_surface_hash": self.support_surface_hash,
                    "router_hashes": tuple(
                        (row.outer_target_id, row.router_hash) for row in routers
                    ),
                    "support_labels_consumed": True,
                    "evaluation_labels_consumed": False,
                }
            ),
        )

    def for_outer(self, outer_target_id: str) -> FittedSupportRouter:
        for row in self.routers:
            if row.outer_target_id == outer_target_id:
                return row
        raise ProtocolError("HARP v16 fitted support router lacks target H.")

    def public_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_v16_support_router_fit_state_v1",
            "support_surface_hash": self.support_surface_hash,
            "routers": [row.public_payload() for row in self.routers],
            "state_hash": self.state_hash,
            "support_labels_consumed": True,
            "evaluation_labels_consumed": False,
        }


def _as_router_config(value: object | None) -> RouterFitConfig:
    if value is None:
        return RouterFitConfig()
    if isinstance(value, RouterFitConfig):
        return value
    raw = getattr(value, "model", value)
    if not isinstance(raw, Mapping):
        raise ProtocolError("HARP v16 router configuration is not a mapping.")
    names = tuple(RouterFitConfig.__dataclass_fields__)
    try:
        selected = {name: raw[name] for name in names if name in raw}
        return RouterFitConfig(**selected)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("HARP v16 router configuration is malformed.") from exc


def build_support_outcome_artifact(
    bundles: Sequence[SupportTargetMenuBundle],
    support_labels_by_outer: Mapping[str, Sequence[object]],
) -> ArtifactValue:
    """Join center-scoped support capabilities without retaining raw labels."""

    bundle_rows = tuple(sorted(bundles, key=lambda row: row.outer_target_id))
    expected = tuple(row.outer_target_id for row in bundle_rows)
    if (
        not bundle_rows
        or len(set(expected)) != len(expected)
        or tuple(sorted(str(key) for key in support_labels_by_outer)) != expected
    ):
        raise ProtocolError("HARP v16 support label shards do not match menu targets.")
    attachments = tuple(
        attach_support_outcome_inventory(
            bundle,
            support_labels_by_outer[bundle.outer_target_id],
        )
        for bundle in bundle_rows
    )
    outcomes = tuple(
        (row.outer_target_id, row.action_outcomes) for row in attachments
    )
    profiles = tuple(
        (row.outer_target_id, row.case_profiles) for row in attachments
    )
    surface = SupportOutcomeSurface(bundle_rows, outcomes, profiles)
    numeric = np.asarray(
        [
            (
                row.bacc_gain,
                float(row.harmed),
                row.brier_delta,
                row.log_loss_delta,
            )
            for _outer, rows in surface.outcomes_by_outer
            for row in rows
        ],
        dtype=np.float64,
    ).reshape((-1, 4))
    offsets = [0]
    for _outer, rows in surface.outcomes_by_outer:
        offsets.append(offsets[-1] + len(rows))
    body = surface.public_payload()
    body = {
        **body,
        "raw_support_labels_persisted": False,
        "endpoint_array_columns": [
            "bacc_gain",
            "harm_indicator",
            "brier_delta",
            "log_loss_delta",
        ],
    }
    return ArtifactValue(
        state=surface,
        manifest={**body, "artifact_hash": canonical_hash(body)},
        arrays={
            "support_endpoint_values": numeric,
            "outer_outcome_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def _require_support_surface(value: ArtifactValue) -> SupportOutcomeSurface:
    if not isinstance(value, ArtifactValue) or not isinstance(
        value.state, SupportOutcomeSurface
    ):
        raise ProtocolError("HARP v16 fitting requires a typed support artifact.")
    if value.manifest.get("surface_hash") != value.state.surface_hash:
        raise ProtocolError("HARP v16 support artifact hash drifted.")
    return value.state


def _model_set_hash(routers: Sequence[FittedSupportRouter]) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_v16_target_local_model_set_v1",
            "models": tuple(
                (row.outer_target_id, row.endpoint_model.model_hash)
                for row in routers
            ),
            "evaluation_labels_consumed": False,
        }
    )


def _fit_support_router_task(
    task: tuple[
        tuple[LabelFreeCaseMenu, ...],
        tuple[SupportActionOutcome, ...],
        tuple[SupportCaseClassProfile, ...],
        tuple[str, ...],
        RouterFitConfig,
    ],
) -> FittedSupportRouter:
    """Spawn-safe, CUDA-blind unit of H-local router fitting."""

    menus, outcomes, profiles, candidates, config = task
    return fit_support_router(
        menus,
        outcomes,
        config=config,
        case_profiles=profiles,
        candidate_source_ids=candidates,
    )


def build_support_router_artifact(
    support: ArtifactValue,
    *,
    config: RouterFitConfig | Mapping[str, object] | object | None = None,
) -> ArtifactValue:
    """Fit one target-local router per H and persist its full OOF replay."""

    surface = _require_support_surface(support)
    selected_config = _as_router_config(config)
    tasks = tuple(
        (
            tuple(surface.bundle_for(outer).support_menus),
            rows,
            surface.case_profiles_for(outer),
            surface.bundle_for(outer).candidate_source_ids,
            selected_config,
        )
        for outer, rows in surface.outcomes_by_outer
    )
    runtime = getattr(config, "runtime", None)
    if isinstance(runtime, Mapping):
        pool_plan = dict(science_pool_plan(runtime))
        receipt = execute_science_jobs(
            tasks,
            _fit_support_router_task,
            weights=tuple(
                max(1, len(task[1])) * max(1, len(task[0])) ** 2 for task in tasks
            ),
            workers=int(runtime["science_workers"]),
            threads_per_worker=int(runtime["science_blas_threads_per_worker"]),
        )
        routers = tuple(receipt.values)
        science_execution: Mapping[str, object] = {
            **pool_plan,
            "worker_count_used": receipt.worker_count,
            "batch_ordinals": [list(row) for row in receipt.batch_ordinals],
        }
    else:
        routers = tuple(_fit_support_router_task(task) for task in tasks)
        science_execution = {
            "schema_version": "midogpp_harp_v16_science_pool_receipt_v1",
            "mode": "deterministic_in_process_unit_test_fallback",
            "worker_count_used": 1,
            "cuda_visible_to_workers": False,
            "nested_pools_used": False,
        }
    state = SupportRouterFitState(routers, surface.surface_hash)
    oof_rows = tuple(
        (
            record.prediction.predicted_gain,
            record.prediction.predicted_harm_probability,
            record.prediction.predicted_brier_delta,
            record.prediction.predicted_log_loss_delta,
            record.outcome.bacc_gain,
            float(record.outcome.harmed),
            record.outcome.brier_delta,
            record.outcome.log_loss_delta,
        )
        for router in routers
        for record in router.support_crossfit.records
    )
    if any(
        len(row) != len(_SUPPORT_OOF_ARRAY_COLUMNS) for row in oof_rows
    ):
        raise ProtocolError("HARP v16 support OOF replay columns drifted.")
    oof_values = np.asarray(oof_rows, dtype=np.float64).reshape(
        (-1, len(_SUPPORT_OOF_ARRAY_COLUMNS))
    )
    offsets = [0]
    for router in routers:
        offsets.append(offsets[-1] + len(router.support_crossfit.records))
    model_hash = _model_set_hash(routers)
    body = {
        **state.public_payload(),
        "model_hash": model_hash,
        "router_config": selected_config.public_payload(),
        "oof_array_columns": list(_SUPPORT_OOF_ARRAY_COLUMNS),
        "all_preprocessing_refit_inside_support_case_crossfit": True,
        "hyperparameter_search_performed": False,
        "target_evaluation_features_used_for_fit": False,
        "target_evaluation_labels_used": False,
        "support_oof_case_prediction_count": sum(
            len(router.support_crossfit.case_predictions) for router in routers
        ),
        "support_exact_b_control_count": sum(
            sum(not row.prediction.action_predictions for row in router.support_crossfit.case_predictions)
            for router in routers
        ),
        "science_execution": dict(science_execution),
    }
    return ArtifactValue(
        state=state,
        manifest={**body, "artifact_hash": canonical_hash(body)},
        arrays={
            "support_oof_values": oof_values,
            "outer_oof_offsets": np.asarray(offsets, dtype=np.int64),
        },
    )


def build_support_target_routes(
    bundles: Sequence[SupportTargetMenuBundle],
    fitted: ArtifactValue,
    *,
    target_action_hash: str | None = None,
) -> PrelabelRouteSet:
    """Apply fitted H-local routers to full-test menus before truth opens."""

    if not isinstance(fitted, ArtifactValue) or not isinstance(
        fitted.state, SupportRouterFitState
    ):
        raise ProtocolError("HARP v16 target routing requires fitted support routers.")
    if fitted.manifest.get("state_hash") != fitted.state.state_hash:
        raise ProtocolError("HARP v16 fitted router artifact drifted.")
    return build_support_prelabel_route_set(
        bundles,
        fitted.state.routers,
        target_action_hash=target_action_hash,
    )


def report_support_router_artifact(fitted: ArtifactValue) -> Mapping[str, object]:
    """Expose a compact JSON-safe support fit/admission report."""

    if not isinstance(fitted, ArtifactValue) or not isinstance(
        fitted.state, SupportRouterFitState
    ):
        raise ProtocolError("HARP v16 router report requires a fitted artifact.")
    state = fitted.state
    report = {
        "schema_version": "midogpp_harp_v16_support_router_report_v1",
        "support_surface_hash": state.support_surface_hash,
        "state_hash": state.state_hash,
        "model_hash": fitted.manifest.get("model_hash"),
        "outer_targets": [row.outer_target_id for row in state.routers],
        "per_outer": [
            {
                "outer_target_id": row.outer_target_id,
                "support_case_count": len(row.support_case_ids),
                "router_hash": row.router_hash,
                "model_hash": row.endpoint_model.model_hash,
                "support_crossfit_hash": row.support_crossfit.result_hash,
                "admission": row.admission.public_payload(),
            }
            for row in state.routers
        ],
        "support_labels_consumed": True,
        "evaluation_labels_consumed": False,
    }
    return report


__all__ = (
    "SupportOutcomeSurface",
    "SupportRouterFitState",
    "build_support_outcome_artifact",
    "build_support_router_artifact",
    "build_support_target_routes",
    "report_support_router_artifact",
)
