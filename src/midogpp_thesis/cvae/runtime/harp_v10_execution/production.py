"""Production orchestration for the fenced HARP v10 terminal diagnostic.

The class in this module owns phase-to-service wiring only.  Physical
generation, label-free compatibility, source-label joining, artifact
serialization, source-active model fitting, whole-policy admission, target
scoring, exact-top-1 routing, terminal evaluation, and resource control live
behind dedicated module boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from .compatibility_adapter import (
    build_compatibility_artifact,
    compatibility_state_from_artifact,
)
from .contracts import (
    ArtifactValue,
    FrozenRouteReceipt,
    LabelFreeOuterMenu,
    LabelFreeTargetMenu,
    PrelabelRouteSet,
    TerminalEvaluation,
)
from .crossfit_contracts import FoldConditionedSourceSurface
from .crossfit_effective_menus import (
    FoldConditionedEffectiveSurface,
    build_fold_conditioned_effective_surface,
)
from .crossfit_durability import (
    SourceCrossfitSurfaceReceipt,
    persist_source_crossfit_surface,
)
from .crossfit_surface import (
    bind_crossfit_prediction_folds_to_target_menus,
    fold_conditioned_physical_plan,
    materialize_fold_conditioned_source_surface,
)
from .model_adapter import (
    build_source_only_admission,
    fit_outer_routers,
    predict_target_evidence,
)
from .physical import (
    build_physical_plan,
    build_target_only_physical_plan,
    materialize_physical_outer_menus,
    materialize_physical_target_menus,
    validate_physical_inputs,
)
from .production_validation import validate_model_config
from .routing_artifacts import build_prelabel_route_set
from .science_pool import science_pool_plan
from .source_development import build_source_development_artifact
from .source_model_artifacts import (
    build_source_admission_artifact,
    build_source_router_artifact,
)
from .target_action_artifacts import build_complete_target_action_artifact
from .terminal import evaluate_terminal_routes
from .workstation import inspect_harp_v10_workstation


class HarpV10ProductionPipeline:
    """Concrete workstation pipeline over the seven fenced HARP v10 inputs."""

    def __init__(self, *, development_role: str, evaluation_role: str) -> None:
        self._development_role = str(development_role)
        self._evaluation_role = str(evaluation_role)
        self._last_menus: tuple[LabelFreeOuterMenu, ...] = ()
        self._last_source_crossfit: FoldConditionedSourceSurface | None = None
        self._last_source_crossfit_effective: FoldConditionedEffectiveSurface | None = None
        self._last_source_crossfit_receipt: SourceCrossfitSurfaceReceipt | None = None

    def preflight(self, config: object, cache: object) -> Mapping[str, object]:
        physical = validate_physical_inputs(config, cache)
        validate_model_config(config)
        science = dict(science_pool_plan(getattr(config, "runtime")))
        live = dict(inspect_harp_v10_workstation(getattr(config, "runtime")))
        return {
            **live,
            "schema_version": "midogpp_harp_v10_workstation_preflight_v3",
            "physical_input_receipt": physical.public_payload(),
            "physical_plan": build_physical_plan(),
            "target_only_physical_plan": build_target_only_physical_plan(),
            "source_crossfit_physical_plan": dict(
                fold_conditioned_physical_plan(
                    tuple(str(value) for value in config.protocol["centers"])
                )
            ),
            "science_pool_plan": science,
            "compatibility_computed_while_expert_resident": True,
            "regularization_hyperparameters_predeclared_fixed_preexecution": True,
            "regularization_hyperparameter_selection_performed": False,
            "acceptance_threshold_selected_source_only_inside_nested_lodo": True,
            "source_crossfit_physical_identity_axes": [
                "outer_H",
                "heldout_q",
                "current_query_r",
            ],
            "heldout_q_physically_excluded_before_classifier_fit": True,
        }

    def materialize_label_free_source_crossfit_surface(
        self,
        config: object,
        cache: object,
        *,
        outer_targets: Sequence[str],
        scratch_root: Path,
    ) -> FoldConditionedSourceSurface:
        """Build the H/q/r substrate consumed by nested source-only fitting."""

        surface = materialize_fold_conditioned_source_surface(
            config,
            cache,
            outer_targets=tuple(str(value) for value in outer_targets),
            scratch_root=Path(scratch_root),
            source_role=self._development_role,
            evaluation_role=self._evaluation_role,
        )
        self._last_source_crossfit = surface
        self._last_source_crossfit_receipt = persist_source_crossfit_surface(
            Path(scratch_root) / "source_crossfit_surface", surface
        )
        return surface

    @property
    def last_source_crossfit_surface(self) -> FoldConditionedSourceSurface:
        if self._last_source_crossfit is None:
            raise ProtocolError(
                "HARP v10 source crossfit surface has not been materialized."
            )
        return self._last_source_crossfit

    @property
    def last_source_crossfit_receipt(self) -> SourceCrossfitSurfaceReceipt:
        if self._last_source_crossfit_receipt is None:
            raise ProtocolError(
                "HARP v10 durable source crossfit receipt is unavailable."
            )
        return self._last_source_crossfit_receipt

    def bind_source_crossfit_predictions(
        self,
        menus: Sequence[LabelFreeTargetMenu | LabelFreeOuterMenu],
        surface: FoldConditionedSourceSurface,
    ) -> tuple[LabelFreeOuterMenu, ...]:
        bound = bind_crossfit_prediction_folds_to_target_menus(surface, menus)
        self._last_source_crossfit = surface
        self._last_menus = bound
        return bound

    def build_label_free_source_crossfit_effective_surface(
        self, surface: FoldConditionedSourceSurface
    ) -> FoldConditionedEffectiveSurface:
        effective = build_fold_conditioned_effective_surface(surface)
        self._last_source_crossfit = surface
        self._last_source_crossfit_effective = effective
        return effective

    def materialize_label_free_target_menus(
        self,
        config: object,
        cache: object,
        *,
        outer_targets: Sequence[str],
        scratch_root: Path,
    ) -> tuple[LabelFreeTargetMenu, ...]:
        """Build only target C-{H}; source blocks come from H/q/r crossfit."""

        return materialize_physical_target_menus(
            config,
            cache,
            outer_targets=tuple(str(value) for value in outer_targets),
            scratch_root=Path(scratch_root),
            development_role=self._development_role,
            evaluation_role=self._evaluation_role,
        )

    def materialize_label_free_outer_menus(
        self,
        config: object,
        cache: object,
        *,
        outer_targets: Sequence[str],
        scratch_root: Path,
    ) -> Sequence[LabelFreeOuterMenu]:
        menus = materialize_physical_outer_menus(
            config,
            cache,
            outer_targets=tuple(str(value) for value in outer_targets),
            scratch_root=Path(scratch_root),
            development_role=self._development_role,
            evaluation_role=self._evaluation_role,
        )
        self._last_menus = tuple(menus)
        return menus

    def materialize_label_free_support_compatibility(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        cache: object,
        *,
        config: object,
        scratch_root: Path,
    ) -> ArtifactValue:
        self._last_menus = tuple(menus)
        return build_compatibility_artifact(
            menus,
            cache,
            config=config,
            scratch_root=Path(scratch_root),
            development_role=self._development_role,
            evaluation_role=self._evaluation_role,
        )

    def build_development_case_surface(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        compatibility: ArtifactValue,
        development_labels: object,
        *,
        config: object,
    ) -> ArtifactValue:
        self._last_menus = tuple(menus)
        return build_source_development_artifact(
            menus,
            compatibility,
            development_labels,
            config=config,
            compatibility_loader=compatibility_state_from_artifact,
        )

    def fit_source_only_router(
        self,
        development: ArtifactValue,
        compatibility: ArtifactValue,
        *,
        config: object,
    ) -> ArtifactValue:
        return build_source_router_artifact(
            development,
            compatibility,
            config=config,
            compatibility_loader=compatibility_state_from_artifact,
            fit_fn=fit_outer_routers,
        )

    def admit_source_only_router(
        self,
        fitted: ArtifactValue,
        development: ArtifactValue,
        *,
        config: object,
    ) -> ArtifactValue:
        return build_source_admission_artifact(
            fitted,
            development,
            config=config,
            admission_fn=build_source_only_admission,
        )

    def build_complete_target_case_actions(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        compatibility: ArtifactValue,
        fit: ArtifactValue,
        admission: ArtifactValue,
        *,
        config: object,
    ) -> ArtifactValue:
        self._last_menus = tuple(menus)
        return build_complete_target_action_artifact(
            menus,
            compatibility,
            fit,
            admission,
            config=config,
            compatibility_loader=compatibility_state_from_artifact,
            predict_fn=predict_target_evidence,
        )

    def route_case_actions(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        target_actions: ArtifactValue,
        fit: ArtifactValue,
        admission: ArtifactValue,
        *,
        config: object,
    ) -> PrelabelRouteSet:
        self._last_menus = tuple(menus)
        return build_prelabel_route_set(
            menus,
            target_actions,
            fit,
            admission,
            config=config,
        )

    def evaluate_terminal(
        self,
        routes: PrelabelRouteSet,
        evaluation_truth: object,
        *,
        frozen_receipt: FrozenRouteReceipt,
        artifact_root: Path,
        config: object,
    ) -> TerminalEvaluation:
        if not isinstance(evaluation_truth, Mapping):
            raise ProtocolError("HARP v10 evaluation truth must be role-scoped.")
        if not self._last_menus:
            raise ProtocolError(
                "HARP v10 terminal evaluation lacks sealed physical menus."
            )
        return evaluate_terminal_routes(
            routes,
            evaluation_truth,
            menus=self._last_menus,
            frozen_receipt=frozen_receipt,
            artifact_root=Path(artifact_root),
            config_hash=getattr(config, "config_hash", None),
        )


__all__ = ("HarpV10ProductionPipeline",)
