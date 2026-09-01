"""Production orchestration for the fenced HARP v6 terminal diagnostic.

The class in this module owns phase-to-service wiring only.  Physical
generation, label-free compatibility, source-label joining, artifact
serialization, model fitting, learnability admission, target opportunity
filtering, routing composition, terminal evaluation, and resource control live
behind dedicated module boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ...protocol import ProtocolError
from ...routing.compatibility_conditioned_directional_router import (
    select_baseline_anchored_route,
)
from .compatibility_adapter import (
    build_compatibility_artifact,
    compatibility_state_from_artifact,
)
from .contracts import (
    ArtifactValue,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    TerminalEvaluation,
)
from .model_adapter import (
    build_source_only_admission,
    fit_outer_routers,
    predict_target_evidence,
)
from .physical import (
    build_physical_plan,
    materialize_physical_outer_menus,
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
from .workstation import inspect_harp_v6_workstation


class HarpV6ProductionPipeline:
    """Concrete workstation pipeline over the seven fenced HARP v6 inputs."""

    def __init__(self, *, development_role: str, evaluation_role: str) -> None:
        self._development_role = str(development_role)
        self._evaluation_role = str(evaluation_role)
        self._last_menus: tuple[LabelFreeOuterMenu, ...] = ()

    def preflight(self, config: object, cache: object) -> Mapping[str, object]:
        physical = validate_physical_inputs(config, cache)
        validate_model_config(config)
        science = dict(science_pool_plan(getattr(config, "runtime")))
        live = dict(inspect_harp_v6_workstation(getattr(config, "runtime")))
        return {
            **live,
            "schema_version": "midogpp_harp_v6_workstation_preflight_v3",
            "physical_input_receipt": physical.public_payload(),
            "physical_plan": build_physical_plan(),
            "science_pool_plan": science,
            "compatibility_computed_while_expert_resident": True,
            "policy_hyperparameters_frozen_preexecution": True,
        }

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
            select_fn=select_baseline_anchored_route,
        )

    def evaluate_terminal(
        self,
        routes: PrelabelRouteSet,
        evaluation_truth: object,
        *,
        config: object,
    ) -> TerminalEvaluation:
        del config
        if not isinstance(evaluation_truth, Mapping):
            raise ProtocolError("HARP v6 evaluation truth must be role-scoped.")
        if not self._last_menus:
            raise ProtocolError(
                "HARP v6 terminal evaluation lacks sealed physical menus."
            )
        return evaluate_terminal_routes(
            routes,
            evaluation_truth,
            menus=self._last_menus,
        )


__all__ = ("HarpV6ProductionPipeline",)
