"""Production service boundary for the HARP v21 pooled source router.

The physical phase is deliberately shared by source train-q and target Test-H
inference: one classifier fit predicts the concatenated, role-sealed frame and
the resulting bytes are split at the authenticated offset.  Only after those
two role surfaces and the fixed-bank independence proof are durable may the
runner pass center-scoped Train-q labels into the methods below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .action_capacity import build_action_capacity_certificate
from .contracts import (
    ArtifactValue,
    FrozenRouteReceipt,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    TerminalEvaluation,
)
from .physical import (
    build_physical_plan,
    materialize_physical_outer_menus,
    validate_physical_inputs,
)
from .physical_contracts import PhysicalInputReceipt
from .support_compatibility import (
    CaseLocalCompatibilitySurface,
    build_case_local_compatibility_surface,
)
from .support_model_artifacts import (
    SupportRouterFitState,
    build_support_outcome_artifact,
    build_support_router_artifact,
    build_support_target_routes,
)
from .support_target_adapter import (
    SupportTargetMenuBundle,
    compile_support_target_menus,
)
from .terminal import evaluate_terminal_routes
from .workstation import inspect_harp_v21_workstation


def _centers(config: object) -> tuple[str, ...]:
    try:
        values = tuple(str(value) for value in config.protocol["centers"])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ProtocolError("HARP v21 config lacks its target-center universe.") from exc
    if values != tuple(CENTERS):
        raise ProtocolError("HARP v21 target-center universe drifted.")
    return values


def _policy_hash(state: SupportRouterFitState) -> str:
    """Return the identity of the sole pooled policy without re-wrapping it."""

    return state.policy.policy_hash


def _bind_model_artifact(
    value: ArtifactValue, *, config_hash: str, centers: Sequence[str]
) -> ArtifactValue:
    if not isinstance(value.state, SupportRouterFitState):
        raise ProtocolError("HARP v21 fitted model state is untyped.")
    body = dict(value.manifest)
    body.pop("artifact_hash", None)
    if (
        body.get("model_hash") is None
        or body.get("support_surface_hash") != value.state.support_surface_hash
    ):
        raise ProtocolError("HARP v21 fitted model manifest drifted.")
    body.update(
        {
            "config_hash": config_hash,
            "expected_center_ids": list(centers),
            "policy_hash": _policy_hash(value.state),
            "source_train_development_only": True,
            "pooled_policy_count": 1,
            "target_evaluation_features_used_for_fit": False,
            "target_evaluation_labels_used": False,
        }
    )
    return ArtifactValue(
        state=value.state,
        manifest={**body, "artifact_hash": canonical_hash(body)},
        arrays=value.arrays,
    )


def _target_action_artifact(
    bundles: Sequence[SupportTargetMenuBundle],
    fitted: ArtifactValue,
    *,
    config_hash: str,
    centers: Sequence[str],
) -> ArtifactValue:
    if not isinstance(fitted.state, SupportRouterFitState):
        raise ProtocolError("HARP v21 target action construction lacks fitted routers.")
    if not fitted.state.policy.admitted:
        raise ProtocolError("HARP v21 target action construction requires source policy admission.")
    rows = tuple(sorted(bundles, key=lambda row: row.outer_target_id))
    if tuple(row.outer_target_id for row in rows) != tuple(centers):
        raise ProtocolError("HARP v21 target action center inventory drifted.")
    case_rows = tuple(
        (bundle.outer_target_id, menu.case_id, menu.menu_hash)
        for bundle in rows
        for menu in bundle.target_menus
    )
    body = {
        "schema_version": "midogpp_harp_v21_target_action_set_v1",
        "config_hash": config_hash,
        "expected_center_ids": list(centers),
        "model_hash": fitted.manifest["model_hash"],
        "policy_hash": fitted.manifest["policy_hash"],
        "physical_outer_menu_hashes": {
            row.outer_target_id: row.physical_menu.menu_hash for row in rows
        },
        "target_effective_menu_hashes": {
            row.outer_target_id: row.target_menu_hash for row in rows
        },
        "case_menu_rows": [list(row) for row in case_rows],
        "case_menu_payloads": [menu.public_payload() for bundle in rows for menu in bundle.target_menus],
        "label_free_candidate_inputs_persisted_for_replay": True,
        "target_case_count": len(case_rows),
        "physical_component_cache": "B_U_FULL_AND_HXE_D01_D10",
        "soft_topk_probability_blends_allowed": True,
        "action_families": ["B", "U_FULL", "D01_ONLY", "D10_ONLY", "BOTH"],
        "unselected_branch_preserves_exact_B_bytes": True,
        "case_conditional_action_selection": True,
        "all_k_lambda_probability_matrices_persisted": False,
        "zero_additional_classifier_or_gpu_fits_for_soft_arms": True,
        "evaluation_labels_consumed": False,
    }
    target_hash = canonical_hash(body)
    manifest = {**body, "target_action_hash": target_hash}
    return ArtifactValue(
        state=rows,
        manifest={**manifest, "artifact_hash": canonical_hash(manifest)},
        arrays={},
    )


class HarpV21ProductionPipeline:
    """Concrete, workstation-optimized v21 numerical services."""

    def __init__(self, *, development_role: str, evaluation_role: str) -> None:
        if (
            development_role != "harp_source_train_development"
            or evaluation_role != "harp_full_test_evaluation"
        ):
            raise ProtocolError("HARP v21 production roles drifted.")
        self.development_role = development_role
        self.evaluation_role = evaluation_role
        self._input_receipt: PhysicalInputReceipt | None = None
        self._compatibility: CaseLocalCompatibilitySurface | None = None

    @property
    def physical_input_receipt(self) -> PhysicalInputReceipt:
        if self._input_receipt is None:
            raise ProtocolError("HARP v21 physical inputs have not been validated.")
        return self._input_receipt

    @property
    def compatibility_surface(self) -> CaseLocalCompatibilitySurface:
        if self._compatibility is None:
            raise ProtocolError("HARP v21 compatibility surface has not been built.")
        return self._compatibility

    def preflight(self, config: object, cache: object) -> Mapping[str, object]:
        centers = _centers(config)
        live = dict(inspect_harp_v21_workstation(config.runtime))
        inputs = validate_physical_inputs(config, cache)
        self._input_receipt = inputs
        plan = dict(build_physical_plan())
        capacity = dict(build_action_capacity_certificate(centers=centers))
        body = {
            "schema_version": "midogpp_harp_v21_workstation_preflight_v1",
            "status": "PASS",
            "config_hash": config.config_hash,
            "cache_hash": cache.cache_hash,
            "persistent_gpu_workers": 2,
            "gpu_devices": ["cuda:0", "cuda:1"],
            "classifier_workers": 4,
            "classifier_blas_threads_per_worker": 3,
            "science_workers": 4,
            "science_blas_threads_per_worker": 1,
            "probability_transport_dtype": "float32",
            "scientific_reduction_dtype": "float64",
            "physical_expert_weight": 1.0,
            "tf32_enabled": False,
            "amp_enabled": False,
            "parent_cuda_context_created": False,
            "shared_validated_menu_index": True,
            "source_train_target_classifier_task_count": 81,
            "joint_support_target_classifier_task_count": 81,
            "total_classifier_fit_count": 810,
            "physical_layout": "NINE_SOURCE_q_PLUS_NINE_TARGET_H",
            "source_H_q_r_crossfit_used": False,
            "H_q_r_seven_expert_folds_used": False,
            "soft_arm_gpu_task_count": 0,
            "soft_arm_classifier_fit_count": 0,
            "support_independence_attestation_hash": (
                inputs.support_independence.attestation_hash
            ),
            "bank_independence_attestation_hash": (
                inputs.support_independence.attestation_hash
            ),
            "physical_input_receipt": inputs.public_payload(),
            "physical_plan": plan,
            "action_capacity_certificate": capacity,
            "live_workstation": live,
            "labels_consumed": False,
        }
        return MappingProxyType({**body, "preflight_hash": canonical_hash(body)})

    def materialize_label_free_outer_menus(
        self,
        config: object,
        cache: object,
        *,
        outer_targets: Sequence[str],
        scratch_root: Path,
    ) -> tuple[LabelFreeOuterMenu, ...]:
        rows = materialize_physical_outer_menus(
            config,
            cache,
            outer_targets=outer_targets,
            scratch_root=Path(scratch_root),
            development_role=self.development_role,
            evaluation_role=self.evaluation_role,
        )
        self._input_receipt = validate_physical_inputs(config, cache)
        return rows

    def compile_label_free_support_target_menus(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        *,
        scratch_root: Path,
    ) -> tuple[tuple[SupportTargetMenuBundle, ...], ArtifactValue]:
        rows = tuple(menus)
        compatibility = build_case_local_compatibility_surface(
            rows, scratch_root=Path(scratch_root)
        )
        bundles = tuple(
            compile_support_target_menus(
                menu,
                compatibility_features=compatibility.for_outer(
                    menu.outer_target_id
                ),
            )
            for menu in rows
        )
        self._compatibility = compatibility
        return bundles, compatibility.artifact()

    def compile_label_free_source_target_menus(
        self,
        menus: Sequence[LabelFreeOuterMenu],
        *,
        scratch_root: Path,
    ) -> tuple[tuple[SupportTargetMenuBundle, ...], ArtifactValue]:
        return self.compile_label_free_support_target_menus(
            menus, scratch_root=scratch_root
        )

    def build_support_case_surface(
        self,
        bundles: Sequence[SupportTargetMenuBundle],
        support_labels_by_outer: Mapping[str, Sequence[object]],
    ) -> ArtifactValue:
        return build_support_outcome_artifact(bundles, support_labels_by_outer)

    def build_source_train_case_surface(
        self,
        bundles: Sequence[SupportTargetMenuBundle],
        source_labels_by_center: Mapping[str, Sequence[object]],
    ) -> ArtifactValue:
        return build_support_outcome_artifact(bundles, source_labels_by_center)

    def fit_pooled_source_router(
        self, support: ArtifactValue, *, config: object
    ) -> ArtifactValue:
        """Fit exactly one selected-policy router over all source-q cases."""

        fitted = build_support_router_artifact(support, config=config)
        return _bind_model_artifact(
            fitted,
            config_hash=config.config_hash,
            centers=_centers(config),
        )

    def build_complete_target_case_actions(
        self,
        bundles: Sequence[SupportTargetMenuBundle],
        fitted: ArtifactValue,
        *,
        config: object,
    ) -> ArtifactValue:
        return _target_action_artifact(
            bundles,
            fitted,
            config_hash=config.config_hash,
            centers=_centers(config),
        )

    def route_case_actions(
        self,
        bundles: Sequence[SupportTargetMenuBundle],
        fitted: ArtifactValue,
        target_actions: ArtifactValue,
    ) -> PrelabelRouteSet:
        target_hash = target_actions.manifest.get("target_action_hash")
        routes = build_support_target_routes(
            bundles,
            fitted,
            target_action_hash=(None if target_hash is None else str(target_hash)),
        )
        if (
            routes.model_hash != fitted.manifest.get("model_hash")
            or routes.policy_hash != fitted.manifest.get("policy_hash")
            or routes.target_action_hash != target_hash
        ):
            raise ProtocolError("HARP v21 routed case bindings drifted.")
        return routes

    def evaluate_terminal(
        self,
        routes: PrelabelRouteSet,
        evaluation_truth: object,
        *,
        frozen_receipt: FrozenRouteReceipt,
        artifact_root: Path,
        config: object,
        menus: Sequence[LabelFreeOuterMenu],
    ) -> TerminalEvaluation:
        if not isinstance(evaluation_truth, Mapping):
            raise ProtocolError("HARP v21 evaluation truth is not a mapping.")
        return evaluate_terminal_routes(
            routes,
            evaluation_truth,
            menus=menus,
            frozen_receipt=frozen_receipt,
            artifact_root=Path(artifact_root),
            config_hash=config.config_hash,
        )


__all__ = ("HarpV21ProductionPipeline",)
