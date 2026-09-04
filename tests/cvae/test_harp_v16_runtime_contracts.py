from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v16 import (
    ActionFamily,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SurfaceRole,
)
from midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v16.contracts import (
    float32_probability_hex,
)
from midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v16.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.cvae.runtime.harp_v16_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    array_bytes_sha256,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.gpu_surface import (
    _validate_support_binding,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.physical import (
    _support_binding,
    build_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.physical_contracts import (
    StagedFrames,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.support_compatibility import (
    _context_index,
)
from midogpp_thesis.cvae.runtime.harp_v16_execution.support_target_adapter import (
    SupportTargetMenuBundle,
    route_target_bundle,
)


SUPPORT_ROLE = "target_train_support"
TARGET_ROLE = "target_test_evaluation"


def test_v16_retains_the_bounded_workstation_execution_topology() -> None:
    plan = build_physical_plan()

    assert plan["classifier_task_count"] == 81
    assert plan["persistent_gpu_workers"] == 2
    assert plan["max_inflight_source_tasks"] == 4
    assert plan["classifier_workers"] == 4
    assert plan["classifier_blas_threads_per_worker"] == 3
    assert plan["max_inflight_classifier_tasks"] == 8
    assert plan["transport_dtype"] == "float32"
    assert plan["reduction_dtype"] == "float64"
    assert plan["classifier_fit_reused_across_support_and_target"] is True


def _physical_support_binding(tmp_path):
    frame_path = (tmp_path / "frames.npy").resolve()
    values = np.zeros((2 * len(CENTERS), COMMON_OUTPUT_DIM), dtype=np.float32)
    np.save(frame_path, values, allow_pickle=False)
    contexts = {}
    samples = {}
    cases = {}
    rows = []
    cursor = 0
    for role in (SUPPORT_ROLE, TARGET_ROLE):
        for center in CENTERS:
            sample_id = f"{role}-H{center}-sample"
            case_id = f"{role}-H{center}-case"
            contexts[(role, center)] = (cursor, cursor + 1)
            samples[(role, center)] = (sample_id,)
            cases[(role, center)] = (case_id,)
            rows.append(
                SimpleNamespace(
                    split_role=role,
                    center=center,
                    case_id=case_id,
                )
            )
            cursor += 1
    frames = StagedFrames(
        path=frame_path,
        receipt_path=(tmp_path / "frame-receipt.json").resolve(),
        contexts=contexts,
        sample_ids=samples,
        case_ids=cases,
        sha256=sha256_file(frame_path),
        provenance_hash="a" * 64,
        receipt_hash="b" * 64,
        receipt_sha256="c" * 64,
    )
    config = SimpleNamespace(
        config_hash="d" * 64,
        protocol={"feature_backbone": "Virchow2_3840"},
        expected_hashes={
            "development_manifest_sha256": "e" * 64,
            "evaluation_manifest_sha256": "f" * 64,
        },
    )
    cache = SimpleNamespace(
        rows=tuple(rows),
        cache_hash="1" * 64,
        content_sha256="2" * 64,
    )
    return _support_binding(
        config,
        cache,
        frames=frames,
        development_role=SUPPORT_ROLE,
        evaluation_role=TARGET_ROLE,
    )


def _compatibility_payload(binding):
    replicas = []
    context_rows = tuple(binding["contexts"])
    case_by_role_center = {
        (str(row["role"]), str(row["center"])): tuple(row["case_ids"])
        for row in context_rows
    }
    for source in CENTERS:
        for seed in TRAINING_SEEDS:
            contexts = []
            for role in (SUPPORT_ROLE, TARGET_ROLE):
                for query in CENTERS:
                    cases = case_by_role_center[(role, query)]
                    contexts.append(
                        {
                            "role": role,
                            "query_center": query,
                            "case_order": list(cases),
                            "per_case_energy_float32": [float(seed)],
                            "case_count": len(cases),
                            "exact_nelbo": False,
                            "labels_consumed": False,
                        }
                    )
            replicas.append(
                {
                    "source_center": source,
                    "training_seed": seed,
                    "contexts": contexts,
                }
            )
    body = {
        "schema_version": "midogpp_harp_v16_role_qualified_compatibility_surface_v2",
        "support_binding": binding,
        "support_binding_hash": binding["support_binding_hash"],
        "training_seeds": list(TRAINING_SEEDS),
        "replicas": replicas,
        "all_replicas_used_without_selection": True,
        "computed_while_expert_resident": True,
        "exact_nelbo": False,
        "labels_consumed": False,
        "evaluation_labels_consumed": False,
    }
    return {**body, "compatibility_hash": canonical_hash(body)}


def test_physical_support_binding_round_trips_through_both_consumers(tmp_path) -> None:
    binding = _physical_support_binding(tmp_path)

    assert binding["support_role"] == SUPPORT_ROLE
    assert "source_role" not in binding
    assert _validate_support_binding(binding)["support_role"] == SUPPORT_ROLE

    _contexts, support_role, target_role = _context_index(
        _compatibility_payload(binding)
    )
    assert (support_role, target_role) == (SUPPORT_ROLE, TARGET_ROLE)


def test_stale_source_role_alias_is_not_accepted_by_v16(tmp_path) -> None:
    binding = dict(_physical_support_binding(tmp_path))
    binding.pop("support_binding_hash")
    binding["source_role"] = binding.pop("support_role")
    binding["support_binding_hash"] = canonical_hash(binding)

    with pytest.raises(ProtocolError, match="label-free support binding drifted"):
        _validate_support_binding(binding)


def _target_route_bundle() -> SupportTargetMenuBundle:
    outer = CENTERS[0]
    support_case = "support-case"
    target_case = "target-case"
    support_samples = ("support-sample-0", "support-sample-1")
    target_samples = ("target-sample-0", "target-sample-1")
    support_baseline = np.asarray((0.3, 0.7), dtype=np.float32)
    support_uniform = np.asarray((0.6, 0.4), dtype=np.float32)
    target_baseline = np.asarray((0.2, 0.7), dtype=np.float32)
    target_uniform = np.asarray((0.8, 0.1), dtype=np.float32)

    def block(
        role: str,
        kind: ActionKind,
        sample_ids: tuple[str, ...],
        case_id: str,
        probabilities: np.ndarray,
    ) -> LabelFreeActionBlock:
        return LabelFreeActionBlock(
            surface_role=role,
            outer_target_id=outer,
            query_center_id=outer,
            action_kind=kind,
            selected_source_id=None,
            sample_ids=sample_ids,
            case_ids=(case_id,) * len(sample_ids),
            probabilities=probabilities,
        )

    physical = LabelFreeOuterMenu(
        outer_target_id=outer,
        blocks=tuple(
            sorted(
                (
                    block(
                        "support",
                        ActionKind.B,
                        support_samples,
                        support_case,
                        support_baseline,
                    ),
                    block(
                        "support",
                        ActionKind.U,
                        support_samples,
                        support_case,
                        support_uniform,
                    ),
                    block(
                        "target",
                        ActionKind.B,
                        target_samples,
                        target_case,
                        target_baseline,
                    ),
                    block(
                        "target",
                        ActionKind.U,
                        target_samples,
                        target_case,
                        target_uniform,
                    ),
                ),
                key=lambda row: row.key,
            )
        ),
        lineage={"fixture": "target-route-contract"},
    )
    baseline_hex = float32_probability_hex(
        tuple(float(value) for value in target_baseline)
    )
    routed_action = LabelFreeAction(
        outer_target_id=outer,
        case_id=target_case,
        surface_role=SurfaceRole.TARGET_EVALUATION,
        action_id="U:D01",
        family=ActionFamily.U,
        direction=Direction.D01,
        candidate_source_id=None,
        feature_names=("proxy_signal",),
        feature_values=(1.0,),
        baseline_probability_hex=baseline_hex,
        action_probability_hex=float32_probability_hex((0.8, 0.7)),
    )
    support_menu = LabelFreeCaseMenu(
        outer_target_id=outer,
        case_id=support_case,
        surface_role=SurfaceRole.TARGET_TRAIN_SUPPORT,
        baseline_probability_hex=float32_probability_hex(
            tuple(float(value) for value in support_baseline)
        ),
        actions=(),
    )
    target_menu = LabelFreeCaseMenu(
        outer_target_id=outer,
        case_id=target_case,
        surface_role=SurfaceRole.TARGET_EVALUATION,
        baseline_probability_hex=baseline_hex,
        actions=(routed_action,),
    )
    return SupportTargetMenuBundle(
        physical_menu=physical,
        candidate_source_ids=tuple(center for center in CENTERS if center != outer),
        action_identity_hash="3" * 64,
        feature_schema_hash="4" * 64,
        support_menus=(support_menu,),
        target_menus=(target_menu,),
        support_case_samples=((support_case, support_samples),),
        target_case_samples=((target_case, target_samples),),
        support_menu_hash="5" * 64,
        target_menu_hash="6" * 64,
        bundle_hash="7" * 64,
    )


def _target_router(bundle: SupportTargetMenuBundle, *, route: bool) -> SimpleNamespace:
    def decide(menu: LabelFreeCaseMenu) -> SimpleNamespace:
        action = menu.actions[0]
        selected_action_id = action.action_id if route else "B"
        probability_hex = (
            action.action_probability_hex if route else menu.baseline_probability_hex
        )
        return SimpleNamespace(
            selected_action_id=selected_action_id,
            probability_hex=probability_hex,
            exact_b_fallback=not route,
            reason=(
                "ROUTED_SUPPORT_CERTIFIED_HIERARCHICAL_EXACT_ACTION"
                if route
                else "EXACT_B_SUPPORT_POLICY_NOT_ADMITTED"
            ),
            public_payload=lambda: {
                "selected_action_id": selected_action_id,
                "probability_hex": list(probability_hex),
                "exact_b_fallback": not route,
            },
        )

    return SimpleNamespace(
        outer_target_id=bundle.outer_target_id,
        router_hash="8" * 64,
        admission=SimpleNamespace(admitted=route),
        endpoint_model=SimpleNamespace(is_null=False),
        route=decide,
    )


def test_route_target_bundle_preserves_exact_baseline_bytes_and_hash() -> None:
    bundle = _target_route_bundle()
    router = _target_router(bundle, route=False)

    routed = route_target_bundle(bundle, router)
    repeated = route_target_bundle(bundle, router)

    assert len(routed) == 1
    case = routed[0]
    expected_baseline = bundle.physical_menu.target_block(
        ActionKind.B
    ).probabilities
    expected_uniform = bundle.physical_menu.target_block(
        ActionKind.U
    ).probabilities
    assert case.selected_kind is ActionKind.B
    assert case.baseline_probabilities.tobytes() == expected_baseline.tobytes()
    assert case.uniform_probabilities.tobytes() == expected_uniform.tobytes()
    assert case.selected_probabilities.tobytes() == expected_baseline.tobytes()
    assert case.routed_probabilities.tobytes() == expected_baseline.tobytes()
    assert array_bytes_sha256(case.baseline_probabilities) == array_bytes_sha256(
        expected_baseline
    )
    assert case.decision_hash == repeated[0].decision_hash


def test_route_target_bundle_preserves_exact_routed_action_and_controls() -> None:
    bundle = _target_route_bundle()
    router = _target_router(bundle, route=True)

    case = route_target_bundle(bundle, router)[0]
    action = bundle.target_menus[0].actions[0]
    expected_selected = np.frombuffer(
        b"".join(bytes.fromhex(value) for value in action.action_probability_hex),
        dtype="<f4",
    )

    assert case.selected_kind is ActionKind.U
    assert case.direction == "D01"
    assert case.component_action_ids == ("U:D01",)
    assert case.component_weights == (1.0,)
    assert case.selected_probabilities.tobytes() == expected_selected.tobytes()
    assert case.routed_probabilities.tobytes() == expected_selected.tobytes()
    assert case.component_probabilities[0].tobytes() == expected_selected.tobytes()
    assert case.baseline_probabilities.tobytes() == bundle.physical_menu.target_block(
        ActionKind.B
    ).probabilities.tobytes()
    assert case.uniform_probabilities.tobytes() == bundle.physical_menu.target_block(
        ActionKind.U
    ).probabilities.tobytes()
    assert array_bytes_sha256(case.selected_probabilities) == array_bytes_sha256(
        expected_selected
    )
