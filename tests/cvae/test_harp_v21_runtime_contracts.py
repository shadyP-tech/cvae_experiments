from __future__ import annotations

import ast
from pathlib import Path
import struct

import numpy as np

from midogpp_thesis.cvae.runtime.harp_v21_execution.contracts import (
    ActionKind,
    PrelabelRouteSet,
    RoutedCase,
    reconstruct_selected_probability_blend,
    reconstruct_shrunk_probability_blend,
)
from midogpp_thesis.cvae.runtime.harp_v21_execution.hash_contracts import (
    runtime_hash_contract_payload,
)
from midogpp_thesis.cvae.runtime.harp_v21_execution.physical import (
    build_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v21_execution.science_pool import (
    science_pool_plan,
)
from midogpp_thesis.cvae.runtime.harp_v21_execution.stores import (
    read_prelabel_routes,
    write_prelabel_routes,
)
from midogpp_thesis.cvae.runtime.harp_v21_execution.terminal import (
    _terminal_case_statuses,
)


def _scalar_branchwise(
    baseline: np.ndarray,
    components: tuple[np.ndarray, ...],
    action_ids: tuple[str, ...],
    *,
    k: int,
    mixing_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected_hex: list[bytes] = []
    final_hex: list[bytes] = []
    for ordinal, baseline_value in enumerate(baseline):
        suffix = ":D01" if baseline_value < np.float32(0.5) else ":D10"
        active = tuple(
            component
            for component, action_id in zip(components, action_ids, strict=True)
            if action_id.endswith(suffix)
        )
        if all(component[ordinal].tobytes() == baseline[ordinal].tobytes() for component in active):
            selected_hex.append(baseline[ordinal].tobytes())
            final_hex.append(baseline[ordinal].tobytes())
            continue
        accumulator = 0.0
        for component in active:
            accumulator += (1.0 / float(k)) * float(component[ordinal])
        selected_cell = struct.pack("<f", accumulator)
        selected_value = struct.unpack("<f", selected_cell)[0]
        selected_hex.append(selected_cell)
        final_hex.append(
            struct.pack(
                "<f",
                (1.0 - mixing_lambda) * float(baseline_value)
                + mixing_lambda * selected_value,
            )
        )
    return (
        np.frombuffer(b"".join(selected_hex), dtype="<f4").copy(),
        np.frombuffer(b"".join(final_hex), dtype="<f4").copy(),
    )


def test_v21_preserves_the_bounded_81_task_810_fit_workstation_plan() -> None:
    plan = build_physical_plan()

    assert plan["source_train_context_count"] == 9
    assert plan["target_context_count"] == 9
    assert plan["candidate_count_per_context"] == 8
    assert plan["stream_job_count"] == 27
    assert plan["classifier_task_count"] == 81
    assert plan["classifier_fit_count"] == 810
    assert plan["persistent_gpu_workers"] == 2
    assert plan["classifier_workers"] == 4
    assert plan["classifier_blas_threads_per_worker"] == 3
    assert plan["transport_dtype"] == "float32"
    assert plan["reduction_dtype"] == "float64"
    assert plan["soft_arm_gpu_task_count"] == 0
    assert plan["soft_arm_classifier_fit_count"] == 0
    assert plan["classifier_fit_reused_across_support_and_target"] is True


def test_science_pool_is_spawned_cuda_blind_and_phase_disjoint() -> None:
    plan = science_pool_plan(
        {
            "science_workers": 4,
            "science_blas_threads_per_worker": 1,
            "multiprocessing_start_method": "spawn",
        }
    )

    assert plan["workers"] == 4
    assert plan["blas_threads_per_worker"] == 1
    assert plan["cuda_visible_to_workers"] is False
    assert plan["nested_pools_used"] is False


def test_vectorized_branchwise_soft_topk_matches_scalar_two_round_contract() -> None:
    baseline = np.asarray((0.2, 0.4, 0.5, 0.8, 0.49, 0.51), dtype=np.float32)
    components = (
        np.asarray((0.61, 0.63, 0.5, 0.8, 0.71, 0.51), dtype=np.float32),
        np.asarray((0.79, 0.67, 0.5, 0.8, 0.69, 0.51), dtype=np.float32),
        np.asarray((0.2, 0.4, 0.31, 0.21, 0.49, 0.29), dtype=np.float32),
        np.asarray((0.2, 0.4, 0.37, 0.33, 0.49, 0.35), dtype=np.float32),
    )
    action_ids = (
        "HXE:1:D01",
        "HXE:2:D01",
        "HXE:1:D10",
        "HXE:2:D10",
    )
    weights = (0.5, 0.5, 0.5, 0.5)

    selected = reconstruct_selected_probability_blend(
        components,
        weights,
        baseline_probabilities=baseline,
        component_action_ids=action_ids,
    )
    routed = reconstruct_shrunk_probability_blend(baseline, selected, 0.75)
    scalar_selected, scalar_routed = _scalar_branchwise(
        baseline, components, action_ids, k=2, mixing_lambda=0.75
    )

    assert selected.tobytes() == scalar_selected.tobytes()
    assert routed.tobytes() == scalar_routed.tobytes()


def test_single_active_branch_accepts_multiple_topk_components_and_copies_b() -> None:
    baseline = np.asarray((0.1, 0.2, 0.3), dtype=np.float32)
    components = (
        np.asarray((0.7, 0.2, 0.8), dtype=np.float32),
        np.asarray((0.9, 0.2, 0.6), dtype=np.float32),
    )
    action_ids = ("HXE:1:D01", "HXE:2:D01")

    selected = reconstruct_selected_probability_blend(
        components,
        (0.5, 0.5),
        baseline_probabilities=baseline,
        component_action_ids=action_ids,
    )
    scalar, _ = _scalar_branchwise(
        baseline, components, action_ids, k=2, mixing_lambda=1.0
    )

    assert selected.tobytes() == scalar.tobytes()
    assert selected[1].tobytes() == baseline[1].tobytes()


def test_exact_b_and_multi_component_recipes_round_trip(tmp_path: Path) -> None:
    baseline = np.asarray((0.2, 0.8), dtype=np.float32)
    uniform = np.asarray((0.6, 0.4), dtype=np.float32)
    d01 = (
        np.asarray((0.6, 0.8), dtype=np.float32),
        np.asarray((0.8, 0.8), dtype=np.float32),
    )
    d10 = (
        np.asarray((0.2, 0.4), dtype=np.float32),
        np.asarray((0.2, 0.2), dtype=np.float32),
    )
    ids = ("HXE:1:D01", "HXE:2:D01", "HXE:1:D10", "HXE:2:D10")
    components = (*d01, *d10)
    weights = (0.5, 0.5, 0.5, 0.5)
    selected = reconstruct_selected_probability_blend(
        components,
        weights,
        baseline_probabilities=baseline,
        component_action_ids=ids,
    )
    routed = reconstruct_shrunk_probability_blend(baseline, selected, 0.5)
    exact_b = RoutedCase(
        outer_target_id="0",
        case_id="a",
        sample_ids=("a0", "a1"),
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="EXACT_B_FALLBACK",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=baseline,
        routed_probabilities=baseline,
    )
    soft = RoutedCase(
        outer_target_id="0",
        case_id="b",
        sample_ids=("b0", "b1"),
        selected_kind=ActionKind.SOFT_TOPK_PROBABILITY_BLEND,
        selected_source_id=None,
        reason="ROUTED_POOLED_SOURCE_POLICY",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=selected,
        routed_probabilities=routed,
        direction="MIXED",
        shrinkage=0.5,
        component_action_ids=ids,
        component_weights=weights,
        component_probabilities=components,
        decision_payload={
            "selection_status": "ROUTE_SELECTED",
            "probability_status": "CHANGED",
            "prediction_status": "CHANGED",
            "utility_status": "NOT_OPENED",
        },
    )
    routes = PrelabelRouteSet(
        cases=(exact_b, soft),
        policy_hash="1" * 64,
        model_hash="2" * 64,
        target_action_hash="3" * 64,
    )

    write_prelabel_routes(tmp_path / "routes", routes)
    loaded = read_prelabel_routes(tmp_path / "routes")

    assert loaded.route_hash == routes.route_hash
    assert loaded.cases[0].recipe_kind == "EXACT_B"
    assert loaded.cases[0].routed_probabilities.tobytes() == baseline.tobytes()
    assert loaded.cases[1].recipe_kind == "SOFT_TOPK_PROBABILITY_BLEND"
    assert loaded.cases[1].recipe_hash == soft.recipe_hash
    assert len(loaded.cases[1].component_probabilities) == 4


def test_terminal_case_statuses_separate_selection_probability_prediction_and_utility() -> None:
    baseline = np.asarray((0.2, 0.8), dtype=np.float32)
    uniform = np.asarray((0.5, 0.5), dtype=np.float32)
    components = (
        np.asarray((0.8, 0.8), dtype=np.float32),
        np.asarray((0.2, 0.2), dtype=np.float32),
    )
    action_ids = ("HXE:1:D01", "HXE:1:D10")
    selected = reconstruct_selected_probability_blend(
        components,
        (1.0, 1.0),
        baseline_probabilities=baseline,
        component_action_ids=action_ids,
    )
    routed = RoutedCase(
        outer_target_id="0",
        case_id="a",
        sample_ids=("a0", "a1"),
        selected_kind=ActionKind.SOFT_TOPK_PROBABILITY_BLEND,
        selected_source_id=None,
        reason="ROUTED_POOLED_SOURCE_POLICY",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=selected,
        routed_probabilities=selected,
        direction="MIXED",
        shrinkage=1.0,
        component_action_ids=action_ids,
        component_weights=(1.0, 1.0),
        component_probabilities=components,
        decision_payload={"donor_entropy": 0.75},
    )
    fallback = RoutedCase(
        outer_target_id="0",
        case_id="b",
        sample_ids=("b0", "b1"),
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="EXACT_B_FALLBACK",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=baseline,
        routed_probabilities=baseline,
    )
    routes = PrelabelRouteSet(
        cases=(routed, fallback),
        policy_hash="4" * 64,
        model_hash="5" * 64,
        target_action_hash="6" * 64,
    )
    truth = {
        ("0", "a", "a0"): 1,
        ("0", "a", "a1"): 0,
        ("0", "b", "b0"): 0,
        ("0", "b", "b1"): 1,
    }

    rows = _terminal_case_statuses(routes, truth)

    assert rows[0]["route_selected"] is True
    assert rows[0]["probability_changed"] is True
    assert rows[0]["prediction_changed"] is True
    assert rows[0]["utility_success"] is True
    assert rows[0]["donor_entropy"] == 0.75
    assert rows[1]["route_selected"] is False
    assert rows[1]["probability_changed"] is False
    assert rows[1]["prediction_changed"] is False
    assert rows[1]["utility_success"] is False


def test_runtime_is_predecessor_free_and_has_no_late_torch_thread_setter() -> None:
    runtime_root = (
        Path(__file__).resolve().parents[2]
        / "src/midogpp_thesis/cvae/runtime/harp_v21_execution"
    )
    forbidden = ("harp_v17_execution", "pooled_pairwise_selected_policy_router_v17")
    for path in runtime_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert all(token not in text for token in forbidden), path
        tree = ast.parse(text, filename=str(path))
        assert not any(
            isinstance(node, ast.Attribute)
            and node.attr == "set_num_interop_threads"
            for node in ast.walk(tree)
        ), path

    hashes = runtime_hash_contract_payload()
    assert hashes["soft_topk_probability_mixtures_allowed"] is True
    assert hashes["soft_arm_gpu_or_classifier_fits"] == 0


def test_full_embedding_store_is_immutable_and_preserves_all_3840_columns(tmp_path):
    from dataclasses import replace
    from test_harp_v21_support_runtime import _physical_menu
    from midogpp_thesis.cvae.runtime.harp_v21_execution.stores import write_label_free_outer_menu, read_label_free_outer_menu
    from midogpp_thesis.cvae.runtime.harp_v21_execution.support_target_adapter import compile_support_target_menus
    original = _physical_menu("0")
    blocks = {role: np.array(values, copy=True) for role, values in original.patch_features.items()}
    blocks["target"][0, 3839] = 123.25
    menu = replace(original, patch_features=blocks)
    expected = menu.patch_features["target"].tobytes()
    blocks["target"][0, 3839] = -9.
    assert menu.patch_features["target"].tobytes() == expected
    import pytest
    with pytest.raises(ValueError):
        menu.patch_features["target"].setflags(write=True)
    write_label_free_outer_menu(tmp_path, menu)
    restored = read_label_free_outer_menu(tmp_path)
    assert restored.menu_hash == menu.menu_hash
    assert restored.patch_features["target"].tobytes() == expected
    case = compile_support_target_menus(restored).target_menus[0]
    assert isinstance(case.patch_features, np.ndarray)
    assert case.patch_features.dtype == np.dtype("<f4")
    assert case.patch_features.shape == (4, 3840)
    assert case.patch_features[0, 3839] == 123.25
    with pytest.raises(ValueError):
        case.patch_features.setflags(write=True)
