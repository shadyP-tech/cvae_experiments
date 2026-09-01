from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v6_execution import workstation
from midogpp_thesis.cvae.runtime.harp_v6_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
    PrelabelRouteSet,
    RoutedCase,
    compose_directional_soft_probability,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.physical import (
    build_physical_plan,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.science_pool import (
    lpt_batches,
    science_pool_plan,
)
from midogpp_thesis.cvae.runtime.harp_v6_execution.stores import (
    read_label_free_outer_menu,
    read_prelabel_routes,
    write_label_free_outer_menu,
    write_prelabel_routes,
)


def test_exact_nine_seed_dispersion_survives_compact_menu_store(
    tmp_path: Path,
) -> None:
    sample_ids = ("s0", "s1")
    case_ids = ("c0", "c0")
    probabilities = np.asarray([0.25, 0.75], dtype=np.float32)
    dispersion = np.asarray([0.01, 0.02], dtype=np.float32)
    blocks = []
    for role, query in (("development", "8"), ("target", "9")):
        for kind, source in (
            (ActionKind.B, None),
            (ActionKind.U, None),
            (ActionKind.HXE, "0"),
        ):
            blocks.append(
                LabelFreeActionBlock(
                    surface_role=role,
                    outer_target_id="9",
                    query_center_id=query,
                    action_kind=kind,
                    selected_source_id=source,
                    sample_ids=sample_ids,
                    case_ids=case_ids,
                    probabilities=probabilities,
                    seed_dispersion=dispersion,
                )
            )
    menu = LabelFreeOuterMenu(
        outer_target_id="9",
        blocks=tuple(sorted(blocks, key=lambda block: block.key)),
        lineage={"test": True},
    )

    write_label_free_outer_menu(tmp_path / "menu", menu)
    restored = read_label_free_outer_menu(tmp_path / "menu")

    assert restored.menu_hash == menu.menu_hash
    assert all(
        block.seed_dispersion.tobytes() == dispersion.tobytes()
        for block in restored.blocks
    )


def _runtime() -> dict[str, object]:
    return {
        "profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "global_parent_blas_threads": 1,
        "classifier_workers": 4,
        "classifier_blas_threads_per_worker": 3,
        "science_workers": 4,
        "science_blas_threads_per_worker": 1,
        "multiprocessing_start_method": "spawn",
        "parent_cuda_context_created": False,
        "late_torch_interop_setter_used": False,
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "memory_mapped_surfaces": True,
        "bounded_inflight_batches_per_gpu": 2,
        "bounded_inflight_classifier_tasks_per_worker": 2,
        "bounded_inflight_science_tasks_per_worker": 1,
        "scratch_root": "/data/local/fixed_bank_harp_router_v6",
    }


def test_v6_plan_separates_gpu_classifier_and_science_pools() -> None:
    plan = build_physical_plan()
    science = science_pool_plan(_runtime())

    assert plan["persistent_gpu_workers"] == 2
    assert plan["max_inflight_source_tasks"] == 4
    assert plan["classifier_workers"] == 4
    assert plan["classifier_blas_threads_per_worker"] == 3
    assert plan["max_inflight_classifier_tasks"] == 8
    assert plan["compatibility_computed_while_expert_resident"] is True
    assert science == {
        "schema_version": "midogpp_harp_v6_science_pool_plan_v1",
        "workers": 4,
        "blas_threads_per_worker": 1,
        "cuda_visible_to_workers": False,
        "nested_pools_used": False,
        "scheduling": "deterministic_lpt_outer_target_direction_batches",
    }


def test_lpt_batching_is_deterministic_and_balanced() -> None:
    first = lpt_batches((10, 9, 8, 7, 2, 1), workers=4)
    second = lpt_batches((10, 9, 8, 7, 2, 1), workers=4)
    assert first == second
    assert sorted(value for batch in first for value in batch) == list(range(6))


def test_directional_soft_route_round_trips_and_off_is_exact_b(tmp_path: Path) -> None:
    baseline = np.asarray([0.2, 0.7, 0.4], dtype=np.float32)
    uniform = np.asarray([0.6, 0.3, 0.8], dtype=np.float32)
    expert_a = np.asarray([0.9, 0.1, 0.7], dtype=np.float32)
    expert_b = np.asarray([0.7, 0.2, 0.6], dtype=np.float32)
    selected, routed = compose_directional_soft_probability(
        baseline,
        (expert_a, expert_b),
        (0.75, 0.25),
        direction="D01",
        shrinkage=0.5,
    )
    active = RoutedCase(
        outer_target_id="9",
        case_id="case-a",
        sample_ids=("a", "b", "c"),
        selected_kind=ActionKind.HXE,
        selected_source_id="0",
        reason="BASELINE_SAFE_PAIRWISE_TOPK",
        baseline_probabilities=baseline,
        uniform_probabilities=uniform,
        selected_probabilities=selected,
        routed_probabilities=routed,
        direction="D01",
        shrinkage=0.5,
        component_action_ids=("HXE:0", "HXE:1"),
        component_weights=(0.75, 0.25),
        component_probabilities=(expert_a, expert_b),
    )
    off = RoutedCase(
        outer_target_id="9",
        case_id="case-b",
        sample_ids=("d",),
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="EXACT_B_FALLBACK",
        baseline_probabilities=baseline[:1],
        uniform_probabilities=uniform[:1],
        selected_probabilities=baseline[:1],
        routed_probabilities=baseline[:1],
    )
    routes = PrelabelRouteSet(
        cases=(active, off),
        policy_hash="a" * 64,
        model_hash="b" * 64,
        target_action_hash="c" * 64,
    )
    write_prelabel_routes(tmp_path / "routes", routes)
    reconstructed = read_prelabel_routes(tmp_path / "routes")

    assert reconstructed.route_hash == routes.route_hash
    assert reconstructed.cases[0].routed_probabilities[1:2].tobytes() == baseline[
        1:2
    ].tobytes()
    assert reconstructed.cases[1].routed_probabilities.tobytes() == baseline[
        :1
    ].tobytes()


def test_live_preflight_requires_global_single_thread_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in workstation._EXPECTED_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(workstation, "_available_cpu_count", lambda: 24)
    monkeypatch.setattr(workstation, "_physical_ram_bytes", lambda: 135 * 1024**3)
    monkeypatch.setattr(
        workstation, "_safe_nearest_existing_parent", lambda _path: tmp_path
    )
    monkeypatch.setattr(workstation.os, "access", lambda *_args: True)
    monkeypatch.setattr(
        workstation.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=7 * 1024**4),
    )
    monkeypatch.setattr(
        workstation,
        "_nvidia_smi_rows",
        lambda: tuple(
            {
                "index": index,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24_576,
                "memory_free_mib": 24_000,
            }
            for index in (0, 1)
        ),
    )
    monkeypatch.setattr(workstation, "_package_versions", lambda: {"numpy": "test"})

    report = workstation.inspect_harp_v6_workstation(_runtime())
    assert report["status"] == "PASS"
    assert report["science_workers"] == 4
    assert report["science_blas_threads_per_worker"] == 1
    assert report["thread_environment"]["OMP_NUM_THREADS"] == "1"

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    with pytest.raises(ProtocolError, match="launch through workspace run"):
        workstation.inspect_harp_v6_workstation(_runtime())


def test_gpu_surface_never_sets_late_torch_interop_threads() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "src/midogpp_thesis/cvae/runtime/harp_v6_execution/gpu_surface.py"
    ).read_text(encoding="utf-8")
    assert "set_num_interop_threads" not in source
    assert "score_variational_compatibility(" in source
    assert "compatibility_computed_while_expert_resident" in source
