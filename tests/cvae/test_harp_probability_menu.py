from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_probability_menu import (
    DEVELOPMENT_SURFACE,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    HarpActionSpec,
    HarpPredictionCell,
    HarpRouteDecision,
    HarpWorkstationContract,
    build_all_development_actions,
    build_all_target_actions,
    build_development_action_menu,
    build_target_action_menu,
    compose_harp_action,
    route_harp_probability_vector,
    seal_harp_prediction_menu,
)


LINEAGE = {
    "bank_hash": "a" * 16,
    "generation_lock_hash": "b" * 16,
    "source_cache_hash": "c" * 16,
    "frame_hash": "d" * 64,
    "classifier_hash": "e" * 16,
    "scaler_state_hash": "f" * 16,
}


def _probabilities(action_ordinal: int, seed_ordinal: int) -> np.ndarray:
    return np.asarray(
        [
            0.10 + 0.01 * action_ordinal + 0.001 * seed_ordinal,
            0.30 + 0.01 * action_ordinal + 0.001 * seed_ordinal,
            0.70 - 0.005 * action_ordinal - 0.001 * seed_ordinal,
        ],
        dtype=np.float32,
    )


def _sealed_target_menu():
    actions = build_target_action_menu("0")
    cells = []
    for action_ordinal, action in enumerate(actions):
        for seed_ordinal, (training_seed, generation_seed) in enumerate(
            EXACT_NINE_SEED_PAIRS
        ):
            cells.append(
                HarpPredictionCell(
                    action=action,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    row_ids=("row-0", "row-1", "row-2"),
                    case_ids=("case-0", "case-1", "case-1"),
                    probabilities=_probabilities(action_ordinal, seed_ordinal),
                    composition_hash=action.action_hash,
                    **LINEAGE,
                )
            )
    return seal_harp_prediction_menu(actions, cells)


def test_action_menus_enforce_all_role_exclusions() -> None:
    development = build_development_action_menu("0", "1")
    assert len(development) == 9
    assert development[0].is_exact_b
    assert development[1].is_uniform_topup
    assert development[0].source_order == ("2", "3", "5", "6", "7", "8", "9")
    assert {action.selected_source_id for action in development[2:]} == set(
        development[0].source_order
    )
    assert all("0" not in action.source_order for action in development)
    assert all("1" not in action.source_order for action in development)

    target = build_target_action_menu("0")
    assert len(target) == 10
    assert target[0].is_exact_b
    assert target[1].is_uniform_topup
    assert target[0].source_order == ("1", "2", "3", "5", "6", "7", "8", "9")
    assert all("0" not in action.source_order for action in target)

    with pytest.raises(ProtocolError, match="distinct H and q"):
        build_development_action_menu("0", "0")
    with pytest.raises(ProtocolError, match="exclude H and q"):
        HarpActionSpec(
            surface_kind=DEVELOPMENT_SURFACE,
            outer_target_id="0",
            query_center_id="1",
            selected_source_id="1",
        )
    with pytest.raises(ProtocolError, match="q == H"):
        HarpActionSpec(
            surface_kind=TARGET_SURFACE,
            outer_target_id="0",
            query_center_id="1",
        )


def test_global_menus_include_complete_b_u_hxe_universes() -> None:
    development = build_all_development_actions()
    target = build_all_target_actions()
    assert len(development) == 9 * 8 * 9 == 648
    assert len(target) == 9 * 10 == 90
    assert sum(action.is_exact_b for action in development) == 72
    assert sum(action.is_uniform_topup for action in development) == 72
    assert sum(action.is_exact_b for action in target) == 9
    assert sum(action.is_uniform_topup for action in target) == 9


def test_development_u_and_hxe_are_matched_budget_b_is_distinct() -> None:
    menu = build_development_action_menu("0", "1")
    assert menu[0].geometry.base_total_per_class == 1008
    assert menu[0].geometry.final_total_per_class == 1134
    assert menu[0].residual_action is None
    assert menu[1].residual_action is not None
    assert menu[2].residual_action is not None
    assert sum(menu[1].residual_action.topup_counts.values()) == 126
    assert sum(menu[2].residual_action.topup_counts.values()) == 126
    assert set(menu[1].residual_action.topup_counts.values()) == {18}
    for label in (0, 1):
        for source in menu[1].source_order:
            window = menu[1].residual_action.windows_by_class[label][source]
            assert window.base_stop == window.topup_start == 144
            assert window.topup_stop == 162


def test_action_composition_reuses_exact_base_and_disjoint_tail_geometry() -> None:
    menu = build_target_action_menu("0")
    source_blocks = {}
    labels = np.concatenate(
        (np.zeros(270, dtype=np.int64), np.ones(270, dtype=np.int64))
    )
    for ordinal, source in enumerate(menu[0].source_order):
        embeddings = np.arange(1080, dtype=np.float32).reshape(540, 2)
        source_blocks[source] = {
            "embeddings": embeddings + np.float32(ordinal * 2000),
            "labels": labels,
        }

    base = compose_harp_action(
        source_blocks, menu[0], shuffle_seed_by_class={0: 17, 1: 42}
    )
    uniform = compose_harp_action(
        source_blocks, menu[1], shuffle_seed_by_class={0: 17, 1: 42}
    )
    tail = compose_harp_action(
        source_blocks, menu[2], shuffle_seed_by_class={0: 17, 1: 42}
    )
    assert base.embeddings.shape == (2048, 2)
    assert uniform.embeddings.shape == (2304, 2)
    assert tail.embeddings.shape == (2304, 2)
    assert base.total_per_class == 1024
    assert uniform.total_per_class == 1152
    assert tail.total_per_class == 1152
    assert menu[1].residual_action is not None
    assert set(menu[1].residual_action.topup_counts.values()) == {16}
    for label in (0, 1):
        for source in menu[1].source_order:
            windows = menu[1].residual_action.windows_by_class[label][source]
            assert (windows.base_start, windows.base_stop) == (0, 128)
            assert (windows.topup_start, windows.topup_stop) == (128, 144)
    assert base.composition_hash != tail.composition_hash
    assert base.composition_hash != uniform.composition_hash


def test_complete_global_seal_and_exact_nine_are_deterministic() -> None:
    first = _sealed_target_menu()
    second = _sealed_target_menu()
    assert first.status == "SEALED_COMPLETE_LABEL_FREE_HARP_MENU"
    assert first.labels_consumed is False
    assert first.seal_hash == second.seal_hash
    assert first.prediction_store_hash == second.prediction_store_hash
    assert len(first.cells) == 10 * 9
    assert all(cell.probabilities.dtype == np.float32 for cell in first.cells)
    assert all(not cell.probabilities.flags.writeable for cell in first.cells)
    assert all(len(cell.probability_bytes_sha256) == 64 for cell in first.cells)
    assert first.action_for(
        surface_kind=TARGET_SURFACE,
        outer_target_id="0",
        query_center_id="0",
        selected_source_id=None,
    ).action_id == "B"
    assert first.action_for(
        surface_kind=TARGET_SURFACE,
        outer_target_id="0",
        query_center_id="0",
        selected_source_id=None,
        action_id="U",
    ).is_uniform_topup

    exact = first.exact_nine(first.actions[0])
    expected = np.mean(
        np.stack([_probabilities(0, ordinal) for ordinal in range(9)]).astype(
            np.float64
        ),
        axis=0,
        dtype=np.float64,
    )
    assert exact.dtype == np.float64
    assert not exact.flags.writeable
    np.testing.assert_array_equal(exact, expected)


def test_incomplete_menu_cannot_be_routed_or_sealed() -> None:
    complete = _sealed_target_menu()
    with pytest.raises(ProtocolError, match="globally complete"):
        seal_harp_prediction_menu(complete.actions, complete.cells[:-1])
    with pytest.raises(ProtocolError, match="complete prediction-menu seal"):
        route_harp_probability_vector(object(), ())  # type: ignore[arg-type]


def test_routing_blends_only_eligible_rows_and_copies_fallback_bytes() -> None:
    menu = _sealed_target_menu()
    policy_hash = "1" * 16
    decisions = (
        HarpRouteDecision(
            surface_kind=TARGET_SURFACE,
            outer_target_id="0",
            query_center_id="0",
            row_id="row-0",
            case_id="case-0",
            eligible=False,
            selected_source_id=None,
            lambda_value=0.0,
            direction="NO_DISAGREEMENT",
            decision_reason="exact_b_fallback",
            policy_hash=policy_hash,
            prediction_menu_seal_hash=menu.seal_hash,
        ),
        HarpRouteDecision(
            surface_kind=TARGET_SURFACE,
            outer_target_id="0",
            query_center_id="0",
            row_id="row-1",
            case_id="case-1",
            eligible=True,
            selected_source_id="1",
            lambda_value=0.5,
            direction="D01",
            decision_reason="conservative_gain_and_loss_gates_passed",
            policy_hash=policy_hash,
            prediction_menu_seal_hash=menu.seal_hash,
        ),
        HarpRouteDecision(
            surface_kind=TARGET_SURFACE,
            outer_target_id="0",
            query_center_id="0",
            row_id="row-2",
            case_id="case-1",
            eligible=False,
            selected_source_id=None,
            lambda_value=0.0,
            direction="ALL_MARGINS",
            decision_reason="support_gate_failed",
            policy_hash=policy_hash,
            prediction_menu_seal_hash=menu.seal_hash,
        ),
    )
    routed = route_harp_probability_vector(menu, decisions)
    baseline = menu.exact_nine(menu.actions[0])
    reference = menu.exact_nine(menu.actions[1])
    expert = menu.exact_nine(menu.actions[2])

    assert routed.fallback_byte_identity is True
    assert routed.routed_probabilities[[0, 2]].view(np.uint64).tolist() == baseline[
        [0, 2]
    ].view(np.uint64).tolist()
    assert routed.routed_probabilities[1] == (
        np.float64(0.5) * reference[1] + np.float64(0.5) * expert[1]
    )
    np.testing.assert_array_equal(routed.reference_probabilities, reference)
    assert not routed.routed_probabilities.flags.writeable
    routed.assert_valid()


def test_lambda_one_is_byte_exact_physical_hxe_endpoint() -> None:
    menu = _sealed_target_menu()
    policy_hash = "2" * 16
    decisions = tuple(
        HarpRouteDecision(
            surface_kind=TARGET_SURFACE,
            outer_target_id="0",
            query_center_id="0",
            row_id=f"row-{ordinal}",
            case_id="case-0" if ordinal == 0 else "case-1",
            eligible=ordinal == 0,
            selected_source_id="1" if ordinal == 0 else None,
            lambda_value=1.0 if ordinal == 0 else 0.0,
            direction="D01" if ordinal == 0 else "NO_DISAGREEMENT",
            decision_reason="physical_endpoint" if ordinal == 0 else "exact_b_fallback",
            policy_hash=policy_hash,
            prediction_menu_seal_hash=menu.seal_hash,
        )
        for ordinal in range(3)
    )
    routed = route_harp_probability_vector(menu, decisions)
    physical_hxe = menu.exact_nine(menu.actions[2])
    assert routed.routed_probabilities[0].tobytes() == physical_hxe[0].tobytes()


def test_workstation_and_public_types_are_label_free() -> None:
    contract = HarpWorkstationContract()
    assert contract.multiprocessing_start_method == "spawn"
    assert contract.parent_cuda_context_created is False
    assert contract.late_torch_interop_setter_used is False
    assert contract.transport_dtype == "float32"
    assert contract.scientific_reduction_dtype == "float64"
    with pytest.raises(ProtocolError, match="workstation execution"):
        HarpWorkstationContract(parent_cuda_context_created=True)
    with pytest.raises(ProtocolError, match="workstation execution"):
        HarpWorkstationContract(late_torch_interop_setter_used=True)
    with pytest.raises(ProtocolError, match="workstation execution"):
        HarpWorkstationContract(multiprocessing_start_method="fork")

    prediction_fields = {field.name for field in fields(HarpPredictionCell)}
    decision_fields = {field.name for field in fields(HarpRouteDecision)}
    assert "labels" not in prediction_fields
    assert "truth" not in prediction_fields
    assert "labels" not in decision_fields
    assert "truth" not in decision_fields

    package_root = Path(
        "src/midogpp_thesis/cvae/runtime/harp_probability_menu"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package_root.glob("*.py"))
    ).lower()
    assert "set_num_interop_threads" not in source
    assert "import torch" not in source
    assert "cvae.diagnostics" not in source
