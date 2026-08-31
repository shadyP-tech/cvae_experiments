from __future__ import annotations

import json
import pickle
import statistics
import struct
from types import MappingProxyType

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import (
    modeling,
    target_action_compiler as compiler,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.runner import (
    _target_action_surface_payload,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_model import (
    HarpActionModelBank,
    HarpLodoFoldAudit,
    HarpOutcomeModel,
    HarpRidgeModel,
    HarpSupportCell,
    HarpTargetAction,
    LAMBDA_GRID,
)
from midogpp_thesis.cvae.routing.harp_action_surface import ACTION_FEATURE_NAMES
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.routing.harp_portfolio import HarpPolicyConfig
from midogpp_thesis.cvae.runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    UNIFORM_ACTION_ID,
    HarpPredictionCell,
    HarpPredictionMenuSeal,
    HarpActionSpec,
    build_all_target_actions,
    route_harp_probability_vector,
    seal_harp_prediction_menu,
)
from midogpp_thesis.cvae.runtime.harp_probability_menu.hashing import raw_array_sha256
from midogpp_thesis.cvae.runtime.harp_probability_menu.indexed import (
    validated_target_menu_view,
)


LINEAGE = {
    "bank_hash": "a" * 16,
    "generation_lock_hash": "b" * 16,
    "source_cache_hash": "c" * 16,
    "frame_hash": "d" * 64,
    "classifier_hash": "e" * 16,
    "scaler_state_hash": "f" * 16,
}


def _all_center_menu() -> HarpPredictionMenuSeal:
    actions = build_all_target_actions()
    cells: list[HarpPredictionCell] = []
    for action_ordinal, action in enumerate(actions):
        center_ordinal = CENTERS.index(action.outer_target_id)
        row_ids = (
            f"sample-z-{action.outer_target_id}",
            f"sample-a-{action.outer_target_id}",
            f"sample-m-{action.outer_target_id}",
        )
        case_ids = (
            f"case-b-{action.outer_target_id}",
            f"case-a-{action.outer_target_id}",
            f"case-a-{action.outer_target_id}",
        )
        for seed_ordinal, (training_seed, generation_seed) in enumerate(
            EXACT_NINE_SEED_PAIRS
        ):
            probabilities = np.asarray(
                [
                    0.36
                    + 0.0011 * action_ordinal
                    + 0.00013 * seed_ordinal,
                    0.64
                    - 0.0009 * action_ordinal
                    - 0.00011 * seed_ordinal,
                    0.49
                    + 0.00017 * center_ordinal
                    + 0.00021 * (action_ordinal % 10)
                    + 0.00003 * seed_ordinal,
                ],
                dtype=np.float32,
            )
            cells.append(
                HarpPredictionCell(
                    action=action,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    row_ids=row_ids,
                    case_ids=case_ids,
                    probabilities=probabilities,
                    composition_hash=action.action_hash,
                    **LINEAGE,
                )
            )
    return seal_harp_prediction_menu(actions, cells)


def _legacy_feature_values(
    baseline_members: np.ndarray,
    expert_members: np.ndarray,
    baseline: float,
    expert: float,
    lam: float,
) -> tuple[float, ...]:
    action = (1.0 - lam) * baseline + lam * expert
    member_actions = (1.0 - lam) * baseline_members + lam * expert_members
    expert_flips = float(
        np.mean((expert_members >= 0.5) != (baseline_members >= 0.5), dtype=np.float64)
    )
    action_flips = float(
        np.mean((member_actions >= 0.5) != (baseline_members >= 0.5), dtype=np.float64)
    )
    dispersion = float(
        statistics.pstdev(
            tuple(float(value) for value in expert_members - baseline_members)
        )
    )
    return (
        baseline,
        expert,
        action,
        abs(baseline - 0.5),
        abs(expert - 0.5),
        abs(action - 0.5),
        expert - baseline,
        abs(expert - baseline),
        action - baseline,
        abs(action - baseline),
        expert_flips,
        action_flips,
        dispersion,
        lam,
    )


def _legacy_direction(baseline: float, action: float) -> str:
    before, after = baseline >= 0.5, action >= 0.5
    if not before and after:
        return "D01"
    if before and not after:
        return "D10"
    return "ALL_MARGINS"


def _legacy_target_actions(menu: HarpPredictionMenuSeal) -> tuple[HarpTargetAction, ...]:
    """Frozen old scalar semantics without its redundant validation scans."""

    cells_by_hash = {
        action.action_hash: tuple(
            cell for cell in menu.cells if cell.action.action_hash == action.action_hash
        )
        for action in menu.actions
    }

    def exact(action) -> np.ndarray:
        stacked = np.stack(
            [cell.probabilities for cell in cells_by_hash[action.action_hash]], axis=0
        ).astype(np.float64, copy=False)
        result = np.ascontiguousarray(
            np.mean(stacked, axis=0, dtype=np.float64), dtype=np.float64
        )
        result.setflags(write=False)
        return result

    output: list[HarpTargetAction] = []
    for center in CENTERS:
        actions = tuple(
            action
            for action in menu.actions
            if action.surface_kind == TARGET_SURFACE
            and action.outer_target_id == center
            and action.query_center_id == center
        )
        baseline_action = next(
            action for action in actions if action.action_id == BASE_ACTION_ID
        )
        reference_action = next(
            action for action in actions if action.action_id == UNIFORM_ACTION_ID
        )
        source_actions = tuple(
            action for action in actions if action.selected_source_id is not None
        )
        fallback_cells = cells_by_hash[baseline_action.action_hash]
        reference_cells = cells_by_hash[reference_action.action_hash]
        fallback = exact(baseline_action)
        reference = exact(reference_action)
        row_ids, case_ids = reference_cells[0].row_ids, reference_cells[0].case_ids
        for ordinal, (sample_id, case_id) in enumerate(
            zip(row_ids, case_ids, strict=True)
        ):
            reference_members = np.asarray(
                [cell.probabilities[ordinal] for cell in reference_cells],
                dtype=np.float64,
            )
            reference_probability = float(reference[ordinal])
            members = []
            for receipt_action in actions:
                values = np.asarray(
                    [
                        cell.probabilities[ordinal]
                        for cell in cells_by_hash[receipt_action.action_hash]
                    ],
                    dtype=np.float32,
                )
                members.append(
                    {
                        "action_hash": receipt_action.action_hash,
                        "member_probability_bytes_sha256": raw_array_sha256(values),
                    }
                )
            receipt = canonical_hash(
                {
                    "schema_version": (
                        "midogpp_harp_stage90_target_exact_nine_receipt_v1"
                    ),
                    "prediction_menu_seal_hash": menu.seal_hash,
                    "outer_target_id": center,
                    "sample_id": sample_id,
                    "case_id": case_id,
                    "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
                    "actions": members,
                    "labels_consumed": False,
                }
            )
            for action in source_actions:
                expert_cells = cells_by_hash[action.action_hash]
                expert_members = np.asarray(
                    [cell.probabilities[ordinal] for cell in expert_cells],
                    dtype=np.float64,
                )
                expert_probability = float(exact(action)[ordinal])
                for lam in LAMBDA_GRID:
                    features = _legacy_feature_values(
                        reference_members,
                        expert_members,
                        reference_probability,
                        expert_probability,
                        lam,
                    )
                    output.append(
                        HarpTargetAction(
                            outer_target_id=center,
                            target_query_id=center,
                            candidate_source_id=str(action.selected_source_id),
                            case_id=case_id,
                            sample_id=sample_id,
                            lambda_value=lam,
                            direction=_legacy_direction(
                                reference_probability, features[2]
                            ),
                            feature_names=ACTION_FEATURE_NAMES,
                            feature_values=features,
                            baseline_probability_bytes=struct.pack(
                                "<d", reference_probability
                            ),
                            operational_fallback_probability_bytes=struct.pack(
                                "<d", float(fallback[ordinal])
                            ),
                            expert_probability=expert_probability,
                            ensemble_size=len(EXACT_NINE_SEED_PAIRS),
                            ensemble_receipt_hash=receipt,
                            prediction_seal_hash=menu.seal_hash,
                            compatibility_shrinkage=1.0,
                        )
                    )
    return tuple(output)


def test_validated_target_view_preserves_seed_order_and_exact_nine_bits() -> None:
    menu = _all_center_menu()
    view = validated_target_menu_view(menu)
    assert view.labels_consumed is False
    assert len(view.index_hash) == 64
    for action in menu.actions:
        cells = view.cells_for(action)
        assert tuple(
            (cell.training_seed, cell.generation_seed) for cell in cells
        ) == EXACT_NINE_SEED_PAIRS
        expected = np.ascontiguousarray(
            np.mean(
                np.stack([cell.probabilities for cell in cells], axis=0).astype(
                    np.float64, copy=False
                ),
                axis=0,
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
        observed = view.exact_nine(action)
        assert observed.tobytes(order="C") == expected.tobytes(order="C")
        assert not observed.flags.writeable


def test_public_router_rejects_internal_prevalidated_view() -> None:
    menu = _all_center_menu()
    view = validated_target_menu_view(menu)
    with pytest.raises(ProtocolError, match="complete prediction-menu seal"):
        route_harp_probability_vector(view, ())  # type: ignore[arg-type]
    probabilities = menu.cells[0].probabilities
    probabilities.setflags(write=True)
    probabilities[0] = np.float32(probabilities[0] + np.float32(0.01))
    with pytest.raises(ProtocolError, match="bytes drifted"):
        route_harp_probability_vector(menu, ())


def test_validated_target_view_is_immutable_nonserializable_and_identity_scoped() -> None:
    menu = _all_center_menu()
    view = validated_target_menu_view(menu)
    action = view.actions_for_center(CENTERS[0])[0]
    with pytest.raises(TypeError):
        view._cells_by_action_hash[action.action_hash] = ()  # type: ignore[index]
    with pytest.raises(AttributeError, match="immutable"):
        view._seal_hash = "0" * 64  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del view._frozen  # type: ignore[misc]
    with pytest.raises(TypeError, match="non-serializable"):
        pickle.dumps(view)

    foreign = HarpActionSpec(
        surface_kind=action.surface_kind,
        outer_target_id=action.outer_target_id,
        query_center_id=action.query_center_id,
        action_id=action.action_id,
    )
    assert foreign.action_hash == action.action_hash
    assert foreign is not action
    with pytest.raises(ProtocolError, match="exact sealed target action member"):
        view.exact_nine(foreign)


@pytest.mark.parametrize(
    "cache_name",
    ("exact", "cells", "center_actions", "lookup", "identities"),
)
def test_final_validation_rejects_internal_cache_replacement(cache_name: str) -> None:
    menu = _all_center_menu()
    view = validated_target_menu_view(menu)
    action = view.actions_for_center(CENTERS[0])[0]
    action_hash = action.action_hash
    if cache_name == "exact":
        values = dict(view._exact_nine_by_action_hash)
        replacement = np.frombuffer(
            np.zeros_like(values[action_hash]).tobytes(order="C"), dtype=np.float64
        )
        values[action_hash] = replacement
        object.__setattr__(
            view, "_exact_nine_by_action_hash", MappingProxyType(values)
        )
    elif cache_name == "cells":
        values = dict(view._cells_by_action_hash)
        values[action_hash] = values[action_hash][:-1]
        object.__setattr__(view, "_cells_by_action_hash", MappingProxyType(values))
    elif cache_name == "center_actions":
        values = dict(view._actions_by_center)
        values[CENTERS[0]] = values[CENTERS[0]][:-1]
        object.__setattr__(view, "_actions_by_center", MappingProxyType(values))
    elif cache_name == "lookup":
        values = dict(view._action_lookup)
        values.pop((action.outer_target_id, action.selected_source_id, action.action_id))
        object.__setattr__(view, "_action_lookup", MappingProxyType(values))
    else:
        values = dict(view._identities_by_action_hash)
        rows, cases = values[action_hash]
        values[action_hash] = (tuple(f"tampered-{row}" for row in rows), cases)
        object.__setattr__(
            view, "_identities_by_action_hash", MappingProxyType(values)
        )

    with pytest.raises(ProtocolError, match="cache|index drifted"):
        view.assert_fully_valid()


def test_compiler_is_byte_equivalent_and_uses_two_full_validations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu = _all_center_menu()
    expected = _legacy_target_actions(menu)
    calls = 0
    original_assert_valid = HarpPredictionMenuSeal.assert_valid

    def counted_assert_valid(self: HarpPredictionMenuSeal) -> None:
        nonlocal calls
        calls += 1
        original_assert_valid(self)

    def forbidden_lookup(*_args, **_kwargs):
        raise AssertionError("optimized compiler used a public linear/revalidating lookup")

    monkeypatch.setattr(HarpPredictionMenuSeal, "assert_valid", counted_assert_valid)
    monkeypatch.setattr(HarpPredictionMenuSeal, "cells_for", forbidden_lookup)
    monkeypatch.setattr(HarpPredictionMenuSeal, "exact_nine", forbidden_lookup)
    observed = compiler.build_target_actions(menu)

    assert calls == 2
    assert len(observed) == len(expected) == len(CENTERS) * 3 * 8 * 4
    assert observed == expected
    expected_payload = _target_action_surface_payload(expected, menu.seal_hash)
    observed_payload = _target_action_surface_payload(observed, menu.seal_hash)
    assert observed_payload == expected_payload
    expected_bytes = (
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    observed_bytes = (
        json.dumps(observed_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert observed_bytes == expected_bytes


def test_compiler_final_validation_rejects_midcall_probability_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu = _all_center_menu()
    original_build = compiler._build_from_view

    def mutate_after_build(view):
        output = original_build(view)
        probabilities = menu.cells[0].probabilities
        probabilities.setflags(write=True)
        probabilities[0] = np.float32(probabilities[0] + np.float32(0.01))
        return output

    monkeypatch.setattr(compiler, "_build_from_view", mutate_after_build)
    with pytest.raises(ProtocolError, match="bytes drifted"):
        compiler.build_target_actions(menu)


def _routing_bank(outer: str) -> HarpActionModelBank:
    candidates = tuple(sorted(set(CENTERS) - {outer}))
    dimension = 1 + len(ACTION_FEATURE_NAMES) + len(candidates)

    def ridge(*, donor: str | None, intercept: float) -> HarpRidgeModel:
        coefficients = np.zeros(dimension, dtype=np.float64)
        coefficients[0] = intercept
        return HarpRidgeModel(
            feature_names=ACTION_FEATURE_NAMES,
            candidate_levels=candidates,
            feature_mean=np.zeros(len(ACTION_FEATURE_NAMES), dtype=np.float64),
            feature_scale=np.ones(len(ACTION_FEATURE_NAMES), dtype=np.float64),
            coefficients=coefficients,
            normal_inverse=np.eye(dimension, dtype=np.float64) * 1.0e-6,
            alpha=0.1,
            training_query_ids=("inner-a", "inner-b", "inner-c"),
            training_source_ids=candidates,
            training_case_ids=("fit-case-a", "fit-case-b"),
            excluded_donor_ids=() if donor is None else (donor,),
        )

    donors = ("donor-a", "donor-b", "donor-c")
    models = []
    for outcome, intercept in (("gain", 0.20), ("brier", -0.10), ("log_loss", -0.10)):
        models.append(
            HarpOutcomeModel(
                outcome=outcome,
                direction="ALL_MARGINS",
                full_model=ridge(donor=None, intercept=intercept),
                delete_donor_models=tuple(
                    (
                        donor,
                        ridge(
                            donor=donor,
                            intercept=intercept + 0.001 * donor_ordinal,
                        ),
                    )
                    for donor_ordinal, donor in enumerate(donors)
                ),
                nested_lodo_audit=(
                    HarpLodoFoldAudit(
                        "audit-heldout",
                        ("inner-a", "inner-b", "inner-c"),
                        candidates,
                        0.1,
                        0.01,
                    ),
                ),
            )
        )
    support = tuple(
        HarpSupportCell(candidate, lam, direction, 4, 16, (0, 1))
        for candidate in candidates
        for lam in LAMBDA_GRID
        for direction in ("D01", "D10", "ALL_MARGINS")
    )
    return HarpActionModelBank(
        outer_target_id=outer,
        feature_names=ACTION_FEATURE_NAMES,
        prediction_seal_hashes=("1" * 64,),
        response_receipt_hashes=("2" * 64,),
        models=tuple(models),
        support_cells=support,
    )


def _vector_bytes(vectors) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            vector.routed_vector_seal_hash,
            vector.baseline_probabilities.tobytes(order="C"),
            vector.reference_probabilities.tobytes(order="C"),
            vector.selected_action_probabilities.tobytes(order="C"),
            vector.routed_probabilities.tobytes(order="C"),
        )
        for vector in vectors
    )


def test_paired_routing_matches_two_independent_role_calls_and_scores_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    menu = _all_center_menu()
    actions = compiler.build_target_actions(menu)
    banks = tuple(_routing_bank(center) for center in CENTERS)
    policy = HarpPolicyConfig(max_leverage=1.0)
    fitted_policy_hash = "3" * 64

    original_score = modeling.score_harp_actions
    calls = 0
    original_menu_assert = HarpPredictionMenuSeal.assert_valid
    full_validations = 0

    def counted_score(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_score(*args, **kwargs)

    def counted_menu_assert(self: HarpPredictionMenuSeal) -> None:
        nonlocal full_validations
        full_validations += 1
        original_menu_assert(self)

    monkeypatch.setattr(modeling, "score_harp_actions", counted_score)
    monkeypatch.setattr(HarpPredictionMenuSeal, "assert_valid", counted_menu_assert)
    predictive = modeling.select_and_route(
        menu,
        banks,
        actions,
        policy=policy,
        fitted_policy_hash=fitted_policy_hash,
    )
    physical = modeling.select_and_route(
        menu,
        banks,
        actions,
        policy=policy,
        fitted_policy_hash=fitted_policy_hash,
        physical_lambda_one_only=True,
    )
    separate_calls = calls
    separate_validations = full_validations

    calls = 0
    full_validations = 0
    paired = modeling.select_and_route_pair(
        menu,
        banks,
        actions,
        policy=policy,
        fitted_policy_hash=fitted_policy_hash,
    )

    assert separate_calls == 2 * len(CENTERS)
    assert separate_validations == 4
    assert calls == len(CENTERS)
    assert full_validations == 2
    assert paired[0] == predictive[0]
    assert paired[2] == physical[0]
    assert _vector_bytes(paired[1]) == _vector_bytes(predictive[1])
    assert _vector_bytes(paired[3]) == _vector_bytes(physical[1])
