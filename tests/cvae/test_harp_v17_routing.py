from __future__ import annotations

import ast
import json
from pathlib import Path
import pickle
import struct

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.pooled_pairwise_selected_policy_router_v17 import (
    AdmissionStatus,
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    RouterFitConfig,
    SealedOOFSelection,
    SupportTruthCapability,
    SurfaceRole,
    build_baseline_composite,
    build_exact_u_composite,
    build_pairwise_comparisons,
    build_soft_topk_composite,
    build_source_only_admission,
    center_stratified_folds,
    fit_feature_transform,
    fit_pairwise_ranker,
    fit_source_router,
    float32_probability_hex,
    nested_source_crossfit,
    route_decision_report,
    route_target_cases,
    score_selected_composite,
)


def _action(
    role: SurfaceRole,
    center: str,
    case: str,
    arm: str,
    direction: Direction,
    donor: str | None,
    baseline: tuple[float, ...],
    probability: tuple[float, ...],
    feature: float,
) -> LabelFreeAction:
    samples = tuple(f"{case}:s{index}" for index in range(len(baseline)))
    return LabelFreeAction(
        surface_role=role,
        center_id=center,
        case_id=case,
        arm_id=arm,
        direction=direction,
        donor_id=donor,
        feature_names=("compatibility", "margin_shift"),
        feature_values=(feature, feature * 0.5),
        sample_ids=samples,
        baseline_probability_hex=float32_probability_hex(baseline),
        action_probability_hex=float32_probability_hex(probability),
    )


def _menu(
    role: SurfaceRole,
    center: str,
    case: str,
    *,
    donors: tuple[str, ...],
    baseline: tuple[float, ...] = (0.2, 0.3, 0.7, 0.8),
) -> LabelFreeCaseMenu:
    actions: list[LabelFreeAction] = []
    for ordinal, donor in enumerate(donors):
        actions.append(
            _action(
                role,
                center,
                case,
                f"D01::{donor}",
                Direction.D01,
                donor,
                baseline,
                (0.2, 0.8 - 0.05 * ordinal, 0.7, 0.8),
                2.0 - ordinal,
            )
        )
        actions.append(
            _action(
                role,
                center,
                case,
                f"D10::{donor}",
                Direction.D10,
                donor,
                baseline,
                (0.2, 0.3, 0.7, 0.2 + 0.05 * ordinal),
                2.0 - ordinal,
            )
        )
    actions.append(
        _action(
            role,
            center,
            case,
            "U_FULL",
            Direction.FULL,
            None,
            baseline,
            (0.1, 0.6, 0.6, 0.4),
            0.0,
        )
    )
    samples = tuple(f"{case}:s{index}" for index in range(len(baseline)))
    return LabelFreeCaseMenu(
        surface_role=role,
        center_id=center,
        case_id=case,
        sample_ids=samples,
        baseline_probability_hex=float32_probability_hex(baseline),
        actions=tuple(actions),
    )


def _source_surface(
    *, centers: int = 3, cases_per_center: int = 4
) -> tuple[tuple[LabelFreeCaseMenu, ...], SupportTruthCapability]:
    center_ids = tuple(f"C{index}" for index in range(centers))
    menus: list[LabelFreeCaseMenu] = []
    truth: dict[tuple[str, str], dict[str, int]] = {}
    for center in center_ids:
        donors = tuple(value for value in center_ids if value != center)
        for ordinal in range(cases_per_center):
            case = f"{center}:case{ordinal}"
            menu = _menu(
                SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,
                center,
                case,
                donors=donors,
            )
            menus.append(menu)
            truth[(center, case)] = dict(zip(menu.sample_ids, (0, 1, 1, 0), strict=True))
    return tuple(menus), SupportTruthCapability(truth)


def _u_only_positive_surface(
    *, role: SurfaceRole = SurfaceRole.SOURCE_TRAIN_DEVELOPMENT
) -> tuple[tuple[LabelFreeCaseMenu, ...], SupportTruthCapability | None]:
    """Build cases where exact U helps and every directional action has zero gain."""

    center_ids = ("C0", "C1", "C2")
    menus: list[LabelFreeCaseMenu] = []
    truth: dict[tuple[str, str], dict[str, int]] = {}
    baseline = (0.1, 0.4, 0.6, 0.9)
    labels = (0, 1, 1, 0)
    for center in center_ids:
        donors = tuple(value for value in center_ids if value != center)
        for ordinal in range(4 if role is SurfaceRole.SOURCE_TRAIN_DEVELOPMENT else 1):
            case = f"{center}:u-only:{ordinal}"
            actions: list[LabelFreeAction] = []
            for donor_ordinal, donor in enumerate(donors):
                # Both actions are active label-free, but remain on B's side of
                # the decision boundary and therefore have exactly zero BACC gain.
                actions.append(
                    _action(
                        role,
                        center,
                        case,
                        f"D01::{donor}",
                        Direction.D01,
                        donor,
                        baseline,
                        (0.1, 0.45 - 0.01 * donor_ordinal, 0.6, 0.9),
                        1.0 - donor_ordinal,
                    )
                )
                actions.append(
                    _action(
                        role,
                        center,
                        case,
                        f"D10::{donor}",
                        Direction.D10,
                        donor,
                        baseline,
                        (0.1, 0.4, 0.6, 0.55 + 0.01 * donor_ordinal),
                        1.0 - donor_ordinal,
                    )
                )
            actions.append(
                _action(
                    role,
                    center,
                    case,
                    "U_FULL",
                    Direction.FULL,
                    None,
                    baseline,
                    (0.1, 0.9, 0.9, 0.1),
                    3.0,
                )
            )
            menu = LabelFreeCaseMenu(
                surface_role=role,
                center_id=center,
                case_id=case,
                sample_ids=tuple(f"{case}:s{index}" for index in range(len(baseline))),
                baseline_probability_hex=float32_probability_hex(baseline),
                actions=tuple(actions),
            )
            menus.append(menu)
            if role is SurfaceRole.SOURCE_TRAIN_DEVELOPMENT:
                truth[(center, case)] = dict(
                    zip(menu.sample_ids, labels, strict=True)
                )
    capability = SupportTruthCapability(truth) if truth else None
    return tuple(menus), capability


def _small_config(**updates: object) -> RouterFitConfig:
    values: dict[str, object] = {
        "outer_folds": 2,
        "inner_folds": 2,
        "opportunity_ridge_alphas": (1.0,),
        "ranker_ridge_alphas": (1.0,),
        "k_values": (1,),
        "lambda_values": (1.0,),
        "route_thresholds": (0.0,),
        "maximum_numeric_features": 2,
        "required_source_case_count": None,
        "required_source_center_count": None,
        "minimum_cases_per_center": 2,
        "minimum_routed_oof_cases": 2,
        "minimum_routed_oof_centers": 2,
        "minimum_routed_oof_cases_per_center": 1,
        "bootstrap_replicates": 32,
    }
    values.update(updates)
    return RouterFitConfig(**values)


def test_predecessor_free_roles_direction_and_json_contract() -> None:
    assert Direction.FULL.value == "FULL"
    assert SurfaceRole.SOURCE_TRAIN_DEVELOPMENT.value == "SOURCE_TRAIN_DEVELOPMENT"
    package = Path(
        "src/midogpp_thesis/cvae/routing/pooled_pairwise_selected_policy_router_v17"
    )
    forbidden = (
        "hierarchical_support_action_risk_router_v16",
        "policy_calibrated_residual_router",
    )
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = "\n".join(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )
        assert not any(token in imported for token in forbidden)
    menus, capability = _source_surface()
    assert json.loads(json.dumps(menus[0].public_payload()))["center_id"] == "C0"
    assert "(0, 1, 1, 0)" not in repr(capability)
    assert "memory-only" in repr(capability)


def test_branchwise_bytes_unused_branch_and_fewer_than_k() -> None:
    menu = _menu(
        SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,
        "C0",
        "case",
        donors=("C1", "C2"),
        baseline=(0.2, 0.3, 0.4, 0.1),
    )
    composite = build_soft_topk_composite(
        menu,
        d01_ranked_actions=("D01::C1",),
        d10_ranked_actions=(),
        k=1,
        mixing_lambda=0.5,
    )
    assert composite.d10_action_ids == ()
    # Cell zero is unchanged by the selected action and must be exact B bytes.
    assert composite.probability_hex[0] == menu.baseline_probability_hex[0]
    selected = struct.unpack("<f", bytes.fromhex(menu.action_for("D01::C1").action_probability_hex[1]))[0]
    baseline = struct.unpack("<f", bytes.fromhex(menu.baseline_probability_hex[1]))[0]
    expected = struct.pack("<f", 0.5 * float(baseline) + 0.5 * float(selected)).hex()
    assert composite.probability_hex[1] == expected

    mixed = _menu(
        SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,
        "C0",
        "mixed",
        donors=("C1",),
    )
    with pytest.raises(ProtocolError, match="fewer than K"):
        build_soft_topk_composite(
            mixed,
            d01_ranked_actions=("D01::C1",),
            d10_ranked_actions=("D10::C1",),
            k=2,
            mixing_lambda=1.0,
        )


def test_exact_u_and_selected_only_truth_capability() -> None:
    menus, capability = _source_surface()
    menu = menus[0]
    exact_u = build_exact_u_composite(menu)
    assert exact_u.kind is CompositeKind.U_FULL
    assert exact_u.probability_hex == menu.full_action.action_probability_hex
    assert exact_u.composite_hash != build_baseline_composite(menu).composite_hash
    training = tuple(
        (row.center_id, row.case_id)
        for row in menus
        if row is not menu
    )
    seal = SealedOOFSelection(
        outer_fold=0,
        composite=exact_u,
        requested_arm_id="U_FULL",
        route_score=1.0,
        route_threshold=0.0,
        training_case_keys=training,
        model_hash="0" * 64,
    )
    score = score_selected_composite(capability, seal)
    assert score.route_selected and score.utility_success
    assert score.bacc_gain > 0.0
    assert capability.selected_score_count == 1
    public = json.dumps(capability.public_payload(), sort_keys=True)
    assert "(0, 1, 1, 0)" not in public
    with pytest.raises(ProtocolError, match="cannot be serialized"):
        pickle.dumps(capability)
    with pytest.raises(ProtocolError, match="sealed selected"):
        capability.score_selected(exact_u)  # type: ignore[arg-type]


def test_pair_comparisons_have_one_case_weight_and_duplication_invariance() -> None:
    menus, capability = _source_surface()
    profiles, outcomes = capability.derive_training_surface(menus)
    transform = fit_feature_transform(menus, maximum_numeric_features=2)
    comparisons = build_pairwise_comparisons(menus, outcomes, transform=transform)
    weights: dict[tuple[str, str], float] = {}
    for row in comparisons:
        key = (row.center_id, row.case_id)
        weights[key] = weights.get(key, 0.0) + row.case_weight
    assert set(weights) == {(row.center_id, row.case_id) for row in menus}
    assert all(value == pytest.approx(1.0 / 12.0) for value in weights.values())
    original = fit_pairwise_ranker(comparisons, alpha=1.0, transform=transform)
    duplicated = fit_pairwise_ranker(
        (*comparisons, *comparisons), alpha=1.0, transform=transform
    )
    assert duplicated.coefficients == original.coefficients
    assert duplicated.ranker_hash == original.ranker_hash


def test_center_stratification_is_deterministic_and_balanced() -> None:
    menus, _ = _source_surface(cases_per_center=7)
    keys = tuple((row.center_id, row.case_id) for row in menus)
    first = center_stratified_folds(keys, fold_count=5, namespace="test")
    second = center_stratified_folds(tuple(reversed(keys)), fold_count=5, namespace="test")
    assert first == second
    for center in ("C0", "C1", "C2"):
        counts = [sum(key[0] == center for key in fold) for fold in first]
        assert max(counts) - min(counts) <= 1


def test_outer_heldout_poison_cannot_change_its_selection() -> None:
    menus, clean = _source_surface()
    config = _small_config()
    clean_profiles, clean_outcomes = clean.derive_training_surface(menus)
    clean_result = nested_source_crossfit(
        menus,
        clean_profiles,
        clean_outcomes,
        clean,
        config=config,
    )
    held_center, held_case = clean_result.outer_fold_case_keys[0][0]
    poisoned_labels: dict[tuple[str, str], dict[str, int]] = {}
    for menu in menus:
        labels = (0, 1, 1, 0)
        if (menu.center_id, menu.case_id) == (held_center, held_case):
            labels = (1, 0, 0, 1)
        poisoned_labels[(menu.center_id, menu.case_id)] = dict(
            zip(menu.sample_ids, labels, strict=True)
        )
    poisoned = SupportTruthCapability(poisoned_labels)
    poison_profiles, poison_outcomes = poisoned.derive_training_surface(menus)
    poison_result = nested_source_crossfit(
        menus,
        poison_profiles,
        poison_outcomes,
        poisoned,
        config=config,
    )
    clean_selection = clean_result.selection_for(held_center, held_case)
    poison_selection = poison_result.selection_for(held_center, held_case)
    assert clean_selection.selection_hash == poison_selection.selection_hash
    assert (held_center, held_case) not in clean_selection.training_case_keys
    assert clean_result.records != poison_result.records


def test_zero_frontier_returns_before_bootstrap() -> None:
    menus, capability = _source_surface(centers=2, cases_per_center=3)
    records = []
    for index, menu in enumerate(menus):
        composite = build_baseline_composite(menu)
        training = tuple(
            (row.center_id, row.case_id)
            for row in menus
            if row is not menu
        )
        seal = SealedOOFSelection(
            outer_fold=index % 2,
            composite=composite,
            requested_arm_id="B",
            route_score=0.0,
            route_threshold=0.0,
            training_case_keys=training,
            model_hash="1" * 64,
        )
        records.append(score_selected_composite(capability, seal))
    admission = build_source_only_admission(records, config=_small_config())
    assert admission.status is AdmissionStatus.NO_NONZERO_SAFE_OOF_COVERAGE
    assert admission.public_payload()["status"] == "NO_NONZERO_SAFE_OOF_COVERAGE"
    assert not admission.admitted
    assert not admission.bootstrap_performed
    assert admission.bounds is None


def test_pooled_fit_target_routing_and_report_fields() -> None:
    menus, capability = _source_surface()
    policy = fit_source_router(menus, capability, config=_small_config())
    assert len(policy.training_case_keys) == 12
    assert len(policy.oof_records) == 12
    assert policy.model_hash == policy.model.model_hash
    assert policy.public_payload()["pooled_known_center_policy_count"] == 1
    target = tuple(
        _menu(
            SurfaceRole.TARGET_EVALUATION,
            center,
            f"{center}:test",
            donors=tuple(value for value in ("C0", "C1", "C2") if value != center),
        )
        for center in ("C0", "C1", "C2")
    )
    decisions = route_target_cases(policy, target)
    report = route_decision_report(decisions)
    assert report["case_count"] == 3
    assert set(
        (
            "route_selected_count",
            "probability_changed_count",
            "prediction_changed_count",
            "utility_success_count",
            "mean_selected_donor_entropy",
        )
    ).issubset(report)
    assert report["utility_success_count"] is None
    assert all(row.utility_success is None for row in decisions)
    json.dumps(policy.public_payload(), sort_keys=True)


def test_explicit_u_head_routes_u_when_directional_actions_have_no_gain() -> None:
    menus, capability = _u_only_positive_surface()
    assert capability is not None
    profiles, outcomes = capability.derive_training_surface(menus)
    directional = tuple(
        row for row in outcomes if row.action.direction is not Direction.FULL
    )
    uniform = tuple(row for row in outcomes if row.action.direction is Direction.FULL)
    assert directional and all(row.bacc_gain == 0.0 for row in directional)
    assert len(uniform) == len(menus)
    assert all(row.bacc_gain > 0.0 for row in uniform)

    policy = fit_source_router(
        menus,
        capability,
        config=_small_config(),
        case_profiles=profiles,
        action_outcomes=outcomes,
    )
    assert policy.crossfit.final_arm.kind is CompositeKind.U_FULL
    assert policy.admitted
    assert all(row.selection.composite.kind is CompositeKind.U_FULL for row in policy.oof_records)
    prediction = policy.model.predict_menu(menus[0])
    assert prediction.directional_route_score == pytest.approx(0.0, abs=1.0e-12)
    assert prediction.u_full_route_score > 0.0
    assert prediction.route_score_for(CompositeKind.U_FULL) == prediction.u_full_route_score
    assert prediction.public_payload()["u_full_score_source"] == (
        "explicit_exact_u_outcome_head"
    )
    assert prediction.public_payload()["selected_action_family_route_score"] is True
    assert policy.model.public_payload()["u_full_opportunity"]["direction"] == "FULL"

    target, target_capability = _u_only_positive_surface(role=SurfaceRole.TARGET_EVALUATION)
    assert target_capability is None
    decisions = route_target_cases(policy, target)
    assert decisions
    assert all(row.requested_arm_id == "U_FULL" for row in decisions)
    assert all(row.composite.kind is CompositeKind.U_FULL for row in decisions)
    assert all(row.route_score > 0.0 for row in decisions)
