from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_ENSEMBLE_METRIC_COUNT,
    EXPECTED_PLAN_CELL_COUNT,
    EXPECTED_SEED_CELL_COUNT,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    MATCHED_BUDGET_PER_CLASS,
    PERMUTATION_ACTION_ID,
    PRIMARY_ENDPOINT,
    SUPPORT_ACTION_ID,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    FrozenActionPayload,
    PredictionCell,
    PredictionSealCapability,
    build_evaluation_plan,
    evaluate_sealed_predictions,
    expected_action_ids,
    legal_sources,
    score_sealed_predictions,
    seal_predictions,
    tail_action_id,
    tail_source,
    validate_prediction_seal,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _class_counts(counts: dict[str, int]) -> dict[int, dict[str, int]]:
    return {0: dict(counts), 1: dict(counts)}


def _action(
    target: str,
    action_id: str,
    counts: dict[str, int],
    *,
    ranks: dict[str, float] | None = None,
    permutation: dict[str, str] | None = None,
) -> FrozenActionPayload:
    return FrozenActionPayload(
        target_center=target,
        action_id=action_id,
        source_counts_by_class=_class_counts(counts),
        action_hash=f"hash::{target}::{action_id}",
        mean_normalized_midrank_by_source=ranks or {},
        source_identity_permutation=permutation or {},
    )


def _target_actions(target: str) -> dict[str, FrozenActionPayload]:
    sources = legal_sources(target)
    support_ranks = {
        source: index / float(len(sources) - 1)
        for index, source in enumerate(sources)
    }
    global_ranks = {
        source: 1.0 - support_ranks[source] for source in sources
    }
    support_counts = {source: 128 for source in sources}
    support_counts[sources[0]] += 128
    global_counts = {source: 128 for source in sources}
    global_counts[sources[-1]] += 128

    permutation = {
        source: sources[(index + 1) % len(sources)]
        for index, source in enumerate(sources)
    }
    permutation_counts = {
        permutation[source]: support_counts[source] for source in sources
    }
    permutation_ranks = {
        permutation[source]: support_ranks[source] for source in sources
    }
    actions = {
        BASE_ACTION_ID: _action(
            target,
            BASE_ACTION_ID,
            {source: 128 for source in sources},
        ),
        UNIFORM_ACTION_ID: _action(
            target,
            UNIFORM_ACTION_ID,
            {source: 144 for source in sources},
        ),
        GLOBAL_ACTION_ID: _action(
            target,
            GLOBAL_ACTION_ID,
            global_counts,
            ranks=global_ranks,
        ),
        SUPPORT_ACTION_ID: _action(
            target,
            SUPPORT_ACTION_ID,
            support_counts,
            ranks=support_ranks,
        ),
        PERMUTATION_ACTION_ID: _action(
            target,
            PERMUTATION_ACTION_ID,
            permutation_counts,
            ranks=permutation_ranks,
            permutation=permutation,
        ),
    }
    for source in sources:
        counts = {candidate: 128 for candidate in sources}
        counts[source] += 128
        action_id = tail_action_id(source)
        actions[action_id] = _action(target, action_id, counts)
    return actions


def _actions() -> dict[str, dict[str, FrozenActionPayload]]:
    return {target: _target_actions(target) for target in CENTERS}


def _rows() -> dict[str, tuple[str, ...]]:
    return {
        target: tuple(f"row::{target}::{index}" for index in range(4))
        for target in CENTERS
    }


def _probabilities_for(cell: object) -> np.ndarray:
    action_id = str(getattr(cell, "action_id"))
    target = str(getattr(cell, "target_center"))
    training_seed = int(getattr(cell, "training_seed"))
    generation_seed = int(getattr(cell, "generation_seed"))
    perfect = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    three_quarters = np.asarray([0.1, 0.8, 0.7, 0.9], dtype=np.float64)
    half = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    if action_id == SUPPORT_ACTION_ID:
        return perfect
    if action_id == GLOBAL_ACTION_ID:
        return three_quarters
    if action_id == PERMUTATION_ACTION_ID and target == CENTERS[0]:
        seed_index = list(
            (train, generation)
            for train in TRAINING_SEEDS
            for generation in GENERATION_SEEDS
        ).index((training_seed, generation_seed))
        # Five barely-correct cells and four confidently wrong cells make the
        # descriptive seed mean positive while the probability ensemble is 0.
        if seed_index < 5:
            return np.asarray([0.49, 0.49, 0.51, 0.51], dtype=np.float64)
        return np.asarray([0.9, 0.9, 0.1, 0.1], dtype=np.float64)
    source = tail_source(action_id)
    if source is not None:
        sources = legal_sources(target)
        if source == sources[0]:
            return three_quarters
        if source == sources[1]:
            return perfect
        return half
    return half


def _predictions(plan: object) -> tuple[PredictionCell, ...]:
    rows = _rows()
    return tuple(
        PredictionCell(
            target_center=cell.target_center,
            training_seed=cell.training_seed,
            generation_seed=cell.generation_seed,
            action_id=cell.action_id,
            action_hash=cell.action_hash,
            evaluation_row_ids=rows[cell.target_center],
            probabilities=_probabilities_for(cell),
        )
        for cell in plan.cells
    )


def _labels() -> dict[str, int]:
    return {
        row_id: (0 if index < 2 else 1)
        for target, rows in _rows().items()
        for index, row_id in enumerate(rows)
    }


def test_plan_expands_every_target_action_and_seed_with_locked_budgets() -> None:
    plan = build_evaluation_plan(
        _actions(), evaluation_row_ids_by_target=_rows()
    )

    assert len(plan.cells) == EXPECTED_PLAN_CELL_COUNT == 1053
    assert len({cell.key for cell in plan.cells}) == EXPECTED_PLAN_CELL_COUNT
    assert plan.primary_endpoint == PRIMARY_ENDPOINT
    for target in CENTERS:
        actions = plan.actions_by_target[target]
        assert tuple(action.action_id for action in actions) == expected_action_ids(
            target
        )
        assert len(actions) == 13
        assert actions[0].budget_per_class == 1024
        assert all(
            action.budget_per_class == MATCHED_BUDGET_PER_CLASS
            for action in actions[1:]
        )
        assert all(
            target not in action.source_counts_by_class[0]
            for action in actions
        )


def test_plan_rejects_incomplete_menu_and_budget_drift() -> None:
    actions = _actions()
    actions[CENTERS[0]].pop(tail_action_id(legal_sources(CENTERS[0])[0]))
    with pytest.raises(ProtocolError, match="every H x e"):
        build_evaluation_plan(actions)

    actions = _actions()
    target = CENTERS[0]
    uniform = actions[target][UNIFORM_ACTION_ID]
    bad_counts = {
        label: dict(uniform.source_counts_by_class[label]) for label in (0, 1)
    }
    bad_counts[0][legal_sources(target)[0]] += 1
    actions[target][UNIFORM_ACTION_ID] = replace(
        uniform, source_counts_by_class=bad_counts
    )
    with pytest.raises(ProtocolError, match="budget|budgets"):
        build_evaluation_plan(actions)


def test_opaque_seal_requires_complete_exact_row_action_seed_coverage() -> None:
    plan = build_evaluation_plan(
        _actions(), evaluation_row_ids_by_target=_rows()
    )
    predictions = _predictions(plan)
    with pytest.raises(ProtocolError, match="every target/action/seed"):
        seal_predictions(plan, predictions[:-1])

    drifted = list(predictions)
    drifted[-1] = replace(
        drifted[-1],
        evaluation_row_ids=tuple(reversed(drifted[-1].evaluation_row_ids)),
    )
    with pytest.raises(ProtocolError, match="row plan|row order"):
        seal_predictions(plan, drifted)

    with pytest.raises(TypeError):
        PredictionSealCapability(None, None)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="requires an issued"):
        score_sealed_predictions(object(), _labels())  # type: ignore[arg-type]

    seal = seal_predictions(plan, predictions)
    summary = validate_prediction_seal(seal, expected_plan=plan)
    assert summary.prediction_cell_count == EXPECTED_PLAN_CELL_COUNT
    assert summary.action_seed_coverage_complete is True
    assert summary.labels_opened is False

    labels_with_extra = {**_labels(), "support-row": 0}
    with pytest.raises(ProtocolError, match="exactly cover"):
        score_sealed_predictions(seal, labels_with_extra)


def test_primary_ensemble_contrasts_center_inference_and_oracles() -> None:
    plan = build_evaluation_plan(
        _actions(), evaluation_row_ids_by_target=_rows()
    )
    seal = seal_predictions(plan, _predictions(plan))
    result = evaluate_sealed_predictions(seal, _labels())

    assert result.primary_endpoint == PRIMARY_ENDPOINT
    assert result.policy_update_emitted is False
    assert len(result.scored.seed_cell_metrics) == EXPECTED_PLAN_CELL_COUNT
    assert len(result.scored.ensemble_metrics) == EXPECTED_ENSEMBLE_METRIC_COUNT
    assert all(
        row.descriptive_only and row.endpoint_role.endswith("descriptive_only")
        for row in result.scored.seed_cell_metrics
    )
    assert all(
        row.primary_endpoint
        and row.endpoint == PRIMARY_ENDPOINT
        and row.seed_cell_count == EXPECTED_SEED_CELL_COUNT
        for row in result.scored.ensemble_metrics
    )

    by_center_contrast = {
        (row.target_center, row.contrast_id): row
        for row in result.center_contrasts
    }
    first_sp = by_center_contrast[(CENTERS[0], "S-P")]
    assert first_sp.probability_ensemble_bacc_delta == pytest.approx(1.0)
    assert first_sp.descriptive_seed_cell_mean_bacc_delta == pytest.approx(
        1.0 - 5.0 / 9.0
    )

    inferred = {row.contrast_id: row for row in result.contrast_inference}
    assert set(inferred) == {"S-U", "S-G", "G-U", "U-B", "S-B", "S-P"}
    assert inferred["S-U"].center_count == 9
    assert inferred["S-U"].contrast_role == (
        "predeclared_primary_center_contrast"
    )
    assert inferred["G-U"].contrast_role == (
        "predeclared_secondary_center_contrast"
    )
    assert inferred["S-U"].wins == 9
    assert inferred["S-U"].ties == 0
    assert inferred["U-B"].wins == 0
    assert inferred["U-B"].ties == 9
    assert inferred["U-B"].losses == 0
    assert inferred["S-U"].one_sided_95_lcb == pytest.approx(0.5)
    assert inferred["S-U"].two_sided_95_ci_low == pytest.approx(0.5)
    assert inferred["S-U"].two_sided_95_ci_high == pytest.approx(0.5)

    assert len(result.oracle_diagnostics) == len(CENTERS)
    assert all(not row.top1_agreement for row in result.oracle_diagnostics)
    assert all(
        row.oracle_headroom_bacc == pytest.approx(0.25)
        and row.normalized_oracle_gap == pytest.approx(0.5)
        and row.diagnostic_only
        and not row.may_update_frozen_policy
        and row.prediction_seal_hash == seal.seal_hash
        for row in result.oracle_diagnostics
    )


def test_fresh_core_has_no_consumed_stage70_or_stage90_imports() -> None:
    import pathlib

    package = pathlib.Path(
        "src/midogpp_thesis/cvae/frozen_policy_downstream/residual_topup_fresh"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in package.glob("*.py")
    )
    assert "diagnostics.residual_topup_router" not in source
    assert "frozen_policy_downstream.scoring" not in source
    assert "frozen_policy_downstream.contracts" not in source


def test_undefined_oracle_spearman_uses_finite_zero_sentinel() -> None:
    plan = build_evaluation_plan(
        _actions(), evaluation_row_ids_by_target=_rows()
    )
    predictions = tuple(
        replace(cell, probabilities=np.asarray([0.1, 0.2, 0.3, 0.4]))
        for cell in _predictions(plan)
    )
    result = evaluate_sealed_predictions(
        seal_predictions(plan, predictions), _labels()
    )
    assert all(
        diagnostic.support_score_utility_spearman == 0.0
        and diagnostic.spearman_defined is False
        for diagnostic in result.oracle_diagnostics
    )
