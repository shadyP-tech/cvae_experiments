from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.config import (
    HarpStage90V9Config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    SOURCE_DEVELOPMENT_ROLE,
    TARGET_EVALUATION_ROLE,
    V9_CACHE_IDENTITY,
    HarpCacheRow,
    HarpConsumedCacheIndex,
    _authenticate_frozen_route_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v9.preparation_contracts import (
    EXPECTED_CASE_COUNT,
    EXPECTED_SOURCE_TRAIN_CASE_COUNT,
    EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER,
    EXPECTED_SOURCE_TRAIN_ROW_COUNT,
    EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER,
    EXPECTED_TARGET_TEST_CASE_COUNT,
    EXPECTED_TARGET_TEST_CASES_BY_CENTER,
    EXPECTED_TARGET_TEST_ROW_COUNT,
    EXPECTED_TARGET_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9 import (
    Direction,
    LabelFreeAction,
    PairwiseFitConfig,
    SourceActionOutcome,
    fit_source_lodo,
    float32_probability_hex,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.contracts import (
    FrozenRouteReceipt,
)
from midogpp_thesis.cvae.runtime.harp_v9_execution.physical_actions import (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    build_all_development_actions,
    build_all_target_actions,
    build_development_action_menu,
    build_target_action_menu,
)


OUTER = "9"
SOURCE_CENTERS = ("0", "1", "2", "3", "5")
FEATURE_NAMES = ("budget_signal", "allocation_signal")
BASELINE = float32_probability_hex((0.2, 0.7, 0.3, 0.6))
D01_PROBABILITY = float32_probability_hex((0.8, 0.7, 0.3, 0.6))
D10_PROBABILITY = float32_probability_hex((0.2, 0.3, 0.3, 0.6))
HXE_PROBABILITY = float32_probability_hex((0.65, 0.7, 0.3, 0.6))


def _action(
    *,
    query: str,
    case: str,
    action_id: str,
    direction: Direction,
    features: tuple[float, float],
    probability: tuple[str, ...],
    kind: str = "U",
    candidate: str | None = None,
) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=OUTER,
        query_center_id=query,
        case_id=case,
        action_id=action_id,
        action_kind=kind,
        direction=direction,
        candidate_source_id=candidate,
        feature_names=FEATURE_NAMES,
        feature_values=features,
        baseline_probability_hex=BASELINE,
        action_probability_hex=probability,
    )


def _source_surface() -> tuple[SourceActionOutcome, ...]:
    rows: list[SourceActionOutcome] = []
    for center_index, center in enumerate(SOURCE_CENTERS):
        candidates = tuple(value for value in SOURCE_CENTERS if value != center)
        for ordinal in range(3):
            case = f"source-{center}-{ordinal}"
            preference = 1.0 if ordinal % 2 == 0 else -1.0
            activity = 1.0 + 0.1 * ordinal
            candidate = candidates[(center_index + ordinal) % len(candidates)]
            actions = (
                (
                    _action(
                        query=center,
                        case=case,
                        action_id="u-d01",
                        direction=Direction.D01,
                        features=(activity, preference),
                        probability=D01_PROBABILITY,
                    ),
                    0.13 if preference > 0.0 else 0.04,
                ),
                (
                    _action(
                        query=center,
                        case=case,
                        action_id="u-d10",
                        direction=Direction.D10,
                        features=(activity, -preference),
                        probability=D10_PROBABILITY,
                    ),
                    0.12 if preference < 0.0 else 0.03,
                ),
                (
                    _action(
                        query=center,
                        case=case,
                        action_id=f"hxe-{candidate}",
                        direction=Direction.D01,
                        features=(activity + 0.25, preference + 0.2),
                        probability=HXE_PROBABILITY,
                        kind="HXE",
                        candidate=candidate,
                    ),
                    0.08 + 0.005 * ordinal,
                ),
            )
            rows.extend(
                SourceActionOutcome(
                    action=action,
                    bacc_gain=gain,
                    brier_delta=-gain / 2.0,
                    log_delta=-gain / 3.0,
                )
                for action, gain in actions
            )
    return tuple(rows)


def _typed_config(tmp_path: Path, *, config_hash: str) -> HarpStage90V9Config:
    return HarpStage90V9Config(
        source_path=tmp_path / "synthetic-v9.yaml",
        artifact_root=str(tmp_path / "output"),
        input_locations={},
        expected_hashes={},
        execution_authorized=True,
        protocol={"centers": list(CENTERS)},
        model={},
        runtime={},
        claim_boundary={},
        config_hash=config_hash,
    )


def _full_test_cache(tmp_path: Path) -> HarpConsumedCacheIndex:
    rows = tuple(
        HarpCacheRow(
            center=CENTERS[index % len(CENTERS)],
            case_id=f"test-case-{index:03d}",
            sample_id=f"eval-{index:03d}",
            split_role=TARGET_EVALUATION_ROLE,
            split_row_index=index,
            embedding_file="synthetic.npy",
            embedding_row_index=index,
        )
        for index in range(EXPECTED_TARGET_TEST_CASE_COUNT)
    )
    return HarpConsumedCacheIndex(
        root=tmp_path,
        rows=rows,
        shards={},
        member_sha256={},
        content_sha256="1" * 64,
        cache_hash="2" * 64,
    )


def _frozen_receipt(
    *, config_hash: str, case_count: int, cache: HarpConsumedCacheIndex
) -> FrozenRouteReceipt:
    ordered_cases = tuple(
        sorted(
            {
                (row.center, row.case_id)
                for row in cache.rows
                if row.split_role == TARGET_EVALUATION_ROLE
            }
        )
    )
    samples = {
        key: [
            row.sample_id
            for row in cache.rows
            if row.split_role == TARGET_EVALUATION_ROLE
            and (row.center, row.case_id) == key
        ]
        for key in ordered_cases
    }
    return FrozenRouteReceipt(
        seal_hash="3" * 64,
        config_hash=config_hash,
        route_hash="4" * 64,
        policy_hash="5" * 64,
        model_hash="6" * 64,
        target_action_hash="7" * 64,
        validation_bundle_hash="8" * 64,
        independent_validation_hashes=("9" * 64, "a" * 64),
        expected_center_ids=CENTERS,
        case_count=case_count,
        ordered_case_identity_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v9_ordered_target_case_identity_v1",
                "ordered_cases": [list(value) for value in ordered_cases],
            }
        ),
        ordered_sample_identity_hash=canonical_hash(
            {
                "schema_version": "midogpp_harp_v9_ordered_target_sample_identity_v1",
                "ordered_case_samples": [
                    {
                        "outer_target_id": center,
                        "case_id": case,
                        "sample_ids": samples[(center, case)],
                    }
                    for center, case in ordered_cases
                ],
            }
        ),
    )


def test_v9_contract_uses_all_source_train_cases_and_all_test_cases() -> None:
    assert SOURCE_DEVELOPMENT_ROLE == "harp_source_train_development"
    assert TARGET_EVALUATION_ROLE == "harp_full_test_evaluation"
    assert DEVELOPMENT_ROLE == SOURCE_DEVELOPMENT_ROLE
    assert EVALUATION_ROLE == TARGET_EVALUATION_ROLE
    assert V9_CACHE_IDENTITY.artifact_id == (
        "midogpp_stage90_harp_source_train_full_test_cache_v9"
    )

    assert EXPECTED_SOURCE_TRAIN_ROW_COUNT == 9_648
    assert EXPECTED_SOURCE_TRAIN_CASE_COUNT == 216
    assert EXPECTED_TARGET_TEST_ROW_COUNT == 9_928
    assert EXPECTED_TARGET_TEST_CASE_COUNT == 218
    assert EXPECTED_CASE_COUNT == 216 + 218
    assert sum(EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER.values()) == 9_648
    assert sum(EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER.values()) == 216
    assert sum(EXPECTED_TARGET_TEST_ROWS_BY_CENTER.values()) == 9_928
    assert sum(EXPECTED_TARGET_TEST_CASES_BY_CENTER.values()) == 218
    assert 106 not in {
        EXPECTED_SOURCE_TRAIN_CASE_COUNT,
        EXPECTED_TARGET_TEST_CASE_COUNT,
    }
    assert 112 not in {
        EXPECTED_SOURCE_TRAIN_CASE_COUNT,
        EXPECTED_TARGET_TEST_CASE_COUNT,
    }


def test_v9_physical_menus_exclude_outer_h_and_pseudo_target_q() -> None:
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            menu = build_development_action_menu(outer, query)
            expected_sources = tuple(
                center for center in CENTERS if center not in {outer, query}
            )
            assert tuple(row.action_id for row in menu[:2]) == (
                BASE_ACTION_ID,
                UNIFORM_ACTION_ID,
            )
            assert tuple(row.selected_source_id for row in menu[2:]) == expected_sources
            assert all(row.source_order == expected_sources for row in menu)
            assert all(row.outer_target_id == outer for row in menu)
            assert all(row.query_center_id == query for row in menu)

        target_menu = build_target_action_menu(outer)
        target_sources = tuple(center for center in CENTERS if center != outer)
        assert tuple(row.selected_source_id for row in target_menu[2:]) == target_sources
        assert all(row.source_order == target_sources for row in target_menu)
        assert all(row.query_center_id == outer for row in target_menu)

    assert len(build_all_development_actions()) == 9 * 8 * (2 + 7)
    assert len(build_all_target_actions()) == 9 * (2 + 8)


def test_q_outcome_poison_cannot_change_q_oof_routes() -> None:
    surface = _source_surface()
    heldout_q = SOURCE_CENTERS[0]
    poisoned = tuple(
        replace(
            row,
            bacc_gain=-7.0 - index,
            brier_delta=3.0 + index,
            log_delta=4.0 + index,
        )
        if row.action.query_center_id == heldout_q
        else row
        for index, row in enumerate(surface)
    )
    config = (PairwiseFitConfig(pairwise_alpha=0.1, residual_alpha=0.1),)

    clean_fit = fit_source_lodo(surface, config_grid=config)
    poisoned_fit = fit_source_lodo(poisoned, config_grid=config)

    clean_model_hash = dict(clean_fit.heldout_model_hashes)[heldout_q]
    poisoned_model_hash = dict(poisoned_fit.heldout_model_hashes)[heldout_q]
    clean_predictions = tuple(
        row.prediction_hash
        for row in clean_fit.oof_predictions
        if row.query_center_id == heldout_q
    )
    poisoned_predictions = tuple(
        row.prediction_hash
        for row in poisoned_fit.oof_predictions
        if row.query_center_id == heldout_q
    )

    assert clean_model_hash == poisoned_model_hash
    assert clean_predictions == poisoned_predictions
    assert clean_predictions
    assert all(
        heldout_q in row.excluded_center_ids
        and OUTER in row.excluded_center_ids
        and heldout_q not in row.training_center_ids
        and heldout_q not in row.training_candidate_ids
        for row in clean_fit.oof_predictions
        if row.query_center_id == heldout_q
    )


def test_evaluation_capability_requires_exact_all_test_case_count(
    tmp_path: Path,
) -> None:
    config_hash = "b" * 64
    config = _typed_config(tmp_path, config_hash=config_hash)
    cache = _full_test_cache(tmp_path)
    receipt = _frozen_receipt(
        config_hash=config_hash,
        case_count=EXPECTED_TARGET_TEST_CASE_COUNT,
        cache=cache,
    )

    _authenticate_frozen_route_receipt(config, cache, receipt)

    with pytest.raises(ProtocolError, match="not evaluation-bound"):
        _authenticate_frozen_route_receipt(
            config,
            cache,
            replace(receipt, case_count=EXPECTED_TARGET_TEST_CASE_COUNT - 1),
        )


def test_evaluation_capability_rejects_same_count_case_or_sample_substitution(
    tmp_path: Path,
) -> None:
    config_hash = "b" * 64
    config = _typed_config(tmp_path, config_hash=config_hash)
    cache = _full_test_cache(tmp_path)
    receipt = _frozen_receipt(
        config_hash=config_hash,
        case_count=EXPECTED_TARGET_TEST_CASE_COUNT,
        cache=cache,
    )
    case_substituted = replace(
        cache,
        rows=(
            replace(cache.rows[0], case_id="substituted-test-case"),
            *cache.rows[1:],
        ),
    )
    sample_substituted = replace(
        cache,
        rows=(
            replace(cache.rows[0], sample_id="substituted-eval-sample"),
            *cache.rows[1:],
        ),
    )
    with pytest.raises(ProtocolError, match="not evaluation-bound"):
        _authenticate_frozen_route_receipt(config, case_substituted, receipt)
    with pytest.raises(ProtocolError, match="not evaluation-bound"):
        _authenticate_frozen_route_receipt(config, sample_substituted, receipt)


def test_evaluation_capability_rejects_delete_duplicate_same_row_count(
    tmp_path: Path,
) -> None:
    config_hash = "b" * 64
    config = _typed_config(tmp_path, config_hash=config_hash)
    cache = _full_test_cache(tmp_path)
    receipt = _frozen_receipt(
        config_hash=config_hash,
        case_count=EXPECTED_TARGET_TEST_CASE_COUNT,
        cache=cache,
    )
    duplicate_after_delete = replace(
        cache,
        rows=(
            replace(cache.rows[0], case_id=cache.rows[len(CENTERS)].case_id),
            *cache.rows[1:],
        ),
    )
    assert len(duplicate_after_delete.rows) == len(cache.rows)
    with pytest.raises(ProtocolError, match="not evaluation-bound"):
        _authenticate_frozen_route_receipt(config, duplicate_after_delete, receipt)
