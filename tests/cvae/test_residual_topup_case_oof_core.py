from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.contracts import (
    BASE_ACTION_ID,
    CENTERS,
    EXPECTED_ACTION_COUNT_PER_TARGET,
    EXPECTED_CASE_OOF_FOLD_COUNT,
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPECTED_PROXY_SCORE_COUNT,
    EXPECTED_SEALED_PREDICTION_CELL_COUNT,
    EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT,
    GENERATION_SEEDS,
    GLOBAL_ACTION_ID,
    GLOBAL_QUERY_ROLE,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    SUPPORT_QUERY_ROLE,
    TRAINING_SEEDS,
    UNIFORM_ACTION_ID,
    ProxyScoreRow,
    candidate_sources,
    expected_action_ids,
    global_candidate_sources,
    tail_action_id,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.config import (
    DOWNSTREAM_CLASSIFIER,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.partitions import (
    build_case_oof_surface,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.planning import (
    build_case_oof_plan,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.prediction_planning import (
    EXPECTED_PREDICTION_TASK_COUNT,
    build_prediction_tasks,
    write_evaluation_scratch,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof import (
    prediction_worker,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.ranking import (
    build_rank_surface,
    normalized_midranks,
)
from midogpp_thesis.cvae.protocol import ProtocolError


@dataclass(frozen=True)
class _Row:
    row_ordinal: int
    manifest_row_index: int
    sample_id: str
    case_id: str
    center: str
    partition_role: str


def _base_partition() -> SimpleNamespace:
    support: dict[str, tuple[_Row, ...]] = {}
    evaluation: dict[str, tuple[_Row, ...]] = {}
    ordinal = 0
    for center in CENTERS:
        support_rows: list[_Row] = []
        evaluation_rows: list[_Row] = []
        for case_index in range(2):
            case_id = f"center-{center}-support-{case_index}"
            for row_index in range(2):
                support_rows.append(
                    _Row(
                        row_ordinal=ordinal,
                        manifest_row_index=ordinal,
                        sample_id=f"{case_id}-sample-{row_index}",
                        case_id=case_id,
                        center=center,
                        partition_role="support",
                    )
                )
                ordinal += 1
        evaluation_case_count = 2 if center == CENTERS[-1] else 3
        for case_index in range(evaluation_case_count):
            case_id = f"center-{center}-evaluation-{case_index}"
            for row_index in range(2):
                evaluation_rows.append(
                    _Row(
                        row_ordinal=ordinal,
                        manifest_row_index=ordinal,
                        sample_id=f"{case_id}-sample-{row_index}",
                        case_id=case_id,
                        center=center,
                        partition_role="evaluation",
                    )
                )
                ordinal += 1
        support[center] = tuple(support_rows)
        evaluation[center] = tuple(evaluation_rows)
    return SimpleNamespace(
        support_rows_by_center=support,
        evaluation_rows_by_center=evaluation,
        lock_hash="a" * 16,
    )


def _proxy_rows(crossfit) -> tuple[ProxyScoreRow, ...]:
    rows: list[ProxyScoreRow] = []
    for target in CENTERS:
        sources = candidate_sources(target)
        support_cases = sorted(
            {row.case_id for row in crossfit.fixed_support_rows_by_center[target]}
        )
        for case_index, case_id in enumerate(support_cases):
            for source_index, source in enumerate(sources):
                for seed_index, seed in enumerate(TRAINING_SEEDS):
                    energy = float(source_index * 10 + case_index + seed_index)
                    if target == "0" and case_index == 0:
                        if source == sources[0]:
                            energy = (0.0, 0.0, 9.0)[seed_index]
                        elif source == sources[1]:
                            energy = 2.0
                    rows.append(
                        ProxyScoreRow(
                            outer_target=target,
                            query_role=SUPPORT_QUERY_ROLE,
                            query_center=target,
                            case_id=case_id,
                            candidate_source=source,
                            training_seed=seed,
                            row_count=2,
                            proxy_energy=energy,
                        )
                    )
        for query in sources:
            query_cases = sorted(
                {
                    row.case_id
                    for row in crossfit.fixed_support_rows_by_center[query]
                }
            )
            candidates = global_candidate_sources(target, query)
            for case_index, case_id in enumerate(query_cases):
                for source_index, source in enumerate(candidates):
                    for seed_index, seed in enumerate(TRAINING_SEEDS):
                        rows.append(
                            ProxyScoreRow(
                                outer_target=target,
                                query_role=GLOBAL_QUERY_ROLE,
                                query_center=query,
                                case_id=case_id,
                                candidate_source=source,
                                training_seed=seed,
                                row_count=2,
                                proxy_energy=float(
                                    source_index * 10
                                    + case_index
                                    + seed_index
                                ),
                            )
                        )
    assert len(rows) == EXPECTED_PROXY_SCORE_COUNT
    return tuple(rows)


@pytest.fixture(scope="module")
def crossfit():
    return build_case_oof_surface(
        _base_partition(),
        config_contract_hash="b" * 16,
    )


@pytest.fixture(scope="module")
def ranks(crossfit):
    return build_rank_surface(_proxy_rows(crossfit), crossfit)


@pytest.fixture(scope="module")
def plan(ranks, crossfit):
    return build_case_oof_plan(
        ranks,
        crossfit,
        config_contract_hash="b" * 16,
    )


def test_fixed_support_surface_has_exact_whole_case_oof_geometry(crossfit) -> None:
    assert len(crossfit.folds) == EXPECTED_CASE_OOF_FOLD_COUNT == 26
    assert tuple(crossfit.folds_by_target) == CENTERS
    heldout_samples: list[str] = []
    for fold in crossfit.folds:
        assert {row.case_id for row in fold.heldout_rows} == {
            fold.heldout_case_id
        }
        assert {row.partition_role for row in fold.heldout_rows} == {
            "evaluation"
        }
        assert fold.fixed_support_rows == crossfit.fixed_support_rows_by_center[
            fold.target_center
        ]
        assert len(fold.fixed_support_case_ids) == 2
        assert fold.heldout_case_id not in fold.fixed_support_case_ids
        heldout_samples.extend(row.sample_id for row in fold.heldout_rows)
    expected_evaluation_samples = {
        row.sample_id
        for center in CENTERS
        for row in crossfit.evaluation_rows_by_center[center]
    }
    assert set(heldout_samples) == expected_evaluation_samples
    assert len(heldout_samples) == len(set(heldout_samples))
    assert crossfit.lock_payload["fixed_support_cases_never_scored"] is True
    assert (
        crossfit.lock_payload["other_evaluation_embeddings_used_for_route"]
        is False
    )


def test_surface_fails_closed_on_support_or_global_case_identity_drift() -> None:
    base = _base_partition()
    support = dict(base.support_rows_by_center)
    support["0"] = tuple(
        row for row in support["0"] if row.case_id.endswith("support-0")
    )
    with pytest.raises(ProtocolError, match="fixed-support boundary"):
        build_case_oof_surface(
            SimpleNamespace(
                support_rows_by_center=support,
                evaluation_rows_by_center=base.evaluation_rows_by_center,
                lock_hash=base.lock_hash,
            ),
            config_contract_hash="b" * 16,
        )

    support = dict(base.support_rows_by_center)
    duplicate_case = support["0"][0].case_id
    support["1"] = (
        replace(support["1"][0], case_id=duplicate_case),
        replace(support["1"][1], case_id=duplicate_case),
        *support["1"][2:],
    )
    with pytest.raises(ProtocolError, match="globally center/role unique"):
        build_case_oof_surface(
            SimpleNamespace(
                support_rows_by_center=support,
                evaluation_rows_by_center=base.evaluation_rows_by_center,
                lock_hash=base.lock_hash,
            ),
            config_contract_hash="b" * 16,
        )


def test_true_midranks_preserve_ties_and_fail_closed() -> None:
    ranks = normalized_midranks({"a": 1.0, "b": 1.0, "c": 3.0})
    assert dict(ranks) == {"a": 0.25, "b": 0.25, "c": 1.0}
    with pytest.raises(TypeError):
        ranks["a"] = 0.0  # type: ignore[index]
    with pytest.raises(ProtocolError, match="finite"):
        normalized_midranks({"a": 1.0, "b": float("nan")})


def test_rank_surface_averages_three_replicas_before_fixed_case_ballots(
    ranks,
) -> None:
    target = ranks["0"]
    first_ballot = target.support_summary.ballots[0]
    sources = candidate_sources("0")
    assert first_ballot.mean_proxy_energy_by_source[sources[0]] == 3.0
    assert first_ballot.mean_proxy_energy_by_source[sources[1]] == 2.0
    assert (
        first_ballot.normalized_midrank_by_source[sources[1]]
        < first_ballot.normalized_midrank_by_source[sources[0]]
    )
    assert set(target.support_summary.ballot_count_by_source.values()) == {2}
    assert set(target.global_summary.ballot_count_by_source.values()) == {14}
    for ballot in target.global_summary.ballots:
        assert ballot.outer_target not in ballot.candidate_sources
        assert ballot.query_center not in ballot.candidate_sources
    for summary in (target.global_summary, target.support_summary):
        assert all(
            summary.priority_by_source[source]
            == pytest.approx(
                1.0 - summary.mean_normalized_midrank_by_source[source]
            )
            for source in summary.candidate_sources
        )


def test_rank_surface_rejects_evaluation_scores_and_incomplete_replicas(
    crossfit,
) -> None:
    rows = list(_proxy_rows(crossfit))
    with pytest.raises(ProtocolError, match="identity drifted"):
        replace(rows[0], evaluation_embeddings_used=True)
    with pytest.raises(ProtocolError, match="grid drifted|incomplete"):
        build_rank_surface(rows[:-1], crossfit)


def test_plan_freezes_complete_target_actions_and_reuses_s_across_folds(
    plan,
    crossfit,
) -> None:
    assert plan.action_count == EXPECTED_FROZEN_ACTION_COUNT == 117
    assert EXPECTED_UNIQUE_CLASSIFIER_FIT_COUNT == 1053
    assert EXPECTED_SEALED_PREDICTION_CELL_COUNT == 3042
    assert len(plan.actions_by_fold) == 26
    assert plan.action_library_payload["action_count"] == 117
    assert plan.lock_payload["support_rank_fixed_across_target_folds"] is True
    for target in CENTERS:
        actions = plan.actions_for_target(target)
        assert len(actions) == EXPECTED_ACTION_COUNT_PER_TARGET == 13
        assert tuple(action.action_id for action in actions) == expected_action_ids(
            target
        )
        for fold in crossfit.folds_by_target[target]:
            assert plan.actions_by_fold[fold.fold_id] is actions

        base = plan.action(target, BASE_ACTION_ID)
        uniform = plan.action(target, UNIFORM_ACTION_ID)
        global_action = plan.action(target, GLOBAL_ACTION_ID)
        support = plan.action(target, SUPPORT_ACTION_ID)
        permutation = plan.action(target, PERMUTATION_ACTION_ID)
        assert base.core_action is None
        assert base.topup_total_per_class == 0
        assert set(base.topup_counts_by_source.values()) == {0}
        assert all(
            set(counts.values()) == {128}
            for counts in base.final_counts_by_class.values()
        )
        assert set(uniform.topup_counts_by_source.values()) == {16}
        assert support.action_hash != permutation.action_hash
        for action in (global_action, support, permutation):
            assert action.core_action is not None
            assert dict(action.core_action.calibrated_energy_by_source) == {}
            best = min(
                action.source_order,
                key=action.mean_normalized_midrank_by_source.__getitem__,
            )
            worst = max(
                action.source_order,
                key=action.mean_normalized_midrank_by_source.__getitem__,
            )
            assert (
                action.direction_weights_by_source[best]
                >= action.direction_weights_by_source[worst]
            )
            if (
                action.mean_normalized_midrank_by_source[best]
                < action.mean_normalized_midrank_by_source[worst]
            ):
                assert (
                    action.direction_weights_by_source[best]
                    > action.direction_weights_by_source[worst]
                )
        for source in candidate_sources(target):
            tail = plan.action(target, tail_action_id(source))
            assert tail.selected_source == source
            assert tail.topup_counts_by_source[source] == 128
            assert sum(tail.topup_counts_by_source.values()) == 128


def test_action_and_plan_hashes_are_deterministic_and_tamper_evident(
    ranks,
    crossfit,
    plan,
) -> None:
    repeated = build_case_oof_plan(
        ranks,
        crossfit,
        config_contract_hash="b" * 16,
    )
    assert repeated.lock_hash == plan.lock_hash
    assert repeated.action_library_hash == plan.action_library_hash
    assert repeated.action("0", SUPPORT_ACTION_ID).action_hash == plan.action(
        "0", SUPPORT_ACTION_ID
    ).action_hash
    with pytest.raises(ProtocolError, match="hash"):
        replace(plan.action("0", SUPPORT_ACTION_ID), action_hash="0" * 64)


def test_prediction_tasks_and_worker_bind_frozen_action_and_fold_contracts(
    plan,
    crossfit,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_array_path = tmp_path / "source.npy"
    source_array = np.zeros(
        (
            len(CENTERS) * len(TRAINING_SEEDS) * len(GENERATION_SEEDS),
            2 * prediction_worker.MAX_SOURCE_PREFIX_PER_CLASS,
            1,
        ),
        dtype=np.float32,
    )
    np.save(source_array_path, source_array, allow_pickle=False)
    source_index_rows = []
    block_ordinal = 0
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                source_index_rows.append(
                    {
                        "source_center": source,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "block_ordinal": block_ordinal,
                        "stream_id": (
                            f"source-{source}-train-{training_seed}-"
                            f"generation-{generation_seed}"
                        ),
                        "expert_lock_hash": source * 64,
                    }
                )
                block_ordinal += 1
    scratch = write_evaluation_scratch(
        tmp_path / "evaluation.npy",
        tmp_path / "evaluation.json",
        frame=SimpleNamespace(
            embeddings_for=lambda rows: np.zeros((len(rows), 1), dtype=np.float32)
        ),
        crossfit=crossfit,
    )
    config = SimpleNamespace(
        contract_hash="b" * 16,
        classifier=DOWNSTREAM_CLASSIFIER,
    )
    source_cache = SimpleNamespace(
        array_path=source_array_path,
        index_rows=tuple(source_index_rows),
    )

    tasks = build_prediction_tasks(
        config,
        "g" * 16,
        source_cache,
        plan,
        crossfit,
        source_cache_lock_hash="s" * 16,
        scratch=scratch,
        scratch_path=tmp_path / "evaluation.npy",
        checkpoint_root=tmp_path / "checkpoints",
    )

    assert len(tasks) == EXPECTED_PREDICTION_TASK_COUNT == 81
    for task in tasks:
        target = task["target_center"]
        assert tuple(fold["fold_id"] for fold in task["folds"]) == tuple(
            fold.fold_id for fold in crossfit.folds_by_target[target]
        )
        assert all(
            action["outer_target"] == target for action in task["actions"]
        )

    def _fake_fit(
        spec,
        train_embeddings,
        train_labels,
        eval_embeddings,
        *,
        threads,
    ):
        assert threads == 3
        return {
            "predictions": np.zeros(len(eval_embeddings), dtype=np.uint8),
            "probabilities": np.full(
                len(eval_embeddings), 0.25, dtype=np.float32
            ),
            "n_iter": (1,),
            "converged": True,
            "classifier_config_hash": spec.config_hash,
            "scaler_state_hash": "f" * 64,
        }

    monkeypatch.setattr(prediction_worker, "COMMON_OUTPUT_DIM", 1)
    monkeypatch.setattr(prediction_worker, "_fit_classifier", _fake_fit)
    task = tasks[0]
    prediction_worker.prediction_task(task)
    checkpoint = prediction_worker.load_prediction_checkpoint(
        Path(task["checkpoint_json_path"]),
        Path(task["checkpoint_npz_path"]),
        task=task,
    )
    expected_folds = tuple(crossfit.folds_by_target[task["target_center"]])
    cells = tuple(checkpoint["cells"])
    assert len(cells) == len(expected_folds) * EXPECTED_ACTION_COUNT_PER_TARGET
    for fold in expected_folds:
        fold_cells = tuple(
            cell for cell in cells if cell["fold_id"] == fold.fold_id
        )
        assert len(fold_cells) == EXPECTED_ACTION_COUNT_PER_TARGET
        for cell in fold_cells:
            metadata = cell["metadata"]
            assert metadata["fold_ordinal"] == fold.fold_ordinal
            assert metadata["heldout_case_id"] == fold.heldout_case_id
            assert (
                metadata["evaluation_row_identity_hash"]
                == fold.heldout_row_identity_hash
            )
            assert metadata["fold_hash"] == fold.fold_hash


def test_core_has_no_dependency_on_prior_stage90_router_or_policy_runtime() -> None:
    package = (
        Path(__file__).parents[2]
        / "src/midogpp_thesis/cvae/diagnostics/residual_topup_case_oof"
    )
    for name in ("contracts.py", "partitions.py", "ranking.py", "planning.py"):
        source = (package / name).read_text(encoding="utf-8")
        assert "residual_topup_router" not in source
        assert "residual_topup_policy" not in source
