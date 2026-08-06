from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.dense_residual_router.contracts import (
    ACTION_IDS,
    CENTERS,
    EXPECTED_TARGET_PREDICTION_CELL_COUNT,
    EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT,
    GENERATION_SEEDS,
    RHO_VALUES,
    TRAINING_SEEDS,
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router import execution
from midogpp_thesis.cvae.diagnostics.dense_residual_router.execution import (
    CompatibilitySurface,
    LabelFreeValidationFrame,
    PartitionSurface,
    compute_compatibility_surface,
    materialize_target_predictions,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.prediction_io import (
    PREDICTION_INDEX_COLUMNS,
    PredictionAccumulator,
    read_prediction_store,
    write_prediction_store,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.selection import (
    choose_diagnostic_action,
    summarize_development_actions,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router import runner
from midogpp_thesis.cvae.protocol import ProtocolError


def _prediction_metadata() -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": "midogpp_dense_residual_prediction_cell_v1",
        "phase": "target",
        "outer_target": "0",
        "query_center": "0",
        "action_id": "rho_0.00",
        "arm_role": "control",
        "training_seed": 17,
        "generation_seed": 17,
        "candidate_sources_json": '["1","2","3","5","6","7","8","9"]',
        "calibrated_energy_json": "{}",
        "requested_rho": 0.0,
        "applied_rho": 0.0,
        "effective_source_count": 8.0,
        "active_constraints_json": '["rho_zero_uniform"]',
        "weights_json": "{}",
        "allocations_json": "{}",
        "shuffle_seed_by_class_json": '{"0":1,"1":2}',
        "composition_hash": "a" * 16,
        "classifier_config_hash": "b" * 16,
        "scaler_state_hash": "c" * 16,
        "classifier_classes_json": "[0,1]",
        "classifier_n_iter_json": "[3]",
        "classifier_converged": True,
        "evaluation_row_ids_json": '["s1","s2"]',
        "evaluation_row_identity_hash": "d" * 16,
    }
    assert set(PREDICTION_INDEX_COLUMNS).difference(values) == {
        "cell_ordinal",
        "prediction_offset_start",
        "prediction_offset_stop",
        "prediction_sha256",
        "probability_sha256",
        "labels_available_to_fit_or_predict",
        "seed_selection_performed",
    }
    return values


def test_flat_prediction_store_round_trip(tmp_path) -> None:
    accumulator = PredictionAccumulator()
    accumulator.append(
        predictions=np.asarray([0, 1], dtype=np.uint8),
        probabilities=np.asarray([0.2, 0.8], dtype=np.float32),
        metadata=_prediction_metadata(),
    )
    store = accumulator.finish()
    path = tmp_path / "predictions.npz"
    write_prediction_store(path, store)
    loaded = read_prediction_store(path, store.index_rows)
    assert loaded.y_pred.tolist() == [0, 1]
    assert loaded.prob_pos.tolist() == pytest.approx([0.2, 0.8])


def test_workload_geometry_is_derived_from_frozen_contract() -> None:
    seed_cell_count = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
    assert EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT == (
        len(CENTERS) * (len(CENTERS) - 1) * len(ACTION_IDS) * seed_cell_count
    )
    assert EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT == (
        len(CENTERS) * len(ACTION_IDS) * seed_cell_count
    )
    assert EXPECTED_TOTAL_CLASSIFIER_FIT_COUNT == (
        EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
        + EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT
    )
    assert EXPECTED_TARGET_PREDICTION_CELL_COUNT == (
        len(CENTERS) * (len(ACTION_IDS) + 1) * seed_cell_count
    )


def test_fixed_action_objective_can_select_positive_residual() -> None:
    outer = "0"
    queries = tuple(center for center in CENTERS if center != outer)
    metrics = []
    index = []
    bacc_by_action = {
        "rho_0.00": 0.70,
        "rho_0.25": 0.72,
        "rho_0.50": 0.69,
    }
    for action_id, rho in zip(ACTION_IDS, RHO_VALUES, strict=True):
        for query in queries:
            sources = tuple(center for center in CENTERS if center not in {outer, query})
            uniform = 1.0 / len(sources)
            weights = {source: uniform for source in sources}
            if rho > 0:
                weights[sources[0]] += 0.01 * rho
                weights[sources[1]] -= 0.01 * rho
            for training_seed in (17, 42, 101):
                for generation_seed in (17, 42, 101):
                    base = {
                        "phase": "development",
                        "outer_target": outer,
                        "query_center": query,
                        "action_id": action_id,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                    }
                    metrics.append(
                        {
                            **base,
                            "bacc": bacc_by_action[action_id],
                            "macro_f1": bacc_by_action[action_id],
                        }
                    )
                    index.append(
                        {
                            **base,
                            "weights_json": __import__("json").dumps(weights),
                        }
                    )
    summaries = summarize_development_actions(metrics, index, outer_target=outer)
    selected = choose_diagnostic_action(summaries, outer_target=outer)
    assert selected["selected_action_id"] == "rho_0.25"
    assert selected["selected_mean_paired_bacc_delta_vs_control"] == pytest.approx(0.02)
    assert selected["fallback_applied"] is False


def test_compatibility_execution_never_scores_query_evaluation_rows(monkeypatch) -> None:
    rows = []
    rows_by_center = {}
    support_by_center = {}
    evaluation_by_center = {}
    ordinal = 0
    for center in CENTERS:
        center_rows = []
        support_rows = []
        for index, role in enumerate(("support", "support", "evaluation")):
            row = ValidationRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=ordinal,
                sample_id=f"{center}-s{index}",
                case_id=f"{center}-c{index}",
                center=center,
                partition_role=role,
            )
            rows.append(row)
            center_rows.append(row)
            if role == "support":
                support_rows.append(row)
            ordinal += 1
        rows_by_center[center] = tuple(center_rows)
        support_by_center[center] = tuple(support_rows)
        evaluation_by_center[center] = (center_rows[-1],)
    frame = LabelFreeValidationFrame(
        embeddings=np.ones((len(rows), 3840), dtype=np.float32),
        rows=tuple(
            ValidationRowIdentity(
                row_ordinal=row.row_ordinal,
                manifest_row_index=row.manifest_row_index,
                sample_id=row.sample_id,
                case_id=row.case_id,
                center=row.center,
                partition_role="evaluation",
            )
            for row in rows
        ),
        rows_by_center={
            center: tuple(
                ValidationRowIdentity(
                    row_ordinal=row.row_ordinal,
                    manifest_row_index=row.manifest_row_index,
                    sample_id=row.sample_id,
                    case_id=row.case_id,
                    center=row.center,
                    partition_role="evaluation",
                )
                for row in center_rows
            )
            for center, center_rows in rows_by_center.items()
        },
        cache_binding={"status": "test"},
    )
    partitions = PartitionSurface(
        support_rows_by_center=support_by_center,
        evaluation_rows_by_center=evaluation_by_center,
        table_rows=(),
        lock_payload={"support_partition_lock_hash": "1" * 16},
    )
    calls = []

    def fake_expert_loader(*args, source_center, training_seed, **kwargs):
        return SimpleNamespace(source_center=source_center, training_seed=training_seed)

    def fake_score(expert, embeddings, case_ids):
        assert len(case_ids) == 2
        assert all(not case_id.endswith("c2") for case_id in case_ids)
        calls.extend(case_ids)
        values = np.asarray([1.0, 2.0], dtype=float)
        return SimpleNamespace(
            case_order=tuple(case_ids),
            per_case=dict(zip(case_ids, values, strict=True)),
            per_class_energy={0: values, 1: values + 0.1},
            per_class_reconstruction_mse={0: values, 1: values},
            per_class_normalized_ps_kl={0: values * 0.0, 1: values * 0.0},
        )

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.dense_residual_router.execution.load_routing_authorized_expert",
        fake_expert_loader,
    )
    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.dense_residual_router.execution.score_variational_compatibility",
        fake_score,
    )
    config = SimpleNamespace(
        expert_bank_root="bank",
        compatibility_device="cpu",
    )
    surface = compute_compatibility_surface(config, frame, partitions)
    assert calls
    assert all(row["query_partition_role"] == "support" for row in surface.case_rows)
    assert len(surface.score_rows) == len(CENTERS) * (len(CENTERS) - 1)


def test_target_control_alias_reuses_rho0_classifier_fit(monkeypatch) -> None:
    rows = tuple(
        ValidationRowIdentity(
            row_ordinal=index,
            manifest_row_index=index,
            sample_id=f"sample-{center}",
            case_id=f"case-{center}",
            center=center,
        )
        for index, center in enumerate(CENTERS)
    )
    frame = LabelFreeValidationFrame(
        embeddings=np.ones((len(rows), 3840), dtype=np.float32),
        rows=rows,
        rows_by_center={center: (rows[index],) for index, center in enumerate(CENTERS)},
        cache_binding={"status": "test"},
    )
    partitions = PartitionSurface(
        support_rows_by_center={center: () for center in CENTERS},
        evaluation_rows_by_center={
            center: (rows[index],) for index, center in enumerate(CENTERS)
        },
        table_rows=(),
        lock_payload={"support_partition_lock_hash": "1" * 16},
    )
    compatibility = CompatibilitySurface(
        calibrated_energy_by_query={
            query: {
                source: float(source_index)
                for source_index, source in enumerate(CENTERS)
                if source != query
            }
            for query in CENTERS
        },
        case_rows=(),
        score_rows=(),
    )
    key_map = {
        (source, training_seed, generation_seed): SimpleNamespace(
            stream_id=f"stream-{source}-{training_seed}-{generation_seed}"
        )
        for source in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    fit_calls = 0
    composition_calls = 0

    def fake_fit(*args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return {
            "predictions": np.asarray([0], dtype=np.uint8),
            "probabilities": np.asarray([0.5], dtype=np.float32),
            "classes": (0, 1),
            "n_iter": (1,),
            "converged": True,
            "classifier_config_hash": "2" * 16,
            "scaler_state_hash": "3" * 16,
        }

    def fake_compose(*args, **kwargs):
        nonlocal composition_calls
        composition_calls += 1
        return SimpleNamespace(
            embeddings=np.zeros((2, 1), dtype=np.float32),
            labels=np.asarray([0, 1], dtype=np.int64),
            composition_hash="4" * 16,
        )

    monkeypatch.setattr(execution, "_generation_key_map", lambda lock: key_map)
    monkeypatch.setattr(
        execution,
        "_generate_seed_cell_blocks",
        lambda *args, **kwargs: {source: object() for source in CENTERS},
    )
    monkeypatch.setattr(execution, "compose_prefix_blocks", fake_compose)
    monkeypatch.setattr(execution, "_fit_classifier", fake_fit)
    config = SimpleNamespace(generation_device="cpu")
    generation_lock = SimpleNamespace(generation_lock_hash="5" * 16)

    surface = materialize_target_predictions(
        config,
        generation_lock,
        frame,
        partitions,
        compatibility,
    )

    assert fit_calls == EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT
    assert composition_calls == EXPECTED_TARGET_UNIQUE_CLASSIFIER_FIT_COUNT
    assert len(surface.store.index_rows) == EXPECTED_TARGET_PREDICTION_CELL_COUNT
    by_key = {
        (
            row["outer_target"],
            row["arm_role"],
            row["action_id"],
            row["training_seed"],
            row["generation_seed"],
        ): row
        for row in surface.store.index_rows
    }
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                selected = by_key[
                    (target, "selected", "rho_0.00", training_seed, generation_seed)
                ]
                control = by_key[
                    (target, "control", "rho_0.00", training_seed, generation_seed)
                ]
                assert selected["prediction_sha256"] == control["prediction_sha256"]
                assert selected["probability_sha256"] == control["probability_sha256"]
                assert selected["composition_hash"] == control["composition_hash"]


def test_validation_report_publisher_calls_expensive_validator_once(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def fake_validate(root, *, config, allow_pending=False):
        calls.append((root, config, allow_pending))
        return {"status": "PASS", "reconstructed": True}

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.dense_residual_router.validation."
        "validate_dense_residual_router_bundle",
        fake_validate,
    )
    config = SimpleNamespace()
    (tmp_path / "reports").mkdir()

    runner._validate_and_publish_report_once(tmp_path, config=config)

    assert calls == [(tmp_path, config, True)]
    payload = __import__("json").loads(
        (tmp_path / "reports/validation_report.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "PASS"
    assert payload["checks"]["reconstructed"] is True


def test_runner_rejects_unresolved_artifact_uris() -> None:
    unresolved = SimpleNamespace(
        artifact_root=Path("output:/dense-router"),
        expert_bank_root=Path("artifact:/expert-bank"),
        generation_lock_root=Path("artifact:/generation-lock"),
        validation_cache_root=Path("artifact:/validation-cache"),
        validation_manifest_path=Path("artifact:/validation-manifest/manifest.csv"),
    )
    with pytest.raises(ProtocolError, match="workspace-resolved config"):
        runner._assert_workspace_resolved_paths(unresolved)
