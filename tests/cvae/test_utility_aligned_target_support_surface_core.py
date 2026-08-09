from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
    SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
    SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS,
    TargetCandidateComponents,
    build_target_feature_production,
    target_feature_production_from_payload,
)
from midogpp_thesis.cvae.routing.utility_aligned.target_features import target_sources
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface import runner
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface import (
    production as target_support_production_module,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.artifact_writer import (
    feature_payload,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.contracts import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.inputs import (
    parse_support_rows,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_checkpoint import (
    load_action_probe_checkpoint,
    sha256_file,
    write_action_probe_checkpoint,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_contracts import (
    ACTION_SHIFT_LOCK_SCHEMA,
    ActionProbeCheckpoint,
    ActionProbeRuntime,
    ActionProbeTask,
    TargetSupportActionShiftRow,
    workstation_action_probe_runtime,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface import (
    action_probe_execution as action_probe_execution_module,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface import (
    action_probe_worker as action_probe_worker_module,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_execution import (
    execute_or_resume_action_probes,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_surface import (
    ACTION_SHIFT_LOCK_KEYS,
    build_action_shift_lock,
    build_task_action_shift_rows,
    validate_action_shift_surface,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.action_probe_worker import (
    compose_target_action,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.artifact_writer import (
    write_csv,
    write_json,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.bundle_validation import (
    _validate_action_shifts,
)
from midogpp_thesis.cvae.routing.utility_aligned_target_support_surface.config import (
    load_utility_aligned_target_support_surface_config,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.config import CLASSIFIER
from midogpp_thesis.cvae.routing.utility_aligned_identities import CENTERS


SEEDS = (17, 42, 101)


def _fixture_case_ensemble_shift_hash(
    target: str,
    source: str,
    case_id: str,
    case_row_identity_hash: str,
) -> str:
    base_hashes = tuple(
        canonical_sha256([target, source, left, right, case_id, "base"])
        for left, right in ENSEMBLE_SEED_KEYS
    )
    tail_hashes = tuple(
        canonical_sha256([target, source, left, right, case_id, "tail"])
        for left, right in ENSEMBLE_SEED_KEYS
    )
    base_ensemble = canonical_sha256([target, source, case_id, "base-ensemble"])
    tail_ensemble = canonical_sha256([target, source, case_id, "tail-ensemble"])
    difference = canonical_sha256([target, source, case_id, "difference"])
    return canonical_sha256(
        {
            "schema_version": SUPPORT_ACTION_PROBABILITY_SHIFT_SCHEMA,
            "row_identity_hash": case_row_identity_hash,
            "seed_pair_count": 9,
            "seed_keys": [list(key) for key in ENSEMBLE_SEED_KEYS],
            "base_component_vector_hashes": list(base_hashes),
            "tail_component_vector_hashes": list(tail_hashes),
            "per_seed_mean_absolute_shifts": [0.1] * 9,
            "technical_seed_spread_semantics": (
                SUPPORT_ACTION_TECHNICAL_SEED_SPREAD_SEMANTICS
            ),
            "technical_seed_values_may_feed_model": False,
            "base_ensemble_probability_sha256": base_ensemble,
            "tail_ensemble_probability_sha256": tail_ensemble,
            "ensemble_absolute_difference_sha256": difference,
            "value": 0.1,
            "seed_standard_deviation": 0.0,
            "seed_minimum": 0.1,
            "seed_maximum": 0.1,
            "seed_range": 0.0,
            "scalar_name": SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
            "scalar_semantics": SUPPORT_ACTION_PROBABILITY_SHIFT_SEMANTICS,
            "labels_used": False,
        }
    )


def _action_task(
    tmp_path: Path,
    *,
    ordinal: int = 0,
    target: str = "0",
    training_seed: int = 17,
    generation_seed: int = 17,
    two_rows_per_case: bool = True,
) -> ActionProbeTask:
    support_case_ids = tuple(
        f"{target}-case-{case:02d}"
        for case in range(8)
        for _ in range(2 if two_rows_per_case else 1)
    )
    support_sample_ids = tuple(
        f"{target}-sample-{index:02d}" for index in range(len(support_case_ids))
    )
    support_path = tmp_path / f"support-{target}.npy"
    if not support_path.is_file():
        with support_path.open("wb") as handle:
            np.save(
                handle,
                np.zeros((len(support_case_ids), 3840), dtype=np.float32),
                allow_pickle=False,
            )
    sources = tuple(center for center in CENTERS if center != target)
    return ActionProbeTask(
        task_ordinal=ordinal,
        target_id=target,
        training_seed=training_seed,
        generation_seed=generation_seed,
        candidate_sources=sources,
        support_array_path=str(support_path),
        support_file_sha256=sha256_file(support_path),
        support_partition_hash=canonical_sha256(
            {"target": target, "support": list(support_sample_ids)}
        ),
        support_case_ids=support_case_ids,
        support_sample_ids=support_sample_ids,
        source_array_path_by_source={
            source: str(tmp_path / f"source-{source}.npy") for source in sources
        },
        source_file_sha256_by_source={source: "a" * 64 for source in sources},
        generated_cache_hash="b" * 16,
        classifier_payload=CLASSIFIER.to_payload(),
        runtime=workstation_action_probe_runtime(),
        checkpoint_root=str(tmp_path / "checkpoints"),
    )


def _components(target: str = "0") -> dict[str, TargetCandidateComponents]:
    case_ids = tuple(f"case-{index:02d}" for index in range(8))
    result = {}
    for source_index, source in enumerate(target_sources(target)):
        reconstruction = {
            seed: {
                label: np.linspace(0.1, 0.8, 8, dtype=np.float64)
                + 0.01 * source_index
                + 0.001 * label
                for label in (0, 1)
            }
            for seed in SEEDS
        }
        kl = {
            seed: {
                label: np.linspace(0.05, 0.4, 8, dtype=np.float64)
                + 0.005 * source_index
                + 0.001 * label
                for label in (0, 1)
            }
            for seed in SEEDS
        }
        support_means = {
            case_id: np.full(3840, float(index), dtype=np.float64)
            for index, case_id in enumerate(case_ids)
        }
        generated_means = {
            (training_seed, generation_seed): np.full(
                3840,
                float(source_index) + training_seed / 1000 + generation_seed / 10000,
                dtype=np.float64,
            )
            for training_seed in SEEDS
            for generation_seed in SEEDS
        }
        result[source] = TargetCandidateComponents(
            candidate_source=source,
            reconstruction_by_training_seed=reconstruction,
            normalized_ps_kl_by_training_seed=kl,
            support_case_mean_embeddings=support_means,
            generated_mean_by_seed_pair=generated_means,
            metadata_similarity=0.5,
        )
    return result


def _production():
    return build_target_feature_production(
        target_id="0",
        case_ids=tuple(f"case-{index:02d}" for index in range(8)),
        components_by_source=_components(),
        bootstrap_seed=60920000,
        bootstrap_replicate_count=32,
    )


def test_label_free_target_features_use_exact_seed_grid_and_case_mmd_bootstrap() -> None:
    production = _production()

    assert len(production.point_rows) == 8 * 3 * 3
    assert len(production.bootstrap_surfaces) == 32
    point_mmd = production.point_rows[0].distribution_mmd
    bootstrap_mmd = {
        surface.rows[0].distribution_mmd for surface in production.bootstrap_surfaces
    }
    assert any(value != point_mmd for value in bootstrap_mmd)
    assert len(bootstrap_mmd) > 1


def test_target_components_reject_noncanonical_training_seed() -> None:
    components = _components()
    source = next(iter(components))
    value = components[source]
    bad = dict(value.reconstruction_by_training_seed)
    bad[19] = bad.pop(17)
    with pytest.raises(ProtocolError, match="exact seeds"):
        replace(value, reconstruction_by_training_seed=bad)


def test_target_payload_malformed_numeric_fails_as_protocol_error() -> None:
    payload = dict(feature_payload(_production()))
    point_rows = [dict(value) for value in payload["point_rows"]]
    point_rows[0]["training_seed"] = {"not": "numeric"}
    payload["point_rows"] = point_rows
    payload["target_feature_hash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "target_feature_hash"}
    )
    with pytest.raises(ProtocolError, match="malformed|numeric"):
        target_feature_production_from_payload(payload)


def test_target_support_reservation_rejects_cross_center_case_reuse() -> None:
    centers = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    cases = {
        center: [f"{center}-case-{index:02d}" for index in range(8)]
        for center in centers
    }
    cases["1"][0] = cases["0"][0]
    rows = {
        center: [
            {
                "row_ordinal": index,
                "sample_id": f"{center}-sample-{index}",
                "case_id": case_id,
                "center": center,
                "cache_shard_path": f"shards/{center}.npy",
                "cache_row_index": index,
            }
            for index, case_id in enumerate(values)
        ]
        for center, values in cases.items()
    }
    with pytest.raises(ProtocolError, match="unique cases"):
        parse_support_rows(
            {
                "support_case_ids_by_center": cases,
                "support_rows_by_center": rows,
            }
        )


def test_target_support_complete_fast_path_and_incomplete_complete_guard(
    monkeypatch, tmp_path: Path
) -> None:
    config = SimpleNamespace(artifact_root=tmp_path)
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "validate_target_support_surface_bundle",
        lambda root: {"status": "COMPLETE", "root": str(root)},
    )
    monkeypatch.setattr(
        runner,
        "require_target_support_inputs_ready",
        lambda _config: (_ for _ in ()).throw(AssertionError("fresh input reopened")),
    )
    result = runner.run_utility_aligned_target_support_surface(
        config, workspace_validator=lambda _config: None
    )
    assert result["status"] == "COMPLETE"

    missing = tmp_path / REQUIRED_FILES[-1]
    missing.unlink()
    (tmp_path / "reports/run_state.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="COMPLETE artifact is incomplete"):
        runner.run_utility_aligned_target_support_surface(
            config, workspace_validator=lambda _config: None
        )


def test_action_probe_geometry_is_exact_base_plus_single_source_tail() -> None:
    sources = tuple(center for center in CENTERS if center != "0")
    shared = np.zeros((540, 3840), dtype=np.float32)
    arrays = {source: shared for source in sources}

    base_x, base_y = compose_target_action(arrays, selected_source=None)
    tail_x, tail_y = compose_target_action(arrays, selected_source=sources[3])

    assert base_x.shape == (2 * 8 * 128, 3840)
    assert tail_x.shape == (2 * (8 * 128 + 128), 3840)
    assert np.bincount(base_y).tolist() == [1024, 1024]
    assert np.bincount(tail_y).tolist() == [1152, 1152]


def test_action_probe_checkpoint_is_hash_valid_and_recoverable(
    tmp_path: Path,
) -> None:
    task = _action_task(tmp_path)
    probabilities = np.full(
        (9, len(task.support_case_ids)), 0.25, dtype=np.float32
    )
    written = write_action_probe_checkpoint(task, probabilities)
    assert load_action_probe_checkpoint(task) == written

    probability_path = Path(task.checkpoint_root) / written.probability_member
    probability_path.write_bytes(b"tampered")
    with pytest.raises(ProtocolError, match="failed validation"):
        load_action_probe_checkpoint(task)

    recovered = write_action_probe_checkpoint(task, probabilities)
    assert load_action_probe_checkpoint(task) == recovered


def test_action_probe_resume_skips_all_81_completed_tasks(tmp_path: Path) -> None:
    tasks = []
    ordinal = 0
    for target in CENTERS:
        for training_seed in SEEDS:
            for generation_seed in SEEDS:
                task = _action_task(
                    tmp_path,
                    ordinal=ordinal,
                    target=target,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    two_rows_per_case=False,
                )
                tasks.append(task)
                write_action_probe_checkpoint(
                    task,
                    np.full((9, 8), 0.5, dtype=np.float32),
                )
                ordinal += 1

    resumed = execute_or_resume_action_probes(
        tasks,
        runtime=workstation_action_probe_runtime(),
    )

    assert len(resumed) == 81
    assert tuple(value.task_hash for value in resumed) == tuple(
        task.task_hash for task in tasks
    )


def test_action_probe_runtime_reaches_executor_and_task_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = workstation_action_probe_runtime()
    tasks = tuple(
        _action_task(
            tmp_path,
            ordinal=ordinal,
            target=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
            two_rows_per_case=False,
        )
        for ordinal, (target, training_seed, generation_seed) in enumerate(
            (target, training_seed, generation_seed)
            for target in CENTERS
            for training_seed in SEEDS
            for generation_seed in SEEDS
        )
    )
    observed: dict[str, object] = {"submitted": []}

    class _Future:
        def __init__(self, task: ActionProbeTask) -> None:
            self.task = task

        def result(self) -> ActionProbeCheckpoint:
            return ActionProbeCheckpoint(
                task_hash=self.task.task_hash,
                checkpoint_hash="c" * 64,
                probability_member="unused.npy",
                probability_file_sha256="d" * 64,
                action_ids=(
                    "B",
                    *tuple(
                        f"Hxe::{source}" for source in self.task.candidate_sources
                    ),
                ),
                support_row_count=len(self.task.support_case_ids),
            )

    class _Pool:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            observed["max_workers"] = max_workers
            observed["mp_context"] = mp_context

        def __enter__(self) -> "_Pool":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(self, _worker: object, task: ActionProbeTask) -> _Future:
            submitted = observed["submitted"]
            assert isinstance(submitted, list)
            submitted.append(task.runtime)
            return _Future(task)

    monkeypatch.setattr(
        action_probe_execution_module,
        "load_action_probe_checkpoint",
        lambda _task: None,
    )
    monkeypatch.setattr(
        action_probe_execution_module,
        "ProcessPoolExecutor",
        _Pool,
    )
    monkeypatch.setattr(
        action_probe_execution_module,
        "as_completed",
        lambda futures: tuple(futures),
    )
    monkeypatch.setattr(
        action_probe_execution_module.mp,
        "get_context",
        lambda method: observed.setdefault("start_method", method),
    )

    completed = execute_or_resume_action_probes(tasks, runtime=runtime)

    assert len(completed) == runtime.task_count == 81
    assert observed["max_workers"] == runtime.classifier_workers == 4
    assert observed["start_method"] == runtime.multiprocessing_start_method == "spawn"
    assert observed["submitted"] == [runtime] * runtime.task_count

    restored = pickle.loads(pickle.dumps(tasks[0]))
    assert restored.task_hash == tasks[0].task_hash
    assert restored.runtime == runtime

    with pytest.raises(ProtocolError, match="runtime drifted"):
        ActionProbeRuntime(
            classifier_workers=2,
            threads_per_worker=6,
            task_count=81,
            fit_count=729,
            multiprocessing_start_method="spawn",
        )
    with pytest.raises(ProtocolError, match="task hash drifted"):
        replace(tasks[0], task_hash="0" * 64)
    with pytest.raises(ProtocolError, match="runtime drifted"):
        ActionProbeRuntime(
            classifier_workers=4,
            threads_per_worker=3,
            task_count=81,
            fit_count=729,
            multiprocessing_start_method="spawn",
            gpu_cpu_overlap_allowed=True,
        )


def test_action_probe_runtime_reaches_threadpool_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _action_task(tmp_path)
    support = np.zeros((len(task.support_case_ids), 3840), dtype=np.float32)
    source = np.zeros((540, 3840), dtype=np.float32)
    limits_seen: list[int] = []
    fit_calls: list[int] = []

    class _LimitContext:
        def __init__(self, limits: int) -> None:
            limits_seen.append(limits)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    def _fit(*_args: object, **_kwargs: object) -> SimpleNamespace:
        fit_calls.append(1)
        probabilities = np.full((len(support), 2), 0.5, dtype=np.float64)
        return SimpleNamespace(
            probabilities=probabilities,
            classes=(0, 1),
            converged=True,
            classifier_config_hash=CLASSIFIER.config_hash,
            scaler_state_hash="sealed",
        )

    monkeypatch.setattr(
        action_probe_worker_module,
        "_load_support",
        lambda *_args, **_kwargs: support,
    )
    monkeypatch.setattr(
        action_probe_worker_module,
        "_load_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        action_probe_worker_module,
        "compose_target_action",
        lambda *_args, **_kwargs: (
            np.zeros((2, 3840), dtype=np.float32),
            np.asarray([0, 1], dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        action_probe_worker_module,
        "fit_logistic_classifier",
        _fit,
    )
    monkeypatch.setattr(
        action_probe_worker_module,
        "write_action_probe_checkpoint",
        lambda task, _probabilities: ActionProbeCheckpoint(
            task_hash=task.task_hash,
            checkpoint_hash="c" * 64,
            probability_member="unused.npy",
            probability_file_sha256="d" * 64,
            action_ids=(
                "B",
                *tuple(f"Hxe::{value}" for value in task.candidate_sources),
            ),
            support_row_count=len(task.support_case_ids),
        ),
    )
    import threadpoolctl

    monkeypatch.setattr(
        threadpoolctl,
        "threadpool_limits",
        lambda *, limits: _LimitContext(limits),
    )

    result = action_probe_worker_module.action_probe_worker(task)

    assert result.task_hash == task.task_hash
    assert limits_seen == [task.runtime.threads_per_worker] == [3]
    assert len(fit_calls) == task.runtime.fits_per_task == 9


def test_action_shift_aggregates_within_case_and_lock_rejects_label_gates(
    tmp_path: Path,
) -> None:
    tasks = tuple(
        _action_task(
            tmp_path,
            ordinal=index,
            training_seed=training_seed,
            generation_seed=generation_seed,
        )
        for index, (training_seed, generation_seed) in enumerate(
            (left, right) for left in SEEDS for right in SEEDS
        )
    )
    checkpoints = []
    for index, task in enumerate(tasks):
        probabilities = np.full(
            (9, len(task.support_case_ids)), 0.5, dtype=np.float32
        )
        direction = 0.4 if index < 4 else -0.4 if index < 8 else 0.0
        probabilities[1:] += direction
        checkpoints.append(write_action_probe_checkpoint(task, probabilities))

    rows = build_task_action_shift_rows(tasks, tuple(checkpoints))

    assert len(rows) == 8 * 8 * 9
    assert max(
        row.descriptive_seed_mean_absolute_positive_probability_shift
        for row in rows
    ) == pytest.approx(0.4)
    assert rows[0].case_ensemble_mean_absolute_positive_probability_shift == pytest.approx(
        0.0, abs=1.0e-7
    )
    assert rows[0].query_id == rows[0].outer_target_id == "0"
    assert rows[0].labels_used is False

    table = tmp_path / "target_support_action_shifts.csv"
    write_csv(table, [row.to_payload() for row in rows])
    lock = build_action_shift_lock(
        rows=rows,
        table_path=table,
        support_reservation_hash="1" * 64,
        target_support_cache_binding_hash="2" * 64,
        source_generation_lock_hash="3" * 64,
        generated_cache_hash="4" * 64,
        runtime=workstation_action_probe_runtime(),
    )
    assert set(lock) == ACTION_SHIFT_LOCK_KEYS
    assert lock["schema_version"] == ACTION_SHIFT_LOCK_SCHEMA
    validate_action_shift_surface(rows=rows, lock=lock, table_path=table)

    bad = dict(lock)
    bad["seeds_selected_by_support"] = True
    bad["shift_lock_hash"] = canonical_sha256(
        {key: value for key, value in bad.items() if key != "shift_lock_hash"}
    )
    with pytest.raises(ProtocolError, match="lock drifted"):
        validate_action_shift_surface(rows=rows, lock=bad, table_path=table)

    bad_topology = dict(lock)
    bad_topology["action_geometry_hash"] = "f" * 64
    bad_topology["shift_lock_hash"] = canonical_sha256(
        {
            key: value
            for key, value in bad_topology.items()
            if key != "shift_lock_hash"
        }
    )
    with pytest.raises(ProtocolError, match="lock drifted"):
        validate_action_shift_surface(
            rows=rows,
            lock=bad_topology,
            table_path=table,
        )


def test_target_support_v2_config_locks_workstation_action_probe() -> None:
    config = load_utility_aligned_target_support_surface_config(
        Path(
            "experiments/midogpp/stages/60_routing_and_composition/configs/"
            "uniform_b_v2_utility_aligned_target_support_surface_v1.yaml"
        )
    )

    assert config.runtime["action_probe_classifier_workers"] == 4
    assert config.runtime["action_probe_threads_per_worker"] == 3
    assert config.runtime["action_probe_task_count"] == 81
    assert config.runtime["action_probe_fit_count"] == 729
    assert config.runtime["target_local_scalar_name"] == (
        "mean_support_row_absolute_exact_nine_ensemble_probability_shift_v2"
    )
    assert config.action_probe_runtime.to_payload() == {
        "schema_version": "midogpp_target_support_action_probe_runtime_v1",
        "classifier_workers": 4,
        "threads_per_worker": 3,
        "task_count": 81,
        "fits_per_task": 9,
        "fit_count": 729,
        "multiprocessing_start_method": "spawn",
        "execution_order": "source_generation_then_cpu_action_probe",
        "gpu_cpu_overlap_allowed": False,
    }


def test_target_support_production_finishes_generation_before_cpu_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = workstation_action_probe_runtime()
    config = SimpleNamespace(action_probe_runtime=runtime)
    inputs = object()
    generated = object()
    productions = (object(),)
    events: list[str] = []

    def _build(_config: object) -> tuple[object, object, tuple[object, ...]]:
        events.extend(("source_generation_started", "source_generation_completed"))
        return inputs, generated, productions

    def _probe(
        actual_inputs: object,
        actual_generated: object,
        *,
        execution_root: Path,
        runtime: ActionProbeRuntime,
    ) -> tuple[object, ...]:
        assert events == ["source_generation_started", "source_generation_completed"]
        assert actual_inputs is inputs
        assert actual_generated is generated
        assert execution_root == tmp_path
        assert runtime == config.action_probe_runtime
        events.append("cpu_action_probe_started")
        return (object(),)

    monkeypatch.setattr(
        target_support_production_module,
        "build_all_target_features",
        _build,
    )
    monkeypatch.setattr(
        target_support_production_module,
        "execution_root_for",
        lambda _config: tmp_path,
    )
    monkeypatch.setattr(
        target_support_production_module,
        "materialize_target_action_shifts",
        _probe,
    )
    monkeypatch.setattr(
        target_support_production_module,
        "persist_target_support_artifact",
        lambda *_args: tmp_path,
    )
    monkeypatch.setattr(
        target_support_production_module,
        "validate_target_support_surface_bundle",
        lambda root: events.append(f"validated:{root.name}"),
    )

    root = target_support_production_module.materialize_target_support_surface(config)

    assert root == tmp_path
    assert events == [
        "source_generation_started",
        "source_generation_completed",
        "cpu_action_probe_started",
        f"validated:{tmp_path.name}",
    ]


def test_action_shift_bundle_reconstructs_exact_target_case_grid(
    tmp_path: Path,
) -> None:
    reservation_hash = "1" * 64
    cache_binding_hash = "2" * 64
    source_generation_lock_hash = "3" * 64
    upstream_cache_hash = "d" * 16
    cases = {
        target: [f"{target}-case-{index:02d}" for index in range(8)]
        for target in CENTERS
    }
    support_rows = {
        target: [
            {
                "row_ordinal": index,
                "sample_id": f"{target}-sample-{index:02d}",
                "case_id": case_id,
                "center": target,
                "cache_shard_path": f"shards/{target}.npy",
                "cache_row_index": index,
            }
            for index, case_id in enumerate(cases[target])
        ]
        for target in CENTERS
    }
    rows = []
    for target in CENTERS:
        partition_hash = canonical_sha256(
            {
                "schema_version": "midogpp_target_support_partition_bridge_v1",
                "parent_reservation_hash": reservation_hash,
                "target": target,
                "case_ids": cases[target],
                "ordered_sample_ids": [
                    value["sample_id"] for value in support_rows[target]
                ],
                "ordered_case_ids": [
                    value["case_id"] for value in support_rows[target]
                ],
            }
        )
        for source in CENTERS:
            if source == target:
                continue
            for training_seed in SEEDS:
                for generation_seed in SEEDS:
                    for case_id in cases[target]:
                        case_index = cases[target].index(case_id)
                        case_identity_hash = canonical_sha256(
                            {
                                "schema_version": (
                                    "midogpp_utility_aligned_target_support_"
                                    "case_rows_v1"
                                ),
                                "outer_target_id": target,
                                "case_id": case_id,
                                "ordered_sample_ids": [
                                    support_rows[target][case_index]["sample_id"]
                                ],
                            }
                        )
                        rows.append(
                            TargetSupportActionShiftRow(
                                outer_target_id=target,
                                query_id=target,
                                candidate_source=source,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                case_id=case_id,
                                support_partition_hash=partition_hash,
                                case_row_identity_hash=case_identity_hash,
                                support_row_count=1,
                                base_probability_sha256="a" * 64,
                                tail_probability_sha256="b" * 64,
                                base_component_vector_hash=canonical_sha256(
                                    [
                                        target,
                                        source,
                                        training_seed,
                                        generation_seed,
                                        case_id,
                                        "base",
                                    ]
                                ),
                                tail_component_vector_hash=canonical_sha256(
                                    [
                                        target,
                                        source,
                                        training_seed,
                                        generation_seed,
                                        case_id,
                                        "tail",
                                    ]
                                ),
                                descriptive_seed_mean_absolute_positive_probability_shift=0.1,
                                case_ensemble_mean_absolute_positive_probability_shift=0.1,
                                case_base_ensemble_probability_sha256=canonical_sha256(
                                    [target, source, case_id, "base-ensemble"]
                                ),
                                case_tail_ensemble_probability_sha256=canonical_sha256(
                                    [target, source, case_id, "tail-ensemble"]
                                ),
                                case_ensemble_absolute_difference_sha256=canonical_sha256(
                                    [target, source, case_id, "difference"]
                                ),
                                case_ensemble_shift_hash=_fixture_case_ensemble_shift_hash(
                                    target,
                                    source,
                                    case_id,
                                    case_identity_hash,
                                ),
                            )
                        )
    ordered = tuple(sorted(rows, key=lambda row: row.row_key))
    table = tmp_path / "tables/target_support_action_shifts.csv"
    write_csv(table, [row.to_payload() for row in ordered])
    generated_binding = canonical_sha256(
        {
            "schema_version": "midogpp_target_support_generated_cache_binding_v1",
            "upstream_cache_hash": upstream_cache_hash,
        }
    )
    lock = build_action_shift_lock(
        rows=ordered,
        table_path=table,
        support_reservation_hash=reservation_hash,
        target_support_cache_binding_hash=cache_binding_hash,
        source_generation_lock_hash=source_generation_lock_hash,
        generated_cache_hash=generated_binding,
        runtime=workstation_action_probe_runtime(),
    )
    write_json(tmp_path / "manifests/target_support_action_shifts_lock.json", lock)

    validated = _validate_action_shifts(
        tmp_path,
        reservation={
            "reservation_hash": reservation_hash,
            "support_case_ids_by_center": cases,
            "support_rows_by_center": support_rows,
        },
        cache={"cache_binding_hash": cache_binding_hash},
        generation={
            "source_generation_lock_hash": source_generation_lock_hash,
            "generated_cache_hash": upstream_cache_hash,
        },
    )

    assert validated["row_count"] == 9 * 8 * 9 * 8
