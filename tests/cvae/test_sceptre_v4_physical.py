from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.physical import (
    prediction_surface as predictions,
    source_streams as streams,
    worker_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.persistence import (
    source_store_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution import (
    worker_runtime as execution_worker_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.physical.prediction_surface import (
    CANDIDATE_EXCLUSION_SENTINEL,
    FitOutcome,
    LOCKED_CLASSIFIER_SPEC,
    PRODUCTION_PREDICTION_GEOMETRY,
    PredictionGeometry,
    PredictionRuntimeTestMode,
    exact_b_source_centers,
    materialize_prediction_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.physical.source_streams import (
    PRODUCTION_SOURCE_GEOMETRY,
    SourceGeometry,
    SourceRuntimeTestMode,
    materialize_source_streams,
)
from midogpp_thesis.cvae.protocol import ProtocolError


TEST_GEOMETRY = SourceGeometry(
    centers=("0", "1", "2"),
    training_seeds=(17, 42),
    generation_seeds=(17, 42),
    rows_per_class=4,
    feature_dim=3,
)

TEST_PREDICTION_GEOMETRY = PredictionGeometry(
    centers=TEST_GEOMETRY.centers,
    training_seeds=TEST_GEOMETRY.training_seeds,
    generation_seeds=TEST_GEOMETRY.generation_seeds,
    source_rows_per_class=TEST_GEOMETRY.rows_per_class,
    exact_b_prefix_per_source_class=2,
    feature_dim=TEST_GEOMETRY.feature_dim,
    rows_by_center=(("0", 2), ("1", 2), ("2", 2)),
)


@dataclass(frozen=True)
class _Key:
    source_center: str
    training_seed: int
    generation_seed: int
    stream_id: str
    expert_lock_hash: str


@dataclass(frozen=True)
class _Lock:
    generation_lock_hash: str = "test-generation-lock"


@dataclass(frozen=True)
class _Config:
    expert_bank_root: Path
    config_hash: str = "test-config-hash"
    classifier: object = LOCKED_CLASSIFIER_SPEC
    runtime: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", {} if self.runtime is None else self.runtime)


@dataclass(frozen=True)
class _Row:
    row_ordinal: int
    evaluation_row_id: str
    case_id: str
    center: str


@dataclass(frozen=True)
class _Frame:
    embeddings: np.ndarray
    rows_by_center: object
    cache_binding_hash: str = "test-cache-binding"
    cache_binding: object = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cache_binding",
            {
                "labels_persisted": False,
                "manifest_opened": False,
                "sample_paths_persisted": False,
            }
            if self.cache_binding is None
            else self.cache_binding,
        )

    def embeddings_for(self, rows: object) -> np.ndarray:
        ordinals = [row.row_ordinal for row in rows]
        return np.ascontiguousarray(self.embeddings[ordinals], dtype=np.float32)


def _keys() -> tuple[_Key, ...]:
    return tuple(
        _Key(
            source,
            training_seed,
            generation_seed,
            f"stream-{source}-{training_seed}-{generation_seed}",
            f"expert-{source}-{training_seed}",
        )
        for source in TEST_GEOMETRY.centers
        for training_seed in TEST_GEOMETRY.training_seeds
        for generation_seed in TEST_GEOMETRY.generation_seeds
    )


def _generate_block(key: _Key, rows_per_class: int, device: str) -> np.ndarray:
    assert device in ("cuda:0", "cuda:1")
    source_shift = int(key.source_center) * 0.03
    seed_shift = (key.training_seed + key.generation_seed) * 0.0001
    negative = np.asarray(
        [
            [-1.0 - source_shift - seed_shift - row * 0.01, -0.8, -0.3]
            for row in range(rows_per_class)
        ],
        dtype=np.float32,
    )
    positive = np.asarray(
        [
            [1.0 + source_shift + seed_shift + row * 0.01, 0.8, 0.3]
            for row in range(rows_per_class)
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(np.concatenate((negative, positive)), dtype=np.float32)


def _nonfinite_block(key: _Key, rows_per_class: int, device: str) -> np.ndarray:
    values = _generate_block(key, rows_per_class, device)
    values[0, 0] = np.nan
    return values


def _must_not_generate(key: _Key, rows_per_class: int, device: str) -> np.ndarray:
    raise AssertionError("sealed source store attempted regeneration")


def _must_not_fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    classifier: object,
) -> FitOutcome:
    raise AssertionError("sealed prediction surface attempted a refit")


def _nonconverged_fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    classifier: object,
) -> FitOutcome:
    probabilities = np.full(len(eval_x), 0.5, dtype=np.float32)
    return FitOutcome(
        positive_probabilities=probabilities,
        predictions=np.ones(len(eval_x), dtype=np.uint8),
        converged=False,
        classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        scaler_state_hash="synthetic-scaler",
    )


def _nonfinite_fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    classifier: object,
) -> FitOutcome:
    probabilities = np.full(len(eval_x), 0.5, dtype=np.float32)
    probabilities[0] = np.nan
    return FitOutcome(
        positive_probabilities=probabilities,
        predictions=np.ones(len(eval_x), dtype=np.uint8),
        converged=True,
        classifier_config_hash=LOCKED_CLASSIFIER_SPEC.config_hash,
        scaler_state_hash="synthetic-scaler",
    )


def _source_test_mode(
    generator: object = _generate_block,
) -> SourceRuntimeTestMode:
    return SourceRuntimeTestMode(
        geometry=TEST_GEOMETRY,
        generation_keys=_keys(),
        generate_block=generator,
    )


def _frame() -> _Frame:
    rows_by_center: dict[str, tuple[_Row, ...]] = {}
    rows: list[_Row] = []
    embeddings: list[list[float]] = []
    ordinal = 0
    for center in TEST_GEOMETRY.centers:
        center_rows = []
        for local, sign in enumerate((-1.0, 1.0)):
            row = _Row(
                ordinal,
                f"eval-{center}-{local}",
                f"case-{center}-{local}",
                center,
            )
            rows.append(row)
            center_rows.append(row)
            embeddings.append([sign, sign * 0.75, sign * 0.25])
            ordinal += 1
        rows_by_center[center] = tuple(center_rows)
    return _Frame(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        rows_by_center=rows_by_center,
    )


def test_production_geometry_and_public_api_are_frozen() -> None:
    assert PRODUCTION_SOURCE_GEOMETRY.array_shape == (81, 2048, 3840)
    assert PRODUCTION_PREDICTION_GEOMETRY.candidate_shape == (9, 9, 9928)
    assert PRODUCTION_PREDICTION_GEOMETRY.exact_b_shape == (9, 9928)
    assert PRODUCTION_PREDICTION_GEOMETRY.fit_count == 162
    assert PRODUCTION_PREDICTION_GEOMETRY.seed_cells == (
        (17, 17),
        (17, 42),
        (17, 101),
        (42, 17),
        (42, 42),
        (42, 101),
        (101, 17),
        (101, 42),
        (101, 101),
    )
    assert exact_b_source_centers("0") == ("1", "2", "3", "5", "6", "7", "8", "9")
    for function in (materialize_source_streams, materialize_prediction_surface):
        names = set(inspect.signature(function).parameters)
        assert "labels" not in names
        assert "manifest_path" not in names
        assert "target_path" not in names
    generation_worker_source = inspect.getsource(
        streams._production_generation_worker
    )
    initializer_source = inspect.getsource(worker_runtime.initialize_gpu_worker)
    executor_source = inspect.getsource(streams._execute_gpu_tasks)
    cpu_executor_source = inspect.getsource(predictions._execute_cpu_tasks)
    assert "torch.set_num_threads(1)" in initializer_source
    assert "torch.set_num_interop_threads(1)" in initializer_source
    assert "torch.set_num_threads" not in generation_worker_source
    assert "torch.set_num_interop_threads" not in generation_worker_source
    assert "initializer=initialize_gpu_worker" in executor_source
    assert "max_workers=CPU_PREDICTION_WORKERS" in cpu_executor_source
    assert "initializer=_initialize_cpu_worker" in cpu_executor_source
    assert execution_worker_runtime.initialize_gpu_worker is worker_runtime.initialize_gpu_worker
    assert execution_worker_runtime.assert_gpu_worker_ready is worker_runtime.assert_gpu_worker_ready


def test_small_source_store_is_deterministic_read_only_and_not_regenerated(
    tmp_path: Path,
) -> None:
    config = _Config(tmp_path / "bank")
    lock = _Lock()
    root = tmp_path / "physical"
    store = materialize_source_streams(
        config,
        lock,
        root=root,
        test_mode=_source_test_mode(),
    )
    values = np.load(store.array_path, mmap_mode="r", allow_pickle=False)
    assert values.shape == TEST_GEOMETRY.array_shape
    assert values.dtype == np.float32
    assert values.flags.writeable is False
    assert tuple(record.key for record in store.records) == tuple(
        (source, training, generation)
        for source in TEST_GEOMETRY.centers
        for training in TEST_GEOMETRY.training_seeds
        for generation in TEST_GEOMETRY.generation_seeds
    )
    original_receipt = store.receipt_hash
    binding = source_store_payload(store)
    assert binding["receipt_hash"] == store.receipt_hash
    assert binding["stream_count"] == TEST_GEOMETRY.stream_count
    assert binding["labels_opened"] is False
    replay = materialize_source_streams(
        config,
        lock,
        root=root,
        test_mode=_source_test_mode(_must_not_generate),
    )
    assert replay.receipt_hash == original_receipt


def test_source_nonfinite_and_partial_checkpoint_fail_closed(tmp_path: Path) -> None:
    config = _Config(tmp_path / "bank")
    lock = _Lock()
    with pytest.raises(ProtocolError, match="invalid values"):
        materialize_source_streams(
            config,
            lock,
            root=tmp_path / "nonfinite",
            test_mode=_source_test_mode(_nonfinite_block),
        )
    assert not (
        tmp_path / "nonfinite" / "arrays" / "sceptre_v4_source_streams.npy"
    ).exists()

    partial_root = tmp_path / "partial"
    partial = (
        partial_root
        / "checkpoints"
        / "sceptre_v4_source_streams"
        / "source_0_train_17.npy"
    )
    partial.parent.mkdir(parents=True)
    with partial.open("wb") as handle:
        np.save(handle, np.zeros((2, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ProtocolError, match="partial; refusing refit"):
        materialize_source_streams(
            config,
            lock,
            root=partial_root,
            test_mode=_source_test_mode(),
        )


def test_prediction_surface_has_exact_geometry_and_sealed_replay(tmp_path: Path) -> None:
    config = _Config(tmp_path / "bank")
    lock = _Lock()
    root = tmp_path / "physical"
    source_store = materialize_source_streams(
        config,
        lock,
        root=root,
        test_mode=_source_test_mode(),
    )
    mode = PredictionRuntimeTestMode(geometry=TEST_PREDICTION_GEOMETRY)
    surface = materialize_prediction_surface(
        config,
        source_store,
        _frame(),
        root=root,
        test_mode=mode,
    )
    candidate = surface.candidate_probabilities
    exact_b = surface.exact_b_probabilities
    assert candidate.shape == (4, 3, 6)
    assert exact_b.shape == (4, 6)
    assert candidate.dtype == exact_b.dtype == np.float32
    assert candidate.flags.writeable is exact_b.flags.writeable is False
    assert np.isfinite(candidate).all() and np.isfinite(exact_b).all()
    for source_ordinal, source in enumerate(TEST_PREDICTION_GEOMETRY.centers):
        forbidden = TEST_PREDICTION_GEOMETRY.row_slice(source)
        assert np.all(
            candidate[:, source_ordinal, forbidden]
            == CANDIDATE_EXCLUSION_SENTINEL
        )
        allowed = np.concatenate(
            (
                candidate[:, source_ordinal, : forbidden.start].reshape(-1),
                candidate[:, source_ordinal, forbidden.stop :].reshape(-1),
            )
        )
        assert np.all((allowed >= 0.0) & (allowed <= 1.0))
    assert np.all((exact_b >= 0.0) & (exact_b <= 1.0))
    assert surface.receipt["fit_count"] == 24
    assert surface.index["seed_cell_order"] == [[17, 17], [17, 42], [42, 17], [42, 42]]
    assert surface.index["candidate_source_order"] == ["0", "1", "2"]
    assert surface.index["candidate_target_exclusion_mode"] == "MASKED_BEFORE_SCORING"
    assert surface.receipt["target_expert_excluded_from_every_candidate_score"] is True

    replay = materialize_prediction_surface(
        config,
        source_store,
        _frame(),
        root=root,
        test_mode=PredictionRuntimeTestMode(
            geometry=TEST_PREDICTION_GEOMETRY,
            fit_predict=_must_not_fit,
        ),
    )
    assert replay.receipt_hash == surface.receipt_hash
    serialized = json.loads(surface.index_path.read_text(encoding="utf-8"))
    keys: set[str] = set()

    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            keys.update(str(key) for key in value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(serialized)
    assert {"manifest_path", "sample_path", "target_path"}.isdisjoint(keys)


def test_exact_b_rejects_any_declared_source_order_that_includes_h() -> None:
    blocks = {
        center: _generate_block(
            _Key(center, 17, 17, f"stream-{center}", f"expert-{center}"),
            TEST_GEOMETRY.rows_per_class,
            "cuda:0",
        )
        for center in TEST_GEOMETRY.centers
    }
    with pytest.raises(ProtocolError, match="included H"):
        predictions._compose_exact_b(
            blocks,
            target_center="0",
            source_order=("0", "1"),
            geometry=TEST_PREDICTION_GEOMETRY,
        )
    values, truth, _ = predictions._compose_exact_b(
        blocks,
        target_center="0",
        source_order=("1", "2"),
        geometry=TEST_PREDICTION_GEOMETRY,
    )
    assert values.shape == (8, 3)
    assert truth.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


@pytest.mark.parametrize("fit_predict", [_nonconverged_fit, _nonfinite_fit])
def test_prediction_fit_failures_are_terminal(
    tmp_path: Path, fit_predict: object
) -> None:
    config = _Config(tmp_path / "bank")
    lock = _Lock()
    root = tmp_path / "source"
    source_store = materialize_source_streams(
        config,
        lock,
        root=root,
        test_mode=_source_test_mode(),
    )
    with pytest.raises(ProtocolError, match="failed convergence or value checks"):
        materialize_prediction_surface(
            config,
            source_store,
            _frame(),
            root=tmp_path / f"prediction-{fit_predict.__name__}",
            test_mode=PredictionRuntimeTestMode(
                geometry=TEST_PREDICTION_GEOMETRY,
                fit_predict=fit_predict,
            ),
        )


def test_v4_physical_schemas_and_import_fence_are_owned(tmp_path: Path) -> None:
    config = _Config(tmp_path / "bank")
    root = tmp_path / "physical"
    source = materialize_source_streams(
        config,
        _Lock(),
        root=root,
        test_mode=_source_test_mode(),
    )
    surface = materialize_prediction_surface(
        config,
        source,
        _frame(),
        root=root,
        test_mode=PredictionRuntimeTestMode(geometry=TEST_PREDICTION_GEOMETRY),
    )
    assert source.receipt["schema_version"].startswith(
        "midogpp_sceptre_v4_physical_"
    )
    assert surface.index["schema_version"].startswith(
        "midogpp_sceptre_v4_physical_"
    )
    assert surface.receipt["schema_version"].startswith(
        "midogpp_sceptre_v4_physical_"
    )
    package_root = Path(streams.__file__).resolve().parent
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package_root.glob("*.py"))
    )
    assert "fixed_bank_sceptre_router_v1" not in text
    assert "fixed_bank_sceptre_router_v2" not in text
    assert "fixed_bank_sceptre_router_v3" not in text
    assert "midogpp_sceptre_v1" not in text
    assert "midogpp_sceptre_v2" not in text
    assert "midogpp_sceptre_v3" not in text


def test_checkpoint_and_final_replay_are_bound_to_one_attempt(tmp_path: Path) -> None:
    config = _Config(tmp_path / "bank")
    root = tmp_path / "physical"
    source = materialize_source_streams(
        config,
        _Lock(),
        root=root,
        attempt_id="attempt-a",
        test_mode=_source_test_mode(),
    )
    assert source.attempt_id == "attempt-a"
    with pytest.raises(ProtocolError, match="receipt failed validation"):
        materialize_source_streams(
            config,
            _Lock(),
            root=root,
            attempt_id="attempt-b",
            test_mode=_source_test_mode(_must_not_generate),
        )
    with pytest.raises(ProtocolError, match="attempt binding drifted"):
        materialize_prediction_surface(
            config,
            source,
            _frame(),
            root=root,
            attempt_id="attempt-b",
            test_mode=PredictionRuntimeTestMode(
                geometry=TEST_PREDICTION_GEOMETRY
            ),
        )


def test_physical_prediction_rejects_a_frame_that_exposes_labels(
    tmp_path: Path,
) -> None:
    config = _Config(tmp_path / "bank")
    source = materialize_source_streams(
        config,
        _Lock(),
        root=tmp_path / "source",
        test_mode=_source_test_mode(),
    )
    frame = _frame()
    object.__setattr__(frame, "labels", (0, 1, 0, 1, 0, 1))
    with pytest.raises(ProtocolError, match="exposes labels"):
        materialize_prediction_surface(
            config,
            source,
            frame,
            root=tmp_path / "prediction",
            test_mode=PredictionRuntimeTestMode(
                geometry=TEST_PREDICTION_GEOMETRY
            ),
        )
