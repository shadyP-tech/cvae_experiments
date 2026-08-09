from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.exact_tail_utility_surface import (
    ensemble_artifact_io,
    ensemble_scoring,
    probability_surface,
    support_shift_surface,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.config import (
    CLASSIFIER,
    CONFIG_SCHEMA_VERSION,
    load_exact_tail_utility_surface_config,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.contracts import (
    BASE_ACTION_ID,
    EXPECTED_COARSE_TASK_COUNT,
    EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT,
    EXPECTED_UTILITY_ROW_COUNT,
    expected_ensemble_endpoint_keys,
    expected_utility_keys,
    tail_action_id,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.prediction_checkpoint_store import (
    atomic_save_npz,
    checkpoint_metadata_path,
    checkpoint_path,
    load_checkpoint,
    write_checkpoint,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.prediction_contracts import (
    CHECKPOINT_SCHEMA,
    PredictionWorkerInput,
)
from midogpp_thesis.cvae.routing.exact_tail_utility_surface.runtime import (
    coarse_prediction_tasks,
)
from midogpp_thesis.cvae.routing.utility_aligned.ensemble_contracts import (
    ENSEMBLE_SEED_KEYS,
    SUPPORT_ACTION_PROBABILITY_SHIFT_NAME,
    SeedProbabilityVector,
)
from midogpp_thesis.cvae.routing.utility_aligned.ensemble_endpoint import (
    scored_ensemble_utility_response_from_payload,
    score_nine_seed_probability_ensemble,
    support_action_probability_shift,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_exact_tail_utility_surface_v1.yaml"
)


class _FakeSeal:
    def __init__(self, cells: tuple[object, ...]) -> None:
        self.cells = cells
        self.seal_hash = "1" * 16
        self.config_contract_hash = "2" * 16
        self.development_manifest_sha256 = "3" * 64
        self.prediction_index_sha256 = "4" * 64
        self.prediction_arrays_sha256 = "5" * 64
        self.partition_hash_by_center = {"1": "6" * 16}
        self.evaluation_row_hash_by_center = {"1": "7" * 16}
        self.support_row_hash_by_center = {"1": "8" * 16}
        self.verification_count = 0

    def verify_complete(self) -> None:
        self.verification_count += 1


def _vectors(
    values: tuple[np.ndarray, ...], *, row_hash: str, role: str
) -> tuple[SeedProbabilityVector, ...]:
    assert len(values) == len(ENSEMBLE_SEED_KEYS)
    return tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash=row_hash,
            prediction_provenance_hash=f"{role}::{ordinal}",
            positive_class_probabilities=value,
        )
        for ordinal, ((training_seed, generation_seed), value) in enumerate(
            zip(ENSEMBLE_SEED_KEYS, values, strict=True)
        )
    )


def _write_payload_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    assert rows
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _single_endpoint_fixture(monkeypatch: pytest.MonkeyPatch):
    identity = ("0", "1", "2")
    monkeypatch.setattr(
        ensemble_scoring, "expected_ensemble_endpoint_keys", lambda: (identity,)
    )
    monkeypatch.setattr(
        ensemble_scoring, "EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT", 1
    )
    base_values = tuple(
        np.asarray([0.51, 0.51], dtype=np.float32)
        if ordinal < 5
        else np.asarray([0.0, 1.0], dtype=np.float32)
        for ordinal in range(9)
    )
    tail_values = tuple(
        np.asarray([0.9, 0.9], dtype=np.float32) for _ in range(9)
    )
    probabilities: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    cells: list[object] = []
    for action_id, values in (
        (BASE_ACTION_ID, base_values),
        (tail_action_id(identity[2]), tail_values),
    ):
        for (training_seed, generation_seed), value in zip(
            ENSEMBLE_SEED_KEYS, values, strict=True
        ):
            key = (
                identity[0],
                identity[1],
                action_id,
                training_seed,
                generation_seed,
            )
            probabilities[key] = value
            cells.append(
                SimpleNamespace(
                    key=key,
                    evaluation_row_identity_hash="7" * 16,
                    probability_sha256=probability_surface.array_sha256(value),
                )
            )
    seal = _FakeSeal(tuple(cells))
    predictions = SimpleNamespace(
        seal=seal,
        probabilities_by_key=probabilities,
    )
    partition = SimpleNamespace(
        reservation_hash="6" * 16,
        evaluation_rows=(
            SimpleNamespace(case_id="case-0"),
            SimpleNamespace(case_id="case-1"),
        ),
    )
    labels = SimpleNamespace(
        prediction_seal_hash=seal.seal_hash,
        manifest_sha256=seal.development_manifest_sha256,
        row_hash_by_center={"1": "7" * 16},
        labels_by_center={"1": (0, 1)},
    )
    return identity, seal, predictions, {"1": partition}, labels


def test_predeclared_counts_and_descriptive_legacy_table_are_frozen() -> None:
    config = load_exact_tail_utility_surface_config(CONFIG_PATH)

    assert CONFIG_SCHEMA_VERSION == "midogpp_exact_tail_utility_surface_config_v2"
    assert EXPECTED_COARSE_TASK_COUNT == 648
    assert EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT == 504
    assert EXPECTED_UTILITY_ROW_COUNT == 4536
    assert len(expected_ensemble_endpoint_keys()) == 504
    assert len(set(expected_ensemble_endpoint_keys())) == 504
    assert len(expected_utility_keys()) == 4536
    assert len(set(expected_utility_keys())) == 4536
    assert len(REQUIRED_FILES) == 21
    assert "tables/exact_tail_utility.csv" in REQUIRED_FILES
    assert ensemble_scoring.ENSEMBLE_ENDPOINT_TABLE_MEMBER in REQUIRED_FILES
    assert ensemble_scoring.ENSEMBLE_ENDPOINT_LOCK_MEMBER in REQUIRED_FILES
    assert support_shift_surface.SUPPORT_SHIFT_TABLE_MEMBER in REQUIRED_FILES
    assert support_shift_surface.SUPPORT_SHIFT_LOCK_MEMBER in REQUIRED_FILES
    assert config.protocol["primary_utility_endpoint"] == (
        "all_nine_seed_probability_ensemble_bacc_delta"
    )
    assert config.protocol["ensemble_seed_pair_count"] == 9
    assert config.protocol["ensemble_threshold"] == 0.5
    assert config.protocol["ensemble_utility_row_count"] == 504
    assert config.protocol["per_seed_utility_role"] == "descriptive_only"
    assert config.protocol["per_seed_rows_may_feed_model"] is False
    assert config.runtime.to_payload()["classifier_fit_count"] == 5184


def test_exact_nine_means_probabilities_before_one_threshold_counterexample() -> None:
    values = tuple(
        np.asarray([0.51, 0.51], dtype=np.float32)
        if ordinal < 5
        else np.asarray([0.0, 1.0], dtype=np.float32)
        for ordinal in range(9)
    )
    vectors = _vectors(values, row_hash="row::evaluation", role="base")

    endpoint = score_nine_seed_probability_ensemble(vectors, [0, 1])
    majority_vote = (
        np.mean(
            np.stack(
                [
                    vector.positive_class_probabilities >= 0.5
                    for vector in vectors
                ]
            ),
            axis=0,
        )
        >= 0.5
    ).astype(np.uint8)

    assert endpoint.seed_keys == ENSEMBLE_SEED_KEYS
    assert endpoint.predictions.tolist() == [0, 1]
    assert endpoint.balanced_accuracy == pytest.approx(1.0)
    assert majority_vote.tolist() == [1, 1]
    assert endpoint.mean_positive_probabilities.flags.writeable is False
    assert endpoint.predictions.flags.writeable is False


def test_exact_nine_fails_closed_on_order_missing_duplicate_and_hash_changes() -> None:
    values = tuple(
        np.asarray([0.1 + ordinal / 100.0, 0.9], dtype=np.float32)
        for ordinal in range(9)
    )
    vectors = _vectors(values, row_hash="row::evaluation", role="base")
    original = score_nine_seed_probability_ensemble(vectors, [0, 1])

    with pytest.raises(ProtocolError, match="exactly nine"):
        score_nine_seed_probability_ensemble(vectors[:-1], [0, 1])
    with pytest.raises(ProtocolError, match="duplicate"):
        score_nine_seed_probability_ensemble((*vectors[:-1], vectors[-2]), [0, 1])
    with pytest.raises(ProtocolError, match="canonical training-major"):
        score_nine_seed_probability_ensemble(
            (vectors[1], vectors[0], *vectors[2:]), [0, 1]
        )

    changed_values = list(values)
    changed_values[0] = np.asarray([0.49, 0.9], dtype=np.float32)
    changed_vectors = _vectors(
        tuple(changed_values), row_hash="row::evaluation", role="base"
    )
    changed = score_nine_seed_probability_ensemble(changed_vectors, [0, 1])
    assert changed_vectors[0].probability_hash != vectors[0].probability_hash
    assert changed.endpoint_hash != original.endpoint_hash


def test_sealed_probability_surface_rejects_missing_or_tampered_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ("0", "1", BASE_ACTION_ID, 17, 17)
    value = np.asarray([0.2, 0.8], dtype=np.float32)
    cell = SimpleNamespace(
        key=key,
        probability_sha256=probability_surface.array_sha256(value),
    )
    seal = _FakeSeal((cell,))
    monkeypatch.setattr(
        probability_surface, "expected_prediction_keys", lambda: (key,)
    )

    observed = probability_surface.SealedProbabilitySurface({key: value}, seal)
    assert observed.probabilities_by_key[key].flags.writeable is False
    with pytest.raises(ProtocolError, match="coverage"):
        probability_surface.SealedProbabilitySurface({}, seal)
    with pytest.raises(ProtocolError, match="bytes drifted"):
        probability_surface.SealedProbabilitySurface(
            {key: np.asarray([0.3, 0.8], dtype=np.float32)}, seal
        )


def test_stage60_endpoint_row_lock_parser_and_label_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    identity, seal, predictions, partitions, labels = _single_endpoint_fixture(
        monkeypatch
    )

    rows = ensemble_scoring.score_exact_tail_ensemble_endpoints(
        predictions, labels, partitions
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.row_key == identity
    assert row.base_bacc == pytest.approx(1.0)
    assert row.tail_bacc == pytest.approx(0.5)
    assert row.delta_bacc == pytest.approx(-0.5)
    assert row.seed_pair_count == 9
    assert seal.verification_count == 1

    parsed = scored_ensemble_utility_response_from_payload(row.to_payload())
    assert parsed.row_key == identity
    assert parsed.source_endpoint_row_hash == row.endpoint_row_hash
    with pytest.raises(ProtocolError, match="row hash drifted"):
        replace(row, endpoint_row_hash="0" * 16)

    monkeypatch.setattr(
        ensemble_artifact_io,
        "expected_ensemble_endpoint_keys",
        lambda: (identity,),
    )
    table_path = tmp_path / "endpoint.csv"
    _write_payload_csv(table_path, (row.to_payload(),))
    assert ensemble_artifact_io.load_ensemble_endpoint_rows(table_path) == rows
    tampered_row = row.to_payload()
    tampered_row["tail_bacc"] = 0.75
    _write_payload_csv(table_path, (tampered_row,))
    with pytest.raises(ProtocolError, match="metrics drifted"):
        ensemble_artifact_io.load_ensemble_endpoint_rows(table_path)

    lock = ensemble_scoring.build_ensemble_endpoint_lock(
        seal=seal,
        rows=rows,
        endpoint_table_sha256="9" * 64,
    )
    path = tmp_path / "endpoint-lock.json"
    path.write_text(json.dumps(lock.to_payload()), encoding="utf-8")
    assert ensemble_scoring.load_ensemble_endpoint_lock(path) == lock
    tampered_lock = lock.to_payload()
    tampered_lock["endpoint_table_sha256"] = "a" * 64
    path.write_text(json.dumps(tampered_lock), encoding="utf-8")
    with pytest.raises(ProtocolError, match="lock hash drifted"):
        ensemble_scoring.load_ensemble_endpoint_lock(path)

    wrong_rows = SimpleNamespace(
        prediction_seal_hash=labels.prediction_seal_hash,
        manifest_sha256=labels.manifest_sha256,
        row_hash_by_center={"1": "f" * 16},
        labels_by_center=labels.labels_by_center,
    )
    with pytest.raises(ProtocolError, match="label rows escaped"):
        ensemble_scoring.score_exact_tail_ensemble_endpoints(
            predictions, wrong_rows, partitions
        )

    missing = dict(predictions.probabilities_by_key)
    missing.pop(next(iter(missing)))
    with pytest.raises(ProtocolError, match="probability cell is missing"):
        ensemble_scoring.score_exact_tail_ensemble_endpoints(
            SimpleNamespace(seal=seal, probabilities_by_key=missing),
            labels,
            partitions,
        )


def test_support_action_shift_surface_is_exact_nine_sealed_and_label_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    endpoint_key = ("0", "1", "2")
    utility_keys = tuple(
        (*endpoint_key, training_seed, generation_seed)
        for training_seed, generation_seed in ENSEMBLE_SEED_KEYS
    )
    monkeypatch.setattr(
        support_shift_surface,
        "expected_ensemble_endpoint_keys",
        lambda: (endpoint_key,),
    )
    monkeypatch.setattr(
        support_shift_surface, "expected_utility_keys", lambda: utility_keys
    )
    monkeypatch.setattr(support_shift_surface, "EXPECTED_UTILITY_ROW_COUNT", 9)

    base = tuple(
        np.asarray([0.5, 0.5, 0.5], dtype=np.float32) for _ in range(9)
    )
    tail = (
        tuple(np.asarray([0.9, 0.9, 0.9], dtype=np.float32) for _ in range(4))
        + tuple(np.asarray([0.1, 0.1, 0.1], dtype=np.float32) for _ in range(4))
        + (np.asarray([0.5, 0.5, 0.5], dtype=np.float32),)
    )
    probabilities: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    cells: list[object] = []
    for action_id, values in (
        (BASE_ACTION_ID, base),
        (tail_action_id(endpoint_key[2]), tail),
    ):
        for (training_seed, generation_seed), value in zip(
            ENSEMBLE_SEED_KEYS, values, strict=True
        ):
            key = (
                endpoint_key[0],
                endpoint_key[1],
                action_id,
                training_seed,
                generation_seed,
            )
            probabilities[key] = value
            cells.append(
                SimpleNamespace(
                    key=key,
                    support_row_identity_hash="8" * 16,
                    support_probability_sha256=probability_surface.array_sha256(
                        value
                    ),
                )
            )
    seal = _FakeSeal(tuple(cells))
    predictions = SimpleNamespace(
        seal=seal,
        support_probabilities_by_key=probabilities,
    )
    partition = SimpleNamespace(
        reservation_hash="6" * 16,
        support_rows=(object(), object(), object()),
        support_case_ids=("case-0", "case-1", "case-2"),
    )

    rows = support_shift_surface.build_support_action_shift_rows(
        predictions, {"1": partition}
    )
    assert tuple(row.row_key for row in rows) == utility_keys
    assert len(rows) == 9
    assert max(row.descriptive_seed_mean_absolute_shift for row in rows) == pytest.approx(0.4)
    assert all(
        row.candidate_ensemble_mean_absolute_shift
        == pytest.approx(0.0, abs=1.0e-7)
        for row in rows
    )
    assert len({row.candidate_aggregate_shift_hash for row in rows}) == 1
    assert all(
        row.scalar_name == SUPPORT_ACTION_PROBABILITY_SHIFT_NAME
        and row.scalar_semantics
        == support_shift_surface.SUPPORT_SHIFT_ROW_SCALAR_SEMANTICS
        and row.labels_used is False
        and row.support_labels_available is False
        and row.target_labels_used is False
        for row in rows
    )
    with pytest.raises(ProtocolError, match="row hash drifted"):
        replace(rows[0], descriptive_seed_mean_absolute_shift=0.2)

    monkeypatch.setattr(
        ensemble_artifact_io, "expected_utility_keys", lambda: utility_keys
    )
    table_path = tmp_path / "support-shifts.csv"
    _write_payload_csv(
        table_path, tuple(row.to_payload() for row in rows)
    )
    assert ensemble_artifact_io.load_support_shift_rows(table_path) == rows
    tampered_row = rows[0].to_payload()
    tampered_row["base_support_probability_sha256"] = "0" * 64
    _write_payload_csv(
        table_path,
        (tampered_row, *(row.to_payload() for row in rows[1:])),
    )
    with pytest.raises(ProtocolError, match="row hash drifted"):
        ensemble_artifact_io.load_support_shift_rows(table_path)

    lock = support_shift_surface.build_support_action_shift_lock(
        seal=seal,
        rows=rows,
        shift_table_sha256="9" * 64,
    )
    path = tmp_path / "support-shift-lock.json"
    path.write_text(json.dumps(lock.to_payload()), encoding="utf-8")
    assert support_shift_surface.load_support_action_shift_lock(path) == lock
    payload = lock.to_payload()
    payload["labels_used"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="violates its contract"):
        support_shift_surface.load_support_action_shift_lock(path)

    core_shift = support_action_probability_shift(
        _vectors(base, row_hash="support::rows", role="base-support"),
        _vectors(tail, row_hash="support::rows", role="tail-support"),
    )
    assert np.mean(core_shift.per_seed_mean_absolute_shifts) > 0.35
    assert core_shift.value == pytest.approx(0.0, abs=1.0e-7)
    assert core_shift.to_payload()["labels_used"] is False


def test_support_probabilities_roundtrip_and_resume_tamper_detection(
    tmp_path: Path,
) -> None:
    task = coarse_prediction_tasks()[0]
    evaluation_path = tmp_path / "evaluation.npy"
    support_path = tmp_path / "support.npy"
    np.save(
        evaluation_path,
        np.zeros((3, 2), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        support_path,
        np.ones((2, 2), dtype=np.float32),
        allow_pickle=False,
    )
    item = PredictionWorkerInput(
        task=task,
        cache_root=str(tmp_path / "cache"),
        source_records=(),
        evaluation_array_path=str(evaluation_path),
        evaluation_row_identity_hash="a" * 64,
        partition_hash="b" * 64,
        source_cache_hash="c" * 64,
        classifier_payload=CLASSIFIER.to_payload(),
        checkpoint_root=str(tmp_path / "resume"),
        support_array_path=str(support_path),
        support_row_identity_hash="d" * 16,
    )
    predictions = np.zeros((8, 3), dtype=np.uint8)
    probabilities = np.full((8, 3), 0.5, dtype=np.float32)
    support_probabilities = np.full((8, 2), 0.25, dtype=np.float32)
    prediction_hashes = {
        action_id: probability_surface.array_sha256(predictions[index])
        for index, action_id in enumerate(task.action_ids)
    }
    probability_hashes = {
        action_id: probability_surface.array_sha256(probabilities[index])
        for index, action_id in enumerate(task.action_ids)
    }
    support_hashes = {
        action_id: probability_surface.array_sha256(
            support_probabilities[index]
        )
        for index, action_id in enumerate(task.action_ids)
    }
    compositions = {action_id: "e" * 64 for action_id in task.action_ids}
    scalers = {action_id: "f" * 64 for action_id in task.action_ids}

    written = write_checkpoint(
        item,
        classifier_config_hash=CLASSIFIER.config_hash,
        predictions=predictions,
        probabilities=probabilities,
        support_probabilities=support_probabilities,
        action_prediction_sha256=prediction_hashes,
        action_probability_sha256=probability_hashes,
        action_support_probability_sha256=support_hashes,
        action_composition_sha256=compositions,
        action_scaler_state_hash=scalers,
        evaluation_row_count=3,
        support_row_count=2,
    )
    assert load_checkpoint(item) == written
    with np.load(checkpoint_path(item), allow_pickle=False) as payload:
        assert set(payload.files) == {
            "predictions",
            "probabilities",
            "support_probabilities",
        }
        np.testing.assert_array_equal(
            payload["support_probabilities"], support_probabilities
        )
    metadata = json.loads(
        checkpoint_metadata_path(item).read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == CHECKPOINT_SCHEMA
    assert metadata["support_array_present"] is True
    assert metadata["support_labels_available_to_fit_or_predict"] is False
    assert metadata["action_support_probability_sha256"] == support_hashes

    tampered_support = support_probabilities.copy()
    tampered_support[0, 0] = np.float32(0.75)
    atomic_save_npz(
        checkpoint_path(item),
        predictions=predictions,
        probabilities=probabilities,
        support_probabilities=tampered_support,
    )
    with pytest.raises(ProtocolError, match="COMPLETE checkpoint binding drifted"):
        load_checkpoint(item)
