from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.data.features.stage70_test_cache import CACHE_ARTIFACT_ID
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.frozen_policy_downstream.bootstrap import (
    paired_descriptive_bootstrap,
)
from midogpp_thesis.cvae.frozen_policy_downstream.bundle import (
    seal_prediction_pass,
    write_authorization_phase,
    write_scored_bundle,
)
from midogpp_thesis.cvae.frozen_policy_downstream.composition import (
    compose_policy_replicate,
)
from midogpp_thesis.cvae.frozen_policy_downstream.contracts import (
    CONTROL_ARM,
    FEATURE_DIM,
    METADATA_ARM,
    POLICY_ARMS,
    UTILITY_ARM,
    MaterializationAssignment,
    PolicyReplicate,
    ScoringLabels,
    SyntheticComposition,
    TargetFrame,
    array_bundle_sha256,
    array_sha256,
)
from midogpp_thesis.cvae.frozen_policy_downstream.contrasts import (
    build_descriptive_contrasts,
)
from midogpp_thesis.cvae.frozen_policy_downstream.prediction import (
    PersistedPredictionPass,
    run_label_free_prediction_pass,
)
from midogpp_thesis.cvae.frozen_policy_downstream.prediction_seal import (
    PredictionSealBinding,
    TargetIdentity,
)
from midogpp_thesis.cvae.frozen_policy_downstream.scoring import (
    CaseConfusionRow,
    score_persisted_predictions,
)
from midogpp_thesis.cvae.frozen_policy_downstream.target_loader import (
    _open_scoring_labels_with_expected_binding,
    open_scoring_labels_after_prediction_seal,
)
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.contracts import (
    FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
from midogpp_thesis.cvae.generation.generation import GeneratedBlock
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    FittedClassifierResult,
)


def _replicate(
    arm: str,
    *,
    target: str = "0",
    training_seed: int = 17,
    generation_seed: int = 17,
    stream_id: str = "stream",
) -> PolicyReplicate:
    exact = arm == UTILITY_ARM
    assignment_id = "control-assignment" if exact else f"{arm}-assignment"
    return PolicyReplicate(
        policy_id=arm,
        policy_lock_hash=f"{arm}-lock",
        policy_plan_hash=f"{arm}-plan",
        assignment_table_hash=f"{arm}-assignments",
        replicate_id=f"rep-{target}-{training_seed}-{generation_seed}",
        target_center=target,
        training_seed=training_seed,
        generation_seed=generation_seed,
        assignments=(
            MaterializationAssignment(
                assignment_id=assignment_id,
                policy_id=arm,
                target_center=target,
                training_seed=training_seed,
                generation_seed=generation_seed,
                source_center="1" if target != "1" else "0",
                source_stream_id=stream_id,
                source_ordinal=0,
                source_budget_per_class=1024,
                prior_method="aggregate",
                selection_source="frozen",
                exact_equal_union_fallback=exact,
                equal_union_assignment_id="control-assignment" if exact else "",
            ),
        ),
        class_shuffle_seed_by_label={"0": 101, "1": 202},
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rebind_prediction_transaction(root: Path) -> None:
    index_path = root / "manifests/prediction_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["prediction_metadata_hash"] = stable_hash(index["records"])
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    index_sha = _sha256_file(root / "manifests/prediction_index.json")
    arrays_sha = _sha256_file(root / "arrays/target_predictions.npz")
    for relative in (
        "manifests/prediction_seal.json",
        "reports/phase_02_predictions_persisted.json",
    ):
        path = root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["prediction_index_sha256"] = index_sha
        payload["prediction_arrays_sha256"] = arrays_sha
        payload["prediction_metadata_hash"] = index["prediction_metadata_hash"]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rebound_capability(
    original: PersistedPredictionPass,
    root: Path,
) -> PersistedPredictionPass:
    return replace(
        original,
        artifact_root=root,
        phase_01_sha256=_sha256_file(
            root / "reports/phase_01_authorization_complete.json"
        ),
        prediction_index_sha256=_sha256_file(
            root / "manifests/prediction_index.json"
        ),
        prediction_arrays_sha256=_sha256_file(
            root / "arrays/target_predictions.npz"
        ),
        prediction_seal_sha256=_sha256_file(
            root / "manifests/prediction_seal.json"
        ),
        phase_02_sha256=_sha256_file(
            root / "reports/phase_02_predictions_persisted.json"
        ),
    )


def _canonical_target_frames() -> dict[str, TargetFrame]:
    frames: dict[str, TargetFrame] = {}
    global_offset = 0
    for center in CENTERS:
        count = EXPECTED_TEST_ROWS_BY_CENTER[center]
        row_ids = tuple(
            "eval_" + hashlib.sha256(f"{center}:{offset}".encode()).hexdigest()
            for offset in range(count)
        )
        row_indices = tuple(range(global_offset, global_offset + count))
        case_ids = tuple(
            f"case-{center}-{offset % 2}" for offset in range(count)
        )
        frames[center] = TargetFrame(
            target_center=center,
            evaluation_row_ids=row_ids,
            contract_row_indices=row_indices,
            case_ids=case_ids,
            embeddings=np.zeros((count, FEATURE_DIM), dtype=np.float32),
            row_order_hash=stable_hash(list(row_ids)),
            content_hash=stable_hash(
                {"center": center, "row_ids": list(row_ids)}
            ),
        )
        global_offset += count
    assert global_offset == EXPECTED_TEST_ROWS
    return frames


def _strict_prediction_binding(
    *,
    frames: dict[str, TargetFrame],
    replicates: tuple[PolicyReplicate, ...],
    manifest_sha256: str,
    classifier_config_hash: str,
) -> PredictionSealBinding:
    return PredictionSealBinding(
        final_authorization_artifact_id=FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
        final_authorization_hash="1" * 16,
        final_authorization_content_hash="2" * 16,
        authorization_protocol_hash="3" * 16,
        identity_lock_hash="4" * 16,
        evaluation_plan_hash="5" * 16,
        reservation_content_hash="6" * 16,
        reservation_identity_lock_hash="7" * 16,
        target_evaluation_reservation_id="reservation_" + "8" * 64,
        target_evaluation_reservation_protocol_hash="9" * 64,
        target_identity_table_hash="a" * 64,
        target_cache_artifact_id=CACHE_ARTIFACT_ID,
        target_cache_content_hash="b" * 16,
        target_cache_row_order_hash="c" * 64,
        target_cache_shard_sha256_by_center={
            center: hashlib.sha256(f"shard:{center}".encode()).hexdigest()
            for center in CENTERS
        },
        target_cache_rows_by_center=dict(EXPECTED_TEST_ROWS_BY_CENTER),
        cache_extractor_protocol_hash="d" * 64,
        scoring_manifest_sha256=manifest_sha256,
        classifier_config_hash=classifier_config_hash,
        identities_by_center={
            center: tuple(
                TargetIdentity(row_id, row_index, case_id)
                for row_id, row_index, case_id in zip(
                    frame.evaluation_row_ids,
                    frame.contract_row_indices,
                    frame.case_ids,
                    strict=True,
                )
            )
            for center, frame in frames.items()
        },
        replicate_id_by_cell={
            (
                row.policy_id,
                row.target_center,
                row.training_seed,
                row.generation_seed,
            ): row.replicate_id
            for row in replicates
        },
    )


def _refresh_record_hash(row: dict[str, object]) -> None:
    row["prediction_cell_hash"] = stable_hash(
        {key: value for key, value in row.items() if key != "prediction_cell_hash"}
    )


def test_composition_slices_prefix_within_each_class() -> None:
    key = SourceGenerationKey(
        source_center="1",
        training_seed=17,
        generation_seed=17,
        expert_lock_hash="expert",
        stream_id="stream",
        class_seed_by_label={"0": 1, "1": 2},
    )
    class_zero = np.repeat(
        np.arange(1024, dtype=np.float32)[:, None], FEATURE_DIM, axis=1
    )
    class_one = np.repeat(
        (10_000 + np.arange(1024, dtype=np.float32))[:, None], FEATURE_DIM, axis=1
    )
    embeddings = np.concatenate((class_zero, class_one), axis=0)
    labels = np.concatenate(
        (np.zeros(1024, dtype=np.int64), np.ones(1024, dtype=np.int64))
    )
    block = GeneratedBlock(
        key=key,
        embeddings=embeddings,
        labels=labels,
        output_sha256=array_bundle_sha256(embeddings, labels),
    )

    observed = compose_policy_replicate(_replicate(CONTROL_ARM), {"stream": block})

    assert observed.embeddings.shape == (2048, FEATURE_DIM)
    assert np.sum(observed.labels == 0) == 1024
    assert np.sum(observed.labels == 1) == 1024
    assert np.max(observed.embeddings[:1024, 0]) < 10_000
    assert np.min(observed.embeddings[1024:, 0]) >= 10_000


def test_target_frame_rejects_legacy_label_encoded_identifier() -> None:
    with pytest.raises(Exception, match="legacy label encoding"):
        TargetFrame(
            target_center="0",
            evaluation_row_ids=("002__ann1__y1",),
            contract_row_indices=(1,),
            case_ids=("002",),
            embeddings=np.zeros((1, FEATURE_DIM), dtype=np.float32),
            row_order_hash="row",
            content_hash="content",
        )


def test_prediction_deduplicates_only_exact_utility_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    base = _replicate(CONTROL_ARM)
    train = np.zeros((2048, FEATURE_DIM), dtype=np.float32)
    train[1024:] = 1.0
    labels = np.concatenate(
        (np.zeros(1024, dtype=np.int64), np.ones(1024, dtype=np.int64))
    )
    shared = SyntheticComposition(
        replicate=base,
        embeddings=train,
        labels=labels,
        pre_shuffle_sha256_by_label={"0": "pre0", "1": "pre1"},
        post_shuffle_sha256_by_label={"0": "post0", "1": "post1"},
        train_content_sha256=array_bundle_sha256(train, labels),
        composition_manifest_hash="f" * 16,
    )
    monkeypatch.setattr(
        "midogpp_thesis.cvae.frozen_policy_downstream.prediction.compose_policy_replicate",
        lambda _replicate, _blocks: shared,
    )
    replicates = tuple(
        _replicate(
            arm,
            target=target,
            training_seed=training_seed,
            generation_seed=generation_seed,
        )
        for arm in POLICY_ARMS
        for target in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    frames = _canonical_target_frames()
    calls = 0
    classifier_spec = ClassifierSpec(
        C=0.01,
        max_iter=3000,
        random_state=23,
    )

    def fake_fit(*args, **_kwargs) -> FittedClassifierResult:
        nonlocal calls
        calls += 1
        n_rows = int(np.asarray(args[2]).shape[0])
        predictions = np.arange(n_rows, dtype=np.int64) % 2
        probabilities = np.where(
            predictions[:, None] == np.asarray([0, 1])[None, :],
            0.9,
            0.1,
        ).astype(np.float64)
        return FittedClassifierResult(
            predictions=predictions,
            probabilities=probabilities,
            classes=(0, 1),
            n_iter=(1,),
            converged=True,
            classifier_config_hash=classifier_spec.config_hash,
            scaler_state_hash="e" * 16,
        )

    observed = run_label_free_prediction_pass(
        replicates=replicates,
        source_blocks={},
        target_frames=frames,
        classifier_spec=classifier_spec,
        classifier_fit=fake_fit,
    )

    assert calls == 162
    assert observed.classifier_fit_count == 162
    assert observed.prediction_reuse_count == 81
    assert sum(cell.reused_from_policy_id == CONTROL_ARM for cell in observed.cells) == 81
    manifest_path = tmp_path.parent / f"{tmp_path.name}_scoring_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("label", "case_id", "center", "split"),
        )
        writer.writeheader()
        for center in CENTERS:
            for offset, case_id in enumerate(frames[center].case_ids):
                writer.writerow(
                    {
                        "label": offset % 2,
                        "case_id": case_id,
                        "center": center,
                        "split": "test",
                    }
                )
    manifest_sha = _sha256_file(manifest_path)
    binding = _strict_prediction_binding(
        frames=frames,
        replicates=replicates,
        manifest_sha256=manifest_sha,
        classifier_config_hash=classifier_spec.config_hash,
    )
    write_authorization_phase(tmp_path, binding=binding)
    sealed = seal_prediction_pass(
        tmp_path,
        observed,
        expected_binding=binding,
    )
    scoring_labels = _open_scoring_labels_with_expected_binding(
        sealed,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha,
        expected_binding=binding,
    )
    assert len(scoring_labels.labels) == EXPECTED_TEST_ROWS

    with pytest.raises(ProtocolError, match="persisted prediction capability"):
        open_scoring_labels_after_prediction_seal(
            tmp_path,  # type: ignore[arg-type]
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            final_authorization_root=tmp_path,
            target_cache_root=tmp_path,
        )

    missing_root = tmp_path.parent / f"{tmp_path.name}_missing"
    shutil.copytree(tmp_path, missing_root)
    (missing_root / "manifests/prediction_index.json").unlink()
    with pytest.raises(ProtocolError, match="missing"):
        _open_scoring_labels_with_expected_binding(
            replace(sealed, artifact_root=missing_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    duplicate_root = tmp_path.parent / f"{tmp_path.name}_duplicate"
    shutil.copytree(tmp_path, duplicate_root)
    duplicate_index_path = duplicate_root / "manifests/prediction_index.json"
    duplicate_index = json.loads(duplicate_index_path.read_text(encoding="utf-8"))
    duplicate_index["records"][1] = {
        **duplicate_index["records"][0],
        "ordinal": 1,
    }
    _refresh_record_hash(duplicate_index["records"][1])
    duplicate_index_path.write_text(
        json.dumps(duplicate_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_prediction_transaction(duplicate_root)
    with pytest.raises(ProtocolError, match="reused|duplicated"):
        _open_scoring_labels_with_expected_binding(
            _rebound_capability(sealed, duplicate_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    tampered_root = tmp_path.parent / f"{tmp_path.name}_tampered"
    shutil.copytree(tmp_path, tampered_root)
    arrays_path = tampered_root / "arrays/target_predictions.npz"
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["prediction_000"][0] = 1 - arrays["prediction_000"][0]
    np.savez_compressed(arrays_path, **arrays)
    _rebind_prediction_transaction(tampered_root)
    with pytest.raises(ProtocolError, match="archive content"):
        _open_scoring_labels_with_expected_binding(
            _rebound_capability(sealed, tampered_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    reduced_root = tmp_path.parent / f"{tmp_path.name}_reduced"
    shutil.copytree(tmp_path, reduced_root)
    reduced_index_path = reduced_root / "manifests/prediction_index.json"
    reduced_index = json.loads(reduced_index_path.read_text(encoding="utf-8"))
    reduced_arrays_path = reduced_root / "arrays/target_predictions.npz"
    with np.load(reduced_arrays_path, allow_pickle=False) as archive:
        reduced_arrays = {
            key: np.array(archive[key], copy=True) for key in archive.files
        }
    for row in reduced_index["records"]:
        if row["target_center"] != CENTERS[0]:
            continue
        row["evaluation_row_ids"] = row["evaluation_row_ids"][:-1]
        row["contract_row_indices"] = row["contract_row_indices"][:-1]
        row["case_ids"] = row["case_ids"][:-1]
        row["row_count"] -= 1
        identities = [
            {
                "evaluation_row_id": row_id,
                "contract_row_index": row_index,
                "case_id": case_id,
            }
            for row_id, row_index, case_id in zip(
                row["evaluation_row_ids"],
                row["contract_row_indices"],
                row["case_ids"],
                strict=True,
            )
        ]
        row["target_identity_hash"] = stable_hash(identities)
        row["target_row_order_hash"] = stable_hash(row["evaluation_row_ids"])
        pred_key = row["prediction_array_key"]
        prob_key = row["probability_array_key"]
        reduced_arrays[pred_key] = reduced_arrays[pred_key][:-1]
        reduced_arrays[prob_key] = reduced_arrays[prob_key][:-1]
        row["prediction_sha256"] = array_sha256(reduced_arrays[pred_key])
        row["probability_sha256"] = array_sha256(reduced_arrays[prob_key])
        _refresh_record_hash(row)
    np.savez_compressed(reduced_arrays_path, **reduced_arrays)
    reduced_index_path.write_text(
        json.dumps(reduced_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_prediction_transaction(reduced_root)
    with pytest.raises(ProtocolError, match="identity coverage"):
        _open_scoring_labels_with_expected_binding(
            _rebound_capability(sealed, reduced_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    probability_root = tmp_path.parent / f"{tmp_path.name}_probability_shape"
    shutil.copytree(tmp_path, probability_root)
    probability_index_path = probability_root / "manifests/prediction_index.json"
    probability_index = json.loads(
        probability_index_path.read_text(encoding="utf-8")
    )
    probability_arrays_path = probability_root / "arrays/target_predictions.npz"
    with np.load(probability_arrays_path, allow_pickle=False) as archive:
        probability_arrays = {
            key: np.array(archive[key], copy=True) for key in archive.files
        }
    probability_arrays["probability_000"] = probability_arrays[
        "probability_000"
    ][:, 0]
    probability_index["records"][0]["probability_sha256"] = array_sha256(
        probability_arrays["probability_000"]
    )
    _refresh_record_hash(probability_index["records"][0])
    np.savez_compressed(probability_arrays_path, **probability_arrays)
    probability_index_path.write_text(
        json.dumps(probability_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_prediction_transaction(probability_root)
    with pytest.raises(ProtocolError, match="content/geometry"):
        _open_scoring_labels_with_expected_binding(
            _rebound_capability(sealed, probability_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    metadata_root = tmp_path.parent / f"{tmp_path.name}_metadata"
    shutil.copytree(tmp_path, metadata_root)
    metadata_index_path = metadata_root / "manifests/prediction_index.json"
    metadata_index = json.loads(metadata_index_path.read_text(encoding="utf-8"))
    metadata_index["records"][0]["classifier_config_hash"] = "f" * 16
    _refresh_record_hash(metadata_index["records"][0])
    metadata_index_path.write_text(
        json.dumps(metadata_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_prediction_transaction(metadata_root)
    with pytest.raises(ProtocolError, match="classifier provenance"):
        _open_scoring_labels_with_expected_binding(
            _rebound_capability(sealed, metadata_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    missing_phase_root = tmp_path.parent / f"{tmp_path.name}_missing_phase"
    shutil.copytree(tmp_path, missing_phase_root)
    (missing_phase_root / "reports/phase_01_authorization_complete.json").unlink()
    with pytest.raises(ProtocolError, match="phase-01|missing"):
        _open_scoring_labels_with_expected_binding(
            replace(sealed, artifact_root=missing_phase_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    for mismatched_binding in (
        replace(binding, final_authorization_hash="f" * 16),
        replace(binding, target_cache_content_hash="e" * 16),
    ):
        with pytest.raises(ProtocolError, match="phase-01 canonical binding"):
            _open_scoring_labels_with_expected_binding(
                sealed,
                manifest_path=manifest_path,
                expected_manifest_sha256=manifest_sha,
                expected_binding=mismatched_binding,
            )

    symlink_root = tmp_path.parent / f"{tmp_path.name}_symlink"
    shutil.copytree(tmp_path, symlink_root)
    symlink_target = tmp_path.parent / f"{tmp_path.name}_arrays_target.npz"
    shutil.copy2(tmp_path / "arrays/target_predictions.npz", symlink_target)
    (symlink_root / "arrays/target_predictions.npz").unlink()
    (symlink_root / "arrays/target_predictions.npz").symlink_to(symlink_target)
    with pytest.raises(ProtocolError, match="symlink"):
        _open_scoring_labels_with_expected_binding(
            replace(sealed, artifact_root=symlink_root),
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            expected_binding=binding,
        )

    with pytest.raises(ProtocolError, match="scoring-label provenance"):
        score_persisted_predictions(
            sealed,
            ScoringLabels(
                evaluation_row_ids=scoring_labels.evaluation_row_ids,
                labels=np.asarray(scoring_labels.labels, dtype=np.int64),
                label_manifest_sha256=scoring_labels.label_manifest_sha256,
            ),  # type: ignore[arg-type]
        )

    # Mutable computation-time arrays cannot bypass the disk-only scoring edge.
    observed.cells[0].predictions[:] = 1 - observed.cells[0].predictions

    scored = score_persisted_predictions(sealed, scoring_labels)
    summaries, deltas = build_descriptive_contrasts(scored.metrics)
    assert len(scored.metrics) == 243
    assert len(summaries) == 3
    assert len(deltas) == 162
    assert all(
        row.bacc_delta == 0.0
        for row in deltas
        if row.policy_id == UTILITY_ARM
    )

    bootstrap = paired_descriptive_bootstrap(
        scored.case_confusions,
        seed=42,
        valid_replicates=1,
        max_attempts=5000,
    )
    write_scored_bundle(
        tmp_path,
        scored=scored,
        summaries=summaries,
        deltas=deltas,
        bootstrap=bootstrap,
        final_authorization_hash=binding.final_authorization_hash,
    )
    assert json.loads(
        (tmp_path / "reports/phase_04_scoring_complete.json").read_text(
            encoding="utf-8"
        )
    )["authorization_binding_hash"] == sealed.authorization_binding_hash


def test_paired_bootstrap_retains_crossed_seeds_and_counts_rejections() -> None:
    rows: list[CaseConfusionRow] = []
    for arm in POLICY_ARMS:
        for center in CENTERS:
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    for duplicate in range(2):
                        rows.extend(
                            (
                            CaseConfusionRow(
                                policy_id=arm,
                                target_center=center,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                case_id=f"{center}-negative-{duplicate}",
                                tn=1,
                                fp=0,
                                fn=0,
                                tp=0,
                            ),
                            CaseConfusionRow(
                                policy_id=arm,
                                target_center=center,
                                training_seed=training_seed,
                                generation_seed=generation_seed,
                                case_id=f"{center}-positive-{duplicate}",
                                tn=0,
                                fp=0,
                                fn=1 if arm == METADATA_ARM else 0,
                                tp=0 if arm == METADATA_ARM else 1,
                            ),
                            )
                        )

    first = paired_descriptive_bootstrap(
        rows,
        seed=42,
        valid_replicates=20,
        max_attempts=2000,
    )
    second = paired_descriptive_bootstrap(
        rows,
        seed=42,
        valid_replicates=20,
        max_attempts=2000,
    )

    assert first == second
    assert first[0].observed_mean_bacc_delta == -0.5
    assert first[1].observed_mean_bacc_delta == 0.0
    assert first[0].attempted_replicates == (
        first[0].valid_replicates + first[0].rejected_replicates
    )
    assert first[0].rejected_replicates > 0
