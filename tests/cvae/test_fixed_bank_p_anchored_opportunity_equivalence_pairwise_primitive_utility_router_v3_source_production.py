from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import canonical_hash
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import CENTERS
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.hashing import file_sha256
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.bundle_writer import write_source_training_bundle
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.held_actions import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    actions_for_held_pair,
    canonical_held_action_library,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.predictions import (
    HeldBaseProbabilityBlock,
    HeldPredictionInventory,
    _BLOCK_GATE,
    _INVENTORY_GATE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.scheduling import build_held_prediction_tasks
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.source_frame import (
    SourceOutcomeRow,
    SourceProbabilitySeal,
    _PROBABILITY_SEAL_GATE,
    load_canonical_source_cache,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.worker import _validate_task


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def test_held_action_library_freezes_mass_policy_and_exact_36_by_9_menu() -> None:
    library = canonical_held_action_library()
    assert len(library.pair_action_hashes) == 36
    assert {len(hashes) for _pair, hashes in library.pair_action_hashes} == {9}
    assert library.mass_policy.b_u_normalization == 8.0 / 7.0
    assert library.mass_policy.a1_normalization == 72.0 / 65.0
    actions = actions_for_held_pair("0", "1")
    assert tuple(actions[0].counts_by_class["0"].values()) == (128,) * 7
    assert tuple(actions[1].counts_by_class["0"].values()) == (144,) * 7
    assert set(actions[0].sample_weight_by_source.values()) == {8.0 / 7.0}
    assert sum(
        actions[2].counts_by_class["0"][source]
        * actions[2].sample_weight_by_source[source]
        for source in actions[2].sample_weight_by_source
    ) == 1152.0
    assert actions[0].to_payload()["scaler_fit_used_sample_weight"] is False


def test_trust_bearing_prediction_receipts_reject_public_constructor_bypass() -> None:
    library = canonical_held_action_library()
    with pytest.raises(ProtocolError, match="factory-only"):
        SourceProbabilitySeal(
            source_frame_hash="1" * 64,
            source_stream_lock_hash="2" * 16,
            held_action_library_sha256=library.library_hash,
            held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
            oriented_block_receipts=tuple(
                (h, q, canonical_hash((h, q)))
                for h in CENTERS
                for q in CENTERS
                if q != h
            ),
        )


def test_producer_source_seal_is_live_derived_and_optional_value_only_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "build_source_seal",
        lambda: SimpleNamespace(combined_source_sha256="a" * 64),
    )
    with pytest.raises(ProtocolError, match="live producer source seal drifted"):
        orchestrator.produce_source_supervision_bundle(
            source_cache_root=tmp_path / "absent-cache",
            expert_bank_root=tmp_path / "absent-bank",
            generation_lock_root=tmp_path / "absent-lock",
            output_root=tmp_path / "output",
            scratch_parent=tmp_path,
            expected_producer_source_seal_sha256="b" * 64,
        )


def test_planner_emits_exact_324_plain_tasks_and_worker_rejects_rehashed_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production import scheduling

    source_path = tmp_path / "source.npy"
    eval_path = tmp_path / "eval.npy"
    source_path.write_bytes(b"source")
    eval_path.write_bytes(b"eval")

    class FakeClassifier:
        def to_payload(self):
            return {
                "family": "sklearn_logistic_regression",
                "C": 0.01,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 3000,
                "class_weight": None,
                "random_state": 23,
                "l1_ratio": None,
                "threshold_policy": "predict",
                "scaler_fit": "synthetic_train_only",
            }

    class FakeConfig:
        classifier = FakeClassifier()

    class FakeRecord:
        def __init__(self, center: str, training: int, generation: int, ordinal: int):
            self.key = (center, training, generation)
            self._payload = {
                "block_ordinal": ordinal,
                "source_center": center,
                "training_seed": training,
                "generation_seed": generation,
                "stream_id": f"stream-{center}-{training}-{generation}",
                "expert_lock_hash": "1" * 16,
                "rows_per_class": 270,
                "row_count": 540,
                "feature_dim": 3840,
                "output_sha256": "2" * 64,
            }

        def to_payload(self):
            return dict(self._payload)

    class FakeSource:
        def __init__(self):
            self.source_array_path = source_path
            self.lock_payload = {"source_array_sha256": "3" * 64}
            self.lock_hash = "4" * 16
            self.records = tuple(
                FakeRecord(center, training, generation, ordinal)
                for ordinal, (center, training, generation) in enumerate(
                    product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS)
                )
            )

    monkeypatch.setattr(scheduling, "SourceProductionRuntimeConfig", FakeConfig)
    monkeypatch.setattr(scheduling, "FrozenSourceStreamCache", FakeSource)
    views = []
    for index, center in enumerate(CENTERS):
        row_ids = (f"source_row_{center}",)
        views.append(
            {
                "center": center,
                "start": index,
                "stop": index + 1,
                "source_row_ids": list(row_ids),
                "source_cache_row_indices": [index],
                "row_identity_hash": canonical_hash(row_ids),
                "slice_sha256": "5" * 64,
            }
        )
    scratch = {
        "labels_present": False,
        "target_rows_present": False,
        "source_frame_hash": "6" * 64,
        "array_path": str(eval_path),
        "array_sha256": "7" * 64,
        "views": views,
    }
    tasks = build_held_prediction_tasks(
        FakeConfig(),
        FakeSource(),
        scratch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert len(tasks) == 36 * 3 * 3
    assert {task["threads_per_fit"] for task in tasks} == {1}
    assert {len(task["candidate_sources"]) for task in tasks} == {7}
    assert {len(task["actions"]) for task in tasks} == {9}
    assert {tuple(view["center"] for view in task["evaluation_views"]) for task in tasks} == {
        pair for pair, _hashes in canonical_held_action_library().pair_action_hashes
    }
    _validate_task(tasks[0])
    drift = dict(tasks[0])
    drift["candidate_sources"] = list(reversed(drift["candidate_sources"]))
    drift["task_hash"] = canonical_hash(
        {
            key: value
            for key, value in drift.items()
            if key not in {"task_hash", "checkpoint_npz_path", "checkpoint_json_path"}
        }
    )
    with pytest.raises(ProtocolError, match="task boundary"):
        _validate_task(drift)

    resumed = build_held_prediction_tasks(
        FakeConfig(),
        FakeSource(),
        scratch,
        checkpoint_root=tmp_path / "checkpoints",
    )
    assert resumed == tasks
    (tmp_path / "checkpoints" / "foreign.checkpoint").write_text("unsafe")
    with pytest.raises(ProtocolError, match="inventory drifted"):
        build_held_prediction_tasks(
            FakeConfig(),
            FakeSource(),
            scratch,
            checkpoint_root=tmp_path / "checkpoints",
        )


def test_exact_five_file_source_loader_keeps_outcomes_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production import source_frame

    root = tmp_path / "cache"
    tensor_path = root / "embeddings/train.pt"
    tensor_path.parent.mkdir(parents=True)
    frozen = {
        "schema_version": "midogpp_uniform_b_canonical_cache_protocol_v1",
        "cache_name": "canonical",
        "representation_id": "annotation_jpeg_fixed_center_b_v3",
        "transformation": "lossless_center_shard_concatenation_in_manifest_order",
        "split": "train",
        "eligible_centers": list(CENTERS),
        "row_count": 9,
        "feature_dim": 3,
        "source_shard_sha256": {center: "x" for center in CENTERS},
        "labels_used_for_feature_construction": False,
        "test_rows_present": False,
        "validation_rows_present": False,
    }
    frozen["protocol_hash"] = stable_hash(frozen)
    torch.save(
        {
            "embeddings": torch.arange(27, dtype=torch.float32).reshape(9, 3),
            "metadata": [
                {
                    "sample_id": f"raw-{center}",
                    "case_id": f"case-{center}",
                    "label": index % 2,
                    "split": "train",
                    "center": center,
                    "contract_row_index": index,
                }
                for index, center in enumerate(CENTERS)
            ],
            "feature_extractor": {
                "schema_version": "midogpp_uniform_b_canonical_feature_extractor_v1",
                "model": "Virchow2",
                "dataset": "MIDOG++",
                "representation_id": "annotation_jpeg_fixed_center_b_v3",
                "feature_dim": 3,
                "pooling": "fixed_center_rows6to9_cols6to9",
                "source_protocol_hash": frozen["protocol_hash"],
            },
        },
        tensor_path,
    )
    _json(root / "manifests/frozen_cache_protocol.json", frozen)
    _json(
        root / "reports/cache_builder_report.json",
        {
            "schema_version": "midogpp_uniform_b_canonical_cache_builder_v1",
            "status": "PASS",
            "representation_id": "annotation_jpeg_fixed_center_b_v3",
            "split": "train",
            "row_count": 9,
            "feature_dim": 3,
            "source_shards": 9,
            "numeric_transformation": "none",
            "independent_validation_status": "PASS",
        },
    )
    _json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_uniform_b_canonical_cache_validation_v1",
            "status": "PASS",
            "validator": "validate_uniform_b_canonical_train_cache",
            "checks": {
                "status": "PASS",
                "row_count": 9,
                "center_count": 9,
                "feature_dim": 3,
                "numeric_identity": "EXACT",
            },
        },
    )
    indexed_members = (
        "embeddings/train.pt",
        "manifests/frozen_cache_protocol.json",
        "reports/cache_builder_report.json",
        "reports/validation_report.json",
    )
    index = {
        "schema_version": "midogpp_uniform_b_canonical_cache_content_index_v1",
        "files": [
            {"path": member, "sha256": file_sha256(root / member)}
            for member in indexed_members
        ],
    }
    index["content_hash"] = stable_hash(index)
    _json(root / "manifests/content_index.json", index)
    hashes = tuple(
        (member, file_sha256(root / member))
        for member in (
            "embeddings/train.pt",
            "manifests/frozen_cache_protocol.json",
            "manifests/content_index.json",
            "reports/cache_builder_report.json",
            "reports/validation_report.json",
        )
    )
    monkeypatch.setattr(source_frame, "SOURCE_CACHE_FILE_HASHES", hashes)
    monkeypatch.setattr(source_frame, "RAW_SOURCE_ROW_COUNT", 9)
    monkeypatch.setattr(source_frame, "RAW_SOURCE_CASE_COUNT", 9)
    monkeypatch.setattr(source_frame, "SOURCE_FEATURE_DIM", 3)
    admitted = load_canonical_source_cache(root)
    assert admitted.frame.embeddings.shape == (9, 3)
    assert not hasattr(admitted.frame.rows[0], "outcome")
    with pytest.raises(ProtocolError, match="before prediction sealing"):
        admitted.open_source_outcomes(object())
    library = canonical_held_action_library()
    seal = SourceProbabilitySeal(
        source_frame_hash=admitted.frame.frame_hash,
        source_stream_lock_hash="8" * 16,
        held_action_library_sha256=library.library_hash,
        held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
        oriented_block_receipts=tuple(
            (h, q, canonical_hash((h, q)))
            for h in CENTERS
            for q in CENTERS
            if q != h
        ),
        _factory_token=_PROBABILITY_SEAL_GATE,
    )
    outcomes = admitted.open_source_outcomes(seal)
    assert tuple(row.outcome for row in outcomes) == tuple(index % 2 for index in range(9))


def test_small_contract_atomic_six_member_write_and_reconstructive_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle import contracts, parsing
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production import bundle_writer

    counts = {
        "RAW_SOURCE_ROW_COUNT": 9,
        "RAW_SOURCE_CASE_COUNT": 9,
        "HELD_POOL_BLOCK_COUNT": 72,
        "LOGICAL_SOURCE_ROW_COUNT": 72,
        "LOGICAL_SOURCE_CASE_GROUP_COUNT": 72,
    }
    for module in (contracts, parsing, bundle_writer):
        for name, value in counts.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, value)
    library = canonical_held_action_library()
    task_hashes = tuple(canonical_hash(("task", index)) for index in range(9))
    blocks = []
    for h in CENTERS:
        for q in CENTERS:
            if h == q:
                continue
            candidates = tuple(center for center in CENTERS if center not in {h, q})
            blocks.append(
                HeldBaseProbabilityBlock(
                    outer_target_center=h,
                    query_center=q,
                    row_ids=(f"source_row_{q}",),
                    source_cache_row_indices=(CENTERS.index(q),),
                    probabilities_by_base=(
                        ("B", np.asarray([0.4], dtype=np.float32)),
                        ("U", np.asarray([0.6], dtype=np.float32)),
                        *(
                            (
                                f"A1::source={center}",
                                np.asarray([0.2 + 0.05 * index], dtype=np.float32),
                            )
                            for index, center in enumerate(candidates)
                        ),
                    ),
                    seed_task_hashes=task_hashes,
                    source_frame_hash="1" * 64,
                    source_stream_lock_hash="2" * 16,
                    held_action_library_sha256=library.library_hash,
                    held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
                    _factory_token=_BLOCK_GATE,
                )
            )
    seal = SourceProbabilitySeal(
        source_frame_hash="1" * 64,
        source_stream_lock_hash="2" * 16,
        held_action_library_sha256=library.library_hash,
        held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
        oriented_block_receipts=tuple(
            (block.outer_target_center, block.query_center, block.block_hash)
            for block in blocks
        ),
        _factory_token=_PROBABILITY_SEAL_GATE,
    )
    predictions = HeldPredictionInventory(
        tuple(blocks), seal, _factory_token=_INVENTORY_GATE
    )
    outcomes = tuple(
        SourceOutcomeRow(
            source_cache_row_index=index,
            source_row_id=f"source_row_{center}",
            case_id=f"case-{center}",
            center=center,
            outcome=index % 2,
        )
        for index, center in enumerate(CENTERS)
    )
    output = tmp_path / "source-bundle"
    result = write_source_training_bundle(
        output,
        predictions=predictions,
        source_outcomes=outcomes,
        producer_source_seal_sha256="3" * 64,
    )
    assert result.production_receipt.read_back_validated is True
    assert result.surface.receipt.row_count == 72
    assert tuple(
        sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
    ) == tuple(sorted(contracts.SOURCE_SUPERVISION_MEMBERS))
    with pytest.raises(ProtocolError, match="fresh"):
        write_source_training_bundle(
            output,
            predictions=predictions,
            source_outcomes=outcomes,
            producer_source_seal_sha256="3" * 64,
        )
