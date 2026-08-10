from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.actions import (
    actions_for_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.constants import (
    MIDOGPP_CENTERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
    prediction_runtime as runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
    prediction_store,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
    execution_adapter,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.case_partitions import (
    CaseIdentityRow,
    build_case_oof_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.experiment_contracts import (
    OOF_PARTITION_NAMESPACE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as _TestRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.prediction_contracts import (
    canonical_action_hashes,
    canonical_cell_keys,
    package_scaler_state_hash,
    prediction_store_hash,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, sha256_array
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM


def _runtime_payload() -> dict[str, object]:
    return {
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "phase_disjoint_gpu_and_cpu_pools": True,
        "scientific_reductions_dtype": "float64",
        "target_task_count": 81,
        "target_probability_cell_count": 1458,
        "maximum_total_classifier_fit_count": 1458,
    }


def test_prediction_runtime_public_api_remains_stable() -> None:
    assert runtime.__all__ == (
        "ActionPredictionStore",
        "EXPECTED_CELL_COUNT",
        "EXPECTED_TASK_COUNT",
        "GlobalActionPredictionSeal",
        "PredictionCell",
        "load_global_action_prediction_seal",
        "materialize_action_predictions",
    )


def test_runtime_profile_is_exactly_four_by_three_spawn() -> None:
    runtime._assert_runtime(_runtime_payload())
    for key, bad in (
        ("classifier_workers", 5),
        ("classifier_threads_per_worker", 4),
        ("multiprocessing_start_method", "fork"),
        ("target_probability_cell_count", 729),
    ):
        payload = _runtime_payload()
        payload[key] = bad
        with pytest.raises(ProtocolError):
            runtime._assert_runtime(payload)


def test_prediction_index_guards_reject_drift_before_store_reconstruction(
    tmp_path: Path,
) -> None:
    atomic_json(tmp_path / runtime.GLOBAL_PREDICTION_SEAL_MEMBER, {})
    base = {
        "schema_version": "midogpp_actionability_prediction_index_v1",
        "cell_count": 1458,
        "labels_consumed": False,
        "target_expert_used": False,
    }
    for field, drifted_value in (
        ("cell_count", 1457),
        ("labels_consumed", True),
        ("target_expert_used", True),
    ):
        drifted = {**base, field: drifted_value}
        atomic_json(
            tmp_path / runtime.PREDICTION_INDEX_MEMBER,
            {**drifted, "index_hash": canonical_hash(drifted)},
        )
        with pytest.raises(ProtocolError, match="lineage"):
            runtime.load_global_action_prediction_seal(tmp_path)


def test_prediction_archive_rejects_unindexed_or_missing_members() -> None:
    canonical = tuple(
        f"cell_{ordinal:04d}" for ordinal in range(runtime.EXPECTED_CELL_COUNT)
    )
    prediction_store._validate_archive_members(canonical)

    with pytest.raises(ProtocolError, match="archive members drifted"):
        prediction_store._validate_archive_members((*canonical, "raw_labels"))
    with pytest.raises(ProtocolError, match="archive members drifted"):
        prediction_store._validate_archive_members(canonical[:-1])


def test_a1_reuses_a0_rows_and_changes_only_logistic_fit_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "COMMON_OUTPUT_DIM", 3)
    target = "0"
    candidates = tuple(center for center in MIDOGPP_CENTERS if center != target)
    blocks = {
        source: np.arange(540 * 3, dtype=np.float32).reshape(540, 3)
        + 10_000.0 * ordinal
        for ordinal, source in enumerate(candidates)
    }
    actions = {action.action_id: action for action in actions_for_target(target)}
    source = candidates[0]
    a0 = actions[f"A0::source={source}"].to_payload()
    a1 = actions[f"A1::source={source}"].to_payload()

    a0_x, a0_y, a0_w, a0_hash = runtime._compose_action(blocks, a0, candidates)
    a1_x, a1_y, a1_w, a1_hash = runtime._compose_action(blocks, a1, candidates)

    assert np.array_equal(a0_x, a1_x)
    assert np.array_equal(a0_y, a1_y)
    assert not np.array_equal(a0_w, a1_w)
    assert a0_hash != a1_hash
    for label in (0, 1):
        mask = a1_y == label
        assert np.isclose(a1_w[mask].sum(), 1152.0)
        assert np.isclose(a0_w[a0_y == label].sum(), 1152.0)


def test_action_library_payload_is_exact_and_target_ordered() -> None:
    library = {target: actions_for_target(target) for target in MIDOGPP_CENTERS}
    payload, library_hash = runtime._validate_action_library(library)

    assert tuple(payload) == MIDOGPP_CENTERS
    assert all(len(payload[target]) == 18 for target in MIDOGPP_CENTERS)
    assert len(library_hash) == 64

    reversed_library = dict(reversed(tuple(library.items())))
    with pytest.raises(ProtocolError):
        runtime._validate_action_library(reversed_library)

    bad_action = copy.copy(library[MIDOGPP_CENTERS[0]][0])
    object.__setattr__(bad_action, "action_hash", "a" * 16)
    short_hash_library = {
        **library,
        MIDOGPP_CENTERS[0]: (bad_action, *library[MIDOGPP_CENTERS[0]][1:]),
    }
    with pytest.raises(ProtocolError):
        runtime._validate_action_library(short_hash_library)


def test_prediction_cells_require_sha256_and_store_uses_canonical_cell_order() -> None:
    probability = np.asarray([0.25], dtype=np.float32)
    probability_hash = sha256_array(probability)
    prediction_hash = sha256_array(np.asarray([0], dtype=np.uint8))
    common = {
        "probabilities": probability,
        "probability_sha256": probability_hash,
        "predictions_sha256": prediction_hash,
        "composition_hash": canonical_hash("composition"),
        "scaler_state_hash": canonical_hash("scaler"),
        "fit_provenance_hash": canonical_hash("fit"),
    }
    first_key = canonical_cell_keys()[0]
    with pytest.raises(ProtocolError):
        runtime.PredictionCell(
            target_center=first_key[0],
            action_id=first_key[1],
            action_hash="a" * 16,
            training_seed=first_key[2],
            generation_seed=first_key[3],
            row_identity_hash=canonical_hash(["row", first_key[0]]),
            **common,
        )

    cells = tuple(
        runtime.PredictionCell(
            target_center=target,
            action_id=action,
            action_hash=canonical_action_hashes()[key],
            training_seed=training,
            generation_seed=generation,
            row_identity_hash=canonical_hash(["row", target]),
            **common,
        )
        for key in canonical_cell_keys()
        for target, action, training, generation in (key,)
    )
    rows = {center: (f"row-{center}",) for center in MIDOGPP_CENTERS}
    cases = {center: (f"case-{center}",) for center in MIDOGPP_CENTERS}
    lineage = {
        "source_stream_lock_hash": "1" * 16,
        "action_library_hash": canonical_hash("action-library"),
        "target_cache_binding_hash": canonical_hash("target-cache"),
    }
    store = runtime.ActionPredictionStore(
        cells=cells,
        rows_by_center=rows,
        case_ids_by_center=cases,
        **lineage,
        store_hash=prediction_store_hash(
            cells,
            rows_by_center=rows,
            case_ids_by_center=cases,
            **lineage,
        ),
    )
    assert tuple(cell.key for cell in store.cells) == canonical_cell_keys()

    short_library_lineage = {**lineage, "action_library_hash": "2" * 16}
    with pytest.raises(ProtocolError, match="topology"):
        runtime.ActionPredictionStore(
            cells=cells,
            rows_by_center=rows,
            case_ids_by_center=cases,
            **short_library_lineage,
            store_hash=prediction_store_hash(
                cells,
                rows_by_center=rows,
                case_ids_by_center=cases,
                **short_library_lineage,
            ),
        )

    seal_unhashed = {
        "schema_version": "midogpp_actionability_global_prediction_seal_v1",
        "status": "SEALED_ALL_1458_LABEL_FREE_ACTIONABILITY_CELLS",
        "prediction_store_hash": store.store_hash,
        "source_stream_lock_hash": store.source_stream_lock_hash,
        "action_library_hash": store.action_library_hash,
        "target_cache_binding_hash": store.target_cache_binding_hash,
        "cell_count": 1458,
        "task_count": 81,
        "physical_action_count_per_target": 18,
        "labels_opened": False,
        "target_expert_used": False,
        "seed_selection_used": False,
        "a1_sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
    }
    runtime.GlobalActionPredictionSeal(
        store=store,
        seal_payload={
            **seal_unhashed,
            "global_prediction_seal_hash": canonical_hash(seal_unhashed),
        },
        arrays_path=Path("arrays.npz"),
        index_path=Path("index.json"),
        seal_path=Path("seal.json"),
    )
    for field, drifted_value in (
        ("task_count", 80),
        ("physical_action_count_per_target", 17),
        ("seed_selection_used", True),
        ("a1_sample_weight_scope", "all_pipeline_steps"),
        ("scaler_fit_used_sample_weight", True),
    ):
        drifted = {**seal_unhashed, field: drifted_value}
        with pytest.raises(ProtocolError, match="global prediction seal"):
            runtime.GlobalActionPredictionSeal(
                store=store,
                seal_payload={
                    **drifted,
                    "global_prediction_seal_hash": canonical_hash(drifted),
                },
                arrays_path=Path("arrays.npz"),
                index_path=Path("index.json"),
                seal_path=Path("seal.json"),
            )

    reordered = (cells[1], cells[0], *cells[2:])
    with pytest.raises(ProtocolError, match="topology"):
        runtime.ActionPredictionStore(
            cells=reordered,
            rows_by_center=rows,
            case_ids_by_center=cases,
            **lineage,
            store_hash=prediction_store_hash(
                reordered,
                rows_by_center=rows,
                case_ids_by_center=cases,
                **lineage,
            ),
        )


def test_package_owned_hash_bindings_use_full_sha256() -> None:
    row = _TestRowIdentity(
        row_ordinal=0,
        manifest_row_index=0,
        evaluation_row_id="row-0",
        case_id="case-0",
        center=MIDOGPP_CENTERS[0],
    )
    rows_by_center = {
        center: ((row,) if center == MIDOGPP_CENTERS[0] else ())
        for center in MIDOGPP_CENTERS
    }
    frame = LabelFreeTestFrame(
        embeddings=np.zeros((1, COMMON_OUTPUT_DIM), dtype=np.float32),
        rows=(row,),
        rows_by_center=rows_by_center,
        cache_binding={"cache": "consumed-test"},
    )

    assert frame.cache_binding_hash == canonical_hash({"cache": "consumed-test"})
    assert len(frame.cache_binding_hash) == 64
    assert runtime._target_cache_binding_hash(frame) == frame.cache_binding_hash
    with pytest.raises(ProtocolError, match="cache binding"):
        runtime._target_cache_binding_hash(
            SimpleNamespace(cache_binding_hash="a" * 16)
        )

    scaler_hash = package_scaler_state_hash("a" * 16)
    assert len(scaler_hash) == 64
    with pytest.raises(ProtocolError, match="scaler-state"):
        package_scaler_state_hash("not-a-neutral-hash")


def test_preflight_adapts_only_the_shared_hardware_probe(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"  # type: ignore[operator]
    (root / "reports").mkdir(parents=True)
    observed: dict[str, object] = {}

    def fake_probe(probe_root, *, runtime, expected_scratch_root):
        observed.update(runtime)
        assert expected_scratch_root == execution_adapter.SCRATCH_ROOT
        return {"schema_version": "probe", "status": "PASS"}

    monkeypatch.setattr(execution_adapter, "_preflight", fake_probe)
    configured = {
        **_runtime_payload(),
        "generation_devices": ["cuda:0", "cuda:1"],
        "cuda_visible_devices": "0,1",
        "source_workers_per_device": 1,
        "source_generation_worker_count": 2,
        "persistent_source_workers": True,
        "launch_blas_threads": 1,
        "tf32_enabled": False,
        "amp_enabled": False,
        "generated_cache_format": "float32_npy_memmap",
        "source_job_count": 27,
        "source_stream_count": 81,
        "source_prefix_rows_per_class": 270,
        "model_workers": 4,
        "model_threads_per_worker": 3,
        "bootstrap_workers": 4,
        "bootstrap_threads_per_worker": 3,
        "probability_surface_format": "sealed_compressed_float32_npz",
        "probability_materialization_device": "cpu",
        "physical_actions_per_target_task": 18,
        "logical_actions_per_target": 19,
        "target_unique_classifier_fit_count": 1458,
        "parent_cuda_context_forbidden_during_cpu_phase": True,
        "scratch_preference": [execution_adapter.SCRATCH_ROOT, "artifact_parent"],
        "resume_policy": (
            "hash_validated_source_prediction_task_resume_plus_"
            "deterministic_phase_replay"
        ),
    }

    report = execution_adapter.run_label_free_workstation_preflight(
        root, runtime=configured
    )

    assert observed["target_probability_cell_count"] == 729
    assert observed["resume_policy"] == "hash_validated_atomic_phase_and_task_checkpoints"
    assert report["target_probability_cell_count"] == 1458
    assert report["physical_actions_per_target_task"] == 18


def test_partition_has_its_own_namespace_and_exact_case_coverage() -> None:
    identities = tuple(
        CaseIdentityRow(center, f"case-{ordinal}", f"row-{center}-{ordinal}")
        for center in MIDOGPP_CENTERS
        for ordinal in range(5)
    )

    first = build_case_oof_partition(
        identities, partition_seed=90902029, expected_total_case_count=45
    )
    second = build_case_oof_partition(
        tuple(reversed(identities)),
        partition_seed=90902029,
        expected_total_case_count=45,
    )

    assert first.partition_hash == second.partition_hash
    assert first.partition_namespace == OOF_PARTITION_NAMESPACE
    assert first.to_payload()["schema_version"] == (
        "fixed_bank_actionability_case_oof_partition_v1"
    )
    assert len(first.folds) == 45
    assert all(len(row.evaluation_case_ids) == 1 for row in first.folds)
