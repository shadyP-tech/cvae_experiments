from __future__ import annotations

from collections import OrderedDict
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.constants import (
    B_ACTION_ID,
    CENTERS,
    OOF_FOLD_COUNT,
    PERMUTATION_COUNT,
    a1_action_id,
    candidate_sources,
    decision_action_ids,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.decisions import (
    select_global_static_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.persistence import (
    ACTION_SCORE_FIELDS,
    SELECTION_FIELDS,
    TERMINAL_CHECKPOINT_MEMBER,
    finalize_terminal_checkpoint,
    load_terminal_checkpoint,
    persist_global_static,
    persist_terminal_checkpoint,
    remove_validated_terminal_checkpoint,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.prediction_adapter import (
    _stage_canonical_source_nonrepairing,
    run_workstation_preflight,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.probability_surfaces import (
    build_prediction_row_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.products import (
    BinaryLabelRow,
    BinaryPredictionRow,
    CaseActionCounts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.runner import (
    _finalize_bundle,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.scoring import (
    score_case_action_counts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.terminal import (
    _evaluate_null_statistics,
    _persist_or_validate_null_array,
    load_null_selection_plan_seal,
    seal_null_selection_plans,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    sha256_array,
)
from midogpp_thesis.cvae.runtime.preflight import REQUIRED_DISTRIBUTIONS
from midogpp_thesis.common.hashing import stable_hash


SHA = "a" * 64


def _preflight_payload(*, free_mib: int) -> dict[str, object]:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.experiment_contracts import (
        SCRATCH_ROOT,
    )

    return {
        "schema_version": "midogpp_label_free_workstation_preflight_v1",
        "status": "PASS",
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_classifier_worker": 3,
        "target_action_identity_count": 90,
        "target_probability_cell_count": 810,
        "target_unique_classifier_fit_count": 810,
        "maximum_total_classifier_fit_count": 810,
        "gpu_then_cpu_phase_order": True,
        "phase_disjoint_gpu_and_cpu_pools": True,
        "parent_cuda_initialized": False,
        "tf32_enabled": False,
        "amp_enabled": False,
        "scratch_preference": [SCRATCH_ROOT, "artifact_parent"],
        "available_cpu_affinity_count": 24,
        "physical_ram_bytes": 128 * 1024**3,
        "disk_probe_path": "/artifact-filesystem",
        "disk_free_bytes_at_launch": 100 * 1024**3,
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "cuda_visible_devices": "0,1",
        "package_versions": {name: "test" for name in REQUIRED_DISTRIBUTIONS},
        "gpus": [
            {
                "index": index,
                "name": "NVIDIA RTX A5000",
                "memory_total_mib": 24_576,
                "memory_free_mib": free_mib,
            }
            for index in (0, 1)
        ],
    }


def test_preflight_retry_reprobes_on_artifact_filesystem_and_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.prediction_adapter as adapter

    root = tmp_path / "bundle"
    probes: list[Path] = []

    def fake_probe(probe_root: Path, **_kwargs: object) -> dict[str, object]:
        probes.append(probe_root)
        payload = _preflight_payload(free_mib=20_000 + len(probes))
        atomic_json(probe_root / "reports/workstation_preflight.json", payload)
        return payload

    monkeypatch.setattr(adapter, "run_label_free_workstation_preflight", fake_probe)
    runtime = canonical_runtime_payload()
    admitted = run_workstation_preflight(root, runtime=runtime)
    report = root / "reports/workstation_preflight.json"
    admitted_bytes = report.read_bytes()

    replayed = run_workstation_preflight(root, runtime=runtime)

    assert replayed == admitted
    assert report.read_bytes() == admitted_bytes
    assert len(probes) == 2
    assert probes[0] == root
    assert probes[1].parent.parent == root.parent
    assert probes[1].parent.name.startswith(".midogpp-s4-preflight-reprobe-")


def test_s4_canonical_source_stage_rejects_tamper_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    scratch = tmp_path / "scratch"
    destination = tmp_path / "canonical"
    members = (
        source_runtime.SOURCE_ARRAY_MEMBER,
        source_runtime.SOURCE_INDEX_MEMBER,
        source_runtime.SOURCE_LOCK_MEMBER,
    )
    for member, payload in zip(members, (b"array", b"index", b"lock"), strict=True):
        path = scratch / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    cache = SimpleNamespace(
        root=scratch,
        lock_payload={
            "source_array_sha256": source_runtime.sha256_file(
                scratch / members[0]
            ),
            "source_stream_index_sha256": source_runtime.sha256_file(
                scratch / members[1]
            ),
            "config_contract_hash": "contract",
            "generation_lock_hash": "generation",
        },
    )
    array = destination / members[0]
    array.parent.mkdir(parents=True, exist_ok=True)
    array.write_bytes(b"tampered-canonical")
    tampered = array.read_bytes()
    monkeypatch.setattr(
        source_runtime,
        "stage_frozen_source_streams",
        lambda *_args, **_kwargs: pytest.fail("tampered canonical must not be staged"),
    )

    with pytest.raises(ProtocolError, match="differs; refusing repair"):
        _stage_canonical_source_nonrepairing(cache, destination=destination)
    assert array.read_bytes() == tampered


def test_s4_canonical_source_stage_rejects_nested_parent_symlink(
    tmp_path: Path,
) -> None:
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    scratch = tmp_path / "scratch"
    destination = tmp_path / "canonical"
    external = tmp_path / "external"
    external.mkdir()
    destination.mkdir()
    (destination / "arrays").symlink_to(external, target_is_directory=True)
    cache = SimpleNamespace(
        root=scratch,
        lock_payload={
            "source_array_sha256": "0" * 64,
            "source_stream_index_sha256": "1" * 64,
            "config_contract_hash": "contract",
            "generation_lock_hash": "generation",
        },
    )
    (scratch / source_runtime.SOURCE_LOCK_MEMBER).parent.mkdir(
        parents=True, exist_ok=True
    )
    (scratch / source_runtime.SOURCE_LOCK_MEMBER).write_bytes(b"lock")

    with pytest.raises(ProtocolError, match="source parent is unsafe"):
        _stage_canonical_source_nonrepairing(cache, destination=destination)
    assert not tuple(external.iterdir())


def test_materialize_sources_reuses_valid_canonical_trio_before_gpu_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.prediction_adapter as adapter
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    root = tmp_path / "canonical"
    for member in (
        source_runtime.SOURCE_ARRAY_MEMBER,
        source_runtime.SOURCE_INDEX_MEMBER,
        source_runtime.SOURCE_LOCK_MEMBER,
    ):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")
    sentinel = object()
    monkeypatch.setattr(adapter, "load_frozen_source_streams", lambda *_a, **_k: sentinel)
    monkeypatch.setattr(
        adapter,
        "_owned_scratch_base",
        lambda **_kwargs: pytest.fail("valid canonical source must skip GPU scratch"),
    )

    observed = adapter.materialize_sources(
        SimpleNamespace(contract_hash="contract"),
        SimpleNamespace(generation_lock_hash="generation"),
        root=root,
    )
    assert observed is sentinel


def test_neutral_source_all_final_branch_validates_and_cleans_all_27_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.common.hashing import stable_hash
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    root = tmp_path / "source"
    for member in (
        source_runtime.SOURCE_ARRAY_MEMBER,
        source_runtime.SOURCE_INDEX_MEMBER,
        source_runtime.SOURCE_LOCK_MEMBER,
    ):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed-final")
    checkpoint = root / source_runtime.CHECKPOINT_DIRECTORY
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(source_runtime, "SOURCE_ROWS_PER_CLASS", 1)
    monkeypatch.setattr(source_runtime, "COMMON_OUTPUT_DIM", 2)

    tasks = []
    by_key = {}
    for task_ordinal, (source, training_seed) in enumerate(
        (source, seed)
        for source in CENTERS
        for seed in (17, 42, 101)
    ):
        keys = tuple(
            SimpleNamespace(
                generation_seed=generation_seed,
                stream_id=f"stream-{source}-{training_seed}-{generation_seed}",
                expert_lock_hash=f"expert-{source}-{training_seed}",
            )
            for generation_seed in (17, 42, 101)
        )
        stem = f"source_{source}_train_{training_seed}"
        array_path = checkpoint / f"{stem}.npy"
        values = np.full((3, 2, 2), task_ordinal, dtype=np.float32)
        with array_path.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        records = []
        for ordinal, key in enumerate(keys):
            output_hash = source_runtime._array_bundle_sha256(values[ordinal])
            records.append(
                {
                    "generation_seed": key.generation_seed,
                    "stream_id": key.stream_id,
                    "expert_lock_hash": key.expert_lock_hash,
                    "output_sha256": output_hash,
                    "array_sha256": sha256_array(values[ordinal]),
                }
            )
            by_key[(source, training_seed, key.generation_seed)] = SimpleNamespace(
                stream_id=key.stream_id,
                expert_lock_hash=key.expert_lock_hash,
                output_sha256=output_hash,
            )
        task = {
            "schema_version": "midogpp_frozen_source_stream_task_v1",
            "task_ordinal": task_ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "generation_keys": keys,
            "device": ("cuda:0", "cuda:1")[task_ordinal % 2],
            "expert_bank_root": str(tmp_path / "bank"),
            "checkpoint_path": str(checkpoint / f"{stem}.json"),
            "array_path": str(array_path),
            "config_contract_hash": "config",
            "generation_lock_hash": "generation",
            "labels_available": False,
            "amp_enabled": False,
            "tf32_enabled": False,
        }
        unhashed = {
            "schema_version": "midogpp_frozen_source_stream_checkpoint_v1",
            "status": "COMPLETE",
            "config_contract_hash": "config",
            "generation_lock_hash": "generation",
            "task_ordinal": task_ordinal,
            "source_center": source,
            "training_seed": training_seed,
            "device": task["device"],
            "array_path": str(array_path),
            "array_file_sha256": source_runtime.sha256_file(array_path),
            "records": records,
            "labels_consumed": False,
            "source_expert_updated": False,
            "tf32_disabled": True,
            "amp_disabled": True,
            "float32_outputs": True,
        }
        atomic_json(
            Path(str(task["checkpoint_path"])),
            {**unhashed, "checkpoint_hash": stable_hash(unhashed)},
        )
        tasks.append(task)

    cache = SimpleNamespace(by_key=by_key)
    config = SimpleNamespace(contract_hash="config", runtime={}, expert_bank_root=tmp_path)
    generation_lock = SimpleNamespace(generation_lock_hash="generation")
    monkeypatch.setattr(source_runtime, "_assert_runtime", lambda _runtime: None)
    monkeypatch.setattr(
        source_runtime,
        "load_frozen_source_streams",
        lambda *_args, **_kwargs: cache,
    )
    monkeypatch.setattr(
        source_runtime, "_build_tasks", lambda *_args, **_kwargs: tuple(tasks)
    )

    assert (
        source_runtime.materialize_frozen_source_streams(
            config, generation_lock, root=root
        )
        is cache
    )
    assert not checkpoint.exists()


@pytest.mark.parametrize("drift", ("unknown", "tampered"))
def test_neutral_source_completed_checkpoint_cleanup_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    root = tmp_path / "source"
    checkpoint = root / source_runtime.CHECKPOINT_DIRECTORY
    checkpoint.mkdir(parents=True)
    if drift == "unknown":
        changed = checkpoint / "foreign.bin"
        changed.write_bytes(b"foreign")
        error = "unknown member"
    else:
        changed = checkpoint / "source_0_train_17.npy"
        changed.write_bytes(b"not-an-npy")
        error = "unreadable"
        task = {
            "source_center": "0",
            "training_seed": 17,
            "generation_keys": tuple(
                SimpleNamespace(generation_seed=seed) for seed in (17, 42, 101)
            ),
            "checkpoint_path": str(checkpoint / "source_0_train_17.json"),
            "array_path": str(changed),
        }
        monkeypatch.setattr(
            source_runtime, "_build_tasks", lambda *_a, **_k: (task,)
        )
    config = SimpleNamespace(contract_hash="config", expert_bank_root=tmp_path)
    generation = SimpleNamespace(generation_lock_hash="generation")
    cache = SimpleNamespace(by_key={})

    with pytest.raises(ProtocolError, match=error):
        source_runtime._cleanup_completed_checkpoint_remnants(
            config, generation, root=root, cache=cache
        )
    assert changed.exists()


def test_neutral_source_task_array_only_crash_predecessor_is_nonrepairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    monkeypatch.setattr(source_runtime, "SOURCE_ROWS_PER_CLASS", 1)
    monkeypatch.setattr(source_runtime, "COMMON_OUTPUT_DIM", 2)
    keys = tuple(
        SourceGenerationKey(
            source_center="0",
            training_seed=17,
            generation_seed=seed,
            expert_lock_hash="expert-lock",
            stream_id=f"stream-{seed}",
            class_seed_by_label={"0": seed, "1": seed + 1},
        )
        for seed in (17, 42, 101)
    )
    embeddings = {
        key.generation_seed: np.full(
            (2, 2), ordinal + 0.25, dtype=np.float32
        )
        for ordinal, key in enumerate(keys)
    }
    expected = np.ascontiguousarray(
        np.stack([embeddings[key.generation_seed] for key in keys]),
        dtype=np.float32,
    )

    def fake_generate(
        _expert: object,
        key: SourceGenerationKey,
        *,
        per_class: int,
        device: str,
    ) -> SimpleNamespace:
        assert per_class == 1
        assert device == "cpu"
        values = embeddings[key.generation_seed]
        return SimpleNamespace(
            key=key,
            embeddings=values,
            output_sha256=source_runtime._array_bundle_sha256(values),
        )

    monkeypatch.setattr(
        source_runtime,
        "load_routing_authorized_expert",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(source_runtime, "generate_source_block", fake_generate)

    def task(directory: Path) -> dict[str, object]:
        return {
            "schema_version": "midogpp_frozen_source_stream_task_v1",
            "task_ordinal": 0,
            "source_center": "0",
            "training_seed": 17,
            "generation_keys": keys,
            "device": "cpu",
            "expert_bank_root": str(tmp_path / "bank"),
            "checkpoint_path": str(directory / "source_0_train_17.json"),
            "array_path": str(directory / "source_0_train_17.npy"),
            "config_contract_hash": "config",
            "generation_lock_hash": "generation",
            "labels_available": False,
            "amp_enabled": False,
            "tf32_enabled": False,
        }

    valid_task = task(tmp_path / "valid")
    valid_array = Path(str(valid_task["array_path"]))
    valid_array.parent.mkdir(parents=True)
    with valid_array.open("wb") as handle:
        np.save(handle, expected, allow_pickle=False)
    admitted_bytes = valid_array.read_bytes()

    payload = source_runtime._generate_task(valid_task)

    assert valid_array.read_bytes() == admitted_bytes
    assert Path(str(valid_task["checkpoint_path"])).is_file()
    assert (
        source_runtime._load_checkpoint(
            Path(str(valid_task["checkpoint_path"])), task=valid_task
        )["checkpoint_hash"]
        == payload["checkpoint_hash"]
    )

    changed_task = task(tmp_path / "changed")
    changed_array = Path(str(changed_task["array_path"]))
    changed_array.parent.mkdir(parents=True)
    changed = expected.copy()
    changed[0, 0, 0] += 1.0
    with changed_array.open("wb") as handle:
        np.save(handle, changed, allow_pickle=False)
    changed_bytes = changed_array.read_bytes()

    with pytest.raises(ProtocolError, match="array differs; refusing repair"):
        source_runtime._generate_task(changed_task)
    assert changed_array.read_bytes() == changed_bytes
    assert not Path(str(changed_task["checkpoint_path"])).exists()


def test_neutral_source_partial_array_and_index_are_nonrepairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.runtime import frozen_source_streams as source_runtime

    monkeypatch.setattr(source_runtime, "SOURCE_ROWS_PER_CLASS", 1)
    monkeypatch.setattr(source_runtime, "COMMON_OUTPUT_DIM", 2)
    monkeypatch.setattr(source_runtime, "EXPECTED_STREAM_COUNT", 3)
    task_array = tmp_path / "task.npy"
    values = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    with task_array.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    records = tuple(
        {
            "generation_seed": seed,
            "stream_id": f"stream-{seed}",
            "expert_lock_hash": "expert",
            "output_sha256": source_runtime._array_bundle_sha256(values[index]),
        }
        for index, seed in enumerate((17, 42, 101))
    )
    task = {"source_center": "0", "training_seed": 17}
    completed = {
        ("0", 17): {"array_path": str(task_array), "records": records}
    }
    final = tmp_path / "arrays/frozen_source_streams.npy"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"changed-final-array")
    changed_bytes = final.read_bytes()
    with pytest.raises(ProtocolError, match="array differs; refusing repair"):
        source_runtime._materialize_array(
            final, tasks=(task,), completed=completed
        )
    assert final.read_bytes() == changed_bytes

    index = tmp_path / "manifests/index.json"
    atomic_json(index, {"status": "changed"})
    index_bytes = index.read_bytes()
    with pytest.raises(ProtocolError, match="JSON differs; refusing repair"):
        source_runtime._persist_or_validate_json(index, {"status": "expected"})
    assert index.read_bytes() == index_bytes


def test_changed_partial_preflight_fails_before_retry_probe_or_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.prediction_adapter as adapter

    root = tmp_path / "bundle"
    calls = 0

    def fake_probe(probe_root: Path, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        payload = _preflight_payload(free_mib=20_000)
        atomic_json(probe_root / "reports/workstation_preflight.json", payload)
        return payload

    monkeypatch.setattr(adapter, "run_label_free_workstation_preflight", fake_probe)
    runtime = canonical_runtime_payload()
    run_workstation_preflight(root, runtime=runtime)
    changed = _preflight_payload(free_mib=20_000)
    changed["classifier_workers"] = 99
    report = root / "reports/workstation_preflight.json"
    atomic_json(report, changed)
    changed_bytes = report.read_bytes()

    with pytest.raises(ProtocolError, match="preflight topology drifted"):
        run_workstation_preflight(root, runtime=runtime)

    assert calls == 1
    assert report.read_bytes() == changed_bytes


def _global_static_donors() -> OrderedDict[str, tuple[CaseActionCounts, ...]]:
    result: OrderedDict[str, tuple[CaseActionCounts, ...]] = OrderedDict()
    for source in candidate_sources("0"):
        action = a1_action_id(source)
        rows = []
        for query in CENTERS:
            if query in {"0", source}:
                continue
            case_id = f"q{query}"
            rows.append(CaseActionCounts(query, case_id, B_ACTION_ID, 10, 5, 10, 5))
            correct = 8 if source == "1" else 4
            rows.append(CaseActionCounts(query, case_id, action, 10, correct, 10, correct))
        result[action] = tuple(rows)
    return result


def test_live_score_persistence_and_replay_use_generic_score_fields(
    tmp_path: Path,
) -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_support_static_router.validation_science import (
        _action_score_rows,
        _assert_table,
        _selection_row,
    )

    selection = select_global_static_action(
        "0", _global_static_donors(), prerequisite_seal_hash=SHA
    )
    persist_global_static(
        tmp_path,
        selections=(selection,),
        seal_payload={"test_seal": SHA},
    )
    _assert_table(
        tmp_path / "tables/global_static_action_scores.csv",
        _action_score_rows(selection, fold_ordinal=-1),
        ACTION_SCORE_FIELDS,
    )
    _assert_table(
        tmp_path / "tables/global_static_selections.csv",
        (_selection_row(selection, fold_ordinal=-1),),
        SELECTION_FIELDS,
    )
    with (tmp_path / "tables/global_static_action_scores.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        fields = tuple(csv.DictReader(handle).fieldnames or ())
    assert fields == ACTION_SCORE_FIELDS
    assert "score_type" in fields
    assert "pooled_bacc" not in fields


def _terminal_result() -> dict[str, object]:
    return {
        "method_decisions": [
            {
                "target_center": "0",
                "fold_ordinal": 0,
                "case_id": "case",
                "method_id": "B",
                "action_id": "B",
                "route_decision_hash": SHA,
                "evaluation_labels_used_for_decision": False,
                "row_hash": SHA,
            }
        ],
        "terminal_case_confusions": [
            {
                "target_center": "0",
                "fold_ordinal": 0,
                "case_id": "case",
                "method_id": "B",
                "action_id": "B",
                "n_positive": 1,
                "true_positive": 1,
                "n_negative": 1,
                "true_negative": 1,
                "row_hash": SHA,
            }
        ],
        "terminal_center_metrics": [
            {
                "target_center": "0",
                "method_id": "B",
                "case_count": 1,
                "n_positive": 1,
                "true_positive": 1,
                "n_negative": 1,
                "true_negative": 1,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "exact_bacc": 1.0,
                "row_hash": SHA,
            }
        ],
        "terminal_contrasts": [
            {
                "contrast_id": "S4-B",
                "method_id": "S4",
                "baseline_id": "B",
                "estimate": 0.0,
                "ci_low": 0.0,
                "ci_high": 0.0,
                "center_estimates": [0.0] * 9,
                "outer_n": 9,
                "outer_df": 8,
                "descriptive_only": True,
                "confirmatory_p_value": False,
                "pass_gate_used": False,
                "row_hash": SHA,
            }
        ],
        "null_route_selection_counts": [
            {
                "target_center": "0",
                "fold_ordinal": 0,
                "action_id": "B",
                "selection_count": 10_000,
                "replicate_count": 10_000,
                "route_null_selection_hash": SHA,
            }
        ],
        "action_identity_null_summary": {
            "exchangeability_claimed": False,
            "confirmatory_p_value": False,
            "pass_gate_used": False,
        },
        "action_identity_null_seal": {"null_seal_hash": SHA},
        "sealed_terminal_evaluation": {
            "sealed_result_hash": SHA,
            "raw_labels_persisted": False,
        },
    }


def test_terminal_checkpoint_roundtrip_is_atomic_nonrepairing_and_schema_exact(
    tmp_path: Path,
) -> None:
    result = _terminal_result()
    kwargs = {
        "result": result,
        "capability_report": {"status": "PASS"},
        "leakage_report": {"status": "PASS"},
        "publication_decision": {"decision": "DO_NOT_PROMOTE"},
        "runtime_summary": {"status": "PASS"},
    }
    written = persist_terminal_checkpoint(tmp_path, **kwargs)
    assert load_terminal_checkpoint(tmp_path) == written
    checkpoint_bytes = (tmp_path / TERMINAL_CHECKPOINT_MEMBER).read_bytes()
    assert persist_terminal_checkpoint(tmp_path, **kwargs) == written
    assert (tmp_path / TERMINAL_CHECKPOINT_MEMBER).read_bytes() == checkpoint_bytes

    changed = dict(kwargs)
    changed["runtime_summary"] = {"status": "CHANGED"}
    with pytest.raises(ProtocolError, match="refusing repair"):
        persist_terminal_checkpoint(tmp_path, **changed)

    finalize_terminal_checkpoint(tmp_path)
    assert (tmp_path / "tables/method_decisions.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0] == (
        "target_center,fold_ordinal,case_id,method_id,action_id,"
        "route_decision_hash,evaluation_labels_used_for_decision,row_hash"
    )
    remove_validated_terminal_checkpoint(tmp_path)
    assert not (tmp_path / "checkpoints").exists()


def test_compact_null_npz_is_reused_byte_exact_and_changed_matrix_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "arrays/action_identity_null_selections.npz"
    matrix = np.zeros((PERMUTATION_COUNT, 45), dtype=np.uint8)
    _persist_or_validate_null_array(path, matrix)
    first_bytes = path.read_bytes()
    _persist_or_validate_null_array(path, matrix.copy())
    assert path.read_bytes() == first_bytes
    changed = matrix.copy()
    changed[0, 0] = 1
    with pytest.raises(ProtocolError, match="differs; refusing repair"):
        _persist_or_validate_null_array(path, changed)
    assert path.read_bytes() == first_bytes


def test_tampered_compact_null_array_is_rejected_before_evaluation_barrier(
    tmp_path: Path,
) -> None:
    plans = []
    for target in CENTERS:
        selection = SimpleNamespace(action_id=B_ACTION_ID)
        selections = (selection,) * PERMUTATION_COUNT
        for fold_ordinal in range(OOF_FOLD_COUNT):
            plans.append(
                SimpleNamespace(
                    target_center=target,
                    fold_ordinal=fold_ordinal,
                    permutation_count=PERMUTATION_COUNT,
                    plan_hash=(f"{len(plans) + 1:064x}"),
                    selections=selections,
                )
            )
    decision_hash = "d" * 64
    partition_hash = "e" * 64
    seal_null_selection_plans(
        tmp_path,
        plans=plans,
        decision_seal_hash=decision_hash,
        partition_hash=partition_hash,
    )
    array_path = tmp_path / "arrays/action_identity_null_selections.npz"
    changed = np.zeros((PERMUTATION_COUNT, 45), dtype=np.uint8)
    changed[0, 0] = 1
    with array_path.open("wb") as handle:
        np.savez_compressed(handle, selected_action_index=changed)

    with pytest.raises(ProtocolError, match="pre-evaluation null selection seal drifted"):
        load_null_selection_plan_seal(
            tmp_path,
            plans=plans,
            decision_seal_hash=decision_hash,
            partition_hash=partition_hash,
        )

def test_prediction_row_index_matches_full_surface_scoring_without_full_scan() -> None:
    rows = []
    labels = []
    for case_id in ("case-a", "case-b"):
        for ordinal, value in enumerate((0, 1)):
            sample_id = f"{case_id}-sample-{ordinal}"
            labels.append(BinaryLabelRow("0", case_id, sample_id, value))
            for action_ordinal, action in enumerate(physical_action_ids("0")):
                probability = 0.8 if (value + action_ordinal) % 2 else 0.2
                rows.append(
                    BinaryPredictionRow(
                        "0", case_id, sample_id, action, probability, SHA
                    )
                )
    index = build_prediction_row_index(rows, surface_hash=SHA)
    scoped_labels = tuple(row for row in labels if row.case_id == "case-a")
    gathered = index.for_labels(scoped_labels)

    assert len(gathered) == 10 * len(scoped_labels)
    assert len(gathered) < len(rows)
    assert score_case_action_counts(rows, scoped_labels) == score_case_action_counts(
        gathered, scoped_labels
    )


@pytest.mark.parametrize("tampered", (False, True))
def test_prediction_task_npz_only_crash_predecessor_is_nonrepairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered: bool
) -> None:
    from midogpp_thesis.cvae.runtime import fixed_bank_a1_prediction_worker as worker

    actions = [
        {"action_id": f"action-{index}", "action_hash": f"hash-{index}"}
        for index in range(10)
    ]
    task = {
        "schema_version": "test-task-v1",
        "task_id": "task",
        "target_center": "0",
        "training_seed": 17,
        "generation_seed": 17,
        "target_start": 0,
        "target_stop": 2,
        "target_row_identity_hash": SHA,
        "labels_available": False,
        "target_expert_available": False,
        "classifier": {},
        "threads_per_fit": 3,
        "candidate_sources": list(candidate_sources("0")),
        "actions": actions,
        "checkpoint_npz_path": str(tmp_path / "checkpoint.npz"),
        "checkpoint_json_path": str(tmp_path / "checkpoint.json"),
    }
    unhashed = {
        key: value
        for key, value in task.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    task["task_hash"] = stable_hash(unhashed)
    probabilities = np.asarray((0.25, 0.75), dtype=np.float32)
    expected_matrix = np.stack([probabilities] * 10).astype(np.float32)
    existing = expected_matrix.copy()
    if tampered:
        existing[0, 0] = np.float32(0.5)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    with npz_path.open("wb") as handle:
        np.savez(handle, probabilities=existing)
    original_bytes = npz_path.read_bytes()

    monkeypatch.setattr(
        worker,
        "load_task_arrays",
        lambda _task: ({}, np.zeros((2, 2), dtype=np.float32)),
    )
    monkeypatch.setattr(worker, "classifier_from_payload", lambda _raw: object())
    monkeypatch.setattr(
        worker,
        "compose_action",
        lambda *_args: (
            np.zeros((2, 2), dtype=np.float32),
            np.asarray((0, 1), dtype=np.uint8),
            np.ones(2, dtype=np.float64),
            "composition",
        ),
    )
    fitted = SimpleNamespace(
        probabilities=np.asarray(((0.75, 0.25), (0.25, 0.75))),
        classes=(0, 1),
        converged=True,
        classifier_config_hash="classifier",
        scaler_state_hash="scaler",
    )
    monkeypatch.setattr(worker, "fit_logistic_classifier", lambda *_a, **_k: fitted)

    if tampered:
        with pytest.raises(ProtocolError, match="array differs; refusing repair"):
            worker.execute_prediction_task(task)
        assert not Path(str(task["checkpoint_json_path"])).exists()
    else:
        worker.execute_prediction_task(task)
        assert Path(str(task["checkpoint_json_path"])).is_file()
        assert worker.load_prediction_checkpoint(task) is not None
    assert npz_path.read_bytes() == original_bytes


@pytest.mark.parametrize("tampered", (False, True))
def test_target_scratch_array_only_crash_predecessor_is_nonrepairing(
    tmp_path: Path, tampered: bool
) -> None:
    from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
    from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
        CHECKPOINT_DIRECTORY,
    )
    from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_planning import (
        write_target_scratch,
    )

    rows_by_center = {
        center: (
            SimpleNamespace(
                evaluation_row_id=f"row-{center}",
                case_id=f"case-{center}",
                center=center,
            ),
        )
        for center in CENTERS
    }
    embeddings = {
        id(row): np.full(
            COMMON_OUTPUT_DIM,
            float(index),
            dtype=np.float32,
        )
        for index, center in enumerate(CENTERS)
        for row in rows_by_center[center]
    }

    class Frame:
        def __init__(self) -> None:
            self.rows_by_center = rows_by_center

        def embeddings_for(self, rows: object) -> np.ndarray:
            return np.stack([embeddings[id(row)] for row in rows])

    frame = Frame()
    first = write_target_scratch(tmp_path, frame, SHA, "binding")
    manifest = tmp_path / CHECKPOINT_DIRECTORY / "target_scratch.json"
    array = tmp_path / CHECKPOINT_DIRECTORY / "target_embeddings.npy"
    manifest.unlink()
    if tampered:
        values = np.load(array, allow_pickle=False)
        values[0, 0] = np.float32(values[0, 0] + 1.0)
        with array.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
    predecessor_bytes = array.read_bytes()

    if tampered:
        with pytest.raises(ProtocolError, match="array differs; refusing repair"):
            write_target_scratch(tmp_path, frame, SHA, "binding")
        assert not manifest.exists()
    else:
        replayed = write_target_scratch(tmp_path, frame, SHA, "binding")
        assert replayed == first
        assert manifest.is_file()
    assert array.read_bytes() == predecessor_bytes


def test_vectorized_null_terminal_statistic_matches_scalar_replay() -> None:
    folds = []
    counts: dict[tuple[str, str, str], CaseActionCounts] = {}
    for target in CENTERS:
        for fold_ordinal in range(OOF_FOLD_COUNT):
            case_id = f"{target}-eval-{fold_ordinal}"
            folds.append(
                SimpleNamespace(
                    target_center=target,
                    fold_ordinal=fold_ordinal,
                    evaluation_case_ids=(case_id,),
                )
            )
            for action_ordinal, action in enumerate(decision_action_ids(target)):
                n_positive = 11 + fold_ordinal
                n_negative = 13 + fold_ordinal
                counts[(target, case_id, action)] = CaseActionCounts(
                    target,
                    case_id,
                    action,
                    n_positive,
                    min(n_positive, 3 + action_ordinal + fold_ordinal),
                    n_negative,
                    min(n_negative, 4 + 2 * action_ordinal + fold_ordinal),
                )
    partition = SimpleNamespace(folds=tuple(folds))
    matrix = np.random.default_rng(90912026).integers(
        0, 9, size=(PERMUTATION_COUNT, 45), dtype=np.uint8
    )
    observed = 0.0125
    vectorized = _evaluate_null_statistics(
        partition=partition,
        counts=counts,
        matrix=matrix,
        observed=observed,
    )

    baseline_by_center = {}
    for center_ordinal, target in enumerate(CENTERS):
        selected_folds = folds[
            center_ordinal * OOF_FOLD_COUNT : (center_ordinal + 1) * OOF_FOLD_COUNT
        ]
        baseline_rows = [
            counts[(target, fold.evaluation_case_ids[0], B_ACTION_ID)]
            for fold in selected_folds
        ]
        baseline_by_center[target] = 0.5 * (
            sum(row.true_positive for row in baseline_rows)
            / sum(row.n_positive for row in baseline_rows)
            + sum(row.true_negative for row in baseline_rows)
            / sum(row.n_negative for row in baseline_rows)
        )
    scalar_values = []
    for replicate in range(PERMUTATION_COUNT):
        center_scores = []
        for center_ordinal, target in enumerate(CENTERS):
            chosen = []
            for fold_ordinal in range(OOF_FOLD_COUNT):
                route_ordinal = center_ordinal * OOF_FOLD_COUNT + fold_ordinal
                fold = folds[route_ordinal]
                action = decision_action_ids(target)[int(matrix[replicate, route_ordinal])]
                chosen.append(counts[(target, fold.evaluation_case_ids[0], action)])
            center_scores.append(
                0.5
                * (
                    sum(row.true_positive for row in chosen)
                    / sum(row.n_positive for row in chosen)
                    + sum(row.true_negative for row in chosen)
                    / sum(row.n_negative for row in chosen)
                )
            )
        scalar_values.append(
            float(np.mean(center_scores, dtype=np.float64))
            - float(np.mean(tuple(baseline_by_center.values()), dtype=np.float64))
        )
    scalar = np.asarray(scalar_values, dtype=np.float64)

    assert vectorized["null_statistics_sha256"] == sha256_array(scalar)
    assert vectorized["null_mean"] == pytest.approx(float(np.mean(scalar)))
    assert vectorized["null_quantile_0_025"] == pytest.approx(
        float(np.quantile(scalar, 0.025))
    )
    assert vectorized["null_quantile_0_5"] == pytest.approx(
        float(np.quantile(scalar, 0.5))
    )
    assert vectorized["null_quantile_0_975"] == pytest.approx(
        float(np.quantile(scalar, 0.975))
    )
    assert vectorized["exceedance_count"] == int(np.count_nonzero(scalar >= observed))


def test_finalization_keeps_validation_pending_until_report_is_persisted(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    checks = {"status": "PASS", "content_hash": "sealed"}
    config = SimpleNamespace(contract_hash="contract", source_path=tmp_path / "config.yaml")
    protocol = SimpleNamespace(contract_hash="protocol")

    def write_content(_root: Path, **kwargs: object) -> None:
        assert kwargs == {
            "config_contract_hash": "contract",
            "protocol_contract_hash": "protocol",
        }
        events.append("content")

    def validate(_root: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs["allow_pending_validation"] is True
        events.append(
            f"validate:skip_fresh={kwargs.get('skip_fresh_process_report', False)}"
        )
        return checks

    def fresh(_root: Path, **kwargs: object) -> dict[str, object]:
        assert kwargs["config_path"] == config.source_path
        events.append("fresh")
        return {"status": "PASS"}

    def persist_fresh(_root: Path, payload: object) -> None:
        assert payload == {"status": "PASS"}
        events.append("persist_fresh")

    def persist_validation(_root: Path, payload: object) -> None:
        assert payload == checks
        events.append("persist_validation")

    def state(_root: Path, **kwargs: object) -> None:
        assert kwargs == {"status": "COMPLETE", "phase": "COMPLETE"}
        events.append("complete_state")

    def completed(_root: Path, **kwargs: object) -> None:
        assert kwargs == {"config": config, "expected_checks": checks}
        events.append("default_completed_validation")

    observed = _finalize_bundle(
        tmp_path,
        config=config,
        protocol=protocol,
        write_content_index_fn=write_content,
        validate_bundle_fn=validate,
        run_fresh_fn=fresh,
        persist_fresh_fn=persist_fresh,
        persist_validation_fn=persist_validation,
        write_state_fn=state,
        assert_completed_fn=completed,
    )

    assert observed == checks
    assert events == [
        "content",
        "validate:skip_fresh=True",
        "fresh",
        "persist_fresh",
        "validate:skip_fresh=False",
        "persist_validation",
        "complete_state",
        "default_completed_validation",
    ]


def test_content_index_rejects_rehashed_claim_or_row_schema_tamper(
    tmp_path: Path,
) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed::{member}\n".encode())
    write_content_index(
        tmp_path,
        config_contract_hash="contract",
        protocol_contract_hash="protocol",
    )
    index_path = tmp_path / "manifests/content_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["promotion_eligible"] = True
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    payload["content_hash"] = canonical_hash(unhashed)
    atomic_json(index_path, payload)
    with pytest.raises(ProtocolError, match="content index header drifted"):
        validate_content_index(
            tmp_path,
            config_contract_hash="contract",
            protocol_contract_hash="protocol",
        )

    payload["promotion_eligible"] = False
    payload["members"][0]["extra"] = "forbidden"
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    payload["content_hash"] = canonical_hash(unhashed)
    atomic_json(index_path, payload)
    with pytest.raises(ProtocolError, match="row is malformed"):
        validate_content_index(
            tmp_path,
            config_contract_hash="contract",
            protocol_contract_hash="protocol",
        )
