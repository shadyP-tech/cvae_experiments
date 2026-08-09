from __future__ import annotations

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.artifact_io import (
    atomic_json,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.feature_production import (
    _validated_development,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.contracts import (
    CaseAwareFeatureSurface,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.runner import (
    CaseAwareProxyAuditRunnerDependencies,
    run_utility_aligned_case_aware_proxy_information_audit,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_case_aware_proxy_information_audit.response_production import (
    _descriptive_seed_rows,
    _validated_feature_lock,
)
from midogpp_thesis.cvae.routing.residual_topup.hashing import (
    array_sha256,
    canonical_sha256,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router import (
    source_cache_planning,
    source_cache_worker,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.source_cache_store import (
    sha256_array,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=(root / "inputs/bank").resolve(),
        generation_lock_root=(root / "inputs/generation").resolve(),
        test_cache_root=(root / "inputs/cache").resolve(),
        test_manifest_path=(root / "inputs/manifest/manifest.csv").resolve(),
        test_consumption_ledger_path=(root / "inputs/ledger/report.json").resolve(),
        metadata_profile_root=(root / "inputs/metadata").resolve(),
        input_artifact_ids=("a", "b", "c", "d", "e", "f"),
        output_artifact_id="output",
        runtime={
            "generation_devices": ["cuda:0", "cuda:1"],
            "classifier_workers": 4,
            "classifier_threads_per_worker": 3,
            "scratch_preference": ["/data/local", "artifact_parent"],
        },
        protocol={
            "support_split_seed": 20_260_809,
            "support_partition_namespace": "case-aware-test",
        },
        contract_hash="a" * 64,
        fixed_support_case_count_per_center=8,
    )


def _launch_files(root: Path) -> None:
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def test_runner_persists_and_rereads_feature_lock_before_test_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "case_aware_proxy_information_audit"
    _launch_files(root)
    config = _config(root)
    phases: list[str] = []
    events: list[str] = []
    partitions = SimpleNamespace(lock_hash="b" * 16)
    cache = SimpleNamespace(source_records=(object(),) * 81)
    development = SimpleNamespace(
        seal=SimpleNamespace(prediction_seal_hash="c" * 16),
        store=SimpleNamespace(cells=(object(),) * 5_184),
    )
    features = SimpleNamespace(rows=(object(),) * 504)
    feature_lock = {"case_aware_feature_lock_hash": "d" * 64}
    responses = SimpleNamespace(
        surface=SimpleNamespace(rows=(object(),) * 504),
        descriptive_seed_rows=({},) * 4_536,
    )
    audit = SimpleNamespace(
        crossfit=SimpleNamespace(
            predictions=(object(),) * 7_056,
            fold_audits=(
                SimpleNamespace(
                    family_id="family",
                    response_name="exact_bacc_delta",
                    predicted_row_key=("0", "1", "2"),
                ),
            )
            * 7_056,
        ),
        query_metrics=(object(),) * 1_008,
        outer_metrics=(object(),) * 126,
        family_summaries=(object(),) * 14,
    )

    def persist_prelabel(_root: Path, **_kwargs: object) -> None:
        events.append("feature_lock_persisted")
        atomic_json(_root / "manifests/proxy_feature_lock.json", feature_lock)

    def open_labels(*_args: object, **_kwargs: object) -> object:
        assert events == ["feature_lock_persisted"]
        events.append("test_labels_opened")
        return object()

    deps = CaseAwareProxyAuditRunnerDependencies(
        validate_inputs=lambda _config: None,
        validate_workspace=lambda _config: {},
        validate_provenance=lambda _root, _config: {},
        load_locks=lambda _config: SimpleNamespace(generation=object()),
        load_frame=lambda _config: object(),
        validate_firewall=lambda _config, _frame: {},
        build_partitions=lambda *_args, **_kwargs: partitions,
        persist_initial=lambda *_args, **_kwargs: None,
        preflight=lambda *_args, **_kwargs: {"status": "PASS"},
        materialize_source=lambda *_args, **_kwargs: cache,
        validate_source=lambda *_args, **_kwargs: {
            "source_cache_lock_hash": "e" * 16
        },
        stage_source=lambda value, **_kwargs: value,
        materialize_development=lambda *_args, **_kwargs: development,
        validate_development_seal=lambda _capability: {},
        load_metadata=lambda _config: {},
        produce_features=lambda *_args, **_kwargs: features,
        build_feature_lock=lambda *_args, **_kwargs: feature_lock,
        persist_prelabel=persist_prelabel,
        open_development_labels=open_labels,
        produce_responses=lambda *_args, **_kwargs: responses,
        run_audit=lambda *_args, **_kwargs: audit,
        build_fold_lock=lambda *_args, **_kwargs: {
            "crossfit_fold_lock_hash": "f" * 64
        },
        persist_postseal=lambda *_args, **_kwargs: None,
        write_index=lambda *_args, **_kwargs: {},
        validate_bundle=lambda *_args, **_kwargs: {"status": "PASS"},
        persist_validation=lambda *_args, **_kwargs: None,
        write_state=lambda *_args, **_kwargs: None,
        phase_observer=phases.append,
    )
    assert run_utility_aligned_case_aware_proxy_information_audit(
        config, artifact_root=root, dependencies=deps
    ) == root
    assert events == ["feature_lock_persisted", "test_labels_opened"]
    assert phases.index("case_aware_features") < phases.index("test_labels")
    assert not any("target" in phase for phase in phases)


def test_real_format_sixteen_hex_development_seal_is_accepted() -> None:
    partitions = SimpleNamespace(lock_hash="1" * 16)
    store = SimpleNamespace(role="development", partition_lock_hash="1" * 16)
    capability = SimpleNamespace(
        store=store,
        seal=SimpleNamespace(
            prediction_seal_hash="a1b2c3d4e5f60718",
            partition_lock_hash="1" * 16,
        ),
    )
    observed_store, observed_hash = _validated_development(capability, partitions)
    assert observed_store is store
    assert observed_hash == "a1b2c3d4e5f60718"


def test_descriptive_seed_rows_bind_uint8_evaluation_label_bytes() -> None:
    base = SimpleNamespace(
        positive_class_probabilities=np.asarray([0.2, 0.8]),
        training_seed=17,
        generation_seed=17,
        vector_hash="a" * 64,
    )
    tail = SimpleNamespace(
        positive_class_probabilities=np.asarray([0.3, 0.7]),
        training_seed=17,
        generation_seed=17,
        vector_hash="b" * 64,
    )
    labels = np.asarray([0, 1], dtype=np.int64)
    common = dict(
        row_key=("0", "1", "2"),
        base_vectors=(base,),
        tail_vectors=(tail,),
        evaluation_row_hash="c" * 64,
        feature_surface_hash="d" * 64,
        feature_lock_hash="e" * 64,
        prediction_seal_hash="f" * 16,
    )
    row = _descriptive_seed_rows(labels=labels, **common)[0]
    assert row["evaluation_label_sha256"] == array_sha256(
        np.ascontiguousarray(labels, dtype=np.uint8)
    )
    tampered = _descriptive_seed_rows(labels=labels[::-1], **common)[0]
    assert tampered["evaluation_label_sha256"] != row["evaluation_label_sha256"]
    assert tampered["row_hash"] != row["row_hash"]


def test_response_boundary_requires_distinct_persisted_feature_lock_hash() -> None:
    row = SimpleNamespace(feature_row_hash="1" * 64)
    surface = CaseAwareFeatureSurface(
        rows=(row,), row_keys=(), surface_hash="2" * 64
    )
    unhashed = {
        "schema_version": "midogpp_stage90_case_aware_feature_lock_v1",
        "status": "SEALED_BEFORE_TEST_LABEL_ACCESS",
        "feature_surface_hash": surface.surface_hash,
        "ordered_feature_row_hashes": [row.feature_row_hash],
        "feature_row_count": 1,
        "development_prediction_seal_hash": "3" * 16,
        "support_partition_lock_hash": "4" * 16,
        "test_labels_opened": False,
        "support_labels_used": False,
        "evaluation_probabilities_used_as_features": False,
    }
    feature_lock = {
        **unhashed,
        "case_aware_feature_lock_hash": canonical_sha256(unhashed),
    }
    observed = _validated_feature_lock(
        surface,
        feature_lock,
        expected_prediction_seal_hash="3" * 16,
        expected_partition_lock_hash="4" * 16,
    )
    assert observed == feature_lock["case_aware_feature_lock_hash"]
    assert observed != surface.surface_hash
    with pytest.raises(ProtocolError, match="feature lock drifted"):
        _validated_feature_lock(
            surface,
            {
                **feature_lock,
                "case_aware_feature_lock_hash": surface.surface_hash,
            },
            expected_prediction_seal_hash="3" * 16,
            expected_partition_lock_hash="4" * 16,
        )


def test_source_task_identity_preserves_legacy_two_case_omission_and_binds_v2_eight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = tuple(
        SimpleNamespace(
            source_center=center,
            training_seed=training_seed,
            generation_seed=generation_seed,
            stream_id=f"{center}-{training_seed}-{generation_seed}",
            expert_lock_hash="a" * 64,
        )
        for center, training_seed, generation_seed in product(
            CENTERS, TRAINING_SEEDS, GENERATION_SEEDS
        )
    )
    monkeypatch.setattr(
        source_cache_planning, "source_generation_plan", lambda _lock: keys
    )
    generation_lock = SimpleNamespace(generation_lock_hash="b" * 64)
    common = dict(
        checkpoint_root=tmp_path / "checkpoints",
        support_array_path=tmp_path / "support.npy",
        support_index_path=tmp_path / "support.json",
        support_scratch_hash="c" * 16,
    )
    legacy = SimpleNamespace(expert_bank_root=tmp_path, contract_hash="d" * 64)
    case_aware = SimpleNamespace(
        expert_bank_root=tmp_path,
        contract_hash="d" * 64,
        fixed_support_case_count_per_center=8,
    )
    legacy_tasks, _ = source_cache_planning.build_source_tasks(
        legacy, generation_lock, **common
    )
    v2_tasks, _ = source_cache_planning.build_source_tasks(
        case_aware, generation_lock, **common
    )
    assert all(
        "fixed_support_case_count_per_center" not in task for task in legacy_tasks
    )
    assert {
        task["fixed_support_case_count_per_center"] for task in v2_tasks
    } == {8}


def test_source_worker_rejects_scratch_task_support_count_drift() -> None:
    support = np.zeros((8, 3_840), dtype=np.float32)
    unhashed = {
        "shape": list(support.shape),
        "dtype": str(support.dtype),
        "array_sha256": sha256_array(support),
        "fixed_support_case_count_per_center": 8,
        "labels_consumed": False,
        "evaluation_embeddings_consumed": False,
    }
    index = {**unhashed, "support_scratch_hash": stable_hash(unhashed)}
    valid_task = {
        "support_scratch_hash": index["support_scratch_hash"],
        "fixed_support_case_count_per_center": 8,
    }
    source_cache_worker._validate_support_scratch(support, index, valid_task)
    with pytest.raises(ProtocolError, match="support scratch"):
        source_cache_worker._validate_support_scratch(
            support,
            index,
            {**valid_task, "fixed_support_case_count_per_center": 2},
        )
