from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only import (
    development_prediction_contracts as development_contracts,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.constants import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.development_actions import (
    DEVELOPMENT_ACTION_COUNT_PER_TASK,
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
    DEVELOPMENT_ORIENTED_CONTEXT_COUNT,
    DEVELOPMENT_PHYSICAL_TASK_COUNT,
    MASS_NORMALIZATION_BY_ACTION_KIND,
    TARGET_EFFECTIVE_MASS_BY_ACTION_KIND,
    development_action_library_payload,
    development_actions_for,
    development_candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.development_prediction_contracts import (
    DEVELOPMENT_PREDICTION_STATUS,
    DevelopmentSourcePredictionSeal,
    canonical_logical_cell_keys,
    canonical_physical_cell_keys,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.development_prediction_plans import (
    build_development_source_tasks,
    validate_development_source_task,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.development_prediction_runtime import (
    materialize_composite_prelabel_prediction_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


def test_strict_source_oof_topology_distinguishes_physical_and_logical_counts() -> None:
    assert DEVELOPMENT_PHYSICAL_TASK_COUNT == 324
    assert DEVELOPMENT_ORIENTED_CONTEXT_COUNT == 648
    assert DEVELOPMENT_ACTION_COUNT_PER_TASK == 16
    assert DEVELOPMENT_CLASSIFIER_FIT_COUNT == 5_184
    assert DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT == 10_368
    assert len(canonical_physical_cell_keys()) == 5_184
    assert len(canonical_logical_cell_keys()) == 10_368
    payload = development_action_library_payload()
    assert payload["unordered_excluded_pair_fit_reuse"] is True
    assert payload["physical_fit_count"] == 5_184
    assert payload["logical_prediction_cell_count"] == 10_368


def test_every_strict_action_excludes_h_and_q_and_matches_target_mass() -> None:
    for target in CENTERS:
        for query in CENTERS:
            if query == target:
                continue
            expected_sources = development_candidate_sources(target, query)
            assert len(expected_sources) == 7
            assert target not in expected_sources and query not in expected_sources
            for action in development_actions_for(target, query):
                assert tuple(action.counts_by_class[0]) == expected_sources
                assert tuple(action.counts_by_class[1]) == expected_sources
                assert tuple(action.sample_weight_by_source) == expected_sources
                kind = (
                    action.action_id
                    if action.action_id in ("B", "U")
                    else str(action.geometry_id)
                )
                assert (
                    action.logistic_mass_normalization
                    == MASS_NORMALIZATION_BY_ACTION_KIND[kind]
                )
                assert action.scaler_fit_used_sample_weight is False
                for label in (0, 1):
                    mass = sum(
                        action.counts_by_class[label][source]
                        * action.sample_weight_by_source[source]
                        for source in expected_sources
                    )
                    assert mass == pytest.approx(
                        TARGET_EFFECTIVE_MASS_BY_ACTION_KIND[kind], abs=1e-12
                    )


def test_unordered_pair_reuses_physical_hash_but_binds_oriented_h_q() -> None:
    forward = development_actions_for("0", "1")
    reverse = development_actions_for("1", "0")
    assert [row.action_id for row in forward] == [row.action_id for row in reverse]
    assert [row.action_hash for row in forward] == [row.action_hash for row in reverse]
    assert all(
        left.orientation_hash != right.orientation_hash
        for left, right in zip(forward, reverse, strict=True)
    )
    assert all(left.excluded_pair == ("0", "1") for left in forward)
    assert all(right.excluded_pair == ("0", "1") for right in reverse)


def test_strict_cell_keys_have_exact_canonical_order() -> None:
    physical = canonical_physical_cell_keys()
    logical = canonical_logical_cell_keys()
    assert physical[0] == ("0", "1", "B", TRAINING_SEEDS[0], GENERATION_SEEDS[0])
    assert physical[1] == ("0", "1", "U", TRAINING_SEEDS[0], GENERATION_SEEDS[0])
    assert physical[-1][:2] == ("8", "9")
    assert logical[0] == ("0", "1", "B", TRAINING_SEEDS[0], GENERATION_SEEDS[0])
    assert logical[16] == (
        "0",
        "1",
        "B",
        TRAINING_SEEDS[0],
        GENERATION_SEEDS[1],
    )
    assert logical[16 * 9] == (
        "0",
        "2",
        "B",
        TRAINING_SEEDS[0],
        GENERATION_SEEDS[0],
    )
    assert len(set(physical)) == len(physical)
    assert len(set(logical)) == len(logical)


def test_task_plan_has_324_pair_symmetric_fits_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    class _Record:
        def to_payload(self) -> dict[str, object]:
            return {"fixture": True}

    offsets = {
        center: {
            "start": ordinal * 2,
            "stop": ordinal * 2 + 2,
            "row_count": 2,
            "row_identity_hash": canonical_hash([center, "rows"]),
            "embedding_slice_sha256": canonical_hash([center, "embeddings"]),
        }
        for ordinal, center in enumerate(CENTERS)
    }
    scratch = {
        "array_path": tmp_path / "source.npy",
        "array_file_sha256": "a" * 64,
        "array_sha256": "b" * 64,
        "shape": [18, 3840],
        "dtype": "float32",
        "cache_binding_hash": "c" * 64,
        "offsets": offsets,
    }
    generated = SimpleNamespace(
        records=(_Record(),),
        lock_hash="d" * 64,
        source_array_path=tmp_path / "generated.npy",
        lock_payload={"source_array_sha256": "e" * 64},
    )
    config = SimpleNamespace(
        contract_hash="f" * 64,
        classifier={"fixture": True},
        runtime={"threads_per_worker": 3},
    )
    tasks = build_development_source_tasks(
        config,
        generated,
        scratch=scratch,
        action_library_hash="1" * 64,
        root=tmp_path,
    )
    assert len(tasks) == 324
    first = tasks[0]
    assert first["excluded_pair"] == ["0", "1"]
    assert first["candidate_sources"] == list(
        development_candidate_sources("0", "1")
    )
    assert first["physical_fit_count"] == 16
    assert first["logical_prediction_count"] == 32
    assert [
        (view["outer_target"], view["query_center"])
        for view in first["evaluation_views"]
    ] == [("0", "1"), ("1", "0")]
    validate_development_source_task(first)

    tampered = dict(first)
    tampered["candidate_sources"] = ["0", *first["candidate_sources"][:-1]]
    unhashed = {
        key: value
        for key, value in tampered.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    tampered["task_hash"] = canonical_hash(unhashed)
    with pytest.raises(ProtocolError):
        validate_development_source_task(tampered)

    orientation_tamper = dict(first)
    orientation_tamper["evaluation_views"] = [
        dict(view) for view in first["evaluation_views"]
    ]
    orientation_tamper["evaluation_views"][0]["orientation_hashes"] = [
        "0" * 64,
        *orientation_tamper["evaluation_views"][0]["orientation_hashes"][1:],
    ]
    unhashed = {
        key: value
        for key, value in orientation_tamper.items()
        if key not in {"task_hash", "checkpoint_json_path", "checkpoint_npz_path"}
    }
    orientation_tamper["task_hash"] = canonical_hash(unhashed)
    with pytest.raises(ProtocolError):
        validate_development_source_task(orientation_tamper)


def test_composite_prelabel_seal_binds_both_banks_and_exclusion_flags(
    tmp_path: Path,
) -> None:
    source_binding = "a" * 64
    strict_bank = SimpleNamespace(
        seal_hash="b" * 64,
        action_library_hash="c" * 64,
        source_cache_binding_hash=source_binding,
    )
    strict_store = SimpleNamespace(
        store_hash="d" * 64,
        frame_cache_binding_hash=source_binding,
        frame_role="source",
    )
    strict = SimpleNamespace(
        seal_hash="e" * 64,
        classifier_bank=strict_bank,
        source_store=strict_store,
    )
    target = SimpleNamespace(
        seal_hash="f" * 64,
        action_library_hash="1" * 64,
        source_cache_binding_hash=source_binding,
        seal_payload={
            "status": "SEALED_1458_SOURCE_ONLY_ACTION_CLASSIFIERS",
            "fit_count": 1_458,
            "test_cache_admitted": False,
        },
    )
    composite = materialize_composite_prelabel_prediction_seal(
        strict, target, root=tmp_path
    )
    assert composite.source_store is strict_store
    assert composite.target_classifier_bank is target
    assert composite.seal_payload[
        "query_excluded_from_every_source_composition"
    ] is True
    assert composite.seal_payload[
        "outer_target_excluded_from_every_source_composition"
    ] is True
    assert composite.seal_payload["unordered_excluded_pair_fit_reuse"] is True
    assert composite.seal_payload["total_physical_classifier_fit_count"] == 6_642

    target_drift = SimpleNamespace(**vars(target))
    target_drift.seal_hash = "2" * 64
    with pytest.raises(ProtocolError):
        materialize_composite_prelabel_prediction_seal(
            strict, target_drift, root=tmp_path
        )


def test_strict_source_seal_rejects_rehashed_config_drift(tmp_path: Path) -> None:
    arrays_path = tmp_path / "probabilities.npz"
    index_path = tmp_path / "prediction-index.json"
    seal_path = tmp_path / "prediction-seal.json"
    arrays_path.write_bytes(b"array-fixture")
    index_path.write_bytes(b"index-fixture")
    config_hash = "3" * 64
    binding_hash = "4" * 64
    bank = SimpleNamespace(
        action_library_hash="5" * 64,
        seal_hash="6" * 64,
        source_cache_binding_hash=binding_hash,
        config_contract_hash=config_hash,
    )
    store = SimpleNamespace(
        action_library_hash=bank.action_library_hash,
        development_classifier_bank_seal_hash=bank.seal_hash,
        frame_cache_binding_hash=binding_hash,
        store_hash="7" * 64,
    )
    unhashed = {
        "schema_version": "midogpp_strict_source_oof_prediction_seal_v1",
        "status": DEVELOPMENT_PREDICTION_STATUS,
        "config_contract_hash": config_hash,
        "classifier_bank_seal_hash": bank.seal_hash,
        "source_prediction_store_hash": store.store_hash,
        "source_prediction_array_sha256": sha256_file(arrays_path),
        "source_prediction_index_sha256": sha256_file(index_path),
        "physical_fit_count": 5_184,
        "logical_source_prediction_cell_count": 10_368,
        "source_labels_opened": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
    }
    payload = {
        **unhashed,
        "source_prediction_seal_hash": canonical_hash(unhashed),
    }
    valid = DevelopmentSourcePredictionSeal(
        classifier_bank=bank,
        source_store=store,
        seal_payload=payload,
        arrays_path=arrays_path,
        index_path=index_path,
        seal_path=seal_path,
    )
    assert valid.seal_hash == payload["source_prediction_seal_hash"]

    drifted_unhashed = {**unhashed, "config_contract_hash": "8" * 64}
    drifted = {
        **drifted_unhashed,
        "source_prediction_seal_hash": canonical_hash(drifted_unhashed),
    }
    with pytest.raises(ProtocolError):
        DevelopmentSourcePredictionSeal(
            classifier_bank=bank,
            source_store=store,
            seal_payload=drifted,
            arrays_path=arrays_path,
            index_path=index_path,
            seal_path=seal_path,
        )


def test_source_stream_stable_hash_is_exact_and_seal_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        development_contracts, "DEVELOPMENT_CLASSIFIER_FIT_COUNT", 0
    )
    monkeypatch.setattr(development_contracts, "FEATURE_DIM", 1)
    monkeypatch.setattr(
        development_contracts, "canonical_physical_cell_keys", lambda: ()
    )
    for member, values in (
        (development_contracts.DEVELOPMENT_CLASSIFIER_MEAN_MEMBER, np.empty((0, 1))),
        (development_contracts.DEVELOPMENT_CLASSIFIER_SCALE_MEMBER, np.empty((0, 1))),
        (
            development_contracts.DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER,
            np.empty((0, 1)),
        ),
        (development_contracts.DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER, np.empty((0,))),
    ):
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values.astype(np.float64), allow_pickle=False)

    source_stream_lock_hash = "1" * 16
    action_library_hash = "2" * 64
    source_cache_binding_hash = "3" * 64
    config_contract_hash = "4" * 64
    bank_hash = "5" * 64
    paths = tuple(
        tmp_path / member
        for member in (
            development_contracts.DEVELOPMENT_CLASSIFIER_MEAN_MEMBER,
            development_contracts.DEVELOPMENT_CLASSIFIER_SCALE_MEMBER,
            development_contracts.DEVELOPMENT_CLASSIFIER_COEFFICIENT_MEMBER,
            development_contracts.DEVELOPMENT_CLASSIFIER_INTERCEPT_MEMBER,
        )
    )
    seal_unhashed = {
        "schema_version": "midogpp_strict_source_oof_classifier_bank_seal_v1",
        "status": development_contracts.DEVELOPMENT_CLASSIFIER_STATUS,
        "config_contract_hash": config_contract_hash,
        "classifier_bank_hash": bank_hash,
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "source_cache_binding_hash": source_cache_binding_hash,
        "physical_fit_count": 0,
        "source_labels_available_during_fit": False,
        "test_cache_admitted": False,
        "scaler_mean_file_sha256": sha256_file(paths[0]),
        "scaler_scale_file_sha256": sha256_file(paths[1]),
        "coefficient_file_sha256": sha256_file(paths[2]),
        "intercept_file_sha256": sha256_file(paths[3]),
    }
    seal = {
        **seal_unhashed,
        "development_classifier_bank_seal_hash": canonical_hash(seal_unhashed),
    }
    bank_kwargs = {
        "root": tmp_path,
        "cells": (),
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": action_library_hash,
        "source_cache_binding_hash": source_cache_binding_hash,
        "config_contract_hash": config_contract_hash,
        "bank_hash": bank_hash,
        "seal_payload": seal,
    }
    observed = development_contracts.DevelopmentClassifierBank(**bank_kwargs)
    assert observed.source_stream_lock_hash == source_stream_lock_hash

    with pytest.raises(ProtocolError):
        development_contracts.DevelopmentClassifierBank(
            **{**bank_kwargs, "source_stream_lock_hash": "1" * 64}
        )

    drifted_unhashed = {
        **seal_unhashed,
        "source_stream_lock_hash": "6" * 16,
    }
    drifted_seal = {
        **drifted_unhashed,
        "development_classifier_bank_seal_hash": canonical_hash(drifted_unhashed),
    }
    with pytest.raises(ProtocolError):
        development_contracts.DevelopmentClassifierBank(
            **{**bank_kwargs, "seal_payload": drifted_seal}
        )
