from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.bundle import (
    REQUIRED_FILES, assert_closed_world, validate_content_index, write_content_index,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_router.runner import (
    EnsembleEndpointRunnerDependencies,
    run_utility_aligned_ensemble_endpoint_router_diagnostic,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _write_inventory(root: Path) -> None:
    for member in REQUIRED_FILES:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if member.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_bytes(b"placeholder\n")


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_root=root.resolve(), expert_bank_root=(root / "bank").resolve(),
        generation_lock_root=(root / "generation").resolve(),
        validation_cache_root=(root / "cache").resolve(),
        validation_manifest_path=(root / "manifest.csv").resolve(),
        metadata_profile_root=(root / "metadata").resolve(), contract_hash="contract",
        runtime={}, input_artifact_ids=(),
    )


def test_content_index_detects_member_tamper(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    write_content_index(tmp_path, config_contract_hash="contract")
    validate_content_index(tmp_path, config_contract_hash="contract")
    member = tmp_path / "tables/target_ensemble_metrics.csv"
    member.write_bytes(member.read_bytes() + b"tamper")
    with pytest.raises(ProtocolError, match="content-index member"):
        validate_content_index(tmp_path, config_contract_hash="contract")


def test_closed_world_rejects_extra_member(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    (tmp_path / "reports/extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_complete_fast_path_performs_only_reconstructive_validation(tmp_path: Path) -> None:
    _write_inventory(tmp_path)
    (tmp_path / "reports/run_state.json").write_text(
        json.dumps({"status": "COMPLETE", "phase": "COMPLETE"}) + "\n", encoding="utf-8"
    )
    calls = []
    deps = EnsembleEndpointRunnerDependencies(
        validate_bundle=lambda root, **kwargs: calls.append("validate") or {"status": "PASS"}
    )
    assert run_utility_aligned_ensemble_endpoint_router_diagnostic(
        _config(tmp_path), artifact_root=tmp_path, dependencies=deps
    ) == tmp_path
    assert calls == ["validate"]


def test_incomplete_complete_marker_fails_before_validator(tmp_path: Path) -> None:
    (tmp_path / "provenance").mkdir(parents=True)
    (tmp_path / "config.resolved.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/run_state.json").write_text(
        json.dumps({"status": "COMPLETE", "phase": "COMPLETE"}) + "\n", encoding="utf-8"
    )
    calls = []
    deps = EnsembleEndpointRunnerDependencies(
        validate_bundle=lambda root, **kwargs: calls.append("validate") or {"status": "PASS"}
    )
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        run_utility_aligned_ensemble_endpoint_router_diagnostic(
            _config(tmp_path), artifact_root=tmp_path, dependencies=deps
        )
    assert calls == []


def test_runner_dependency_order_seals_before_each_label_gate(tmp_path: Path) -> None:
    (tmp_path / "provenance").mkdir(parents=True)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "config.resolved.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifests/ensemble_endpoint_target_probe_seal.json").write_text(
        json.dumps({"probe_seal_hash": "p" * 64}) + "\n", encoding="utf-8"
    )
    order = []
    noop = lambda *args, **kwargs: None
    generation = SimpleNamespace(generation_lock_hash="generation")
    locks = SimpleNamespace(generation=generation)
    frame = SimpleNamespace(cache_binding_hash="cache")
    partitions = SimpleNamespace(lock_hash="partitions")
    folds = SimpleNamespace(lock_hash="folds")
    cache = SimpleNamespace(source_records=(), component_records=())
    seed_features = SimpleNamespace(inner_rows=(), target_rows=(), production_hash="seed")
    development = SimpleNamespace(
        seal=SimpleNamespace(prediction_seal_hash="development-seal"),
        store=SimpleNamespace(cells=()),
    )
    inner_shifts = SimpleNamespace(by_candidate={}, lock_hash="inner-shifts")
    probe = SimpleNamespace(cells=())
    target_shifts = SimpleNamespace(by_candidate={}, lock_hash="target-shifts")
    features = SimpleNamespace(surface_hash="features")
    labels = object()
    utility = SimpleNamespace(rows=(), surface_hash="utility")
    models = SimpleNamespace(model_set_hash="models")
    plans = SimpleNamespace(plan_set_hash="plans")
    actions = SimpleNamespace(action_library_hash="actions", action_count=117)
    predictions = SimpleNamespace(cells=(), unique_classifier_fit_count=810, store_hash="store")
    target_capability = SimpleNamespace(
        payload={
            "action_library_hash": "actions", "seal_hash": "target-seal",
            "target_probe_seal_hash": "p" * 64,
        },
        seal_hash="target-seal",
    )
    scores = SimpleNamespace(rows=(), score_set_hash="scores")
    deps = EnsembleEndpointRunnerDependencies(
        validate_workspace=lambda config: {"status": "PASS"},
        validate_provenance=lambda root, config: {}, load_locks=lambda config: locks,
        load_frame=lambda config: frame,
        validate_firewall=lambda config, frame: {"status": "PASS"},
        build_partitions=lambda frame, **kwargs: partitions,
        build_case_folds=lambda partitions, **kwargs: folds,
        run_preflight=lambda root, **kwargs: {"status": "PASS"},
        materialize_source=lambda *args, **kwargs: cache,
        validate_source_lock=lambda *args, **kwargs: {"source_cache_lock_hash": "source-lock"},
        stage_source=lambda cache, **kwargs: cache, load_metadata=lambda config: {},
        produce_seed_features=lambda *args: seed_features,
        materialize_development=lambda *args, **kwargs: development,
        build_inner_shifts=lambda *args: inner_shifts,
        materialize_target_probe=lambda *args, **kwargs: probe,
        build_target_shifts=lambda *args: target_shifts,
        build_features=lambda *args, **kwargs: features,
        open_development_labels=lambda *args, **kwargs: labels,
        score_development=lambda *args: (utility, ()), fit_models=lambda *args: models,
        build_plans=lambda *args: plans, build_actions=lambda *args: actions,
        materialize_target=lambda *args, **kwargs: predictions,
        build_target_seal=lambda *args, **kwargs: target_capability,
        open_target_labels=lambda *args, **kwargs: ({}, {"status": "PASS"}),
        score_terminal=lambda *args: (scores, (), ()), build_contrasts=lambda *args: (),
        infer_contrasts=lambda *args: (), validate_bundle=lambda *args, **kwargs: {"status": "PASS"},
        persist_initial=noop, persist_features=noop, persist_development_router=noop,
        persist_target_seal=noop, persist_terminal=noop, write_index=lambda *args, **kwargs: {},
        persist_validation=noop, write_state=noop, phase_observer=order.append,
    )
    run_utility_aligned_ensemble_endpoint_router_diagnostic(
        _config(tmp_path), artifact_root=tmp_path, dependencies=deps
    )
    assert order == [
        "workspace", "provenance", "firewall", "partitions", "preflight",
        "source_cache", "seed_features", "development_predictions",
        "source_inner_shifts", "target_probe", "target_shifts", "features",
        "development_labels", "development_scoring", "models", "plans", "actions",
        "target_predictions", "target_seal", "target_labels", "terminal_scoring",
        "validation",
    ]
