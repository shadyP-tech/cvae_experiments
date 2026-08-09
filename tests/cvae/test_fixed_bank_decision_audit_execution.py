"""Execution-boundary tests for the terminal fixed-bank decision audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit import (
    execution_adapter,
    source_cache,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.artifact_io import (
    atomic_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.bundle import (
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.experiment_contracts import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    SUPPORT_PARTITION_NAMESPACE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity as FixedTestRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.inputs import (
    assert_input_fence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.ledger import (
    expected_amendment_payload,
    load_validated_ledger_chain,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.partitions import (
    build_fixed_test_partition_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit.runner import (
    FixedBankDecisionAuditRunnerDependencies,
    run_fixed_bank_decision_audit,
)
from midogpp_thesis.cvae.protocol import ProtocolError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AMENDMENT_CONTRACT = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts"
    / "uniform_b_v2_consumed_test_fixed_bank_decision_audit_ledger_amendment_v1.json"
)


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=(root / "inputs/bank").resolve(),
        generation_lock_root=(root / "inputs/generation").resolve(),
        test_cache_root=(root / "inputs/cache").resolve(),
        test_manifest_path=(root / "inputs/manifest/manifest.csv").resolve(),
        test_consumption_ledger_path=(
            root / "inputs/ledger/reports/test_consumption_ledger.json"
        ).resolve(),
        ledger_amendment_path=(root / "inputs/amendment/amendment.json").resolve(),
        metadata_profile_root=(root / "inputs/metadata").resolve(),
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        runtime=canonical_runtime_payload(),
        protocol={
            "support_split_seed": 20_260_809,
            "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        },
        contract_hash="a" * 64,
        fixed_support_case_count_per_center=8,
    )


def _launch_files(root: Path) -> None:
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def _write_inventory(root: Path) -> None:
    for member in REQUIRED_FILES:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder\n")
    (root / "reports/run_state.json").write_text(
        json.dumps({"status": "COMPLETE", "phase": "COMPLETE"}) + "\n",
        encoding="utf-8",
    )


def test_runner_persists_and_rereads_feature_lock_before_test_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fixed_bank_decision_audit"
    _launch_files(root)
    config = _config(root)
    phases: list[str] = []
    events: list[str] = []
    generation_lock = object()
    locks = SimpleNamespace(generation=generation_lock)
    frame = object()
    partitions = SimpleNamespace(lock_hash="b" * 16)
    cache = SimpleNamespace(source_records=(object(),) * 81)
    development = SimpleNamespace(
        seal=SimpleNamespace(prediction_seal_hash="c" * 16),
        store=SimpleNamespace(cells=(object(),) * 5_184),
    )
    features = SimpleNamespace(rows=(object(),) * 504)
    feature_lock = {"fixed_bank_feature_lock_hash": "d" * 64}
    responses = SimpleNamespace(
        rows=(object(),) * 504,
        descriptive_seed_rows=({},) * 4_536,
    )
    exact = SimpleNamespace(
        predictions=(object(),) * 4_536,
        fold_audits=(object(),) * 648,
        result_hash="e" * 64,
    )
    smooth = SimpleNamespace(
        predictions=(object(),) * 1_512,
        fold_audits=(object(),) * 216,
        result_hash="f" * 64,
    )
    audit = SimpleNamespace(
        exact_crossfit=exact,
        smooth_descriptive_crossfit=smooth,
        query_metrics=(object(),) * 648,
        outer_metrics=(object(),) * 81,
        family_summaries=(object(),) * 9,
        abstention_decisions=(object(),) * 648,
        abstention_summaries=(object(),) * 9,
    )

    def firewall(
        observed_config: object, observed_frame: object, observed_locks: object
    ) -> dict[str, object]:
        assert (observed_config, observed_frame, observed_locks) == (
            config,
            frame,
            locks,
        )
        return {"status": "PASS"}

    def persist_prelabel(observed_root: Path, **_kwargs: object) -> None:
        events.append("feature_lock_persisted")
        atomic_json(
            observed_root / "manifests/fixed_bank_feature_lock.json",
            feature_lock,
        )

    def open_labels(*_args: object, **_kwargs: object) -> object:
        assert events == ["feature_lock_persisted"]
        events.append("test_labels_opened")
        return object()

    deps = FixedBankDecisionAuditRunnerDependencies(
        validate_inputs=lambda _config: None,
        validate_workspace=lambda _config: {},
        validate_provenance=lambda _root, _config: {},
        load_locks=lambda _config: locks,
        load_frame=lambda _config: frame,
        validate_firewall=firewall,
        build_partitions=lambda *_args, **_kwargs: partitions,
        persist_initial=lambda *_args, **_kwargs: None,
        preflight=lambda *_args, **_kwargs: {"status": "PASS"},
        materialize_source=lambda observed_config, *args, **kwargs: (
            cache
            if observed_config is config and args[0] is generation_lock
            else pytest.fail("runner did not propagate source inputs")
        ),
        validate_source=lambda *_args, **_kwargs: {
            "source_cache_lock_hash": "1" * 16
        },
        stage_source=lambda value, **_kwargs: value,
        materialize_development=lambda observed_config, *_args, **_kwargs: (
            development
            if observed_config is config
            else pytest.fail("runner did not propagate prediction config")
        ),
        validate_development_seal=lambda _capability: {},
        load_metadata=lambda _config: {},
        produce_features=lambda *_args, **_kwargs: features,
        build_feature_lock=lambda *_args, **_kwargs: feature_lock,
        persist_prelabel=persist_prelabel,
        open_development_labels=open_labels,
        produce_responses=lambda *_args, **_kwargs: responses,
        build_response_lock=lambda *_args, **_kwargs: {
            "fixed_bank_response_lock_hash": "2" * 64
        },
        build_dataset=lambda *_args, **_kwargs: object(),
        run_core=lambda *_args, **_kwargs: audit,
        build_exact_lock=lambda *_args, **_kwargs: {
            "exact_crossfit_lock_hash": "3" * 64
        },
        build_smooth_lock=lambda *_args, **_kwargs: {
            "smooth_descriptive_crossfit_lock_hash": "4" * 64
        },
        persist_postseal=lambda *_args, **_kwargs: None,
        write_index=lambda *_args, **_kwargs: {},
        validate_bundle=lambda *_args, **_kwargs: {"status": "PASS"},
        persist_validation=lambda *_args, **_kwargs: None,
        write_state=lambda *_args, **_kwargs: None,
        phase_observer=phases.append,
    )
    assert run_fixed_bank_decision_audit(
        config, artifact_root=root, dependencies=deps
    ) == root
    assert events == ["feature_lock_persisted", "test_labels_opened"]
    assert phases.index("fixed_bank_features") < phases.index("test_labels")
    assert not any("target" in phase or "action" in phase for phase in phases)


def test_runner_fails_closed_on_internal_firewall_typeerror(tmp_path: Path) -> None:
    root = tmp_path / "fixed_bank_decision_audit_firewall"
    _launch_files(root)
    calls: list[tuple[object, object, object]] = []
    frame = object()
    locks = SimpleNamespace(generation=object())

    def broken_firewall(*args: object) -> object:
        calls.append(args)
        raise TypeError("internal firewall defect")

    deps = FixedBankDecisionAuditRunnerDependencies(
        validate_inputs=lambda _config: None,
        validate_workspace=lambda _config: {},
        validate_provenance=lambda _root, _config: {},
        load_locks=lambda _config: locks,
        load_frame=lambda _config: frame,
        validate_firewall=broken_firewall,
        write_state=lambda *_args, **_kwargs: None,
    )
    config = _config(root)
    with pytest.raises(TypeError, match="internal firewall defect"):
        run_fixed_bank_decision_audit(config, artifact_root=root, dependencies=deps)
    assert calls == [(config, frame, locks)]


def test_complete_fast_path_only_runs_reconstructive_validation(tmp_path: Path) -> None:
    root = tmp_path / "fixed_bank_decision_audit_complete"
    _write_inventory(root)
    calls: list[tuple[Path, dict[str, object]]] = []
    deps = FixedBankDecisionAuditRunnerDependencies(
        validate_inputs=lambda _config: pytest.fail("complete run refit inputs"),
        validate_bundle=lambda observed_root, **kwargs: (
            calls.append((observed_root, kwargs)) or {"status": "PASS"}
        ),
    )
    config = _config(root)
    assert run_fixed_bank_decision_audit(
        config, artifact_root=root, dependencies=deps
    ) == root
    assert calls == [(root, {"config": config})]


def test_runtime_topology_reaches_low_level_preflight_and_worker_adapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = canonical_runtime_payload()
    preflight_calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        execution_adapter,
        "_run_workstation_preflight",
        lambda root, *, runtime: (
            preflight_calls.append((root, runtime)) or {"status": "PASS"}
        ),
    )
    assert execution_adapter.run_workstation_preflight(
        tmp_path, runtime=runtime
    ) == {"status": "PASS"}
    assert preflight_calls == [(tmp_path, runtime)]

    source_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()
    monkeypatch.setattr(
        source_cache,
        "_materialize_source_cache",
        lambda *args, **kwargs: (
            source_calls.append((args, kwargs)) or sentinel
        ),
    )
    config = SimpleNamespace(runtime=runtime)
    root = tmp_path / "fixed_bank_decision_audit_v1"
    assert source_cache.materialize_source_cache(
        config, object(), object(), object(), root=root
    ) is sentinel
    assert source_calls[0][0][0] is config
    assert source_calls[0][0][0].runtime["generation_workers_per_device"] == 1
    assert source_calls[0][0][0].runtime["persistent_source_workers"] is True

    stage_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        source_cache,
        "_stage_source_cache_for_cpu",
        lambda value, **kwargs: stage_calls.append(kwargs) or value,
    )
    assert source_cache.stage_source_cache_for_cpu(
        sentinel,
        scratch_root=Path("/data/local/fixed_bank_decision_audit_v1"),
        canonical_root=root,
    ) is sentinel
    assert stage_calls == [
        {
            "scratch_root": Path("/data/local/fixed_bank_decision_audit_v1"),
            "canonical_root": root,
            "local_stage_directory": "source_cache",
        }
    ]

    drifted = dict(runtime)
    drifted.pop("generation_workers_per_device")
    with pytest.raises(ProtocolError, match="topology drifted"):
        execution_adapter.run_workstation_preflight(tmp_path, runtime=drifted)
    assert len(preflight_calls) == 1


def test_fixed_partition_is_deterministic_eight_case_72_146_surface() -> None:
    rows: list[FixedTestRowIdentity] = []
    rows_by_center: dict[str, tuple[FixedTestRowIdentity, ...]] = {}
    for center in CENTERS:
        center_rows: list[FixedTestRowIdentity] = []
        for case_index in range(EXPECTED_CASE_COUNTS_BY_CENTER[center]):
            ordinal = len(rows)
            row = FixedTestRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=ordinal,
                evaluation_row_id=f"row-{center}-{case_index}",
                case_id=f"case-{center}-{case_index}",
                center=center,
            )
            rows.append(row)
            center_rows.append(row)
        rows_by_center[center] = tuple(center_rows)
    frame = LabelFreeTestFrame(
        embeddings=np.zeros((len(rows), 3_840), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding={"split": "test", "labels_persisted": False},
    )
    first = build_fixed_test_partition_surface(
        frame, config_contract_hash="a" * 64
    )
    second = build_fixed_test_partition_surface(
        frame, config_contract_hash="a" * 64
    )
    assert first.lock_hash == second.lock_hash
    assert first.lock_payload["support_partition_namespace"] == (
        SUPPORT_PARTITION_NAMESPACE
    )
    assert first.lock_payload["support_case_count_total"] == 72
    assert first.lock_payload["evaluation_case_count_total"] == 146
    assert {
        len({row.case_id for row in first.support_rows_by_center[center]})
        for center in CENTERS
    } == {8}
    assert all(
        {row.case_id for row in first.support_rows_by_center[center]}.isdisjoint(
            row.case_id for row in first.evaluation_rows_by_center[center]
        )
        for center in CENTERS
    )


def test_input_fence_and_ledger_amendment_are_exactly_experiment_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "fixed_bank_decision_audit_inputs")
    assert_input_fence(config)
    forbidden = SimpleNamespace(
        **{
            **vars(config),
            "metadata_profile_root": (
                tmp_path / "utility_aligned_case_aware_proxy_information_audit"
            ).resolve(),
        }
    )
    with pytest.raises(ProtocolError, match="prior Stage-90"):
        assert_input_fence(forbidden)

    contract_bytes = AMENDMENT_CONTRACT.read_bytes()
    assert hashlib.sha256(contract_bytes).hexdigest() == (
        EXPECTED_LEDGER_AMENDMENT_SHA256
    )
    assert json.loads(contract_bytes) == expected_amendment_payload()
    assert expected_amendment_payload()["authorized_consumer_experiment_ids"] == [
        EXPERIMENT_ID
    ]

    parent = tmp_path / "parent.json"
    amendment = tmp_path / "amendment.json"
    parent.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_uniform_b_test_consumption_ledger_v1",
                "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
                "split": "test",
                "row_count": 9_928,
                "observed_centers": 9,
                "may_be_reused_as_fresh_representation_selection_evidence": False,
                "may_be_reused_for_descriptive_locked-model_scoring": True,
            }
        ),
        encoding="utf-8",
    )
    amendment.write_text(
        json.dumps(expected_amendment_payload()), encoding="utf-8"
    )
    ledger_config = SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        test_consumption_ledger_path=parent,
        ledger_amendment_path=amendment,
    )

    from midogpp_thesis.cvae.diagnostics.fixed_bank_decision_audit import ledger

    monkeypatch.setattr(
        ledger,
        "sha256_file",
        lambda path: (
            EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
            if path == parent
            else EXPECTED_LEDGER_AMENDMENT_SHA256
        ),
    )
    chain = load_validated_ledger_chain(ledger_config)
    assert chain.amendment["parent_sha256"] == (
        EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
    )
    tampered = {
        **expected_amendment_payload(),
        "authorized_consumer_experiment_ids": ["another.experiment"],
    }
    amendment.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ProtocolError, match="chain or whitelist drifted"):
        load_validated_ledger_chain(ledger_config)
