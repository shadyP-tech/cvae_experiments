from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit.runner import (
    ProxyAuditRunnerDependencies,
    run_utility_aligned_ensemble_endpoint_proxy_information_audit,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=(root / "inputs/bank").resolve(),
        generation_lock_root=(root / "inputs/generation").resolve(),
        validation_cache_root=(root / "inputs/cache").resolve(),
        validation_manifest_path=(root / "inputs/manifest/manifest.csv").resolve(),
        metadata_profile_root=(root / "inputs/metadata").resolve(),
        runtime={"classifier_workers": 4},
        model={"ridge_alpha": 1.0},
        contract_hash="a" * 64,
    )


def _launch_files(root: Path) -> None:
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def test_runner_keeps_proxy_features_prelabel_and_never_builds_target_actions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "proxy_information_audit"
    _launch_files(root)
    config = _config(root)
    phases: list[str] = []
    writes: list[tuple[str, str]] = []
    partition = SimpleNamespace(lock_hash="b" * 64)
    cache = SimpleNamespace(source_records=(1,) * 81)
    development = SimpleNamespace(
        seal=SimpleNamespace(prediction_seal_hash="c" * 64),
        store=SimpleNamespace(cells=(1,) * 5184),
    )
    proxy_rows = ({"proxy_feature_row_hash": "d" * 64},) * 504
    proxy_lock = {"proxy_feature_lock_hash": "e" * 64}
    utility = SimpleNamespace(rows=(object(),) * 504)
    audit = SimpleNamespace(
        fold_lock={"crossfit_fold_lock_hash": "f" * 64},
        result_payload={"audit_result_hash": "1" * 64},
        crossfit_rows=({},),
        query_metric_rows=({},),
        outer_metric_rows=({},),
        family_summary_rows=({},),
    )

    def observe(phase: str) -> None:
        phases.append(phase)

    deps = ProxyAuditRunnerDependencies(
        validate_workspace=lambda _config: {},
        validate_provenance=lambda _root, _config: {},
        load_locks=lambda _config: SimpleNamespace(generation=object()),
        load_frame=lambda _config: object(),
        validate_firewall=lambda _config, _frame: {},
        build_partitions=lambda _frame, **_kwargs: partition,
        persist_initial=lambda *_args, **_kwargs: None,
        preflight=lambda *_args, **_kwargs: {"status": "PASS"},
        materialize_source=lambda *_args, **_kwargs: cache,
        validate_source=lambda *_args, **_kwargs: {
            "source_cache_lock_hash": "2" * 64
        },
        stage_source=lambda value, **_kwargs: value,
        load_metadata=lambda _config: {},
        produce_seed_features=lambda *_args, **_kwargs: object(),
        materialize_development=lambda *_args, **_kwargs: development,
        produce_proxy_features=lambda *_args, **_kwargs: proxy_rows,
        build_proxy_lock=lambda *_args, **_kwargs: proxy_lock,
        persist_prelabel=lambda *_args, **_kwargs: writes.append(
            ("persist", "prelabel")
        ),
        open_development_labels=lambda *_args, **_kwargs: writes.append(
            ("open", "development_labels")
        )
        or object(),
        score_development=lambda *_args, **_kwargs: (utility, (object(),) * 4536),
        run_audit=lambda *_args, **_kwargs: audit,
        persist_postseal=lambda *_args, **_kwargs: None,
        write_index=lambda *_args, **_kwargs: {},
        validate_bundle=lambda *_args, **_kwargs: {"status": "PASS"},
        persist_validation=lambda *_args, **_kwargs: None,
        write_state=lambda *_args, **_kwargs: None,
        phase_observer=observe,
    )
    assert (
        run_utility_aligned_ensemble_endpoint_proxy_information_audit(
            config, artifact_root=root, dependencies=deps
        )
        == root
    )
    assert phases == [
        "workspace",
        "preflight",
        "source_cache",
        "development_predictions",
        "proxy_features",
        "development_labels",
        "proxy_information_audit",
    ]
    assert writes == [("persist", "prelabel"), ("open", "development_labels")]
    assert all("target" not in phase for phase in phases)


def test_complete_fast_path_is_closed_world_and_incomplete_complete_fails(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete_proxy_information_audit"
    for member in REQUIRED_FILES:
        path = complete / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (complete / "reports/run_state.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    calls: list[str] = []
    result = run_utility_aligned_ensemble_endpoint_proxy_information_audit(
        _config(complete),
        artifact_root=complete,
        dependencies=ProxyAuditRunnerDependencies(
            validate_bundle=lambda *_args, **_kwargs: calls.append("validated") or {},
        ),
    )
    assert result == complete
    assert calls == ["validated"]

    incomplete = tmp_path / "incomplete_proxy_information_audit"
    _launch_files(incomplete)
    (incomplete / "reports").mkdir(exist_ok=True)
    (incomplete / "reports/run_state.json").write_text(
        '{"status":"COMPLETE"}\n', encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="closed-world inventory drifted"):
        run_utility_aligned_ensemble_endpoint_proxy_information_audit(
            _config(incomplete), artifact_root=incomplete
        )


def test_content_index_detects_tamper_and_final_closed_world_rejects_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(member + "\n", encoding="utf-8")
    payload = write_content_index(root, config_contract_hash="a" * 64)
    assert payload["member_count"] == len(CONTENT_INDEX_MEMBERS)
    validate_content_index(root, config_contract_hash="a" * 64)
    (root / CONTENT_INDEX_MEMBERS[-1]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="member drifted"):
        validate_content_index(root, config_contract_hash="a" * 64)

    for member in REQUIRED_FILES:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
    checkpoint = root / "checkpoints/orphan.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="checkpoints/orphan.json"):
        assert_closed_world(root, allow_incomplete=False)


def test_reconstructive_validator_checks_content_bytes_before_any_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from midogpp_thesis.cvae.diagnostics.utility_aligned_ensemble_endpoint_proxy_information_audit import (
        validation,
    )

    root = tmp_path / "tampered_proxy_information_audit"
    root.mkdir()
    config = SimpleNamespace(
        contract_hash="a" * 64,
        artifact_root=root.resolve(),
        input_artifact_ids=("input",),
    )
    reached_reconstruction: list[bool] = []
    monkeypatch.setattr(validation, "assert_closed_world", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        validation,
        "load_utility_aligned_ensemble_endpoint_proxy_information_audit_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        validation,
        "validate_content_index",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ProtocolError("content byte tamper")
        ),
    )
    monkeypatch.setattr(
        validation,
        "validate_active_diagnostic_workspace_binding",
        lambda _config: reached_reconstruction.append(True),
    )
    with pytest.raises(ProtocolError, match="content byte tamper"):
        validation.validate_proxy_information_audit_bundle(root, config=config)
    assert reached_reconstruction == []
