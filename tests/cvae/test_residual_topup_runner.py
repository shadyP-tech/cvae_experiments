from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.residual_topup_router import runner
from midogpp_thesis.cvae.diagnostics.residual_topup_router.calibration import (
    QUERY_GAIN_COLUMNS,
    SELECTION_COLUMNS,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.contracts import CENTERS
from midogpp_thesis.cvae.diagnostics.residual_topup_router.partitions import (
    SUPPORT_PARTITION_COLUMNS,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.scoring import (
    DEVELOPMENT_GAIN_COLUMNS,
    ENSEMBLE_METRIC_COLUMNS,
    METRIC_COLUMNS,
    TARGET_DELTA_COLUMNS,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.target_plans import (
    ASSIGNMENT_COLUMNS,
    PLAN_COLUMNS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _row(columns: tuple[str, ...]) -> dict[str, object]:
    return {column: 0 for column in columns}


def _config(root: Path) -> SimpleNamespace:
    classifier = SimpleNamespace(to_payload=lambda: {"family": "test"})
    return SimpleNamespace(
        artifact_root=root,
        expert_bank_root=root / "bank",
        generation_lock_root=root / "generation",
        equal_union_policy_root=root / "policy",
        validation_cache_root=root / "cache",
        validation_manifest_path=root / "manifest.csv",
        input_artifact_ids=("input-a",),
        experiment_id="residual-topup-test",
        contract_hash="config-hash",
        runtime={},
        protocol={},
        actions={},
        selection={},
        classifier=classifier,
        claim_boundary={},
    )


def _install_lightweight_phases(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> tuple[SimpleNamespace, list[str]]:
    events: list[str] = []
    config = _config(root)
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("test: true\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")

    frame = SimpleNamespace(cache_binding_hash="cache-binding")
    partitions = SimpleNamespace(
        table_rows=(_row(SUPPORT_PARTITION_COLUMNS),),
        lock_payload={"support_partition_lock_hash": "partition-lock"},
        lock_hash="partition-lock",
    )
    generation = SimpleNamespace(generation_lock_hash="generation-lock")
    source_cache = SimpleNamespace(
        compatibility_case_rows=({}, {}),
        index_rows=(),
    )
    plans = SimpleNamespace(
        table_rows=(_row(PLAN_COLUMNS),),
        assignment_rows=(_row(ASSIGNMENT_COLUMNS),),
        lock_payload={"router_plan_lock_hash": "plan-lock"},
        lock_hash="plan-lock",
    )
    predictions = SimpleNamespace(
        unique_classifier_fit_count=1539,
        index_rows=(),
    )

    monkeypatch.setattr(
        runner,
        "validate_workspace_provenance",
        lambda *_args, **_kwargs: {"input-a": {"artifact_id": "input-a"}},
    )
    monkeypatch.setattr(
        runner,
        "load_validated_locks",
        lambda *_args, **_kwargs: SimpleNamespace(generation=generation),
    )
    monkeypatch.setattr(
        runner,
        "run_workstation_preflight",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "load_label_free_validation_frame",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        runner,
        "build_partition_surface",
        lambda *_args, **_kwargs: partitions,
    )

    def materialize_sources(*_args: object, **_kwargs: object) -> object:
        events.append("source_cache")
        return source_cache

    monkeypatch.setattr(runner, "materialize_source_cache", materialize_sources)
    monkeypatch.setattr(
        runner,
        "validate_source_cache_lock",
        lambda *_args, **_kwargs: {"source_cache_lock_hash": "source-lock"},
    )
    monkeypatch.setattr(
        runner,
        "build_plan_surface",
        lambda *_args, **_kwargs: plans,
    )

    def materialize_predictions(*_args: object, **_kwargs: object) -> object:
        events.append("predictions")
        return predictions

    monkeypatch.setattr(
        runner,
        "materialize_all_action_predictions",
        materialize_predictions,
    )
    monkeypatch.setattr(
        runner,
        "validate_prediction_store_binding",
        lambda *_args, **_kwargs: events.append("prediction_binding"),
    )

    def build_seal(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("seal")
        seal_path = root / "manifests/global_all_action_prediction_seal.json"
        seal_path.write_text('{"seal_hash":"global-seal"}\n', encoding="utf-8")
        return {"seal_hash": "global-seal"}

    monkeypatch.setattr(runner, "build_global_prediction_seal", build_seal)
    monkeypatch.setattr(
        runner,
        "validate_global_prediction_seal",
        lambda *_args, **_kwargs: (
            events.append("seal_validation") or {"seal_hash": "global-seal"}
        ),
    )

    def open_labels(*_args: object, **_kwargs: object):
        assert events[-1] == "seal_validation"
        assert "seal" in events
        events.append("labels")
        return {"sample": 0}, {"status": "OPENED_AFTER_SEAL"}

    monkeypatch.setattr(
        runner,
        "open_evaluation_labels_after_global_seal",
        open_labels,
    )
    monkeypatch.setattr(
        runner,
        "score_prediction_store",
        lambda *_args, **_kwargs: (_row(METRIC_COLUMNS),),
    )
    monkeypatch.setattr(
        runner,
        "development_paired_gains",
        lambda *_args, **_kwargs: (_row(DEVELOPMENT_GAIN_COLUMNS),),
    )

    selections = []
    for center in CENTERS:
        row = _row(SELECTION_COLUMNS)
        row["outer_target"] = center
        row["selected_action_id"] = "uniform_topup"
        selections.append(row)
    monkeypatch.setattr(
        runner,
        "calibrate_outer_actions",
        lambda *_args, **_kwargs: (
            (_row(QUERY_GAIN_COLUMNS),),
            tuple(selections),
            {"calibration_lock_hash": "calibration-lock"},
        ),
    )
    monkeypatch.setattr(
        runner,
        "target_paired_deltas",
        lambda *_args, **_kwargs: (_row(TARGET_DELTA_COLUMNS),),
    )
    monkeypatch.setattr(
        runner,
        "target_probability_ensemble_metrics",
        lambda *_args, **_kwargs: (_row(ENSEMBLE_METRIC_COLUMNS),),
    )
    monkeypatch.setattr(
        runner,
        "scoring_summary_payload",
        lambda *_args, **_kwargs: {"diagnostic_only": True},
    )

    def write_index(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("content_index")
        return {"content_hash": "content"}

    monkeypatch.setattr(runner, "write_content_index", write_index)
    monkeypatch.setattr(
        runner,
        "_validate_bundle",
        lambda *_args, **_kwargs: events.append("validation") or {"status": "PASS"},
    )
    return config, events


def test_runner_opens_labels_only_after_all_action_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, events = _install_lightweight_phases(monkeypatch, tmp_path)

    result = runner.run_residual_topup_router_diagnostic(config)

    assert result == tmp_path
    assert events.index("predictions") < events.index("seal") < events.index("labels")
    assert events.count("validation") == 2
    state = json.loads(
        (tmp_path / "reports/run_state.json").read_text(encoding="utf-8")
    )
    assert state == {
        "error": None,
        "phase": "COMPLETE",
        "resumable": False,
        "schema_version": "midogpp_residual_topup_run_state_v1",
        "status": "COMPLETE",
    }


def test_runner_does_not_open_labels_when_global_seal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, events = _install_lightweight_phases(monkeypatch, tmp_path)

    def reject_seal(*_args: object, **_kwargs: object) -> object:
        events.append("seal_failed")
        raise ProtocolError("incomplete global prediction surface")

    monkeypatch.setattr(runner, "build_global_prediction_seal", reject_seal)

    with pytest.raises(ProtocolError, match="incomplete global prediction surface"):
        runner.run_residual_topup_router_diagnostic(config)

    assert "labels" not in events
    state = json.loads(
        (tmp_path / "reports/run_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "FAILED"
    assert state["phase"] == "ALL_ACTION_PREDICTIONS"
    assert state["resumable"] is True
    assert state["error"].startswith("ProtocolError:")


def test_runner_rejects_an_unresolved_override_before_creating_output(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(ProtocolError, match="workspace-resolved paths"):
        runner.run_residual_topup_router_diagnostic(
            config,
            artifact_root=Path("relative-output"),
        )

    assert not Path("relative-output").exists()


def test_post_seal_resume_reuses_seal_and_persisted_label_access_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, events = _install_lightweight_phases(monkeypatch, tmp_path)
    successful_score = runner.score_prediction_store
    attempts = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("interrupted after label access")
        return successful_score(*args, **kwargs)

    monkeypatch.setattr(runner, "score_prediction_store", fail_once)
    with pytest.raises(RuntimeError, match="interrupted after label access"):
        runner.run_residual_topup_router_diagnostic(config)

    assert (tmp_path / "manifests/global_all_action_prediction_seal.json").is_file()
    assert (tmp_path / "reports/label_access_report.json").is_file()
    assert events.count("seal") == 1
    assert events.count("labels") == 1

    runner.run_residual_topup_router_diagnostic(config)

    assert events.count("seal") == 1
    assert events.count("seal_validation") >= 2
    assert events.count("labels") == 2
    assert json.loads(
        (tmp_path / "reports/run_state.json").read_text(encoding="utf-8")
    )["status"] == "COMPLETE"


def test_post_seal_resume_rejects_drifted_label_access_report_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _events = _install_lightweight_phases(monkeypatch, tmp_path)
    successful_score = runner.score_prediction_store
    attempts = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("interrupted after label access")
        return successful_score(*args, **kwargs)

    monkeypatch.setattr(runner, "score_prediction_store", fail_once)
    with pytest.raises(RuntimeError, match="interrupted after label access"):
        runner.run_residual_topup_router_diagnostic(config)

    report_path = tmp_path / "reports/label_access_report.json"
    report_path.write_text('{"status":"tampered"}\n', encoding="utf-8")
    tampered_bytes = report_path.read_bytes()

    with pytest.raises(ProtocolError, match="durable JSON drifted"):
        runner.run_residual_topup_router_diagnostic(config)

    assert report_path.read_bytes() == tampered_bytes
