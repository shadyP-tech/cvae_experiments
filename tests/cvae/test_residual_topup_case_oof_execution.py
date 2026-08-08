from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof import runner
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.artifact_io import (
    persist_or_validate_csv,
    read_json,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.validation import (
    _assert_csv,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.contracts import (
    EXPECTED_SEALED_PREDICTION_CELL_COUNT,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.prediction_store import (
    PREDICTION_INDEX_COLUMNS,
    assemble_prediction_store,
    read_prediction_store,
    write_prediction_store,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_case_oof.runner import (
    CaseOOFRunnerDependencies,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_runner_fails_closed_at_data_firewall_before_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("resolved: true\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text(
        "{}\n", encoding="utf-8"
    )
    config = SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=(tmp_path / "bank").resolve(),
        generation_lock_root=(tmp_path / "generation").resolve(),
        equal_union_policy_root=(tmp_path / "policy").resolve(),
        validation_cache_root=(tmp_path / "cache").resolve(),
        validation_manifest_path=(tmp_path / "manifest.csv").resolve(),
        contract_hash="a" * 64,
    )
    events: list[str] = []
    monkeypatch.setattr(
        runner,
        "validate_active_diagnostic_workspace_binding",
        lambda config: events.append("workspace") or {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "validate_workspace_provenance",
        lambda root, config: events.append("provenance") or {},
    )
    monkeypatch.setattr(
        runner,
        "load_validated_locks",
        lambda config: events.append("locks") or object(),
    )
    monkeypatch.setattr(
        runner,
        "load_label_free_validation_frame",
        lambda config: events.append("frame") or object(),
    )

    def _reject(config: object, frame: object) -> object:
        events.append("firewall")
        raise ProtocolError("bank/cache split evidence absent")

    monkeypatch.setattr(runner, "validate_pre_gpu_firewall", _reject)
    monkeypatch.setattr(
        runner,
        "run_workstation_preflight",
        lambda *args, **kwargs: events.append("gpu_preflight"),
    )
    monkeypatch.setattr(
        runner,
        "materialize_source_cache",
        lambda *args, **kwargs: events.append("gpu_source"),
    )

    with pytest.raises(ProtocolError, match="split evidence absent"):
        runner.run_residual_topup_case_oof_diagnostic(config, artifact_root=root)

    assert events == ["workspace", "provenance", "locks", "frame", "firewall"]
    state = read_json(root / "reports/run_state.json")
    assert state["status"] == "FAILED"
    assert state["phase"] == "INITIALIZING"


def test_prediction_store_float32_npz_csv_roundtrip(tmp_path: Path) -> None:
    generated_fields = {
        "cell_ordinal",
        "prediction_offset_start",
        "prediction_offset_stop",
        "prediction_sha256",
        "probability_sha256",
    }
    metadata_fields = tuple(
        field for field in PREDICTION_INDEX_COLUMNS if field not in generated_fields
    )
    cells = []
    for ordinal in range(EXPECTED_SEALED_PREDICTION_CELL_COUNT):
        metadata = {field: "" for field in metadata_fields}
        metadata.update(
            {
                "schema_version": "test_case_oof_prediction_v1",
                "fold_ordinal": ordinal,
                "training_seed": 17,
                "generation_seed": 17,
                "classifier_converged": True,
                "fit_aliased_by_composition_hash": ordinal > 0,
            }
        )
        cells.append(
            {
                "predictions": np.asarray([ordinal % 2], dtype=np.uint8),
                "probabilities": np.asarray(
                    [0.25 if ordinal % 2 == 0 else 0.75], dtype=np.float32
                ),
                "metadata": metadata,
            }
        )
    store = assemble_prediction_store(cells, unique_classifier_fit_count=17)
    write_prediction_store(tmp_path, store)
    loaded = read_prediction_store(tmp_path)

    assert loaded.unique_classifier_fit_count == 17
    assert loaded.y_pred.dtype == np.uint8
    assert loaded.prob_pos.dtype == np.float32
    assert np.array_equal(loaded.y_pred, store.y_pred)
    assert np.array_equal(loaded.prob_pos, store.prob_pos)
    assert loaded.index_rows == tuple(
        {key: str(value) for key, value in row.items()}
        for row in store.index_rows
    )


def test_csv_resume_and_reconstructive_validation_preserve_crlf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resume.csv"
    rows = ({"sample_id": "case-001", "value": 17},)
    columns = ("sample_id", "value")

    persist_or_validate_csv(path, rows, columns=columns)
    assert path.read_bytes() == b"sample_id,value\r\ncase-001,17\r\n"

    persist_or_validate_csv(path, rows, columns=columns)
    _assert_csv(path, rows, columns=columns)


def test_runner_globally_seals_before_label_capability_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("resolved: true\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    config = SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=(tmp_path / "bank").resolve(),
        generation_lock_root=(tmp_path / "generation").resolve(),
        equal_union_policy_root=(tmp_path / "policy").resolve(),
        validation_cache_root=(tmp_path / "cache").resolve(),
        validation_manifest_path=(tmp_path / "manifest.csv").resolve(),
        contract_hash="a" * 64,
        runtime={},
    )
    events: list[str] = []
    generation = SimpleNamespace(generation_lock_hash="b" * 64)
    locks = SimpleNamespace(generation=generation)
    frame = object()
    base = SimpleNamespace(lock_hash="c" * 64)
    crossfit = SimpleNamespace(lock_hash="d" * 64)
    plan = SimpleNamespace(lock_hash="e" * 64)

    class _Source:
        compatibility_case_rows = ()

        def proxy_score_rows(self, crossfit: object) -> tuple[None, ...]:
            return (None,) * 3456

    source = _Source()
    predictions = SimpleNamespace(unique_classifier_fit_count=17)

    monkeypatch.setattr(runner, "build_partition_surface", lambda *a, **k: base)
    monkeypatch.setattr(runner, "build_case_oof_surface", lambda *a, **k: crossfit)
    monkeypatch.setattr(runner, "persist_initial_surfaces", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "validate_source_cache_lock",
        lambda *a, **k: {"source_cache_lock_hash": "f" * 64},
    )
    monkeypatch.setattr(runner, "build_rank_surface", lambda *a, **k: {})
    monkeypatch.setattr(runner, "build_case_oof_plan", lambda *a, **k: plan)
    monkeypatch.setattr(runner, "persist_source_phase", lambda *a, **k: None)
    monkeypatch.setattr(runner, "persist_rank_and_plan_surfaces", lambda *a, **k: None)
    monkeypatch.setattr(
        runner, "validate_prediction_store_binding", lambda *a, **k: None
    )

    def _build_seal(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("seal_build")
        return {"seal_hash": "1" * 64}

    def _validate_seal(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("seal_validate")
        return {"seal_hash": "1" * 64}

    monkeypatch.setattr(runner, "build_global_prediction_seal", _build_seal)
    monkeypatch.setattr(runner, "validate_global_prediction_seal", _validate_seal)
    monkeypatch.setattr(runner, "persist_prediction_phase", lambda *a, **k: None)
    monkeypatch.setattr(
        runner,
        "score_center_seed_cells",
        lambda *a, **k: events.append("score_seed") or (),
    )
    monkeypatch.setattr(
        runner,
        "score_center_probability_ensembles",
        lambda *a, **k: events.append("score_ensemble") or (),
    )
    monkeypatch.setattr(runner, "build_center_contrasts", lambda rows: ())
    monkeypatch.setattr(runner, "infer_center_contrasts", lambda rows: ())
    monkeypatch.setattr(runner, "build_oracle_hxe_diagnostics", lambda *a, **k: ())
    monkeypatch.setattr(runner, "scoring_summary_payload", lambda *a: {})
    monkeypatch.setattr(runner, "leakage_report_payload", lambda **k: {})
    monkeypatch.setattr(runner, "runtime_summary_payload", lambda *a, **k: {})
    monkeypatch.setattr(runner, "publication_decision_payload", lambda summary: {})
    monkeypatch.setattr(runner, "persist_terminal_surfaces", lambda *a, **k: None)
    monkeypatch.setattr(runner, "write_content_index", lambda *a, **k: {})

    def _predictions(*args: object, **kwargs: object) -> object:
        events.append("predictions")
        return predictions

    def _labels(*args: object, **kwargs: object):
        events.append("labels")
        return {}, {"status": "SEALED_LABEL_ACCESS"}

    def _validate_bundle(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("bundle_validate")
        return {"status": "PASS"}

    deps = CaseOOFRunnerDependencies(
        validate_workspace=lambda config: {"status": "PASS"},
        validate_provenance=lambda root, config: {},
        load_locks=lambda config: locks,
        load_frame=lambda config: frame,
        validate_firewall=lambda config, frame: {"status": "PASS"},
        run_preflight=lambda *a, **k: {"status": "PASS"},
        materialize_source=lambda *a, **k: source,
        materialize_predictions=_predictions,
        open_labels=_labels,
        validate_bundle=_validate_bundle,
    )
    output = runner.run_residual_topup_case_oof_diagnostic(
        config, artifact_root=root, dependencies=deps
    )

    assert output == root
    assert events == [
        "predictions",
        "seal_build",
        "seal_validate",
        "labels",
        "score_seed",
        "score_ensemble",
        "bundle_validate",
        "bundle_validate",
    ]
    assert read_json(root / "reports/run_state.json")["status"] == "COMPLETE"


def test_source_cache_public_module_is_a_thin_facade() -> None:
    source = (
        Path(runner.__file__).with_name("source_cache.py").read_text(encoding="utf-8")
    )
    assert len(source.splitlines()) < 80
    assert "residual_topup_router" not in source
    assert "source_cache_execution" in source
    assert "source_cache_validation" in source
