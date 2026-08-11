from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.constants import (
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_SOURCE_ROWS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.development_actions import (
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.runner import (
    run_fixed_bank_disagreement_regret_prediction_only,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.runner_dependencies import (
    PredictionOnlyDependencies,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.bundle import (
    assert_closed_world,
    cleanup_owned_atomic_temps,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json


@pytest.mark.parametrize("fit_fails", (False, True), ids=("complete", "fit-failure"))
def test_runner_freezes_source_models_before_loading_whole_test(
    tmp_path: Path, fit_fails: bool
) -> None:
    root = tmp_path / "bundle"
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("resolved: true\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    phases: list[str] = []
    model_seal_written = False
    test_admission_calls = 0
    test_load_calls = 0

    source_rows = tuple(
        SimpleNamespace(
            center=center,
            case_id=f"source-case-{center}",
            source_row_id=f"src-{center}",
        )
        for center in ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    )
    source_frame = SimpleNamespace(
        rows=source_rows,
        cache_binding={"labels_in_typed_frame": False},
    )
    test_rows = tuple(
        SimpleNamespace(
            center=center,
            case_id=f"test-case-{center}",
            evaluation_row_id=f"eval-{center}",
        )
        for center in ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    )
    test_frame = SimpleNamespace(rows=test_rows)
    target_classifier_bank = SimpleNamespace(seal_hash="9" * 64)
    strict_source_predictions = SimpleNamespace(seal_hash="8" * 64)
    source_predictions = SimpleNamespace(
        seal_hash="a" * 64,
        target_classifier_bank=target_classifier_bank,
    )
    test_predictions = SimpleNamespace(seal_hash="b" * 64)
    development = SimpleNamespace(model_bank_hash="c" * 64)
    inference = SimpleNamespace(frozen_prediction_hash="d" * 64)

    class Capability:
        def __init__(self, _frame: object, *, train_cache_root: Path) -> None:
            assert train_cache_root.is_absolute()

        def open_after_source_prediction_seal(self, seal: object) -> None:
            assert seal is source_predictions
            phases.append("source_labels_opened")

        def labels_for_outer_target(self, target: str) -> tuple[object, ...]:
            return (SimpleNamespace(query_id=f"donor-{target}"),)

        def access_report(self) -> dict[str, object]:
            payload = {
                "schema_version": (
                    "midogpp_prediction_only_source_label_capability_v1"
                ),
                "status": "OPEN_SOURCE_ONLY",
                "source_prediction_seal_hash": "a" * 64,
                "source_oof_classifier_bank_seal_hash": "8" * 64,
                "target_classifier_bank_seal_hash": "9" * 64,
                "source_labels_opened": True,
                "source_labels_opened_after_complete_prediction_seal": True,
                "source_row_count": EXPECTED_SOURCE_ROWS,
                "outer_targets_accessed": list(CENTERS),
                "outer_target_label_excluded": True,
                "query_excluded_from_every_source_action_composition": True,
                "source_oof_physical_classifier_fit_count": (
                    DEVELOPMENT_CLASSIFIER_FIT_COUNT
                ),
                "source_oof_oriented_prediction_cell_count": (
                    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
                ),
                "target_compatible_classifier_fit_count": (
                    EXPECTED_CLASSIFIER_FIT_COUNT
                ),
                "raw_source_labels_persisted": False,
                "raw_sample_ids_persisted": False,
                "test_manifest_opened": False,
                "test_labels_opened": False,
                "test_labels_available": False,
            }
            return {**payload, "access_report_hash": canonical_hash(payload)}

    def fit_development(*_args: object, **_kwargs: object) -> object:
        persisted = read_json(
            root / "manifests/source_label_capability_report.json"
        )
        assert persisted["source_labels_opened"] is True
        assert persisted["test_manifest_opened"] is False
        assert persisted["test_labels_opened"] is False
        phases.append("model_fit_started")
        if fit_fails:
            raise RuntimeError("synthetic model fit failure")
        return development

    def persist_development(path: Path, _products: object) -> None:
        nonlocal model_seal_written
        model_seal_written = True
        atomic_json(
            path / "manifests/model_bank_seal.json",
            {
                "status": "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION",
                "source_labels_only": True,
                "test_cache_admitted": False,
                "target_labels_used": False,
                "regret_model_bank_seal_hash": "e" * 64,
            },
        )
        phases.append("model_bank_sealed")

    def load_test(_config: object, *, admission: object) -> object:
        nonlocal test_load_calls
        test_load_calls += 1
        assert model_seal_written
        assert admission == "admission"
        phases.append("whole_test_loaded")
        return test_frame

    def issue_test_admission(*_args: object) -> str:
        nonlocal test_admission_calls
        test_admission_calls += 1
        return "admission"

    config = SimpleNamespace(
        artifact_root=root,
        expert_bank_root=tmp_path / "bank",
        generation_lock_root=tmp_path / "generation",
        train_cache_root=tmp_path / "train",
        test_cache_root=tmp_path / "test",
        test_consumption_ledger_path=tmp_path / "ledger.json",
        ledger_amendment_path=tmp_path / "amendment.json",
        runtime={},
        contract_hash="config-hash",
        expected_ledger_amendment_sha256="f" * 64,
        expected_test_cache_content_hash="1" * 64,
        expected_test_cache_row_order_hash="2" * 64,
    )
    dependencies = PredictionOnlyDependencies(
        validate_input_fence=lambda _config: None,
        validate_workspace=lambda _config: {"status": "PASS"},
        validate_provenance=lambda _root, _config: {},
        load_locks=lambda _config: SimpleNamespace(generation=object()),
        load_source_frame=lambda _config: source_frame,
        validate_firewall=lambda *_args: {"status": "PASS"},
        persist_initial=lambda *_args, **_kwargs: None,
        preflight=lambda *_args, **_kwargs: {"status": "PASS"},
        materialize_source_streams=lambda *_args, **_kwargs: object(),
        stage_source_streams=lambda source, **_kwargs: source,
        materialize_target_classifier_bank=lambda *_args, **_kwargs: (
            target_classifier_bank
        ),
        materialize_source_oof_predictions=lambda *_args, **_kwargs: (
            strict_source_predictions
        ),
        materialize_prelabel_prediction_seal=lambda *_args, **_kwargs: (
            source_predictions
        ),
        aggregate_source_probabilities=lambda *_args, **_kwargs: (object(),),
        build_contexts=lambda *_args, **_kwargs: object(),
        build_prelabel=lambda *_args, **_kwargs: object(),
        persist_prelabel=lambda *_args, **_kwargs: None,
        build_source_label_capability=Capability,
        fit_development=fit_development,
        persist_development=persist_development,
        issue_test_admission=issue_test_admission,
        load_test_frame=load_test,
        materialize_test_predictions=lambda *_args, **_kwargs: test_predictions,
        aggregate_test_probabilities=lambda *_args, **_kwargs: (object(),),
        build_inference=lambda *_args, **_kwargs: inference,
        persist_inference=lambda *_args, **_kwargs: None,
        build_runtime_summary=lambda **_kwargs: {"status": "PASS"},
        persist_reports=lambda *_args, **_kwargs: None,
        write_content_index=lambda *_args, **_kwargs: None,
        validate_bundle=lambda *_args, **_kwargs: {"status": "PASS"},
        persist_validation=lambda *_args, **_kwargs: None,
        write_state=lambda *_args, **_kwargs: None,
        phase_observer=phases.append,
    )

    if fit_fails:
        with pytest.raises(RuntimeError, match="synthetic model fit failure"):
            run_fixed_bank_disagreement_regret_prediction_only(
                config, artifact_root=root, dependencies=dependencies
            )
        capability_report = read_json(
            root / "manifests/source_label_capability_report.json"
        )
        assert capability_report["source_labels_opened"] is True
        assert capability_report["raw_source_labels_persisted"] is False
        assert capability_report["raw_sample_ids_persisted"] is False
        assert capability_report["test_manifest_opened"] is False
        assert capability_report["test_labels_opened"] is False
        assert capability_report["test_labels_available"] is False
        assert "model_bank_sealed" not in phases
        assert test_admission_calls == 0
        assert test_load_calls == 0
        return

    result = run_fixed_bank_disagreement_regret_prediction_only(
        config, artifact_root=root, dependencies=dependencies
    )

    assert result == root
    assert phases.index("strict_source_oof_prediction_seal") < phases.index(
        "source_labels_opened"
    )
    assert phases.index("target_classifier_bank_seal") < phases.index(
        "source_labels_opened"
    )
    assert phases.index("target_classifier_bank_seal") < phases.index(
        "strict_source_oof_prediction_seal"
    )
    assert phases.index("composite_prelabel_prediction_seal") < phases.index(
        "source_labels_opened"
    )
    assert phases.index("model_bank_sealed") < phases.index("whole_test_loaded")
    assert test_admission_calls == 1
    assert test_load_calls == 1
    assert "test_labels" not in " ".join(phases)


def test_incomplete_bundle_allows_only_owned_prediction_checkpoints(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    owned = (
        "checkpoints/disagreement_regret_source_predictions/tasks/a.json",
        "checkpoints/disagreement_regret_test_predictions/tasks/b.npz",
        "checkpoints/disagreement_regret_strict_source_oof/tasks/c.json",
    )
    for relative in owned:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint\n", encoding="utf-8")
        temp = path.with_name(f"{path.name}.123.tmp")
        temp.write_text("interrupted\n", encoding="utf-8")

    cleanup_owned_atomic_temps(root)
    assert not any(root.rglob("*.tmp"))
    assert_closed_world(root, allow_incomplete=True)
