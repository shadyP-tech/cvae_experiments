from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router import (
    persistence,
    runner,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.artifact_io import (
    persist_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.runner_dependencies import (
    MultiChallengerRouterDependencies,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.science_decisions import (
    _calibration_semantic_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_multi_challenger_hierarchical_flip_router.validation_science import (
    _CALIBRATION_FLOAT_PATHS,
    _CALIBRATION_IGNORED_PATHS,
    _DECISION_FLOAT_PATHS,
    _assert_semantic_equal,
    _assert_decision_table,
    _assert_table_derived,
    _directional_model,
    _validate_persisted_calibration_fingerprints,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.hierarchical_multi_challenger import (
    DirectionalCalibration,
    DirectionalLogitModel,
)
from midogpp_thesis.cvae.routing.threshold_flip_case_router import (
    DirectionSharedCalibration,
)


def test_terminal_checkpoint_is_atomic_and_contains_only_terminal_products(
    tmp_path: Path,
) -> None:
    result = {
        "terminal_case_confusions": ({"case_id": "c", "tp": 1},),
        "terminal_center_metrics": ({"target_center": "0", "bacc": 1.0},),
        "terminal_contrasts": ({"contrast_id": "R_multi-B", "estimate": 0.1},),
        "router_identification_metrics": ({"target_center": "0", "spearman": 0.0},),
        "permutation_metrics": ({"target_center": "0", "action_agreement": 1.0},),
        "menu_oracle_metrics": ({"target_center": "0", "menu_oracle_bacc": 1.0},),
        "sealed_terminal_evaluation": {
            "sealed_result_hash": "a" * 64,
            "raw_labels_persisted": False,
        },
    }
    reports = {
        "capability_report": {"status": "PASS", "raw_labels_persisted": False},
        "leakage_report": {"status": "PASS"},
        "publication_decision": {"decision": "DO_NOT_PROMOTE"},
        "runtime_summary": {"status": "PASS"},
    }
    checkpoint = persistence.persist_terminal_checkpoint(
        tmp_path, result=result, **reports
    )
    assert checkpoint["terminal_products_only"] is True
    assert (tmp_path / persistence.TERMINAL_CHECKPOINT_MEMBER).is_file()
    persistence.finalize_terminal_checkpoint(tmp_path)
    assert (tmp_path / "tables/menu_oracle_metrics.csv").is_file()
    assert (tmp_path / "manifests/sealed_terminal_evaluation.json").is_file()
    persistence.remove_validated_terminal_checkpoint(tmp_path)
    assert not (tmp_path / "checkpoints").exists()


def test_derived_numeric_replay_is_tolerant_but_decision_identity_is_exact() -> None:
    expected = {
        "action_id": "A1::source=7",
        "rank": 1,
        "reason": "positive_winner_runner_up_margin_lcb",
        "predicted_gain": 0.001234567890123,
    }
    replayed = {
        **expected,
        "predicted_gain": expected["predicted_gain"] + 2.0e-15,
    }
    _assert_semantic_equal(
        replayed,
        expected,
        role="decision",
        allowed_float_paths=_DECISION_FLOAT_PATHS,
    )
    with pytest.raises(ProtocolError, match="categorical"):
        _assert_semantic_equal(
            {**replayed, "action_id": "B"},
            expected,
            role="decision",
            allowed_float_paths=_DECISION_FLOAT_PATHS,
        )


def test_unallowlisted_calibration_float_is_exact() -> None:
    expected = {
        "family_calibrations": {
            "G": {
                "0to1": {
                    "offset": 0.125,
                    "offset_variance": 0.25,
                    "alpha": 4.0,
                    "success_count": 3,
                    "calibration_fingerprint": "a" * 64,
                }
            }
        },
        "single_challenger_calibration": {
            "gamma_0to1": 0.5,
            "gamma_1to0": 0.75,
            "calibration_hash": "b" * 64,
        },
    }
    allowed_drift = {
        **expected,
        "family_calibrations": {
            "G": {
                "0to1": {
                    **expected["family_calibrations"]["G"]["0to1"],
                    "offset": 0.125 + 2.0e-15,
                    "calibration_fingerprint": "c" * 64,
                }
            }
        },
    }
    _assert_semantic_equal(
        allowed_drift,
        expected,
        role="directional calibrations",
        allowed_float_paths=_CALIBRATION_FLOAT_PATHS,
        ignored_paths=_CALIBRATION_IGNORED_PATHS,
    )

    unallowlisted_drift = {
        **allowed_drift,
        "family_calibrations": {
            "G": {
                "0to1": {
                    **allowed_drift["family_calibrations"]["G"]["0to1"],
                    "alpha": 4.0 + 2.0e-15,
                }
            }
        },
    }
    with pytest.raises(ProtocolError, match="unallowlisted numeric"):
        _assert_semantic_equal(
            unallowlisted_drift,
            expected,
            role="directional calibrations",
            allowed_float_paths=_CALIBRATION_FLOAT_PATHS,
            ignored_paths=_CALIBRATION_IGNORED_PATHS,
        )


def test_static_decision_gain_remains_exact_even_below_fit_tolerance(
    tmp_path: Path,
) -> None:
    expected = {
        "method_id": "S_static",
        "action_id": "A1::source=7",
        "predicted_gain": 0.001,
        "action_margin": 0.0,
    }
    drifted = {**expected, "predicted_gain": 0.001 + 2.0e-15}
    path = tmp_path / "tables/method_decisions.csv"
    persist_rows(path, (drifted,), tuple(drifted))
    with pytest.raises(ProtocolError, match="unallowlisted numeric"):
        _assert_decision_table(path, (expected,))


def test_raw_calibration_fingerprint_is_checked_before_tolerant_replay(
    tmp_path: Path,
) -> None:
    expected = _calibration_row()
    tampered = deepcopy(expected)
    tampered["family_calibrations"]["G"]["0to1"]["offset"] += 2.0e-12
    path = tmp_path / "tables/directional_calibrations.csv"
    persist_rows(path, (tampered,), tuple(tampered))

    with pytest.raises(ProtocolError, match="fingerprint"):
        _validate_persisted_calibration_fingerprints(tmp_path, (expected,))


def test_raw_fit_fingerprint_is_checked_before_tolerant_replay() -> None:
    expected = DirectionalLogitModel(
        model_target="0",
        family="G",
        direction="0to1",
        feature_names=("feature",),
        feature_mean=(0.125,),
        feature_scale=(1.0,),
        candidate_sources=("1",),
        query_centers=("2",),
        coefficients=(0.125, 0.25, 0.375),
        covariance=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        feature_alpha=1.0,
        source_alpha=1.0,
        query_alpha=1.0,
        intercept_alpha=1.0,
        training_row_count=2,
        training_trial_count=4,
        training_case_clusters=("1::case-a", "2::case-b"),
        provenance_hash="a" * 64,
    ).to_payload()
    tampered = deepcopy(expected)
    tampered["coefficients"][0] += 2.0e-12

    with pytest.raises(ProtocolError, match="fingerprint"):
        _directional_model(tampered)


def test_self_validated_fitted_calibration_drift_within_tolerance_passes(
    tmp_path: Path,
) -> None:
    expected = _calibration_row()
    replayed = deepcopy(expected)
    payload = replayed["family_calibrations"]["G"]["0to1"]
    payload["offset"] += 2.0e-12
    replayed["family_calibrations"]["G"]["0to1"] = DirectionalCalibration(
        direction=payload["direction"],
        offset=payload["offset"],
        offset_variance=payload["offset_variance"],
        success_count=payload["success_count"],
        trial_count=payload["trial_count"],
        row_count=payload["row_count"],
        case_count=payload["case_count"],
        alpha=payload["alpha"],
        menu_hash=payload["menu_hash"],
        valid=payload["valid"],
    ).to_payload()
    path = tmp_path / "tables/directional_calibrations.csv"
    persist_rows(path, (replayed,), tuple(replayed))

    _validate_persisted_calibration_fingerprints(tmp_path, (expected,))
    _assert_table_derived(
        path,
        (expected,),
        role="directional calibrations",
        allowed_float_paths=_CALIBRATION_FLOAT_PATHS,
        ignored_paths=_CALIBRATION_IGNORED_PATHS,
    )


def test_runner_orders_all_seals_before_terminal_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    absolute = tmp_path.resolve()
    config = SimpleNamespace(
        source_path=root / "config.resolved.yaml",
        artifact_root=root,
        expert_bank_root=absolute / "bank",
        generation_lock_root=absolute / "generation",
        test_cache_root=absolute / "cache",
        test_manifest_path=absolute / "manifest.csv",
        test_consumption_ledger_path=absolute / "ledger.json",
        ledger_amendment_path=absolute / "amendment.json",
        input_artifact_ids=tuple(f"input-{index}" for index in range(6)),
        contract_hash="config-hash",
        runtime={},
    )
    events: list[str] = []
    frame = SimpleNamespace(cache_binding_hash="cache-hash")
    partition = SimpleNamespace(partition_hash="partition-hash")
    source = SimpleNamespace(root=root, records=tuple(range(81)), lock_hash="source-hash")
    prediction = SimpleNamespace(
        seal_hash="prediction-hash",
        store=SimpleNamespace(cells=tuple(range(810))),
    )
    probabilities = SimpleNamespace(rows=())
    prelabel = SimpleNamespace(feature_surface_hash="f" * 64)
    donor = SimpleNamespace()
    decisions = SimpleNamespace()
    gate = {"status": "FAIL"}
    terminal = {
        "sealed_terminal_evaluation": {
            "sealed_result_hash": "e" * 64,
            "diagnostic_routing_gate": gate,
        }
    }

    class FakeManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append("manager_constructed")

        def seal_all_fold_plans(self) -> tuple[object, ...]:
            events.append("45_plans_sealed")
            return ()

        def open_terminal_evaluation_labels(self) -> tuple[object, ...]:
            assert "donor_models_persisted" in events
            assert "fold_decisions_persisted" in events
            events.append("terminal_opened")
            return ()

        def report_payload(self) -> dict[str, object]:
            return {"terminal_scoring_opened": True}

    monkeypatch.setattr(runner, "assert_input_fence", lambda config: None)
    monkeypatch.setattr(
        runner,
        "validate_active_diagnostic_workspace_binding",
        lambda config: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "validate_workspace_provenance",
        lambda root, config: {item: {} for item in config.input_artifact_ids},
    )
    monkeypatch.setattr(
        runner,
        "load_validated_locks",
        lambda config: SimpleNamespace(generation=SimpleNamespace()),
    )
    monkeypatch.setattr(runner, "load_label_free_test_frame", lambda config: frame)
    monkeypatch.setattr(
        runner, "validate_pre_gpu_firewall", lambda *args: {"status": "PASS"}
    )
    monkeypatch.setattr(runner, "build_case_partition", lambda *args, **kwargs: partition)
    monkeypatch.setattr(
        runner,
        "run_workstation_preflight",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(runner, "persist_initial_surfaces", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "seed_probability_rows", lambda prediction: ())
    monkeypatch.setattr(runner, "aggregate_exact_nine", lambda rows: probabilities)
    monkeypatch.setattr(runner, "build_prelabel_surface", lambda *args, **kwargs: prelabel)
    monkeypatch.setattr(runner, "persist_prelabel_surfaces", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "MultiChallengerLabelCapabilityManager", FakeManager)
    monkeypatch.setattr(runner, "persist_fold_plans", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "persist_donor_models",
        lambda *args: events.append("donor_models_persisted"),
    )
    monkeypatch.setattr(
        runner,
        "persist_fold_decisions",
        lambda *args: events.append("fold_decisions_persisted"),
    )
    monkeypatch.setattr(
        runner,
        "leakage_report_payload",
        lambda **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        runner,
        "publication_decision_payload",
        lambda *args, **kwargs: {"decision": "DO_NOT_PROMOTE"},
    )
    monkeypatch.setattr(
        runner,
        "runtime_summary_payload",
        lambda **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(runner, "persist_terminal_checkpoint", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "finalize_terminal_checkpoint", lambda root: None)
    monkeypatch.setattr(runner, "remove_validated_terminal_checkpoint", lambda root: None)
    monkeypatch.setattr(runner, "write_content_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "validate_bundle",
        lambda *args, **kwargs: {
            "schema_version": "test",
            "status": "PASS",
        },
    )
    monkeypatch.setattr(runner, "persist_validation_report", lambda *args: None)
    monkeypatch.setattr(runner, "assert_completed_binding", lambda *args, **kwargs: None)

    deps = MultiChallengerRouterDependencies(
        materialize_source=lambda *args, **kwargs: source,
        stage_source=lambda *args, **kwargs: source,
        materialize_predictions=lambda *args, **kwargs: prediction,
        build_donor_models=lambda **kwargs: donor,
        build_fold_decisions=lambda **kwargs: decisions,
        evaluate_terminal=lambda **kwargs: terminal,
        cleanup_staging=lambda *args, **kwargs: events.append("cleanup"),
        phase_observer=events.append,
    )
    assert runner._run(config, artifact_root=root, deps=deps) == root
    assert events.index("45_plans_sealed") < events.index("donor_models_persisted")
    assert events.index("donor_models_persisted") < events.index(
        "fold_decisions_persisted"
    )
    assert events.index("fold_decisions_persisted") < events.index("terminal_opened")
    assert events[-1] == "cleanup"


def _calibration_row() -> dict[str, object]:
    menu_hash = "a" * 64
    family_calibrations = {
        family: {
            direction: DirectionalCalibration(
                direction=direction,
                offset=0.125,
                offset_variance=0.25,
                success_count=3,
                trial_count=6,
                row_count=2,
                case_count=2,
                alpha=4.0,
                menu_hash=menu_hash,
                valid=True,
            ).to_payload()
            for direction in ("0to1", "1to0")
        }
        for family in ("G", "R", "P")
    }
    single = DirectionSharedCalibration(
        gamma_0to1=0.5,
        gamma_1to0=0.75,
        n_positive=3,
        n_negative=3,
        row_count=2,
        valid=True,
    ).to_payload()
    unhashed = {
        "target_center": "0",
        "fold_ordinal": 0,
        "family_calibrations": family_calibrations,
        "single_challenger_calibration": single,
    }
    return {**unhashed, "row_hash": _calibration_semantic_hash(unhashed)}
