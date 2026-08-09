from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.bundle import (
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.inputs import (
    validate_workspace_provenance,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.reports import (
    leakage_report_payload,
    protocol_manifest_payload,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.runner import (
    UtilityAlignedRunnerDependencies,
    run_utility_aligned_exact_tail_router_diagnostic,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_content_index_detects_scientific_member_tamper(tmp_path: Path) -> None:
    _write_required_inventory(tmp_path)
    write_content_index(tmp_path, config_contract_hash="contract")
    validate_content_index(tmp_path, config_contract_hash="contract")

    member = tmp_path / "tables/target_ensemble_metrics.csv"
    member.write_bytes(member.read_bytes() + b"tamper")
    with pytest.raises(ProtocolError, match="content-index member"):
        validate_content_index(tmp_path, config_contract_hash="contract")


def test_closed_world_rejects_unindexed_extra_file(tmp_path: Path) -> None:
    _write_required_inventory(tmp_path)
    (tmp_path / "reports/untracked.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_checkpoints_are_resumable_only_before_complete_publication(
    tmp_path: Path,
) -> None:
    _write_required_inventory(tmp_path)
    checkpoint = tmp_path / "checkpoints/target_predictions/task.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n", encoding="utf-8")

    assert_closed_world(tmp_path, allow_incomplete=True)
    assert_closed_world(
        tmp_path, allow_incomplete=False, allow_pending_validation=True
    )
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_complete_fast_path_validates_and_does_no_scientific_work(
    tmp_path: Path,
) -> None:
    _write_required_inventory(tmp_path)
    _write_json(
        tmp_path / "reports/run_state.json",
        {"status": "COMPLETE", "phase": "COMPLETE"},
    )
    calls: list[str] = []
    deps = UtilityAlignedRunnerDependencies(
        validate_bundle=lambda root, **kwargs: calls.append("validate") or {"status": "PASS"},
        phase_observer=calls.append,
    )

    observed = run_utility_aligned_exact_tail_router_diagnostic(
        _config(tmp_path), artifact_root=tmp_path, dependencies=deps
    )

    assert observed == tmp_path
    assert calls == ["validate"]


def test_incomplete_complete_marker_fails_closed_before_fast_path_validator(
    tmp_path: Path,
) -> None:
    _write_launch_files(tmp_path)
    _write_json(
        tmp_path / "reports/run_state.json",
        {"status": "COMPLETE", "phase": "COMPLETE"},
    )
    calls: list[str] = []
    deps = UtilityAlignedRunnerDependencies(
        validate_bundle=lambda root, **kwargs: calls.append("validate") or {"status": "PASS"}
    )

    with pytest.raises(ProtocolError, match="closed-world inventory"):
        run_utility_aligned_exact_tail_router_diagnostic(
            _config(tmp_path), artifact_root=tmp_path, dependencies=deps
        )
    assert calls == []


def test_runner_dependency_order_freezes_before_terminal_scoring(
    tmp_path: Path,
) -> None:
    _write_launch_files(tmp_path)
    order: list[str] = []
    generation = SimpleNamespace(generation_lock_hash="generation")
    locks = SimpleNamespace(generation=generation)
    frame = SimpleNamespace(cache_binding_hash="cache")
    partitions = SimpleNamespace(lock_hash="partitions")
    folds = SimpleNamespace(lock_hash="folds")
    cache = SimpleNamespace(
        root=tmp_path,
        source_records=(object(),) * 81,
        component_records=(object(),) * 216,
    )
    surfaces = object()
    production = SimpleNamespace(
        surfaces=surfaces,
        inner_rows=(object(),) * 4_536,
        target_rows=(object(),) * 648,
    )
    development = SimpleNamespace(
        seal=SimpleNamespace(prediction_seal_hash="inner-seal", cell_count=5_184)
    )
    models = SimpleNamespace(model_set_hash="models", by_target={str(i): object() for i in range(9)})
    plans = SimpleNamespace(plan_set_hash="plans", by_target={str(i): object() for i in range(9)})
    actions = SimpleNamespace(action_library_hash="actions", action_count=117)
    predictions = SimpleNamespace(cells=(object(),) * 1_053, unique_classifier_fit_count=999)

    noop = lambda *args, **kwargs: None
    deps = UtilityAlignedRunnerDependencies(
        validate_workspace=lambda config: {"status": "PASS"},
        validate_provenance=lambda root, config: {
            artifact_id: {} for artifact_id in config.input_artifact_ids
        },
        load_locks=lambda config: locks,
        load_frame=lambda config: frame,
        validate_firewall=lambda config, value: {"status": "PASS"},
        build_partitions=lambda value, **kwargs: partitions,
        build_case_folds=lambda value, **kwargs: folds,
        run_preflight=lambda root, **kwargs: {"status": "PASS"},
        materialize_source=lambda *args, **kwargs: cache,
        validate_source_lock=lambda *args, **kwargs: {"source_cache_lock_hash": "source-lock"},
        stage_source=lambda value, **kwargs: value,
        load_metadata=lambda config: {},
        produce_features=lambda *args: production,
        materialize_development=lambda *args, **kwargs: development,
        open_development_labels=lambda *args, **kwargs: object(),
        score_development=lambda *args: (object(),) * 4_536,
        fit_models=lambda *args: models,
        build_plans=lambda *args: plans,
        build_actions=lambda *args: actions,
        materialize_target=lambda *args, **kwargs: predictions,
        build_target_seal=lambda *args, **kwargs: {"seal_hash": "target-seal"},
        open_target_labels=lambda *args, **kwargs: ({}, {"status": "PASS"}),
        score_seed_cells=lambda *args: (),
        score_ensembles=lambda *args: (),
        build_contrasts=lambda *args: (),
        infer_contrasts=lambda *args: (),
        build_oracle=lambda *args: (),
        validate_bundle=lambda *args, **kwargs: {"status": "PASS"},
        persist_initial=noop,
        persist_source_features=noop,
        persist_development_router=noop,
        persist_target_seal=noop,
        persist_terminal=noop,
        write_index=lambda *args, **kwargs: {},
        persist_validation=noop,
        write_state=noop,
        phase_observer=order.append,
    )

    run_utility_aligned_exact_tail_router_diagnostic(
        _config(tmp_path), artifact_root=tmp_path, dependencies=deps
    )

    assert order == [
        "workspace",
        "provenance",
        "firewall",
        "partitions",
        "preflight",
        "source_cache",
        "features",
        "inner_predictions",
        "development_labels",
        "development_scoring",
        "models",
        "plans",
        "actions",
        "target_predictions",
        "target_seal",
        "target_labels",
        "terminal_scoring",
        "validation",
    ]


def test_reports_use_truthful_crossfit_then_terminal_scoring_boundary() -> None:
    config = SimpleNamespace(contract_hash="contract")
    manifest = protocol_manifest_payload(
        config,
        input_artifact_hashes={},
        validation_cache_binding_hash="cache",
        firewall={"status": "PASS"},
    )
    leakage = leakage_report_payload(
        support_partition_lock_hash="support",
        case_fold_lock_hash="folds",
        development_prediction_seal_hash="inner",
        model_set_hash="models",
        plan_set_hash="plans",
        action_library_hash="actions",
        target_prediction_seal_hash="target",
        firewall={"status": "PASS"},
    )
    assert manifest["development_crossfit_labels_opened_before_target_action_lock"] is True
    assert manifest["outer_H_development_rows_excluded_from_plan_H"] is True
    assert manifest["target_predictions_sealed_before_terminal_target_scoring"] is True
    assert "target_predictions_sealed_before_target_labels" not in manifest
    assert leakage["terminal_target_scoring_capability_opened_after_target_seal"] is True
    assert leakage["development_crossfit_labels_previously_opened"] is True
    assert "target_labels_opened_only_after_global_target_seal" not in leakage


def test_workspace_provenance_accepts_workspace_lexical_order(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_json(
        tmp_path / "provenance/input_artifacts.json",
        _provenance_payload(config, tuple(sorted(config.input_artifact_ids))),
    )

    observed = validate_workspace_provenance(tmp_path, config)

    assert tuple(observed) == config.input_artifact_ids


@pytest.mark.parametrize(
    "order",
    (
        ("bank", "generation", "policy", "cache", "manifest", "metadata"),
        ("bank", "cache", "generation", "manifest", "metadata", "metadata"),
        ("bank", "cache", "generation", "manifest", "metadata", "undeclared"),
    ),
)
def test_workspace_provenance_rejects_noncanonical_or_drifted_inputs(
    tmp_path: Path,
    order: tuple[str, ...],
) -> None:
    config = _config(tmp_path)
    _write_json(
        tmp_path / "provenance/input_artifacts.json",
        _provenance_payload(config, order),
    )

    with pytest.raises(ProtocolError, match="provenance order drifted"):
        validate_workspace_provenance(tmp_path, config)


def _config(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id="experiment",
        artifact_root=root,
        expert_bank_root=root / "bank",
        generation_lock_root=root / "generation",
        equal_union_policy_root=root / "policy",
        validation_cache_root=root / "cache",
        validation_manifest_path=root / "manifest.csv",
        metadata_profile_root=root / "metadata",
        input_artifact_ids=("bank", "generation", "policy", "cache", "manifest", "metadata"),
        contract_hash="contract",
        runtime={},
    )


def _provenance_payload(
    config: SimpleNamespace,
    order: tuple[str, ...],
) -> dict[str, object]:
    paths = {
        "bank": config.expert_bank_root,
        "generation": config.generation_lock_root,
        "policy": config.equal_union_policy_root,
        "cache": config.validation_cache_root,
        "manifest": config.validation_manifest_path.parent,
        "metadata": config.metadata_profile_root,
        "undeclared": config.artifact_root / "undeclared",
    }
    return {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": config.experiment_id,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "input_artifacts": [
            {
                "artifact_id": artifact_id,
                "resolved_path": str(paths[artifact_id]),
                "exists": True,
                "semantic_identities": {},
                "file_integrity": {},
            }
            for artifact_id in order
        ],
    }


def _write_launch_files(root: Path) -> None:
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text("experiment: {}\n", encoding="utf-8")
    _write_json(root / "provenance/input_artifacts.json", {})


def _write_required_inventory(root: Path) -> None:
    for member in REQUIRED_FILES:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture::{member}\n".encode("utf-8"))
    _write_json(
        root / "reports/run_state.json", {"status": "COMPLETE", "phase": "COMPLETE"}
    )
    _write_json(
        root / "reports/validation_report.json", {"status": "PASS"}
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
