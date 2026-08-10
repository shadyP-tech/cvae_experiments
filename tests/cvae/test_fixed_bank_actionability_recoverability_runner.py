from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.bundle import (
    REQUIRED_FILES,
    cleanup_owned_atomic_temps,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.runner import (
    FixedBankActionabilityRecoverabilityDependencies,
    _exclusive_run_lock,
    run_fixed_bank_actionability_recoverability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
    execution_adapter,
    runner_runtime,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability.persistence import (
    persist_all_decisions,
)
from midogpp_thesis.cvae.runtime.artifact_io import read_json
from midogpp_thesis.cvae.protocol import ProtocolError


_HASH = "a" * 64
_ACTION_HASH = "b" * 64


def _config(root: Path, base: Path) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_root=root.resolve(),
        expert_bank_root=(base / "bank").resolve(),
        generation_lock_root=(base / "generation").resolve(),
        test_cache_root=(base / "cache").resolve(),
        test_manifest_path=(base / "manifest.csv").resolve(),
        test_consumption_ledger_path=(base / "ledger.json").resolve(),
        ledger_amendment_path=(base / "amendment.json").resolve(),
        input_artifact_ids=tuple(f"input-{index}" for index in range(6)),
        runtime={
            "model_workers": 4,
            "model_threads_per_worker": 3,
            "bootstrap_workers": 4,
            "bootstrap_threads_per_worker": 3,
            "multiprocessing_start_method": "spawn",
            "classifier_workers": 4,
            "classifier_threads_per_worker": 3,
            "scratch_preference": [
                "/data/local/fixed_bank_actionability_recoverability_v1",
                "artifact_parent",
            ],
        },
        evaluation={
            "whole_case_cluster_bootstrap_replicates": 10_000,
            "whole_case_cluster_bootstrap_seed": 90_912_029,
        },
        contract_hash=_HASH,
    )


def _launch(root: Path) -> None:
    (root / "provenance").mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text("experiment: test\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")


def test_runner_persists_every_boundary_before_the_next_label_capability(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _launch(root)
    config = _config(root, tmp_path)
    flags = {
        "models_persisted": False,
        "models_recorded": False,
        "pre_support_persisted": False,
        "pre_support_recorded": False,
        "all_decisions_persisted": False,
        "preevaluation_recorded": False,
        "capability_persisted": False,
        "evaluation_opened": False,
        "postseal_persisted": False,
        "index_written": False,
    }
    phases: list[str] = []
    frame = SimpleNamespace(cache_binding_hash=_HASH)
    partition = SimpleNamespace(partition_hash=_HASH)
    source = SimpleNamespace(lock_hash=_HASH, records=(object(),))
    prediction = SimpleNamespace(
        seal_hash=_HASH,
        store=SimpleNamespace(
            action_library_hash=_ACTION_HASH,
            store_hash="c" * 64,
            cells=tuple(range(1458)),
        ),
    )
    prelabel = SimpleNamespace(
        feature_surface_hash="d" * 64,
        permutation_provenance_hash="e" * 64,
    )
    models = SimpleNamespace(model_seals_by_target={})
    pre_support = SimpleNamespace()
    decisions = SimpleNamespace(
        all_decision_hashes={index: _HASH for index in range(495)}
    )

    class _Manager:
        def open_loco_donor_labels(self, target: str) -> tuple[object, ...]:
            assert not flags["pre_support_recorded"]
            return (SimpleNamespace(target=target),)

        def open_fold_support_labels(
            self, target: str, fold: int
        ) -> tuple[object, ...]:
            assert flags["pre_support_recorded"]
            return (SimpleNamespace(target=target, fold=fold),)

        def open_oof_evaluation_labels(self) -> tuple[object, ...]:
            assert flags["preevaluation_recorded"]
            flags["evaluation_opened"] = True
            return (object(),)

        def access_report(self) -> dict[str, object]:
            assert flags["evaluation_opened"]
            return {
                "status": "PASS",
                "evaluation_labels_opened": True,
                "report_hash": "f" * 64,
            }

    manager = _Manager()

    def persist_initial(path: Path, **_: object) -> None:
        target = path / "manifests/action_library.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"action_library_hash": _ACTION_HASH}), encoding="utf-8"
        )

    def persist_prelabel(path: Path, **_: object) -> None:
        manifests = path / "manifests"
        manifests.mkdir(parents=True, exist_ok=True)
        (manifests / "prelabel_feature_seal.json").write_text(
            json.dumps(
                {
                    "feature_surface_hash": prelabel.feature_surface_hash,
                    "permutation_provenance_hash": (
                        prelabel.permutation_provenance_hash
                    ),
                }
            ),
            encoding="utf-8",
        )
        (manifests / "sealed_probability_surface.json").write_text(
            json.dumps({"global_prediction_seal_hash": prediction.seal_hash}),
            encoding="utf-8",
        )

    def persist_models(_: Path, **kwargs: object) -> None:
        assert len(kwargs["utility_products"]) == 9
        assert len(kwargs["target_products"]) == 9
        flags["models_persisted"] = True

    def record_models(capability: object, products: object) -> None:
        assert capability is manager and products is models
        assert flags["models_persisted"]
        flags["models_recorded"] = True

    def build_pre_support(products: object, scope: object) -> object:
        assert products is models and scope is partition
        assert flags["models_recorded"]
        return pre_support

    def persist_pre(_: Path, *, products: object) -> None:
        assert products is pre_support
        flags["pre_support_persisted"] = True

    def record_pre(capability: object, products: object) -> None:
        assert capability is manager and products is pre_support
        assert flags["pre_support_persisted"]
        flags["pre_support_recorded"] = True

    def persist_decisions(_: Path, *, products: object) -> None:
        assert products is decisions
        flags["all_decisions_persisted"] = True

    def record_preevaluation(capability: object, products: object) -> None:
        assert capability is manager and products is decisions
        assert flags["all_decisions_persisted"]
        flags["preevaluation_recorded"] = True

    def persist_capability(path: Path, report: object) -> None:
        assert flags["evaluation_opened"]
        (path / "reports").mkdir(parents=True, exist_ok=True)
        (path / "reports/label_capability_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        flags["capability_persisted"] = True

    def evaluate(*_: object, **__: object) -> object:
        assert flags["capability_persisted"]
        return object()

    def persist_postseal(_: Path, **__: object) -> None:
        assert flags["capability_persisted"]
        flags["postseal_persisted"] = True

    def write_index(_: Path, **__: object) -> None:
        assert flags["postseal_persisted"]
        flags["index_written"] = True

    def validate(_: Path, **__: object) -> dict[str, object]:
        assert flags["index_written"]
        return {"status": "PASS"}

    deps = FixedBankActionabilityRecoverabilityDependencies(
        validate_inputs=lambda _: None,
        validate_workspace=lambda _: {"status": "PASS"},
        validate_provenance=lambda *_: {
            name: {"id": name} for name in config.input_artifact_ids
        },
        load_locks=lambda _: SimpleNamespace(generation=object()),
        load_frame=lambda _: frame,
        validate_firewall=lambda *_: {"status": "PASS"},
        build_partition=lambda *_args, **_kwargs: partition,
        persist_initial=persist_initial,
        preflight=lambda *_args, **_kwargs: {"status": "PASS"},
        materialize_source=lambda *_args, **_kwargs: source,
        stage_source=lambda *_args, **_kwargs: source,
        materialize_predictions=lambda *_args, **_kwargs: prediction,
        build_seed_rows=lambda _: (object(),),
        aggregate_probabilities=lambda _: object(),
        build_prelabel=lambda *_args, **_kwargs: prelabel,
        persist_prelabel=persist_prelabel,
        build_label_manager=lambda *_args, **_kwargs: manager,
        build_utility=lambda *_args, **_kwargs: object(),
        fit_target_model=lambda *_args, **_kwargs: object(),
        combine_models=lambda *_args, **_kwargs: models,
        persist_models=persist_models,
        record_models=record_models,
        build_pre_support=build_pre_support,
        persist_pre_support=persist_pre,
        record_pre_support=record_pre,
        build_support_fold=lambda *_args, **_kwargs: object(),
        combine_decisions=lambda *_args, **_kwargs: decisions,
        persist_decisions=persist_decisions,
        record_preevaluation=record_preevaluation,
        persist_capability=persist_capability,
        evaluate=evaluate,
        persist_postseal=persist_postseal,
        write_index=write_index,
        validate_bundle=validate,
        persist_validation=lambda *_args, **_kwargs: None,
        write_state=lambda *_args, **_kwargs: None,
        phase_observer=phases.append,
    )
    assert run_fixed_bank_actionability_recoverability(
        config, artifact_root=root.resolve(), dependencies=deps
    ) == root.resolve()
    assert all(flags.values())
    assert phases.index("loco_donor_labels_and_models") < phases.index(
        "same_H_support_S_y"
    ) < phases.index("terminal_evaluation_labels") < phases.index("validation")


def test_runner_refuses_automatic_replay_after_terminal_labels_were_opened(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _launch(root)
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/label_capability_report.json").write_text(
        json.dumps({"evaluation_labels_opened": True}), encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="Terminal-label-phase recovery is forbidden"):
        run_fixed_bank_actionability_recoverability(
            _config(root, tmp_path), artifact_root=root.resolve()
        )


def test_decision_persistence_round_trips_support_hash_keys_idempotently(
    tmp_path: Path,
) -> None:
    decision = SimpleNamespace(
        target_center="0",
        case_id="case-0",
        method_id="B",
        action_id="B",
        geometry_id=None,
        predicted_gain=0.0,
        decision_source="global_baseline",
        evaluation_labels_used=False,
    )
    support_score = SimpleNamespace(
        target_center="0",
        fold_ordinal=0,
        geometry_id="A0",
        action_id="U",
        support_exact_bacc=0.5,
    )
    products = SimpleNamespace(
        decisions=(decision,),
        all_decision_hashes={("0", 0, "B", None): _HASH},
        support_action_scores=(support_score,),
        support_product_hashes=(("0", 0, _HASH),),
        pre_support_seal_hash=_HASH,
        all_decisions_seal_hash=_HASH,
        permutation_provenance_hash=_HASH,
        partition_hash=_HASH,
        protocol_contract_hash=_HASH,
    )

    persist_all_decisions(tmp_path, products=products)
    persist_all_decisions(tmp_path, products=products)

    manifest = read_json(tmp_path / "manifests/all_method_decisions_seal.json")
    assert manifest["support_product_hashes"] == [["0", 0, _HASH]]


def test_terminal_marker_recovers_only_index_and_validation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    for member in REQUIRED_FILES:
        if member in {
            "manifests/content_index.json",
            "reports/validation_report.json",
        }:
            continue
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if member == "reports/label_capability_report.json":
            path.write_text(
                json.dumps({"evaluation_labels_opened": True}), encoding="utf-8"
            )
        elif member == "reports/runtime_summary.json":
            path.write_text(
                json.dumps({"local_source_staging": {"used": False}}),
                encoding="utf-8",
            )
        else:
            path.write_text("{}\n", encoding="utf-8")
    calls: list[str] = []

    def write_index(path: Path, **_: object) -> None:
        calls.append("index")
        (path / "manifests/content_index.json").write_text("{}\n", encoding="utf-8")

    def validate(*_: object, **__: object) -> dict[str, object]:
        calls.append("validate")
        return {"status": "PASS"}

    deps = FixedBankActionabilityRecoverabilityDependencies(
        write_index=write_index,
        validate_bundle=validate,
    )
    assert run_fixed_bank_actionability_recoverability(
        _config(root, tmp_path), artifact_root=root.resolve(), dependencies=deps
    ) == root.resolve()
    assert calls == ["index", "validate", "validate"]
    assert (root / "reports/validation_report.json").is_file()


def test_lock_and_atomic_temp_cleanup_are_scope_limited(tmp_path: Path) -> None:
    with _exclusive_run_lock(tmp_path):
        with pytest.raises(ProtocolError, match="already running"):
            with _exclusive_run_lock(tmp_path):
                raise AssertionError("unreachable")
    owned = tmp_path / "tables/method_decisions.csv.123.tmp"
    unrelated = tmp_path / "notes.txt.456.tmp"
    for path in (owned, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial", encoding="utf-8")
    cleanup_owned_atomic_temps(tmp_path)
    assert not owned.exists()
    assert unrelated.is_file()


def test_validated_staging_cleanup_removes_only_the_exact_owned_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "actionability-scratch"
    owned = scratch / "source_cache"
    unrelated = scratch / "keep-me"
    owned.mkdir(parents=True)
    unrelated.mkdir()
    (owned / "member").write_text("owned", encoding="utf-8")
    (unrelated / "member").write_text("unrelated", encoding="utf-8")
    lock = {"source": _HASH}
    canonical = SimpleNamespace(lock_payload=lock)
    monkeypatch.setattr(execution_adapter, "SCRATCH_ROOT", str(scratch))
    monkeypatch.setattr(
        execution_adapter,
        "load_frozen_source_streams",
        lambda *_args, **_kwargs: SimpleNamespace(lock_payload=lock),
    )
    config = SimpleNamespace(
        artifact_root=(tmp_path / "artifact").resolve(),
        contract_hash=_HASH,
        runtime={"scratch_preference": [str(scratch), "artifact_parent"]},
    )
    runner_runtime.cleanup_validated_local_stage(
        config, canonical_source=canonical
    )
    assert not owned.exists()
    assert unrelated.is_dir()
