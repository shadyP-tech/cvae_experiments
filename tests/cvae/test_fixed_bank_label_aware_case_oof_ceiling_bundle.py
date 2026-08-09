from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.runner import (
    DecisionPhaseProducts,
    FixedBankLabelAwareRunnerDependencies,
    run_fixed_bank_label_aware_case_oof_ceiling,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_label_aware_case_oof_ceiling.persistence import (
    persist_and_validate_loco_prior_seals,
    persist_and_validate_preevaluation_seals,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


def test_incomplete_closed_world_allows_only_exact_owned_checkpoint_namespaces(
    tmp_path: Path,
) -> None:
    for member in (
        "checkpoints/frozen_source_streams/source_0.json",
        "checkpoints/label_free_action_predictions/tasks/target_0.json",
    ):
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint", encoding="utf-8")
    assert_closed_world(tmp_path, allow_incomplete=True)

    rogue = tmp_path / "checkpoints/unowned/rogue.json"
    rogue.parent.mkdir(parents=True)
    rogue.write_text("rogue", encoding="utf-8")
    with pytest.raises(ProtocolError, match="extras"):
        assert_closed_world(tmp_path, allow_incomplete=True)


def test_complete_closed_world_rejects_even_owned_checkpoints(tmp_path: Path) -> None:
    _write_required_members(tmp_path)
    checkpoint = tmp_path / "checkpoints/frozen_source_streams/source_0.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")
    with pytest.raises(ProtocolError, match="extras"):
        assert_closed_world(tmp_path, allow_incomplete=False)


def test_content_index_detects_scientific_member_tamper_before_semantic_reads(
    tmp_path: Path,
) -> None:
    _write_content_members(tmp_path)
    write_content_index(tmp_path, config_contract_hash="contract")
    assert validate_content_index(tmp_path, config_contract_hash="contract")["closed_world"] is True
    target = tmp_path / "manifests/ceiling_evaluation.json"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(ProtocolError, match="member drifted"):
        validate_content_index(tmp_path, config_contract_hash="contract")


def test_complete_fast_path_validates_without_refitting(tmp_path: Path) -> None:
    _write_required_members(tmp_path)
    atomic_json(tmp_path / "reports/run_state.json", {"status": "COMPLETE"})
    calls: list[str] = []

    def validate(root: Path, *, config: object) -> dict[str, object]:
        calls.append("validate")
        assert root == tmp_path
        return {"status": "PASS"}

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("COMPLETE fast path attempted scientific work")

    config = SimpleNamespace(
        artifact_root=tmp_path,
        expert_bank_root=tmp_path / "expert",
        generation_lock_root=tmp_path / "generation",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "ledger.json",
        ledger_amendment_path=tmp_path / "amendment.json",
    )
    deps = FixedBankLabelAwareRunnerDependencies(
        validate_bundle=validate,
        materialize_source=forbidden,
        materialize_predictions=forbidden,
    )
    assert run_fixed_bank_label_aware_case_oof_ceiling(
        config, artifact_root=tmp_path, dependencies=deps
    ) == tmp_path
    assert calls == ["validate"]


def test_required_bundle_contains_compact_pre_evaluation_permutation_seal() -> None:
    assert "arrays/permutation_null_actions.npy" in REQUIRED_FILES
    assert "manifests/permutation_null_decision_seal.json" in REQUIRED_FILES
    assert "tables/action_selection_metrics.csv" in REQUIRED_FILES
    assert "tables/permutation_null_actions.csv" not in REQUIRED_FILES


def test_durable_loco_prior_seal_is_required_and_nonrepairing_before_support(
    tmp_path: Path,
) -> None:
    estimate = SimpleNamespace(
        action_id="1",
        other_center_count=7,
        other_center_case_count=100,
        shrunk_mean_gain_vs_b=0.1,
        standard_error=0.01,
        lower_confidence_bound=0.08,
        estimate_hash="e" * 64,
    )
    prior = SimpleNamespace(
        target_center="0",
        global_action_id="1",
        best_candidate_action_id="1",
        prior_hash="a" * 64,
        estimates=(estimate,),
        to_payload=lambda: {
            "target_center": "0",
            "prior_hash": "a" * 64,
            "H_labels_used_in_G_H": False,
        },
    )
    persist_and_validate_loco_prior_seals(tmp_path, (prior,))
    path = tmp_path / "manifests/loco_global_prior_seals.json"
    path.write_text('{"coherent":"tamper"}\n', encoding="utf-8")
    with pytest.raises(ProtocolError, match="will not be repaired"):
        persist_and_validate_loco_prior_seals(tmp_path, (prior,))


def test_durable_pre_eval_seals_and_compact_actions_are_nonrepairing(
    tmp_path: Path,
) -> None:
    estimate = SimpleNamespace(
        action_id="1",
        support_case_count=10,
        prior_mean_gain_vs_g=0.0,
        posterior_mean_gain_vs_g=0.1,
        standard_error=0.01,
        lower_confidence_bound=0.08,
        estimate_hash="e" * 64,
    )
    posterior = SimpleNamespace(
        target_center="0",
        fold_ordinal=0,
        posterior_hash="b" * 64,
        estimates=(estimate,),
        to_payload=lambda: {"target_center": "0", "fold_ordinal": 0, "posterior_hash": "b" * 64},
    )
    decision = SimpleNamespace(
        to_payload=lambda: {"target_center": "0", "fold_ordinal": 0, "decision_hash": "c" * 64}
    )
    decision_seal = SimpleNamespace(
        to_payload=lambda: {"decision_seal_hash": "d" * 64, "fold_decision_count": 45}
    )
    permutation = SimpleNamespace(
        action_ordinals=np.zeros((10_000, 45), dtype=np.uint8),
        to_payload=lambda: {
            "permutation_decision_seal_hash": "f" * 64,
            "partition_hash": "1" * 64,
            "decision_seal_hash": "d" * 64,
            "generated_before_evaluation_label_access": True,
            "permutation_decision_tie_break": (
                "lexicographic_action_id_no_evaluation_utility_access"
            ),
            "evaluation_utility_used_for_permutation_tie_break": False,
        },
    )
    persisted = persist_and_validate_preevaluation_seals(
        tmp_path,
        posteriors=(posterior,),
        decisions=(decision,),
        decision_seal=decision_seal,
        permutation_seal=permutation,
        config_contract_hash="contract",
    )
    assert persisted["permutation_decision_seal_hash"] == "f" * 64
    assert persisted["evaluation_utility_used_for_permutation_tie_break"] is False
    actions_path = tmp_path / "arrays/permutation_null_actions.npy"
    tampered = np.load(actions_path, allow_pickle=False)
    tampered[0, 0] = np.uint8(1)
    with actions_path.open("wb") as handle:
        np.save(handle, tampered, allow_pickle=False)
    with pytest.raises(ProtocolError, match="action array differs"):
        persist_and_validate_preevaluation_seals(
            tmp_path,
            posteriors=(posterior,),
            decisions=(decision,),
            decision_seal=decision_seal,
            permutation_seal=permutation,
            config_contract_hash="contract",
        )


def test_preevaluation_persistence_rejects_permutation_tie_boundary_tamper(
    tmp_path: Path,
) -> None:
    posterior = SimpleNamespace(
        target_center="0",
        fold_ordinal=0,
        posterior_hash="b" * 64,
        estimates=(),
        to_payload=lambda: {
            "target_center": "0",
            "fold_ordinal": 0,
            "posterior_hash": "b" * 64,
        },
    )
    decision = SimpleNamespace(
        to_payload=lambda: {
            "target_center": "0",
            "fold_ordinal": 0,
            "decision_hash": "c" * 64,
        }
    )
    decision_seal = SimpleNamespace(
        to_payload=lambda: {
            "decision_seal_hash": "d" * 64,
            "fold_decision_count": 45,
        }
    )
    tampered = SimpleNamespace(
        action_ordinals=np.zeros((10_000, 45), dtype=np.uint8),
        to_payload=lambda: {
            "permutation_decision_seal_hash": "f" * 64,
            "permutation_decision_tie_break": "evaluation_utility_maximizer",
            "evaluation_utility_used_for_permutation_tie_break": True,
        },
    )
    with pytest.raises(ProtocolError, match="tie boundary drifted"):
        persist_and_validate_preevaluation_seals(
            tmp_path,
            posteriors=(posterior,),
            decisions=(decision,),
            decision_seal=decision_seal,
            permutation_seal=tampered,
            config_contract_hash="contract",
        )


def test_runner_opens_evaluation_only_after_durable_prior_and_decision_seals(
    tmp_path: Path,
) -> None:
    (tmp_path / "provenance").mkdir(parents=True)
    (tmp_path / "config.resolved.yaml").write_text("config", encoding="utf-8")
    (tmp_path / "provenance/input_artifacts.json").write_text("{}", encoding="utf-8")
    order: list[str] = []
    hash64 = "a" * 64
    frame = SimpleNamespace(cache_binding_hash="binding")
    partition = SimpleNamespace(partition_hash=hash64, folds=())
    source = SimpleNamespace(records=tuple(range(81)), lock_hash="source")
    prediction = SimpleNamespace(
        seal_hash=hash64,
        store=SimpleNamespace(cells=tuple(range(729))),
    )
    probabilities = SimpleNamespace(surface_hash=hash64)
    prior = SimpleNamespace(target_center="0", prior_hash=hash64)
    decision_seal = SimpleNamespace(decision_seal_hash=hash64, decisions=tuple(range(45)))
    permutation = SimpleNamespace()

    class Manager:
        def record_loco_prior_seal(self, *args: object) -> None:
            assert "prior_durable" in order
            order.append("prior_recorded")

        def record_preevaluation_seals(self, *args: object, **kwargs: object) -> None:
            assert "preeval_durable" in order
            order.append("preeval_recorded")

        def open_oof_evaluation_labels(self) -> tuple[object, ...]:
            assert order.index("prior_durable") < order.index("preeval_durable")
            order.append("evaluation_opened")
            return ()

        def access_report(self) -> dict[str, object]:
            return {"evaluation_labels_opened": True}

    def persist_probabilities(root: Path, **kwargs: object) -> None:
        atomic_json(root / "manifests/sealed_probability_surface.json", {"surface_hash": hash64})

    def stop_after_postseal(*args: object, **kwargs: object) -> None:
        order.append("postseal")

    def stop_at_index(*args: object, **kwargs: object) -> object:
        raise RuntimeError("stop-after-order-audit")

    config = SimpleNamespace(
        artifact_root=tmp_path,
        expert_bank_root=tmp_path / "expert",
        generation_lock_root=tmp_path / "generation",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "ledger.json",
        ledger_amendment_path=tmp_path / "amendment.json",
        runtime={"scratch_preference": ["/data/local/x"], "generation_devices": ["cuda:0", "cuda:1"]},
        contract_hash="contract",
        input_artifact_ids=(),
        protocol={"centers": []},
        global_prior={},
        posterior={},
        decision={},
        evaluation={},
    )
    deps = FixedBankLabelAwareRunnerDependencies(
        validate_inputs=lambda *_: None,
        validate_workspace=lambda *_: {},
        validate_provenance=lambda *_: {},
        load_locks=lambda *_: SimpleNamespace(generation=object()),
        load_frame=lambda *_: frame,
        validate_firewall=lambda *_: {},
        build_partition=lambda *_args, **_kwargs: partition,
        persist_initial=lambda *_args, **_kwargs: None,
        preflight=lambda *_args, **_kwargs: {},
        materialize_source=lambda *_args, **_kwargs: source,
        stage_source=lambda value, **_kwargs: value,
        materialize_predictions=lambda *_args, **_kwargs: prediction,
        build_seed_rows=lambda *_: (),
        aggregate_probabilities=lambda *_: probabilities,
        persist_probabilities=persist_probabilities,
        build_label_manager=lambda *_args, **_kwargs: Manager(),
        fit_loco_priors=lambda **_kwargs: (prior,),
        persist_loco_priors=lambda *_args, **_kwargs: order.append("prior_durable"),
        fit_fold_decisions=lambda **_kwargs: DecisionPhaseProducts(
            posteriors=(), decisions=tuple(range(45)), decision_seal=decision_seal, permutation_seal=permutation
        ),
        persist_preevaluation=lambda *_args, **_kwargs: (
            order.append("preeval_durable")
            or {"permutation_decision_seal_hash": hash64}
        ),
        evaluate_sealed_decisions=lambda **_kwargs: order.append("evaluated") or {},
        persist_postseal=stop_after_postseal,
        write_index=stop_at_index,
    )
    with pytest.raises(RuntimeError, match="stop-after-order-audit"):
        run_fixed_bank_label_aware_case_oof_ceiling(
            config, artifact_root=tmp_path, dependencies=deps
        )
    assert order[:5] == [
        "prior_durable",
        "prior_recorded",
        "preeval_durable",
        "preeval_recorded",
        "evaluation_opened",
    ]


def _write_content_members(root: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode("utf-8"))


def _write_required_members(root: Path) -> None:
    for member in REQUIRED_FILES:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode("utf-8"))
