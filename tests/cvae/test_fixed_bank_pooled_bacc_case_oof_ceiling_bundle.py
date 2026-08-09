from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    assert_closed_world,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.persistence import (
    persist_and_validate_preevaluation_seals,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.runner import (
    DecisionPhaseProducts,
    EvaluationPhaseProducts,
    FixedBankPooledBaccRunnerDependencies,
    PriorPhaseProducts,
    run_fixed_bank_pooled_bacc_case_oof_ceiling,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.validation import (
    _assert_exact_mapping,
    _require_recomputed_surface_payloads,
    _validate_excluded_control_members,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.core_hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json
from midogpp_thesis.cvae.diagnostics.fixed_bank_pooled_bacc_case_oof_ceiling.reports import (
    publication_decision_payload,
    run_state_payload,
)


def test_required_bundle_persists_sufficient_statistics_but_no_case_bacc() -> None:
    assert "tables/loco_case_action_sufficient_statistics.csv" in REQUIRED_FILES
    assert "tables/fold_support_case_action_sufficient_statistics.csv" in REQUIRED_FILES
    assert "tables/oof_evaluation_case_action_sufficient_statistics.csv" in REQUIRED_FILES
    assert "tables/oof_case_metrics.csv" not in REQUIRED_FILES
    assert "arrays/permutation_null_actions.npy" in REQUIRED_FILES


def test_content_index_distinguishes_diagnostic_actions_from_deployable_capability(
    tmp_path: Path,
) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode())
    payload = write_content_index(tmp_path, config_contract_hash="contract")
    assert payload["diagnostic_action_rows_present"] is True
    assert payload["deployable_policy_or_action_capability_present"] is False


def test_publication_report_uses_the_frozen_v2_claim_role() -> None:
    payload = publication_decision_payload({"scientific_result_hash": "a" * 64})
    assert (
        payload["claim_role"]
        == "pooled_bacc_known_bank_case_oof_information_ceiling"
    )
    assert payload["decision"] == "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    assert payload["policy_update_authorized"] is False


def test_incomplete_closed_world_allows_only_v2_owned_runtime_checkpoints(
    tmp_path: Path,
) -> None:
    for member in (
        "checkpoints/frozen_source_streams/source_0_train_17.json",
        "checkpoints/label_free_action_predictions/target_scratch.json",
        "checkpoints/label_free_action_predictions/tasks/target_0_train_17_generation_17.json",
    ):
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("checkpoint", encoding="utf-8")
    assert_closed_world(tmp_path, allow_incomplete=True)
    rogue = tmp_path / "checkpoints/v1_label_aware/rogue.json"
    rogue.parent.mkdir(parents=True)
    rogue.write_text("rogue", encoding="utf-8")
    with pytest.raises(ProtocolError, match="extras"):
        assert_closed_world(tmp_path, allow_incomplete=True)

    rogue.unlink()
    rogue_inside = (
        tmp_path
        / "checkpoints/label_free_action_predictions/tasks/target_0_train_17_generation_17.tmp"
    )
    rogue_inside.write_text("rogue", encoding="utf-8")
    with pytest.raises(ProtocolError, match="extras"):
        assert_closed_world(tmp_path, allow_incomplete=True)


def test_content_index_detects_tamper_before_semantic_validation(tmp_path: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode())
    write_content_index(tmp_path, config_contract_hash="contract")
    assert validate_content_index(
        tmp_path, config_contract_hash="contract"
    )["closed_world"] is True
    (tmp_path / "manifests/ceiling_evaluation.json").write_text(
        "coherent-looking tamper", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="member drifted"):
        validate_content_index(tmp_path, config_contract_hash="contract")


def test_content_index_rejects_coherently_hashed_extra_field(tmp_path: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode())
    payload = dict(write_content_index(tmp_path, config_contract_hash="contract"))
    payload["policy_update_authorized"] = True
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    payload["content_hash"] = canonical_hash(unhashed)
    # content-index hashing uses canonical compact JSON SHA-256, same byte grammar
    # as canonical_hash for a JSON-only mapping.
    atomic_json(tmp_path / "manifests/content_index.json", payload)
    with pytest.raises(ProtocolError, match="header drifted"):
        validate_content_index(tmp_path, config_contract_hash="contract")


def test_scientific_wrapper_rejects_coherent_extra_authorization_field() -> None:
    expected = {
        "schema_version": "fixed_bank_pooled_bacc_permutation_decision_plan_v2",
        "plan_hash": "a" * 64,
    }
    observed = {**expected, "policy_update_authorized": True}
    with pytest.raises(ProtocolError, match="closed schema or payload drifted"):
        _assert_exact_mapping(observed, expected, role="null manifest")


def test_coherent_sufficient_statistic_rehash_cannot_replace_canonical_rebuild() -> None:
    canonical = {
        "rows": [
            {
                "n_positive": 2,
                "true_positive": 1,
                "n_negative": 3,
                "true_negative": 2,
            }
        ],
        "statistics_surface_hash": "canonical",
    }
    recomputed = SimpleNamespace(to_payload=lambda: canonical)
    tampered = {
        "rows": [
            {
                "n_positive": 2,
                "true_positive": 2,
                "n_negative": 3,
                "true_negative": 2,
            }
        ]
    }
    tampered["statistics_surface_hash"] = canonical_hash(tampered)
    with pytest.raises(ProtocolError, match="canonical labels and probabilities"):
        _require_recomputed_surface_payloads(
            (recomputed,), (tampered,), role="support"
        )


def test_complete_excluded_control_members_are_exact_and_tamper_evident(
    tmp_path: Path,
) -> None:
    expected = {"schema_version": "validation", "status": "PASS"}
    atomic_json(
        tmp_path / "reports/run_state.json",
        run_state_payload("COMPLETE", "COMPLETE"),
    )
    atomic_json(tmp_path / "reports/validation_report.json", expected)
    _validate_excluded_control_members(tmp_path, expected)

    atomic_json(
        tmp_path / "reports/run_state.json",
        run_state_payload("COMPLETE", "tampered-phase"),
    )
    with pytest.raises(ProtocolError, match="run-state or validation report drifted"):
        _validate_excluded_control_members(tmp_path, expected)


def test_completed_validation_report_tamper_is_rejected(tmp_path: Path) -> None:
    expected = {"schema_version": "validation", "status": "PASS"}
    atomic_json(
        tmp_path / "reports/run_state.json",
        run_state_payload("COMPLETE", "COMPLETE"),
    )
    atomic_json(
        tmp_path / "reports/validation_report.json",
        {**expected, "status": "COHERENT_TAMPER"},
    )
    with pytest.raises(ProtocolError, match="run-state or validation report drifted"):
        _validate_excluded_control_members(tmp_path, expected)


def test_first_validation_pass_allows_only_exact_running_phase(tmp_path: Path) -> None:
    expected = {"status": "PASS"}
    atomic_json(
        tmp_path / "reports/run_state.json",
        run_state_payload("RUNNING", "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"),
    )
    _validate_excluded_control_members(tmp_path, expected)
    atomic_json(
        tmp_path / "reports/run_state.json",
        run_state_payload("RUNNING", "FORTY_FIVE_SUPPORT_POSTERIORS_AND_NULL_ACTIONS"),
    )
    with pytest.raises(ProtocolError, match="exact running phase"):
        _validate_excluded_control_members(tmp_path, expected)


def test_null_action_persistence_is_compact_and_nonrepairing(tmp_path: Path) -> None:
    row = SimpleNamespace(
        to_payload=lambda: {
            "target_center": "0",
            "case_id": "case",
            "action_id": "B",
            "n_positive": 1,
            "true_positive": 1,
            "n_negative": 1,
            "true_negative": 1,
            "per_case_bacc_stored": False,
        }
    )
    surfaces = tuple(
        SimpleNamespace(
            rows=(row,),
            label_scope=f"scope-{index}",
            statistics_surface_hash="a" * 64,
            prerequisite_seal_hash="b" * 64,
            to_payload=lambda index=index: {
                "surface": index,
                "rows": [row.to_payload()],
            },
        )
        for index in range(45)
    )
    posteriors = tuple(
        SimpleNamespace(
            to_payload=lambda index=index: {
                "target_center": str(index // 5),
                "fold_ordinal": index % 5,
                "posterior_hash": "c" * 64,
            }
        )
        for index in range(45)
    )
    decisions = tuple(
        SimpleNamespace(
            to_payload=lambda index=index: {
                "target_center": str(index // 5),
                "fold_ordinal": index % 5,
                "decision_hash": "d" * 64,
            }
        )
        for index in range(45)
    )
    decision_seal = SimpleNamespace(
        to_payload=lambda: {
            "decision_seal_hash": "e" * 64,
            "fold_decision_count": 45,
        }
    )
    permutation = SimpleNamespace(
        action_codes=np.zeros((10_000, 45), dtype=np.uint8),
        to_payload=lambda: {
            "plan_hash": "f" * 64,
            "permutation_decision_seal_hash": "f" * 64,
            "sealed_before_evaluation_labels": True,
            "evaluation_labels_used_to_generate_actions": False,
            "baseline_action_permuted": False,
            "candidate_multiset_preserved_per_case": True,
            "evaluation_utility_used_for_permutation_tie_break": False,
        },
    )
    persist_and_validate_preevaluation_seals(
        tmp_path,
        support_surfaces=surfaces,
        posteriors=posteriors,
        decisions=decisions,
        decision_seal=decision_seal,
        permutation_seal=permutation,
        config_contract_hash="contract",
    )
    path = tmp_path / "arrays/permutation_null_actions.npy"
    tampered = np.load(path, allow_pickle=False)
    tampered[0, 0] = np.uint8(1)
    with path.open("wb") as handle:
        np.save(handle, tampered, allow_pickle=False)
    with pytest.raises(ProtocolError, match="differs"):
        persist_and_validate_preevaluation_seals(
            tmp_path,
            support_surfaces=surfaces,
            posteriors=posteriors,
            decisions=decisions,
            decision_seal=decision_seal,
            permutation_seal=permutation,
            config_contract_hash="contract",
        )


def test_runner_opens_evaluation_only_after_prior_observed_and_null_seals(
    tmp_path: Path,
) -> None:
    (tmp_path / "provenance").mkdir(parents=True)
    (tmp_path / "config.resolved.yaml").write_text("config", encoding="utf-8")
    (tmp_path / "provenance/input_artifacts.json").write_text("{}", encoding="utf-8")
    order: list[str] = []
    hash64 = "a" * 64
    frame = SimpleNamespace(cache_binding_hash="binding")
    partition = SimpleNamespace(partition_hash=hash64, folds=())
    source = SimpleNamespace(records=tuple(range(81)), lock_hash=hash64)
    prediction = SimpleNamespace(
        seal_hash=hash64, store=SimpleNamespace(cells=tuple(range(729)))
    )
    probabilities = SimpleNamespace(surface_hash=hash64)
    priors = tuple(
        SimpleNamespace(target_center=center, prior_hash=hash64)
        for center in ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    )
    decisions = tuple(range(45))
    decision_seal = SimpleNamespace(decision_seal_hash=hash64)
    permutation = SimpleNamespace(action_codes=np.zeros((10_000, 45), dtype=np.uint8))

    class Manager:
        def record_loco_prior_seal(self, *_args: object) -> None:
            assert "prior_durable" in order

        def record_preevaluation_seals(self, *_args: object, **_kwargs: object) -> None:
            assert "preeval_durable" in order
            order.append("preeval_recorded")

        def open_oof_evaluation_labels(self) -> tuple[object, ...]:
            assert order.index("prior_durable") < order.index("preeval_durable")
            order.append("evaluation_opened")
            return ()

        def access_report(self) -> dict[str, object]:
            return {"evaluation_labels_opened": True}

    def persist_probabilities(root: Path, **_kwargs: object) -> None:
        atomic_json(
            root / "manifests/sealed_probability_surface.json",
            {"surface_hash": hash64},
        )

    def persist_priors(*_args: object, **_kwargs: object) -> None:
        order.append("prior_durable")

    def fit_decisions(**_kwargs: object) -> DecisionPhaseProducts:
        assert "prior_durable" in order
        return DecisionPhaseProducts(
            support_surfaces=tuple(range(45)),
            posteriors=tuple(range(45)),
            decisions=decisions,
            decision_seal=decision_seal,
            permutation_seal=permutation,
        )

    def persist_preeval(*_args: object, **_kwargs: object) -> dict[str, object]:
        order.append("preeval_durable")
        return {
            "permutation_decision_seal_hash": hash64,
            "null_action_count": 450_000,
        }

    def stop_at_index(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("stop-after-order-audit")

    config = SimpleNamespace(
        artifact_root=tmp_path,
        expert_bank_root=tmp_path / "expert",
        generation_lock_root=tmp_path / "generation",
        test_cache_root=tmp_path / "cache",
        test_manifest_path=tmp_path / "manifest.csv",
        test_consumption_ledger_path=tmp_path / "ledger.json",
        ledger_amendment_path=tmp_path / "amendment.json",
        runtime={"scratch_preference": ["/data/local/v2", "artifact_parent"]},
        contract_hash="contract",
        input_artifact_ids=(),
        protocol={"centers": []},
        global_prior={},
        posterior={},
        decision={},
        evaluation={"permutation_count": 10_000},
    )
    deps = FixedBankPooledBaccRunnerDependencies(
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
        fit_loco_priors=lambda **_kwargs: PriorPhaseProducts(
            statistic_surfaces=tuple(range(9)), priors=priors
        ),
        persist_loco_priors=persist_priors,
        fit_fold_decisions=fit_decisions,
        persist_preevaluation=persist_preeval,
        evaluate_sealed_decisions=lambda **_kwargs: EvaluationPhaseProducts(
            statistics={}, evaluation={"scientific_result_hash": hash64}
        ),
        persist_postseal=lambda *_args, **_kwargs: order.append("postseal"),
        write_index=stop_at_index,
    )
    with pytest.raises(RuntimeError, match="stop-after-order-audit"):
        run_fixed_bank_pooled_bacc_case_oof_ceiling(
            config, artifact_root=tmp_path, dependencies=deps
        )
    assert order[:4] == [
        "prior_durable",
        "preeval_durable",
        "preeval_recorded",
        "evaluation_opened",
    ]


def test_complete_fast_path_validates_without_refitting(tmp_path: Path) -> None:
    for member in REQUIRED_FILES:
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bytes:{member}".encode())
    atomic_json(tmp_path / "reports/run_state.json", {"status": "COMPLETE"})
    calls: list[str] = []

    def validate(root: Path, *, config: object) -> dict[str, object]:
        calls.append("validate")
        return {"status": "PASS"}

    def forbidden(*_args: object, **_kwargs: object) -> object:
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
    deps = FixedBankPooledBaccRunnerDependencies(
        validate_bundle=validate,
        materialize_source=forbidden,
        materialize_predictions=forbidden,
    )
    assert run_fixed_bank_pooled_bacc_case_oof_ceiling(
        config, artifact_root=tmp_path, dependencies=deps
    ) == tmp_path
    assert calls == ["validate"]
