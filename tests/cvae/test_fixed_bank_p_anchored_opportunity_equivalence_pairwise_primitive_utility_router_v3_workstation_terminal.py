from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.dto import (
    PrimitiveWorkerResult,
    PrimitiveWorkerTask,
    assert_pickle_round_trip,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal import (
    issue_terminal_aggregate_capability,
    seal_guarded_preterminal_boundary,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal.contracts import (
    ALLOWED_AGGREGATE_METRICS,
    AggregateOnlyTerminalReceipt,
    _ATTESTATION_TOKEN,
    _issue_artifact_only_preterminal_attestation,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal.evaluator import (
    AggregateOnlyTerminalEvaluator,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal.label_reader import (
    CaseRoutingDiagnostic,
    _VIEW_TOKEN,
    _build_manager_owned_manifest_label_reader,
    _seal_manager_owned_terminal_input,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workstation import (
    CPU_WORKER_ENVIRONMENT,
    preflight_workstation,
    validate_workstation_observation,
    workstation_plan_payload,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_workstation_requires_two_a5000_and_exact_spawn_cpu_contract() -> None:
    receipt = validate_workstation_observation(
        {
            "gpu_count": 2,
            "gpu_names": ["NVIDIA RTX A5000", "NVIDIA RTX A5000"],
            "cpu_count": 32,
            "start_method": "spawn",
        },
        dto_pickle_round_trip_validated=True,
    )
    payload = receipt.to_payload()
    plan = workstation_plan_payload()
    assert payload["persistent_gpu_worker_count"] == 2
    assert payload["spawn_cpu_worker_count"] == 4
    assert payload["cuda_visible_to_cpu_workers"] is False
    assert payload["cpu_worker_environment"] == CPU_WORKER_ENVIRONMENT
    assert plan["prediction_matrix_dtype"] == "<f4"
    assert plan["reduction_dtype"] == "<f8"

    with pytest.raises(ProtocolError, match="workstation plan drifted"):
        validate_workstation_observation(
            {
                "gpu_count": 2,
                "gpu_names": ["RTX 4090", "RTX 4090"],
                "cpu_count": 32,
                "start_method": "spawn",
            },
            dto_pickle_round_trip_validated=True,
        )


def test_live_preflight_performs_exact_primitive_dto_pickle_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.workstation as workstation_module

    monkeypatch.setattr(
        workstation_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="NVIDIA RTX A5000\nNVIDIA RTX A5000\n"
        ),
    )
    monkeypatch.setattr(workstation_module.os, "cpu_count", lambda: 32)
    receipt = preflight_workstation()

    assert receipt.dto_pickle_round_trip_validated is True
    assert receipt.gpu_names == ("NVIDIA RTX A5000", "NVIDIA RTX A5000")


def test_primitive_worker_dtos_are_spawn_pickle_safe() -> None:
    task = PrimitiveWorkerTask(
        task_id="H0-fold0",
        outer_center_id="0",
        inner_fold_id=0,
        row_start=0,
        row_stop=12,
        source_training_surface_hash="a" * 64,
        candidate_pool_receipt_hash="b" * 64,
        compiled_action_surface_hash="c" * 64,
        random_seed=7,
    )
    result = PrimitiveWorkerResult(
        task_id=task.task_id,
        outer_center_id=task.outer_center_id,
        inner_fold_id=task.inner_fold_id,
        worker_pid=123,
        model_receipt_hash="d" * 64,
        source_ordering_receipt_hash="e" * 64,
        ordered_case_ids=("case-1",),
        ordered_action_ids=("P_PROTECTED",),
        ordered_scores=(0.25,),
        exact_p_required=False,
        failure_reason=None,
    )
    assert assert_pickle_round_trip(task) == task
    assert assert_pickle_round_trip(result) == result


def _boundary():
    ledger = "a" * 64
    attestations = tuple(
        _issue_artifact_only_preterminal_attestation(
            sealed_ledger_receipt_hash=ledger,
            artifact_file_sha256="b" * 64,
            artifact_file_identity_sha256="c" * 64,
            validator_runtime_sha256="d" * 64,
            process_pid=100 + index,
            _validator_token=_ATTESTATION_TOKEN,
        )
        for index in range(2)
    )
    return seal_guarded_preterminal_boundary(
        seven_input_contract_hash="1" * 64,
        source_seal_hash="2" * 64,
        source_training_surface_receipt_hash="3" * 64,
        decision_ledger_receipt_hash=ledger,
        attestations=attestations,
        case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
        case_count=218,
        exact_p_fallback_count=109,
    )


def test_terminal_requires_two_distinct_attestations() -> None:
    ledger = "a" * 64
    attestation = _issue_artifact_only_preterminal_attestation(
        sealed_ledger_receipt_hash=ledger,
        artifact_file_sha256="b" * 64,
        artifact_file_identity_sha256="c" * 64,
        validator_runtime_sha256="d" * 64,
        process_pid=100,
        _validator_token=_ATTESTATION_TOKEN,
    )
    with pytest.raises(ProtocolError, match="two distinct artifact-only"):
        seal_guarded_preterminal_boundary(
            seven_input_contract_hash="1" * 64,
            source_seal_hash="2" * 64,
            source_training_surface_receipt_hash="3" * 64,
            decision_ledger_receipt_hash=ledger,
            attestations=(attestation, attestation),
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            case_count=218,
            exact_p_fallback_count=109,
        )


def test_concrete_terminal_evaluator_is_one_shot_and_aggregate_only() -> None:
    boundary = _boundary()
    row_case_ids = tuple(f"case-{index % 218:03d}" for index in range(9_928))
    labels = tuple(index % 2 for index in range(9_928))
    protected = tuple(0.8 if value else 0.2 for value in labels)
    selected = protected
    diagnostics = tuple(
        CaseRoutingDiagnostic(
            case_id=f"case-{index:03d}",
            selected_action_id=(
                "P_PROTECTED" if index < 109 else "B::zero_to_one"
            ),
            oracle_action_id="B::zero_to_one",
            spearman_rank_correlation=0.25,
            normalized_oracle_gap=0.1,
        )
        for index in range(218)
    )
    view = _seal_manager_owned_terminal_input(
        boundary,
        row_case_ids=row_case_ids,
        row_labels=labels,
        selected_probabilities=selected,
        protected_probabilities=protected,
        case_diagnostics=diagnostics,
        _manager_token=_VIEW_TOKEN,
    )
    reader = _build_manager_owned_manifest_label_reader(view)
    with pytest.raises(ProtocolError, match="evaluator bypassed manager"):
        AggregateOnlyTerminalEvaluator(reader)
    with pytest.raises(ProtocolError, match="capability inputs are untyped"):
        issue_terminal_aggregate_capability(boundary, reader=object())
    capability = issue_terminal_aggregate_capability(
        boundary,
        reader=reader,
    )
    receipt = capability.score_aggregates()
    payload = receipt.to_payload()

    assert receipt.evaluated_case_count == 218
    assert receipt.routed_case_count == 109
    assert set(payload["aggregate_metrics"]) >= {
        "selected_balanced_accuracy",
        "selected_brier_score",
        "selected_log_loss",
        "top1_oracle_agreement",
    }
    assert payload["raw_labels_present"] is False
    assert payload["per_row_values_present"] is False
    assert payload["per_case_values_present"] is False
    assert _reconstruct_persisted_aggregate_only_terminal_receipt(payload) == receipt
    tampered = dict(payload)
    tampered["raw_labels_present"] = True
    with pytest.raises(ProtocolError, match="exposed raw values"):
        _reconstruct_persisted_aggregate_only_terminal_receipt(tampered)
    with pytest.raises(ProtocolError, match="replayed"):
        capability.score_aggregates()
    with pytest.raises(TypeError):
        pickle.dumps(capability)
    with pytest.raises(TypeError):
        copy.copy(capability)


def test_terminal_receipt_cannot_be_injected_or_publicly_constructed() -> None:
    boundary = _boundary()
    with pytest.raises(ProtocolError, match="receipt bypassed manager"):
        AggregateOnlyTerminalReceipt(
            boundary_receipt_hash=boundary.receipt_hash,
            decision_ledger_receipt_hash=boundary.decision_ledger_receipt_hash,
            evaluated_case_count=218,
            routed_case_count=109,
            exact_p_fallback_count=109,
            aggregate_metrics=tuple(
                (metric, 0.0) for metric in ALLOWED_AGGREGATE_METRICS
            ),
        )

    import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal as terminal

    for unsafe_name in (
        "AggregateOnlyTerminalReceipt",
        "AggregateOnlyTerminalEvaluator",
        "AggregateOnlyLabelReader",
        "ManagerOwnedManifestLabelReader",
        "build_manager_owned_manifest_label_reader",
        "build_manager_owned_terminal_evaluator",
    ):
        assert not hasattr(terminal, unsafe_name)
