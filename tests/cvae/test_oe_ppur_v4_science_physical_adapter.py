from __future__ import annotations

import ast
from dataclasses import fields
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.action_compiler import (
    BasePredictionSurface,
    canonical_compiler_receipt,
    compile_action_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.candidate_pools import (
    build_final_outer_candidate_pool,
    build_held_center_candidate_pool,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.dto import (
    PrimitiveWorkerResult,
    PrimitiveWorkerTask,
    assert_pickle_round_trip,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.execution.services import (
    CanonicalRouterExecutionRequest,
    CanonicalScientificRouterService,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.folds import (
    build_outer_fold_plan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.identity import (
    CENTERS,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_PROBABILITY_MATRIX_SHAPE,
    EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
    EXPECTED_TEST_ROWS_BY_CENTER,
    SOURCE_CONTENT_LINEAGE_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.physical.actions import (
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.physical.cache_loader import (
    _CACHE_METADATA_FIELDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.physical.compiled_matrix import (
    assemble_compiled_probability_matrix,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.physical.frame import (
    LabelFreeTestFrame,
    TestRowIdentity as _TestRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.physical.runtime_config import (
    physical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.physical.topology import (
    WorkstationTopologyReceipt,
    project_workstation_topology,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.science.admission import (
    exact_p_fail_closed_reason,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.science.outer_orchestration import (
    OuterScienceResult,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.science.target_decision import (
    OuterTargetDecisionInput,
    TargetRowBinding,
    assemble_exact_218_case_decisions,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.science.target_inventory import (
    CANONICAL_TARGET_CASE_INVENTORY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.service_factory import (
    CanonicalScientificServiceFactory,
    prepare_canonical_scientific_service_factory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.source_bundle.constants import (
    EXPECTED_HELD_ACTION_LIBRARY_SHA256,
    EXPECTED_HELD_MASS_POLICY_RECEIPT_SHA256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.source_supervision import (
    SourceSupervisionContractReceipt,
)


SHA = "a" * 64
PACKAGE = (
    Path(__file__).parents[2]
    / "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4"
)


def _inventory() -> tuple[tuple[str, str], ...]:
    return tuple((f"expert-{center}", center) for center in CENTERS)


def _contract() -> SourceSupervisionContractReceipt:
    compiler = canonical_compiler_receipt()
    return SourceSupervisionContractReceipt(
        compiler_receipt_hash=compiler.receipt_hash,
        producer_source_seal_sha256=EXPECTED_SOURCE_PRODUCER_SEAL_SHA256,
        held_action_library_sha256=EXPECTED_HELD_ACTION_LIBRARY_SHA256,
        held_mass_policy_receipt_sha256=(
            EXPECTED_HELD_MASS_POLICY_RECEIPT_SHA256
        ),
    )


def _pools(center: str):
    compiler = canonical_compiler_receipt()
    contract = _contract()
    held = tuple(
        build_held_center_candidate_pool(
            outer_target_center=center,
            held_center=query,
            all_center_ids=CENTERS,
            expert_inventory=_inventory(),
            bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
            source_supervision_contract_hash=contract.contract_hash,
            compiler=compiler,
        )
        for query in CENTERS
        if query != center
    )
    final = build_final_outer_candidate_pool(
        outer_target_center=center,
        all_center_ids=CENTERS,
        expert_inventory=_inventory(),
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        source_supervision_contract_hash=contract.contract_hash,
        compiler=compiler,
    )
    return compiler, contract, held, final


def test_v4_science_is_owned_and_has_no_predecessor_imports() -> None:
    inspected = (
        PACKAGE / "action_compiler.py",
        PACKAGE / "candidate_pools.py",
        PACKAGE / "feature_engineering.py",
        PACKAGE / "folds.py",
        PACKAGE / "service_factory.py",
        PACKAGE / "source_supervision.py",
        *sorted((PACKAGE / "science").glob("*.py")),
        *sorted((PACKAGE / "physical").glob("*.py")),
        *sorted((PACKAGE / "source_bundle").glob("*.py")),
        PACKAGE / "execution/dto.py",
        PACKAGE / "execution/services.py",
    )
    for path in inspected:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = tuple(
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ) + tuple(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any("router_v3" in name for name in imported), path


def test_factory_requires_type_gated_run_admission_without_boolean_bypass() -> None:
    parameters = inspect.signature(
        prepare_canonical_scientific_service_factory
    ).parameters
    assert tuple(parameters) == (
        "config_bundle",
        "source_seal",
        "source_surface",
        "admission",
    )
    assert "execution_authorized" not in parameters
    assert "execution_authorized" not in inspect.signature(
        CanonicalScientificServiceFactory
    ).parameters
    assert "execution_authorized" not in inspect.signature(
        CanonicalScientificRouterService
    ).parameters


def test_historical_source_receipt_algebra_is_exact_and_authority_free() -> None:
    compiler = canonical_compiler_receipt()
    contract = _contract()
    assert (
        compiler.receipt_hash
        == "1a7ef0e577c5bcbc741ebbddad7a1a0f52f26890dbd0563935084290a1b65a51"
    )
    assert contract.artifact_id == SOURCE_CONTENT_LINEAGE_ARTIFACT_ID
    payload = contract.manifest_payload(
        compiler_recomputation_receipt_sha256="b" * 64
    )
    assert payload["target_rows_present"] is False
    assert payload["target_labels_used"] is False
    assert "authority" not in contract.__dataclass_fields__


def test_every_outer_and_inner_pool_excludes_target_h_exactly() -> None:
    compiler, contract, held, final = _pools("0")
    assert "0" not in final.candidate_center_ids
    assert tuple(pool.held_center for pool in held) == tuple(
        center for center in CENTERS if center != "0"
    )
    for pool in held:
        assert "0" not in pool.candidate_center_ids
        assert pool.held_center not in pool.candidate_center_ids
    cases = {center: (f"{center}-a", f"{center}-b") for center in CENTERS}
    plan = build_outer_fold_plan(
        outer_target_center="0",
        cases_by_center=cases,
        held_pool_receipts=held,
        final_pool_receipt=final,
        compiler=compiler,
        source_supervision_contract_hash=contract.contract_hash,
    )
    assert all(scope.H == "0" and scope.J != "0" for scope in plan.scopes)
    assert all(scope.H == "0" and scope.J != "0" for scope in plan.case_crossfit_scopes)


def test_physical_adapter_is_exact_two_gpu_four_spawn_cpu_and_label_free() -> None:
    runtime = physical_runtime_payload()
    assert runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert runtime["classifier_workers"] == 4
    assert runtime["classifier_threads_per_worker"] == 1
    assert runtime["multiprocessing_start_method"] == "spawn"
    assert _CACHE_METADATA_FIELDS == {
        "evaluation_row_id",
        "contract_row_index",
        "case_id",
        "center",
        "split",
    }
    library = action_library_by_target()
    for target, actions in library.items():
        assert len(actions) == 10
        assert all(target not in action.counts_by_class["0"] for action in actions)
        assert all(action.to_payload()["target_expert_excluded"] for action in actions)


def test_scientific_execution_contract_has_no_target_label_or_path_input() -> None:
    request_fields = tuple(field.name for field in fields(CanonicalRouterExecutionRequest))
    frame_fields = tuple(field.name for field in fields(LabelFreeTestFrame))
    row_fields = tuple(field.name for field in fields(_TestRowIdentity))
    assert request_fields == (
        "frame",
        "physical_inputs",
        "upstream_receipt_hash",
        "workstation_receipt",
        "request_hash",
    )
    assert not any("label" in name or "path" in name for name in request_fields)
    assert not any("label" in name or "path" in name for name in frame_fields)
    assert not any("label" in name or "path" in name for name in row_fields)
    assert "label" not in inspect.signature(
        CanonicalScientificRouterService.execute_label_free
    ).parameters


def test_workstation_projection_binds_richer_receipt_without_transporting_it() -> None:
    upstream = SimpleNamespace(
        receipt_hash="1" * 64,
        to_payload=lambda: {
            "hostname": "delli2",
            "gpu_rows": [
                {
                    "index": "0",
                    "name": "NVIDIA RTX A5000",
                    "memory_total_mib": 24_576,
                },
                {
                    "index": "1",
                    "name": "NVIDIA RTX A5000",
                    "memory_total_mib": 24_576,
                },
            ],
            "cpu_count": 32,
            "cpu_worker_count": 4,
            "blas_threads_per_cpu_worker": 1,
        },
    )
    receipt = project_workstation_topology(upstream)
    assert type(receipt) is WorkstationTopologyReceipt
    assert receipt.upstream_receipt_hash == upstream.receipt_hash
    assert receipt.to_payload()["spawn_cpu_worker_count"] == 4
    assert receipt.to_payload()["persistent_gpu_worker_count"] == 2


def test_worker_dtos_are_flat_primitives_and_pickle_safe() -> None:
    task = PrimitiveWorkerTask(
        task_id="H0-fold0",
        outer_center_id="0",
        inner_fold_id=0,
        row_start=0,
        row_stop=12,
        source_training_surface_hash="2" * 64,
        candidate_pool_receipt_hash="3" * 64,
        compiled_action_surface_hash="4" * 64,
        random_seed=23,
    )
    result = PrimitiveWorkerResult(
        task_id=task.task_id,
        outer_center_id=task.outer_center_id,
        inner_fold_id=task.inner_fold_id,
        worker_pid=123,
        model_receipt_hash="5" * 64,
        source_ordering_receipt_hash="6" * 64,
        ordered_case_ids=("case-1",),
        ordered_action_ids=("P_PROTECTED",),
        ordered_scores=(0.25,),
        exact_p_required=False,
        failure_reason=None,
    )
    assert assert_pickle_round_trip(task) == task
    assert assert_pickle_round_trip(result) == result
    for value in (task, result):
        assert all(
            item is None
            or isinstance(item, (str, int, float, bool, tuple))
            for item in (getattr(value, field.name) for field in fields(value))
        )


def test_exact_9928_by_7_matrix_and_218_case_fail_closed_ledger() -> None:
    surfaces = []
    decisions = []
    cursor = 0
    inventory_by_center = {
        center: tuple(
            case
            for value, case in CANONICAL_TARGET_CASE_INVENTORY
            if value == center
        )
        for center in CENTERS
    }
    for center, count in EXPECTED_TEST_ROWS_BY_CENTER:
        compiler, _contract_row, _held, final = _pools(center)
        row_ids = tuple(f"{center}-row-{index:05d}" for index in range(count))
        base = BasePredictionSurface(
            outer_target_center=center,
            evaluated_center=center,
            row_ids=row_ids,
            equal_union_probabilities=(0.40,) * count,
            union_probabilities=(0.60,) * count,
            expert_probabilities=tuple(
                (candidate, (0.55,) * count)
                for candidate in final.candidate_center_ids
            ),
            candidate_pool_receipt_hash=final.receipt_hash,
        )
        surface = compile_action_surface(base, candidate_pool=final, compiler=compiler)
        surfaces.append(surface)
        outer = OuterScienceResult(
            outer_target_center=center,
            plan_hash=SHA,
            source_surface_lineage_hash="7" * 64,
            admitted=False,
            row_posterior_model=None,
            row_oof_predictions=(),
            pairwise_model=None,
            uncertainty_calibration=None,
            admission=None,
            fallback=exact_p_fail_closed_reason(
                outer_target_center=center,
                reason_code="source_model_unavailable",
                evidence_hash="8" * 64,
            ),
        )
        cases = inventory_by_center[center]
        bindings = tuple(
            TargetRowBinding(
                index,
                row_id,
                center,
                cases[index % len(cases)],
            )
            for index, row_id in enumerate(row_ids)
        )
        decisions.append(OuterTargetDecisionInput(outer, surface, bindings, final))
        cursor += count
    matrix = assemble_compiled_probability_matrix(tuple(surfaces))
    ledger = assemble_exact_218_case_decisions(
        tuple(decisions),
        expected_case_inventory=CANONICAL_TARGET_CASE_INVENTORY,
    )
    assert cursor == 9_928
    assert matrix.values.shape == EXPECTED_PROBABILITY_MATRIX_SHAPE
    assert matrix.values.dtype == np.dtype("<f4")
    assert matrix.values.flags.c_contiguous
    assert matrix.values.flags.writeable is False
    assert len(ledger.decisions) == 218
    assert ledger.exact_p_count == 218
    assert all(row.selected_action_id == "P_PROTECTED" for row in ledger.decisions)
