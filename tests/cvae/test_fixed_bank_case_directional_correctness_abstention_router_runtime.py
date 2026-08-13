from __future__ import annotations

import ast
import errno
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router import (
    execution_adapter,
    fresh_process_validation,
    validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.constants import (
    CENTERS,
    DIRECTION_IDS,
    a1_action_id,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.features import (
    build_label_free_case_candidate_features,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.hashing import (
    canonical_hash,
    canonical_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.held_case_plans import (
    HeldCasePlan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.persistence import (
    persist_validation_report,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.products import (
    BinaryLabel,
    DonorDirectionalPrior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_case_directional_correctness_abstention_router.runner_runtime import (
    RouteJobResult,
    assert_no_foreign_or_partial_state,
    execute_route_jobs,
    reject_existing_run_state,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    assert_runtime as assert_fixed_bank_a1_runtime,
)
from midogpp_thesis.cvae.runtime.frozen_source_streams import (
    _assert_runtime as assert_frozen_source_runtime,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics"
    / "fixed_bank_case_directional_correctness_abstention_router"
)

EXPECTED_REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "arrays/frozen_source_streams.npy",
    "arrays/fixed_bank_a1_action_probabilities.npz",
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/frozen_source_stream_index.json",
    "manifests/frozen_source_stream_lock.json",
    "manifests/fixed_bank_a1_prediction_index.json",
    "manifests/fixed_bank_a1_prediction_seal.json",
    "manifests/physical_prelabel_seal.json",
    "manifests/held_case_plan_seal.json",
    "manifests/held_case_feature_seal.json",
    "manifests/donor_prior_seal.json",
    "manifests/route_model_seal.json",
    "manifests/route_decision_seal.json",
    "manifests/aggregate_plan_decision_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/exact_nine_probability_index.csv",
    "tables/held_case_plans.csv",
    "tables/held_case_features.csv",
    "tables/support_response_counts.csv",
    "tables/donor_priors.csv",
    "tables/route_model_fits.csv",
    "tables/route_candidate_scores.csv",
    "tables/route_decisions.csv",
    "tables/method_predictions.csv",
    "tables/descriptive_method_predictions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_method_metrics.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "tables/router_identification_metrics.csv",
    "tables/feature_permutation_summary.csv",
    "reports/workstation_preflight.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

CONTENT_INDEX_EXCLUSIONS = frozenset(
    {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)

ATTESTATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "fresh_python_process_count",
        "independent_fresh_python_processes",
        "process_launches_sequential",
        "persisted_resolved_config_loaded_by_each_process",
        "full_scientific_reconstruction_called_by_each_process",
        "pending_validation_allowed",
        "cuda_visible_devices",
        "worker_thread_environment",
        "parent_process_id",
        "child_process_ids",
        "child_process_results",
        "subprocess_exit_codes",
        "reconstructed_check_payloads_exactly_equal",
        "reconstructed_check_hash",
        "validator_entrypoint",
        "attestation_hash",
    }
)

CHILD_RESULT_KEYS = frozenset(
    {
        "ordinal",
        "process_id",
        "exit_code",
        "reconstructed_check_hash",
        "result_payload_hash",
    }
)


def _write_content_members(root: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed::{member}\n".encode("utf-8"))


def _spawn_route_fixture() -> tuple[ExactNineProbabilitySurface, dict[str, object]]:
    target = "0"
    cases = ("support-a", "support-b", "held")
    rows = []
    for case_id in cases:
        for suffix, baseline in (("low", 0.4), ("high", 0.6)):
            for action_id in physical_action_ids(target):
                probability = baseline
                if action_id == a1_action_id("1") and suffix == "low":
                    probability = 0.6
                elif action_id == a1_action_id("2") and suffix == "high":
                    probability = 0.4
                rows.append(
                    ExactNineProbabilityRow(
                        target,
                        case_id,
                        f"{case_id}-{suffix}",
                        action_id,
                        (probability,) * 9,
                    )
                )
    surface = ExactNineProbabilitySurface(tuple(rows), "spawn-store-v1")
    plan = HeldCasePlan(
        target,
        "held",
        "held",
        ("support-a", "support-b"),
        ("held-low", "held-high"),
        surface.surface_hash,
    )
    labels = tuple(
        BinaryLabel(target, case_id, f"{case_id}-{suffix}", value, "route-support")
        for case_id, values in (("support-a", (1, 0)), ("support-b", (0, 1)))
        for (suffix, _), value in zip((("low", 0.4), ("high", 0.6)), values, strict=True)
    )
    priors = tuple(
        DonorDirectionalPrior(
            target,
            source,
            direction,
            tuple(center for center in CENTERS if center not in {target, source}),
            tuple(
                canonical_hash(
                    {"target": target, "source": source, "direction": direction, "query": query}
                )
                for query in CENTERS
                if query not in {target, source}
            ),
            0,
            1,
            0.0,
        )
        for source in candidate_sources(target)
        for direction in DIRECTION_IDS
    )
    return surface, {
        "plan": plan,
        "support_labels": labels,
        "donor_priors": priors,
        "route_features": build_label_free_case_candidate_features(surface),
    }


def _attested_checks(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[dict[str, object], list[tuple[tuple[str, ...], dict[str, object]]]]:
    expected = {
        "status": "PASS",
        "closed_world": True,
        "content_index_exact": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
    }
    parent_pid = os.getpid()
    worker_payloads = iter(
        (
            {"process_id": parent_pid + 10_001, "checks": expected},
            {"process_id": parent_pid + 10_002, "checks": expected},
        )
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(
        args: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((tuple(args), dict(kwargs)))
        payload = next(worker_payloads)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=canonical_json(payload).decode("utf-8") + "\n",
            stderr="",
        )

    monkeypatch.setattr(fresh_process_validation.subprocess, "run", fake_run)
    attested = fresh_process_validation.require_two_fresh_process_validations(
        root, expected_checks=expected
    )
    return dict(attested), calls


def _imported_modules(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return tuple(modules)


def test_required_files_and_content_index_claims_are_exact() -> None:
    assert REQUIRED_FILES == EXPECTED_REQUIRED_FILES
    assert len(REQUIRED_FILES) == 43
    assert CONTENT_INDEX_MEMBERS == tuple(
        member
        for member in EXPECTED_REQUIRED_FILES
        if member not in CONTENT_INDEX_EXCLUSIONS
    )
    assert len(CONTENT_INDEX_MEMBERS) == 40
    assert set(REQUIRED_FILES).difference(CONTENT_INDEX_MEMBERS) == (
        CONTENT_INDEX_EXCLUSIONS
    )


def test_content_index_binds_all_40_members_and_terminal_claims(tmp_path: Path) -> None:
    _write_content_members(tmp_path)
    payload = write_content_index(
        tmp_path,
        config_contract_hash="config-contract",
        protocol_contract_hash="protocol-contract",
    )

    assert payload["member_count"] == 40
    assert [row["member"] for row in payload["members"]] == list(
        CONTENT_INDEX_MEMBERS
    )
    assert payload["closed_world"] is True
    assert payload["raw_labels_persisted"] is False
    assert payload["image_or_sample_paths_persisted"] is False
    assert payload["terminal_checkpoint_persisted"] is False
    assert (
        payload["previous_stage90_artifact_checkpoint_or_scratch_consumed"]
        is False
    )
    assert payload["publication_decision"] == (
        "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
    )
    assert validate_content_index(
        tmp_path,
        config_contract_hash="config-contract",
        protocol_contract_hash="protocol-contract",
    ) == payload

    drifted = json.loads(json.dumps(payload))
    drifted["previous_stage90_artifact_checkpoint_or_scratch_consumed"] = True
    unsigned = {key: value for key, value in drifted.items() if key != "content_hash"}
    drifted["content_hash"] = canonical_hash(unsigned)
    atomic_json(tmp_path / "manifests/content_index.json", drifted)
    with pytest.raises(ProtocolError, match="content-index header drifted"):
        validate_content_index(
            tmp_path,
            config_contract_hash="config-contract",
            protocol_contract_hash="protocol-contract",
        )


def test_launch_only_state_is_allowed_but_any_product_or_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    for member in ("config.resolved.yaml", "provenance/input_artifacts.json"):
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("launch-only\n", encoding="utf-8")
    assert_no_foreign_or_partial_state(tmp_path)

    product = tmp_path / "tables/held_case_plans.csv"
    product.parent.mkdir(parents=True, exist_ok=True)
    product.write_text("partial\n", encoding="utf-8")
    with pytest.raises(
        ProtocolError,
        match="partial/cross-run state is forbidden|foreign directory",
    ):
        assert_no_foreign_or_partial_state(tmp_path)

    product.unlink()
    checkpoint = (
        tmp_path
        / "checkpoints/frozen_source_streams/source_0_train_17.json"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        ProtocolError,
        match="partial/cross-run state is forbidden|foreign directory",
    ):
        assert_no_foreign_or_partial_state(tmp_path)


@pytest.mark.parametrize("status", ("RUNNING", "FAILED", "COMPLETE"))
def test_every_existing_run_state_is_rejected(tmp_path: Path, status: str) -> None:
    atomic_json(
        tmp_path / "reports/run_state.json",
        {"status": status, "phase": "fixture"},
    )
    with pytest.raises(
        ProtocolError,
        match=rf"cross-run recovery is forbidden; existing status={status}",
    ):
        reject_existing_run_state(tmp_path)


def test_spawn_job_dto_and_result_are_pickle_safe_across_real_process_boundary() -> None:
    context = mp.get_context("spawn")
    try:
        semaphore = context.Semaphore(1)
    except NotImplementedError as exc:
        pytest.skip(f"OS semaphore creation is unavailable: {exc}")
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS}:
            pytest.skip(f"OS semaphore creation is unavailable: {exc}")
        raise
    semaphore.acquire()
    semaphore.release()

    surface, job = _spawn_route_fixture()
    assert pickle.loads(pickle.dumps(surface)) == surface
    assert pickle.loads(pickle.dumps(job)) == job
    try:
        results = execute_route_jobs(
            surface,
            (job,),
            workers=4,
            threads_per_worker=3,
        )
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"OS process spawning is unavailable: {exc}")
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS}:
            pytest.skip(f"OS process spawning is unavailable: {exc}")
        raise

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, RouteJobResult)
    assert pickle.loads(pickle.dumps(result)) == result
    assert result.plan == job["plan"]
    assert len(result.support_responses) == 2 * 8 * 2
    assert len(result.model_fits) == 8 * 2
    assert len(result.candidate_scores) == 4 * 2 * 9
    assert len(result.decisions) == 4
    assert len(result.predictions) == 4 * 2


@pytest.mark.parametrize(
    ("workers", "threads"),
    ((3, 3), (4, 2), (5, 3)),
)
def test_route_pool_rejects_every_noncanonical_topology(
    workers: int, threads: int
) -> None:
    with pytest.raises(ProtocolError, match="worker topology drifted"):
        execute_route_jobs(object(), ({"plan": object()},), workers=workers, threads_per_worker=threads)


def test_two_fresh_process_attestation_has_exact_schema_and_bounded_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attested, calls = _attested_checks(monkeypatch, tmp_path)
    attestation = attested[fresh_process_validation.ATTESTATION_KEY]
    assert isinstance(attestation, dict)
    assert set(attestation) == ATTESTATION_KEYS
    assert attestation["schema_version"] == fresh_process_validation.ATTESTATION_SCHEMA
    assert attestation["fresh_python_process_count"] == 2
    assert attestation["independent_fresh_python_processes"] is True
    assert attestation["process_launches_sequential"] is True
    assert attestation["cuda_visible_devices"] == ""
    assert attestation["subprocess_exit_codes"] == [0, 0]
    assert len(set(attestation["child_process_ids"])) == 2
    assert all(set(row) == CHILD_RESULT_KEYS for row in attestation["child_process_results"])
    assert len(calls) == 2
    expected_command = (
        sys.executable,
        "-m",
        fresh_process_validation.WORKER_MODULE,
        "--worker",
        str(tmp_path.resolve()),
    )
    for command, kwargs in calls:
        assert command == expected_command
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["PYTHONHASHSEED"] == "0"
        assert environment["OMP_NUM_THREADS"] == "1"
        assert environment["OPENBLAS_NUM_THREADS"] == "1"
        assert environment["MKL_NUM_THREADS"] == "1"
        assert environment["VECLIB_MAXIMUM_THREADS"] == "1"
        assert environment["NUMEXPR_NUM_THREADS"] == "1"
        assert environment["BLIS_NUM_THREADS"] == "1"
        assert kwargs["timeout"] == (
            fresh_process_validation.FRESH_PROCESS_TIMEOUT_SECONDS
        )
    assert (
        fresh_process_validation.verify_attested_validation_checks(
            attested,
            expected_reconstructed_checks={
                key: value
                for key, value in attested.items()
                if key != fresh_process_validation.ATTESTATION_KEY
            },
        )
        == attested
    )


def test_validation_report_requires_attestation_and_preserves_reconstructed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attested, _ = _attested_checks(monkeypatch, tmp_path)
    persist_validation_report(tmp_path, attested)
    report = json.loads(
        (tmp_path / "reports/validation_report.json").read_text(encoding="utf-8")
    )
    assert report == {
        "schema_version": "fixed_bank_cdca_validation_report_v1",
        **attested,
    }

    tampered = json.loads(json.dumps(attested))
    attestation = tampered[fresh_process_validation.ATTESTATION_KEY]
    attestation["child_process_results"][0]["result_payload_hash"] = "0" * 64
    unsigned = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    attestation["attestation_hash"] = canonical_hash(unsigned)
    with pytest.raises(ProtocolError, match="attestation drifted"):
        persist_validation_report(tmp_path / "tampered", tampered)

    reconstructed = {
        key: value
        for key, value in attested.items()
        if key != fresh_process_validation.ATTESTATION_KEY
    }
    with pytest.raises(ProtocolError, match="not reconstructive"):
        persist_validation_report(tmp_path / "missing", reconstructed)

    completed = validation._validate_attested_report(tmp_path, reconstructed)
    assert completed == attested
    assert completed["status"] == "PASS"


def test_two_gpu_then_cuda_free_four_by_three_runtime_contract_is_exact() -> None:
    runtime = canonical_runtime_payload()
    assert runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert runtime["persistent_generation_worker_count"] == 2
    assert runtime["multiprocessing_start_method"] == "spawn"
    assert runtime["gpu_generation_phase_precedes_cpu_phase"] is True
    assert runtime["cuda_visible_devices_cleared_before_cpu_phase"] is True
    assert runtime["parent_cuda_context_forbidden"] is True
    assert runtime["phase_disjoint_gpu_and_cpu_pools"] is True
    assert runtime["route_model_workers"] == 4
    assert runtime["classifier_workers"] == 4
    assert runtime["classifier_threads_per_worker"] == 3
    assert runtime["maximum_total_cpu_threads"] == 12
    assert runtime["cross_run_recovery_allowed"] is False
    assert runtime["terminal_recovery_allowed"] is False
    assert runtime["previous_stage90_scratch_reuse_forbidden"] is True
    assert_frozen_source_runtime(runtime)
    assert_fixed_bank_a1_runtime(runtime)

    runner_source = (PACKAGE_ROOT / "runner.py").read_text(encoding="utf-8")
    generation = runner_source.index("materialize_sources)(")
    cuda_free = runner_source.index("enter_cuda_free_cpu_phase()")
    classifier = runner_source.index("materialize_probabilities)(")
    routes = runner_source.index("execute_route_jobs)(")
    assert generation < cuda_free < classifier < routes


def test_dedicated_scratch_is_write_probed_before_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "owned-scratch"
    monkeypatch.setattr(execution_adapter, "SCRATCH_ROOT", str(scratch))
    runtime = canonical_runtime_payload()
    checks = execution_adapter._probe_dedicated_scratch(runtime)
    assert checks["dedicated_scratch_absent_at_launch"] is True
    assert checks["dedicated_scratch_parent_writable"] is True
    assert checks["dedicated_scratch_free_bytes_at_launch"] > 0
    assert not scratch.exists()

    scratch.mkdir()
    with pytest.raises(ProtocolError, match="prior-run or foreign scratch"):
        execution_adapter._probe_dedicated_scratch(runtime)


def test_package_ast_imports_no_other_stage90_diagnostic_package() -> None:
    diagnostics_root = PACKAGE_ROOT.parent
    other_diagnostic_packages = {
        path.name
        for path in diagnostics_root.iterdir()
        if path.is_dir()
        and path.name != PACKAGE_ROOT.name
        and not path.name.startswith("__")
    }
    violations = {}
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        imported = _imported_modules(path)
        forbidden = tuple(
            module
            for module in imported
            if set(module.split(".")) & other_diagnostic_packages
        )
        if forbidden:
            violations[path.name] = forbidden
    assert not violations
