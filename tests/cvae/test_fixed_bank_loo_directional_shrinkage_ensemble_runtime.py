from __future__ import annotations

import ast
import errno
from fractions import Fraction
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble import (
    execution_adapter,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    relative_files,
    validate_content_index,
    write_content_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.config_payloads import (
    canonical_runtime_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    METHOD_IDS,
    NULL_REPLICATES,
    NULL_SEED,
    PRE_TERMINAL_METHOD_IDS,
    TERMINAL_ORACLE_IDS,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.ensemble import (
    DESCRIPTIVE_METHOD_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble import (
    fresh_process_validation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.hashing import (
    canonical_hash,
    canonical_json,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.loo_plans import (
    WholeCaseLooPlan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.nulls import (
    build_candidate_identity_null_plan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.persistence import (
    persist_json,
    persist_validation_report,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.products import (
    BinaryLabel,
    DirectionalGain,
    DonorPrior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.runner_runtime import (
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
    / "fixed_bank_loo_directional_shrinkage_ensemble"
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
    "manifests/loo_plan_seal.json",
    "manifests/donor_prior_seal.json",
    "manifests/endpoint_library_seal.json",
    "manifests/arm_decisions_seal.json",
    "manifests/aggregate_plan_decision_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "manifests/content_index.json",
    "tables/action_library.csv",
    "tables/exact_nine_probability_index.csv",
    "tables/loo_plans.csv",
    "tables/case_action_confusions.csv",
    "tables/directional_gains.csv",
    "tables/donor_priors.csv",
    "tables/endpoint_arms.csv",
    "tables/arm_decisions.csv",
    "tables/control_decisions.csv",
    "tables/method_predictions.csv",
    "tables/descriptive_control_predictions.csv",
    "tables/terminal_case_confusions.csv",
    "tables/terminal_method_metrics.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv",
    "tables/whole_pipeline_delete_one_center.csv",
    "tables/leave_one_arm_ablations.csv",
    "tables/null_statistics.csv",
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

PRIOR_STAGE90_DIAGNOSTIC_MODULE_FRAGMENTS = (
    "fixed_bank_actionability_recoverability",
    "fixed_bank_decision_audit",
    "fixed_bank_disagreement_regret_prediction_only",
    "fixed_bank_hierarchical_residual_stacker",
    "fixed_bank_label_aware_case_oof_ceiling",
    "fixed_bank_labeled_support_case_conditional_flip_router",
    "fixed_bank_multi_challenger_hierarchical_flip_router",
    "fixed_bank_pooled_bacc_case_oof_ceiling",
    "fixed_bank_signed_error_gate",
    "fixed_bank_support_static_router_s4",
)


def _write_content_members(root: Path) -> None:
    for member in CONTENT_INDEX_MEMBERS:
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"sealed::{member}\n".encode("utf-8"))


def _replace_index(root: Path, payload: dict[str, object]) -> None:
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    payload["content_hash"] = canonical_hash(unhashed)
    atomic_json(root / "manifests/content_index.json", payload)


def _imported_modules(path: Path, *, top_level_only: bool = False) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    nodes = tree.body if top_level_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return tuple(modules)


def _spawn_route_fixture() -> tuple[ExactNineProbabilitySurface, dict[str, object]]:
    target = "0"
    support = (
        ("support-a", "sample-a-positive", 1),
        ("support-a", "sample-a-negative", 0),
        ("support-b", "sample-b-positive", 1),
        ("support-b", "sample-b-negative", 0),
    )
    probability_rows = tuple(
        ExactNineProbabilityRow(
            target,
            case_id,
            sample_id,
            action_id,
            (0.4,) * 9,
        )
        for case_id, sample_id, _label in support
        for action_id in physical_action_ids(target)
    )
    surface = ExactNineProbabilitySurface(probability_rows, "spawn-store-v1")
    plan = WholeCaseLooPlan(
        target_center=target,
        case_id="held",
        group_id="held",
        support_case_ids=("support-a", "support-b"),
        evaluation_sample_ids=("held-sample",),
        probability_surface_hash=surface.surface_hash,
    )
    labels = tuple(
        BinaryLabel(
            target,
            case_id,
            sample_id,
            label,
            "route_support::H=0::c=held",
        )
        for case_id, sample_id, label in support
    )
    priors = tuple(
        DonorPrior(
            target,
            source,
            direction,
            tuple(
                DirectionalGain(
                    query_center=query,
                    excluded_case_id=None,
                    source=source,
                    direction=direction,
                    n_positive=1,
                    n_negative=1,
                    favorable_count=0,
                    adverse_count=0,
                    contributing_case_ids=(f"donor-case-{query}",),
                    label_scope=f"donor::{query}",
                )
                for query in CENTERS
                if query not in {target, source}
            ),
        )
        for source in candidate_sources(target)
        for direction in ("zero_to_one", "one_to_zero")
    )
    return surface, {"plan": plan, "support_labels": labels, "donor_priors": priors}


def test_required_files_are_exact_43_and_content_index_exclusions_are_exact() -> None:
    assert REQUIRED_FILES == EXPECTED_REQUIRED_FILES
    assert len(REQUIRED_FILES) == 43
    assert CONTENT_INDEX_MEMBERS == tuple(
        member for member in EXPECTED_REQUIRED_FILES if member not in CONTENT_INDEX_EXCLUSIONS
    )
    assert len(CONTENT_INDEX_MEMBERS) == 40
    assert set(REQUIRED_FILES).difference(CONTENT_INDEX_MEMBERS) == (
        CONTENT_INDEX_EXCLUSIONS
    )


@pytest.mark.parametrize("tamper", ("header", "row", "claim"))
def test_content_index_rejects_coherently_rehashed_tamper(
    tmp_path: Path, tamper: str
) -> None:
    _write_content_members(tmp_path)
    write_content_index(
        tmp_path,
        config_contract_hash="config-contract",
        protocol_contract_hash="protocol-contract",
    )
    index_path = tmp_path / "manifests/content_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if tamper == "header":
        payload["member_count"] = 37
        error = "content-index header drifted"
    elif tamper == "claim":
        payload["publication_decision"] = "PROMOTE"
        error = "content-index header drifted"
    else:
        payload["members"][0]["promotion_eligible"] = True
        error = "content-index row malformed"
    _replace_index(tmp_path, payload)

    with pytest.raises(ProtocolError, match=error):
        validate_content_index(
            tmp_path,
            config_contract_hash="config-contract",
            protocol_contract_hash="protocol-contract",
        )


def test_bundle_inventory_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    link = root / "manifests/protocol_manifest.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)

    with pytest.raises(ProtocolError, match="bundle contains a symlink"):
        relative_files(root)


def test_partial_product_state_is_rejected_even_when_member_is_owned(
    tmp_path: Path,
) -> None:
    for member in (
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "tables/loo_plans.csv",
    ):
        path = tmp_path / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("partial\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="partial/cross-run state is forbidden"):
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


def test_json_persistence_is_native_idempotent_and_nonrepairing(tmp_path: Path) -> None:
    path = tmp_path / "reports/product.json"
    payload = {
        "schema_version": "fixture_v1",
        "ratio": Fraction(3, 5),
        "members": ("a", "b"),
        "nested": {"enabled": False},
    }
    persist_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": "fixture_v1",
        "ratio": [3, 5],
        "members": ["a", "b"],
        "nested": {"enabled": False},
    }
    persist_json(path, payload)

    with pytest.raises(ProtocolError, match="refuses repair"):
        persist_json(path, {**payload, "nested": {"enabled": True}})
    with pytest.raises(ProtocolError, match="not JSON-native"):
        persist_json(tmp_path / "reports/opaque.json", {"opaque": object()})


@pytest.mark.parametrize(
    "payload",
    (
        {"label": 1},
        {"nested": [{"Labels": [0, 1]}]},
        {"sample_path": "/forbidden/sample"},
        {"nested": {"checkpoint_path": "/forbidden/checkpoint"}},
        {"manifest_path": "/forbidden/manifest.csv"},
    ),
)
def test_json_persistence_rejects_label_and_path_keys(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    with pytest.raises(ProtocolError, match="persistence forbids key"):
        persist_json(tmp_path / "reports/forbidden.json", payload)
    assert not (tmp_path / "reports/forbidden.json").exists()


def test_candidate_identity_null_is_deterministic_10000_by_218_by_8_digest() -> None:
    route_keys = tuple(
        (center, f"case-{center}-{ordinal:03d}")
        for center in CENTERS
        for ordinal in range(EXPECTED_CASE_COUNTS_BY_CENTER[center])
    )
    plan = build_candidate_identity_null_plan(route_keys)
    payload = plan.to_payload()
    expected_digest = (
        "7f6268bbd51ea450b3138d4f0289e3e9"
        "738b00df4b220a37650a43588fc4ad72"
    )

    assert NULL_SEED == 20_260_813
    assert NULL_REPLICATES == 10_000
    assert plan.shape == (10_000, 218, 8)
    assert plan.permutation_sha256 == expected_digest
    assert plan.permutation(0, 0) == (3, 4, 5, 1, 6, 0, 7, 2)
    assert plan.permutation(9_999, 217) == (2, 6, 5, 4, 0, 1, 7, 3)
    assert {
        key: value
        for key, value in payload.items()
        if "permutation" in key and key.endswith("sha256")
    } == {"permutation_sha256": expected_digest}
    assert "permutations" not in payload
    assert payload["exchangeability_claimed"] is False
    assert payload["descriptive_only"] is True


def test_canonical_runtime_satisfies_both_neutral_contracts_and_cuda_rule() -> None:
    runtime = canonical_runtime_payload()

    assert runtime["multiprocessing_start_method"] == "spawn"
    assert runtime["parent_cuda_context_forbidden"] is True
    assert runtime["parent_cuda_context_forbidden_during_cpu_phase"] is True
    assert runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert runtime["classifier_workers"] == 4
    assert runtime["classifier_threads_per_worker"] == 3
    assert runtime["cross_run_recovery_allowed"] is False
    assert runtime["terminal_recovery_allowed"] is False
    assert_frozen_source_runtime(runtime)
    assert_fixed_bank_a1_runtime(runtime)

    drifted = {**runtime, "parent_cuda_context_forbidden": False}
    with pytest.raises(ProtocolError, match="two exact float32 GPU streams"):
        assert_frozen_source_runtime(drifted)


def test_package_ast_imports_no_prior_stage90_diagnostic_modules() -> None:
    imports_by_file = {
        path.name: _imported_modules(path) for path in sorted(PACKAGE_ROOT.glob("*.py"))
    }
    violations = {
        filename: tuple(
            module
            for module in modules
            if any(
                fragment in module
                for fragment in PRIOR_STAGE90_DIAGNOSTIC_MODULE_FRAGMENTS
            )
        )
        for filename, modules in imports_by_file.items()
    }
    assert not {filename: modules for filename, modules in violations.items() if modules}


def test_runner_ast_has_no_top_level_validation_or_validation_plan_producer_imports() -> None:
    runner_path = PACKAGE_ROOT / "runner.py"
    top_level_modules = _imported_modules(runner_path, top_level_only=True)
    assert not tuple(
        module for module in top_level_modules if "validation" in module.split(".")
    )
    all_modules = _imported_modules(runner_path)
    assert not tuple(
        module for module in all_modules if module.endswith("validation_plans")
    )
    # Finalization may lazily import only the public validator entrypoint.
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    lazy_validation_imports = tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "validation"
    )
    assert len(lazy_validation_imports) == 1
    assert tuple(alias.name for alias in lazy_validation_imports[0].names) == (
        "validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle",
    )


def test_preterminal_and_descriptive_method_coverage_is_exact() -> None:
    assert PRE_TERMINAL_METHOD_IDS == (
        "B",
        "U",
        "DCSE_LOO",
        "G_directional_matched",
        "DLOO_raw",
        "LOO_frequency_committee",
    )
    assert TERMINAL_ORACLE_IDS == (
        "O_directional_static",
        "O_case_directional",
    )
    assert METHOD_IDS == (*PRE_TERMINAL_METHOD_IDS, *TERMINAL_ORACLE_IDS)
    assert DESCRIPTIVE_METHOD_IDS == (
        "DCSE_hard_vote_descriptive",
        "DCSE_unique_mean_descriptive",
        "uniform_A1_mean_descriptive",
        "DCSE_zero_to_one_only_descriptive",
        "DCSE_one_to_zero_only_descriptive",
    )
    runner_source = (PACKAGE_ROOT / "runner.py").read_text(encoding="utf-8")
    persistence_source = (PACKAGE_ROOT / "persistence.py").read_text(encoding="utf-8")
    validation_source = (PACKAGE_ROOT / "validation_plans.py").read_text(
        encoding="utf-8"
    )
    for method_id in DESCRIPTIVE_METHOD_IDS:
        assert method_id in runner_source or method_id in persistence_source
    assert "9_928 * 5" in runner_source
    assert "9_928 * 5" in persistence_source
    assert "9_928 * 5" in validation_source


def test_runner_attests_two_fresh_replays_before_persisting_validation() -> None:
    source = (PACKAGE_ROOT / "runner.py").read_text(encoding="utf-8")
    attest = source.index("checks = require_two_fresh_process_validations(")
    persist = source.index(
        "(deps.persist_validation or persist_validation_report)(root, checks)"
    )
    complete = source.index(
        'write_state(deps, root, status="COMPLETE", phase="COMPLETE")'
    )
    final_validation = source.rindex("(deps.validate_bundle or _validate_bundle)(")
    assert attest < persist < complete < final_validation


def test_attested_validation_is_bound_before_report_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "schema_version": "fixed_bank_dcse_validation_v1",
        "status": "PASS",
        "all_six_preterminal_methods_reconstructed": True,
        "all_five_descriptive_controls_reconstructed": True,
        "candidate_identity_null_plan_reconstructed": True,
        "exact_topology_and_confusions_compared": True,
        "fitted_numeric_tolerance_used": False,
        "content_index_validated_before_scientific_members": True,
        "two_fresh_cuda_free_process_replays_required": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "image_or_sample_paths_persisted": False,
        "terminal_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    parent_pid = __import__("os").getpid()
    worker_payloads = iter(
        (
            {"process_id": parent_pid + 10_001, "checks": expected},
            {"process_id": parent_pid + 10_002, "checks": expected},
        )
    )

    def fake_worker(_root: Path) -> subprocess.CompletedProcess[str]:
        payload = next(worker_payloads)
        return subprocess.CompletedProcess(
            args=(sys.executable,),
            returncode=0,
            stdout=canonical_json(payload).decode("utf-8") + "\n",
            stderr="",
        )

    monkeypatch.setattr(fresh_process_validation, "_run_worker", fake_worker)
    attested = fresh_process_validation.require_two_fresh_process_validations(
        tmp_path,
        expected_checks=expected,
    )
    assert (
        fresh_process_validation.verify_attested_validation_checks(
            attested,
            expected_reconstructed_checks=expected,
        )
        == attested
    )
    persist_validation_report(tmp_path, attested)
    persisted = json.loads(
        (tmp_path / "reports/validation_report.json").read_text(encoding="utf-8")
    )
    assert persisted == attested

    tampered = json.loads(json.dumps(attested))
    attestation = tampered["fresh_process_validation_attestation"]
    attestation["child_process_results"][0]["result_payload_hash"] = "0" * 64
    unsigned = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    attestation["attestation_hash"] = canonical_hash(unsigned)
    with pytest.raises(ProtocolError, match="attestation drifted"):
        persist_validation_report(tmp_path / "tampered", tampered)

    with pytest.raises(ProtocolError, match="validation report is not reconstructive"):
        persist_validation_report(tmp_path / "missing", expected)


def test_physical_partition_case_count_deduplicates_sample_rows() -> None:
    rows = (
        SimpleNamespace(center="0", case_id="case-a", sample_id="sample-a1"),
        SimpleNamespace(center="0", case_id="case-a", sample_id="sample-a2"),
        SimpleNamespace(center="0", case_id="case-b", sample_id="sample-b1"),
    )
    observed = execution_adapter.physical_partition_hash(SimpleNamespace(rows=rows))
    expected = canonical_hash(
        {
            "schema_version": "fixed_bank_dcse_global_physical_plan_v1",
            "rows": [
                {
                    "target_center": row.center,
                    "case_id": row.case_id,
                    "sample_id": row.sample_id,
                }
                for row in rows
            ],
            "row_count": 3,
            "case_count": 2,
            "labels_used": False,
            "arbitrary_folds_used": False,
        }
    )
    assert observed == expected


@pytest.mark.parametrize(
    ("injection", "error"),
    (
        ("file", "scratch contains foreign state"),
        ("symlink", "scratch contains foreign state"),
        ("unknown_directory", "scratch contains foreign state"),
        ("dirty_generation", "generation scratch is not sealed"),
        ("dirty_prediction", "prediction scratch is not a sealed empty tree"),
        ("missing_generation", "generation scratch is absent"),
    ),
)
def test_scratch_cleanup_rejects_injected_file_or_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injection: str,
    error: str,
) -> None:
    scratch = (tmp_path / "scratch").resolve()
    scratch.mkdir()
    if injection == "file":
        (scratch / "injected.bin").write_bytes(b"foreign")
    elif injection == "symlink":
        external = (tmp_path / "external").resolve()
        external.mkdir()
        (scratch / execution_adapter.LOCAL_GENERATION_DIRECTORY).symlink_to(
            external, target_is_directory=True
        )
    elif injection == "unknown_directory":
        (scratch / "unknown").mkdir()
    elif injection == "dirty_generation":
        generation = scratch / execution_adapter.LOCAL_GENERATION_DIRECTORY
        generation.mkdir()
        (generation / "partial.bin").write_bytes(b"partial")
    else:
        prediction = scratch / execution_adapter.LOCAL_PREDICTION_DIRECTORY
        prediction.mkdir()
        if injection == "dirty_prediction":
            generation = scratch / execution_adapter.LOCAL_GENERATION_DIRECTORY
            (generation / "arrays").mkdir(parents=True)
            (generation / "manifests").mkdir()
            for member in (
                "arrays/frozen_source_streams.npy",
                "manifests/frozen_source_stream_index.json",
                "manifests/frozen_source_stream_lock.json",
            ):
                (generation / member).write_bytes(b"fixture")
            (prediction / "foreign.bin").write_bytes(b"foreign")
    monkeypatch.setattr(execution_adapter, "SCRATCH_ROOT", str(scratch))
    config = SimpleNamespace(
        runtime={"scratch_preference": [str(scratch), "artifact_parent"]}
    )

    with pytest.raises(ProtocolError, match=error):
        execution_adapter.cleanup_validated_scratch(config)
    assert scratch.exists()


def test_two_independent_fresh_processes_have_distinct_pids_and_same_contract() -> None:
    program = (
        "import json, os; "
        "from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.config_payloads "
        "import canonical_runtime_payload; "
        "from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_directional_shrinkage_ensemble.hashing "
        "import canonical_hash; "
        "print(json.dumps({'pid': os.getpid(), 'hash': canonical_hash(canonical_runtime_payload())}))"
    )
    outputs = tuple(
        subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(ROOT / "src"),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            },
        )
        for _ in range(2)
    )
    attestations = tuple(json.loads(result.stdout) for result in outputs)
    assert attestations[0]["pid"] != attestations[1]["pid"]
    assert all(row["pid"] != __import__("os").getpid() for row in attestations)
    assert attestations[0]["hash"] == attestations[1]["hash"]


def test_route_job_crosses_a_real_spawn_boundary() -> None:
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
    try:
        results = execute_route_jobs(
            surface,
            (job,),
            workers=4,
            threads_per_worker=3,
        )
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"OS semaphore creation is unavailable: {exc}")
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EPERM, errno.ENOSYS}:
            pytest.skip(f"OS semaphore creation is unavailable: {exc}")
        raise

    assert len(results) == 1
    pickle.loads(pickle.dumps(results[0]))
    assert results[0].plan == job["plan"]
    assert len(results[0].counts) == 20
    assert len(results[0].gains) == 16
    assert len(results[0].endpoint_decisions) == 18
    assert len(results[0].control_decisions) == 2
    assert all(
        decision.zero_to_one.selected_source is None
        and decision.one_to_zero.selected_source is None
        for decision in results[0].endpoint_decisions
    )
