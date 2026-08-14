from __future__ import annotations

import ast
import csv
from contextlib import nullcontext
import errno
import importlib
import json
import os
from pathlib import Path
import pkgutil
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.constants import (
    CENTERS,
    DIRECTION_IDS,
    candidate_sources,
    physical_action_ids,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.correctness_proxy import (
    build_label_free_features,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.donor_prior import (
    DonorPrior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.probability_surfaces import (
    ExactNineProbabilityRow,
    ExactNineProbabilitySurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.response_products import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.route_worker_runtime import (
    _initialize_route_worker,
    compute_route_job,
    exact_route_blas_scope,
    execute_route_jobs,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.runner_services import (
    RunnerServices,
    read_scoped_manifest_labels,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.actions import (
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.artifact_rows import (
    reject_forbidden_persistence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.artifact_writers import (
    persist_rows,
    read_rows,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.source_surface_runtime import (
    ProbabilityIndexRow,
    runtime_summary_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.split_plans import (
    WholeCaseLooPlan,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.fixed_bank_a1_prediction_contracts import (
    stable_digest,
    validate_action_library,
)
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    evaluation_row_id,
)


SHA = canonical_hash({"runtime-test": 1})
PACKAGE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_loo_opportunity_gated_dual_endpoint_router"
)


def test_neutral_action_library_uses_frozen_16_hex_runtime_digest() -> None:
    _payload, digest = validate_action_library(action_library_by_target())
    assert stable_digest(digest)
    assert len(digest) == 16


def test_runtime_summary_persists_only_path_free_scratch_identity() -> None:
    source = SimpleNamespace(lock_hash=SHA, records=(object(),))
    prediction = SimpleNamespace(
        seal_hash=SHA, store=SimpleNamespace(cells=(object(), object()))
    )
    preflight = {
        "schema_version": "fixed_bank_dual_endpoint_workstation_preflight_v1",
        "status": "PASS",
        "generation_devices": ["cuda:0", "cuda:1"],
        "persistent_gpu_workers": 2,
        "classifier_workers": 4,
        "blas_threads_per_classifier_worker": 3,
        "target_probability_cell_count": 810,
        "scratch_root_id": "fixed_bank_loo_opportunity_gated_dual_endpoint_router_v1",
        "scratch_fallback_role": "artifact_parent",
    }
    runtime = {
        "classifier_workers": 4,
        "route_model_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "resume_policy": "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only",
    }
    payload = runtime_summary_payload(
        source_cache=source,
        prediction=prediction,
        preflight=preflight,
        runtime=runtime,
    )
    reject_forbidden_persistence(payload)
    assert payload["scratch_root_id"] == preflight["scratch_root_id"]


def test_probability_index_row_has_exact_json_schema_and_persists_roundtrip(
    tmp_path: Path,
) -> None:
    row = ProbabilityIndexRow(
        "0",
        "B",
        23,
        (SHA,) * 9,
        SHA,
        SHA,
        SHA,
    )
    expected = {
        "target_center": "0",
        "action_id": "B",
        "row_count": 23,
        "source_cell_probability_sha256": [SHA] * 9,
        "sample_identity_hash": SHA,
        "case_identity_hash": SHA,
        "exact_nine_probability_sha256": SHA,
        "storage_dtype": "float32",
        "reduction_dtype": "float64",
    }
    payload = row.to_payload()
    assert payload == expected
    assert json.loads(json.dumps(payload, allow_nan=False)) == expected
    path = tmp_path / "exact_nine_probability_index.csv"
    assert persist_rows(path, (row,)) == (expected,)
    assert read_rows(path) == (expected,)


def test_no_successor_payload_method_recurses_through_json_native_self() -> None:
    package = importlib.import_module(PACKAGE)
    root = Path(next(iter(package.__path__)))
    offenders: list[tuple[str, int]] = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "to_payload":
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "json_native"
                    and any(isinstance(argument, ast.Name) and argument.id == "self" for argument in call.args)
                ):
                    offenders.append((path.name, call.lineno))
    assert offenders == []


def _route_fixture() -> tuple[object, dict[str, object]]:
    cases = ("held", "support-a", "support-b")
    rows = []
    for case in cases:
        for sample, baseline in (("negative", 0.35), ("positive", 0.65)):
            for action_index, action in enumerate(physical_action_ids("0")):
                probability = baseline
                if action.startswith("A1::"):
                    if sample == "negative":
                        probability = 0.75 if action_index % 2 == 0 else 0.25
                    else:
                        probability = 0.25 if action_index % 2 == 0 else 0.75
                rows.append(
                    ExactNineProbabilityRow(
                        "0", case, f"{case}-{sample}", action, (probability,) * 9
                    )
                )
    surface = ExactNineProbabilitySurface(tuple(rows), SHA)
    plan = WholeCaseLooPlan(
        "0",
        "held",
        "held",
        ("support-a", "support-b"),
        ("held-negative", "held-positive"),
        surface.surface_hash,
    )
    labels = tuple(
        BinaryLabel(
            "0", case, f"{case}-{sample}", int(sample == "positive"), "route-test"
        )
        for case in plan.support_case_ids
        for sample in ("negative", "positive")
    )
    priors = tuple(
        DonorPrior(
            "0",
            source,
            direction,
            tuple(center for center in CENTERS if center not in {"0", source}),
            (SHA,) * 7,
            0,
            1,
            0.0,
        )
        for source in candidate_sources("0")
        for direction in DIRECTION_IDS
    )
    job = {
        "plan": plan,
        "support_labels": labels,
        "donor_priors": priors,
        "route_features": build_label_free_features(surface),
    }
    return surface, job


@pytest.mark.parametrize("threads", (3.0, "3", None, True, 2))
def test_route_thread_contract_rejects_non_exact_integer(threads: object) -> None:
    with pytest.raises(ProtocolError, match="thread count drifted"):
        _initialize_route_worker(object(), threads)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError, match="thread count drifted"):
        with exact_route_blas_scope(threads):  # type: ignore[arg-type]
            pass


def test_exact_route_blas_scope_restores_outer_one_thread_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in names:
        monkeypatch.setenv(name, "1")
    with exact_route_blas_scope(3):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert all(os.environ[name] == "3" for name in names)
    assert all(os.environ[name] == "1" for name in names)


def test_spawned_route_models_are_payload_and_hash_exact_to_serial_replay() -> None:
    surface, job = _route_fixture()
    with exact_route_blas_scope(3):
        serial = compute_route_job(surface, job)
    try:
        (spawned,) = execute_route_jobs(
            surface, (job,), workers=4, threads_per_worker=3
        )
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES, errno.ENOSYS}:
            raise
        pytest.skip("host sandbox denies multiprocessing semaphore admission")
    except NotImplementedError as exc:
        if not any(
            token in str(exc).casefold()
            for token in ("semaphore", "multiprocessing.synchronize")
        ):
            raise
        pytest.skip("host Python lacks multiprocessing semaphore support")
    serial_payload = [row.to_payload() for row in serial.model_fits_primary]
    spawned_payload = [row.to_payload() for row in spawned.model_fits_primary]
    assert spawned_payload == serial_payload
    assert canonical_hash(spawned_payload) == canonical_hash(serial_payload)
    assert len(spawned.robust_arm_decisions) == 18
    assert {row.method_id for row in spawned.robust_arm_decisions} == {
        "R_NINE_ARM_ROBUST",
        "G_DIRECTIONAL_MATCHED",
    }


def test_scoped_loader_never_csv_decodes_excluded_label_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "center,case_id,label\n"
        "0,excluded-a,EXCLUDED_SENTINEL_A\n"
        "0,allowed,1\n"
        "0,excluded-b,EXCLUDED_SENTINEL_B\n",
        encoding="utf-8",
    )
    identities = tuple(
        SimpleNamespace(
            center="0",
            case_id=case,
            sample_id=evaluation_row_id(SHA, ordinal),
            manifest_row_index=ordinal,
        )
        for ordinal, case in enumerate(("excluded-a", "allowed", "excluded-b"))
    )
    decoded: list[str] = []
    real_reader = csv.reader

    def spy(lines: object, *args: object, **kwargs: object):
        values = tuple(lines)  # type: ignore[arg-type]
        decoded.extend(str(value) for value in values)
        return real_reader(values, *args, **kwargs)

    import midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.runner_services as module

    monkeypatch.setattr(module.csv, "reader", spy)
    config = SimpleNamespace(test_manifest_path=manifest, expected_manifest_sha256=SHA)
    allowed = frozenset({("0", "allowed", identities[1].sample_id)})
    (label,) = read_scoped_manifest_labels(
        config, SimpleNamespace(rows=identities), allowed_keys=allowed
    )
    assert label.value == 1
    assert not any("EXCLUDED_SENTINEL" in value for value in decoded)


def test_runner_phase_order_uses_four_cohesive_service_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import midogpp_thesis.cvae.diagnostics.fixed_bank_loo_opportunity_gated_dual_endpoint_router.runner as runner

    phases: list[str] = []
    states: list[tuple[str, str]] = []

    class Admission:
        def admit(self, *_args: object) -> dict[str, object]:
            return {"locks": SimpleNamespace(generation=object()), "frame": object()}

    class Physical:
        def preflight(self, *_args: object) -> dict[str, object]:
            return {"status": "PASS"}

        def source_streams(self, *_args: object) -> object:
            return object()

        def probabilities(self, *_args: object) -> dict[str, object]:
            return {
                "surface": object(),
                "prediction": object(),
                "seal": {"seal_hash": SHA},
            }

    class Science:
        def label_free(self, *_args: object) -> dict[str, object]:
            return {
                "plan_seal": object(),
                "plans": (),
                "features": (),
                "persisted_plan_seal": {},
                "feature_seal": {"seal_hash": SHA},
            }

        def route(self, **_kwargs: object) -> dict[str, object]:
            return {
                "seals": {"aggregate": {"seal_hash": SHA}},
                "directional_support_gains": (),
                "identification_decisions": (),
                "robust_arm_decisions": (),
                "method_predictions": (),
            }

    class Finalization:
        def evaluate(self, **_kwargs: object) -> dict[str, object]:
            return {
                "terminal_seal": {"seal_hash": SHA},
                "diagnostic_summary": {},
            }

        def validate(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"status": "PASS"}

    class Firewall:
        def __init__(self, *_args: object) -> None:
            pass

        def open_terminal_labels(self) -> tuple[object, ...]:
            return ()

        def report_payload(self) -> dict[str, object]:
            return {"status": "PASS"}

    monkeypatch.setattr(runner, "assert_workspace_resolved_paths", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "assert_launch_files", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "exclusive_run_lock", lambda *_a: nullcontext())
    monkeypatch.setattr(runner, "reject_existing_run_state", lambda *_a: None)
    monkeypatch.setattr(runner, "assert_no_foreign_or_partial_state", lambda *_a: None)
    monkeypatch.setattr(runner, "enter_cuda_free_cpu_phase", lambda: None)
    monkeypatch.setattr(runner, "assert_cuda_free_cpu_phase", lambda: None)
    monkeypatch.setattr(runner, "DualEndpointLabelFirewall", Firewall)
    monkeypatch.setattr(runner, "persist_terminal", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "write_content_index", lambda *_a, **_k: {})
    monkeypatch.setattr(
        runner, "require_two_fresh_process_validations", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(runner, "persist_validation_report", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "cleanup_validated_scratch", lambda *_a: None)
    monkeypatch.setattr(runner, "runtime_summary_payload", lambda **_k: {})
    monkeypatch.setattr(
        runner,
        "write_state",
        lambda _s, _r, *, status, phase, **_k: states.append((status, phase)),
    )
    services = RunnerServices(
        admission=Admission(),  # type: ignore[arg-type]
        physical=Physical(),  # type: ignore[arg-type]
        science=Science(),  # type: ignore[arg-type]
        finalization=Finalization(),  # type: ignore[arg-type]
        phase_observer=phases.append,
    )
    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime={},
        contract_hash=SHA,
    )
    assert runner.run_fixed_bank_loo_opportunity_gated_dual_endpoint_router(
        config, services=services
    ) == tmp_path
    assert phases == [
        "input_admission",
        "workstation_preflight",
        "source_generation",
        "physical_probability_seal",
        "label_free_plan_feature_seal",
        "scoped_route_science",
        "terminal_evaluation",
    ]
    assert states[-1] == ("COMPLETE", "COMPLETE")


def test_all_successor_modules_import_and_no_sibling_stage90_imports() -> None:
    package = importlib.import_module(PACKAGE)
    modules = tuple(pkgutil.iter_modules(package.__path__, PACKAGE + "."))
    for module in modules:
        importlib.import_module(module.name)
    root = Path(next(iter(package.__path__)))
    forbidden = (
        "fixed_bank_loo_directional_shrinkage_ensemble",
        "fixed_bank_case_directional_correctness_abstention_router",
        "fixed_bank_support_static_router",
    )
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "/private/tmp" not in source
        tree = ast.parse(source)
        imports = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(name in imported for name in forbidden for imported in imports)
