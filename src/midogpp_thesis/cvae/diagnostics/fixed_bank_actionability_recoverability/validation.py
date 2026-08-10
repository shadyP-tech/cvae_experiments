"""Content-first independent replay validation for the terminal bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.preflight import REQUIRED_DISTRIBUTIONS, REQUIRED_THREAD_ENVIRONMENT
from .aggregation import aggregate_exact_nine_probabilities
from .bundle import assert_closed_world, validate_content_index
from .execution import (
    build_loco_utility_product,
    build_pre_support_decision_products,
    build_prelabel_products,
    build_support_fold_product,
    combine_decision_products,
    combine_model_products,
    fit_target_model_product,
)
from .execution_adapter import (
    build_case_partition,
    load_frozen_source_streams,
    load_global_action_prediction_seal,
    runtime_summary_payload,
    seed_probability_rows,
)
from .hashing import canonical_hash
from .label_capabilities import ActionabilityLabelCapabilityManager
from .persistence import (
    persist_all_decisions,
    persist_and_validate_models,
    persist_initial_surfaces,
    persist_postseal_results,
    persist_pre_support_decisions,
    persist_prelabel_surfaces,
)
from .protocol import canonical_consumed_test_protocol
from .reports import leakage_report_payload
from .sealing import (
    record_durable_model_seals,
    record_durable_preevaluation_seals,
    record_durable_pre_support_seals,
)
from .terminal import evaluate_terminal


_INITIAL_MEMBERS = (
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/case_oof_partition.json",
    "tables/action_library.csv",
    "tables/case_oof_partitions.csv",
)
_PRELABEL_MEMBERS = (
    "tables/seed_probability_rows.csv",
    "tables/aggregated_probability_rows.csv",
    "tables/case_action_features.csv",
    "manifests/sealed_probability_surface.json",
    "manifests/prelabel_feature_seal.json",
    "reports/phase_01_prelabel_seal_complete.json",
)
_MODEL_MEMBERS = (
    "tables/loco_utility_targets.csv",
    "tables/model_fits.csv",
    "tables/model_predictions.csv",
    "manifests/loco_utility_seals.json",
    "manifests/model_seals.json",
)
_DECISION_MEMBERS = (
    "tables/method_decisions.csv",
    "manifests/pre_support_decisions_seal.json",
    "manifests/all_method_decisions_seal.json",
    "manifests/permutation_provenance_seal.json",
)
_TERMINAL_MEMBERS = (
    "tables/terminal_case_confusions.csv",
    "tables/terminal_center_metrics.csv",
    "tables/terminal_method_summary.csv",
    "tables/terminal_contrasts.csv",
    "tables/oracle_rank_metrics.csv",
    "tables/complementarity.csv",
    "tables/rank_stability.csv",
    "tables/permutation_metrics.csv",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "manifests/sealed_terminal_evaluation.json",
)


def validate_fixed_bank_actionability_recoverability_bundle(
    root: str | Path, *, config: object
) -> Mapping[str, object]:
    """Replay every scientific boundary without trusting report summaries."""

    path = Path(root)
    validation_exists = (path / "reports/validation_report.json").is_file()
    assert_closed_world(
        path,
        allow_incomplete=False,
        allow_pending_validation=not validation_exists,
    )
    protocol = canonical_consumed_test_protocol()

    # Content hashes are checked before parsing any scientific member.
    validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.contract_hash,
    )
    _reject_forbidden_persisted_fields(path)

    from .inputs import (
        assert_input_fence,
        load_label_free_test_frame,
        load_validated_locks,
        validate_active_diagnostic_workspace_binding,
        validate_pre_gpu_firewall,
        validate_workspace_provenance,
    )

    assert_input_fence(config)
    workspace = validate_active_diagnostic_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
    firewall["workspace_binding"] = workspace
    partition = build_case_partition(frame, config=config)

    preflight = _validate_preflight(path, runtime=getattr(config, "runtime"))
    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=locks.generation.generation_lock_hash,
    )
    action_manifest = _read_object(path / "manifests/action_library.json")
    prediction = load_global_action_prediction_seal(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_source_lock_hash=source.lock_hash,
        expected_partition_hash=partition.partition_hash,
        expected_action_library_hash=str(action_manifest["action_library_hash"]),
        expected_target_cache_binding_hash=frame.cache_binding_hash,
    )
    seeds = seed_probability_rows(prediction)
    probabilities = aggregate_exact_nine_probabilities(seeds)
    prelabel = build_prelabel_products(
        probabilities, protocol_contract_hash=protocol.contract_hash
    )

    manager = ActionabilityLabelCapabilityManager(
        getattr(config, "test_manifest_path"),
        frame,
        partition,
        global_prediction_seal_hash=prediction.seal_hash,
        label_free_feature_seal_hash=prelabel.feature_surface_hash,
        action_library_hash=prediction.store.action_library_hash,
    )
    runtime = getattr(config, "runtime")
    utilities: list[object] = []
    targets: list[object] = []
    for target in _centers():
        labels = manager.open_loco_donor_labels(target)
        utility = build_loco_utility_product(
            probabilities, labels, outer_target_center=target
        )
        utilities.append(utility)
        targets.append(
            fit_target_model_product(
                prelabel,
                utility,
                workers=int(runtime["model_workers"]),
                threads_per_worker=int(runtime["model_threads_per_worker"]),
                start_method=str(runtime["multiprocessing_start_method"]),
            )
        )
    models = combine_model_products(prelabel, tuple(targets))
    record_durable_model_seals(manager, models)

    pre_support = build_pre_support_decision_products(models, partition)
    record_durable_pre_support_seals(manager, pre_support)
    support_products: list[object] = []
    for target in _centers():
        for fold_ordinal in range(5):
            labels = manager.open_fold_support_labels(target, fold_ordinal)
            support_products.append(
                build_support_fold_product(
                    probabilities,
                    partition,
                    labels,
                    target_center=target,
                    fold_ordinal=fold_ordinal,
                )
            )
    decisions = combine_decision_products(
        pre_support, tuple(support_products), partition
    )
    record_durable_preevaluation_seals(manager, decisions)
    terminal_labels = manager.open_oof_evaluation_labels()
    capability_report = manager.access_report()
    evaluation = evaluate_terminal(
        probabilities,
        decisions,
        terminal_labels,
        partition,
        capability_report=capability_report,
        protocol_contract_hash=protocol.contract_hash,
        bootstrap_replicates=int(
            getattr(config, "evaluation")[
                "whole_case_cluster_bootstrap_replicates"
            ]
        ),
        bootstrap_seed=int(
            getattr(config, "evaluation")["whole_case_cluster_bootstrap_seed"]
        ),
        bootstrap_workers=int(runtime["bootstrap_workers"]),
        bootstrap_threads_per_worker=int(runtime["bootstrap_threads_per_worker"]),
        multiprocessing_start_method=str(runtime["multiprocessing_start_method"]),
    )
    leakage = leakage_report_payload(
        prediction_seal_hash=prediction.seal_hash,
        feature_seal_hash=prelabel.feature_surface_hash,
        action_library_hash=prediction.store.action_library_hash,
        model_seal_count=9 * 2 * 3,
        decision_count=len(decisions.all_decision_hashes),
        capability_report=capability_report,
    )
    observed_runtime = _read_object(path / "reports/runtime_summary.json")
    local_staging = observed_runtime.get("local_source_staging")
    if not isinstance(local_staging, Mapping):
        raise ProtocolError("Runtime summary lacks local staging provenance.")
    nested_preflight = local_staging.get("workstation_preflight")
    if not isinstance(nested_preflight, Mapping) or dict(nested_preflight) != dict(
        preflight
    ):
        raise ProtocolError(
            "Runtime summary is not fully bound to the validated preflight report."
        )
    staging_keys = set(local_staging)
    used = local_staging.get("used")
    status = local_staging.get("status")
    if (
        local_staging.get("attempted") is not True
        or type(used) is not bool
        or staging_keys
        not in (
            {"attempted", "used", "status", "workstation_preflight"},
            {
                "attempted",
                "used",
                "status",
                "failure",
                "workstation_preflight",
            },
        )
        or (
            used is True
            and (status != "STAGED_LOCAL_CPU_CACHE" or "failure" in local_staging)
        )
        or (
            used is False
            and "failure" not in local_staging
            and status != "CANONICAL_ALREADY_LOCAL"
        )
        or (
            used is False
            and "failure" in local_staging
            and (
                status != "CANONICAL_FALLBACK"
                or not isinstance(local_staging.get("failure"), str)
                or not local_staging.get("failure")
            )
        )
    ):
        raise ProtocolError("Runtime local-staging provenance drifted.")
    expected_runtime = runtime_summary_payload(
        source_cache=source,
        prediction_capability=prediction,
        local_staging=local_staging,
        runtime=runtime,
    )

    with tempfile.TemporaryDirectory(
        prefix="actionability-replay-", dir=str(path.parent)
    ) as temporary:
        replay = Path(temporary)
        persist_initial_surfaces(
            replay,
            config=config,
            protocol=protocol,
            provenance=provenance,
            frame=frame,
            firewall=firewall,
            partition=partition,
        )
        _compare_members(path, replay, _INITIAL_MEMBERS)
        persist_prelabel_surfaces(
            replay,
            prediction_capability=prediction,
            seed_rows=seeds,
            probability_surface=probabilities,
            prelabel=prelabel,
        )
        _compare_members(path, replay, _PRELABEL_MEMBERS)
        persist_and_validate_models(
            replay,
            products=models,
            utility_products=tuple(utilities),
            target_products=tuple(targets),
        )
        _compare_members(path, replay, _MODEL_MEMBERS)
        persist_pre_support_decisions(replay, products=pre_support)
        persist_all_decisions(replay, products=decisions)
        _compare_members(path, replay, _DECISION_MEMBERS)
        persist_postseal_results(
            replay,
            evaluation=evaluation,
            capability_report=capability_report,
            leakage_report=leakage,
            runtime_summary=expected_runtime,
        )
        _compare_members(path, replay, _TERMINAL_MEMBERS)

    checks_unhashed = {
        "schema_version": "midogpp_fixed_bank_actionability_recoverability_validation_v1",
        "status": "PASS",
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.contract_hash,
        "partition_hash": partition.partition_hash,
        "source_stream_lock_hash": source.lock_hash,
        "global_prediction_seal_hash": prediction.seal_hash,
        "action_library_hash": prediction.store.action_library_hash,
        "probability_surface_hash": probabilities.surface_hash,
        "feature_surface_hash": prelabel.feature_surface_hash,
        "all_models_seal_hash": models.all_models_seal_hash,
        "pre_support_seal_hash": decisions.pre_support_seal_hash,
        "all_decisions_seal_hash": decisions.all_decisions_seal_hash,
        "permutation_provenance_hash": decisions.permutation_provenance_hash,
        "sealed_result_hash": evaluation.sealed_result_hash,
        "closed_world_inventory": True,
        "content_index_validated_before_scientific_replay": True,
        "source_and_action_probability_arrays_reloaded": True,
        "exact_nine_probability_and_feature_surfaces_recomputed": True,
        "loco_utility_targets_recomputed_from_scoped_labels": True,
        "all_G_R_P_models_refit_with_H_q_e_exclusions": True,
        "all_495_method_decision_seals_recomputed": True,
        "terminal_sufficient_statistics_and_oracles_recomputed": True,
        "workstation_preflight_status": preflight["status"],
        "raw_labels_persisted": False,
        "per_case_bacc_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "routing_or_promotion_authorized": False,
        "may_feed_another_experiment": False,
    }
    checks = {
        **checks_unhashed,
        "validation_hash": canonical_hash(checks_unhashed),
    }
    _validate_run_state(path, validation_exists=validation_exists)
    if validation_exists and _read_object(
        path / "reports/validation_report.json"
    ) != checks:
        raise ProtocolError("Persisted validation report differs from replay.")
    return checks


def _validate_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    report = _read_object(root / "reports/workstation_preflight.json")
    gpus = report.get("gpus")
    packages = report.get("package_versions")
    expected_actionability = {
        "source_generation_devices": ["cuda:0", "cuda:1"],
        "probability_materialization_device": "cpu",
        "probability_materialization_workers": 4,
        "physical_actions_per_target_task": 18,
        "logical_actions_per_target": 19,
        "target_probability_cell_count": 1458,
        "target_unique_classifier_fit_count": 1458,
        "A0_A1_geometry_selected": False,
        "A1_sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
        "resume_strategy": runtime["resume_policy"],
    }
    if (
        report.get("schema_version")
        != "midogpp_label_free_workstation_preflight_v1"
        or report.get("status") != "PASS"
        or report.get("classifier_workers") != 4
        or report.get("blas_threads_per_classifier_worker") != 3
        or report.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or report.get("gpu_then_cpu_phase_order") is not True
        or tuple(report.get("generation_devices", ())) != ("cuda:0", "cuda:1")
        or report.get("persistent_gpu_workers") != 2
        or report.get("parent_cuda_initialized") is not False
        or report.get("tf32_enabled") is not False
        or report.get("amp_enabled") is not False
        or report.get("cuda_visible_devices") != "0,1"
        or report.get("thread_environment") != REQUIRED_THREAD_ENVIRONMENT
        or tuple(report.get("scratch_preference", ()))
        != tuple(runtime["scratch_preference"])
        or any(report.get(key) != value for key, value in expected_actionability.items())
        or not isinstance(packages, Mapping)
        or set(packages) != set(REQUIRED_DISTRIBUTIONS)
        or any(not isinstance(value, str) or not value for value in packages.values())
        or not isinstance(gpus, list)
        or len(gpus) != 2
        or tuple(row.get("index") for row in gpus if isinstance(row, Mapping))
        != (0, 1)
        or any(
            not isinstance(row, Mapping)
            or "RTX A5000" not in str(row.get("name"))
            or int(row.get("memory_free_mib", -1))
            < int(runtime["minimum_gpu_free_mib_per_device"])
            for row in gpus
        )
        or int(report.get("available_cpu_affinity_count", -1))
        < int(runtime["minimum_logical_cpu_count"])
        or int(report.get("physical_ram_bytes", -1))
        < int(runtime["minimum_physical_ram_bytes"])
        or int(report.get("disk_free_bytes_at_launch", -1))
        < int(runtime["minimum_artifact_disk_free_bytes"])
        or Path(str(report.get("disk_probe_path", ""))).resolve()
        != root.resolve()
        or int(runtime["classifier_workers"]) != 4
        or int(runtime["classifier_threads_per_worker"]) != 3
    ):
        raise ProtocolError("Workstation preflight report drifted.")
    return report


def _compare_members(
    observed_root: Path, replay_root: Path, members: tuple[str, ...]
) -> None:
    for member in members:
        observed = observed_root / member
        replayed = replay_root / member
        if not observed.is_file() or not replayed.is_file():
            raise ProtocolError(f"Replay member is absent: {member}.")
        if observed.read_bytes() != replayed.read_bytes():
            raise ProtocolError(f"Persisted member differs from replay: {member}.")


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read diagnostic JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Diagnostic JSON must be an object: {path}.")
    return value


def _reject_forbidden_persisted_fields(root: Path) -> None:
    forbidden_keys = {"labels", "raw_labels", "per_case_bacc"}
    for directory in ("manifests", "reports"):
        for path in sorted((root / directory).glob("*.json")):
            value = _read_object(path)
            if _contains_forbidden_key(value, forbidden_keys):
                raise ProtocolError(
                    f"Bundle persisted forbidden raw-label/per-case fields: {path}."
                )
    for path in sorted((root / "tables").glob("*.csv")):
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
        except (OSError, StopIteration, csv.Error) as exc:
            raise ProtocolError(f"Cannot inspect diagnostic table: {path}.") from exc
        if len(header) != len(set(header)) or forbidden_keys.intersection(header):
            raise ProtocolError(f"Diagnostic table has a forbidden/drifted header: {path}.")


def _contains_forbidden_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return bool(forbidden.intersection(str(key) for key in value)) or any(
            _contains_forbidden_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item, forbidden) for item in value)
    return False


def _validate_run_state(root: Path, *, validation_exists: bool) -> None:
    state = _read_object(root / "reports/run_state.json")
    common = {
        "schema_version": "midogpp_fixed_bank_actionability_recoverability_run_state_v1",
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
    }
    status, phase = state.get("status"), state.get("phase")
    valid_running = {
        "CLOSED_WORLD_CONTENT_FIRST_VALIDATION",
        "TERMINAL_PHASE_VALIDATION_RECOVERY",
        "CLOSED_WORLD_CONTENT_FIRST_VALIDATION_RECOVERY",
    }
    if (
        any(state.get(key) != value for key, value in common.items())
        or status not in {"RUNNING", "COMPLETE"}
        or (status == "RUNNING" and phase not in valid_running)
        or (status == "COMPLETE" and phase != "COMPLETE")
        or (status == "COMPLETE" and not validation_exists)
    ):
        raise ProtocolError("Actionability run state is not validatable.")


def _centers() -> tuple[str, ...]:
    from .constants import MIDOGPP_CENTERS

    return MIDOGPP_CENTERS


__all__ = ("validate_fixed_bank_actionability_recoverability_bundle",)
