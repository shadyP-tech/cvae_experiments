"""Thin, phase-ordered runner for the terminal consumed-test diagnostic."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import (
    assert_closed_world,
    cleanup_owned_atomic_temps,
    write_content_index,
)
from .persistence import (
    persist_all_decisions,
    persist_and_validate_models,
    persist_initial_surfaces,
    persist_label_capability_report,
    persist_postseal_results,
    persist_pre_support_decisions,
    persist_prelabel_surfaces,
    persist_validation_report,
)
from .protocol import canonical_consumed_test_protocol
from .reports import leakage_report_payload
from .runner_dependencies import FixedBankActionabilityRecoverabilityDependencies
from .sealing import (
    record_durable_model_seals,
    record_durable_preevaluation_seals,
    record_durable_pre_support_seals,
)
from .runner_runtime import (
    assert_launch_files as _assert_launch_files,
    assert_persisted_prelabel as _assert_persisted_prelabel,
    assert_workspace_resolved_paths as _assert_workspace_resolved_paths,
    cleanup_validated_local_stage as _cleanup_validated_local_stage,
    enter_cuda_free_cpu_phase as _enter_cuda_free_cpu_phase,
    exclusive_run_lock as _exclusive_run_lock,
    observe as _observe,
    recover_if_possible as _recover_if_possible,
    validate_bundle as _validate_bundle,
    write_state as _write_state,
)

def run_fixed_bank_actionability_recoverability(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    dependencies: FixedBankActionabilityRecoverabilityDependencies | None = None,
) -> Path:
    """Execute the direct-parent, original-six consumed-test authorization once."""

    root = Path(artifact_root or getattr(config, "artifact_root"))
    deps = dependencies or FixedBankActionabilityRecoverabilityDependencies()
    protocol = canonical_consumed_test_protocol()
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)

    from .aggregation import aggregate_exact_nine_probabilities
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
        materialize_probabilities,
        materialize_sources,
        run_label_free_workstation_preflight,
        runtime_summary_payload,
        seed_probability_rows,
        stage_sources_for_cpu,
    )
    from .inputs import (
        assert_input_fence,
        load_label_free_test_frame,
        load_validated_locks,
        validate_active_diagnostic_workspace_binding,
        validate_pre_gpu_firewall,
        validate_workspace_provenance,
    )
    from .label_capabilities import ActionabilityLabelCapabilityManager

    with _exclusive_run_lock(root):
        cleanup_owned_atomic_temps(root)
        assert_closed_world(root, allow_incomplete=True)
        capability_path = root / "reports/label_capability_report.json"
        terminal_path = root / "manifests/sealed_terminal_evaluation.json"
        if (
            capability_path.is_file()
            and read_json(capability_path).get("evaluation_labels_opened") is True
            and not terminal_path.is_file()
        ):
            _write_state(
                deps,
                root,
                status="FAILED",
                phase="TERMINAL_LABEL_PHASE_NONREPLAYABLE",
                error=(
                    "ProtocolError: terminal labels were opened but the terminal "
                    "evaluation commit marker is absent"
                ),
            )
            raise ProtocolError(
                "Terminal-label-phase recovery is forbidden: labels were opened "
                "but no sealed terminal evaluation exists. Preserve the bundle "
                "for audit; do not automatically refit or reopen labels."
            )
        recovered = _recover_if_possible(root, config=config, deps=deps, protocol=protocol)
        if recovered is not None:
            return recovered

        phase = "INITIALIZING"
        _write_state(deps, root, status="RUNNING", phase=phase)
        try:
            _observe(deps, "input_fence")
            (deps.validate_inputs or assert_input_fence)(config)
            workspace = (
                deps.validate_workspace
                or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_test_frame)(config)
            firewall = dict(
                (deps.validate_firewall or validate_pre_gpu_firewall)(
                    config, frame, locks
                )
            )
            firewall["workspace_binding"] = workspace
            partition = (deps.build_partition or build_case_partition)(
                frame, config=config
            )
            (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                protocol=protocol,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partition=partition,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "preflight")
            preflight = (deps.preflight or run_label_free_workstation_preflight)(
                root, runtime=getattr(config, "runtime")
            )

            phase = "FROZEN_SOURCE_STREAMS_TWO_GPU"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "gpu_source_streams")
            canonical_source = (deps.materialize_source or materialize_sources)(
                config, getattr(locks, "generation"), root=root
            )
            source_for_cpu = canonical_source
            staging: dict[str, object] = {
                "attempted": True,
                "used": False,
                "status": "CANONICAL_FALLBACK",
            }
            try:
                source_for_cpu = (deps.stage_source or stage_sources_for_cpu)(
                    canonical_source, config=config, root=root
                )
            except (OSError, ProtocolError) as exc:
                staging["failure"] = f"{type(exc).__name__}: {exc}"
            else:
                staging.update(
                    {
                        "used": source_for_cpu is not canonical_source,
                        "status": (
                            "STAGED_LOCAL_CPU_CACHE"
                            if source_for_cpu is not canonical_source
                            else "CANONICAL_ALREADY_LOCAL"
                        ),
                    }
                )

            _enter_cuda_free_cpu_phase()
            phase = "GLOBAL_ACTION_PROBABILITY_AND_PRELABEL_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "cuda_free_probability_and_prelabel_seal")
            prediction = (deps.materialize_predictions or materialize_probabilities)(
                config, source_for_cpu, frame, partition, root=root
            )
            seed_rows = (deps.build_seed_rows or seed_probability_rows)(prediction)
            probabilities = (
                deps.aggregate_probabilities or aggregate_exact_nine_probabilities
            )(seed_rows)
            prelabel = (deps.build_prelabel or build_prelabel_products)(
                probabilities, protocol_contract_hash=protocol.contract_hash
            )
            (deps.persist_prelabel or persist_prelabel_surfaces)(
                root,
                prediction_capability=prediction,
                seed_rows=seed_rows,
                probability_surface=probabilities,
                prelabel=prelabel,
            )
            _assert_persisted_prelabel(root, prediction=prediction, prelabel=prelabel)
            action_library_hash = str(
                getattr(getattr(prediction, "store"), "action_library_hash")
            )
            if (
                read_json(root / "manifests/action_library.json").get(
                    "action_library_hash"
                )
                != action_library_hash
            ):
                raise ProtocolError("Action-library runtime and manifest hashes differ.")

            manager = (
                deps.build_label_manager or ActionabilityLabelCapabilityManager
            )(
                getattr(config, "test_manifest_path"),
                frame,
                partition,
                global_prediction_seal_hash=str(getattr(prediction, "seal_hash")),
                label_free_feature_seal_hash=str(
                    getattr(prelabel, "feature_surface_hash")
                ),
                action_library_hash=action_library_hash,
            )

            phase = "STRICT_OUTER_H_NESTED_QUERY_G_R_P_MODELS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "loco_donor_labels_and_models")
            utility_products: list[object] = []
            target_products: list[object] = []
            for target in _centers():
                labels = manager.open_loco_donor_labels(target)
                utility = (deps.build_utility or build_loco_utility_product)(
                    probabilities, labels, outer_target_center=target
                )
                target_product = (
                    deps.fit_target_model or fit_target_model_product
                )(
                    prelabel,
                    utility,
                    workers=int(getattr(config, "runtime")["model_workers"]),
                    threads_per_worker=int(
                        getattr(config, "runtime")["model_threads_per_worker"]
                    ),
                    start_method=str(
                        getattr(config, "runtime")["multiprocessing_start_method"]
                    ),
                )
                utility_products.append(utility)
                target_products.append(target_product)
            models = (deps.combine_models or combine_model_products)(
                prelabel, tuple(target_products)
            )
            (deps.persist_models or persist_and_validate_models)(
                root,
                products=models,
                utility_products=tuple(utility_products),
                target_products=tuple(target_products),
            )
            (deps.record_models or record_durable_model_seals)(manager, models)
            _observe(deps, "all_G_R_P_models_durable_before_support")

            phase = "PRE_SUPPORT_B_U_G_R_P_DECISION_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            pre_support = (
                deps.build_pre_support or build_pre_support_decision_products
            )(models, partition)
            (deps.persist_pre_support or persist_pre_support_decisions)(
                root, products=pre_support
            )
            (deps.record_pre_support or record_durable_pre_support_seals)(
                manager, pre_support
            )
            _observe(deps, "all_405_pre_support_decisions_durable")

            phase = "FORTY_FIVE_SUPPORT_FOLDS_AND_ALL_METHOD_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "same_H_support_S_y")
            support_products: list[object] = []
            for target in _centers():
                for fold_ordinal in range(5):
                    labels = manager.open_fold_support_labels(target, fold_ordinal)
                    support_products.append(
                        (deps.build_support_fold or build_support_fold_product)(
                            probabilities,
                            partition,
                            labels,
                            target_center=target,
                            fold_ordinal=fold_ordinal,
                        )
                    )
            decisions = (deps.combine_decisions or combine_decision_products)(
                pre_support, tuple(support_products), partition
            )
            (deps.persist_decisions or persist_all_decisions)(
                root, products=decisions
            )
            (deps.record_preevaluation or record_durable_preevaluation_seals)(
                manager, decisions
            )
            _observe(deps, "all_495_decisions_and_permutation_durable")

            phase = "TERMINAL_POOLED_EXACT_BACC_AND_ORACLES"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "terminal_evaluation_labels")
            terminal_labels = manager.open_oof_evaluation_labels()
            capability_report = manager.access_report()
            (deps.persist_capability or persist_label_capability_report)(
                root, capability_report
            )
            if read_json(root / "reports/label_capability_report.json") != dict(
                capability_report
            ):
                raise ProtocolError(
                    "Terminal label-capability report was not durable before scoring."
                )
            evaluation = (deps.evaluate or _evaluate_terminal)(
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
                bootstrap_workers=int(getattr(config, "runtime")["bootstrap_workers"]),
                bootstrap_threads_per_worker=int(
                    getattr(config, "runtime")["bootstrap_threads_per_worker"]
                ),
                multiprocessing_start_method=str(
                    getattr(config, "runtime")["multiprocessing_start_method"]
                ),
            )
            leakage = leakage_report_payload(
                prediction_seal_hash=str(getattr(prediction, "seal_hash")),
                feature_seal_hash=str(getattr(prelabel, "feature_surface_hash")),
                action_library_hash=action_library_hash,
                model_seal_count=9 * 2 * 3,
                decision_count=len(getattr(decisions, "all_decision_hashes")),
                capability_report=capability_report,
            )
            runtime_summary = runtime_summary_payload(
                source_cache=canonical_source,
                prediction_capability=prediction,
                local_staging={**staging, "workstation_preflight": dict(preflight)},
                runtime=getattr(config, "runtime"),
            )
            (deps.persist_postseal or persist_postseal_results)(
                root,
                evaluation=evaluation,
                capability_report=capability_report,
                leakage_report=leakage,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "validation")
            (deps.write_index or write_content_index)(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.contract_hash,
            )
            checks = (deps.validate_bundle or _validate_bundle)(root, config=config)
            (deps.persist_validation or persist_validation_report)(root, checks)
            _write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            if staging.get("used") is True:
                (deps.cleanup_staging or _cleanup_validated_local_stage)(
                    config, canonical_source=canonical_source
                )
            return root
        except BaseException as exc:
            _write_state(
                deps,
                root,
                status="FAILED",
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _evaluate_terminal(*args: object, **kwargs: object) -> object:
    from .terminal import evaluate_terminal

    return evaluate_terminal(*args, **kwargs)

def _centers() -> tuple[str, ...]:
    from .constants import MIDOGPP_CENTERS

    return MIDOGPP_CENTERS

__all__ = (
    "FixedBankActionabilityRecoverabilityDependencies",
    "run_fixed_bank_actionability_recoverability",
)
