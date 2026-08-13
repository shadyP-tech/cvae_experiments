"""Phase-ordered workstation runner for the terminal CDCA diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .actions import action_library_by_target, build_action_library
from .bundle import write_content_index
from .constants import (
    B_ACTION_ID,
    CENTERS,
    DESCRIPTIVE_METHOD_IDS,
    PRE_TERMINAL_METHOD_IDS,
    U_ACTION_ID,
    candidate_sources,
)
from .donor_priors import compute_donor_priors
from .ensemble import compose_fixed_action_predictions
from .execution_adapter import (
    build_exact_nine_surface,
    cleanup_validated_scratch,
    materialize_probabilities,
    materialize_sources,
    physical_partition_hash,
    probability_index_rows,
    run_label_free_workstation_preflight,
    runtime_summary_payload,
)
from .features import build_label_free_case_candidate_features
from .hashing import canonical_hash
from .held_case_plans import build_held_case_plans, seal_held_case_plans
from .label_capabilities import DirectionalCorrectnessLabelFirewall
from .persistence import (
    persist_initial_surfaces,
    persist_physical_prelabel,
    persist_plans_and_features,
    persist_route_science,
    persist_terminal,
    persist_validation_report,
)
from .protocol import build_frozen_science_protocol
from .reports import leakage_report_payload, publication_decision_payload
from .runner_dependencies import CaseDirectionalRunnerDependencies
from .runner_runtime import (
    assert_cuda_free_cpu_phase,
    assert_launch_files,
    assert_no_foreign_or_partial_state,
    assert_workspace_resolved_paths,
    enter_cuda_free_cpu_phase,
    execute_route_jobs,
    exclusive_run_lock,
    observe,
    reject_existing_run_state,
    write_state,
)
from .scoring import score_directional_gains


def run_fixed_bank_case_directional_correctness_abstention_router(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    dependencies: CaseDirectionalRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or getattr(config, "artifact_root"))
    deps = dependencies or CaseDirectionalRunnerDependencies()
    protocol = build_frozen_science_protocol()
    assert_workspace_resolved_paths(config, root=root)
    for directory in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    assert_launch_files(root, config)
    from .inputs import (
        assert_input_fence,
        load_label_free_test_frame,
        load_validated_locks,
        validate_active_diagnostic_workspace_binding,
        validate_pre_gpu_firewall,
        validate_workspace_provenance,
    )

    with exclusive_run_lock(root):
        reject_existing_run_state(root)
        assert_no_foreign_or_partial_state(root)
        phase = "INPUT_ADMISSION"
        write_state(deps, root, status="RUNNING", phase=phase)
        try:
            observe(deps, "input_admission")
            (deps.validate_inputs or assert_input_fence)(config)
            workspace = (
                deps.validate_workspace or validate_active_diagnostic_workspace_binding
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
            actions = tuple((deps.build_actions or build_action_library)())
            (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                protocol=protocol,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                actions=actions,
            )

            phase = "WORKSTATION_PREFLIGHT"
            write_state(deps, root, status="RUNNING", phase=phase)
            preflight = (deps.preflight or run_label_free_workstation_preflight)(
                root, runtime=getattr(config, "runtime")
            )

            phase = "TWO_PERSISTENT_A5000_GENERATION_WORKERS"
            write_state(deps, root, status="RUNNING", phase=phase)
            source_caches = (deps.materialize_source or materialize_sources)(
                config, locks.generation, root=root
            )
            source_for_cpu = getattr(source_caches, "local", source_caches)

            enter_cuda_free_cpu_phase()
            assert_cuda_free_cpu_phase()
            phase = "FOUR_SPAWNED_CPU_WORKERS_THREE_BLAS_EXACT_810"
            write_state(deps, root, status="RUNNING", phase=phase)
            prediction = (deps.materialize_predictions or materialize_probabilities)(
                config,
                source_for_cpu,
                frame,
                partition_hash=physical_partition_hash(frame),
                action_library=action_library_by_target(),
                root=root,
            )
            surface = (deps.build_probability_surface or build_exact_nine_surface)(
                prediction
            )
            compact_index = (deps.build_probability_index or probability_index_rows)(
                prediction
            )
            physical_seal = (deps.persist_prelabel or persist_physical_prelabel)(
                root,
                prediction=prediction,
                probability_index=compact_index,
                probability_surface_hash=str(surface.surface_hash),
            )

            phase = "LABEL_FREE_FEATURE_AND_218_PLAN_SEALS"
            write_state(deps, root, status="RUNNING", phase=phase)
            plans = tuple(
                (deps.build_plans or build_held_case_plans)(
                    frame.rows,
                    probability_surface_hash=str(surface.surface_hash),
                )
            )
            plan_seal = seal_held_case_plans(
                plans, probability_surface_hash=str(surface.surface_hash)
            )
            features = tuple(
                (deps.build_features or build_label_free_case_candidate_features)(
                    surface
                )
            )
            persisted_plan_seal, feature_seal = persist_plans_and_features(
                root,
                plans=plans,
                plan_seal=plan_seal,
                features=features,
                physical_prelabel_seal_hash=str(physical_seal["seal_hash"]),
            )
            label_firewall = (
                deps.build_label_firewall or DirectionalCorrectnessLabelFirewall
            )(
                plan_seal,
                lambda allowed: _read_manifest_labels(
                    config, frame, allowed_keys=allowed
                ),
            )

            phase = "DONOR_PRIORS_AND_ROUTE_LOCAL_H_MINUS_C_MODELS"
            write_state(deps, root, status="RUNNING", phase=phase)
            all_priors: list[object] = []
            priors_by_target: dict[str, tuple[object, ...]] = {}
            for target in CENTERS:
                gains_by_source = {}
                for source in candidate_sources(target):
                    labels = label_firewall.open_donor_labels(target, source)
                    gains_by_source[source] = score_directional_gains(surface, labels)
                priors = tuple(
                    (deps.compute_priors or compute_donor_priors)(
                        gains_by_source, heldout_center=target
                    )
                )
                priors_by_target[target] = priors
                all_priors.extend(priors)

            jobs = []
            for plan in plans:
                support_labels = label_firewall.open_route_support_labels(
                    plan.target_center, plan.case_id, plan_hash=plan.plan_hash
                )
                route_features = tuple(
                    row
                    for row in features
                    if row.target_center == plan.target_center
                    and row.case_id in {*plan.support_case_ids, plan.case_id}
                )
                jobs.append(
                    {
                        "plan": plan,
                        "support_labels": support_labels,
                        "donor_priors": priors_by_target[plan.target_center],
                        "route_features": route_features,
                    }
                )
            route_results = tuple(
                (deps.execute_route_jobs or execute_route_jobs)(
                    surface,
                    jobs,
                    workers=int(getattr(config, "runtime")["route_model_workers"]),
                    threads_per_worker=int(
                        getattr(config, "runtime")["classifier_threads_per_worker"]
                    ),
                )
            )
            responses = tuple(row for result in route_results for row in result.support_responses)
            models = tuple(row for result in route_results for row in result.model_fits)
            scores = tuple(row for result in route_results for row in result.candidate_scores)
            decisions = tuple(row for result in route_results for row in result.decisions)
            routed_predictions = tuple(
                row for result in route_results for row in result.predictions
            )
            fixed = {
                B_ACTION_ID: compose_fixed_action_predictions(surface, method_id=B_ACTION_ID),
                U_ACTION_ID: compose_fixed_action_predictions(surface, method_id=U_ACTION_ID),
            }
            primary_predictions = tuple(
                row
                for method in PRE_TERMINAL_METHOD_IDS
                for row in (
                    fixed[method]
                    if method in fixed
                    else tuple(
                        value for value in routed_predictions if value.method_id == method
                    )
                )
            )
            descriptive_predictions = tuple(
                row
                for method in DESCRIPTIVE_METHOD_IDS
                for row in routed_predictions
                if row.method_id == method
            )
            if (
                len(primary_predictions) != 9_928 * len(PRE_TERMINAL_METHOD_IDS)
                or len(descriptive_predictions) != 9_928 * len(DESCRIPTIVE_METHOD_IDS)
            ):
                raise ProtocolError("Case-directional prediction topology drifted.")
            for plan in plans:
                route_payload = {
                    "decisions": [
                        row.to_payload()
                        for row in decisions
                        if (row.target_center, row.case_id) == plan.key
                    ],
                    "predictions": [
                        row.to_payload()
                        for row in (*primary_predictions, *descriptive_predictions)
                        if (row.target_center, row.case_id) == plan.key
                    ],
                }
                label_firewall.record_route_decision_seal(
                    plan.target_center, plan.case_id, canonical_hash(route_payload)
                )
            barrier = label_firewall.decision_barrier_payload()
            _, _, _, aggregate_seal = (
                deps.persist_science or persist_route_science
            )(
                root,
                support_responses=responses,
                donor_priors=all_priors,
                model_fits=models,
                candidate_scores=scores,
                decisions=decisions,
                method_predictions=primary_predictions,
                descriptive_predictions=descriptive_predictions,
                held_case_plan_seal_hash=str(persisted_plan_seal["seal_hash"]),
                held_case_feature_seal_hash=str(feature_seal["seal_hash"]),
                route_barrier=barrier,
            )
            aggregate_bindings = aggregate_seal["bindings"]
            if not isinstance(aggregate_bindings, Mapping):
                raise ProtocolError("Case-directional aggregate bindings absent.")
            label_firewall.record_aggregate_plan_decision_seal(
                str(aggregate_seal["seal_hash"]),
                plan_seal_hash=plan_seal.plan_seal_hash,
                decision_barrier_hash=str(barrier["decision_barrier_hash"]),
            )

            phase = "TERMINAL_LABELS_REPORTS_AND_REPLAY"
            write_state(deps, root, status="RUNNING", phase=phase)
            terminal_labels = (
                deps.open_terminal_labels or label_firewall.open_terminal_labels
            )()
            capability = label_firewall.report_payload()
            terminal = (deps.evaluate_terminal or _evaluate_terminal)(
                probability_surface=surface,
                method_predictions=primary_predictions,
                descriptive_predictions=descriptive_predictions,
                decisions=decisions,
                aggregate_plan_decision_seal_hash=str(aggregate_seal["seal_hash"]),
                terminal_labels=terminal_labels,
            )
            terminal_seal = terminal["terminal_seal"]
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction.seal_hash,
                physical_prelabel_seal_hash=str(physical_seal["seal_hash"]),
                held_case_feature_seal_hash=str(feature_seal["seal_hash"]),
                aggregate_plan_decision_seal_hash=str(aggregate_seal["seal_hash"]),
                capability_report=capability,
            )
            publication = publication_decision_payload(
                str(terminal_seal["seal_hash"]),
                descriptive_summary=terminal["descriptive_summary"],
            )
            runtime_summary = runtime_summary_payload(
                source_cache=source_caches,
                prediction=prediction,
                preflight=preflight,
                runtime=getattr(config, "runtime"),
            )
            (deps.persist_terminal or persist_terminal)(
                root,
                result=terminal,
                capability_report=capability,
                leakage_report=leakage,
                publication_decision=publication,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_TWO_FRESH_PROCESS_VALIDATION"
            write_state(deps, root, status="RUNNING", phase=phase)
            (deps.write_content_index or write_content_index)(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.protocol_hash,
            )
            checks = (deps.validate_bundle or _validate_bundle)(
                root, config=config, allow_pending_validation=True
            )
            if deps.validate_bundle is None:
                from .fresh_process_validation import require_two_fresh_process_validations

                checks = require_two_fresh_process_validations(
                    root, expected_checks=checks
                )
            (deps.persist_validation or persist_validation_report)(root, checks)
            write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            cleanup_validated_scratch(config)
            return root
        except BaseException as exc:
            write_state(
                deps,
                root,
                status="FAILED",
                phase=phase,
                error=str(exc),
                error_class=type(exc).__name__,
            )
            raise


def _read_manifest_labels(
    config: object,
    frame: object,
    *,
    allowed_keys: frozenset[tuple[str, str, str]],
) -> Sequence[object]:
    from .products import BinaryLabel

    universe = {(row.center, row.case_id, row.sample_id): row for row in frame.rows}
    if not allowed_keys or not set(allowed_keys) <= set(universe):
        raise ProtocolError("Case-directional label grant escapes sealed rows.")
    ordered = tuple(
        key
        for row in frame.rows
        if (key := (row.center, row.case_id, row.sample_id)) in allowed_keys
    )
    requested = {key: universe[key] for key in ordered}
    found = {}
    manifest_path = Path(getattr(config, "test_manifest_path"))
    manifest_hash = str(getattr(config, "expected_manifest_sha256"))
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for ordinal, raw in enumerate(csv.DictReader(handle)):
                key = (
                    str(raw.get("center", "")),
                    str(raw.get("case_id", "")),
                    evaluation_row_id(manifest_hash, ordinal),
                )
                if key not in requested:
                    continue
                if requested[key].manifest_row_index != ordinal or key in found:
                    raise ProtocolError("Case-directional manifest order drifted.")
                found[key] = BinaryLabel(*key, int(raw["label"]), "scoped_loader")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot load case-directional labels.") from exc
    if set(found) != set(requested):
        raise ProtocolError("Case-directional label coverage drifted.")
    return tuple(found[key] for key in ordered)


def _evaluate_terminal(**kwargs: object) -> Mapping[str, object]:
    from .terminal import evaluate_terminal

    return evaluate_terminal(**kwargs)


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import (
        validate_fixed_bank_case_directional_correctness_abstention_router_bundle,
    )

    return validate_fixed_bank_case_directional_correctness_abstention_router_bundle(
        root, **kwargs
    )


__all__ = ("run_fixed_bank_case_directional_correctness_abstention_router",)
