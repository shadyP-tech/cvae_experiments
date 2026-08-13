"""Phase-ordered runner for the terminal whole-case LOO DCSE diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

from ....data.contract.stage70_target_evaluation.contracts import evaluation_row_id
from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .actions import action_library_by_target, build_action_library
from .bundle import write_content_index
from .donor_priors import compute_donor_priors
from .endpoint_library import build_endpoint_arms
from .ensemble import (
    DESCRIPTIVE_METHOD_IDS,
    compose_control_predictions,
    compose_descriptive_control_predictions,
    compose_method_predictions,
    fixed_physical_method_predictions,
)
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
from .hashing import canonical_hash
from .label_capabilities import LabelCapabilityFirewall
from .loo_plans import build_whole_case_loo_plans, seal_loo_plans
from .persistence import (
    persist_decisions,
    persist_donor_priors,
    persist_endpoint_library,
    persist_initial_surfaces,
    persist_loo_products,
    persist_physical_prelabel,
    persist_terminal,
    persist_validation_report,
)
from .protocol import canonical_consumed_test_protocol
from .reports import leakage_report_payload, publication_decision_payload
from .runner_dependencies import DirectionalShrinkageRunnerDependencies
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
from .scoring import score_case_action_confusions, score_loo_directional_gains
from .nulls import build_candidate_identity_null_plan
from .constants import CENTERS, PRE_TERMINAL_METHOD_IDS, physical_action_ids


def run_fixed_bank_loo_directional_shrinkage_ensemble(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    dependencies: DirectionalShrinkageRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or getattr(config, "artifact_root"))
    deps = dependencies or DirectionalShrinkageRunnerDependencies()
    protocol = canonical_consumed_test_protocol()
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
                deps.validate_workspace
                or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_test_frame)(config)
            firewall_report = dict(
                (deps.validate_firewall or validate_pre_gpu_firewall)(
                    config, frame, locks
                )
            )
            firewall_report["workspace_binding"] = workspace
            actions = tuple((deps.build_actions or build_action_library)())
            action_manifest = (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                protocol=protocol,
                provenance=provenance,
                frame=frame,
                firewall=firewall_report,
                actions=actions,
            )

            phase = "WORKSTATION_PREFLIGHT"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "workstation_preflight")
            preflight = (
                deps.preflight or run_label_free_workstation_preflight
            )(root, runtime=getattr(config, "runtime"))

            phase = "TWO_PERSISTENT_A5000_GENERATION_WORKERS"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "gpu_source_generation")
            source_caches = (deps.materialize_source or materialize_sources)(
                config, locks.generation, root=root
            )
            source_for_cpu = getattr(source_caches, "local", source_caches)

            enter_cuda_free_cpu_phase()
            assert_cuda_free_cpu_phase()
            phase = "FOUR_SPAWNED_CPU_WORKERS_THREE_BLAS_EXACT_810"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "cuda_free_probability_materialization")
            prediction = (
                deps.materialize_predictions or materialize_probabilities
            )(
                config,
                source_for_cpu,
                frame,
                partition_hash=physical_partition_hash(frame),
                action_library=action_library_by_target(),
                root=root,
            )
            surface = (
                deps.build_probability_surface or build_exact_nine_surface
            )(prediction)
            compact_index = (
                deps.build_probability_index or probability_index_rows
            )(prediction)
            physical_seal = (
                deps.persist_prelabel or persist_physical_prelabel
            )(
                root,
                prediction=prediction,
                probability_index=compact_index,
                probability_surface_hash=str(surface.surface_hash),
            )
            if read_json(root / "manifests/physical_prelabel_seal.json") != dict(
                physical_seal
            ):
                raise ProtocolError("Directional-shrinkage prelabel seal readback failed.")

            phase = "GLOBAL_218_WHOLE_CASE_LOO_PLAN_SEAL"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "global_loo_plan_seal")
            plans = tuple(
                (deps.build_plans or build_whole_case_loo_plans)(
                    frame.rows,
                    probability_surface_hash=str(surface.surface_hash),
                )
            )
            plan_seal = seal_loo_plans(
                plans, probability_surface_hash=str(surface.surface_hash)
            )
            label_firewall = (
                deps.build_label_firewall or LabelCapabilityFirewall
            )(
                plan_seal,
                lambda allowed: _read_manifest_labels(
                    config, frame, allowed_keys=allowed
                ),
            )

            phase = "DONOR_PRIORS_AND_218_ROUTE_DECISIONS"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "donor_and_route_scoped_label_capabilities")
            donor_counts: list[object] = []
            all_priors: list[object] = []
            priors_by_target: dict[str, tuple[object, ...]] = {}
            for target in tuple(action_library_by_target()):
                counts_by_source: dict[str, tuple[object, ...]] = {}
                for source_id in (
                    action.selected_source
                    for action in action_library_by_target()[target]
                    if action.selected_source is not None
                ):
                    labels = label_firewall.open_donor_labels(target, source_id)
                    counts = tuple(
                        (deps.score_case_actions or score_case_action_confusions)(
                            surface, labels
                        )
                    )
                    counts_by_source[source_id] = counts
                    donor_counts.extend(counts)
                priors = tuple(
                    (deps.compute_priors or compute_donor_priors)(
                        counts_by_source, heldout_center=target
                    )
                )
                priors_by_target[target] = priors
                all_priors.extend(priors)

            route_jobs: list[dict[str, object]] = []
            for plan in plans:
                support_labels = label_firewall.open_route_support_labels(
                    plan.target_center, plan.case_id, plan_hash=plan.plan_hash
                )
                route_jobs.append(
                    {
                        "plan": plan,
                        "support_labels": tuple(support_labels),
                        "donor_priors": priors_by_target[plan.target_center],
                    }
                )

            route_results = tuple(
                (deps.execute_route_jobs or execute_route_jobs)(
                    surface,
                    route_jobs,
                    workers=int(getattr(config, "runtime")["classifier_workers"]),
                    threads_per_worker=int(
                        getattr(config, "runtime")["classifier_threads_per_worker"]
                    ),
                )
            )
            if len(route_results) != len(plans):
                raise ProtocolError("Directional-shrinkage route result count drifted.")
            repeated_case_counts: list[object] = []
            all_gains: list[object] = []
            all_decisions: list[object] = []
            all_control_decisions: list[object] = []
            for plan, route in zip(plans, route_results, strict=True):
                if route.plan != plan:
                    raise ProtocolError("Directional-shrinkage route result order drifted.")
                repeated_case_counts.extend(route.counts)
                all_gains.extend(route.gains)
                all_decisions.extend(route.endpoint_decisions)
                all_control_decisions.extend(route.control_decisions)

            all_case_counts = _deduplicate_case_action_confusions(
                repeated_case_counts
            )

            plan_artifact_seal = (
                deps.persist_plans or persist_loo_products
            )(
                root,
                plans=plans,
                case_action_confusions=all_case_counts,
                directional_gains=all_gains,
                physical_prelabel_seal_hash=str(physical_seal["seal_hash"]),
            )
            prior_seal = (deps.persist_priors or persist_donor_priors)(
                root,
                priors=all_priors,
                loo_plan_seal_hash=str(plan_artifact_seal["seal_hash"]),
            )
            endpoints = tuple((deps.build_endpoints or build_endpoint_arms)())
            endpoint_seal = (
                deps.persist_endpoints or persist_endpoint_library
            )(
                root,
                endpoints=endpoints,
                donor_prior_seal_hash=str(prior_seal["seal_hash"]),
            )
            endpoint_predictions = tuple(
                (deps.compose_predictions or compose_method_predictions)(
                    surface,
                    tuple(row for row in all_decisions if row.method_id == method),
                    method_id=method,
                )
                for method in ("DCSE_LOO", "G_directional_matched")
            )
            control_predictions = tuple(
                compose_control_predictions(
                    surface,
                    tuple(
                        row
                        for row in all_control_decisions
                        if row.method_id == method
                    ),
                    method_id=method,
                )
                for method in ("DLOO_raw", "LOO_frequency_committee")
            )
            predictions_by_method = {
                "B": fixed_physical_method_predictions(surface, method_id="B"),
                "U": fixed_physical_method_predictions(surface, method_id="U"),
                "DCSE_LOO": endpoint_predictions[0],
                "G_directional_matched": endpoint_predictions[1],
                "DLOO_raw": control_predictions[0],
                "LOO_frequency_committee": control_predictions[1],
            }
            method_predictions = tuple(
                row
                for method in PRE_TERMINAL_METHOD_IDS
                for row in predictions_by_method[method]
            )
            descriptive_control_predictions = tuple(
                compose_descriptive_control_predictions(
                    surface,
                    tuple(
                        row
                        for row in all_decisions
                        if row.method_id == "DCSE_LOO"
                    ),
                )
            )
            if (
                len(descriptive_control_predictions) != 9_928 * 5
                or tuple(
                    dict.fromkeys(
                        row.method_id for row in descriptive_control_predictions
                    )
                )
                != DESCRIPTIVE_METHOD_IDS
            ):
                raise ProtocolError(
                    "Directional-shrinkage descriptive-control topology drifted."
                )
            for plan in plans:
                key = (plan.target_center, plan.case_id)
                route_endpoint = tuple(
                    row
                    for row in all_decisions
                    if (row.target_center, row.case_id) == key
                )
                route_controls = tuple(
                    row
                    for row in all_control_decisions
                    if (row.target_center, row.case_id) == key
                )
                route_predictions = tuple(
                    row
                    for row in method_predictions
                    if (row.target_center, row.case_id) == key
                )
                route_descriptive = tuple(
                    row
                    for row in descriptive_control_predictions
                    if (row.target_center, row.case_id) == key
                )
                route_hash = canonical_hash(
                    {
                        "endpoint_decisions": [
                            row.to_payload() for row in route_endpoint
                        ],
                        "control_decisions": [
                            row.to_payload() for row in route_controls
                        ],
                        "preterminal_method_predictions": [
                            row.to_payload() for row in route_predictions
                        ],
                        "descriptive_control_predictions": [
                            row.to_payload() for row in route_descriptive
                        ],
                    }
                )
                label_firewall.record_route_decision_seal(
                    plan.target_center, plan.case_id, route_hash
                )
            route_barrier = label_firewall.decision_barrier_payload()
            null_plan = (
                deps.build_null_plan or build_candidate_identity_null_plan
            )(
                tuple((plan.target_center, plan.case_id) for plan in plans),
                seed=int(getattr(config, "nulls")["seed"]),
                replicates=int(getattr(config, "nulls")["replicates"]),
            )
            _decision_seal, aggregate_seal = (
                deps.persist_decisions or persist_decisions
            )(
                root,
                decisions=all_decisions,
                control_decisions=all_control_decisions,
                predictions=method_predictions,
                descriptive_control_predictions=descriptive_control_predictions,
                loo_plan_seal_hash=str(plan_artifact_seal["seal_hash"]),
                global_plan_seal_hash=plan_seal.plan_seal_hash,
                donor_prior_seal_hash=str(prior_seal["seal_hash"]),
                endpoint_library_seal_hash=str(endpoint_seal["seal_hash"]),
                route_decision_barrier=route_barrier,
                null_plan=null_plan,
            )
            if read_json(
                root / "manifests/aggregate_plan_decision_seal.json"
            ) != dict(aggregate_seal):
                raise ProtocolError("Directional-shrinkage aggregate barrier readback failed.")
            aggregate_readback = read_json(
                root / "manifests/aggregate_plan_decision_seal.json"
            )
            aggregate_bindings = aggregate_readback.get("bindings")
            if not isinstance(aggregate_bindings, Mapping):
                raise ProtocolError("Directional-shrinkage aggregate bindings absent.")
            if (
                aggregate_bindings.get("global_plan_seal_hash")
                != plan_seal.plan_seal_hash
                or aggregate_bindings.get("route_decision_barrier_hash")
                != route_barrier["decision_barrier_hash"]
                or aggregate_bindings.get("candidate_identity_null_plan_hash")
                != null_plan.plan_hash
                or aggregate_bindings.get(
                    "candidate_identity_null_permutation_sha256"
                )
                != null_plan.permutation_sha256
            ):
                raise ProtocolError(
                    "Directional-shrinkage aggregate barrier binding drifted."
                )
            label_firewall.record_aggregate_plan_decision_seal(
                str(aggregate_readback["seal_hash"]),
                plan_seal_hash=str(
                    aggregate_bindings["global_plan_seal_hash"]
                ),
                decision_barrier_hash=str(
                    aggregate_bindings["route_decision_barrier_hash"]
                ),
            )

            phase = "TERMINAL_LABELS_REPORTS_AND_REPLAY"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "terminal_labels_after_aggregate_barrier")
            terminal_labels = (
                deps.open_terminal_labels or label_firewall.open_terminal_labels
            )()
            capability = label_firewall.report_payload()
            terminal = (deps.evaluate_terminal or _default_terminal_evaluator)(
                probability_surface=surface,
                plans=plans,
                donor_counts=tuple(donor_counts),
                case_action_confusions=tuple(all_case_counts),
                donor_priors=tuple(all_priors),
                arm_decisions=tuple(all_decisions),
                method_predictions=method_predictions,
                descriptive_predictions=descriptive_control_predictions,
                aggregate_plan_decision_seal_hash=str(
                    aggregate_readback["seal_hash"]
                ),
                terminal_labels=terminal_labels,
                config=config,
                null_plan=null_plan,
            )
            terminal_seal = terminal.get("terminal_seal")
            if not isinstance(terminal_seal, Mapping):
                raise ProtocolError("Directional-shrinkage terminal seal is absent.")
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction.seal_hash,
                physical_prelabel_seal_hash=str(physical_seal["seal_hash"]),
                aggregate_plan_decision_seal_hash=str(aggregate_seal["seal_hash"]),
                capability_report=capability,
            )
            publication = publication_decision_payload(
                str(
                    terminal_seal.get(
                        "seal_hash", terminal_seal.get("terminal_seal_hash")
                    )
                ),
                descriptive_success_rubric=terminal_seal.get(
                    "descriptive_success_rubric", {}
                ),
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
                protocol_contract_hash=protocol.contract_hash,
            )
            checks = (deps.validate_bundle or _validate_bundle)(
                root, config=config, allow_pending_validation=True
            )
            if deps.validate_bundle is None:
                from .fresh_process_validation import (
                    require_two_fresh_process_validations,
                )

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
        raise ProtocolError("Directional-shrinkage label grant escapes sealed rows.")
    ordered_keys = tuple(
        (row.center, row.case_id, row.sample_id)
        for row in frame.rows
        if (row.center, row.case_id, row.sample_id) in allowed_keys
    )
    if len(ordered_keys) != len(allowed_keys):
        raise ProtocolError("Directional-shrinkage label grant order drifted.")
    requested = {key: universe[key] for key in ordered_keys}
    found: dict[tuple[str, str, str], BinaryLabel] = {}
    manifest_path = Path(getattr(config, "test_manifest_path"))
    manifest_hash = str(getattr(config, "expected_manifest_sha256"))
    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for ordinal, raw in enumerate(csv.DictReader(handle)):
                center = str(raw.get("center", ""))
                case_id = str(raw.get("case_id", ""))
                sample_id = evaluation_row_id(manifest_hash, ordinal)
                key = (center, case_id, sample_id)
                if key not in requested:
                    continue
                if requested[key].manifest_row_index != ordinal or key in found:
                    raise ProtocolError("Directional-shrinkage manifest order drifted.")
                # Decode a value only after the opaque identity was admitted.
                found[key] = BinaryLabel(
                    center,
                    case_id,
                    sample_id,
                    int(raw["label"]),
                    "manifest_loader",
                )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Cannot load directional-shrinkage labels.") from exc
    if set(found) != set(requested):
        raise ProtocolError("Directional-shrinkage label coverage drifted.")
    return tuple(found[key] for key in ordered_keys)


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import (
        validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle,
    )

    return validate_fixed_bank_loo_directional_shrinkage_ensemble_bundle(
        root, **kwargs
    )


def _default_terminal_evaluator(**kwargs: object) -> Mapping[str, object]:
    try:
        from .terminal import evaluate_terminal
    except ImportError as exc:
        raise ProtocolError(
            "Directional-shrinkage terminal evaluator is unavailable."
        ) from exc
    result = evaluate_terminal(**kwargs)
    if not isinstance(result, Mapping):
        raise ProtocolError("Directional-shrinkage terminal result is not a mapping.")
    return result


def _deduplicate_case_action_confusions(
    repeated: Sequence[object],
) -> tuple[object, ...]:
    """Collapse repeated H-minus-c views to the exact 218 x 10 case surface."""

    by_key: dict[tuple[str, str, str], object] = {}
    payload_by_key: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for row in repeated:
        key = (
            str(getattr(row, "target_center")),
            str(getattr(row, "case_id")),
            str(getattr(row, "action_id")),
        )
        payload = row.to_payload()
        if key in payload_by_key and payload_by_key[key] != payload:
            raise ProtocolError(
                "Directional-shrinkage repeated case/action confusion drifted."
            )
        by_key.setdefault(key, row)
        payload_by_key.setdefault(key, payload)
    ordered = tuple(
        by_key[(center, case_id, action)]
        for center in CENTERS
        for case_id in sorted(
            {case for target, case, _action in by_key if target == center}
        )
        for action in physical_action_ids(center)
    )
    if len(ordered) != 218 * 10 or len(by_key) != len(ordered):
        raise ProtocolError(
            "Directional-shrinkage case/action confusion topology is not 218 x 10."
        )
    return ordered


__all__ = ("run_fixed_bank_loo_directional_shrinkage_ensemble",)
