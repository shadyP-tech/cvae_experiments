"""Thin phase runner for source-trained, label-free test inference."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ...routing.disagreement_regret_core import canonical_workstation_runtime
from ...runtime.artifact_io import read_json
from .bundle import (
    assert_closed_world,
    cleanup_owned_atomic_temps,
    write_content_index,
)
from .development import (
    build_posthoc_source_contexts,
    build_source_prelabel_products,
    fit_source_development_products,
)
from .execution_adapter import (
    aggregate_probability_rows,
    aggregate_source_oof_probability_rows,
    materialize_sources,
    run_label_free_workstation_preflight,
    runtime_summary_payload,
    stage_sources_for_cpu,
)
from .development_prediction_runtime import (
    materialize_composite_prelabel_prediction_seal,
    materialize_development_source_action_predictions,
)
from .experiment_contracts import CENTERS, GEOMETRY_IDS
from .inference import build_test_inference_products
from .inputs import (
    assert_train_test_disjoint,
    load_label_free_source_frame,
    load_label_free_test_frame,
)
from .persistence import (
    persist_development_products,
    persist_inference_products,
    persist_initial_manifest,
    persist_prelabel_products,
    persist_reports,
    persist_validation_report,
)
from .prediction_runtime import (
    issue_test_inference_admission,
    materialize_target_action_classifier_bank,
    materialize_test_action_predictions,
)
from .protocol import (
    assert_prediction_only_diagnostic,
    canonical_prediction_only_protocol,
)
from .reports import leakage_report_payload, publication_decision_payload
from .runner_dependencies import PredictionOnlyDependencies
from .runner_runtime import (
    assert_launch_files,
    assert_workspace_resolved_paths,
    enter_cuda_free_cpu_phase,
    exclusive_run_lock,
    observe,
    recover_complete,
    write_state,
)
from .source_capability import SourceOOFLabelCapability


def run_fixed_bank_disagreement_regret_prediction_only(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    dependencies: PredictionOnlyDependencies | None = None,
) -> Path:
    """Fit on source OOF labels, then predict all test rows without labels."""

    root = Path(artifact_root or getattr(config, "artifact_root"))
    deps = dependencies or PredictionOnlyDependencies()
    protocol = canonical_prediction_only_protocol()
    assert_prediction_only_diagnostic(protocol)
    assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    assert_launch_files(root)

    # Heavy input validators are imported only by the runnable adapter.  The
    # reusable mathematical core remains free of workspace and artifact I/O.
    from .inputs import (
        assert_input_fence,
        load_validated_locks,
        validate_pre_gpu_firewall,
    )
    from .workspace_inputs import (
        validate_active_diagnostic_workspace_binding,
        validate_workspace_provenance,
    )

    with exclusive_run_lock(root):
        cleanup_owned_atomic_temps(root)
        assert_closed_world(root, allow_incomplete=True)
        recovered = recover_complete(root, config=config, dependencies=deps)
        if recovered is not None:
            return recovered

        phase = "INITIALIZING"
        write_state(deps, root, status="RUNNING", phase=phase)
        try:
            observe(deps, "input_fence")
            (deps.validate_input_fence or assert_input_fence)(config)
            workspace = (
                deps.validate_workspace
                or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (
                deps.validate_provenance or validate_workspace_provenance
            )(root, config)
            locks = (deps.load_locks or load_validated_locks)(config)

            # Only the label-free source frame is available before source
            # prediction/model seals.  The test cache is deliberately absent.
            source_frame = (
                deps.load_source_frame or load_label_free_source_frame
            )(config)
            firewall = dict(
                (deps.validate_firewall or validate_pre_gpu_firewall)(
                    config, source_frame, locks
                )
            )
            firewall["workspace_binding"] = workspace
            test_input_binding = _sealed_test_input_binding(config)
            (deps.persist_initial or persist_initial_manifest)(
                root,
                config=config,
                protocol=protocol,
                provenance=provenance,
                source_frame_binding=dict(source_frame.cache_binding),
                test_input_binding=test_input_binding,
                firewall=firewall,
            )

            phase = "WORKSTATION_PREFLIGHT"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "preflight")
            preflight = (deps.preflight or run_label_free_workstation_preflight)(
                root, runtime=getattr(config, "runtime")
            )

            phase = "FROZEN_SOURCE_STREAMS_TWO_GPU"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "gpu_source_streams")
            canonical_sources = (
                deps.materialize_source_streams or materialize_sources
            )(config, getattr(locks, "generation"), root=root)
            source_for_cpu = (
                deps.stage_source_streams or stage_sources_for_cpu
            )(canonical_sources, root=root)

            enter_cuda_free_cpu_phase()
            phase = "STRICT_SOURCE_OOF_AND_TARGET_CLASSIFIER_PRELABEL_SEALS"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "target_classifier_bank_seal")
            target_classifier_bank = (
                deps.materialize_target_classifier_bank
                or materialize_target_action_classifier_bank
            )(config, source_for_cpu, source_frame, root=root)
            observe(deps, "strict_source_oof_prediction_seal")
            strict_source_predictions = (
                deps.materialize_source_oof_predictions
                or materialize_development_source_action_predictions
            )(config, source_for_cpu, source_frame, root=root)
            observe(deps, "composite_prelabel_prediction_seal")
            source_predictions = (
                deps.materialize_prelabel_prediction_seal
                or materialize_composite_prelabel_prediction_seal
            )(
                strict_source_predictions,
                target_classifier_bank,
                root=root,
            )
            source_views = _probability_views(
                deps.aggregate_source_probabilities
                or aggregate_source_oof_probability_rows,
                source_predictions,
                frame_role="source",
            )
            contexts = (deps.build_contexts or build_posthoc_source_contexts)(
                source_frame,
                authorization_hash=str(
                    getattr(config, "expected_ledger_amendment_sha256")
                ),
            )
            prelabel = (deps.build_prelabel or build_source_prelabel_products)(
                source_views,
                source_prediction_seal_hash=source_predictions.seal_hash,
                contexts=contexts,
            )
            (deps.persist_prelabel or persist_prelabel_products)(root, prelabel)

            phase = "SOURCE_LABEL_CAPABILITY_AND_MODEL_BANK_SEAL"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "source_labels_after_prediction_seal")
            capability_factory = (
                deps.build_source_label_capability or SourceOOFLabelCapability
            )
            capability = capability_factory(
                source_frame,
                train_cache_root=Path(getattr(config, "train_cache_root")),
            )
            capability.open_after_source_prediction_seal(source_predictions)
            labels_by_target = {
                target: capability.labels_for_outer_target(target)
                for target in CENTERS
            }
            capability_report = dict(capability.access_report())
            development = (deps.fit_development or fit_source_development_products)(
                prelabel,
                labels_by_outer_target=labels_by_target,
                source_label_capability_report=capability_report,
                runtime=canonical_workstation_runtime(),
            )
            (deps.persist_development or persist_development_products)(
                root, development
            )

            # This durable seal is the sole authority that admits the test
            # cache. It proves source-only fitting and no target-label access.
            model_bank_seal = read_json(root / "manifests/model_bank_seal.json")
            phase = "LABEL_FREE_TEST_ADMISSION_AND_FROZEN_INFERENCE"
            write_state(deps, root, status="RUNNING", phase=phase)
            observe(deps, "test_cache_after_model_bank_seal")
            admission = (
                deps.issue_test_admission or issue_test_inference_admission
            )(source_predictions, model_bank_seal)
            test_frame = (deps.load_test_frame or load_label_free_test_frame)(
                config, admission=admission
            )
            disjointness = assert_train_test_disjoint(source_frame, test_frame)
            test_predictions = (
                deps.materialize_test_predictions
                or materialize_test_action_predictions
            )(config, source_predictions, test_frame, root=root)
            test_views = _probability_views(
                deps.aggregate_test_probabilities or aggregate_probability_rows,
                test_predictions,
                frame_role="test",
            )
            inference = (deps.build_inference or build_test_inference_products)(
                development,
                test_views,
                test_prediction_seal_hash=test_predictions.seal_hash,
                target_cache_content_hash=str(
                    getattr(config, "expected_test_cache_content_hash")
                ),
                target_cache_order_hash=str(
                    getattr(config, "expected_test_cache_row_order_hash")
                ),
            )
            (deps.persist_inference or persist_inference_products)(root, inference)

            phase = "CLOSED_WORLD_PREDICTION_ONLY_VALIDATION"
            write_state(deps, root, status="RUNNING", phase=phase)
            leakage = leakage_report_payload(
                source_prediction_seal_hash=source_predictions.seal_hash,
                test_prediction_seal_hash=test_predictions.seal_hash,
                source_label_capability_report=capability_report,
                model_bank_hash=development.model_bank_hash,
                frozen_test_prediction_hash=inference.frozen_prediction_hash,
            )
            publication = publication_decision_payload(
                frozen_test_prediction_hash=inference.frozen_prediction_hash
            )
            runtime_summary = dict(
                (deps.build_runtime_summary or runtime_summary_payload)(
                    generated_sources=canonical_sources,
                    source_predictions=source_predictions,
                    test_predictions=test_predictions,
                    runtime=getattr(config, "runtime"),
                )
            )
            runtime_summary["workstation_preflight"] = dict(preflight)
            runtime_summary["train_test_disjointness"] = dict(disjointness)
            (deps.persist_reports or persist_reports)(
                root,
                leakage=leakage,
                publication=publication,
                runtime_summary=runtime_summary,
            )
            (deps.write_content_index or write_content_index)(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.contract_hash,
            )
            validator = deps.validate_bundle or _validate_bundle
            checks = validator(root, config=config)
            (deps.persist_validation or persist_validation_report)(root, checks)
            write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            validator(root, config=config)
            if deps.cleanup_staging is not None:
                deps.cleanup_staging(source_for_cpu, canonical_sources)
            return root
        except BaseException as exc:
            write_state(
                deps,
                root,
                status="FAILED",
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _probability_views(
    aggregate: object,
    capability: object,
    *,
    frame_role: str,
) -> dict[tuple[str, str], tuple[object, ...]]:
    if not callable(aggregate):
        raise ProtocolError("Probability aggregation dependency is not callable.")
    return {
        (target, geometry): tuple(
            aggregate(
                capability,
                frame_role=frame_role,
                geometry_id=geometry,
                outer_target_id=target,
            )
        )
        for target in CENTERS
        for geometry in GEOMETRY_IDS
    }


def _sealed_test_input_binding(config: object) -> dict[str, object]:
    return {
        "schema_version": "midogpp_prediction_only_sealed_test_input_v1",
        "row_count": 9_928,
        "feature_dim": 3_840,
        "cache_content_hash": str(
            getattr(config, "expected_test_cache_content_hash")
        ),
        "row_order_hash": str(
            getattr(config, "expected_test_cache_row_order_hash")
        ),
        "cache_opened": False,
        "labels_available": False,
        "scoring_permitted": False,
        "admission_required_after_model_bank_seal": True,
    }


def _validate_bundle(root: Path, *, config: object) -> object:
    from .validation import (
        validate_fixed_bank_disagreement_regret_prediction_only_bundle,
    )

    return validate_fixed_bank_disagreement_regret_prediction_only_bundle(
        root, config=config
    )


__all__ = (
    "PredictionOnlyDependencies",
    "run_fixed_bank_disagreement_regret_prediction_only",
)
