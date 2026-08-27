from __future__ import annotations

from dataclasses import asdict
import csv
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.evidence_builder import (
    build_target_prediction_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.model_freeze import (
    freeze_full_prelabel_router,
    route_frozen_predicted_utility_or_exact_b,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.phase_order import (
    SceptrePhaseManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.development_orchestrator import (
    FrozenDevelopmentReplay,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.authorization_lease import (
    claim_authorization_lease,
    LEASE_MEMBER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.config import (
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.experiment_contracts import (
    EXPECTED_EXECUTION_AMENDMENT_SHA256,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    INPUT_ARTIFACT_IDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.fresh_process_validation import (
    require_two_fresh_final_validations,
    require_two_fresh_preterminal_validations,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.label_broker import (
    RoleLabelBroker,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.outcome_builder import (
    build_role_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.prediction_surface import (
    CANDIDATE_ARRAY_MEMBER,
    CANDIDATE_EXCLUSION_SENTINEL,
    EXACT_B_ARRAY_MEMBER,
    LOCKED_CLASSIFIER_SPEC,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_RECEIPT_MEMBER,
    PRODUCTION_PREDICTION_GEOMETRY,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.persistence import (
    FINAL_FRESH_ATTESTATION_MEMBER,
    VALIDATION_REPORT_MEMBER,
    persist_durable_attestation,
    persist_preterminal_bundle,
    persist_terminal_bundle,
    persist_validation_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.phase_orchestrator import (
    run_routing_phases,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.reports import (
    build_final_reports,
    build_validation_report,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.source_seal import (
    build_source_snapshot_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.terminal_evaluation import (
    evaluate_terminal_surfaces,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.validation import (
    validate_final_bundle,
    validate_preterminal_bundle,
    validate_publication_bundle,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json, read_json, sha256_file
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    evaluation_row_id,
)
from test_sceptre_model_freeze import RAW_SOURCE_RECEIPT, frozen_fixture


def _frame_and_manifest(tmp_path: Path):
    repository = Path(__file__).resolve().parents[2]
    manifest = (
        repository / "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    )
    manifest_sha = sha256_file(manifest)
    manifest_rows = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for manifest_ordinal, raw in enumerate(csv.DictReader(handle)):
            if raw["split"] == "test" and raw["center"] in CENTERS:
                manifest_rows.append(
                    (manifest_ordinal, str(raw["case_id"]), str(raw["center"]))
                )
    rows = [
        row for center in CENTERS for row in manifest_rows if row[2] == center
    ]
    assert len(rows) == EXPECTED_TEST_ROWS
    assert {
        center: sum(row[2] == center for row in rows) for center in CENTERS
    } == EXPECTED_TEST_ROWS_BY_CENTER
    frame_rows = tuple(
        SimpleNamespace(
            row_ordinal=ordinal,
            manifest_row_index=manifest_ordinal,
            sample_id=evaluation_row_id(manifest_sha, manifest_ordinal),
            evaluation_row_id=evaluation_row_id(manifest_sha, manifest_ordinal),
            case_id=case_id,
            center=center,
        )
        for ordinal, (manifest_ordinal, case_id, center) in enumerate(rows)
    )
    frame = SimpleNamespace(rows=frame_rows)
    partition = build_three_role_partition(
        tuple(
            CaseIdentity(row.center, row.case_id, row.sample_id)
            for row in frame_rows
        )
    )
    return frame, manifest, manifest_sha, partition


def _development(frozen_fixture, partition) -> FrozenDevelopmentReplay:
    router = freeze_full_prelabel_router(
        frozen_fixture.models,
        generation_lock=frozen_fixture.lock,
        partition=partition,
        dirichlet_config=frozen_fixture.dirichlet_config,
    )
    decisions = []
    proposals = []
    for model, menu in zip(
        frozen_fixture.models, frozen_fixture.menus, strict=True
    ):
        evidence = build_target_prediction_evidence(
            frozen_fixture.raw,
            target_center=model.outer_target,
            raw_source_receipt_hash=RAW_SOURCE_RECEIPT,
        )
        decision = route_frozen_predicted_utility_or_exact_b(
            model,
            evidence,
            generation_lock=frozen_fixture.lock,
            candidate_menu=menu,
        )
        decisions.append(decision)
        proposals.append(router.bind_g_proposal(decision))
    body = {
        "schema_version": "sceptre_v3_label_free_development_replay_v1",
        "target_order": list(CENTERS),
        "fit_receipts": [
            {
                "outer_target": fit.outer_target,
                "selected_alpha": fit.selected_alpha,
                "outer_evidence_receipt_hash": fit.outer_evidence_receipt_hash,
                "final_training_receipt_hash": (
                    fit.final_model.training_receipt_hash
                ),
                "assessment_hash": canonical_hash(
                    [asdict(row) for row in fit.assessments]
                ),
            }
            for fit in frozen_fixture.fits
        ],
        "model_sha256_by_target": [
            [model.outer_target, model.model_sha256]
            for model in frozen_fixture.models
        ],
        "adaptive_decision_sha256_by_target": [
            [decision.outer_target, decision.decision_sha256]
            for decision in decisions
        ],
        "full_router_sha256": router.full_router_sha256,
        "g_proposal_sha256_by_target": [
            [proposal.target_center, proposal.proposal_sha256]
            for proposal in proposals
        ],
        "source_inner_only": True,
        "test_labels_consumed": False,
        "fresh_evidence": False,
    }
    return FrozenDevelopmentReplay(
        fits=frozen_fixture.fits,
        models=frozen_fixture.models,
        decisions=tuple(decisions),
        router=router,
        proposals=tuple(proposals),
        replay_hash=canonical_hash(body),
    )


def _prediction_members(
    root: Path,
    *,
    frame: object,
    candidate: np.ndarray,
    exact_b: np.ndarray,
    config_hash: str,
    cache_binding_hash: str,
):
    prediction = root / "prediction_store"
    (prediction / "arrays").mkdir(parents=True)
    (prediction / "manifests").mkdir()
    candidate_path = prediction / CANDIDATE_ARRAY_MEMBER
    exact_b_path = prediction / EXACT_B_ARRAY_MEMBER
    with candidate_path.open("wb") as handle:
        np.save(handle, candidate, allow_pickle=False)
    with exact_b_path.open("wb") as handle:
        np.save(handle, exact_b, allow_pickle=False)
    candidate_path.chmod(0o444)
    exact_b_path.chmod(0o444)

    row_ids = [str(row.sample_id) for row in frame.rows]
    row_centers = [str(row.center) for row in frame.rows]
    row_identity_sha256 = canonical_hash(
        [
            {"row_ordinal": ordinal, "row_id": row_id, "center": center}
            for ordinal, (row_id, center) in enumerate(
                zip(row_ids, row_centers, strict=True)
            )
        ]
    )
    fit_rows = []
    for seed_ordinal, (training_seed, generation_seed) in enumerate(
        PRODUCTION_PREDICTION_GEOMETRY.seed_cells
    ):
        within = 0
        for source in CENTERS:
            masked = EXPECTED_TEST_ROWS_BY_CENTER[source]
            body = {
                "family": "single_source",
                "source_center": source,
                "target_center": None,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "source_centers": [source],
                "composition_hash": canonical_hash(["single", source, training_seed]),
                "classifier_config_hash": LOCKED_CLASSIFIER_SPEC.config_hash,
                "scaler_state_hash": canonical_hash(["scaler", source, training_seed]),
                "probability_sha256": canonical_hash(["probability", source, training_seed]),
                "prediction_sha256": canonical_hash(["prediction", source, training_seed]),
                "evaluated_row_count": EXPECTED_TEST_ROWS - masked,
                "excluded_evaluation_center": source,
                "masked_row_count": masked,
                "converged": True,
                "target_expert_excluded": True,
            }
            fit_rows.append(
                {
                    "global_fit_ordinal": len(fit_rows),
                    "seed_cell_ordinal": seed_ordinal,
                    "within_cell_fit_ordinal": within,
                    **body,
                    "fit_sha256": canonical_hash(body),
                }
            )
            within += 1
        for target in CENTERS:
            sources = [source for source in CENTERS if source != target]
            body = {
                "family": "exact_B",
                "source_center": None,
                "target_center": target,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "source_centers": sources,
                "composition_hash": canonical_hash(["exact-B", target, training_seed]),
                "classifier_config_hash": LOCKED_CLASSIFIER_SPEC.config_hash,
                "scaler_state_hash": canonical_hash(["B-scaler", target, training_seed]),
                "probability_sha256": canonical_hash(["B-probability", target, training_seed]),
                "prediction_sha256": canonical_hash(["B-prediction", target, training_seed]),
                "evaluated_row_count": EXPECTED_TEST_ROWS_BY_CENTER[target],
                "excluded_evaluation_center": None,
                "masked_row_count": 0,
                "converged": True,
                "target_expert_excluded": True,
            }
            fit_rows.append(
                {
                    "global_fit_ordinal": len(fit_rows),
                    "seed_cell_ordinal": seed_ordinal,
                    "within_cell_fit_ordinal": within,
                    **body,
                    "fit_sha256": canonical_hash(body),
                }
            )
            within += 1
    fit_hash = canonical_hash(fit_rows)
    geometry = PRODUCTION_PREDICTION_GEOMETRY.to_payload()
    index_body = {
        "schema_version": "midogpp_sceptre_v3_prediction_index_v1",
        "config_hash": config_hash,
        "source_receipt_sha256": "1" * 64,
        "source_array_sha256": "2" * 64,
        "cache_binding_hash": cache_binding_hash,
        "geometry": geometry,
        "classifier": LOCKED_CLASSIFIER_SPEC.to_payload(),
        "classifier_config_hash": LOCKED_CLASSIFIER_SPEC.config_hash,
        "row_ids": row_ids,
        "row_centers": row_centers,
        "row_identity_sha256": row_identity_sha256,
        "fit_rows": fit_rows,
        "fit_index_sha256": fit_hash,
        "fit_count": len(fit_rows),
        "candidate_source_order": list(CENTERS),
        "seed_cell_order": [
            list(value) for value in PRODUCTION_PREDICTION_GEOMETRY.seed_cells
        ],
        "exact_b_target_exclusion_verified": True,
        "candidate_target_exclusion_mode": "MASKED_BEFORE_SCORING",
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "all_seed_cells_retained": True,
        "seed_selection_performed": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
    }
    index = {**index_body, "index_sha256": canonical_hash(index_body)}
    index_path = prediction / PREDICTION_INDEX_MEMBER
    atomic_json(index_path, index)
    receipt_body = {
        "schema_version": "midogpp_sceptre_v3_prediction_receipt_v1",
        "status": "SEALED_ALL_LABEL_FREE_CANDIDATE_AND_EXACT_B_PREDICTIONS",
        "config_hash": config_hash,
        "source_receipt_sha256": "1" * 64,
        "cache_binding_hash": cache_binding_hash,
        "geometry": geometry,
        "classifier_config_hash": LOCKED_CLASSIFIER_SPEC.config_hash,
        "candidate_array_file_sha256": sha256_file(candidate_path),
        "exact_b_array_file_sha256": sha256_file(exact_b_path),
        "prediction_index_file_sha256": sha256_file(index_path),
        "prediction_index_sha256": index["index_sha256"],
        "row_identity_sha256": row_identity_sha256,
        "fit_index_sha256": fit_hash,
        "fit_count": len(fit_rows),
        "candidate_shape": list(candidate.shape),
        "exact_b_shape": list(exact_b.shape),
        "dtype": "float32",
        "npy_memmap_mode": "read_only",
        "cpu_worker_count": 4,
        "blas_threads_per_worker": 1,
        "native_threads_per_worker": 1,
        "top_level_spawn_pool_only": True,
        "target_expert_excluded_from_every_exact_b_fit": True,
        "target_expert_excluded_from_every_candidate_score": True,
        "candidate_exclusion_sentinel": float(CANDIDATE_EXCLUSION_SENTINEL),
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
        "classifier_refit_after_seal": False,
        "seed_selection_performed": False,
        "synthetic_test_mode": False,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_hash(receipt_body)}
    receipt_path = prediction / PREDICTION_RECEIPT_MEMBER
    atomic_json(receipt_path, receipt)
    members = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(prediction.rglob("*"))
        if path.is_file()
    }
    return (
        {
            "schema_version": "sceptre_v3_durable_prediction_store_v1",
            "store_hash": receipt["receipt_sha256"],
            "physical_receipt": receipt,
            "member_sha256": members,
            "candidate_source_order": list(CENTERS),
            "manifest_opened": False,
            "outcomes_available": False,
            "raw_sample_paths_available": False,
        },
        members,
        receipt["receipt_sha256"],
    )


def _launch_binding_and_lease(root: Path, tmp_path: Path):
    repository = Path(__file__).resolve().parents[2]
    canonical_config = repository / (
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
        "uniform_b_v2_consumed_test_fixed_bank_sceptre_router_v3.yaml"
    )
    resolved_config = root / "config.resolved.yaml"
    shutil.copyfile(canonical_config, resolved_config)
    config = load_config(resolved_config)
    artifact_rows = [
        {
            "artifact_id": artifact_id,
            "resolved_path": f"/synthetic/{artifact_id}",
            "exists": True,
            "semantic_identities": {},
            "file_integrity": {},
        }
        for artifact_id in sorted(INPUT_ARTIFACT_IDS)
    ]
    workspace = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": config.experiment_id,
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "selection_used_target_eval_artifacts": False,
        "repository_revision": "synthetic-validator-fixture",
        "repository_dirty": True,
        "repository_status_hash": "synthetic-status",
        "input_artifacts": artifact_rows,
    }
    atomic_json(root / "provenance/input_artifacts.json", workspace)
    provenance = {
        artifact_id: next(
            row for row in artifact_rows if row["artifact_id"] == artifact_id
        )
        for artifact_id in INPUT_ARTIFACT_IDS
    }
    cache_binding_hash = canonical_hash({"cache": "synthetic-canonical-geometry"})
    admission_hash = canonical_hash({"admission": "synthetic"})
    read_only_admission_hash = canonical_hash({"admission": "read-only"})
    worker_runtime_smoke_hash = canonical_hash({"worker": "smoke"})
    body = {
        "schema_version": "sceptre_v3_exact_eight_input_binding_v1",
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "direct_input_count": 8,
        "workspace_provenance": provenance,
        "admission_hash": admission_hash,
        "read_only_admission_hash": read_only_admission_hash,
        "worker_runtime_smoke_hash": worker_runtime_smoke_hash,
        "cache_binding_hash": cache_binding_hash,
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "source_inner_amendment_sha256": EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
        "execution_amendment_sha256": EXPECTED_EXECUTION_AMENDMENT_SHA256,
        "all_inputs_validated_before_authorization_claim": True,
        "target_labels_opened": False,
        "previous_stage90_output_used": False,
        "v2_output_used": False,
        "v2_run_state_used": False,
        "v2_scratch_or_checkpoint_used": False,
        "v2_execution_amendment_used": False,
        "prior_v2_execution_authorization_reused": False,
    }
    input_binding = {**body, "binding_hash": canonical_hash(body)}
    lease_repository = tmp_path / "lease-repository"
    (lease_repository / "artifacts/midogpp/90_oracles_and_diagnostics").mkdir(
        parents=True
    )
    lease = claim_authorization_lease(
        config,
        admission_hash=admission_hash,
        repository_root=lease_repository,
    )
    return (
        config,
        input_binding,
        read_json(lease.root / LEASE_MEMBER),
        cache_binding_hash,
        lease.lease_hash,
    )


def _worker_smoke_receipt() -> dict[str, object]:
    probes = [
        {
            "device": device,
            "device_index": index,
            "process_id": 8000 + index,
            "initializer_invocation_count": 1,
            "torch_intraop_threads": 1,
            "torch_interop_threads": 1,
            "tf32_enabled": False,
            "amp_enabled": False,
            "task_ordinal": ordinal,
        }
        for index, device in enumerate(("cuda:0", "cuda:1"))
        for ordinal in (0, 1)
    ]
    base = {
        "schema_version": "sceptre_v3_gpu_worker_runtime_smoke_v1",
        "status": "PASS",
        "execution_mode": "spawn",
        "gpu_devices": ["cuda:0", "cuda:1"],
        "persistent_worker_count": 2,
        "max_workers_per_pool": 1,
        "task_count_per_worker": 2,
        "initializer_invocation_count_per_worker": 1,
        "same_process_for_repeated_tasks": True,
        "distinct_process_per_gpu": True,
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "child_cuda_context_initialized": True,
        "parent_cuda_context_initialized": False,
        "parent_cuda_state_checked_before_and_after_smoke": True,
        "scientific_gpu_work_performed": False,
        "experts_loaded": False,
        "embeddings_generated": False,
        "target_cache_opened": False,
        "target_manifest_opened": False,
        "target_labels_opened": False,
        "filesystem_mutations": 0,
        "probes": probes,
    }
    return {**base, "worker_runtime_smoke_hash": canonical_hash(base)}


def test_complete_45_fold_lifecycle_reconstructs_without_refit(
    tmp_path: Path, frozen_fixture
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (
        config,
        input_binding,
        lease_payload,
        cache_binding_hash,
        lease_hash,
    ) = _launch_binding_and_lease(root, tmp_path)
    frame, manifest, manifest_sha, partition = _frame_and_manifest(tmp_path)
    development = _development(frozen_fixture, partition)
    manager = SceptrePhaseManager(partition, development.router)
    candidate = np.full((9, 9, len(frame.rows)), 0.5, dtype=np.float32)
    for source_ordinal, source in enumerate(CENTERS):
        row_ordinals = [
            row.row_ordinal for row in frame.rows if row.center == source
        ]
        candidate[:, source_ordinal, row_ordinals] = CANDIDATE_EXCLUSION_SENTINEL
    exact_b = np.full((9, len(frame.rows)), 0.5, dtype=np.float32)
    candidate.setflags(write=False)
    exact_b.setflags(write=False)
    prediction_store, members, store_hash = _prediction_members(
        root,
        frame=frame,
        candidate=candidate,
        exact_b=exact_b,
        config_hash=config.config_hash,
        cache_binding_hash=cache_binding_hash,
    )
    broker = RoleLabelBroker(
        manager=manager,
        partition=partition,
        frame=frame,
        manifest_path=manifest,
        expected_manifest_sha256=manifest_sha,
        prediction_store_hash=store_hash,
        authorization_lease_hash=lease_hash,
    )
    phases = run_routing_phases(
        development,
        partition=partition,
        manager=manager,
        broker=broker,
        candidate_probabilities=candidate,
        exact_b_probabilities=exact_b,
        candidate_source_order=CENTERS,
        prediction_store_hash=store_hash,
    )
    assert len(phases.support_decisions) == 45
    assert len(phases.calibration_decisions) == 45
    assert not phases.uncertainty_decisions
    assert all(
        row.route == EXACT_B_CANDIDATE
        for row in phases.calibration_decisions
    )

    source = build_source_snapshot_payload()
    preterminal = persist_preterminal_bundle(
        root,
        config_hash=config.config_hash,
        input_binding=input_binding,
        source_snapshot=source,
        authorization_lease=lease_payload,
        prediction_store=prediction_store,
        prediction_member_hashes=members,
        partition=partition,
        development=development,
        phases=phases,
    )
    assert validate_preterminal_bundle(root)["status"] == "PASS"
    durable = require_two_fresh_preterminal_validations(root)
    persist_durable_attestation(
        root,
        durable,
        preterminal_content_index_hash=preterminal["content_index_hash"],
    )

    capability = manager.begin_terminal_evaluation(durable)
    broker.activate_terminal(capability)
    surfaces = []
    for target in CENTERS:
        for fold_ordinal in range(5):
            fold = partition.fold(target, fold_ordinal)
            scoped = broker.open_evaluation(target, fold_ordinal, capability)
            model = development.router.model_for_target(target)
            evidence = build_role_evidence(
                scoped,
                fold=fold,
                partition_hash=partition.partition_hash,
                candidate_probabilities=candidate,
                exact_b_probabilities=exact_b,
                candidate_source_order=CENTERS,
                prediction_store_hash=store_hash,
                candidate_menu_hash=model.candidate_menu_hash,
                exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
                phase_capability=capability,
            )
            surfaces.append(evidence.surface)
    result = evaluate_terminal_surfaces(
        phases.route_policy,
        surfaces,
        prediction_store_hash=store_hash,
        terminal_capability_hash=capability.capability_hash,
    )
    assert len(result.folds) == 45
    smoke = _worker_smoke_receipt()
    reports = build_final_reports(
        result,
        input_binding=input_binding,
        source_snapshot=source,
        label_journal=broker.journal_payload(),
        runtime={
            "status": "PASS",
            "worker_runtime_smoke": smoke,
            "worker_runtime_smoke_hash": smoke["worker_runtime_smoke_hash"],
            "runtime_launch_admission_hash": input_binding["admission_hash"],
        },
        prediction_store=prediction_store,
    )
    persist_terminal_bundle(root, result, reports=reports)
    assert validate_final_bundle(root)["terminal_result_hash"] == result.result_hash
    final = require_two_fresh_final_validations(root)
    assert final["fresh_process_count"] == 2
    atomic_json(root / FINAL_FRESH_ATTESTATION_MEMBER, final)
    validation_report = build_validation_report(final)
    atomic_json(root / VALIDATION_REPORT_MEMBER, validation_report)
    validation_index = persist_validation_index(
        root,
        final_attestation_member=FINAL_FRESH_ATTESTATION_MEMBER,
        validation_report_member=VALIDATION_REPORT_MEMBER,
    )
    publication = validate_publication_bundle(root)
    assert publication["postvalidation_index_authenticated"] is True
    assert (
        publication["validation_index_hash"]
        == validation_index["validation_index_hash"]
    )

    tampered = dict(validation_report)
    tampered["status"] = "FAIL"
    atomic_json(root / VALIDATION_REPORT_MEMBER, tampered)
    with pytest.raises(ProtocolError, match="validation report|post-validation"):
        validate_publication_bundle(root)
