"""Orchestrate the exact 36-key Stage-90 paired Variant-B audit."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError

from .artifacts import (
    ensure_audit_directories,
    file_sha256,
    write_content_index,
    write_csv,
    write_json,
)
from .comparison import audit_decision, paired_comparison_rows
from .config import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    CLAIM_SCOPE,
    EVIDENCE_LABEL,
    FIXED_ANTITHETIC_CANDIDATE,
    FIXED_ONE_EPSILON_CANDIDATE,
    LEGACY_CANDIDATE,
    SNAPSHOT_ARTIFACT_ID,
    STAGE,
    AuditConfig,
)
from .execution import run_audit_job
from .entrypoint import (
    AUDIT_CANONICAL_RELATIVE,
    AUDIT_EXPERIMENT_ID,
    assert_workspace_prepared_entrypoint,
)
from .protocol import AuditKeyRecord, key_inventory_hash, validate_key_inventory
from .run_lock import exclusive_artifact_lock
from .snapshot import load_snapshot
from .trace import EpsilonTraceLedger, EpsilonTraceSpec, load_epsilon_trace
from .validation import (
    PROTOCOL_SCHEMA,
    RUNTIME_SUMMARY_SCHEMA,
    SNAPSHOT_BINDING_SCHEMA,
    validate_audit_bundle,
)


def run_b_paired_reparameterization_audit(
    config: AuditConfig,
    *,
    artifact_root: Path,
    resolved_config_path: str | Path | None = None,
) -> Path:
    """Run the diagnostic panel; never export a recipe or selection decision."""

    root = Path(artifact_root).resolve()
    if resolved_config_path is None:
        raise ProtocolError(
            "Stage-90 audit execution requires a workspace-resolved config path."
        )
    assert_workspace_prepared_entrypoint(
        resolved_config_path=resolved_config_path,
        artifact_root=root,
        experiment_id=AUDIT_EXPERIMENT_ID,
        canonical_relative=AUDIT_CANONICAL_RELATIVE,
        input_artifact_ids=(SNAPSHOT_ARTIFACT_ID,),
        expected_input_members={SNAPSHOT_ARTIFACT_ID: config.snapshot_root},
    )
    root = ensure_audit_directories(root)
    (root / "reports/training_diagnostics").mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    snapshot_root = Path(config.snapshot_root).resolve()
    snapshot = load_snapshot(
        snapshot_root / "manifests/snapshot_manifest.json",
        artifact_root=snapshot_root,
        require_hash_promoted=True,
    )
    records = validate_key_inventory(
        snapshot.keys,
        require_publication_hashes=True,
    )
    _validate_config_snapshot_alignment(config, records)
    _require_workspace_snapshot_files(root)

    with exclusive_artifact_lock(root, purpose="stage90_audit"):
        _write_protocol_artifacts(
            root=root,
            config=config,
            snapshot=snapshot,
            records=records,
        )
        _write_run_state(
            root,
            status="RUNNING",
            completed_jobs=0,
            expected_jobs=len(records),
        )
        results = _execute_jobs(
            root=root,
            snapshot_root=snapshot_root,
            snapshot=snapshot,
            config=config,
            records=records,
        )
        _write_result_artifacts(
            root=root,
            config=config,
            snapshot=snapshot,
            records=records,
            results=results,
            wall_seconds=perf_counter() - started,
        )
        write_json(
            root / "reports/validation_report.json",
            {
                "schema_version": "midogpp_b_paired_reparameterization_validation_report_v1",
                "status": "VALIDATING",
                "errors": [],
                "claim_scope": "diagnostic_only",
            },
        )
        _write_run_state(
            root,
            status="VALIDATING",
            completed_jobs=len(results),
            expected_jobs=len(records),
        )
        write_content_index(root)
        report = validate_audit_bundle(root)
        write_json(root / "reports/validation_report.json", report)
        final_status = "COMPLETE" if report["status"] == "PASS" else "FAILED_VALIDATION"
        _write_run_state(
            root,
            status=final_status,
            completed_jobs=len(results),
            expected_jobs=len(records),
        )
        if report["status"] != "PASS":
            raise ProtocolError(
                "Stage-90 paired audit failed bundle validation: "
                + "; ".join(str(value) for value in report.get("errors", ()))
            )
        # The excluded validation/run-state files changed after the first pass;
        # all indexed bytes remain immutable. Re-run for an independent handoff.
        final_report = validate_audit_bundle(root)
        if final_report["status"] != "PASS":
            raise ProtocolError("Final Stage-90 audit validation did not replay.")
        write_json(root / "reports/validation_report.json", final_report)
    return root


def _execute_jobs(
    *,
    root: Path,
    snapshot_root: Path,
    snapshot: object,
    config: AuditConfig,
    records: Sequence[AuditKeyRecord],
) -> list[dict[str, object]]:
    jobs = [
        {
            "snapshot_root": str(snapshot_root),
            "output_root": str(root),
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_protocol_hash": snapshot.protocol_hash,
            "record": record.to_payload(),
            "recipe": asdict(config.recipe),
        }
        for record in records
    ]
    order = {
        candidate: index
        for index, candidate in enumerate(AUDIT_CANDIDATES)
    }
    jobs.sort(
        key=lambda item: (
            AUDIT_CENTERS.index(str(item["record"]["center"])),
            int(item["record"]["initialization_seed"]),
            order[str(item["record"]["candidate"])],
        )
    )
    by_device: dict[str, list[Mapping[str, object]]] = {}
    for job in jobs:
        by_device.setdefault(str(job["record"]["execution_device"]), []).append(job)
    if set(by_device) != {"cuda:0", "cuda:1"}:
        raise ProtocolError("Audit job inventory must use exactly cuda:0 and cuda:1.")
    context = mp.get_context("spawn")
    executors = {
        device: ProcessPoolExecutor(max_workers=1, mp_context=context)
        for device in sorted(by_device)
    }
    futures = []
    results: list[dict[str, object]] = []
    ledger = EpsilonTraceLedger(records)
    records_by_key = {record.key_hash: record for record in records}
    try:
        for device in sorted(by_device):
            futures.extend(
                executors[device].submit(run_audit_job, job)
                for job in by_device[device]
            )
        for future in as_completed(futures):
            result = future.result()
            job_row = _mapping(result.get("job"), "job result")
            key_hash = str(job_row["key_hash"])
            record = records_by_key.get(key_hash)
            if record is None:
                raise ProtocolError("Worker returned an undeclared audit key.")
            loaded = load_epsilon_trace(
                snapshot_root,
                EpsilonTraceSpec(
                    relative_path=record.epsilon_trace_relpath,
                    file_sha256=record.epsilon_trace_sha256,
                    content_sha256=record.epsilon_trace_content_hash,
                    steps=config.recipe.optimizer_steps,
                    batch_size=config.recipe.batch_size,
                    latent_dim=config.recipe.latent_dim,
                ),
            )
            ledger.consume(record, loaded)
            results.append(dict(result))
            write_json(root / "jobs" / f"{key_hash}.json", dict(result))
            write_json(
                root / "reports/training_diagnostics" / f"{key_hash}.json",
                {
                    "schema_version": "midogpp_b_paired_training_diagnostics_v1",
                    "key_hash": key_hash,
                    "rows": result.get("training_diagnostics", ()),
                    "claim_scope": "diagnostic_only",
                    "may_feed_expert_bank": False,
                },
            )
            _write_run_state(
                root,
                status="RUNNING",
                completed_jobs=len(results),
                expected_jobs=len(records),
            )
    except Exception:
        _write_run_state(
            root,
            status="FAILED_EXECUTION",
            completed_jobs=len(results),
            expected_jobs=len(records),
        )
        raise
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
    ledger.assert_complete()
    results.sort(
        key=lambda result: (
            AUDIT_CENTERS.index(str(result["job"]["center"])),
            int(result["job"]["initialization_seed"]),
            order[str(result["job"]["candidate"])],
        )
    )
    return results


def _write_protocol_artifacts(
    *,
    root: Path,
    config: AuditConfig,
    snapshot: object,
    records: Sequence[AuditKeyRecord],
) -> None:
    inventory_hash = key_inventory_hash(records)
    eval_row_inventory_hashes = {
        prepared.center: prepared.evaluation.row_inventory_hash
        for prepared in snapshot.prepared_centers
    }
    write_json(
        root / "manifests/key_inventory.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_key_inventory_v1",
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_manifest_hash": snapshot.manifest_hash,
            "key_inventory_hash": inventory_hash,
            "records": [record.to_payload() for record in records],
        },
    )
    binding = {
        "schema_version": SNAPSHOT_BINDING_SCHEMA,
        "snapshot_artifact_id": SNAPSHOT_ARTIFACT_ID,
        "publication_state": snapshot.publication_state,
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_manifest_hash": snapshot.manifest_hash,
        "key_inventory_hash": inventory_hash,
        "snapshot_content_index_hash": snapshot.content_index_hash,
        "eval_row_inventory_hashes": eval_row_inventory_hashes,
        "historical_paths_read": False,
        "claim_scope": "diagnostic_only",
        **config.claim_firewall.to_payload(),
    }
    write_json(root / "manifests/snapshot_binding.json", binding)
    protocol: dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA,
        "stage": STAGE,
        "evidence_label": EVIDENCE_LABEL,
        "claim_scope": CLAIM_SCOPE,
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_manifest_hash": snapshot.manifest_hash,
        "key_inventory_hash": inventory_hash,
        "eval_row_inventory_hashes": eval_row_inventory_hashes,
        "workspace_snapshot_hashes": {
            "config_resolved_sha256": file_sha256(root / "config.resolved.yaml"),
            "input_artifacts_sha256": file_sha256(
                root / "provenance/input_artifacts.json"
            ),
        },
        "recipe": config.recipe.to_payload(),
        "decision_thresholds": config.decision_thresholds.to_payload(),
        "legacy_used_for_decision": False,
        "controlled_pair_count": 12,
        "key_count": 36,
        "claim_firewall": config.claim_firewall.to_payload(),
        **config.claim_firewall.to_payload(),
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    write_json(root / "manifests/protocol_manifest.json", protocol)


def _write_result_artifacts(
    *,
    root: Path,
    config: AuditConfig,
    snapshot: object,
    records: Sequence[AuditKeyRecord],
    results: Sequence[Mapping[str, object]],
    wall_seconds: float,
) -> None:
    jobs = [dict(_mapping(result.get("job"), "job result")) for result in results]
    traces = [
        dict(_mapping(result.get("trace_audit"), "trace audit"))
        for result in results
    ]
    consumptions = [
        dict(_mapping(result.get("consumption"), "consumption audit"))
        for result in results
    ]
    legacy = [
        dict(_mapping(result.get("legacy_validation"), "legacy validation"))
        for result in results
        if result.get("legacy_validation") is not None
    ]
    all_metrics = [
        dict(_mapping(result.get("metric"), "metric result"))
        for result in results
    ]
    controlled_metrics = [
        row for row in all_metrics if row["candidate"] != LEGACY_CANDIDATE
    ]
    predictions = [
        dict(row)
        for result in results
        for row in _sequence_of_mappings(result.get("predictions"), "predictions")
    ]
    paired = paired_comparison_rows(controlled_metrics)
    decision = audit_decision(
        controlled_metrics,
        thresholds=config.decision_thresholds.to_payload(),
    )
    write_csv(root / "tables/job_inventory.csv", jobs)
    write_csv(root / "tables/replay_trace_audit.csv", traces)
    write_csv(root / "tables/legacy_v2_validation.csv", legacy)
    write_csv(root / "tables/controlled_metrics.csv", controlled_metrics)
    write_csv(root / "tables/paired_comparison.csv", paired)
    write_csv(root / "tables/consumption_audit.csv", consumptions)
    write_csv(root / "tables/decoded_predictions.csv", predictions)
    write_json(root / "reports/audit_decision.json", decision)
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_leakage_v1",
            "status": "PASS",
            "stage": STAGE,
            "evidence_label": EVIDENCE_LABEL,
            "claim_scope": CLAIM_SCOPE,
            "historical_paths_read": False,
            "eval_labels_used_for_cvae_fit": False,
            "eval_labels_used_for_classifier_fit": False,
            "eval_labels_used_for_selection": False,
            "eval_labels_used_for_decode_condition": True,
            "eval_labels_used_for_final_diagnostic_scoring": True,
            "legacy_v2_used_for_decision": False,
            "selection_used_target_eval_artifacts": False,
            "claim_firewall": config.claim_firewall.to_payload(),
            **config.claim_firewall.to_payload(),
        },
    )
    decoder_by_candidate = {
        candidate: sum(
            int(row["decoder_forwards"])
            for row in jobs
            if row["candidate"] == candidate
        )
        for candidate in AUDIT_CANDIDATES
    }
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": RUNTIME_SUMMARY_SCHEMA,
            "stage": STAGE,
            "evidence_label": EVIDENCE_LABEL,
            "claim_scope": CLAIM_SCOPE,
            "job_count": len(jobs),
            "legacy_job_count": sum(
                row["candidate"] == LEGACY_CANDIDATE for row in jobs
            ),
            "controlled_job_count": sum(
                row["candidate"] != LEGACY_CANDIDATE for row in jobs
            ),
            "controlled_pair_count": len(
                {record.pair_id for record in records if not record.is_legacy}
            ),
            "optimizer_updates": sum(int(row["optimizer_steps"]) for row in jobs),
            "legacy_decoder_forwards": decoder_by_candidate[LEGACY_CANDIDATE],
            "fixed_one_epsilon_decoder_forwards": decoder_by_candidate[
                FIXED_ONE_EPSILON_CANDIDATE
            ],
            "antithetic_decoder_forwards": decoder_by_candidate[
                FIXED_ANTITHETIC_CANDIDATE
            ],
            "decoder_forwards": sum(
                int(row["decoder_forwards"]) for row in jobs
            ),
            "epsilon_consumptions": sum(
                int(row["epsilon_consumptions"]) for row in jobs
            ),
            "peak_cuda_bytes_max": max(int(row["peak_cuda_bytes"]) for row in jobs),
            "checkpoint_cache_hits": sum(
                row["checkpoint_cache_status"] == "HIT" for row in jobs
            ),
            "wall_seconds": float(wall_seconds),
            "snapshot_hash": snapshot.snapshot_hash,
            "claim_firewall": config.claim_firewall.to_payload(),
            **config.claim_firewall.to_payload(),
        },
    )


def _validate_config_snapshot_alignment(
    config: AuditConfig,
    records: Sequence[AuditKeyRecord],
) -> None:
    coordinates = {
        (record.center, record.initialization_seed, record.candidate)
        for record in records
    }
    expected = {
        (center, seed, candidate)
        for center in config.centers
        for seed in config.initialization_seeds
        for candidate in config.candidates
    }
    if coordinates != expected:
        raise ProtocolError("Audit config and promoted snapshot key panels differ.")
    if any(
        record.candidate == LEGACY_CANDIDATE and record.pair_id is not None
        for record in records
    ):
        raise ProtocolError("Legacy replay keys cannot enter controlled pairs.")


def _require_workspace_snapshot_files(root: Path) -> None:
    required = (
        root / "config.resolved.yaml",
        root / "provenance/input_artifacts.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProtocolError(
            "Audit execution requires the workspace-prepared config/provenance files: "
            + ", ".join(missing)
        )


def _write_run_state(
    root: Path,
    *,
    status: str,
    completed_jobs: int,
    expected_jobs: int,
) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_run_state_v1",
            "status": status,
            "stage": STAGE,
            "evidence_label": EVIDENCE_LABEL,
            "claim_scope": CLAIM_SCOPE,
            "completed_jobs": int(completed_jobs),
            "expected_jobs": int(expected_jobs),
            "legacy_v2_used_for_decision": False,
            "may_export_recipe_lock": False,
            "may_feed_stage20": False,
            "may_feed_expert_bank": False,
            "may_feed_generation": False,
            "may_feed_routing": False,
            "may_feed_composition": False,
            "may_feed_downstream": False,
            "may_feed_deployable_selection": False,
            "may_tune_or_select": False,
            "may_support_thesis_claim": False,
        },
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Malformed {label}.")
    return value


def _sequence_of_mappings(
    value: object,
    label: str,
) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"Malformed {label} sequence.")
    return [_mapping(row, label) for row in value]


__all__ = ("run_b_paired_reparameterization_audit",)
