"""Bounded last-quarter parameter-averaging diagnostic for the v2 B block."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...metrics import balanced_accuracy
from ...models import ClassConditionedCVAE
from ...preservation.scoring import chance_normalized_preservation
from ..b_adaptation_pilot.case_balanced_sampler import build_balanced_schedule
from ..b_adaptation_pilot.runner import (
    _fit_real_classifier,
    _real_classifier_spec,
    _seed,
)
from ..b_adaptation_pilot.step_training import (
    StepTrainingSpec,
    _configure_determinism,
    model_state_hash,
)
from .config import (
    CENTERS,
    IDENTITY,
    PREDECESSOR_HASHES,
    PREDECESSOR_PROTOCOL_HASH,
    READOUTS,
    SCHEMA,
    TRAINING_SEEDS,
    StabilityConfig,
)
from .tail_training import (
    TailAverageRuntime,
    checkpoint_payload,
    train_with_tail_average,
)


def run_stability_probe(
    config: StabilityConfig, *, artifact_root: Path | None = None
) -> Path:
    """Execute the twelve-fit stability diagnostic and validate its bundle."""

    root = Path(artifact_root or config.artifact_root).resolve()
    _ensure_directories(root)
    stale_failure = root / "tables/job_failures.csv"
    if stale_failure.is_file():
        stale_failure.unlink()
    import fcntl

    lock_handle = (root / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ProtocolError(
            f"Another B-block stability process holds the run lock: {root}"
        ) from exc

    started = perf_counter()
    resolved_config = root / "config.resolved.yaml"
    provenance = root / "provenance/input_artifacts.json"
    if not resolved_config.is_file() or not provenance.is_file():
        raise ProtocolError(
            "The stability probe must be launched through the workspace."
        )
    predecessor = _load_predecessor(config)
    protocol = _protocol_payload(
        config,
        config_resolved_sha256=_file_sha256(resolved_config),
        input_artifacts_sha256=_file_sha256(provenance),
    )
    protocol_hash = stable_hash(protocol)
    protocol["protocol_hash"] = protocol_hash
    _write_json(root / "manifests/frozen_protocol.json", protocol)
    _write_json(root / "reports/predecessor_audit.json", predecessor["audit"])
    _write_csv(root / "tables/frozen_comparators.csv", predecessor["comparators"])

    jobs = _build_jobs(config, protocol_hash, root, predecessor)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    context = mp.get_context("spawn")
    executors = {
        device: ProcessPoolExecutor(max_workers=1, mp_context=context)
        for device in config.devices
    }
    future_jobs: dict[object, Mapping[str, object]] = {}
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    _write_run_state(root, "RUNNING", 0, 0, len(jobs))
    try:
        for device in config.devices:
            for job in (row for row in jobs if row["device"] == device):
                future = executors[device].submit(_run_job, job)
                future_jobs[future] = job
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "schema_version": "midogpp_b_tail_average_job_failure_v1",
                        "center": str(job["center"]),
                        "training_seed": int(job["training_seed"]),
                        "device": str(job["device"]),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    }
                )
            _write_run_state(
                root, "RUNNING", len(results), len(failures), len(jobs)
            )
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)

    if failures:
        _write_csv(root / "tables/job_failures.csv", failures)
        _write_run_state(
            root, "FAILED_WORKER_JOBS", len(results), len(failures), len(jobs)
        )
        raise ProtocolError(
            f"B-block stability probe had {len(failures)} failed job(s); "
            "see tables/job_failures.csv."
        )

    results.sort(key=lambda row: (str(row["center"]), int(row["training_seed"])))
    metrics = [
        dict(metric)
        for result in results
        for metric in result["metrics"]  # type: ignore[index]
    ]
    predictions = [
        dict(prediction)
        for result in results
        for prediction in result["prediction_rows"]  # type: ignore[index]
    ]
    schedule_audit = [
        dict(row)
        for result in results
        for row in result["schedule_audit"]  # type: ignore[index]
    ]
    _write_csv(
        root / "tables/job_inventory.csv",
        [_without_nested(result) for result in results],
    )
    _write_csv(root / "tables/stability_metrics.csv", metrics)
    _write_csv(root / "tables/heldout_predictions.csv", predictions)
    _write_csv(root / "tables/case_class_sampling_audit.csv", schedule_audit)
    decision = stability_decision(
        metrics,
        predecessor["comparators"],
        config.gates,
        endpoint_replay_exact=all(
            bool(result["endpoint_replay_exact"]) for result in results
        ),
    )
    _write_json(root / "reports/stability_decision.json", decision)
    _write_csv(root / "tables/gate_audit.csv", decision["gate_audit"])
    _write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "midogpp_b_tail_average_leakage_report_v1",
            "status": "PASS",
            "surface": "already_consumed_train_case_holdouts",
            "canonical_validation_or_test_features_used": False,
            "fresh_prepared_frames_fitted": False,
            "predecessor_prepared_arrays_reused_exactly": True,
            "heldout_labels_used_for_classifier_fit": False,
            "heldout_labels_used_for_cvae_fit": False,
            "heldout_labels_used_for_scoring": True,
            "heldout_labels_used_for_diagnostic_progression_decision": True,
            "heldout_labels_used_for_confirmation": False,
            "prior_or_generation_evaluated": False,
            "claim_scope": "diagnostic_only",
            "may_feed_expert_bank": False,
        },
    )
    _write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_b_tail_average_runtime_v1",
            "wall_seconds": perf_counter() - started,
            "devices": list(config.devices),
            "workers_per_device": 1,
            "cpu_threads_per_worker": config.cpu_threads_per_worker,
            "n_jobs": len(results),
            "fresh_fits": sum(result["cache_status"] == "miss" for result in results),
            "checkpoint_hits": sum(
                result["cache_status"] == "hit" for result in results
            ),
            "optimizer_steps": sum(int(result["optimizer_steps"]) for result in results),
            "tail_states_averaged": sum(
                int(result["tail_state_count"]) for result in results
            ),
            "peak_cuda_bytes_max": max(
                int(result["peak_cuda_bytes"]) for result in results
            ),
        },
    )
    _write_run_state(root, "VALIDATING", len(results), 0, len(jobs))
    _write_content_index(root)
    from .validation import validate_stability_bundle

    validation = validate_stability_bundle(root)
    _write_json(root / "reports/validation_report.json", validation)
    final_status = "COMPLETE" if validation["status"] == "PASS" else "FAILED_VALIDATION"
    _write_run_state(root, final_status, len(results), 0, len(jobs))
    if validation["status"] != "PASS":
        raise ProtocolError(
            "B-block stability bundle failed validation: "
            + "; ".join(str(value) for value in validation["errors"][:5])
        )
    return root


def _load_predecessor(config: StabilityConfig) -> dict[str, object]:
    root = config.predecessor_root
    jobs = _read_csv(root / "tables/job_inventory.csv")
    metrics = _read_csv(root / "tables/pilot_metrics.csv")
    predictions = _read_csv(root / "tables/heldout_predictions.csv")
    schedule = _read_csv(root / "tables/case_class_sampling_audit.csv")
    content = _json(root / "manifests/content_index.json")
    indexed = {
        str(row["path"]): str(row["sha256"])
        for row in content.get("files", [])
        if isinstance(row, Mapping)
    }
    block_jobs = [
        row
        for row in jobs
        if row["arm"] == config.arm
        and row["center"] in config.centers
        and int(row["training_seed"]) in config.training_seeds
    ]
    expected = {
        (center, str(seed)) for center in config.centers for seed in config.training_seeds
    }
    keys = [(row["center"], row["training_seed"]) for row in block_jobs]
    if len(block_jobs) != 12 or set(keys) != expected or len(keys) != len(set(keys)):
        raise ProtocolError("Predecessor B-block job coverage is not exact 4x3.")

    comparator_roles = {
        ("a_global_pca128", "decode_mu"),
        ("b_joint_pca128", "decode_mu"),
        ("b_block_pca96_32", "decode_mu"),
    }
    comparators = [
        dict(row)
        for row in metrics
        if (row["arm"], row["representation_role"]) in comparator_roles
        and row["center"] in config.centers
        and int(row["training_seed"]) in config.training_seeds
        and row["generation_seed"] == ""
    ]
    if len(comparators) != 36:
        raise ProtocolError("Predecessor comparator coverage is not exact.")
    block_predictions = [
        dict(row)
        for row in predictions
        if row["arm"] == config.arm
        and row["representation_role"] == "decode_mu"
        and row["center"] in config.centers
        and int(row["training_seed"]) in config.training_seeds
    ]
    block_schedule = [
        dict(row)
        for row in schedule
        if row["arm"] == config.arm
        and row["center"] in config.centers
        and int(row["training_seed"]) in config.training_seeds
    ]
    for job in block_jobs:
        center = job["center"]
        prepared_relative = f"prepared/{center}/{config.arm}/arrays.npz"
        prepared = root / prepared_relative
        if (
            not prepared.is_file()
            or prepared_relative not in indexed
            or _file_sha256(prepared) != indexed[prepared_relative]
        ):
            raise ProtocolError(
                f"Predecessor prepared array binding failed: {prepared_relative}"
            )
        checkpoint = Path(job["checkpoint_path"])
        checkpoint_relative = str(checkpoint.resolve().relative_to(root))
        if (
            not checkpoint.is_file()
            or checkpoint_relative not in indexed
            or _file_sha256(checkpoint) != indexed[checkpoint_relative]
        ):
            raise ProtocolError(
                f"Predecessor checkpoint binding failed: {checkpoint_relative}"
            )
        job["prepared_path"] = str(prepared)
        job["prepared_sha256"] = indexed[prepared_relative]
        job["predecessor_checkpoint_sha256"] = indexed[checkpoint_relative]

    audit = {
        "schema_version": "midogpp_b_tail_average_predecessor_audit_v1",
        "status": "PASS",
        "predecessor_root": str(root),
        "predecessor_protocol_hash": PREDECESSOR_PROTOCOL_HASH,
        "predecessor_file_hashes": dict(PREDECESSOR_HASHES),
        "bound_job_count": len(block_jobs),
        "bound_comparator_count": len(comparators),
        "bound_prediction_count": len(block_predictions),
        "bound_schedule_row_count": len(block_schedule),
        "prepared_arrays_reused": True,
        "fresh_training_required": True,
        "predecessor_outcomes_inspected": True,
        "confirmation_eligible": False,
    }
    return {
        "jobs": block_jobs,
        "comparators": comparators,
        "predictions": block_predictions,
        "schedule": block_schedule,
        "audit": audit,
    }


def _build_jobs(
    config: StabilityConfig,
    protocol_hash: str,
    root: Path,
    predecessor: Mapping[str, object],
) -> list[dict[str, object]]:
    predecessor_jobs = predecessor["jobs"]
    predecessor_predictions = predecessor["predictions"]
    predecessor_schedule = predecessor["schedule"]
    jobs: list[dict[str, object]] = []
    for source in predecessor_jobs:  # type: ignore[assignment]
        center = str(source["center"])
        training_seed = int(source["training_seed"])
        expected_predictions = [
            row
            for row in predecessor_predictions  # type: ignore[union-attr]
            if row["center"] == center
            and int(row["training_seed"]) == training_seed
        ]
        expected_schedule = [
            row
            for row in predecessor_schedule  # type: ignore[union-attr]
            if row["center"] == center
            and int(row["training_seed"]) == training_seed
        ]
        jobs.append(
            {
                "center": center,
                "arm": config.arm,
                "training_seed": training_seed,
                "device": str(source["device"]),
                "cpu_threads": config.cpu_threads_per_worker,
                "artifact_root": str(root),
                "protocol_hash": protocol_hash,
                "prepared_path": str(source["prepared_path"]),
                "prepared_sha256": str(source["prepared_sha256"]),
                "frame_hash": str(source["frame_hash"]),
                "fit_row_hash": str(source["fit_row_hash"]),
                "eval_row_hash": str(source["eval_row_hash"]),
                "predecessor_training_key_hash": str(source["training_key_hash"]),
                "predecessor_checkpoint_hash": str(source["checkpoint_hash"]),
                "predecessor_checkpoint_sha256": str(
                    source["predecessor_checkpoint_sha256"]
                ),
                "predecessor_initialization_hash": str(
                    source["initialization_hash"]
                ),
                "predecessor_schedule_hash": str(source["schedule_hash"]),
                "predecessor_posterior_stream_hash": str(
                    source["posterior_stream_hash"]
                ),
                "predecessor_endpoint_metric": _one(
                    [
                        row
                        for row in predecessor["comparators"]  # type: ignore[index]
                        if row["arm"] == config.arm
                        and row["center"] == center
                        and int(row["training_seed"]) == training_seed
                    ],
                    "predecessor endpoint metric",
                ),
                "predecessor_predictions": expected_predictions,
                "predecessor_schedule_rows": expected_schedule,
                "classifier_c": config.classifier_c,
                "minimum_real_bacc": config.minimum_real_bacc,
                "training_spec": _training_spec(config).to_payload(),
                "tail_steps": list(config.tail_steps),
            }
        )
    if {str(job["device"]) for job in jobs} != set(config.devices):
        raise ProtocolError("Predecessor device assignment does not cover both GPUs.")
    return sorted(
        jobs, key=lambda row: (str(row["center"]), int(row["training_seed"]))
    )


def _run_job(job: Mapping[str, object]) -> dict[str, object]:
    import numpy as np
    import torch

    started = perf_counter()
    device = str(job["device"])
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    _configure_determinism(int(job["cpu_threads"]))
    arrays_path = Path(str(job["prepared_path"]))
    if _file_sha256(arrays_path) != str(job["prepared_sha256"]):
        raise ProtocolError("Prepared-array hash changed before worker execution.")
    arrays = np.load(arrays_path, allow_pickle=False)
    required = {
        "x_fit",
        "y_fit",
        "case_fit",
        "sample_fit",
        "x_eval",
        "y_eval",
        "case_eval",
        "sample_eval",
    }
    if set(arrays.files) != required:
        raise ProtocolError("Prepared-array schema differs from v2.")
    x_fit, y_fit = arrays["x_fit"], arrays["y_fit"]
    case_fit, sample_fit = arrays["case_fit"], arrays["sample_fit"]
    x_eval, y_eval = arrays["x_eval"], arrays["y_eval"]
    case_eval, sample_eval = arrays["case_eval"], arrays["sample_eval"]
    center = str(job["center"])
    training_seed = int(job["training_seed"])
    spec = _spec_from_payload(job["training_spec"])  # type: ignore[arg-type]

    schedule_seed = _seed(
        PREDECESSOR_PROTOCOL_HASH, center, training_seed, "case_class_schedule"
    )
    schedule = build_balanced_schedule(
        y_fit,
        case_fit,
        sample_fit,
        steps=spec.optimizer_steps,
        batch_size=spec.batch_size,
        seed=schedule_seed,
    )
    pairing_key = stable_hash(
        {
            "protocol_hash": PREDECESSOR_PROTOCOL_HASH,
            "center": center,
            "training_seed": training_seed,
            "schedule_hash": schedule.stream_hash,
            "paired_across_arms": True,
        }
    )
    training_key = stable_hash(
        {
            "schema_version": "midogpp_b_tail_average_training_key_v1",
            "protocol_hash": job["protocol_hash"],
            "predecessor_protocol_hash": PREDECESSOR_PROTOCOL_HASH,
            "center": center,
            "arm": job["arm"],
            "training_seed": training_seed,
            "prepared_sha256": job["prepared_sha256"],
            "predecessor_checkpoint_sha256": job[
                "predecessor_checkpoint_sha256"
            ],
            "frame_hash": job["frame_hash"],
            "fit_row_hash": job["fit_row_hash"],
            "eval_row_hash": job["eval_row_hash"],
            "schedule_hash": schedule.stream_hash,
            "training_spec_hash": spec.hash,
            "tail_steps": job["tail_steps"],
        }
    )
    metadata = {
        "center": center,
        "arm": job["arm"],
        "training_seed": training_seed,
        "protocol_hash": job["protocol_hash"],
        "predecessor_protocol_hash": PREDECESSOR_PROTOCOL_HASH,
        "predecessor_training_key_hash": job["predecessor_training_key_hash"],
        "predecessor_checkpoint_hash": job["predecessor_checkpoint_hash"],
        "predecessor_checkpoint_sha256": job["predecessor_checkpoint_sha256"],
        "prepared_sha256": job["prepared_sha256"],
        "frame_hash": job["frame_hash"],
        "fit_row_hash": job["fit_row_hash"],
        "eval_row_hash": job["eval_row_hash"],
        "schedule_seed": schedule_seed,
        "schedule_hash": schedule.stream_hash,
        "pairing_key": pairing_key,
        "training_spec": spec.to_payload(),
        "tail_steps": list(job["tail_steps"]),
    }
    checkpoint_path = (
        Path(str(job["artifact_root"])) / "checkpoints" / f"{training_key}.pt"
    )
    runtime, cache_status = _fit_or_load(
        checkpoint_path,
        x_fit=x_fit,
        y_fit=y_fit,
        schedule=schedule,
        spec=spec,
        pairing_key=pairing_key,
        training_key=training_key,
        device=device,
        cpu_threads=int(job["cpu_threads"]),
        tail_steps=job["tail_steps"],  # type: ignore[arg-type]
        metadata=metadata,
    )
    replay_fields = {
        "endpoint_hash": (
            runtime.endpoint_hash,
            str(job["predecessor_checkpoint_hash"]),
        ),
        "initialization_hash": (
            runtime.initialization_hash,
            str(job["predecessor_initialization_hash"]),
        ),
        "schedule_hash": (
            runtime.schedule_hash,
            str(job["predecessor_schedule_hash"]),
        ),
        "posterior_stream_hash": (
            runtime.posterior_stream_hash,
            str(job["predecessor_posterior_stream_hash"]),
        ),
    }
    mismatches = [
        name for name, (observed, expected) in replay_fields.items()
        if observed != expected
    ]
    if mismatches:
        raise ProtocolError(
            "CONTROL_REPLAY_FAILED: " + ", ".join(sorted(mismatches))
        )
    observed_schedule = [
        {
            "center": center,
            "arm": str(job["arm"]),
            "training_seed": training_seed,
            "schedule_hash": schedule.stream_hash,
            "group": group,
            "exposure": exposure,
        }
        for group, exposure in sorted(schedule.case_class_exposure.items())
    ]
    expected_schedule = [
        {
            key: str(row[key])
            for key in (
                "center",
                "arm",
                "training_seed",
                "schedule_hash",
                "group",
                "exposure",
            )
        }
        for row in job["predecessor_schedule_rows"]  # type: ignore[index]
    ]
    normalized_observed = [
        {key: str(value) for key, value in row.items()} for row in observed_schedule
    ]
    if normalized_observed != expected_schedule:
        raise ProtocolError("CONTROL_REPLAY_FAILED: sampling audit differs from v2.")

    metrics, predictions = _evaluate_pair(
        runtime,
        center=center,
        training_seed=training_seed,
        x_fit=x_fit,
        y_fit=y_fit,
        x_eval=x_eval,
        y_eval=y_eval,
        case_eval=case_eval,
        sample_eval=sample_eval,
        classifier_c=float(job["classifier_c"]),
        minimum_real_bacc=float(job["minimum_real_bacc"]),
    )
    terminal = _one(
        [row for row in metrics if row["readout"] == READOUTS[0]],
        "terminal metric",
    )
    expected_metric = job["predecessor_endpoint_metric"]
    for field in (
        "bacc",
        "positive_recall",
        "specificity",
        "preservation_ratio",
        "real_reference_bacc",
    ):
        if abs(float(terminal[field]) - float(expected_metric[field])) > 1e-12:
            raise ProtocolError(
                f"CONTROL_REPLAY_FAILED: endpoint {field} differs from v2."
            )
    for field in ("tp", "fn", "tn", "fp"):
        if int(terminal[field]) != int(expected_metric[field]):
            raise ProtocolError(
                f"CONTROL_REPLAY_FAILED: endpoint {field} differs from v2."
            )
    terminal_predictions = [
        row for row in predictions if row["readout"] == READOUTS[0]
    ]
    expected_predictions = [
        {
            "sample_id": str(row["sample_id"]),
            "case_id": str(row["case_id"]),
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }
        for row in job["predecessor_predictions"]  # type: ignore[index]
    ]
    observed_predictions = [
        {
            "sample_id": str(row["sample_id"]),
            "case_id": str(row["case_id"]),
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        }
        for row in terminal_predictions
    ]
    if observed_predictions != expected_predictions:
        raise ProtocolError("CONTROL_REPLAY_FAILED: endpoint predictions differ from v2.")

    record = {
        "schema_version": "midogpp_b_tail_average_job_v1",
        "center": center,
        "arm": str(job["arm"]),
        "training_seed": training_seed,
        "device": device,
        "training_key_hash": training_key,
        "checkpoint_path": str(checkpoint_path),
        "endpoint_hash": runtime.endpoint_hash,
        "averaged_hash": runtime.averaged_hash,
        "initialization_hash": runtime.initialization_hash,
        "schedule_hash": runtime.schedule_hash,
        "posterior_stream_hash": runtime.posterior_stream_hash,
        "averaging_derivation_hash": runtime.averaging_derivation_hash,
        "tail_state_count": runtime.tail_state_count,
        "tail_start_step": runtime.tail_steps[0],
        "tail_end_step": runtime.tail_steps[-1],
        "optimizer_steps": spec.optimizer_steps,
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
        "cache_status": cache_status,
        "elapsed_seconds": perf_counter() - started,
        "predecessor_training_key_hash": job["predecessor_training_key_hash"],
        "predecessor_checkpoint_hash": job["predecessor_checkpoint_hash"],
        "predecessor_checkpoint_sha256": job["predecessor_checkpoint_sha256"],
        "prepared_sha256": job["prepared_sha256"],
        "frame_hash": job["frame_hash"],
        "fit_row_hash": job["fit_row_hash"],
        "eval_row_hash": job["eval_row_hash"],
        "endpoint_replay_exact": True,
        "metrics": metrics,
        "prediction_rows": predictions,
        "schedule_audit": observed_schedule,
    }
    _write_json(
        Path(str(job["artifact_root"])) / "jobs" / f"{training_key}.json",
        _without_nested(record),
    )
    return record


def _fit_or_load(
    path: Path,
    *,
    x_fit: object,
    y_fit: object,
    schedule: object,
    spec: StepTrainingSpec,
    pairing_key: str,
    training_key: str,
    device: str,
    cpu_threads: int,
    tail_steps: Sequence[int],
    metadata: Mapping[str, object],
) -> tuple[TailAverageRuntime, str]:
    import torch

    if path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload["schema_version"] != "midogpp_b_tail_average_checkpoint_v1"
                or payload["training_key_hash"] != training_key
                or payload["metadata"] != dict(metadata)
                or payload["schedule_hash"] != schedule.stream_hash
                or tuple(int(v) for v in payload["tail_steps"])
                != tuple(int(v) for v in tail_steps)
                or int(payload["tail_state_count"]) != 250
            ):
                raise ProtocolError("Tail-average checkpoint provenance mismatch.")
            expected_steps = [1, *range(100, 1001, 100)]
            diagnostics = payload.get("diagnostics", [])
            if (
                [int(row["step"]) for row in diagnostics] != expected_steps
                or int(diagnostics[-1]["tail_state_count"]) != 250
                or {
                    int(row["step"]): int(row["tail_state_count"])
                    for row in diagnostics
                    if int(row["step"]) in {800, 900, 1000}
                }
                != {800: 50, 900: 150, 1000: 250}
            ):
                raise ProtocolError("Tail-average checkpoint diagnostics mismatch.")
            endpoint = ClassConditionedCVAE(
                input_dim=128, hidden_dim=512, latent_dim=32
            )
            averaged = ClassConditionedCVAE(
                input_dim=128, hidden_dim=512, latent_dim=32
            )
            endpoint.load_state_dict(payload["endpoint_state_dict"], strict=True)
            averaged.load_state_dict(payload["averaged_state_dict"], strict=True)
            if not all(
                torch.isfinite(tensor).all()
                for state in (
                    payload["endpoint_state_dict"],
                    payload["averaged_state_dict"],
                )
                for tensor in state.values()
            ):
                raise ProtocolError("Tail-average checkpoint contains nonfinite state.")
            if (
                model_state_hash(endpoint) != payload["endpoint_hash"]
                or model_state_hash(averaged) != payload["averaged_hash"]
            ):
                raise ProtocolError("Tail-average checkpoint content hash mismatch.")
            expected_derivation = stable_hash(
                {
                    "schema_version": "midogpp_b_tail_average_derivation_v1",
                    "method": "uniform_fp32_online_parameter_mean_v1",
                    "update_timing": "after_optimizer_step",
                    "tail_steps": list(range(751, 1001)),
                    "tail_step_hash": stable_hash(list(range(751, 1001))),
                    "tail_state_count": 250,
                    "accumulator_dtype": "float32",
                    "uniform_weight": 1.0 / 250.0,
                    "endpoint_hash": payload["endpoint_hash"],
                    "averaged_hash": payload["averaged_hash"],
                }
            )
            if payload["averaging_derivation_hash"] != expected_derivation:
                raise ProtocolError("Tail-average checkpoint derivation mismatch.")
            return (
                TailAverageRuntime(
                    endpoint_model=endpoint.to(device),
                    averaged_model=averaged.to(device),
                    device=device,
                    training_key_hash=training_key,
                    endpoint_hash=str(payload["endpoint_hash"]),
                    averaged_hash=str(payload["averaged_hash"]),
                    initialization_hash=str(payload["initialization_hash"]),
                    schedule_hash=str(payload["schedule_hash"]),
                    posterior_stream_hash=str(payload["posterior_stream_hash"]),
                    averaging_derivation_hash=str(
                        payload["averaging_derivation_hash"]
                    ),
                    tail_steps=tuple(int(v) for v in payload["tail_steps"]),
                    tail_state_count=int(payload["tail_state_count"]),
                    diagnostics=tuple(payload["diagnostics"]),
                    peak_cuda_bytes=int(payload.get("peak_cuda_bytes", 0)),
                ),
                "hit",
            )
        except Exception as exc:
            raise ProtocolError(
                f"Existing tail-average checkpoint is invalid: {path}"
            ) from exc
    runtime = train_with_tail_average(
        x_fit,
        y_fit,
        schedule=schedule,  # type: ignore[arg-type]
        spec=spec,
        pairing_key=pairing_key,
        training_key_hash=training_key,
        device=device,
        tail_steps=tail_steps,
        cpu_threads=cpu_threads,
    )
    _atomic_torch_save(path, checkpoint_payload(runtime, metadata=metadata))
    return runtime, "miss"


def _evaluate_pair(
    runtime: TailAverageRuntime,
    *,
    center: str,
    training_seed: int,
    x_fit: object,
    y_fit: object,
    x_eval: object,
    y_eval: object,
    case_eval: object,
    sample_eval: object,
    classifier_c: float,
    minimum_real_bacc: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import numpy as np
    import torch

    classifier = _fit_real_classifier(
        x_fit, y_fit, _real_classifier_spec(classifier_c)
    )
    real_predictions = classifier.predict(x_eval)
    real_bacc = balanced_accuracy(y_eval, real_predictions)
    if real_bacc < minimum_real_bacc:
        raise ProtocolError(
            f"B-block real reference failed denominator at center={center}."
        )
    x_tensor = torch.as_tensor(
        np.asarray(x_eval), dtype=torch.float32, device=runtime.device
    )
    y_tensor = torch.as_tensor(
        np.asarray(y_eval), dtype=torch.long, device=runtime.device
    )
    metrics: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for readout, model in (
        (READOUTS[0], runtime.endpoint_model),
        (READOUTS[1], runtime.averaged_model),
    ):
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(x_tensor, y_tensor)
            decoded = model.decode(mu, y_tensor).cpu().numpy()
        predicted = classifier.predict(decoded)
        metrics.append(
            _metric_row(
                center=center,
                training_seed=training_seed,
                readout=readout,
                truth=y_eval,
                predictions=predicted,
                real_bacc=real_bacc,
                minimum_real_bacc=minimum_real_bacc,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                center=center,
                training_seed=training_seed,
                readout=readout,
                sample_ids=sample_eval,
                case_ids=case_eval,
                truth=y_eval,
                predictions=predicted,
            )
        )
    return metrics, prediction_rows


def _metric_row(
    *,
    center: str,
    training_seed: int,
    readout: str,
    truth: Sequence[int],
    predictions: Sequence[int],
    real_bacc: float,
    minimum_real_bacc: float,
) -> dict[str, object]:
    counts = _confusion(truth, predictions)
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    specificity = counts["tn"] / max(1, counts["tn"] + counts["fp"])
    bacc = 0.5 * (recall + specificity)
    return {
        "schema_version": "midogpp_b_tail_average_metric_v1",
        "center": center,
        "arm": "b_block_pca96_32",
        "training_seed": training_seed,
        "readout": readout,
        "bacc": bacc,
        "positive_recall": recall,
        "specificity": specificity,
        "preservation_ratio": chance_normalized_preservation(
            bacc, real_bacc, minimum_real_bacc=minimum_real_bacc
        ),
        "real_reference_bacc": real_bacc,
        **counts,
        "n_positive": counts["tp"] + counts["fn"],
        "n_negative": counts["tn"] + counts["fp"],
        "heldout_class_used_for_cvae_conditioning": True,
        "heldout_labels_used_for_classifier_fit": False,
        "heldout_labels_used_for_cvae_fit": False,
        "heldout_labels_used_for_scoring": True,
        "heldout_labels_used_for_diagnostic_progression_decision": True,
        "heldout_labels_used_for_confirmation": False,
        "prior_or_generation_metric": False,
        "oracle_eligible": False,
        "claim_scope": "diagnostic_only",
    }


def _prediction_rows(
    *,
    center: str,
    training_seed: int,
    readout: str,
    sample_ids: Sequence[object],
    case_ids: Sequence[object],
    truth: Sequence[int],
    predictions: Sequence[int],
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "midogpp_b_tail_average_prediction_v1",
            "center": center,
            "arm": "b_block_pca96_32",
            "training_seed": training_seed,
            "readout": readout,
            "sample_id": str(sample_id),
            "case_id": str(case_id),
            "y_true": int(y_true),
            "y_pred": int(y_pred),
            "heldout_class_used_for_cvae_conditioning": True,
            "heldout_labels_used_for_classifier_fit": False,
            "heldout_labels_used_for_cvae_fit": False,
            "heldout_labels_used_for_scoring": True,
            "heldout_labels_used_for_diagnostic_progression_decision": True,
            "heldout_labels_used_for_confirmation": False,
            "oracle_eligible": False,
        }
        for sample_id, case_id, y_true, y_pred in zip(
            sample_ids, case_ids, truth, predictions, strict=True
        )
    ]


def stability_decision(
    metrics: Sequence[Mapping[str, object]],
    comparators: Sequence[Mapping[str, object]],
    gates: Mapping[str, float],
    *,
    endpoint_replay_exact: bool,
) -> dict[str, object]:
    """Recompute the frozen diagnostic progression rule."""

    candidate = {
        (str(row["center"]), int(row["training_seed"])): row
        for row in metrics
        if row["readout"] == READOUTS[1]
    }
    endpoint = {
        (str(row["center"]), int(row["training_seed"])): row
        for row in metrics
        if row["readout"] == READOUTS[0]
    }
    comparator_maps = {
        arm: {
            (str(row["center"]), int(row["training_seed"])): row
            for row in comparators
            if row["arm"] == arm
        }
        for arm in ("a_global_pca128", "b_joint_pca128", "b_block_pca96_32")
    }
    expected = sorted(
        (center, seed) for center in CENTERS for seed in TRAINING_SEEDS
    )
    if (
        set(candidate) != set(expected)
        or set(endpoint) != set(expected)
        or any(set(rows) != set(expected) for rows in comparator_maps.values())
    ):
        raise ProtocolError("Stability decision coverage is not exact 4x3.")

    mean = lambda values: sum(values) / len(values)
    ratio = lambda row: float(row["preservation_ratio"])
    field = lambda row, name: float(row[name])
    a_map = comparator_maps["a_global_pca128"]
    joint_map = comparator_maps["b_joint_pca128"]
    predecessor_block = comparator_maps["b_block_pca96_32"]
    center_delta_a = {
        center: mean(
            [
                ratio(candidate[(center, seed)]) - ratio(a_map[(center, seed)])
                for seed in TRAINING_SEEDS
            ]
        )
        for center in CENTERS
    }
    center_delta_joint = {
        center: mean(
            [
                ratio(candidate[(center, seed)])
                - ratio(joint_map[(center, seed)])
                for seed in TRAINING_SEEDS
            ]
        )
        for center in CENTERS
    }
    seed_mean_ratios = {
        seed: mean([ratio(candidate[(center, seed)]) for center in CENTERS])
        for seed in TRAINING_SEEDS
    }
    center_direction_ranges = {
        center: {
            "positive_recall": max(
                field(candidate[(center, seed)], "positive_recall")
                for seed in TRAINING_SEEDS
            )
            - min(
                field(candidate[(center, seed)], "positive_recall")
                for seed in TRAINING_SEEDS
            ),
            "specificity": max(
                field(candidate[(center, seed)], "specificity")
                for seed in TRAINING_SEEDS
            )
            - min(
                field(candidate[(center, seed)], "specificity")
                for seed in TRAINING_SEEDS
            ),
        }
        for center in CENTERS
    }
    maximum_direction_range = max(
        value
        for center in center_direction_ranges.values()
        for value in center.values()
    )
    macro_recall_by_seed = {
        seed: mean(
            [
                field(candidate[(center, seed)], "positive_recall")
                for center in CENTERS
            ]
        )
        for seed in TRAINING_SEEDS
    }
    macro_specificity_by_seed = {
        seed: mean(
            [
                field(candidate[(center, seed)], "specificity")
                for center in CENTERS
            ]
        )
        for seed in TRAINING_SEEDS
    }
    center_terminal_deltas = {
        center: {
            name: mean(
                [
                    field(candidate[(center, seed)], name)
                    - field(endpoint[(center, seed)], name)
                    for seed in TRAINING_SEEDS
                ]
            )
            for name in ("positive_recall", "specificity", "preservation_ratio")
        }
        for center in CENTERS
    }
    observations = {
        "endpoint_replay_exact": 1.0 if endpoint_replay_exact else 0.0,
        "real_reference_valid": 1.0
        if all(
            field(row, "real_reference_bacc") >= 0.60
            for row in candidate.values()
        )
        else 0.0,
        "mean_preservation": mean([ratio(row) for row in candidate.values()]),
        "mean_minus_a_preservation": mean(
            [ratio(candidate[key]) - ratio(a_map[key]) for key in expected]
        ),
        "worst_center_minus_a_preservation": min(center_delta_a.values()),
        "mean_minus_a_bacc": mean(
            [
                field(candidate[key], "bacc") - field(a_map[key], "bacc")
                for key in expected
            ]
        ),
        "mean_minus_a_recall": mean(
            [
                field(candidate[key], "positive_recall")
                - field(a_map[key], "positive_recall")
                for key in expected
            ]
        ),
        "mean_minus_a_specificity": mean(
            [
                field(candidate[key], "specificity")
                - field(a_map[key], "specificity")
                for key in expected
            ]
        ),
        "minimum_seed_mean_preservation": min(seed_mean_ratios.values()),
        "seed_mean_preservation_range": max(seed_mean_ratios.values())
        - min(seed_mean_ratios.values()),
        "mean_center_minus_joint_preservation": mean(
            list(center_delta_joint.values())
        ),
        "strict_center_wins_over_joint": float(
            sum(value > 0.0 for value in center_delta_joint.values())
        ),
        "mean_bacc_delta_vs_terminal": mean(
            [
                field(candidate[key], "bacc")
                - field(predecessor_block[key], "bacc")
                for key in expected
            ]
        ),
        "mean_preservation_delta_vs_terminal": mean(
            [ratio(candidate[key]) - ratio(endpoint[key]) for key in expected]
        ),
        "worst_center_preservation_delta_vs_terminal": min(
            values["preservation_ratio"]
            for values in center_terminal_deltas.values()
        ),
        "mean_recall_delta_vs_terminal": mean(
            [
                field(candidate[key], "positive_recall")
                - field(endpoint[key], "positive_recall")
                for key in expected
            ]
        ),
        "mean_specificity_delta_vs_terminal": mean(
            [
                field(candidate[key], "specificity")
                - field(endpoint[key], "specificity")
                for key in expected
            ]
        ),
        "center_5_mean_recall_delta_vs_terminal": center_terminal_deltas["5"][
            "positive_recall"
        ],
        "center_5_mean_specificity_delta_vs_terminal": center_terminal_deltas["5"][
            "specificity"
        ],
        "center_9_mean_recall_delta_vs_terminal": center_terminal_deltas["9"][
            "positive_recall"
        ],
        "center_9_mean_specificity_delta_vs_terminal": center_terminal_deltas["9"][
            "specificity"
        ],
        "maximum_within_center_class_direction_seed_range": maximum_direction_range,
    }
    checks = [
        ("endpoint_replay_exact", "min", observations["endpoint_replay_exact"], 1.0),
        ("real_reference_valid", "min", observations["real_reference_valid"], 1.0),
        (
            "mean_preservation",
            "min",
            observations["mean_preservation"],
            gates["mean_preservation_min"],
        ),
        (
            "mean_minus_a_preservation",
            "min",
            observations["mean_minus_a_preservation"],
            gates["mean_minus_a_preservation_min"],
        ),
        (
            "worst_center_minus_a_preservation",
            "min",
            observations["worst_center_minus_a_preservation"],
            gates["worst_center_minus_a_preservation_min"],
        ),
        (
            "mean_minus_a_bacc",
            "min",
            observations["mean_minus_a_bacc"],
            gates["mean_minus_a_bacc_min"],
        ),
        (
            "mean_minus_a_recall",
            "min",
            observations["mean_minus_a_recall"],
            gates["mean_minus_a_recall_min"],
        ),
        (
            "mean_minus_a_specificity",
            "min",
            observations["mean_minus_a_specificity"],
            gates["mean_minus_a_specificity_min"],
        ),
        (
            "minimum_seed_mean_preservation",
            "min",
            observations["minimum_seed_mean_preservation"],
            gates["minimum_seed_mean_preservation"],
        ),
        (
            "seed_mean_preservation_range",
            "max",
            observations["seed_mean_preservation_range"],
            gates["maximum_seed_mean_preservation_range"],
        ),
        (
            "mean_center_minus_joint_preservation",
            "min",
            observations["mean_center_minus_joint_preservation"],
            gates["mean_center_minus_joint_preservation_min"],
        ),
        (
            "strict_center_wins_over_joint",
            "min",
            observations["strict_center_wins_over_joint"],
            gates["minimum_strict_center_wins_over_joint"],
        ),
        (
            "mean_bacc_delta_vs_terminal",
            "min",
            observations["mean_bacc_delta_vs_terminal"],
            gates["mean_bacc_delta_vs_terminal_min"],
        ),
        (
            "mean_preservation_delta_vs_terminal",
            "min",
            observations["mean_preservation_delta_vs_terminal"],
            gates["mean_preservation_delta_vs_terminal_min"],
        ),
        (
            "worst_center_preservation_delta_vs_terminal",
            "min",
            observations["worst_center_preservation_delta_vs_terminal"],
            gates["worst_center_preservation_delta_vs_terminal_min"],
        ),
        (
            "mean_recall_delta_vs_terminal",
            "min",
            observations["mean_recall_delta_vs_terminal"],
            gates["mean_recall_delta_vs_terminal_min"],
        ),
        (
            "mean_specificity_delta_vs_terminal",
            "min",
            observations["mean_specificity_delta_vs_terminal"],
            gates["mean_specificity_delta_vs_terminal_min"],
        ),
        (
            "center_5_mean_recall_delta_vs_terminal",
            "min",
            observations["center_5_mean_recall_delta_vs_terminal"],
            gates["center_5_mean_recall_delta_vs_terminal_min"],
        ),
        (
            "center_5_mean_specificity_delta_vs_terminal",
            "min",
            observations["center_5_mean_specificity_delta_vs_terminal"],
            gates["center_5_mean_specificity_delta_vs_terminal_min"],
        ),
        (
            "center_9_mean_recall_delta_vs_terminal",
            "min",
            observations["center_9_mean_recall_delta_vs_terminal"],
            gates["center_9_mean_recall_delta_vs_terminal_min"],
        ),
        (
            "center_9_mean_specificity_delta_vs_terminal",
            "min",
            observations["center_9_mean_specificity_delta_vs_terminal"],
            gates["center_9_mean_specificity_delta_vs_terminal_min"],
        ),
        (
            "maximum_within_center_class_direction_seed_range",
            "max",
            observations["maximum_within_center_class_direction_seed_range"],
            gates["maximum_within_center_class_direction_seed_range"],
        ),
    ]
    gate_audit = [
        {
            "gate": name,
            "direction": direction,
            "observed": observed,
            "threshold": threshold,
            "passed": observed >= threshold if direction == "min" else observed <= threshold,
        }
        for name, direction, observed, threshold in checks
    ]
    passed = all(bool(row["passed"]) for row in gate_audit)
    if not endpoint_replay_exact:
        decision = "CONTROL_REPLAY_FAILED"
    elif passed:
        decision = "TAIL_AVERAGING_STABILIZES_B_BLOCK"
    else:
        decision = "TAIL_AVERAGING_INSUFFICIENT"
    return {
        "schema_version": "midogpp_b_tail_average_decision_v1",
        "decision": decision,
        "all_progression_gates_passed": passed,
        "observations": observations,
        "gate_audit": gate_audit,
        "center_minus_a_preservation": center_delta_a,
        "center_minus_joint_preservation": center_delta_joint,
        "seed_mean_preservation": seed_mean_ratios,
        "within_center_class_direction_seed_ranges": center_direction_ranges,
        "descriptive_stricter_maximum_direction_range_0_10_passed": (
            maximum_direction_range <= 0.10
        ),
        "descriptive_macro_recall_seed_range": max(macro_recall_by_seed.values())
        - min(macro_recall_by_seed.values()),
        "descriptive_macro_specificity_seed_range": max(
            macro_specificity_by_seed.values()
        )
        - min(macro_specificity_by_seed.values()),
        "predecessor_prior_status": "FAILED_V2_NOT_RETESTED",
        "prior_or_generation_evaluated": False,
        "claim_scope": "diagnostic_only",
        "confirmation_eligible": False,
        "may_export_recipe_lock": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "next_step_if_pass": "separately_reviewed_b_block_prior_only_replay",
    }


def _protocol_payload(
    config: StabilityConfig,
    *,
    config_resolved_sha256: str,
    input_artifacts_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "experiment": IDENTITY,
        "centers": list(config.centers),
        "training_seeds": list(config.training_seeds),
        "arm": config.arm,
        "predecessor": dict(config.lineage),
        "replay": {
            "protocol_hash": PREDECESSOR_PROTOCOL_HASH,
            "prepared_arrays": "hash_bound_reuse",
            "schedule_rng_pairing": "exact_v2_namespace",
            "terminal_endpoint": "required_exact_control",
            "fresh_training": True,
        },
        "training": _training_spec(config).to_payload(),
        "tail_averaging": {
            "method": "uniform_fp32_online_parameter_mean_v1",
            "update_timing": "after_optimizer_step",
            "start_step": config.tail_steps[0],
            "end_step": config.tail_steps[-1],
            "stride": 1,
            "expected_state_count": len(config.tail_steps),
            "average_optimizer_state": False,
            "heldout_selection": False,
        },
        "evaluation": {
            "roles": list(READOUTS),
            "classifier_c": config.classifier_c,
            "minimum_real_bacc": config.minimum_real_bacc,
            "prior_or_generation_evaluated": False,
        },
        "decision_gates": dict(config.gates),
        "runtime_policy": {
            "devices": list(config.devices),
            "device_assignment": "exact_predecessor_job_inventory",
            "workers_per_device": 1,
            "cpu_threads_per_worker": config.cpu_threads_per_worker,
            "multiprocessing_start_method": "spawn",
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "workspace_snapshot_hashes": {
            "config_resolved_sha256": config_resolved_sha256,
            "input_artifacts_sha256": input_artifacts_sha256,
        },
        "component_hashes": _component_hashes(),
        "library_versions": _library_versions(),
        "test_or_validation_split_used": False,
        "predecessor_outcomes_inspected": True,
        "claim_scope": "diagnostic_only",
        "confirmation_eligible": False,
        "may_export_recipe_lock": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
    }


def _training_spec(config: StabilityConfig) -> StepTrainingSpec:
    return StepTrainingSpec(
        optimizer_steps=config.optimizer_steps,
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta_final=config.beta_final,
        kl_warmup_steps=config.kl_warmup_steps,
        gradient_clip_norm=config.gradient_clip_norm,
    )


def _spec_from_payload(payload: Mapping[str, object]) -> StepTrainingSpec:
    return StepTrainingSpec(
        optimizer_steps=int(payload["optimizer_steps"]),
        batch_size=int(payload["batch_size"]),
        hidden_dim=int(payload["hidden_dim"]),
        latent_dim=int(payload["latent_dim"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        beta_final=float(payload["beta_final"]),
        kl_warmup_steps=int(payload["kl_warmup_steps"]),
        gradient_clip_norm=float(payload["gradient_clip_norm"]),
    )


def _confusion(
    truth: Sequence[int], predictions: Sequence[int]
) -> dict[str, int]:
    pairs = [(int(t), int(p)) for t, p in zip(truth, predictions, strict=True)]
    return {
        "tp": sum(t == 1 and p == 1 for t, p in pairs),
        "fn": sum(t == 1 and p == 0 for t, p in pairs),
        "tn": sum(t == 0 and p == 0 for t, p in pairs),
        "fp": sum(t == 0 and p == 1 for t, p in pairs),
    }


def _one(rows: Sequence[Mapping[str, object]], label: str) -> Mapping[str, object]:
    if len(rows) != 1:
        raise ProtocolError(f"Expected one {label}, observed {len(rows)}.")
    return rows[0]


def _component_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    hashes = {
        name: _file_sha256(directory / name)
        for name in (
            "__init__.py",
            "config.py",
            "runner.py",
            "tail_training.py",
            "validation.py",
        )
    }
    dependencies = {
        "dependency/case_balanced_sampler.py": directory.parent
        / "b_adaptation_pilot/case_balanced_sampler.py",
        "dependency/step_training.py": directory.parent
        / "b_adaptation_pilot/step_training.py",
        "dependency/adaptation_runner.py": directory.parent
        / "b_adaptation_pilot/runner.py",
        "dependency/models/__init__.py": directory.parents[1]
        / "models/__init__.py",
        "dependency/models/cvae.py": directory.parents[1] / "models/cvae.py",
        "dependency/preservation_scoring.py": directory.parents[1]
        / "preservation/scoring.py",
    }
    hashes.update({name: _file_sha256(path) for name, path in dependencies.items()})
    return hashes


def _library_versions() -> dict[str, str]:
    import numpy
    import sklearn
    import torch

    return {
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
    }


def _ensure_directories(root: Path) -> None:
    for relative in ("reports", "manifests", "tables", "checkpoints", "jobs"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _without_nested(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"metrics", "prediction_rows", "schedule_audit"}
    }


def _write_run_state(
    root: Path, status: str, completed: int, failed: int, expected: int
) -> None:
    _write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_b_tail_average_run_state_v1",
            "status": status,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "expected_jobs": expected,
            "claim_scope": "diagnostic_only",
            "may_export_recipe_lock": False,
            "may_feed_expert_bank": False,
        },
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Cannot write empty stability table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected JSON object: {path}")
    return value


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_content_index(root: Path) -> None:
    from .validation import INDEX_EXCLUSIONS

    files = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if (
            path.is_file()
            and relative not in INDEX_EXCLUSIONS
            and ".tmp" not in path.name
        ):
            files.append(
                {
                    "path": relative,
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_json(
        root / "manifests/content_index.json",
        {
            "schema_version": "midogpp_b_tail_average_content_index_v1",
            "files": files,
        },
    )


__all__ = ("run_stability_probe", "stability_decision")
