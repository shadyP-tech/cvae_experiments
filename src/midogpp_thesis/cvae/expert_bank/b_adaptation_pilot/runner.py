"""Orchestration for the bounded canonical-B source-expert adaptation pilot."""

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
from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ....real_features.classifier_reference.protocol import ProtocolError
from ....real_features.classifier_reference.real_feature_frame import (
    load_midogpp_real_feature_frame,
)
from ...metrics import balanced_accuracy
from ...models import ClassConditionedCVAE
from ...preservation.scoring import chance_normalized_preservation
from .block_frame import bridge_a_prefix, fit_pilot_frame
from .case_balanced_sampler import build_balanced_schedule
from .case_split import deterministic_case_holdout
from .config import PILOT_ARMS, PilotConfig
from .conservative_prior import (
    CONDITIONAL_PRIOR_FAMILY,
    STANDARD_PRIOR_FAMILY,
    fit_shrunk_diagonal_prior,
    sample_prior,
)
from .step_training import (
    PilotRuntime,
    StepTrainingSpec,
    checkpoint_payload,
    model_state_hash,
    train_fixed_steps,
)


def run_pilot(config: PilotConfig, *, artifact_root: Path | None = None) -> Path:
    import numpy as np

    root = Path(artifact_root or config.artifact_root).resolve()
    _ensure_directories(root)
    started = perf_counter()
    b_frame = load_midogpp_real_feature_frame(
        manifest_path=config.manifest_path,
        feature_cache_path=config.b_feature_cache_path,
        expected_feature_dim=config.expected_b_dim,
        allow_excluded_center_omission=True,
    )
    b_ids = tuple(row.sample_id for row in b_frame.rows)
    a_embeddings = _load_aligned_a_bridge(config.a_feature_cache_path, b_ids)
    bridge = bridge_a_prefix(b_frame.embeddings, a_embeddings)
    input_hashes = {
        "manifest_hash": b_frame.manifest_hash,
        "b_feature_cache_hash": b_frame.feature_cache_hash,
        "a_feature_cache_hash": _file_sha256(config.a_feature_cache_path),
    }
    protocol_hash = _protocol_hash(config, input_hashes)
    _write_json(root / "reports/a_prefix_bridge.json", bridge)
    _write_json(root / "manifests/frozen_protocol.json", _protocol_payload(config, input_hashes))

    preparation_rows: list[dict[str, object]] = []
    preparation_started = perf_counter()
    for center in config.centers:
        center_indices = [
            index for index, row in enumerate(b_frame.rows) if row.center == center
        ]
        labels = [b_frame.rows[index].label for index in center_indices]
        cases = [b_frame.rows[index].case_id for index in center_indices]
        samples = [b_frame.rows[index].sample_id for index in center_indices]
        holdout = deterministic_case_holdout(
            cases,
            labels,
            validation_fraction=config.validation_fraction,
            seed=config.case_split_seed,
        )
        absolute_fit = [center_indices[index] for index in holdout.fit_indices]
        absolute_eval = [center_indices[index] for index in holdout.eval_indices]
        fit_sample_ids = [b_frame.rows[index].sample_id for index in absolute_fit]
        eval_sample_ids = [b_frame.rows[index].sample_id for index in absolute_eval]
        fit_hash = stable_hash(fit_sample_ids)
        eval_hash = stable_hash(eval_sample_ids)
        for arm in config.arms:
            frame = fit_pilot_frame(
                arm,
                np.asarray(b_frame.embeddings)[absolute_fit],
                fit_sample_hash=fit_hash,
            )
            x_fit = frame.transform(np.asarray(b_frame.embeddings)[absolute_fit])
            x_eval = frame.transform(np.asarray(b_frame.embeddings)[absolute_eval])
            original_fit = np.asarray(b_frame.embeddings)[absolute_fit, : frame.input_dim]
            reconstructed_fit = frame.inverse_transform(x_fit)
            block_mse = []
            cursor = 0
            for block in frame.blocks:
                width = block.stop - block.start
                mse = float(
                    np.mean(
                        (
                            reconstructed_fit[:, cursor : cursor + width]
                            - original_fit[:, block.start : block.stop]
                        )
                        ** 2
                    )
                )
                block_mse.append(mse)
                cursor += width
            prepared_path = root / "prepared" / center / arm / "arrays.npz"
            prepared_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_save_npz(
                prepared_path,
                x_fit=x_fit,
                y_fit=np.asarray([b_frame.rows[i].label for i in absolute_fit], dtype=np.int64),
                case_fit=np.asarray([b_frame.rows[i].case_id for i in absolute_fit], dtype=str),
                sample_fit=np.asarray(fit_sample_ids, dtype=str),
                x_eval=x_eval,
                y_eval=np.asarray([b_frame.rows[i].label for i in absolute_eval], dtype=np.int64),
                case_eval=np.asarray([b_frame.rows[i].case_id for i in absolute_eval], dtype=str),
                sample_eval=np.asarray(eval_sample_ids, dtype=str),
            )
            frame_path = root / "prepared" / center / arm / "frame_state.json"
            _write_json(frame_path, frame.to_payload())
            preparation_rows.append(
                {
                    "center": center,
                    "arm": arm,
                    "frame_hash": frame.state_hash,
                    "fit_row_hash": fit_hash,
                    "eval_row_hash": eval_hash,
                    "n_fit_rows": len(absolute_fit),
                    "n_eval_rows": len(absolute_eval),
                    "n_fit_cases": len(holdout.fit_cases),
                    "n_eval_cases": len(holdout.eval_cases),
                    "fit_cases": json.dumps(holdout.fit_cases),
                    "eval_cases": json.dumps(holdout.eval_cases),
                    "explained_variance_ratio": sum(
                        block.explained_variance_ratio_sum for block in frame.blocks
                    ),
                    "block_reconstruction_mse": json.dumps(block_mse),
                    "prepared_path": str(prepared_path),
                }
            )
    _write_csv(root / "tables/frame_preparation.csv", preparation_rows)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    jobs = []
    prep_by_key = {
        (str(row["center"]), str(row["arm"])): row for row in preparation_rows
    }
    for center in config.centers:
        for arm in config.arms:
            for training_seed in config.training_seeds:
                prep = prep_by_key[(center, arm)]
                jobs.append(
                    {
                        "center": center,
                        "arm": arm,
                        "training_seed": training_seed,
                        "prepared_path": prep["prepared_path"],
                        "frame_hash": prep["frame_hash"],
                        "fit_row_hash": prep["fit_row_hash"],
                        "eval_row_hash": prep["eval_row_hash"],
                        "protocol_hash": protocol_hash,
                        "artifact_root": str(root),
                        "device": "",
                        "cpu_threads": config.cpu_threads_per_worker,
                        "generation_seeds": config.generation_seeds,
                        "generated_per_class": config.generated_per_class,
                        "classifier_c": config.classifier_c,
                        "minimum_real_bacc": config.minimum_real_bacc,
                        "training_spec": _training_spec(config).to_payload(),
                        "conditional_prior": {
                            "rho": config.conditional_prior_rho,
                            "min_rows": config.conditional_prior_min_rows,
                            "min_cases": config.conditional_prior_min_cases,
                            "variance_clip": config.conditional_prior_variance_clip,
                            "max_condition_number": config.conditional_prior_max_condition_number,
                        },
                    }
                )
    jobs.sort(key=lambda row: (str(row["center"]), str(row["arm"]), int(row["training_seed"])))
    device_jobs = {device: [] for device in config.devices}
    for index, job in enumerate(jobs):
        device = config.devices[index % len(config.devices)]
        job["device"] = device
        device_jobs[device].append(job)
    results: list[dict[str, object]] = []
    context = mp.get_context("spawn")
    executors = {
        device: ProcessPoolExecutor(max_workers=1, mp_context=context)
        for device in config.devices
    }
    futures = []
    try:
        for device in config.devices:
            futures.extend(
                executors[device].submit(_run_job, job)
                for job in device_jobs[device]
            )
        for future in as_completed(futures):
            results.append(future.result())
            _write_json(
                root / "reports/run_state.json",
                {
                    "schema_version": "midogpp_b_adaptation_run_state_v1",
                    "status": "RUNNING",
                    "completed_jobs": len(results),
                    "expected_jobs": len(jobs),
                },
            )
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
    results.sort(key=lambda row: (row["center"], row["arm"], row["training_seed"]))
    metric_rows = [
        dict(metric)
        for result in results
        for metric in result["metrics"]  # type: ignore[index]
    ]
    _write_csv(root / "tables/job_inventory.csv", [_without_metrics(row) for row in results])
    _write_csv(root / "tables/pilot_metrics.csv", metric_rows)
    schedule_rows = [
        dict(row)
        for result in results
        for row in result["schedule_audit"]  # type: ignore[index]
    ]
    _write_csv(root / "tables/case_class_sampling_audit.csv", schedule_rows)
    decision = _decision(metric_rows, results, minimum_real_bacc=config.minimum_real_bacc)
    _write_json(root / "reports/pilot_decision.json", decision)
    validation = validate_pilot_bundle(root)
    _write_json(root / "reports/validation_report.json", validation)
    _write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": "midogpp_b_adaptation_runtime_v1",
            "wall_seconds": perf_counter() - started,
            "preparation_seconds": perf_counter() - preparation_started,
            "devices": list(config.devices),
            "n_jobs": len(results),
            "optimizer_steps": sum(int(row["optimizer_steps"]) for row in results),
            "peak_cuda_bytes_max": max(int(row["peak_cuda_bytes"]) for row in results),
        },
    )
    _write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_b_adaptation_run_state_v1",
            "status": "COMPLETE" if validation["status"] == "PASS" else "FAILED_VALIDATION",
            "completed_jobs": len(results),
            "expected_jobs": len(jobs),
            "claim_scope": "diagnostic_only",
            "may_export_recipe_lock": False,
            "may_feed_expert_bank": False,
        },
    )
    _write_content_index(root)
    return root


def _run_job(job: Mapping[str, object]) -> dict[str, object]:
    import numpy as np
    import torch

    started = perf_counter()
    if str(job["device"]).startswith("cuda"):
        torch.cuda.set_device(str(job["device"]))
    arrays = np.load(str(job["prepared_path"]), allow_pickle=False)
    x_fit = arrays["x_fit"]
    y_fit = arrays["y_fit"]
    case_fit = arrays["case_fit"]
    sample_fit = arrays["sample_fit"]
    x_eval = arrays["x_eval"]
    y_eval = arrays["y_eval"]
    center = str(job["center"])
    arm = str(job["arm"])
    training_seed = int(job["training_seed"])
    protocol_hash = str(job["protocol_hash"])
    schedule_seed = _seed(protocol_hash, center, training_seed, "case_class_schedule")
    schedule = build_balanced_schedule(
        y_fit,
        case_fit,
        sample_fit,
        steps=1000,
        batch_size=128,
        seed=schedule_seed,
    )
    spec = _spec_from_payload(job["training_spec"])  # type: ignore[arg-type]
    pairing_key = stable_hash(
        {
            "protocol_hash": protocol_hash,
            "center": center,
            "training_seed": training_seed,
            "schedule_hash": schedule.stream_hash,
            "paired_across_arms": True,
        }
    )
    training_key = stable_hash(
        {
            "schema_version": "midogpp_b_adaptation_training_key_v1",
            "protocol_hash": protocol_hash,
            "center": center,
            "arm": arm,
            "training_seed": training_seed,
            "fit_row_hash": job["fit_row_hash"],
            "frame_hash": job["frame_hash"],
            "schedule_hash": schedule.stream_hash,
            "training_spec_hash": spec.hash,
        }
    )
    root = Path(str(job["artifact_root"]))
    checkpoint_path = root / "checkpoints" / f"{training_key}.pt"
    runtime, cache_status = _fit_or_load_runtime(
        checkpoint_path,
        x_fit=x_fit,
        y_fit=y_fit,
        schedule=schedule,
        spec=spec,
        pairing_key=pairing_key,
        training_key=training_key,
        device=str(job["device"]),
        cpu_threads=int(job["cpu_threads"]),
        metadata={
            "center": center,
            "arm": arm,
            "training_seed": training_seed,
            "frame_hash": job["frame_hash"],
            "fit_row_hash": job["fit_row_hash"],
            "eval_row_hash": job["eval_row_hash"],
            "protocol_hash": protocol_hash,
        },
    )
    metrics, prior_record = _evaluate(
        runtime,
        x_fit=x_fit,
        y_fit=y_fit,
        case_fit=case_fit,
        x_eval=x_eval,
        y_eval=y_eval,
        center=center,
        arm=arm,
        training_seed=training_seed,
        generation_seeds=tuple(int(v) for v in job["generation_seeds"]),  # type: ignore[arg-type]
        generated_per_class=int(job["generated_per_class"]),
        classifier_c=float(job["classifier_c"]),
        minimum_real_bacc=float(job["minimum_real_bacc"]),
        prior_config=job["conditional_prior"],  # type: ignore[arg-type]
    )
    job_record = {
        "schema_version": "midogpp_b_adaptation_job_v1",
        "center": center,
        "arm": arm,
        "training_seed": training_seed,
        "device": str(job["device"]),
        "training_key_hash": training_key,
        "checkpoint_hash": runtime.checkpoint_hash,
        "checkpoint_path": str(checkpoint_path),
        "frame_hash": job["frame_hash"],
        "fit_row_hash": job["fit_row_hash"],
        "eval_row_hash": job["eval_row_hash"],
        "schedule_hash": schedule.stream_hash,
        "initialization_hash": runtime.initialization_hash,
        "posterior_stream_hash": runtime.posterior_stream_hash,
        "optimizer_steps": spec.optimizer_steps,
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
        "cache_status": cache_status,
        "elapsed_seconds": perf_counter() - started,
        "prior_state_hash": prior_record["state_hash"],
        "prior_realized_family": prior_record["realized_family"],
        "metrics": metrics,
        "schedule_audit": [
            {
                "center": center,
                "arm": arm,
                "training_seed": training_seed,
                "schedule_hash": schedule.stream_hash,
                "group": group,
                "exposure": exposure,
            }
            for group, exposure in sorted(schedule.case_class_exposure.items())
        ],
    }
    _write_json(root / "jobs" / f"{training_key}.json", _without_metrics(job_record))
    _write_json(root / "priors" / f"{training_key}.json", prior_record)
    return job_record


def _fit_or_load_runtime(
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
    metadata: Mapping[str, object],
) -> tuple[PilotRuntime, str]:
    import torch

    if path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload["schema_version"] != "midogpp_b_adaptation_checkpoint_v1"
                or payload["training_key_hash"] != training_key
                or payload["metadata"] != dict(metadata)
            ):
                raise ProtocolError("Checkpoint provenance mismatch.")
            model = ClassConditionedCVAE(input_dim=128, hidden_dim=512, latent_dim=32)
            model.load_state_dict(payload["state_dict"], strict=True)
            if model_state_hash(model) != payload["checkpoint_hash"]:
                raise ProtocolError("Checkpoint content hash mismatch.")
            model = model.to(device)
            return (
                PilotRuntime(
                    model=model,
                    device=device,
                    training_key_hash=training_key,
                    checkpoint_hash=str(payload["checkpoint_hash"]),
                    initialization_hash=str(payload["initialization_hash"]),
                    schedule_hash=str(payload["schedule_hash"]),
                    posterior_stream_hash=str(payload["posterior_stream_hash"]),
                    diagnostics=tuple(payload["diagnostics"]),
                    peak_cuda_bytes=int(payload.get("peak_cuda_bytes", 0)),
                ),
                "hit",
            )
        except Exception as exc:
            raise ProtocolError(f"Existing pilot checkpoint is invalid: {path}") from exc
    runtime = train_fixed_steps(
        x_fit,
        y_fit,
        schedule=schedule,  # type: ignore[arg-type]
        spec=spec,
        pairing_key=pairing_key,
        training_key_hash=training_key,
        device=device,
        cpu_threads=cpu_threads,
    )
    payload = checkpoint_payload(runtime, metadata=metadata)
    _atomic_torch_save(path, payload)
    return runtime, "miss"


def _evaluate(
    runtime: PilotRuntime,
    *,
    x_fit: object,
    y_fit: object,
    case_fit: object,
    x_eval: object,
    y_eval: object,
    center: str,
    arm: str,
    training_seed: int,
    generation_seeds: tuple[int, ...],
    generated_per_class: int,
    classifier_c: float,
    minimum_real_bacc: float,
    prior_config: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import numpy as np
    import torch

    classifier_spec = ClassifierSpec(
        C=classifier_c,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced",
        random_state=23,
        threshold_policy="fixed_0_5",
    )
    classifier = _fit_real_classifier(x_fit, y_fit, classifier_spec)
    real_predictions = classifier.predict(x_eval)
    real_bacc = balanced_accuracy(y_eval, real_predictions)
    if real_bacc < minimum_real_bacc:
        raise ProtocolError(
            f"Pilot real reference failed denominator floor for center={center}, arm={arm}."
        )
    device = torch.device(runtime.device)
    x_tensor = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_fit, dtype=torch.long, device=device)
    x_eval_tensor = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
    y_eval_tensor = torch.as_tensor(y_eval, dtype=torch.long, device=device)
    runtime.model.eval()
    with torch.no_grad():
        mu, logvar = runtime.model.encode(x_tensor, y_tensor)
        eval_mu, eval_logvar = runtime.model.encode(x_eval_tensor, y_eval_tensor)
        decoded_eval_mu = runtime.model.decode(eval_mu, y_eval_tensor).cpu().numpy()
    mu_np = mu.cpu().numpy()
    logvar_np = logvar.cpu().numpy()
    conditional = fit_shrunk_diagonal_prior(
        mu_np,
        logvar_np,
        y_fit,
        case_fit,
        rho=float(prior_config["rho"]),
        variance_clip=tuple(float(v) for v in prior_config["variance_clip"]),  # type: ignore[arg-type]
        min_rows=int(prior_config["min_rows"]),
        min_cases=int(prior_config["min_cases"]),
        max_condition_number=float(prior_config["max_condition_number"]),
        source_state_hash=runtime.checkpoint_hash,
    )
    metrics = [
        _metric_row(
            center, arm, training_seed, None, "real", real_bacc,
            real_predictions, y_eval, real_bacc, minimum_real_bacc,
        )
    ]
    decoded_predictions = classifier.predict(decoded_eval_mu)
    decoded_bacc = balanced_accuracy(y_eval, decoded_predictions)
    metrics.append(
        _metric_row(
            center, arm, training_seed, None, "decode_mu", decoded_bacc,
            decoded_predictions, y_eval, real_bacc, minimum_real_bacc,
        )
    )
    labels_generated = np.asarray(
        [0] * generated_per_class + [1] * generated_per_class, dtype=np.int64
    )
    for generation_seed in generation_seeds:
        epsilon_seed = _seed(center, training_seed, generation_seed, "posterior_eval")
        generator = torch.Generator(
            device="cuda" if runtime.device.startswith("cuda") else "cpu"
        ).manual_seed(epsilon_seed)
        with torch.no_grad():
            epsilon = torch.randn(
                eval_mu.shape,
                generator=generator,
                dtype=eval_mu.dtype,
                device=eval_mu.device,
            )
            posterior_decoded = runtime.model.decode(
                eval_mu + epsilon * torch.exp(0.5 * eval_logvar), y_eval_tensor
            ).cpu().numpy()
        posterior_predictions = classifier.predict(posterior_decoded)
        posterior_bacc = balanced_accuracy(y_eval, posterior_predictions)
        metrics.append(
            _metric_row(
                center, arm, training_seed, generation_seed, "posterior_sample",
                posterior_bacc, posterior_predictions, y_eval,
                real_bacc, minimum_real_bacc,
            )
        )
        for role, prior in (
            ("prior_standard", None),
            ("prior_conditional", conditional),
        ):
            z = sample_prior(
                prior,
                labels_generated,
                latent_dim=runtime.model.latent_dim,
                seed=_seed(center, training_seed, generation_seed, "prior_eval"),
            )
            with torch.no_grad():
                generated = runtime.model.decode(
                    torch.as_tensor(z, dtype=torch.float32, device=device),
                    torch.as_tensor(labels_generated, dtype=torch.long, device=device),
                ).cpu().numpy()
            predictions = classifier.predict(generated)
            requested_class_bacc = balanced_accuracy(labels_generated, predictions)
            row = _metric_row(
                center, arm, training_seed, generation_seed, role,
                requested_class_bacc, predictions, labels_generated,
                real_bacc, minimum_real_bacc,
            )
            row["requested_prior_family"] = (
                STANDARD_PRIOR_FAMILY
                if prior is None
                else CONDITIONAL_PRIOR_FAMILY
            )
            row["realized_prior_family"] = (
                STANDARD_PRIOR_FAMILY
                if prior is None
                else conditional.realized_family
            )
            row["oracle_eligible"] = False
            metrics.append(row)
    prior_record = {
        "schema_version": "midogpp_b_adaptation_prior_record_v1",
        "training_key_hash": runtime.training_key_hash,
        "requested_family": conditional.requested_family,
        "realized_family": conditional.realized_family,
        "fallback_reason": conditional.fallback_reason,
        "n_rows_by_class": dict(conditional.n_rows_by_class),
        "n_cases_by_class": dict(conditional.n_cases_by_class),
        "condition_number_by_class": dict(conditional.condition_number_by_class),
        "source_state_hash": conditional.source_state_hash,
        "state_hash": conditional.state_hash,
        "claim_scope": "diagnostic_only",
        "may_feed_model_recipe": False,
        "may_feed_deployable_selection": False,
    }
    return metrics, prior_record


def _fit_real_classifier(
    x_fit: object,
    y_fit: object,
    spec: ClassifierSpec,
) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(**spec.to_sklearn_kwargs()),
    )
    classifier.fit(x_fit, y_fit)
    fitted = classifier[-1]
    if any(int(value) >= spec.max_iter for value in fitted.n_iter_):
        raise ProtocolError("Pilot frozen real classifier did not converge.")
    return classifier


def _metric_row(
    center: str,
    arm: str,
    training_seed: int,
    generation_seed: int | None,
    role: str,
    bacc: float,
    predictions: Sequence[int],
    truth: Sequence[int],
    real_bacc: float,
    minimum_real_bacc: float,
) -> dict[str, object]:
    y_true = [int(v) for v in truth]
    y_pred = [int(v) for v in predictions]
    recall = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred)) / max(
        1, sum(t == 1 for t in y_true)
    )
    specificity = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred)) / max(
        1, sum(t == 0 for t in y_true)
    )
    ratio = (
        1.0
        if role == "real"
        else chance_normalized_preservation(
            bacc, real_bacc, minimum_real_bacc=minimum_real_bacc
        )
    )
    return {
        "schema_version": "midogpp_b_adaptation_metric_v1",
        "center": center,
        "arm": arm,
        "training_seed": training_seed,
        "generation_seed": "" if generation_seed is None else generation_seed,
        "representation_role": role,
        "bacc": bacc,
        "positive_recall": recall,
        "specificity": specificity,
        "preservation_ratio": ratio,
        "real_reference_bacc": real_bacc,
        "target_labels_used_for_scoring_only": True,
        "claim_scope": "diagnostic_only",
    }


def _decision(
    metrics: Sequence[Mapping[str, object]],
    jobs: Sequence[Mapping[str, object]],
    *,
    minimum_real_bacc: float,
) -> dict[str, object]:
    def rows(arm: str, role: str) -> list[Mapping[str, object]]:
        return [
            row for row in metrics
            if row["arm"] == arm and row["representation_role"] == role
        ]

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    real_valid = all(float(row["bacc"]) >= minimum_real_bacc for row in metrics if row["representation_role"] == "real")
    block_decode = rows("b_block_pca96_32", "decode_mu")
    a_decode = rows("a_global_pca128", "decode_mu")
    joint_decode = rows("b_joint_pca128", "decode_mu")
    keyed = lambda collection: {
        (str(row["center"]), int(row["training_seed"])): row for row in collection
    }
    block_map, a_map, joint_map = keyed(block_decode), keyed(a_decode), keyed(joint_decode)
    common = sorted(set(block_map).intersection(a_map).intersection(joint_map))
    ratio_delta_a = [
        float(block_map[key]["preservation_ratio"]) - float(a_map[key]["preservation_ratio"])
        for key in common
    ]
    center_delta_a = {
        center: mean([
            float(block_map[key]["preservation_ratio"]) - float(a_map[key]["preservation_ratio"])
            for key in common if key[0] == center
        ])
        for center in sorted({key[0] for key in common})
    }
    seed_means = {
        seed: mean([
            float(block_map[key]["preservation_ratio"])
            for key in common if key[1] == seed
        ])
        for seed in sorted({key[1] for key in common})
    }
    adaptation = (
        real_valid
        and len(jobs) == 36
        and all(int(job["optimizer_steps"]) == 1000 for job in jobs)
        and mean([float(row["preservation_ratio"]) for row in block_decode]) >= 0.80
        and mean(ratio_delta_a) >= -0.02
        and min(center_delta_a.values()) >= -0.05
        and mean([
            float(block_map[key]["bacc"]) - float(a_map[key]["bacc"]) for key in common
        ]) >= -0.01
        and mean([
            float(block_map[key]["positive_recall"]) - float(a_map[key]["positive_recall"])
            for key in common
        ]) >= -0.05
        and mean([
            float(block_map[key]["specificity"]) - float(a_map[key]["specificity"])
            for key in common
        ]) >= -0.05
        and min(seed_means.values()) >= 0.75
        and max(seed_means.values()) - min(seed_means.values()) <= 0.05
    )
    joint_center_delta = {
        center: mean([
            float(block_map[key]["preservation_ratio"]) - float(joint_map[key]["preservation_ratio"])
            for key in common if key[0] == center
        ])
        for center in center_delta_a
    }
    block_justified = (
        adaptation
        and mean(list(joint_center_delta.values())) >= 0.01
        and sum(delta > 0 for delta in joint_center_delta.values()) >= 3
    )
    conditional_rows = rows("b_block_pca96_32", "prior_conditional")
    conditional_jobs = [
        job for job in jobs if job["arm"] == "b_block_pca96_32"
    ]
    conditional_center = {
        center: mean([
            float(row["bacc"]) for row in conditional_rows if row["center"] == center
        ])
        for center in center_delta_a
    }
    prior_viable = (
        all(job["prior_realized_family"] == CONDITIONAL_PRIOR_FAMILY for job in conditional_jobs)
        and mean(list(conditional_center.values())) >= 0.70
        and min(conditional_center.values()) >= 0.60
        and min(
            min(float(row["positive_recall"]), float(row["specificity"]))
            for row in conditional_rows
        ) >= 0.55
    )
    if block_justified:
        category = "PROCEED_TO_FULL_B_SOURCE_INNER_RECIPE_STUDY"
    elif adaptation:
        category = "B_FEASIBLE_BLOCK_AWARE_NOT_JUSTIFIED"
    else:
        category = "B_ADAPTATION_NOT_FEASIBLE"
    return {
        "schema_version": "midogpp_b_adaptation_decision_v1",
        "decision": category,
        "b_adaptation_feasible": adaptation,
        "block_aware_justified": block_justified,
        "conservative_prior_viable": prior_viable,
        "mean_block_decode_ratio": mean([
            float(row["preservation_ratio"]) for row in block_decode
        ]),
        "mean_block_minus_a_ratio": mean(ratio_delta_a),
        "worst_center_block_minus_a_ratio": min(center_delta_a.values()),
        "block_minus_joint_by_center": joint_center_delta,
        "block_seed_mean_ratios": seed_means,
        "claim_scope": "diagnostic_only",
        "may_export_recipe_lock": False,
        "may_feed_expert_bank": False,
        "next_step_if_pass": "separately_reviewed_full_b_source_inner_recipe_study",
    }


def validate_pilot_bundle(root: Path) -> dict[str, object]:
    jobs = _read_csv(root / "tables/job_inventory.csv")
    metrics = _read_csv(root / "tables/pilot_metrics.csv")
    errors = []
    if len(jobs) != 36:
        errors.append(f"expected 36 jobs, found {len(jobs)}")
    if sum(int(row["optimizer_steps"]) for row in jobs) != 36000:
        errors.append("optimizer-step coverage differs from 36,000")
    expected_metrics = 36 * (2 + 3 + 3 + 3)
    if len(metrics) != expected_metrics:
        errors.append(f"expected {expected_metrics} metric rows, found {len(metrics)}")
    for center in ("2", "5", "6", "9"):
        for seed in ("17", "42", "101"):
            subset = [
                row for row in jobs
                if row["center"] == center and row["training_seed"] == seed
            ]
            if len(subset) != 3:
                errors.append(f"paired arm coverage missing for center={center},seed={seed}")
                continue
            if len({row["schedule_hash"] for row in subset}) != 1:
                errors.append(f"schedule pairing failed for center={center},seed={seed}")
            if len({row["initialization_hash"] for row in subset}) != 1:
                errors.append(f"initialization pairing failed for center={center},seed={seed}")
            if len({row["posterior_stream_hash"] for row in subset}) != 1:
                errors.append(f"posterior stream pairing failed for center={center},seed={seed}")
    return {
        "schema_version": "midogpp_b_adaptation_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "n_jobs": len(jobs),
        "n_metrics": len(metrics),
        "claim_scope": "diagnostic_only",
    }


def _protocol_payload(config: PilotConfig, input_hashes: Mapping[str, str]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_b_adaptation_frozen_protocol_v1",
        "experiment": config.name,
        "centers": list(config.centers),
        "arms": list(config.arms),
        "training_seeds": list(config.training_seeds),
        "generation_seeds": list(config.generation_seeds),
        "representation": {
            "b_dim": config.expected_b_dim,
            "global_slice": [0, config.global_dim],
            "local_slice": [config.global_dim, config.expected_b_dim],
            "joint_pca_dim": config.pca_dim,
            "block_pca_dims": [config.block_global_pca_dim, config.block_local_pca_dim],
        },
        "training": _training_spec(config).to_payload(),
        "input_hashes": dict(input_hashes),
        "test_or_validation_split_used": False,
        "train_case_holdout_consumed_for_model_selection": True,
        "claim_scope": "diagnostic_only",
        "may_export_recipe_lock": False,
        "may_feed_expert_bank": False,
    }


def _protocol_hash(config: PilotConfig, input_hashes: Mapping[str, str]) -> str:
    return stable_hash(_protocol_payload(config, input_hashes))


def _training_spec(config: PilotConfig) -> StepTrainingSpec:
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


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _load_aligned_a_bridge(path: Path, sample_ids: Sequence[str]) -> object:
    import numpy as np
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or "embeddings" not in payload or "metadata" not in payload:
        raise ProtocolError("Canonical-A bridge cache is malformed.")
    embeddings = np.asarray(payload["embeddings"])
    metadata = payload["metadata"]
    if embeddings.ndim != 2 or embeddings.shape[1] != 2560 or len(metadata) != len(embeddings):
        raise ProtocolError("Canonical-A bridge cache has the wrong shape.")
    by_id = {}
    for index, row in enumerate(metadata):
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in by_id:
            raise ProtocolError("Canonical-A bridge cache has missing or duplicate sample IDs.")
        by_id[sample_id] = index
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ProtocolError(f"Canonical-A bridge is missing B sample IDs: {missing[:5]}")
    return embeddings[[by_id[sample_id] for sample_id in sample_ids]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_directories(root: Path) -> None:
    for relative in ("reports", "manifests", "tables", "prepared", "checkpoints", "jobs", "priors"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _without_metrics(row: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"metrics", "schedule_audit"}}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Cannot write empty pilot table: {path}")
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


def _atomic_save_npz(path: Path, **arrays: object) -> None:
    import numpy as np

    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _write_content_index(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "content_index.json" and ".tmp" not in path.name:
            files.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
    _write_json(
        root / "manifests/content_index.json",
        {
            "schema_version": "midogpp_b_adaptation_content_index_v1",
            "files": files,
        },
    )


__all__ = ("run_pilot", "validate_pilot_bundle")
