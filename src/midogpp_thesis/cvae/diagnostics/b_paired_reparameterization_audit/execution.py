"""One-key execution for the Stage-90 paired Variant-B audit."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.fixed_step_training import (
    FixedStepTrainingRuntime,
    StepTrainingSpec,
    checkpoint_payload,
    model_state_hash,
    train_fixed_steps,
)
from midogpp_thesis.cvae.models import ClassConditionedCVAE
from midogpp_thesis.cvae.protocol import ProtocolError

from .artifacts import torch_save
from .comparison import (
    FIXED_ANTITHETIC,
    LEGACY_REPLAY,
    fit_frozen_classifier,
    prediction_digest,
    score_decoded_mean,
)
from .config import FrozenBRecipe
from .protocol import AuditKeyRecord, key_record_from_mapping
from .snapshot_io import (
    canonical_mapping_hash,
    load_prepared_bundle,
    load_schedule,
)
from .trace import EpsilonTraceSpec, load_epsilon_trace


def run_audit_job(job: Mapping[str, object]) -> dict[str, object]:
    """Execute one immutable key and return only serializable evidence."""

    import torch

    started = perf_counter()
    snapshot_root = Path(str(job["snapshot_root"]))
    output_root = Path(str(job["output_root"]))
    snapshot_hash = str(job["snapshot_hash"])
    snapshot_protocol_hash = str(job["snapshot_protocol_hash"])
    record = key_record_from_mapping(_mapping(job["record"]))
    recipe = FrozenBRecipe(**dict(_mapping(job["recipe"])))
    prepared = load_prepared_bundle(
        snapshot_root / record.prepared_relpath,
        expected_file_sha256=record.prepared_sha256,
        expected_content_hash=record.prepared_content_hash,
    )
    schedule, _ = load_schedule(
        snapshot_root / record.schedule_relpath,
        expected_file_sha256=record.schedule_sha256,
        expected_content_hash=record.schedule_content_hash,
        labels=prepared["y_fit"],
        case_ids=prepared["case_fit"],
        sample_ids=prepared["sample_fit"],
    )
    trace = load_epsilon_trace(
        snapshot_root,
        EpsilonTraceSpec(
            relative_path=record.epsilon_trace_relpath,
            file_sha256=record.epsilon_trace_sha256,
            content_sha256=record.epsilon_trace_content_hash,
            steps=recipe.optimizer_steps,
            batch_size=recipe.batch_size,
            latent_dim=recipe.latent_dim,
        ),
    )
    if record.execution_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ProtocolError(
                f"Audit key requires unavailable CUDA device {record.execution_device}."
            )
        torch.cuda.set_device(record.execution_device)

    pairing_key = _pairing_key(
        record=record,
        snapshot_protocol_hash=snapshot_protocol_hash,
        schedule_hash=schedule.stream_hash,
    )
    estimator = (
        "antithetic_epsilon"
        if record.candidate == FIXED_ANTITHETIC
        else "one_epsilon"
    )
    spec = _training_spec(recipe)
    checkpoint_path = output_root / "checkpoints" / f"{record.key_hash}.pt"
    runtime, checkpoint_cache_status = _fit_or_load(
        checkpoint_path,
        record=record,
        snapshot_hash=snapshot_hash,
        prepared=prepared,
        schedule=schedule,
        epsilon_trace=trace.values,
        pairing_key=pairing_key,
        estimator=estimator,
        spec=spec,
    )
    classifier = fit_frozen_classifier(prepared["x_fit"], prepared["y_fit"])
    real_predictions = [
        int(value) for value in classifier.predict(prepared["x_eval"])
    ]
    real_reference_bacc = _balanced_accuracy(
        prepared["y_eval"],
        real_predictions,
    )
    score = score_decoded_mean(
        runtime=runtime,
        classifier=classifier,
        x_eval=prepared["x_eval"],
        y_eval=prepared["y_eval"],
        sample_ids=prepared["sample_eval"],
        case_ids=prepared["case_eval"],
        center=record.center,
        training_seed=record.initialization_seed,
        candidate=record.candidate,
        real_reference_bacc=real_reference_bacc,
        minimum_real_bacc=0.60,
    )
    metric = dict(
        score.metric,
        key_hash=record.key_hash,
        pair_id="" if record.pair_id is None else record.pair_id,
        execution_device=record.execution_device,
    )
    predictions = [
        dict(
            row,
            real_reference_y_pred=int(real_prediction),
            key_hash=record.key_hash,
            pair_id="" if record.pair_id is None else record.pair_id,
            execution_device=record.execution_device,
        )
        for row, real_prediction in zip(
            score.predictions,
            real_predictions,
            strict=True,
        )
    ]
    observed_prediction_hash = prediction_digest(predictions)
    metric_subset = {
        key: metric[key]
        for key in (
            "bacc",
            "positive_recall",
            "specificity",
            "fn",
            "fp",
            "tn",
            "tp",
        )
    }
    observed_metric_hash = canonical_mapping_hash(metric_subset)
    job_row: dict[str, object] = {
        "schema_version": "midogpp_b_paired_reparameterization_job_v1",
        "center": record.center,
        "initialization_seed": record.initialization_seed,
        "candidate": record.candidate,
        "execution_device": record.execution_device,
        "key_hash": record.key_hash,
        "pair_id": "" if record.pair_id is None else record.pair_id,
        "prepared_sha256": record.prepared_sha256,
        "prepared_content_hash": record.prepared_content_hash,
        "schedule_sha256": record.schedule_sha256,
        "schedule_content_hash": record.schedule_content_hash,
        "epsilon_trace_sha256": record.epsilon_trace_sha256,
        "epsilon_trace_content_hash": record.epsilon_trace_content_hash,
        "initialization_hash": runtime.initialization_hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "schedule_hash": runtime.schedule_hash,
        "posterior_stream_hash": runtime.posterior_stream_hash,
        "optimizer_steps": runtime.optimizer_steps,
        "decoder_forwards": runtime.decoder_forwards,
        "posterior_estimator": estimator,
        "epsilon_consumptions": 1,
        "checkpoint_cache_status": checkpoint_cache_status,
        "cache_status": "COMPLETED",
        "status": "PASS",
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
        "elapsed_seconds": perf_counter() - started,
        "claim_scope": "diagnostic_only",
        "may_export_recipe_lock": False,
        "may_feed_expert_bank": False,
        "may_feed_generation": False,
        "may_feed_routing": False,
        "may_feed_downstream": False,
    }
    trace_audit = {
        "schema_version": "midogpp_b_paired_reparameterization_trace_audit_v1",
        "center": record.center,
        "initialization_seed": record.initialization_seed,
        "candidate": record.candidate,
        "key_hash": record.key_hash,
        "prepared_file_match": True,
        "prepared_content_match": True,
        "schedule_file_match": True,
        "schedule_content_match": True,
        "epsilon_file_match": True,
        "epsilon_content_match": runtime.epsilon_trace_hash
        == record.epsilon_trace_content_hash,
        "trace_consumption_count": 1,
        "status": "PASS",
        "claim_scope": "diagnostic_only",
    }
    consumption = {
        "schema_version": "midogpp_b_paired_reparameterization_consumption_v1",
        "center": record.center,
        "initialization_seed": record.initialization_seed,
        "candidate": record.candidate,
        "key_hash": record.key_hash,
        "epsilon_consumption_count": 1,
        "optimizer_steps": runtime.optimizer_steps,
        "decoder_forwards": runtime.decoder_forwards,
        "status": "PASS",
        "claim_scope": "diagnostic_only",
    }
    legacy_validation = None
    if record.candidate == LEGACY_REPLAY:
        legacy_validation = _legacy_validation(
            record=record,
            runtime=runtime,
            observed_prediction_hash=observed_prediction_hash,
            observed_metric_hash=observed_metric_hash,
            metric_subset=metric_subset,
        )
        if legacy_validation["status"] != "PASS":
            job_row["status"] = "FAIL"
    if not trace_audit["epsilon_content_match"]:
        trace_audit["status"] = "FAIL"
        consumption["status"] = "FAIL"
        job_row["status"] = "FAIL"
    return {
        "job": job_row,
        "metric": metric,
        "predictions": predictions,
        "trace_audit": trace_audit,
        "consumption": consumption,
        "legacy_validation": legacy_validation,
        "training_diagnostics": [dict(row) for row in runtime.diagnostics],
    }


def _fit_or_load(
    path: Path,
    *,
    record: AuditKeyRecord,
    snapshot_hash: str,
    prepared: Mapping[str, object],
    schedule: object,
    epsilon_trace: object,
    pairing_key: str,
    estimator: str,
    spec: StepTrainingSpec,
) -> tuple[FixedStepTrainingRuntime, str]:
    import torch

    metadata = {
        "audit_key_hash": record.key_hash,
        "snapshot_hash": snapshot_hash,
        "candidate": record.candidate,
        "center": record.center,
        "initialization_seed": record.initialization_seed,
        "execution_device": record.execution_device,
        "prepared_content_hash": record.prepared_content_hash,
        "schedule_content_hash": record.schedule_content_hash,
        "epsilon_trace_content_hash": record.epsilon_trace_content_hash,
        "posterior_estimator": estimator,
    }
    if path.is_file():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("schema_version")
                != "midogpp_b_paired_reparameterization_checkpoint_v1"
                or payload.get("metadata") != metadata
                or payload.get("training_key_hash") != record.key_hash
            ):
                raise ProtocolError("Cached audit checkpoint provenance mismatch.")
            model = ClassConditionedCVAE(
                input_dim=128,
                hidden_dim=spec.hidden_dim,
                latent_dim=spec.latent_dim,
                num_hidden_layers=2,
            )
            model.load_state_dict(payload["state_dict"], strict=True)
            if model_state_hash(model) != payload.get("checkpoint_hash"):
                raise ProtocolError("Cached audit checkpoint content hash mismatch.")
            model = model.to(record.execution_device)
            runtime = FixedStepTrainingRuntime(
                model=model,
                device=record.execution_device,
                training_key_hash=record.key_hash,
                checkpoint_hash=str(payload["checkpoint_hash"]),
                initialization_hash=str(payload["initialization_hash"]),
                schedule_hash=str(payload["schedule_hash"]),
                posterior_stream_hash=str(payload["posterior_stream_hash"]),
                diagnostics=tuple(payload["diagnostics"]),
                peak_cuda_bytes=int(payload.get("peak_cuda_bytes", 0)),
                optimizer_steps=int(payload["optimizer_steps"]),
                decoder_forwards=int(payload["decoder_forwards"]),
                epsilon_trace_hash=str(payload["epsilon_trace_hash"]),
            )
            _validate_runtime(record, runtime, spec=spec, estimator=estimator)
            return runtime, "HIT"
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise ProtocolError(f"Existing audit checkpoint is invalid: {path}") from exc
    runtime = train_fixed_steps(
        prepared["x_fit"],
        prepared["y_fit"],
        schedule=schedule,
        spec=spec,
        pairing_key=pairing_key,
        posterior_stream_key=pairing_key,
        training_key_hash=record.key_hash,
        device=record.execution_device,
        cpu_threads=1,
        posterior_estimator=estimator,
        epsilon_trace=epsilon_trace,
        epsilon_trace_hash=record.epsilon_trace_content_hash,
    )
    _validate_runtime(record, runtime, spec=spec, estimator=estimator)
    payload = checkpoint_payload(runtime, metadata=metadata)
    payload["schema_version"] = "midogpp_b_paired_reparameterization_checkpoint_v1"
    torch_save(path, payload)
    return runtime, "MISS"


def _validate_runtime(
    record: AuditKeyRecord,
    runtime: FixedStepTrainingRuntime,
    *,
    spec: StepTrainingSpec,
    estimator: str,
) -> None:
    expected_forwards = spec.optimizer_steps * (
        2 if estimator == "antithetic_epsilon" else 1
    )
    if (
        runtime.training_key_hash != record.key_hash
        or runtime.schedule_hash == ""
        or runtime.epsilon_trace_hash != record.epsilon_trace_content_hash
        or runtime.optimizer_steps != spec.optimizer_steps
        or runtime.decoder_forwards != expected_forwards
    ):
        raise ProtocolError("Audit runtime violated its exact update/forward contract.")


def _legacy_validation(
    *,
    record: AuditKeyRecord,
    runtime: FixedStepTrainingRuntime,
    observed_prediction_hash: str,
    observed_metric_hash: str,
    metric_subset: Mapping[str, object],
) -> dict[str, object]:
    expected_metric = dict(record.legacy_expected_decode_metric or {})
    metric_values_match = (
        set(expected_metric) == set(metric_subset)
        and all(
            int(metric_subset[key]) == int(expected_metric[key])
            for key in ("fn", "fp", "tn", "tp")
        )
        and all(
            abs(float(metric_subset[key]) - float(expected_metric[key])) <= 1e-12
            for key in ("bacc", "positive_recall", "specificity")
        )
    )
    expected = {
        "initialization": str(record.legacy_expected_initialization_hash),
        "checkpoint": str(record.legacy_expected_checkpoint_hash),
        "prediction": str(record.legacy_expected_prediction_hash),
        "metric": str(record.legacy_expected_metric_hash),
        "schedule": str(record.legacy_historical_schedule_hash),
        "posterior": str(record.legacy_historical_posterior_stream_hash),
    }
    observed = {
        "initialization": runtime.initialization_hash,
        "checkpoint": runtime.checkpoint_hash,
        "prediction": observed_prediction_hash,
        "metric": observed_metric_hash,
        "schedule": runtime.schedule_hash,
        "posterior": runtime.posterior_stream_hash,
    }
    matches = {
        name: expected[name] == observed[name]
        for name in expected
    }
    matches["metric"] = matches["metric"] and metric_values_match
    status = "PASS" if all(matches.values()) else "FAIL"
    row: dict[str, object] = {
        "schema_version": "midogpp_b_paired_reparameterization_legacy_validation_v1",
        "center": record.center,
        "initialization_seed": record.initialization_seed,
        "candidate": record.candidate,
        "key_hash": record.key_hash,
        "historical_training_key_hash": record.legacy_historical_training_key_hash,
        "historical_frame_hash": record.legacy_historical_frame_hash,
        "historical_fit_row_hash": record.legacy_historical_fit_row_hash,
        "historical_eval_row_hash": record.legacy_historical_eval_row_hash,
        "metric_values_match": metric_values_match,
        "status": status,
        "comparison_eligible": False,
        "replay_validation_only": True,
        "claim_scope": "diagnostic_only",
    }
    for name in (
        "initialization",
        "checkpoint",
        "prediction",
        "metric",
        "schedule",
        "posterior",
    ):
        row[f"expected_{name}_hash"] = expected[name]
        row[f"observed_{name}_hash"] = observed[name]
        row[f"{name}_match"] = matches[name]
    return row


def _pairing_key(
    *,
    record: AuditKeyRecord,
    snapshot_protocol_hash: str,
    schedule_hash: str,
) -> str:
    if record.candidate == LEGACY_REPLAY:
        if schedule_hash != record.legacy_historical_schedule_hash:
            raise ProtocolError("Loaded legacy schedule identity drifted.")
        return stable_hash(
            {
                "protocol_hash": snapshot_protocol_hash,
                "center": record.center,
                "training_seed": record.initialization_seed,
                "schedule_hash": schedule_hash,
                "paired_across_arms": True,
            }
        )
    if not record.pair_id:
        raise ProtocolError("Controlled audit key lacks a pair identity.")
    return stable_hash(
        {
            "schema_version": "midogpp_b_controlled_initialization_key_v1",
            "pair_id": record.pair_id,
            "center": record.center,
            "initialization_seed": record.initialization_seed,
        }
    )


def _training_spec(recipe: FrozenBRecipe) -> StepTrainingSpec:
    return StepTrainingSpec(
        optimizer_steps=recipe.optimizer_steps,
        batch_size=recipe.batch_size,
        hidden_dim=recipe.hidden_dim,
        latent_dim=recipe.latent_dim,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        beta_final=recipe.beta_final,
        kl_warmup_steps=recipe.kl_warmup_steps,
        gradient_clip_norm=recipe.gradient_clip_norm,
    )


def _balanced_accuracy(truth: object, predictions: object) -> float:
    from midogpp_thesis.cvae.metrics import balanced_accuracy

    return float(balanced_accuracy(truth, predictions))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProtocolError("Audit job payload contains a malformed mapping.")
    return value


__all__ = ("run_audit_job",)
