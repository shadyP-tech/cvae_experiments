"""Source-local training, aggregate-prior fitting, and paired v2 generation."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ....common.hashing import stable_hash
from ...block_frame import PilotFeatureFrame, fit_pilot_frame
from ...geco import GECOController
from ...generation_samplers import (
    AggregatePosteriorSampler, FULL_SAMPLER, fit_aggregate_posterior_sampler,
    sample_latents, standard_normal_sampler,
)
from ...keyed_training import (
    FIXED_BETA, GECO, KeyedTrainingSpec, KeyedTrainingState, attach_geco,
    derived_seed, initialize_training_state, model_state_hash, run_keyed_steps,
    stream_hash, training_state_hash,
)
from ...models import ClassConditionedCVAE
from ...protocol import ProtocolError
from ...schedules import build_balanced_schedule
from ..independent_source import IndependentSourceData
from ..uniform_b_task_geometry.generation import GeneratedBlock
from .config import OptimizedPriorConfig
from .contracts import FRAME, P0, PS, Q, QM, R, OptimizedTrainingKey


@dataclass(frozen=True)
class OptimizedSourceFrame:
    source_center: str
    source_row_hash: str
    frame: PilotFeatureFrame

    def __post_init__(self) -> None:
        if (
            self.frame.arm != FRAME or self.frame.input_dim != 3840
            or self.frame.output_dim != 256
            or self.frame.fit_sample_hash != self.source_row_hash
        ):
            raise ProtocolError("Optimized source-frame identity is invalid.")

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_optimized_source_frame_v2",
            "source_center": self.source_center,
            "source_row_hash": self.source_row_hash,
            "fit_scope": "source_center_rows_only",
            "outer_or_inner_rows_used": False,
            "frame": self.frame.to_payload(),
        }


def fit_optimized_source_frame(source: IndependentSourceData) -> OptimizedSourceFrame:
    if source.embeddings.shape != (len(source.labels), 3840):
        raise ProtocolError("Optimized frame requires aligned 3840-D source rows.")
    return OptimizedSourceFrame(
        source_center=source.center,
        source_row_hash=source.row_hash,
        frame=fit_pilot_frame(FRAME, source.embeddings, fit_sample_hash=source.row_hash),
    )


class TorchOptimizedFrame:
    def __init__(self, source_frame: OptimizedSourceFrame, *, device: str) -> None:
        self.output_dim = source_frame.frame.output_dim
        self.blocks = tuple(
            {
                "start": block.start,
                "stop": block.stop,
                "width": block.output_dim,
                "mean": torch.as_tensor(block.scaler_mean, dtype=torch.float32, device=device),
                "scale": torch.as_tensor(block.scaler_scale, dtype=torch.float32, device=device),
                "pca_mean": torch.as_tensor(block.pca_mean, dtype=torch.float32, device=device),
                "components": torch.as_tensor(block.pca_components, dtype=torch.float32, device=device),
            }
            for block in source_frame.frame.blocks
        )

    def inverse_transform(self, projected: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 2 or projected.shape[1] != self.output_dim:
            raise ProtocolError("Optimized inverse transform expects [n,256].")
        cursor = 0
        outputs = []
        for block in self.blocks:
            width = int(block["width"])
            z = projected[:, cursor : cursor + width]
            scaled = z @ block["components"] + block["pca_mean"]
            outputs.append(scaled * block["scale"] + block["mean"])
            cursor += width
        return torch.cat(outputs, dim=1)


@dataclass
class OptimizedTrainingRuntime:
    state: KeyedTrainingState
    training_key: OptimizedTrainingKey
    schedule_hash: str
    warmup_state_hash: str
    final_stream_hash: str
    geco_target: float


def train_optimized_checkpoint(
    projected: np.ndarray,
    labels: tuple[int, ...],
    case_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    *,
    source_identity_hash: str,
    config: OptimizedPriorConfig,
    training_key: OptimizedTrainingKey,
    device: str,
) -> OptimizedTrainingRuntime:
    x = np.asarray(projected, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.shape != (len(y), config.pca_output_dim) or set(y.tolist()) != {0, 1}:
        raise ProtocolError("Optimized source training arrays are invalid.")
    pairing_key = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_optimized_pairing_v2",
            "training_key_hash": training_key.hash,
            "source_identity_hash": source_identity_hash,
            "parent_checkpoint": "none",
        }
    )
    schedule = build_balanced_schedule(
        y, case_ids, sample_ids,
        steps=config.total_steps,
        batch_size=config.batch_size,
        seed=derived_seed(pairing_key, "balanced_schedule"),
    )
    spec = KeyedTrainingSpec(
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        beta_final=config.beta_final,
        gradient_clip_norm=config.gradient_clip_norm,
        cpu_threads=1,
    )
    state = initialize_training_state(
        input_dim=config.pca_output_dim,
        spec=spec,
        pairing_key=pairing_key,
        device=device,
        num_hidden_layers=config.num_hidden_layers,
    )
    run_keyed_steps(
        state, x, y, schedule=schedule, spec=spec,
        end_step=config.warmup_steps, stream_key=pairing_key, objective=FIXED_BETA,
    )
    warmup_hash = state.state_hash
    target = _mean_distortion(state, x, y) * config.geco_target_slack
    if not np.isfinite(target) or target <= 0:
        raise ProtocolError("Optimized GECO target is invalid.")
    attach_geco(
        state,
        GECOController(
            target=float(target),
            ema_decay=config.geco_ema_decay,
            dual_step_size=config.geco_dual_step_size,
            initial_multiplier=config.geco_initial_multiplier,
            minimum_multiplier=config.geco_minimum_multiplier,
            maximum_multiplier=config.geco_maximum_multiplier,
        ),
    )
    run_keyed_steps(
        state, x, y, schedule=schedule, spec=spec,
        end_step=config.total_steps, stream_key=pairing_key, objective=GECO,
    )
    return OptimizedTrainingRuntime(
        state=state,
        training_key=training_key,
        schedule_hash=schedule.stream_hash,
        warmup_state_hash=warmup_hash,
        final_stream_hash=stream_hash(state),
        geco_target=float(target),
    )


def _mean_distortion(state: KeyedTrainingState, x: np.ndarray, y: np.ndarray) -> float:
    state.model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x, dtype=torch.float32, device=state.device)
        yb = torch.as_tensor(y, dtype=torch.long, device=state.device)
        mu, _ = state.model.encode(xb, yb)
        value = F.mse_loss(state.model.decode(mu, yb), xb, reduction="none").mean()
    state.model.train()
    return float(value.detach().cpu())


@dataclass(frozen=True)
class RuntimePlan:
    scoring_workers: int
    training_devices: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_optimized_runtime_plan_v2",
            "scoring_workers": self.scoring_workers,
            "training_devices": list(self.training_devices),
            "cpu_topology": "12_physical_cores_24_threads",
            "gpu_topology": "two_independent_rtx_a5000_24gb",
            "one_training_process_per_gpu": len(self.training_devices) > 1,
            "cpu_threads_per_training_process": 1,
            "parallel_pca_frame_workers": min(4, self.scoring_workers),
            "mixed_precision": False,
            "tf32": False,
            "cross_gpu_communication": False,
            "scientific_contract_unchanged": True,
        }


def resolve_runtime_plan(config: OptimizedPriorConfig) -> RuntimePlan:
    workers = int(os.environ.get("MIDOGPP_OPTIMIZED_PRIOR_SCORING_WORKERS", config.runtime_scoring_workers))
    raw = os.environ.get("MIDOGPP_OPTIMIZED_PRIOR_TRAINING_DEVICES", "").strip()
    devices = tuple(item.strip() for item in raw.split(",") if item.strip()) if raw else config.runtime_training_devices
    if not 1 <= workers <= 24 or not devices or len(devices) != len(set(devices)):
        raise ProtocolError("Optimized runtime controls are invalid.")
    if any(not (item == "cpu" or item.startswith("cuda:")) for item in devices):
        raise ProtocolError("Training devices must be cpu or explicit cuda:N values.")
    cuda_indices = [int(item.split(":", 1)[1]) for item in devices if item.startswith("cuda:")]
    if cuda_indices:
        if not torch.cuda.is_available() or max(cuda_indices) >= torch.cuda.device_count():
            raise ProtocolError("Configured optimized-prior CUDA device is unavailable.")
    return RuntimePlan(workers, tuple(devices))


@dataclass(frozen=True)
class TrainingTask:
    source_center: str
    labels: tuple[int, ...]
    case_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    source_identity_hash: str
    projected: np.ndarray
    training_key: OptimizedTrainingKey
    config: OptimizedPriorConfig
    device: str
    root: Path

    @property
    def key(self) -> tuple[str, int]:
        return self.source_center, self.training_key.training_seed


@dataclass(frozen=True)
class TrainingResult:
    source_center: str
    training_seed: int
    device: str
    checkpoint_record: Mapping[str, object]
    resumed: bool
    elapsed_seconds: float

    @property
    def key(self) -> tuple[str, int]:
        return self.source_center, self.training_seed


def train_panel(
    *, root: Path, config: OptimizedPriorConfig,
    sources: Mapping[str, IndependentSourceData],
    projected: Mapping[str, np.ndarray],
    keys: Mapping[tuple[str, int], OptimizedTrainingKey],
    runtime: RuntimePlan,
) -> tuple[TrainingResult, ...]:
    order = tuple((center, seed) for center in config.heldout_centers for seed in config.training_seeds)
    assignments = {
        key: runtime.training_devices[index % len(runtime.training_devices)]
        for index, key in enumerate(order)
    }
    tasks = {
        key: TrainingTask(
            source_center=key[0],
            labels=sources[key[0]].labels,
            case_ids=sources[key[0]].case_ids,
            sample_ids=sources[key[0]].sample_ids,
            source_identity_hash=sources[key[0]].identity_hash,
            projected=projected[key[0]],
            training_key=keys[key],
            config=config,
            device=assignments[key],
            root=root,
        )
        for key in order
    }
    if len(runtime.training_devices) == 1:
        results = tuple(_run_training_task(tasks[key]) for key in order)
    else:
        context = mp.get_context("spawn")
        executors = {device: ProcessPoolExecutor(max_workers=1, mp_context=context) for device in runtime.training_devices}
        futures: dict[tuple[str, int], Future[TrainingResult]] = {}
        try:
            for key in order:
                futures[key] = executors[assignments[key]].submit(_run_training_task, tasks[key])
            results = tuple(futures[key].result() for key in order)
        finally:
            for executor in executors.values():
                executor.shutdown(wait=True, cancel_futures=True)
    if tuple(item.key for item in results) != order:
        raise ProtocolError("Optimized training scheduler changed canonical order.")
    return results


def _run_training_task(task: TrainingTask) -> TrainingResult:
    started = perf_counter()
    loaded = load_checkpoint(task.root, task.training_key.hash, task.config, device=task.device)
    resumed = loaded is not None
    if loaded is None:
        runtime = train_optimized_checkpoint(
            task.projected, task.labels, task.case_ids, task.sample_ids,
            source_identity_hash=task.source_identity_hash,
            config=task.config, training_key=task.training_key,
            device=task.device,
        )
        record = save_checkpoint(task.root, runtime)
    else:
        _, record = loaded
    return TrainingResult(
        source_center=task.source_center,
        training_seed=task.training_key.training_seed,
        device=task.device,
        checkpoint_record=record,
        resumed=resumed,
        elapsed_seconds=perf_counter() - started,
    )


def save_checkpoint(root: Path, runtime: OptimizedTrainingRuntime) -> dict[str, object]:
    key = runtime.training_key.hash
    state_path = root / "runtime_cache/uniform_b_optimized_prior/states" / f"{key}.pt"
    record_path = root / "runtime_cache/uniform_b_optimized_prior/by_key" / f"{key}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "midogpp_uniform_b_optimized_state_v2",
        "model": {name: tensor.detach().cpu() for name, tensor in runtime.state.model.state_dict().items()},
        "optimizer": runtime.state.optimizer.state_dict(),
        "controller": runtime.state.controller.state_payload() if runtime.state.controller else None,
        "completed_step": runtime.state.completed_step,
        "initialization_hash": runtime.state.initialization_hash,
        "stream_records": list(runtime.state.stream_records),
        "diagnostics": list(runtime.state.diagnostics),
    }
    temporary = state_path.with_suffix(f".pt.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, state_path)
    record = {
        "schema_version": "midogpp_uniform_b_optimized_checkpoint_record_v2",
        "training_key_hash": key,
        "source_center": runtime.training_key.source_center,
        "training_seed": runtime.training_key.training_seed,
        "checkpoint_hash": model_state_hash(runtime.state.model),
        "training_state_hash": training_state_hash(runtime.state),
        "completed_step": runtime.state.completed_step,
        "relative_path": state_path.relative_to(root).as_posix(),
        "file_sha256": _file_hash(state_path),
        "schedule_hash": runtime.schedule_hash,
        "warmup_state_hash": runtime.warmup_state_hash,
        "final_stream_hash": runtime.final_stream_hash,
        "geco_target": runtime.geco_target,
        "fresh_source_only_training": True,
        "parent_checkpoint_used": False,
    }
    _atomic_json(record_path, record)
    return record


def load_checkpoint(
    root: Path, key: str, config: OptimizedPriorConfig, *, device: str
) -> tuple[KeyedTrainingState, dict[str, object]] | None:
    record_path = root / "runtime_cache/uniform_b_optimized_prior/by_key" / f"{key}.json"
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    state_path = root / str(record.get("relative_path", ""))
    if (
        record.get("fresh_source_only_training") is not True
        or record.get("parent_checkpoint_used") is not False
        or int(record.get("completed_step", -1)) != config.total_steps
        or not state_path.is_file() or _file_hash(state_path) != record.get("file_sha256")
    ):
        raise ProtocolError("Optimized checkpoint record violates its firewall.")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    model = ClassConditionedCVAE(
        input_dim=config.pca_output_dim, hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim, num_hidden_layers=config.num_hidden_layers,
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(payload["optimizer"])
    controller = GECOController.from_state_payload(payload["controller"])
    state = KeyedTrainingState(
        model=model, optimizer=optimizer, controller=controller, device=device,
        completed_step=int(payload["completed_step"]),
        initialization_hash=str(payload["initialization_hash"]),
        stream_records=[dict(row) for row in payload.get("stream_records", ())],
        diagnostics=[dict(row) for row in payload.get("diagnostics", ())],
    )
    if model_state_hash(model) != record["checkpoint_hash"] or training_state_hash(state) != record["training_state_hash"]:
        raise ProtocolError("Restored optimized checkpoint identity mismatch.")
    return state, record


def fit_source_sampler(
    model: ClassConditionedCVAE, projected: np.ndarray, labels: np.ndarray,
    *, source_row_hash: str, config: OptimizedPriorConfig, device: str,
) -> tuple[AggregatePosteriorSampler, AggregatePosteriorSampler, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(projected, dtype=torch.float32, device=device)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        mu, logvar = model.encode(x, y)
    mu_np = mu.detach().cpu().numpy().astype(np.float32)
    logvar_np = logvar.detach().cpu().numpy().astype(np.float32)
    fitted = fit_aggregate_posterior_sampler(
        mu_np, logvar_np, labels, family=FULL_SAMPLER,
        source_row_hash=source_row_hash,
        min_class_count=config.sampler_min_class_count,
        max_condition_number=config.sampler_max_condition_number,
    )
    effective = fitted if fitted.requested_family_realized_for_both_classes else standard_normal_sampler(
        latent_dim=config.latent_dim, source_row_hash=source_row_hash
    )
    return fitted, effective, mu_np, logvar_np


def generate_paired_blocks(
    model: ClassConditionedCVAE,
    source_frame: OptimizedSourceFrame,
    projected: np.ndarray,
    labels: np.ndarray,
    *, source_center: str, training_seed: int, generation_seed: int,
    checkpoint_hash: str, config: OptimizedPriorConfig, device: str,
) -> tuple[dict[str, GeneratedBlock], dict[str, object]]:
    fitted, effective, mu, logvar = fit_source_sampler(
        model, projected, labels, source_row_hash=source_frame.source_row_hash,
        config=config, device=device,
    )
    per_class = config.total_generation_per_class
    out_labels = np.asarray([0] * per_class + [1] * per_class, dtype=np.int64)
    stream_root = stable_hash({
        "schema_version": "midogpp_uniform_b_optimized_generation_stream_v2",
        "source_center": source_center, "training_seed": training_seed,
        "generation_seed": generation_seed, "checkpoint_hash": checkpoint_hash,
        "per_class": per_class, "outer_or_inner_identity_present": False,
    })
    standard = standard_normal_sampler(latent_dim=config.latent_dim, source_row_hash=source_frame.source_row_hash)
    noise_seed = derived_seed(stream_root, "paired_latent_noise")
    latent_by_arm: dict[str, np.ndarray] = {
        P0: np.asarray(sample_latents(standard, out_labels, seed=noise_seed), dtype=np.float32),
        PS: np.asarray(sample_latents(effective, out_labels, seed=noise_seed), dtype=np.float32),
    }
    rng = np.random.default_rng(derived_seed(stream_root, "posterior_rows"))
    indices = np.concatenate([
        rng.choice(np.flatnonzero(labels == cls), size=per_class, replace=True)
        for cls in (0, 1)
    ]).astype(np.int64)
    epsilon = np.random.default_rng(derived_seed(stream_root, "posterior_epsilon")).normal(
        size=(len(indices), config.latent_dim)
    ).astype(np.float32)
    latent_by_arm[Q] = mu[indices] + epsilon * np.exp(0.5 * logvar[indices])
    latent_by_arm[QM] = mu[indices]
    adapter = TorchOptimizedFrame(source_frame, device=device)
    y = torch.as_tensor(out_labels, dtype=torch.long, device=device)
    blocks: dict[str, GeneratedBlock] = {}
    model.eval()
    with torch.no_grad():
        for arm in (P0, PS, Q, QM):
            z = torch.as_tensor(latent_by_arm[arm], dtype=torch.float32, device=device)
            common = adapter.inverse_transform(model.decode(z, y))
            if common.shape != (2 * per_class, 3840) or not torch.isfinite(common).all():
                raise ProtocolError(f"Optimized generation arm {arm} produced invalid rows.")
            arm_stream = stable_hash({"stream_root": stream_root, "arm": arm})
            blocks[arm] = GeneratedBlock(
                source_center=source_center, arm=arm, training_seed=training_seed,
                generation_seed=generation_seed,
                embeddings=common.detach().cpu().numpy().astype(np.float32),
                labels=out_labels.copy(), per_class=per_class,
                checkpoint_hash=checkpoint_hash, frame_hash=source_frame.state_hash,
                stream_hash=arm_stream, kind="prior" if arm in {P0, PS} else "posterior",
            )
        frame_common = adapter.inverse_transform(
            torch.as_tensor(projected[indices], dtype=torch.float32, device=device)
        )
        if frame_common.shape != (2 * per_class, 3840) or not torch.isfinite(frame_common).all():
            raise ProtocolError("PCA-only reconstruction produced invalid rows.")
        blocks[R] = GeneratedBlock(
            source_center=source_center, arm=R, training_seed=training_seed,
            generation_seed=generation_seed,
            embeddings=frame_common.detach().cpu().numpy().astype(np.float32),
            labels=out_labels.copy(), per_class=per_class,
            checkpoint_hash=checkpoint_hash, frame_hash=source_frame.state_hash,
            stream_hash=stable_hash({"stream_root": stream_root, "arm": R}),
            kind="frame_reconstruction",
        )
    audit = {
        "schema_version": "midogpp_uniform_b_optimized_sampler_audit_v2",
        "source_center": source_center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "requested_family": fitted.requested_family,
        "requested_family_realized_for_both_classes": fitted.requested_family_realized_for_both_classes,
        "effective_ps_family": FULL_SAMPLER if fitted.requested_family_realized_for_both_classes else "standard_normal_all_or_none_fallback",
        "realized_family_by_class": fitted.realized_family_by_class(),
        "fallback_reason_by_class": fitted.fallback_reason_by_class(),
        "sampler_state_hash": fitted.state_hash,
        "partial_class_fallback_allowed": False,
        "source_only_fit": True,
        "outer_or_inner_rows_used": False,
    }
    return blocks, audit


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = (
    "OptimizedSourceFrame", "RuntimePlan", "TrainingResult",
    "fit_optimized_source_frame", "fit_source_sampler", "generate_paired_blocks",
    "load_checkpoint", "resolve_runtime_plan", "train_optimized_checkpoint", "train_panel",
)
