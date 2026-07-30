"""Paired BF/BG/BM/BT training over independently prepared source data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ....common.hashing import stable_hash
from ...geco import GECOController
from ...keyed_training import (
    FIXED_BETA,
    GECO,
    KeyedTrainingSpec,
    KeyedTrainingState,
    attach_geco,
    clone_training_state,
    derived_seed,
    initialize_training_state,
    run_keyed_steps,
    stream_hash,
    torch_generator,
)
from ...protocol import ProtocolError
from ...schedules import build_balanced_schedule
from .config import UniformBTaskGeometryConfig
from .contracts import ARMS, BF, BG, BM, BT
from .task_geometry import TaskGeometryState
from .task_loss import (
    FrozenTaskObjective,
    TaskLossWeights,
    TaskTermScales,
    calibrate_task_scales,
)


@dataclass
class ArmRuntime:
    arm: str
    state: KeyedTrainingState
    training_key_hash: str
    task_lock_hash: str
    branch_start_hash: str
    final_stream_hash: str


@dataclass(frozen=True)
class SourcePanelRuntime:
    arms: Mapping[str, ArmRuntime]
    schedule_hash: str
    shared_initialization_hash: str
    warmup_state_hash: str
    task_branch_state_hash: str
    task_scales: TaskTermScales
    task_lock_hash: str


def train_source_panel(
    projected: Sequence[Sequence[float]],
    labels: Sequence[int],
    case_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    geometry: TaskGeometryState,
    config: UniformBTaskGeometryConfig,
    source_center: str,
    training_seed: int,
    source_identity_hash: str,
    frame_hash: str,
) -> SourcePanelRuntime:
    """Train the exact paired panel; source centers remain independent."""

    x = np.asarray(projected, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 128 or set(y.tolist()) != {0, 1}:
        raise ProtocolError("Uniform-B source-panel arrays are invalid.")
    pairing_key = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_panel_pairing_v1",
            "source_center": str(source_center),
            "training_seed": int(training_seed),
            "source_identity_hash": source_identity_hash,
            "frame_hash": frame_hash,
            "config_hash": config.contract_hash,
        }
    )
    schedule = build_balanced_schedule(
        y,
        case_ids,
        sample_ids,
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
    )
    warm = initialize_training_state(
        input_dim=128,
        spec=spec,
        pairing_key=pairing_key,
        device=config.device,
    )
    shared_initialization_hash = warm.initialization_hash
    run_keyed_steps(
        warm,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=config.warmup_steps,
        stream_key=pairing_key,
        objective=FIXED_BETA,
    )
    warmup_state_hash = warm.state_hash
    target = (
        _mean_source_distortion(warm.model, x, y, warm.device)
        * config.geco_target_slack
    )
    if not np.isfinite(target) or target <= 0.0:
        raise ProtocolError("Source-only GECO target is invalid.")
    controller = GECOController(
        target=float(target),
        ema_decay=config.geco_ema_decay,
        dual_step_size=config.geco_dual_step_size,
        initial_multiplier=config.geco_initial_multiplier,
        minimum_multiplier=config.geco_minimum_multiplier,
        maximum_multiplier=config.geco_maximum_multiplier,
    )

    bf = clone_training_state(warm)
    run_keyed_steps(
        bf,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=config.total_steps,
        stream_key=pairing_key,
        objective=FIXED_BETA,
    )

    geco_branch = clone_training_state(warm)
    attach_geco(geco_branch, controller)
    run_keyed_steps(
        geco_branch,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=config.task_start_step,
        stream_key=pairing_key,
        objective=GECO,
    )
    task_branch_state_hash = geco_branch.state_hash
    calibration_labels = torch.as_tensor(
        [0] * (config.batch_size // 2) + [1] * (config.batch_size // 2),
        dtype=torch.long,
        device=geco_branch.device,
    )
    calibration_generator = torch_generator(
        geco_branch.device,
        derived_seed(pairing_key, "task_scale_calibration"),
    )
    calibration_z = torch.randn(
        (config.batch_size, config.latent_dim),
        generator=calibration_generator,
        dtype=torch.float32,
        device=geco_branch.device,
    )
    with torch.no_grad():
        calibration_generated = geco_branch.model.decode(
            calibration_z,
            calibration_labels,
        )
        scales = calibrate_task_scales(
            calibration_generated,
            calibration_labels,
            geometry,
            cdf_temperature=config.cdf_temperature,
            device=geco_branch.device,
        )
    task_lock_hash = stable_hash(
        {
            "schema_version": "midogpp_uniform_b_task_lock_v1",
            "geometry_hash": geometry.state_hash,
            "scale_hash": scales.state_hash,
            "task_start_step": config.task_start_step,
            "total_steps": config.total_steps,
            "weights": {
                "mmd": config.mmd_weight,
                "margin": config.margin_weight,
                "gradient": config.gradient_weight,
                "global": config.task_weight,
            },
            "calibration_seed": derived_seed(
                pairing_key,
                "task_scale_calibration",
            ),
            "source_only": True,
        }
    )
    zero_task = FrozenTaskObjective(
        geometry,
        scales=scales,
        weights=TaskLossWeights(
            mmd=config.mmd_weight,
            margin=config.margin_weight,
            gradient=config.gradient_weight,
        ),
        cdf_temperature=config.cdf_temperature,
        device=geco_branch.device,
    )
    mmd_task = FrozenTaskObjective(
        geometry,
        scales=scales,
        weights=TaskLossWeights(
            mmd=config.mmd_weight,
            margin=0.0,
            gradient=0.0,
        ),
        cdf_temperature=config.cdf_temperature,
        device=geco_branch.device,
    )
    full_task = zero_task
    branch_states = {
        BG: clone_training_state(geco_branch),
        BM: clone_training_state(geco_branch),
        BT: clone_training_state(geco_branch),
    }
    if len({state.state_hash for state in branch_states.values()}) != 1:
        raise ProtocolError("BG/BM/BT branch states are not identical.")
    for arm, auxiliary, weight in (
        (BG, zero_task, 0.0),
        (BM, mmd_task, config.task_weight),
        (BT, full_task, config.task_weight),
    ):
        run_keyed_steps(
            branch_states[arm],
            x,
            y,
            schedule=schedule,
            spec=spec,
            end_step=config.total_steps,
            stream_key=pairing_key,
            objective=GECO,
            auxiliary=auxiliary,
            auxiliary_weight=weight,
        )
    # Callback identities differ intentionally; stochastic identities must not.
    stripped = {
        arm: stable_hash(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"auxiliary_identity_hash", "objective"}
                }
                for row in branch_states[arm].stream_records
                if int(row["step"]) > config.task_start_step
            ]
        )
        for arm in (BG, BM, BT)
    }
    if len(set(stripped.values())) != 1:
        raise ProtocolError("BG/BM/BT final-phase stochastic streams diverged.")
    states = {BF: bf, **branch_states}
    runtimes = {
        arm: ArmRuntime(
            arm=arm,
            state=states[arm],
            training_key_hash=stable_hash(
                {
                    "pairing_key": pairing_key,
                    "arm": arm,
                    "task_lock_hash": (
                        task_lock_hash if arm in {BM, BT} else "none"
                    ),
                }
            ),
            task_lock_hash=task_lock_hash if arm in {BM, BT} else "none",
            branch_start_hash=(
                task_branch_state_hash if arm in {BG, BM, BT} else warmup_state_hash
            ),
            final_stream_hash=stream_hash(states[arm]),
        )
        for arm in ARMS
    }
    return SourcePanelRuntime(
        arms=runtimes,
        schedule_hash=schedule.stream_hash,
        shared_initialization_hash=shared_initialization_hash,
        warmup_state_hash=warmup_state_hash,
        task_branch_state_hash=task_branch_state_hash,
        task_scales=scales,
        task_lock_hash=task_lock_hash,
    )


def _mean_source_distortion(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
) -> float:
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x, dtype=torch.float32, device=device)
        yb = torch.as_tensor(y, dtype=torch.long, device=device)
        mu, _ = model.encode(xb, yb)  # type: ignore[attr-defined]
        decoded = model.decode(mu, yb)  # type: ignore[attr-defined]
        value = F.mse_loss(decoded, xb, reduction="none").mean(dim=1).mean()
    model.train()
    return float(value.detach().cpu())


__all__ = (
    "ArmRuntime",
    "SourcePanelRuntime",
    "train_source_panel",
)
