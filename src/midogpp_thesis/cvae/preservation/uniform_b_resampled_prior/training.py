"""Fresh source-local BG training without task-geometry branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    derived_seed,
    initialize_training_state,
    run_keyed_steps,
    stream_hash,
)
from ...protocol import ProtocolError
from ...schedules import build_balanced_schedule
from .config import UniformBResampledPriorConfig
from .contracts import SourceTrainingKey


@dataclass
class BGTrainingRuntime:
    state: KeyedTrainingState
    training_key: SourceTrainingKey
    schedule_hash: str
    initialization_hash: str
    warmup_state_hash: str
    final_stream_hash: str
    geco_target: float


def train_fresh_bg_checkpoint(
    projected: Sequence[Sequence[float]],
    labels: Sequence[int],
    case_ids: Sequence[str],
    sample_ids: Sequence[str],
    *,
    config: UniformBResampledPriorConfig,
    training_key: SourceTrainingKey,
    source_identity_hash: str,
    device: str,
) -> BGTrainingRuntime:
    x = np.asarray(projected, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 128 or set(y.tolist()) != {0, 1}:
        raise ProtocolError("Fresh BG source arrays are invalid.")
    pairing_key = stable_hash(
        {
            "schema_version": "midogpp_resampled_prior_bg_pairing_v1",
            "training_key_hash": training_key.hash,
            "source_identity_hash": source_identity_hash,
            "fresh_training": True,
            "parent_checkpoint_hash": "none",
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
    state = initialize_training_state(
        input_dim=128,
        spec=spec,
        pairing_key=pairing_key,
        device=device,
    )
    initialization_hash = state.initialization_hash
    run_keyed_steps(
        state,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=config.warmup_steps,
        stream_key=pairing_key,
        objective=FIXED_BETA,
    )
    warmup_state_hash = state.state_hash
    target = _mean_source_distortion(state, x, y) * config.geco_target_slack
    if not np.isfinite(target) or target <= 0.0:
        raise ProtocolError("Fresh BG GECO target is invalid.")
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
        state,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=config.total_steps,
        stream_key=pairing_key,
        objective=GECO,
    )
    return BGTrainingRuntime(
        state=state,
        training_key=training_key,
        schedule_hash=schedule.stream_hash,
        initialization_hash=initialization_hash,
        warmup_state_hash=warmup_state_hash,
        final_stream_hash=stream_hash(state),
        geco_target=float(target),
    )


def _mean_source_distortion(
    state: KeyedTrainingState,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    state.model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x, dtype=torch.float32, device=state.device)
        yb = torch.as_tensor(y, dtype=torch.long, device=state.device)
        mu, _ = state.model.encode(xb, yb)
        decoded = state.model.decode(mu, yb)
        value = F.mse_loss(decoded, xb, reduction="none").mean(dim=1).mean()
    state.model.train()
    return float(value.detach().cpu())


__all__ = ("BGTrainingRuntime", "train_fresh_bg_checkpoint")
