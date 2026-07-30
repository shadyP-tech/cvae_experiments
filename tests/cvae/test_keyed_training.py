from __future__ import annotations

import numpy as np

from midogpp_thesis.cvae.keyed_training import (
    FIXED_BETA,
    KeyedTrainingSpec,
    clone_training_state,
    initialize_training_state,
    run_keyed_steps,
)
from midogpp_thesis.cvae.schedules import build_balanced_schedule


def test_keyed_training_clones_complete_branch_state() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 8)).astype(np.float32)
    y = np.asarray([0] * 20 + [1] * 20, dtype=np.int64)
    cases = [f"case-{index // 2}" for index in range(40)]
    samples = [f"sample-{index}" for index in range(40)]
    schedule = build_balanced_schedule(
        y,
        cases,
        samples,
        steps=4,
        batch_size=8,
        seed=11,
    )
    spec = KeyedTrainingSpec(
        batch_size=8,
        hidden_dim=16,
        latent_dim=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta_final=1e-3,
        gradient_clip_norm=5.0,
    )
    state = initialize_training_state(
        input_dim=8,
        spec=spec,
        pairing_key="test-pair",
        device="cpu",
    )
    run_keyed_steps(
        state,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=2,
        stream_key="test-stream",
        objective=FIXED_BETA,
    )
    clone = clone_training_state(state)
    assert clone.state_hash == state.state_hash
    assert clone.completed_step == 2
    run_keyed_steps(
        clone,
        x,
        y,
        schedule=schedule,
        spec=spec,
        end_step=4,
        stream_key="test-stream",
        objective=FIXED_BETA,
    )
    assert clone.completed_step == 4
    assert state.completed_step == 2
