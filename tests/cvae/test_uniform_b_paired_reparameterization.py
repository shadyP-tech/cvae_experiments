from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from midogpp_thesis.cvae.fixed_step_training import (
    FixedStepTrainingRuntime,
    PilotRuntime,
    StepTrainingSpec,
    checkpoint_payload,
    epsilon_trace_content_hash,
    train_fixed_steps,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.schedules import (
    build_fold_fixed_schedule,
)


def test_fold_fixed_schedule_omits_training_seed_and_candidate() -> None:
    labels = np.asarray([0, 1] * 16)
    cases = np.asarray([f"case-{index // 2}" for index in range(32)])
    samples = np.asarray([f"sample-{index}" for index in range(32)])
    first = build_fold_fixed_schedule(
        labels, cases, samples, steps=3, batch_size=8, center="2", fit_row_hash="fit", recipe_version="v1"
    )
    second = build_fold_fixed_schedule(
        labels, cases, samples, steps=3, batch_size=8, center="2", fit_row_hash="fit", recipe_version="v1"
    )
    assert first.stream_hash == second.stream_hash
    np.testing.assert_array_equal(first.batches, second.batches)


def test_antithetic_training_uses_one_update_and_two_decoder_forwards() -> None:
    rng = np.random.default_rng(3)
    labels = np.asarray([0, 1] * 16)
    cases = np.asarray([f"case-{index // 2}" for index in range(32)])
    samples = np.asarray([f"sample-{index}" for index in range(32)])
    schedule = build_fold_fixed_schedule(
        labels, cases, samples, steps=2, batch_size=8, center="2", fit_row_hash="fit", recipe_version="v1"
    )
    runtime = train_fixed_steps(
        rng.normal(size=(32, 128)).astype("float32"), labels, schedule=schedule,
        spec=StepTrainingSpec(optimizer_steps=2, batch_size=8, hidden_dim=8, latent_dim=2),
        pairing_key="initialization", training_key_hash="test", device="cpu",
        posterior_estimator="antithetic_epsilon", posterior_stream_key="fold-fixed-epsilon",
    )
    assert runtime.optimizer_steps == 2
    assert runtime.decoder_forwards == 4
    assert len(runtime.epsilon_trace_hash) == 64
    assert len(runtime.diagnostics) == 2
    assert runtime.diagnostics[-1]["optimizer_steps"] == 2
    assert runtime.diagnostics[-1]["decoder_forwards"] == 4
    assert runtime.diagnostics[-1]["epsilon_trace_hash"] == runtime.epsilon_trace_hash


def test_fixed_step_training_infers_projected_input_width() -> None:
    rng = np.random.default_rng(4)
    labels = np.asarray([0, 1] * 16)
    cases = np.asarray([f"case-{index // 2}" for index in range(32)])
    samples = np.asarray([f"sample-{index}" for index in range(32)])
    schedule = build_fold_fixed_schedule(
        labels,
        cases,
        samples,
        steps=1,
        batch_size=8,
        center="2",
        fit_row_hash="fit",
        recipe_version="v1",
    )
    runtime = train_fixed_steps(
        rng.normal(size=(32, 7)).astype("float32"),
        labels,
        schedule=schedule,
        spec=StepTrainingSpec(
            optimizer_steps=1,
            batch_size=8,
            hidden_dim=8,
            latent_dim=2,
        ),
        pairing_key="initialization",
        training_key_hash="test",
        device="cpu",
    )

    assert runtime.model.input_dim == 7


def test_explicit_epsilon_trace_is_verified_and_audited() -> None:
    rng = np.random.default_rng(5)
    labels = np.asarray([0, 1] * 16)
    cases = np.asarray([f"case-{index // 2}" for index in range(32)])
    samples = np.asarray([f"sample-{index}" for index in range(32)])
    schedule = build_fold_fixed_schedule(
        labels,
        cases,
        samples,
        steps=2,
        batch_size=8,
        center="2",
        fit_row_hash="fit",
        recipe_version="v1",
    )
    trace = rng.normal(size=(2, 8, 2))
    trace_hash = epsilon_trace_content_hash(trace)
    runtime = train_fixed_steps(
        rng.normal(size=(32, 128)).astype("float32"),
        labels,
        schedule=schedule,
        spec=StepTrainingSpec(
            optimizer_steps=2,
            batch_size=8,
            hidden_dim=8,
            latent_dim=2,
        ),
        pairing_key="initialization",
        training_key_hash="test",
        device="cpu",
        posterior_estimator="one_epsilon",
        posterior_stream_key="legacy-stream-identity",
        epsilon_trace=trace,
        epsilon_trace_hash=trace_hash,
    )
    assert runtime.optimizer_steps == 2
    assert runtime.decoder_forwards == 2
    assert runtime.epsilon_trace_hash == trace_hash
    assert runtime.posterior_stream_hash != trace_hash
    payload = checkpoint_payload(runtime, metadata={"test": True})
    assert payload["optimizer_steps"] == 2
    assert payload["decoder_forwards"] == 2
    assert payload["epsilon_trace_hash"] == trace_hash


def test_explicit_trace_exactly_replays_the_legacy_seeded_objective() -> None:
    rng = np.random.default_rng(11)
    labels = np.asarray([0, 1] * 16)
    cases = np.asarray([f"case-{index // 2}" for index in range(32)])
    samples = np.asarray([f"sample-{index}" for index in range(32)])
    schedule = build_fold_fixed_schedule(
        labels,
        cases,
        samples,
        steps=2,
        batch_size=8,
        center="2",
        fit_row_hash="fit",
        recipe_version="v1",
    )
    embeddings = rng.normal(size=(32, 128)).astype("float32")
    spec = StepTrainingSpec(
        optimizer_steps=2,
        batch_size=8,
        hidden_dim=8,
        latent_dim=2,
    )
    stream_key = "fold-fixed-epsilon"
    trace = np.stack(
        [
            torch.randn(
                (8, 2),
                generator=torch.Generator().manual_seed(
                    _derived_seed(stream_key, step, "posterior")
                ),
            ).numpy()
            for step in (1, 2)
        ]
    )
    common = {
        "schedule": schedule,
        "spec": spec,
        "pairing_key": "initialization",
        "training_key_hash": "test",
        "device": "cpu",
        "posterior_stream_key": stream_key,
    }
    legacy = train_fixed_steps(embeddings, labels, **common)
    replay = train_fixed_steps(
        embeddings,
        labels,
        epsilon_trace=trace,
        epsilon_trace_hash=epsilon_trace_content_hash(trace),
        **common,
    )
    assert replay.checkpoint_hash == legacy.checkpoint_hash
    assert replay.posterior_stream_hash == legacy.posterior_stream_hash
    assert replay.epsilon_trace_hash == legacy.epsilon_trace_hash


def test_explicit_epsilon_trace_rejects_wrong_shape_or_hash() -> None:
    rng = np.random.default_rng(7)
    labels = np.asarray([0, 1] * 16)
    cases = np.asarray([f"case-{index // 2}" for index in range(32)])
    samples = np.asarray([f"sample-{index}" for index in range(32)])
    schedule = build_fold_fixed_schedule(
        labels,
        cases,
        samples,
        steps=2,
        batch_size=8,
        center="2",
        fit_row_hash="fit",
        recipe_version="v1",
    )
    kwargs = {
        "schedule": schedule,
        "spec": StepTrainingSpec(
            optimizer_steps=2,
            batch_size=8,
            hidden_dim=8,
            latent_dim=2,
        ),
        "pairing_key": "initialization",
        "training_key_hash": "test",
        "device": "cpu",
    }
    embeddings = rng.normal(size=(32, 128)).astype("float32")
    with pytest.raises(ProtocolError, match="must have shape"):
        train_fixed_steps(
            embeddings,
            labels,
            epsilon_trace=np.zeros((2, 8, 3), dtype=np.float32),
            **kwargs,
        )
    with pytest.raises(ProtocolError, match="content hash mismatch"):
        train_fixed_steps(
            embeddings,
            labels,
            epsilon_trace=np.zeros((2, 8, 2), dtype=np.float32),
            epsilon_trace_hash="not-the-content-hash",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"optimizer_steps": 0},
        {"batch_size": 7},
        {"latent_dim": 0},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"beta_final": float("nan")},
        {"gradient_clip_norm": 0.0},
    ),
)
def test_fixed_step_training_spec_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ProtocolError):
        StepTrainingSpec(**kwargs)


def test_recovered_pilot_modules_are_thin_compatibility_exports() -> None:
    from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.case_balanced_sampler import (
        build_fold_fixed_schedule as recovered_build_fold_fixed_schedule,
    )
    from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.step_training import (
        PilotRuntime as RecoveredPilotRuntime,
        train_fixed_steps as recovered_train_fixed_steps,
    )

    assert recovered_build_fold_fixed_schedule is build_fold_fixed_schedule
    assert recovered_train_fixed_steps is train_fixed_steps
    assert FixedStepTrainingRuntime is PilotRuntime is RecoveredPilotRuntime


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
