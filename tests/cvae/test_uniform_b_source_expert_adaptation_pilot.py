from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.block_frame import (
    bridge_a_prefix,
    fit_pilot_frame,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.case_balanced_sampler import (
    build_balanced_schedule,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.case_split import (
    deterministic_case_holdout,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.conservative_prior import (
    CONDITIONAL_PRIOR_FAMILY,
    STANDARD_PRIOR_FAMILY,
    fit_shrunk_diagonal_prior,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.runner import (
    _real_reference_audit,
    _run_job,
    run_pilot,
)
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.step_training import (
    StepTrainingSpec,
    beta_for_step,
    train_fixed_steps,
)
from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.config import load_pilot_config
from midogpp_thesis.cvae.expert_bank.b_adaptation_pilot.validation import validate_final_bundle


def test_three_frames_are_128d_train_only_and_block_ordered() -> None:
    rng = np.random.default_rng(7)
    fit = rng.normal(size=(140, 3840)).astype(np.float32)
    eval_ = rng.normal(size=(11, 3840)).astype(np.float32)
    block = fit_pilot_frame(
        "b_block_pca96_32",
        fit,
        fit_sample_hash="fit-only",
        pca_n_oversamples=10,
    )
    assert block.to_payload()["pca_policy"]["n_oversamples"] == 10
    projected = block.transform(eval_)
    assert projected.shape == (11, 128)
    assert block.fit_sample_hash == "fit-only"
    assert block.state_hash == stable_hash(block.to_payload())
    assert [(state.start, state.stop, state.output_dim) for state in block.blocks] == [
        (0, 2560, 96),
        (2560, 3840, 32),
    ]
    assert block.inverse_transform(block.transform(eval_)).shape == (11, 3840)


def test_b_prefix_bridge_passes_exact_and_rejects_drift() -> None:
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20, 2560))
    b = np.concatenate([a.copy(), rng.normal(size=(20, 1280))], axis=1)
    assert bridge_a_prefix(b, a)["status"] == "PASS"
    b[:, :2560] += 0.1
    with pytest.raises(ProtocolError, match="bridge"):
        bridge_a_prefix(b, a)


def test_case_split_is_deterministic_disjoint_and_two_class() -> None:
    cases = [f"case-{index}" for index in range(12) for _ in range(4)]
    labels = [value for _ in range(12) for value in (0, 0, 1, 1)]
    first = deterministic_case_holdout(
        cases, labels, validation_fraction=0.2, seed=2718
    )
    second = deterministic_case_holdout(
        cases, labels, validation_fraction=0.2, seed=2718
    )
    assert first == second
    assert set(first.fit_cases).isdisjoint(first.eval_cases)
    assert {labels[index] for index in first.fit_indices} == {0, 1}
    assert {labels[index] for index in first.eval_indices} == {0, 1}


def test_schedule_has_exact_class_quota_is_paired_and_case_balanced() -> None:
    labels = np.asarray([0] * 12 + [1] * 12)
    cases = np.asarray(
        ["a"] * 4 + ["b"] * 4 + ["c"] * 4
        + ["d"] * 4 + ["e"] * 4 + ["f"] * 4
    )
    samples = np.asarray([f"s{index}" for index in range(24)])
    first = build_balanced_schedule(
        labels, cases, samples, steps=80, batch_size=8, seed=19
    )
    repeated = build_balanced_schedule(
        labels, cases, samples, steps=80, batch_size=8, seed=19
    )
    changed = build_balanced_schedule(
        labels, cases, samples, steps=80, batch_size=8, seed=20
    )
    assert first.stream_hash == repeated.stream_hash
    assert first.stream_hash != changed.stream_hash
    for batch in np.asarray(first.batches):
        assert int((labels[batch] == 0).sum()) == 4
        assert int((labels[batch] == 1).sum()) == 4
    for label, eligible in ((0, ("a", "b", "c")), (1, ("d", "e", "f"))):
        exposure = [first.case_class_exposure[f"{label}:{case}"] for case in eligible]
        assert max(exposure) - min(exposure) < 50


def test_fixed_shrinkage_formula_and_standard_fallback() -> None:
    rng = np.random.default_rng(2)
    labels = np.asarray([0] * 70 + [1] * 70)
    cases = np.asarray(
        [f"c0-{index % 7}" for index in range(70)]
        + [f"c1-{index % 7}" for index in range(70)]
    )
    mu = rng.normal(size=(140, 3))
    logvar = np.zeros((140, 3))
    prior = fit_shrunk_diagonal_prior(
        mu,
        logvar,
        labels,
        cases,
        source_state_hash="checkpoint",
    )
    assert prior.realized_family == CONDITIONAL_PRIOR_FAMILY
    expected_mean = 0.25 * mu[labels == 0].mean(axis=0)
    expected_var = np.clip(
        0.75 + 0.25 * (mu[labels == 0].var(axis=0) + 1.0), 0.25, 4.0
    )
    assert np.allclose(np.asarray(prior.means)[0], expected_mean)
    assert np.allclose(np.asarray(prior.variances)[0], expected_var)
    fallback = fit_shrunk_diagonal_prior(
        mu[:20],
        logvar[:20],
        np.asarray([0] * 10 + [1] * 10),
        np.asarray([f"x-{index}" for index in range(20)]),
        source_state_hash="checkpoint",
    )
    assert fallback.realized_family == STANDARD_PRIOR_FAMILY
    assert np.array_equal(np.asarray(fallback.means), np.zeros((2, 3)))
    assert np.array_equal(np.asarray(fallback.variances), np.ones((2, 3)))


def test_fixed_step_training_pairs_stream_and_uses_step_warmup() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 128)).astype(np.float32)
    labels = np.asarray([0] * 20 + [1] * 20)
    cases = np.asarray(
        [f"a-{index % 5}" for index in range(20)]
        + [f"b-{index % 5}" for index in range(20)]
    )
    samples = np.asarray([f"s-{index}" for index in range(40)])
    schedule = build_balanced_schedule(
        labels, cases, samples, steps=4, batch_size=8, seed=31
    )
    spec = StepTrainingSpec(
        optimizer_steps=4,
        batch_size=8,
        hidden_dim=8,
        latent_dim=3,
        kl_warmup_steps=2,
    )
    first = train_fixed_steps(
        x,
        labels,
        schedule=schedule,
        spec=spec,
        pairing_key="paired",
        training_key_hash=stable_hash({"arm": "a"}),
        device="cpu",
    )
    second = train_fixed_steps(
        x + 0.1,
        labels,
        schedule=schedule,
        spec=spec,
        pairing_key="paired",
        training_key_hash=stable_hash({"arm": "b"}),
        device="cpu",
    )
    assert first.initialization_hash == second.initialization_hash
    assert first.posterior_stream_hash == second.posterior_stream_hash
    assert first.schedule_hash == second.schedule_hash
    assert first.checkpoint_hash != second.checkpoint_hash
    assert beta_for_step(spec, 1) == spec.beta_final / 2
    assert beta_for_step(spec, 2) == spec.beta_final
    assert first.diagnostics[-1]["step"] == 4


def test_tiny_job_runs_all_roles_and_resumes_without_retraining(tmp_path) -> None:
    rng = np.random.default_rng(13)
    labels_fit = np.asarray([0] * 20 + [1] * 20)
    labels_eval = np.asarray([0] * 8 + [1] * 8)
    x_fit = rng.normal(scale=0.25, size=(40, 128)).astype(np.float32)
    x_eval = rng.normal(scale=0.25, size=(16, 128)).astype(np.float32)
    x_fit[:, 0] += labels_fit * 3.0
    x_eval[:, 0] += labels_eval * 3.0
    prepared = tmp_path / "arrays.npz"
    np.savez_compressed(
        prepared,
        x_fit=x_fit,
        y_fit=labels_fit,
        case_fit=np.asarray(
            [f"n-{index % 5}" for index in range(20)]
            + [f"p-{index % 5}" for index in range(20)]
        ),
        sample_fit=np.asarray([f"fit-{index}" for index in range(40)]),
        x_eval=x_eval,
        y_eval=labels_eval,
        case_eval=np.asarray([f"eval-{index // 2}" for index in range(16)]),
        sample_eval=np.asarray([f"eval-{index}" for index in range(16)]),
    )
    spec = StepTrainingSpec(
        optimizer_steps=4,
        batch_size=8,
        hidden_dim=8,
        latent_dim=3,
        kl_warmup_steps=2,
    )
    arrays = np.load(prepared, allow_pickle=False)
    expected_real_reference = _real_reference_audit(
        center="2",
        arm="b_block_pca96_32",
        x_fit=arrays["x_fit"],
        y_fit=arrays["y_fit"],
        case_fit=arrays["case_fit"],
        x_eval=arrays["x_eval"],
        y_eval=arrays["y_eval"],
        case_eval=arrays["case_eval"],
        classifier_c=0.01,
        minimum_real_bacc=0.60,
    )
    job = {
        "center": "2",
        "arm": "b_block_pca96_32",
        "training_seed": 17,
        "prepared_path": str(prepared),
        "frame_hash": "frame",
        "fit_row_hash": "fit",
        "eval_row_hash": "eval",
        "protocol_hash": "protocol",
        "artifact_root": str(tmp_path / "artifact"),
        "device": "cpu",
        "cpu_threads": 1,
        "generation_seeds": (17,),
        "generated_per_class": 8,
        "classifier_c": 0.01,
        "minimum_real_bacc": 0.60,
        "expected_real_reference": expected_real_reference,
        "training_spec": spec.to_payload(),
        "conditional_prior": {
            "rho": 0.25,
            "min_rows": 64,
            "min_cases": 5,
            "variance_clip": (0.25, 4.0),
            "max_condition_number": 1e4,
        },
    }
    first = _run_job(job)
    second = _run_job(job)
    assert first["optimizer_steps"] == 4
    assert len(first["metrics"]) == 5
    assert {row["representation_role"] for row in first["metrics"]} == {
        "real", "decode_mu", "posterior_sample", "prior_standard", "prior_conditional"
    }
    assert first["cache_status"] == "miss"
    assert second["cache_status"] == "hit"
    assert first["checkpoint_hash"] == second["checkpoint_hash"]


def test_v2_locks_the_reviewed_randomized_pca_policy() -> None:
    config = load_pilot_config(
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
        "uniform_b_source_expert_adaptation_pilot_v2.yaml"
    )
    assert config.name == "uniform_b_source_expert_adaptation_pilot_v2"
    assert config.pca_svd_solver == "randomized"
    assert config.pca_random_state == 0
    assert config.pca_n_oversamples == 10
    assert config.pca_iterated_power == 4


def test_v1_is_terminally_nonrunnable_even_via_direct_cli_path(tmp_path) -> None:
    config = load_pilot_config(
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
        "uniform_b_source_expert_adaptation_pilot_v1.yaml"
    )
    with pytest.raises(ProtocolError, match="terminally failed"):
        run_pilot(config, artifact_root=tmp_path / "must-not-start")
    assert not (tmp_path / "must-not-start").exists()


def test_v2_amendment_drift_fails_closed(tmp_path) -> None:
    source = Path(
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
        "uniform_b_source_expert_adaptation_pilot_v2.yaml"
    ).read_text(encoding="utf-8")
    drifted = tmp_path / "drifted-v2.yaml"
    drifted.write_text(
        source.replace("confirmation_eligible: false", "confirmation_eligible: true"),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="amendment"):
        load_pilot_config(drifted)


def test_config_and_validator_fail_closed(tmp_path) -> None:
    source = Path(
        "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
        "uniform_b_source_expert_adaptation_pilot_v1.yaml"
    ).read_text(encoding="utf-8")
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(source.replace("optimizer_steps: 1000", "optimizer_steps: 999"), encoding="utf-8")
    with pytest.raises(ProtocolError, match="optimizer_steps"):
        load_pilot_config(drifted)
    report = validate_final_bundle(tmp_path / "missing-bundle")
    assert report["status"] == "FAIL"
    assert report["errors"]

