from __future__ import annotations

from pathlib import Path

import torch

from midogpp_thesis.cvae.expert_bank.b_stability_probe.config import (
    CENTERS,
    READOUTS,
    TRAINING_SEEDS,
    load_stability_config,
)
from midogpp_thesis.cvae.expert_bank.b_stability_probe.runner import (
    stability_decision,
)
from midogpp_thesis.cvae.expert_bank.b_stability_probe.tail_training import (
    _update_online_parameter_mean,
)
from midogpp_thesis.cvae.expert_bank.b_stability_probe.validation import (
    _independent_decision,
)
from midogpp_thesis.cvae.models import ClassConditionedCVAE


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_block_tail_average_stability_probe_v1.yaml"
)


def test_frozen_config_loads_and_binds_v2() -> None:
    config = load_stability_config(CONFIG)
    assert config.centers == CENTERS
    assert config.training_seeds == TRAINING_SEEDS
    assert config.tail_steps == tuple(range(751, 1001))
    assert config.predecessor_root.name == "uniform_b_source_expert_adaptation_pilot_v2"


def test_online_parameter_mean_is_uniform_and_does_not_mutate_model() -> None:
    model = ClassConditionedCVAE(
        input_dim=128, hidden_dim=8, latent_dim=2, num_hidden_layers=2
    )
    accumulator: dict[str, torch.Tensor] = {}
    original = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    _update_online_parameter_mean(accumulator, model, 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    _update_online_parameter_mean(accumulator, model, 2)
    for name, parameter in model.named_parameters():
        assert torch.allclose(accumulator[name], original[name] + 1.0)
        assert torch.allclose(parameter, original[name] + 2.0)


def test_decision_requires_uniformity_gate() -> None:
    config = load_stability_config(CONFIG)
    comparators = []
    metrics = []
    for center in CENTERS:
        for seed in TRAINING_SEEDS:
            for arm, ratio, bacc in (
                ("a_global_pca128", 0.90, 0.75),
                ("b_joint_pca128", 0.85, 0.74),
                ("b_block_pca96_32", 0.95, 0.79),
            ):
                comparators.append(
                    _row(center, seed, arm, "decode_mu", ratio, bacc, 0.75, 0.75)
                )
            metrics.append(
                _row(
                    center,
                    seed,
                    "b_block_pca96_32",
                    READOUTS[0],
                    0.95,
                    0.79,
                    0.75,
                    0.75,
                    readout=True,
                )
            )
            metrics.append(
                _row(
                    center,
                    seed,
                    "b_block_pca96_32",
                    READOUTS[1],
                    0.96,
                    0.80,
                    0.76,
                    0.76,
                    readout=True,
                )
            )
    passed = stability_decision(
        metrics, comparators, config.gates, endpoint_replay_exact=True
    )
    assert passed["decision"] == "TAIL_AVERAGING_STABILIZES_B_BLOCK"

    unstable = [dict(row) for row in metrics]
    target = next(
        row
        for row in unstable
        if row["center"] == "9"
        and row["training_seed"] == 101
        and row["readout"] == READOUTS[1]
    )
    target["positive_recall"] = 0.50
    failed = stability_decision(
        unstable, comparators, config.gates, endpoint_replay_exact=True
    )
    assert failed["decision"] == "TAIL_AVERAGING_INSUFFICIENT"
    assert (
        failed["observations"][
            "maximum_within_center_class_direction_seed_range"
        ]
        > 0.15
    )

    uniformly_degraded = [dict(row) for row in metrics]
    for row in uniformly_degraded:
        if row["center"] == "9" and row["readout"] == READOUTS[1]:
            row["specificity"] = 0.60
    stable_but_worse = stability_decision(
        uniformly_degraded, comparators, config.gates, endpoint_replay_exact=True
    )
    assert stable_but_worse["decision"] == "TAIL_AVERAGING_INSUFFICIENT"
    failed_gates = {
        row["gate"] for row in stable_but_worse["gate_audit"] if not row["passed"]
    }
    assert "center_9_mean_specificity_delta_vs_terminal" in failed_gates
    assert (
        stable_but_worse["observations"][
            "maximum_within_center_class_direction_seed_range"
        ]
        <= 0.15
    )
    independent = _independent_decision(
        uniformly_degraded,
        comparators,
        config.gates,
        endpoint_replay_exact=True,
    )
    assert independent["observations"] == stable_but_worse["observations"]
    assert independent["gate_audit"] == stable_but_worse["gate_audit"]


def _row(
    center: str,
    seed: int,
    arm: str,
    role: str,
    ratio: float,
    bacc: float,
    recall: float,
    specificity: float,
    *,
    readout: bool = False,
) -> dict[str, object]:
    row = {
        "center": center,
        "training_seed": seed,
        "arm": arm,
        "preservation_ratio": ratio,
        "bacc": bacc,
        "positive_recall": recall,
        "specificity": specificity,
        "real_reference_bacc": 0.80,
    }
    row["readout" if readout else "representation_role"] = role
    return row
