from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.uniform_b_nonlinear_probe.config import (
    EXPECTED_CANDIDATES,
    load_nonlinear_probe_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_nonlinear_probe.estimator import (
    effective_gamma,
    median_distance_fit,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_nonlinear_probe.statistics import (
    progression_decision,
    summarize_and_select,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_nystroem_nonlinear_probe_v1.yaml"
)


def test_nonlinear_probe_grid_and_claim_are_frozen() -> None:
    config = load_nonlinear_probe_config(CONFIG)
    assert len(config.candidates) == EXPECTED_CANDIDATES == 36
    assert config.width_multipliers == (0.5, 1.0, 2.0)
    assert config.components == (256, 512, 1024)
    assert config.logistic_cs == (0.01, 0.1, 1.0, 10.0)
    assert config.runtime.pair_jobs == 4
    assert config.runtime.threads_per_job == 3
    assert config.claim_boundary["validation_predictions_generated"] is False


def test_width_multiplier_maps_to_sigma_not_gamma_multiplier() -> None:
    median = 4.0
    assert effective_gamma(0.5, median) == pytest.approx(1.0 / 8.0)
    assert effective_gamma(1.0, median) == pytest.approx(1.0 / 32.0)
    assert effective_gamma(2.0, median) == pytest.approx(1.0 / 128.0)


def test_gamma_sample_is_deterministic_and_fit_key_namespaced() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(80, 12)).astype(np.float32)
    ids = np.asarray([f"s{i:03d}" for i in range(80)])
    first = median_distance_fit(x, ids, seed=42017, cap=32, fit_key="pair:0,1")
    replay = median_distance_fit(x, ids, seed=42017, cap=32, fit_key="pair:0,1")
    other = median_distance_fit(x, ids, seed=42017, cap=32, fit_key="pair:0,2")
    assert first == replay
    assert first["gamma_sample_row_hash"] != other["gamma_sample_row_hash"]


def test_selector_tie_break_prefers_smaller_basis_then_width_one() -> None:
    config = load_nonlinear_probe_config(CONFIG)
    centers = ("0", "1")
    candidates = (
        next(
            candidate
            for candidate in config.candidates
            if candidate.width_multiplier == 2
            and candidate.n_components == 256
            and candidate.logistic_c == 0.01
        ),
        next(
            candidate
            for candidate in config.candidates
            if candidate.width_multiplier == 1
            and candidate.n_components == 256
            and candidate.logistic_c == 0.01
        ),
        next(
            candidate
            for candidate in config.candidates
            if candidate.width_multiplier == 1
            and candidate.n_components == 512
            and candidate.logistic_c == 0.01
        ),
    )
    cells = []
    for outer, inner in (("0", "1"), ("1", "0")):
        for candidate in candidates:
            cells.append(
                {
                    "outer_center": outer,
                    "inner_center": inner,
                    "candidate_id": candidate.candidate_id,
                    "bacc": 0.8,
                    "positive_recall": 0.8,
                    "specificity": 0.8,
                }
            )
    summaries, selected = summarize_and_select(cells, candidates, centers)
    assert len(summaries) == 6
    assert selected["0"]["width_multiplier"] == 1.0
    assert selected["0"]["n_components"] == 256


def test_progression_gate_is_equal_center_and_conjunctive() -> None:
    config = load_nonlinear_probe_config(CONFIG)
    comparisons = [
        {
            "delta_bacc": 0.02,
            "delta_positive_recall": 0.01,
            "delta_specificity": 0.0,
        }
        for _ in range(9)
    ]
    stability = [
        {"landmark_seed": seed, "delta_bacc": 0.015}
        for seed in config.stability_landmark_seeds
        for _ in range(9)
    ]
    bootstrap = {"percentile_2_5": -0.5}
    decision = progression_decision(comparisons, stability, bootstrap, config.gate)
    assert decision["passed"] is True
    assert decision["bootstrap_is_supportive_not_conjunctive"] is True
    comparisons[0]["delta_bacc"] = -0.02
    decision = progression_decision(comparisons, stability, bootstrap, config.gate)
    assert decision["passed"] is False


def test_config_rejects_validation_scoring(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace(
        "validation_predictions_generated: false",
        "validation_predictions_generated: true",
    )
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProtocolError, match="claim boundary"):
        load_nonlinear_probe_config(path)
