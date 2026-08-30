from __future__ import annotations

import inspect

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_compatibility import (
    CompatibilityAblationFold,
    calibrate_own_source_energy,
    class_marginal_variational_energy,
    decide_compatibility_ablation,
    shrink_eligible_weight,
)


def test_variational_energy_is_label_free_and_class_marginalized() -> None:
    assert "label" not in inspect.signature(class_marginal_variational_energy).parameters
    surface = class_marginal_variational_energy(
        source_center="1",
        training_seed=17,
        row_ids=("r0", "r1"),
        case_ids=("c0", "c1"),
        reconstruction_distortion=((1.0, 2.0), (0.5, 0.5)),
        prior_rate=((0.0, 0.0), (0.5, 1.5)),
        beta=1.0,
    )
    expected0 = -np.log(0.5 * np.exp(-1.0) + 0.5 * np.exp(-2.0))
    assert surface.per_row[0] == pytest.approx(expected0)
    assert surface.exact_nelbo is False
    assert surface.labels_consumed is False
    with pytest.raises(ValueError):
        surface.per_row[0] = 2.0


def test_variational_energy_rejects_bad_geometry_and_negative_distortion() -> None:
    with pytest.raises(ProtocolError):
        class_marginal_variational_energy(
            source_center="1",
            training_seed=17,
            row_ids=("r0",),
            case_ids=("c0",),
            reconstruction_distortion=((-1.0, 1.0),),
            prior_rate=((0.0, 0.0),),
        )


def test_calibration_requires_complete_source_seed_cartesian_product() -> None:
    query = {
        ("1", 17): {"q0": 3.0, "q1": 5.0},
        ("1", 42): {"q0": 2.0, "q1": 4.0},
        ("2", 17): {"q0": 1.0, "q1": 3.0},
        ("2", 42): {"q0": 2.0, "q1": 2.0},
    }
    own = {
        key: {"s0": 1.0, "s1": 2.0, "s2": 3.0} for key in query
    }
    result = calibrate_own_source_energy(
        query,
        own,
        candidate_sources=("2", "1"),
        training_seeds=(17, 42),
    )
    assert tuple(result.mean_z_by_source) == ("1", "2")
    assert result.labels_consumed is False
    with pytest.raises(ProtocolError):
        calibrate_own_source_energy(
            {key: value for key, value in query.items() if key != ("2", 42)},
            own,
            candidate_sources=("1", "2"),
            training_seeds=(17, 42),
        )


def test_compatibility_only_shrinks_an_action_model_decision() -> None:
    blocked = shrink_eligible_weight(
        action_model_eligible=False,
        original_weight=1.0,
        calibrated_z=-100.0,
        enabled=True,
    )
    assert blocked.final_weight == 0.0
    assert blocked.compatibility_authorized_action is False

    unchanged = shrink_eligible_weight(
        action_model_eligible=True,
        original_weight=0.75,
        calibrated_z=1.0,
        enabled=True,
    )
    assert unchanged.final_weight == pytest.approx(0.75)

    shrunk = shrink_eligible_weight(
        action_model_eligible=True,
        original_weight=0.75,
        calibrated_z=3.0,
        enabled=True,
    )
    assert 0.0 < shrunk.final_weight < 0.75

    abstained = shrink_eligible_weight(
        action_model_eligible=True,
        original_weight=0.75,
        calibrated_z=4.0,
        enabled=True,
    )
    assert abstained.final_weight == 0.0


def test_compatibility_defaults_off_unless_held_query_ablation_improves_safely() -> None:
    unsafe = tuple(
        CompatibilityAblationFold(str(index), 4, 0.01, 0.01, -0.01)
        for index in range(4)
    )
    decision = decide_compatibility_ablation(unsafe)
    assert decision.enabled is False
    assert "brier_noninferiority_failed" in decision.rejection_reasons

    safe = tuple(
        CompatibilityAblationFold(str(index), 4, 0.02, -0.01, -0.01)
        for index in range(4)
    )
    decision = decide_compatibility_ablation(safe)
    assert decision.enabled is True
    assert decision.rejection_reasons == ()
    assert decision.target_support_labels_used is False
    assert decision.target_evaluation_labels_used is False
