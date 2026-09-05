"""Primitive descriptors must use exactly the surface the runtime executes."""
import numpy as np
import pytest

from midogpp_thesis.cvae.routing.risk_aligned_router_v20.contracts import Direction
from midogpp_thesis.cvae.runtime.harp_v20_execution.mechanism_features import (
    LABEL_FREE_FEATURE_NAMES, feature_values,
)


@pytest.mark.parametrize("direction,active", [
    (Direction.D01, (True, False)),
    (Direction.D10, (False, True)),
    (Direction.FULL, (True, True)),
])
def test_directional_flip_and_mass_describe_only_executed_surface(direction, active):
    baseline = np.asarray((.4, .6), dtype=np.float32)
    challenger = np.asarray((.7, .3), dtype=np.float32)
    mask = np.asarray(active)
    dispersion = np.asarray((.1, .2), dtype=np.float32)
    result = dict(zip(LABEL_FREE_FEATURE_NAMES, feature_values(
        baseline, challenger, dispersion, dispersion,
        active=mask, direction=direction, compatibility=(0., 0., 0., 0.),
    ), strict=True))
    expected = baseline.copy()
    expected[mask] = challenger[mask]
    assert result['threshold_flip_fraction'] == np.mean((baseline >= .5) != (expected >= .5))
    assert result['direction_aligned_branch_mass'] == pytest.approx(
        np.mean(np.abs(expected.astype(np.float64) - baseline)))


def test_unexecuted_direction_cannot_change_primitive_action_features():
    baseline = np.asarray((.4, .6), dtype=np.float32)
    mask = np.asarray((True, False))
    dispersion = np.zeros(2, dtype=np.float32)
    def describe(second):
        return feature_values(baseline, np.asarray((.7, second), dtype=np.float32),
            dispersion, dispersion, active=mask, direction=Direction.D01,
            compatibility=(0., 0., 0., 0.))
    assert describe(.01) == describe(.99)
