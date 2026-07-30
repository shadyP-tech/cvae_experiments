from pathlib import Path

import numpy as np

from midogpp_thesis.real_features.classifier_reference.uniform_b_robust_interaction_probe.config import (
    BILINEAR_RANKS,
    ROBUST_OBJECTIVES,
    load_robust_interaction_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_robust_interaction_probe.models import (
    group_sample_weights,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_robust_interaction_probe_v1.yaml"
)


def test_robust_interaction_grid_is_bounded() -> None:
    config = load_robust_interaction_config(CONFIG)
    assert config.robust_objectives == ROBUST_OBJECTIVES
    assert config.bilinear_ranks == BILINEAR_RANKS
    assert config.global_dim + config.local_dim == 3840
    assert config.bilinear_epochs == 1
    assert config.gpu_devices == (0, 1)
    assert config.claim_boundary["validation_scored"] is False


def test_group_weights_equalize_center_class_mass() -> None:
    centers = np.asarray(["0", "0", "0", "1", "1", "1", "1"])
    labels = np.asarray([0, 0, 1, 0, 1, 1, 1])
    groups = sorted({(str(center), int(label)) for center, label in zip(centers, labels)})
    mass = {group: 1.0 / len(groups) for group in groups}
    weights = group_sample_weights(centers, labels, mass)
    totals = {
        group: float(np.sum(weights[(centers == group[0]) & (labels == group[1])]))
        for group in groups
    }
    assert np.allclose(list(totals.values()), list(totals.values())[0])
    assert np.isclose(np.mean(weights), 1.0)
