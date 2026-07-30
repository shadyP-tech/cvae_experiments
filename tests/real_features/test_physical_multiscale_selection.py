from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
import yaml

from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec
from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    canonical_matched_reference_specs,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.config import (
    GateConfig,
    REPRESENTATION_DIMS,
    load_physical_multiscale_pilot_config,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.selection import (
    choose_representation_from_vectors,
    select_representation_for_outer,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.frames import (
    MultiRepresentationFrame,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


CONFIG = Path(
    "experiments/midogpp/stages/10_real_feature_reference/configs/"
    "physical_multiscale_center_pooling_pilot_v1.yaml"
)


def test_frozen_config_contains_literal_ordered_thirty_candidate_pool(
    tmp_path: Path,
) -> None:
    config = load_physical_multiscale_pilot_config(CONFIG)

    assert config.expected_selector_cells == 2160
    assert config.expected_candidate_summaries == 270
    assert len(config.classifier_specs) * len(REPRESENTATION_DIMS) == 30

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["classifier_grid"]["literal_candidate_ids"] = payload["classifier_grid"][
        "literal_candidate_ids"
    ][:-1]
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="literal ordered 30-candidate"):
        load_physical_multiscale_pilot_config(drifted)


def test_gate_falls_back_to_a_and_breaks_passing_tie_by_lower_dimension() -> None:
    spec = canonical_matched_reference_specs(classifier_seed=23)[0]
    centers = ("0", "1", "2", "3", "5", "6", "7", "8")
    baseline = {center: 0.60 for center in centers}
    passing = {center: 0.63 for center in centers}
    selected = {
        "canonical_a": (spec, baseline),
        "jpeg_center_b": (spec, passing),
        "physical_multiscale_center_c": (spec, passing),
    }

    chosen = choose_representation_from_vectors(
        selected,
        centers=centers,
        gate=GateConfig(),
    )
    assert chosen[0] == "jpeg_center_b"
    assert chosen[5] is True

    selected["jpeg_center_b"] = (spec, {**passing, "0": 0.58})
    selected["physical_multiscale_center_c"] = (
        spec,
        {center: 0.61 for center in centers},
    )
    fallback = choose_representation_from_vectors(
        selected,
        centers=centers,
        gate=GateConfig(),
    )
    assert fallback[0] == "canonical_a"
    assert fallback[2:] == (0.0, 0.0, 0, False)


def test_selector_scores_complete_representation_spec_inner_matrix_without_h() -> None:
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    centers = ("1",) * 4 + ("2",) * 4
    base = np.asarray(
        [[-2.0, 0.0], [2.0, 0.0], [-1.0, 1.0], [1.0, 1.0]] * 2,
        dtype=float,
    )
    frame = MultiRepresentationFrame(
        sample_ids=tuple(f"s{index}" for index in range(8)),
        case_ids=tuple(f"case{index}" for index in range(8)),
        labels=labels,
        centers=centers,
        embeddings={
            "canonical_a": base,
            "jpeg_center_b": base,
            "physical_multiscale_center_c": base,
        },
    )
    spec = ClassifierSpec(
        C=1.0,
        max_iter=5000,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
    )

    decision, cells, summaries = select_representation_for_outer(
        frame,
        outer_target_center="0",
        source_centers=("1", "2"),
        classifier_specs=(spec,),
        gate=GateConfig(),
    )

    assert decision.selected_representation == "canonical_a"
    assert len(cells) == 3 * 1 * 2
    assert len(summaries) == 3
    assert {row["inner_pseudo_target_center"] for row in cells} == {"1", "2"}
    assert all(row["fit_used_target_center"] is False for row in cells)
