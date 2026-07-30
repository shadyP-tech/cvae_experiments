from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    CANONICAL_GRID_HASH,
    canonical_matched_reference_specs,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.config import (
    REPRESENTATION_DIMS,
    REPRESENTATION_ORDER,
    load_physical_multiscale_pilot_config,
    representation_candidate_grid_hash,
    representation_candidate_payload,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.profiles import (
    ANNOTATION_LOCAL_POOLING_PILOT_V2,
    ANNOTATION_LOCAL_PROFILE_V2,
    CENTER_POOLING_PILOT_V1,
    CENTER_POOLING_PROFILE_V1,
    CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
    CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    PHYSICAL_MULTISCALE_PROFILES,
    V1_CANDIDATE_GRID_HASH,
    V2_CANDIDATE_GRID_HASH,
    V3_CANDIDATE_GRID_HASH,
    get_physical_multiscale_profile,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


CONFIG_ROOT = Path(
    "experiments/midogpp/stages/10_real_feature_reference/configs"
)
V1_CONFIG = CONFIG_ROOT / "physical_multiscale_center_pooling_pilot_v1.yaml"
V2_CONFIG = (
    CONFIG_ROOT / "physical_multiscale_annotation_local_pooling_pilot_v2.yaml"
)
V3_CONFIG = (
    CONFIG_ROOT
    / "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml"
)


def test_v1_profile_preserves_identity_order_and_literal_hash() -> None:
    specs = canonical_matched_reference_specs(classifier_seed=23)
    config = load_physical_multiscale_pilot_config(V1_CONFIG)

    assert config.profile is CENTER_POOLING_PROFILE_V1
    assert config.profile.profile_id == CENTER_POOLING_PILOT_V1
    assert REPRESENTATION_ORDER == (
        "canonical_a",
        "jpeg_center_b",
        "physical_multiscale_center_c",
    )
    assert REPRESENTATION_DIMS == {
        "canonical_a": 2560,
        "jpeg_center_b": 3840,
        "physical_multiscale_center_c": 11520,
    }
    assert tuple(
        row["candidate_id"] for row in representation_candidate_payload(specs)
    ) == tuple(
        f"{representation_id}:{spec.config_hash}"
        for representation_id in REPRESENTATION_ORDER
        for spec in specs
    )
    assert representation_candidate_grid_hash(specs) == V1_CANDIDATE_GRID_HASH
    assert V1_CANDIDATE_GRID_HASH == "b572cc680b088ecd"


def test_v2_profile_freezes_exact_ordered_three_by_ten_pool() -> None:
    specs = canonical_matched_reference_specs(classifier_seed=23)
    config = load_physical_multiscale_pilot_config(V2_CONFIG)
    candidates = representation_candidate_payload(specs, config.profile)

    assert config.profile is ANNOTATION_LOCAL_PROFILE_V2
    assert config.profile.profile_id == ANNOTATION_LOCAL_POOLING_PILOT_V2
    assert config.representation_order == (
        "canonical_a",
        "annotation_jpeg_fixed_center_b_v2",
        "physical_multiscale_annotation_local_c_v2",
    )
    assert dict(config.representation_dims) == {
        "canonical_a": 2560,
        "annotation_jpeg_fixed_center_b_v2": 3840,
        "physical_multiscale_annotation_local_c_v2": 11520,
    }
    assert len(candidates) == 3 * 10
    expected_representation_ids = tuple(
        representation_id
        for representation_id in config.representation_order
        for _ in range(10)
    )
    assert tuple(
        row["representation_id"] for row in candidates
    ) == expected_representation_ids
    assert [spec.config_hash for spec in config.classifier_specs] == [
        spec.config_hash for spec in specs
    ]
    assert CANONICAL_GRID_HASH == "5abd0897d02bdcaa"
    assert (
        representation_candidate_grid_hash(specs, config.profile)
        == V2_CANDIDATE_GRID_HASH
        == "fec13ae0471e3481"
    )


def test_v3_profile_freezes_distinct_ordered_three_by_ten_pool() -> None:
    specs = canonical_matched_reference_specs(classifier_seed=23)
    config = load_physical_multiscale_pilot_config(V3_CONFIG)
    candidates = representation_candidate_payload(specs, config.profile)

    assert config.profile is CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3
    assert (
        config.profile.profile_id
        == CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3
    )
    assert config.representation_order == (
        "canonical_a",
        "annotation_jpeg_fixed_center_b_v3",
        "physical_multiscale_clipped_bbox_annotation_local_c_v3",
    )
    assert dict(config.representation_dims) == {
        "canonical_a": 2560,
        "annotation_jpeg_fixed_center_b_v3": 3840,
        "physical_multiscale_clipped_bbox_annotation_local_c_v3": 11520,
    }
    assert len(candidates) == 3 * 10
    assert tuple(row["candidate_id"] for row in candidates) == tuple(
        f"{representation_id}:{spec.config_hash}"
        for representation_id in config.representation_order
        for spec in specs
    )
    assert (
        representation_candidate_grid_hash(specs, config.profile)
        == V3_CANDIDATE_GRID_HASH
        == "2f651b2f8bd53c1a"
    )
    assert V3_CANDIDATE_GRID_HASH not in {
        V1_CANDIDATE_GRID_HASH,
        V2_CANDIDATE_GRID_HASH,
    }


def test_only_three_immutable_profiles_are_accepted() -> None:
    assert PHYSICAL_MULTISCALE_PROFILES == (
        CENTER_POOLING_PROFILE_V1,
        ANNOTATION_LOCAL_PROFILE_V2,
        CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    )
    with pytest.raises(FrozenInstanceError):
        CENTER_POOLING_PROFILE_V1.profile_id = "drifted"  # type: ignore[misc]
    with pytest.raises(TypeError):
        CENTER_POOLING_PROFILE_V1.representation_dims["other"] = 1  # type: ignore[index]
    with pytest.raises(ProtocolError, match="Unsupported physical multiscale profile"):
        get_physical_multiscale_profile("arbitrary_profile")


def test_configs_reject_arbitrary_profile_and_representation(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    payload["experiment"]["profile_id"] = "arbitrary_profile"
    arbitrary_profile = tmp_path / "arbitrary_profile.yaml"
    arbitrary_profile.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="Unsupported physical multiscale profile"):
        load_physical_multiscale_pilot_config(arbitrary_profile)

    payload = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    payload["representations"]["arbitrary_representation"] = {"feature_dim": 17}
    arbitrary_representation = tmp_path / "arbitrary_representation.yaml"
    arbitrary_representation.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="exact ordered profile representations"):
        load_physical_multiscale_pilot_config(arbitrary_representation)


@pytest.mark.parametrize(
    "config_path",
    (V1_CONFIG, V2_CONFIG, V3_CONFIG),
)
def test_selector_stays_equal_center_bacc_and_auroc_is_descriptive(
    config_path: Path,
) -> None:
    config = load_physical_multiscale_pilot_config(config_path)

    assert config.selector_metric == "bacc"
    assert config.selector_aggregation == "equal_center_arithmetic_mean"
    assert config.profile.selector_decision_metric == "equal_center_mean_bacc"
    assert config.auroc_role == "descriptive_only"
    assert "auroc" in config.profile.descriptive_metrics
    assert config.gate.mean_delta_min == 0.02
    assert config.gate.strict_win_delta_min == 1.0e-12
    assert config.gate.strict_win_count_min == 6
    assert config.gate.worst_delta_min == -0.01
