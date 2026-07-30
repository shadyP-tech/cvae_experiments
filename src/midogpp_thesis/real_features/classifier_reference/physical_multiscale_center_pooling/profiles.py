"""Immutable representation profiles for the versioned Stage-10 pilots."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash

from ..classifiers import ClassifierSpec
from ..protocol import ProtocolError


CENTER_POOLING_PILOT_V1 = "physical_multiscale_center_pooling_pilot_v1"
ANNOTATION_LOCAL_POOLING_PILOT_V2 = (
    "physical_multiscale_annotation_local_pooling_pilot_v2"
)
CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3 = (
    "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3"
)

V1_CANDIDATE_GRID_HASH = "b572cc680b088ecd"
V2_CANDIDATE_GRID_HASH = "fec13ae0471e3481"
V3_CANDIDATE_GRID_HASH = "2f651b2f8bd53c1a"


@dataclass(frozen=True)
class RepresentationDefinition:
    """One representation identity and its frozen feature dimension."""

    representation_id: str
    feature_dim: int


@dataclass(frozen=True)
class PhysicalMultiscaleProfile:
    """Protocol identity for one exact A/B/C candidate surface."""

    profile_id: str
    experiment_name: str
    representations: tuple[RepresentationDefinition, ...]
    expected_candidate_grid_hash: str
    selector_metric: str = "bacc"
    selector_aggregation: str = "equal_center_arithmetic_mean"
    auroc_role: str = "descriptive_only"
    descriptive_metrics: tuple[str, ...] = ("macro_f1", "auroc")

    @property
    def representation_order(self) -> tuple[str, ...]:
        return tuple(item.representation_id for item in self.representations)

    @property
    def representation_dims(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                item.representation_id: int(item.feature_dim)
                for item in self.representations
            }
        )

    @property
    def selector_decision_metric(self) -> str:
        return "equal_center_mean_bacc"


CENTER_POOLING_PROFILE_V1 = PhysicalMultiscaleProfile(
    profile_id=CENTER_POOLING_PILOT_V1,
    experiment_name=CENTER_POOLING_PILOT_V1,
    representations=(
        RepresentationDefinition("canonical_a", 2560),
        RepresentationDefinition("jpeg_center_b", 3840),
        RepresentationDefinition("physical_multiscale_center_c", 11520),
    ),
    expected_candidate_grid_hash=V1_CANDIDATE_GRID_HASH,
)

ANNOTATION_LOCAL_PROFILE_V2 = PhysicalMultiscaleProfile(
    profile_id=ANNOTATION_LOCAL_POOLING_PILOT_V2,
    experiment_name=ANNOTATION_LOCAL_POOLING_PILOT_V2,
    representations=(
        RepresentationDefinition("canonical_a", 2560),
        RepresentationDefinition("annotation_jpeg_fixed_center_b_v2", 3840),
        RepresentationDefinition("physical_multiscale_annotation_local_c_v2", 11520),
    ),
    expected_candidate_grid_hash=V2_CANDIDATE_GRID_HASH,
)

CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3 = PhysicalMultiscaleProfile(
    profile_id=CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
    experiment_name=CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
    representations=(
        RepresentationDefinition("canonical_a", 2560),
        RepresentationDefinition("annotation_jpeg_fixed_center_b_v3", 3840),
        RepresentationDefinition(
            "physical_multiscale_clipped_bbox_annotation_local_c_v3",
            11520,
        ),
    ),
    expected_candidate_grid_hash=V3_CANDIDATE_GRID_HASH,
)

PHYSICAL_MULTISCALE_PROFILES = (
    CENTER_POOLING_PROFILE_V1,
    ANNOTATION_LOCAL_PROFILE_V2,
    CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
)
_PROFILES_BY_ID = MappingProxyType(
    {profile.profile_id: profile for profile in PHYSICAL_MULTISCALE_PROFILES}
)


def get_physical_multiscale_profile(profile_id: str) -> PhysicalMultiscaleProfile:
    """Resolve one accepted immutable Stage-10 pilot profile."""

    try:
        return _PROFILES_BY_ID[str(profile_id)]
    except KeyError as exc:
        raise ProtocolError(
            f"Unsupported physical multiscale profile: {profile_id!r}"
        ) from exc


def profile_candidate_payload(
    profile: PhysicalMultiscaleProfile,
    specs: Sequence[ClassifierSpec],
) -> tuple[dict[str, object], ...]:
    """Return the exact ordered 3x10 representation/classifier pool."""

    canonical_profile = get_physical_multiscale_profile(profile.profile_id)
    if profile != canonical_profile:
        raise ProtocolError(
            f"Physical multiscale profile definition drifted: {profile.profile_id!r}"
        )
    return tuple(
        {
            "candidate_id": f"{representation.representation_id}:{spec.config_hash}",
            "representation_id": representation.representation_id,
            "feature_dim": int(representation.feature_dim),
            "classifier": spec.to_payload(),
        }
        for representation in canonical_profile.representations
        for spec in specs
    )


def profile_candidate_grid_hash(
    profile: PhysicalMultiscaleProfile,
    specs: Sequence[ClassifierSpec],
) -> str:
    return stable_hash(profile_candidate_payload(profile, specs))


def assert_profile_candidate_grid(
    profile: PhysicalMultiscaleProfile,
    specs: Sequence[ClassifierSpec],
) -> str:
    """Fail closed if either frozen profile's literal candidate identity drifts."""

    actual = profile_candidate_grid_hash(profile, specs)
    if actual != profile.expected_candidate_grid_hash:
        raise ProtocolError(
            "Physical multiscale profile candidate grid drifted: "
            f"profile={profile.profile_id!r} "
            f"expected={profile.expected_candidate_grid_hash} actual={actual}"
        )
    return actual
