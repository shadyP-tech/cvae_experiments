"""Scientific identities for the frozen MIDOG++ metadata compatibility proxy."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError


EXPERIMENT_ID = "midogpp.routing_compatibility.uniform_b_v2_metadata_exact_match_lock.v1"
EXPERIMENT_NAME = "uniform_b_v2_metadata_exact_match_compatibility_v1"
OUTPUT_ARTIFACT_ID = "midogpp_output_uniform_b_v2_metadata_exact_match_compatibility_v1"
INPUT_ARTIFACT_ID = "midogpp_routing_metadata_profiles_v1"
CLAIM_SCOPE = "routing_compatibility_only"

DOMAIN_MAPPING_MEMBER = "domain_mapping.json"
DOMAIN_MAPPING_SHA256 = (
    "79d703ccf3085ae3968698c2ac44a3eabc2713b434762cc6b2fd2fa90126a211"
)
DOMAIN_AXIS = "tumor_type|lab_or_origin|scanner_model"
ORDERED_AXES = ("tumor_type", "lab_or_origin", "scanner_model")
ALL_SOURCE_CENTERS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
ELIGIBLE_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTERS = ("4",)
SOURCES_PER_TARGET = 8
EXPECTED_PROFILE_COUNT = 9
EXPECTED_SCORE_COUNT = 72

SCORING_FAMILY = "unweighted_component_exact_match_count"
SCORING_NAMESPACE = "uniform_b_v2_metadata_exact_match_compatibility_v1"
COMPATIBILITY_DECISION = "FROZEN_AS_METADATA_PROXY_COMPATIBILITY_INPUT"
PUBLICATION_STATE = "COMPATIBILITY_LOCKED_FOR_SEPARATE_STAGE60_POLICY"

# Frozen from the canonical config and SHA-256-pinned domain mapping.
EXPECTED_CONFIG_CONTRACT_HASH = "89191838fbb3f1c8"
EXPECTED_METADATA_PROFILE_TABLE_HASH = "eee8dececd62bef8"
EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH = "aec9e0b5b09a1fe5"
EXPECTED_METADATA_PROFILE_LOCK_HASH = "de23d1c8de734503"
EXPECTED_COMPATIBILITY_LOCK_HASH = "4b46b3d157b07781"


@dataclass(frozen=True)
class MetadataProfile:
    """Sanitized routing-time values; deliberately contains no center/domain ID."""

    tumor_type: str
    lab_or_origin: str
    scanner_model: str

    def __post_init__(self) -> None:
        for axis, value in zip(ORDERED_AXES, self.values, strict=True):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ProtocolError(f"Metadata profile value is invalid for {axis}.")
            if "|" in value:
                raise ProtocolError(f"Metadata profile value contains the domain separator: {axis}.")

    @property
    def values(self) -> tuple[str, str, str]:
        return (self.tumor_type, self.lab_or_origin, self.scanner_model)

    def to_payload(self) -> dict[str, object]:
        return {
            "tumor_type": self.tumor_type,
            "lab_or_origin": self.lab_or_origin,
            "scanner_model": self.scanner_model,
        }


@dataclass(frozen=True)
class CompatibilityScore:
    """One directional, target-excluded metadata proxy score."""

    target_center: str
    source_center: str
    tumor_type_exact_match: int
    lab_or_origin_exact_match: int
    scanner_model_exact_match: int
    exact_match_count: int

    def __post_init__(self) -> None:
        if (
            self.target_center not in ELIGIBLE_CENTERS
            or self.source_center not in ELIGIBLE_CENTERS
            or self.target_center == self.source_center
        ):
            raise ProtocolError("Compatibility score must bind an eligible target-excluded pair.")
        components = (
            self.tumor_type_exact_match,
            self.lab_or_origin_exact_match,
            self.scanner_model_exact_match,
        )
        if any(type(value) is not int or value not in (0, 1) for value in components):
            raise ProtocolError("Metadata exact-match components must be integer zero or one.")
        if type(self.exact_match_count) is not int or self.exact_match_count != sum(components):
            raise ProtocolError("Metadata exact-match count must equal its unweighted components.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_score_v1",
            "target_center": self.target_center,
            "source_center": self.source_center,
            "tumor_type_exact_match": self.tumor_type_exact_match,
            "lab_or_origin_exact_match": self.lab_or_origin_exact_match,
            "scanner_model_exact_match": self.scanner_model_exact_match,
            "exact_match_count": self.exact_match_count,
            "score_minimum": 0,
            "score_maximum": 3,
            "target_expert_excluded": True,
            "proxy_only": True,
        }


@dataclass(frozen=True)
class MetadataCompatibilityLock:
    """Immutable, self-hashing metadata compatibility lock."""

    _payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self._payload, Mapping):
            raise ProtocolError("Metadata compatibility lock must be a mapping.")
        frozen = _deep_freeze(self._payload)
        if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
            raise ProtocolError("Metadata compatibility lock must be a mapping.")
        payload = _deep_thaw(frozen)
        if not isinstance(payload, dict):  # pragma: no cover - guarded above
            raise ProtocolError("Metadata compatibility lock must be an object.")
        observed = payload.get("compatibility_lock_hash")
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "compatibility_lock_hash"
        }
        expected = _expected_compatibility_lock_unhashed_payload()
        if unhashed != expected:
            raise ProtocolError("Metadata compatibility-lock semantic identity drifted.")
        if observed != stable_hash(unhashed):
            raise ProtocolError("Metadata compatibility-lock hash drifted.")
        if observed != EXPECTED_COMPATIBILITY_LOCK_HASH:
            raise ProtocolError("Metadata compatibility-lock frozen identity drifted.")
        object.__setattr__(self, "_payload", frozen)

    @property
    def compatibility_lock_hash(self) -> str:
        return str(self._payload["compatibility_lock_hash"])

    @property
    def metadata_profile_lock_hash(self) -> str:
        return str(self._payload["metadata_profile_lock_hash"])

    @property
    def metadata_profile_table_hash(self) -> str:
        return str(self._payload["metadata_profile_table_hash"])

    @property
    def compatibility_score_table_hash(self) -> str:
        return str(self._payload["compatibility_score_table_hash"])

    def to_payload(self) -> dict[str, object]:
        payload = _deep_thaw(self._payload)
        if not isinstance(payload, dict):  # pragma: no cover - construction invariant
            raise ProtocolError("Metadata compatibility lock must be an object.")
        return payload


def candidate_sources(target_center: str) -> tuple[str, ...]:
    """Return the canonical target-excluded source order."""

    rendered = str(target_center)
    if rendered not in ELIGIBLE_CENTERS:
        raise ProtocolError(f"Unknown eligible target center: {target_center!r}.")
    return tuple(center for center in ELIGIBLE_CENTERS if center != rendered)


def _expected_compatibility_lock_unhashed_payload() -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_uniform_b_v2_metadata_exact_match_compatibility_lock_v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
        "input_artifact_id": INPUT_ARTIFACT_ID,
        "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
        "metadata_profile_lock_hash": EXPECTED_METADATA_PROFILE_LOCK_HASH,
        "metadata_profile_table_hash": EXPECTED_METADATA_PROFILE_TABLE_HASH,
        "compatibility_score_table_hash": EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
        "scoring_family": SCORING_FAMILY,
        "scoring_namespace": SCORING_NAMESPACE,
        "ordered_axes": list(ORDERED_AXES),
        "component_weights": {axis: 1 for axis in ORDERED_AXES},
        "scorer_inputs": "metadata_profile_values_only",
        "center_or_domain_ids_passed_to_scorer": False,
        "directionality": "all_ordered_target_source_pairs",
        "target_expert_excluded": True,
        "eligible_target_count": len(ELIGIBLE_CENTERS),
        "sources_per_target": SOURCES_PER_TARGET,
        "ordered_score_count": EXPECTED_SCORE_COUNT,
        "minimum_score": 0,
        "maximum_score": len(ORDERED_AXES),
        "metadata_score_is_proxy_only": True,
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
        "nelbo_computed": False,
        "true_utility_computed": False,
    }


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("Metadata compatibility lock keys must be strings.")
            copied[key] = _deep_freeze(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ProtocolError("Metadata compatibility lock contains a non-JSON value.")


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


OUTPUT_SEMANTIC_IDENTITIES = {
    "metadata_compatibility_contract": (
        "midogpp_uniform_b_v2_metadata_exact_match_compatibility_lock_v1"
    ),
    "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
    "metadata_profile_lock_hash": EXPECTED_METADATA_PROFILE_LOCK_HASH,
    "compatibility_lock_hash": EXPECTED_COMPATIBILITY_LOCK_HASH,
    "metadata_profile_table_hash": EXPECTED_METADATA_PROFILE_TABLE_HASH,
    "compatibility_score_table_hash": EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
}


__all__ = (
    "ALL_SOURCE_CENTERS",
    "CLAIM_SCOPE",
    "COMPATIBILITY_DECISION",
    "DOMAIN_AXIS",
    "DOMAIN_MAPPING_MEMBER",
    "DOMAIN_MAPPING_SHA256",
    "ELIGIBLE_CENTERS",
    "EXCLUDED_CENTERS",
    "EXPECTED_COMPATIBILITY_LOCK_HASH",
    "EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH",
    "EXPECTED_CONFIG_CONTRACT_HASH",
    "EXPECTED_METADATA_PROFILE_LOCK_HASH",
    "EXPECTED_METADATA_PROFILE_TABLE_HASH",
    "EXPECTED_PROFILE_COUNT",
    "EXPECTED_SCORE_COUNT",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "INPUT_ARTIFACT_ID",
    "MetadataCompatibilityLock",
    "MetadataProfile",
    "CompatibilityScore",
    "ORDERED_AXES",
    "OUTPUT_ARTIFACT_ID",
    "OUTPUT_SEMANTIC_IDENTITIES",
    "PUBLICATION_STATE",
    "SCORING_FAMILY",
    "SCORING_NAMESPACE",
    "SOURCES_PER_TARGET",
    "candidate_sources",
)
