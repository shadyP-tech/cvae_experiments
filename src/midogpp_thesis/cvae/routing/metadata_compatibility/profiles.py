"""Narrow parsing and sanitization of the hash-pinned domain mapping."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .contracts import (
    ALL_SOURCE_CENTERS,
    DOMAIN_AXIS,
    DOMAIN_MAPPING_SHA256,
    ELIGIBLE_CENTERS,
    EXCLUDED_CENTERS,
    EXPECTED_PROFILE_COUNT,
    MetadataProfile,
)


@dataclass(frozen=True)
class FrozenDomainMapping:
    """The only two fields authorized for parsing from domain_mapping.json."""

    domain_axis: str
    domain_name_to_id: Mapping[str, str]


def read_frozen_domain_mapping(
    path: str | Path,
    *,
    expected_sha256: str = DOMAIN_MAPPING_SHA256,
) -> FrozenDomainMapping:
    """Hash-check the file, then parse only its axis and name-to-ID mapping."""

    source = Path(path)
    if not source.is_file() or _sha256_file(source) != expected_sha256:
        raise ProtocolError("Frozen routing metadata mapping SHA-256 drifted.")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot parse frozen routing metadata mapping.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("Frozen routing metadata mapping must be an object.")

    # Deliberately do not inspect domains, counts, status, labels, samples, or splits.
    domain_axis = raw.get("domain_axis")
    raw_mapping = raw.get("domain_name_to_id")
    if domain_axis != DOMAIN_AXIS or not isinstance(raw_mapping, Mapping):
        raise ProtocolError("Frozen routing metadata fields drifted.")
    mapping: dict[str, str] = {}
    for raw_name, raw_id in raw_mapping.items():
        if not isinstance(raw_name, str) or not isinstance(raw_id, str):
            raise ProtocolError("Frozen routing metadata mapping must contain strings only.")
        if raw_name in mapping or raw_id in mapping.values():
            raise ProtocolError("Frozen routing metadata mapping contains duplicate identities.")
        mapping[raw_name] = raw_id
    if len(mapping) != len(ALL_SOURCE_CENTERS) or set(mapping.values()) != set(
        ALL_SOURCE_CENTERS
    ):
        raise ProtocolError("Frozen routing metadata center coverage drifted.")
    return FrozenDomainMapping(
        domain_axis=str(domain_axis),
        domain_name_to_id=MappingProxyType(mapping),
    )


def derive_metadata_profiles(
    source: FrozenDomainMapping | str | Path,
    *,
    expected_sha256: str = DOMAIN_MAPPING_SHA256,
) -> dict[str, MetadataProfile]:
    """Derive exactly nine ID-keyed profiles whose values contain no IDs."""

    frozen = (
        source
        if isinstance(source, FrozenDomainMapping)
        else read_frozen_domain_mapping(source, expected_sha256=expected_sha256)
    )
    if frozen.domain_axis != DOMAIN_AXIS:
        raise ProtocolError("Frozen routing metadata axis drifted.")
    profiles: dict[str, MetadataProfile] = {}
    for raw_name, center_id in frozen.domain_name_to_id.items():
        components = tuple(component.strip() for component in raw_name.split("|"))
        if len(components) != 3 or any(not component for component in components):
            raise ProtocolError("Frozen routing metadata domain name is not a three-axis profile.")
        if center_id in EXCLUDED_CENTERS:
            continue
        if center_id not in ELIGIBLE_CENTERS or center_id in profiles:
            raise ProtocolError("Frozen routing metadata eligible-center mapping drifted.")
        profiles[center_id] = MetadataProfile(*components)
    ordered = {center: profiles[center] for center in ELIGIBLE_CENTERS if center in profiles}
    if tuple(ordered) != ELIGIBLE_CENTERS or len(ordered) != EXPECTED_PROFILE_COUNT:
        raise ProtocolError("Sanitized metadata profile coverage drifted.")
    if any(center in ordered for center in EXCLUDED_CENTERS):
        raise ProtocolError("Excluded center 4 must never be emitted as a metadata profile.")
    return ordered


def metadata_profile_rows(
    profiles: Mapping[str, MetadataProfile],
) -> tuple[dict[str, object], ...]:
    _validate_profiles(profiles)
    return tuple(
        {
            "schema_version": "midogpp_uniform_b_v2_metadata_profile_v1",
            "center_id": center,
            **profiles[center].to_payload(),
        }
        for center in ELIGIBLE_CENTERS
    )


def _validate_profiles(profiles: Mapping[str, MetadataProfile]) -> None:
    if tuple(profiles) != ELIGIBLE_CENTERS or len(profiles) != EXPECTED_PROFILE_COUNT:
        raise ProtocolError("Metadata profiles must use the canonical eligible-center order.")
    if not all(isinstance(profile, MetadataProfile) for profile in profiles.values()):
        raise ProtocolError("Metadata profile mapping contains a non-profile value.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "FrozenDomainMapping",
    "derive_metadata_profiles",
    "metadata_profile_rows",
    "read_frozen_domain_mapping",
)
