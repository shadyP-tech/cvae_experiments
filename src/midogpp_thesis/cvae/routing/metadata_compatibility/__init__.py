"""Fail-closed MIDOG++ metadata compatibility proxy artifact."""

from .config import MetadataCompatibilityConfig, load_metadata_compatibility_config
from .contracts import (
    CompatibilityScore,
    MetadataCompatibilityLock,
    MetadataProfile,
)
from .locks import read_compatibility_lock, read_metadata_profile_lock
from .profiles import derive_metadata_profiles, read_frozen_domain_mapping
from .runner import run_metadata_compatibility_lock
from .scoring import (
    compatibility_score_table_hash,
    derive_compatibility_scores,
    metadata_profile_table_hash,
    score_profile_values,
)
from .table_io import (
    read_compatibility_scores_table,
    read_metadata_profiles_table,
)
from .validation import (
    validate_metadata_compatibility_bundle,
    validate_metadata_compatibility_provenance,
)


__all__ = (
    "CompatibilityScore",
    "MetadataCompatibilityConfig",
    "MetadataCompatibilityLock",
    "MetadataProfile",
    "compatibility_score_table_hash",
    "derive_compatibility_scores",
    "derive_metadata_profiles",
    "load_metadata_compatibility_config",
    "metadata_profile_table_hash",
    "read_compatibility_lock",
    "read_compatibility_scores_table",
    "read_frozen_domain_mapping",
    "read_metadata_profile_lock",
    "read_metadata_profiles_table",
    "run_metadata_compatibility_lock",
    "score_profile_values",
    "validate_metadata_compatibility_bundle",
    "validate_metadata_compatibility_provenance",
)
