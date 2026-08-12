"""Thin façade for reconstructive validation of persisted router science.

Each phase lives in a dedicated module.  This façade preserves the original
public entrypoint and typed phase APIs while keeping orchestration explicit.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .validation_science_common import decode_hashed_row as _decode_hashed_row
from .validation_science_contracts import (
    DevelopmentScienceValidation,
    FeatureScienceValidation,
    PrelabelScienceValidation,
    ScientificPartitionContext,
    TerminalScienceValidation,
)
from .validation_science_development import (
    validate_development_science,
    validate_partition_context,
)
from .validation_science_features import (
    feature_row as _feature_row,
    validate_feature_science,
)
from .validation_science_prelabel import validate_prelabel_science
from .validation_science_terminal import (
    validate_logical_endpoint_aliases as _validate_logical_endpoint_aliases,
    validate_terminal_science,
)


def validate_scientific_surfaces(root: str | Path) -> Mapping[str, object]:
    """Replay all persisted scientific surfaces under one narrow entrypoint."""

    base = Path(root)
    partitions = validate_partition_context(base)
    development = validate_development_science(base, partitions)
    features = validate_feature_science(base, partitions)
    prelabel = validate_prelabel_science(
        base,
        partitions=partitions,
        development=development,
        features=features,
    )
    terminal = validate_terminal_science(
        base,
        partitions=partitions,
        prelabel=prelabel,
    )
    return MappingProxyType(
        {
            "development_response_count": development.response_count,
            "development_response_set_hash": development.response_set_hash,
            "source_feature_count": features.source_feature_count,
            "target_feature_count": features.target_feature_count,
            "source_feature_surface_set_hash": features.source_surface_set_hash,
            "model_set_hash": prelabel.model_set_hash,
            "policy_set_hash": prelabel.policy_set_hash,
            "action_library_hash": prelabel.action_library_hash,
            "terminal_score_count": terminal.score_count,
            "terminal_score_set_hash": terminal.score_set_hash,
            "terminal_contrast_count": terminal.contrast_count,
            "terminal_inference_hash": terminal.inference_hash,
        }
    )


__all__ = (
    "DevelopmentScienceValidation",
    "FeatureScienceValidation",
    "PrelabelScienceValidation",
    "ScientificPartitionContext",
    "TerminalScienceValidation",
    "validate_development_science",
    "validate_feature_science",
    "validate_partition_context",
    "validate_prelabel_science",
    "validate_scientific_surfaces",
    "validate_terminal_science",
)
