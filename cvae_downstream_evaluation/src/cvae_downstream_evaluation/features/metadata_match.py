"""Metadata-match feature helpers."""

from __future__ import annotations

from typing import Mapping

from . import assert_no_target_identity_features


def metadata_match_features(features: Mapping[str, float]) -> dict[str, float]:
    assert_no_target_identity_features(tuple(features.keys()))
    return {str(key): float(value) for key, value in features.items()}
