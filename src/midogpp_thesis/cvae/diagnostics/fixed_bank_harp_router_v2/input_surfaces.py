"""HARP v2 cache identity over the shared role-pure input reader."""

from __future__ import annotations

from ..fixed_bank_harp_router_v1.input_surfaces import (
    CACHE_INDEX,
    CACHE_ROWS,
    CONTENT_INDEX,
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpCacheRow,
    HarpConsumedCacheIdentity,
    HarpConsumedCacheIndex,
    _read_label_manifest,
    load_development_labels,
    load_evaluation_truth,
)
from .config import HarpStage90V2Config


V2_CACHE_IDENTITY = HarpConsumedCacheIdentity(
    artifact_id="midogpp_stage90_harp_consumed_test_cache_v2",
    cache_schema="midogpp_harp_consumed_test_label_blind_frame_cache_v2",
    row_schema="midogpp_harp_consumed_test_frame_row_v2",
    content_schema="midogpp_harp_consumed_test_content_index_v2",
)


def load_cache_index(config: HarpStage90V2Config) -> HarpConsumedCacheIndex:
    from ..fixed_bank_harp_router_v1.input_surfaces import load_cache_index as shared_load

    return shared_load(config, cache_identity=V2_CACHE_IDENTITY)


__all__ = (
    "CACHE_INDEX",
    "CACHE_ROWS",
    "CONTENT_INDEX",
    "DEVELOPMENT_ROLE",
    "EVALUATION_ROLE",
    "HarpCacheRow",
    "HarpConsumedCacheIndex",
    "V2_CACHE_IDENTITY",
    "load_cache_index",
    "load_development_labels",
    "load_evaluation_truth",
)
