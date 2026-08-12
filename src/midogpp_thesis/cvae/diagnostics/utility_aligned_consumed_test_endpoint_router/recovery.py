"""Composite dispatch for the endpoint router's registered retry boundaries."""

from __future__ import annotations

from pathlib import Path

from .finalization_recovery import (
    detect_complete_endpoint_router_revalidation,
    detect_feature_partition_binding_finalization_recovery,
)
from .initialization_recovery import detect_initializing_cache_identity_recovery


def detect_registered_endpoint_router_recovery(root: Path) -> bool:
    """Recognize one exact pre-label or finalization-only recovery snapshot."""

    if detect_initializing_cache_identity_recovery(root):
        return True
    if detect_feature_partition_binding_finalization_recovery(root):
        return True
    return detect_complete_endpoint_router_revalidation(root)


__all__ = ("detect_registered_endpoint_router_recovery",)
