"""Compatibility facade for frozen-router and G-proposal persistence.

The immutable nine-target bundle lives in :mod:`.frozen_router_bundle`.
Label-free G-proposal persistence lives in :mod:`.g_proposal_persistence`.
"""

from .frozen_router_bundle import (
    FULL_ROUTER_ROLE,
    FULL_ROUTER_SCHEMA,
    FrozenPrelabelRouter,
    FrozenRouterBundle,
    FullRouterReplayReceipt,
    freeze_full_prelabel_router,
    replay_full_prelabel_router,
)
from .g_proposal_persistence import FrozenGProposal


__all__ = (
    "FULL_ROUTER_ROLE",
    "FULL_ROUTER_SCHEMA",
    "FrozenGProposal",
    "FrozenPrelabelRouter",
    "FrozenRouterBundle",
    "FullRouterReplayReceipt",
    "freeze_full_prelabel_router",
    "replay_full_prelabel_router",
)

