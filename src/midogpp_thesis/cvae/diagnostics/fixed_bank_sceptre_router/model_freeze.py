"""Compatibility imports for SCEPTRE freeze and replay contracts.

Target-specific adaptive-model persistence lives in
:mod:`.adaptive_model_freeze`; complete nine-target router and G-proposal
persistence lives in :mod:`.router_bundle_freeze`.
"""

from .adaptive_model_freeze import (
    AdaptiveUtilityDecision,
    AdaptiveUtilityExactBFallback,
    AdaptiveUtilityRanking,
    AdaptiveUtilityRoute,
    EXACT_UTILITY_TIE_REASON,
    FROZEN_MODEL_PUBLICATION_STATUS,
    FROZEN_MODEL_ROLE,
    FROZEN_MODEL_SCHEMA,
    FrozenAdaptiveUtilityModel,
    FrozenModelReplayReceipt,
    INVALID_UTILITY_EVIDENCE_REASON,
    MISSING_UTILITY_EVIDENCE_REASON,
    PREDICTED_UTILITY_POLICY_ID,
    PREDICTED_UTILITY_SEMANTICS,
    freeze_adaptive_utility_model,
    replay_frozen_adaptive_utility_model,
    route_frozen_predicted_utility_or_exact_b,
)
from .router_bundle_freeze import (
    FULL_ROUTER_ROLE,
    FULL_ROUTER_SCHEMA,
    FrozenGProposal,
    FrozenPrelabelRouter,
    FrozenRouterBundle,
    FullRouterReplayReceipt,
    freeze_full_prelabel_router,
    replay_full_prelabel_router,
)


__all__ = (
    "AdaptiveUtilityDecision",
    "AdaptiveUtilityExactBFallback",
    "AdaptiveUtilityRanking",
    "AdaptiveUtilityRoute",
    "EXACT_UTILITY_TIE_REASON",
    "FROZEN_MODEL_PUBLICATION_STATUS",
    "FROZEN_MODEL_ROLE",
    "FROZEN_MODEL_SCHEMA",
    "FULL_ROUTER_ROLE",
    "FULL_ROUTER_SCHEMA",
    "FrozenAdaptiveUtilityModel",
    "FrozenGProposal",
    "FrozenModelReplayReceipt",
    "FrozenPrelabelRouter",
    "FrozenRouterBundle",
    "FullRouterReplayReceipt",
    "INVALID_UTILITY_EVIDENCE_REASON",
    "MISSING_UTILITY_EVIDENCE_REASON",
    "PREDICTED_UTILITY_POLICY_ID",
    "PREDICTED_UTILITY_SEMANTICS",
    "freeze_adaptive_utility_model",
    "freeze_full_prelabel_router",
    "replay_frozen_adaptive_utility_model",
    "replay_full_prelabel_router",
    "route_frozen_predicted_utility_or_exact_b",
)

