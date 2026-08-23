"""Terminal opening evaluation after a durable P-DCAPS preterminal seal."""

from __future__ import annotations

from typing import Sequence

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
)
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    TerminalLabelCapability,
)
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    SealedActionSurfaceSet,
)
from ...fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.method_controls import (
    ComposedAdmissionControlledPrediction,
)
from ..identity import canonical_hash, require_sha256
from .contracts import TerminalEvaluationResult
from .diagnostics import build_router_diagnostics
from .inference import exact_shared_center_max_sign_flip
from .scoring import score_composed_methods


def evaluate_terminal(
    *,
    identity_results: Sequence[OuterActionPolicyResult],
    surface_set: SealedActionSurfaceSet,
    compositions: Sequence[ComposedAdmissionControlledPrediction],
    capabilities: Sequence[TerminalLabelCapability],
    preterminal_seal_hash: str,
) -> TerminalEvaluationResult:
    """Score target labels exactly once; return only aggregates and identities."""

    seal_hash = require_sha256(preterminal_seal_hash, "preterminal seal")
    grants = tuple(capabilities)
    if (
        tuple(row.center for row in grants) != CENTERS
        or any(row.preterminal_seal_hash != seal_hash for row in grants)
    ):
        raise ProtocolError("P-DCAPS terminal capability seal drifted.")
    method_rows, center_rows, case_rows, center_metrics = score_composed_methods(
        compositions, grants
    )
    selection = exact_shared_center_max_sign_flip(center_metrics)
    diagnostics = build_router_diagnostics(
        identity_results=identity_results,
        surface_set=surface_set,
        compositions=compositions,
        capabilities=grants,
        center_rows=center_rows,
        case_rows=case_rows,
    )
    label_identity_hash = canonical_hash(
        {
            "schema_version": "pdcaps_v4_terminal_label_identity_v1",
            "preterminal_seal_hash": seal_hash,
            "center_capabilities": tuple(
                (row.center, row.key_order_hash, row.capability_hash) for row in grants
            ),
            "raw_labels_persisted": False,
        }
    )
    return TerminalEvaluationResult(
        method_rows,
        center_rows,
        case_rows,
        selection,
        diagnostics,
        seal_hash,
        label_identity_hash,
    )


__all__ = ("evaluate_terminal",)
