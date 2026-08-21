"""Thin phase coordinator for the terminal CBPUPR diagnostic."""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidate_orchestration import SealedCandidateProducts, build_sealed_candidates
from .constants import CENTERS, PRIMARY_FINGERPRINT_CONTROL_ID, PRIMARY_METHOD_ID
from .decision_orchestration import SealedDecisionProducts, build_sealed_decisions
from .hashing import canonical_hash
from .terminal_diagnostics import GateFunnel, build_gate_funnel


@dataclass(frozen=True)
class PreterminalResult:
    candidates: SealedCandidateProducts
    decisions: SealedDecisionProducts
    gate_funnel: GateFunnel
    preterminal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preterminal_hash",
            canonical_hash(
                {
                    "schema_version": "fixed_bank_cbpupr_preterminal_result_v1",
                    "candidate_seal_hash": self.candidates.target_candidate_seal_hash,
                    "pre_evaluation_seal_hash": self.candidates.pre_evaluation_seal_hash,
                    "replay_calibration_seal_hash": (
                        self.decisions.replay_calibration_seal_hash
                    ),
                    "aggregate_seal_hash": self.decisions.aggregate_seal_hash,
                    "gate_funnel_hash": self.gate_funnel.funnel_hash,
                    "target_label_used": False,
                }
            ),
        )

    def diagnostic_summary(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_diagnostic_summary_v1",
            "preterminal_hash": self.preterminal_hash,
            "outer_route_count": len(self.candidates.plan_seal.outer_plans),
            "target_posterior_model_fit_count": len(
                self.candidates.posterior_models
            ),
            "pseudo_posterior_model_fit_count": 0,
            "pseudo_posterior_reference_count": len(
                self.candidates.pseudo_posterior_references
            ),
            "target_candidate_runtime_count": len(
                self.candidates.target_candidates
            ),
            "pseudo_candidate_runtime_count": len(
                self.candidates.pseudo_candidates
            ),
            "donor_replay_count": len(self.decisions.donor_replays),
            "policy_replay_count": len(self.decisions.policy_replays),
            "gate_funnel": self.gate_funnel.to_payload(),
            "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
            "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
            "formal_claim_authorized": False,
            "may_feed_another_experiment": False,
            "all_fitted_DTO_outputs_replayed_during_validation": True,
            "optimizer_refit_during_bundle_validation": False,
            "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
        }


def build_preterminal_result(
    surface: object,
    label_loader: object,
    *,
    use_processes: bool = True,
) -> PreterminalResult:
    candidates = build_sealed_candidates(
        surface, label_loader, use_processes=use_processes
    )
    decisions = build_sealed_decisions(candidates)
    primary_candidates = tuple(
        row
        for row in candidates.target_candidates
        if row.control_id == PRIMARY_FINGERPRINT_CONTROL_ID
    )
    primary_decisions = tuple(
        row
        for row in decisions.route_decisions
        if row.method_id == PRIMARY_METHOD_ID and row.center in CENTERS
    )
    funnel = build_gate_funnel(primary_candidates, primary_decisions)
    return PreterminalResult(candidates, decisions, funnel)


__all__ = ("PreterminalResult", "build_preterminal_result")
