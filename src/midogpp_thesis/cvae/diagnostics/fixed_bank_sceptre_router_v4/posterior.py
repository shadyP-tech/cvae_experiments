"""V4-owned adapter for paired whole-case Bayesian bootstrap evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.uncertainty import (
    ActionUncertaintySummary,
    DirichletBootstrapConfig,
    RolePredictionSurface,
    paired_dirichlet_route_decision,
)
from .support_posterior import SupportPosteriorDecision


@dataclass(frozen=True, slots=True)
class PairedCandidatePosterior:
    """Calibration evidence for the exact support-selected candidate."""

    target_center: str
    fold_ordinal: int
    fold_hash: str
    partition_hash: str
    calibration_case_set_hash: str
    routing_context_hash: str
    proposal_set_hash: str
    support_decision_hash: str
    candidate_center: str
    prediction_surface_hash: str
    bootstrap_config_hash: str
    shared_weight_draw_hash: str
    action_summaries: tuple[ActionUncertaintySummary, ...]
    candidate_summary_hash: str
    joint_acceptance_probability: float
    posterior_hash: str = ""

    def __post_init__(self) -> None:
        target = str(self.target_center)
        candidate = str(self.candidate_center)
        if candidate not in legal_routing_sources(target):
            raise ProtocolError("SCEPTRE v4 posterior candidate is illegal.")
        rows = tuple(self.action_summaries)
        if not rows or any(
            not isinstance(row, ActionUncertaintySummary) for row in rows
        ):
            raise ProtocolError("SCEPTRE v4 posterior summaries are invalid.")
        try:
            selected = next(row for row in rows if row.action_id == candidate)
        except StopIteration as exc:
            raise ProtocolError("SCEPTRE v4 posterior lost its candidate.") from exc
        if (
            selected.summary_hash != self.candidate_summary_hash
            or selected.joint_acceptance_probability
            != float(self.joint_acceptance_probability)
        ):
            raise ProtocolError("SCEPTRE v4 posterior summary binding drifted.")
        for value, role in (
            (self.fold_hash, "posterior fold"),
            (self.partition_hash, "posterior partition"),
            (self.calibration_case_set_hash, "posterior calibration cases"),
            (self.routing_context_hash, "posterior routing context"),
            (self.proposal_set_hash, "posterior proposal set"),
            (self.support_decision_hash, "posterior support decision"),
            (self.prediction_surface_hash, "posterior prediction surface"),
            (self.bootstrap_config_hash, "posterior bootstrap config"),
            (self.shared_weight_draw_hash, "posterior shared weights"),
            (self.candidate_summary_hash, "posterior candidate summary"),
        ):
            require_sha256(value, role)
        body = self._payload_without_hash(target, candidate, rows)
        expected = canonical_hash(body)
        if self.posterior_hash and self.posterior_hash != expected:
            raise ProtocolError("SCEPTRE v4 posterior hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_center", candidate)
        object.__setattr__(self, "action_summaries", rows)
        object.__setattr__(self, "posterior_hash", expected)

    def _payload_without_hash(
        self,
        target: str | None = None,
        candidate: str | None = None,
        rows: tuple[ActionUncertaintySummary, ...] | None = None,
    ) -> dict[str, object]:
        summaries = self.action_summaries if rows is None else rows
        return {
            "schema_version": "sceptre_v4_paired_candidate_posterior_v1",
            "target_center": self.target_center if target is None else target,
            "fold_ordinal": self.fold_ordinal,
            "fold_hash": self.fold_hash,
            "partition_hash": self.partition_hash,
            "calibration_case_set_hash": self.calibration_case_set_hash,
            "routing_context_hash": self.routing_context_hash,
            "proposal_set_hash": self.proposal_set_hash,
            "support_decision_hash": self.support_decision_hash,
            "candidate_center": (
                self.candidate_center if candidate is None else candidate
            ),
            "prediction_surface_hash": self.prediction_surface_hash,
            "bootstrap_config_hash": self.bootstrap_config_hash,
            "shared_weight_draw_hash": self.shared_weight_draw_hash,
            "action_summary_hashes": [row.summary_hash for row in summaries],
            "candidate_summary_hash": self.candidate_summary_hash,
            "joint_acceptance_probability": (
                self.joint_acceptance_probability
            ),
            "whole_case_shared_draws": True,
            "seed_cells_are_nuisance_replications": True,
            "candidate_was_support_selected_not_G_top1": True,
            "raw_labels_persisted": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._payload_without_hash(),
            "action_summaries": [row.to_payload() for row in self.action_summaries],
            "posterior_hash": self.posterior_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PairedCandidatePosterior":
        try:
            raw_summaries = payload["action_summaries"]
            if not isinstance(raw_summaries, list):
                raise TypeError("summary list")
            summaries = tuple(
                _action_summary_from_payload(row)
                for row in raw_summaries
                if isinstance(row, Mapping)
            )
            if len(summaries) != len(raw_summaries) or payload.get(
                "action_summary_hashes"
            ) != [row.summary_hash for row in summaries]:
                raise ProtocolError("SCEPTRE v4 posterior summary binding drifted.")
            value = cls(
                target_center=str(payload["target_center"]),
                fold_ordinal=int(payload["fold_ordinal"]),
                fold_hash=str(payload["fold_hash"]),
                partition_hash=str(payload["partition_hash"]),
                calibration_case_set_hash=str(payload["calibration_case_set_hash"]),
                routing_context_hash=str(payload["routing_context_hash"]),
                proposal_set_hash=str(payload["proposal_set_hash"]),
                support_decision_hash=str(payload["support_decision_hash"]),
                candidate_center=str(payload["candidate_center"]),
                prediction_surface_hash=str(payload["prediction_surface_hash"]),
                bootstrap_config_hash=str(payload["bootstrap_config_hash"]),
                shared_weight_draw_hash=str(payload["shared_weight_draw_hash"]),
                action_summaries=summaries,
                candidate_summary_hash=str(payload["candidate_summary_hash"]),
                joint_acceptance_probability=float(
                    payload["joint_acceptance_probability"]
                ),
                posterior_hash=str(payload["posterior_hash"]),
            )
        except ProtocolError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE v4 posterior payload is malformed.") from exc
        if value.to_payload() != dict(payload):
            raise ProtocolError("SCEPTRE v4 posterior payload drifted.")
        return value


def _action_summary_from_payload(
    payload: Mapping[str, object],
) -> ActionUncertaintySummary:
    try:
        value = ActionUncertaintySummary(
            action_id=str(payload["action_id"]),
            point_bacc=float(payload["point_bacc"]),
            point_brier=float(payload["point_brier"]),
            point_log_loss=float(payload["point_log_loss"]),
            bootstrap_expected_bacc=float(payload["bootstrap_expected_bacc"]),
            bootstrap_expected_brier=float(payload["bootstrap_expected_brier"]),
            bootstrap_expected_log_loss=float(
                payload["bootstrap_expected_log_loss"]
            ),
            bacc_superiority_probability=float(
                payload["bacc_superiority_probability"]
            ),
            brier_noninferiority_probability=float(
                payload["brier_noninferiority_probability"]
            ),
            log_loss_noninferiority_probability=float(
                payload["log_loss_noninferiority_probability"]
            ),
            joint_acceptance_probability=float(
                payload["joint_acceptance_probability"]
            ),
            summary_hash=str(payload["summary_hash"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE v4 uncertainty summary is malformed.") from exc
    if value.to_payload() != dict(payload):
        raise ProtocolError("SCEPTRE v4 uncertainty summary drifted.")
    return value


def build_calibration_posterior(
    surface: RolePredictionSurface,
    support: SupportPosteriorDecision,
    *,
    routing_context_hash: str,
    config: DirichletBootstrapConfig,
) -> PairedCandidatePosterior:
    """Run paired case bootstrap for exactly the support-selected member."""

    if not isinstance(surface, RolePredictionSurface) or surface.role != "CALIBRATION":
        raise ProtocolError("SCEPTRE v4 posterior requires calibration surface.")
    if not isinstance(support, SupportPosteriorDecision):
        raise ProtocolError("SCEPTRE v4 posterior requires support decision.")
    selected = support.selected_candidate
    if selected is None or support.fallback_required:
        raise ProtocolError("SCEPTRE v4 support fallback has no posterior.")
    if (
        surface.target_center != support.target_center
        or surface.fold.fold_ordinal != support.fold_ordinal
        or surface.fold.fold_hash != support.fold_hash
        or surface.partition_hash != support.partition_hash
        or surface.fold.case_set_hash("CALIBRATION")
        != support.calibration_case_set_hash
        or surface.g_proposal_hash != support.proposal_set_hash
        or surface.predecessor_decision_hash != support.decision_hash
        or surface.candidate_menu_hash != support.candidate_menu_hash
        or surface.exact_b_control_receipt_hash
        != support.exact_b_control_receipt_hash
    ):
        raise ProtocolError("SCEPTRE v4 calibration surface lineage drifted.")

    # Reuse only the sealed numeric case-bootstrap kernel.  The v4 adapter
    # discards its legacy single-G route semantics and persists a v4-owned
    # support-selected posterior contract.
    numeric = paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate=selected,
        support_selected_candidate=selected,
        config=config,
    )
    summary = numeric.summaries_by_action[selected]
    return PairedCandidatePosterior(
        target_center=support.target_center,
        fold_ordinal=support.fold_ordinal,
        fold_hash=support.fold_hash,
        partition_hash=support.partition_hash,
        calibration_case_set_hash=support.calibration_case_set_hash,
        routing_context_hash=require_sha256(
            routing_context_hash, "routing context"
        ),
        proposal_set_hash=support.proposal_set_hash,
        support_decision_hash=support.decision_hash,
        candidate_center=selected,
        prediction_surface_hash=numeric.prediction_surface_hash,
        bootstrap_config_hash=numeric.bootstrap_config_hash,
        shared_weight_draw_hash=numeric.shared_weight_draw_hash,
        action_summaries=numeric.action_summaries,
        candidate_summary_hash=summary.summary_hash,
        joint_acceptance_probability=summary.joint_acceptance_probability,
    )


__all__ = ("PairedCandidatePosterior", "build_calibration_posterior")
