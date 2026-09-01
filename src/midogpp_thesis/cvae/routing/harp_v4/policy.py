"""Deterministic case-level B/U/Hxe hierarchical policy for HARP v4."""

from __future__ import annotations

from dataclasses import dataclass

from ...protocol import ProtocolError
from .contracts import ActionKind, CaseTargetAction, Comparison, PolicyConfig
from .fitting import HarpV4Fit
from .scoring import ConservativeScore, score_comparison


@dataclass(frozen=True, kw_only=True)
class CaseActionSet:
    baseline: CaseTargetAction
    uniform: CaseTargetAction
    experts: tuple[CaseTargetAction, ...]
    expected_candidate_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.baseline, CaseTargetAction)
            or not isinstance(self.uniform, CaseTargetAction)
            or self.baseline.action_kind is not ActionKind.B
            or self.uniform.action_kind is not ActionKind.U
            or not self.experts
            or any(
                not isinstance(value, CaseTargetAction)
                or value.action_kind is not ActionKind.HXE
                for value in self.experts
            )
        ):
            raise ProtocolError("A HARP v4 case requires B, U, and physical Hxe actions.")
        actions = (self.baseline, self.uniform, *self.experts)
        reference = self.baseline
        if any(
            action.case_key != reference.case_key
            or action.sample_ids != reference.sample_ids
            or action.feature_names != reference.feature_names
            or action.prediction_seal_hash != reference.prediction_seal_hash
            for action in actions
        ):
            raise ProtocolError("HARP v4 case actions drifted in case, samples, schema, or seal.")
        source_ids = tuple(action.candidate_source_id for action in self.experts)
        expected = tuple(str(value) for value in self.expected_candidate_source_ids)
        if (
            expected != tuple(sorted(set(expected)))
            or not expected
            or reference.outer_target_id in expected
            or source_ids != expected
        ):
            raise ProtocolError(
                "HARP v4 requires the complete sealed physical candidate universe."
            )


@dataclass(frozen=True)
class ActionAudit:
    action_id: str
    action_kind: ActionKind
    candidate_source_id: str | None
    comparison_scores: tuple[ConservativeScore, ...]
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.action_id
            or self.action_kind is ActionKind.B
            and self.comparison_scores
            or self.action_kind is ActionKind.U
            and tuple(score.comparison for score in self.comparison_scores)
            != (Comparison.U_VS_B,)
            or self.action_kind is ActionKind.HXE
            and tuple(score.comparison for score in self.comparison_scores)
            != (Comparison.HXE_VS_B, Comparison.HXE_VS_U)
            or self.eligible != (not self.rejection_reasons)
        ):
            raise ProtocolError("HARP v4 per-action audit is malformed.")


@dataclass(frozen=True)
class CaseRoutingDecision:
    outer_target_id: str
    case_id: str
    sample_ids: tuple[str, ...]
    baseline_probability_bytes: tuple[bytes, ...]
    output_probability_bytes: tuple[bytes, ...]
    selected_kind: ActionKind
    selected_source_id: str | None
    reason: str
    prediction_seal_hash: str
    action_audits: tuple[ActionAudit, ...]

    def __post_init__(self) -> None:
        if (
            not self.outer_target_id
            or not self.case_id
            or not self.sample_ids
            or len(self.sample_ids) != len(self.baseline_probability_bytes)
            or len(self.sample_ids) != len(self.output_probability_bytes)
            or not self.reason
            or not self.prediction_seal_hash
            or not self.action_audits
        ):
            raise ProtocolError("HARP v4 case decision is malformed.")
        object.__setattr__(self, "selected_kind", ActionKind(self.selected_kind))
        if self.selected_kind is ActionKind.B:
            if (
                self.selected_source_id is not None
                or self.output_probability_bytes != self.baseline_probability_bytes
            ):
                raise ProtocolError("Exact-B fallback must retain byte-identical probabilities.")
        elif self.selected_kind is ActionKind.U:
            if self.selected_source_id is not None:
                raise ProtocolError("A U decision cannot name an expert source.")
        elif not self.selected_source_id:
            raise ProtocolError("A physical Hxe decision must name its expert source.")

    @property
    def routed_to_expert(self) -> bool:
        return self.selected_kind is ActionKind.HXE


def _audit_for_u(score: ConservativeScore) -> ActionAudit:
    return ActionAudit(
        action_id="U",
        action_kind=ActionKind.U,
        candidate_source_id=None,
        comparison_scores=(score,),
        eligible=score.eligible,
        rejection_reasons=score.rejection_reasons,
    )


def _audit_for_expert(
    action: CaseTargetAction,
    versus_b: ConservativeScore,
    versus_u: ConservativeScore,
) -> ActionAudit:
    reasons = tuple(
        f"{score.comparison.value}:{reason}"
        for score in (versus_b, versus_u)
        for reason in score.rejection_reasons
    )
    return ActionAudit(
        action_id=action.action_id,
        action_kind=ActionKind.HXE,
        candidate_source_id=action.candidate_source_id,
        comparison_scores=(versus_b, versus_u),
        eligible=not reasons,
        rejection_reasons=reasons,
    )


def route_case(
    actions: CaseActionSet,
    fit: HarpV4Fit,
    *,
    config: PolicyConfig = PolicyConfig(),
) -> CaseRoutingDecision:
    if not isinstance(actions, CaseActionSet) or not isinstance(fit, HarpV4Fit):
        raise ProtocolError("HARP v4 routing requires typed case actions and fit.")
    if actions.baseline.outer_target_id != fit.outer_target_id:
        raise ProtocolError("HARP v4 case escaped its outer-target fit.")
    baseline_audit = ActionAudit("B", ActionKind.B, None, (), True, ())
    uniform_score = score_comparison(
        fit, actions.uniform, Comparison.U_VS_B, config=config
    )
    uniform_audit = _audit_for_u(uniform_score)
    expert_rows: list[
        tuple[CaseTargetAction, ConservativeScore, ConservativeScore, ActionAudit]
    ] = []
    for action in actions.experts:
        versus_b = score_comparison(
            fit, action, Comparison.HXE_VS_B, config=config
        )
        versus_u = score_comparison(
            fit, action, Comparison.HXE_VS_U, config=config
        )
        expert_rows.append(
            (action, versus_b, versus_u, _audit_for_expert(action, versus_b, versus_u))
        )
    admitted = tuple(value for value in expert_rows if value[3].eligible)
    audits = (baseline_audit, uniform_audit, *(value[3] for value in expert_rows))
    if admitted:
        action, versus_b, versus_u, _ = min(
            admitted,
            key=lambda value: (
                -min(
                    value[
                        1
                    ].geometry_adjusted_bounds.case_equal_bacc_contribution_gain_lower,
                    value[
                        2
                    ].geometry_adjusted_bounds.case_equal_bacc_contribution_gain_lower,
                ),
                max(
                    value[1].geometry_adjusted_bounds.brier_upper,
                    value[2].geometry_adjusted_bounds.brier_upper,
                ),
                max(
                    value[1].geometry_adjusted_bounds.log_loss_upper,
                    value[2].geometry_adjusted_bounds.log_loss_upper,
                ),
                value[0].candidate_source_id or "",
            ),
        )
        return CaseRoutingDecision(
            outer_target_id=action.outer_target_id,
            case_id=action.case_id,
            sample_ids=action.sample_ids,
            baseline_probability_bytes=actions.baseline.probability_bytes,
            output_probability_bytes=action.probability_bytes,
            selected_kind=ActionKind.HXE,
            selected_source_id=action.candidate_source_id,
            reason="PHYSICAL_EXPERT_SAFE_VS_B_AND_U",
            prediction_seal_hash=action.prediction_seal_hash,
            action_audits=audits,
        )
    if uniform_audit.eligible:
        return CaseRoutingDecision(
            outer_target_id=actions.uniform.outer_target_id,
            case_id=actions.uniform.case_id,
            sample_ids=actions.uniform.sample_ids,
            baseline_probability_bytes=actions.baseline.probability_bytes,
            output_probability_bytes=actions.uniform.probability_bytes,
            selected_kind=ActionKind.U,
            selected_source_id=None,
            reason="UNIFORM_SAFE_VS_B",
            prediction_seal_hash=actions.uniform.prediction_seal_hash,
            action_audits=audits,
        )
    return CaseRoutingDecision(
        outer_target_id=actions.baseline.outer_target_id,
        case_id=actions.baseline.case_id,
        sample_ids=actions.baseline.sample_ids,
        baseline_probability_bytes=actions.baseline.probability_bytes,
        output_probability_bytes=actions.baseline.probability_bytes,
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="EXACT_B_FALLBACK_NO_HIERARCHICALLY_SAFE_ACTION",
        prediction_seal_hash=actions.baseline.prediction_seal_hash,
        action_audits=audits,
    )


__all__ = (
    "ActionAudit",
    "CaseActionSet",
    "CaseRoutingDecision",
    "route_case",
)
