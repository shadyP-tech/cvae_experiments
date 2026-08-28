"""Typed partition, decision, seal, and route-policy reconstruction."""

from __future__ import annotations

from typing import Mapping

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS

from ....protocol import ProtocolError
from ...fixed_bank_sceptre_router.hashing import canonical_hash
from ...fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    ThreeRolePartition,
    build_three_role_partition,
)
from ...fixed_bank_sceptre_router.seals import (
    EXPECTED_DECISION_KEYS,
    build_global_decision_seal,
)
from ..confirmation_gate import ConfirmationDecision
from ..phase_manager import ProposalSetFoldReceipt
from ..posterior import PairedCandidatePosterior
from ..proposal_set import FrozenCandidateSetProposal
from ..route_policy import FrozenRoutePolicy
from ..support_posterior import SupportPosteriorDecision
from .validation_journal import validate_label_journal, validate_preterminal_journal


def reconstruct_partition(
    payload: Mapping[str, object], policy: FrozenRoutePolicy
) -> ThreeRolePartition:
    """Rebuild the whole-case three-role split and bind it to the policy."""

    raw_identities = payload.get("identities")
    raw_folds = payload.get("folds")
    if not isinstance(raw_identities, list) or not isinstance(raw_folds, list):
        raise ProtocolError("SCEPTRE v4 persisted partition is malformed.")
    try:
        identities = tuple(
            CaseIdentity(
                str(row["target_center"]),
                str(row["case_id"]),
                str(row["sample_id"]),
            )
            for row in raw_identities
            if isinstance(row, Mapping)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE v4 persisted identities drifted.") from exc
    if len(identities) != len(raw_identities):
        raise ProtocolError("SCEPTRE v4 persisted identities are incomplete.")
    replay = build_three_role_partition(identities, expected_total_case_count=218)
    expected_folds = [
        {
            "target_center": fold.target_center,
            "fold_ordinal": fold.fold_ordinal,
            "selection_case_ids": list(fold.selection_case_ids),
            "calibration_case_ids": list(fold.calibration_case_ids),
            "evaluation_case_ids": list(fold.evaluation_case_ids),
            "fold_hash": fold.fold_hash,
        }
        for fold in replay.folds
    ]
    if (
        payload.get("partition_hash") != replay.partition_hash
        or payload.get("partition_seed") != replay.partition_seed
        or raw_folds != expected_folds
        or policy.partition_hash != replay.partition_hash
        or payload.get("whole_case_roles_disjoint") is not True
        or payload.get("evaluation_cases_exactly_once") is not True
    ):
        raise ProtocolError("SCEPTRE v4 persisted partition does not replay.")
    return replay


def validate_decision_graph(
    *,
    index: Mapping[str, object],
    bundle: Mapping[str, object],
    development: Mapping[str, object],
    phases: Mapping[str, object],
    journal: Mapping[str, object],
    policy: FrozenRoutePolicy,
    partition: ThreeRolePartition,
) -> None:
    """Rehydrate every decision DTO and replay its global seal lineage."""

    events = journal.get("events")
    if (
        bundle["partition"].get("case_count") != 218
        or len(bundle["partition"].get("folds", ())) != 45
        or len(bundle.get("proposal_sets", ())) != 9
        or len(bundle.get("support_decisions", ())) != 45
        or len(bundle.get("confirmation_decisions", ())) != 45
        or phases.get("route_policy_hash") != policy.policy_artifact_hash
        or phases.get("phase_hash")
        != canonical_hash({key: value for key, value in phases.items() if key != "phase_hash"})
        or index.get("route_policy_hash") != policy.policy_artifact_hash
        or index.get("policy_seal_hash") != policy.policy_seal_hash
        or policy.routing_context_hash != development.get("routing_context_hash")
        or not isinstance(events, list)
        or any(
            not isinstance(row, Mapping)
            or row.get("raw_labels_persisted") is not False
            or str(row.get("event", "")).startswith("EVALUATION_LABELS")
            for row in events
        )
        or journal.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("SCEPTRE v4 preterminal scientific graph drifted.")
    _validate_hashed_rows(
        bundle,
        development,
        phases,
        journal,
        policy=policy,
        partition=partition,
    )


def _validate_hashed_rows(
    bundle: Mapping[str, object],
    development: Mapping[str, object],
    phases: Mapping[str, object],
    journal: Mapping[str, object],
    *,
    policy: FrozenRoutePolicy,
    partition: ThreeRolePartition,
) -> None:
    context = bundle.get("routing_context")
    proposals = bundle.get("proposal_sets")
    support = bundle.get("support_decisions")
    posteriors = bundle.get("calibration_posteriors")
    confirmations = bundle.get("confirmation_decisions")
    if (
        not isinstance(context, Mapping)
        or not isinstance(proposals, list)
        or not isinstance(support, list)
        or not isinstance(posteriors, list)
        or not isinstance(confirmations, list)
    ):
        raise ProtocolError("SCEPTRE v4 persisted decision rows are malformed.")
    _require_payload_hash(context, "context_hash")
    _require_payload_hash(development, "replay_hash")
    try:
        typed_proposals = tuple(
            FrozenCandidateSetProposal.from_payload(row)
            for row in proposals
            if isinstance(row, Mapping)
        )
        typed_support = tuple(
            SupportPosteriorDecision.from_payload(row)
            for row in support
            if isinstance(row, Mapping)
        )
        typed_posteriors = tuple(
            PairedCandidatePosterior.from_payload(row)
            for row in posteriors
            if isinstance(row, Mapping)
        )
        typed_confirmations = tuple(
            ConfirmationDecision.from_payload(row)
            for row in confirmations
            if isinstance(row, Mapping)
        )
    except ProtocolError:
        raise
    if (
        len(typed_proposals) != len(proposals)
        or len(typed_support) != len(support)
        or len(typed_posteriors) != len(posteriors)
        or len(typed_confirmations) != len(confirmations)
    ):
        raise ProtocolError("SCEPTRE v4 persisted typed decisions are incomplete.")

    proposal_by_target = {
        row.target_center: row.proposal_set_hash for row in typed_proposals
    }
    if tuple(proposal_by_target) != CENTERS:
        raise ProtocolError("SCEPTRE v4 persisted proposals drifted.")
    proposal_receipts = {
        key: ProposalSetFoldReceipt(
            target_center=key[0],
            fold_ordinal=key[1],
            partition_hash=str(bundle["partition"]["partition_hash"]),
            routing_context_hash=str(context["context_hash"]),
            proposal_set_hash=proposal_by_target[key[0]],
        ).receipt_hash
        for key in EXPECTED_DECISION_KEYS
    }
    proposal_seal = build_global_decision_seal(
        "G_RANKED_CANDIDATE_SETS_LABEL_FREE", proposal_receipts
    )
    support_by_key = {
        (row.target_center, row.fold_ordinal): row for row in typed_support
    }
    confirmation_by_key = {
        (row.target_center, row.fold_ordinal): row for row in typed_confirmations
    }
    posterior_by_key = {
        (row.target_center, row.fold_ordinal): row for row in typed_posteriors
    }
    if (
        tuple(support_by_key) != EXPECTED_DECISION_KEYS
        or tuple(confirmation_by_key) != EXPECTED_DECISION_KEYS
        or set(posterior_by_key)
        != {
            key
            for key, row in support_by_key.items()
            if row.selected_candidate is not None
        }
    ):
        raise ProtocolError("SCEPTRE v4 persisted decision coverage drifted.")
    for key in EXPECTED_DECISION_KEYS:
        target, fold_ordinal = key
        fold = partition.fold(target, fold_ordinal)
        proposal = next(row for row in typed_proposals if row.target_center == target)
        support_row = support_by_key[key]
        confirmation = confirmation_by_key[key]
        posterior = posterior_by_key.get(key)
        if (
            support_row.fold_hash != fold.fold_hash
            or support_row.partition_hash != partition.partition_hash
            or support_row.selection_case_set_hash != fold.case_set_hash("SELECTION")
            or support_row.calibration_case_set_hash
            != fold.case_set_hash("CALIBRATION")
            or support_row.evaluation_case_set_hash
            != fold.case_set_hash("EVALUATION")
            or support_row.routing_context_hash != context.get("context_hash")
            or support_row.proposal_set_hash != proposal.proposal_set_hash
            or support_row.candidate_menu_hash != proposal.candidate_menu_hash
            or support_row.exact_b_control_receipt_hash
            != proposal.exact_b_control_receipt_hash
            or confirmation.fold_hash != support_row.fold_hash
            or confirmation.partition_hash != support_row.partition_hash
            or confirmation.selection_case_set_hash
            != support_row.selection_case_set_hash
            or confirmation.calibration_case_set_hash
            != support_row.calibration_case_set_hash
            or confirmation.evaluation_case_set_hash
            != support_row.evaluation_case_set_hash
            or confirmation.routing_context_hash
            != support_row.routing_context_hash
            or confirmation.proposal_set_hash != support_row.proposal_set_hash
            or confirmation.support_decision_hash != support_row.decision_hash
            or confirmation.candidate_menu_hash != support_row.candidate_menu_hash
            or confirmation.exact_b_control_receipt_hash
            != support_row.exact_b_control_receipt_hash
            or confirmation.support_selected_candidate
            != support_row.selected_candidate
            or (posterior is None and confirmation.posterior_hash is not None)
            or (
                posterior is not None
                and (
                    posterior.fold_hash != support_row.fold_hash
                    or posterior.partition_hash != support_row.partition_hash
                    or posterior.calibration_case_set_hash
                    != support_row.calibration_case_set_hash
                    or posterior.routing_context_hash
                    != support_row.routing_context_hash
                    or posterior.proposal_set_hash != support_row.proposal_set_hash
                    or posterior.support_decision_hash != support_row.decision_hash
                    or posterior.candidate_center != support_row.selected_candidate
                    or confirmation.posterior_hash != posterior.posterior_hash
                )
            )
        ):
            raise ProtocolError("SCEPTRE v4 persisted decision lineage drifted.")

    support_hashes = {key: row.decision_hash for key, row in support_by_key.items()}
    support_seal = build_global_decision_seal(
        "S_Y_SUPPORT_SELECTED_MEMBER_OR_EXACT_B",
        support_hashes,
        predecessor_seal_hash=proposal_seal.seal_hash,
    )
    confirmation_hashes = {
        key: row.decision_hash for key, row in confirmation_by_key.items()
    }
    policy_seal = build_global_decision_seal(
        "A_CONFIRM_SAME_SUPPORT_MEMBER_OR_EXACT_B",
        confirmation_hashes,
        predecessor_seal_hash=support_seal.seal_hash,
    )
    validate_label_journal(journal)
    validate_preterminal_journal(journal, support_by_key)
    expected_route_rows = tuple(
        (
            key[0],
            key[1],
            proposal_by_target[key[0]],
            support_by_key[key].decision_hash,
            support_by_key[key].selected_candidate,
            confirmation_by_key[key].route,
            confirmation_by_key[key].decision_hash,
        )
        for key in EXPECTED_DECISION_KEYS
    )
    if (
        development.get("routing_context_hash") != context.get("context_hash")
        or development.get("proposal_set_sha256_by_target")
        != [[target, proposal_by_target[target]] for target in CENTERS]
        or phases.get("proposal_set_seal_hash") != proposal_seal.seal_hash
        or phases.get("support_seal_hash") != support_seal.seal_hash
        or phases.get("policy_seal_hash") != policy_seal.seal_hash
        or phases.get("support_decision_hashes")
        != [support_hashes[key] for key in EXPECTED_DECISION_KEYS]
        or phases.get("calibration_posterior_hashes")
        != [row.posterior_hash for row in typed_posteriors]
        or phases.get("confirmation_decision_hashes")
        != [confirmation_hashes[key] for key in EXPECTED_DECISION_KEYS]
        or phases.get("label_journal_hash") != journal.get("journal_hash")
        or policy.partition_hash != partition.partition_hash
        or policy.routing_context_hash != context.get("context_hash")
        or policy.proposal_set_seal_hash != proposal_seal.seal_hash
        or policy.support_seal_hash != support_seal.seal_hash
        or policy.policy_seal_hash != policy_seal.seal_hash
        or policy.route_rows != expected_route_rows
    ):
        raise ProtocolError("SCEPTRE v4 persisted decision seals drifted.")


def _require_payload_hash(payload: Mapping[str, object], key: str) -> None:
    body = {name: value for name, value in payload.items() if name != key}
    if payload.get(key) != canonical_hash(body):
        raise ProtocolError(f"SCEPTRE v4 persisted {key} drifted.")


def _decision_hashes_by_key(
    rows: list[object], hash_key: str
) -> dict[tuple[str, int], str]:
    """Retain the exact inventory helper for focused forensic checks."""

    try:
        result = {
            (str(row["target_center"]), int(row["fold_ordinal"])): str(row[hash_key])
            for row in rows
            if isinstance(row, Mapping)
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("SCEPTRE v4 persisted decision keys drifted.") from exc
    if set(result) != set(EXPECTED_DECISION_KEYS) or len(result) != len(rows):
        raise ProtocolError("SCEPTRE v4 persisted decision coverage drifted.")
    return result


__all__ = ("reconstruct_partition", "validate_decision_graph")
