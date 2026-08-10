"""Fold sealing plus same-H support-only S_y selection."""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from ...protocol import ProtocolError
from .case_partitions import CaseOOFPartition
from .constants import GEOMETRY_IDS, HARD_THRESHOLD, MIDOGPP_CENTERS, U_ACTION_ID, candidate_sources, geometry_action_id
from .contracts import AggregatedProbabilityRow, BinaryPredictionRow, ExactNineProbabilitySurface, MethodDecision
from .decision_contracts import (
    DecisionProducts,
    FoldActionScore,
    FoldDecisionSeal,
    PRE_SUPPORT_GEOMETRY_METHODS,
    PreSupportDecisionProducts,
    SupportFoldProduct,
)
from .decisions import build_pre_support_decisions, build_support_static_decisions
from .execution_support import coerce_labels, decision_payload, partition_cases
from .hashing import canonical_hash
from .metrics import pooled_exact_bacc, score_case_confusions
from .model_execution import ModelProducts


def build_pre_support_decision_products(
    models: ModelProducts, partition: CaseOOFPartition
) -> PreSupportDecisionProducts:
    """Freeze B/U/G/R/P decisions and their 405 fold-level seals."""

    decisions: list[MethodDecision] = []
    for target in MIDOGPP_CENTERS:
        decisions.extend(
            build_pre_support_decisions(
                target_center=target,
                case_ids=partition_cases(partition, target),
                action_scores=tuple(row for row in models.scores if row.target_center == target),
            )
        )
    canonical_decisions = tuple(sorted(decisions))
    fold_seals: list[FoldDecisionSeal] = []
    for fold in partition.folds:
        method_contexts = (("B", None),) + tuple(
            (method, geometry)
            for geometry in GEOMETRY_IDS
            for method in PRE_SUPPORT_GEOMETRY_METHODS
        )
        eval_cases = set(fold.evaluation_case_ids)
        for method, geometry in method_contexts:
            selected = tuple(
                row for row in canonical_decisions
                if row.target_center == fold.target_center
                and row.case_id in eval_cases
                and row.method_id == method
                and row.geometry_id == geometry
            )
            if len(selected) != len(eval_cases):
                raise ProtocolError("A pre-support fold decision surface is incomplete.")
            digest = canonical_hash(
                {
                    "schema_version": "fixed_bank_actionability_fold_decision_seal_v1",
                    "fold_hash": fold.fold_hash,
                    "method_id": method,
                    "geometry_id": geometry,
                    "decisions": [decision_payload(row) for row in selected],
                    "evaluation_labels_used": False,
                }
            )
            fold_seals.append(
                FoldDecisionSeal(
                    fold.target_center, fold.fold_ordinal, method, geometry, len(selected), digest
                )
            )
    canonical_seals = tuple(sorted(fold_seals))
    hashes = MappingProxyType({row.key: row.decision_hash for row in canonical_seals})
    if len(hashes) != 405:
        raise ProtocolError("Pre-support fold seal topology must contain 405 cells.")
    seal_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_pre_support_seal_v1",
            "partition_hash": partition.partition_hash,
            "fold_seals": [row.to_payload() for row in canonical_seals],
            "model_seal_hash": models.all_models_seal_hash,
            "permutation_provenance_hash": models.permutation_provenance_hash,
            "probability_surface_hash": models.probability_surface_hash,
        }
    )
    return PreSupportDecisionProducts(
        canonical_decisions, canonical_seals, hashes, seal_hash,
        models.permutation_provenance_hash, partition.partition_hash,
        models.protocol_contract_hash, models.probability_surface_hash,
    )


def _hard_predictions(rows: Sequence[AggregatedProbabilityRow]) -> tuple[BinaryPredictionRow, ...]:
    return tuple(
        BinaryPredictionRow(
            row.target_center, row.case_id, row.sample_id, row.action_id,
            int(row.probability_mean >= HARD_THRESHOLD),
        )
        for row in rows
    )


def build_support_fold_product(
    probabilities: ExactNineProbabilitySurface,
    partition: CaseOOFPartition,
    labels: Sequence[object],
    *,
    target_center: str,
    fold_ordinal: int,
) -> SupportFoldProduct:
    """Use one exact same-H support scope to freeze both parallel S_y arms."""

    target, ordinal = str(target_center), int(fold_ordinal)
    fold = partition.fold(target, ordinal)
    scoped = coerce_labels(labels, expected_scope="target_support")
    support_cases = set(fold.support_case_ids)
    expected_keys = {
        (row.target_center, row.case_id, row.sample_id)
        for row in partition.identities
        if row.target_center == target and row.case_id in support_cases
    }
    if {row.sample_key for row in scoped} != expected_keys:
        raise ProtocolError("Support capability is not the exact whole-case fold scope.")
    if {row.label for row in scoped} != {0, 1}:
        raise ProtocolError("Support fold must contain both classes after pooling.")
    allowed_actions = {
        U_ACTION_ID,
        *(geometry_action_id(g, e) for g in GEOMETRY_IDS for e in candidate_sources(target)),
    }
    probability_rows = tuple(
        row for row in probabilities.rows
        if row.target_center == target and row.case_id in support_cases and row.action_id in allowed_actions
    )
    counts = score_case_confusions(_hard_predictions(probability_rows), scoped)
    decisions, scores, seals = [], [], []
    label_identity_hash = canonical_hash(
        [[row.target_center, row.case_id, row.sample_id, row.label] for row in scoped]
    )
    for geometry in GEOMETRY_IDS:
        actions = (
            U_ACTION_ID,
            *(geometry_action_id(geometry, source) for source in candidate_sources(target)),
        )
        geometry_counts = tuple(row for row in counts if row.action_id in actions)
        selected = build_support_static_decisions(
            target_center=target, geometry_id=geometry,
            support_counts=geometry_counts, evaluation_case_ids=fold.evaluation_case_ids,
        )
        decisions.extend(selected)
        for action in actions:
            metric = pooled_exact_bacc(tuple(row for row in geometry_counts if row.action_id == action))
            scores.append(FoldActionScore(target, ordinal, geometry, action, metric.exact_bacc))
        digest = canonical_hash(
            {
                "schema_version": "fixed_bank_actionability_support_decision_seal_v1",
                "fold_hash": fold.fold_hash,
                "geometry_id": geometry,
                "decisions": [decision_payload(row) for row in selected],
                "support_label_identity_hash": label_identity_hash,
                "evaluation_labels_used": False,
            }
        )
        seals.append(FoldDecisionSeal(target, ordinal, "S_y", geometry, len(selected), digest))
    support_label_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_support_label_surface_v1",
            "target_center": target,
            "fold_ordinal": ordinal,
            "labels": [[row.target_center, row.case_id, row.sample_id, row.label] for row in scoped],
        }
    )
    return SupportFoldProduct(
        target, ordinal, tuple(sorted(decisions)), tuple(scores), tuple(seals),
        support_label_hash, fold.fold_hash,
    )


def combine_decision_products(
    pre_support: PreSupportDecisionProducts,
    support_folds: Sequence[SupportFoldProduct],
    partition: CaseOOFPartition,
) -> DecisionProducts:
    folds = tuple(support_folds)
    expected_keys = tuple((target, ordinal) for target in MIDOGPP_CENTERS for ordinal in range(5))
    if tuple((row.target_center, row.fold_ordinal) for row in folds) != expected_keys:
        raise ProtocolError("Support fold products must cover all 45 cells canonically.")
    for row in folds:
        if row.fold_hash != partition.fold(row.target_center, row.fold_ordinal).fold_hash:
            raise ProtocolError("Support fold partition provenance drifted.")
    support_decisions = tuple(row for fold in folds for row in fold.decisions)
    by_case = {(row.target_center, row.case_id, row.geometry_id): row for row in support_decisions}
    expected = {
        (target, case_id, geometry)
        for target in MIDOGPP_CENTERS for case_id in partition_cases(partition, target)
        for geometry in GEOMETRY_IDS
    }
    if set(by_case) != expected or len(by_case) != len(support_decisions):
        raise ProtocolError("Every whole case needs exactly one OOF S_y decision per geometry.")
    all_seals = dict(pre_support.pre_support_decision_hashes)
    for fold in folds:
        for seal in fold.geometry_seals:
            if seal.key in all_seals:
                raise ProtocolError("Support seal collides with a pre-support method.")
            all_seals[seal.key] = seal.decision_hash
    all_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_actionability_all_decisions_seal_v1",
            "partition_hash": partition.partition_hash,
            "pre_support_seal_hash": pre_support.pre_support_seal_hash,
            "support_product_hashes": [row.support_product_hash for row in folds],
            "decision_seals": [[*key, value] for key, value in sorted(all_seals.items())],
            "permutation_provenance_hash": pre_support.permutation_provenance_hash,
            "probability_surface_hash": pre_support.probability_surface_hash,
            "evaluation_labels_used": False,
        }
    )
    return DecisionProducts(
        tuple((*pre_support.decisions, *support_decisions)),
        tuple(row for fold in folds for row in fold.action_scores),
        tuple((row.target_center, row.fold_ordinal, row.support_product_hash) for row in folds),
        pre_support.pre_support_decision_hashes, pre_support.pre_support_seal_hash,
        all_seals, all_hash, pre_support.permutation_provenance_hash,
        partition.partition_hash, pre_support.protocol_contract_hash,
        pre_support.probability_surface_hash,
    )


__all__ = (
    "build_pre_support_decision_products", "build_support_fold_product", "combine_decision_products",
)
