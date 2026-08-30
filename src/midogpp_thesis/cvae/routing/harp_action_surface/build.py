"""Pure HARP builders with exact-nine and independent-case enforcement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import math
import statistics

from ...protocol import ProtocolError
from ..harp_protocol.contracts import canonical_id
from ..harp_protocol.hashing import canonical_hash
from ..harp_protocol.label_access import HarpSourceLabelRow, OpenedHarpSourceLabels
from .contracts import (
    ACTION_FEATURE_NAMES,
    ACTION_LAMBDAS,
    ENSEMBLE_SEED_COUNT,
    HarpActionFeatureRow,
    HarpActionFeatureSurface,
    HarpDirectionalResponseRow,
    HarpDirectionalResponseSurface,
    HarpDisagreementRow,
    HarpProbabilityEnsembleRow,
    HarpProbabilityEnsembleSurface,
    HarpProbabilityRow,
    HarpProbabilitySurface,
    SourceClassPriorReceipt,
    _direction,
    action_feature_values,
    outer_scoped_label_collection_hash,
)


def build_probability_surface(rows: Sequence[HarpProbabilityRow]) -> HarpProbabilitySurface:
    typed = tuple(rows)
    if not typed or any(not isinstance(row, HarpProbabilityRow) for row in typed):
        raise ProtocolError("HARP probability builder requires typed seed cells.")
    seals = {row.prediction_seal_hash for row in typed}
    if len(seals) != 1:
        raise ProtocolError("HARP probability seed cells require one prediction seal.")
    return HarpProbabilitySurface(
        rows=tuple(sorted(typed, key=lambda row: row.row_key)),
        prediction_seal_hash=next(iter(seals)),
    )


def build_probability_ensemble_surface(
    seed_surface: HarpProbabilitySurface,
    *,
    expected_seed_ids: Sequence[str],
) -> HarpProbabilityEnsembleSurface:
    """Aggregate exact-nine seeds per sample and bind equal-case mass receipts."""

    if not isinstance(seed_surface, HarpProbabilitySurface):
        raise ProtocolError("HARP ensemble construction requires a typed seed surface.")
    seeds = tuple(canonical_id(value, name="seed") for value in expected_seed_ids)
    if len(seeds) != ENSEMBLE_SEED_COUNT or seeds != tuple(sorted(set(seeds))):
        raise ProtocolError("HARP ensemble construction requires exact-nine canonical seeds.")

    by_sample: dict[tuple[str, str, str, str, str, str], list[HarpProbabilityRow]] = defaultdict(list)
    for row in seed_surface.rows:
        key = (
            row.outer_target,
            row.pseudo_query,
            row.candidate_source,
            row.inner_donor or "",
            row.case_id,
            row.sample_id,
        )
        by_sample[key].append(row)

    checked_samples: dict[tuple[str, str, str, str, str], list[tuple[str, tuple[HarpProbabilityRow, ...]]]] = defaultdict(list)
    for key, cells in by_sample.items():
        ordered = tuple(sorted(cells, key=lambda row: row.seed_id))
        if tuple(row.seed_id for row in ordered) != seeds:
            raise ProtocolError("HARP sample lacks the exact sealed nine-seed inventory.")
        checked_samples[key[:-1]].append((key[-1], ordered))

    ensembles: list[HarpProbabilityEnsembleRow] = []
    for case_key, sample_cells in checked_samples.items():
        ordered_samples = tuple(sorted(sample_cells, key=lambda item: item[0]))
        sample_ids = tuple(item[0] for item in ordered_samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ProtocolError("HARP case aggregation duplicates sample identities.")
        case_receipt = canonical_hash(
            {
                "schema_version": "midogpp_harp_equal_case_total_mass_receipt_v2",
                "role_key": list(case_key),
                "sample_ids": list(sample_ids),
                "seed_ids": list(seeds),
                "seed_cell_row_hashes": [
                    row.row_hash for _, rows in ordered_samples for row in rows
                ],
                "sample_weight": 1.0 / len(sample_ids),
                "samples_are_model_rows": True,
                "equal_total_mass_per_independent_case": True,
            }
        )
        outer, query, source, donor_key, case_id = case_key
        for sample_id, sample_rows in ordered_samples:
            baseline_members = tuple(row.baseline_probability for row in sample_rows)
            expert_members = tuple(row.expert_probability for row in sample_rows)
            ensembles.append(
                HarpProbabilityEnsembleRow(
                    outer_target=outer,
                    pseudo_query=query,
                    candidate_source=source,
                    inner_donor=donor_key or None,
                    case_id=case_id,
                    sample_id=sample_id,
                    case_sample_ids=sample_ids,
                    seed_ids=seeds,
                    baseline_member_probabilities=baseline_members,
                    expert_member_probabilities=expert_members,
                    baseline_probability=statistics.fmean(baseline_members),
                    expert_probability=statistics.fmean(expert_members),
                    seed_dispersion=statistics.pstdev(
                        tuple(
                            expert - baseline
                            for baseline, expert in zip(
                                baseline_members, expert_members, strict=True
                            )
                        )
                    ),
                    case_aggregation_receipt_hash=case_receipt,
                    prediction_seal_hash=seed_surface.prediction_seal_hash,
                )
            )
    return HarpProbabilityEnsembleSurface(
        rows=tuple(sorted(ensembles, key=lambda row: row.row_key)),
        seed_surface_hash=seed_surface.surface_hash,
        expected_seed_ids=seeds,
        prediction_seal_hash=seed_surface.prediction_seal_hash,
    )


def build_action_feature_surface(
    ensemble_surface: HarpProbabilityEnsembleSurface,
    *,
    action_lambdas: Sequence[float] = ACTION_LAMBDAS,
) -> HarpActionFeatureSurface:
    if not isinstance(ensemble_surface, HarpProbabilityEnsembleSurface):
        raise ProtocolError(
            "HARP features require exact-nine case ensembles, never seed cells."
        )
    lambdas = tuple(float(value) for value in action_lambdas)
    if lambdas != ACTION_LAMBDAS:
        raise ProtocolError("HARP action builder requires the locked lambda portfolio.")
    rows: list[HarpActionFeatureRow] = []
    for ensemble in ensemble_surface.rows:
        for lam in lambdas:
            action = (
                ensemble.expert_probability
                if lam == 1.0
                else (
                    (1.0 - lam) * ensemble.baseline_probability
                    + lam * ensemble.expert_probability
                )
            )
            rows.append(
                HarpActionFeatureRow(
                    outer_target=ensemble.outer_target,
                    pseudo_query=ensemble.pseudo_query,
                    candidate_source=ensemble.candidate_source,
                    inner_donor=ensemble.inner_donor,
                    case_id=ensemble.case_id,
                    sample_id=ensemble.sample_id,
                    case_sample_ids=ensemble.case_sample_ids,
                    action_lambda=lam,
                    direction=_direction(ensemble.baseline_probability, action),
                    baseline_probability=ensemble.baseline_probability,
                    expert_probability=ensemble.expert_probability,
                    action_probability=action,
                    feature_names=ACTION_FEATURE_NAMES,
                    feature_values=action_feature_values(ensemble, lam),
                    ensemble_receipt_hash=ensemble.ensemble_receipt_hash,
                    case_aggregation_receipt_hash=ensemble.case_aggregation_receipt_hash,
                    prediction_seal_hash=ensemble.prediction_seal_hash,
                )
            )
    return HarpActionFeatureSurface(
        rows=tuple(sorted(rows, key=lambda row: row.row_key)),
        ensemble_surface_hash=ensemble_surface.surface_hash,
        prediction_seal_hash=ensemble_surface.prediction_seal_hash,
    )


def build_disagreement_rows(
    feature_surface: HarpActionFeatureSurface,
) -> tuple[HarpDisagreementRow, ...]:
    if not isinstance(feature_surface, HarpActionFeatureSurface):
        raise ProtocolError("HARP disagreement rows require typed case features.")
    return tuple(
        HarpDisagreementRow(
            outer_target=row.outer_target,
            pseudo_query=row.pseudo_query,
            candidate_source=row.candidate_source,
            inner_donor=row.inner_donor,
            case_id=row.case_id,
            sample_id=row.sample_id,
            action_lambda=row.action_lambda,
            direction=row.direction,
            ensemble_receipt_hash=row.ensemble_receipt_hash,
            feature_hash=row.feature_hash,
        )
        for row in feature_surface.rows
        if row.direction in ("D01", "D10")
    )


def build_directional_response_surface(
    feature_surface: HarpActionFeatureSurface,
    labels: OpenedHarpSourceLabels,
    *,
    log_loss_epsilon: float = 1.0e-6,
) -> HarpDirectionalResponseSurface:
    if not isinstance(feature_surface, HarpActionFeatureSurface) or not isinstance(
        labels, OpenedHarpSourceLabels
    ):
        raise ProtocolError("HARP responses require typed case features and source labels.")
    if not 0.0 < log_loss_epsilon < 0.5:
        raise ProtocolError("HARP log-loss epsilon must lie in (0,0.5).")

    # Close the label surface against the already sealed query-frame identity
    # before any denominator or response transform is constructed.  Minimal
    # unit surfaces need not represent every center, but every represented
    # pseudo-query must have exact key equality (no missing or extra rows).
    expected_by_context: dict[
        tuple[str, str], set[tuple[str, str, str]]
    ] = defaultdict(set)
    for row in feature_surface.rows:
        expected_by_context[(row.outer_target, row.pseudo_query)].add(
            (row.pseudo_query, row.case_id, row.sample_id)
        )
    outer_targets = tuple(sorted({outer for outer, _ in expected_by_context}))
    scoped_by_outer = {
        outer: labels.scope_for_outer_target(outer) for outer in outer_targets
    }
    observed_by_context: dict[
        tuple[str, str], set[tuple[str, str, str]]
    ] = defaultdict(set)
    for outer, scoped in scoped_by_outer.items():
        for row in scoped.rows:
            context = (outer, row.center)
            if context in expected_by_context:
                observed_by_context[context].add(row.row_key)
    if any(
        observed_by_context[context] != expected
        for context, expected in expected_by_context.items()
    ):
        raise ProtocolError(
            "HARP source-label keys differ from the sealed development menu."
        )

    label_cases_by_outer = {
        outer: _label_cases(scoped.rows)
        for outer, scoped in scoped_by_outer.items()
    }
    label_index_by_outer = {
        outer: {
            (row.center, row.case_id, row.sample_id): row.label
            for row in scoped.rows
        }
        for outer, scoped in scoped_by_outer.items()
    }
    contexts = sorted(
        {(row.outer_target, row.pseudo_query) for row in feature_surface.rows}
    )
    receipts: dict[tuple[str, str], SourceClassPriorReceipt] = {}
    for outer, query in contexts:
        scoped = scoped_by_outer[outer]
        scoped_cases = label_cases_by_outer[outer]
        query_cases = {
            key[1]: value for key, value in scoped_cases.items() if key[0] == query
        }
        positives = sum(any(label == 1 for label in value.values()) for value in query_cases.values())
        negatives = sum(any(label == 0 for label in value.values()) for value in query_cases.values())
        if positives <= 0 or negatives <= 0:
            raise ProtocolError("HARP source query lacks both independent case classes.")
        total_cases = len(query_cases)
        receipts[(outer, query)] = SourceClassPriorReceipt(
            outer_target=outer,
            pseudo_query=query,
            positive_case_count=positives,
            negative_case_count=negatives,
            positive_weight=total_cases / (2.0 * positives),
            negative_weight=total_cases / (2.0 * negatives),
            case_sample_counts=tuple(
                sorted((case_id, len(sample_truth)) for case_id, sample_truth in query_cases.items())
            ),
            case_class_sample_counts=tuple(
                sorted(
                    (case_id, truth, sum(value == truth for value in sample_truth.values()))
                    for case_id, sample_truth in query_cases.items()
                    for truth in (0, 1)
                    if any(value == truth for value in sample_truth.values())
                )
            ),
            label_surface_hash=scoped.label_surface_hash,
        )

    responses: list[HarpDirectionalResponseRow] = []
    for feature in feature_surface.rows:
        scoped = scoped_by_outer[feature.outer_target]
        case = label_cases_by_outer[feature.outer_target].get(
            (feature.pseudo_query, feature.case_id)
        )
        truth = label_index_by_outer[feature.outer_target].get(
            (feature.pseudo_query, feature.case_id, feature.sample_id)
        )
        if case is None or truth is None:
            raise ProtocolError("HARP sample feature has no aligned source-inner truth.")
        sample_truth = case
        sample_ids = tuple(sorted(sample_truth))
        if sample_truth.get(feature.sample_id) != truth or sample_ids != feature.case_sample_ids:
            raise ProtocolError("HARP case prediction/label sample coverage drifted.")
        receipt = receipts[(feature.outer_target, feature.pseudo_query)]
        class_case_weight = receipt.positive_weight if truth == 1 else receipt.negative_weight
        case_sample_count = len(sample_truth)
        case_class_sample_count = sum(value == truth for value in sample_truth.values())
        if case_class_sample_count <= 0:
            raise ProtocolError("HARP source case/class denominator is empty.")
        # The ridge supplies 1/n_samples(case) row mass.  This multiplier
        # cancels that count, assigns equal mass within each case/class, and
        # applies the Stage-70 class-over-cases factor.
        weight = class_case_weight * case_sample_count / case_class_sample_count
        baseline_correct = int(feature.baseline_probability >= 0.5) == truth
        action_correct = int(feature.action_probability >= 0.5) == truth
        responses.append(
            HarpDirectionalResponseRow(
                outer_target=feature.outer_target,
                pseudo_query=feature.pseudo_query,
                candidate_source=feature.candidate_source,
                inner_donor=feature.inner_donor,
                case_id=feature.case_id,
                sample_id=feature.sample_id,
                action_lambda=feature.action_lambda,
                direction=feature.direction,
                truth_class=truth,
                weighted_correctness_surrogate=weight
                * (float(action_correct) - float(baseline_correct)),
                brier_delta=(feature.action_probability - truth) ** 2
                - (feature.baseline_probability - truth) ** 2,
                log_loss_delta=_binary_log_loss(
                    feature.action_probability, truth, epsilon=log_loss_epsilon
                )
                - _binary_log_loss(
                    feature.baseline_probability, truth, epsilon=log_loss_epsilon
                ),
                denominator_receipt_hash=receipt.receipt_hash,
                ensemble_receipt_hash=feature.ensemble_receipt_hash,
                case_aggregation_receipt_hash=feature.case_aggregation_receipt_hash,
                feature_hash=feature.feature_hash,
                label_surface_hash=scoped.label_surface_hash,
            )
        )
    scoped_collection_hash = outer_scoped_label_collection_hash(
        tuple(
            (outer, scoped_by_outer[outer].label_surface_hash)
            for outer in outer_targets
        )
    )
    return HarpDirectionalResponseSurface(
        rows=tuple(sorted(responses, key=lambda row: row.row_key)),
        feature_surface_hash=feature_surface.surface_hash,
        label_surface_hash=scoped_collection_hash,
        receipts=tuple(receipts[key] for key in sorted(receipts)),
    )


def _label_cases(
    rows: Sequence[HarpSourceLabelRow],
) -> dict[tuple[str, str], dict[str, int]]:
    grouped: dict[tuple[str, str], list[HarpSourceLabelRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.center, row.case_id)].append(row)
    result: dict[tuple[str, str], dict[str, int]] = {}
    for key, case_rows in grouped.items():
        sample_truth = {row.sample_id: row.label for row in case_rows}
        if len(sample_truth) != len(case_rows):
            raise ProtocolError("HARP source case contains duplicate samples.")
        result[key] = dict(sorted(sample_truth.items()))
    return result


def _binary_log_loss(probability: float, label: int, *, epsilon: float) -> float:
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return -(label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped))


__all__ = (
    "build_action_feature_surface",
    "build_directional_response_surface",
    "build_disagreement_rows",
    "build_probability_ensemble_surface",
    "build_probability_surface",
)
