"""Typed-to-JSON serializers for HARP probability, feature, and response surfaces."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import statistics
from typing import Any

from ...protocol import ProtocolError
from ...runtime.harp_probability_menu import (
    EXACT_NINE_SEED_PAIRS,
    UNIFORM_ACTION_ID,
    HarpPredictionMenuSeal,
)
from ..harp_protocol.hashing import canonical_hash
from ..harp_protocol.label_access import HarpSourceLabelRow
from .build import build_probability_surface
from .contracts import (
    ACTION_FEATURE_NAMES,
    ACTION_LAMBDAS,
    HarpActionFeatureSurface,
    HarpDirectionalResponseSurface,
    HarpProbabilityRow,
)
from .transport import seed_id


def development_seed_surface(menu: HarpPredictionMenuSeal):
    rows: list[HarpProbabilityRow] = []
    by_context: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for action in menu.actions:
        by_context[(action.outer_target_id, action.query_center_id)].append(action)
    for (outer, query), actions in sorted(by_context.items()):
        reference = next((action for action in actions if action.is_uniform_topup), None)
        if reference is None:
            raise ProtocolError("HARP development context lacks matched-budget U.")
        reference_cells = {
            (cell.training_seed, cell.generation_seed): cell
            for cell in menu.cells_for(reference)
        }
        for action in actions:
            if action.selected_source_id is None:
                continue
            for candidate_cell in menu.cells_for(action):
                pair = (candidate_cell.training_seed, candidate_cell.generation_seed)
                base_cell = reference_cells[pair]
                if (
                    candidate_cell.row_ids != base_cell.row_ids
                    or candidate_cell.case_ids != base_cell.case_ids
                ):
                    raise ProtocolError("HARP U/Hxe seed cells are not row aligned.")
                for index, (sample, case) in enumerate(
                    zip(base_cell.row_ids, base_cell.case_ids, strict=True)
                ):
                    rows.append(
                        HarpProbabilityRow(
                            outer_target=outer,
                            pseudo_query=query,
                            candidate_source=str(action.selected_source_id),
                            inner_donor=None,
                            case_id=case,
                            sample_id=sample,
                            seed_id=seed_id(pair),
                            baseline_probability=float(base_cell.probabilities[index]),
                            expert_probability=float(candidate_cell.probabilities[index]),
                            prediction_seal_hash=menu.seal_hash,
                        )
                    )
    return build_probability_surface(rows)


def action_feature_payload(surface: HarpActionFeatureSurface) -> dict[str, object]:
    rows = [
        {
            "outer_target": row.outer_target,
            "pseudo_query": row.pseudo_query,
            "candidate_source": row.candidate_source,
            "inner_donor": row.inner_donor,
            "case_id": row.case_id,
            "sample_id": row.sample_id,
            "case_sample_ids": list(row.case_sample_ids),
            "action_lambda": row.action_lambda,
            "direction": row.direction,
            "feature_names": list(row.feature_names),
            "feature_values": list(row.feature_values),
            "baseline_probability": row.baseline_probability,
            "expert_probability": row.expert_probability,
            "action_probability": row.action_probability,
            "ensemble_receipt_hash": row.ensemble_receipt_hash,
            "case_weight_receipt_hash": row.case_aggregation_receipt_hash,
            "prediction_seal_hash": row.prediction_seal_hash,
            "seed_count": row.seed_count,
            "feature_hash": row.feature_hash,
        }
        for row in surface.rows
    ]
    return {
        "schema_version": "midogpp_harp_action_feature_artifact_v2",
        "surface_hash": surface.surface_hash,
        "ensemble_surface_hash": surface.ensemble_surface_hash,
        "prediction_seal_hash": surface.prediction_seal_hash,
        "feature_names": list(ACTION_FEATURE_NAMES),
        "lambda_grid": list(ACTION_LAMBDAS),
        "row_count": len(rows),
        "rows": rows,
        "seed_cells_may_feed_model": False,
        "case_equal_weighting_required": True,
        "predictive_reference_action_id": UNIFORM_ACTION_ID,
        "probability_ensemble_semantics": "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe",
        "lambda_one_is_physical_hxe_endpoint": True,
        "labels_used": False,
    }


def response_payload(surface: HarpDirectionalResponseSurface) -> dict[str, object]:
    rows = [
        {
            "outer_target": row.outer_target,
            "pseudo_query": row.pseudo_query,
            "candidate_source": row.candidate_source,
            "inner_donor": row.inner_donor,
            "case_id": row.case_id,
            "sample_id": row.sample_id,
            "action_lambda": row.action_lambda,
            "direction": row.direction,
            "truth_class": row.truth_class,
            "weighted_correctness_surrogate": row.weighted_correctness_surrogate,
            "brier_delta": row.brier_delta,
            "log_loss_delta": row.log_loss_delta,
            "denominator_receipt_hash": row.denominator_receipt_hash,
            "ensemble_receipt_hash": row.ensemble_receipt_hash,
            "case_weight_receipt_hash": row.case_aggregation_receipt_hash,
            "feature_hash": row.feature_hash,
            "outer_scoped_label_surface_hash": row.label_surface_hash,
            "response_hash": row.response_hash,
        }
        for row in surface.rows
    ]
    return {
        "schema_version": "midogpp_harp_directional_response_artifact_v3",
        "surface_hash": surface.surface_hash,
        "feature_surface_hash": surface.feature_surface_hash,
        "label_surface_hash": surface.label_surface_hash,
        "label_surface_semantics": "collection_of_outer_H_scoped_hashes",
        "outer_label_surface_hashes": [
            [outer, label_hash]
            for outer, label_hash in sorted(
                {
                    receipt.outer_target: receipt.label_surface_hash
                    for receipt in surface.receipts
                }.items()
            )
        ],
        "receipts": [
            {
                "outer_target": receipt.outer_target,
                "pseudo_query": receipt.pseudo_query,
                "positive_case_count": receipt.positive_case_count,
                "negative_case_count": receipt.negative_case_count,
                "positive_weight": receipt.positive_weight,
                "negative_weight": receipt.negative_weight,
                "case_sample_counts": [list(value) for value in receipt.case_sample_counts],
                "case_class_sample_counts": [
                    list(value) for value in receipt.case_class_sample_counts
                ],
                "label_surface_hash": receipt.label_surface_hash,
                "receipt_hash": receipt.receipt_hash,
            }
            for receipt in surface.receipts
        ],
        "receipt_hashes": [receipt.receipt_hash for receipt in surface.receipts],
        "row_count": len(rows),
        "rows": rows,
        "source_development_labels_used_for_scoring_only": True,
        "target_labels_used": False,
        "case_equal_weighting_required": True,
        "response_reference_action_id": UNIFORM_ACTION_ID,
        "responses_are_hxe_predictive_ensemble_minus_u": True,
    }


def target_support_payload(menu: HarpPredictionMenuSeal) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    by_outer: dict[str, list[Any]] = defaultdict(list)
    for action in menu.actions:
        by_outer[action.outer_target_id].append(action)
    for outer, actions in sorted(by_outer.items()):
        reference = next((action for action in actions if action.is_uniform_topup), None)
        if reference is None:
            raise ProtocolError("HARP target context lacks matched-budget U.")
        base_cells = menu.cells_for(reference)
        row_ids, case_ids = menu.identities_for(reference)
        case_samples = {
            case: tuple(
                sorted(
                    row
                    for row, row_case in zip(row_ids, case_ids, strict=True)
                    if row_case == case
                )
            )
            for case in sorted(set(case_ids))
        }
        for action in actions:
            if action.selected_source_id is None:
                continue
            expert_cells = menu.cells_for(action)
            if menu.identities_for(action) != (row_ids, case_ids):
                raise ProtocolError("HARP target U/Hxe identities are not aligned.")
            for index, (sample, case) in enumerate(zip(row_ids, case_ids, strict=True)):
                baseline_members = tuple(float(cell.probabilities[index]) for cell in base_cells)
                expert_members = tuple(float(cell.probabilities[index]) for cell in expert_cells)
                ensemble_receipt = canonical_hash(
                    {
                        "schema_version": "midogpp_harp_target_exact_nine_sample_v2",
                        "outer_target": outer,
                        "candidate_source": action.selected_source_id,
                        "case_id": case,
                        "sample_id": sample,
                        "seed_pairs": [list(pair) for pair in EXACT_NINE_SEED_PAIRS],
                        "baseline_members": list(baseline_members),
                        "expert_members": list(expert_members),
                        "prediction_menu_hash": menu.seal_hash,
                        "predictive_reference_action_id": UNIFORM_ACTION_ID,
                    }
                )
                case_receipt = canonical_hash(
                    {
                        "schema_version": "midogpp_harp_target_equal_case_mass_v1",
                        "outer_target": outer,
                        "case_id": case,
                        "sample_ids": list(case_samples[case]),
                        "sample_weight": 1.0 / len(case_samples[case]),
                    }
                )
                baseline_mean = statistics.fmean(baseline_members)
                expert_mean = statistics.fmean(expert_members)
                dispersion = statistics.pstdev(
                    tuple(
                        e - b
                        for b, e in zip(baseline_members, expert_members, strict=True)
                    )
                )
                for lam in ACTION_LAMBDAS:
                    action_probability = (
                        expert_mean
                        if lam == 1.0
                        else (1.0 - lam) * baseline_mean + lam * expert_mean
                    )
                    direction = hard_direction(baseline_mean, action_probability)
                    feature_values = target_feature_values(
                        baseline_members, expert_members, lam, dispersion
                    )
                    unhashed = {
                        "outer_target": outer,
                        "candidate_source": action.selected_source_id,
                        "case_id": case,
                        "sample_id": sample,
                        "case_sample_ids": list(case_samples[case]),
                        "action_lambda": lam,
                        "direction": direction,
                        "feature_names": list(ACTION_FEATURE_NAMES),
                        "feature_values": list(feature_values),
                        "baseline_probability": baseline_mean,
                        "expert_probability": expert_mean,
                        "action_probability": action_probability,
                        "ensemble_receipt_hash": ensemble_receipt,
                        "case_weight_receipt_hash": case_receipt,
                        "seed_count": 9,
                        "label_free": True,
                    }
                    rows.append({**unhashed, "feature_hash": canonical_hash(unhashed)})
    rows.sort(
        key=lambda row: (
            str(row["outer_target"]),
            str(row["candidate_source"]),
            str(row["case_id"]),
            str(row["sample_id"]),
            float(row["action_lambda"]),
        )
    )
    surface_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_target_support_feature_surface_v2",
            "prediction_menu_hash": menu.seal_hash,
            "feature_hashes": [row["feature_hash"] for row in rows],
            "seed_cells_may_feed_model": False,
            "target_support_labels_used": False,
            "predictive_reference_action_id": UNIFORM_ACTION_ID,
        }
    )
    return {
        "schema_version": "midogpp_harp_target_support_feature_artifact_v2",
        "surface_hash": surface_hash,
        "prediction_menu_hash": menu.seal_hash,
        "feature_names": list(ACTION_FEATURE_NAMES),
        "lambda_grid": list(ACTION_LAMBDAS),
        "row_count": len(rows),
        "rows": rows,
        "seed_cells_may_feed_model": False,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "predictive_reference_action_id": UNIFORM_ACTION_ID,
        "probability_ensemble_semantics": "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe",
        "lambda_one_is_physical_hxe_endpoint": True,
    }


def target_feature_values(
    baseline_members: tuple[float, ...],
    expert_members: tuple[float, ...],
    lam: float,
    dispersion: float,
) -> tuple[float, ...]:
    baseline = statistics.fmean(baseline_members)
    expert = statistics.fmean(expert_members)
    action = expert if lam == 1.0 else (1.0 - lam) * baseline + lam * expert
    expert_flips = statistics.fmean(
        float(int(e >= 0.5) != int(b >= 0.5))
        for b, e in zip(baseline_members, expert_members, strict=True)
    )
    action_flips = statistics.fmean(
        float(int(((1.0 - lam) * b + lam * e) >= 0.5) != int(b >= 0.5))
        for b, e in zip(baseline_members, expert_members, strict=True)
    )
    return (
        baseline,
        expert,
        action,
        abs(baseline - 0.5),
        abs(expert - 0.5),
        abs(action - 0.5),
        expert - baseline,
        abs(expert - baseline),
        action - baseline,
        abs(action - baseline),
        expert_flips,
        action_flips,
        dispersion,
        lam,
    )


def hard_direction(baseline: float, action: float) -> str:
    pair = (int(baseline >= 0.5), int(action >= 0.5))
    return "D01" if pair == (0, 1) else "D10" if pair == (1, 0) else "ALL_MARGINS"


def read_fresh_source_labels(path: Path) -> tuple[HarpSourceLabelRow, ...]:
    required = ("center", "case_id", "sample_id", "label")
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ProtocolError("HARP fresh source manifest is unreadable.") from exc
    rows: list[HarpSourceLabelRow] = []
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != required:
            raise ProtocolError("HARP fresh source-label manifest schema drifted.")
        for raw in reader:
            try:
                label = int(str(raw["label"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError("HARP fresh source label is malformed.") from exc
            rows.append(
                HarpSourceLabelRow(
                    center=str(raw["center"]),
                    case_id=str(raw["case_id"]),
                    sample_id=str(raw["sample_id"]),
                    label=label,
                )
            )
    return tuple(rows)


__all__ = (
    "action_feature_payload",
    "development_seed_surface",
    "read_fresh_source_labels",
    "response_payload",
    "target_support_payload",
)
