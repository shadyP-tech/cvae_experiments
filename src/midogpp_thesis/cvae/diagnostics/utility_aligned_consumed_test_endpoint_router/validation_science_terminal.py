"""Terminal endpoint, contrast, aggregate and oracle reconstruction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ...metrics import spearman
from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from .artifact_io import read_json
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    PRIMARY_CONTRASTS,
    ROUTED_ACTION_ID,
    candidate_sources,
    expected_target_action_ids,
    h_x_e_action_id,
)
from .experiment_contracts import (
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER,
)
from .inference import (
    AggregateCenterContrast,
    CenterBaccContrast,
    OracleRankDiagnostic,
    summarize_center_contrasts,
)
from .prediction_contracts import TARGET_ROLE as TARGET_PREDICTION_ROLE
from .prediction_store import load_prediction_store
from .validation_science_common import (
    decode_hashed_row,
    decode_inference_row,
    nullable_text,
    read_csv,
)
from .validation_science_contracts import (
    SCORE_FIELDS,
    PrelabelScienceValidation,
    ScientificPartitionContext,
    TerminalScienceValidation,
)


def validate_terminal_science(
    root: str | Path,
    *,
    partitions: ScientificPartitionContext,
    prelabel: PrelabelScienceValidation,
) -> TerminalScienceValidation:
    base = Path(root)
    score_rows = tuple(
        decode_hashed_row(raw, SCORE_FIELDS, "score_hash", "terminal score")
        for raw in read_csv(base / "tables/terminal_endpoint_scores.csv")
    )
    expected_keys = tuple(
        (target, action_id)
        for target in CENTERS
        for action_id in expected_target_action_ids(target)
    )
    if tuple(
        (row.get("target_center"), row.get("action_id")) for row in score_rows
    ) != expected_keys:
        raise ProtocolError("Terminal endpoint-score key geometry drifted.")
    score_by_key: dict[tuple[str, str], Mapping[str, object]] = {}
    target_store = load_prediction_store(base, phase=TARGET_PREDICTION_ROLE)
    if target_store.partition_lock_hash != partitions.support_partition_lock_hash:
        raise ProtocolError("Terminal prediction partition lock drifted.")
    for row in score_rows:
        target = str(row["target_center"])
        action_id = str(row["action_id"])
        if (
            row.get("action_hash")
            != prelabel.action_hash_by_key[(target, action_id)]
            or row.get("policy_hash") != prelabel.policy_hash_by_target[target]
            or row.get("support_partition_lock_hash")
            != partitions.support_partition_lock_hash
            or row.get("evaluation_partition_hash")
            != partitions.evaluation_identity_hash_by_center[target]
            or row.get("evaluation_row_identity_hash")
            != partitions.evaluation_identity_hash_by_center[target]
            or row.get("global_target_prediction_seal_hash")
            != prelabel.global_target_prediction_seal_hash
            or row.get("global_prelabel_seal_hash")
            != prelabel.global_prelabel_seal_hash
            or int(row.get("evaluation_case_count", -1))
            != EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER[target]
            or int(row.get("evaluation_row_count", -1))
            != EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER[target]
            or int(row.get("observed_class_0_row_count", -1))
            + int(row.get("observed_class_1_row_count", -1))
            != EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER[target]
            or not 0.0 <= float(row.get("balanced_accuracy", math.nan)) <= 1.0
            or row.get("primary_endpoint")
            != "exact_nine_probability_mean_then_threshold_bacc"
            or row.get(
                "same_outer_H_evaluation_labels_opened_after_plan_and_global_seal"
            ) is not True
            or row.get("terminal_scores_may_update_plan") is not False
            or row.get("inference_unit") != "target_center"
            or row.get("technical_seed_cells_are_independent_units") is not False
            or row.get("consumed_test_diagnostic_only") is not True
        ):
            raise ProtocolError("Terminal endpoint-score binding drifted.")
        validate_terminal_endpoint_lineage(
            row,
            target_store,
            physical_action_id=prelabel.effective_action_by_key[
                (target, action_id)
            ],
        )
        score_by_key[(target, action_id)] = row
    for target in CENTERS:
        selected = tuple(
            score_by_key[(target, action_id)]
            for action_id in expected_target_action_ids(target)
        )
        if (
            len({row["evaluation_row_identity_hash"] for row in selected}) != 1
            or len({row["evaluation_label_hash"] for row in selected}) != 1
        ):
            raise ProtocolError("Terminal endpoint label/row lineage drifted.")
        validate_logical_endpoint_aliases(
            target,
            score_by_key,
            {
                logical: prelabel.effective_action_by_key[(target, logical)]
                for logical in (
                    GLOBAL_ACTION_ID,
                    ROUTED_ACTION_ID,
                    PERMUTATION_ACTION_ID,
                )
            },
        )

    action_hash_by_key = {
        f"{target}::{action_id}": prelabel.action_hash_by_key[(target, action_id)]
        for target, action_id in expected_keys
    }
    score_set_unhashed = {
        "schema_version": "midogpp_consumed_test_terminal_endpoint_score_set_v1",
        "centers": list(CENTERS),
        "score_hashes": [str(row["score_hash"]) for row in score_rows],
        "action_library_hash": prelabel.action_library_hash,
        "policy_set_hash": prelabel.policy_set_hash,
        "action_hash_by_key": action_hash_by_key,
        "policy_hash_by_target": dict(prelabel.policy_hash_by_target),
        "global_target_prediction_seal_hash": (
            prelabel.global_target_prediction_seal_hash
        ),
        "global_prelabel_seal_hash": prelabel.global_prelabel_seal_hash,
        "score_count": len(score_rows),
        "same_outer_H_labels_opened_only_after_global_seal": True,
        "terminal_scores_may_update_plan": False,
    }
    score_set_hash = canonical_sha256(score_set_unhashed)

    center_rows = tuple(
        center_contrast(raw)
        for raw in read_csv(base / "tables/center_contrasts.csv")
    )
    expected_center_keys = tuple(
        (target, contrast_id)
        for target in CENTERS
        for contrast_id, _left, _right in PRIMARY_CONTRASTS
    )
    if tuple(
        (row.target_id, row.contrast_id) for row in center_rows
    ) != expected_center_keys:
        raise ProtocolError("Center contrast geometry drifted.")
    for row in center_rows:
        expected_left = float(
            score_by_key[(row.target_id, row.left_action_id)]["balanced_accuracy"]
        )
        expected_right = float(
            score_by_key[(row.target_id, row.right_action_id)]["balanced_accuracy"]
        )
        if (
            row.score_set_hash != score_set_hash
            or not math.isclose(
                row.left_bacc, expected_left, rel_tol=0.0, abs_tol=1e-15
            )
            or not math.isclose(
                row.right_bacc, expected_right, rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ProtocolError("Center contrast/score reconstruction drifted.")
    expected_aggregate = summarize_center_contrasts(
        center_rows, score_set_hash=score_set_hash
    )
    observed_aggregate = tuple(
        aggregate_contrast(raw)
        for raw in read_csv(base / "tables/aggregate_contrasts.csv")
    )
    if [row.to_payload() for row in observed_aggregate] != [
        row.to_payload() for row in expected_aggregate
    ]:
        raise ProtocolError("Aggregate center formula reconstruction drifted.")

    observed_oracle = tuple(
        oracle_row(raw)
        for raw in read_csv(base / "tables/oracle_rank_diagnostics.csv")
    )
    expected_oracle = tuple(
        rebuild_oracle(target, score_by_key, prelabel, score_set_hash)
        for target in CENTERS
    )
    if [row.to_payload() for row in observed_oracle] != [
        row.to_payload() for row in expected_oracle
    ]:
        raise ProtocolError("Oracle-rank formula reconstruction drifted.")

    inference_unhashed = {
        "schema_version": "midogpp_consumed_test_terminal_inference_v1",
        "score_set_hash": score_set_hash,
        "center_contrast_hashes": [row.contrast_hash for row in center_rows],
        "aggregate_contrast_hashes": [
            row.summary_hash for row in expected_aggregate
        ],
        "oracle_rank_diagnostic_hashes": [
            row.diagnostic_hash for row in expected_oracle
        ],
        "inference_unit": "target_center",
        "Hxe_may_feed_plan": False,
        "terminal_scores_may_update_plan": False,
    }
    inference_hash = canonical_sha256(inference_unhashed)
    sealed = read_json(base / "manifests/sealed_terminal_evaluation.json")
    if (
        sealed.get("terminal_score_set_hash") != score_set_hash
        or sealed.get("terminal_inference_hash") != inference_hash
        or sealed.get("target_prediction_seal_hash")
        != prelabel.global_target_prediction_seal_hash
        or sealed.get("global_prelabel_seal_hash")
        != prelabel.global_prelabel_seal_hash
        or sealed.get(
            "same_outer_H_evaluation_labels_used_for_plan_H"
        ) is not False
        or sealed.get("support_labels_used") is not False
        or sealed.get("terminal_only_no_plan_or_policy_update") is not True
    ):
        raise ProtocolError("Sealed terminal-evaluation lineage drifted.")
    return TerminalScienceValidation(
        score_count=len(score_rows),
        score_set_hash=score_set_hash,
        contrast_count=len(center_rows),
        inference_hash=inference_hash,
    )


def center_contrast(raw: Mapping[str, str]) -> CenterBaccContrast:
    payload = decode_inference_row(
        raw,
        int_fields=set(),
        float_fields={"left_bacc", "right_bacc", "paired_bacc_delta"},
        bool_fields={
            "terminal_scores_may_update_plan", "consumed_test_diagnostic_only"
        },
    )
    row = CenterBaccContrast(
        target_id=str(payload["target_center"]),
        contrast_id=str(payload["contrast_id"]),
        left_action_id=str(payload["left_action_id"]),
        right_action_id=str(payload["right_action_id"]),
        left_bacc=float(payload["left_bacc"]),
        right_bacc=float(payload["right_bacc"]),
        paired_bacc_delta=float(payload["paired_bacc_delta"]),
        score_set_hash=str(payload["score_set_hash"]),
        contrast_hash=str(payload["contrast_hash"]),
    )
    if payload != row.to_payload():
        raise ProtocolError("Center contrast CSV projection drifted.")
    return row


def aggregate_contrast(raw: Mapping[str, str]) -> AggregateCenterContrast:
    payload = decode_inference_row(
        raw,
        int_fields={"center_count", "degrees_of_freedom"},
        float_fields={
            "equal_center_mean_delta", "sample_standard_deviation",
            "standard_error", "two_sided_ci95_lower", "two_sided_ci95_upper",
            "one_sided_lcb95", "two_sided_p_value",
        },
        bool_fields={
            "technical_seed_cells_are_independent_units",
            "terminal_scores_may_update_plan",
            "consumed_test_diagnostic_only",
        },
    )
    row = AggregateCenterContrast(
        contrast_id=str(payload["contrast_id"]),
        left_action_id=str(payload["left_action_id"]),
        right_action_id=str(payload["right_action_id"]),
        center_count=int(payload["center_count"]),
        degrees_of_freedom=int(payload["degrees_of_freedom"]),
        equal_center_mean_delta=float(payload["equal_center_mean_delta"]),
        sample_standard_deviation=float(payload["sample_standard_deviation"]),
        standard_error=float(payload["standard_error"]),
        two_sided_ci95_lower=float(payload["two_sided_ci95_lower"]),
        two_sided_ci95_upper=float(payload["two_sided_ci95_upper"]),
        one_sided_lcb95=float(payload["one_sided_lcb95"]),
        two_sided_p_value=float(payload["two_sided_p_value"]),
        center_delta_hash=str(payload["center_delta_hash"]),
        score_set_hash=str(payload["score_set_hash"]),
        summary_hash=str(payload["summary_hash"]),
    )
    if payload != row.to_payload():
        raise ProtocolError("Aggregate contrast CSV projection drifted.")
    return row


def oracle_row(raw: Mapping[str, str]) -> OracleRankDiagnostic:
    payload = decode_inference_row(
        raw,
        int_fields={"routed_candidate_oracle_rank"},
        float_fields={
            "routed_candidate_normalized_rank", "base_bacc",
            "routed_endpoint_bacc", "routed_candidate_hxe_bacc",
            "oracle_hxe_bacc", "normalized_oracle_gap",
        },
        bool_fields={
            "routed_top1_exact_agreement", "routed_top1_tie_agreement",
            "Hxe_may_feed_plan", "terminal_scores_may_update_plan",
            "consumed_test_diagnostic_only",
        },
        json_fields={"oracle_source_ids"},
        nullable_fields={
            "routed_executed_source", "predicted_gain_hxe_bacc_spearman"
        },
    )
    rho = payload["predicted_gain_hxe_bacc_spearman"]
    row = OracleRankDiagnostic(
        target_id=str(payload["target_center"]),
        routed_candidate_source=str(payload["routed_candidate_source"]),
        routed_executed_source=nullable_text(payload["routed_executed_source"]),
        routed_executed_action_id=str(payload["routed_executed_action_id"]),
        oracle_source_ids=tuple(
            str(value) for value in payload["oracle_source_ids"]
        ),
        routed_candidate_oracle_rank=int(payload["routed_candidate_oracle_rank"]),
        routed_candidate_normalized_rank=float(
            payload["routed_candidate_normalized_rank"]
        ),
        routed_top1_exact_agreement=bool(payload["routed_top1_exact_agreement"]),
        routed_top1_tie_agreement=bool(payload["routed_top1_tie_agreement"]),
        predicted_gain_hxe_bacc_spearman=(
            None if rho is None else float(rho)
        ),
        base_bacc=float(payload["base_bacc"]),
        routed_endpoint_bacc=float(payload["routed_endpoint_bacc"]),
        routed_candidate_hxe_bacc=float(payload["routed_candidate_hxe_bacc"]),
        oracle_hxe_bacc=float(payload["oracle_hxe_bacc"]),
        normalized_oracle_gap=float(payload["normalized_oracle_gap"]),
        policy_hash=str(payload["policy_hash"]),
        score_set_hash=str(payload["score_set_hash"]),
        diagnostic_hash=str(payload["diagnostic_hash"]),
    )
    if payload != row.to_payload():
        raise ProtocolError("Oracle-rank CSV projection drifted.")
    return row


def rebuild_oracle(
    target: str,
    scores: Mapping[tuple[str, str], Mapping[str, object]],
    prelabel: PrelabelScienceValidation,
    score_set_hash: str,
) -> OracleRankDiagnostic:
    sources = candidate_sources(target)
    utility = {
        source: float(
            scores[(target, h_x_e_action_id(source))]["balanced_accuracy"]
        )
        for source in sources
    }
    maximum, minimum = max(utility.values()), min(utility.values())
    oracle_sources = tuple(
        source
        for source in sources
        if math.isclose(utility[source], maximum, rel_tol=0.0, abs_tol=1e-15)
    )
    proposal = prelabel.routed_candidate_by_target[target]
    rank = 1 + sum(
        value > utility[proposal] + 1e-15 for value in utility.values()
    )
    denominator = maximum - minimum
    gap = 0.0 if denominator <= 0.0 else (
        maximum - utility[proposal]
    ) / denominator
    predicted = prelabel.routed_prediction_by_target[target]
    rho = spearman(
        [predicted[source] for source in sources],
        [utility[source] for source in sources],
    )
    rho_value = None if not math.isfinite(rho) else float(rho)
    unhashed = {
        "schema_version": "midogpp_consumed_test_oracle_rank_diagnostic_v1",
        "target_center": target,
        "routed_candidate_source": proposal,
        "routed_executed_source": prelabel.routed_executed_source_by_target[target],
        "routed_executed_action_id": prelabel.selected_action_by_target[target],
        "oracle_source_ids": list(oracle_sources),
        "routed_candidate_oracle_rank": rank,
        "routed_candidate_normalized_rank": (rank - 1) / 7.0,
        "routed_top1_exact_agreement": proposal == min(oracle_sources),
        "routed_top1_tie_agreement": proposal in oracle_sources,
        "predicted_gain_hxe_bacc_spearman": rho_value,
        "base_bacc": float(scores[(target, BASE_ACTION_ID)]["balanced_accuracy"]),
        "routed_endpoint_bacc": float(
            scores[(target, ROUTED_ACTION_ID)]["balanced_accuracy"]
        ),
        "routed_candidate_hxe_bacc": utility[proposal],
        "oracle_hxe_bacc": maximum,
        "normalized_oracle_gap": gap,
        "policy_hash": prelabel.policy_hash_by_target[target],
        "score_set_hash": score_set_hash,
        "Hxe_may_feed_plan": False,
        "terminal_scores_may_update_plan": False,
        "consumed_test_diagnostic_only": True,
    }
    return OracleRankDiagnostic(
        target_id=target,
        routed_candidate_source=proposal,
        routed_executed_source=prelabel.routed_executed_source_by_target[target],
        routed_executed_action_id=prelabel.selected_action_by_target[target],
        oracle_source_ids=oracle_sources,
        routed_candidate_oracle_rank=rank,
        routed_candidate_normalized_rank=(rank - 1) / 7.0,
        routed_top1_exact_agreement=proposal == min(oracle_sources),
        routed_top1_tie_agreement=proposal in oracle_sources,
        predicted_gain_hxe_bacc_spearman=rho_value,
        base_bacc=float(scores[(target, BASE_ACTION_ID)]["balanced_accuracy"]),
        routed_endpoint_bacc=float(
            scores[(target, ROUTED_ACTION_ID)]["balanced_accuracy"]
        ),
        routed_candidate_hxe_bacc=utility[proposal],
        oracle_hxe_bacc=maximum,
        normalized_oracle_gap=gap,
        policy_hash=prelabel.policy_hash_by_target[target],
        score_set_hash=score_set_hash,
        diagnostic_hash=canonical_sha256(unhashed),
    )


def validate_logical_endpoint_aliases(
    target: str,
    scores: Mapping[tuple[str, str], Mapping[str, object]],
    effective_action_by_logical: Mapping[str, str],
) -> None:
    """Require logical G/R/P reports to be exact aliases of frozen physics."""

    lineage_fields = (
        "evaluation_row_identity_hash", "evaluation_label_hash", "endpoint_hash",
        "evaluation_case_count", "evaluation_row_count",
        "observed_class_0_row_count", "observed_class_1_row_count",
        "balanced_accuracy", "primary_endpoint",
    )
    for logical in (GLOBAL_ACTION_ID, ROUTED_ACTION_ID, PERMUTATION_ACTION_ID):
        physical = effective_action_by_logical.get(logical)
        if physical is None or (target, physical) not in scores:
            raise ProtocolError("Logical endpoint has no frozen physical action.")
        logical_row = scores[(target, logical)]
        physical_row = scores[(target, physical)]
        if any(
            logical_row.get(field) != physical_row.get(field)
            for field in lineage_fields
        ):
            raise ProtocolError("Logical endpoint differs from its physical action.")


def validate_terminal_endpoint_lineage(
    row: Mapping[str, object],
    store: object,
    *,
    physical_action_id: str,
) -> None:
    vectors = store.vectors(
        outer_target=str(row["target_center"]),
        query_center=str(row["target_center"]),
        action_id=physical_action_id,
        role="evaluation",
    )
    mean_probability = np.mean(
        np.stack([vector.positive_class_probabilities for vector in vectors]),
        axis=0,
        dtype=np.float64,
    )
    prediction = (mean_probability >= 0.5).astype(np.uint8)
    endpoint_unhashed = {
        "schema_version": "midogpp_utility_aligned_probability_ensemble_endpoint_v1",
        "row_identity_hash": vectors[0].row_identity_hash,
        "label_sha256": row["evaluation_label_hash"],
        "seed_pair_count": 9,
        "seed_keys": [
            [training_seed, generation_seed]
            for training_seed, generation_seed in (
                (17, 17), (17, 42), (17, 101),
                (42, 17), (42, 42), (42, 101),
                (101, 17), (101, 42), (101, 101),
            )
        ],
        "component_vector_hashes": [vector.vector_hash for vector in vectors],
        "row_count": len(prediction),
        "mean_probability_sha256": array_sha256(mean_probability),
        "prediction_sha256": array_sha256(prediction),
        "balanced_accuracy": row["balanced_accuracy"],
        "threshold": 0.5,
        "endpoint_semantics": (
            "arithmetic_mean_of_exact_nine_positive_class_probability_vectors_"
            "thresholded_at_0_5_then_balanced_accuracy"
        ),
    }
    if (
        row.get("evaluation_row_identity_hash") != vectors[0].row_identity_hash
        or row.get("evaluation_row_count") != len(prediction)
        or row.get("endpoint_hash") != canonical_sha256(endpoint_unhashed)
    ):
        raise ProtocolError("Terminal endpoint/prediction-store lineage drifted.")


__all__ = (
    "aggregate_contrast", "center_contrast", "oracle_row", "rebuild_oracle",
    "validate_logical_endpoint_aliases", "validate_terminal_endpoint_lineage",
    "validate_terminal_science",
)
