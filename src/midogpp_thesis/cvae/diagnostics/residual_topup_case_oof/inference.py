"""Center-level contrasts and sealed Hxe oracle diagnostics."""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from ...metrics import spearman
from ...protocol import ProtocolError
from .contracts import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
    candidate_sources,
    tail_action_id,
)


CONTRASTS = (
    ("S-U", SUPPORT_ACTION_ID, UNIFORM_ACTION_ID, "primary"),
    ("S-G", SUPPORT_ACTION_ID, GLOBAL_ACTION_ID, "primary"),
    ("G-U", GLOBAL_ACTION_ID, UNIFORM_ACTION_ID, "secondary"),
    ("U-B", UNIFORM_ACTION_ID, BASE_ACTION_ID, "secondary"),
    ("S-B", SUPPORT_ACTION_ID, BASE_ACTION_ID, "secondary"),
    ("S-P", SUPPORT_ACTION_ID, PERMUTATION_ACTION_ID, "permutation_diagnostic"),
)

PRIMARY_CONTRAST_COLUMNS = (
    "schema_version",
    "target_center",
    "contrast_id",
    "left_action_id",
    "right_action_id",
    "left_bacc",
    "right_bacc",
    "bacc_delta",
    "left_macro_f1",
    "right_macro_f1",
    "macro_f1_delta_descriptive",
    "contrast_role",
    "inference_unit",
    "primary_endpoint",
    "diagnostic_only",
)

CONTRAST_INFERENCE_COLUMNS = (
    "schema_version",
    "contrast_id",
    "left_action_id",
    "right_action_id",
    "contrast_role",
    "center_count",
    "mean_bacc_delta",
    "sample_standard_deviation",
    "two_sided_95_ci_low",
    "two_sided_95_ci_high",
    "one_sided_95_lcb",
    "center_wins",
    "center_ties",
    "center_losses",
    "mean_positive",
    "one_sided_lcb_positive",
    "technical_seed_repeats_are_not_independent_units",
    "diagnostic_only",
)

ORACLE_HXE_COLUMNS = (
    "schema_version",
    "target_center",
    "source_count",
    "support_top1_source",
    "oracle_top1_source",
    "support_score_utility_spearman",
    "spearman_defined",
    "top1_agreement",
    "support_selected_Hxe_bacc",
    "oracle_Hxe_bacc",
    "oracle_headroom_bacc",
    "oracle_utility_range_bacc",
    "normalized_oracle_gap",
    "support_action_bacc",
    "Hxe_bacc_by_source_json",
    "support_midrank_by_source_json",
    "oracle_matrix_role",
    "may_update_policy",
    "diagnostic_only",
)


def build_center_contrasts(
    ensemble_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    by_key = {
        (str(row["target_center"]), str(row["action_id"])): row
        for row in ensemble_rows
    }
    if len(by_key) != len(ensemble_rows):
        raise ProtocolError("Case-OOF ensemble metrics duplicate.")
    output: list[dict[str, object]] = []
    for target in CENTERS:
        for contrast_id, left_id, right_id, role in CONTRASTS:
            try:
                left = by_key[(target, left_id)]
                right = by_key[(target, right_id)]
            except KeyError as exc:
                raise ProtocolError("Case-OOF contrast metrics are incomplete.") from exc
            output.append(
                {
                    "schema_version": "midogpp_residual_topup_case_oof_center_contrast_v1",
                    "target_center": target,
                    "contrast_id": contrast_id,
                    "left_action_id": left_id,
                    "right_action_id": right_id,
                    "left_bacc": float(left["bacc"]),
                    "right_bacc": float(right["bacc"]),
                    "bacc_delta": float(left["bacc"]) - float(right["bacc"]),
                    "left_macro_f1": float(left["macro_f1"]),
                    "right_macro_f1": float(right["macro_f1"]),
                    "macro_f1_delta_descriptive": float(left["macro_f1"])
                    - float(right["macro_f1"]),
                    "contrast_role": role,
                    "inference_unit": "target_center",
                    "primary_endpoint": "all_nine_seed_probability_ensemble_bacc",
                    "diagnostic_only": True,
                }
            )
    return tuple(output)


def infer_center_contrasts(
    contrast_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    output: list[dict[str, object]] = []
    for contrast_id, left_id, right_id, role in CONTRASTS:
        rows = [
            row for row in contrast_rows if str(row["contrast_id"]) == contrast_id
        ]
        if tuple(str(row["target_center"]) for row in rows) != CENTERS:
            raise ProtocolError("Case-OOF contrast center coverage drifted.")
        values = np.asarray([float(row["bacc_delta"]) for row in rows])
        if not np.isfinite(values).all():
            raise ProtocolError("Case-OOF contrast contains non-finite values.")
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        standard_error = standard_deviation / math.sqrt(len(values))
        two_sided = float(student_t.ppf(0.975, df=len(values) - 1))
        one_sided = float(student_t.ppf(0.95, df=len(values) - 1))
        output.append(
            {
                "schema_version": "midogpp_residual_topup_case_oof_contrast_inference_v1",
                "contrast_id": contrast_id,
                "left_action_id": left_id,
                "right_action_id": right_id,
                "contrast_role": role,
                "center_count": len(values),
                "mean_bacc_delta": mean,
                "sample_standard_deviation": standard_deviation,
                "two_sided_95_ci_low": mean - two_sided * standard_error,
                "two_sided_95_ci_high": mean + two_sided * standard_error,
                "one_sided_95_lcb": mean - one_sided * standard_error,
                "center_wins": int(np.sum(values > 0.0)),
                "center_ties": int(np.sum(values == 0.0)),
                "center_losses": int(np.sum(values < 0.0)),
                "mean_positive": mean > 0.0,
                "one_sided_lcb_positive": mean - one_sided * standard_error > 0.0,
                "technical_seed_repeats_are_not_independent_units": True,
                "diagnostic_only": True,
            }
        )
    return tuple(output)


def build_oracle_hxe_diagnostics(
    ensemble_rows: Sequence[Mapping[str, object]],
    *,
    rank_surface: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Diagnose proxy rank against sealed Hxe utility without emitting actions."""

    by_key = {
        (str(row["target_center"]), str(row["action_id"])): row
        for row in ensemble_rows
    }
    output: list[dict[str, object]] = []
    for target in CENTERS:
        sources = candidate_sources(target)
        try:
            support_summary = rank_surface[target].support_summary
            support_action = by_key[(target, SUPPORT_ACTION_ID)]
        except (KeyError, AttributeError) as exc:
            raise ProtocolError("Case-OOF support/oracle surface is incomplete.") from exc
        midranks = {
            source: float(
                support_summary.mean_normalized_midrank_by_source[source]
            )
            for source in sources
        }
        utilities: dict[str, float] = {}
        for source in sources:
            try:
                utilities[source] = float(
                    by_key[(target, tail_action_id(source))]["bacc"]
                )
            except KeyError as exc:
                raise ProtocolError("Case-OOF Hxe matrix is incomplete.") from exc
        support_top1 = min(sources, key=lambda source: (midranks[source], source))
        oracle_top1 = min(
            sources, key=lambda source: (-utilities[source], source)
        )
        aligned_scores = [-midranks[source] for source in sources]
        utility_values = np.asarray(
            [utilities[source] for source in sources], dtype=np.float64
        )
        correlation = float(spearman(aligned_scores, utility_values.tolist()))
        correlation_defined = math.isfinite(correlation)
        oracle_utility = float(np.max(utility_values))
        selected_utility = utilities[support_top1]
        utility_range = float(np.max(utility_values) - np.min(utility_values))
        headroom = max(0.0, oracle_utility - selected_utility)
        normalized_gap = headroom / utility_range if utility_range > 0.0 else 0.0
        output.append(
            {
                "schema_version": "midogpp_residual_topup_case_oof_oracle_Hxe_v1",
                "target_center": target,
                "source_count": len(sources),
                "support_top1_source": support_top1,
                "oracle_top1_source": oracle_top1,
                "support_score_utility_spearman": correlation
                if correlation_defined
                else 0.0,
                "spearman_defined": correlation_defined,
                "top1_agreement": support_top1 == oracle_top1,
                "support_selected_Hxe_bacc": selected_utility,
                "oracle_Hxe_bacc": oracle_utility,
                "oracle_headroom_bacc": headroom,
                "oracle_utility_range_bacc": utility_range,
                "normalized_oracle_gap": normalized_gap,
                "support_action_bacc": float(support_action["bacc"]),
                "Hxe_bacc_by_source_json": _compact(utilities),
                "support_midrank_by_source_json": _compact(midranks),
                "oracle_matrix_role": "diagnostic_only_no_policy_or_fallback_update",
                "may_update_policy": False,
                "diagnostic_only": True,
            }
        )
    return tuple(output)


def mechanism_interpretation(
    inference_rows: Sequence[Mapping[str, object]],
) -> str:
    by_id = {str(row["contrast_id"]): row for row in inference_rows}
    try:
        s_u = float(by_id["S-U"]["mean_bacc_delta"])
        s_g = float(by_id["S-G"]["mean_bacc_delta"])
    except KeyError as exc:
        raise ProtocolError("Case-OOF primary inference rows are absent.") from exc
    if s_u > 0.0 and s_g > 0.0:
        return "TARGET_SPECIFIC_SUPPORT_MECHANISM_SIGN_CRITERION_MET"
    if s_u > 0.0 and s_g <= 0.0:
        return "GLOBAL_SOURCE_PREFERENCE_ONLY_NOT_TARGET_SPECIFIC"
    return "SUPPORT_ROUTING_MECHANISM_NOT_SUPPORTED"


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = (
    "CONTRASTS",
    "CONTRAST_INFERENCE_COLUMNS",
    "ORACLE_HXE_COLUMNS",
    "PRIMARY_CONTRAST_COLUMNS",
    "build_center_contrasts",
    "build_oracle_hxe_diagnostics",
    "infer_center_contrasts",
    "mechanism_interpretation",
)
