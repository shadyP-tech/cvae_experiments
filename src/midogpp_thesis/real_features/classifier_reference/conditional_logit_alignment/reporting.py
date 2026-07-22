"""Pure decision-summary and Markdown rendering for the CLA diagnostic."""

from __future__ import annotations

from collections import Counter
import math
from typing import Mapping, Sequence

from ..artifacts import stable_hash
from ..protocol import ProtocolError
from .schema import (
    CLA_CLAIM_SCOPE,
    DECISION_SUMMARY_SCHEMA_VERSION,
    PRIMARY_CONTRAST,
)


def build_decision_summary(
    outer_results: Sequence[Mapping[str, object]],
    outer_comparison: Sequence[Mapping[str, object]],
    gamma_summary: Sequence[Mapping[str, object]],
    *,
    design_hash: str,
    table_bundle_hash: str,
    protocol_hash: str,
    numerical_epsilon: float = 1e-12,
    pass_min_nonnegative_center_deltas: int = 5,
) -> dict[str, object]:
    """Recompute the non-adoptive scientific decision from persisted rows."""

    if not outer_results or not outer_comparison or not gamma_summary:
        raise ProtocolError("CLA decision summary requires all decision tables.")
    selected = [row for row in outer_results if row.get("evaluation_role") == "selected"]
    gamma0 = [row for row in outer_results if row.get("evaluation_role") == "gamma0"]
    selected_gamma_rows = [row for row in gamma_summary if _as_bool(row.get("selected"))]
    if (
        len(selected) != len(outer_comparison)
        or len(gamma0) != len(outer_comparison)
        or len(selected_gamma_rows) != len(outer_comparison)
    ):
        raise ProtocolError("CLA decision inputs have inconsistent held-out coverage.")

    deltas = [_finite_float(row.get("delta_bacc"), "delta_bacc") for row in outer_comparison]
    macro_deltas = [
        _finite_float(row.get("delta_macro_f1"), "delta_macro_f1")
        for row in outer_comparison
    ]
    mean_delta = sum(deltas) / float(len(deltas))
    minimum_selected_bacc = min(
        _finite_float(row.get("heldout_bacc"), "heldout_bacc")
        for row in selected
    )
    minimum_gamma0_bacc = min(
        _finite_float(row.get("heldout_bacc"), "heldout_bacc")
        for row in gamma0
    )
    nonnegative = sum(delta >= 0.0 for delta in deltas)
    pass_mean = mean_delta > float(numerical_epsilon)
    pass_minimum = minimum_selected_bacc >= (
        minimum_gamma0_bacc - float(numerical_epsilon)
    )
    pass_count = nonnegative >= int(pass_min_nonnegative_center_deltas)
    passed = bool(pass_mean and pass_minimum and pass_count)
    decision = (
        "PASS_DIAGNOSTIC_ONLY"
        if passed
        else (
            "WEAK_PASS_DIAGNOSTIC_ONLY"
            if pass_mean
            else "NEGATIVE_RESULT_DIAGNOSTIC_ONLY"
        )
    )
    gamma_distribution = Counter(
        _canonical_number(row.get("gamma")) for row in selected_gamma_rows
    )

    payload: dict[str, object] = {
        "schema_version": DECISION_SUMMARY_SCHEMA_VERSION,
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "decision": decision,
        "design_hash": str(design_hash),
        "table_bundle_hash": str(table_bundle_hash),
        "protocol_hash": str(protocol_hash),
        "primary_contrast": PRIMARY_CONTRAST,
        "primary_metric": "equal_center_mean_bacc",
        "n_heldout_centers": len(outer_comparison),
        "mean_selected_bacc": _mean(selected, "heldout_bacc"),
        "mean_gamma0_bacc": _mean(gamma0, "heldout_bacc"),
        "mean_delta_bacc": mean_delta,
        "minimum_selected_bacc": minimum_selected_bacc,
        "minimum_gamma0_bacc": minimum_gamma0_bacc,
        "minimum_bacc_margin": minimum_selected_bacc - minimum_gamma0_bacc,
        "nonnegative_center_delta_count": nonnegative,
        "mean_delta_macro_f1": sum(macro_deltas) / float(len(macro_deltas)),
        "pass_requires_positive_mean_delta": True,
        "pass_requires_nonworse_minimum_center": True,
        "pass_min_nonnegative_center_deltas": int(
            pass_min_nonnegative_center_deltas
        ),
        "numerical_epsilon": float(numerical_epsilon),
        "positive_mean_delta_passed": pass_mean,
        "nonworse_minimum_center_passed": pass_minimum,
        "nonnegative_center_count_passed": pass_count,
        "selected_gamma_distribution": dict(sorted(gamma_distribution.items())),
        "claim_scope": CLA_CLAIM_SCOPE,
        "diagnostic_only": True,
        "non_adoptive": True,
        "adoption_enabled": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "uses_generated_embeddings": False,
        "uses_cvae_checkpoint": False,
        "uses_router": False,
        "uses_expert_bank": False,
        "uses_nelbo": False,
        "forbidden_claims": [
            "causal_domain_shift_removal",
            "cvae_preservation",
            "prior_quality",
            "expert_compatibility",
            "routing_or_composition",
            "generation_or_synthetic_downstream_utility",
        ],
    }
    payload["decision_hash"] = stable_hash(payload)
    return payload


def render_decision_report(summary: Mapping[str, object]) -> str:
    """Render the deterministic, claim-disciplined human-readable report."""

    distribution = summary.get("selected_gamma_distribution")
    if not isinstance(distribution, Mapping):
        raise ProtocolError("CLA decision summary lacks selected gamma distribution.")
    gamma_text = ", ".join(
        f"`{gamma}`: {int(count)}" for gamma, count in distribution.items()
    )
    return "\n".join(
        [
            "# Conditional-Logit Alignment Diagnostic",
            "",
            f"Decision: `{summary['decision']}`.",
            "",
            "This is a diagnostic-only, non-adoptive Stage 10 real-feature result.",
            "",
            "| quantity | value |",
            "| --- | ---: |",
            f"| held-out centers | {int(summary['n_heldout_centers'])} |",
            f"| selected mean BACC | {float(summary['mean_selected_bacc']):.12f} |",
            f"| gamma-0 mean BACC | {float(summary['mean_gamma0_bacc']):.12f} |",
            f"| selected minus gamma-0 mean BACC | {float(summary['mean_delta_bacc']):.12f} |",
            f"| minimum selected-center BACC | {float(summary['minimum_selected_bacc']):.12f} |",
            f"| minimum gamma-0-center BACC | {float(summary['minimum_gamma0_bacc']):.12f} |",
            f"| minimum BACC margin | {float(summary['minimum_bacc_margin']):.12f} |",
            f"| nonnegative center deltas | {int(summary['nonnegative_center_delta_count'])} |",
            "",
            f"Selected gamma counts: {gamma_text}.",
            "",
            "Held-out-center labels were scoring-only. Gamma selection used only "
            "source-inner pseudo-target labels.",
            "",
            "This result cannot adopt a Stage 20+ recipe or policy and cannot "
            "support CVAE, prior, expert, NELBO, routing, composition, generation, "
            "or synthetic-utility claims.",
            "",
            f"Design hash: `{summary['design_hash']}`.",
            "",
            f"Table bundle hash: `{summary['table_bundle_hash']}`.",
            "",
            f"Protocol hash: `{summary['protocol_hash']}`.",
            "",
            f"Decision hash: `{summary['decision_hash']}`.",
            "",
        ]
    )


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    if not rows:
        raise ProtocolError(f"CLA decision has no rows for {field}.")
    return sum(_finite_float(row.get(field), field) for row in rows) / float(
        len(rows)
    )


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"CLA decision field {label} is not numeric.") from exc
    if not math.isfinite(number):
        raise ProtocolError(f"CLA decision field {label} is not finite.")
    return number


def _canonical_number(value: object) -> str:
    number = _finite_float(value, "gamma")
    return format(number, ".17g")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


__all__ = ["build_decision_summary", "render_decision_report"]
