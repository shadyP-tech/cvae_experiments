"""Immutable source-inner recipe selection for prior recovery."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.artifacts import stable_hash
from ...real_features.classifier_reference.protocol import ProtocolError
from ..generation_samplers import DIAGONAL_SAMPLER, FULL_SAMPLER, STANDARD_SAMPLER
from ..objectives import ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE


RECIPE_LOCK_SCHEMA_VERSION = "midogpp_prior_recovery_recipe_lock_v1"


@dataclass(frozen=True)
class InnerCenterMetric:
    outer_target_center: str
    inner_pseudo_target_center: str
    arm: str
    sampler_family: str
    objective_id: str
    prior_ratio: float
    decode_bacc: float
    posterior_bacc: float
    real_reference_bacc: float
    valid: bool = True
    task_fisher_valid: bool = True
    sampler_viable: bool = True
    realized_sampler_by_class: Mapping[str, str] | None = None
    fallback_reason_by_class: Mapping[str, str] | None = None


@dataclass(frozen=True)
class RecipeLock:
    outer_target_center: str
    status: str
    primary_arm: str
    objective_id: str
    sampler_family: str
    alpha: float
    beta_final: float
    generation_seeds: tuple[int, ...]
    inner_centers: tuple[str, ...]
    gate_summary: Mapping[str, object]
    classifier_grid_hash: str
    protocol_hash: str
    source_metric_table_hash: str
    fit_center_sets_hash: str
    recipe_contract_hash: str
    selection_bundle_hash: str
    reason: str = ""

    @property
    def may_feed_model_recipe(self) -> bool:
        return self.status == "VALID"

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload(include_hash=False))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": RECIPE_LOCK_SCHEMA_VERSION,
            "outer_target_center": self.outer_target_center,
            "status": self.status,
            "primary_arm": self.primary_arm,
            "objective_id": self.objective_id,
            "sampler_family": self.sampler_family,
            "alpha": self.alpha,
            "beta_final": self.beta_final,
            "generation_seeds": list(self.generation_seeds),
            "inner_centers": list(self.inner_centers),
            "gate_summary": dict(self.gate_summary),
            "classifier_grid_hash": self.classifier_grid_hash,
            "protocol_hash": self.protocol_hash,
            "source_metric_table_hash": self.source_metric_table_hash,
            "fit_center_sets_hash": self.fit_center_sets_hash,
            "recipe_contract_hash": self.recipe_contract_hash,
            "selection_bundle_hash": self.selection_bundle_hash,
            "reason": self.reason,
            "method": "fully_nested_prior_recovery_recipe_lock_v1",
            "claim_role": "cvae_recipe_lock",
            "row_role": "source_inner_selection",
            "leakage_status": "PASS" if self.status == "VALID" else "INVALID",
            "support_labels_used": False,
            "oracle_eligible": False,
            "selection_source": "fully_nested_source_inner",
            "source_inner_labels_used_for_selection": True,
            "target_eval_labels_used_for_scoring_only": False,
            "target_eval_labels_used_for_selection": False,
            "may_feed_model_recipe": self.may_feed_model_recipe,
            "may_feed_deployable_selection": False,
            "routing_performed": False,
            "composition_performed": False,
            "query_object": "none",
        }
        if include_hash:
            payload["recipe_lock_hash"] = self.hash
        return payload


def select_recipe_lock(
    metrics: Sequence[InnerCenterMetric],
    *,
    outer_target_center: str,
    expected_inner_centers: Sequence[str],
    generation_seeds: Sequence[int],
    beta_final: float,
    classifier_grid_hash: str,
    protocol_hash: str,
    fit_center_sets_hash: str,
    recipe_contract_hash: str,
    selection_bundle_hash: str,
    source_metric_table_hash: str,
    gate_min_ratio_improvement: float = 0.05,
    gate_min_inner_wins: int = 6,
    sampler_tie_margin: float = 0.01,
    task_increment_min_ratio: float = 0.01,
    safety_max_bacc_regression: float = 0.01,
    minimum_real_bacc: float = 0.55,
    require_task_factorial: bool = False,
) -> RecipeLock:
    outer = str(outer_target_center)
    expected = tuple(str(center) for center in expected_inner_centers)
    rows = [row for row in metrics if row.outer_target_center == outer]
    invalid_reason = _invalid_reason(rows, expected, minimum_real_bacc)
    if invalid_reason:
        return _lock(
            outer,
            status="INVALID",
            primary_arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler_family=STANDARD_SAMPLER,
            generation_seeds=generation_seeds,
            beta_final=beta_final,
            expected=expected,
            summary={},
            classifier_grid_hash=classifier_grid_hash,
            protocol_hash=protocol_hash,
            metric_hash=source_metric_table_hash,
            fit_center_sets_hash=fit_center_sets_hash,
            recipe_contract_hash=recipe_contract_hash,
            selection_bundle_hash=selection_bundle_hash,
            reason=invalid_reason,
        )
    baseline = _by_center(rows, arm="A", sampler_family=STANDARD_SAMPLER, expected=expected)
    candidates: list[tuple[str, dict[str, object]]] = []
    for family in (DIAGONAL_SAMPLER, FULL_SAMPLER):
        candidate = _by_center(rows, arm="C", sampler_family=family, expected=expected, allow_missing=True)
        if candidate is None or any(not row.valid or not row.sampler_viable for row in candidate.values()):
            continue
        summary = _paired_summary(candidate, baseline)
        summary["realized_sampler_by_inner"] = {
            center: dict(row.realized_sampler_by_class or {})
            for center, row in candidate.items()
        }
        summary["fallback_reason_by_inner"] = {
            center: dict(row.fallback_reason_by_class or {})
            for center, row in candidate.items()
        }
        if (
            float(summary["mean_delta"]) >= gate_min_ratio_improvement
            and int(summary["strict_wins"]) >= gate_min_inner_wins
        ):
            candidates.append((family, summary))
    if not candidates:
        return _lock(
            outer,
            status="VALID",
            primary_arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler_family=STANDARD_SAMPLER,
            generation_seeds=generation_seeds,
            beta_final=beta_final,
            expected=expected,
            summary={"sampler_gate": "NO_PASS"},
            classifier_grid_hash=classifier_grid_hash,
            protocol_hash=protocol_hash,
            metric_hash=source_metric_table_hash,
            fit_center_sets_hash=fit_center_sets_hash,
            recipe_contract_hash=recipe_contract_hash,
            selection_bundle_hash=selection_bundle_hash,
            reason="conditional_sampler_gate_not_met",
        )
    candidates.sort(key=lambda item: (-float(item[1]["mean_delta"]), item[0] != DIAGONAL_SAMPLER))
    selected_family, sampler_summary = candidates[0]
    if len(candidates) > 1 and abs(float(candidates[0][1]["mean_delta"]) - float(candidates[1][1]["mean_delta"])) <= sampler_tie_margin:
        selected_family = DIAGONAL_SAMPLER
        sampler_summary = next(summary for family, summary in candidates if family == DIAGONAL_SAMPLER)
    conditional = _by_center(rows, arm="C", sampler_family=selected_family, expected=expected)
    task_standard = _by_center(rows, arm="B", sampler_family=STANDARD_SAMPLER, expected=expected, allow_missing=True)
    task_conditional = _by_center(rows, arm="D", sampler_family=selected_family, expected=expected, allow_missing=True)
    task_valid = task_standard is not None and task_conditional is not None and all(
        row.task_fisher_valid and row.valid and row.sampler_viable
        for row in rows
        if row.arm in {"B", "D"}
    )
    if require_task_factorial and not task_valid:
        return _lock(
            outer,
            status="INVALID",
            primary_arm="A",
            objective_id=ISOTROPIC_OBJECTIVE,
            sampler_family=STANDARD_SAMPLER,
            generation_seeds=generation_seeds,
            beta_final=beta_final,
            expected=expected,
            summary={"sampler": sampler_summary, "task_fisher": {"eligible": False}},
            classifier_grid_hash=classifier_grid_hash,
            protocol_hash=protocol_hash,
            metric_hash=source_metric_table_hash,
            fit_center_sets_hash=fit_center_sets_hash,
            recipe_contract_hash=recipe_contract_hash,
            selection_bundle_hash=selection_bundle_hash,
            reason="incomplete_task_fisher_factorial",
        )
    selected_arm = "C"
    selected_objective = ISOTROPIC_OBJECTIVE
    task_summary: dict[str, object] = {"eligible": False}
    if task_valid:
        d_vs_a = _paired_summary(task_conditional, baseline)
        d_vs_c = _paired_summary(task_conditional, conditional)
        decode_regression = _mean(row.decode_bacc for row in task_standard.values()) - _mean(
            row.decode_bacc for row in baseline.values()
        )
        posterior_regression = _mean(row.posterior_bacc for row in task_standard.values()) - _mean(
            row.posterior_bacc for row in baseline.values()
        )
        safety = decode_regression >= -safety_max_bacc_regression and posterior_regression >= -safety_max_bacc_regression
        task_summary = {
            "eligible": True,
            "d_vs_a": d_vs_a,
            "d_vs_c": d_vs_c,
            "b_vs_a_decode_delta": decode_regression,
            "b_vs_a_posterior_delta": posterior_regression,
            "safety_pass": safety,
        }
        if (
            float(d_vs_a["mean_delta"]) >= gate_min_ratio_improvement
            and int(d_vs_a["strict_wins"]) >= gate_min_inner_wins
            and float(d_vs_c["mean_delta"]) > task_increment_min_ratio
            and safety
        ):
            selected_arm = "D"
            selected_objective = TASK_FISHER_OBJECTIVE
    return _lock(
        outer,
        status="VALID",
        primary_arm=selected_arm,
        objective_id=selected_objective,
        sampler_family=selected_family,
        generation_seeds=generation_seeds,
        beta_final=beta_final,
        expected=expected,
        summary={"sampler": sampler_summary, "task_fisher": task_summary},
        classifier_grid_hash=classifier_grid_hash,
        protocol_hash=protocol_hash,
        metric_hash=source_metric_table_hash,
        fit_center_sets_hash=fit_center_sets_hash,
        recipe_contract_hash=recipe_contract_hash,
        selection_bundle_hash=selection_bundle_hash,
        reason="",
    )


def write_recipe_lock(path: Path, lock: RecipeLock) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_recipe_lock(path: Path) -> RecipeLock:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError("RecipeLock payload must be a mapping.")
    return recipe_lock_from_payload(payload)


def recipe_lock_from_payload(payload: Mapping[str, object]) -> RecipeLock:
    if payload.get("schema_version") != RECIPE_LOCK_SCHEMA_VERSION:
        raise ProtocolError("Unexpected RecipeLock schema version.")
    lock = RecipeLock(
        outer_target_center=str(payload["outer_target_center"]),
        status=str(payload["status"]),
        primary_arm=str(payload["primary_arm"]),
        objective_id=str(payload["objective_id"]),
        sampler_family=str(payload["sampler_family"]),
        alpha=float(payload["alpha"]),
        beta_final=float(payload["beta_final"]),
        generation_seeds=tuple(int(value) for value in payload["generation_seeds"]),
        inner_centers=tuple(str(value) for value in payload["inner_centers"]),
        gate_summary=dict(payload["gate_summary"]),
        classifier_grid_hash=str(payload["classifier_grid_hash"]),
        protocol_hash=str(payload["protocol_hash"]),
        source_metric_table_hash=str(payload["source_metric_table_hash"]),
        fit_center_sets_hash=str(payload["fit_center_sets_hash"]),
        recipe_contract_hash=str(payload["recipe_contract_hash"]),
        selection_bundle_hash=str(payload["selection_bundle_hash"]),
        reason=str(payload.get("reason", "")),
    )
    if str(payload.get("recipe_lock_hash")) != lock.hash:
        raise ProtocolError("RecipeLock hash mismatch.")
    return lock


def _invalid_reason(rows: Sequence[InnerCenterMetric], expected: Sequence[str], minimum_real_bacc: float) -> str:
    if not rows:
        return "no_source_inner_metrics"
    observed = {row.inner_pseudo_target_center for row in rows if row.arm == "A" and row.sampler_family == STANDARD_SAMPLER}
    if observed != set(expected):
        return "incomplete_inner_center_coverage"
    baseline = [
        row for row in rows if row.arm == "A" and row.sampler_family == STANDARD_SAMPLER
    ]
    if any(not row.valid for row in baseline):
        return "invalid_baseline_metric_row"
    if any(
        not math.isfinite(row.real_reference_bacc)
        or row.real_reference_bacc < minimum_real_bacc
        for row in baseline
    ):
        return "invalid_real_reference_denominator"
    return ""


def _by_center(
    rows: Sequence[InnerCenterMetric],
    *,
    arm: str,
    sampler_family: str,
    expected: Sequence[str],
    allow_missing: bool = False,
) -> dict[str, InnerCenterMetric] | None:
    selected = {
        row.inner_pseudo_target_center: row
        for row in rows
        if row.arm == arm and row.sampler_family == sampler_family
    }
    if set(selected) != set(expected):
        if allow_missing:
            return None
        raise ProtocolError(f"Incomplete {arm}/{sampler_family} inner-center coverage.")
    return selected


def _paired_summary(
    candidate: Mapping[str, InnerCenterMetric],
    baseline: Mapping[str, InnerCenterMetric],
) -> dict[str, object]:
    deltas = {center: candidate[center].prior_ratio - baseline[center].prior_ratio for center in baseline}
    return {
        "mean_delta": _mean(deltas.values()),
        "strict_wins": sum(value > 0.0 for value in deltas.values()),
        "center_deltas": deltas,
    }


def _mean(values: Sequence[float] | object) -> float:
    items = [float(value) for value in values]  # type: ignore[arg-type]
    return sum(items) / float(len(items))


def _lock(
    outer: str,
    *,
    status: str,
    primary_arm: str,
    objective_id: str,
    sampler_family: str,
    generation_seeds: Sequence[int],
    beta_final: float,
    expected: Sequence[str],
    summary: Mapping[str, object],
    classifier_grid_hash: str,
    protocol_hash: str,
    metric_hash: str,
    fit_center_sets_hash: str,
    recipe_contract_hash: str,
    selection_bundle_hash: str,
    reason: str,
) -> RecipeLock:
    return RecipeLock(
        outer_target_center=outer,
        status=status,
        primary_arm=primary_arm,
        objective_id=objective_id,
        sampler_family=sampler_family,
        alpha=1.0 if objective_id == TASK_FISHER_OBJECTIVE else 0.0,
        beta_final=float(beta_final),
        generation_seeds=tuple(int(value) for value in generation_seeds),
        inner_centers=tuple(expected),
        gate_summary=dict(summary),
        classifier_grid_hash=classifier_grid_hash,
        protocol_hash=protocol_hash,
        source_metric_table_hash=metric_hash,
        fit_center_sets_hash=fit_center_sets_hash,
        recipe_contract_hash=recipe_contract_hash,
        selection_bundle_hash=selection_bundle_hash,
        reason=reason,
    )
