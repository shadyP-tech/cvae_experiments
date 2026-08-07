"""Plan and assignment-table validation for the antisymmetric router."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.antisymmetric_residual_mmd import (
    LABEL_USE_SEMANTICS,
    PROXY_CLAIM_ROLE,
    ROBUST_OBJECTIVE_SEMANTICS,
    SOLVER_METHOD,
    WEIGHT_SEMANTICS,
    build_antisymmetric_allocation,
)
from .bundle_validation import _read_csv, _truthy
from .contracts import (
    ARM_ROLES,
    CENTERS,
    EXPECTED_CROSS_FIT_FOLD_COUNT,
    GENERATION_SEEDS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    candidate_sources,
)


def _validate_plans(
    plans: Mapping[str, Mapping[str, object]],
    *,
    crossfit: object,
) -> None:
    if len(plans) != EXPECTED_CROSS_FIT_FOLD_COUNT:
        raise ProtocolError("Antisymmetric plan count drifted.")
    workspace_hashes: dict[str, set[tuple[str, str, str, str, str]]] = {
        target: set() for target in CENTERS
    }
    for fold in crossfit.folds:
        plan = plans.get(fold.fold_id)
        if not isinstance(plan, Mapping):
            raise ProtocolError("Antisymmetric plan is missing for a fold.")
        candidates = candidate_sources(fold.target_center)
        uniform = {source: 1.0 / len(candidates) for source in candidates}
        delta = _float_mapping(plan.get("delta"), candidates)
        class_zero = _float_mapping(plan.get("class_0_weights"), candidates)
        class_one = _float_mapping(plan.get("class_1_weights"), candidates)
        control = _float_mapping(plan.get("control_weights"), candidates)
        solution = plan.get("solution")
        if not isinstance(solution, Mapping):
            raise ProtocolError("Antisymmetric solution payload is malformed.")
        if (
            plan.get("fold_hash") != fold.fold_hash
            or plan.get("target_center") != fold.target_center
            or plan.get("heldout_case_id") != fold.heldout_case_id
            or plan.get("heldout_row_identity_hash") != fold.heldout_row_identity_hash
            or plan.get("router_support_row_identity_hash")
            != fold.router_support_row_identity_hash
            or tuple(plan.get("router_support_case_ids", ()))
            != fold.router_support_case_ids
            or tuple(plan.get("candidate_sources", ())) != candidates
            or control != uniform
            or any(
                not np.isclose(
                    class_zero[source] + class_one[source],
                    2.0 * uniform[source],
                    atol=1e-10,
                    rtol=0.0,
                )
                for source in candidates
            )
            or any(
                not np.isclose(
                    class_zero[source] - uniform[source],
                    delta[source],
                    atol=1e-10,
                    rtol=0.0,
                )
                for source in candidates
            )
        ):
            raise ProtocolError(
                "Antisymmetric plan violates paired-weight semantics."
            )
        workspace_hash = tuple(
            str(plan.get(key, ""))
            for key in (
                "preprocessing_hash",
                "candidate_pool_fit_hash",
                "kernel_map_hash",
                "prior_model_hash",
                "prior_fit_pool_hash",
            )
        )
        if any(not value for value in workspace_hash):
            raise ProtocolError(
                "Antisymmetric target-workspace provenance is incomplete."
            )
        workspace_hashes[fold.target_center].add(workspace_hash)
        zero = np.asarray([class_zero[source] for source in candidates])
        one = np.asarray([class_one[source] for source in candidates])
        if (
            not np.isclose(zero.sum(), 1.0, atol=1e-10, rtol=0.0)
            or not np.isclose(one.sum(), 1.0, atol=1e-10, rtol=0.0)
            or min(zero.min(), one.min()) < -1e-10
            or max(zero.max(), one.max()) > 0.25 + 1e-10
            or np.abs(zero - 0.125).sum() > 0.25 + 1e-8
            or np.abs(one - 0.125).sum() > 0.25 + 1e-8
            or 1.0 / np.dot(zero, zero) < 6.0 - 1e-8
            or 1.0 / np.dot(one, one) < 6.0 - 1e-8
        ):
            raise ProtocolError("Antisymmetric plan escaped solver constraints.")
        allocation = build_antisymmetric_allocation(
            delta, total_per_class=TOTAL_PER_CLASS
        )
        routed = _allocations(plan.get("routed_allocations_by_class"), candidates)
        control_allocations = _allocations(
            plan.get("control_allocations_by_class"), candidates
        )
        expected_routed = {
            "0": dict(allocation.class_0_allocations),
            "1": dict(allocation.class_1_allocations),
        }
        if (
            routed != expected_routed
            or control_allocations
            != {
                str(label): {source: 128 for source in candidates}
                for label in (0, 1)
            }
            or plan.get("allocation_hash") != allocation.allocation_hash
            or bool(plan.get("used_uniform_fallback"))
            != (plan.get("fallback_reason") is not None)
            or plan.get("heldout_case_embeddings_used_for_own_route") is not False
            or plan.get("support_labels_used") is not False
            or plan.get("claim_role") != "label_free_proxy_compatibility_only"
            or plan.get("proxy_is_nelbo_compatibility") is not False
            or plan.get("proxy_is_downstream_utility") is not False
            or tuple(solution.get("candidate_sources", ())) != candidates
            or solution.get("solver_method") != SOLVER_METHOD
            or solution.get("weight_semantics") != WEIGHT_SEMANTICS
            or solution.get("objective_semantics") != ROBUST_OBJECTIVE_SEMANTICS
            or solution.get("claim_role") != PROXY_CLAIM_ROLE
            or solution.get("label_use_semantics") != LABEL_USE_SEMANTICS
            or any(
                solution.get(key) is not False
                for key in (
                    "labels_used",
                    "support_labels_used",
                    "target_labels_used",
                    "evaluation_labels_used",
                    "downstream_utility_used",
                    "downstream_utility_claimed",
                    "promotion_eligible",
                )
            )
        ):
            raise ProtocolError(
                "Antisymmetric plan allocation or claim boundary drifted."
            )
    if any(len(values) != 1 for values in workspace_hashes.values()):
        raise ProtocolError(
            "Antisymmetric target kernel workspace was not reused across case folds."
        )


def _validate_assignment_table(path: Path) -> None:
    rows = _read_csv(path)
    expected_count = (
        EXPECTED_CROSS_FIT_FOLD_COUNT
        * len(ARM_ROLES)
        * len(TRAINING_SEEDS)
        * len(GENERATION_SEEDS)
        * 2
        * 8
    )
    if len(rows) != expected_count:
        raise ProtocolError("Antisymmetric assignment-table row count drifted.")
    totals: dict[tuple[str, str, int, int, int], int] = {}
    for row in rows:
        key = (
            str(row["fold_id"]),
            str(row["arm_role"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
            int(row["class_label"]),
        )
        totals[key] = totals.get(key, 0) + int(row["prefix_count"])
        if (
            str(row["arm_role"]) not in ARM_ROLES
            or str(row["source_center"]) == str(row["target_center"])
            or not _truthy(row["target_expert_excluded"])
            or _truthy(row["seed_selected"])
            or not 0 < int(row["prefix_count"]) <= MAX_SOURCE_PREFIX_PER_CLASS
        ):
            raise ProtocolError("Antisymmetric assignment row escaped its boundary.")
    if (
        len(totals)
        != EXPECTED_CROSS_FIT_FOLD_COUNT * len(ARM_ROLES) * 9 * 2
        or set(totals.values()) != {TOTAL_PER_CLASS}
    ):
        raise ProtocolError("Antisymmetric assignment class totals drifted.")


def _float_mapping(
    value: object,
    candidates: Sequence[str],
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(str(key) for key in value) != set(
        candidates
    ):
        raise ProtocolError("Antisymmetric source-weight mapping is malformed.")
    parsed = {source: float(value[source]) for source in candidates}
    if any(not np.isfinite(number) for number in parsed.values()):
        raise ProtocolError("Antisymmetric source-weight mapping is nonfinite.")
    return parsed


def _allocations(
    value: object,
    candidates: Sequence[str],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise ProtocolError("Antisymmetric allocation mapping is malformed.")
    output: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        raw = value.get(str(label), value.get(label))
        if not isinstance(raw, Mapping) or set(str(key) for key in raw) != set(
            candidates
        ):
            raise ProtocolError("Antisymmetric class allocation is malformed.")
        output[str(label)] = {source: int(raw[source]) for source in candidates}
    return output


__all__: tuple[str, ...] = ()
