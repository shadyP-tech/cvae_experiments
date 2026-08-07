from __future__ import annotations

from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.conditional_contrast_mmd_router import (
    plan_validation,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SOURCES = ("source-a", "source-b")
CONTROL = {"source-a": 0.5, "source-b": 0.5}
CONTROL_ALLOCATION = {"source-a": 512, "source-b": 512}


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        proxy={
            "family": "class_conditional_contrast_mmd_kmm",
            "class_weights": [0.5, 0.5],
            "contrast_weight": 1.0,
            "maximum_uniform_l1": 0.25,
            "minimum_soft_class_mass_per_case": 1.0,
            "minimum_soft_class_effective_rows_per_case": 2.0,
            "conditional_component_nonworsening_required": True,
            "previous_pooled_mmd_output_used": False,
            "duplicate_direction_cosine": 0.999,
            "duplicate_weight_l1": 0.01,
        },
        conditional_contrast=SimpleNamespace(
            class_weights=(0.5, 0.5),
            contrast_weight=1.0,
            maximum_uniform_l1=0.25,
            minimum_soft_class_mass_per_case=1.0,
            minimum_soft_class_effective_rows_per_case=2.0,
            component_tolerance=1.0e-10,
        ),
    )


def _plan() -> dict[str, object]:
    routed = {"source-a": 0.6, "source-b": 0.4}
    diagnostics = {
        "proxy_family": "class_conditional_contrast_mmd_kmm",
        "class_weights": [0.5, 0.5],
        "contrast_weight": 1.0,
        "maximum_uniform_l1": 0.25,
        "observed_uniform_l1": 0.2,
        "routed_components": {
            "class_0_weighted_mmd_squared": 0.2,
            "class_1_weighted_mmd_squared": 0.2,
            "contrast_weighted_mmd_squared": 0.1,
            "conditional_discrepancy": 0.5,
        },
        "uniform_components": {
            "class_0_weighted_mmd_squared": 0.3,
            "class_1_weighted_mmd_squared": 0.3,
            "contrast_weighted_mmd_squared": 0.2,
            "conditional_discrepancy": 0.8,
        },
        "support_quality_passed": True,
        "component_nonworsening_passed": True,
        "casewise_component_improvement_passed": True,
        "soft_class_mass_by_case": {
            "case-a": [2.0, 2.0],
            "case-b": [1.5, 2.5],
        },
        "soft_class_effective_rows_by_case": {
            "case-a": [2.1, 2.2],
            "case-b": [2.3, 2.4],
        },
        "pooled_direction_cosine": 0.5,
        "pooled_weight_l1_distance": 0.4,
        "duplicate_pooled_direction": False,
        "previous_pooled_mmd_output_used": False,
        "conditional_fallback_reason": None,
    }
    return {
        "final_weights": routed,
        "control_weights": dict(CONTROL),
        "mmd_allocations_per_class": {"source-a": 614, "source-b": 410},
        "control_allocations_per_class": dict(CONTROL_ALLOCATION),
        "used_uniform_fallback": False,
        "fallback_reason": None,
        "base_solution": {
            "uniform_weights": dict(CONTROL),
            "weights": routed,
            "delta": {"source-a": 0.1, "source-b": -0.1},
            "mmd_squared": 0.5,
            "uniform_mmd_squared": 0.8,
            "used_uniform_fallback": False,
        },
        "conditional_contrast_diagnostics": diagnostics,
    }


def _validate(plan: dict[str, object]) -> None:
    plan_validation.validate_conditional_plan(
        plan,
        target="target",
        sources=SOURCES,
        final_weights=plan["final_weights"],
        control_weights=plan["control_weights"],
        final_allocations=plan["mmd_allocations_per_class"],
        control_allocations=plan["control_allocations_per_class"],
        used_uniform_fallback=plan["used_uniform_fallback"],
        expected_support_case_row_counts={"case-a": 4, "case-b": 4},
        config=_config(),
    )


def test_conditional_plan_validation_accepts_a_consistent_nonuniform_plan() -> None:
    _validate(_plan())


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda plan: plan["conditional_contrast_diagnostics"].update(
                {"previous_pooled_mmd_output_used": True}
            ),
            "objective declaration",
        ),
        (
            lambda plan: plan["conditional_contrast_diagnostics"].update(
                {"observed_uniform_l1": 0.1}
            ),
            "uniform-L1 audit",
        ),
        (
            lambda plan: plan["conditional_contrast_diagnostics"].update(
                {"component_nonworsening_passed": False}
            ),
            "component-nonworsening claim",
        ),
        (
            lambda plan: plan["conditional_contrast_diagnostics"].update(
                {"duplicate_pooled_direction": True}
            ),
            "pooled-direction claim",
        ),
    ),
)
def test_conditional_plan_validation_rejects_inconsistent_audits(
    mutation: object,
    message: str,
) -> None:
    plan = _plan()
    mutation(plan)
    with pytest.raises(ProtocolError, match=message):
        _validate(plan)


def test_failed_conditional_gate_requires_exact_equal_union_fallback() -> None:
    plan = _plan()
    diagnostics = plan["conditional_contrast_diagnostics"]
    diagnostics["soft_class_mass_by_case"]["case-a"][0] = 0.5
    diagnostics["soft_class_mass_by_case"]["case-a"][1] = 3.5
    diagnostics["support_quality_passed"] = False
    diagnostics["conditional_fallback_reason"] = (
        "insufficient_soft_class_support_uniform"
    )
    with pytest.raises(ProtocolError, match="fallback is not exact equal union"):
        _validate(plan)

    plan["final_weights"] = dict(CONTROL)
    plan["mmd_allocations_per_class"] = dict(CONTROL_ALLOCATION)
    plan["used_uniform_fallback"] = True
    plan["fallback_reason"] = "insufficient_soft_class_support_uniform"
    _validate(plan)
