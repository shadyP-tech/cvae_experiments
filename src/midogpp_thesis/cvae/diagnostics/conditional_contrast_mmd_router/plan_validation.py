"""Fail-closed plan validation for conditional contrast-MMD routing."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ..mmd_kmm_router.config import MMDKMMRouterDiagnosticConfig
from ..mmd_kmm_router.contracts import SUPPORT_CASE_COUNT


_CONDITIONAL_DIAGNOSTIC_KEYS = frozenset(
    {
        "proxy_family",
        "class_weights",
        "contrast_weight",
        "maximum_uniform_l1",
        "observed_uniform_l1",
        "routed_components",
        "uniform_components",
        "support_quality_passed",
        "component_nonworsening_passed",
        "casewise_component_improvement_passed",
        "soft_class_mass_by_case",
        "soft_class_effective_rows_by_case",
        "pooled_direction_cosine",
        "pooled_weight_l1_distance",
        "duplicate_pooled_direction",
        "previous_pooled_mmd_output_used",
        "conditional_fallback_reason",
    }
)
_CONDITIONAL_COMPONENT_KEYS = (
    "class_0_weighted_mmd_squared",
    "class_1_weighted_mmd_squared",
    "contrast_weighted_mmd_squared",
)
_CONDITIONAL_COMPONENT_TOTAL = "conditional_discrepancy"


def validate_conditional_plan(
    plan: Mapping[str, object],
    *,
    target: str,
    sources: tuple[str, ...],
    final_weights: Mapping[str, float],
    control_weights: Mapping[str, float],
    final_allocations: Mapping[str, int],
    control_allocations: Mapping[str, int],
    used_uniform_fallback: bool,
    expected_support_case_row_counts: Mapping[str, int],
    config: MMDKMMRouterDiagnosticConfig,
) -> None:
    raw = plan.get("conditional_contrast_diagnostics")
    if not isinstance(raw, Mapping) or set(raw) != _CONDITIONAL_DIAGNOSTIC_KEYS:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} diagnostics are incomplete."
        )
    conditional = config.conditional_contrast
    frozen_proxy = {
        "family": "class_conditional_contrast_mmd_kmm",
        "class_weights": [0.5, 0.5],
        "contrast_weight": 1.0,
        "maximum_uniform_l1": 0.25,
        "minimum_soft_class_mass_per_case": 1.0,
        "minimum_soft_class_effective_rows_per_case": 2.0,
        "conditional_component_nonworsening_required": True,
        "previous_pooled_mmd_output_used": False,
    }
    if any(config.proxy.get(key) != value for key, value in frozen_proxy.items()):
        raise ProtocolError("Conditional MMD/KMM frozen proxy config drifted.")
    try:
        reported_class_weights = tuple(float(value) for value in raw["class_weights"])
    except (TypeError, ValueError):
        reported_class_weights = ()
    if (
        raw.get("proxy_family") != frozen_proxy["family"]
        or reported_class_weights != tuple(conditional.class_weights)
        or not _numeric_equal(raw.get("contrast_weight"), conditional.contrast_weight)
        or not _numeric_equal(
            raw.get("maximum_uniform_l1"), conditional.maximum_uniform_l1
        )
        or raw.get("previous_pooled_mmd_output_used") is not False
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} objective declaration drifted."
        )

    expected_cases = tuple(sorted(str(value) for value in expected_support_case_row_counts))
    if (
        len(expected_cases) != SUPPORT_CASE_COUNT
        or len(set(expected_cases)) != SUPPORT_CASE_COUNT
        or any(not case_id for case_id in expected_cases)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in expected_support_case_row_counts.values()
        )
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} expected support cases drifted."
        )
    soft_mass = _conditional_pair_mapping(
        raw.get("soft_class_mass_by_case"),
        role=f"target {target} soft class mass",
    )
    soft_effective = _conditional_pair_mapping(
        raw.get("soft_class_effective_rows_by_case"),
        role=f"target {target} soft class effective rows",
    )
    if (
        tuple(sorted(soft_mass)) != expected_cases
        or tuple(sorted(soft_effective)) != expected_cases
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} support-case coverage drifted."
        )
    for case_id in expected_cases:
        row_count = int(expected_support_case_row_counts[case_id])
        mass_pair = soft_mass[case_id]
        effective_pair = soft_effective[case_id]
        if (
            not np.isclose(sum(mass_pair), row_count, atol=1e-8, rtol=0.0)
            or any(value > row_count + 1e-8 for value in mass_pair)
            or any(
                value < 1.0 - 1e-8 or value > row_count + 1e-8
                for value in effective_pair
            )
        ):
            raise ProtocolError(
                f"Conditional MMD/KMM target {target} support-quality values are impossible."
            )
    reconstructed_support_quality = all(
        value >= conditional.minimum_soft_class_mass_per_case
        for pair in soft_mass.values()
        for value in pair
    ) and all(
        value >= conditional.minimum_soft_class_effective_rows_per_case
        for pair in soft_effective.values()
        for value in pair
    )
    reported_support_quality = _strict_bool(
        raw.get("support_quality_passed"),
        f"target {target} conditional support-quality flag",
    )
    if reported_support_quality != reconstructed_support_quality:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} support-quality claim drifted."
        )

    base = plan.get("base_solution")
    if not isinstance(base, Mapping):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} base solution is malformed."
        )
    base_uniform = _conditional_weight_mapping(
        base.get("uniform_weights"), sources, f"target {target} base uniform weights"
    )
    base_weights = _conditional_weight_mapping(
        base.get("weights"), sources, f"target {target} base routed weights"
    )
    base_delta = _conditional_delta_mapping(
        base.get("delta"), sources, f"target {target} base weight delta"
    )
    if any(
        not np.isclose(base_uniform[source], control_weights[source], atol=1e-12, rtol=0.0)
        or not np.isclose(
            base_delta[source],
            base_weights[source] - base_uniform[source],
            atol=1e-12,
            rtol=0.0,
        )
        for source in sources
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} base-solution anchor drifted."
        )
    observed_l1 = _finite_float(
        raw.get("observed_uniform_l1"), f"target {target} observed uniform L1"
    )
    reconstructed_l1 = float(
        sum(abs(base_weights[source] - base_uniform[source]) for source in sources)
    )
    if not np.isclose(observed_l1, reconstructed_l1, atol=1e-12, rtol=1e-12):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} uniform-L1 audit drifted."
        )

    routed_components = _conditional_component_mapping(
        raw.get("routed_components"), f"target {target} routed components"
    )
    uniform_components = _conditional_component_mapping(
        raw.get("uniform_components"), f"target {target} uniform components"
    )
    tolerance = float(conditional.component_tolerance)
    reconstructed_nonworsening = all(
        routed_components[key] <= uniform_components[key] + tolerance
        for key in _CONDITIONAL_COMPONENT_KEYS
    )
    reported_nonworsening = _strict_bool(
        raw.get("component_nonworsening_passed"),
        f"target {target} conditional component-nonworsening flag",
    )
    if reported_nonworsening != reconstructed_nonworsening:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} component-nonworsening claim drifted."
        )
    for components, role in (
        (routed_components, "routed"),
        (uniform_components, "uniform"),
    ):
        component_sum = sum(components[key] for key in _CONDITIONAL_COMPONENT_KEYS)
        if not np.isclose(
            components[_CONDITIONAL_COMPONENT_TOTAL],
            component_sum,
            atol=max(tolerance, 1e-12),
            rtol=1e-10,
        ):
            raise ProtocolError(
                f"Conditional MMD/KMM target {target} {role} component sum drifted."
            )
    if not np.isclose(
        _finite_float(base.get("mmd_squared"), f"target {target} base MMD"),
        routed_components[_CONDITIONAL_COMPONENT_TOTAL],
        atol=max(tolerance, 1e-12),
        rtol=1e-10,
    ) or not np.isclose(
        _finite_float(
            base.get("uniform_mmd_squared"), f"target {target} uniform base MMD"
        ),
        uniform_components[_CONDITIONAL_COMPONENT_TOTAL],
        atol=max(tolerance, 1e-12),
        rtol=1e-10,
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} component/MMD identity drifted."
        )

    casewise_passed = _strict_bool(
        raw.get("casewise_component_improvement_passed"),
        f"target {target} casewise component-improvement flag",
    )
    base_used_uniform = _strict_bool(
        base.get("used_uniform_fallback"),
        f"target {target} base uniform-fallback flag",
    )
    base_reason = base.get("fallback_reason")
    if base_used_uniform != (
        isinstance(base_reason, str) and bool(base_reason.strip())
    ) or base_used_uniform and any(
        not np.isclose(
            base_weights[source], base_uniform[source], atol=1e-12, rtol=0.0
        )
        for source in sources
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} base fallback drifted."
        )
    if base_used_uniform and casewise_passed:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} casewise claim contradicts its uniform base."
        )
    pooled_cosine = _finite_float(
        raw.get("pooled_direction_cosine"),
        f"target {target} pooled-direction cosine",
    )
    pooled_l1 = _finite_float(
        raw.get("pooled_weight_l1_distance"),
        f"target {target} pooled-weight L1",
    )
    if (
        pooled_cosine < -1.0 - 1e-12
        or pooled_cosine > 1.0 + 1e-12
        or not 0.0 <= pooled_l1 <= 2.0 + 1e-12
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} pooled-direction audit is invalid."
        )
    duplicate_pooled = _strict_bool(
        raw.get("duplicate_pooled_direction"),
        f"target {target} duplicate pooled-direction flag",
    )
    reconstructed_duplicate = not base_used_uniform and (
        pooled_cosine >= float(config.proxy["duplicate_direction_cosine"])
        or pooled_l1 <= float(config.proxy["duplicate_weight_l1"])
    )
    if duplicate_pooled != reconstructed_duplicate:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} pooled-direction claim drifted."
        )

    reason = raw.get("conditional_fallback_reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} fallback reason is malformed."
        )
    if base_used_uniform and reason != base_reason:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} did not preserve its base fallback reason."
        )
    trust_region_passed = observed_l1 <= conditional.maximum_uniform_l1
    conditional_gate_failed = (
        not reconstructed_support_quality
        or not trust_region_passed
        or not reconstructed_nonworsening
        or not casewise_passed
        or reconstructed_duplicate
    )
    if conditional_gate_failed and reason is None:
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} failed a gate without abstaining."
        )
    if reason is not None:
        if (
            not used_uniform_fallback
            or plan.get("fallback_reason") != reason
            or final_weights != control_weights
            or final_allocations != control_allocations
        ):
            raise ProtocolError(
                f"Conditional MMD/KMM target {target} fallback is not exact equal union."
            )
    elif not used_uniform_fallback and any(
        not np.isclose(final_weights[source], base_weights[source], atol=1e-12, rtol=0.0)
        for source in sources
    ):
        raise ProtocolError(
            f"Conditional MMD/KMM target {target} accepted weights drifted from its base solution."
        )


def _conditional_pair_mapping(
    value: object, *, role: str
) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Conditional MMD/KMM {role} is malformed.")
    output: dict[str, tuple[float, float]] = {}
    for key, raw_pair in value.items():
        if not isinstance(key, str) or not key:
            raise ProtocolError(f"Conditional MMD/KMM {role} has an invalid case id.")
        if not isinstance(raw_pair, (list, tuple)):
            raise ProtocolError(f"Conditional MMD/KMM {role} has invalid values.")
        try:
            pair = tuple(float(item) for item in raw_pair)
        except (TypeError, ValueError):
            pair = ()
        if len(pair) != 2 or any(
            not np.isfinite(item) or item <= 0.0 for item in pair
        ):
            raise ProtocolError(f"Conditional MMD/KMM {role} has invalid values.")
        output[key] = (pair[0], pair[1])
    return output


def _conditional_component_mapping(value: object, role: str) -> dict[str, float]:
    expected = {*_CONDITIONAL_COMPONENT_KEYS, _CONDITIONAL_COMPONENT_TOTAL}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProtocolError(f"Conditional MMD/KMM {role} is malformed.")
    output = {key: _finite_float(value[key], f"{role} {key}") for key in expected}
    if any(item < 0.0 for item in output.values()):
        raise ProtocolError(f"Conditional MMD/KMM {role} contains a negative loss.")
    return output


def _conditional_weight_mapping(
    value: object, sources: tuple[str, ...], role: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != sources:
        raise ProtocolError(f"Conditional MMD/KMM {role} source coverage drifted.")
    output = {
        source: _finite_float(value[source], f"{role} {source}")
        for source in sources
    }
    vector = np.asarray([output[source] for source in sources])
    if np.any(vector < 0.0) or not np.isclose(
        vector.sum(), 1.0, atol=1e-10, rtol=0.0
    ):
        raise ProtocolError(f"Conditional MMD/KMM {role} violates the simplex.")
    return output


def _conditional_delta_mapping(
    value: object, sources: tuple[str, ...], role: str
) -> dict[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != sources:
        raise ProtocolError(f"Conditional MMD/KMM {role} source coverage drifted.")
    return {
        source: _finite_float(value[source], f"{role} {source}")
        for source in sources
    }


def _finite_float(value: object, role: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"Conditional MMD/KMM {role} is not numeric.")
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Conditional MMD/KMM {role} is not numeric.") from exc
    if not np.isfinite(output):
        raise ProtocolError(f"Conditional MMD/KMM {role} is not finite.")
    return output


def _numeric_equal(value: object, expected: float) -> bool:
    try:
        return not isinstance(value, bool) and float(value) == float(expected)
    except (TypeError, ValueError):
        return False


def _strict_bool(value: object, role: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"MMD/KMM {role} is not boolean.")
    return value


__all__ = ("validate_conditional_plan",)
