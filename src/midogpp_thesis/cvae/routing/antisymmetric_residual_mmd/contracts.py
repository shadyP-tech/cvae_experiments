"""Fail-closed contracts for label-free antisymmetric residual MMD routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError


PROXY_CLAIM_ROLE = "proxy_compatibility_only"
LABEL_USE_SEMANTICS = "label_free_unlabeled_target_support_only"
WEIGHT_SEMANTICS = "uniform_anchored_antisymmetric_binary_class_residual"
ROBUST_OBJECTIVE_SEMANTICS = (
    "variant_equal_mean_conditional_loss_plus_worst_variant_epigraph_plus_l2"
)
SOLVER_METHOD = "scipy_slsqp_epigraph_with_exact_l1_halfspaces"

DEFAULT_WORST_VARIANT_PENALTY = 1.0
DEFAULT_L2_SHRINKAGE = 0.01
DEFAULT_MAX_SOURCE_WEIGHT = 0.25
DEFAULT_MIN_EFFECTIVE_SOURCES = 6.0
DEFAULT_MAX_UNIFORM_L1 = 0.25
DEFAULT_MIN_SOFT_CLASS_MASS_PER_CASE = 1.0
DEFAULT_MIN_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE = 1.0
DEFAULT_MIN_ROBUST_IMPROVEMENT = 1.0e-10
DEFAULT_VARIANT_WORSENING_TOLERANCE = 1.0e-10
DEFAULT_SOLVER_TOLERANCE = 1.0e-12
DEFAULT_MAX_ITERATIONS = 2000


@dataclass(frozen=True)
class AntisymmetricResidualConfig:
    """Scientific and numerical controls for the pure proxy optimizer.

    The density limits may only be made stricter than the thesis defaults.
    In particular, callers cannot relax the per-source cap above ``0.25``, the
    effective-source floor below ``6``, or the per-class uniform L1 radius
    above ``0.25``.
    """

    worst_variant_penalty: float = DEFAULT_WORST_VARIANT_PENALTY
    l2_shrinkage: float = DEFAULT_L2_SHRINKAGE
    max_source_weight: float = DEFAULT_MAX_SOURCE_WEIGHT
    minimum_effective_sources: float = DEFAULT_MIN_EFFECTIVE_SOURCES
    maximum_uniform_l1: float = DEFAULT_MAX_UNIFORM_L1
    minimum_soft_class_mass_per_case: float = (
        DEFAULT_MIN_SOFT_CLASS_MASS_PER_CASE
    )
    minimum_soft_class_effective_rows_per_case: float = (
        DEFAULT_MIN_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE
    )
    minimum_robust_improvement: float = DEFAULT_MIN_ROBUST_IMPROVEMENT
    variant_worsening_tolerance: float = DEFAULT_VARIANT_WORSENING_TOLERANCE
    solver_tolerance: float = DEFAULT_SOLVER_TOLERANCE
    max_iterations: int = DEFAULT_MAX_ITERATIONS

    def __post_init__(self) -> None:
        try:
            numeric = {
                "worst_variant_penalty": float(self.worst_variant_penalty),
                "l2_shrinkage": float(self.l2_shrinkage),
                "max_source_weight": float(self.max_source_weight),
                "minimum_effective_sources": float(
                    self.minimum_effective_sources
                ),
                "maximum_uniform_l1": float(self.maximum_uniform_l1),
                "minimum_soft_class_mass_per_case": float(
                    self.minimum_soft_class_mass_per_case
                ),
                "minimum_soft_class_effective_rows_per_case": float(
                    self.minimum_soft_class_effective_rows_per_case
                ),
                "minimum_robust_improvement": float(
                    self.minimum_robust_improvement
                ),
                "variant_worsening_tolerance": float(
                    self.variant_worsening_tolerance
                ),
                "solver_tolerance": float(self.solver_tolerance),
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError(
                "Antisymmetric residual MMD configuration is invalid."
            ) from exc
        if (
            any(not math.isfinite(value) for value in numeric.values())
            or numeric["worst_variant_penalty"] < 0.0
            or numeric["l2_shrinkage"] < 0.0
            or not (
                0.0
                < numeric["max_source_weight"]
                <= DEFAULT_MAX_SOURCE_WEIGHT
            )
            or numeric["minimum_effective_sources"]
            < DEFAULT_MIN_EFFECTIVE_SOURCES
            or not (
                0.0
                < numeric["maximum_uniform_l1"]
                <= DEFAULT_MAX_UNIFORM_L1
            )
            or numeric["minimum_soft_class_mass_per_case"] <= 0.0
            or numeric["minimum_soft_class_effective_rows_per_case"] <= 0.0
            or numeric["minimum_robust_improvement"] < 0.0
            or numeric["variant_worsening_tolerance"] < 0.0
            or numeric["solver_tolerance"] <= 0.0
            or isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or int(self.max_iterations) <= 0
        ):
            raise ProtocolError(
                "Antisymmetric residual MMD configuration is invalid."
            )
        for field_name, value in numeric.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "max_iterations", int(self.max_iterations))


@dataclass(frozen=True)
class AntisymmetricVariantDiagnostic:
    """Conditional proxy loss comparison for one named robust variant."""

    axis: str
    variant_id: str
    uniform_components: Mapping[str, float]
    proposed_components: Mapping[str, float]
    final_components: Mapping[str, float]
    proposed_improvement: float
    final_improvement: float
    proposed_worsened: bool
    claim_role: str = PROXY_CLAIM_ROLE
    labels_used: bool = False

    def __post_init__(self) -> None:
        if (
            not str(self.axis)
            or not str(self.variant_id)
            or self.claim_role != PROXY_CLAIM_ROLE
            or self.labels_used is not False
        ):
            raise ProtocolError("Antisymmetric variant diagnostic is invalid.")
        object.__setattr__(
            self,
            "uniform_components",
            MappingProxyType(_component_mapping(self.uniform_components)),
        )
        object.__setattr__(
            self,
            "proposed_components",
            MappingProxyType(_component_mapping(self.proposed_components)),
        )
        object.__setattr__(
            self,
            "final_components",
            MappingProxyType(_component_mapping(self.final_components)),
        )


@dataclass(frozen=True)
class AntisymmetricAxisDiagnostic:
    """Aggregate robust-loss audit for one perturbation axis."""

    axis: str
    variant_ids: tuple[str, ...]
    uniform_mean_loss: float
    proposed_mean_loss: float
    final_mean_loss: float
    uniform_worst_loss: float
    proposed_worst_loss: float
    final_worst_loss: float
    minimum_proposed_variant_improvement: float
    maximum_proposed_variant_worsening: float
    all_proposed_variants_nonworsening: bool
    claim_role: str = PROXY_CLAIM_ROLE
    labels_used: bool = False


@dataclass(frozen=True)
class AntisymmetricResidualSolution:
    """Auditable class-paired weights and robust proxy diagnostics."""

    candidate_sources: tuple[str, ...]
    uniform_weights: Mapping[str, float]
    proposed_delta: Mapping[str, float]
    proposed_class_0_weights: Mapping[str, float]
    proposed_class_1_weights: Mapping[str, float]
    delta: Mapping[str, float]
    class_0_weights: Mapping[str, float]
    class_1_weights: Mapping[str, float]
    robust_objective: float
    uniform_robust_objective: float
    proposed_robust_improvement: float
    robust_improvement: float
    proposed_mean_conditional_loss: float
    final_mean_conditional_loss: float
    uniform_mean_conditional_loss: float
    proposed_worst_conditional_loss: float
    final_worst_conditional_loss: float
    uniform_worst_conditional_loss: float
    l2_penalty_value: float
    class_0_effective_source_count: float
    class_1_effective_source_count: float
    maximum_source_weight: float
    class_0_uniform_l1: float
    class_1_uniform_l1: float
    used_uniform_fallback: bool
    fallback_reason: str | None
    solver_success: bool
    solver_message: str
    solver_iterations: int
    solver_version: str
    variant_diagnostics: tuple[AntisymmetricVariantDiagnostic, ...]
    axis_diagnostics: Mapping[str, AntisymmetricAxisDiagnostic]
    support_quality_passed: bool
    all_variants_nonworsening: bool
    solver_method: str = SOLVER_METHOD
    weight_semantics: str = WEIGHT_SEMANTICS
    objective_semantics: str = ROBUST_OBJECTIVE_SEMANTICS
    claim_role: str = PROXY_CLAIM_ROLE
    label_use_semantics: str = LABEL_USE_SEMANTICS
    labels_used: bool = False
    support_labels_used: bool = False
    target_labels_used: bool = False
    evaluation_labels_used: bool = False
    downstream_utility_used: bool = False
    downstream_utility_claimed: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        sources = tuple(str(source) for source in self.candidate_sources)
        if (
            not sources
            or len(set(sources)) != len(sources)
            or self.solver_method != SOLVER_METHOD
            or self.weight_semantics != WEIGHT_SEMANTICS
            or self.objective_semantics != ROBUST_OBJECTIVE_SEMANTICS
            or self.claim_role != PROXY_CLAIM_ROLE
            or self.label_use_semantics != LABEL_USE_SEMANTICS
            or self.labels_used is not False
            or self.support_labels_used is not False
            or self.target_labels_used is not False
            or self.evaluation_labels_used is not False
            or self.downstream_utility_used is not False
            or self.downstream_utility_claimed is not False
            or self.promotion_eligible is not False
        ):
            raise ProtocolError("Antisymmetric residual solution is invalid.")
        for field_name in (
            "uniform_weights",
            "proposed_delta",
            "proposed_class_0_weights",
            "proposed_class_1_weights",
            "delta",
            "class_0_weights",
            "class_1_weights",
        ):
            values = _source_mapping(getattr(self, field_name), sources, field_name)
            object.__setattr__(self, field_name, MappingProxyType(values))
        axes = {str(key): value for key, value in self.axis_diagnostics.items()}
        if set(axes) != {diagnostic.axis for diagnostic in self.variant_diagnostics}:
            raise ProtocolError("Antisymmetric axis diagnostic coverage drifted.")
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "axis_diagnostics", MappingProxyType(axes))

    @property
    def weights_by_class(self) -> Mapping[int, Mapping[str, float]]:
        """Return the two final mappings without implying label consumption."""

        return MappingProxyType(
            {0: self.class_0_weights, 1: self.class_1_weights}
        )

    @property
    def d(self) -> Mapping[str, float]:
        """Short mathematical alias for the final antisymmetric residual."""

        return self.delta

    @property
    def proxy_only(self) -> bool:
        return True

    @property
    def label_free(self) -> bool:
        return True

    @property
    def base_diagnostic(self) -> AntisymmetricVariantDiagnostic:
        for diagnostic in self.variant_diagnostics:
            if diagnostic.axis == "base":
                return diagnostic
        raise ProtocolError("Antisymmetric solution has no base diagnostic.")


_COMPONENT_KEYS = (
    "class_0_weighted_mmd_squared",
    "class_1_weighted_mmd_squared",
    "contrast_weighted_mmd_squared",
    "conditional_discrepancy",
)


def _component_mapping(values: Mapping[str, float]) -> dict[str, float]:
    output = {str(key): float(value) for key, value in values.items()}
    if set(output) != set(_COMPONENT_KEYS) or any(
        not math.isfinite(value) or value < -1.0e-12 for value in output.values()
    ):
        raise ProtocolError("Antisymmetric component losses are invalid.")
    return {key: max(0.0, output[key]) for key in _COMPONENT_KEYS}


def _source_mapping(
    values: Mapping[str, float], sources: tuple[str, ...], role: str
) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in values.items()}
    if set(normalized) != set(sources) or any(
        not math.isfinite(value) for value in normalized.values()
    ):
        raise ProtocolError(f"Antisymmetric {role} source mapping is invalid.")
    return {source: normalized[source] for source in sources}


__all__ = (
    "AntisymmetricAxisDiagnostic",
    "AntisymmetricResidualConfig",
    "AntisymmetricResidualSolution",
    "AntisymmetricVariantDiagnostic",
    "DEFAULT_L2_SHRINKAGE",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_SOURCE_WEIGHT",
    "DEFAULT_MAX_UNIFORM_L1",
    "DEFAULT_MIN_EFFECTIVE_SOURCES",
    "DEFAULT_MIN_ROBUST_IMPROVEMENT",
    "DEFAULT_MIN_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE",
    "DEFAULT_MIN_SOFT_CLASS_MASS_PER_CASE",
    "DEFAULT_SOLVER_TOLERANCE",
    "DEFAULT_VARIANT_WORSENING_TOLERANCE",
    "DEFAULT_WORST_VARIANT_PENALTY",
    "LABEL_USE_SEMANTICS",
    "PROXY_CLAIM_ROLE",
    "ROBUST_OBJECTIVE_SEMANTICS",
    "SOLVER_METHOD",
    "WEIGHT_SEMANTICS",
)
