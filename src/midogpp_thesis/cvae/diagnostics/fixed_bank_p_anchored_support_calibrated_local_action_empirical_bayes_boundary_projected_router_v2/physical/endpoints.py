"""Independent protected-P reconstruction and SCALE-BP challenger endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from .contracts import (
    ACTION_IDS,
    CENTERS,
    DIRECTIONS,
    HARD_THRESHOLD,
    PORTFOLIO_I_WEIGHT,
    PORTFOLIO_R_WEIGHT,
    ROBUST_ARM_COUNT,
    array_sha256,
    candidate_sources,
    probability_vector,
)
from .store import PhysicalStoreAdapter, SEED_PAIRS


ROBUST_ARM_GRID = tuple(
    (k, weight)
    for k in (4, 5, 6)
    for weight in (1.0 / 2.0, 3.0 / 5.0, 7.0 / 10.0)
)


@dataclass(frozen=True, slots=True)
class RouteEndpointPlan:
    """Sealed support-scoped decisions used only to reconstruct protected P.

    ``case_id`` is the case whose endpoint surface is being reconstructed.  A
    final route excludes only that case.  A route-local validation surface also
    excludes the outer held case and every member of the validation fold; the
    complete, canonical exclusion tuple is retained here so those two surfaces
    cannot share a plan hash accidentally.
    """

    target_center: str
    case_id: str
    identification_sources: Mapping[str, str | None]
    robust_arm_sources: Mapping[str, tuple[str | None, ...]]
    support_scope_hash: str
    source_excluded_centers: tuple[str, ...]
    support_excluded_case_ids: tuple[str, ...]
    outer_held_case_id: str
    derivation_hashes: tuple[str, ...] = ()
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target, case = str(self.target_center), str(self.case_id)
        identification = {
            str(direction): None if source is None else str(source)
            for direction, source in self.identification_sources.items()
        }
        robust = {
            str(direction): tuple(
                None if source is None else str(source) for source in sources
            )
            for direction, sources in self.robust_arm_sources.items()
        }
        hashes = tuple(str(value) for value in self.derivation_hashes)
        excluded = tuple(str(center) for center in self.source_excluded_centers)
        excluded_cases = tuple(str(value) for value in self.support_excluded_case_ids)
        outer_held = str(self.outer_held_case_id)
        if excluded != tuple(center for center in CENTERS if center in set(excluded)):
            raise GovernanceError("SCALE-BP v2 source-exclusion order drifted.")
        legal = set(candidate_sources(target)) - set(excluded)
        if (
            not case
            or not self.support_scope_hash
            or tuple(identification) != DIRECTIONS
            or tuple(robust) != DIRECTIONS
            or any(source is not None and source not in legal for source in identification.values())
            or any(len(sources) != ROBUST_ARM_COUNT for sources in robust.values())
            or any(
                source is not None and source not in legal
                for sources in robust.values()
                for source in sources
            )
            or target not in excluded
            or case not in excluded_cases
            or outer_held not in excluded_cases
            or not outer_held
            or len(excluded_cases) != len(set(excluded_cases))
            or len(excluded) != len(set(excluded))
            or any(center not in (*candidate_sources(target), target) for center in excluded)
            or len(hashes) != len(set(hashes))
        ):
            raise GovernanceError("SCALE-BP v2 protected-P endpoint plan drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "identification_sources", MappingProxyType(identification))
        object.__setattr__(self, "robust_arm_sources", MappingProxyType(robust))
        object.__setattr__(self, "source_excluded_centers", excluded)
        object.__setattr__(self, "support_excluded_case_ids", excluded_cases)
        object.__setattr__(self, "outer_held_case_id", outer_held)
        object.__setattr__(self, "derivation_hashes", hashes)
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_protected_p_endpoint_plan_v2",
                    "target_center": target,
                    "case_id": case,
                    "identification_sources": identification,
                    "robust_arm_sources": robust,
                    "support_scope_hash": self.support_scope_hash,
                    "source_excluded_centers": excluded,
                    "support_excluded_case_ids": excluded_cases,
                    "outer_held_case_id": outer_held,
                    "derivation_hashes": hashes,
                    "evaluation_case_excluded": True,
                    "outer_held_case_excluded": True,
                    "own_validation_fold_excluded_when_applicable": True,
                    "target_expert_excluded": True,
                    "challenger_endpoints_affected": False,
                    "robust_duplicate_arms_preserved": True,
                }
            ),
        )


def derive_route_endpoint_plan(
    *,
    target_center: object,
    case_id: object,
    identification_scores: Mapping[str, Mapping[str, float]],
    support_directional_gains: Mapping[str, Mapping[str, float]],
    donor_directional_priors: Mapping[str, Mapping[str, float]],
    support_scope_hash: object,
    source_excluded_centers: tuple[str, ...] | None = None,
    support_excluded_case_ids: tuple[str, ...] | None = None,
    outer_held_case_id: object | None = None,
    derivation_hashes: tuple[str, ...] = (),
    tie_tolerance: float = 1.0e-12,
) -> RouteEndpointPlan:
    """Freeze protected-P decisions from freshly reconstructed H-minus-c maps."""

    target, evaluation_case = str(target_center), str(case_id)
    excluded = (
        (target,)
        if source_excluded_centers is None
        else tuple(str(center) for center in source_excluded_centers)
    )
    if target not in excluded:
        raise GovernanceError("SCALE-BP v2 target must be source-excluded.")
    excluded_cases = (
        (evaluation_case,)
        if support_excluded_case_ids is None
        else tuple(str(value) for value in support_excluded_case_ids)
    )
    outer_held = (
        evaluation_case if outer_held_case_id is None else str(outer_held_case_id)
    )
    if (
        not evaluation_case
        or evaluation_case not in excluded_cases
        or not outer_held
        or outer_held not in excluded_cases
        or len(excluded_cases) != len(set(excluded_cases))
    ):
        raise GovernanceError("SCALE-BP v2 support-case exclusion scope drifted.")
    sources = tuple(
        source for source in candidate_sources(target) if source not in set(excluded)
    )
    if len(sources) < 3:
        raise GovernanceError("SCALE-BP v2 endpoint plan lacks legal sources.")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise GovernanceError("SCALE-BP v2 endpoint-plan tie tolerance drifted.")

    def checked(
        surface: Mapping[str, Mapping[str, float]], role: str
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        if tuple(surface) != DIRECTIONS:
            raise GovernanceError(f"SCALE-BP v2 {role} direction surface drifted.")
        for direction in DIRECTIONS:
            row = {
                str(source): float(value)
                for source, value in surface[direction].items()
            }
            if tuple(row) != sources or not all(
                math.isfinite(value) for value in row.values()
            ):
                raise GovernanceError(f"SCALE-BP v2 {role} source surface drifted.")
            result[direction] = row
        return result

    identification = checked(identification_scores, "identification score")
    support = checked(support_directional_gains, "support gain")
    donor = checked(donor_directional_priors, "donor prior")
    selected_identification: dict[str, str | None] = {}
    robust: dict[str, tuple[str | None, ...]] = {}
    for direction in DIRECTIONS:
        eligible_sources = tuple(
            source for source in sources if support[direction][source] > 0.0
        )
        maximum = (
            max(identification[direction][source] for source in eligible_sources)
            if eligible_sources
            else -math.inf
        )
        tied = tuple(
            source
            for source in eligible_sources
            if maximum - identification[direction][source] <= tie_tolerance
        )
        selected_identification[direction] = (
            min(tied, key=int) if maximum > tie_tolerance else None
        )
        ranked = tuple(
            sorted(
                sources,
                key=lambda source: (-donor[direction][source], int(source)),
            )
        )
        arm_sources: list[str | None] = []
        for k, weight in ROBUST_ARM_GRID:
            menu: list[tuple[str | None, float]] = [(None, 0.0)]
            menu.extend(
                (
                    source,
                    weight * support[direction][source]
                    + (1.0 - weight) * donor[direction][source],
                )
                for source in ranked[:k]
            )
            best = max(score for _, score in menu)
            tied_sources = tuple(
                source for source, score in menu if best - score <= tie_tolerance
            )
            arm_sources.append(
                min(
                    tied_sources,
                    key=lambda source: -1 if source is None else int(source),
                )
            )
        robust[direction] = tuple(arm_sources)
    return RouteEndpointPlan(
        target,
        evaluation_case,
        selected_identification,
        robust,
        str(support_scope_hash),
        excluded,
        excluded_cases,
        outer_held,
        derivation_hashes,
    )


@dataclass(frozen=True, slots=True, eq=False)
class CaseEndpointSurface:
    target_center: str
    case_id: str
    sample_ids: tuple[str, ...]
    challenger_probabilities: Mapping[str, np.ndarray]
    seed_challenger_probabilities: Mapping[str, np.ndarray]
    protected_component_probabilities: Mapping[str, np.ndarray]
    seed_protected_component_probabilities: Mapping[str, np.ndarray]
    available_sources: tuple[str, ...]
    source_excluded_centers: tuple[str, ...]
    support_excluded_case_ids: tuple[str, ...]
    outer_held_case_id: str
    physical_view_hashes: tuple[str, ...]
    plan_hash: str
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(str(value) for value in self.sample_ids)
        challengers = {
            str(name): probability_vector(values, expected_length=len(samples))
            for name, values in self.challenger_probabilities.items()
        }
        seed_challengers = {
            str(name): np.ascontiguousarray(values, dtype=np.float64)
            for name, values in self.seed_challenger_probabilities.items()
        }
        protected = {
            str(name): probability_vector(values, expected_length=len(samples))
            for name, values in self.protected_component_probabilities.items()
        }
        seed_protected = {
            str(name): np.ascontiguousarray(values, dtype=np.float64)
            for name, values in self.seed_protected_component_probabilities.items()
        }
        hashes = tuple(str(value) for value in self.physical_view_hashes)
        available = tuple(str(value) for value in self.available_sources)
        excluded = tuple(str(value) for value in self.source_excluded_centers)
        excluded_cases = tuple(str(value) for value in self.support_excluded_case_ids)
        outer_held = str(self.outer_held_case_id)
        if (
            not samples
            or len(set(samples)) != len(samples)
            or tuple(challengers) != ACTION_IDS
            or tuple(seed_challengers) != ACTION_IDS
            or tuple(protected)
            != ("I_PROTECTED", "R_PROTECTED", "P_PROTECTED")
            or tuple(seed_protected)
            != ("I_PROTECTED", "R_PROTECTED", "P_PROTECTED")
            or any(
                values.shape != (len(SEED_PAIRS), len(samples))
                for values in (*seed_challengers.values(), *seed_protected.values())
            )
            or not all(
                np.isfinite(values).all()
                for values in (*seed_challengers.values(), *seed_protected.values())
            )
            or not all(
                np.allclose(
                    challengers[action],
                    np.mean(seed_challengers[action], axis=0, dtype=np.float64),
                    rtol=0.0,
                    atol=8.0 * np.finfo(np.float64).eps,
                )
                for action in ACTION_IDS
            )
            or not all(
                np.array_equal(
                    protected[name],
                    np.mean(seed_protected[name], axis=0, dtype=np.float64),
                )
                for name in ("I_PROTECTED", "R_PROTECTED")
            )
            or not np.array_equal(
                protected["P_PROTECTED"],
                PORTFOLIO_I_WEIGHT * protected["I_PROTECTED"]
                + PORTFOLIO_R_WEIGHT * protected["R_PROTECTED"],
            )
            or not np.array_equal(
                seed_protected["P_PROTECTED"],
                PORTFOLIO_I_WEIGHT * seed_protected["I_PROTECTED"]
                + PORTFOLIO_R_WEIGHT * seed_protected["R_PROTECTED"],
            )
            or available != tuple(
                source
                for source in candidate_sources(self.target_center)
                if source not in set(excluded)
            )
            or self.target_center not in excluded
            or self.case_id not in excluded_cases
            or outer_held not in excluded_cases
            or len(excluded_cases) != len(set(excluded_cases))
            or len(hashes) != 2 + len(available)
            or len(set(hashes)) != len(hashes)
        ):
            raise GovernanceError("SCALE-BP v2 case endpoint surface drifted.")
        for values in (*seed_challengers.values(), *seed_protected.values()):
            values.setflags(write=False)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(
            self, "challenger_probabilities", MappingProxyType(challengers)
        )
        object.__setattr__(
            self,
            "seed_challenger_probabilities",
            MappingProxyType(seed_challengers),
        )
        object.__setattr__(
            self,
            "protected_component_probabilities",
            MappingProxyType(protected),
        )
        object.__setattr__(
            self,
            "seed_protected_component_probabilities",
            MappingProxyType(seed_protected),
        )
        object.__setattr__(self, "physical_view_hashes", hashes)
        object.__setattr__(self, "available_sources", available)
        object.__setattr__(self, "source_excluded_centers", excluded)
        object.__setattr__(self, "support_excluded_case_ids", excluded_cases)
        object.__setattr__(self, "outer_held_case_id", outer_held)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_case_endpoint_surface_v3",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "sample_ids": samples,
                    "challenger_sha256": {
                        key: array_sha256(value)
                        for key, value in challengers.items()
                    },
                    "seed_challenger_sha256": {
                        key: array_sha256(value)
                        for key, value in seed_challengers.items()
                    },
                    "protected_component_sha256": {
                        key: array_sha256(value)
                        for key, value in protected.items()
                    },
                    "seed_protected_component_sha256": {
                        key: array_sha256(value)
                        for key, value in seed_protected.items()
                    },
                    "physical_view_hashes": hashes,
                    "available_sources": available,
                    "source_excluded_centers": excluded,
                    "support_excluded_case_ids": excluded_cases,
                    "outer_held_case_id": outer_held,
                    "plan_hash": self.plan_hash,
                    "portfolio_formula": "3/5*I_PROTECTED+2/5*R_PROTECTED",
                    "challenger_geometry": (
                        "B|directional_A1_extreme|median_U_plus_eight_A1"
                    ),
                    "labels_used_in_composition": False,
                }
            ),
        )

    @property
    def protected_p(self) -> np.ndarray:
        return self.protected_component_probabilities["P_PROTECTED"]

    def challenger(self, family: object, direction: object) -> np.ndarray:
        action_id = f"{family}::{direction}"
        try:
            return self.challenger_probabilities[action_id]
        except KeyError as exc:
            raise GovernanceError("SCALE-BP v2 challenger action is unknown.") from exc


def reconstruct_case_surface(
    store: PhysicalStoreAdapter,
    plan: RouteEndpointPlan,
) -> CaseEndpointSurface:
    """Reconstruct protected P and independent challengers from the 810 cells."""

    if plan.case_id not in store.case_ids(plan.target_center):
        raise GovernanceError("SCALE-BP v2 route plan is not bound to the store.")
    available_sources = tuple(
        source
        for source in candidate_sources(plan.target_center)
        if source not in set(plan.source_excluded_centers)
    )
    actions = (
        "B",
        "U",
        *(
            f"A1::source={source}"
            for source in available_sources
        ),
    )
    views = {
        action: store.exact_nine_view(
            plan.target_center, action, case_id=plan.case_id
        )
        for action in actions
    }
    sample_orders = {view.sample_ids for view in views.values()}
    if len(sample_orders) != 1:
        raise GovernanceError("SCALE-BP v2 physical views disagree on row order.")
    sample_ids = next(iter(sample_orders))
    seed_by_action = {
        action: np.asarray(view.seed_probabilities, dtype=np.float64)
        for action, view in views.items()
    }
    mean_by_action = {
        action: np.asarray(view.mean_probability, dtype=np.float64)
        for action, view in views.items()
    }
    b_mean = mean_by_action["B"]
    b_seeds = seed_by_action["B"]
    baseline_zero = b_mean < HARD_THRESHOLD

    protected_i_seed = np.array(b_seeds, copy=True)
    for direction in DIRECTIONS:
        source = plan.identification_sources[direction]
        if source is None:
            continue
        mask = baseline_zero if direction == "zero_to_one" else ~baseline_zero
        protected_i_seed[:, mask] = seed_by_action[f"A1::source={source}"][:, mask]
    robust_arm_seed: list[np.ndarray] = []
    for arm_index in range(ROBUST_ARM_COUNT):
        arm = np.array(b_seeds, copy=True)
        for direction in DIRECTIONS:
            source = plan.robust_arm_sources[direction][arm_index]
            if source is None:
                continue
            mask = baseline_zero if direction == "zero_to_one" else ~baseline_zero
            arm[:, mask] = seed_by_action[f"A1::source={source}"][:, mask]
        robust_arm_seed.append(arm)
    protected_r_seed = np.mean(
        np.stack(robust_arm_seed, axis=0), axis=0, dtype=np.float64
    )
    protected_p_seed = (
        PORTFOLIO_I_WEIGHT * protected_i_seed
        + PORTFOLIO_R_WEIGHT * protected_r_seed
    )

    a1_actions = tuple(
        action for action in actions if action.startswith("A1::source=")
    )
    a1_mean = np.stack(
        [mean_by_action[action] for action in a1_actions], axis=0
    )
    max_index = np.argmax(a1_mean, axis=0)
    min_index = np.argmin(a1_mean, axis=0)
    i_zero_seed = np.stack(
        [
            seed_by_action[a1_actions[index]][:, row]
            for row, index in enumerate(max_index)
        ],
        axis=1,
    )
    i_one_seed = np.stack(
        [
            seed_by_action[a1_actions[index]][:, row]
            for row, index in enumerate(min_index)
        ],
        axis=1,
    )
    robust_actions = ("U", *a1_actions)
    robust_mean = np.stack(
        [mean_by_action[action] for action in robust_actions], axis=0
    )
    robust_order = np.argsort(robust_mean, axis=0, kind="stable")
    middle = len(robust_actions) // 2
    upper_index = robust_order[middle]
    lower_index = robust_order[middle if len(robust_actions) % 2 else middle - 1]
    robust_seed = np.stack(
        [
            0.5
            * (
                seed_by_action[robust_actions[lower]][:, row]
                + seed_by_action[robust_actions[upper]][:, row]
            )
            if lower != upper
            else seed_by_action[robust_actions[upper]][:, row]
            for row, (lower, upper) in enumerate(zip(lower_index, upper_index, strict=True))
        ],
        axis=1,
    )
    seed_challengers = {
        "B::zero_to_one": b_seeds,
        "B::one_to_zero": b_seeds,
        "I::zero_to_one": i_zero_seed,
        "I::one_to_zero": i_one_seed,
        "R::zero_to_one": robust_seed,
        "R::one_to_zero": robust_seed,
    }
    i_zero = np.max(a1_mean, axis=0)
    i_one = np.min(a1_mean, axis=0)
    robust_median = np.median(robust_mean, axis=0)
    challengers = {
        "B::zero_to_one": b_mean,
        "B::one_to_zero": b_mean,
        "I::zero_to_one": i_zero,
        "I::one_to_zero": i_one,
        "R::zero_to_one": robust_median,
        "R::one_to_zero": robust_median,
    }
    protected = {
        "I_PROTECTED": np.mean(
            protected_i_seed, axis=0, dtype=np.float64
        ),
        "R_PROTECTED": np.mean(
            protected_r_seed, axis=0, dtype=np.float64
        ),
    }
    protected["P_PROTECTED"] = (
        PORTFOLIO_I_WEIGHT * protected["I_PROTECTED"]
        + PORTFOLIO_R_WEIGHT * protected["R_PROTECTED"]
    )
    return CaseEndpointSurface(
        plan.target_center,
        plan.case_id,
        sample_ids,
        challengers,
        seed_challengers,
        protected,
        {
            "I_PROTECTED": protected_i_seed,
            "R_PROTECTED": protected_r_seed,
            "P_PROTECTED": protected_p_seed,
        },
        available_sources,
        plan.source_excluded_centers,
        plan.support_excluded_case_ids,
        plan.outer_held_case_id,
        tuple(view.view_hash for view in views.values()),
        plan.plan_hash,
    )


__all__ = (
    "CaseEndpointSurface",
    "ROBUST_ARM_GRID",
    "RouteEndpointPlan",
    "derive_route_endpoint_plan",
    "reconstruct_case_surface",
)
