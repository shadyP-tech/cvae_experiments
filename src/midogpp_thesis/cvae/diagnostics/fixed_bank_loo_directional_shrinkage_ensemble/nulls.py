"""Prelabel-sealed candidate-identity null plan and descriptive controls.

The null is deliberately a *candidate block* scramble: for every held route a
single permutation is applied to the eight support-score identities and that
same permutation is used for both B-defined directions.  B, G, the physical
probability surface, and all canonical decisions remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    DIRECTION_IDS,
    EXPECTED_TOTAL_CASE_COUNT,
    NULL_REPLICATES,
    NULL_SEED,
    K_GRID,
    W_FRACTION_GRID,
    TIE_TOLERANCE,
    candidate_sources,
)
from .hashing import canonical_hash


NULL_ALGORITHM = "splitmix64_route_candidate_block_permutation_v1"
_UINT64_MASK = (1 << 64) - 1
_REP_MULTIPLIER = 0x9E3779B97F4A7C15
_SOURCE_MULTIPLIER = 0xD1B54A32D192ED03
_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB


@dataclass(frozen=True)
class CandidateIdentityNullPlan:
    seed: int
    replicates: int
    route_keys: tuple[tuple[str, str], ...]
    permutation_sha256: str
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        route_keys = tuple((str(target), str(case)) for target, case in self.route_keys)
        if self.seed != NULL_SEED or self.replicates != NULL_REPLICATES:
            raise ProtocolError("DCSE candidate-identity null constants drifted.")
        if (
            len(route_keys) != EXPECTED_TOTAL_CASE_COUNT
            or len(set(route_keys)) != EXPECTED_TOTAL_CASE_COUNT
            or any(target not in CENTERS or not case for target, case in route_keys)
        ):
            raise ProtocolError("DCSE null plan must bind the ordered 218 route keys.")
        if len(self.permutation_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.permutation_sha256
        ):
            raise ProtocolError("DCSE null permutation digest is not SHA-256.")
        object.__setattr__(self, "route_keys", route_keys)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.replicates, len(self.route_keys), 8

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_candidate_identity_null_plan_v1",
            "seed": self.seed,
            "replicates": self.replicates,
            "algorithm": NULL_ALGORITHM,
            "shape": list(self.shape),
            "dtype": "uint8",
            "matrix_shape": list(self.shape),
            "matrix_dtype": "uint8",
            "route_keys": [list(key) for key in self.route_keys],
            "candidate_order_by_target": {
                target: list(candidate_sources(target)) for target in CENTERS
            },
            "direction_ids": list(DIRECTION_IDS),
            "permutation_unit": "one_route_local_candidate_block_per_replicate",
            "permutation_scope": "one_route_local_eight_candidate_block",
            "same_candidate_permutation_for_paired_directions": True,
            "same_permutation_for_paired_directions": True,
            "scrambled_surface": "support_S_candidate_identities_only",
            "support_S_scramble_order": (
                "permute_each_route_case_additive_block_then_pool_H_minus_c"
            ),
            "baseline_B_fixed": True,
            "donor_G_fixed": True,
            "donor_prior_G_fixed": True,
            "physical_probability_surface_fixed": True,
            "permutation_sha256": self.permutation_sha256,
            "plan_sealed_before_terminal_labels": True,
            "canonical_decisions_may_change": False,
            "canonical_decisions_fixed": True,
            "canonical_endpoint_and_method_decisions_fixed": True,
            "null_specific_endpoint_selections_recomputed": True,
            "null_replicate_endpoint_selections_recomputed": True,
            "null_can_train_tune_rank_or_select": False,
            "exchangeability_claimed": False,
            "p_value_computed": False,
            "confirmatory_gate_defined": False,
            "descriptive_only": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_hash": self.plan_hash}

    def materialize(self) -> np.ndarray:
        matrix = _materialize_permutations(
            route_keys=self.route_keys, seed=self.seed, replicates=self.replicates
        )
        if hashlib.sha256(matrix.tobytes(order="C")).hexdigest() != self.permutation_sha256:
            raise ProtocolError("DCSE regenerated null permutation digest drifted.")
        return matrix

    def permutation(self, replicate: int, route_ordinal: int) -> tuple[int, ...]:
        if replicate not in range(self.replicates) or route_ordinal not in range(len(self.route_keys)):
            raise ProtocolError("DCSE null permutation index lies outside the sealed plan.")
        return _permutation_for(self.seed, replicate, self.route_keys[route_ordinal])


def build_candidate_identity_null_plan(
    route_keys: Sequence[tuple[str, str]] | Sequence[object],
    *,
    seed: int = NULL_SEED,
    replicates: int = NULL_REPLICATES,
) -> CandidateIdentityNullPlan:
    canonical = tuple(
        (
            str(getattr(value, "target_center", value[0] if isinstance(value, tuple) else "")),
            str(getattr(value, "case_id", value[1] if isinstance(value, tuple) else "")),
        )
        for value in route_keys
    )
    if seed != NULL_SEED or replicates != NULL_REPLICATES:
        raise ProtocolError("DCSE null plan seed/replicate count drifted.")
    if len(canonical) != EXPECTED_TOTAL_CASE_COUNT or len(set(canonical)) != len(canonical):
        raise ProtocolError("DCSE null plan requires exactly 218 ordered routes.")
    hasher = hashlib.sha256()
    # Hash the exact regenerated uint8 stream without retaining it.  SplitMix64
    # is specified here rather than relying on a NumPy RNG implementation, so
    # replay is independent of NumPy version while still vectorizing all 218
    # route permutations in a replicate.
    for replicate in range(replicates):
        hasher.update(
            _permutation_block(seed, replicate, canonical).tobytes(order="C")
        )
    return CandidateIdentityNullPlan(seed, replicates, canonical, hasher.hexdigest())


def _permutation_for(seed: int, replicate: int, route_key: tuple[str, str]) -> tuple[int, ...]:
    salt = _route_salt(route_key)
    ranked = sorted(
        range(8),
        key=lambda source_ordinal: _splitmix64_scalar(
            (int(seed) ^ salt ^ (replicate * _REP_MULTIPLIER)
             ^ (source_ordinal * _SOURCE_MULTIPLIER))
            & _UINT64_MASK
        ),
    )
    return tuple(ranked)


def _route_salt(route_key: tuple[str, str]) -> int:
    target, case_id = route_key
    digest = hashlib.sha256(
        f"fixed-bank-dcse-null-route-v1::{target}::{case_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _splitmix64_scalar(value: int) -> int:
    value = (value + _REP_MULTIPLIER) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * _MIX_MULTIPLIER_1) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _MIX_MULTIPLIER_2) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _splitmix64_array(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = values + np.uint64(_REP_MULTIPLIER)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            _MIX_MULTIPLIER_1
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            _MIX_MULTIPLIER_2
        )
        return values ^ (values >> np.uint64(31))


def _permutation_block(
    seed: int, replicate: int, route_keys: Sequence[tuple[str, str]]
) -> np.ndarray:
    salts = np.asarray([_route_salt(key) for key in route_keys], dtype=np.uint64)
    sources = np.arange(8, dtype=np.uint64)
    with np.errstate(over="ignore"):
        keys = (
            np.uint64(seed)
            ^ salts[:, None]
            ^ (np.uint64(replicate) * np.uint64(_REP_MULTIPLIER))
            ^ (sources[None, :] * np.uint64(_SOURCE_MULTIPLIER))
        )
    scores = _splitmix64_array(keys)
    # Stable sorting makes the (astronomically unlikely) uint64 collision
    # deterministic by the canonical numeric candidate ordinal.
    return np.argsort(scores, axis=1, kind="stable").astype(np.uint8, copy=False)


def _materialize_permutations(
    *, route_keys: Sequence[tuple[str, str]], seed: int, replicates: int
) -> np.ndarray:
    matrix = np.empty((replicates, len(route_keys), 8), dtype=np.uint8)
    for replicate in range(replicates):
        matrix[replicate] = _permutation_block(seed, replicate, route_keys)
    matrix.setflags(write=False)
    return matrix


def descriptive_null_statistics(
    plan: CandidateIdentityNullPlan,
    *,
    observed_statistic: float,
    replicate_statistics: Sequence[float] | np.ndarray,
) -> tuple[dict[str, object], ...]:
    """Summarize all sealed replicates without an exchangeability claim."""

    values = np.asarray(replicate_statistics, dtype=np.float64)
    if values.shape != (plan.replicates,) or not np.isfinite(values).all():
        raise ProtocolError(
            "DCSE null summary requires one finite statistic per sealed replicate."
        )
    quantiles = np.quantile(values, (0.025, 0.5, 0.975), method="linear")

    return (
        {
            "schema_version": "fixed_bank_dcse_candidate_identity_null_summary_v1",
            "null_family": "candidate_identity_scrambling",
            "plan_hash": plan.plan_hash,
            "seed": plan.seed,
            "replicates": plan.replicates,
            "shape": list(plan.shape),
            "permutation_sha256": plan.permutation_sha256,
            "observed_statistic": float(observed_statistic),
            "null_mean": float(np.mean(values, dtype=np.float64)),
            "null_sd": float(np.std(values, ddof=1, dtype=np.float64)),
            "null_min": float(np.min(values)),
            "null_q025": float(quantiles[0]),
            "null_median": float(quantiles[1]),
            "null_q975": float(quantiles[2]),
            "null_max": float(np.max(values)),
            "p_value": None,
            "exchangeability_claimed": False,
            "descriptive_only": True,
            "is_gate": False,
            "all_replicates_evaluated": True,
        },
    )


def validate_candidate_identity_null_plan_contract(
    plan: CandidateIdentityNullPlan,
    config_nulls: Mapping[str, object],
) -> dict[str, object]:
    """Match the executable null plan to the exact frozen config semantics."""

    payload = plan.to_payload()
    config_to_plan = {
        "algorithm": "algorithm",
        "replicates": "replicates",
        "seed": "seed",
        "matrix_shape": "matrix_shape",
        "matrix_dtype": "matrix_dtype",
        "permutation_scope": "permutation_scope",
        "same_permutation_for_paired_directions": (
            "same_permutation_for_paired_directions"
        ),
        "scrambled_surface": "scrambled_surface",
        "baseline_B_fixed": "baseline_B_fixed",
        "donor_prior_G_fixed": "donor_prior_G_fixed",
        "physical_probability_surface_fixed": (
            "physical_probability_surface_fixed"
        ),
        "canonical_endpoint_and_method_decisions_fixed": (
            "canonical_endpoint_and_method_decisions_fixed"
        ),
        "null_replicate_endpoint_selections_recomputed": (
            "null_replicate_endpoint_selections_recomputed"
        ),
    }
    if any(
        config_key not in config_nulls
        or payload.get(plan_key) != config_nulls[config_key]
        for config_key, plan_key in config_to_plan.items()
    ):
        raise ProtocolError(
            "DCSE candidate-identity null plan/config contract drifted."
        )
    if payload.get("support_S_scramble_order") != (
        "permute_each_route_case_additive_block_then_pool_H_minus_c"
    ):
        raise ProtocolError(
            "DCSE candidate-identity null block/pool order drifted."
        )
    return {
        "candidate_identity_null_contract_exact": True,
        "algorithm": payload["algorithm"],
        "shape": payload["matrix_shape"],
        "dtype": payload["matrix_dtype"],
        "scrambled_surface": payload["scrambled_surface"],
        "support_S_scramble_order": payload["support_S_scramble_order"],
        "plan_hash": plan.plan_hash,
        "permutation_sha256": plan.permutation_sha256,
    }


def select_scrambled_endpoints_scalar(
    support_values: Sequence[Sequence[Fraction]],
    prior_values: Sequence[Sequence[Fraction]],
    prior_rankings: Sequence[Sequence[int]],
    permutation: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exact reference selection for one route and one block permutation.

    Returned values are local numeric candidate ordinals; ``-1`` denotes OFF.
    Candidate ordinals are required to follow numeric source order.
    """

    support = tuple(tuple(Fraction(value) for value in row) for row in support_values)
    prior = tuple(tuple(Fraction(value) for value in row) for row in prior_values)
    rankings = tuple(tuple(int(value) for value in row) for row in prior_rankings)
    shuffled = tuple(int(value) for value in permutation)
    if (
        len(support) != 2
        or any(len(row) != 8 for row in support)
        or len(prior) != 2
        or any(len(row) != 8 for row in prior)
        or len(rankings) != 2
        or any(sorted(row) != list(range(8)) for row in rankings)
        or sorted(shuffled) != list(range(8))
    ):
        raise ProtocolError("DCSE scalar null endpoint surface is malformed.")
    output: list[tuple[int, ...]] = []
    tolerance = Fraction(1, 10**12)
    for direction in range(2):
        selected: list[int] = []
        scrambled_support = tuple(support[direction][shuffled[index]] for index in range(8))
        for k in K_GRID:
            retained = rankings[direction][:k]
            for weight in W_FRACTION_GRID:
                values = {
                    source: weight * scrambled_support[source]
                    + (1 - weight) * prior[direction][source]
                    for source in retained
                }
                maximum = max(Fraction(0), *values.values())
                if maximum <= tolerance:
                    selected.append(-1)
                else:
                    selected.append(
                        min(
                            source
                            for source, value in values.items()
                            if maximum - value <= tolerance
                        )
                    )
        output.append(tuple(selected))
    return output[0], output[1]


def select_scrambled_endpoints_vectorized(
    support_values: np.ndarray,
    prior_values: np.ndarray,
    prior_rankings: np.ndarray,
    permutations: np.ndarray,
) -> np.ndarray:
    """Vectorized null endpoint selection with scalar-reference semantics.

    Shapes are ``support/prior/rankings=(R,2,8)`` and
    ``permutations=(P,R,8)``.  The result is int8 ``(P,R,2,9)`` with OFF=-1.
    """

    support = np.asarray(support_values, dtype=np.float64)
    prior = np.asarray(prior_values, dtype=np.float64)
    rankings = np.asarray(prior_rankings, dtype=np.int64)
    shuffled = np.asarray(permutations, dtype=np.int64)
    if (
        support.ndim != 3
        or support.shape[1:] != (2, 8)
        or prior.shape != support.shape
        or rankings.shape != support.shape
        or shuffled.ndim != 3
        or shuffled.shape[1:] != (support.shape[0], 8)
        or not np.isfinite(support).all()
        or not np.isfinite(prior).all()
        or np.any(np.sort(rankings, axis=2) != np.arange(8))
        or np.any(np.sort(shuffled, axis=2) != np.arange(8))
    ):
        raise ProtocolError("DCSE vectorized null endpoint surface is malformed.")
    replicate_count = shuffled.shape[0]
    route_count = support.shape[0]
    support_broadcast = np.broadcast_to(
        support[None, :, :, :], (replicate_count, route_count, 2, 8)
    )
    scrambled_support = np.take_along_axis(
        support_broadcast,
        np.broadcast_to(
            shuffled[:, :, None, :], (replicate_count, route_count, 2, 8)
        ),
        axis=3,
    )
    return select_endpoint_values_vectorized(
        scrambled_support, prior, rankings
    )


def select_endpoint_values_vectorized(
    support_values: np.ndarray,
    prior_values: np.ndarray,
    prior_rankings: np.ndarray,
) -> np.ndarray:
    """Select nine endpoints from already-scrambled support values.

    ``support_values`` is ``(P,R,2,8)``.  Priors and their exact-G ranking are
    fixed ``(R,2,8)``.  This separation lets the terminal null first scramble
    each route/case sufficient-statistic block, pool H-minus-c, and only then
    execute the endpoint pipeline.
    """

    scrambled_support = np.asarray(support_values, dtype=np.float64)
    prior = np.asarray(prior_values, dtype=np.float64)
    rankings = np.asarray(prior_rankings, dtype=np.int64)
    if (
        scrambled_support.ndim != 4
        or scrambled_support.shape[2:] != (2, 8)
        or prior.shape != scrambled_support.shape[1:]
        or rankings.shape != prior.shape
        or not np.isfinite(scrambled_support).all()
        or not np.isfinite(prior).all()
        or np.any(np.sort(rankings, axis=2) != np.arange(8))
    ):
        raise ProtocolError("DCSE vectorized null endpoint values are malformed.")
    replicate_count, route_count = scrambled_support.shape[:2]
    selected = np.full((replicate_count, route_count, 2, 9), -1, dtype=np.int8)
    arm_ordinal = 0
    for k in K_GRID:
        retained = rankings[:, :, :k]
        support_retained = np.take_along_axis(
            scrambled_support,
            np.broadcast_to(retained[None, :, :, :], (replicate_count, route_count, 2, k)),
            axis=3,
        )
        prior_retained = np.take_along_axis(prior, retained, axis=2)
        for weight in W_FRACTION_GRID:
            scores = float(weight) * support_retained + float(1 - weight) * prior_retained[None, :, :, :]
            maximum = np.maximum(0.0, np.max(scores, axis=3))
            active = maximum > TIE_TOLERANCE
            eligible = maximum[:, :, :, None] - scores <= TIE_TOLERANCE
            numeric = np.where(
                eligible,
                retained[None, :, :, :],
                np.int64(99),
            )
            winner = np.min(numeric, axis=3).astype(np.int8)
            selected[:, :, :, arm_ordinal] = np.where(active, winner, -1)
            arm_ordinal += 1
    selected.setflags(write=False)
    return selected


__all__ = (
    "CandidateIdentityNullPlan",
    "NULL_ALGORITHM",
    "build_candidate_identity_null_plan",
    "descriptive_null_statistics",
    "select_endpoint_values_vectorized",
    "select_scrambled_endpoints_scalar",
    "select_scrambled_endpoints_vectorized",
    "validate_candidate_identity_null_plan_contract",
)
