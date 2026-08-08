"""Immutable contracts for fresh, label-free residual top-up proxy ranking.

The contracts in this package stop at fixed proxy ranks and priorities.  They
do not describe a composed synthetic dataset or a downstream measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping

from ...protocol import ProtocolError


GLOBAL_PSEUDOQUERY_ROLE = "global_pseudoquery"
TARGET_SUPPORT_ROLE = "target_support"
QUERY_ROLES = (GLOBAL_PSEUDOQUERY_ROLE, TARGET_SUPPORT_ROLE)

FIXED_TRAINING_SEEDS = (17, 42, 101)
GLOBAL_POLICY_ID = "G"
SUPPORT_POLICY_ID = "S"

PROXY_ENERGY_SEMANTICS = (
    "proxy_only_fixed_half_class_marginalized_common_space_reconstruction_mse_"
    "plus_latent_dim_normalized_analytic_ps_prior_kl"
)
REPLICA_AGGREGATION_SEMANTICS = (
    "arithmetic_mean_of_fixed_three_training_replica_proxy_energies_before_"
    "each_case_ballot"
)
NORMALIZED_MIDRANK_SEMANTICS = (
    "lower_proxy_energy_is_better_true_average_tie_midrank_minus_one_divided_"
    "by_ballot_size_minus_one"
)
GLOBAL_AGGREGATION_SEMANTICS = (
    "case_equal_leave_outer_target_and_pseudoquery_out_mean_normalized_"
    "proxy_midrank"
)
SUPPORT_AGGREGATION_SEMANTICS = (
    "case_equal_unlabeled_target_support_mean_normalized_proxy_midrank"
)
PRIORITY_SEMANTICS = "one_minus_mean_normalized_proxy_midrank"
PERMUTATION_SEMANTICS = (
    "canonical_source_order_nonzero_cyclic_identity_permutation_control"
)
POLICY_SEMANTICS = (
    "fresh_label_free_fixed_global_and_target_support_proxy_rank_summary_only"
)


@dataclass(frozen=True)
class FreshProxyScoreRow:
    """One fresh proxy-energy cell with explicit negative attestations."""

    outer_target: str
    query_role: str
    query_center: str
    case_id: str
    candidate_source: str
    training_seed: int
    proxy_energy: float
    labels_consumed: bool
    evaluation_overlap: bool
    source_expert_updated: bool
    proxy_energy_semantics: str = PROXY_ENERGY_SEMANTICS

    def __post_init__(self) -> None:
        identifiers = (
            self.outer_target,
            self.query_role,
            self.query_center,
            self.case_id,
            self.candidate_source,
        )
        if any(
            not isinstance(value, str)
            or not value
            or value.strip() != value
            for value in identifiers
        ):
            raise ProtocolError(
                "Fresh proxy score identifiers must be nonempty canonical strings."
            )
        if self.query_role not in QUERY_ROLES:
            raise ProtocolError("Fresh proxy score query role is invalid.")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, Integral)
            or int(self.training_seed) not in FIXED_TRAINING_SEEDS
        ):
            raise ProtocolError("Fresh proxy score training-seed drift detected.")
        if isinstance(self.proxy_energy, bool):
            raise ProtocolError("Fresh proxy energy must be finite.")
        try:
            energy = float(self.proxy_energy)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Fresh proxy energy must be finite.") from exc
        if not math.isfinite(energy):
            raise ProtocolError("Fresh proxy energy must be finite.")
        if type(self.labels_consumed) is not bool or self.labels_consumed:
            raise ProtocolError(
                "Fresh proxy scores require an explicit no-label attestation."
            )
        if type(self.evaluation_overlap) is not bool or self.evaluation_overlap:
            raise ProtocolError(
                "Fresh proxy scores require an explicit no-evaluation-overlap "
                "attestation."
            )
        if type(self.source_expert_updated) is not bool or self.source_expert_updated:
            raise ProtocolError(
                "Fresh proxy scores require an explicit no-source-expert-update "
                "attestation."
            )
        if self.proxy_energy_semantics != PROXY_ENERGY_SEMANTICS:
            raise ProtocolError("Fresh proxy-energy semantics drifted.")
        if self.candidate_source == self.outer_target:
            raise ProtocolError("Outer target H leaked into the candidate pool.")
        if self.query_role == GLOBAL_PSEUDOQUERY_ROLE:
            if self.query_center == self.outer_target:
                raise ProtocolError(
                    "Global pseudoquery q must differ from outer target H."
                )
            if self.candidate_source == self.query_center:
                raise ProtocolError(
                    "Global pseudoquery q leaked into its candidate pool."
                )
        elif self.query_center != self.outer_target:
            raise ProtocolError(
                "Target-support rows must query their own outer target H."
            )
        object.__setattr__(self, "training_seed", int(self.training_seed))
        object.__setattr__(self, "proxy_energy", energy)

    @property
    def label_free_attested(self) -> bool:
        return not self.labels_consumed

    @property
    def evaluation_disjoint_attested(self) -> bool:
        return not self.evaluation_overlap

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target": self.outer_target,
            "query_role": self.query_role,
            "query_center": self.query_center,
            "case_id": self.case_id,
            "candidate_source": self.candidate_source,
            "training_seed": self.training_seed,
            "proxy_energy": self.proxy_energy,
            "labels_consumed": self.labels_consumed,
            "evaluation_overlap": self.evaluation_overlap,
            "source_expert_updated": self.source_expert_updated,
            "proxy_energy_semantics": self.proxy_energy_semantics,
        }


@dataclass(frozen=True)
class CaseProxyBallot:
    """One case ballot after fixed-replica averaging."""

    outer_target: str
    query_role: str
    query_center: str
    case_id: str
    candidate_sources: tuple[str, ...]
    mean_proxy_energy_by_source: Mapping[str, float]
    normalized_midrank_by_source: Mapping[str, float]
    training_seeds: tuple[int, ...] = FIXED_TRAINING_SEEDS
    proxy_energy_semantics: str = PROXY_ENERGY_SEMANTICS
    replica_aggregation_semantics: str = REPLICA_AGGREGATION_SEMANTICS
    normalized_midrank_semantics: str = NORMALIZED_MIDRANK_SEMANTICS
    labels_consumed: bool = False
    evaluation_overlap: bool = False
    source_expert_updated: bool = False

    def __post_init__(self) -> None:
        sources = canonical_source_ids(self.candidate_sources)
        if sources != self.candidate_sources or len(sources) < 2:
            raise ProtocolError("Case proxy ballot candidate order is invalid.")
        energies = _finite_source_mapping(
            self.mean_proxy_energy_by_source,
            expected_sources=sources,
            value_name="case mean proxy energy",
        )
        ranks = _finite_source_mapping(
            self.normalized_midrank_by_source,
            expected_sources=sources,
            value_name="normalized proxy midrank",
        )
        if any(value < 0.0 or value > 1.0 for value in ranks.values()):
            raise ProtocolError("Normalized proxy midranks must lie in [0, 1].")
        if tuple(self.training_seeds) != FIXED_TRAINING_SEEDS:
            raise ProtocolError("Case proxy ballot training-seed drift detected.")
        if (
            self.proxy_energy_semantics != PROXY_ENERGY_SEMANTICS
            or self.replica_aggregation_semantics
            != REPLICA_AGGREGATION_SEMANTICS
            or self.normalized_midrank_semantics != NORMALIZED_MIDRANK_SEMANTICS
        ):
            raise ProtocolError("Case proxy ballot semantics drifted.")
        if (
            self.labels_consumed
            or self.evaluation_overlap
            or self.source_expert_updated
        ):
            raise ProtocolError("Case proxy ballot attestations failed closed.")
        object.__setattr__(
            self, "mean_proxy_energy_by_source", MappingProxyType(energies)
        )
        object.__setattr__(
            self, "normalized_midrank_by_source", MappingProxyType(ranks)
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target": self.outer_target,
            "query_role": self.query_role,
            "query_center": self.query_center,
            "case_id": self.case_id,
            "candidate_sources": list(self.candidate_sources),
            "mean_proxy_energy_by_source": dict(
                self.mean_proxy_energy_by_source
            ),
            "normalized_midrank_by_source": dict(
                self.normalized_midrank_by_source
            ),
            "training_seeds": list(self.training_seeds),
            "proxy_energy_semantics": self.proxy_energy_semantics,
            "replica_aggregation_semantics": self.replica_aggregation_semantics,
            "normalized_midrank_semantics": self.normalized_midrank_semantics,
            "labels_consumed": self.labels_consumed,
            "evaluation_overlap": self.evaluation_overlap,
            "source_expert_updated": self.source_expert_updated,
        }


@dataclass(frozen=True)
class ProxyRankSummary:
    """Immutable per-target G or S mean-rank and priority summary."""

    outer_target: str
    policy_id: str
    candidate_sources: tuple[str, ...]
    mean_normalized_midrank_by_source: Mapping[str, float]
    priority_by_source: Mapping[str, float]
    ballot_count_by_source: Mapping[str, int]
    query_centers: tuple[str, ...]
    case_count_by_query_center: Mapping[str, int]
    ballots: tuple[CaseProxyBallot, ...]
    aggregation_semantics: str
    training_seeds: tuple[int, ...] = FIXED_TRAINING_SEEDS
    priority_semantics: str = PRIORITY_SEMANTICS
    labels_consumed: bool = False
    evaluation_overlap: bool = False
    source_expert_updated: bool = False

    def __post_init__(self) -> None:
        sources = canonical_source_ids(self.candidate_sources)
        if sources != self.candidate_sources:
            raise ProtocolError("Proxy-rank summary candidate order is invalid.")
        if self.policy_id not in {GLOBAL_POLICY_ID, SUPPORT_POLICY_ID}:
            raise ProtocolError("Proxy-rank summary policy identity is invalid.")
        expected_aggregation = (
            GLOBAL_AGGREGATION_SEMANTICS
            if self.policy_id == GLOBAL_POLICY_ID
            else SUPPORT_AGGREGATION_SEMANTICS
        )
        if self.aggregation_semantics != expected_aggregation:
            raise ProtocolError("Proxy-rank aggregation semantics drifted.")
        ranks = _finite_source_mapping(
            self.mean_normalized_midrank_by_source,
            expected_sources=sources,
            value_name="mean normalized proxy midrank",
        )
        priorities = _finite_source_mapping(
            self.priority_by_source,
            expected_sources=sources,
            value_name="proxy-rank priority",
        )
        if any(value < 0.0 or value > 1.0 for value in ranks.values()):
            raise ProtocolError("Mean normalized proxy midranks must lie in [0, 1].")
        if any(value < 0.0 or value > 1.0 for value in priorities.values()):
            raise ProtocolError("Proxy-rank priorities must lie in [0, 1].")
        if any(
            not math.isclose(
                priorities[source],
                1.0 - ranks[source],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            for source in sources
        ):
            raise ProtocolError("Proxy-rank priority must equal one minus mean rank.")
        counts = _positive_count_mapping(
            self.ballot_count_by_source,
            expected_keys=sources,
            value_name="source ballot count",
        )
        queries = canonical_source_ids(self.query_centers)
        if queries != self.query_centers:
            raise ProtocolError("Proxy-rank query-center order is invalid.")
        case_counts = _positive_count_mapping(
            self.case_count_by_query_center,
            expected_keys=queries,
            value_name="query case count",
        )
        if not self.ballots:
            raise ProtocolError("Proxy-rank summary requires case ballots.")
        if tuple(self.training_seeds) != FIXED_TRAINING_SEEDS:
            raise ProtocolError("Proxy-rank summary training-seed drift detected.")
        if self.priority_semantics != PRIORITY_SEMANTICS:
            raise ProtocolError("Proxy-rank priority semantics drifted.")
        if (
            self.labels_consumed
            or self.evaluation_overlap
            or self.source_expert_updated
        ):
            raise ProtocolError("Proxy-rank summary attestations failed closed.")
        object.__setattr__(
            self, "mean_normalized_midrank_by_source", MappingProxyType(ranks)
        )
        object.__setattr__(self, "priority_by_source", MappingProxyType(priorities))
        object.__setattr__(
            self, "ballot_count_by_source", MappingProxyType(counts)
        )
        object.__setattr__(
            self, "case_count_by_query_center", MappingProxyType(case_counts)
        )

    @property
    def mean_rank_by_source(self) -> Mapping[str, float]:
        """Short alias for the normalized mean-rank mapping."""

        return self.mean_normalized_midrank_by_source

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target": self.outer_target,
            "policy_id": self.policy_id,
            "candidate_sources": list(self.candidate_sources),
            "mean_normalized_midrank_by_source": dict(
                self.mean_normalized_midrank_by_source
            ),
            "priority_by_source": dict(self.priority_by_source),
            "ballot_count_by_source": dict(self.ballot_count_by_source),
            "query_centers": list(self.query_centers),
            "case_count_by_query_center": dict(
                self.case_count_by_query_center
            ),
            "ballots": [ballot.to_payload() for ballot in self.ballots],
            "aggregation_semantics": self.aggregation_semantics,
            "training_seeds": list(self.training_seeds),
            "priority_semantics": self.priority_semantics,
            "labels_consumed": self.labels_consumed,
            "evaluation_overlap": self.evaluation_overlap,
            "source_expert_updated": self.source_expert_updated,
        }


@dataclass(frozen=True)
class TargetProxyPolicySummary:
    """Action-free per-target G/S proxy policy summary."""

    outer_target: str
    candidate_sources: tuple[str, ...]
    global_summary: ProxyRankSummary
    support_summary: ProxyRankSummary
    source_identity_permutation: Mapping[str, str]
    permutation_index: int
    training_seeds: tuple[int, ...] = FIXED_TRAINING_SEEDS
    policy_semantics: str = POLICY_SEMANTICS
    labels_consumed: bool = False
    evaluation_overlap: bool = False
    source_expert_updated: bool = False
    actions_constructed: bool = False

    def __post_init__(self) -> None:
        sources = canonical_source_ids(self.candidate_sources)
        if sources != self.candidate_sources or self.outer_target in sources:
            raise ProtocolError("Target proxy-policy source pool is invalid.")
        if (
            self.global_summary.outer_target != self.outer_target
            or self.support_summary.outer_target != self.outer_target
            or self.global_summary.policy_id != GLOBAL_POLICY_ID
            or self.support_summary.policy_id != SUPPORT_POLICY_ID
            or self.global_summary.candidate_sources != sources
            or self.support_summary.candidate_sources != sources
        ):
            raise ProtocolError("Target proxy-policy summaries are misaligned.")
        permutation = {str(key): str(value) for key, value in self.source_identity_permutation.items()}
        if (
            tuple(permutation) != sources
            or set(permutation.values()) != set(sources)
            or any(permutation[source] == source for source in sources)
        ):
            raise ProtocolError(
                "Source-identity permutation control must be a canonical derangement."
            )
        if (
            isinstance(self.permutation_index, bool)
            or not isinstance(self.permutation_index, Integral)
            or not 1 <= int(self.permutation_index) < len(sources)
        ):
            raise ProtocolError("Source-identity permutation index is invalid.")
        expected = {
            source: sources[(index + int(self.permutation_index)) % len(sources)]
            for index, source in enumerate(sources)
        }
        if permutation != expected:
            raise ProtocolError("Source-identity permutation mapping drifted.")
        if tuple(self.training_seeds) != FIXED_TRAINING_SEEDS:
            raise ProtocolError("Target proxy-policy training-seed drift detected.")
        if self.policy_semantics != POLICY_SEMANTICS:
            raise ProtocolError("Target proxy-policy semantics drifted.")
        if (
            self.labels_consumed
            or self.evaluation_overlap
            or self.source_expert_updated
            or self.actions_constructed
        ):
            raise ProtocolError("Target proxy-policy boundary failed closed.")
        object.__setattr__(
            self, "source_identity_permutation", MappingProxyType(permutation)
        )
        object.__setattr__(self, "permutation_index", int(self.permutation_index))

    @property
    def g_summary(self) -> ProxyRankSummary:
        return self.global_summary

    @property
    def s_summary(self) -> ProxyRankSummary:
        return self.support_summary

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_target": self.outer_target,
            "candidate_sources": list(self.candidate_sources),
            "global_summary": self.global_summary.to_payload(),
            "support_summary": self.support_summary.to_payload(),
            "source_identity_permutation": dict(self.source_identity_permutation),
            "permutation_index": self.permutation_index,
            "permutation_semantics": PERMUTATION_SEMANTICS,
            "training_seeds": list(self.training_seeds),
            "policy_semantics": self.policy_semantics,
            "labels_consumed": self.labels_consumed,
            "evaluation_overlap": self.evaluation_overlap,
            "source_expert_updated": self.source_expert_updated,
            "actions_constructed": self.actions_constructed,
        }


def canonical_source_ids(values: Iterable[object]) -> tuple[str, ...]:
    """Return unique source identifiers in the repository's canonical order."""

    sources: set[str] = set()
    for raw_source in values:
        source = str(raw_source)
        if not source or source.strip() != source or source in sources:
            raise ProtocolError(
                "Proxy-policy source identifiers must be unique and nonempty."
            )
        sources.add(source)
    if not sources:
        raise ProtocolError("Proxy-policy source identifiers cannot be empty.")
    return tuple(sorted(sources))


def _finite_source_mapping(
    values: Mapping[object, object],
    *,
    expected_sources: tuple[str, ...],
    value_name: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ProtocolError(f"{value_name.capitalize()} must be a mapping.")
    normalized: dict[str, float] = {}
    try:
        for raw_source, raw_value in values.items():
            source = str(raw_source)
            if not source or source.strip() != source or source in normalized:
                raise ProtocolError(f"{value_name.capitalize()} source keys are invalid.")
            if isinstance(raw_value, bool):
                raise ProtocolError(f"{value_name.capitalize()} must be finite.")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ProtocolError(f"{value_name.capitalize()} must be finite.")
            normalized[source] = value
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{value_name.capitalize()} must be finite.") from exc
    if tuple(sorted(normalized)) != expected_sources:
        raise ProtocolError(f"{value_name.capitalize()} grid is incomplete.")
    return {source: normalized[source] for source in expected_sources}


def _positive_count_mapping(
    values: Mapping[object, object],
    *,
    expected_keys: tuple[str, ...],
    value_name: str,
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ProtocolError(f"{value_name.capitalize()} must be a mapping.")
    normalized: dict[str, int] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        if (
            not key
            or key.strip() != key
            or key in normalized
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, Integral)
            or int(raw_value) <= 0
        ):
            raise ProtocolError(f"{value_name.capitalize()} is invalid.")
        normalized[key] = int(raw_value)
    if tuple(sorted(normalized)) != expected_keys:
        raise ProtocolError(f"{value_name.capitalize()} grid is incomplete.")
    return {key: normalized[key] for key in expected_keys}


__all__ = (
    "FIXED_TRAINING_SEEDS",
    "GLOBAL_AGGREGATION_SEMANTICS",
    "GLOBAL_POLICY_ID",
    "GLOBAL_PSEUDOQUERY_ROLE",
    "NORMALIZED_MIDRANK_SEMANTICS",
    "PERMUTATION_SEMANTICS",
    "POLICY_SEMANTICS",
    "PRIORITY_SEMANTICS",
    "PROXY_ENERGY_SEMANTICS",
    "QUERY_ROLES",
    "REPLICA_AGGREGATION_SEMANTICS",
    "SUPPORT_AGGREGATION_SEMANTICS",
    "SUPPORT_POLICY_ID",
    "TARGET_SUPPORT_ROLE",
    "CaseProxyBallot",
    "FreshProxyScoreRow",
    "ProxyRankSummary",
    "TargetProxyPolicySummary",
    "canonical_source_ids",
)
