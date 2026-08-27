"""Immutable, label-free contracts for SCEPTRE source-family routing.

The selectable scientific unit is one source center.  Training and generation
seeds remain locked replications of that unit and can never become candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    legal_routing_sources,
)
from ...generation.contracts import SourceGenerationKey
from ...protocol import ProtocolError


PROXY_SCORE_SEMANTICS = "PROXY_ENERGY_RANK"
RAW_ROUTE_POLICY_ID = "G_PROXY_ENERGY_RANK"
EXACT_B_CONTROL_ID = "B"
EXACT_TIE_REASON = "EXACT_SCORE_TIE"
UNSUPPORTED_EVIDENCE_REASON = "UNSUPPORTED_EVIDENCE"
NONFINITE_EVIDENCE_REASON = "NONFINITE_EVIDENCE"
INVALID_EVIDENCE_REASON = "INVALID_EVIDENCE"
EXACT_B_FALLBACK_REASONS = frozenset(
    {
        EXACT_TIE_REASON,
        UNSUPPORTED_EVIDENCE_REASON,
        NONFINITE_EVIDENCE_REASON,
        INVALID_EVIDENCE_REASON,
    }
)


def _identifier(value: object, name: str) -> str:
    identifier = str(value)
    if not identifier or identifier.strip() != identifier:
        raise ProtocolError(f"SCEPTRE {name} is invalid.")
    return identifier


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"SCEPTRE {name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"SCEPTRE {name} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"SCEPTRE {name} must be finite.")
    return parsed


def _canonical_stream_key(key: SourceGenerationKey) -> SourceGenerationKey:
    if not isinstance(key, SourceGenerationKey):
        raise ProtocolError("SCEPTRE source families require SourceGenerationKey rows.")
    raw_class_seeds = dict(key.class_seed_by_label)
    if set(raw_class_seeds) != {"0", "1"} or any(
        isinstance(value, bool) for value in raw_class_seeds.values()
    ):
        raise ProtocolError("SCEPTRE source stream class-seed coverage drifted.")
    try:
        class_seeds = MappingProxyType(
            {label: int(raw_class_seeds[label]) for label in ("0", "1")}
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("SCEPTRE source stream class seeds are invalid.") from exc
    return SourceGenerationKey(
        source_center=_identifier(key.source_center, "source center"),
        training_seed=int(key.training_seed),
        generation_seed=int(key.generation_seed),
        expert_lock_hash=_identifier(key.expert_lock_hash, "expert lock hash"),
        stream_id=_identifier(key.stream_id, "stream id"),
        class_seed_by_label=class_seeds,
        max_samples_per_class=int(key.max_samples_per_class),
        equal_union_prefix_per_class=int(key.equal_union_prefix_per_class),
    )


@dataclass(frozen=True)
class SourceFamily:
    """One selectable source-center family with an exact three-by-three grid."""

    target_center: str
    source_center: str
    generation_lock_hash: str
    stream_keys: tuple[SourceGenerationKey, ...]
    family_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        source = _identifier(self.source_center, "source center")
        lock_hash = _identifier(self.generation_lock_hash, "generation lock hash")
        if target not in CENTERS or source not in CENTERS:
            raise ProtocolError("SCEPTRE source family contains an unknown center.")
        if source == target:
            raise ProtocolError("SCEPTRE source family included the target expert.")

        keys = tuple(_canonical_stream_key(key) for key in self.stream_keys)
        expected_grid = {
            (training_seed, generation_seed)
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        }
        observed_grid = {
            (key.training_seed, key.generation_seed) for key in keys
        }
        if len(keys) != len(expected_grid) or observed_grid != expected_grid:
            raise ProtocolError(
                "SCEPTRE source family must contain the exact three-by-three seed grid."
            )
        if any(key.source_center != source for key in keys):
            raise ProtocolError("SCEPTRE source family mixed source centers.")
        if len({key.stream_id for key in keys}) != len(keys):
            raise ProtocolError("SCEPTRE source family contains duplicate stream ids.")
        for training_seed in TRAINING_SEEDS:
            expert_hashes = {
                key.expert_lock_hash
                for key in keys
                if key.training_seed == training_seed
            }
            if len(expert_hashes) != 1:
                raise ProtocolError(
                    "SCEPTRE generation replicas do not share one expert per training seed."
                )
        canonical_keys = tuple(
            sorted(
                keys,
                key=lambda key: (key.training_seed, key.generation_seed),
            )
        )
        unhashed = {
            "schema_version": "midogpp_sceptre_source_family_v1",
            "scientific_unit": "source_center",
            "target_center": target,
            "source_center": source,
            "generation_lock_hash": lock_hash,
            "training_seeds_are_replications": True,
            "generation_seeds_are_replications": True,
            "stream_keys": [key.to_payload() for key in canonical_keys],
        }
        expected_hash = stable_hash(unhashed)
        if self.family_hash and self.family_hash != expected_hash:
            raise ProtocolError("SCEPTRE source-family semantic hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "source_center", source)
        object.__setattr__(self, "generation_lock_hash", lock_hash)
        object.__setattr__(self, "stream_keys", canonical_keys)
        object.__setattr__(self, "family_hash", expected_hash)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_source_family_v1",
            "scientific_unit": "source_center",
            "target_center": self.target_center,
            "source_center": self.source_center,
            "generation_lock_hash": self.generation_lock_hash,
            "training_seeds_are_replications": True,
            "generation_seeds_are_replications": True,
            "stream_keys": [key.to_payload() for key in self.stream_keys],
            "family_hash": self.family_hash,
        }


@dataclass(frozen=True)
class CandidateMenu:
    """The exact eight-family candidate menu for one held-out target."""

    target_center: str
    generation_lock_hash: str
    families: tuple[SourceFamily, ...]
    menu_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        lock_hash = _identifier(self.generation_lock_hash, "generation lock hash")
        if target not in CENTERS:
            raise ProtocolError("SCEPTRE candidate menu contains an unknown target.")
        families = tuple(self.families)
        expected_sources = legal_routing_sources(target)
        observed_sources = tuple(family.source_center for family in families)
        if observed_sources != expected_sources:
            raise ProtocolError(
                "SCEPTRE candidate menu is not the canonical target-excluded source pool."
            )
        if any(
            not isinstance(family, SourceFamily)
            or family.target_center != target
            or family.generation_lock_hash != lock_hash
            for family in families
        ):
            raise ProtocolError("SCEPTRE candidate-family lineage drifted.")
        unhashed = {
            "schema_version": "midogpp_sceptre_candidate_menu_v1",
            "target_center": target,
            "generation_lock_hash": lock_hash,
            "candidate_unit": "source_center_family",
            "target_expert_excluded": True,
            "seed_selection_allowed": False,
            "family_hashes": [family.family_hash for family in families],
        }
        expected_hash = stable_hash(unhashed)
        if self.menu_hash and self.menu_hash != expected_hash:
            raise ProtocolError("SCEPTRE candidate-menu semantic hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "generation_lock_hash", lock_hash)
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "menu_hash", expected_hash)

    @property
    def candidate_sources(self) -> tuple[str, ...]:
        return tuple(family.source_center for family in self.families)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_candidate_menu_v1",
            "target_center": self.target_center,
            "generation_lock_hash": self.generation_lock_hash,
            "candidate_unit": "source_center_family",
            "target_expert_excluded": True,
            "seed_selection_allowed": False,
            "family_hashes": [family.family_hash for family in self.families],
            "menu_hash": self.menu_hash,
        }


@dataclass(frozen=True)
class FamilyProxyScore:
    """A label-free proxy energy averaged over the three training replicas."""

    target_center: str
    source_center: str
    training_replica_scores: Mapping[int, float]
    mean_proxy_energy: float | None = None
    exact_nelbo: bool = False
    labels_consumed: bool = False
    score_semantics: str = PROXY_SCORE_SEMANTICS
    score_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        source = _identifier(self.source_center, "source center")
        if target not in CENTERS or source not in CENTERS or source == target:
            raise ProtocolError("SCEPTRE proxy score violates target exclusion.")
        if type(self.exact_nelbo) is not bool or self.exact_nelbo:
            raise ProtocolError("SCEPTRE proxy energy must not be represented as exact NELBO.")
        if type(self.labels_consumed) is not bool or self.labels_consumed:
            raise ProtocolError("SCEPTRE proxy energy must remain label-free.")
        if self.score_semantics != PROXY_SCORE_SEMANTICS:
            raise ProtocolError("SCEPTRE proxy score semantics drifted.")

        try:
            raw_scores = dict(self.training_replica_scores)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("SCEPTRE training-replica scores are invalid.") from exc
        if set(raw_scores) != set(TRAINING_SEEDS):
            raise ProtocolError(
                "SCEPTRE proxy score requires exactly training seeds 17, 42, and 101."
            )
        scores = {
            seed: _finite(raw_scores[seed], "training-replica proxy score")
            for seed in TRAINING_SEEDS
        }
        mean = math.fsum(scores[seed] for seed in TRAINING_SEEDS) / float(
            len(TRAINING_SEEDS)
        )
        if self.mean_proxy_energy is not None and _finite(
            self.mean_proxy_energy, "mean proxy energy"
        ) != mean:
            raise ProtocolError("SCEPTRE mean proxy energy does not replay exactly.")
        frozen_scores = MappingProxyType(scores)
        unhashed = {
            "schema_version": "midogpp_sceptre_family_proxy_score_v1",
            "target_center": target,
            "source_center": source,
            "training_replica_scores": {
                str(seed): scores[seed] for seed in TRAINING_SEEDS
            },
            "aggregation": "arithmetic_mean_over_fixed_training_seeds",
            "mean_proxy_energy": mean,
            "score_semantics": PROXY_SCORE_SEMANTICS,
            "lower_is_better": True,
            "exact_nelbo": False,
            "labels_consumed": False,
        }
        expected_hash = stable_hash(unhashed)
        if self.score_hash and self.score_hash != expected_hash:
            raise ProtocolError("SCEPTRE proxy-score semantic hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "source_center", source)
        object.__setattr__(self, "training_replica_scores", frozen_scores)
        object.__setattr__(self, "mean_proxy_energy", mean)
        object.__setattr__(self, "score_hash", expected_hash)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_family_proxy_score_v1",
            "target_center": self.target_center,
            "source_center": self.source_center,
            "training_replica_scores": {
                str(seed): self.training_replica_scores[seed]
                for seed in TRAINING_SEEDS
            },
            "aggregation": "arithmetic_mean_over_fixed_training_seeds",
            "mean_proxy_energy": self.mean_proxy_energy,
            "score_semantics": self.score_semantics,
            "lower_is_better": True,
            "exact_nelbo": self.exact_nelbo,
            "labels_consumed": self.labels_consumed,
            "score_hash": self.score_hash,
        }


def _expected_normalized_midranks(
    sources: tuple[str, ...], values: Mapping[str, float]
) -> dict[str, float]:
    ordered = sorted(sources, key=lambda source: (values[source], source))
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        tied_value = values[ordered[start]]
        while stop < len(ordered) and values[ordered[stop]] == tied_value:
            stop += 1
        midrank = (float(start + 1) + float(stop)) / 2.0
        normalized = (midrank - 1.0) / float(len(ordered) - 1)
        for source in ordered[start:stop]:
            result[source] = normalized
        start = stop
    return result


@dataclass(frozen=True)
class RankedWinnerSet:
    """A complete exact-tie-preserving lower-is-better ranking."""

    target_center: str
    candidate_sources: tuple[str, ...]
    mean_proxy_energy_by_source: Mapping[str, float]
    normalized_midrank_by_source: Mapping[str, float]
    score_hash_by_source: Mapping[str, str]
    winner_sources: tuple[str, ...]
    lower_is_better: bool = True
    ranking_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        sources = tuple(_identifier(value, "candidate source") for value in self.candidate_sources)
        if target not in CENTERS or sources != legal_routing_sources(target):
            raise ProtocolError("SCEPTRE ranked candidate pool drifted.")
        if type(self.lower_is_better) is not bool or not self.lower_is_better:
            raise ProtocolError("SCEPTRE proxy ranking must be lower-is-better.")
        if set(self.mean_proxy_energy_by_source) != set(sources):
            raise ProtocolError("SCEPTRE proxy ranking score grid is incomplete.")
        values = {
            source: _finite(
                self.mean_proxy_energy_by_source[source], "ranked proxy energy"
            )
            for source in sources
        }
        if set(self.normalized_midrank_by_source) != set(sources):
            raise ProtocolError("SCEPTRE proxy midrank grid is incomplete.")
        observed_ranks = {
            source: _finite(
                self.normalized_midrank_by_source[source], "normalized midrank"
            )
            for source in sources
        }
        expected_ranks = _expected_normalized_midranks(sources, values)
        if observed_ranks != expected_ranks:
            raise ProtocolError("SCEPTRE true-midrank replay failed.")
        if set(self.score_hash_by_source) != set(sources):
            raise ProtocolError("SCEPTRE score-hash grid is incomplete.")
        score_hashes = {
            source: _identifier(self.score_hash_by_source[source], "score hash")
            for source in sources
        }
        best = min(values.values())
        expected_winners = tuple(source for source in sources if values[source] == best)
        winners = tuple(_identifier(value, "winner source") for value in self.winner_sources)
        if winners != expected_winners:
            raise ProtocolError("SCEPTRE winner set is incomplete or order-drifted.")
        unhashed = {
            "schema_version": "midogpp_sceptre_ranked_winner_set_v1",
            "target_center": target,
            "candidate_sources": list(sources),
            "mean_proxy_energy_by_source": values,
            "normalized_midrank_by_source": expected_ranks,
            "score_hash_by_source": score_hashes,
            "winner_sources": list(winners),
            "lower_is_better": True,
            "tie_semantics": "complete_exact_winner_set",
        }
        expected_hash = stable_hash(unhashed)
        if self.ranking_hash and self.ranking_hash != expected_hash:
            raise ProtocolError("SCEPTRE ranking semantic hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "mean_proxy_energy_by_source", MappingProxyType(values))
        object.__setattr__(
            self,
            "normalized_midrank_by_source",
            MappingProxyType(expected_ranks),
        )
        object.__setattr__(self, "score_hash_by_source", MappingProxyType(score_hashes))
        object.__setattr__(self, "winner_sources", winners)
        object.__setattr__(self, "ranking_hash", expected_hash)

    @property
    def has_unique_winner(self) -> bool:
        return len(self.winner_sources) == 1

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_ranked_winner_set_v1",
            "target_center": self.target_center,
            "candidate_sources": list(self.candidate_sources),
            "mean_proxy_energy_by_source": dict(self.mean_proxy_energy_by_source),
            "normalized_midrank_by_source": dict(self.normalized_midrank_by_source),
            "score_hash_by_source": dict(self.score_hash_by_source),
            "winner_sources": list(self.winner_sources),
            "lower_is_better": self.lower_is_better,
            "tie_semantics": "complete_exact_winner_set",
            "ranking_hash": self.ranking_hash,
        }


@dataclass(frozen=True)
class RawRoute:
    """A raw source-family route issued only for a unique proxy winner."""

    target_center: str
    candidate_sources: tuple[str, ...]
    selected_source_center: str
    candidate_menu_hash: str
    ranking_hash: str
    policy_id: str = RAW_ROUTE_POLICY_ID
    raw_route_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        sources = tuple(_identifier(value, "candidate source") for value in self.candidate_sources)
        selected = _identifier(self.selected_source_center, "selected source center")
        if target not in CENTERS or sources != legal_routing_sources(target):
            raise ProtocolError("SCEPTRE raw-route candidate pool drifted.")
        if selected not in sources or selected == target:
            raise ProtocolError("SCEPTRE raw route violates target exclusion.")
        if self.policy_id != RAW_ROUTE_POLICY_ID:
            raise ProtocolError("SCEPTRE raw-route policy identity drifted.")
        menu_hash = _identifier(self.candidate_menu_hash, "candidate menu hash")
        ranking_hash = _identifier(self.ranking_hash, "ranking hash")
        unhashed = {
            "schema_version": "midogpp_sceptre_raw_route_v1",
            "target_center": target,
            "candidate_sources": list(sources),
            "selected_source_center": selected,
            "candidate_menu_hash": menu_hash,
            "ranking_hash": ranking_hash,
            "policy_id": RAW_ROUTE_POLICY_ID,
            "route_unit": "source_center_family",
        }
        expected_hash = stable_hash(unhashed)
        if self.raw_route_hash and self.raw_route_hash != expected_hash:
            raise ProtocolError("SCEPTRE raw-route semantic hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "selected_source_center", selected)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "ranking_hash", ranking_hash)
        object.__setattr__(self, "raw_route_hash", expected_hash)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_raw_route_v1",
            "target_center": self.target_center,
            "candidate_sources": list(self.candidate_sources),
            "selected_source_center": self.selected_source_center,
            "candidate_menu_hash": self.candidate_menu_hash,
            "ranking_hash": self.ranking_hash,
            "policy_id": self.policy_id,
            "route_unit": "source_center_family",
            "raw_route_hash": self.raw_route_hash,
        }


@dataclass(frozen=True)
class ExactBFallback:
    """An exact equal-union fallback caused by tie or inadmissible evidence."""

    target_center: str
    candidate_sources: tuple[str, ...]
    winner_sources: tuple[str, ...]
    candidate_menu_hash: str
    ranking_hash: str
    reason: str = EXACT_TIE_REASON
    control_id: str = EXACT_B_CONTROL_ID
    fallback_hash: str = ""

    def __post_init__(self) -> None:
        target = _identifier(self.target_center, "target center")
        sources = tuple(_identifier(value, "candidate source") for value in self.candidate_sources)
        winners = tuple(_identifier(value, "winner source") for value in self.winner_sources)
        if target not in CENTERS or sources != legal_routing_sources(target):
            raise ProtocolError("SCEPTRE fallback candidate pool drifted.")
        reason = _identifier(self.reason, "fallback reason")
        if reason not in EXACT_B_FALLBACK_REASONS:
            raise ProtocolError("SCEPTRE exact-B fallback reason drifted.")
        if reason == EXACT_TIE_REASON:
            if len(winners) < 2 or any(winner not in sources for winner in winners):
                raise ProtocolError(
                    "SCEPTRE exact-tie fallback requires the complete tied winner set."
                )
            if tuple(source for source in sources if source in set(winners)) != winners:
                raise ProtocolError("SCEPTRE exact-tie winner order drifted.")
        elif winners:
            raise ProtocolError(
                "SCEPTRE inadmissible-evidence fallback cannot invent winner sources."
            )
        if self.control_id != EXACT_B_CONTROL_ID:
            raise ProtocolError("SCEPTRE exact-B fallback identity drifted.")
        menu_hash = _identifier(self.candidate_menu_hash, "candidate menu hash")
        ranking_hash = _identifier(self.ranking_hash, "ranking hash")
        unhashed = {
            "schema_version": "midogpp_sceptre_exact_b_fallback_v1",
            "target_center": target,
            "candidate_sources": list(sources),
            "winner_sources": list(winners),
            "candidate_menu_hash": menu_hash,
            "ranking_hash": ranking_hash,
            "reason": reason,
            "control_id": EXACT_B_CONTROL_ID,
            "fake_tie_breaking": False,
        }
        expected_hash = stable_hash(unhashed)
        if self.fallback_hash and self.fallback_hash != expected_hash:
            raise ProtocolError("SCEPTRE exact-B fallback semantic hash drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "candidate_sources", sources)
        object.__setattr__(self, "winner_sources", winners)
        object.__setattr__(self, "candidate_menu_hash", menu_hash)
        object.__setattr__(self, "ranking_hash", ranking_hash)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "fallback_hash", expected_hash)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_sceptre_exact_b_fallback_v1",
            "target_center": self.target_center,
            "candidate_sources": list(self.candidate_sources),
            "winner_sources": list(self.winner_sources),
            "candidate_menu_hash": self.candidate_menu_hash,
            "ranking_hash": self.ranking_hash,
            "reason": self.reason,
            "control_id": self.control_id,
            "fake_tie_breaking": False,
            "fallback_hash": self.fallback_hash,
        }


RouteDecision = RawRoute | ExactBFallback


__all__ = (
    "CandidateMenu",
    "EXACT_B_CONTROL_ID",
    "EXACT_B_FALLBACK_REASONS",
    "EXACT_TIE_REASON",
    "ExactBFallback",
    "FamilyProxyScore",
    "INVALID_EVIDENCE_REASON",
    "NONFINITE_EVIDENCE_REASON",
    "PROXY_SCORE_SEMANTICS",
    "RAW_ROUTE_POLICY_ID",
    "RankedWinnerSet",
    "RawRoute",
    "RouteDecision",
    "SourceFamily",
    "UNSUPPORTED_EVIDENCE_REASON",
)
