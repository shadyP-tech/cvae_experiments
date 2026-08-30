"""Leakage-safe fold identities for HARP development.

Every constructor validates the outer target ``H``, pseudo-query ``q``,
candidate expert ``e`` and (when present) inner donor ``r`` before downstream
code can normalize features, fit a model, or inspect an outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .hashing import canonical_hash


def canonical_id(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ProtocolError(f"HARP {name} must be a string identity.")
    if not value or value.strip() != value or "\x00" in value:
        raise ProtocolError(f"HARP {name} must be nonempty and canonical.")
    return value


def canonical_centers(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ProtocolError("HARP center universe must be an ordered sequence.")
    centers = tuple(canonical_id(value, name="center") for value in values)
    if len(centers) < 4 or len(centers) != len(set(centers)):
        raise ProtocolError("HARP center universe requires at least four unique centers.")
    if centers != tuple(sorted(centers)):
        raise ProtocolError("HARP center universe must use canonical sorted order.")
    return centers


def validate_hqe(
    *, outer_target: object, pseudo_query: object, candidate_source: object
) -> tuple[str, str, str]:
    outer = canonical_id(outer_target, name="outer target H")
    query = canonical_id(pseudo_query, name="pseudo-query q")
    source = canonical_id(candidate_source, name="candidate source e")
    if outer == query:
        raise ProtocolError("HARP pseudo-query q must differ from outer target H.")
    if source in {outer, query}:
        raise ProtocolError("HARP candidate source e must exclude outer H and query q.")
    return outer, query, source


def validate_hqer(
    *,
    outer_target: object,
    pseudo_query: object,
    candidate_source: object,
    inner_donor: object,
) -> tuple[str, str, str, str]:
    outer, query, source = validate_hqe(
        outer_target=outer_target,
        pseudo_query=pseudo_query,
        candidate_source=candidate_source,
    )
    donor = canonical_id(inner_donor, name="inner donor r")
    if donor in {outer, query, source}:
        raise ProtocolError("HARP inner donor r must exclude H, q, and e.")
    return outer, query, source, donor


def development_queries(
    outer_target: object, *, centers: object
) -> tuple[str, ...]:
    universe = canonical_centers(centers)
    outer = canonical_id(outer_target, name="outer target H")
    if outer not in universe:
        raise ProtocolError("HARP outer target H is outside the center universe.")
    return tuple(center for center in universe if center != outer)


def legal_sources(
    *, outer_target: object, pseudo_query: object, centers: object
) -> tuple[str, ...]:
    universe = canonical_centers(centers)
    outer = canonical_id(outer_target, name="outer target H")
    query = canonical_id(pseudo_query, name="pseudo-query q")
    if outer not in universe or query not in universe or outer == query:
        raise ProtocolError("HARP H/q identities are illegal for this center universe.")
    return tuple(center for center in universe if center not in {outer, query})


def legal_inner_donors(
    *,
    outer_target: object,
    pseudo_query: object,
    candidate_source: object,
    centers: object,
) -> tuple[str, ...]:
    universe = canonical_centers(centers)
    outer, query, source = validate_hqe(
        outer_target=outer_target,
        pseudo_query=pseudo_query,
        candidate_source=candidate_source,
    )
    if any(value not in universe for value in (outer, query, source)):
        raise ProtocolError("HARP H/q/e identities are outside the center universe.")
    return tuple(
        center for center in universe if center not in {outer, query, source}
    )


@dataclass(frozen=True)
class HarpOuterFold:
    outer_target: str
    centers: tuple[str, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = canonical_centers(self.centers)
        outer = canonical_id(self.outer_target, name="outer target H")
        if outer not in centers:
            raise ProtocolError("HARP outer target H is outside its fold universe.")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "outer_target", outer)
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_outer_fold_v1",
                    "outer_target": outer,
                    "centers": list(centers),
                    "outer_target_excluded_before_transform": True,
                }
            ),
        )

    @property
    def queries(self) -> tuple[str, ...]:
        return development_queries(self.outer_target, centers=self.centers)


@dataclass(frozen=True)
class HarpNestedFold:
    outer_target: str
    pseudo_query: str
    candidate_source: str
    inner_donor: str
    centers: tuple[str, ...]
    fold_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = canonical_centers(self.centers)
        outer, query, source, donor = validate_hqer(
            outer_target=self.outer_target,
            pseudo_query=self.pseudo_query,
            candidate_source=self.candidate_source,
            inner_donor=self.inner_donor,
        )
        if any(value not in centers for value in (outer, query, source, donor)):
            raise ProtocolError("HARP nested-fold identity is outside its center universe.")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "outer_target", outer)
        object.__setattr__(self, "pseudo_query", query)
        object.__setattr__(self, "candidate_source", source)
        object.__setattr__(self, "inner_donor", donor)
        object.__setattr__(
            self,
            "fold_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_nested_fold_v1",
                    "outer_target": outer,
                    "pseudo_query": query,
                    "candidate_source": source,
                    "inner_donor": donor,
                    "centers": list(centers),
                    "all_four_roles_distinct_before_transform": True,
                }
            ),
        )


__all__ = (
    "HarpNestedFold",
    "HarpOuterFold",
    "canonical_centers",
    "canonical_id",
    "development_queries",
    "legal_inner_donors",
    "legal_sources",
    "validate_hqe",
    "validate_hqer",
)
