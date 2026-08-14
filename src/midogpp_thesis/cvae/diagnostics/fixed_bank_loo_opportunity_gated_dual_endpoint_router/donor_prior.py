"""Exact equal-center donor priors with the q-not-in-{H,e} firewall."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, candidate_sources
from .hashing import canonical_hash, require_sha256
from .response_scoring import CaseActionConfusion, DirectionalGain, directional_hard_flip_gain


@dataclass(frozen=True, order=True)
class DonorPrior:
    heldout_center: str
    source: str
    direction: str
    query_centers: tuple[str, ...]
    query_gain_hashes: tuple[str, ...]
    numerator: int
    denominator: int
    value: float
    prior_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        queries = tuple(str(value) for value in self.query_centers)
        hashes = tuple(require_sha256(value, "query_gain_hash") for value in self.query_gain_hashes)
        expected = tuple(center for center in CENTERS if center not in {self.heldout_center, self.source})
        denominator = int(self.denominator)
        if (
            self.source not in candidate_sources(self.heldout_center)
            or self.direction not in DIRECTION_IDS
            or queries != expected
            or len(hashes) != len(expected)
            or denominator <= 0
        ):
            raise ProtocolError("OGDE donor prior scope drifted.")
        exact = Fraction(int(self.numerator), denominator)
        if float(self.value) != float(exact):
            raise ProtocolError("OGDE donor prior fraction drifted.")
        object.__setattr__(self, "query_centers", queries)
        object.__setattr__(self, "query_gain_hashes", hashes)
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "value", float(exact))
        object.__setattr__(self, "prior_hash", canonical_hash(self._unhashed()))

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_donor_prior_v1",
            "heldout_center": self.heldout_center,
            "source": self.source,
            "direction": self.direction,
            "query_centers": list(self.query_centers),
            "query_gain_hashes": list(self.query_gain_hashes),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "query_excludes_heldout_and_source": True,
            "equal_center_mean": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prior_hash": self.prior_hash}


def compute_donor_prior(
    rows: Sequence[CaseActionConfusion],
    *,
    heldout_center: object,
    source: object,
    direction: object,
) -> DonorPrior:
    heldout = str(heldout_center)
    candidate = str(source)
    direction_id = str(direction)
    if candidate not in candidate_sources(heldout) or direction_id not in DIRECTION_IDS:
        raise ProtocolError("OGDE donor prior target/source/direction drifted.")
    queries = tuple(center for center in CENTERS if center not in {heldout, candidate})
    gains: list[DirectionalGain] = []
    for query in queries:
        query_rows = tuple(
            row
            for row in rows
            if row.target_center == query and row.action_id.endswith(f"={candidate}")
        )
        cases = tuple(sorted({row.case_id for row in query_rows}))
        scopes = {row.label_scope for row in query_rows}
        if not cases or len(scopes) != 1:
            raise ProtocolError("OGDE donor grant lacks one legal query center.")
        gains.append(
            directional_hard_flip_gain(
                query_rows,
                query_center=query,
                source=candidate,
                direction=direction_id,
                contributing_case_ids=cases,
                label_scope=next(iter(scopes)),
            )
        )
    exact = sum((gain.exact for gain in gains), Fraction(0)) / len(gains)
    return DonorPrior(
        heldout,
        candidate,
        direction_id,
        queries,
        tuple(gain.gain_hash for gain in gains),
        exact.numerator,
        exact.denominator,
        float(exact),
    )


def compute_donor_priors(
    counts_by_source: Mapping[str, Sequence[CaseActionConfusion]],
    *,
    heldout_center: object,
) -> tuple[DonorPrior, ...]:
    heldout = str(heldout_center)
    expected_sources = candidate_sources(heldout)
    if tuple(counts_by_source) != expected_sources:
        raise ProtocolError("OGDE donor grants require canonical source-keyed coverage.")
    output = tuple(
        compute_donor_prior(
            counts_by_source[source],
            heldout_center=heldout,
            source=source,
            direction=direction,
        )
        for source in expected_sources
        for direction in DIRECTION_IDS
    )
    if any(heldout in row.query_centers or row.source in row.query_centers for row in output):
        raise ProtocolError("OGDE donor prior contains forbidden H/e query labels.")
    return output


__all__ = ("DonorPrior", "compute_donor_prior", "compute_donor_priors")
