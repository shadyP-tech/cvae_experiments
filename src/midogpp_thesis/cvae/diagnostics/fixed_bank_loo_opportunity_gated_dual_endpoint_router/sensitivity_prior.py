"""Terminal-only delete-center donor priors; never used by the primary route."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, candidate_sources
from .hashing import canonical_hash
from .response_products import CaseActionConfusion
from .response_scoring import directional_hard_flip_gain


@dataclass(frozen=True, order=True)
class DeleteCenterDonorPrior:
    heldout_center: str
    deleted_query_center: str
    source: str
    direction: str
    query_centers: tuple[str, ...]
    numerator: int
    denominator: int
    value: float
    prior_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        expected = tuple(
            center
            for center in CENTERS
            if center not in {self.heldout_center, self.source, self.deleted_query_center}
        )
        exact = Fraction(int(self.numerator), int(self.denominator))
        if (
            self.deleted_query_center not in CENTERS
            or self.source not in candidate_sources(self.heldout_center)
            or self.direction not in DIRECTION_IDS
            or tuple(self.query_centers) != expected
            or not expected
            or float(self.value) != float(exact)
        ):
            raise ProtocolError("OGDE delete-center donor prior drifted.")
        object.__setattr__(self, "numerator", exact.numerator)
        object.__setattr__(self, "denominator", exact.denominator)
        object.__setattr__(self, "value", float(exact))
        object.__setattr__(self, "prior_hash", canonical_hash(self.to_payload()))

    @property
    def exact(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_delete_center_donor_prior_v1",
            "heldout_center": self.heldout_center,
            "deleted_query_center": self.deleted_query_center,
            "source": self.source,
            "direction": self.direction,
            "query_centers": list(self.query_centers),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "terminal_sensitivity_only": True,
        }


def compute_delete_center_donor_priors(
    terminal_confusions: Sequence[CaseActionConfusion],
    *,
    heldout_center: object,
    deleted_query_center: object,
) -> tuple[DeleteCenterDonorPrior, ...]:
    heldout, deleted = str(heldout_center), str(deleted_query_center)
    rows = tuple(terminal_confusions)
    output: list[DeleteCenterDonorPrior] = []
    for source in candidate_sources(heldout):
        queries = tuple(center for center in CENTERS if center not in {heldout, source, deleted})
        for direction in DIRECTION_IDS:
            gains = []
            for query in queries:
                scoped = tuple(
                    row
                    for row in rows
                    if row.target_center == query and row.action_id.endswith(f"={source}")
                )
                cases = tuple(sorted({row.case_id for row in scoped}))
                scopes = {row.label_scope for row in scoped}
                if not cases or len(scopes) != 1:
                    raise ProtocolError("OGDE delete-center prior lacks a legal query center.")
                gains.append(
                    directional_hard_flip_gain(
                        scoped,
                        query_center=query,
                        source=source,
                        direction=direction,
                        contributing_case_ids=cases,
                        label_scope=next(iter(scopes)),
                    ).exact
                )
            exact = sum(gains, Fraction(0)) / len(gains)
            output.append(
                DeleteCenterDonorPrior(
                    heldout,
                    deleted,
                    source,
                    direction,
                    queries,
                    exact.numerator,
                    exact.denominator,
                    float(exact),
                )
            )
    return tuple(output)


__all__ = ("DeleteCenterDonorPrior", "compute_delete_center_donor_priors")
