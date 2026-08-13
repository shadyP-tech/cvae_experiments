"""Equal-center donor priors with the exact `q not in {H,e}` fence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, candidate_sources
from .products import DirectionalGain, DonorDirectionalPrior


def compute_donor_priors(
    directional_gains_by_source: Mapping[str, Sequence[DirectionalGain]],
    *,
    heldout_center: object,
) -> tuple[DonorDirectionalPrior, ...]:
    """Average the seven eligible query-center gains for each `(H,e,d)`.

    A source-keyed mapping is mandatory because the label capability is granted
    independently for each `(H,e)` donor scope.
    """

    heldout = str(heldout_center)
    expected_sources = candidate_sources(heldout)
    if set(directional_gains_by_source) != set(expected_sources):
        raise ProtocolError("Abstention-router donor source grant topology drifted.")
    output: list[DonorDirectionalPrior] = []
    for source in expected_sources:
        grant = tuple(directional_gains_by_source[source])
        eligible_queries = tuple(center for center in CENTERS if center not in {heldout, source})
        if any(row.query_center in {heldout, source} for row in grant):
            raise ProtocolError("Abstention-router donor grant contains H or e labels.")
        for direction in DIRECTION_IDS:
            selected = tuple(
                sorted(
                    (
                        row
                        for row in grant
                        if row.source == source and row.direction == direction
                    ),
                    key=lambda row: CENTERS.index(row.query_center),
                )
            )
            if (
                tuple(row.query_center for row in selected) != eligible_queries
                or len({row.query_center for row in selected}) != len(selected)
            ):
                raise ProtocolError(
                    "Abstention-router donor prior requires exactly q outside H and e."
                )
            exact = sum((row.fraction for row in selected), Fraction(0, 1)) / len(
                selected
            )
            output.append(
                DonorDirectionalPrior(
                    heldout,
                    source,
                    direction,
                    eligible_queries,
                    tuple(row.gain_hash for row in selected),
                    exact.numerator,
                    exact.denominator,
                    float(exact),
                )
            )
    return tuple(output)


compute_donor_directional_priors = compute_donor_priors


__all__ = ("compute_donor_directional_priors", "compute_donor_priors")
