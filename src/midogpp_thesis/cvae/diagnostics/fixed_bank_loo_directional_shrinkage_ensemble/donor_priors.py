"""Strict H/e-excluded, equal-center directional donor priors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import CENTERS, DIRECTION_IDS, candidate_sources
from .products import CaseActionConfusion, DirectionalGain, DonorPrior
from .scoring import directional_hard_flip_gain


def compute_donor_prior(
    rows: Sequence[CaseActionConfusion],
    *,
    heldout_center: str,
    source: str,
    direction: str,
) -> DonorPrior:
    """Compute G_d(H,e) as the exact equal-center mean over q outside {H,e}."""

    heldout = str(heldout_center)
    candidate = str(source)
    if heldout not in CENTERS or candidate not in candidate_sources(heldout):
        raise ProtocolError("DCSE donor prior violates the target-expert exclusion.")
    if direction not in DIRECTION_IDS:
        raise ProtocolError("DCSE donor prior direction drifted.")
    legal_queries = tuple(center for center in CENTERS if center not in {heldout, candidate})
    gains: list[DirectionalGain] = []
    for query in legal_queries:
        query_cases = tuple(
            sorted(
                {
                    row.case_id
                    for row in rows
                    if row.target_center == query and row.action_id.endswith(f"={candidate}")
                }
            )
        )
        if not query_cases:
            raise ProtocolError("DCSE donor prior lacks one or more legal query centers.")
        gains.append(
            directional_hard_flip_gain(
                rows,
                query_center=query,
                source=candidate,
                direction=direction,
                contributing_case_ids=query_cases,
                label_scope=f"donor::heldout_H={heldout}::source_e={candidate}::query_q={query}",
            )
        )
    return DonorPrior(heldout, candidate, direction, tuple(gains))


def compute_donor_priors(
    rows: Sequence[CaseActionConfusion]
    | Mapping[str, Sequence[CaseActionConfusion]],
    *,
    heldout_center: str,
) -> tuple[DonorPrior, ...]:
    target = str(heldout_center)
    if isinstance(rows, Mapping):
        expected = candidate_sources(target)
        if tuple(rows) != expected:
            raise ProtocolError(
                "DCSE donor count mapping must contain one canonical q-notin-H-e "
                "capability per source."
            )
        scoped_by_source = {source: tuple(rows[source]) for source in expected}
    else:
        # This form is useful for synthetic science fixtures. Production
        # orchestration should pass the source-keyed mapping so each label I/O
        # capability is physically scoped before decoding.
        values = tuple(rows)
        scoped_by_source = {source: values for source in candidate_sources(target)}
    return tuple(
        compute_donor_prior(
            scoped_by_source[source],
            heldout_center=target,
            source=source,
            direction=direction,
        )
        for source in candidate_sources(target)
        for direction in DIRECTION_IDS
    )


def donor_prior_index(
    priors: Sequence[DonorPrior],
) -> Mapping[tuple[str, str], DonorPrior]:
    rows = tuple(priors)
    if not rows:
        raise ProtocolError("DCSE donor-prior index cannot be empty.")
    targets = {row.heldout_center for row in rows}
    if len(targets) != 1:
        raise ProtocolError("DCSE donor-prior index cannot mix held-out centers.")
    target = next(iter(targets))
    expected = tuple(
        (source, direction)
        for source in candidate_sources(target)
        for direction in DIRECTION_IDS
    )
    result = {(row.source, row.direction): row for row in rows}
    if tuple(result) != expected:
        raise ProtocolError("DCSE donor-prior index lacks exact source/direction coverage.")
    return MappingProxyType(result)


__all__ = (
    "compute_donor_prior",
    "compute_donor_priors",
    "donor_prior_index",
)
