"""Hash-bound evidence for H/J-recomposed pseudo endpoint surfaces.

The pseudo endpoints are derived before pseudo-case evaluation labels open.  We
persist only aggregate donor priors, capability identities, and endpoint
probabilities.  Raw labels and per-sample paths never cross this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .candidate_runtime import SOURCE_EXCLUSION_ROLE
from .constants import CENTERS, DIRECTION_IDS, candidate_sources
from .contracts import EndpointCasePrediction
from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True)
class PseudoSourcePriorEvidence:
    """One outer-H/target-J donor-prior surface with scoped capability lineage."""

    outer_center: str
    target_center: str
    priors: Mapping[tuple[str, str], float]
    capability_hashes: Mapping[str, str]
    source_prior_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        target = str(self.target_center)
        sources = candidate_sources(target)
        values = {
            (str(source), str(direction)): float(value)
            for (source, direction), value in self.priors.items()
        }
        capabilities = {
            str(source): require_sha256(value, "source_prior_capability_hash")
            for source, value in self.capability_hashes.items()
        }
        expected_keys = tuple(
            (source, direction)
            for source in sources
            for direction in DIRECTION_IDS
        )
        expected_capability_sources = tuple(
            source for source in sources if source != outer
        )
        if (
            outer not in CENTERS
            or target not in CENTERS
            or outer == target
            or tuple(values) != expected_keys
            or tuple(capabilities) != expected_capability_sources
            or any(values[(outer, direction)] != 0.0 for direction in DIRECTION_IDS)
        ):
            raise ProtocolError("CBPUPR pseudo source-prior evidence drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "priors", MappingProxyType(values))
        object.__setattr__(self, "capability_hashes", MappingProxyType(capabilities))
        object.__setattr__(
            self,
            "source_prior_hash",
            canonical_hash(self._unhashed_payload()),
        )

    def _unhashed_payload(self) -> dict[str, object]:
        sources = candidate_sources(self.target_center)
        return {
            "schema_version": "fixed_bank_cbpupr_pseudo_source_prior_v1",
            "outer_center": self.outer_center,
            "target_center": self.target_center,
            "source_excluded_centers": sorted(
                (self.outer_center, self.target_center)
            ),
            "source_excluded_centers_role": SOURCE_EXCLUSION_ROLE,
            "capability_hashes": [
                [source, self.capability_hashes[source]]
                for source in sources
                if source != self.outer_center
            ],
            "priors": [
                [source, direction, self.priors[(source, direction)]]
                for source in sources
                for direction in DIRECTION_IDS
            ],
            "support_labels_used_indirectly": True,
            "raw_labels_persisted": False,
            "target_evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._unhashed_payload()
        return {**payload, "source_prior_hash": self.source_prior_hash}


@dataclass(frozen=True)
class PseudoEndpointEvidence:
    """One H/J/d endpoint prediction bound to its H-excluded prior surface."""

    outer_center: str
    prediction: EndpointCasePrediction
    source_prior_hash: str
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        prediction = self.prediction
        if (
            outer not in CENTERS
            or outer == prediction.center
            or not isinstance(prediction, EndpointCasePrediction)
        ):
            raise ProtocolError("CBPUPR pseudo endpoint evidence drifted.")
        require_sha256(self.source_prior_hash, "pseudo_source_prior_hash")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(
            self,
            "evidence_hash",
            canonical_hash(self._unhashed_payload()),
        )

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_pseudo_endpoint_evidence_v1",
            "outer_center": self.outer_center,
            "target_center": self.prediction.center,
            "case_id": self.prediction.case_id,
            "sample_ids": list(self.prediction.sample_ids),
            "source_excluded_centers": sorted(
                (self.outer_center, self.prediction.center)
            ),
            "source_excluded_centers_role": SOURCE_EXCLUSION_ROLE,
            "source_prior_hash": self.source_prior_hash,
            "state_hash": self.prediction.state_hash,
            "prediction_hash": self.prediction.prediction_hash,
            "support_labels_used_indirectly": True,
            "raw_labels_persisted": False,
            "target_evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        payload = self._unhashed_payload()
        return {**payload, "evidence_hash": self.evidence_hash}


def build_pseudo_source_prior_evidence(
    *,
    outer_center: str,
    target_center: str,
    priors: Mapping[tuple[str, str], float],
    capability_events: Sequence[Mapping[str, object]],
) -> PseudoSourcePriorEvidence:
    """Bind each non-H prior source to its exact firewall event payload."""

    outer, target = str(outer_center), str(target_center)
    hashes: dict[str, str] = {}
    for source in candidate_sources(target):
        if source == outer:
            continue
        role = f"source_prior::outer_H={outer}::J={target}::source={source}"
        matches = [row for row in capability_events if row.get("role") == role]
        if len(matches) != 1:
            raise ProtocolError("CBPUPR pseudo source-prior capability drifted.")
        event = dict(matches[0])
        if (
            event.get("outer_target_center") != outer
            or event.get("target_center") != target
            or event.get("case_id") is not None
            or event.get("excluded_centers")
            != sorted((outer, target, source))
            or event.get("raw_labels_persisted") is not False
        ):
            raise ProtocolError("CBPUPR pseudo source-prior event scope drifted.")
        hashes[source] = canonical_hash(event)
    return PseudoSourcePriorEvidence(outer, target, priors, hashes)


__all__ = (
    "PseudoEndpointEvidence",
    "PseudoSourcePriorEvidence",
    "build_pseudo_source_prior_evidence",
)
