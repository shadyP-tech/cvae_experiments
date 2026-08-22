"""Facade for semantic validation of persisted transport diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from ...runtime.artifact_io import read_json
from .constants import CENTERS
from .hashing import canonical_hash
from .transport_geometry import NumericTransportAudit, StructuralTransportGate
from .validation_candidates import CandidateTopology
from .validation_origin import PhysicalOriginTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail
from .validation_transport_numeric import validate_numeric_transport_rows
from .validation_transport_structural import validate_structural_transport_rows


_TRANSPORT_SCHEMA = "fixed_bank_cbpupr_transport_diagnostics_v1"


@dataclass(frozen=True)
class TransportTopology:
    """Validated, immutable transport products keyed by target center."""

    structural_by_center: Mapping[str, StructuralTransportGate]
    numeric_by_center: Mapping[str, NumericTransportAudit]
    primary_fingerprint_hash_by_center: Mapping[str, str]
    raw_numeric_source_recomputation_available: bool = field(
        default=True, init=False
    )
    validation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        structural = dict(self.structural_by_center)
        numeric = dict(self.numeric_by_center)
        fingerprints = dict(self.primary_fingerprint_hash_by_center)
        expected = set(CENTERS)
        if (
            set(structural) != expected
            or set(numeric) != expected
            or set(fingerprints) != expected
        ):
            fail("transport validation topology")
        for name, value in (
            ("structural_by_center", structural),
            ("numeric_by_center", numeric),
            ("primary_fingerprint_hash_by_center", fingerprints),
        ):
            object.__setattr__(self, name, MappingProxyType(value))
        object.__setattr__(
            self,
            "validation_hash",
            canonical_hash(
                {
                    "schema_version": "fixed_bank_cbpupr_transport_validation_v1",
                    "structural_gate_hashes": [
                        structural[center].gate_hash for center in CENTERS
                    ],
                    "numeric_audit_hashes": [
                        numeric[center].audit_hash for center in CENTERS
                    ],
                    "primary_fingerprint_hashes": [
                        fingerprints[center] for center in CENTERS
                    ],
                    "raw_numeric_source_recomputation_available": True,
                    "numeric_transport_is_authorization_gate": False,
                }
            ),
        )


def validate_transport_diagnostics(
    root: Path,
    *,
    rows: Sequence[Row],
    topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    origin: PhysicalOriginTopology,
) -> TransportTopology:
    """Validate the exact nine structural plus nine numeric persisted rows."""

    observed = tuple(rows)
    payload = read_json(root / "tables/transport_diagnostics.json")
    if (
        payload.get("schema_version") != _TRANSPORT_SCHEMA
        or payload.get("row_count") != len(observed)
        or payload.get("rows") != list(observed)
        or len(observed) != 2 * len(CENTERS)
    ):
        fail("transport diagnostics table envelope")
    structural_rows = observed[: len(CENTERS)]
    numeric_rows = observed[len(CENTERS) :]
    if (
        tuple(str(row.get("target_center", "")) for row in structural_rows)
        != CENTERS
        or tuple(str(row.get("target_center", "")) for row in numeric_rows)
        != CENTERS
    ):
        fail("transport diagnostic row order")

    structural = validate_structural_transport_rows(
        root,
        rows=structural_rows,
        topology=topology,
        candidate_topology=candidate_topology,
    )
    numeric, fingerprints = validate_numeric_transport_rows(
        root,
        rows=numeric_rows,
        topology=topology,
        source_fingerprints=origin.fingerprints,
    )
    return TransportTopology(structural, numeric, fingerprints)


__all__ = ("TransportTopology", "validate_transport_diagnostics")
