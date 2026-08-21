"""Numeric transport replay from independently reconstructed fingerprints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path

import numpy as np

from .constants import (
    BLOCKED_FINGERPRINT_CONTROL_ID,
    CENTERS,
    FINGERPRINT_FEATURE_COUNT,
    PRIMARY_FINGERPRINT_CONTROL_ID,
    TRANSPORT_MAD_SCALE,
    TRANSPORT_SCALE_FLOOR,
)
from .hashing import require_sha256
from .physical_fingerprint import fingerprint_feature_names
from .posterior_contracts import CONTROL_IDS, PhysicalFingerprintSurface
from .transport_geometry import (
    NumericDimensionAudit,
    NumericTransportAudit,
    audit_numeric_transport,
)
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list, table_rows


_NUMERIC_FIELDS = frozenset(
    "target_center reference_centers dimensions "
    "active_continuous_dimension_count zero_scale_dimensions "
    "zero_scale_novelty_dimensions sparse_pattern_mismatch_count l2_distance "
    "maximum_absolute_distance authorization_gate audit_hash".split()
)
_DIMENSION_FIELDS = frozenset(
    "feature_name feature_kind target_value reference_median reference_mad active "
    "standardized_distance zero_scale_novelty sparse_reference_rate".split()
)


def validate_numeric_transport_rows(
    root: Path,
    *,
    rows: Sequence[Row],
    topology: PlanPosteriorTopology,
    source_fingerprints: Mapping[tuple[str, str], PhysicalFingerprintSurface],
) -> tuple[dict[str, NumericTransportAudit], dict[str, str]]:
    """Rebuild numeric semantics and return audits plus primary hash lineage."""

    observed = tuple(rows)
    if (
        len(observed) != len(CENTERS)
        or tuple(str(row.get("target_center", "")) for row in observed) != CENTERS
    ):
        fail("numeric transport row rectangle")
    fingerprints = _validate_fingerprint_lineage(root, topology)
    expected_source_keys = {
        (center, control) for center in CENTERS for control in CONTROL_IDS
    }
    if set(source_fingerprints) != expected_source_keys:
        fail("numeric transport source fingerprint rectangle")
    primary = {
        center: source_fingerprints[(center, PRIMARY_FINGERPRINT_CONTROL_ID)]
        for center in CENTERS
    }
    names = tuple(
        f"physical_fingerprint_dimension_{index}"
        for index in range(FINGERPRINT_FEATURE_COUNT)
    )
    audits: dict[str, NumericTransportAudit] = {}
    for center, row in zip(CENTERS, observed, strict=True):
        if primary[center].fingerprint_hash != fingerprints[center]:
            fail("numeric transport reconstructed fingerprint lineage")
        internally_reconstructed = _reconstruct_row(row, center=center)
        independently_replayed = audit_numeric_transport(
            target_center=center,
            target_vector=np.mean(
                primary[center].feature_values, axis=0, dtype=np.float64
            ),
            reference_vectors_by_center={
                other: primary[other].feature_values
                for other in CENTERS
                if other != center
            },
            feature_names=names,
        )
        if internally_reconstructed.to_payload() != independently_replayed.to_payload():
            fail("numeric transport primitive-fingerprint replay")
        audits[center] = independently_replayed
    return audits, fingerprints


def _validate_fingerprint_lineage(
    root: Path, topology: PlanPosteriorTopology
) -> dict[str, str]:
    rows = table_rows(root, "physical_fingerprints")
    indexed = index_rows(
        rows, ("center", "control_id"), "transport physical fingerprints"
    )
    expected = {(center, control) for center in CENTERS for control in CONTROL_IDS}
    if set(indexed) != expected:
        fail("transport physical fingerprint rectangle")
    for center in CENTERS:
        case_count = len(topology.cases_by_center[center])
        sample_count = sum(
            len(string_list(topology.plans[(center, case)], "evaluation_sample_ids"))
            for case in topology.cases_by_center[center]
        )
        for control in CONTROL_IDS:
            row = indexed[(center, control)]
            if (
                row.get("schema_version")
                != "fixed_bank_cbpupr_fingerprint_summary_v1"
                or row.get("sample_count") != sample_count
                or row.get("case_count") != case_count
                or string_list(row, "feature_names")
                != fingerprint_feature_names(center)
                or row.get("raw_feature_rows_persisted") is not False
                or row.get("labels_used") is not False
            ):
                fail("transport physical fingerprint lineage")
            require_sha256(row.get("feature_array_sha256"), "fingerprint array hash")
            require_sha256(row.get("source_surface_hash"), "fingerprint source hash")
            require_sha256(row.get("fingerprint_hash"), "fingerprint hash")
            for case in topology.cases_by_center[center]:
                if (
                    topology.models[(center, case, control)].get("fingerprint_hash")
                    != row.get("fingerprint_hash")
                ):
                    fail("transport model/fingerprint lineage")
        if (
            indexed[(center, PRIMARY_FINGERPRINT_CONTROL_ID)].get(
                "source_surface_hash"
            )
            != indexed[(center, BLOCKED_FINGERPRINT_CONTROL_ID)].get(
                "source_surface_hash"
            )
        ):
            fail("transport fingerprint control source lineage")
    return {
        center: str(
            indexed[(center, PRIMARY_FINGERPRINT_CONTROL_ID)]["fingerprint_hash"]
        )
        for center in CENTERS
    }


def _reconstruct_row(row: Row, *, center: str) -> NumericTransportAudit:
    if (
        set(row) != _NUMERIC_FIELDS
        or row.get("target_center") != center
        or row.get("authorization_gate") is not False
    ):
        fail("numeric transport row schema")
    references = _string_sequence(row.get("reference_centers"), "reference centers")
    expected_references = tuple(sorted(value for value in CENTERS if value != center))
    if references != expected_references:
        fail("numeric transport reference centers")
    raw_dimensions = row.get("dimensions")
    if (
        not isinstance(raw_dimensions, list)
        or len(raw_dimensions) != FINGERPRINT_FEATURE_COUNT
    ):
        fail("numeric transport dimension count")

    dimensions: list[NumericDimensionAudit] = []
    distances: list[float] = []
    zero_scale: list[str] = []
    novelty: list[str] = []
    for index, raw in enumerate(raw_dimensions):
        dimension, distance = _reconstruct_dimension(raw, index=index)
        dimensions.append(dimension)
        if distance is None:
            zero_scale.append(dimension.feature_name)
            if dimension.zero_scale_novelty:
                novelty.append(dimension.feature_name)
        else:
            distances.append(distance)

    expected_l2 = float(np.linalg.norm(np.asarray(distances, dtype=np.float64)))
    expected_maximum = max((abs(value) for value in distances), default=0.0)
    if (
        _strict_integer(
            row.get("active_continuous_dimension_count"), "active dimension count"
        )
        != len(distances)
        or _string_sequence(row.get("zero_scale_dimensions"), "zero-scale dimensions")
        != tuple(zero_scale)
        or _string_sequence(
            row.get("zero_scale_novelty_dimensions"), "novelty dimensions"
        )
        != tuple(novelty)
        or _strict_integer(
            row.get("sparse_pattern_mismatch_count"), "sparse mismatch count"
        )
        != 0
        or _finite_number(row.get("l2_distance"), "transport L2 distance")
        != expected_l2
        or _finite_number(
            row.get("maximum_absolute_distance"), "transport maximum distance"
        )
        != expected_maximum
    ):
        fail("numeric transport aggregate semantics")
    expected = NumericTransportAudit(
        center,
        references,
        tuple(dimensions),
        len(distances),
        tuple(zero_scale),
        tuple(novelty),
        0,
        expected_l2,
        expected_maximum,
    )
    require_sha256(row.get("audit_hash"), "persisted numeric transport hash")
    if dict(row) != expected.to_payload():
        fail("numeric transport hash/payload")
    return expected


def _reconstruct_dimension(
    raw: object, *, index: int
) -> tuple[NumericDimensionAudit, float | None]:
    if not isinstance(raw, Mapping) or set(raw) != _DIMENSION_FIELDS:
        fail("numeric transport dimension schema")
    name = f"physical_fingerprint_dimension_{index}"
    if (
        raw.get("feature_name") != name
        or raw.get("feature_kind") != "continuous"
        or raw.get("sparse_reference_rate") is not None
    ):
        fail("numeric transport dimension identity")
    target = _finite_number(raw.get("target_value"), "transport target value")
    median = _finite_number(raw.get("reference_median"), "transport reference median")
    mad = _finite_number(raw.get("reference_mad"), "transport reference MAD")
    if mad < 0.0:
        fail("numeric transport reference MAD")
    active = _strict_bool(raw.get("active"), "transport active flag")
    novelty = _strict_bool(raw.get("zero_scale_novelty"), "transport novelty flag")
    if mad <= TRANSPORT_SCALE_FLOOR:
        expected_novelty = not math.isclose(
            target, median, rel_tol=0.0, abs_tol=TRANSPORT_SCALE_FLOOR
        )
        if (
            active
            or raw.get("standardized_distance") is not None
            or novelty != expected_novelty
        ):
            fail("numeric zero-scale semantics")
        distance = None
    else:
        distance = _finite_number(
            raw.get("standardized_distance"), "transport standardized distance"
        )
        expected_distance = (target - median) / (TRANSPORT_MAD_SCALE * mad)
        if not active or novelty or distance != expected_distance:
            fail("numeric standardized-distance semantics")
    dimension = NumericDimensionAudit(
        name, "continuous", target, median, mad, active, distance, novelty, None
    )
    if dict(raw) != dimension.to_payload():
        fail("numeric transport dimension payload")
    return dimension, distance


def _strict_bool(value: object, role: str) -> bool:
    if type(value) is not bool:
        fail(role)
    return value


def _strict_integer(value: object, role: str) -> int:
    if type(value) is not int or value < 0:
        fail(role)
    return value


def _finite_number(value: object, role: str) -> float:
    if type(value) not in (int, float):
        fail(role)
    result = float(value)
    if not math.isfinite(result):
        fail(role)
    return result


def _string_sequence(value: object, role: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        fail(role)
    return tuple(value)


__all__ = ("validate_numeric_transport_rows",)
