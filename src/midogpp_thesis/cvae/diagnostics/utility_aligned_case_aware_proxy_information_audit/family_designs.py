"""Predeclared compact case-aware and control family designs."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256
from .contracts import (
    CASE_AWARE_HYBRID_COMPACT,
    CASE_BALANCED_RICH_COMPACT,
    CASE_BALANCED_SHIFT_COMPACT,
    CENTERS,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
    CYCLIC_PERMUTATION_SHIFT,
    EQUAL_UNION_NULL,
    FAMILY_IDS,
    METADATA_ONLY_CONTROL,
    POOLED_ROW_WEIGHTED_SHIFT_CONTROL,
    CaseAwareFeatureSurface,
    ProxyFamilyDesign,
    ProxyFamilySpec,
    candidate_sources,
)


FAMILY_PREDICTORS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        EQUAL_UNION_NULL: (),
        METADATA_ONLY_CONTROL: ("metadata_similarity",),
        POOLED_ROW_WEIGHTED_SHIFT_CONTROL: (
            "pooled_row_weighted_abs_shift",
        ),
        CASE_BALANCED_SHIFT_COMPACT: (
            "equal_case_abs_shift",
            "case_abs_shift_sd",
            "equal_case_signed_margin",
        ),
        CASE_BALANCED_RICH_COMPACT: (
            "case_balanced_reconstruction_z",
            "case_balanced_kl_z",
            "case_balanced_log_mmd_z",
        ),
        CASE_AWARE_HYBRID_COMPACT: (
            "metadata_similarity",
            "case_balanced_log_mmd_z",
            "equal_case_abs_shift",
        ),
        CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL: (
            "cyclic_equal_case_signed_margin",
            "cyclic_case_balanced_flip_rate",
            "cyclic_case_balanced_entropy_change",
        ),
    }
)

# Lowercase alias is convenient for declarative config adapters.
family_predictors = FAMILY_PREDICTORS

PROXY_FAMILY_SPECS: Mapping[str, ProxyFamilySpec] = MappingProxyType(
    {
        family_id: ProxyFamilySpec(
            family_id=family_id,
            predictor_names=FAMILY_PREDICTORS[family_id],
            family_role=(
                "screening_candidate"
                if family_id
                in {
                    CASE_BALANCED_SHIFT_COMPACT,
                    CASE_BALANCED_RICH_COMPACT,
                    CASE_AWARE_HYBRID_COMPACT,
                }
                else "control"
            ),
            cyclic_shift=(
                CYCLIC_PERMUTATION_SHIFT
                if family_id == CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL
                else None
            ),
        )
        for family_id in FAMILY_IDS
    }
)


def build_family_designs(
    surface: CaseAwareFeatureSurface,
) -> Mapping[str, ProxyFamilyDesign]:
    """Materialize all fixed designs with at most three predictors each."""

    if not isinstance(surface, CaseAwareFeatureSurface):
        raise ProtocolError("Family designs require a typed feature surface.")
    rows = {row.row_key: row for row in surface.rows}
    transforms = _within_query_transforms(surface)
    designs: dict[str, ProxyFamilyDesign] = {}
    for family_id in FAMILY_IDS:
        spec = PROXY_FAMILY_SPECS[family_id]
        values: list[tuple[float, ...]] = []
        provenance: list[str] = []
        for key in surface.row_keys:
            row = rows[key]
            source_row = row
            if family_id == CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL:
                sources = candidate_sources(row.outer_target_id, row.query_id)
                source_index = sources.index(row.candidate_source)
                donor = sources[
                    (source_index + CYCLIC_PERMUTATION_SHIFT) % len(sources)
                ]
                source_row = rows[(row.outer_target_id, row.query_id, donor)]
                vector = (
                    source_row.equal_case_signed_margin,
                    source_row.case_balanced_flip_rate,
                    source_row.case_balanced_entropy_change,
                )
            else:
                vector = tuple(
                    _predictor_value(name, row, transforms)
                    for name in spec.predictor_names
                )
            values.append(vector)
            provenance.append(source_row.feature_row_hash)
        matrix = np.asarray(values, dtype=np.float64).reshape(
            len(surface.rows), spec.predictor_count
        )
        unhashed = {
            "schema_version": "midogpp_stage90_case_aware_family_design_v1",
            "family_spec": spec.to_payload(),
            "feature_surface_hash": surface.surface_hash,
            "row_keys": [list(key) for key in surface.row_keys],
            "source_row_hashes": provenance,
            "values_sha256": array_sha256(matrix),
            "case_balanced_candidate_list_z_transforms": True,
            "label_free_candidate_list_transforms_may_be_transductive": True,
            "utility_or_evaluation_labels_used": False,
        }
        designs[family_id] = ProxyFamilyDesign(
            spec=spec,
            row_keys=surface.row_keys,
            values=matrix,
            source_row_hashes=tuple(provenance),
            design_hash=canonical_sha256(unhashed),
        )
    if tuple(designs) != FAMILY_IDS or any(
        design.spec.predictor_count > 3 for design in designs.values()
    ):
        raise ProtocolError("Predeclared family coverage/capacity drifted.")
    return MappingProxyType(designs)


build_proxy_family_designs = build_family_designs


def _within_query_transforms(
    surface: CaseAwareFeatureSurface,
) -> Mapping[tuple[tuple[str, str, str], str], float]:
    output: dict[tuple[tuple[str, str, str], str], float] = {}
    by_key = {row.row_key: row for row in surface.rows}
    raw_to_z = (
        ("case_balanced_reconstruction", "case_balanced_reconstruction_z"),
        ("case_balanced_kl", "case_balanced_kl_z"),
        ("case_balanced_log_mmd", "case_balanced_log_mmd_z"),
    )
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            keys = tuple(
                (outer, query, source) for source in candidate_sources(outer, query)
            )
            for raw_name, transformed_name in raw_to_z:
                transformed = _within_group_z(
                    [float(getattr(by_key[key], raw_name)) for key in keys]
                )
                for key, value in zip(keys, transformed, strict=True):
                    output[(key, transformed_name)] = float(value)
    return MappingProxyType(output)


def _predictor_value(
    name: str,
    row: object,
    transforms: Mapping[tuple[tuple[str, str, str], str], float],
) -> float:
    if name.endswith("_z"):
        return transforms[(row.row_key, name)]  # type: ignore[attr-defined]
    try:
        return float(getattr(row, name))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Unknown/non-numeric family predictor {name}.") from exc


def _within_group_z(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("Within-query z transform requires finite values.")
    centered = array - float(np.mean(array, dtype=np.float64))
    rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
    if rms <= float(np.sqrt(np.finfo(np.float64).eps)):
        return np.zeros_like(centered)
    return centered / rms


__all__ = (
    "FAMILY_PREDICTORS",
    "PROXY_FAMILY_SPECS",
    "build_family_designs",
    "build_proxy_family_designs",
    "family_predictors",
)
