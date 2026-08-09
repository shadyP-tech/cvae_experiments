"""Label-free adapters and compact fixed-family feature designs."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXACT_FAMILY_IDS,
    PERMUTATION_CONTROL_FAMILY_IDS,
    SMOOTH_DESCRIPTIVE_FAMILY_IDS,
    candidate_sources,
    expected_row_keys,
)
from .model_contracts import FamilyDesign, family_spec
from .row_contracts import (
    FixedBankDataset,
    FixedBankFeatureRow,
    FixedBankResponseRow,
)


def build_fixed_bank_dataset(
    feature_rows: Sequence[FixedBankFeatureRow],
    response_rows: Sequence[FixedBankResponseRow],
) -> FixedBankDataset:
    """Canonicalize and validate a complete fixed-bank surface."""

    features = _ordered(feature_rows, FixedBankFeatureRow, "feature")
    responses = _ordered(response_rows, FixedBankResponseRow, "response")
    return FixedBankDataset(
        feature_rows=features,  # type: ignore[arg-type]
        response_rows=responses,  # type: ignore[arg-type]
    )


def build_exact_family_designs(
    dataset: FixedBankDataset,
    family_ids: Sequence[str] = EXACT_FAMILY_IDS,
) -> Mapping[str, FamilyDesign]:
    return _build_designs(dataset, family_ids, allowed=EXACT_FAMILY_IDS)


def build_smooth_descriptive_designs(
    dataset: FixedBankDataset,
    family_ids: Sequence[str] = SMOOTH_DESCRIPTIVE_FAMILY_IDS,
) -> Mapping[str, FamilyDesign]:
    return _build_designs(
        dataset, family_ids, allowed=SMOOTH_DESCRIPTIVE_FAMILY_IDS
    )


def _build_designs(
    dataset: FixedBankDataset,
    family_ids: Sequence[str],
    *,
    allowed: Sequence[str],
) -> Mapping[str, FamilyDesign]:
    if not isinstance(dataset, FixedBankDataset):
        raise ProtocolError("Family designs require a typed fixed-bank dataset.")
    selected = tuple(family_ids)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(value not in allowed for value in selected)
    ):
        raise ProtocolError("Fixed-bank family selection drifted from predeclaration.")
    rows = {row.row_key: row for row in dataset.feature_rows}
    transforms = _within_query_rich_z(dataset.feature_rows)
    output: dict[str, FamilyDesign] = {}
    for family_id in selected:
        spec = family_spec(family_id)
        values: list[tuple[float, ...]] = []
        source_hashes: list[str] = []
        donor_hashes: list[str] = []
        permuted = family_id in PERMUTATION_CONTROL_FAMILY_IDS
        for key in dataset.row_keys:
            source = rows[key]
            donor = rows[_donor_key(key)] if permuted else source
            vector = tuple(
                _predictor_value(name, donor, transforms) for name in spec.predictor_names
            )
            values.append(vector)
            source_hashes.append(source.feature_row_hash)
            donor_hashes.append(donor.feature_row_hash)
        matrix = np.asarray(values, dtype=np.float64).reshape(
            len(dataset.row_keys), len(spec.predictor_names)
        )
        output[family_id] = FamilyDesign(
            spec=spec,
            row_keys=dataset.row_keys,
            values=matrix,
            source_feature_row_hashes=tuple(source_hashes),
            donor_feature_row_hashes=tuple(donor_hashes),
            feature_surface_hash=dataset.feature_surface_hash,
        )
    return MappingProxyType(output)


def blocked_permutation_donor_keys(
    dataset: FixedBankDataset,
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(dataset, FixedBankDataset):
        raise ProtocolError("Permutation donors require a typed dataset.")
    return tuple(_donor_key(key) for key in dataset.row_keys)


def _donor_key(key: tuple[str, str, str]) -> tuple[str, str, str]:
    outer, query, source = key
    sources = candidate_sources(outer, query)
    index = sources.index(source)
    donor = sources[(index + 1) % len(sources)]
    return outer, query, donor


def _within_query_rich_z(
    rows: Sequence[FixedBankFeatureRow],
) -> Mapping[tuple[tuple[str, str, str], str], float]:
    by_key = {row.row_key: row for row in rows}
    output: dict[tuple[tuple[str, str, str], str], float] = {}
    names = (
        ("case_balanced_reconstruction", "case_balanced_reconstruction_z"),
        ("case_balanced_kl", "case_balanced_kl_z"),
        ("case_balanced_log_mmd", "case_balanced_log_mmd_z"),
    )
    for outer in CENTERS:
        for query in (value for value in CENTERS if value != outer):
            keys = tuple(
                (outer, query, source) for source in candidate_sources(outer, query)
            )
            for raw_name, transformed_name in names:
                raw = np.asarray(
                    [getattr(by_key[key], raw_name) for key in keys],
                    dtype=np.float64,
                )
                centered = raw - float(np.mean(raw, dtype=np.float64))
                rms = float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))
                transformed = (
                    np.zeros_like(centered)
                    if rms <= float(np.sqrt(np.finfo(np.float64).eps))
                    else centered / rms
                )
                for key, value in zip(keys, transformed, strict=True):
                    output[(key, transformed_name)] = float(value)
    return MappingProxyType(output)


def _predictor_value(
    name: str,
    row: FixedBankFeatureRow,
    transforms: Mapping[tuple[tuple[str, str, str], str], float],
) -> float:
    if name.endswith("_z"):
        return transforms[(row.row_key, name)]
    try:
        return float(getattr(row, name))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Unknown fixed-bank predictor {name}.") from exc


def _ordered(rows: Sequence[object], expected_type: type, role: str) -> tuple[object, ...]:
    if any(not isinstance(row, expected_type) for row in rows):
        raise ProtocolError(f"Fixed-bank {role} surface contains an untyped row.")
    by_key = {row.row_key: row for row in rows}  # type: ignore[attr-defined]
    keys = expected_row_keys()
    if len(by_key) != len(rows) or set(by_key) != set(keys):
        raise ProtocolError(f"Fixed-bank {role} surface coverage drifted.")
    return tuple(by_key[key] for key in keys)


__all__ = (
    "blocked_permutation_donor_keys",
    "build_exact_family_designs",
    "build_fixed_bank_dataset",
    "build_smooth_descriptive_designs",
)
