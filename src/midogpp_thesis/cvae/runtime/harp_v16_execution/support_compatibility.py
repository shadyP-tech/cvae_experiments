"""Case-local, label-free compatibility features for HARP v16.

The resident expert workers score every Train-H and Test-H case while the
corresponding frozen CVAE is on its assigned GPU.  This module performs the
small deterministic CPU reduction that turns those raw energies into the
four candidate-relative features consumed by the H-local router.  It never
opens outcomes and it calibrates every expert only against its own Train-e
energy distribution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash, require_sha256
from ..artifact_io import read_json, sha256_file
from .contracts import ArtifactValue, LabelFreeOuterMenu
from .gpu_surface import COMPATIBILITY_MEMBER


FeatureKey = tuple[str, str, str]
FEATURE_COLUMNS = (
    "mean_own_calibrated_energy_z",
    "training_seed_dispersion_z",
    "reciprocal_candidate_rank",
    "candidate_rank_margin",
)


def _surface_identity(
    by_outer: Mapping[
        str, Mapping[FeatureKey, tuple[float, float, float, float]]
    ],
    *,
    raw_compatibility_hash: str,
    raw_file_sha256: str,
) -> str:
    body = {
        "schema_version": "midogpp_harp_v16_case_local_compatibility_surface_v1",
        "raw_compatibility_hash": raw_compatibility_hash,
        "raw_file_sha256": raw_file_sha256,
        "outer_hashes": {
            outer: canonical_hash(
                tuple((key, value) for key, value in sorted(rows.items()))
            )
            for outer, rows in by_outer.items()
        },
        "own_source_train_calibration_only": True,
        "labels_consumed": False,
        "evaluation_labels_consumed": False,
    }
    return canonical_hash(body)


def _normalize_surface(
    value: Mapping[str, Mapping[FeatureKey, tuple[float, float, float, float]]],
) -> Mapping[str, Mapping[FeatureKey, tuple[float, float, float, float]]]:
    if not isinstance(value, Mapping) or tuple(value) != tuple(CENTERS):
        raise ProtocolError("HARP v16 compatibility surface lacks a target center.")
    output: dict[
        str, Mapping[FeatureKey, tuple[float, float, float, float]]
    ] = {}
    for outer in CENTERS:
        raw_rows = value.get(outer)
        if not isinstance(raw_rows, Mapping) or not raw_rows:
            raise ProtocolError("HARP v16 compatibility target surface is empty.")
        candidates = tuple(center for center in CENTERS if center != outer)
        rows: dict[FeatureKey, tuple[float, float, float, float]] = {}
        coverage: dict[tuple[str, str], set[str]] = {}
        for raw_key, raw_features in raw_rows.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 3:
                raise ProtocolError("HARP v16 compatibility feature key is malformed.")
            role, case_id, source = raw_key
            if (
                role not in {"support", "target"}
                or type(case_id) is not str
                or not case_id
                or type(source) is not str
                or source not in candidates
                or not isinstance(raw_features, (tuple, list))
                or len(raw_features) != len(FEATURE_COLUMNS)
            ):
                raise ProtocolError("HARP v16 compatibility feature row is malformed.")
            features = tuple(float(member) for member in raw_features)
            if not all(math.isfinite(member) for member in features):
                raise ProtocolError("HARP v16 compatibility feature is nonfinite.")
            key = (role, case_id, source)
            if key in rows:
                raise ProtocolError("HARP v16 compatibility feature row is duplicated.")
            rows[key] = features  # type: ignore[assignment]
            coverage.setdefault((role, case_id), set()).add(source)
        if (
            {role for role, _case in coverage} != {"support", "target"}
            or any(sources != set(candidates) for sources in coverage.values())
        ):
            raise ProtocolError("HARP v16 compatibility candidate coverage drifted.")
        output[outer] = MappingProxyType(
            {key: rows[key] for key in sorted(rows)}
        )
    return MappingProxyType(output)


def _artifact_projection(
    surface: "CaseLocalCompatibilitySurface",
) -> tuple[list[dict[str, object]], np.ndarray, dict[str, object]]:
    metadata: list[dict[str, object]] = []
    values: list[tuple[float, float, float, float]] = []
    for outer, rows in surface.by_outer.items():
        for (role, case_id, source), row in rows.items():
            metadata.append(
                {
                    "outer_target_id": outer,
                    "surface_role": role,
                    "case_id": case_id,
                    "candidate_source_id": source,
                }
            )
            values.append(row)
    array = np.asarray(values, dtype=np.float64).reshape((-1, len(FEATURE_COLUMNS)))
    body = {
        "schema_version": "midogpp_harp_v16_case_local_compatibility_features_v1",
        "raw_compatibility_hash": surface.raw_compatibility_hash,
        "raw_file_sha256": surface.raw_file_sha256,
        "rows": metadata,
        "feature_columns": list(FEATURE_COLUMNS),
        "own_calibration_role": "target_train_support",
        "case_local_support_and_target_features": True,
        "exact_nelbo": False,
        "labels_consumed": False,
        "evaluation_labels_consumed": False,
        "compatibility_feature_hash": surface.surface_hash,
    }
    return metadata, array, body


def _robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ProtocolError("HARP v16 own-source compatibility calibration is malformed.")
    location = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - location)))
    floor = math.sqrt(np.finfo(np.float64).eps)
    if not math.isfinite(scale) or scale <= floor:
        scale = float(np.std(array, dtype=np.float64))
    if not math.isfinite(scale) or scale <= floor:
        scale = max(1e-6, abs(location) * 1e-12)
    return location, scale


def _case_inventory(
    menus: Sequence[LabelFreeOuterMenu], *, physical_role: str
) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for menu in menus:
        blocks = tuple(
            block for block in menu.blocks if block.surface_role == physical_role
        )
        if not blocks:
            raise ProtocolError("HARP v16 compatibility lacks a physical role menu.")
        cases = tuple(dict.fromkeys(blocks[0].case_ids))
        if not cases or any(tuple(dict.fromkeys(row.case_ids)) != cases for row in blocks[1:]):
            raise ProtocolError("HARP v16 compatibility physical case inventory drifted.")
        output[menu.outer_target_id] = cases
    if tuple(output) != tuple(CENTERS):
        raise ProtocolError("HARP v16 compatibility outer-target order drifted.")
    return output


def _context_index(
    raw: Mapping[str, object],
) -> tuple[
    dict[tuple[str, int, str, str], Mapping[str, object]],
    str,
    str,
]:
    body = {key: value for key, value in raw.items() if key != "compatibility_hash"}
    binding = raw.get("support_binding")
    replicas = raw.get("replicas")
    if (
        raw.get("schema_version")
        != "midogpp_harp_v16_role_qualified_compatibility_surface_v2"
        or raw.get("compatibility_hash") != canonical_hash(body)
        or tuple(raw.get("training_seeds", ())) != tuple(TRAINING_SEEDS)
        or raw.get("all_replicas_used_without_selection") is not True
        or raw.get("computed_while_expert_resident") is not True
        or raw.get("exact_nelbo") is not False
        or raw.get("labels_consumed") is not False
        or raw.get("evaluation_labels_consumed") is not False
        or not isinstance(binding, Mapping)
        or not isinstance(replicas, list)
    ):
        raise ProtocolError("HARP v16 resident compatibility surface drifted.")
    support_role = str(binding.get("support_role", ""))
    target_role = str(binding.get("target_role", ""))
    binding_body = {
        key: value for key, value in binding.items() if key != "support_binding_hash"
    }
    binding_hash = binding.get("support_binding_hash")
    if (
        support_role != "target_train_support"
        or target_role != "target_test_evaluation"
        or binding.get("schema_version")
        != "midogpp_harp_v16_role_qualified_label_free_binding_v2"
        or type(binding_hash) is not str
        or binding_hash != canonical_hash(binding_body)
        or raw.get("support_binding_hash") != binding_hash
        or binding.get("labels_present") is not False
        or binding.get("evaluation_labels_included") is not False
    ):
        raise ProtocolError("HARP v16 compatibility role boundary drifted.")
    contexts: dict[tuple[str, int, str, str], Mapping[str, object]] = {}
    observed_replicas: list[tuple[str, int]] = []
    for replica in replicas:
        if not isinstance(replica, Mapping):
            raise ProtocolError("HARP v16 compatibility replica is malformed.")
        source = str(replica.get("source_center", ""))
        seed = replica.get("training_seed")
        rows = replica.get("contexts")
        if (
            source not in CENTERS
            or type(seed) is not int
            or seed not in TRAINING_SEEDS
            or (source, seed) in observed_replicas
            or not isinstance(rows, list)
        ):
            raise ProtocolError("HARP v16 compatibility replica grid drifted.")
        observed_replicas.append((source, seed))
        observed_contexts: list[tuple[str, str]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProtocolError("HARP v16 compatibility context is malformed.")
            role = str(row.get("role", ""))
            query = str(row.get("query_center", ""))
            case_order = row.get("case_order")
            energies = row.get("per_case_energy_float32")
            if (
                role not in {support_role, target_role}
                or query not in CENTERS
                or (role, query) in observed_contexts
                or not isinstance(case_order, list)
                or not isinstance(energies, list)
                or not case_order
                or any(type(value) is not str or not value for value in case_order)
                or len(case_order) != len(energies)
                or len(set(case_order)) != len(case_order)
                or row.get("case_count") != len(case_order)
                or row.get("exact_nelbo") is not False
                or row.get("labels_consumed") is not False
            ):
                raise ProtocolError("HARP v16 compatibility context geometry drifted.")
            values = np.asarray(energies, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ProtocolError("HARP v16 compatibility energy is nonfinite.")
            contexts[(source, seed, role, query)] = row
            observed_contexts.append((role, query))
        expected = tuple(
            (role, center)
            for role in (support_role, target_role)
            for center in CENTERS
        )
        if tuple(observed_contexts) != expected:
            raise ProtocolError("HARP v16 compatibility context inventory drifted.")
    if tuple(observed_replicas) != tuple(
        (source, seed) for source in CENTERS for seed in TRAINING_SEEDS
    ):
        raise ProtocolError("HARP v16 compatibility replica coverage is incomplete.")
    return contexts, support_role, target_role


@dataclass(frozen=True, slots=True)
class CaseLocalCompatibilitySurface:
    by_outer: Mapping[str, Mapping[FeatureKey, tuple[float, float, float, float]]]
    raw_compatibility_hash: str
    raw_file_sha256: str
    surface_hash: str

    def __post_init__(self) -> None:
        normalized = _normalize_surface(self.by_outer)
        raw_hash = require_sha256(
            self.raw_compatibility_hash,
            name="HARP v16 raw compatibility semantic hash",
        )
        raw_sha = require_sha256(
            self.raw_file_sha256,
            name="HARP v16 raw compatibility file SHA-256",
        )
        observed = require_sha256(
            self.surface_hash, name="HARP v16 compatibility feature hash"
        )
        expected = _surface_identity(
            normalized,
            raw_compatibility_hash=raw_hash,
            raw_file_sha256=raw_sha,
        )
        if observed != expected:
            raise ProtocolError("HARP v16 compatibility feature identity drifted.")
        object.__setattr__(self, "by_outer", normalized)
        object.__setattr__(self, "raw_compatibility_hash", raw_hash)
        object.__setattr__(self, "raw_file_sha256", raw_sha)
        object.__setattr__(self, "surface_hash", observed)

    def for_outer(
        self, outer_target_id: str
    ) -> Mapping[FeatureKey, tuple[float, float, float, float]]:
        try:
            return self.by_outer[str(outer_target_id)]
        except KeyError as exc:
            raise ProtocolError("HARP v16 compatibility target center is absent.") from exc

    def artifact(self) -> ArtifactValue:
        _metadata, values, body = _artifact_projection(self)
        artifact = ArtifactValue(
            state=self,
            manifest={**body, "artifact_hash": canonical_hash(body)},
            arrays={"compatibility_values": values},
        )
        validate_case_local_compatibility_artifact(artifact)
        return artifact


def validate_case_local_compatibility_artifact(value: ArtifactValue) -> str:
    """Validate both the semantic feature surface and its durable projection.

    This accepts an in-memory value or an opaque-store reconstruction, whose
    ``state`` is deliberately unavailable.
    """

    if not isinstance(value, ArtifactValue):
        raise ProtocolError("HARP v16 compatibility artifact is untyped.")
    manifest = dict(value.manifest)
    artifact_hash = require_sha256(
        manifest.pop("artifact_hash", None),
        name="HARP v16 compatibility artifact hash",
    )
    if canonical_hash(manifest) != artifact_hash:
        raise ProtocolError("HARP v16 compatibility artifact identity drifted.")
    raw_hash = require_sha256(
        manifest.get("raw_compatibility_hash"),
        name="HARP v16 raw compatibility semantic hash",
    )
    raw_sha = require_sha256(
        manifest.get("raw_file_sha256"),
        name="HARP v16 raw compatibility file SHA-256",
    )
    feature_hash = require_sha256(
        manifest.get("compatibility_feature_hash"),
        name="HARP v16 compatibility feature hash",
    )
    rows = manifest.get("rows")
    values = value.arrays.get("compatibility_values")
    if (
        manifest.get("schema_version")
        != "midogpp_harp_v16_case_local_compatibility_features_v1"
        or manifest.get("feature_columns") != list(FEATURE_COLUMNS)
        or manifest.get("own_calibration_role") != "target_train_support"
        or manifest.get("case_local_support_and_target_features") is not True
        or manifest.get("exact_nelbo") is not False
        or manifest.get("labels_consumed") is not False
        or manifest.get("evaluation_labels_consumed") is not False
        or not isinstance(rows, list)
        or not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("float64")
        or values.shape != (len(rows), len(FEATURE_COLUMNS))
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("HARP v16 compatibility artifact geometry drifted.")
    by_outer: dict[
        str, dict[FeatureKey, tuple[float, float, float, float]]
    ] = {center: {} for center in CENTERS}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "outer_target_id",
            "surface_role",
            "case_id",
            "candidate_source_id",
        }:
            raise ProtocolError("HARP v16 compatibility artifact row is malformed.")
        outer = row.get("outer_target_id")
        role = row.get("surface_role")
        case_id = row.get("case_id")
        source = row.get("candidate_source_id")
        if type(outer) is not str or outer not in by_outer:
            raise ProtocolError("HARP v16 compatibility artifact target is malformed.")
        key = (role, case_id, source)
        if (
            type(role) is not str
            or type(case_id) is not str
            or type(source) is not str
            or key in by_outer[outer]
        ):
            raise ProtocolError("HARP v16 compatibility artifact key is malformed.")
        by_outer[outer][key] = tuple(float(member) for member in values[index])  # type: ignore[assignment]
    surface = CaseLocalCompatibilitySurface(
        by_outer=by_outer,
        raw_compatibility_hash=raw_hash,
        raw_file_sha256=raw_sha,
        surface_hash=feature_hash,
    )
    expected_rows, expected_values, expected_body = _artifact_projection(surface)
    if (
        rows != expected_rows
        or not np.array_equal(values, expected_values)
        or manifest != expected_body
    ):
        raise ProtocolError("HARP v16 compatibility artifact projection drifted.")
    return feature_hash


def build_case_local_compatibility_surface(
    menus: Sequence[LabelFreeOuterMenu], *, scratch_root: Path
) -> CaseLocalCompatibilitySurface:
    """Reduce the resident energy grid without opening any outcome labels."""

    menu_rows = tuple(menus)
    if tuple(menu.outer_target_id for menu in menu_rows) != tuple(CENTERS):
        raise ProtocolError("HARP v16 compatibility menu universe drifted.")
    support_cases = _case_inventory(menu_rows, physical_role="support")
    target_cases = _case_inventory(menu_rows, physical_role="target")
    path = Path(scratch_root).resolve() / "source_streams" / COMPATIBILITY_MEMBER
    if not path.is_file() or path.is_symlink():
        raise ProtocolError("HARP v16 resident compatibility file is absent or unsafe.")
    raw = read_json(path)
    contexts, support_role, target_role = _context_index(raw)

    own_calibration: dict[tuple[str, int], tuple[float, float]] = {}
    for source in CENTERS:
        for seed in TRAINING_SEEDS:
            own = contexts[(source, seed, support_role, source)]
            own_calibration[(source, seed)] = _robust_location_scale(
                tuple(float(value) for value in own["per_case_energy_float32"])
            )

    by_outer: dict[
        str, dict[FeatureKey, tuple[float, float, float, float]]
    ] = {}
    for outer in CENTERS:
        candidates = tuple(center for center in CENTERS if center != outer)
        scoped: dict[FeatureKey, tuple[float, float, float, float]] = {}
        for physical_role, raw_role, expected_cases in (
            ("support", support_role, support_cases[outer]),
            ("target", target_role, target_cases[outer]),
        ):
            per_candidate: dict[str, dict[str, tuple[float, float]]] = {}
            for source in candidates:
                case_seed_z: dict[str, list[float]] = {
                    case_id: [] for case_id in expected_cases
                }
                for seed in TRAINING_SEEDS:
                    context = contexts[(source, seed, raw_role, outer)]
                    cases = tuple(str(value) for value in context["case_order"])
                    energies = tuple(
                        float(value) for value in context["per_case_energy_float32"]
                    )
                    if cases != expected_cases:
                        raise ProtocolError(
                            "HARP v16 compatibility cases escaped the physical menu."
                        )
                    location, scale = own_calibration[(source, seed)]
                    for case_id, energy in zip(cases, energies, strict=True):
                        case_seed_z[case_id].append((energy - location) / scale)
                per_candidate[source] = {
                    case_id: (
                        float(np.mean(values, dtype=np.float64)),
                        float(np.std(values, dtype=np.float64)),
                    )
                    for case_id, values in case_seed_z.items()
                }
            for case_id in expected_cases:
                order = tuple(
                    sorted(
                        candidates,
                        key=lambda source: (per_candidate[source][case_id][0], source),
                    )
                )
                best = per_candidate[order[0]][case_id][0]
                runner_up = per_candidate[order[1]][case_id][0]
                rank_by_source = {
                    source: rank for rank, source in enumerate(order, start=1)
                }
                for source in candidates:
                    mean, std = per_candidate[source][case_id]
                    rank = rank_by_source[source]
                    scoped[(physical_role, case_id, source)] = (
                        mean,
                        std,
                        1.0 / float(rank),
                        runner_up - mean if rank == 1 else best - mean,
                    )
        by_outer[outer] = scoped
    raw_hash = str(raw["compatibility_hash"])
    raw_file_sha = sha256_file(path)
    return CaseLocalCompatibilitySurface(
        by_outer=by_outer,
        raw_compatibility_hash=raw_hash,
        raw_file_sha256=raw_file_sha,
        surface_hash=_surface_identity(
            by_outer,
            raw_compatibility_hash=raw_hash,
            raw_file_sha256=raw_file_sha,
        ),
    )


__all__ = (
    "CaseLocalCompatibilitySurface",
    "FEATURE_COLUMNS",
    "FeatureKey",
    "build_case_local_compatibility_surface",
    "validate_case_local_compatibility_artifact",
)
