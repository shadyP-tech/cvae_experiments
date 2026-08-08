"""Typed label-free target feature and whole-case bootstrap construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .features import build_distributional_feature_surface
from .row_contracts import (
    MIN_TARGET_SUPPORT_CASES,
    TARGET_ROLE,
    CaseBootstrapPlan,
    build_case_bootstrap_plan,
)
from .surface_contracts import CandidateFeatureRow, FeatureSurface


MINIMUM_SUPPORT_CASE_COUNT = MIN_TARGET_SUPPORT_CASES


def target_sources(target: object) -> tuple[str, ...]:
    rendered = str(target)
    if rendered not in CENTERS:
        raise ProtocolError("Target center is outside the frozen universe.")
    return tuple(center for center in CENTERS if center != rendered)


@dataclass(frozen=True)
class TargetCandidateComponents:
    """Unlabeled per-row components for all exact replicas of one source."""

    candidate_source: str
    reconstruction_by_training_seed: Mapping[int, Mapping[int, np.ndarray]]
    normalized_ps_kl_by_training_seed: Mapping[int, Mapping[int, np.ndarray]]
    support_case_mean_embeddings: Mapping[str, np.ndarray]
    generated_mean_by_seed_pair: Mapping[tuple[int, int], np.ndarray]
    metadata_similarity: float

    def __post_init__(self) -> None:
        reconstruction = _component_grid(
            self.reconstruction_by_training_seed, "reconstruction"
        )
        kl = _component_grid(
            self.normalized_ps_kl_by_training_seed, "normalized PS KL"
        )
        if any(
            reconstruction[seed][label].shape != kl[seed][label].shape
            for seed in TRAINING_SEEDS
            for label in (0, 1)
        ):
            raise ProtocolError("Target reconstruction/KL row geometry drifted.")
        support_means = {
            str(case_id): np.asarray(value, dtype=np.float64)
            for case_id, value in self.support_case_mean_embeddings.items()
        }
        if not support_means or any(
            not case_id
            or value.shape != (COMMON_OUTPUT_DIM,)
            or not np.isfinite(value).all()
            for case_id, value in support_means.items()
        ):
            raise ProtocolError(
                "Target support case means must stay in the common feature frame."
            )
        generated_means = {
            (int(key[0]), int(key[1])): np.asarray(value, dtype=np.float64)
            for key, value in self.generated_mean_by_seed_pair.items()
        }
        if set(generated_means) != set(product(TRAINING_SEEDS, GENERATION_SEEDS)) or any(
            value.shape != (COMMON_OUTPUT_DIM,) or not np.isfinite(value).all()
            for value in generated_means.values()
        ):
            raise ProtocolError("Target generated means require all exact seed pairs.")
        for value in (*support_means.values(), *generated_means.values()):
            value.setflags(write=False)
        metadata = float(self.metadata_similarity)
        if not np.isfinite(metadata) or not 0.0 <= metadata <= 1.0:
            raise ProtocolError("Target metadata similarity must lie in [0,1].")
        object.__setattr__(
            self, "reconstruction_by_training_seed", MappingProxyType(reconstruction)
        )
        object.__setattr__(
            self, "normalized_ps_kl_by_training_seed", MappingProxyType(kl)
        )
        object.__setattr__(
            self, "support_case_mean_embeddings", MappingProxyType(support_means)
        )
        object.__setattr__(
            self, "generated_mean_by_seed_pair", MappingProxyType(generated_means)
        )
        object.__setattr__(self, "metadata_similarity", metadata)


@dataclass(frozen=True)
class TargetFeatureProduction:
    target_id: str
    bootstrap_plan: CaseBootstrapPlan
    point_rows: tuple[CandidateFeatureRow, ...]
    point_surface: FeatureSurface
    bootstrap_surfaces: tuple[FeatureSurface, ...]

    def __post_init__(self) -> None:
        if (
            self.bootstrap_plan.target_id != self.target_id
            or self.point_surface.outer_target_id != self.target_id
            or self.point_surface.role != TARGET_ROLE
            or len(self.point_rows) != 72
            or self.point_surface.case_bootstrap_replicate is not None
            or len(self.bootstrap_surfaces) != self.bootstrap_plan.replicate_count
            or any(
                surface.case_bootstrap_replicate != replicate
                for surface, replicate in zip(
                    self.bootstrap_surfaces,
                    self.bootstrap_plan.replicates,
                    strict=True,
                )
            )
        ):
            raise ProtocolError("Target feature production escaped its bootstrap plan.")


def build_target_feature_production(
    *,
    target_id: object,
    case_ids: Sequence[object],
    components_by_source: Mapping[str, TargetCandidateComponents],
    bootstrap_seed: int,
    bootstrap_replicate_count: int = 32,
) -> TargetFeatureProduction:
    """Construct 8x9 TARGET rows and >=32 typed whole-case bootstrap surfaces.

    This API accepts no label, utility, prediction, or evaluation argument.
    Resampling is exclusively over the canonical independent case-ID universe.
    """

    target = str(target_id)
    if target not in CENTERS:
        raise ProtocolError("Target feature center is outside the frozen universe.")
    raw_cases = tuple(str(value) for value in case_ids)
    if not raw_cases or any(not value for value in raw_cases):
        raise ProtocolError("Target support rows require case identities.")
    case_universe = tuple(sorted(set(raw_cases)))
    if len(case_universe) < MINIMUM_SUPPORT_CASE_COUNT:
        raise ProtocolError("Target feature construction requires at least eight cases.")
    expected_sources = target_sources(target)
    components = {str(key): value for key, value in components_by_source.items()}
    if tuple(components) != expected_sources or any(
        not isinstance(value, TargetCandidateComponents)
        or value.candidate_source != source
        for source, value in components.items()
    ):
        raise ProtocolError("Target feature component source coverage drifted.")
    for value in components.values():
        if any(
            len(value.reconstruction_by_training_seed[seed][label]) != len(raw_cases)
            for seed in TRAINING_SEEDS
            for label in (0, 1)
        ):
            raise ProtocolError("Target feature components and case IDs do not align.")
        if set(value.support_case_mean_embeddings) != set(case_universe):
            raise ProtocolError("Target support case-mean coverage drifted.")

    plan = build_case_bootstrap_plan(
        target_id=target,
        support_case_ids=tuple(case_universe),
        bootstrap_seed=bootstrap_seed,
        replicate_count=bootstrap_replicate_count,
    )
    point_rows = _build_rows(
        target=target,
        raw_case_ids=raw_cases,
        selected_case_ids=plan.support_case_ids,
        support_partition_hash=plan.support_partition_hash,
        components=components,
    )
    point_surface = build_distributional_feature_surface(point_rows)
    bootstrap_surfaces: list[FeatureSurface] = []
    for replicate in plan.replicates:
        rows = _build_rows(
            target=target,
            raw_case_ids=raw_cases,
            selected_case_ids=replicate.sampled_case_ids,
            support_partition_hash=replicate.support_partition_hash,
            components=components,
        )
        bootstrap_surfaces.append(
            build_distributional_feature_surface(
                rows, case_bootstrap_replicate=replicate
            )
        )
    return TargetFeatureProduction(
        target_id=target,
        bootstrap_plan=plan,
        point_rows=point_rows,
        point_surface=point_surface,
        bootstrap_surfaces=tuple(bootstrap_surfaces),
    )


def target_feature_production_from_payload(raw: object) -> TargetFeatureProduction:
    """Independently reconstruct a serialized target feature family."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "target_id", "case_bootstrap_plan", "point_rows", "point_surface_hash",
        "bootstrap_surfaces", "target_feature_hash",
    }:
        raise ProtocolError("Target feature-set schema drifted.")
    from ..residual_topup.hashing import canonical_sha256

    try:
        if raw.get("target_feature_hash") != canonical_sha256(
            {key: value for key, value in raw.items() if key != "target_feature_hash"}
        ):
            raise ProtocolError("Target feature-set hash drifted.")
        plan_payload = raw["case_bootstrap_plan"]
        if not isinstance(plan_payload, Mapping):
            raise ProtocolError("Target case-bootstrap plan is malformed.")
        support_ids = plan_payload.get("support_case_ids", ())
        if not isinstance(support_ids, Sequence) or isinstance(support_ids, (str, bytes)):
            raise ProtocolError("Target case-bootstrap support IDs are malformed.")
        plan = build_case_bootstrap_plan(
            target_id=str(raw["target_id"]),
            support_case_ids=tuple(support_ids),
            bootstrap_seed=int(plan_payload.get("bootstrap_seed", -1)),
            replicate_count=int(plan_payload.get("replicate_count", -1)),
        )
        if plan.to_payload() != dict(plan_payload):
            raise ProtocolError("Target case-bootstrap plan was fabricated or drifted.")
        point_rows = tuple(_candidate_from_payload(value) for value in _objects(raw["point_rows"]))
        point = build_distributional_feature_surface(point_rows)
        if point.surface_hash != raw["point_surface_hash"]:
            raise ProtocolError("Target point feature surface hash drifted.")
        values = _objects(raw["bootstrap_surfaces"])
        if len(values) != plan.replicate_count:
            raise ProtocolError("Target bootstrap feature count drifted.")
        bootstraps = []
        for replicate, value in zip(plan.replicates, values, strict=True):
            if set(value) != {"replicate_index", "replicate_hash", "rows", "surface_hash"} or value.get("replicate_index") != replicate.replicate_index or value.get("replicate_hash") != replicate.replicate_hash:
                raise ProtocolError("Target bootstrap replicate order drifted.")
            surface = build_distributional_feature_surface(
                tuple(_candidate_from_payload(item) for item in _objects(value["rows"])),
                case_bootstrap_replicate=replicate,
            )
            if surface.surface_hash != value["surface_hash"]:
                raise ProtocolError("Target bootstrap surface hash drifted.")
            bootstraps.append(surface)
        return TargetFeatureProduction(
            target_id=str(raw["target_id"]), bootstrap_plan=plan,
            point_rows=point_rows, point_surface=point,
            bootstrap_surfaces=tuple(bootstraps),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Target feature-set payload is malformed.") from exc


def _candidate_from_payload(raw: Mapping[str, object]) -> CandidateFeatureRow:
    expected = {
        "schema_version", "role", "outer_target_id", "query_id", "candidate_source",
        "training_seed", "generation_seed", "replicate_id", "candidate_source_count",
        "support_partition_hash", "support_case_count", "reconstruction_mean",
        "reconstruction_std", "reconstruction_q25", "reconstruction_q50",
        "reconstruction_q75", "kl_mean", "kl_std", "kl_q25", "kl_q50", "kl_q75",
        "replica_disagreement", "distribution_mmd", "metadata_similarity",
        "feature_semantics", "row_hash",
    }
    if set(raw) != expected:
        raise ProtocolError("Serialized target feature row schema drifted.")
    try:
        row = CandidateFeatureRow(
            role=str(raw["role"]), outer_target_id=str(raw["outer_target_id"]),
            query_id=str(raw["query_id"]), candidate_source=str(raw["candidate_source"]),
            training_seed=int(raw["training_seed"]), generation_seed=int(raw["generation_seed"]),
            candidate_source_count=int(raw["candidate_source_count"]),
            support_partition_hash=str(raw["support_partition_hash"]),
            support_case_count=int(raw["support_case_count"]),
            reconstruction_mean=float(raw["reconstruction_mean"]), reconstruction_std=float(raw["reconstruction_std"]),
            reconstruction_q25=float(raw["reconstruction_q25"]), reconstruction_q50=float(raw["reconstruction_q50"]), reconstruction_q75=float(raw["reconstruction_q75"]),
            kl_mean=float(raw["kl_mean"]), kl_std=float(raw["kl_std"]),
            kl_q25=float(raw["kl_q25"]), kl_q50=float(raw["kl_q50"]), kl_q75=float(raw["kl_q75"]),
            replica_disagreement=float(raw["replica_disagreement"]), distribution_mmd=float(raw["distribution_mmd"]),
            metadata_similarity=float(raw["metadata_similarity"]), feature_semantics=str(raw["feature_semantics"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Serialized target feature row numeric field drifted.") from exc
    if raw["schema_version"] != "midogpp_utility_aligned_candidate_feature_row_v1" or raw["replicate_id"] != row.replicate_id or raw["row_hash"] != row.row_hash:
        raise ProtocolError("Serialized target feature row identity drifted.")
    return row


def _objects(raw: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or any(not isinstance(value, Mapping) for value in raw):
        raise ProtocolError("Serialized target feature sequence is malformed.")
    return tuple(raw)  # type: ignore[return-value]


def _build_rows(
    *,
    target: str,
    raw_case_ids: Sequence[str],
    selected_case_ids: Sequence[str],
    support_partition_hash: str,
    components: Mapping[str, TargetCandidateComponents],
) -> tuple[CandidateFeatureRow, ...]:
    selected = tuple(selected_case_ids)
    if not selected or any(value not in set(raw_case_ids) for value in selected):
        raise ProtocolError("Target bootstrap attempted to fabricate a support case.")
    case_count = len(selected)
    rows: list[CandidateFeatureRow] = []
    for source in target_sources(target):
        component = components[source]
        replica_energy_by_seed = {
            seed: _selected_mean(
                _case_equal_energy(
                    component.reconstruction_by_training_seed[seed],
                    component.normalized_ps_kl_by_training_seed[seed],
                    raw_case_ids,
                ),
                selected,
            )
            for seed in TRAINING_SEEDS
        }
        for training_seed in TRAINING_SEEDS:
            reconstruction_cases = _case_equal_component(
                component.reconstruction_by_training_seed[training_seed],
                raw_case_ids,
            )
            kl_cases = _case_equal_component(
                component.normalized_ps_kl_by_training_seed[training_seed],
                raw_case_ids,
            )
            reconstruction_stats = _stats(
                np.asarray(
                    [reconstruction_cases[case_id] for case_id in selected],
                    dtype=np.float64,
                )
            )
            kl_stats = _stats(
                np.asarray(
                    [kl_cases[case_id] for case_id in selected], dtype=np.float64
                )
            )
            disagreement = float(
                np.std(tuple(replica_energy_by_seed.values()), ddof=0)
            )
            for generation_seed in GENERATION_SEEDS:
                rows.append(
                    CandidateFeatureRow(
                        role=TARGET_ROLE,
                        outer_target_id=target,
                        query_id=target,
                        candidate_source=source,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        candidate_source_count=8,
                        support_partition_hash=support_partition_hash,
                        support_case_count=case_count,
                        reconstruction_mean=reconstruction_stats[0],
                        reconstruction_std=reconstruction_stats[1],
                        reconstruction_q25=reconstruction_stats[2],
                        reconstruction_q50=reconstruction_stats[3],
                        reconstruction_q75=reconstruction_stats[4],
                        kl_mean=kl_stats[0],
                        kl_std=kl_stats[1],
                        kl_q25=kl_stats[2],
                        kl_q50=kl_stats[3],
                        kl_q75=kl_stats[4],
                        replica_disagreement=disagreement,
                        distribution_mmd=_resampled_linear_kernel_mmd2(
                            component.support_case_mean_embeddings,
                            component.generated_mean_by_seed_pair[
                                (training_seed, generation_seed)
                            ],
                            selected,
                        ),
                        metadata_similarity=component.metadata_similarity,
                    )
                )
    if len(rows) != 72:
        raise ProtocolError("Target feature row geometry drifted from 8x9.")
    return tuple(rows)


def _component_grid(
    raw: Mapping[int, Mapping[int, np.ndarray]], role: str
) -> dict[int, Mapping[int, np.ndarray]]:
    if set(raw) != set(TRAINING_SEEDS):
        raise ProtocolError(f"Target {role} requires exact seeds 17,42,101.")
    result: dict[int, Mapping[int, np.ndarray]] = {}
    for seed in TRAINING_SEEDS:
        classes = raw[seed]
        if set(classes) != {0, 1}:
            raise ProtocolError(f"Target {role} requires both class hypotheses.")
        values = {
            label: np.asarray(classes[label], dtype=np.float64) for label in (0, 1)
        }
        if (
            any(value.ndim != 1 or not len(value) for value in values.values())
            or values[0].shape != values[1].shape
            or any(
                not np.isfinite(value).all() or np.any(value < 0.0)
                for value in values.values()
            )
        ):
            raise ProtocolError(f"Target {role} row components are invalid.")
        for value in values.values():
            value.setflags(write=False)
        result[seed] = MappingProxyType(values)
    return result


def _case_equal_component(
    per_class: Mapping[int, np.ndarray], case_ids: Sequence[str]
) -> Mapping[str, float]:
    values = 0.5 * (
        np.asarray(per_class[0], dtype=np.float64)
        + np.asarray(per_class[1], dtype=np.float64)
    )
    return _case_means(values, case_ids)


def _case_equal_energy(
    reconstruction: Mapping[int, np.ndarray],
    kl: Mapping[int, np.ndarray],
    case_ids: Sequence[str],
) -> Mapping[str, float]:
    log_half = float(np.log(0.5))
    energy_0 = np.asarray(reconstruction[0]) + np.asarray(kl[0])
    energy_1 = np.asarray(reconstruction[1]) + np.asarray(kl[1])
    row_energy = -np.logaddexp(log_half - energy_0, log_half - energy_1)
    return _case_means(row_energy, case_ids)


def _case_means(values: np.ndarray, case_ids: Sequence[str]) -> Mapping[str, float]:
    cases = tuple(case_ids)
    if values.shape != (len(cases),):
        raise ProtocolError("Target case component alignment drifted.")
    result = {
        case_id: float(np.mean(values[np.asarray([v == case_id for v in cases])]))
        for case_id in sorted(set(cases))
    }
    if not result or not np.isfinite(tuple(result.values())).all():
        raise ProtocolError("Target case-equal summaries are invalid.")
    return MappingProxyType(result)


def _selected_mean(values: Mapping[str, float], selected: Sequence[str]) -> float:
    return float(np.mean([values[case_id] for case_id in selected], dtype=np.float64))


def _resampled_linear_kernel_mmd2(
    support_case_means: Mapping[str, np.ndarray],
    generated_mean: np.ndarray,
    selected: Sequence[str],
) -> float:
    support_mean = np.mean(
        np.stack([support_case_means[case_id] for case_id in selected]),
        axis=0,
        dtype=np.float64,
    )
    difference = support_mean - np.asarray(generated_mean, dtype=np.float64)
    value = float(np.dot(difference, difference))
    if not np.isfinite(value) or value < 0.0:
        raise ProtocolError("Target resampled linear-kernel MMD squared is invalid.")
    return value


def _stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ProtocolError("Target bootstrap distribution is invalid.")
    quantiles = np.quantile(values, (0.25, 0.5, 0.75))
    return (
        float(np.mean(values, dtype=np.float64)),
        float(np.std(values, ddof=0)),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
    )


__all__ = (
    "TargetCandidateComponents",
    "TargetFeatureProduction",
    "build_target_feature_production",
    "target_feature_production_from_payload",
)
