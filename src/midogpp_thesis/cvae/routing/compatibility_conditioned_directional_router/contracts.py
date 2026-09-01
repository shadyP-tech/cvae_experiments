"""Typed contracts for compatibility-conditioned directional routing.

Target contracts are intentionally outcome-free.  Only source-development
contracts contain endpoint effects, which makes it impossible to pass target
evaluation labels into fitting, uncertainty calibration, selection, or
composition by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import struct
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash, probability_bytes_hash, require_sha256


TRAINING_SEEDS = (17, 42, 101)
ALPHA_GRID = (0.1, 1.0, 10.0)
ENDPOINTS = ("bacc_gain", "brier_delta", "log_delta")
_FORBIDDEN_PRETERMINAL_FEATURE_TOKENS = (
    "label",
    "truth",
    "outcome",
    "oracle",
    "bacc",
    "brier",
    "log_loss",
    "evaluation_endpoint",
)


class ActionKind(str, Enum):
    B = "B"
    U = "U"
    HXE = "HXE"


class Direction(str, Enum):
    D01 = "D01"
    D10 = "D10"
    ALL = "ALL_MARGINS"


def canonical_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProtocolError(f"{name} must be a canonical nonempty string.")
    return value


def finite(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite.")
    return 0.0 if result == 0.0 else result


def canonical_names_values(
    names: Sequence[str], values: Sequence[float]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    normalized_names = tuple(canonical_text(value, name="feature name") for value in names)
    normalized_values = tuple(finite(value, name="feature value") for value in values)
    if (
        not normalized_names
        or len(normalized_names) != len(normalized_values)
        or len(set(normalized_names)) != len(normalized_names)
    ):
        raise ProtocolError("Directional-router features must be unique and aligned.")
    return normalized_names, normalized_values


def canonical_probability_bytes(values: Sequence[bytes]) -> tuple[bytes, ...]:
    output: list[bytes] = []
    for raw in values:
        if type(raw) is not bytes or len(raw) != 4:
            raise ProtocolError("Probabilities must retain exact little-endian float32 bytes.")
        value = struct.unpack("<f", raw)[0]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProtocolError("Probability cells must lie in [0,1].")
        output.append(raw)
    if not output:
        raise ProtocolError("A probability surface cannot be empty.")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class CandidatePoolReceipt:
    """Role-complete candidate pool for one outer target H and query q.

    Source development uses ``C \\ {H, q}``.  Target inference has ``q == H``
    and uses ``C \\ {H}``.  There is no caller-selectable relaxation.
    """

    outer_target_id: str
    query_center_id: str
    all_center_ids: tuple[str, ...]
    candidate_center_ids: tuple[str, ...]
    bank_lock_hash: str
    pool_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="outer target H")
        q = canonical_text(self.query_center_id, name="query center q")
        centers = tuple(sorted(canonical_text(value, name="center") for value in self.all_center_ids))
        candidates = tuple(
            sorted(canonical_text(value, name="candidate center") for value in self.candidate_center_ids)
        )
        if (
            len(set(centers)) != len(centers)
            or len(set(candidates)) != len(candidates)
            or h not in centers
            or q not in centers
        ):
            raise ProtocolError("Candidate-pool center inventory is malformed.")
        excluded = {h} if q == h else {h, q}
        expected = tuple(value for value in centers if value not in excluded)
        if candidates != expected or not candidates:
            scope = "C-minus-H" if q == h else "C-minus-H-minus-q"
            raise ProtocolError(f"Candidate pool is not the exact {scope} inventory.")
        bank_hash = require_sha256(self.bank_lock_hash, name="expert-bank lock hash")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "query_center_id", q)
        object.__setattr__(self, "all_center_ids", centers)
        object.__setattr__(self, "candidate_center_ids", candidates)
        object.__setattr__(self, "bank_lock_hash", bank_hash)
        object.__setattr__(
            self,
            "pool_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_candidate_pool_v1",
                    "outer_target_H": h,
                    "query_q": q,
                    "all_centers": centers,
                    "candidate_centers": candidates,
                    "role_exclusion": "C_MINUS_H" if q == h else "C_MINUS_H_MINUS_Q",
                    "bank_lock_hash": bank_hash,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def target_scope(self) -> bool:
        return self.query_center_id == self.outer_target_id


@dataclass(frozen=True, slots=True)
class SupportPartitionReceipt:
    """Disjoint unlabeled target support/evaluation case identities."""

    center_id: str
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    support_manifest_hash: str
    evaluation_manifest_hash: str
    partition_hash: str = field(init=False)
    labels_consumed: bool = False

    def __post_init__(self) -> None:
        center = canonical_text(self.center_id, name="partition center")
        support = tuple(sorted(canonical_text(value, name="support case") for value in self.support_case_ids))
        evaluation = tuple(
            sorted(canonical_text(value, name="evaluation case") for value in self.evaluation_case_ids)
        )
        if (
            not support
            or not evaluation
            or len(set(support)) != len(support)
            or len(set(evaluation)) != len(evaluation)
            or set(support).intersection(evaluation)
            or self.labels_consumed is not False
        ):
            raise ProtocolError("Support and evaluation cases must be nonempty, unique, and disjoint.")
        support_manifest = require_sha256(self.support_manifest_hash, name="support manifest hash")
        evaluation_manifest = require_sha256(
            self.evaluation_manifest_hash, name="evaluation manifest hash"
        )
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_case_ids", evaluation)
        object.__setattr__(self, "support_manifest_hash", support_manifest)
        object.__setattr__(self, "evaluation_manifest_hash", evaluation_manifest)
        object.__setattr__(
            self,
            "partition_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_support_partition_v1",
                    "center_id": center,
                    "support_case_ids": support,
                    "evaluation_case_ids": evaluation,
                    "support_manifest_hash": support_manifest,
                    "evaluation_manifest_hash": evaluation_manifest,
                    "whole_case_split": True,
                    "labels_consumed": False,
                }
            ),
        )

    @property
    def support_hash(self) -> str:
        return canonical_hash(
            {
                "center_id": self.center_id,
                "support_case_ids": self.support_case_ids,
                "support_manifest_hash": self.support_manifest_hash,
                "partition_hash": self.partition_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class ReplicaEnergyInput:
    """Label-free query and own-source energy for one fixed expert replica."""

    candidate_source_id: str
    training_seed: int
    query_case_equal_energy: float
    own_source_location: float
    own_source_scale: float
    checkpoint_hash: str
    source_frame_hash: str
    sampler_hash: str
    exact_nelbo: bool = False
    labels_consumed: bool = False

    def __post_init__(self) -> None:
        source = canonical_text(self.candidate_source_id, name="compatibility candidate")
        seed = int(self.training_seed)
        scale = finite(self.own_source_scale, name="own-source energy scale")
        if (
            seed not in TRAINING_SEEDS
            or scale <= 0.0
            or self.exact_nelbo is not False
            or self.labels_consumed is not False
        ):
            raise ProtocolError("Compatibility replica semantics drifted.")
        object.__setattr__(self, "candidate_source_id", source)
        object.__setattr__(self, "training_seed", seed)
        object.__setattr__(
            self, "query_case_equal_energy", finite(self.query_case_equal_energy, name="query energy")
        )
        object.__setattr__(
            self, "own_source_location", finite(self.own_source_location, name="own-source location")
        )
        object.__setattr__(self, "own_source_scale", scale)
        for name in ("checkpoint_hash", "source_frame_hash", "sampler_hash"):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))

    @property
    def calibrated_z(self) -> float:
        return (self.query_case_equal_energy - self.own_source_location) / self.own_source_scale


@dataclass(frozen=True, slots=True)
class CompatibilityReceipt:
    """All-three-seed, label-free compatibility summary for one candidate."""

    outer_target_id: str
    query_center_id: str
    candidate_source_id: str
    candidate_pool_hash: str
    support_partition_hash: str
    support_hash: str
    support_manifest_hash: str
    replica_scores: tuple[ReplicaEnergyInput, ...]
    mean_z: float
    std_z: float
    rank: int
    rank_margin: float
    exact_nelbo: bool = False
    labels_consumed: bool = False
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="compatibility outer H")
        q = canonical_text(self.query_center_id, name="compatibility query q")
        source = canonical_text(self.candidate_source_id, name="compatibility candidate")
        replicas = tuple(sorted(self.replica_scores, key=lambda row: row.training_seed))
        if (
            tuple(row.training_seed for row in replicas) != TRAINING_SEEDS
            or any(row.candidate_source_id != source for row in replicas)
            or self.rank < 1
            or self.exact_nelbo is not False
            or self.labels_consumed is not False
        ):
            raise ProtocolError("Compatibility receipt must average exactly seeds 17/42/101.")
        mean_z = finite(self.mean_z, name="compatibility mean z")
        std_z = finite(self.std_z, name="compatibility std z")
        margin = finite(self.rank_margin, name="compatibility rank margin")
        expected_mean = sum(row.calibrated_z for row in replicas) / len(replicas)
        expected_std = math.sqrt(
            sum((row.calibrated_z - expected_mean) ** 2 for row in replicas) / len(replicas)
        )
        if (
            std_z < 0.0
            or not math.isclose(mean_z, expected_mean, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(std_z, expected_std, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ProtocolError("Compatibility receipt statistics drifted from replica scores.")
        for name in (
            "candidate_pool_hash",
            "support_partition_hash",
            "support_hash",
            "support_manifest_hash",
        ):
            object.__setattr__(self, name, require_sha256(getattr(self, name), name=name))
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "query_center_id", q)
        object.__setattr__(self, "candidate_source_id", source)
        object.__setattr__(self, "replica_scores", replicas)
        object.__setattr__(self, "mean_z", mean_z)
        object.__setattr__(self, "std_z", std_z)
        object.__setattr__(self, "rank_margin", margin)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_conditioned_directional_receipt_v1",
                    "H": h,
                    "q": q,
                    "candidate_source": source,
                    "candidate_pool_hash": self.candidate_pool_hash,
                    "support_partition_hash": self.support_partition_hash,
                    "support_hash": self.support_hash,
                    "support_manifest_hash": self.support_manifest_hash,
                    "training_seeds": TRAINING_SEEDS,
                    "replicas": tuple(
                        {
                            "seed": row.training_seed,
                            "query_energy": row.query_case_equal_energy,
                            "own_location": row.own_source_location,
                            "own_scale": row.own_source_scale,
                            "calibrated_z": row.calibrated_z,
                            "checkpoint_hash": row.checkpoint_hash,
                            "source_frame_hash": row.source_frame_hash,
                            "sampler_hash": row.sampler_hash,
                        }
                        for row in replicas
                    ),
                    "mean_z": mean_z,
                    "std_z": std_z,
                    "rank": self.rank,
                    "rank_margin": margin,
                    "energy_semantics": "variational_compatibility_proxy_not_exact_nelbo",
                    "exact_nelbo": False,
                    "labels_consumed": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CandidateFeatureVector:
    """One label-free, candidate-aware action descriptor."""

    outer_target_id: str
    query_center_id: str
    case_id: str
    action_id: str
    action_kind: ActionKind
    direction: Direction
    candidate_source_id: str | None
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    candidate_pool_hash: str
    probability_hash: str
    compatibility_receipt_hash: str | None
    feature_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="feature outer H")
        q = canonical_text(self.query_center_id, name="feature query q")
        case = canonical_text(self.case_id, name="feature case")
        action_id = canonical_text(self.action_id, name="feature action")
        try:
            kind = ActionKind(self.action_kind)
            direction = Direction(self.direction)
        except (TypeError, ValueError) as exc:
            raise ProtocolError("Candidate feature action semantics are invalid.") from exc
        candidate = self.candidate_source_id
        compatibility_hash = self.compatibility_receipt_hash
        if kind is ActionKind.HXE:
            candidate = canonical_text(candidate, name="candidate source")
            compatibility_hash = require_sha256(
                compatibility_hash, name="compatibility receipt hash"
            )
        elif kind is ActionKind.U:
            if candidate is not None or compatibility_hash is not None:
                raise ProtocolError("Uniform actions cannot carry expert compatibility.")
        else:
            raise ProtocolError("Protected B is implicit and cannot be a challenger feature row.")
        names, values = canonical_names_values(self.feature_names, self.feature_values)
        if any(
            any(token in name.lower() for token in _FORBIDDEN_PRETERMINAL_FEATURE_TOKENS)
            for name in names
        ):
            raise ProtocolError("Preterminal candidate features cannot encode endpoint outcomes.")
        pool_hash = require_sha256(self.candidate_pool_hash, name="candidate pool hash")
        surface_hash = require_sha256(self.probability_hash, name="probability hash")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "query_center_id", q)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "action_id", action_id)
        object.__setattr__(self, "action_kind", kind)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "candidate_source_id", candidate)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "candidate_pool_hash", pool_hash)
        object.__setattr__(self, "probability_hash", surface_hash)
        object.__setattr__(self, "compatibility_receipt_hash", compatibility_hash)
        object.__setattr__(
            self,
            "feature_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_feature_v1",
                    "H": h,
                    "q": q,
                    "case": case,
                    "action_id": action_id,
                    "action_kind": kind.value,
                    "direction": direction.value,
                    "candidate_source": candidate,
                    "feature_names": names,
                    "feature_values": values,
                    "candidate_pool_hash": pool_hash,
                    "probability_hash": surface_hash,
                    "compatibility_receipt_hash": compatibility_hash,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class EndpointEffects:
    """Source-development effects relative to protected B."""

    bacc_gain: float
    brier_delta: float
    log_delta: float

    def __post_init__(self) -> None:
        for name in ENDPOINTS:
            object.__setattr__(self, name, finite(getattr(self, name), name=name))

    def as_tuple(self) -> tuple[float, float, float]:
        return self.bacc_gain, self.brier_delta, self.log_delta


@dataclass(frozen=True, slots=True)
class SourceActionObservation:
    """Source-only response row; target cases cannot instantiate this contract."""

    feature: CandidateFeatureVector
    candidate_pool: CandidatePoolReceipt
    effects: EndpointEffects
    source_response_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.feature, CandidateFeatureVector)
            or not isinstance(self.candidate_pool, CandidatePoolReceipt)
            or not isinstance(self.effects, EndpointEffects)
            or self.candidate_pool.target_scope
            or self.feature.outer_target_id != self.candidate_pool.outer_target_id
            or self.feature.query_center_id != self.candidate_pool.query_center_id
            or self.feature.candidate_pool_hash != self.candidate_pool.pool_hash
            or (
                self.feature.action_kind is ActionKind.HXE
                and self.feature.candidate_source_id not in self.candidate_pool.candidate_center_ids
            )
        ):
            raise ProtocolError("Source action observation violated strict outer-H/query exclusion.")
        object.__setattr__(
            self,
            "source_response_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_source_response_v1",
                    "feature_hash": self.feature.feature_hash,
                    "candidate_pool_hash": self.candidate_pool.pool_hash,
                    "effects": self.effects.as_tuple(),
                    "response_scope": "SOURCE_DEVELOPMENT_ONLY",
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TargetAction:
    """Sealed label-free target action with exact float32 probabilities."""

    feature: CandidateFeatureVector
    candidate_pool: CandidatePoolReceipt
    sample_ids: tuple[str, ...]
    probability_bytes: tuple[bytes, ...]
    prediction_seal_hash: str
    target_action_hash: str = field(init=False)

    def __post_init__(self) -> None:
        samples = tuple(canonical_text(value, name="target sample") for value in self.sample_ids)
        probabilities = canonical_probability_bytes(self.probability_bytes)
        if (
            not isinstance(self.feature, CandidateFeatureVector)
            or not isinstance(self.candidate_pool, CandidatePoolReceipt)
            or not self.candidate_pool.target_scope
            or self.feature.outer_target_id != self.candidate_pool.outer_target_id
            or self.feature.query_center_id != self.candidate_pool.query_center_id
            or self.feature.candidate_pool_hash != self.candidate_pool.pool_hash
            or len(samples) != len(probabilities)
            or len(set(samples)) != len(samples)
            or self.feature.probability_hash != probability_bytes_hash(probabilities)
            or (
                self.feature.action_kind is ActionKind.HXE
                and self.feature.candidate_source_id not in self.candidate_pool.candidate_center_ids
            )
        ):
            raise ProtocolError("Target action is not a sealed C-minus-H label-free surface.")
        seal = require_sha256(self.prediction_seal_hash, name="prediction seal hash")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "probability_bytes", probabilities)
        object.__setattr__(self, "prediction_seal_hash", seal)
        object.__setattr__(
            self,
            "target_action_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_target_action_v1",
                    "feature_hash": self.feature.feature_hash,
                    "candidate_pool_hash": self.candidate_pool.pool_hash,
                    "sample_ids": samples,
                    "probability_hash": probability_bytes_hash(probabilities),
                    "prediction_seal_hash": seal,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldLoss:
    held_center_id: str
    alpha: float
    hurdle_log_loss: float
    pairwise_mse: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "held_center_id", canonical_text(self.held_center_id, name="held K"))
        object.__setattr__(self, "alpha", finite(self.alpha, name="alpha"))
        object.__setattr__(self, "hurdle_log_loss", finite(self.hurdle_log_loss, name="hurdle loss"))
        object.__setattr__(self, "pairwise_mse", finite(self.pairwise_mse, name="pairwise loss"))
        if self.alpha <= 0 or self.hurdle_log_loss < 0 or self.pairwise_mse < 0:
            raise ProtocolError("Nested source-center fold losses are malformed.")


@dataclass(frozen=True, slots=True)
class HurdlePairwiseModel:
    """Candidate-aware generalized action model; B has an exact zero score."""

    outer_target_id: str
    feature_names: tuple[str, ...]
    normalization_mean: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    design_names: tuple[str, ...]
    hurdle_coefficients: tuple[float, ...]
    pairwise_coefficients: tuple[float, ...]
    endpoint_coefficients: tuple[tuple[str, tuple[float, ...]], ...]
    selected_alpha: float
    alpha_grid: tuple[float, ...]
    fold_losses: tuple[FoldLoss, ...]
    training_query_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    training_case_count: int
    training_row_hash: str
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="model outer H")
        names, means = canonical_names_values(self.feature_names, self.normalization_mean)
        scales = tuple(finite(value, name="normalization scale") for value in self.normalization_scale)
        design = tuple(canonical_text(value, name="design name") for value in self.design_names)
        hurdle = tuple(finite(value, name="hurdle coefficient") for value in self.hurdle_coefficients)
        pairwise = tuple(finite(value, name="pairwise coefficient") for value in self.pairwise_coefficients)
        endpoints = tuple(
            (
                str(name),
                tuple(finite(value, name=f"{name} endpoint coefficient") for value in values),
            )
            for name, values in self.endpoint_coefficients
        )
        queries = tuple(sorted(canonical_text(value, name="training query") for value in self.training_query_ids))
        candidates = tuple(
            sorted(canonical_text(value, name="training candidate") for value in self.training_candidate_ids)
        )
        grid = tuple(float(value) for value in self.alpha_grid)
        if (
            len(scales) != len(names)
            or any(value <= 0.0 for value in scales)
            or not design
            or len(set(design)) != len(design)
            or len(hurdle) != len(design)
            or len(pairwise) != len(design)
            or tuple(name for name, _ in endpoints) != ENDPOINTS
            or any(len(values) != len(design) for _, values in endpoints)
            or grid != tuple(sorted(set(grid)))
            or self.selected_alpha not in grid
            or not self.fold_losses
            or not queries
            or h in queries
            or h in candidates
            or self.training_case_count < 1
        ):
            raise ProtocolError("Hurdle-pairwise model contract is malformed.")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "normalization_mean", means)
        object.__setattr__(self, "normalization_scale", scales)
        object.__setattr__(self, "design_names", design)
        object.__setattr__(self, "hurdle_coefficients", hurdle)
        object.__setattr__(self, "pairwise_coefficients", pairwise)
        object.__setattr__(self, "endpoint_coefficients", endpoints)
        object.__setattr__(self, "alpha_grid", grid)
        object.__setattr__(self, "training_query_ids", queries)
        object.__setattr__(self, "training_candidate_ids", candidates)
        object.__setattr__(self, "training_row_hash", require_sha256(self.training_row_hash, name="training row hash"))
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_hurdle_pairwise_model_v1",
                    "outer_target_H": h,
                    "feature_names": names,
                    "normalization_mean": means,
                    "normalization_scale": scales,
                    "design_names": design,
                    "hurdle_coefficients": hurdle,
                    "pairwise_coefficients": pairwise,
                    "endpoint_coefficients": endpoints,
                    "selected_alpha": self.selected_alpha,
                    "alpha_grid": grid,
                    "fold_losses": self.fold_losses,
                    "training_query_ids": queries,
                    "training_candidate_ids": candidates,
                    "training_case_count": self.training_case_count,
                    "training_row_hash": self.training_row_hash,
                    "target_labels_used": False,
                }
            ),
        )

    def endpoint_coefficients_for(self, endpoint: str) -> tuple[float, ...]:
        try:
            return dict(self.endpoint_coefficients)[endpoint]
        except KeyError as exc:
            raise ProtocolError(f"Unknown endpoint: {endpoint}") from exc


@dataclass(frozen=True, slots=True)
class ActionPrediction:
    feature: CandidateFeatureVector
    opportunity_probability: float
    ranking_score: float
    predicted_effects: EndpointEffects
    model_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.feature, CandidateFeatureVector) or not isinstance(
            self.predicted_effects, EndpointEffects
        ):
            raise ProtocolError("Action prediction requires typed label-free inputs.")
        probability = finite(self.opportunity_probability, name="opportunity probability")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Opportunity probability must lie in [0,1].")
        score = finite(self.ranking_score, name="ranking score")
        model_hash = require_sha256(self.model_hash, name="model hash")
        object.__setattr__(self, "opportunity_probability", probability)
        object.__setattr__(self, "ranking_score", score)
        object.__setattr__(self, "model_hash", model_hash)
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_action_prediction_v1",
                    "feature_hash": self.feature.feature_hash,
                    "opportunity_probability": probability,
                    "ranking_score": score,
                    "predicted_effects": self.predicted_effects.as_tuple(),
                    "model_hash": model_hash,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceOOFPrediction:
    """Role-complete source-center-OOF prediction with source effects only."""

    held_center_id: str
    prediction: ActionPrediction
    observed: EndpointEffects
    fold_training_query_ids: tuple[str, ...]
    fold_training_candidate_ids: tuple[str, ...]
    fold_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        held = canonical_text(self.held_center_id, name="OOF held center")
        queries = tuple(
            sorted(canonical_text(value, name="OOF training query") for value in self.fold_training_query_ids)
        )
        candidates = tuple(
            sorted(
                canonical_text(value, name="OOF training candidate")
                for value in self.fold_training_candidate_ids
            )
        )
        if (
            not isinstance(self.prediction, ActionPrediction)
            or not isinstance(self.observed, EndpointEffects)
            or held in queries
            or held in candidates
            or self.prediction.feature.query_center_id != held
        ):
            raise ProtocolError("Source OOF prediction leaked its held query center.")
        fold_hash = require_sha256(self.fold_hash, name="OOF fold hash")
        object.__setattr__(self, "held_center_id", held)
        object.__setattr__(self, "fold_training_query_ids", queries)
        object.__setattr__(self, "fold_training_candidate_ids", candidates)
        object.__setattr__(self, "fold_hash", fold_hash)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_source_oof_prediction_v1",
                    "held_center": held,
                    "prediction_hash": self.prediction.prediction_hash,
                    "observed": self.observed.as_tuple(),
                    "fold_training_queries": queries,
                    "fold_training_candidates": candidates,
                    "fold_hash": fold_hash,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OOFEndpointRow:
    """Source-only out-of-fold prediction used for exact-group bounds."""

    query_center_id: str
    case_id: str
    action_key: str
    comparator_key: str
    predicted: EndpointEffects
    observed: EndpointEffects
    fold_model_hash: str

    def __post_init__(self) -> None:
        for name in ("query_center_id", "case_id", "action_key", "comparator_key"):
            object.__setattr__(self, name, canonical_text(getattr(self, name), name=name))
        if self.action_key == self.comparator_key:
            raise ProtocolError("Endpoint calibration requires distinct action/comparator keys.")
        if not isinstance(self.predicted, EndpointEffects) or not isinstance(self.observed, EndpointEffects):
            raise ProtocolError("Endpoint OOF rows require typed effects.")
        object.__setattr__(self, "fold_model_hash", require_sha256(self.fold_model_hash, name="fold model hash"))


@dataclass(frozen=True, slots=True)
class EndpointCalibrationCell:
    action_key: str
    comparator_key: str
    bacc_overprediction_quantile: float
    brier_underprediction_quantile: float
    log_underprediction_quantile: float
    source_center_ids: tuple[str, ...]
    row_count: int
    cell_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action = canonical_text(self.action_key, name="calibration action")
        comparator = canonical_text(self.comparator_key, name="calibration comparator")
        centers = tuple(sorted(canonical_text(value, name="calibration center") for value in self.source_center_ids))
        values = tuple(
            finite(getattr(self, name), name=name)
            for name in (
                "bacc_overprediction_quantile",
                "brier_underprediction_quantile",
                "log_underprediction_quantile",
            )
        )
        if action == comparator or len(set(centers)) != len(centers) or not centers or self.row_count < len(centers):
            raise ProtocolError("Endpoint calibration cell is malformed.")
        object.__setattr__(self, "action_key", action)
        object.__setattr__(self, "comparator_key", comparator)
        object.__setattr__(self, "source_center_ids", centers)
        object.__setattr__(
            self,
            "cell_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_endpoint_cell_v1",
                    "action_key": action,
                    "comparator_key": comparator,
                    "bacc_overprediction_quantile": values[0],
                    "brier_underprediction_quantile": values[1],
                    "log_underprediction_quantile": values[2],
                    "source_center_ids": centers,
                    "row_count": self.row_count,
                    "pooled_fallback": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class EndpointCalibration:
    quantile: float
    cells: tuple[EndpointCalibrationCell, ...]
    source_oof_hash: str
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        quantile = finite(self.quantile, name="uncertainty quantile")
        cells = tuple(sorted(self.cells, key=lambda row: (row.action_key, row.comparator_key)))
        keys = tuple((row.action_key, row.comparator_key) for row in cells)
        if not 0.5 < quantile < 1.0 or not cells or len(set(keys)) != len(keys):
            raise ProtocolError("Endpoint uncertainty calibration is malformed.")
        source_hash = require_sha256(self.source_oof_hash, name="source OOF hash")
        object.__setattr__(self, "quantile", quantile)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "source_oof_hash", source_hash)
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_endpoint_calibration_v1",
                    "quantile": quantile,
                    "cell_hashes": tuple(row.cell_hash for row in cells),
                    "source_oof_hash": source_hash,
                    "grouping": "endpoint_by_action_by_comparator",
                    "pooled_fallback": False,
                    "target_labels_used": False,
                }
            ),
        )

    def cell(self, action_key: str, comparator_key: str) -> EndpointCalibrationCell:
        key = (str(action_key), str(comparator_key))
        for row in self.cells:
            if (row.action_key, row.comparator_key) == key:
                return row
        raise ProtocolError(
            f"No exact endpoint calibration for action/comparator {key}; pooled fallback is forbidden."
        )


@dataclass(frozen=True, slots=True)
class EndpointBounds:
    bacc_lcb: float
    brier_ucb: float
    log_ucb: float

    def __post_init__(self) -> None:
        for name in ("bacc_lcb", "brier_ucb", "log_ucb"):
            object.__setattr__(self, name, finite(getattr(self, name), name=name))


@dataclass(frozen=True, slots=True)
class BoundedActionEvidence:
    prediction: ActionPrediction
    comparator_key: str
    bounds: EndpointBounds
    uncertainty_calibration_hash: str
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, ActionPrediction) or not isinstance(self.bounds, EndpointBounds):
            raise ProtocolError("Bounded evidence requires a typed prediction and bounds.")
        comparator = canonical_text(self.comparator_key, name="evidence comparator")
        calibration_hash = require_sha256(
            self.uncertainty_calibration_hash, name="uncertainty calibration hash"
        )
        object.__setattr__(self, "comparator_key", comparator)
        object.__setattr__(self, "uncertainty_calibration_hash", calibration_hash)
        object.__setattr__(
            self,
            "evidence_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_bounded_evidence_v1",
                    "prediction_hash": self.prediction.prediction_hash,
                    "comparator_key": comparator,
                    "bounds": (
                        self.bounds.bacc_lcb,
                        self.bounds.brier_ucb,
                        self.bounds.log_ucb,
                    ),
                    "uncertainty_calibration_hash": calibration_hash,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def safe_vs_baseline(self) -> bool:
        return (
            self.comparator_key == ActionKind.B.value
            and self.bounds.bacc_lcb > 0.0
            and self.bounds.brier_ucb <= 0.0
            and self.bounds.log_ucb <= 0.0
        )


@dataclass(frozen=True, slots=True)
class SourceAdmissionCandidate:
    action_id: str
    predicted_score: float
    opportunity_probability: float
    safe_selected: bool
    observed: EndpointEffects

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", canonical_text(self.action_id, name="admission action"))
        object.__setattr__(self, "predicted_score", finite(self.predicted_score, name="predicted score"))
        probability = finite(self.opportunity_probability, name="opportunity probability")
        if not 0.0 <= probability <= 1.0:
            raise ProtocolError("Admission opportunity probability must lie in [0,1].")
        if not isinstance(self.observed, EndpointEffects):
            raise ProtocolError("Admission candidates require source-only observed effects.")
        object.__setattr__(self, "opportunity_probability", probability)


@dataclass(frozen=True, slots=True)
class SourceAdmissionCase:
    query_center_id: str
    case_id: str
    candidates: tuple[SourceAdmissionCandidate, ...]

    def __post_init__(self) -> None:
        candidates = tuple(sorted(self.candidates, key=lambda row: row.action_id))
        if not candidates or len({row.action_id for row in candidates}) != len(candidates):
            raise ProtocolError("Source admission case candidates are empty or duplicated.")
        object.__setattr__(self, "query_center_id", canonical_text(self.query_center_id, name="admission center"))
        object.__setattr__(self, "case_id", canonical_text(self.case_id, name="admission case"))
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class AdmissionThresholds:
    minimum_center_count: int = 4
    minimum_case_count: int = 12
    minimum_sign_accuracy: float = 0.55
    minimum_top1_accuracy: float = 0.35
    minimum_delete_center_tau: float = 0.0
    minimum_safe_coverage: float = 0.05
    maximum_harmful_selected: int = 0
    maximum_proper_loss_violations: int = 0

    def __post_init__(self) -> None:
        if (
            self.minimum_center_count < 3
            or self.minimum_case_count < 1
            or not 0.5 < self.minimum_sign_accuracy <= 1.0
            or not 0.0 < self.minimum_top1_accuracy <= 1.0
            or not -1.0 <= self.minimum_delete_center_tau < 1.0
            or not 0.0 < self.minimum_safe_coverage <= 1.0
            or self.maximum_harmful_selected < 0
            or self.maximum_proper_loss_violations < 0
        ):
            raise ProtocolError("Learnability admission thresholds are invalid or vacuous.")


@dataclass(frozen=True, slots=True)
class LearnabilityAdmission:
    passed: bool
    center_ids: tuple[str, ...]
    case_count: int
    sign_accuracy: float
    top1_accuracy: float
    minimum_delete_center_tau: float
    safe_coverage: float
    selected_count: int
    harmful_selected_count: int
    proper_loss_violation_count: int
    reasons: tuple[str, ...]
    source_oof_hash: str
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = tuple(sorted(canonical_text(value, name="admission center") for value in self.center_ids))
        reasons = tuple(canonical_text(value, name="admission reason") for value in self.reasons)
        metrics = tuple(
            finite(getattr(self, name), name=name)
            for name in ("sign_accuracy", "top1_accuracy", "minimum_delete_center_tau", "safe_coverage")
        )
        if (
            not centers
            or len(set(centers)) != len(centers)
            or bool(self.passed) == bool(reasons)
            or self.case_count < 1
            or min(self.selected_count, self.harmful_selected_count, self.proper_loss_violation_count) < 0
            or not 0.0 <= metrics[0] <= 1.0
            or not 0.0 <= metrics[1] <= 1.0
            or not -1.0 <= metrics[2] <= 1.0
            or not 0.0 <= metrics[3] <= 1.0
        ):
            raise ProtocolError("Learnability admission report is malformed.")
        source_hash = require_sha256(self.source_oof_hash, name="admission source OOF hash")
        object.__setattr__(self, "center_ids", centers)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "source_oof_hash", source_hash)
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_source_only_admission_v1",
                    "passed": self.passed,
                    "centers": centers,
                    "case_count": self.case_count,
                    "metrics": metrics,
                    "selected_count": self.selected_count,
                    "harmful_selected_count": self.harmful_selected_count,
                    "proper_loss_violation_count": self.proper_loss_violation_count,
                    "reasons": reasons,
                    "source_oof_hash": source_hash,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    outer_target_id: str
    case_id: str
    enabled: bool
    selected_direction: Direction | None
    selected_action_ids: tuple[str, ...]
    selected_weights: tuple[float, ...]
    mixture_lambda: float
    reason: str
    admission_hash: str
    evidence_hashes: tuple[str, ...]
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h = canonical_text(self.outer_target_id, name="decision outer H")
        case = canonical_text(self.case_id, name="decision case")
        actions = tuple(canonical_text(value, name="selected action") for value in self.selected_action_ids)
        weights = tuple(finite(value, name="selected weight") for value in self.selected_weights)
        direction = None if self.selected_direction is None else Direction(self.selected_direction)
        mixture_lambda = finite(self.mixture_lambda, name="mixture lambda")
        evidence = tuple(require_sha256(value, name="evidence hash") for value in self.evidence_hashes)
        if (
            not 0.0 <= mixture_lambda <= 1.0
            or len(actions) != len(weights)
            or len(set(actions)) != len(actions)
            or (
                self.enabled
                and (
                    not actions
                    or direction is None
                    or mixture_lambda <= 0.0
                    or any(value <= 0.0 for value in weights)
                    or not math.isclose(sum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12)
                    or len(evidence) != len(actions)
                )
            )
            or (
                not self.enabled
                and (actions or weights or direction is not None or mixture_lambda != 0.0 or evidence)
            )
        ):
            raise ProtocolError("Routing decision is internally inconsistent.")
        admission_hash = require_sha256(self.admission_hash, name="admission hash")
        object.__setattr__(self, "outer_target_id", h)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "selected_direction", direction)
        object.__setattr__(self, "selected_action_ids", actions)
        object.__setattr__(self, "selected_weights", weights)
        object.__setattr__(self, "mixture_lambda", mixture_lambda)
        object.__setattr__(self, "reason", canonical_text(self.reason, name="decision reason"))
        object.__setattr__(self, "admission_hash", admission_hash)
        object.__setattr__(self, "evidence_hashes", evidence)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_route_decision_v1",
                    "H": h,
                    "case": case,
                    "enabled": self.enabled,
                    "selected_direction": None if direction is None else direction.value,
                    "selected_action_ids": actions,
                    "selected_weights": weights,
                    "mixture_lambda": mixture_lambda,
                    "reason": self.reason,
                    "admission_hash": admission_hash,
                    "evidence_hashes": evidence,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CompositionReceipt:
    decision_hash: str
    baseline_probability_hash: str
    selected_probability_hashes: tuple[str, ...]
    output_probability_hash: str
    exact_baseline_fallback: bool
    opposite_branch_preserved: bool
    composition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        decision = require_sha256(self.decision_hash, name="decision hash")
        baseline = require_sha256(self.baseline_probability_hash, name="baseline probability hash")
        selected = tuple(require_sha256(value, name="selected probability hash") for value in self.selected_probability_hashes)
        output = require_sha256(self.output_probability_hash, name="output probability hash")
        if self.exact_baseline_fallback and (selected or output != baseline):
            raise ProtocolError("Exact-B fallback must preserve the protected bytes exactly.")
        if self.opposite_branch_preserved is not True:
            raise ProtocolError("Directional composition must preserve the opposite branch.")
        object.__setattr__(self, "decision_hash", decision)
        object.__setattr__(self, "baseline_probability_hash", baseline)
        object.__setattr__(self, "selected_probability_hashes", selected)
        object.__setattr__(self, "output_probability_hash", output)
        object.__setattr__(
            self,
            "composition_hash",
            canonical_hash(
                {
                    "schema_version": "compatibility_directional_composition_v1",
                    "decision_hash": decision,
                    "baseline_probability_hash": baseline,
                    "selected_probability_hashes": selected,
                    "output_probability_hash": output,
                    "exact_baseline_fallback": self.exact_baseline_fallback,
                    "opposite_branch_preserved": True,
                    "anchor": "B",
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CompositionResult:
    sample_ids: tuple[str, ...]
    output_probability_bytes: tuple[bytes, ...]
    receipt: CompositionReceipt

    def __post_init__(self) -> None:
        samples = tuple(canonical_text(value, name="composition sample") for value in self.sample_ids)
        probabilities = canonical_probability_bytes(self.output_probability_bytes)
        if (
            len(samples) != len(probabilities)
            or len(set(samples)) != len(samples)
            or not isinstance(self.receipt, CompositionReceipt)
            or probability_bytes_hash(probabilities) != self.receipt.output_probability_hash
        ):
            raise ProtocolError("Composition result escaped its probability receipt.")
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "output_probability_bytes", probabilities)


__all__ = (
    "ALPHA_GRID",
    "ENDPOINTS",
    "TRAINING_SEEDS",
    "ActionKind",
    "ActionPrediction",
    "AdmissionThresholds",
    "BoundedActionEvidence",
    "CandidateFeatureVector",
    "CandidatePoolReceipt",
    "CompatibilityReceipt",
    "CompositionReceipt",
    "CompositionResult",
    "Direction",
    "EndpointBounds",
    "EndpointCalibration",
    "EndpointCalibrationCell",
    "EndpointEffects",
    "FoldLoss",
    "HurdlePairwiseModel",
    "LearnabilityAdmission",
    "OOFEndpointRow",
    "ReplicaEnergyInput",
    "RoutingDecision",
    "SourceActionObservation",
    "SourceOOFPrediction",
    "SourceAdmissionCandidate",
    "SourceAdmissionCase",
    "SupportPartitionReceipt",
    "TargetAction",
    "canonical_names_values",
    "canonical_probability_bytes",
    "canonical_text",
    "finite",
)
