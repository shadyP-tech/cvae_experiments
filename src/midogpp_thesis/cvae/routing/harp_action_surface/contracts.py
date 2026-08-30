"""Closed contracts for HARP probabilities, case ensembles, and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics

from ...protocol import ProtocolError
from ..harp_protocol.contracts import canonical_id, validate_hqe, validate_hqer
from ..harp_protocol.hashing import canonical_hash, require_sha256


ENSEMBLE_SEED_COUNT = 9
ACTION_LAMBDAS = (0.25, 0.5, 0.75, 1.0)
DIRECTIONS = ("D01", "D10", "ALL_MARGINS")
ACTION_FEATURE_NAMES = (
    "baseline_probability",
    "expert_probability",
    "action_probability",
    "baseline_margin",
    "expert_margin",
    "action_margin",
    "signed_expert_delta",
    "absolute_expert_delta",
    "signed_action_delta",
    "absolute_action_delta",
    "expert_hard_disagreement_fraction",
    "action_hard_flip_fraction",
    "seed_dispersion",
    "action_lambda",
)
RESPONSE_SEMANTICS = (
    "source_standardized_u_relative_hxe_predictive_ensemble_weighted_correctness_"
    "surrogate_and_proper_loss_deltas"
)


def outer_scoped_label_collection_hash(
    values: tuple[tuple[str, str], ...],
) -> str:
    """Hash only per-H label views; no H row may enter its own view hash."""

    pairs = tuple(
        (
            canonical_id(outer, name="outer target H"),
            require_sha256(value, name="outer-scoped label surface hash"),
        )
        for outer, value in values
    )
    if not pairs or pairs != tuple(sorted(set(pairs))):
        raise ProtocolError("HARP outer-scoped label hash collection drifted.")
    return canonical_hash(
        {
            "schema_version": (
                "midogpp_harp_outer_scoped_label_surface_collection_v1"
            ),
            "outer_label_surface_hashes": [list(value) for value in pairs],
            "outer_target_rows_excluded_before_hashing": True,
            "target_labels_used": False,
        }
    )


def _probability(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"HARP {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ProtocolError(f"HARP {name} must lie in [0,1].")
    return 0.0 if result == 0.0 else result


def _finite(value: object, *, name: str) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"HARP {name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"HARP {name} must be finite.")
    return 0.0 if result == 0.0 else result


def _roles(
    outer: object, query: object, source: object, donor: object | None
) -> tuple[str, str, str, str | None]:
    if donor is None:
        h, q, e = validate_hqe(
            outer_target=outer, pseudo_query=query, candidate_source=source
        )
        return h, q, e, None
    return validate_hqer(
        outer_target=outer,
        pseudo_query=query,
        candidate_source=source,
        inner_donor=donor,
    )


def _donor_key(value: str | None) -> str:
    return value or ""


def _direction(baseline: float, action: float) -> str:
    before, after = int(baseline >= 0.5), int(action >= 0.5)
    if (before, after) == (0, 1):
        return "D01"
    if (before, after) == (1, 0):
        return "D10"
    return "ALL_MARGINS"


@dataclass(frozen=True)
class HarpProbabilityRow:
    """One seed/sample cell; sealed and label-free but never model-feeding."""

    outer_target: str
    pseudo_query: str
    candidate_source: str
    inner_donor: str | None
    case_id: str
    sample_id: str
    seed_id: str
    baseline_probability: float
    expert_probability: float
    prediction_seal_hash: str
    label_free: bool = True
    model_feeding: bool = False
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h, q, e, r = _roles(
            self.outer_target, self.pseudo_query, self.candidate_source, self.inner_donor
        )
        case = canonical_id(self.case_id, name="case")
        sample = canonical_id(self.sample_id, name="sample")
        seed = canonical_id(self.seed_id, name="seed")
        baseline = _probability(self.baseline_probability, name="baseline probability")
        expert = _probability(self.expert_probability, name="expert probability")
        seal = require_sha256(self.prediction_seal_hash, name="prediction_seal_hash")
        if self.label_free is not True or self.model_feeding is not False:
            raise ProtocolError("HARP seed cells are label-free descriptive inputs only.")
        for name, value in (
            ("outer_target", h),
            ("pseudo_query", q),
            ("candidate_source", e),
            ("inner_donor", r),
            ("case_id", case),
            ("sample_id", sample),
            ("seed_id", seed),
            ("baseline_probability", baseline),
            ("expert_probability", expert),
            ("prediction_seal_hash", seal),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "row_hash", canonical_hash(self.to_payload()))

    @property
    def row_key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.candidate_source,
            _donor_key(self.inner_donor),
            self.case_id,
            self.sample_id,
            self.seed_id,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_harp_probability_seed_cell_v3",
            "outer_target": self.outer_target,
            "pseudo_query": self.pseudo_query,
            "candidate_source": self.candidate_source,
            "inner_donor": self.inner_donor,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "seed_id": self.seed_id,
            "baseline_probability": self.baseline_probability,
            "expert_probability": self.expert_probability,
            "prediction_seal_hash": self.prediction_seal_hash,
            "label_free": True,
            "model_feeding": False,
            "predictive_reference_action_id": "U",
            "candidate_physical_action_kind": "Hxe",
        }


@dataclass(frozen=True)
class HarpProbabilitySurface:
    rows: tuple[HarpProbabilityRow, ...]
    prediction_seal_hash: str
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        seal = require_sha256(self.prediction_seal_hash, name="prediction_seal_hash")
        if not rows or any(not isinstance(row, HarpProbabilityRow) for row in rows):
            raise ProtocolError("HARP probability surface requires typed seed cells.")
        if rows != tuple(sorted(rows, key=lambda row: row.row_key)) or len(
            {row.row_key for row in rows}
        ) != len(rows):
            raise ProtocolError("HARP probability seed cells are not canonical and unique.")
        if any(row.prediction_seal_hash != seal for row in rows):
            raise ProtocolError("HARP seed cells escaped their prediction seal.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "prediction_seal_hash", seal)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_probability_seed_surface_v3",
                    "prediction_seal_hash": seal,
                    "row_hashes": [row.row_hash for row in rows],
                    "seed_cells_model_feeding": False,
                    "label_free": True,
                    "predictive_reference_action_id": "U",
                }
            ),
        )


@dataclass(frozen=True)
class HarpProbabilityEnsembleRow:
    """One sample after exact-nine aggregation, bound to an equal-case receipt."""

    outer_target: str
    pseudo_query: str
    candidate_source: str
    inner_donor: str | None
    case_id: str
    sample_id: str
    case_sample_ids: tuple[str, ...]
    seed_ids: tuple[str, ...]
    baseline_member_probabilities: tuple[float, ...]
    expert_member_probabilities: tuple[float, ...]
    baseline_probability: float
    expert_probability: float
    seed_dispersion: float
    case_aggregation_receipt_hash: str
    prediction_seal_hash: str
    seed_count: int = ENSEMBLE_SEED_COUNT
    model_feeding: bool = True
    ensemble_receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h, q, e, r = _roles(
            self.outer_target, self.pseudo_query, self.candidate_source, self.inner_donor
        )
        case = canonical_id(self.case_id, name="case")
        sample = canonical_id(self.sample_id, name="sample")
        samples = tuple(canonical_id(value, name="case sample") for value in self.case_sample_ids)
        seeds = tuple(canonical_id(value, name="seed") for value in self.seed_ids)
        if not samples or samples != tuple(sorted(set(samples))) or sample not in samples:
            raise ProtocolError("HARP sample must belong to its canonical equal-case inventory.")
        if len(seeds) != ENSEMBLE_SEED_COUNT or seeds != tuple(sorted(set(seeds))):
            raise ProtocolError("HARP model rows require the exact-nine canonical seed inventory.")
        baseline_members = tuple(
            _probability(value, name="baseline member")
            for value in self.baseline_member_probabilities
        )
        expert_members = tuple(
            _probability(value, name="expert member")
            for value in self.expert_member_probabilities
        )
        if len(baseline_members) != ENSEMBLE_SEED_COUNT or len(expert_members) != ENSEMBLE_SEED_COUNT:
            raise ProtocolError("HARP probability ensemble must contain exactly nine members.")
        baseline = _probability(self.baseline_probability, name="baseline ensemble")
        expert = _probability(self.expert_probability, name="expert ensemble")
        dispersion = statistics.pstdev(
            tuple(
                right - left
                for left, right in zip(baseline_members, expert_members, strict=True)
            )
        )
        if not math.isclose(baseline, statistics.fmean(baseline_members), abs_tol=1e-15) or not math.isclose(
            expert, statistics.fmean(expert_members), abs_tol=1e-15
        ):
            raise ProtocolError("HARP ensemble means drifted from exact-nine members.")
        if not math.isclose(
            _finite(self.seed_dispersion, name="seed dispersion"), dispersion, abs_tol=1e-15
        ):
            raise ProtocolError("HARP seed dispersion drifted from exact-nine members.")
        if self.seed_count != ENSEMBLE_SEED_COUNT or self.model_feeding is not True:
            raise ProtocolError("Only exact-nine case ensembles may feed HARP models.")
        for name in ("case_aggregation_receipt_hash", "prediction_seal_hash"):
            require_sha256(getattr(self, name), name=name)
        for name, value in (
            ("outer_target", h),
            ("pseudo_query", q),
            ("candidate_source", e),
            ("inner_donor", r),
            ("case_id", case),
            ("sample_id", sample),
            ("case_sample_ids", samples),
            ("seed_ids", seeds),
            ("baseline_member_probabilities", baseline_members),
            ("expert_member_probabilities", expert_members),
            ("baseline_probability", baseline),
            ("expert_probability", expert),
            ("seed_dispersion", dispersion),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "ensemble_receipt_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_exact_nine_case_ensemble_v2",
                    "row_key": list(self.row_key),
                    "sample_id": sample,
                    "case_sample_ids": list(samples),
                    "seed_ids": list(seeds),
                    "baseline_member_probabilities": list(baseline_members),
                    "expert_member_probabilities": list(expert_members),
                    "case_aggregation_receipt_hash": self.case_aggregation_receipt_hash,
                    "prediction_seal_hash": self.prediction_seal_hash,
                    "seed_count": ENSEMBLE_SEED_COUNT,
                    "model_feeding": True,
                    "model_observation_unit": "sample_with_equal_case_total_mass",
                    "predictive_reference_action_id": "U",
                }
            ),
        )

    @property
    def row_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.candidate_source,
            _donor_key(self.inner_donor),
            self.case_id,
            self.sample_id,
        )


@dataclass(frozen=True)
class HarpProbabilityEnsembleSurface:
    rows: tuple[HarpProbabilityEnsembleRow, ...]
    seed_surface_hash: str
    expected_seed_ids: tuple[str, ...]
    prediction_seal_hash: str
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        seeds = tuple(self.expected_seed_ids)
        for name in ("seed_surface_hash", "prediction_seal_hash"):
            require_sha256(getattr(self, name), name=name)
        if len(seeds) != ENSEMBLE_SEED_COUNT or seeds != tuple(sorted(set(seeds))):
            raise ProtocolError("HARP ensemble surface requires one exact-nine seed inventory.")
        if not rows or any(not isinstance(row, HarpProbabilityEnsembleRow) for row in rows):
            raise ProtocolError("HARP ensemble surface requires typed case rows.")
        if rows != tuple(sorted(rows, key=lambda row: row.row_key)) or len(
            {row.row_key for row in rows}
        ) != len(rows):
            raise ProtocolError("HARP ensemble case rows are not canonical and unique.")
        if any(
            row.seed_ids != seeds or row.prediction_seal_hash != self.prediction_seal_hash
            for row in rows
        ):
            raise ProtocolError("HARP ensemble row escaped seed/seal coverage.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "expected_seed_ids", seeds)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_probability_ensemble_surface_v2",
                    "seed_surface_hash": self.seed_surface_hash,
                    "expected_seed_ids": list(seeds),
                    "prediction_seal_hash": self.prediction_seal_hash,
                    "ensemble_receipt_hashes": [row.ensemble_receipt_hash for row in rows],
                    "model_observation_unit": "sample_with_equal_case_total_mass",
                    "seed_count": ENSEMBLE_SEED_COUNT,
                    "predictive_reference_action_id": "U",
                }
            ),
        )


@dataclass(frozen=True)
class HarpActionFeatureRow:
    outer_target: str
    pseudo_query: str
    candidate_source: str
    inner_donor: str | None
    case_id: str
    sample_id: str
    case_sample_ids: tuple[str, ...]
    action_lambda: float
    direction: str
    baseline_probability: float
    expert_probability: float
    action_probability: float
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]
    ensemble_receipt_hash: str
    case_aggregation_receipt_hash: str
    prediction_seal_hash: str
    seed_count: int = ENSEMBLE_SEED_COUNT
    label_free: bool = True
    feature_hash: str = field(init=False)

    def __post_init__(self) -> None:
        h, q, e, r = _roles(
            self.outer_target, self.pseudo_query, self.candidate_source, self.inner_donor
        )
        case = canonical_id(self.case_id, name="case")
        sample = canonical_id(self.sample_id, name="sample")
        samples = tuple(canonical_id(value, name="case sample") for value in self.case_sample_ids)
        if not samples or samples != tuple(sorted(set(samples))) or sample not in samples:
            raise ProtocolError("HARP feature sample escaped its equal-case inventory.")
        lam = _finite(self.action_lambda, name="action lambda")
        if lam not in ACTION_LAMBDAS:
            raise ProtocolError("HARP action lambda is outside the locked portfolio.")
        baseline = _probability(self.baseline_probability, name="baseline probability")
        expert = _probability(self.expert_probability, name="expert probability")
        action = _probability(self.action_probability, name="action probability")
        expected_action = expert if lam == 1.0 else (1.0 - lam) * baseline + lam * expert
        if not math.isclose(action, expected_action, abs_tol=1e-15):
            raise ProtocolError(
                "HARP action probability drifted from the post-classifier predictive ensemble."
            )
        if lam == 1.0 and action != expert:
            raise ProtocolError("HARP lambda=1 must equal the physical Hxe endpoint exactly.")
        if self.direction != _direction(baseline, action):
            raise ProtocolError("HARP direction drifted from aggregate hard predictions.")
        names = tuple(self.feature_names)
        values = tuple(_finite(value, name="feature value") for value in self.feature_values)
        if names != ACTION_FEATURE_NAMES or len(values) != len(names):
            raise ProtocolError("HARP action feature schema is not closed-world.")
        if self.seed_count != ENSEMBLE_SEED_COUNT or self.label_free is not True:
            raise ProtocolError("HARP model features require label-free exact-nine ensembles.")
        for name in (
            "ensemble_receipt_hash",
            "case_aggregation_receipt_hash",
            "prediction_seal_hash",
        ):
            require_sha256(getattr(self, name), name=name)
        for name, value in (
            ("outer_target", h),
            ("pseudo_query", q),
            ("candidate_source", e),
            ("inner_donor", r),
            ("case_id", case),
            ("sample_id", sample),
            ("case_sample_ids", samples),
            ("action_lambda", lam),
            ("baseline_probability", baseline),
            ("expert_probability", expert),
            ("action_probability", action),
            ("feature_names", names),
            ("feature_values", values),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "feature_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_case_action_feature_v3",
                    "row_key": list(self.row_key),
                    "sample_id": sample,
                    "case_sample_ids": list(samples),
                    "direction": self.direction,
                    "probabilities": [baseline, expert, action],
                    "feature_names": list(names),
                    "feature_values": list(values),
                    "ensemble_receipt_hash": self.ensemble_receipt_hash,
                    "case_aggregation_receipt_hash": self.case_aggregation_receipt_hash,
                    "prediction_seal_hash": self.prediction_seal_hash,
                    "seed_count": ENSEMBLE_SEED_COUNT,
                    "label_free": True,
                    "predictive_reference_action_id": "U",
                    "lambda_one_is_physical_hxe_endpoint": True,
                }
            ),
        )

    @property
    def row_key(self) -> tuple[str, str, str, str, str, str, float]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.candidate_source,
            _donor_key(self.inner_donor),
            self.case_id,
            self.sample_id,
            self.action_lambda,
        )

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Current model row only; sibling samples are bound by the case receipt."""

        return (self.sample_id,)


@dataclass(frozen=True)
class HarpActionFeatureSurface:
    rows: tuple[HarpActionFeatureRow, ...]
    ensemble_surface_hash: str
    prediction_seal_hash: str
    surface_hash: str = field(init=False)

    @property
    def probability_surface_hash(self) -> str:
        return self.ensemble_surface_hash

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        for name in ("ensemble_surface_hash", "prediction_seal_hash"):
            require_sha256(getattr(self, name), name=name)
        if not rows or any(not isinstance(row, HarpActionFeatureRow) for row in rows):
            raise ProtocolError("HARP feature surface requires typed case rows.")
        if rows != tuple(sorted(rows, key=lambda row: row.row_key)) or len(
            {row.row_key for row in rows}
        ) != len(rows):
            raise ProtocolError("HARP case features are not canonical and unique.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_action_feature_surface_v3",
                    "ensemble_surface_hash": self.ensemble_surface_hash,
                    "prediction_seal_hash": self.prediction_seal_hash,
                    "feature_names": list(ACTION_FEATURE_NAMES),
                    "action_lambdas": list(ACTION_LAMBDAS),
                    "feature_hashes": [row.feature_hash for row in rows],
                    "model_observation_unit": "sample_with_equal_case_total_mass",
                    "seed_cells_model_feeding": False,
                    "predictive_reference_action_id": "U",
                    "probability_ensemble_semantics": "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe",
                }
            ),
        )


@dataclass(frozen=True)
class HarpDisagreementRow:
    outer_target: str
    pseudo_query: str
    candidate_source: str
    inner_donor: str | None
    case_id: str
    sample_id: str
    action_lambda: float
    direction: str
    ensemble_receipt_hash: str
    feature_hash: str
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _roles(self.outer_target, self.pseudo_query, self.candidate_source, self.inner_donor)
        canonical_id(self.case_id, name="case")
        canonical_id(self.sample_id, name="sample")
        if self.direction not in ("D01", "D10") or self.action_lambda not in ACTION_LAMBDAS:
            raise ProtocolError("HARP disagreement rows require a locked hard-flip action.")
        require_sha256(self.ensemble_receipt_hash, name="ensemble_receipt_hash")
        require_sha256(self.feature_hash, name="feature_hash")
        object.__setattr__(
            self,
            "row_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_case_disagreement_v2",
                    "row_key": list(self.row_key),
                    "direction": self.direction,
                    "ensemble_receipt_hash": self.ensemble_receipt_hash,
                    "feature_hash": self.feature_hash,
                }
            ),
        )

    @property
    def row_key(self) -> tuple[str, str, str, str, str, str, float]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.candidate_source,
            _donor_key(self.inner_donor),
            self.case_id,
            self.sample_id,
            self.action_lambda,
        )


@dataclass(frozen=True)
class SourceClassPriorReceipt:
    outer_target: str
    pseudo_query: str
    positive_case_count: int
    negative_case_count: int
    positive_weight: float
    negative_weight: float
    case_sample_counts: tuple[tuple[str, int], ...]
    case_class_sample_counts: tuple[tuple[str, int, int], ...]
    label_surface_hash: str
    receipt_hash: str = field(init=False)

    @property
    def positive_count(self) -> int:
        return self.positive_case_count

    @property
    def negative_count(self) -> int:
        return self.negative_case_count

    def __post_init__(self) -> None:
        h = canonical_id(self.outer_target, name="outer H")
        q = canonical_id(self.pseudo_query, name="query q")
        case_counts = tuple(
            (canonical_id(case, name="source case"), int(count))
            for case, count in self.case_sample_counts
        )
        class_counts = tuple(
            (canonical_id(case, name="source case"), int(label), int(count))
            for case, label, count in self.case_class_sample_counts
        )
        if h == q or any(
            type(value) is not int or value <= 0
            for value in (self.positive_case_count, self.negative_case_count)
        ) or (
            not case_counts
            or case_counts != tuple(sorted(case_counts))
            or len({case for case, _ in case_counts}) != len(case_counts)
            or any(count <= 0 for _, count in case_counts)
            or not class_counts
            or class_counts != tuple(sorted(class_counts))
            or len({(case, label) for case, label, _ in class_counts}) != len(class_counts)
            or any(label not in (0, 1) or count <= 0 for _, label, count in class_counts)
        ):
            raise ProtocolError("HARP source case denominators require H != q and both classes.")
        total_cases = len(case_counts)
        observed_positive_cases = len({case for case, label, _ in class_counts if label == 1})
        observed_negative_cases = len({case for case, label, _ in class_counts if label == 0})
        if (
            observed_positive_cases != self.positive_case_count
            or observed_negative_cases != self.negative_case_count
            or {
                case: sum(count for row_case, _, count in class_counts if row_case == case)
                for case, _ in case_counts
            }
            != dict(case_counts)
        ):
            raise ProtocolError("HARP source case/class sample denominators drifted.")
        positive_weight = total_cases / (2.0 * self.positive_case_count)
        negative_weight = total_cases / (2.0 * self.negative_case_count)
        if not math.isclose(self.positive_weight, positive_weight, abs_tol=1e-15) or not math.isclose(
            self.negative_weight, negative_weight, abs_tol=1e-15
        ):
            raise ProtocolError("HARP source case weights drifted from denominators.")
        require_sha256(self.label_surface_hash, name="label_surface_hash")
        object.__setattr__(self, "case_sample_counts", case_counts)
        object.__setattr__(self, "case_class_sample_counts", class_counts)
        object.__setattr__(
            self,
            "receipt_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_source_case_prior_receipt_v4",
                    "outer_target": h,
                    "pseudo_query": q,
                    "positive_case_count": self.positive_case_count,
                    "negative_case_count": self.negative_case_count,
                    "positive_weight": positive_weight,
                    "negative_weight": negative_weight,
                    "total_case_count": total_cases,
                    "case_sample_counts": [list(value) for value in case_counts],
                    "case_class_sample_counts": [list(value) for value in class_counts],
                    "label_surface_hash": self.label_surface_hash,
                    "independent_case_denominators": True,
                    "mixed_label_cases_supported": True,
                    "estimand": "case_equal_balanced_accuracy_delta",
                    "outer_h_excluded_before_counting": True,
                }
            ),
        )


@dataclass(frozen=True)
class HarpDirectionalResponseRow:
    outer_target: str
    pseudo_query: str
    candidate_source: str
    inner_donor: str | None
    case_id: str
    sample_id: str
    action_lambda: float
    direction: str
    truth_class: int
    weighted_correctness_surrogate: float
    brier_delta: float
    log_loss_delta: float
    denominator_receipt_hash: str
    ensemble_receipt_hash: str
    case_aggregation_receipt_hash: str
    feature_hash: str
    label_surface_hash: str
    seed_count: int = ENSEMBLE_SEED_COUNT
    response_semantics: str = RESPONSE_SEMANTICS
    response_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _roles(self.outer_target, self.pseudo_query, self.candidate_source, self.inner_donor)
        canonical_id(self.case_id, name="case")
        canonical_id(self.sample_id, name="sample")
        if self.action_lambda not in ACTION_LAMBDAS or self.direction not in DIRECTIONS:
            raise ProtocolError("HARP response action identity escaped its schema.")
        if type(self.truth_class) is not int or self.truth_class not in (0, 1):
            raise ProtocolError("HARP response truth class must be binary.")
        weighted = _finite(
            self.weighted_correctness_surrogate, name="weighted-correctness surrogate"
        )
        brier = _finite(self.brier_delta, name="Brier delta")
        log_loss = _finite(self.log_loss_delta, name="log-loss delta")
        if not -1.0 <= brier <= 1.0 or self.seed_count != ENSEMBLE_SEED_COUNT:
            raise ProtocolError("HARP response metric/seed contract drifted.")
        for name in (
            "denominator_receipt_hash",
            "ensemble_receipt_hash",
            "case_aggregation_receipt_hash",
            "feature_hash",
            "label_surface_hash",
        ):
            require_sha256(getattr(self, name), name=name)
        if self.response_semantics != RESPONSE_SEMANTICS:
            raise ProtocolError("HARP response semantics drifted.")
        object.__setattr__(self, "weighted_correctness_surrogate", weighted)
        object.__setattr__(self, "brier_delta", brier)
        object.__setattr__(self, "log_loss_delta", log_loss)
        object.__setattr__(
            self,
            "response_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_case_directional_response_v3",
                    "row_key": list(self.row_key),
                    "direction": self.direction,
                    "truth_class": self.truth_class,
                    "weighted_correctness_surrogate": weighted,
                    "brier_delta": brier,
                    "log_loss_delta": log_loss,
                    "denominator_receipt_hash": self.denominator_receipt_hash,
                    "ensemble_receipt_hash": self.ensemble_receipt_hash,
                    "case_aggregation_receipt_hash": self.case_aggregation_receipt_hash,
                    "feature_hash": self.feature_hash,
                    "label_surface_hash": self.label_surface_hash,
                    "seed_count": ENSEMBLE_SEED_COUNT,
                    "response_semantics": self.response_semantics,
                }
            ),
        )

    @property
    def row_key(self) -> tuple[str, str, str, str, str, str, float]:
        return (
            self.outer_target,
            self.pseudo_query,
            self.candidate_source,
            _donor_key(self.inner_donor),
            self.case_id,
            self.sample_id,
            self.action_lambda,
        )

    @property
    def class_prior_receipt_hash(self) -> str:
        return self.denominator_receipt_hash


@dataclass(frozen=True)
class HarpDirectionalResponseSurface:
    rows: tuple[HarpDirectionalResponseRow, ...]
    feature_surface_hash: str
    label_surface_hash: str
    receipts: tuple[SourceClassPriorReceipt, ...]
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        receipts = tuple(self.receipts)
        for name in ("feature_surface_hash", "label_surface_hash"):
            require_sha256(getattr(self, name), name=name)
        if not rows or rows != tuple(sorted(rows, key=lambda row: row.row_key)) or len(
            {row.row_key for row in rows}
        ) != len(rows):
            raise ProtocolError("HARP response rows are not canonical sample actions.")
        if not receipts or receipts != tuple(
            sorted(receipts, key=lambda item: (item.outer_target, item.pseudo_query))
        ):
            raise ProtocolError("HARP denominator receipts are not canonical.")
        receipt_hashes = {item.receipt_hash for item in receipts}
        outer_label_hashes: dict[str, str] = {}
        for receipt in receipts:
            previous = outer_label_hashes.setdefault(
                receipt.outer_target, receipt.label_surface_hash
            )
            if previous != receipt.label_surface_hash:
                raise ProtocolError(
                    "HARP one outer target crossed scoped label surfaces."
                )
        expected_label_collection_hash = outer_scoped_label_collection_hash(
            tuple(sorted(outer_label_hashes.items()))
        )
        if any(
            row.denominator_receipt_hash not in receipt_hashes
            or row.label_surface_hash
            != outer_label_hashes.get(row.outer_target)
            for row in rows
        ) or self.label_surface_hash != expected_label_collection_hash:
            raise ProtocolError("HARP response escaped its source-only receipts.")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(
            self,
            "surface_hash",
            canonical_hash(
                {
                    "schema_version": "midogpp_harp_case_response_surface_v4",
                    "feature_surface_hash": self.feature_surface_hash,
                    "label_surface_hash": self.label_surface_hash,
                    "receipt_hashes": [item.receipt_hash for item in receipts],
                    "response_hashes": [item.response_hash for item in rows],
                    "model_observation_unit": "sample_with_equal_case_total_mass",
                    "seed_cells_model_feeding": False,
                    "response_semantics": RESPONSE_SEMANTICS,
                    "target_labels_used": False,
                    "response_reference_action_id": "U",
                }
            ),
        )


def action_feature_values(
    ensemble: HarpProbabilityEnsembleRow, lam: float
) -> tuple[float, ...]:
    baseline, expert = ensemble.baseline_probability, ensemble.expert_probability
    action = expert if lam == 1.0 else (1.0 - lam) * baseline + lam * expert
    expert_flips = statistics.fmean(
        float(int(expert_member >= 0.5) != int(baseline_member >= 0.5))
        for baseline_member, expert_member in zip(
            ensemble.baseline_member_probabilities,
            ensemble.expert_member_probabilities,
            strict=True,
        )
    )
    action_flips = statistics.fmean(
        float(
            int(((1.0 - lam) * baseline_member + lam * expert_member) >= 0.5)
            != int(baseline_member >= 0.5)
        )
        for baseline_member, expert_member in zip(
            ensemble.baseline_member_probabilities,
            ensemble.expert_member_probabilities,
            strict=True,
        )
    )
    return (
        baseline,
        expert,
        action,
        abs(baseline - 0.5),
        abs(expert - 0.5),
        abs(action - 0.5),
        expert - baseline,
        abs(expert - baseline),
        action - baseline,
        abs(action - baseline),
        expert_flips,
        action_flips,
        ensemble.seed_dispersion,
        lam,
    )


__all__ = (
    "ACTION_FEATURE_NAMES",
    "ACTION_LAMBDAS",
    "DIRECTIONS",
    "ENSEMBLE_SEED_COUNT",
    "RESPONSE_SEMANTICS",
    "HarpActionFeatureRow",
    "HarpActionFeatureSurface",
    "HarpDirectionalResponseRow",
    "HarpDirectionalResponseSurface",
    "HarpDisagreementRow",
    "HarpProbabilityEnsembleRow",
    "HarpProbabilityEnsembleSurface",
    "HarpProbabilityRow",
    "HarpProbabilitySurface",
    "SourceClassPriorReceipt",
    "action_feature_values",
    "outer_scoped_label_collection_hash",
)
