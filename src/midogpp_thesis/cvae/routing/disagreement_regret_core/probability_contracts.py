"""Label-free probability and feature contracts for disagreement regret."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from ...protocol import ProtocolError
from ._validation import _canonical_id, _finite_probability
from .hashing import canonical_sha256, is_sha256


FEATURE_NAMES = (
    "disagreement_rate",
    "positive_flip_rate",
    "negative_flip_rate",
    "signed_logit_delta_mean",
    "absolute_logit_delta_mean",
    "signed_logit_delta_on_flip_mean",
    "absolute_logit_delta_on_flip_mean",
    "control_margin_on_flip_mean",
    "action_margin_on_flip_mean",
    "baseline_margin_mean",
    "candidate_rank_fraction_mean",
    "candidate_gap_from_best_mean",
    "action_probability_sd_mean",
    "control_probability_sd_mean",
    "hard_vote_fraction_mean",
)

DEVELOPMENT_COMPOSITE_SURFACE_ROLE = "development_composite"
SOURCE_OOF_TRAINING_SURFACE_ROLE = "source_oof_training_only"
LABEL_FREE_INFERENCE_SURFACE_ROLE = "label_free_inference_only"
_SURFACE_ROLES = (
    DEVELOPMENT_COMPOSITE_SURFACE_ROLE,
    SOURCE_OOF_TRAINING_SURFACE_ROLE,
    LABEL_FREE_INFERENCE_SURFACE_ROLE,
)


@dataclass(frozen=True)
class ProbabilityRow:
    """One label-free mean prediction from a sealed action surface."""

    query_id: str
    case_id: str
    sample_id: str
    action_id: str
    source_id: str | None
    probability: float
    probability_sd: float
    hard_vote_fraction: float
    prediction_seal_hash: str
    label_free: bool = True

    def __post_init__(self) -> None:
        for name in ("query_id", "case_id", "sample_id", "action_id"):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        if self.source_id is not None:
            object.__setattr__(
                self, "source_id", _canonical_id(self.source_id, name="source_id")
            )
        object.__setattr__(
            self,
            "probability",
            _finite_probability(self.probability, name="probability"),
        )
        deviation = float(self.probability_sd)
        if not math.isfinite(deviation) or deviation < 0.0:
            raise ProtocolError("probability_sd must be finite and nonnegative.")
        object.__setattr__(self, "probability_sd", deviation)
        vote = _finite_probability(self.hard_vote_fraction, name="hard_vote_fraction")
        if vote < 0.5:
            raise ProtocolError("hard_vote_fraction must describe the winning hard vote.")
        object.__setattr__(self, "hard_vote_fraction", vote)
        if not is_sha256(self.prediction_seal_hash):
            raise ProtocolError("prediction_seal_hash must be a lowercase SHA-256.")
        if self.label_free is not True:
            raise ProtocolError("Probability rows must be label-free.")

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return (self.query_id, self.case_id, self.sample_id, self.action_id)

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.query_id, self.case_id, self.sample_id)


@dataclass(frozen=True)
class SourceOOFLabelRow:
    """Synthetic or separately authorized source-OOF training outcome."""

    query_id: str
    case_id: str
    sample_id: str
    label: int
    role: str = "source_oof_training_only"

    def __post_init__(self) -> None:
        for name in ("query_id", "case_id", "sample_id"):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        if type(self.label) is not int or self.label not in (0, 1):
            raise ProtocolError("Source-OOF labels must be binary integers.")
        object.__setattr__(self, "label", int(self.label))
        if self.role != "source_oof_training_only":
            raise ProtocolError("Target/support/evaluation labels are forbidden in this core.")

    @property
    def row_key(self) -> tuple[str, str, str]:
        return (self.query_id, self.case_id, self.sample_id)


@dataclass(frozen=True)
class DisagreementRow:
    query_id: str
    case_id: str
    sample_id: str
    action_id: str
    source_id: str | None
    flip_direction: int
    action_probability: float
    control_probability: float
    baseline_probability: float
    signed_logit_delta: float
    action_margin: float
    control_margin: float
    candidate_rank_fraction: float
    candidate_gap_from_best: float
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("query_id", "case_id", "sample_id", "action_id"):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        if self.source_id is not None:
            object.__setattr__(
                self, "source_id", _canonical_id(self.source_id, name="source_id")
            )
        if type(self.flip_direction) is not int or self.flip_direction not in (-1, 1):
            raise ProtocolError("Sparse disagreement rows require a nonzero hard flip.")
        numeric = (
            self.action_probability,
            self.control_probability,
            self.baseline_probability,
            self.signed_logit_delta,
            self.action_margin,
            self.control_margin,
            self.candidate_rank_fraction,
            self.candidate_gap_from_best,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ProtocolError("Disagreement rows must be finite.")
        for name in ("action_probability", "control_probability", "baseline_probability"):
            _finite_probability(getattr(self, name), name=name)
        if not 0.0 <= self.action_margin <= 0.5 or not 0.0 <= self.control_margin <= 0.5:
            raise ProtocolError("Disagreement margins must lie in [0, 0.5].")
        if not 0.0 <= self.candidate_rank_fraction <= 1.0:
            raise ProtocolError("Candidate rank fraction must lie in [0, 1].")
        if self.candidate_gap_from_best < 0.0:
            raise ProtocolError("Candidate gap from best cannot be negative.")
        expected_direction = int(self.action_probability >= 0.5) - int(
            self.control_probability >= 0.5
        )
        if expected_direction != self.flip_direction:
            raise ProtocolError("Disagreement direction drifted from hard predictions.")
        if not math.isclose(
            self.action_margin,
            abs(self.action_probability - 0.5),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ) or not math.isclose(
            self.control_margin,
            abs(self.control_probability - 0.5),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ProtocolError("Disagreement margins drifted from probabilities.")
        epsilon = 1.0e-6
        action = min(max(self.action_probability, epsilon), 1.0 - epsilon)
        control = min(max(self.control_probability, epsilon), 1.0 - epsilon)
        expected_logit_delta = math.log(action / (1.0 - action)) - math.log(
            control / (1.0 - control)
        )
        if not math.isclose(
            self.signed_logit_delta,
            expected_logit_delta,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ProtocolError("Disagreement logit delta drifted from probabilities.")
        object.__setattr__(
            self,
            "row_hash",
            canonical_sha256(
                {
                    "schema_version": "midogpp_disagreement_sample_row_v1",
                    "query_id": self.query_id,
                    "case_id": self.case_id,
                    "sample_id": self.sample_id,
                    "action_id": self.action_id,
                    "source_id": self.source_id,
                    "flip_direction": self.flip_direction,
                    "action_probability": self.action_probability,
                    "control_probability": self.control_probability,
                    "baseline_probability": self.baseline_probability,
                    "signed_logit_delta": self.signed_logit_delta,
                    "action_margin": self.action_margin,
                    "control_margin": self.control_margin,
                    "candidate_rank_fraction": self.candidate_rank_fraction,
                    "candidate_gap_from_best": self.candidate_gap_from_best,
                }
            ),
        )

    @property
    def row_key(self) -> tuple[str, str, str, str]:
        return (self.query_id, self.case_id, self.sample_id, self.action_id)

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return (self.query_id, self.case_id, self.sample_id)


@dataclass(frozen=True)
class CaseActionFeatureRow:
    query_id: str
    case_id: str
    action_id: str
    source_id: str | None
    values: tuple[float, ...]
    sample_count: int
    disagreement_count: int
    prediction_seal_hash: str
    feature_origin_action_id: str | None = None
    label_free: bool = True
    feature_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("query_id", "case_id", "action_id"):
            object.__setattr__(self, name, _canonical_id(getattr(self, name), name=name))
        if self.source_id is not None:
            object.__setattr__(
                self, "source_id", _canonical_id(self.source_id, name="source_id")
            )
        values = tuple(float(value) for value in self.values)
        if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
            raise ProtocolError("Case-action feature vectors must be complete and finite.")
        object.__setattr__(self, "values", values)
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ProtocolError("sample_count must be a positive integer.")
        if (
            type(self.disagreement_count) is not int
            or not 0 <= self.disagreement_count <= self.sample_count
        ):
            raise ProtocolError("disagreement_count must lie within sample_count.")
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "disagreement_count", int(self.disagreement_count))
        if not is_sha256(self.prediction_seal_hash):
            raise ProtocolError("Feature rows require a sealed prediction surface.")
        origin = self.action_id if self.feature_origin_action_id is None else _canonical_id(
            self.feature_origin_action_id, name="feature_origin_action_id"
        )
        object.__setattr__(self, "feature_origin_action_id", origin)
        if self.label_free is not True:
            raise ProtocolError("Case-action features must be label-free.")
        payload = self._unhashed_payload()
        object.__setattr__(self, "feature_hash", canonical_sha256(payload))

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_disagreement_regret_case_feature_v1",
            "query_id": self.query_id,
            "case_id": self.case_id,
            "action_id": self.action_id,
            "source_id": self.source_id,
            "feature_names": list(FEATURE_NAMES),
            "values": list(self.values),
            "sample_count": self.sample_count,
            "disagreement_count": self.disagreement_count,
            "prediction_seal_hash": self.prediction_seal_hash,
            "feature_origin_action_id": self.feature_origin_action_id,
            "label_free": True,
        }

    @property
    def row_key(self) -> tuple[str, str, str]:
        return (self.query_id, self.case_id, self.action_id)


@dataclass(frozen=True)
class DisagreementFeatureSurface:
    rows: tuple[CaseActionFeatureRow, ...]
    disagreements: tuple[DisagreementRow, ...]
    baseline_action_id: str
    control_action_id: str
    candidate_source_by_action: Mapping[str, str]
    prediction_seal_hash: str
    sample_keys: tuple[tuple[str, str, str], ...]
    development_context_hash: str
    dataset_family: str
    outer_target_id: str
    surface_role: str = DEVELOPMENT_COMPOSITE_SURFACE_ROLE
    family: str = "R"
    parent_surface_hash: str | None = None
    surface_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        disagreements = tuple(self.disagreements)
        if not rows:
            raise ProtocolError("A disagreement feature surface cannot be empty.")
        if any(not isinstance(row, CaseActionFeatureRow) for row in rows):
            raise ProtocolError("Feature surfaces require typed rows.")
        if any(not isinstance(row, DisagreementRow) for row in disagreements):
            raise ProtocolError("Feature surfaces require typed disagreement rows.")
        if len({row.row_key for row in rows}) != len(rows):
            raise ProtocolError("Feature surfaces contain duplicate case-action rows.")
        if tuple(sorted(rows, key=lambda row: row.row_key)) != rows:
            raise ProtocolError("Feature surface rows must use canonical ordering.")
        if tuple(sorted(disagreements, key=lambda row: row.row_key)) != disagreements:
            raise ProtocolError("Disagreement rows must use canonical ordering.")
        if len({row.row_key for row in disagreements}) != len(disagreements):
            raise ProtocolError("Disagreement surfaces contain duplicate rows.")
        sample_keys = tuple(
            (
                _canonical_id(query, name="sample query_id"),
                _canonical_id(case, name="sample case_id"),
                _canonical_id(sample, name="sample_id"),
            )
            for query, case, sample in self.sample_keys
        )
        if not sample_keys or sample_keys != tuple(sorted(sample_keys)):
            raise ProtocolError("Feature sample keys must be nonempty and canonically ordered.")
        if len(set(sample_keys)) != len(sample_keys):
            raise ProtocolError("Feature sample keys contain duplicates.")
        baseline = _canonical_id(self.baseline_action_id, name="baseline_action_id")
        control = _canonical_id(self.control_action_id, name="control_action_id")
        if baseline == control:
            raise ProtocolError("Baseline and control actions must differ.")
        mapping = {
            _canonical_id(action, name="candidate action"): _canonical_id(
                source, name="candidate source"
            )
            for action, source in self.candidate_source_by_action.items()
        }
        if not mapping or len(set(mapping.values())) != len(mapping):
            raise ProtocolError("Candidate actions require unique known-bank source IDs.")
        if baseline in mapping or control in mapping:
            raise ProtocolError("B/U controls cannot be candidate-source actions.")
        if not is_sha256(self.prediction_seal_hash):
            raise ProtocolError("Feature surface prediction seal must be SHA-256.")
        if not is_sha256(self.development_context_hash):
            raise ProtocolError("Feature surface context identity must be SHA-256.")
        dataset_family = _canonical_id(self.dataset_family, name="dataset_family")
        outer_target = _canonical_id(self.outer_target_id, name="outer_target_id")
        if self.surface_role not in _SURFACE_ROLES:
            raise ProtocolError("Feature surface role drifted.")
        if self.family not in ("G", "R", "P"):
            raise ProtocolError("Feature surface family must be G, R, or P.")
        if self.family == "R":
            if self.parent_surface_hash is not None:
                raise ProtocolError("Aligned R is the parent feature surface.")
        elif not is_sha256(self.parent_surface_hash):
            raise ProtocolError("G/P controls require their aligned parent surface hash.")
        if self.family != "R" and disagreements:
            raise ProtocolError("G/P controls cannot carry aligned sparse disagreement rows.")
        if self.family == "G" and any(
            row.disagreement_count != 0
            or row.feature_origin_action_id != row.action_id
            or any(value != 0.0 for value in row.values)
            for row in rows
        ):
            raise ProtocolError("G controls must contain exact zero, self-origin features.")
        if {row.prediction_seal_hash for row in rows} != {self.prediction_seal_hash}:
            raise ProtocolError("Feature rows drifted from the prediction seal.")
        count_by_case: dict[tuple[str, str], int] = {}
        for query, case, _sample in sample_keys:
            count_by_case[(query, case)] = count_by_case.get((query, case), 0) + 1
        feature_cases = {(row.query_id, row.case_id) for row in rows}
        if set(count_by_case) != feature_cases:
            raise ProtocolError("Feature sample identity and case rows are misaligned.")
        actions_by_case: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            actions_by_case.setdefault((row.query_id, row.case_id), set()).add(
                row.action_id
            )
        for (query_id, _case_id), observed_actions in actions_by_case.items():
            expected_actions = {
                baseline,
                *(
                    action_id
                    for action_id, source_id in mapping.items()
                    if source_id != query_id
                ),
            }
            if observed_actions != expected_actions:
                raise ProtocolError(
                    "Every feature case requires B plus every legal non-query candidate."
                )
        if any(
            row.sample_count != count_by_case[(row.query_id, row.case_id)] for row in rows
        ):
            raise ProtocolError("Feature sample_count drifted from exact sample identities.")
        query_ids = {row.query_id for row in rows}
        if self.surface_role == SOURCE_OOF_TRAINING_SURFACE_ROLE:
            if outer_target in query_ids:
                raise ProtocolError("Source-OOF training surfaces cannot contain target rows.")
        elif self.surface_role == LABEL_FREE_INFERENCE_SURFACE_ROLE:
            if query_ids != {outer_target}:
                raise ProtocolError("Label-free inference surfaces must be target-only.")
        elif outer_target not in query_ids:
            raise ProtocolError("Feature surface lacks its declared outer target query.")
        row_by_key = {row.row_key: row for row in rows}
        for row in rows:
            if row.action_id == baseline:
                if row.source_id is not None:
                    raise ProtocolError("Feature B rows cannot carry source identity.")
            elif (
                mapping.get(row.action_id) != row.source_id
                or row.source_id == row.query_id
                or row.source_id == outer_target
            ):
                raise ProtocolError("Feature candidate identity violates H/query exclusion.")
        disagreement_count_by_key: dict[tuple[str, str, str], int] = {}
        sample_key_set = set(sample_keys)
        for row in disagreements:
            case_action_key = (row.query_id, row.case_id, row.action_id)
            feature = row_by_key.get(case_action_key)
            if (
                feature is None
                or row.sample_key not in sample_key_set
                or row.source_id != feature.source_id
            ):
                raise ProtocolError("Sparse disagreement lineage drifted from feature rows.")
            disagreement_count_by_key[case_action_key] = (
                disagreement_count_by_key.get(case_action_key, 0) + 1
            )
        if self.family == "R" and any(
            row.disagreement_count != disagreement_count_by_key.get(row.row_key, 0)
            for row in rows
        ):
            raise ProtocolError("Sparse disagreement counts drifted from feature rows.")
        if self.family == "P":
            rows_by_case: dict[tuple[str, str], list[CaseActionFeatureRow]] = {}
            for row in rows:
                rows_by_case.setdefault((row.query_id, row.case_id), []).append(row)
            for case_rows in rows_by_case.values():
                baseline_rows = [
                    row for row in case_rows if row.action_id == baseline
                ]
                candidate_rows = [
                    row for row in case_rows if row.action_id != baseline
                ]
                if (
                    len(baseline_rows) != 1
                    or baseline_rows[0].feature_origin_action_id != baseline
                    or len(candidate_rows) < 2
                    or any(
                        row.feature_origin_action_id == row.action_id
                        for row in candidate_rows
                    )
                    or {row.feature_origin_action_id for row in candidate_rows}
                    != {row.action_id for row in candidate_rows}
                ):
                    raise ProtocolError(
                        "P controls require a complete within-case candidate derangement."
                    )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "disagreements", disagreements)
        object.__setattr__(self, "baseline_action_id", baseline)
        object.__setattr__(self, "control_action_id", control)
        object.__setattr__(self, "candidate_source_by_action", dict(sorted(mapping.items())))
        object.__setattr__(self, "sample_keys", sample_keys)
        object.__setattr__(self, "dataset_family", dataset_family)
        object.__setattr__(self, "outer_target_id", outer_target)
        payload = {
            "schema_version": "midogpp_disagreement_feature_surface_v1",
            "row_hashes": [row.feature_hash for row in rows],
            "disagreement_hashes": [row.row_hash for row in disagreements],
            "baseline_action_id": baseline,
            "control_action_id": control,
            "candidate_source_by_action": dict(sorted(mapping.items())),
            "prediction_seal_hash": self.prediction_seal_hash,
            "sample_keys": [list(key) for key in sample_keys],
            "development_context_hash": self.development_context_hash,
            "dataset_family": dataset_family,
            "outer_target_id": outer_target,
            "surface_role": self.surface_role,
            "family": self.family,
            "parent_surface_hash": self.parent_surface_hash,
        }
        object.__setattr__(self, "surface_hash", canonical_sha256(payload))

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.query_id for row in self.rows}))


__all__ = (
    "FEATURE_NAMES",
    "DEVELOPMENT_COMPOSITE_SURFACE_ROLE",
    "SOURCE_OOF_TRAINING_SURFACE_ROLE",
    "LABEL_FREE_INFERENCE_SURFACE_ROLE",
    "CaseActionFeatureRow",
    "DisagreementFeatureSurface",
    "DisagreementRow",
    "ProbabilityRow",
    "SourceOOFLabelRow",
)
