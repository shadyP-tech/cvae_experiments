"""All-nine-seed probability-ensemble scoring and endpoint lock contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ...protocol import ProtocolError
from ..utility_aligned.ensemble_contracts import SeedProbabilityVector
from ..utility_aligned.ensemble_endpoint import build_ensemble_utility_response
from .contracts import (
    BASE_ACTION_ID,
    EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT,
    PRIMARY_METRIC,
    DevelopmentPartition,
    expected_ensemble_endpoint_keys,
    legal_sources,
    tail_action_id,
)
from .label_access import OpenedDevelopmentLabels
from .probability_surface import array_sha256
from .scoring import SealedPredictionSurface
from .seals import GlobalPredictionSeal


ENSEMBLE_ENDPOINT_TABLE_MEMBER = "tables/exact_tail_ensemble_endpoints.csv"
ENSEMBLE_ENDPOINT_LOCK_MEMBER = (
    "manifests/exact_tail_ensemble_endpoints_lock.json"
)
ENSEMBLE_ENDPOINT_ROW_SCHEMA = "midogpp_exact_tail_ensemble_endpoint_row_v1"
ENSEMBLE_ENDPOINT_LOCK_SCHEMA = "midogpp_exact_tail_ensemble_endpoints_lock_v1"
ENSEMBLE_SEED_PAIR_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
ENSEMBLE_THRESHOLD = 0.5
ENSEMBLE_AGGREGATION = (
    "arithmetic_mean_of_exact_nine_float32_probability_vectors_then_threshold_once"
)
ENSEMBLE_RESPONSE_SEMANTICS = (
    "bacc_of_all_nine_seed_probability_ensemble_exact_tail_minus_"
    "bacc_of_all_nine_seed_probability_ensemble_exact_base"
)
ENSEMBLE_ENDPOINT_ROLE = "predeclared_all_nine_seed_probability_ensemble"
PRIMARY_UTILITY_ENDPOINT = "all_nine_seed_probability_ensemble_bacc_delta"


@dataclass(frozen=True)
class ScoredExactTailEnsembleEndpointRow:
    """One operational source-inner endpoint; seed cells are technical repeats."""

    outer_target: str
    pseudo_query: str
    candidate_source: str
    base_bacc: float
    tail_bacc: float
    delta_bacc: float
    evaluation_row_count: int
    evaluation_case_count: int
    evaluation_row_hash: str
    support_partition_hash: str
    prediction_seal_hash: str
    evaluation_label_sha256: str
    base_probability_cell_hashes_hash: str
    tail_probability_cell_hashes_hash: str
    base_ensemble_probability_sha256: str
    tail_ensemble_probability_sha256: str
    base_ensemble_prediction_sha256: str
    tail_ensemble_prediction_sha256: str
    base_endpoint_hash: str
    tail_endpoint_hash: str
    ensemble_utility_response_hash: str
    endpoint_row_hash: str
    seed_pair_count: int = ENSEMBLE_SEED_PAIR_COUNT
    threshold: float = ENSEMBLE_THRESHOLD
    primary_metric: str = PRIMARY_METRIC
    aggregation_semantics: str = ENSEMBLE_AGGREGATION
    response_semantics: str = ENSEMBLE_RESPONSE_SEMANTICS
    endpoint_role: str = ENSEMBLE_ENDPOINT_ROLE
    target_support_labels_used: bool = False
    target_evaluation_labels_used: bool = False
    seed_selection_performed: bool = False

    def __post_init__(self) -> None:
        if self.pseudo_query == self.outer_target or self.candidate_source not in legal_sources(
            outer_target=self.outer_target, pseudo_query=self.pseudo_query
        ):
            raise ProtocolError("Exact-tail ensemble endpoint violates H/q/e exclusions.")
        metrics = (float(self.base_bacc), float(self.tail_bacc), float(self.delta_bacc))
        if (
            not all(math.isfinite(value) for value in metrics)
            or not 0.0 <= self.base_bacc <= 1.0
            or not 0.0 <= self.tail_bacc <= 1.0
            or not math.isclose(
                self.delta_bacc,
                self.tail_bacc - self.base_bacc,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ProtocolError("Exact-tail ensemble endpoint metrics drifted.")
        if self.evaluation_row_count <= 0 or self.evaluation_case_count <= 0:
            raise ProtocolError("Exact-tail ensemble endpoint has empty evaluation coverage.")
        for value, role, lengths in (
            (self.evaluation_row_hash, "evaluation-row hash", {16}),
            (self.support_partition_hash, "support-partition hash", {16}),
            (self.prediction_seal_hash, "prediction-seal hash", {16}),
            (self.evaluation_label_sha256, "evaluation-label SHA-256", {64}),
            (
                self.base_probability_cell_hashes_hash,
                "base probability-cell hash",
                {16},
            ),
            (
                self.tail_probability_cell_hashes_hash,
                "tail probability-cell hash",
                {16},
            ),
            (
                self.base_ensemble_probability_sha256,
                "base ensemble-probability SHA-256",
                {64},
            ),
            (
                self.tail_ensemble_probability_sha256,
                "tail ensemble-probability SHA-256",
                {64},
            ),
            (
                self.base_ensemble_prediction_sha256,
                "base ensemble-prediction SHA-256",
                {64},
            ),
            (
                self.tail_ensemble_prediction_sha256,
                "tail ensemble-prediction SHA-256",
                {64},
            ),
            (self.base_endpoint_hash, "base endpoint hash", {64}),
            (self.tail_endpoint_hash, "tail endpoint hash", {64}),
            (
                self.ensemble_utility_response_hash,
                "ensemble utility-response hash",
                {64},
            ),
            (self.endpoint_row_hash, "endpoint-row hash", {16}),
        ):
            _require_hash(value, role, lengths)
        if (
            self.seed_pair_count != ENSEMBLE_SEED_PAIR_COUNT
            or self.threshold != ENSEMBLE_THRESHOLD
            or self.primary_metric != PRIMARY_METRIC
            or self.aggregation_semantics != ENSEMBLE_AGGREGATION
            or self.response_semantics != ENSEMBLE_RESPONSE_SEMANTICS
            or self.endpoint_role != ENSEMBLE_ENDPOINT_ROLE
            or self.target_support_labels_used is not False
            or self.target_evaluation_labels_used is not False
            or self.seed_selection_performed is not False
        ):
            raise ProtocolError("Exact-tail ensemble endpoint contract drifted.")
        if self.endpoint_row_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail ensemble endpoint row hash drifted.")

    @property
    def row_key(self) -> tuple[str, str, str]:
        return self.outer_target, self.pseudo_query, self.candidate_source

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": ENSEMBLE_ENDPOINT_ROW_SCHEMA,
            "outer_target": self.outer_target,
            "pseudo_query": self.pseudo_query,
            "candidate_source": self.candidate_source,
            "base_bacc": self.base_bacc,
            "tail_bacc": self.tail_bacc,
            "delta_bacc": self.delta_bacc,
            "evaluation_row_count": self.evaluation_row_count,
            "evaluation_case_count": self.evaluation_case_count,
            "evaluation_row_hash": self.evaluation_row_hash,
            "support_partition_hash": self.support_partition_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "evaluation_label_sha256": self.evaluation_label_sha256,
            "base_probability_cell_hashes_hash": (
                self.base_probability_cell_hashes_hash
            ),
            "tail_probability_cell_hashes_hash": (
                self.tail_probability_cell_hashes_hash
            ),
            "base_ensemble_probability_sha256": (
                self.base_ensemble_probability_sha256
            ),
            "tail_ensemble_probability_sha256": (
                self.tail_ensemble_probability_sha256
            ),
            "base_ensemble_prediction_sha256": (
                self.base_ensemble_prediction_sha256
            ),
            "tail_ensemble_prediction_sha256": (
                self.tail_ensemble_prediction_sha256
            ),
            "base_endpoint_hash": self.base_endpoint_hash,
            "tail_endpoint_hash": self.tail_endpoint_hash,
            "ensemble_utility_response_hash": self.ensemble_utility_response_hash,
            "seed_pair_count": self.seed_pair_count,
            "seed_pairs_hash": _seed_pairs_hash(),
            "threshold": self.threshold,
            "primary_metric": self.primary_metric,
            "primary_utility_endpoint": PRIMARY_UTILITY_ENDPOINT,
            "aggregation_semantics": self.aggregation_semantics,
            "response_semantics": self.response_semantics,
            "endpoint_role": self.endpoint_role,
            "development_labels_used_for_scoring_only": True,
            "technical_seed_repeats_are_not_independent_units": True,
            "target_support_labels_used": self.target_support_labels_used,
            "target_evaluation_labels_used": self.target_evaluation_labels_used,
            "seed_selection_performed": self.seed_selection_performed,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "endpoint_row_hash": self.endpoint_row_hash}


@dataclass(frozen=True)
class ExactTailEnsembleEndpointLock:
    """Separate lock binding the complete 504-row ensemble endpoint table."""

    config_contract_hash: str
    prediction_seal_hash: str
    prediction_index_sha256: str
    prediction_arrays_sha256: str
    probability_cell_surface_hash: str
    endpoint_keys_hash: str
    endpoint_row_hashes_hash: str
    endpoint_table_sha256: str
    endpoint_row_count: int
    endpoint_lock_hash: str
    schema_version: str = ENSEMBLE_ENDPOINT_LOCK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ENSEMBLE_ENDPOINT_LOCK_SCHEMA:
            raise ProtocolError("Exact-tail ensemble endpoint-lock schema drifted.")
        if self.endpoint_row_count != EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT:
            raise ProtocolError("Exact-tail ensemble endpoint-lock row count drifted.")
        for value, role, lengths in (
            (self.config_contract_hash, "config hash", {16, 64}),
            (self.prediction_seal_hash, "prediction-seal hash", {16}),
            (self.prediction_index_sha256, "prediction-index SHA-256", {64}),
            (self.prediction_arrays_sha256, "prediction-array SHA-256", {64}),
            (self.probability_cell_surface_hash, "probability-cell surface hash", {16}),
            (self.endpoint_keys_hash, "endpoint-key hash", {16}),
            (self.endpoint_row_hashes_hash, "endpoint-row hash", {16}),
            (self.endpoint_table_sha256, "endpoint-table SHA-256", {64}),
            (self.endpoint_lock_hash, "endpoint-lock hash", {16}),
        ):
            _require_hash(value, role, lengths)
        if self.endpoint_keys_hash != _endpoint_keys_hash():
            raise ProtocolError("Exact-tail endpoint identity grid escaped its lock.")
        if self.endpoint_lock_hash != stable_hash(self._unhashed_payload()):
            raise ProtocolError("Exact-tail ensemble endpoint-lock hash drifted.")

    def _unhashed_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "config_contract_hash": self.config_contract_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "prediction_index_sha256": self.prediction_index_sha256,
            "prediction_arrays_sha256": self.prediction_arrays_sha256,
            "probability_cell_surface_hash": self.probability_cell_surface_hash,
            "endpoint_keys_hash": self.endpoint_keys_hash,
            "endpoint_row_hashes_hash": self.endpoint_row_hashes_hash,
            "endpoint_table_sha256": self.endpoint_table_sha256,
            "endpoint_row_count": self.endpoint_row_count,
            "seed_pair_count": ENSEMBLE_SEED_PAIR_COUNT,
            "seed_pairs_hash": _seed_pairs_hash(),
            "threshold": ENSEMBLE_THRESHOLD,
            "primary_metric": PRIMARY_METRIC,
            "primary_utility_endpoint": PRIMARY_UTILITY_ENDPOINT,
            "per_seed_utility_role": "descriptive_only",
            "per_seed_rows_may_feed_model": False,
            "aggregation_semantics": ENSEMBLE_AGGREGATION,
            "response_semantics": ENSEMBLE_RESPONSE_SEMANTICS,
            "all_predictions_sealed_before_development_labels": True,
            "development_labels_used_for_scoring_only": True,
            "technical_seed_repeats_are_not_independent_units": True,
            "outer_target_excluded_from_query_and_source_roles": True,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "seed_selection_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed_payload(), "endpoint_lock_hash": self.endpoint_lock_hash}


def score_exact_tail_ensemble_endpoints(
    predictions: SealedPredictionSurface,
    labels: OpenedDevelopmentLabels,
    partitions: Mapping[str, DevelopmentPartition],
) -> tuple[ScoredExactTailEnsembleEndpointRow, ...]:
    """Mean each exact canonical 3x3 probability grid, then threshold once."""

    seal = predictions.seal
    seal.verify_complete()
    if (
        labels.prediction_seal_hash != seal.seal_hash
        or labels.manifest_sha256 != seal.development_manifest_sha256
    ):
        raise ProtocolError("Exact-tail ensemble labels bind another prediction surface.")
    normalized_partitions = {
        str(center): partition for center, partition in partitions.items()
    }
    cell_by_key = {cell.key: cell for cell in seal.cells}
    rows: list[ScoredExactTailEnsembleEndpointRow] = []
    for outer, query, source in expected_ensemble_endpoint_keys():
        partition = normalized_partitions.get(query)
        if (
            partition is None
            or partition.reservation_hash != seal.partition_hash_by_center[query]
            or labels.row_hash_by_center.get(query)
            != seal.evaluation_row_hash_by_center[query]
        ):
            raise ProtocolError(
                "Exact-tail ensemble partition or label rows escaped the seal."
            )
        truth = np.asarray(labels.labels_by_center[query], dtype=np.uint8)
        base_keys = _action_seed_keys(outer, query, BASE_ACTION_ID)
        tail_keys = _action_seed_keys(outer, query, tail_action_id(source))
        base_probabilities = _exact_nine_vectors(
            predictions, base_keys, truth.shape, cell_by_key
        )
        tail_probabilities = _exact_nine_vectors(
            predictions, tail_keys, truth.shape, cell_by_key
        )
        response = build_ensemble_utility_response(
            outer_target_id=outer,
            query_id=query,
            candidate_source=source,
            base_vectors=base_probabilities,
            tail_vectors=tail_probabilities,
            labels=truth,
            support_partition_hash=partition.reservation_hash,
            evaluation_partition_hash=seal.evaluation_row_hash_by_center[query],
            prediction_seal_hash=seal.seal_hash,
            support_eval_disjoint=True,
            predictions_sealed_before_labels=True,
            source_expert_frozen=True,
            target_labels_used_for_routing=False,
        )
        base_mean = response.base_endpoint.mean_positive_probabilities
        tail_mean = response.tail_endpoint.mean_positive_probabilities
        base_prediction = response.base_endpoint.predictions
        tail_prediction = response.tail_endpoint.predictions
        base_bacc = response.base_bacc
        tail_bacc = response.tail_bacc
        values: dict[str, object] = {
            "outer_target": outer,
            "pseudo_query": query,
            "candidate_source": source,
            "base_bacc": base_bacc,
            "tail_bacc": tail_bacc,
            "delta_bacc": tail_bacc - base_bacc,
            "evaluation_row_count": len(truth),
            "evaluation_case_count": len(
                {row.case_id for row in partition.evaluation_rows}
            ),
            "evaluation_row_hash": seal.evaluation_row_hash_by_center[query],
            "support_partition_hash": partition.reservation_hash,
            "prediction_seal_hash": seal.seal_hash,
            "evaluation_label_sha256": response.base_endpoint.label_hash,
            "base_probability_cell_hashes_hash": stable_hash(
                [cell_by_key[key].probability_sha256 for key in base_keys]
            ),
            "tail_probability_cell_hashes_hash": stable_hash(
                [cell_by_key[key].probability_sha256 for key in tail_keys]
            ),
            "base_ensemble_probability_sha256": array_sha256(base_mean),
            "tail_ensemble_probability_sha256": array_sha256(tail_mean),
            "base_ensemble_prediction_sha256": array_sha256(base_prediction),
            "tail_ensemble_prediction_sha256": array_sha256(tail_prediction),
            "base_endpoint_hash": response.base_endpoint.endpoint_hash,
            "tail_endpoint_hash": response.tail_endpoint.endpoint_hash,
            "ensemble_utility_response_hash": response.row_hash,
            "endpoint_row_hash": "",
        }
        provisional = ScoredExactTailEnsembleEndpointRow.__new__(
            ScoredExactTailEnsembleEndpointRow
        )
        for key, value in values.items():
            object.__setattr__(provisional, key, value)
        for key, value in _endpoint_defaults().items():
            object.__setattr__(provisional, key, value)
        values["endpoint_row_hash"] = stable_hash(provisional._unhashed_payload())
        rows.append(ScoredExactTailEnsembleEndpointRow(**values))  # type: ignore[arg-type]
    if (
        len(rows) != EXPECTED_ENSEMBLE_ENDPOINT_ROW_COUNT
        or tuple(row.row_key for row in rows) != expected_ensemble_endpoint_keys()
    ):
        raise ProtocolError("Exact-tail ensemble endpoint coverage drifted.")
    return tuple(rows)


def build_ensemble_endpoint_lock(
    *,
    seal: GlobalPredictionSeal,
    rows: Sequence[ScoredExactTailEnsembleEndpointRow],
    endpoint_table_sha256: str,
) -> ExactTailEnsembleEndpointLock:
    endpoint_rows = tuple(rows)
    if tuple(row.row_key for row in endpoint_rows) != expected_ensemble_endpoint_keys():
        raise ProtocolError("Exact-tail ensemble endpoint rows are not canonical.")
    if any(row.prediction_seal_hash != seal.seal_hash for row in endpoint_rows):
        raise ProtocolError("Exact-tail ensemble endpoint rows mix prediction seals.")
    values: dict[str, object] = {
        "config_contract_hash": seal.config_contract_hash,
        "prediction_seal_hash": seal.seal_hash,
        "prediction_index_sha256": seal.prediction_index_sha256,
        "prediction_arrays_sha256": seal.prediction_arrays_sha256,
        "probability_cell_surface_hash": stable_hash(
            [cell.probability_sha256 for cell in seal.cells]
        ),
        "endpoint_keys_hash": _endpoint_keys_hash(),
        "endpoint_row_hashes_hash": stable_hash(
            [row.endpoint_row_hash for row in endpoint_rows]
        ),
        "endpoint_table_sha256": str(endpoint_table_sha256),
        "endpoint_row_count": len(endpoint_rows),
        "endpoint_lock_hash": "",
        "schema_version": ENSEMBLE_ENDPOINT_LOCK_SCHEMA,
    }
    provisional = ExactTailEnsembleEndpointLock.__new__(
        ExactTailEnsembleEndpointLock
    )
    for key, value in values.items():
        object.__setattr__(provisional, key, value)
    values["endpoint_lock_hash"] = stable_hash(provisional._unhashed_payload())
    return ExactTailEnsembleEndpointLock(**values)  # type: ignore[arg-type]


def load_ensemble_endpoint_lock(
    path: str | Path,
) -> ExactTailEnsembleEndpointLock:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot load exact-tail ensemble endpoint lock.") from exc
    required = set(
        ExactTailEnsembleEndpointLock.__new__(
            ExactTailEnsembleEndpointLock
        ).__class__.__dataclass_fields__
    )
    required.update(
        {
            "seed_pair_count",
            "seed_pairs_hash",
            "threshold",
            "primary_metric",
            "primary_utility_endpoint",
            "per_seed_utility_role",
            "per_seed_rows_may_feed_model",
            "aggregation_semantics",
            "response_semantics",
            "all_predictions_sealed_before_development_labels",
            "development_labels_used_for_scoring_only",
            "technical_seed_repeats_are_not_independent_units",
            "outer_target_excluded_from_query_and_source_roles",
            "target_support_labels_used",
            "target_evaluation_labels_used",
            "seed_selection_performed",
        }
    )
    if set(raw) != required or any(
        (
            raw.get("schema_version") != ENSEMBLE_ENDPOINT_LOCK_SCHEMA,
            raw.get("seed_pair_count") != ENSEMBLE_SEED_PAIR_COUNT,
            raw.get("seed_pairs_hash") != _seed_pairs_hash(),
            raw.get("threshold") != ENSEMBLE_THRESHOLD,
            raw.get("primary_metric") != PRIMARY_METRIC,
            raw.get("primary_utility_endpoint") != PRIMARY_UTILITY_ENDPOINT,
            raw.get("per_seed_utility_role") != "descriptive_only",
            raw.get("per_seed_rows_may_feed_model") is not False,
            raw.get("aggregation_semantics") != ENSEMBLE_AGGREGATION,
            raw.get("response_semantics") != ENSEMBLE_RESPONSE_SEMANTICS,
            raw.get("all_predictions_sealed_before_development_labels") is not True,
            raw.get("development_labels_used_for_scoring_only") is not True,
            raw.get("technical_seed_repeats_are_not_independent_units") is not True,
            raw.get("outer_target_excluded_from_query_and_source_roles") is not True,
            raw.get("target_support_labels_used") is not False,
            raw.get("target_evaluation_labels_used") is not False,
            raw.get("seed_selection_performed") is not False,
        )
    ):
        raise ProtocolError("Exact-tail ensemble endpoint lock violates its contract.")
    return ExactTailEnsembleEndpointLock(
        config_contract_hash=str(raw["config_contract_hash"]),
        prediction_seal_hash=str(raw["prediction_seal_hash"]),
        prediction_index_sha256=str(raw["prediction_index_sha256"]),
        prediction_arrays_sha256=str(raw["prediction_arrays_sha256"]),
        probability_cell_surface_hash=str(raw["probability_cell_surface_hash"]),
        endpoint_keys_hash=str(raw["endpoint_keys_hash"]),
        endpoint_row_hashes_hash=str(raw["endpoint_row_hashes_hash"]),
        endpoint_table_sha256=str(raw["endpoint_table_sha256"]),
        endpoint_row_count=int(raw["endpoint_row_count"]),
        endpoint_lock_hash=str(raw["endpoint_lock_hash"]),
        schema_version=str(raw["schema_version"]),
    )


def _action_seed_keys(
    outer: str, query: str, action_id: str
) -> tuple[tuple[str, str, str, int, int], ...]:
    return tuple(
        (outer, query, action_id, training_seed, generation_seed)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )


def _exact_nine_vectors(
    predictions: SealedPredictionSurface,
    keys: Sequence[tuple[str, str, str, int, int]],
    expected_shape: tuple[int, ...],
    cell_by_key: Mapping[tuple[str, str, str, int, int], object],
) -> tuple[SeedProbabilityVector, ...]:
    if len(keys) != ENSEMBLE_SEED_PAIR_COUNT or len(set(keys)) != len(keys):
        raise ProtocolError("Exact-tail ensemble requires exactly nine seed cells.")
    try:
        probabilities = tuple(predictions.probabilities_by_key[key] for key in keys)
    except KeyError as exc:
        raise ProtocolError("Exact-tail ensemble probability cell is missing.") from exc
    if any(vector.shape != expected_shape for vector in probabilities):
        raise ProtocolError("Exact-tail ensemble probability row geometry drifted.")
    try:
        return tuple(
            SeedProbabilityVector(
                training_seed=key[3],
                generation_seed=key[4],
                row_identity_hash=str(
                    getattr(cell_by_key[key], "evaluation_row_identity_hash")
                ),
                prediction_provenance_hash=str(
                    getattr(cell_by_key[key], "probability_sha256")
                ),
                positive_class_probabilities=probability,
            )
            for key, probability in zip(keys, probabilities, strict=True)
        )
    except (KeyError, AttributeError) as exc:
        raise ProtocolError("Exact-tail ensemble probability seal cell is missing.") from exc


def _endpoint_defaults() -> dict[str, object]:
    return {
        "seed_pair_count": ENSEMBLE_SEED_PAIR_COUNT,
        "threshold": ENSEMBLE_THRESHOLD,
        "primary_metric": PRIMARY_METRIC,
        "aggregation_semantics": ENSEMBLE_AGGREGATION,
        "response_semantics": ENSEMBLE_RESPONSE_SEMANTICS,
        "endpoint_role": ENSEMBLE_ENDPOINT_ROLE,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "seed_selection_performed": False,
    }


def _seed_pairs_hash() -> str:
    return stable_hash(
        [
            [training_seed, generation_seed]
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        ]
    )


def _endpoint_keys_hash() -> str:
    return stable_hash([list(key) for key in expected_ensemble_endpoint_keys()])


def _require_hash(value: object, role: str, lengths: set[int]) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Exact-tail ensemble {role} is malformed.")


__all__ = (
    "ENSEMBLE_AGGREGATION",
    "ENSEMBLE_ENDPOINT_LOCK_MEMBER",
    "ENSEMBLE_ENDPOINT_LOCK_SCHEMA",
    "ENSEMBLE_ENDPOINT_ROLE",
    "ENSEMBLE_ENDPOINT_ROW_SCHEMA",
    "ENSEMBLE_ENDPOINT_TABLE_MEMBER",
    "ENSEMBLE_RESPONSE_SEMANTICS",
    "ENSEMBLE_SEED_PAIR_COUNT",
    "ENSEMBLE_THRESHOLD",
    "PRIMARY_UTILITY_ENDPOINT",
    "ExactTailEnsembleEndpointLock",
    "ScoredExactTailEnsembleEndpointRow",
    "build_ensemble_endpoint_lock",
    "load_ensemble_endpoint_lock",
    "score_exact_tail_ensemble_endpoints",
)
