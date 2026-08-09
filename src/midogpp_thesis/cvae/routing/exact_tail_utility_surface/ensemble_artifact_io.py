"""Strict persisted-table readers for exact-tail ensemble artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from ...protocol import ProtocolError
from .contracts import expected_ensemble_endpoint_keys, expected_utility_keys
from .ensemble_scoring import (
    ENSEMBLE_ENDPOINT_ROW_SCHEMA,
    ScoredExactTailEnsembleEndpointRow,
)
from .support_shift_surface import (
    SUPPORT_SHIFT_ROW_SCHEMA,
    ExactTailSupportActionShiftRow,
    validate_support_shift_group_bindings,
)


def load_ensemble_endpoint_rows(
    path: str | Path,
) -> tuple[ScoredExactTailEnsembleEndpointRow, ...]:
    """Load the complete canonical 504-row endpoint table fail-closed."""

    rows: list[ScoredExactTailEnsembleEndpointRow] = []
    expected = {
        "schema_version",
        "outer_target",
        "pseudo_query",
        "candidate_source",
        "base_bacc",
        "tail_bacc",
        "delta_bacc",
        "evaluation_row_count",
        "evaluation_case_count",
        "evaluation_row_hash",
        "support_partition_hash",
        "prediction_seal_hash",
        "evaluation_label_sha256",
        "base_probability_cell_hashes_hash",
        "tail_probability_cell_hashes_hash",
        "base_ensemble_probability_sha256",
        "tail_ensemble_probability_sha256",
        "base_ensemble_prediction_sha256",
        "tail_ensemble_prediction_sha256",
        "base_endpoint_hash",
        "tail_endpoint_hash",
        "ensemble_utility_response_hash",
        "seed_pair_count",
        "seed_pairs_hash",
        "threshold",
        "primary_metric",
        "primary_utility_endpoint",
        "aggregation_semantics",
        "response_semantics",
        "endpoint_role",
        "development_labels_used_for_scoring_only",
        "technical_seed_repeats_are_not_independent_units",
        "target_support_labels_used",
        "target_evaluation_labels_used",
        "seed_selection_performed",
        "endpoint_row_hash",
    }
    for raw in _csv(Path(path)):
        if (
            set(raw) != expected
            or raw["schema_version"] != ENSEMBLE_ENDPOINT_ROW_SCHEMA
            or not _true(raw["development_labels_used_for_scoring_only"])
            or not _true(
                raw["technical_seed_repeats_are_not_independent_units"]
            )
        ):
            raise ProtocolError("Exact-tail ensemble endpoint CSV schema drifted.")
        row = ScoredExactTailEnsembleEndpointRow(
            outer_target=raw["outer_target"],
            pseudo_query=raw["pseudo_query"],
            candidate_source=raw["candidate_source"],
            base_bacc=float(raw["base_bacc"]),
            tail_bacc=float(raw["tail_bacc"]),
            delta_bacc=float(raw["delta_bacc"]),
            evaluation_row_count=int(raw["evaluation_row_count"]),
            evaluation_case_count=int(raw["evaluation_case_count"]),
            evaluation_row_hash=raw["evaluation_row_hash"],
            support_partition_hash=raw["support_partition_hash"],
            prediction_seal_hash=raw["prediction_seal_hash"],
            evaluation_label_sha256=raw["evaluation_label_sha256"],
            base_probability_cell_hashes_hash=raw[
                "base_probability_cell_hashes_hash"
            ],
            tail_probability_cell_hashes_hash=raw[
                "tail_probability_cell_hashes_hash"
            ],
            base_ensemble_probability_sha256=raw[
                "base_ensemble_probability_sha256"
            ],
            tail_ensemble_probability_sha256=raw[
                "tail_ensemble_probability_sha256"
            ],
            base_ensemble_prediction_sha256=raw[
                "base_ensemble_prediction_sha256"
            ],
            tail_ensemble_prediction_sha256=raw[
                "tail_ensemble_prediction_sha256"
            ],
            base_endpoint_hash=raw["base_endpoint_hash"],
            tail_endpoint_hash=raw["tail_endpoint_hash"],
            ensemble_utility_response_hash=raw[
                "ensemble_utility_response_hash"
            ],
            endpoint_row_hash=raw["endpoint_row_hash"],
            seed_pair_count=int(raw["seed_pair_count"]),
            threshold=float(raw["threshold"]),
            primary_metric=raw["primary_metric"],
            aggregation_semantics=raw["aggregation_semantics"],
            response_semantics=raw["response_semantics"],
            endpoint_role=raw["endpoint_role"],
            target_support_labels_used=_true(raw["target_support_labels_used"]),
            target_evaluation_labels_used=_true(
                raw["target_evaluation_labels_used"]
            ),
            seed_selection_performed=_true(raw["seed_selection_performed"]),
        )
        payload = row.to_payload()
        if (
            raw["seed_pairs_hash"] != payload["seed_pairs_hash"]
            or raw["primary_utility_endpoint"]
            != payload["primary_utility_endpoint"]
        ):
            raise ProtocolError("Exact-tail ensemble seed-pair identity drifted.")
        rows.append(row)
    if tuple(row.row_key for row in rows) != expected_ensemble_endpoint_keys():
        raise ProtocolError("Exact-tail ensemble endpoint CSV key grid drifted.")
    return tuple(rows)


def load_support_shift_rows(
    path: str | Path,
) -> tuple[ExactTailSupportActionShiftRow, ...]:
    """Load the complete canonical 4,536-row label-free shift table."""

    rows: list[ExactTailSupportActionShiftRow] = []
    expected = {
        "schema_version",
        "outer_target",
        "pseudo_query",
        "candidate_source",
        "training_seed",
        "generation_seed",
        "descriptive_seed_mean_absolute_shift",
        "candidate_ensemble_mean_absolute_shift",
        "support_row_count",
        "support_case_count",
        "support_row_hash",
        "support_partition_hash",
        "prediction_seal_hash",
        "base_support_probability_sha256",
        "tail_support_probability_sha256",
        "base_component_vector_hash",
        "tail_component_vector_hash",
        "candidate_base_ensemble_probability_sha256",
        "candidate_tail_ensemble_probability_sha256",
        "candidate_ensemble_absolute_difference_sha256",
        "candidate_aggregate_shift_hash",
        "scalar_name",
        "scalar_semantics",
        "row_role",
        "descriptive_seed_value_may_feed_model",
        "labels_used",
        "support_labels_available",
        "target_labels_used",
        "seed_selection_performed",
        "shift_row_hash",
    }
    for raw in _csv(Path(path)):
        if (
            set(raw) != expected
            or raw["schema_version"] != SUPPORT_SHIFT_ROW_SCHEMA
            or raw["descriptive_seed_value_may_feed_model"] != "False"
        ):
            raise ProtocolError("Exact-tail support action-shift CSV schema drifted.")
        rows.append(
            ExactTailSupportActionShiftRow(
                outer_target=raw["outer_target"],
                pseudo_query=raw["pseudo_query"],
                candidate_source=raw["candidate_source"],
                training_seed=int(raw["training_seed"]),
                generation_seed=int(raw["generation_seed"]),
                descriptive_seed_mean_absolute_shift=float(
                    raw["descriptive_seed_mean_absolute_shift"]
                ),
                candidate_ensemble_mean_absolute_shift=float(
                    raw["candidate_ensemble_mean_absolute_shift"]
                ),
                support_row_count=int(raw["support_row_count"]),
                support_case_count=int(raw["support_case_count"]),
                support_row_hash=raw["support_row_hash"],
                support_partition_hash=raw["support_partition_hash"],
                prediction_seal_hash=raw["prediction_seal_hash"],
                base_support_probability_sha256=raw[
                    "base_support_probability_sha256"
                ],
                tail_support_probability_sha256=raw[
                    "tail_support_probability_sha256"
                ],
                base_component_vector_hash=raw["base_component_vector_hash"],
                tail_component_vector_hash=raw["tail_component_vector_hash"],
                candidate_base_ensemble_probability_sha256=raw[
                    "candidate_base_ensemble_probability_sha256"
                ],
                candidate_tail_ensemble_probability_sha256=raw[
                    "candidate_tail_ensemble_probability_sha256"
                ],
                candidate_ensemble_absolute_difference_sha256=raw[
                    "candidate_ensemble_absolute_difference_sha256"
                ],
                candidate_aggregate_shift_hash=raw[
                    "candidate_aggregate_shift_hash"
                ],
                shift_row_hash=raw["shift_row_hash"],
                scalar_name=raw["scalar_name"],
                scalar_semantics=raw["scalar_semantics"],
                row_role=raw["row_role"],
                labels_used=_true(raw["labels_used"]),
                support_labels_available=_true(
                    raw["support_labels_available"]
                ),
                target_labels_used=_true(raw["target_labels_used"]),
                seed_selection_performed=_true(raw["seed_selection_performed"]),
            )
        )
    if tuple(row.row_key for row in rows) != expected_utility_keys():
        raise ProtocolError("Exact-tail support action-shift CSV key grid drifted.")
    validate_support_shift_group_bindings(rows)
    return tuple(rows)


def _csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(
                set(reader.fieldnames)
            ):
                raise ProtocolError(f"Exact-tail CSV header drifted: {path}.")
            return tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read exact-tail CSV: {path}.") from exc


def _true(value: object) -> bool:
    if value not in (True, "True"):
        if value in (False, "False"):
            return False
        raise ProtocolError("Exact-tail persisted boolean is malformed.")
    return True


__all__ = ("load_ensemble_endpoint_rows", "load_support_shift_rows")
