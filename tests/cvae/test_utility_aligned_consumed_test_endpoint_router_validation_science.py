from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.utility_aligned import (
    EnsembleCandidateFeatureRow,
    ScoredEnsembleUtilityResponse,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.contracts import (
    CENTERS,
    candidate_sources,
    inner_candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.validation_science import (
    ScientificPartitionContext,
    _decode_hashed_row,
    _feature_row,
    _validate_logical_endpoint_aliases,
    validate_development_science,
    validate_terminal_science,
)


SHA = "a" * 64
DEVELOPMENT_SEAL = "b" * 64


def test_flat_csv_decoder_is_lossless_and_feature_hash_bound() -> None:
    row = EnsembleCandidateFeatureRow(
        role="fresh_target_support",
        outer_target_id="0",
        query_id="0",
        candidate_source="1",
        candidate_source_count=8,
        support_partition_hash=SHA,
        support_case_count=8,
        seed_row_hashes=tuple(f"seed-{index}" for index in range(9)),
        feature_mean_by_name=_feature_values(0.1),
        feature_seed_standard_deviation_by_name=_feature_values(0.01),
        target_local_scalar=0.02,
        target_local_scalar_name="local",
        target_local_scalar_semantics="label-free",
        target_local_scalar_seed_standard_deviation=0.003,
        target_local_scalar_provenance_hash=SHA,
    )
    raw = _csv_row({**row.to_payload(), "row_hash": row.row_hash})
    assert _feature_row(raw).to_payload() == row.to_payload()

    raw["target_local_scalar"] = "0.03"
    with pytest.raises(ProtocolError, match="row hash drifted"):
        _feature_row(raw)


def test_model_hash_tamper_is_rejected() -> None:
    payload = {
        "schema_version": "midogpp_consumed_test_endpoint_router_models_v1",
        "outer_target_id": "0",
        "model_hashes_by_role": {"G": SHA, "R": SHA, "P": SHA},
        "cardinality_transfer_hash": SHA,
        "source_feature_surface_hash": SHA,
        "development_response_set_hash": SHA,
        "training_response_count": 56,
        "response_unit": "candidate_H_q_e_exact_nine_probability_ensemble",
        "alpha_tuning_endpoint": "mean_normalized_oracle_regret",
        "strict_H_q_e_exclusion": True,
        "same_outer_H_evaluation_labels_used_for_fit": False,
        "support_labels_used_for_fit": False,
        "target_features_used_for_fit": False,
        "seed_rows_are_independent_observations": False,
    }
    raw = _csv_row({**payload, "model_hash": canonical_sha256(payload)})
    decoded = _decode_hashed_row(
        raw, set(raw), "model_hash", "model"
    )
    assert decoded["training_response_count"] == 56
    raw["training_response_count"] = "55"
    with pytest.raises(ProtocolError, match="model_hash drifted"):
        _decode_hashed_row(raw, set(raw), "model_hash", "model")


def test_action_hash_tamper_is_rejected() -> None:
    payload = {
        "schema_version": "midogpp_consumed_test_frozen_endpoint_action_v1",
        "outer_target_id": "0", "query_id": "0", "action_id": "B",
        "action_role": "consumed_test_target_static_endpoint",
        "effective_action_id": "B", "selected_source": None,
        "geometry": {"source_order": list(candidate_sources("0"))},
        "topup_counts_by_source": {source: 0 for source in candidate_sources("0")},
        "realized_total_per_class": 1024, "core_action_hash": None,
        "policy_hash": SHA, "diagnostic_control": True, "target_static": True,
        "case_router_used": False, "labels_used_to_build": False,
        "terminal_scores_used_to_build": False, "diagnostic_only": True,
    }
    raw = _csv_row({**payload, "action_hash": canonical_sha256(payload)})
    _decode_hashed_row(raw, set(raw), "action_hash", "action")
    raw["effective_action_id"] = "U"
    with pytest.raises(ProtocolError, match="action_hash drifted"):
        _decode_hashed_row(raw, set(raw), "action_hash", "action")


def test_duplicate_development_H_q_e_is_rejected(tmp_path: Path) -> None:
    rows = _development_rows()
    rows[-1] = dict(rows[0])
    _write_csv(tmp_path / "tables/development_endpoint_responses.csv", rows)
    _write_json(
        tmp_path / "manifests/development_prediction_seal.json",
        {"development_prediction_seal_hash": DEVELOPMENT_SEAL},
    )
    with pytest.raises(ProtocolError, match="duplicate H/q/e"):
        validate_development_science(tmp_path, _partition_context())


def test_terminal_target_score_tamper_fails_before_inference(tmp_path: Path) -> None:
    rows = [_terminal_score_row("0", "B", balanced_accuracy=0.6)]
    rows[0]["balanced_accuracy"] = "0.7"
    _write_csv(tmp_path / "tables/terminal_endpoint_scores.csv", rows)
    with pytest.raises(ProtocolError, match="terminal score score_hash drifted"):
        validate_terminal_science(
            tmp_path,
            partitions=_partition_context(),
            prelabel=_minimal_prelabel(),
        )


def test_logical_endpoint_alias_tamper_is_rejected() -> None:
    common = {
        "evaluation_row_identity_hash": SHA, "evaluation_label_hash": SHA,
        "endpoint_hash": "endpoint", "evaluation_case_count": 15,
        "evaluation_row_count": 874, "observed_class_0_row_count": 437,
        "observed_class_1_row_count": 437, "balanced_accuracy": 0.6,
        "primary_endpoint": "exact_nine_probability_mean_then_threshold_bacc",
    }
    scores = {
        ("0", "B"): dict(common),
        ("0", "Hxe::1"): dict(common),
        ("0", "G"): dict(common),
        ("0", "R"): dict(common),
        ("0", "P"): dict(common),
    }
    effective = {"G": "Hxe::1", "R": "B", "P": "Hxe::1"}
    _validate_logical_endpoint_aliases("0", scores, effective)
    scores[("0", "R")]["endpoint_hash"] = "tampered"
    with pytest.raises(ProtocolError, match="differs from its physical action"):
        _validate_logical_endpoint_aliases("0", scores, effective)


def test_aggregate_formula_hash_rejects_tamper() -> None:
    payload = {
        "schema_version": "midogpp_consumed_test_aggregate_center_contrast_v1",
        "contrast_id": "R-B", "left_action_id": "R", "right_action_id": "B",
        "center_count": 9, "degrees_of_freedom": 8,
        "equal_center_mean_delta": 0.01, "sample_standard_deviation": 0.02,
        "standard_error": 0.02 / 3.0, "two_sided_ci95_lower": -0.01,
        "two_sided_ci95_upper": 0.03, "one_sided_lcb95": -0.005,
        "two_sided_p_value": 0.2, "center_delta_hash": SHA,
        "score_set_hash": SHA, "inference_unit": "target_center",
        "technical_seed_cells_are_independent_units": False,
        "terminal_scores_may_update_plan": False,
        "consumed_test_diagnostic_only": True,
    }
    from midogpp_thesis.cvae.diagnostics.utility_aligned_consumed_test_endpoint_router.inference import AggregateCenterContrast

    with pytest.raises(ProtocolError, match="boundary drifted"):
        AggregateCenterContrast(
            contrast_id="R-B", left_action_id="R", right_action_id="B",
            center_count=9, degrees_of_freedom=8,
            equal_center_mean_delta=0.01, sample_standard_deviation=0.02,
            standard_error=0.02 / 3.0, two_sided_ci95_lower=-0.01,
            two_sided_ci95_upper=0.03, one_sided_lcb95=-0.004,
            two_sided_p_value=0.2, center_delta_hash=SHA,
            score_set_hash=SHA, summary_hash=canonical_sha256(payload),
        )


def _development_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for outer in CENTERS:
        for query in candidate_sources(outer):
            for source in inner_candidate_sources(outer, query):
                row = ScoredEnsembleUtilityResponse(
                    outer_target_id=outer, query_id=query,
                    candidate_source=source, candidate_source_count=7,
                    support_partition_hash=f"support::{query}",
                    evaluation_partition_hash=f"evaluation::{query}",
                    prediction_seal_hash=DEVELOPMENT_SEAL,
                    evaluation_row_identity_hash=f"evaluation::{query}",
                    evaluation_label_hash=f"labels::{query}",
                    base_endpoint_hash=f"base::{outer}::{query}",
                    tail_endpoint_hash=f"tail::{outer}::{query}::{source}",
                    base_probability_cell_hashes_hash=f"bc::{outer}::{query}",
                    tail_probability_cell_hashes_hash=f"tc::{outer}::{query}::{source}",
                    base_ensemble_probability_hash=f"bp::{outer}::{query}",
                    tail_ensemble_probability_hash=f"tp::{outer}::{query}::{source}",
                    base_ensemble_prediction_hash=f"by::{outer}::{query}",
                    tail_ensemble_prediction_hash=f"ty::{outer}::{query}::{source}",
                    source_response_hash=None, source_endpoint_row_hash=None,
                    base_component_vector_hashes=tuple(
                        f"base::{outer}::{query}::{index}" for index in range(9)
                    ),
                    tail_component_vector_hashes=tuple(
                        f"tail::{outer}::{query}::{source}::{index}" for index in range(9)
                    ),
                    base_bacc=0.5, tail_bacc=0.51,
                    support_eval_disjoint=True, predictions_sealed_before_labels=True,
                    source_expert_frozen=True,
                )
                rows.append(_csv_row({**row.to_payload(), "row_hash": row.row_hash}))
    return rows


def _partition_context() -> ScientificPartitionContext:
    return ScientificPartitionContext(
        support_case_ids_by_center={center: tuple(f"c{center}-{i}" for i in range(8)) for center in CENTERS},
        support_row_identity_hash_by_center={center: f"support::{center}" for center in CENTERS},
        support_feature_hash_by_center={center: f"support::{center}" for center in CENTERS},
        evaluation_identity_hash_by_center={center: f"evaluation::{center}" for center in CENTERS},
        partition_hash_by_center={center: f"partition::{center}" for center in CENTERS},
        support_partition_lock_hash=SHA,
    )


@dataclass(frozen=True)
class _Prelabel:
    action_hash_by_key: dict[tuple[str, str], str]
    policy_hash_by_target: dict[str, str]
    effective_action_by_key: dict[tuple[str, str], str]
    global_target_prediction_seal_hash: str
    global_prelabel_seal_hash: str


def _minimal_prelabel() -> _Prelabel:
    return _Prelabel(
        action_hash_by_key={("0", "B"): SHA},
        policy_hash_by_target={"0": SHA},
        effective_action_by_key={("0", "B"): "B"},
        global_target_prediction_seal_hash=SHA,
        global_prelabel_seal_hash=SHA,
    )


def _terminal_score_row(target: str, action_id: str, *, balanced_accuracy: float) -> dict[str, str]:
    payload = {
        "schema_version": "midogpp_consumed_test_terminal_endpoint_score_v1",
        "target_center": target, "action_id": action_id, "action_hash": SHA,
        "policy_hash": SHA, "support_partition_lock_hash": SHA,
        "evaluation_partition_hash": f"evaluation::{target}",
        "global_target_prediction_seal_hash": SHA, "global_prelabel_seal_hash": SHA,
        "evaluation_row_identity_hash": SHA, "evaluation_label_hash": SHA,
        "endpoint_hash": SHA, "evaluation_case_count": 15,
        "evaluation_row_count": 874, "observed_class_0_row_count": 437,
        "observed_class_1_row_count": 437, "balanced_accuracy": balanced_accuracy,
        "primary_endpoint": "exact_nine_probability_mean_then_threshold_bacc",
        "same_outer_H_evaluation_labels_opened_after_plan_and_global_seal": True,
        "terminal_scores_may_update_plan": False, "inference_unit": "target_center",
        "technical_seed_cells_are_independent_units": False,
        "consumed_test_diagnostic_only": True,
    }
    return _csv_row({**payload, "score_hash": canonical_sha256(payload)})


def _feature_values(value: float) -> dict[str, float]:
    names = (
        "reconstruction_mean", "reconstruction_std", "reconstruction_q25",
        "reconstruction_q50", "reconstruction_q75", "kl_mean", "kl_std",
        "kl_q25", "kl_q50", "kl_q75", "replica_disagreement",
        "distribution_mmd", "metadata_similarity",
    )
    return {name: value for name in names}


def _csv_row(payload: dict[str, object]) -> dict[str, str]:
    return {
        key: (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (dict, list, tuple))
            else "" if value is None else str(value)
        )
        for key, value in payload.items()
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
