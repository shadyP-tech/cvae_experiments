"""Content-first, non-repairing validation for a completed ceiling bundle."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np

from ...generation.contracts import EXPECTED_GENERATION_LOCK_HASH
from ...protocol import ProtocolError
from ...runtime.frozen_source_streams import load_frozen_source_streams
from ...runtime.label_free_action_predictions import load_global_prediction_seal
from .artifact_io import read_json
from .bundle import assert_closed_world, validate_content_index
from .core_hashing import canonical_hash, require_sha256
from .experiment_contracts import CENTERS, EXPECTED_CENTER_FOLD_COUNT
from .permutation_plan import PermutationDecisionPlan


_PERMUTATION_DECISION_TIE_BREAK = (
    "lexicographic_action_id_no_evaluation_utility_access"
)


def validate_fixed_bank_label_aware_case_oof_ceiling_bundle(
    root: Path,
    *,
    config: object,
) -> Mapping[str, object]:
    """Validate the content index before opening scientific result JSON."""

    assert_closed_world(root, allow_incomplete=False, allow_pending_validation=True)
    content = validate_content_index(root, config_contract_hash=str(config.contract_hash))

    protocol = read_json(root / "manifests/protocol_manifest.json")
    partition = read_json(root / "manifests/case_oof_partition.json")
    probability_surface = read_json(root / "manifests/sealed_probability_surface.json")
    priors = read_json(root / "manifests/loco_global_prior_seals.json")
    posteriors = read_json(root / "manifests/fold_posterior_seals.json")
    decisions = read_json(root / "manifests/fold_decisions.json")
    decision_seal = read_json(root / "manifests/all_fold_decisions_seal.json")
    permutation_seal = read_json(root / "manifests/permutation_null_decision_seal.json")
    evaluation = read_json(root / "manifests/ceiling_evaluation.json")
    capability = read_json(root / "reports/label_capability_report.json")
    leakage = read_json(root / "reports/leakage_report.json")
    publication = read_json(root / "reports/publication_decision.json")
    runtime = read_json(root / "reports/runtime_summary.json")
    action_selection_table = _read_csv(
        root / "tables/action_selection_metrics.csv"
    )

    source_cache = load_frozen_source_streams(
        root,
        expected_config_hash=str(config.contract_hash),
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    predictions = load_global_prediction_seal(
        root,
        expected_config_hash=str(config.contract_hash),
        expected_source_lock_hash=source_cache.lock_hash,
        expected_partition_lock_hash=str(partition.get("partition_hash", "")),
        expected_target_cache_binding_hash=str(
            protocol.get("test_cache_binding_hash", "")
        ),
    )
    require_sha256(str(partition.get("partition_hash", "")), "partition_hash")
    require_sha256(str(probability_surface.get("surface_hash", "")), "probability_surface_hash")
    permutation_array_path = root / "arrays/permutation_null_actions.npy"
    permutation_actions = np.load(permutation_array_path, allow_pickle=False)
    permutation_hash = str(
        permutation_seal.get(
            "permutation_decision_seal_hash",
            permutation_seal.get(
                "permutation_seal_hash", permutation_seal.get("plan_hash", "")
            ),
        )
    )
    require_sha256(permutation_hash, "permutation_decision_seal_hash")
    raw_priors = priors.get("priors")
    raw_posteriors = posteriors.get("posteriors")
    raw_decisions = decisions.get("decisions")
    if (
        protocol.get("experiment_id") != config.experiment_id
        or protocol.get("config_contract_hash") != config.contract_hash
        or protocol.get("support_labels_used") is not True
        or protocol.get("evaluation_labels_inaccessible_until_all_decisions_sealed") is not True
        or partition.get("fold_count") != 5
        or len(partition.get("folds", [])) != EXPECTED_CENTER_FOLD_COUNT
        or partition.get("evaluation_case_coverage_exactly_once") is not True
        or probability_surface.get("global_prediction_seal_hash") != predictions.seal_hash
        or probability_surface.get("probability_store_hash") != predictions.store.store_hash
        or predictions.store.target_cache_binding_hash
        != protocol.get("test_cache_binding_hash")
        or predictions.seal_payload.get("target_cache_binding_hash")
        != protocol.get("test_cache_binding_hash")
        or probability_surface.get("predictions_globally_sealed_before_labels") is not True
        or not isinstance(raw_priors, list)
        or len(raw_priors) != len(CENTERS)
        or not isinstance(raw_posteriors, list)
        or len(raw_posteriors) != EXPECTED_CENTER_FOLD_COUNT
        or not isinstance(raw_decisions, list)
        or len(raw_decisions) != EXPECTED_CENTER_FOLD_COUNT
        or decision_seal.get("fold_decision_count") != EXPECTED_CENTER_FOLD_COUNT
        or decision_seal.get("all_fold_decisions_sealed_before_evaluation_labels") is not True
        or capability.get("status") != "PASS"
        or capability.get("evaluation_labels_opened") is not True
        or capability.get("fold_decision_count") != EXPECTED_CENTER_FOLD_COUNT
        or leakage.get("status") != "PASS"
        or leakage.get("H_labels_used_in_G_H") is not False
        or leakage.get("G_H_shared_across_H") is not False
        or leakage.get("target_expert_used") is not False
        or publication.get("decision") != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or publication.get("promotion_eligible") is not False
        or publication.get("may_feed_another_stage90") is not False
        or publication.get("scientific_result_hash") != evaluation.get("scientific_result_hash")
        or runtime.get("classifier_cell_count") != 729
        or evaluation.get("consumed_test_data") is not True
        or evaluation.get("fresh_evidence") is not False
        or evaluation.get("diagnostic_only") is not True
        or evaluation.get("may_feed_later_stage") is not False
        or permutation_actions.shape != (10_000, EXPECTED_CENTER_FOLD_COUNT)
        or permutation_actions.dtype != np.uint8
        or np.any(permutation_actions > 8)
        or permutation_seal.get("action_array_member") != "arrays/permutation_null_actions.npy"
        or permutation_seal.get("action_array_sha256") != _sha256_file(permutation_array_path)
        or permutation_seal.get("action_array_value_sha256") != _sha256_array(permutation_actions)
        or permutation_seal.get("action_codes_sha256")
        != hashlib.sha256(permutation_actions.tobytes(order="C")).hexdigest()
        or permutation_seal.get("partition_hash") != partition.get("partition_hash")
        or permutation_seal.get("probability_surface_hash") != probability_surface.get("surface_hash")
        or permutation_seal.get("observed_decision_seal_hash") != decision_seal.get("decision_seal_hash")
        or permutation_seal.get("config_contract_hash") != config.contract_hash
        or permutation_seal.get("sealed_before_evaluation_labels") is not True
        or permutation_seal.get("evaluation_labels_used_to_generate_actions") is not False
        or permutation_seal.get("permutation_decision_tie_break")
        != _PERMUTATION_DECISION_TIE_BREAK
        or permutation_seal.get("evaluation_utility_used_for_permutation_tie_break")
        is not False
    ):
        raise ProtocolError("Label-aware scientific bundle invariants drifted.")
    _validate_action_selection_table(
        action_selection_table,
        evaluation.get("action_selection_rows"),
        total_case_count=evaluation.get("total_case_count"),
    )
    prior_targets = tuple(str(value.get("target_center")) for value in raw_priors if isinstance(value, Mapping))
    posterior_keys = tuple(
        (str(value.get("target_center")), int(value.get("fold_ordinal", -1)))
        for value in raw_posteriors
        if isinstance(value, Mapping)
    )
    decision_keys = tuple(
        (str(value.get("target_center")), int(value.get("fold_ordinal", -1)))
        for value in raw_decisions
        if isinstance(value, Mapping)
    )
    expected_keys = tuple((center, fold) for center in CENTERS for fold in range(5))
    if prior_targets != CENTERS or posterior_keys != expected_keys or decision_keys != expected_keys:
        raise ProtocolError("Label-aware prior/posterior/decision order drifted.")
    for prior in raw_priors:
        _validate_direct_hash(prior, "prior_hash")
        if prior.get("H_labels_used_in_G_H") is not False or prior.get("G_H_shared_across_H") is not False:
            raise ProtocolError("Label-aware G_H claim flags drifted.")
    for posterior in raw_posteriors:
        _validate_direct_hash(posterior, "posterior_hash")
        if posterior.get("evaluation_labels_used") is not False or posterior.get("smooth_response_used") is not False:
            raise ProtocolError("Fold posterior label/smooth boundary drifted.")
    decision_seal_hash = str(decision_seal.get("decision_seal_hash", ""))
    _validate_direct_hash(decision_seal, "decision_seal_hash")
    _validate_direct_hash(evaluation, "scientific_result_hash")
    PermutationDecisionPlan(
        action_codes=np.ascontiguousarray(permutation_actions, dtype=np.uint8),
        permutation_seed=int(permutation_seal["permutation_seed"]),
        permutation_count=int(permutation_seal["permutation_count"]),
        fold_keys=tuple(
            (str(value[0]), int(value[1])) for value in permutation_seal["fold_keys"]
        ),
        partition_hash=str(permutation_seal["partition_hash"]),
        probability_surface_hash=str(permutation_seal["probability_surface_hash"]),
        support_input_hash=str(permutation_seal["support_input_hash"]),
        plan_hash=str(permutation_seal["plan_hash"]),
    )
    if capability.get("all_decisions_seal_hash") != decision_seal_hash:
        raise ProtocolError("Label capability differs from the all-decision seal.")
    if capability.get("permutation_decision_seal_hash") != permutation_hash:
        raise ProtocolError("Label capability differs from the permutation-decision seal.")

    return {
        "schema_version": "midogpp_label_aware_case_oof_validation_v1",
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": config.contract_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "global_prediction_seal_hash": predictions.seal_hash,
        "probability_surface_hash": probability_surface["surface_hash"],
        "decision_seal_hash": decision_seal_hash,
        "permutation_decision_seal_hash": permutation_hash,
        "loco_prior_count": len(raw_priors),
        "fold_posterior_count": len(raw_posteriors),
        "fold_decision_count": len(raw_decisions),
        "content_index_validated_before_scientific_members": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "terminal_consumed_test_diagnostic_only": True,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_stage90": False,
    }


def _validate_direct_hash(payload: object, hash_field: str) -> None:
    if not isinstance(payload, Mapping):
        raise ProtocolError("Hashed label-aware record is malformed.")
    unhashed = {str(key): value for key, value in payload.items() if key != hash_field}
    if payload.get(hash_field) != canonical_hash(unhashed):
        raise ProtocolError(f"Label-aware {hash_field} drifted.")


def _sha256_file(path: Path) -> str:
    from ...runtime.artifact_io import sha256_file

    return sha256_file(path)


def _sha256_array(values: np.ndarray) -> str:
    from ...runtime.artifact_io import sha256_array

    return sha256_array(values)


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolError(f"Cannot read label-aware table: {path}.") from exc


def _validate_action_selection_table(
    observed: tuple[dict[str, str], ...], expected: object, *, total_case_count: object
) -> None:
    expected_keys = tuple(
        (method, action)
        for method in ("G_H", "R")
        for action in ("B", *CENTERS)
    )
    if (
        not isinstance(expected, list)
        or len(expected) != len(expected_keys)
        or isinstance(total_case_count, bool)
        or not isinstance(total_case_count, int)
        or total_case_count <= 0
    ):
        raise ProtocolError("Action-selection evaluation rows are absent.")
    if (
        len(observed) != len(expected_keys)
        or tuple((row.get("method_id"), row.get("action_id")) for row in observed)
        != expected_keys
        or tuple(
            (str(row.get("method_id")), str(row.get("action_id")))
            for row in expected
            if isinstance(row, Mapping)
        )
        != expected_keys
    ):
        raise ProtocolError("Action-selection table coverage/order drifted.")
    try:
        for csv_row, json_row in zip(observed, expected, strict=True):
            if not isinstance(json_row, Mapping):
                raise ProtocolError("Action-selection evaluation row is malformed.")
            if (
                csv_row.get("row_hash") != str(json_row.get("row_hash"))
                or int(csv_row.get("selection_count", -1))
                != int(json_row.get("selection_count", -2))
                or int(csv_row.get("total_case_count", -1))
                != int(json_row.get("total_case_count", -2))
                or int(json_row.get("total_case_count", -1)) != total_case_count
                or float(csv_row.get("selection_share", "nan"))
                != float(json_row.get("selection_share", "nan"))
            ):
                raise ProtocolError(
                    "Action-selection table differs from sealed evaluation."
                )
        for method in ("G_H", "R"):
            if sum(
                int(row["selection_count"])
                for row in expected
                if isinstance(row, Mapping) and row.get("method_id") == method
            ) != total_case_count:
                raise ProtocolError("Action-selection counts lack exact case coverage.")
    except (TypeError, ValueError, KeyError) as exc:
        raise ProtocolError("Action-selection table is malformed.") from exc


__all__ = ("validate_fixed_bank_label_aware_case_oof_ceiling_bundle",)
