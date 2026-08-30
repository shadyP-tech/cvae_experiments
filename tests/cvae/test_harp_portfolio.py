from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import struct

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_model import (
    LAMBDA_GRID,
    HarpActionScore,
    HarpSupportCell,
    HarpTargetAction,
    HarpTrainingObservation,
    model_bank_collection_from_payload,
    score_harp_actions,
    training_observation_surface_payload,
)
from midogpp_thesis.cvae.routing.harp_action_surface import (
    ACTION_FEATURE_NAMES,
    HarpActionInferenceBinding,
)
from midogpp_thesis.cvae.routing.harp_portfolio import (
    HarpPolicyConfig,
    select_harp_physical_portfolio,
    select_harp_portfolio,
)
from midogpp_thesis.cvae.routing.harp_portfolio.production import (
    ProductionPolicyLockAdapter,
    policy_input_binding,
)
from midogpp_thesis.cvae.routing.harp_protocol.hashing import canonical_hash
from midogpp_thesis.cvae.routing.harp_stage60 import POLICY_LOCK, load_harp_stage60_config
from midogpp_thesis.cvae.routing.harp_stage60.config import HarpInputReadiness


BASELINE = struct.pack("<d", 0.37)


def _score(
    source: str,
    lam: float,
    *,
    gain: float = 0.1,
    harmful: bool = False,
    rho: float = 1.0,
    reference_bytes: bytes = BASELINE,
    fallback_bytes: bytes | None = None,
) -> HarpActionScore:
    action = HarpTargetAction(
        outer_target_id="H",
        target_query_id="H",
        candidate_source_id=source,
        case_id="case",
        sample_id="sample",
        lambda_value=lam,
        direction="ALL_MARGINS",
        feature_names=("margin", "seed_dispersion"),
        feature_values=(0.1, 0.02),
        baseline_probability_bytes=reference_bytes,
        operational_fallback_probability_bytes=fallback_bytes,
        expert_probability=0.8,
        ensemble_size=9,
        ensemble_receipt_hash="e" * 64,
        prediction_seal_hash="f" * 64,
        compatibility_shrinkage=rho,
    )
    loss = 0.02 if harmful else -0.02
    return HarpActionScore(
        action,
        (gain, gain, gain, gain),
        (loss, loss, loss, loss),
        (loss, loss, loss, loss),
        (0.1, 0.1, 0.1, 0.1),
        HarpSupportCell(source, lam, "ALL_MARGINS", 4, 16, (0, 1)),
        ("0", "1", "2", "3"),
    )


def test_policy_is_deterministic_and_uses_frozen_tie_order() -> None:
    rows = tuple(_score(source, lam) for source in ("B", "A") for lam in LAMBDA_GRID)
    config = HarpPolicyConfig(max_leverage=1.0)
    first = select_harp_portfolio(rows, config=config)
    second = select_harp_portfolio(tuple(reversed(rows)), config=config)
    assert first == second
    assert first[0].selected_source_id == "A"
    assert first[0].selected_lambda == 0.25
    assert first[0].routed


def test_exact_b_fallback_is_byte_identical() -> None:
    rows = tuple(_score(source, lam, gain=-0.1, harmful=True) for source in ("A", "B") for lam in LAMBDA_GRID)
    decision = select_harp_portfolio(rows)[0]
    assert not decision.routed
    assert decision.output_probability_bytes == BASELINE
    assert decision.output_probability_bytes is decision.baseline_probability_bytes


def test_u_is_predictive_reference_while_b_remains_operational_fallback() -> None:
    reference_u = struct.pack("<d", 0.50)
    fallback_b = struct.pack("<d", 0.37)
    rejected = tuple(
        _score(
            source,
            lam,
            gain=-0.1,
            harmful=True,
            reference_bytes=reference_u,
            fallback_bytes=fallback_b,
        )
        for source in ("A", "B")
        for lam in LAMBDA_GRID
    )
    fallback = select_harp_portfolio(rejected)[0]
    assert fallback.output_probability_bytes == fallback_b
    assert fallback.baseline_probability_bytes == fallback_b

    admitted = tuple(
        _score(
            source,
            lam,
            reference_bytes=reference_u,
            fallback_bytes=fallback_b,
        )
        for source in ("A", "B")
        for lam in LAMBDA_GRID
    )
    routed = select_harp_portfolio(admitted)[0]
    assert routed.routed is True
    assert routed.selected_lambda == 0.25
    assert struct.unpack("<d", routed.output_probability_bytes)[0] == pytest.approx(
        0.75 * 0.50 + 0.25 * 0.80
    )


def test_incomplete_lambda_grid_is_rejected() -> None:
    rows = tuple(_score("A", lam) for lam in LAMBDA_GRID[:-1])
    with pytest.raises(ProtocolError, match="complete frozen lambda grid"):
        select_harp_portfolio(rows)


def test_support_envelope_rejects_source_winner_without_reranking_runner_up() -> None:
    rows = tuple(
        _score(
            source,
            lam,
            gain=0.2 if source == "A" else 0.1,
            rho=0.0 if source == "A" else 1.0,
        )
        for source in ("A", "B")
        for lam in LAMBDA_GRID
    )
    decision = select_harp_portfolio(
        rows,
        config=HarpPolicyConfig(
            max_leverage=1.0,
            min_compatibility_shrinkage=0.5,
        ),
    )[0]
    assert decision.routed is False
    assert decision.selected_source_id is None
    assert decision.output_probability_bytes == BASELINE
    assert decision.reason == (
        "EXACT_B_FALLBACK_SUPPORT_ENVELOPE_REJECTED_SOURCE_WINNER"
    )


def test_support_envelope_cannot_reverse_source_evidence_ranking() -> None:
    rows = tuple(
        _score(
            source,
            lam,
            gain=0.2 if source == "A" else 0.1,
            rho=0.25 if source == "A" else 1.0,
        )
        for source in ("A", "B")
        for lam in LAMBDA_GRID
    )
    decision = select_harp_portfolio(
        rows,
        config=HarpPolicyConfig(max_leverage=1.0),
    )[0]
    assert decision.routed is True
    assert decision.selected_source_id == "A"
    assert decision.gain_lower == pytest.approx(0.05)


def test_physical_selector_requires_only_complete_lambda_one_endpoints() -> None:
    rows = tuple(
        _score(source, 1.0, gain=0.2 if source == "A" else 0.1)
        for source in ("A", "B")
    )
    decision = select_harp_physical_portfolio(rows)[0]
    assert decision.routed is True
    assert decision.selected_source_id == "A"
    assert decision.selected_lambda == 1.0

    with pytest.raises(ProtocolError, match="complete frozen lambda grid"):
        select_harp_physical_portfolio(
            tuple(_score("A", lam) for lam in LAMBDA_GRID)
        )


def test_physical_support_veto_falls_back_without_selecting_runner_up() -> None:
    rows = tuple(
        _score(
            source,
            1.0,
            gain=0.2 if source == "A" else 0.1,
            rho=0.0 if source == "A" else 1.0,
        )
        for source in ("A", "B")
    )
    decision = select_harp_physical_portfolio(
        rows,
        config=HarpPolicyConfig(min_compatibility_shrinkage=0.5),
    )[0]
    assert decision.routed is False
    assert decision.selected_source_id is None
    assert decision.selected_lambda is None
    assert decision.output_probability_bytes == BASELINE
    assert decision.reason == (
        "EXACT_B_FALLBACK_SUPPORT_ENVELOPE_REJECTED_SOURCE_WINNER"
    )


def test_selection_surface_has_no_target_truth_parameter() -> None:
    parameters = inspect.signature(select_harp_portfolio).parameters
    assert tuple(parameters) == ("scores", "config")
    assert tuple(inspect.signature(select_harp_physical_portfolio).parameters) == (
        "scores",
        "config",
    )


def _training_rows() -> tuple[HarpTrainingObservation, ...]:
    centers = ("0", "1", "2", "3", "4", "H")
    rows: list[HarpTrainingObservation] = []
    for outer in centers:
        donors = tuple(center for center in centers if center != outer)
        for q_index, query in enumerate(donors):
            for e_index, source in enumerate(donors):
                if query == source:
                    continue
                for case_index in range(4):
                    for sample_index in range(2):
                        for lam in LAMBDA_GRID:
                            feature_values = [0.0] * len(ACTION_FEATURE_NAMES)
                            feature_values[0] = 0.4
                            feature_values[1] = 0.4 + 0.01 * (e_index - q_index)
                            feature_values[2] = (1.0 - lam) * feature_values[0] + lam * feature_values[1]
                            feature_values[12] = 0.02 + 0.001 * sample_index
                            feature_values[13] = lam
                            feature = tuple(feature_values)
                            rows.append(HarpTrainingObservation(
                                outer_target_id=outer, pseudo_query_id=query,
                                candidate_source_id=source, case_id=f"case-{case_index}",
                                sample_id=f"sample-{sample_index}", lambda_value=lam,
                                direction="ALL_MARGINS", feature_names=ACTION_FEATURE_NAMES,
                                feature_values=feature,
                                weighted_correctness_surrogate=0.04 + 0.03 * lam + 0.01 * feature[0],
                                brier_delta=-0.02 * lam, log_loss_delta=-0.03 * lam,
                                truth_class=sample_index, ensemble_size=9,
                                ensemble_receipt_hash="1" * 64, prediction_seal_hash="2" * 64,
                                case_aggregation_receipt_hash="0" * 64,
                                response_receipt_hash="3" * 64,
                            ))
    return tuple(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _target_support_surface(centers: tuple[str, ...]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for outer in centers:
        for source in centers:
            if source == outer:
                continue
            for case_index in range(2):
                sample = f"support-{outer}-{source}-{case_index}"
                case = f"support-case-{outer}-{case_index}"
                for lam in LAMBDA_GRID:
                    values = [0.0] * len(ACTION_FEATURE_NAMES)
                    values[0] = 0.4
                    values[1] = 0.6
                    values[2] = 0.4 + 0.2 * lam
                    values[12] = 0.02
                    values[13] = lam
                    unhashed: dict[str, object] = {
                        "outer_target": outer,
                        "candidate_source": source,
                        "case_id": case,
                        "sample_id": sample,
                        "case_sample_ids": [sample],
                        "action_lambda": lam,
                        "direction": "ALL_MARGINS",
                        "feature_names": list(ACTION_FEATURE_NAMES),
                        "feature_values": values,
                        "baseline_probability": 0.4,
                        "expert_probability": 0.6,
                        "action_probability": 0.4 + 0.2 * lam,
                        "ensemble_receipt_hash": "a" * 64,
                        "case_weight_receipt_hash": "b" * 64,
                        "seed_count": 9,
                        "label_free": True,
                    }
                    rows.append({**unhashed, "feature_hash": canonical_hash(unhashed)})
    rows.sort(
        key=lambda row: (
            str(row["outer_target"]),
            str(row["candidate_source"]),
            str(row["case_id"]),
            str(row["sample_id"]),
            float(row["action_lambda"]),
        )
    )
    prediction_menu_hash = "c" * 64
    surface_hash = canonical_hash(
        {
            "schema_version": "midogpp_harp_target_support_feature_surface_v2",
            "prediction_menu_hash": prediction_menu_hash,
            "feature_hashes": [row["feature_hash"] for row in rows],
            "seed_cells_may_feed_model": False,
            "target_support_labels_used": False,
            "predictive_reference_action_id": "U",
        }
    )
    return {
        "schema_version": "midogpp_harp_target_support_feature_artifact_v2",
        "surface_hash": surface_hash,
        "prediction_menu_hash": prediction_menu_hash,
        "feature_names": list(ACTION_FEATURE_NAMES),
        "lambda_grid": list(LAMBDA_GRID),
        "row_count": len(rows),
        "rows": rows,
        "seed_cells_may_feed_model": False,
        "target_support_labels_used": False,
        "target_evaluation_labels_used": False,
        "predictive_reference_action_id": "U",
        "probability_ensemble_semantics": "post_classifier_predictive_p_lambda=(1-lambda)*p_U+lambda*p_Hxe",
        "lambda_one_is_physical_hxe_endpoint": True,
    }


def test_production_policy_lock_reloads_models_scores_and_preserves_exact_b(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "experiments/midogpp/stages/60_routing_and_composition/configs/uniform_b_v2_harp_policy_lock_v1.yaml"
    base = load_harp_stage60_config(config_path)
    action_root = tmp_path / "action"
    support_root = tmp_path / "support"
    exact_root = tmp_path / "exact"
    support_reservation = tmp_path / "support-reservation"
    fresh_reservation = tmp_path / "fresh-reservation"
    paths = {
        "action_surface_root": action_root,
        "exact_b_policy_root": exact_root,
        "target_support_surface_root": support_root,
        "target_support_reservation_root": support_reservation,
        "fresh_target_reservation_root": fresh_reservation,
        "readiness_attestation_path": fresh_reservation / "manifests/harp_policy_inputs_attestation.json",
    }
    config = replace(
        base,
        artifact_root=tmp_path / "output",
        input_paths=paths,
        protocol={
            **dict(base.protocol),
            "center_universe": ("0", "1", "2", "3", "4", "H"),
            "input_status": "ready",
        },
        model={
            **dict(base.model), "ridge_alphas": (0.1,), "minimum_positive_gain": 999.0,
            "maximum_leverage": 1.0, "minimum_compatibility_shrinkage": 0.0,
        },
        runtime={**dict(base.runtime), "model_workers": 1, "model_threads_per_worker": 1},
    )
    for surface_root, surface, experiment in (
        (action_root, "uniform-b-v2-harp-action-surface", "action"),
        (support_root, "uniform-b-v2-harp-target-support-surface", "support"),
    ):
        _write_json(surface_root / "reports/run_state.json", {
            "schema_version": "midogpp_harp_run_state_v1", "status": "COMPLETE",
            "surface": surface, "experiment_id": experiment,
            "product_hash": "4" * 64, "validation_hash": "5" * 64,
            "target_support_labels_used": False, "target_evaluation_labels_used": False,
        })
    training = training_observation_surface_payload(_training_rows(), feature_surface_hash="6" * 64, response_surface_hash="7" * 64)
    _write_json(action_root / "surfaces/harp_training_observations.json", training)
    source_binding = canonical_hash(
        {
            "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
            "source_cache_lock_sha256": "3" * 64,
            "source_cache_index_sha256": "4" * 64,
            "source_stream_content_hash": "c" * 64,
        }
    )
    binding = HarpActionInferenceBinding(
        expert_bank_semantic_id="8" * 16,
        generation_semantic_id="9" * 16,
        source_stream_lock_semantic_id="a" * 16,
        source_stream_index_semantic_id="b" * 16,
        source_stream_content_semantic_id="c" * 64,
        classifier_config_semantic_id="d" * 16,
        source_stream_artifact_binding_semantic_id=source_binding,
        classifier_contract_semantic_id="f" * 64,
        global_prediction_seal_semantic_id="a" * 64,
        feature_surface_semantic_id="6" * 64,
        response_surface_semantic_id="7" * 64,
        expert_bank_index_file_sha256="1" * 64,
        generation_lock_file_sha256="2" * 64,
        source_cache_lock_file_sha256="3" * 64,
        source_cache_index_file_sha256="4" * 64,
    )
    _write_json(
        action_root / "manifests/harp_action_inference_binding.json",
        binding.to_payload(),
    )
    _write_json(
        support_root / "surfaces/target_support_features.json",
        _target_support_surface(("0", "1", "2", "3", "4", "H")),
    )
    exact_unhashed = {"schema_version": "midogpp_uniform_b_v2_equal_union_policy_lock_v1", "upstreams": {"generation_lock_hash": "d" * 64}}
    _write_json(exact_root / "manifests/policy_lock.json", {**exact_unhashed, "policy_lock_hash": canonical_hash(exact_unhashed)})
    for reservation_root, artifact_id in (
        (support_reservation, "midogpp_harp_target_support_reservation_v1"),
        (fresh_reservation, "midogpp_harp_fresh_target_reservation_v1"),
    ):
        reservation_unhashed = {
            "schema_version": f"{artifact_id}", "artifact_id": artifact_id,
            "dataset_family": "MIDOG++", "status": "ACTIVE",
            "fresh_unconsumed_surface": True, "labels_opened": False,
            "labels_present": False, "target_evaluation_rows_present": False,
            "consumed_test_used": False, "consumed_validation_used": False,
            "consumed_stage70_used": False, "consumed_stage90_used": False,
        }
        _write_json(reservation_root / "manifests/reservation.json", {**reservation_unhashed, "reservation_hash": canonical_hash(reservation_unhashed)})
    input_hash, reservation_hash, _members = policy_input_binding(config)
    readiness = HarpInputReadiness(POLICY_LOCK.surface, POLICY_LOCK.experiment_id, input_hash, reservation_hash, "d" * 64, "e" * 64, "f" * 64)
    adapter = ProductionPolicyLockAdapter()
    adapter.preflight(config, readiness)
    seal = adapter.materialize_and_seal_label_free_menu(config, readiness)
    product = adapter.build_product(config, seal, None)
    adapter.persist_product(config, seal, product)
    receipt = adapter.validate_completed_bundle(config)
    assert receipt.status == "COMPLETE"
    for member in (
        "manifests/model_lock.json", "manifests/delete_donor_lock.json",
        "manifests/action_library.json", "manifests/target_policy_lock.json",
        "manifests/policy_lock.json", "manifests/content_index.json",
        "reports/leakage_report.json", "reports/validation_report.json", "reports/run_state.json",
    ):
        assert (config.artifact_root / member).is_file()

    lock = json.loads((config.artifact_root / "manifests/policy_lock.json").read_text(encoding="utf-8"))
    bank = next(
        value
        for value in model_bank_collection_from_payload(lock["model_bank_collection"])
        if value.outer_target_id == "H"
    )
    actions = tuple(HarpTargetAction(
        outer_target_id="H", target_query_id="H", candidate_source_id="0",
        case_id="target-case", sample_id="target-sample", lambda_value=lam,
        direction="ALL_MARGINS", feature_names=bank.feature_names,
        feature_values=tuple(
            0.25 if name == "action_lambda" else 0.02 if name == "seed_dispersion" else 0.1
            for name in bank.feature_names
        ), baseline_probability_bytes=BASELINE,
        expert_probability=0.8, ensemble_size=9, ensemble_receipt_hash="e" * 64,
        prediction_seal_hash="f" * 64,
    ) for lam in LAMBDA_GRID)
    scores = score_harp_actions(bank, actions)
    decision = select_harp_portfolio(scores, config=HarpPolicyConfig(
        gain_threshold=float(lock["minimum_positive_gain"]), max_leverage=float(lock["maximum_leverage"]),
        min_compatibility_shrinkage=float(lock["minimum_compatibility_shrinkage"]),
    ))[0]
    assert not decision.routed
    assert decision.output_probability_bytes == BASELINE

    # Completed fast paths must re-hash every authoritative member, including
    # the run-state commit marker, before accepting the lock.
    state_path = config.artifact_root / "reports/run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["experiment_id"] = "tampered"
    _write_json(state_path, state)
    with pytest.raises(ProtocolError, match="content index"):
        adapter.validate_completed_bundle(config)
