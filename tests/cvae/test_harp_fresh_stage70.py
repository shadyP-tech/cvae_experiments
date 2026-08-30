from __future__ import annotations

from dataclasses import fields
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.frozen_policy_downstream.fresh_runtime_contract import (
    DOWNSTREAM_CLASSIFIER,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh import (
    HarpFreshEvaluationCapability,
    HarpFreshPredictionOutput,
    HarpFreshReservation,
    HarpFreshRunner,
    HarpFreshStage70Config,
    HarpFreshTargetCache,
    HarpFreshTargetFrame,
    HarpFrozenPolicyMetadata,
    bind_frozen_harp_policy,
    issue_harp_fresh_evaluation_capability,
    load_frozen_harp_policy,
    load_harp_fresh_stage70_config,
    load_harp_fresh_target,
    materialize_harp_fresh_probability_menu,
    reconstruct_frozen_harp_policy_receipt,
    score_harp_fresh_routes,
    select_and_seal_harp_fresh_routes,
    validate_harp_fresh_completed_bundle,
    write_harp_fresh_content_index,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh.config import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    PATH_KEYS,
    canonical_claim_boundary_payload,
    canonical_evaluation_payload,
    canonical_harp_runtime_payload,
    canonical_protocol_payload,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh.production_prediction import (
    _task_plan,
)
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh import bundle as harp_bundle
from midogpp_thesis.cvae.frozen_policy_downstream.harp_fresh.validation import (
    _validate_prediction_checkpoints,
)
from midogpp_thesis.cvae.generation.contracts import COMMON_OUTPUT_DIM
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_model import (
    HarpActionModelBank,
    HarpLodoFoldAudit,
    HarpOutcomeModel,
    HarpRidgeModel,
    HarpSupportCell,
    model_bank_collection_payload,
)
from midogpp_thesis.cvae.routing.harp_action_surface import (
    ACTION_FEATURE_NAMES,
    HarpActionInferenceBinding,
)
from midogpp_thesis.cvae.routing.harp_action_surface.lineage import (
    HarpAuthoritativeLineage,
)
from midogpp_thesis.cvae.routing.harp_portfolio.support_envelope import (
    HarpSupportEnvelope,
    HarpSupportEnvelopeCell,
)
from midogpp_thesis.cvae.routing.harp_protocol.hashing import canonical_hash
from midogpp_thesis.cvae.runtime.harp_probability_menu import LAMBDA_GRID
from midogpp_thesis.cvae.runtime.harp_probability_menu.hashing import raw_array_sha256


_SELECTION_ORDER = [
    "gain_lower_desc",
    "brier_upper_asc",
    "log_loss_upper_asc",
    "lambda_asc",
    "source_id_asc",
]


def _ridge(
    *, candidate_levels: tuple[str, ...], excluded: tuple[str, ...], intercept: float
) -> HarpRidgeModel:
    dimension = 1 + len(ACTION_FEATURE_NAMES) + len(candidate_levels)
    coefficients = np.zeros(dimension, dtype=np.float64)
    coefficients[0] = intercept
    retained = tuple(center for center in CENTERS if center not in excluded)
    return HarpRidgeModel(
        feature_names=ACTION_FEATURE_NAMES,
        candidate_levels=candidate_levels,
        feature_mean=np.zeros(len(ACTION_FEATURE_NAMES), dtype=np.float64),
        feature_scale=np.ones(len(ACTION_FEATURE_NAMES), dtype=np.float64),
        coefficients=coefficients,
        normal_inverse=np.zeros((dimension, dimension), dtype=np.float64),
        alpha=1.0,
        training_query_ids=retained,
        training_source_ids=retained,
        training_case_ids=("source-case",),
        excluded_donor_ids=excluded,
    )


def _bank(outer: str, *, favorable: bool = True) -> HarpActionModelBank:
    sources = tuple(center for center in CENTERS if center != outer)
    donors = sources[:4]
    models = []
    for outcome, intercept in (
        ("gain", 0.2 if favorable else -0.2),
        ("brier", -0.1 if favorable else 0.1),
        ("log_loss", -0.1 if favorable else 0.1),
    ):
        models.append(
            HarpOutcomeModel(
                outcome,
                "ALL_MARGINS",
                _ridge(candidate_levels=sources, excluded=(), intercept=intercept),
                tuple(
                    (
                        donor,
                        _ridge(
                            candidate_levels=tuple(
                                source for source in sources if source != donor
                            ),
                            excluded=(donor,),
                            intercept=intercept,
                        ),
                    )
                    for donor in donors
                ),
                (
                    HarpLodoFoldAudit(
                        donors[0],
                        tuple(center for center in CENTERS if center != donors[0]),
                        tuple(center for center in CENTERS if center != donors[0]),
                        1.0,
                        0.0,
                    ),
                ),
            )
        )
    supports = tuple(
        HarpSupportCell(source, lam, "ALL_MARGINS", 4, 16, (0, 1))
        for source in sources
        for lam in LAMBDA_GRID
    )
    return HarpActionModelBank(
        outer,
        ACTION_FEATURE_NAMES,
        ("1" * 64,),
        ("2" * 64,),
        tuple(models),
        supports,
    )


def _policy_payload(reservation_hash: str, *, favorable: bool = True) -> dict[str, object]:
    banks = tuple(_bank(center, favorable=favorable) for center in CENTERS)
    library: dict[str, object] = {
        "schema_version": "midogpp_harp_action_library_v2",
        "candidate_sources_by_target": {
            target: [source for source in CENTERS if source != target]
            for target in CENTERS
        },
        "lambda_grid": list(LAMBDA_GRID),
        "directions": ["D01", "D10", "ALL_MARGINS"],
        "feature_names": list(ACTION_FEATURE_NAMES),
        "probability_endpoint": "exact_nine_seed_ensemble_float64",
        "predictive_reference_action_id": "U",
        "operational_fallback_action_id": "B",
        "lambda_semantics": "post_classifier_predictive_probability_ensemble_not_generated_distribution",
        "lambda_one_is_physical_hxe_endpoint": True,
        "selection_order": _SELECTION_ORDER,
    }
    library["action_library_hash"] = canonical_hash(library)
    source_content_hash = "d" * 64
    source_lock_sha256 = "3" * 64
    source_index_sha256 = "4" * 64
    source_artifact_binding = canonical_hash(
        {
            "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
            "source_cache_lock_sha256": source_lock_sha256,
            "source_cache_index_sha256": source_index_sha256,
            "source_stream_content_hash": source_content_hash,
        }
    )
    binding = HarpActionInferenceBinding(
        expert_bank_semantic_id="9" * 16,
        generation_semantic_id="a" * 16,
        source_stream_lock_semantic_id="b" * 16,
        source_stream_index_semantic_id="c" * 16,
        source_stream_content_semantic_id=source_content_hash,
        classifier_config_semantic_id="e" * 16,
        source_stream_artifact_binding_semantic_id=source_artifact_binding,
        classifier_contract_semantic_id="5" * 64,
        global_prediction_seal_semantic_id="6" * 64,
        feature_surface_semantic_id="7" * 64,
        response_surface_semantic_id="8" * 64,
        expert_bank_index_file_sha256="1" * 64,
        generation_lock_file_sha256="2" * 64,
        source_cache_lock_file_sha256=source_lock_sha256,
        source_cache_index_file_sha256=source_index_sha256,
    )
    envelope = HarpSupportEnvelope(
        support_surface_semantic_id="f" * 64,
        maximum_allowed_leverage=1.0,
        cells=tuple(
            HarpSupportEnvelopeCell(
                outer_target_id=outer,
                candidate_source_id=source,
                q95_case_max_leverage=0.5,
                maximum_case_leverage=0.5,
                compatibility_shrinkage=1.0,
                case_count=2,
                row_count=8,
            )
            for outer in CENTERS
            for source in CENTERS
            if source != outer
        ),
    )
    payload: dict[str, object] = {
        "schema_version": "midogpp_harp_policy_lock_v2",
        "artifact_id": "midogpp_output_uniform_b_v2_harp_policy_lock_v1",
        "experiment_id": "midogpp.routing_and_composition.uniform_b_v2_harp_policy_lock.v1",
        "status": "FROZEN_BEFORE_TARGET_EVALUATION",
        "dataset_family": "MIDOG++",
        "config_contract_hash": "3" * 64,
        "prelabel_seal_hash": "4" * 64,
        "action_surface_product_hash": "5" * 64,
        "target_support_surface_product_hash": "6" * 64,
        "exact_b_policy_lock_hash": "7" * 16,
        "support_reservation_hash": "8" * 64,
        "fresh_target_reservation_hash": reservation_hash,
        "lambda_grid": list(LAMBDA_GRID),
        "ridge_alphas": [0.1],
        "gain_kappa": 1.0,
        "loss_kappa": 1.0,
        "minimum_paired_cases": 16,
        "minimum_donor_centers": 4,
        "minimum_truth_classes": 2,
        "minimum_positive_gain": 0.0,
        "maximum_brier_delta": 0.0,
        "maximum_log_loss_delta": 0.0,
        "maximum_leverage": 1.0,
        "minimum_compatibility_shrinkage": 0.0,
        "probability_endpoint": "exact_nine_seed_ensemble",
        "matched_budget_reference_action": "U",
        "utility_deltas_reference_action": "U",
        "lambda_semantics": "post_classifier_predictive_probability_ensemble_not_generated_distribution",
        "physical_expert_routing_primary_lambda": 1.0,
        "operational_fallback_action": "B",
        "case_equal_weighting": True,
        "delete_donor_predictions": True,
        "proper_loss_noninferiority": True,
        "exact_b_byte_identical_fallback": True,
        "policy_accepts_outcomes": False,
        "target_support_outcomes_used": False,
        "target_support_feature_geometry_used_for_shrink_only": True,
        "support_predicted_outcomes_used": False,
        "target_evaluation_outcomes_used": False,
        "stage50_artifacts_used": False,
        "stage90_artifacts_used": False,
        "action_inference_binding": binding.to_payload(),
        "action_inference_binding_sha256": binding.binding_sha256,
        "support_compatibility_envelope": envelope.to_payload(),
        "support_compatibility_envelope_sha256": envelope.envelope_sha256,
        "model_bank_collection": model_bank_collection_payload(banks),
        "action_library": library,
    }
    payload["policy_lock_hash"] = canonical_hash(payload)
    return payload


def _write_policy(tmp_path: Path, reservation_hash: str, *, favorable: bool = True) -> Path:
    root = tmp_path / "policy"
    path = root / "manifests/policy_lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(_policy_payload(reservation_hash, favorable=favorable), sort_keys=True),
        encoding="utf-8",
    )
    return root


def _fresh_cache() -> HarpFreshTargetCache:
    support = {center: (f"support-{center}",) for center in CENTERS}
    evaluation = {
        center: (f"negative-{center}", f"positive-{center}") for center in CENTERS
    }
    reservation = HarpFreshReservation(
        reservation_id="fresh-reservation",
        support_case_ids_by_center=support,
        evaluation_case_ids_by_center=evaluation,
    )
    frames = {
        center: HarpFreshTargetFrame(
            center=center,
            embeddings=np.zeros((2, COMMON_OUTPUT_DIM), dtype=np.float32),
            row_ids=(f"row-{center}-0", f"row-{center}-1"),
            case_ids=evaluation[center],
        )
        for center in CENTERS
    }
    return HarpFreshTargetCache(reservation=reservation, frames_by_center=frames)


def _config_payload() -> dict[str, object]:
    paths = {
        key: value
        for key, value in zip(
            PATH_KEYS,
            (
                f"artifact://{INPUT_ARTIFACT_IDS[0]}",
                f"artifact://{INPUT_ARTIFACT_IDS[1]}",
                f"artifact://{INPUT_ARTIFACT_IDS[2]}",
                f"artifact://{INPUT_ARTIFACT_IDS[3]}",
                f"artifact://{INPUT_ARTIFACT_IDS[4]}",
                f"artifact://{INPUT_ARTIFACT_IDS[5]}/manifest.csv",
            ),
            strict=True,
        )
    }
    return {
        "schema_version": CONFIG_SCHEMA,
        "experiment": {
            "id": EXPERIMENT_ID,
            "artifact_root": f"output://{OUTPUT_ARTIFACT_ID}",
            "output_artifact_id": OUTPUT_ARTIFACT_ID,
        },
        "inputs": {"artifact_ids": list(INPUT_ARTIFACT_IDS), "paths": paths},
        "protocol": canonical_protocol_payload(),
        "classifier": DOWNSTREAM_CLASSIFIER.to_payload(),
        "runtime": canonical_harp_runtime_payload(),
        "evaluation": canonical_evaluation_payload(),
        "claim_boundary": canonical_claim_boundary_payload(),
    }


def test_config_hash_is_path_independent_and_locations_are_closed(tmp_path: Path) -> None:
    payload = _config_payload()
    left = tmp_path / "left/config.yaml"
    right = tmp_path / "right/config.yaml"
    left.parent.mkdir()
    right.parent.mkdir()
    text = yaml.safe_dump(payload, sort_keys=False)
    left.write_text(text, encoding="utf-8")
    right.write_text(text, encoding="utf-8")
    first = load_harp_fresh_stage70_config(left)
    second = load_harp_fresh_stage70_config(right)
    assert isinstance(first, HarpFreshStage70Config)
    assert first.contract_hash == second.contract_hash
    assert first.source_path != second.source_path
    drifted = _config_payload()
    drifted["inputs"]["paths"]["policy_root"] = "/tmp/metadata-only"  # type: ignore[index]
    left.write_text(yaml.safe_dump(drifted, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="path identities"):
        load_harp_fresh_stage70_config(left)


def test_policy_loader_requires_reconstructable_banks_library_thresholds_and_lineage(
    tmp_path: Path,
) -> None:
    cache = _fresh_cache()
    root = _write_policy(tmp_path, cache.reservation.reservation_hash)
    policy = load_frozen_harp_policy(
        root,
        expected_fresh_reservation_hash=cache.reservation.reservation_hash,
    )
    assert policy.production_ready is True
    assert policy.model_bank_collection_hash is not None
    assert policy.action_library_hash is not None
    assert len(policy.policy_receipt_hash) == 64

    path = root / "manifests/policy_lock.json"
    metadata_only = json.loads(path.read_text(encoding="utf-8"))
    metadata_only.pop("model_bank_collection")
    metadata_only["policy_lock_hash"] = canonical_hash(
        {key: value for key, value in metadata_only.items() if key != "policy_lock_hash"}
    )
    path.write_text(json.dumps(metadata_only), encoding="utf-8")
    with pytest.raises(ProtocolError, match="model-bank collection"):
        load_frozen_harp_policy(
            root,
            expected_fresh_reservation_hash=cache.reservation.reservation_hash,
        )


def test_stage70_loads_binding_built_from_actual_stage60_lineage_shape(
    tmp_path: Path,
) -> None:
    cache = _fresh_cache()
    source_content = "d" * 64
    source_lock_file = "3" * 64
    source_index_file = "4" * 64
    source_binding = canonical_hash(
        {
            "schema_version": "midogpp_harp_source_stream_artifact_binding_v1",
            "source_cache_lock_sha256": source_lock_file,
            "source_cache_index_sha256": source_index_file,
            "source_stream_content_hash": source_content,
        }
    )
    lineage = HarpAuthoritativeLineage(
        bank_semantic_lock_hash="9" * 16,
        generation_semantic_lock_hash="a" * 16,
        source_stream_lock_hash="b" * 16,
        source_stream_index_hash="c" * 16,
        source_stream_content_hash=source_content,
        classifier_config_hash="e" * 16,
        expert_bank_index_sha256="1" * 64,
        generation_lock_file_sha256="2" * 64,
        source_cache_lock_sha256=source_lock_file,
        source_cache_index_sha256=source_index_file,
        source_stream_artifact_binding_hash=source_binding,
        classifier_contract_sha256="5" * 64,
        receipt_hash="f" * 64,
    )
    binding = HarpActionInferenceBinding.from_stage60_lineage(
        lineage,
        global_prediction_seal_semantic_id="6" * 64,
        feature_surface_semantic_id="7" * 64,
        response_surface_semantic_id="8" * 64,
    )
    payload = _policy_payload(cache.reservation.reservation_hash)
    payload["action_inference_binding"] = binding.to_payload()
    payload["action_inference_binding_sha256"] = binding.binding_sha256
    payload["policy_lock_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "policy_lock_hash"}
    )
    path = tmp_path / "stage60-shaped/manifests/policy_lock.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    receipt = reconstruct_frozen_harp_policy_receipt(
        path,
        expected_fresh_reservation_hash=cache.reservation.reservation_hash,
    )
    assert receipt.action_inference_binding_sha256 == binding.binding_sha256
    assert receipt.execution_lineage.expert_bank_index_sha256 == "1" * 64


def test_full_policy_scores_complete_label_free_menu_and_center_inference(
    tmp_path: Path,
) -> None:
    cache = _fresh_cache()
    policy = load_frozen_harp_policy(
        _write_policy(tmp_path, cache.reservation.reservation_hash),
        expected_fresh_reservation_hash=cache.reservation.reservation_hash,
    )

    def predictor(action, training_seed, generation_seed, frame):
        source_ordinal = 0 if action.selected_source_id is None else CENTERS.index(action.selected_source_id) + 1
        base = np.asarray(
            [0.40, 0.60] if action.is_exact_b else [0.45, 0.55],
            dtype=np.float32,
        )
        if action.selected_source_id is not None:
            base = np.asarray([0.75, 0.25], dtype=np.float32)
        values = base + np.float32(0.0001 * (training_seed % 10 + generation_seed % 10 + source_ordinal))
        return HarpFreshPredictionOutput(
            probabilities=np.ascontiguousarray(values, dtype=np.float32),
            composition_hash=action.action_hash,
            scaler_state_hash="e" * 16,
        )

    menu = materialize_harp_fresh_probability_menu(policy, cache, predictor)
    assert len(menu.actions) == 90
    assert len(menu.cells) == 810
    physical_decisions = policy.select_all_physical_routes(menu, cache)
    assert len(physical_decisions) == 2 * len(CENTERS)
    assert all(
        not decision.eligible or decision.lambda_value == 1.0
        for decision in physical_decisions
    )
    assert all(
        decision.decision_reason.startswith("PHYSICAL_LAMBDA_ONE_ABLATION::")
        for decision in physical_decisions
    )
    seal = select_and_seal_harp_fresh_routes(
        policy,
        cache,
        menu,
        durable_bundle_hash="f" * 64,
        independent_validation_hashes=("1" * 64, "2" * 64),
    )
    assert all(decision.labels_consumed is False for vector in seal.routed_vectors for decision in vector.decisions)
    assert len(seal.physical_ablation_vectors) == len(CENTERS)
    assert seal.physical_ablation_row_keys == seal.row_keys
    assert len(seal.physical_ablation_reference_preserving_sha256) == len(CENTERS)
    assert all(
        raw_array_sha256(values) == expected
        for values, expected in zip(
            seal.physical_ablation_reference_preserving_vectors,
            seal.physical_ablation_reference_preserving_sha256,
            strict=True,
        )
    )
    labels = {
        key: int(key[1].startswith("positive")) for key in seal.row_keys
    }
    capability = issue_harp_fresh_evaluation_capability(
        seal,
        labels_by_row_key=labels,
        reservation_hash=cache.reservation.reservation_hash,
        target_cache_hash=cache.cache_hash,
        authorization_hash="3" * 64,
    )
    result = score_harp_fresh_routes(seal, capability)
    assert tuple(row.center for row in result.center_metrics) == CENTERS
    assert len(result.center_inference) == 3
    assert all(len(row.center_deltas) == 9 for row in result.center_inference)
    assert all(row.inference_unit == "target_center" for row in result.center_inference)
    assert all(row.seed_cells_are_inference_units is False for row in result.center_inference)
    assert all(
        diagnostic.physical_matched_action_count == len(CENTERS)
        and diagnostic.frozen_lambda_one_policy_reference_preserving is True
        and 0.0 <= diagnostic.frozen_lambda_one_policy_route_rate <= 1.0
        and all(
            action_id.endswith("lambda=1.00")
            for action_id in diagnostic.best_physical_action_ids
        )
        for diagnostic in result.oracle_diagnostics.center_diagnostics
    )
    with pytest.raises(ProtocolError, match="already consumed"):
        capability.consume(seal)


def test_rejected_physical_ablation_seals_exact_b_but_scores_exact_u(
    tmp_path: Path,
) -> None:
    cache = _fresh_cache()
    policy = load_frozen_harp_policy(
        _write_policy(
            tmp_path,
            cache.reservation.reservation_hash,
            favorable=False,
        ),
        expected_fresh_reservation_hash=cache.reservation.reservation_hash,
    )

    def predictor(action, training_seed, generation_seed, frame):
        del training_seed, generation_seed, frame
        if action.is_exact_b:
            values = np.asarray([0.2, 0.8], dtype=np.float32)
        elif action.selected_source_id is None:  # matched-budget U
            values = np.asarray([0.4, 0.6], dtype=np.float32)
        else:
            values = np.asarray([0.9, 0.1], dtype=np.float32)
        return HarpFreshPredictionOutput(
            probabilities=values,
            composition_hash=action.action_hash,
            scaler_state_hash="e" * 16,
        )

    menu = materialize_harp_fresh_probability_menu(policy, cache, predictor)
    seal = select_and_seal_harp_fresh_routes(
        policy,
        cache,
        menu,
        durable_bundle_hash="f" * 64,
        independent_validation_hashes=("1" * 64, "2" * 64),
    )
    for operational, reference_preserving in zip(
        seal.physical_ablation_vectors,
        seal.physical_ablation_reference_preserving_vectors,
        strict=True,
    ):
        assert all(not decision.eligible for decision in operational.decisions)
        assert np.array_equal(
            operational.routed_probabilities.view(np.uint64),
            operational.baseline_probabilities.view(np.uint64),
        )
        assert np.array_equal(
            reference_preserving.view(np.uint64),
            operational.reference_probabilities.view(np.uint64),
        )

    labels = {key: int(key[1].startswith("positive")) for key in seal.row_keys}
    capability = issue_harp_fresh_evaluation_capability(
        seal,
        labels_by_row_key=labels,
        reservation_hash=cache.reservation.reservation_hash,
        target_cache_hash=cache.cache_hash,
        authorization_hash="3" * 64,
    )
    result = score_harp_fresh_routes(seal, capability)
    for diagnostic in result.oracle_diagnostics.center_diagnostics:
        assert diagnostic.frozen_lambda_one_policy_route_rate == 0.0
        assert (
            diagnostic.frozen_lambda_one_policy_balanced_accuracy_delta_vs_u
            == pytest.approx(0.0)
        )
        assert diagnostic.frozen_lambda_one_policy_brier_delta_vs_u == pytest.approx(
            0.0
        )
        assert (
            diagnostic.frozen_lambda_one_policy_log_loss_delta_vs_u
            == pytest.approx(0.0)
        )


def test_callback_policy_is_test_only_and_production_task_plan_is_exact_810(
    tmp_path: Path,
) -> None:
    cache = _fresh_cache()
    metadata = HarpFrozenPolicyMetadata(
        policy_lock_hash="4" * 64,
        fresh_reservation_hash=cache.reservation.reservation_hash,
        bank_hash="5" * 16,
        generation_lock_hash="6" * 16,
        source_cache_hash="7" * 16,
        classifier_hash="8" * 16,
    )
    callback = bind_frozen_harp_policy(metadata, lambda *_: ())
    assert callback.production_ready is False
    with pytest.raises(ProtocolError, match="callback-only"):
        HarpFreshRunner(callback, cache)

    policy = load_frozen_harp_policy(
        _write_policy(tmp_path, cache.reservation.reservation_hash),
        expected_fresh_reservation_hash=cache.reservation.reservation_hash,
    )
    first = _task_plan(
        cache,
        policy=policy,
        source_stream_content_hash=policy.metadata.source_cache_hash,
        classifier_hash=policy.metadata.classifier_hash,
    )
    second = _task_plan(
        cache,
        policy=policy,
        source_stream_content_hash=policy.metadata.source_cache_hash,
        classifier_hash=policy.metadata.classifier_hash,
    )
    assert len(first) == 810
    assert [row.task_hash for row in first] == [row.task_hash for row in second]
    assert len({row.task_id for row in first}) == 810


def test_public_fresh_policy_and_label_free_apis_do_not_accept_truth() -> None:
    assert tuple(inspect.signature(load_frozen_harp_policy).parameters) == (
        "policy_lock_path",
        "expected_fresh_reservation_hash",
    )
    forbidden = {"labels", "truth", "outcomes", "targets"}
    metadata_fields = {field.name for field in fields(HarpFrozenPolicyMetadata)}
    assert not metadata_fields.intersection(forbidden)
    assert "_labels" not in {field.name for field in fields(HarpFreshTargetCache)}
    assert HarpFreshEvaluationCapability.__module__.endswith("label_access")
    assert "_sha256_file(binding.scoring_manifest_path)" not in inspect.getsource(
        load_harp_fresh_target
    )


def test_content_index_excludes_commit_files_and_false_complete_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _config_payload()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_harp_fresh_stage70_config(config_path)
    root = tmp_path / "bundle"
    (root / "reports").mkdir(parents=True)
    (root / "authoritative.txt").write_text("sealed", encoding="utf-8")
    (root / "reports/run_state.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    (root / "reports/validation_report.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    fsync_calls: list[int] = []
    monkeypatch.setattr(harp_bundle.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))
    write_harp_fresh_content_index(root)
    assert len(fsync_calls) == 2  # temporary file, then containing directory
    index = json.loads((root / "manifests/content_index.json").read_text(encoding="utf-8"))
    indexed = {row["path"] for row in index["files"]}
    assert "authoritative.txt" in indexed
    assert "reports/run_state.json" not in indexed
    assert "reports/validation_report.json" not in indexed
    assert "manifests/content_index.json" not in indexed
    with pytest.raises(ProtocolError, match="incomplete"):
        validate_harp_fresh_completed_bundle(root, config=config)


def test_independent_validator_reconstructs_all_810_raw_prediction_cells(
    tmp_path: Path,
) -> None:
    cache = _fresh_cache()
    policy = load_frozen_harp_policy(
        _write_policy(tmp_path, cache.reservation.reservation_hash),
        expected_fresh_reservation_hash=cache.reservation.reservation_hash,
    )

    def predictor(action, training_seed, generation_seed, frame):
        offset = np.float32(
            0.00001
            * (training_seed + generation_seed + (0 if action.is_exact_b else 1))
        )
        return HarpFreshPredictionOutput(
            probabilities=np.ascontiguousarray(
                np.asarray([0.2, 0.8], dtype=np.float32) + offset,
                dtype=np.float32,
            ),
            composition_hash=action.action_hash,
            scaler_state_hash="e" * 16,
        )

    sealed = materialize_harp_fresh_probability_menu(policy, cache, predictor)
    checkpoint_root = tmp_path / "bundle/checkpoints/predictions"
    for cell in sealed.cells:
        source = (
            cell.action.action_id
            if cell.action.selected_source_id is None
            else cell.action.selected_source_id
        )
        task_id = (
            f"H_{cell.action.outer_target_id}__e_{source}__"
            f"train_{cell.training_seed}__gen_{cell.generation_seed}"
        )
        probability_member = f"arrays/{task_id}.probabilities.npy"
        probability_path = checkpoint_root / probability_member
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(probability_path, cell.probabilities, allow_pickle=False)
        frame = cache.frames_by_center[cell.action.outer_target_id]
        task = {
            "schema_version": "midogpp_harp_fresh_prediction_task_v2",
            "task_id": task_id,
            "outer_target_id": cell.action.outer_target_id,
            "selected_source_id": cell.action.selected_source_id,
            "action_id": cell.action.action_id,
            "training_seed": cell.training_seed,
            "generation_seed": cell.generation_seed,
            "action_hash": cell.action.action_hash,
            "frame_hash": cell.frame_hash,
            "target_embedding_bytes_sha256": raw_array_sha256(frame.embeddings),
            "row_count": len(cell.row_ids),
            "policy_lock_hash": policy.metadata.policy_lock_hash,
            "source_stream_content_hash": policy.metadata.source_cache_hash,
            "target_cache_hash": cache.cache_hash,
            "classifier_config_hash": policy.metadata.classifier_hash,
        }
        receipt = {
            **task,
            "schema_version": "midogpp_harp_fresh_prediction_task_result_v2",
            "task_hash": canonical_hash(task),
            "probability_member": probability_member,
            "probability_file_sha256": hashlib.sha256(
                probability_path.read_bytes()
            ).hexdigest(),
            "probability_bytes_sha256": cell.probability_bytes_sha256,
            "composition_hash": cell.composition_hash,
            "classifier_config_hash": cell.classifier_hash,
            "scaler_state_hash": cell.scaler_state_hash,
            "classifier_converged": True,
            "labels_available_to_fit_or_predict": False,
        }
        receipt["result_hash"] = canonical_hash(receipt)
        receipt_path = checkpoint_root / f"tasks/{task_id}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    menu = {
        "actions": [action.to_payload() for action in sealed.actions],
        "prediction_cells": [cell.to_payload() for cell in sealed.cells],
        "seed_pairs": [[a, b] for a, b in ((17, 17), (17, 42), (17, 101), (42, 17), (42, 42), (42, 101), (101, 17), (101, 42), (101, 101))],
        "workstation": sealed.workstation.to_payload(),
        "workstation_hash": sealed.workstation.runtime_hash,
        "prediction_checkpoint_root": "checkpoints/predictions",
        "action_menu_hash": sealed.action_menu_hash,
        "prediction_store_hash": sealed.prediction_store_hash,
        "prediction_menu_seal_hash": sealed.seal_hash,
    }
    root = tmp_path / "bundle"
    _validate_prediction_checkpoints(root, menu)
    first = checkpoint_root / "arrays/H_0__e_B__train_17__gen_17.probabilities.npy"
    np.save(first, np.asarray([0.3, 0.7], dtype=np.float32), allow_pickle=False)
    with pytest.raises(ProtocolError, match="file bytes"):
        _validate_prediction_checkpoints(root, menu)
