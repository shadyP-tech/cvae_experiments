from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only import input_contracts
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only import inputs as prediction_inputs
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only import source_capability
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.actions import (
    action_library_payload,
    actions_for_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.constants import (
    CENTERS,
    CLASSIFIER_COEFFICIENT_MEMBER,
    CLASSIFIER_INTERCEPT_MEMBER,
    CLASSIFIER_MEAN_MEMBER,
    CLASSIFIER_SCALE_MEMBER,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    SOURCE_ARRAY_MEMBER,
    SOURCE_INDEX_MEMBER,
    SOURCE_SEAL_MEMBER,
    TEST_ARRAY_MEMBER,
    TEST_INDEX_MEMBER,
    TEST_SEAL_MEMBER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.execution_adapter import (
    issue_test_inference_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.input_contracts import (
    LabelFreeSourceFrame,
    SourceRowIdentity,
    opaque_source_row_id,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.experiment_contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPECTED_TRAIN_CACHE_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.prediction_contracts import (
    canonical_cell_keys,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.prediction_tasks import (
    fit_action_classifier,
    predict_probability_batched,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_disagreement_regret_prediction_only.source_capability import (
    SourceOOFLabelCapability,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
)


CACHE_HASH = "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"


def test_action_library_has_exact_1458_fit_topology_and_durable_phase_members() -> None:
    assert all(len(actions_for_target(target)) == 18 for target in CENTERS)
    assert len(canonical_cell_keys()) == EXPECTED_CLASSIFIER_FIT_COUNT == 1458
    assert len({key for key in canonical_cell_keys()}) == 1458
    payload = action_library_payload()
    assert len(payload["action_library_hash"]) == 64
    assert payload["geometry_selection_used"] is False
    assert (
        CLASSIFIER_MEAN_MEMBER,
        CLASSIFIER_SCALE_MEMBER,
        CLASSIFIER_COEFFICIENT_MEMBER,
        CLASSIFIER_INTERCEPT_MEMBER,
    ) == (
        "arrays/action_classifier_scaler_mean.npy",
        "arrays/action_classifier_scaler_scale.npy",
        "arrays/action_classifier_coefficients.npy",
        "arrays/action_classifier_intercepts.npy",
    )
    assert (SOURCE_ARRAY_MEMBER, SOURCE_INDEX_MEMBER, SOURCE_SEAL_MEMBER) == (
        "arrays/source_action_probabilities.npz",
        "manifests/source_prediction_index.json",
        "manifests/source_prediction_seal.json",
    )
    assert (TEST_ARRAY_MEMBER, TEST_INDEX_MEMBER, TEST_SEAL_MEMBER) == (
        "arrays/test_action_probabilities.npz",
        "manifests/test_prediction_index.json",
        "manifests/test_prediction_seal.json",
    )


def test_source_identity_projection_removes_outcome_encoding() -> None:
    first = opaque_source_row_id("002__002__ann10__y1", cache_sha256=CACHE_HASH)
    second = opaque_source_row_id("002__002__ann10__y0", cache_sha256=CACHE_HASH)
    assert first.startswith("src_") and len(first) == 68
    assert "y1" not in first and "y0" not in first
    assert first != second
    with pytest.raises(ProtocolError):
        opaque_source_row_id("sample", cache_sha256="short")


def test_frozen_classifier_parameters_predict_without_refit() -> None:
    rng = np.random.default_rng(5)
    x = np.zeros((48, 3840), dtype=np.float32)
    x[:, :8] = rng.normal(size=(48, 8)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0.0).astype(np.uint8)
    spec = ClassifierSpec(
        C=0.01,
        penalty="l2",
        solver="lbfgs",
        max_iter=3000,
        class_weight=None,
        random_state=23,
        l1_ratio=None,
        threshold_policy="predict",
        scaler_fit="synthetic_train_only",
    )
    fitted = fit_action_classifier(x, y, spec=spec, sample_weight=None)
    evaluation = np.zeros((7, 3840), dtype=np.float32)
    evaluation[:, :8] = rng.normal(size=(7, 8)).astype(np.float32)
    observed = predict_probability_batched(
        evaluation,
        fitted["mean"],
        fitted["scale"],
        fitted["coefficient"],
        float(fitted["intercept"]),
        batch_rows=256,
    )
    logits = (
        (evaluation.astype(np.float64) - fitted["mean"])
        / fitted["scale"]
    ) @ fitted["coefficient"] + float(fitted["intercept"])
    expected = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)


def test_source_labels_open_only_after_complete_source_seal_and_exclude_h(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row_count = 2 * len(CENTERS)
    monkeypatch.setattr(input_contracts, "EXPECTED_SOURCE_ROWS", row_count)
    monkeypatch.setattr(source_capability, "EXPECTED_SOURCE_ROWS", row_count)
    raw_rows: list[dict[str, str]] = []
    typed_rows: list[SourceRowIdentity] = []
    by_center: dict[str, tuple[SourceRowIdentity, ...]] = {}
    ordinal = 0
    for center in CENTERS:
        center_rows: list[SourceRowIdentity] = []
        for label in (0, 1):
            raw_id = f"case{center}_{label}__y{label}"
            case_id = f"case{center}"
            raw_rows.append(
                {
                    "sample_id": raw_id,
                    "case_id": case_id,
                    "center": center,
                    "split": "train",
                    "label": str(label),
                }
            )
            row = SourceRowIdentity(
                row_ordinal=ordinal,
                cache_row_index=ordinal,
                source_row_id=opaque_source_row_id(raw_id, cache_sha256=CACHE_HASH),
                case_id=case_id,
                center=center,
            )
            typed_rows.append(row)
            center_rows.append(row)
            ordinal += 1
        by_center[center] = tuple(center_rows)
    train_root = tmp_path / "train-cache"
    (train_root / "embeddings").mkdir(parents=True)
    (train_root / "embeddings/train.pt").write_bytes(b"train-only-fixture")
    monkeypatch.setattr(source_capability, "sha256_file", lambda _path: CACHE_HASH)
    monkeypatch.setattr(
        source_capability,
        "load_cache_rows",
        lambda *_args, **_kwargs: SimpleNamespace(
            cache_sha256=CACHE_HASH,
            metadata=tuple(raw_rows),
        ),
    )
    frame = LabelFreeSourceFrame(
        embeddings=np.zeros((row_count, 3840), dtype=np.float32),
        rows=tuple(typed_rows),
        rows_by_center=by_center,
        cache_binding={"fixture": True},
    )
    capability = SourceOOFLabelCapability(
        frame,
        train_cache_root=train_root,
        expected_train_cache_sha256=CACHE_HASH,
    )
    with pytest.raises(ProtocolError):
        capability.labels_for_outer_target("0")
    bank = SimpleNamespace(
        source_cache_binding_hash=frame.cache_binding_hash,
        seal_hash="b" * 64,
    )
    target_bank = SimpleNamespace(
        source_cache_binding_hash=frame.cache_binding_hash,
        seal_hash="c" * 64,
    )
    store = SimpleNamespace(
        frame_role="source", frame_cache_binding_hash=frame.cache_binding_hash
    )
    seal = SimpleNamespace(
        seal_hash="a" * 64,
        classifier_bank=bank,
        target_classifier_bank=target_bank,
        source_store=store,
        seal_payload={
            "status": (
                "SEALED_STRICT_SOURCE_OOF_AND_TARGET_CLASSIFIER_BANK_BEFORE_LABELS"
            ),
            "source_labels_opened": False,
            "test_cache_admitted": False,
            "strict_source_oof_classifier_bank_seal_hash": bank.seal_hash,
            "target_classifier_bank_seal_hash": target_bank.seal_hash,
            "strict_source_physical_fit_count": 5_184,
            "strict_source_logical_prediction_cell_count": 10_368,
            "target_classifier_fit_count": 1_458,
            "query_excluded_from_every_source_composition": True,
        },
    )
    capability.open_after_source_prediction_seal(seal)
    labels = capability.labels_for_outer_target("0")
    assert len(labels) == row_count - 2
    assert {row.query_id for row in labels} == set(CENTERS) - {"0"}
    assert all(row.query_id != "0" for row in labels)
    with pytest.raises(ProtocolError):
        capability.access_report()
    for target in CENTERS[1:]:
        scoped = capability.labels_for_outer_target(target)
        assert {row.query_id for row in scoped} == set(CENTERS) - {target}
    report = dict(capability.access_report())
    assert report["status"] == "OPEN_SOURCE_ONLY"
    assert report["source_labels_opened"] is True
    assert report["source_labels_opened_after_complete_prediction_seal"] is True
    assert report["test_labels_opened"] is False
    assert report["test_labels_available"] is False
    assert report["raw_source_labels_persisted"] is False
    assert report["outer_targets_accessed"] == list(CENTERS)
    with pytest.raises(ProtocolError):
        capability.labels_for_outer_target("0")
    with pytest.raises(ProtocolError):
        capability.open_after_source_prediction_seal(seal)


def test_test_admission_requires_source_only_regret_model_bank() -> None:
    source = SimpleNamespace(
        seal_hash="a" * 64,
        classifier_bank=SimpleNamespace(seal_hash="b" * 64),
        seal_payload={
            "test_cache_admitted": False,
            "source_labels_opened": False,
        },
    )
    token = issue_test_inference_admission(
        source,
        {
            "regret_model_bank_seal_hash": "c" * 64,
            "status": "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION",
            "source_labels_only": True,
            "test_cache_admitted": False,
            "target_labels_used": False,
        },
    )
    assert token.target_labels_available is False
    assert token.test_scoring_permitted is False
    assert token.action_classifier_bank_seal_hash == "b" * 64
    with pytest.raises(ProtocolError):
        issue_test_inference_admission(
            source,
            {
                "regret_model_bank_seal_hash": "c" * 64,
                "status": "SEALED_SOURCE_ONLY_BEFORE_TEST_ADMISSION",
                "source_labels_only": True,
                "test_cache_admitted": True,
                "target_labels_used": False,
            },
        )


def _input_config_fixture() -> SimpleNamespace:
    return SimpleNamespace(
        experiment_id=EXPERIMENT_ID,
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
        expert_bank_root=Path("/safe/promoted_bank"),
        generation_lock_root=Path("/safe/generation_lock"),
        train_cache_root=Path("/safe/train_cache"),
        test_cache_root=Path("/safe/test_cache"),
        test_consumption_ledger_path=Path("/safe/test_ledger.json"),
        ledger_amendment_path=Path("/safe/ledger_amendment.json"),
        expected_bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        expected_generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
        expected_train_cache_sha256=EXPECTED_TRAIN_CACHE_SHA256,
        expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        expected_test_cache_semantic_id=EXPECTED_TEST_CACHE_SEMANTIC_ID,
        expected_test_cache_representation_id=EXPECTED_TEST_CACHE_REPRESENTATION_ID,
        expected_test_cache_content_hash=EXPECTED_TEST_CACHE_CONTENT_HASH,
        expected_test_cache_row_order_hash=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        expected_test_consumption_ledger_sha256=EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        expected_ledger_amendment_sha256=EXPECTED_LEDGER_AMENDMENT_SHA256,
    )


def test_input_fence_requires_exact_local_aliases() -> None:
    config = _input_config_fixture()
    prediction_inputs.assert_input_fence(config)
    drifted = SimpleNamespace(
        **{
            **vars(config),
            "train_cache_root": Path("/safe/fixed_bank_actionability_recoverability"),
        }
    )
    with pytest.raises(ProtocolError):
        prediction_inputs.assert_input_fence(drifted)


def test_pre_gpu_firewall_is_source_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _input_config_fixture()
    records = [
        {
            "fresh_source_only_training": True,
            "parent_checkpoint_used": False,
        }
        for _ in range(27)
    ]
    monkeypatch.setattr(prediction_inputs, "load_promotion_config", lambda _path: object())
    monkeypatch.setattr(
        prediction_inputs,
        "validate_promoted_bank",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "all_experts_source_only": True,
        },
    )
    monkeypatch.setattr(
        prediction_inputs,
        "_json",
        lambda _path: {
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "records": records,
        },
    )
    monkeypatch.setattr(
        prediction_inputs,
        "load_validated_stage70_test_cache",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("test cache opened before model-bank seal")
        ),
    )
    generation = SimpleNamespace(
        bank_lock_hash=EXPECTED_BANK_LOCK_HASH,
        generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH,
    )
    locks = prediction_inputs.ValidatedLocks(
        generation=generation,
        test_consumption_ledger={},
        ledger_amendment={
            "parent_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
            "previous_stage90_outputs_used": False,
            "previous_stage90_scratch_or_checkpoints_used": False,
            "previous_prediction_surfaces_used": False,
            "target_cache_is_label_free": True,
            "no_target_label_capability_created": True,
        },
    )
    source_frame = SimpleNamespace(
        cache_binding={
            "split": "train",
            "row_count": 9_648,
            "feature_dim": 3_840,
            "cache_sha256": EXPECTED_TRAIN_CACHE_SHA256,
            "labels_in_typed_frame": False,
            "historical_sample_ids_persisted": False,
            "source_label_field_accessed_by_projection_code": False,
            "source_labels_physically_present_in_input_metadata": True,
            "single_consumer_alias_only": True,
        }
    )
    report = prediction_inputs.validate_pre_gpu_firewall(config, source_frame, locks)
    assert report["status"] == "PASS"
    assert report["test_cache_opened"] is False
    assert report["source_labels_opened"] is False
    assert report["gpu_work_authorized_for_source_streams_only"] is True
