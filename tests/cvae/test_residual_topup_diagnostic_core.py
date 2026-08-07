from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.residual_topup_router.bundle import (
    CONTENT_INDEX_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.calibration import (
    calibrate_outer_actions,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.contracts import (
    CENTERS,
    ENERGY_TOPUP_ACTION_ID,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    UNIFORM_TOPUP_ACTION_ID,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.reports import (
    phase_completion_payload,
    publication_decision_payload,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router.scoring import (
    score_prediction_store,
)
from midogpp_thesis.cvae.diagnostics.residual_topup_router import (
    seals,
    source_cache_validation,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _gain_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outer in CENTERS:
        for query in CENTERS:
            if query == outer:
                continue
            gain = 0.01 if outer == CENTERS[0] else 0.0
            for training_seed in TRAINING_SEEDS:
                for generation_seed in GENERATION_SEEDS:
                    rows.append(
                        {
                            "outer_target": outer,
                            "query_center": query,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "paired_bacc_gain": gain,
                        }
                    )
    return rows


def test_fixed_lcb_gate_uses_query_clusters_and_exact_uniform_fallback() -> None:
    query_rows, selections, lock = calibrate_outer_actions(
        _gain_rows(),
        config_contract_hash="config-hash",
        global_prediction_seal_hash="seal-hash",
    )

    assert len(query_rows) == len(CENTERS) * (len(CENTERS) - 1)
    assert len(selections) == len(CENTERS)
    by_target = {str(row["outer_target"]): row for row in selections}
    assert by_target[CENTERS[0]]["selected_action_id"] == ENERGY_TOPUP_ACTION_ID
    assert by_target[CENTERS[0]]["fallback_applied"] is False
    for target in CENTERS[1:]:
        assert by_target[target]["selected_action_id"] == UNIFORM_TOPUP_ACTION_ID
        assert by_target[target]["fallback_applied"] is True
    assert all(row["query_cluster_count"] == len(CENTERS) - 1 for row in selections)
    assert all(row["target_H_labels_used_for_selection"] is False for row in selections)
    assert lock["target_H_labels_used"] is False
    assert lock["hyperparameters_fitted"] is False


def test_fixed_lcb_gate_rejects_incomplete_outer_fold() -> None:
    with pytest.raises(ProtocolError, match="coverage"):
        calibrate_outer_actions(
            _gain_rows()[:-1],
            config_contract_hash="config-hash",
            global_prediction_seal_hash="seal-hash",
        )


def test_report_helpers_keep_terminal_diagnostic_claim_boundary() -> None:
    phase = phase_completion_payload(
        "phase_02_all_actions_sealed",
        config_contract_hash="config-hash",
        bindings={"seal_hash": "abc"},
        counts={"prediction_cell_count": 1539},
        labels_opened=False,
    )
    assert phase["status"] == "COMPLETE"
    assert phase["labels_opened"] is False
    assert phase["diagnostic_only"] is True

    decision = publication_decision_payload({"diagnostic_only": True})
    assert decision["promotion_eligible"] is False
    assert decision["may_feed_stage60"] is False
    assert decision["may_feed_stage70"] is False
    assert decision["may_feed_deployable_selection"] is False


def test_mutable_run_state_is_outside_the_scientific_content_seal() -> None:
    assert "reports/run_state.json" not in CONTENT_INDEX_MEMBERS
    assert "reports/validation_report.json" not in CONTENT_INDEX_MEMBERS
    assert "reports/publication_decision.json" in CONTENT_INDEX_MEMBERS


def test_independent_source_cache_validation_rehashes_persisted_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = ("0", "1")
    monkeypatch.setattr(source_cache_validation, "CENTERS", centers)
    monkeypatch.setattr(source_cache_validation, "TRAINING_SEEDS", (17,))
    monkeypatch.setattr(source_cache_validation, "GENERATION_SEEDS", (17,))
    monkeypatch.setattr(source_cache_validation, "MAX_SOURCE_PREFIX_PER_CLASS", 2)
    monkeypatch.setattr(source_cache_validation, "COMMON_FEATURE_DIM", 2)
    keys = tuple(
        SimpleNamespace(
            source_center=center,
            training_seed=17,
            generation_seed=17,
            stream_id=f"stream-{center}",
            expert_lock_hash=f"expert-{center}",
        )
        for center in centers
    )
    monkeypatch.setattr(
        source_cache_validation,
        "source_generation_plan",
        lambda _lock: keys,
    )
    array = np.arange(16, dtype=np.float32).reshape(2, 4, 2)
    array_path = tmp_path / "source.npy"
    np.save(array_path, array, allow_pickle=False)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    index_rows = tuple(
        {
            "block_ordinal": ordinal,
            "source_center": center,
            "training_seed": 17,
            "generation_seed": 17,
            "stream_id": f"stream-{center}",
            "expert_lock_hash": f"expert-{center}",
            "samples_per_class": 2,
            "row_count": 4,
            "feature_dim": 2,
            "output_sha256": source_cache_validation._array_bundle_sha256(
                array[ordinal], labels
            ),
        }
        for ordinal, center in enumerate(centers)
    )
    energy_rows = tuple(
        {
            "source_center": source,
            "training_seed": 17,
            "query_center": query,
            "case_id": case,
            "row_count": 1,
            "marginal_variational_energy": 1.0,
            "class_0_energy": 1.0,
            "class_1_energy": 1.0,
            "class_0_common_reconstruction_mse": 1.0,
            "class_1_common_reconstruction_mse": 1.0,
            "class_0_normalized_ps_kl": 1.0,
            "class_1_normalized_ps_kl": 1.0,
            "query_partition_role": "support",
            "class_prior_json": "[0.5,0.5]",
            "labels_used": False,
            "exact_nelbo_claimed": False,
        }
        for source in centers
        for query in centers
        for case in (f"{query}-a", f"{query}-b")
    )
    cache = SimpleNamespace(
        array_path=array_path,
        index_rows=index_rows,
        compatibility_case_rows=energy_rows,
    )
    partitions = SimpleNamespace(
        support_rows_by_center={
            query: (
                SimpleNamespace(case_id=f"{query}-a"),
                SimpleNamespace(case_id=f"{query}-b"),
            )
            for query in centers
        }
    )
    checks = source_cache_validation.validate_source_cache_contents(
        cache,
        generation_lock=object(),
        partitions=partitions,
    )
    assert checks["all_source_block_hashes_verified"] is True

    tampered = array.copy()
    tampered[0, 0, 0] += 1.0
    np.save(array_path, tampered, allow_pickle=False)
    with pytest.raises(ProtocolError, match="source block binding"):
        source_cache_validation.validate_source_cache_contents(
            cache,
            generation_lock=object(),
            partitions=partitions,
        )


def test_metric_rows_distinguish_calibration_from_terminal_scoring() -> None:
    index_rows = (
        {
            "phase": "development",
            "outer_target": "0",
            "query_center": "1",
            "action_id": "uniform_topup",
            "arm_role": "development_action",
            "budget_role": "matched_topup_primary",
            "training_seed": 17,
            "generation_seed": 17,
            "evaluation_row_ids_json": '["a","b"]',
        },
        {
            "phase": "target",
            "outer_target": "0",
            "query_center": "0",
            "action_id": "base_only",
            "arm_role": "target_action",
            "budget_role": "base_budget_reference",
            "training_seed": 17,
            "generation_seed": 17,
            "evaluation_row_ids_json": '["a","b"]',
        },
    )
    store = SimpleNamespace(
        index_rows=index_rows,
        slice_for=lambda _row: (
            np.asarray([0, 1], dtype=np.uint8),
            np.asarray([0.1, 0.9], dtype=np.float32),
        ),
    )
    rows = score_prediction_store(store, labels_by_sample_id={"a": 0, "b": 1})
    assert rows[0]["metric_role"] == "q_not_H_diagnostic_calibration_and_scoring"
    assert rows[1]["metric_role"] == "terminal_descriptive_scoring_only"
    assert all(row["labels_used_only_after_global_prediction_seal"] for row in rows)
    assert all(row["target_H_labels_used_for_own_selection"] is False for row in rows)


def test_label_capability_revalidates_current_plan_before_manifest_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_touched = False

    def reject_drift(*_args: object, **_kwargs: object) -> object:
        raise ProtocolError("router plan lock drifted")

    def touch_manifest(*_args: object, **_kwargs: object) -> object:
        nonlocal manifest_touched
        manifest_touched = True
        return ()

    monkeypatch.setattr(seals, "validate_global_prediction_seal", reject_drift)
    monkeypatch.setattr(seals, "_stream_labels", touch_manifest)
    config = SimpleNamespace(
        contract_hash="config-hash",
        validation_manifest_path=tmp_path / "manifest.csv",
    )
    with pytest.raises(ProtocolError, match="plan lock drifted"):
        seals.open_evaluation_labels_after_global_seal(
            config,
            SimpleNamespace(lock_hash="partition-lock"),
            SimpleNamespace(lock_hash="drifted-plan-lock"),
            object(),
            root=tmp_path,
        )
    assert manifest_touched is False
