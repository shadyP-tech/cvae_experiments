from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.artifacts import (
    file_sha256,
    write_content_index,
    write_csv,
    write_json,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.comparison import (
    audit_decision,
    metric_row,
    paired_comparison_rows,
    prediction_digest,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.config import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    CLAIM_SCOPE,
    CONTROLLED_CANDIDATES,
    EVIDENCE_LABEL,
    FIXED_ANTITHETIC_CANDIDATE,
    FIXED_ONE_EPSILON_CANDIDATE,
    INITIALIZATION_SEEDS,
    LEGACY_CANDIDATE,
    SNAPSHOT_ARTIFACT_ID,
    STAGE,
    ClaimFirewall,
    DecisionThresholds,
    FrozenBRecipe,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.protocol import (
    AuditKeyRecord,
    build_key_record,
    key_inventory_hash,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.snapshot import (
    HASH_PROMOTED,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.validation import (
    PROTOCOL_SCHEMA,
    RUNTIME_SUMMARY_SCHEMA,
    SNAPSHOT_BINDING_SCHEMA,
    assert_valid_audit_bundle,
    validate_audit_bundle,
)


def test_complete_bundle_passes_exact_accounting_and_firewalls(tmp_path: Path) -> None:
    root = _build_valid_bundle(tmp_path / "audit")

    report = assert_valid_audit_bundle(root)

    assert report["status"] == "PASS"
    assert report["counts"] == {
        "key_count": 36,
        "controlled_pair_count": 12,
        "job_count": 36,
        "optimizer_updates": 36_000,
        "decoder_forwards": 48_000,
        "epsilon_consumptions": 36,
        "replay_trace_rows": 36,
        "legacy_validation_rows": 12,
        "controlled_metric_rows": 24,
        "decoded_prediction_rows": 288,
        "paired_comparison_rows": 12,
        "consumption_rows": 36,
    }


def test_decoder_accounting_fails_closed_even_with_refreshed_index(
    tmp_path: Path,
) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    jobs = _read_csv(root / "tables/job_inventory.csv")
    antithetic = next(
        row
        for row in jobs
        if row["candidate"] == FIXED_ANTITHETIC_CANDIDATE
    )
    antithetic["decoder_forwards"] = "1999"
    write_csv(root / "tables/job_inventory.csv", jobs)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("job execution accounting mismatches" in error for error in report["errors"])


def test_controlled_pair_initialization_must_be_identical(
    tmp_path: Path,
) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    jobs = _read_csv(root / "tables/job_inventory.csv")
    antithetic = next(
        row
        for row in jobs
        if row["center"] == "2"
        and row["initialization_seed"] == "17"
        and row["candidate"] == FIXED_ANTITHETIC_CANDIDATE
    )
    antithetic["initialization_hash"] = _sha256("different-initialization")
    write_csv(root / "tables/job_inventory.csv", jobs)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("initialization differs within pair" in error for error in report["errors"])


def test_legacy_rows_cannot_enter_decision(tmp_path: Path) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    decision_path = root / "reports/audit_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["legacy_v2_used_for_decision"] = True
    write_json(decision_path, decision)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("decision does not independently recompute" in error for error in report["errors"])


def test_content_index_detects_post_index_tampering(tmp_path: Path) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    runtime_path = root / "reports/runtime_summary.json"
    runtime_path.write_text(runtime_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("content-index" in error for error in report["errors"])


def test_macro_f1_tamper_fails_even_with_refreshed_index(tmp_path: Path) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    metrics = _read_csv(root / "tables/controlled_metrics.csv")
    metrics[0]["macro_f1"] = "0.123"
    write_csv(root / "tables/controlled_metrics.csv", metrics)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("metric/confusion arithmetic" in error for error in report["errors"])


def test_prediction_tamper_fails_even_with_refreshed_index(tmp_path: Path) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    predictions = _read_csv(root / "tables/decoded_predictions.csv")
    predictions[0]["y_pred"] = "0" if predictions[0]["y_pred"] == "1" else "1"
    write_csv(root / "tables/decoded_predictions.csv", predictions)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("prediction" in error.lower() for error in report["errors"])


def test_coordinated_eval_inventory_tamper_fails_snapshot_binding(
    tmp_path: Path,
) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    predictions = _read_csv(root / "tables/decoded_predictions.csv")
    for row in predictions:
        if row["center"] == "2" and row["sample_id"] == "2-sample-0":
            row["sample_id"] = "2-replaced-sample-0"
    write_csv(root / "tables/decoded_predictions.csv", predictions)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any(
        "promoted snapshot eval rows" in error
        for error in report["errors"]
    )


def test_real_reference_denominator_tamper_fails_prediction_binding(
    tmp_path: Path,
) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    metrics = _read_csv(root / "tables/controlled_metrics.csv")
    metrics[0]["real_reference_bacc"] = "0.875"
    metrics[0]["preservation_ratio"] = str(
        (float(metrics[0]["bacc"]) - 0.5) / (0.875 - 0.5)
    )
    write_csv(root / "tables/controlled_metrics.csv", metrics)
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any(
        "Real-reference BACC disagrees" in error
        for error in report["errors"]
    )


def test_stale_per_key_member_fails_even_with_refreshed_index(
    tmp_path: Path,
) -> None:
    root = _build_valid_bundle(tmp_path / "audit")
    (root / "jobs").mkdir()
    write_json(root / "jobs/stale-key.json", {"status": "stale"})
    write_content_index(root)

    report = validate_audit_bundle(root)

    assert report["status"] == "FAIL"
    assert any("stale or undeclared" in error for error in report["errors"])


def _build_valid_bundle(root: Path) -> Path:
    for directory in ("manifests", "provenance", "reports", "tables"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text(
        "schema_version: fixture\nclaim_scope: diagnostic_only\n",
        encoding="utf-8",
    )
    write_json(
        root / "provenance/input_artifacts.json",
        {
            "schema_version": "fixture",
            "input_artifact_ids": [SNAPSHOT_ARTIFACT_ID],
            "historical_paths_read": False,
        },
    )
    write_json(root / "reports/validation_report.json", {"status": "PENDING"})

    records = _key_records()
    inventory_hash = key_inventory_hash(records)
    eval_row_inventory_hashes = {
        center: _fixture_eval_row_inventory_hash(center)
        for center in AUDIT_CENTERS
    }
    write_json(
        root / "manifests/key_inventory.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_key_inventory_v1",
            "key_inventory_hash": inventory_hash,
            "records": [record.to_payload() for record in records],
        },
    )
    record_by_coordinate = {
        (record.center, record.initialization_seed, record.candidate): record
        for record in records
    }
    write_json(
        root / "manifests/snapshot_binding.json",
        {
            "schema_version": SNAPSHOT_BINDING_SCHEMA,
            "snapshot_artifact_id": SNAPSHOT_ARTIFACT_ID,
            "publication_state": HASH_PROMOTED,
            "snapshot_hash": "b" * 16,
            "snapshot_manifest_hash": "a" * 16,
            "key_inventory_hash": inventory_hash,
            "eval_row_inventory_hashes": eval_row_inventory_hashes,
            "historical_paths_read": False,
            "may_feed_deployable_selection": False,
        },
    )
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "stage": STAGE,
        "evidence_label": EVIDENCE_LABEL,
        "claim_scope": CLAIM_SCOPE,
        "snapshot_hash": "b" * 16,
        "snapshot_manifest_hash": "a" * 16,
        "key_inventory_hash": inventory_hash,
        "eval_row_inventory_hashes": eval_row_inventory_hashes,
        "workspace_snapshot_hashes": {
            "config_resolved_sha256": file_sha256(root / "config.resolved.yaml"),
            "input_artifacts_sha256": file_sha256(
                root / "provenance/input_artifacts.json"
            ),
        },
        "recipe": FrozenBRecipe().to_payload(),
        "legacy_used_for_decision": False,
        "claim_firewall": ClaimFirewall().to_payload(),
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    write_json(root / "manifests/protocol_manifest.json", protocol)

    jobs: list[dict[str, object]] = []
    replay: list[dict[str, object]] = []
    legacy: list[dict[str, object]] = []
    consumption: list[dict[str, object]] = []
    for index, record in enumerate(records):
        coordinate = (record.center, record.initialization_seed, record.candidate)
        initialization_hash = (
            _sha256(f"controlled-init-{record.center}-{record.initialization_seed}")
            if not record.is_legacy
            else _sha256(f"legacy-init-{record.center}-{record.initialization_seed}")
        )
        checkpoint_hash = (
            str(record.legacy_expected_checkpoint_hash)
            if record.is_legacy
            else _sha256(f"checkpoint-{record.key_hash}")
        )
        antithetic = record.candidate == FIXED_ANTITHETIC_CANDIDATE
        decoder_forwards = 2000 if antithetic else 1000
        job = {
            "center": record.center,
            "initialization_seed": record.initialization_seed,
            "candidate": record.candidate,
            "execution_device": record.execution_device,
            "key_hash": record.key_hash,
            "pair_id": "" if record.pair_id is None else record.pair_id,
            "prepared_sha256": record.prepared_sha256,
            "prepared_content_hash": record.prepared_content_hash,
            "schedule_sha256": record.schedule_sha256,
            "schedule_content_hash": record.schedule_content_hash,
            "epsilon_trace_sha256": record.epsilon_trace_sha256,
            "epsilon_trace_content_hash": record.epsilon_trace_content_hash,
            "initialization_hash": initialization_hash,
            "checkpoint_hash": checkpoint_hash,
            "schedule_hash": (
                record.legacy_historical_schedule_hash
                if record.is_legacy
                else stable_hash({"runtime-schedule": record.center})
            ),
            "posterior_stream_hash": (
                record.legacy_historical_posterior_stream_hash
                if record.is_legacy
                else stable_hash(
                    {
                        "runtime-posterior": [
                            record.center,
                            record.initialization_seed,
                        ]
                    }
                )
            ),
            "optimizer_steps": 1000,
            "decoder_forwards": decoder_forwards,
            "posterior_estimator": (
                "antithetic_epsilon" if antithetic else "one_epsilon"
            ),
            "epsilon_consumptions": 1,
            "cache_status": "COMPLETED",
            "status": "PASS",
            "claim_scope": CLAIM_SCOPE,
        }
        jobs.append(job)
        replay.append(
            {
                "center": record.center,
                "initialization_seed": record.initialization_seed,
                "candidate": record.candidate,
                "key_hash": record.key_hash,
                "prepared_file_match": True,
                "prepared_content_match": True,
                "schedule_file_match": True,
                "schedule_content_match": True,
                "epsilon_file_match": True,
                "epsilon_content_match": True,
                "trace_consumption_count": 1,
                "status": "PASS",
            }
        )
        consumption.append(
            {
                "center": record.center,
                "initialization_seed": record.initialization_seed,
                "candidate": record.candidate,
                "key_hash": record.key_hash,
                "epsilon_consumption_count": 1,
                "optimizer_steps": 1000,
                "decoder_forwards": decoder_forwards,
                "status": "PASS",
            }
        )
        if record.is_legacy:
            legacy_hashes = {
                "initialization": initialization_hash,
                "checkpoint": checkpoint_hash,
                "prediction": str(record.legacy_expected_prediction_hash),
                "metric": str(record.legacy_expected_metric_hash),
                "schedule": stable_hash({"schedule": coordinate}),
                "posterior": stable_hash({"posterior": coordinate}),
            }
            legacy.append(
                {
                    "center": record.center,
                    "initialization_seed": record.initialization_seed,
                    "candidate": record.candidate,
                    "key_hash": record.key_hash,
                    **{
                        column: value
                        for field, digest in legacy_hashes.items()
                        for column, value in (
                            (f"expected_{field}_hash", digest),
                            (f"observed_{field}_hash", digest),
                            (f"{field}_match", True),
                        )
                    },
                    "metric_values_match": True,
                    "comparison_eligible": False,
                    "replay_validation_only": True,
                    "claim_scope": CLAIM_SCOPE,
                    "status": "PASS",
                }
            )
    write_csv(root / "tables/job_inventory.csv", jobs)
    write_csv(root / "tables/replay_trace_audit.csv", replay)
    write_csv(root / "tables/legacy_v2_validation.csv", legacy)
    write_csv(root / "tables/consumption_audit.csv", consumption)

    metrics: list[dict[str, object]] = []
    decoded_predictions: list[dict[str, object]] = []
    for center in AUDIT_CENTERS:
        for seed in INITIALIZATION_SEEDS:
            for candidate in AUDIT_CANDIDATES:
                record = record_by_coordinate[(center, seed, candidate)]
                prediction_rows = _fixture_prediction_rows(
                    center=center,
                    candidate=candidate,
                )
                decoded_predictions.extend(
                    {
                        **row,
                        "key_hash": record.key_hash,
                        "pair_id": "" if record.pair_id is None else record.pair_id,
                        "training_seed": seed,
                        "candidate": candidate,
                        "center": center,
                        "representation_role": "decode_mu",
                        "eval_label_role": (
                            "final_diagnostic_scoring_and_decode_condition_only"
                        ),
                        "selection_source": "none",
                        "oracle_eligible": False,
                        "claim_scope": CLAIM_SCOPE,
                    }
                    for row in prediction_rows
                )
                if candidate in CONTROLLED_CANDIDATES:
                    metrics.append(
                        {
                            **metric_row(
                                center=center,
                                training_seed=seed,
                                candidate=candidate,
                                truth=tuple(
                                    int(row["y_true"]) for row in prediction_rows
                                ),
                                predicted=tuple(
                                    int(row["y_pred"]) for row in prediction_rows
                                ),
                                real_reference_bacc=0.75,
                                minimum_real_bacc=0.60,
                            ),
                            "key_hash": record.key_hash,
                            "pair_id": record.pair_id,
                        }
                    )
    write_csv(root / "tables/controlled_metrics.csv", metrics)
    write_csv(root / "tables/decoded_predictions.csv", decoded_predictions)
    paired = paired_comparison_rows(metrics)
    for row in paired:
        coordinate = (str(row["center"]), int(row["training_seed"]))
        row["pair_id"] = record_by_coordinate[
            (*coordinate, FIXED_ONE_EPSILON_CANDIDATE)
        ].pair_id
    write_csv(root / "tables/paired_comparison.csv", paired)
    write_json(
        root / "reports/audit_decision.json",
        audit_decision(metrics, thresholds=DecisionThresholds().to_payload()),
    )
    write_json(
        root / "reports/leakage_provenance_report.json",
        {
            "schema_version": "fixture",
            "status": "PASS",
            "stage": STAGE,
            "evidence_label": EVIDENCE_LABEL,
            "claim_scope": CLAIM_SCOPE,
            "historical_paths_read": False,
            "eval_labels_used_for_cvae_fit": False,
            "eval_labels_used_for_classifier_fit": False,
            "eval_labels_used_for_selection": False,
            "eval_labels_used_for_decode_condition": True,
            "eval_labels_used_for_final_diagnostic_scoring": True,
            "claim_firewall": ClaimFirewall().to_payload(),
        },
    )
    write_json(
        root / "reports/runtime_summary.json",
        {
            "schema_version": RUNTIME_SUMMARY_SCHEMA,
            "stage": STAGE,
            "evidence_label": EVIDENCE_LABEL,
            "claim_scope": CLAIM_SCOPE,
            "job_count": 36,
            "legacy_job_count": 12,
            "controlled_job_count": 24,
            "controlled_pair_count": 12,
            "optimizer_updates": 36_000,
            "legacy_decoder_forwards": 12_000,
            "fixed_one_epsilon_decoder_forwards": 12_000,
            "antithetic_decoder_forwards": 24_000,
            "decoder_forwards": 48_000,
            "epsilon_consumptions": 36,
            "claim_firewall": ClaimFirewall().to_payload(),
        },
    )
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "fixture",
            "status": "VALIDATING",
            "stage": STAGE,
            "evidence_label": EVIDENCE_LABEL,
            "claim_scope": CLAIM_SCOPE,
            "may_feed_deployable_selection": False,
        },
    )
    write_content_index(root)
    return root


def _key_records() -> tuple[AuditKeyRecord, ...]:
    output: list[AuditKeyRecord] = []
    for center in AUDIT_CENTERS:
        center_index = AUDIT_CENTERS.index(center)
        prepared_sha = _sha256(f"prepared-file-{center}")
        prepared_content = _sha256(f"prepared-content-{center}")
        fixed_schedule_sha = _sha256(f"fixed-schedule-file-{center}")
        fixed_schedule_content = _sha256(f"fixed-schedule-content-{center}")
        fixed_epsilon_sha = _sha256(f"fixed-epsilon-file-{center}")
        fixed_epsilon_content = _sha256(f"fixed-epsilon-content-{center}")
        for seed in INITIALIZATION_SEEDS:
            seed_index = INITIALIZATION_SEEDS.index(seed)
            execution_device = f"cuda:{(center_index + seed_index) % 2}"
            for candidate in AUDIT_CANDIDATES:
                legacy = candidate == LEGACY_CANDIDATE
                expected_metric = {
                    "bacc": 0.75,
                    "positive_recall": 0.75,
                    "specificity": 0.75,
                    "fn": 1,
                    "fp": 1,
                    "tn": 3,
                    "tp": 3,
                }
                output.append(
                    build_key_record(
                        center=center,
                        initialization_seed=seed,
                        execution_device=execution_device,
                        candidate=candidate,
                        prepared_relpath=f"prepared/{center}.npz",
                        prepared_sha256=prepared_sha,
                        prepared_content_hash=prepared_content,
                        schedule_relpath=(
                            f"schedules/legacy-{center}-{seed}.npz"
                            if legacy
                            else f"schedules/fixed-{center}.npz"
                        ),
                        schedule_sha256=(
                            _sha256(f"legacy-schedule-file-{center}-{seed}")
                            if legacy
                            else fixed_schedule_sha
                        ),
                        schedule_content_hash=(
                            _sha256(f"legacy-schedule-content-{center}-{seed}")
                            if legacy
                            else fixed_schedule_content
                        ),
                        epsilon_trace_relpath=(
                            f"epsilon/legacy-{center}-{seed}.npy"
                            if legacy
                            else f"epsilon/fixed-{center}.npy"
                        ),
                        epsilon_trace_sha256=(
                            _sha256(f"legacy-epsilon-file-{center}-{seed}")
                            if legacy
                            else fixed_epsilon_sha
                        ),
                        epsilon_trace_content_hash=(
                            _sha256(f"legacy-epsilon-content-{center}-{seed}")
                            if legacy
                            else fixed_epsilon_content
                        ),
                        snapshot_manifest_hash="a" * 16,
                        legacy_expected_checkpoint_hash=(
                            _sha256(f"legacy-checkpoint-{center}-{seed}")
                            if legacy
                            else None
                        ),
                        legacy_expected_prediction_hash=(
                            prediction_digest(
                                _fixture_prediction_rows(
                                    center=center,
                                    candidate=LEGACY_CANDIDATE,
                                )
                            )
                            if legacy
                            else None
                        ),
                        legacy_expected_metric_hash=(
                            _canonical_sha256(expected_metric)
                            if legacy
                            else None
                        ),
                        legacy_expected_initialization_hash=(
                            _sha256(f"legacy-init-{center}-{seed}")
                            if legacy
                            else None
                        ),
                        legacy_historical_training_key_hash=(
                            stable_hash({"legacy-training-key": [center, seed]})
                            if legacy
                            else None
                        ),
                        legacy_historical_schedule_hash=(
                            stable_hash({"schedule": (center, seed, candidate)})
                            if legacy
                            else None
                        ),
                        legacy_historical_posterior_stream_hash=(
                            stable_hash({"posterior": (center, seed, candidate)})
                            if legacy
                            else None
                        ),
                        legacy_historical_frame_hash=(
                            stable_hash({"frame": [center, seed]})
                            if legacy
                            else None
                        ),
                        legacy_historical_fit_row_hash=(
                            stable_hash({"fit": [center, seed]})
                            if legacy
                            else None
                        ),
                        legacy_historical_eval_row_hash=(
                            stable_hash({"eval": [center, seed]})
                            if legacy
                            else None
                        ),
                        legacy_expected_decode_metric=(
                            expected_metric if legacy else None
                        ),
                    )
                )
    return tuple(output)


def _fixture_prediction_rows(
    *,
    center: str,
    candidate: str,
) -> list[dict[str, object]]:
    truth = (1, 1, 1, 1, 0, 0, 0, 0)
    predicted = (
        (1, 1, 1, 0, 0, 0, 0, 1)
        if candidate == LEGACY_CANDIDATE
        else truth
    )
    real_reference_predicted = (1, 1, 1, 0, 0, 0, 0, 1)
    return [
        {
            "schema_version": "fixture_prediction_evidence_v1",
            "sample_id": f"{center}-sample-{index}",
            "case_id": f"{center}-case-{index // 2}",
            "y_true": y_true,
            "y_pred": y_pred,
            "real_reference_y_pred": real_reference_y_pred,
        }
        for index, (y_true, y_pred, real_reference_y_pred) in enumerate(
            zip(truth, predicted, real_reference_predicted, strict=True)
        )
    ]


def _fixture_eval_row_inventory_hash(center: str) -> str:
    rows = _fixture_prediction_rows(
        center=center,
        candidate=FIXED_ONE_EPSILON_CANDIDATE,
    )
    return _canonical_sha256(
        {
            "schema_version": "midogpp_b_prepared_row_inventory_v1",
            "rows": [
                {
                    "sample_id": row["sample_id"],
                    "case_id": row["case_id"],
                    "label": row["y_true"],
                }
                for row in rows
            ],
        }
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
