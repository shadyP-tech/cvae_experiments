from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from midogpp_thesis.cvae.preservation.prior_recovery import (
    run_outer_prior_recovery,
    run_source_inner_prior_recovery,
)
from midogpp_thesis.cvae.preservation.prior_recovery_artifacts import (
    validate_outer_bundle,
    validate_source_inner_bundle,
)
from midogpp_thesis.cvae.preservation.prior_recovery_artifact_shared import (
    _validate_workspace_provenance,
)
from midogpp_thesis.cvae.preservation.prior_recovery_common import selection_evidence_hash
from midogpp_thesis.cvae.objectives import TASK_FISHER_OBJECTIVE
from midogpp_thesis.cvae.preservation.prior_recovery_config import (
    load_prior_recovery_config,
    outer_decision_contract_hash,
    recipe_contract_hash,
)
from midogpp_thesis.cvae.preservation.runtime import EvaluationKey, GenerationKey
from midogpp_thesis.cvae.preservation.source_inner_selection import (
    load_recipe_lock,
    write_recipe_lock,
)
from midogpp_thesis.real_features.classifier_reference.matched_reference import (
    MatchedReferenceConfig,
    run_matched_reference,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from tests.cvae.prior_recovery_test_support import (
    prior_recovery_config,
    write_prior_recovery_fixture,
)


def test_artifacts_are_separate_and_tamper_evident(tmp_path: Path) -> None:
    source_root, outer_root = _run_bundles(tmp_path)
    validate_source_inner_bundle(source_root)
    validate_outer_bundle(outer_root)
    assert (source_root / "tables/checkpoint_reuse_audit.csv").is_file()
    assert not (outer_root / "manifests/recipe_locks/0.json").exists()
    assert json.loads(
        (source_root / "manifests/recipe_locks/0.json").read_text(encoding="utf-8")
    )["may_feed_model_recipe"] is True

    changed_metric = _copy_bundle(source_root, tmp_path / "changed-metric")
    metric_path = changed_metric / "tables/source_inner_metrics.csv"
    rows = _read_csv(metric_path)
    rows[0]["bacc"] = "0.123456"
    _write_csv(metric_path, rows)
    with pytest.raises(ProtocolError, match="evidence bundle hash mismatch"):
        validate_source_inner_bundle(changed_metric)

    changed_source_role = _copy_bundle(source_root, tmp_path / "changed-source-role")
    source_role_path = changed_source_role / "tables/source_inner_metrics.csv"
    source_role_rows = _read_csv(source_role_path)
    source_role_rows[0]["representation_role"] = "oracle"
    source_role_rows[0]["row_role"] = "oracle"
    _rebind_source_evidence(changed_source_role, source_role_rows)
    with pytest.raises(ProtocolError, match="undeclared representation role"):
        validate_source_inner_bundle(changed_source_role)

    changed_source_selection = _copy_bundle(
        source_root,
        tmp_path / "changed-source-selection",
    )
    source_selection_rows = _read_csv(
        changed_source_selection / "tables/source_inner_metrics.csv"
    )
    source_selection_rows[0]["selection_source"] = "forged"
    _rebind_source_evidence(changed_source_selection, source_selection_rows)
    with pytest.raises(ProtocolError, match="selection identity"):
        validate_source_inner_bundle(changed_source_selection)

    changed_classifier_grid = _copy_bundle(
        source_root,
        tmp_path / "changed-classifier-grid",
    )
    tuning_path = changed_classifier_grid / "tables/nested_classifier_tuning.csv"
    tuning_rows = _read_csv(tuning_path)
    forged_spec = json.loads(tuning_rows[0]["classifier_spec"])
    forged_spec["C"] = 10.0
    tuning_rows[0]["classifier_spec"] = json.dumps(forged_spec, sort_keys=True)
    _write_csv(tuning_path, tuning_rows)
    _rebind_source_evidence(
        changed_classifier_grid,
        _read_csv(changed_classifier_grid / "tables/source_inner_metrics.csv"),
    )
    with pytest.raises(ProtocolError, match="tuning identity or fold coverage"):
        validate_source_inner_bundle(changed_classifier_grid)

    changed_source_generation = _copy_bundle(
        source_root,
        tmp_path / "changed-source-generation",
    )
    source_generation_rows = _read_csv(
        changed_source_generation / "tables/source_inner_metrics.csv"
    )
    source_generation_rows[0]["generation_key_hash"] = "forged"
    _rebind_source_evidence(changed_source_generation, source_generation_rows)
    with pytest.raises(ProtocolError, match="generation key does not recompute"):
        validate_source_inner_bundle(changed_source_generation)

    changed_source_evaluation = _copy_bundle(
        source_root,
        tmp_path / "changed-source-evaluation",
    )
    source_evaluation_rows = _read_csv(
        changed_source_evaluation / "tables/source_inner_metrics.csv"
    )
    source_evaluation_rows[0]["evaluation_key_hash"] = "forged"
    _rebind_source_evidence(changed_source_evaluation, source_evaluation_rows)
    with pytest.raises(ProtocolError, match="evaluation key does not recompute"):
        validate_source_inner_bundle(changed_source_evaluation)

    changed_source_budget = _copy_bundle(
        source_root,
        tmp_path / "changed-source-budget",
    )
    source_budget_rows = _read_csv(
        changed_source_budget / "tables/source_inner_metrics.csv"
    )
    budget_row = next(
        row
        for row in source_budget_rows
        if row["arm"] == "C" and row["representation_role"] == "prior"
    )
    counts = [int(value) for value in json.loads(budget_row["generation_class_counts"])]
    counts[0] += 1
    budget_row["generation_class_counts"] = json.dumps(counts)
    generation_hash = GenerationKey(
        source_state_hash=budget_row["sampler_state_hash"],
        generation_seed=int(budget_row["generation_seed"]),
        class_count_vector=(counts[0], counts[1]),
        representation_role=budget_row["representation_role"],
    ).hash
    source_protocol = json.loads(
        (changed_source_budget / "manifests/protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    budget_row["generation_key_hash"] = generation_hash
    budget_row["evaluation_key_hash"] = EvaluationKey(
        generated_artifact_hash=generation_hash,
        frozen_classifier_spec_hash=budget_row["classifier_spec_hash"],
        eval_center=budget_row["inner_pseudo_target_center"],
        eval_row_hash=budget_row["eval_row_hash"],
        metric_schema_version="chance_corrected_bacc_preservation_v1",
        protocol_hash=source_protocol["protocol_hash"],
    ).hash
    _rebind_source_evidence(changed_source_budget, source_budget_rows)
    with pytest.raises(ProtocolError, match="unequal generation class-count budgets"):
        validate_source_inner_bundle(changed_source_budget)

    changed_source_protocol = _copy_bundle(
        source_root,
        tmp_path / "changed-source-protocol",
    )
    source_protocol_path = changed_source_protocol / "manifests/protocol_manifest.json"
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    source_protocol["outer_target_rows_passed_to_training_or_selection"] = True
    source_protocol_path.write_text(
        json.dumps(source_protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="protocol semantic contract"):
        validate_source_inner_bundle(changed_source_protocol)

    changed_source_pairing = _copy_bundle(
        source_root,
        tmp_path / "changed-source-pairing",
    )
    checkpoint_index_path = changed_source_pairing / "manifests/checkpoint_index.json"
    checkpoint_index = json.loads(checkpoint_index_path.read_text(encoding="utf-8"))
    task_record = next(
        record
        for record in checkpoint_index["records"]
        if record["objective_id"] == TASK_FISHER_OBJECTIVE
    )
    task_record["initialization_hash"] = "0" * 64
    checkpoint_index_path.write_text(
        json.dumps(checkpoint_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_path = (
        changed_source_pairing
        / "checkpoints/by_training_key"
        / f"{task_record['training_key_hash']}.json"
    )
    sidecar_path.write_text(
        json.dumps(task_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rebind_source_evidence(
        changed_source_pairing,
        _read_csv(changed_source_pairing / "tables/source_inner_metrics.csv"),
    )
    with pytest.raises(ProtocolError, match="initialization pairing failed"):
        validate_source_inner_bundle(changed_source_pairing)

    changed_timing = _copy_bundle(source_root, tmp_path / "changed-timing")
    timing_path = changed_timing / "tables/runtime_timings.csv"
    timing_rows = _read_csv(timing_path)
    timing_rows[0]["elapsed_seconds"] = "-1"
    _write_csv(timing_path, timing_rows)
    with pytest.raises(ProtocolError, match="finite and nonnegative"):
        validate_source_inner_bundle(changed_timing)

    changed_lock = _copy_bundle(source_root, tmp_path / "changed-lock")
    lock_path = changed_lock / "manifests/recipe_locks/0.json"
    lock = load_recipe_lock(lock_path)
    write_recipe_lock(lock_path, replace(lock, protocol_hash="tampered"))
    with pytest.raises(ProtocolError, match="does not recompute"):
        validate_source_inner_bundle(changed_lock)

    changed_checkpoint = _copy_bundle(outer_root, tmp_path / "changed-checkpoint")
    index = json.loads(
        (changed_checkpoint / "manifests/checkpoint_index.json").read_text(encoding="utf-8")
    )
    (changed_checkpoint / index["records"][0]["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(ProtocolError, match="provenance file hash mismatch"):
        validate_outer_bundle(changed_checkpoint)

    changed_ratio = _copy_bundle(outer_root, tmp_path / "changed-ratio")
    ratio_path = changed_ratio / "tables/preservation_metrics.csv"
    ratio_rows = _read_csv(ratio_path)
    ratio_rows[0]["preservation_ratio"] = "0.999"
    _write_csv(ratio_path, ratio_rows)
    with pytest.raises(ProtocolError, match="preservation ratio"):
        validate_outer_bundle(changed_ratio)

    changed_decision = _copy_bundle(outer_root, tmp_path / "changed-decision")
    decision_path = changed_decision / "reports/decision_report.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["mean_policy_preservation_ratio"] = 999.0
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="decision report does not recompute"):
        validate_outer_bundle(changed_decision)

    changed_lock_binding = _copy_bundle(outer_root, tmp_path / "changed-lock-binding")
    binding_path = changed_lock_binding / "tables/preservation_metrics.csv"
    binding_rows = _read_csv(binding_path)
    binding_rows[0]["recipe_lock_hash"] = "tampered"
    _write_csv(binding_path, binding_rows)
    with pytest.raises(ProtocolError, match="embedded RecipeLock"):
        validate_outer_bundle(changed_lock_binding)

    changed_generation = _copy_bundle(outer_root, tmp_path / "changed-generation-key")
    generation_path = changed_generation / "tables/preservation_metrics.csv"
    generation_rows = _read_csv(generation_path)
    generation_rows[0]["generation_key_hash"] = "tampered"
    _write_csv(generation_path, generation_rows)
    with pytest.raises(ProtocolError, match="generation key does not recompute"):
        validate_outer_bundle(changed_generation)

    changed_sampler = _copy_bundle(outer_root, tmp_path / "changed-sampler")
    sampler_path = changed_sampler / "tables/sampler_realizations.csv"
    sampler_rows = _read_csv(sampler_path)
    sampler_rows[0]["condition_number"] = "999.0"
    _write_csv(sampler_path, sampler_rows)
    with pytest.raises(ProtocolError, match="Sampler state hash"):
        validate_outer_bundle(changed_sampler)

    changed_reference = _copy_bundle(outer_root, tmp_path / "changed-reference")
    reference_path = changed_reference / "manifests/protocol_manifest.json"
    reference_protocol = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_protocol["real_reference_bacc_by_center"]["0"] = 0.123
    reference_path.write_text(
        json.dumps(reference_protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="frozen-reference identity hash"):
        validate_outer_bundle(changed_reference)

    changed_outer_protocol = _copy_bundle(
        outer_root,
        tmp_path / "changed-outer-protocol",
    )
    outer_protocol_path = changed_outer_protocol / "manifests/protocol_manifest.json"
    outer_protocol = json.loads(outer_protocol_path.read_text(encoding="utf-8"))
    outer_protocol["target_eval_labels_used_for_selection"] = True
    outer_protocol_path.write_text(
        json.dumps(outer_protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="protocol semantic contract"):
        validate_outer_bundle(changed_outer_protocol)

    changed_role = _copy_bundle(outer_root, tmp_path / "changed-role")
    role_path = changed_role / "tables/preservation_metrics.csv"
    role_rows = _read_csv(role_path)
    forged = dict(role_rows[0])
    forged["representation_role"] = "oracle"
    forged["row_role"] = "oracle"
    role_rows.append(forged)
    _write_csv(role_path, role_rows)
    with pytest.raises(ProtocolError, match="undeclared representation role"):
        validate_outer_bundle(changed_role)


def test_complete_outer_workspace_provenance_binds_registered_inputs(tmp_path: Path) -> None:
    config_source = (
        Path(__file__).resolve().parents[2]
        / "experiments/midogpp/stages/20_cvae_preservation/configs/prior_recovery_outer_v1.yaml"
    )
    config = load_prior_recovery_config(config_source, expected_mode="outer")
    root = tmp_path / "complete-outer"
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text(
        config_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    hashes = {
        "manifest": "a" * 64,
        "cache": "b" * 64,
        "reference": "c" * 64,
        "source_protocol": "d" * 64,
        "source_selection": "e" * 64,
    }
    protocol = {
        "coverage_mode": "complete",
        "recipe_contract_hash": recipe_contract_hash(config),
        "outer_decision_contract_hash": outer_decision_contract_hash(config),
        "manifest_hash": hashes["manifest"],
        "feature_cache_hash": hashes["cache"],
        "real_reference_protocol_file_sha256": hashes["reference"],
        "source_inner_protocol_file_sha256": hashes["source_protocol"],
        "source_selection_evidence_file_sha256": hashes["source_selection"],
    }
    manifest = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": "midogpp.cvae.prior_recovery_outer.v1",
        "stage": "20_cvae_preservation",
        "claim_scope": "cvae_preservation_only",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            _input_artifact(
                "midogpp_dataset_contract_annotation_patch_v1",
                {"manifest.csv": hashes["manifest"]},
            ),
            _input_artifact(
                "midogpp_virchow2_xyxy_feature_cache_seed42",
                {"embeddings/train.pt": hashes["cache"]},
            ),
            _input_artifact(
                "midogpp_output_eligible_tuned_real_reference_v2",
                {"manifests/protocol_manifest.json": hashes["reference"]},
            ),
            _input_artifact(
                "midogpp_output_cvae_prior_recovery_source_inner_v1",
                {
                    "manifests/protocol_manifest.json": hashes["source_protocol"],
                    "manifests/selection_evidence_manifest.json": hashes[
                        "source_selection"
                    ],
                },
            ),
        ],
    }
    manifest_path = root / "provenance/input_artifacts.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _validate_workspace_provenance(root, protocol=protocol, mode="outer")

    manifest["input_artifacts"][2]["file_integrity"]["files"][0]["computed"][
        "sha256"
    ] = "f" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="Upstream source/reference file identity"):
        _validate_workspace_provenance(root, protocol=protocol, mode="outer")

    manifest["input_artifacts"][2]["file_integrity"]["files"][0]["computed"][
        "sha256"
    ] = hashes["reference"]
    manifest["input_artifacts"].append(dict(manifest["input_artifacts"][0]))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="artifact IDs differ"):
        _validate_workspace_provenance(root, protocol=protocol, mode="outer")


def _run_bundles(tmp_path: Path) -> tuple[Path, Path]:
    manifest, cache = write_prior_recovery_fixture(tmp_path / "fixture")
    source_config = prior_recovery_config(
        mode="source_inner",
        artifact_root=tmp_path / "source",
        manifest=manifest,
        cache=cache,
    )
    source_root = run_source_inner_prior_recovery(source_config)
    reference_root = run_matched_reference(
        MatchedReferenceConfig(
            name="eligible_tuned_real_reference_v2",
            artifact_root=tmp_path / "reference",
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=("0",),
            expected_feature_dim=6,
            allow_partial_test_coverage=True,
        )
    )
    outer_config = prior_recovery_config(
        mode="outer",
        artifact_root=tmp_path / "outer",
        manifest=manifest,
        cache=cache,
        reference=reference_root,
        locks=source_root,
    )
    return source_root, run_outer_prior_recovery(outer_config)


def _copy_bundle(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rebind_source_evidence(root: Path, rows: list[dict[str, str]]) -> None:
    bundle_hash = selection_evidence_hash(
        metric_rows=rows,
        nested_reference_rows=_read_csv(root / "tables/nested_real_references.csv"),
        nested_tuning_rows=_read_csv(root / "tables/nested_classifier_tuning.csv"),
        sampler_rows=_read_csv(root / "tables/sampler_realizations.csv"),
        identity_rows=_read_csv(root / "tables/identity_overlap_audit.csv"),
        protocol_manifest=json.loads(
            (root / "manifests/protocol_manifest.json").read_text(encoding="utf-8")
        ),
        checkpoint_index=json.loads(
            (root / "manifests/checkpoint_index.json").read_text(encoding="utf-8")
        ),
        task_fisher_index=json.loads(
            (root / "manifests/task_fisher_index.json").read_text(encoding="utf-8")
        ),
        feature_frame_index=json.loads(
            (root / "manifests/feature_frame_index.json").read_text(encoding="utf-8")
        ),
    )
    for row in rows:
        row["selection_bundle_hash"] = bundle_hash
    _write_csv(root / "tables/source_inner_metrics.csv", rows)
    evidence_path = root / "manifests/selection_evidence_manifest.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["selection_bundle_hash"] = bundle_hash
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _input_artifact(artifact_id: str, files: dict[str, str]) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "exists": True,
        "semantic_identities_are_file_hashes": False,
        "file_integrity": {
            "status": "HASHES_RECORDED_NO_EXPECTATIONS",
            "files": [
                {
                    "path": path,
                    "exists": True,
                    "computed": {"sha256": digest},
                    "expected": None,
                }
                for path, digest in files.items()
            ],
        },
    }
