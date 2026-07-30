from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    INITIALIZATION_SEEDS,
    load_audit_config,
    load_snapshot_build_config,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.entrypoint import (
    SNAPSHOT_CANONICAL_RELATIVE,
    SNAPSHOT_EXPERIMENT_ID,
    assert_workspace_prepared_entrypoint,
)
from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.runner import (
    run_b_paired_reparameterization_audit,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.workspace.runtime import MidogppWorkspace


SNAPSHOT_EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_paired_reparameterization_snapshot.v1"
)
AUDIT_EXPERIMENT_ID = "midogpp.oracle.uniform_b_paired_reparameterization_audit.v1"
SNAPSHOT_ARTIFACT_ID = (
    "midogpp_stage90_uniform_b_paired_reparameterization_snapshot_v1"
)
AUDIT_ARTIFACT_ID = "midogpp_output_uniform_b_paired_reparameterization_audit_v1"
CACHE_ARTIFACT_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
CONFIG_ROOT = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "midogpp"
    / "stages"
    / "90_oracles_and_diagnostics"
    / "configs"
)

NON_DIAGNOSTIC_REUSE = {
    "real_feature_reference_evidence",
    "cvae_preservation_evidence",
    "expert_bank_evidence",
    "generation_evidence",
    "all_candidate_utility_diagnostic",
    "synthetic_downstream_utility_evidence",
    "routing_evidence",
    "expert_selection_evidence",
    "nelbo_compatibility_evidence",
}
SNAPSHOT_REQUIRED_FILES = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/snapshot_manifest.json",
    "manifests/key_inventory.json",
    "manifests/content_index.json",
    "reports/leakage_report.json",
    "reports/validation_report.json",
    "reports/run_state.json",
}
AUDIT_REQUIRED_FILES = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/snapshot_binding.json",
    "manifests/key_inventory.json",
    "manifests/content_index.json",
    "reports/run_state.json",
    "reports/leakage_provenance_report.json",
    "reports/validation_report.json",
    "reports/audit_decision.json",
    "reports/runtime_summary.json",
    "tables/job_inventory.csv",
    "tables/replay_trace_audit.csv",
    "tables/legacy_v2_validation.csv",
    "tables/controlled_metrics.csv",
    "tables/paired_comparison.csv",
    "tables/consumption_audit.csv",
    "tables/decoded_predictions.csv",
}


def test_stage90_snapshot_and_audit_are_registered_audit_only() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()

    snapshot = workspace.get_experiment(SNAPSHOT_EXPERIMENT_ID)
    audit = workspace.get_experiment(AUDIT_EXPERIMENT_ID)
    snapshot_output = workspace.artifacts[SNAPSHOT_ARTIFACT_ID]
    audit_output = workspace.artifacts[AUDIT_ARTIFACT_ID]

    assert snapshot.stage == audit.stage == "90_oracles_and_diagnostics"
    assert snapshot.status == audit.status == "diagnostic"
    assert snapshot.claim_scope == audit.claim_scope == "diagnostic_only"
    assert snapshot.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        CACHE_ARTIFACT_ID,
    )
    assert audit.input_artifact_ids == (SNAPSHOT_ARTIFACT_ID,)
    assert snapshot.runner_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert audit.runner_env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert snapshot_output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/inputs/"
        "uniform_b_paired_reparameterization_snapshot_v1"
    )
    assert audit_output.canonical_path == (
        "artifacts/midogpp/90_oracles_and_diagnostics/"
        "uniform_b_paired_reparameterization_audit/v1"
    )

    for output, required in (
        (snapshot_output, SNAPSHOT_REQUIRED_FILES),
        (audit_output, AUDIT_REQUIRED_FILES),
    ):
        assert output.evidence_label == "AUDIT_ONLY"
        assert output.claim_scope == "diagnostic_only"
        assert set(output.required_files) == required
        assert set(output.forbidden_reuse) == NON_DIAGNOSTIC_REUSE
        assert output.may_feed_recipe_selection is False
        assert output.may_feed_deployable_selection is False


def test_canonical_b_cache_is_bound_to_exact_workstation_hashes() -> None:
    cache = MidogppWorkspace.load().artifacts[CACHE_ARTIFACT_ID]

    assert cache.stage == "derived_features"
    assert cache.availability == "workstation_only"
    assert cache.canonical_path == (
        "datasets/midogpp/derived/features/virchow2/"
        "uniform_b_canonical_reference_v1/seed42"
    )
    assert {item.algorithm for item in cache.expected_file_hashes.values()} == {"sha256"}
    assert cache.semantic_identities["feature_cache_hash"] == (
        "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
    )
    assert {
        path: expectation.digest
        for path, expectation in cache.expected_file_hashes.items()
    } == {
        "embeddings/train.pt": (
            "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
        ),
        "manifests/frozen_cache_protocol.json": (
            "a4faf27a427cfb424789e5592048aa748a057f37124566d46b8b6c557e2bfe69"
        ),
        "manifests/content_index.json": (
            "307991668f11454da69e3798feb23a2e899e1a00c2ee5132b031e7f7fb9ab82e"
        ),
        "reports/cache_builder_report.json": (
            "3e3c40449196dc6db9fe0ab982defa86afb1094e3d958e944875396bc363b0ec"
        ),
        "reports/validation_report.json": (
            "e8b69f557ea92ac8e70a20e504150aba1c947f2b47f735b34e3ca7147efcf6b7"
        ),
    }


def test_snapshot_config_uses_only_canonical_inputs_and_inert_legacy_strings() -> None:
    path = CONFIG_ROOT / "uniform_b_paired_reparameterization_snapshot_v1.yaml"
    payload = _read_yaml(path)
    config = load_snapshot_build_config(path)

    assert payload["inputs"] == {
        "manifest_path": (
            "artifact://midogpp_dataset_contract_annotation_patch_v1/manifest.csv"
        ),
        "b_feature_cache_path": (
            "artifact://midogpp_virchow2_uniform_b_canonical_train_cache_seed42/"
            "embeddings/train.pt"
        ),
    }
    lineage = payload["historical_lineage"]
    assert lineage["historical_paths_read"] is False
    assert str(lineage["predecessor_root_provenance_only"]).startswith("/home/")
    assert "://" not in str(lineage["predecessor_root_provenance_only"])
    assert all(
        len(str(digest)) == 64
        for digest in lineage["predecessor_bundle_hashes"].values()
    )

    legacy = payload["legacy_expectations"]
    expected_coordinates = {
        (center, seed) for center in AUDIT_CENTERS for seed in INITIALIZATION_SEEDS
    }
    assert len(legacy) == 12
    assert len(config.legacy_expectations) == 12
    assert config.historical_lineage.historical_paths_read is False
    assert {(str(row["center"]), int(row["training_seed"])) for row in legacy} == (
        expected_coordinates
    )
    assert all(len(str(row["checkpoint_hash"])) == 64 for row in legacy)
    assert all(
        len(str(row["expected_decode_prediction_sha256"])) == 64 for row in legacy
    )


def test_audit_config_has_one_snapshot_input_and_exact_36_key_product() -> None:
    path = CONFIG_ROOT / "uniform_b_paired_reparameterization_audit_v1.yaml"
    payload = _read_yaml(path)
    config = load_audit_config(path)

    artifact_uris = tuple(_artifact_uris(payload))
    assert artifact_uris == (
        "artifact://midogpp_stage90_uniform_b_paired_reparameterization_snapshot_v1",
    )
    assert config.snapshot_artifact_id == SNAPSHOT_ARTIFACT_ID
    coordinates = {
        (center, seed, candidate)
        for center in config.centers
        for seed in config.initialization_seeds
        for candidate in config.candidates
    }
    assert config.centers == AUDIT_CENTERS
    assert config.initialization_seeds == INITIALIZATION_SEEDS
    assert config.candidates == AUDIT_CANDIDATES
    assert len(coordinates) == 36
    assert len(tuple(row for row in coordinates if row[2] != AUDIT_CANDIDATES[0])) == 24


def test_stage90_entrypoint_rejects_output_outside_registered_root(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "experiments/midogpp").mkdir(parents=True)
    (repo / "experiments/midogpp/registry.yaml").write_text(
        "schema_version: fixture\n",
        encoding="utf-8",
    )
    output = repo / "arbitrary-output"
    _write_workspace_entrypoint_fixture(output)

    with pytest.raises(ProtocolError, match="registered canonical output"):
        assert_workspace_prepared_entrypoint(
            resolved_config_path=output / "config.resolved.yaml",
            artifact_root=output,
            experiment_id=SNAPSHOT_EXPERIMENT_ID,
            canonical_relative=SNAPSHOT_CANONICAL_RELATIVE,
            input_artifact_ids=("canonical-input",),
            expected_input_members={"canonical-input": output / "input/member.bin"},
        )


def test_stage90_entrypoint_rejects_unregistered_absolute_input(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "experiments/midogpp").mkdir(parents=True)
    (repo / "experiments/midogpp/registry.yaml").write_text(
        "schema_version: fixture\n",
        encoding="utf-8",
    )
    output = repo / SNAPSHOT_CANONICAL_RELATIVE
    _write_workspace_entrypoint_fixture(output)
    unregistered = tmp_path / "unregistered/member.bin"

    with pytest.raises(ProtocolError, match="escapes workspace artifact"):
        assert_workspace_prepared_entrypoint(
            resolved_config_path=output / "config.resolved.yaml",
            artifact_root=output,
            experiment_id=SNAPSHOT_EXPERIMENT_ID,
            canonical_relative=SNAPSHOT_CANONICAL_RELATIVE,
            input_artifact_ids=("canonical-input",),
            expected_input_members={"canonical-input": unregistered},
        )


def test_audit_runner_rejects_before_writing_unregistered_output(
    tmp_path: Path,
) -> None:
    config = load_audit_config(
        CONFIG_ROOT / "uniform_b_paired_reparameterization_audit_v1.yaml"
    )
    output = tmp_path / "unregistered-output"

    with pytest.raises(ProtocolError, match="workspace-prepared"):
        run_b_paired_reparameterization_audit(
            config,
            artifact_root=output,
            resolved_config_path=tmp_path / "not-a-workspace-config.yaml",
        )

    assert not output.exists()


def _write_workspace_entrypoint_fixture(output: Path) -> None:
    (output / "provenance").mkdir(parents=True)
    (output / "config.resolved.yaml").write_text(
        "schema_version: fixture\n",
        encoding="utf-8",
    )
    input_root = output / "input"
    input_root.mkdir()
    (input_root / "member.bin").write_bytes(b"fixture")
    (output / "provenance/input_artifacts.json").write_text(
        json.dumps(
            {
                "schema_version": "midogpp_input_artifacts_v2",
                "experiment_id": SNAPSHOT_EXPERIMENT_ID,
                "selection_used_target_eval_artifacts": False,
                "input_artifacts": [
                    {
                        "artifact_id": "canonical-input",
                        "resolved_path": str(input_root),
                        "exists": True,
                        "file_integrity": {"files": []},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _read_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _artifact_uris(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _artifact_uris(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_uris(child)
    elif isinstance(value, str) and value.startswith("artifact://"):
        yield value
