from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from midogpp_thesis.workspace.cli import build_parser
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


def test_repository_workspace_validates() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()

    assert workspace.get_experiment("midogpp.cvae.tuned_classifier_preservation.v1").claim_scope == (
        "cvae_preservation_only"
    )


def test_repository_experiments_tree_is_declarative_only() -> None:
    workspace = MidogppWorkspace.load()

    assert not list((workspace.repo_root / "experiments" / "midogpp").rglob("*.py"))


def test_workspace_runtime_is_owned_by_canonical_package() -> None:
    assert MidogppWorkspace.__module__ == "midogpp_thesis.workspace.runtime"


def test_central_command_uses_canonical_package_launcher() -> None:
    workspace = MidogppWorkspace.load()

    command = workspace.central_command("midogpp.cvae.tuned_classifier_preservation.v1")

    assert "-m midogpp_thesis workspace run" in command
    assert "experiments/midogpp/cli.py" not in command


@pytest.mark.parametrize(
    ("argv", "expected_command"),
    (
        (["validate"], "validate"),
        (["list"], "list"),
        (["prepare", "example.experiment"], "prepare"),
        (["run", "example.experiment"], "run"),
    ),
)
def test_workspace_cli_exposes_canonical_lifecycle_commands(
    argv: list[str], expected_command: str
) -> None:
    assert build_parser().parse_args(argv).command == expected_command


def test_dataset_contract_resolves_to_frozen_local_artifact() -> None:
    workspace = MidogppWorkspace.load()

    resolved = workspace.resolve_artifact("midogpp_dataset_contract_annotation_patch_v1")

    assert resolved.name == "annotation_patch_v1"
    assert (resolved / "dataset_contract.json").is_file()


def test_feature_cache_resolves_canonical_path_only() -> None:
    workspace = MidogppWorkspace.load()

    resolved = workspace.resolve_artifact(
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        require_exists=False,
    )

    assert resolved.as_posix().endswith(
        "datasets/midogpp/derived/features/virchow2/annotation_patch_xyxy/seed42"
    )


def test_artifact_member_never_falls_back_from_canonical_root(tmp_path: Path) -> None:
    workspace = MidogppWorkspace.load()
    catalog = deepcopy(workspace.catalog_payload)
    entry = next(
        item
        for item in catalog["artifacts"]
        if item["artifact_id"] == "midogpp_virchow2_xyxy_feature_cache_seed42"
    )
    canonical = tmp_path / "canonical"
    physical = tmp_path / "legacy"
    canonical.mkdir()
    (physical / "embeddings").mkdir(parents=True)
    expected = physical / "embeddings" / "train.pt"
    expected.write_bytes(b"cache")
    entry["canonical_path"] = str(canonical)
    entry["physical_path"] = str(physical)
    entry["required_files"] = ["embeddings/train.pt"]
    compatible = MidogppWorkspace(
        repo_root=workspace.repo_root,
        registry=workspace.registry_payload,
        catalog=catalog,
        workspace=workspace.workspace_payload,
        protocol_defaults=workspace.protocol_defaults_payload,
    )

    with pytest.raises(WorkspaceError, match="is unavailable"):
        compatible.resolve_value(
            "artifact://midogpp_virchow2_xyxy_feature_cache_seed42/embeddings/train.pt",
            require_inputs=True,
        )

    assert compatible.resolve_artifact(
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        require_exists=False,
    ) == canonical
    assert expected.is_file()


def test_canonical_config_uris_resolve_without_changing_claim_scope() -> None:
    workspace = MidogppWorkspace.load()
    config = workspace.repo_root / (
        "experiments/midogpp/stages/20_cvae_preservation/configs/"
        "tuned_classifier_preservation_v1.yaml"
    )
    import yaml

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    resolved = workspace.resolve_value(payload, require_inputs=False, used_inputs=set())

    assert resolved["experiment"]["artifact_root"].endswith(
        "artifacts/midogpp/20_cvae_preservation/"
        "virchow2_cvae_midogpp_tuned_classifier_preservation_v1/seed42"
    )
    assert resolved["inputs"]["manifest_path"].endswith(
        "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    )
    assert resolved["claim_boundary"]["forbidden"].startswith("No routing")


def test_signal_control_config_resolves_to_stage_10_output() -> None:
    workspace = MidogppWorkspace.load()
    config = workspace.repo_root / (
        "experiments/midogpp/stages/10_real_feature_reference/configs/"
        "virchow2_signal_controls_v1.yaml"
    )
    import yaml

    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    resolved = workspace.resolve_value(payload, require_inputs=False, used_inputs=set())

    assert resolved["inputs"]["manifest_path"].endswith(
        "datasets/midogpp/contract/annotation_patch_v1/manifest.csv"
    )
    assert resolved["output"]["artifacts_root"].endswith(
        "artifacts/midogpp/10_real_feature_reference/"
        "midogpp_virchow2_real_feature_signal_controls/v1"
    )
    assert "CVAE preservation" in resolved["claim_boundary"]["forbidden"]


def test_prepare_writes_resolved_snapshot_for_argument_driven_runner(tmp_path: Path) -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    registry["experiments"] = [
        entry
        for entry in registry["experiments"]
        if entry["experiment_id"] == "midogpp.real_feature.tuned_classifier.seed42"
    ]
    catalog = deepcopy(source.catalog_payload)
    contract_root = tmp_path / "contract"
    feature_root = tmp_path / "feature_cache"
    contract_root.mkdir()
    (feature_root / "embeddings").mkdir(parents=True)
    (contract_root / "manifest.csv").write_text("sample_id\n", encoding="utf-8")
    (feature_root / "embeddings" / "train.pt").write_bytes(b"cache")
    for entry in catalog["artifacts"]:
        if entry["artifact_id"] == "midogpp_dataset_contract_annotation_patch_v1":
            entry["physical_path"] = str(contract_root)
            entry["required_files"] = ["manifest.csv"]
            entry["authoritative_files"] = []
            entry["expected_file_hashes"] = {
                "manifest.csv": {
                    "algorithm": "sha256",
                    "digest": hashlib.sha256(b"sample_id\n").hexdigest(),
                }
            }
        elif entry["artifact_id"] == "midogpp_virchow2_xyxy_feature_cache_seed42":
            entry["canonical_path"] = str(feature_root)
            entry.pop("physical_path", None)
            entry["required_files"] = ["embeddings/train.pt"]
            entry["expected_file_hashes"] = {
                "embeddings/train.pt": {
                    "algorithm": "sha256",
                    "digest": hashlib.sha256(b"cache").hexdigest(),
                }
            }
    workspace = MidogppWorkspace(
        repo_root=tmp_path,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )

    prepared = workspace.prepare("midogpp.real_feature.tuned_classifier.seed42")

    import yaml

    snapshot = yaml.safe_load(prepared.resolved_config_path.read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == "midogpp_resolved_command_v1"
    assert snapshot["experiment"]["claim_scope"] == "real_feature_transfer_only"
    assert str(contract_root / "manifest.csv") in prepared.argv
    assert str(feature_root / "embeddings" / "train.pt") in prepared.argv
    assert prepared.input_manifest_path.is_file()
    manifest = json.loads(prepared.input_manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "midogpp_input_artifacts_v2"
    for artifact in manifest["input_artifacts"]:
        assert artifact["semantic_identities_are_file_hashes"] is False
        assert artifact["file_integrity"]["status"] == "EXPECTED_FILE_HASHES_MATCH"
        assert all(row["verification"] == "MATCH" for row in artifact["file_integrity"]["files"])


def test_planned_experiment_cannot_prepare() -> None:
    workspace = MidogppWorkspace.load()

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.prepare("midogpp.expert_bank.provenance_clean.v1", require_inputs=False)


def test_router_cannot_consume_diagnostic_utility_artifact() -> None:
    workspace = MidogppWorkspace.load()
    registry = deepcopy(workspace.registry_payload)
    catalog = deepcopy(workspace.catalog_payload)
    catalog["artifacts"].append(
        {
            "artifact_id": "test_router_output",
            "stage": "60_routing_and_composition",
            "canonical_path": "artifacts/midogpp/60_routing_and_composition/test/v1",
            "migration": "canonical_output",
            "availability": "generated_on_run",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": "routing_and_composition",
        }
    )
    registry["experiments"].append(
        {
            "experiment_id": "test.unsafe.router",
            "stage": "60_routing_and_composition",
            "status": "active",
            "claim_scope": "routing_and_composition",
            "output_artifact_id": "test_router_output",
            "input_artifact_ids": ["midogpp_phase1_virchow2_late_import_seed42"],
            "runner": {"argv": ["{python}", "-c", "pass"]},
        }
    )
    unsafe = MidogppWorkspace(
        repo_root=workspace.repo_root,
        registry=registry,
        catalog=catalog,
        workspace=workspace.workspace_payload,
        protocol_defaults=workspace.protocol_defaults_payload,
    )

    with pytest.raises(WorkspaceError, match="consumes forbidden upstream stage"):
        unsafe.validate()


def test_artifact_uri_rejects_parent_traversal() -> None:
    workspace = MidogppWorkspace.load()

    with pytest.raises(WorkspaceError, match="escapes its root"):
        workspace.resolve_value(
            "artifact://midogpp_dataset_contract_annotation_patch_v1/../../secret.txt",
            require_inputs=True,
        )


def _workspace_with_router_input(
    input_artifact: dict[str, object],
    *,
    claim_scope_exception: str | None = None,
) -> MidogppWorkspace:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    artifact_id = str(input_artifact["artifact_id"])
    catalog["artifacts"].extend(
        [
            input_artifact,
            {
                "artifact_id": "test_router_output",
                "stage": "60_routing_and_composition",
                "canonical_path": "artifacts/midogpp/60_routing_and_composition/test/v1",
                "migration": "canonical_output",
                "availability": "generated_on_run",
                "evidence_label": "TODO_VERIFY_ARTIFACT",
                "claim_scope": "routing_and_composition",
            },
        ]
    )
    experiment: dict[str, object] = {
        "experiment_id": "test.router",
        "stage": "60_routing_and_composition",
        "status": "active",
        "claim_scope": "routing_and_composition",
        "output_artifact_id": "test_router_output",
        "input_artifact_ids": [artifact_id],
        "runner": {"argv": ["{python}", "-c", "pass"]},
    }
    if claim_scope_exception is not None:
        experiment["input_claim_scope_exceptions"] = {artifact_id: claim_scope_exception}
    registry["experiments"].append(experiment)
    return MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )


def test_router_rejects_artifact_marked_not_for_deployable_selection() -> None:
    workspace = _workspace_with_router_input(
        {
            "artifact_id": "test_not_deployable",
            "stage": "30_expert_bank",
            "physical_path": "unused",
            "evidence_label": "PASS",
            "claim_scope": "expert_bank_construction_only",
            "may_feed_deployable_selection": False,
        }
    )

    with pytest.raises(WorkspaceError, match="may_feed_deployable_selection=true"):
        workspace.validate()


def test_router_rejects_preservation_claim_scope_without_explicit_exception() -> None:
    workspace = _workspace_with_router_input(
        {
            "artifact_id": "test_preservation_only",
            "stage": "20_cvae_preservation",
            "physical_path": "unused",
            "evidence_label": "PASS",
            "claim_scope": "cvae_preservation_only",
            "may_feed_deployable_selection": True,
        }
    )

    with pytest.raises(WorkspaceError, match="claim_scope 'cvae_preservation_only' is incompatible"):
        workspace.validate()


def test_artifact_specific_claim_scope_exception_requires_rationale_and_can_allow_scope() -> None:
    input_artifact = {
        "artifact_id": "test_reviewed_preservation_input",
        "stage": "20_cvae_preservation",
        "physical_path": "unused",
        "evidence_label": "PASS",
        "claim_scope": "cvae_preservation_only",
        "may_feed_deployable_selection": True,
    }
    reviewed = _workspace_with_router_input(
        input_artifact,
        claim_scope_exception="Protocol review approved this artifact-specific non-evidentiary input.",
    )

    reviewed.validate()

    blank = _workspace_with_router_input(input_artifact, claim_scope_exception=" ")
    with pytest.raises(WorkspaceError, match="requires a non-empty rationale"):
        blank.validate()


def test_claim_scope_exception_does_not_bypass_forbidden_reuse() -> None:
    workspace = _workspace_with_router_input(
        {
            "artifact_id": "test_preservation_forbidden_reuse",
            "stage": "20_cvae_preservation",
            "physical_path": "unused",
            "evidence_label": "PASS",
            "claim_scope": "cvae_preservation_only",
            "may_feed_deployable_selection": True,
            "forbidden_reuse": ["routing_evidence"],
        },
        claim_scope_exception="Scope exception reviewed; reuse prohibition remains binding.",
    )

    with pytest.raises(WorkspaceError, match="forbids reuse as"):
        workspace.validate()


def test_prepare_rejects_expected_file_hash_mismatch_before_writing(tmp_path: Path) -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    registry["experiments"] = [
        entry
        for entry in registry["experiments"]
        if entry["experiment_id"] == "midogpp.real_feature.tuned_classifier.seed42"
    ]
    catalog = deepcopy(source.catalog_payload)
    contract_root = tmp_path / "contract"
    feature_root = tmp_path / "feature"
    contract_root.mkdir()
    (feature_root / "embeddings").mkdir(parents=True)
    (contract_root / "manifest.csv").write_bytes(b"changed")
    (feature_root / "embeddings" / "train.pt").write_bytes(b"cache")
    for entry in catalog["artifacts"]:
        if entry["artifact_id"] == "midogpp_dataset_contract_annotation_patch_v1":
            entry["physical_path"] = str(contract_root)
            entry["required_files"] = ["manifest.csv"]
            entry["authoritative_files"] = []
            entry["expected_file_hashes"] = {
                "manifest.csv": {"algorithm": "sha256", "digest": "0" * 64}
            }
        elif entry["artifact_id"] == "midogpp_virchow2_xyxy_feature_cache_seed42":
            entry["canonical_path"] = str(feature_root)
            entry.pop("physical_path", None)
            entry["required_files"] = ["embeddings/train.pt"]
            entry["expected_file_hashes"] = {
                "embeddings/train.pt": {
                    "algorithm": "sha256",
                    "digest": hashlib.sha256(b"cache").hexdigest(),
                }
            }
    workspace = MidogppWorkspace(
        repo_root=tmp_path,
        registry=registry,
        catalog=catalog,
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )

    with pytest.raises(WorkspaceError, match="file hash mismatch for manifest.csv"):
        workspace.prepare("midogpp.real_feature.tuned_classifier.seed42")

    assert not (tmp_path / "artifacts" / "midogpp").exists()
