from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import shutil

import pytest
import yaml

from midogpp_thesis.data.physical_multiscale.config import load_build_config
from midogpp_thesis.data.physical_multiscale.config_v2 import load_build_config_v2
from midogpp_thesis.data.physical_multiscale.config_v3 import load_build_config_v3
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.profiles import (
    ANNOTATION_LOCAL_POOLING_PILOT_V2,
    CENTER_POOLING_PILOT_V1,
    CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3,
)
from midogpp_thesis.real_features.classifier_reference import cli as classifier_cli
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling import (
    runner as pilot_runner,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.workspace_binding import (
    ANNOTATION_LOCAL_WORKSPACE_BINDING_V2,
    CENTER_POOLING_WORKSPACE_BINDING_V1,
    CLIPPED_BBOX_ANNOTATION_LOCAL_WORKSPACE_BINDING_V3,
    PHYSICAL_MULTISCALE_WORKSPACE_BINDINGS,
    WORKSPACE_BINDINGS_BY_PROFILE_ID,
    get_physical_multiscale_workspace_binding,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


EXPERIMENT_ID = "midogpp.real_feature.physical_multiscale_center_pooling_pilot.v1"
EXPERIMENT_ID_V2 = (
    "midogpp.real_feature.physical_multiscale_annotation_local_pooling_pilot.v2"
)
EXPERIMENT_ID_V3 = (
    "midogpp.real_feature.physical_multiscale_clipped_bbox_annotation_local_pooling_pilot.v3"
)


def test_physical_multiscale_workspace_uses_split_ownership_and_stays_planned() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)

    assert experiment.status == "planned"
    assert experiment.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_dataset_contract_physical_multiscale_center_pooling_pilot_v1",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_virchow2_jpeg_center_pooling_3840_seed42",
        "midogpp_virchow2_physical_multiscale_center_pooling_11520_seed42",
        "midogpp_output_eligible_tuned_real_reference_v2",
    )
    contract = workspace.artifacts[
        "midogpp_dataset_contract_physical_multiscale_center_pooling_pilot_v1"
    ]
    b_cache = workspace.artifacts[
        "midogpp_virchow2_jpeg_center_pooling_3840_seed42"
    ]
    output = workspace.artifacts[experiment.output_artifact_id]
    assert Path(contract.canonical_path).is_relative_to("datasets/midogpp/contract")
    assert Path(b_cache.canonical_path).is_relative_to(
        "datasets/midogpp/derived/features"
    )
    assert Path(output.canonical_path).is_relative_to("artifacts/midogpp")
    assert "cvae_preservation_evidence" in contract.forbidden_reuse
    assert "routing_evidence" in b_cache.forbidden_reuse
    assert "real_feature_reference_evidence" in output.forbidden_reuse

    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.prepare(EXPERIMENT_ID, require_inputs=False)


def test_dataset_build_config_resolves_catalog_inputs_without_path_overrides() -> None:
    config = load_build_config(
        "datasets/midogpp/configs/physical_multiscale_center_pooling_pilot_v1.yaml",
        require_inputs=False,
    )

    assert config.contract_root.is_relative_to(
        config.repo_root / "datasets/midogpp/contract"
    )
    assert config.b_cache_root.is_relative_to(
        config.repo_root / "datasets/midogpp/derived/features"
    )
    assert config.c_cache_root.is_relative_to(
        config.repo_root / "datasets/midogpp/derived/features"
    )
    assert config.raw_root == config.repo_root / "datasets/midogpp/raw/MIDOGpp"
    assert config.bridge_minimum_cosine == 0.99999
    assert config.bridge_maximum_relative_l2 == 0.001
    assert config.bridge_minimum_prediction_agreement == 0.999
    assert config.bridge_maximum_equal_center_bacc_delta == 0.001


def test_annotation_local_v2_is_distinct_planned_and_stage_firewalled() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID_V2)

    assert experiment.status == "planned"
    assert experiment.output_artifact_id == (
        "midogpp_output_real_feature_physical_multiscale_annotation_local_pooling_pilot_v2"
    )
    assert experiment.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_dataset_contract_physical_multiscale_annotation_local_pooling_pilot_v2",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_virchow2_annotation_local_pooling_bc_bundle_seed42_v2",
        "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v2_seed42",
        "midogpp_virchow2_physical_multiscale_annotation_local_pooling_11520_v2_seed42",
        "midogpp_output_eligible_tuned_real_reference_v2",
    )
    protected = (
        "midogpp_dataset_contract_physical_multiscale_annotation_local_pooling_pilot_v2",
        "midogpp_virchow2_annotation_local_pooling_bc_bundle_seed42_v2",
        "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v2_seed42",
        "midogpp_virchow2_physical_multiscale_annotation_local_pooling_11520_v2_seed42",
    )
    forbidden = {
        "cvae_preservation_evidence",
        "expert_bank_evidence",
        "generation_evidence",
        "all_candidate_utility_diagnostic",
        "synthetic_downstream_utility_evidence",
        "routing_evidence",
        "expert_selection_evidence",
        "nelbo_compatibility_evidence",
    }
    for artifact_id in protected:
        artifact = workspace.artifacts[artifact_id]
        assert forbidden.issubset(set(artifact.forbidden_reuse))
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False
    with pytest.raises(WorkspaceError, match="status='planned'"):
        workspace.prepare(EXPERIMENT_ID_V2, require_inputs=False)


def test_clipped_bbox_v3_is_distinct_diagnostic_and_stage_firewalled() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID_V3)

    assert experiment.status == "diagnostic"
    assert experiment.runnable is True
    assert experiment.output_artifact_id == (
        "midogpp_output_real_feature_physical_multiscale_clipped_bbox_"
        "annotation_local_pooling_pilot_v3"
    )
    assert experiment.input_artifact_ids == (
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_dataset_contract_physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3",
        "midogpp_virchow2_xyxy_feature_cache_seed42",
        "midogpp_virchow2_clipped_bbox_annotation_local_pooling_bc_bundle_seed42_v3",
        "midogpp_virchow2_annotation_jpeg_fixed_center_pooling_3840_v3_seed42",
        "midogpp_virchow2_physical_multiscale_clipped_bbox_annotation_local_pooling_11520_v3_seed42",
        "midogpp_output_eligible_tuned_real_reference_v2",
    )
    protected = (
        experiment.input_artifact_ids[1],
        *experiment.input_artifact_ids[3:6],
    )
    forbidden = {
        "cvae_preservation_evidence",
        "expert_bank_evidence",
        "generation_evidence",
        "all_candidate_utility_diagnostic",
        "synthetic_downstream_utility_evidence",
        "routing_evidence",
        "expert_selection_evidence",
        "nelbo_compatibility_evidence",
    }
    for artifact_id in protected:
        artifact = workspace.artifacts[artifact_id]
        assert forbidden.issubset(set(artifact.forbidden_reuse))
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False
        assert set(artifact.required_files).issubset(artifact.expected_file_hashes)
        assert artifact.evidence_label == "AUDIT_ONLY"


def test_profile_workspace_bindings_are_immutable_and_exact() -> None:
    assert PHYSICAL_MULTISCALE_WORKSPACE_BINDINGS == (
        CENTER_POOLING_WORKSPACE_BINDING_V1,
        ANNOTATION_LOCAL_WORKSPACE_BINDING_V2,
        CLIPPED_BBOX_ANNOTATION_LOCAL_WORKSPACE_BINDING_V3,
    )
    expected = {
        CENTER_POOLING_PILOT_V1: EXPERIMENT_ID,
        ANNOTATION_LOCAL_POOLING_PILOT_V2: EXPERIMENT_ID_V2,
        CLIPPED_BBOX_ANNOTATION_LOCAL_POOLING_PILOT_V3: EXPERIMENT_ID_V3,
    }
    assert {
        profile_id: binding.experiment_id
        for profile_id, binding in WORKSPACE_BINDINGS_BY_PROFILE_ID.items()
    } == expected
    for profile_id, experiment_id in expected.items():
        binding = get_physical_multiscale_workspace_binding(profile_id)
        assert binding.experiment_id == experiment_id
        assert binding.profile_id == profile_id
    with pytest.raises(FrozenInstanceError):
        CENTER_POOLING_WORKSPACE_BINDING_V1.output_id = "drifted"  # type: ignore[misc]
    with pytest.raises(TypeError):
        WORKSPACE_BINDINGS_BY_PROFILE_ID["other"] = (  # type: ignore[index]
            CENTER_POOLING_WORKSPACE_BINDING_V1
        )


def test_v2_failed_audit_and_v3_audit_are_stage90_only() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()

    v2 = workspace.artifacts[
        "midogpp_physical_multiscale_annotation_local_pooling_v2_failed_geometry_audit"
    ]
    v3 = workspace.artifacts[
        "midogpp_physical_multiscale_clipped_bbox_annotation_local_pooling_v3_geometry_audit"
    ]
    for artifact in (v2, v3):
        assert artifact.stage == "90_oracles_and_diagnostics"
        assert artifact.evidence_label == "AUDIT_ONLY"
        assert artifact.claim_scope == "diagnostic_only"
        assert artifact.may_feed_recipe_selection is False
        assert artifact.may_feed_deployable_selection is False


def test_annotation_local_dataset_config_resolves_one_atomic_cache_parent() -> None:
    config = load_build_config_v2(
        "datasets/midogpp/configs/physical_multiscale_annotation_local_pooling_pilot_v2.yaml",
        require_inputs=False,
    )

    assert config.contract_root.is_relative_to(
        config.repo_root / "datasets/midogpp/contract"
    )
    assert config.b_cache_root.parent == config.cache_bundle_root
    assert config.c_cache_root.parent == config.cache_bundle_root
    assert config.b_cache_root.name == "b_3840"
    assert config.c_cache_root.name == "c_11520"
    assert config.expected_row_count == 9648
    assert config.expected_slide_count == 216


def test_clipped_bbox_v3_dataset_config_freezes_qc_and_atomic_parent() -> None:
    config = load_build_config_v3(
        "datasets/midogpp/configs/"
        "physical_multiscale_clipped_bbox_annotation_local_pooling_pilot_v3.yaml",
        require_inputs=False,
    )

    assert config.contract_root.is_relative_to(
        config.repo_root / "datasets/midogpp/contract"
    )
    assert config.b_cache_root.parent == config.cache_bundle_root
    assert config.c_cache_root.parent == config.cache_bundle_root
    assert config.minimum_clipped_bbox_area_fraction == 0.25
    assert config.expected_clipped_bbox_count == 84
    assert config.expected_row_count == 9648
    assert config.expected_slide_count == 216


def test_v3_stage10_config_consumes_every_declared_registry_input() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID_V3)
    path = Path(experiment.config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    used: set[str] = set()

    workspace.resolve_value(
        payload,
        require_inputs=False,
        used_inputs=used,
    )

    assert used == set(experiment.input_artifact_ids)
    assert (
        "midogpp_virchow2_clipped_bbox_annotation_local_pooling_bc_bundle_seed42_v3"
        in used
    )


@pytest.mark.parametrize("fail", (False, True))
def test_v3_registered_prepare_cli_runner_uses_one_atomic_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    source_root = Path.cwd()
    shutil.copytree(
        source_root / "experiments" / "midogpp",
        tmp_path / "experiments" / "midogpp",
    )
    registry_path = tmp_path / "experiments" / "midogpp" / "registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for experiment in registry["experiments"]:
        if experiment["experiment_id"] == EXPERIMENT_ID_V3:
            experiment["status"] = "diagnostic"
            break
    else:  # pragma: no cover - registry drift gives a clearer assertion.
        raise AssertionError("v3 experiment is absent from the copied registry")
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    workspace = MidogppWorkspace.load(tmp_path)
    prepared = workspace.prepare(EXPERIMENT_ID_V3, require_inputs=False)
    resolved_bytes = prepared.resolved_config_path.read_bytes()
    provenance_bytes = prepared.input_manifest_path.read_bytes()
    final_root = prepared.artifact_root
    assert final_root.is_dir()

    monkeypatch.setattr(
        pilot_runner,
        "validate_production_workspace_binding",
        lambda _config: None,
    )

    def execute_in_stage(
        config: object,
        *,
        production_binding_validated: bool = False,
    ) -> Path:
        stage = config.artifact_root  # type: ignore[attr-defined]
        assert production_binding_validated is True
        assert not final_root.exists()
        assert (stage / "config.resolved.yaml").read_bytes() == resolved_bytes
        assert (
            stage / "provenance" / "input_artifacts.json"
        ).read_bytes() == provenance_bytes
        (stage / "claim-bearing.txt").write_text("partial", encoding="utf-8")
        if fail:
            raise RuntimeError("injected registered execution failure")
        return stage

    monkeypatch.setattr(
        pilot_runner,
        "_run_physical_multiscale_pilot_in_place",
        execute_in_stage,
    )
    command_index = prepared.argv.index("real-feature-classifier")
    cli_args = list(prepared.argv[command_index + 1 :])

    if fail:
        with pytest.raises(
            RuntimeError,
            match="injected registered execution failure",
        ):
            classifier_cli.main(cli_args)
        assert not final_root.exists()
        quarantines = tuple(
            final_root.parent.glob(f".{final_root.name}.quarantine-*")
        )
        assert len(quarantines) == 1
        published = quarantines[0]
    else:
        assert classifier_cli.main(cli_args) == 0
        assert final_root.is_dir()
        assert not final_root.with_name(f".{final_root.name}.staging").exists()
        published = final_root

    assert (published / "config.resolved.yaml").read_bytes() == resolved_bytes
    assert (
        published / "provenance" / "input_artifacts.json"
    ).read_bytes() == provenance_bytes
    assert (published / "claim-bearing.txt").read_text(encoding="utf-8") == "partial"
