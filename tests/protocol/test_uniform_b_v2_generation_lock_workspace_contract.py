from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from midogpp_thesis.workspace.runtime import MidogppWorkspace, WorkspaceError


EXPERIMENT_ID = "midogpp.prior_and_generation.uniform_b_v2_generation_lock.v1"
OUTPUT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
BANK_ID = "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
CONFIG_PATH = (
    "experiments/midogpp/stages/40_prior_and_generation/configs/"
    "uniform_b_v2_generation_lock_v1.yaml"
)
LOCK_SCOPE = "generation_settings_and_frame_lock"
REQUIRED_BUNDLE = {
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/generation_lock.json",
    "manifests/source_generation_plan.json",
    "manifests/equal_union_replicate_plan.json",
    "manifests/content_index.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/generation_health.csv",
}


def test_generation_lock_is_registered_as_the_selectable_stage40_edge() -> None:
    workspace = MidogppWorkspace.load()

    workspace.validate()
    experiment = workspace.get_experiment(EXPERIMENT_ID)
    artifact = workspace.artifacts[OUTPUT_ID]
    stage40 = workspace.stages["40_prior_and_generation"]
    stage60 = workspace.stages["60_routing_and_composition"]
    stage70 = workspace.stages["70_frozen_policy_downstream"]

    assert experiment.status == "active"
    assert experiment.runnable is True
    assert experiment.stage == "40_prior_and_generation"
    assert experiment.claim_scope == LOCK_SCOPE
    assert experiment.config_path == CONFIG_PATH
    assert experiment.output_artifact_id == OUTPUT_ID
    assert experiment.input_artifact_ids == (BANK_ID,)
    assert experiment.runner_argv == (
        "{python}",
        "-m",
        "midogpp_thesis",
        "cvae-generation",
        "uniform-b-v2-generation-lock",
        "--config",
        "{resolved_config}",
        "--artifact-root",
        f"output://{OUTPUT_ID}",
    )
    assert artifact.claim_scope == LOCK_SCOPE
    assert artifact.migration == "canonical_output"
    assert artifact.semantic_identities["generation_lock_hash"] == "34e551425710362e"
    assert artifact.may_feed_recipe_selection is False
    assert artifact.may_feed_deployable_selection is True
    assert set(artifact.required_files) == REQUIRED_BUNDLE
    assert "routing_evidence" not in artifact.forbidden_reuse
    assert "expert_selection_evidence" not in artifact.forbidden_reuse
    assert "nelbo_compatibility_evidence" not in artifact.forbidden_reuse
    assert "synthetic_downstream_utility_evidence" not in artifact.forbidden_reuse
    assert set(stage40["allowed_claim_scopes"]) == {
        "generation_diagnostics_only",
        LOCK_SCOPE,
    }
    assert LOCK_SCOPE in stage60["allowed_input_claim_scopes"]
    assert "generation_diagnostics_only" not in stage60["allowed_input_claim_scopes"]
    assert LOCK_SCOPE in stage70["allowed_input_claim_scopes"]


def test_generation_lock_config_freezes_the_full_nonselective_contract() -> None:
    workspace = MidogppWorkspace.load()
    config = yaml.safe_load((workspace.repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    used_inputs: set[str] = set()
    resolved = workspace.resolve_value(config, require_inputs=False, used_inputs=used_inputs)

    assert used_inputs == {BANK_ID}
    assert resolved["experiment"]["artifact_root"].endswith(
        "artifacts/midogpp/40_prior_and_generation/uniform_b_v2_generation_lock/v1"
    )
    inputs = config["inputs"]
    assert inputs == {
        "bank_root": f"artifact://{BANK_ID}",
        "bank_artifact_id": BANK_ID,
        "expected_bank_lock_hash": "9972a41dcd4814cd",
        "expected_control_lock_hash": "cddbcc3b3343fe38",
        "expected_bank_index_sha256": (
            "5bc46728fd66d5c2c8a72d3da58cc6721e6c0b72c7291ed7b3b6931a4bcc41c9"
        ),
        "expected_control_sha256": (
            "3cae13d5755f27643e0387b1f106f7745b470fb7aafe8d46be41677a1cd8eedd"
        ),
        "expected_content_index_sha256": (
            "6b74fe794bd30cf6c1e42190427e506d1ff50ecd9280b9dcfee2a7592ec6a318"
        ),
        "expected_content_hash": "fb1f1194be44ca41",
    }
    generation = config["generation_contract"]
    assert generation["centers"] == ["0", "1", "2", "3", "5", "6", "7", "8", "9"]
    assert generation["training_seeds"] == [17, 42, 101]
    assert generation["generation_seeds"] == [17, 42, 101]
    assert generation["seed_pairing"] == "cartesian_product"
    assert generation["total_per_class"] == 1024
    assert generation["sources_per_target"] == 8
    assert generation["source_budget_per_class"] == 128
    assert generation["budget_applies_independently_per_replicate"] is True
    assert generation["source_budgets_split_across_seeds"] is False
    assert generation["target_expert_excluded"] is True
    assert generation["target_conditioned_source_weighting"] is False
    assert generation["no_expert_selection"] is True
    assert generation["no_seed_selection"] is True
    assert generation["expected_source_plan_rows"] == 81
    assert generation["expected_target_replicate_rows"] == 81
    assert all(
        target not in candidates and len(candidates) == 8
        for target, candidates in generation["candidate_sources_by_target"].items()
    )
    assert config["model"] == {
        "family": "conditional_variational_autoencoder",
        "input_dim": 256,
        "hidden_dim": 1024,
        "latent_dim": 64,
        "num_hidden_layers": 3,
        "class_conditioning_dim": 2,
        "frozen_checkpoint_required": True,
        "training_allowed": False,
    }
    assert config["source_frame"]["family"] == "source_specific_pca"
    assert config["source_frame"]["model_space_dim"] == 256
    assert config["source_frame"]["reconstructed_embedding_dim"] == 3840
    assert config["source_frame"]["fit_scope"] == "source_center_rows_only"
    assert config["source_frame"]["refit_allowed"] is False
    assert config["aggregate_prior"]["family"] == (
        "class_conditional_shrinkage_full_total_moment"
    )
    assert config["aggregate_prior"]["refit_allowed"] is False
    assert config["deterministic_rng"]["global_rng_forbidden"] is True
    assert config["deterministic_rng"]["namespaces"] == {
        "latent_draw": "uniform_b_v2_source_stream_v1",
        "equal_union_shuffle": "uniform_b_v2_composition_shuffle_v1",
        "health_probe": "uniform_b_v2_generation_lock.health_probe.v1",
    }
    assert config["deterministic_rng"]["latent_draw_key_fields"] == [
        "namespace",
        "bank_lock_hash",
        "expert_lock_hash",
        "generation_seed",
        "class_label",
    ]
    assert config["classifier"] == {
        "family": "sklearn_logistic_regression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 3000,
        "class_weight": None,
        "random_state": 23,
        "threshold_policy": "predict",
        "scaler": "sklearn.preprocessing.StandardScaler",
        "scaler_fit": "synthetic_train_only",
        "fit_in_stage_40": False,
    }
    assert config["health_probe"]["samples_per_class"] == 1
    assert config["health_probe"]["expected_rows"] == 162
    assert config["runtime"]["default_device"] == "cpu"
    claim = config["claim_boundary"]
    assert claim["strict_claim_firewall"] is True
    assert claim["claim_scope"] == LOCK_SCOPE
    assert claim["may_feed_deployable_selection"] is True
    assert all(
        claim[key] is False
        for key in (
            "target_data_used",
            "target_support_used",
            "target_labels_used",
            "target_evaluation_labels_used",
            "routing_evidence_computed",
            "routing_quality_claimed",
            "nelbo_computed",
            "expert_selection_performed",
            "source_weighting_learned",
            "classifier_fit_performed",
            "downstream_utility_computed",
            "stage20_bacc_reused_as_stage40_result",
            "eight_source_control_scored",
        )
    )


def test_generation_lock_is_a_valid_stage60_input() -> None:
    workspace = _workspace_with_stage60_consumer(input_artifact_id=OUTPUT_ID)

    workspace.validate()


def test_stage60_rejects_generation_diagnostics_even_if_marked_selectable() -> None:
    source = MidogppWorkspace.load()
    registry = deepcopy(source.registry_payload)
    catalog = deepcopy(source.catalog_payload)
    diagnostic_id = "test_stage40_generation_diagnostics"
    catalog["artifacts"].append(
        {
            "artifact_id": diagnostic_id,
            "stage": "40_prior_and_generation",
            "canonical_path": (
                "artifacts/midogpp/40_prior_and_generation/test_generation_diagnostics/v1"
            ),
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": "generation_diagnostics_only",
            "may_feed_deployable_selection": True,
        }
    )
    candidate = _workspace_with_stage60_consumer(
        input_artifact_id=diagnostic_id,
        source=source,
        registry=registry,
        catalog=catalog,
    )

    with pytest.raises(WorkspaceError, match="generation_diagnostics_only.*incompatible"):
        candidate.validate()


def test_stage60_rejects_generation_lock_not_marked_for_deployable_selection() -> None:
    source = MidogppWorkspace.load()
    catalog = deepcopy(source.catalog_payload)
    _artifact_payload(catalog, OUTPUT_ID)["may_feed_deployable_selection"] = False
    candidate = _workspace_with_stage60_consumer(
        input_artifact_id=OUTPUT_ID,
        source=source,
        catalog=catalog,
    )

    with pytest.raises(WorkspaceError, match="may_feed_deployable_selection=true"):
        candidate.validate()


def test_stage40_rejects_generation_lock_output_claim_mismatch() -> None:
    source = MidogppWorkspace.load()
    catalog = deepcopy(source.catalog_payload)
    _artifact_payload(catalog, OUTPUT_ID)["claim_scope"] = "generation_diagnostics_only"
    candidate = _clone_workspace(source=source, catalog=catalog)

    with pytest.raises(WorkspaceError, match="output claim_scope.*does not match"):
        candidate.validate()


@pytest.mark.parametrize("forbidden_purpose", ("routing_evidence", "nelbo_compatibility_evidence"))
def test_stage60_rejects_lock_that_forbids_required_routing_reuse(
    forbidden_purpose: str,
) -> None:
    source = MidogppWorkspace.load()
    catalog = deepcopy(source.catalog_payload)
    lock = _artifact_payload(catalog, OUTPUT_ID)
    lock["forbidden_reuse"] = [*lock.get("forbidden_reuse", ()), forbidden_purpose]
    candidate = _workspace_with_stage60_consumer(
        input_artifact_id=OUTPUT_ID,
        source=source,
        catalog=catalog,
    )

    with pytest.raises(WorkspaceError, match="forbids reuse"):
        candidate.validate()


def _workspace_with_stage60_consumer(
    *,
    input_artifact_id: str,
    source: MidogppWorkspace | None = None,
    registry: dict[str, object] | None = None,
    catalog: dict[str, object] | None = None,
) -> MidogppWorkspace:
    source = source or MidogppWorkspace.load()
    registry = registry or deepcopy(source.registry_payload)
    catalog = catalog or deepcopy(source.catalog_payload)
    output_id = "test_uniform_b_v2_router_output"
    catalog["artifacts"].append(
        {
            "artifact_id": output_id,
            "stage": "60_routing_and_composition",
            "canonical_path": (
                "artifacts/midogpp/60_routing_and_composition/test_uniform_b_v2_router/v1"
            ),
            "availability": "generated_on_run",
            "migration": "canonical_output",
            "evidence_label": "TODO_VERIFY_ARTIFACT",
            "claim_scope": "routing_and_composition",
        }
    )
    registry["experiments"].append(
        {
            "experiment_id": "test.uniform_b_v2.router",
            "stage": "60_routing_and_composition",
            "status": "planned",
            "claim_scope": "routing_and_composition",
            "output_artifact_id": output_id,
            "input_artifact_ids": [input_artifact_id],
            "runner": {"argv": ["{python}", "-c", "pass"]},
        }
    )
    return _clone_workspace(source=source, registry=registry, catalog=catalog)


def _artifact_payload(catalog: dict[str, object], artifact_id: str) -> dict[str, object]:
    artifacts = catalog["artifacts"]
    assert isinstance(artifacts, list)
    return next(item for item in artifacts if item["artifact_id"] == artifact_id)


def _clone_workspace(
    *,
    source: MidogppWorkspace,
    registry: dict[str, object] | None = None,
    catalog: dict[str, object] | None = None,
) -> MidogppWorkspace:
    return MidogppWorkspace(
        repo_root=source.repo_root,
        registry=registry or deepcopy(source.registry_payload),
        catalog=catalog or deepcopy(source.catalog_payload),
        workspace=source.workspace_payload,
        protocol_defaults=source.protocol_defaults_payload,
    )
