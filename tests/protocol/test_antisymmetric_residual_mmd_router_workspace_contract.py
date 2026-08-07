from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.config import (
    load_antisymmetric_residual_mmd_config,
)
from midogpp_thesis.cvae.diagnostics.antisymmetric_residual_mmd_router.contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router_v1.yaml"
)


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_antisymmetric_config_cli_and_workstation_profile_are_frozen() -> None:
    config = load_antisymmetric_residual_mmd_config(CONFIG)
    assert config.experiment_id == EXPERIMENT_ID
    assert config.proxy["family"] == "antisymmetric_class_residual_robust_mmd"
    assert config.proxy["weight_parameterization"] == (
        "w_class0=uniform+delta;w_class1=uniform-delta"
    )
    assert config.proxy["residual_l1_enforced_inside_solver"] is True
    assert config.protocol["heldout_case_excluded_from_own_route"] is True
    assert config.protocol["cohort_unlabeled_embeddings_used_for_other_case_routes"] is True
    assert config.protocol["evaluation_embeddings_available_to_router"] is True
    assert config.protocol[
        "cohort_evaluation_embeddings_available_for_other_case_routes"
    ] is True
    assert config.protocol[
        "heldout_evaluation_embeddings_available_to_own_route"
    ] is False
    assert config.claim_boundary["cross_fitted_transductive_diagnostic"] is True
    assert config.claim_boundary["proxy_is_nelbo_compatibility"] is False
    assert config.runtime["generation_devices"] == ["cuda:0", "cuda:1"]
    assert config.runtime["cuda_visible_devices"] == "0,1"
    assert config.runtime["classifier_workers"] == 4
    assert config.runtime["classifier_threads_per_worker"] == 3
    assert config.runtime["tf32_disabled_in_gpu_workers"] is True
    assert config.runtime["minimum_gpu_free_mib_per_device"] == 18000
    assert config.runtime["minimum_artifact_disk_free_bytes"] == 8 * 1024**3
    assert config.runtime["maximum_unique_classifier_fit_count"] == 315
    parsed = build_parser().parse_args(
        (
            "antisymmetric-residual-mmd-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/antisymmetric-residual-mmd",
        )
    )
    assert parsed.surface == "antisymmetric-residual-mmd-router-diagnostic"


def test_antisymmetric_registry_and_catalog_are_terminally_fenced() -> None:
    registry = _yaml(ROOT / "experiments/midogpp/registry.yaml")
    experiments = {str(row["experiment_id"]): row for row in registry["experiments"]}
    experiment = experiments[EXPERIMENT_ID]
    assert tuple(experiment["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert experiment["claim_scope"] == "diagnostic_only"
    assert experiment["runner"]["argv"][4] == "antisymmetric-residual-mmd-router-diagnostic"
    assert (
        experiment["runner"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    )
    assert not any("stage90" in value and "antisymmetric" not in value for value in experiment["input_artifact_ids"])

    catalog = _yaml(ROOT / "experiments/midogpp/artifact_catalog.yaml")
    artifacts = {str(row["artifact_id"]): row for row in catalog["artifacts"]}
    for artifact_id in (
        VALIDATION_CACHE_ARTIFACT_ID,
        VALIDATION_MANIFEST_ARTIFACT_ID,
    ):
        artifact = artifacts[artifact_id]
        semantics = artifact["semantic_identities"]
        assert semantics["fresh_evidence"] == "false"
        assert semantics["authorized_consumer_experiment_ids"] == EXPERIMENT_ID
        assert artifact["may_feed_recipe_selection"] is False
        assert artifact["may_feed_deployable_selection"] is False

    source_cache = artifacts["midogpp_stage90_mmd_kmm_router_validation_cache_v1"]
    assert artifacts[VALIDATION_CACHE_ARTIFACT_ID]["expected_file_hashes"] == source_cache["expected_file_hashes"]
    source_manifest = artifacts["midogpp_stage90_mmd_kmm_router_validation_manifest_v1"]
    assert artifacts[VALIDATION_MANIFEST_ARTIFACT_ID]["expected_file_hashes"] == source_manifest["expected_file_hashes"]

    output = artifacts[OUTPUT_ARTIFACT_ID]
    semantics = output["semantic_identities"]
    assert semantics["promotion_eligible"] == "false"
    assert semantics["may_feed_stage60"] == "false"
    assert semantics["may_feed_stage70"] == "false"
    assert "routing_evidence" in output["forbidden_reuse"]
    assert "nelbo_compatibility_evidence" in output["forbidden_reuse"]
    assert "oracle_and_diagnostic_evidence" in output["forbidden_reuse"]
