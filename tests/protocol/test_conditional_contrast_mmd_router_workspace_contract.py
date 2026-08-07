from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.conditional_contrast_mmd_router.config import (
    load_conditional_contrast_mmd_router_config,
)
from midogpp_thesis.cvae.diagnostics.mmd_kmm_router.profiles import (
    CONDITIONAL_PROFILE,
    CONDITIONAL_ROUTER_MODE,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_validation_conditional_contrast_mmd_router_v1.yaml"
)


def _yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_conditional_profile_config_and_cli_are_frozen() -> None:
    config = load_conditional_contrast_mmd_router_config(CONFIG)
    assert config.profile == CONDITIONAL_PROFILE
    assert config.router_mode == CONDITIONAL_ROUTER_MODE
    assert config.proxy["family"] == "class_conditional_contrast_mmd_kmm"
    assert config.proxy["class_weights"] == [0.5, 0.5]
    assert config.proxy["contrast_weight"] == 1.0
    assert config.proxy["maximum_uniform_l1"] == 0.25
    assert config.proxy["pooled_reference_regularization"] == 0.05
    assert config.proxy["pooled_reference_source"].endswith(
        "before_stability_gates"
    )
    assert config.proxy["previous_pooled_mmd_output_used"] is False
    assert config.runtime == {
        "workstation_profile": "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        "generation_devices": ["cuda:0", "cuda:1"],
        "kernel_devices": ["cuda:0", "cuda:1"],
        "generation_workers_per_device": 1,
        "kernel_workers_per_device": 1,
        "classifier_workers": 4,
        "classifier_threads_per_worker": 3,
        "multiprocessing_start_method": "spawn",
        "generated_cache_format": "float32_npy_memmap",
        "kernel_batch_rows": 1024,
        "one_expert_per_gpu_at_a_time": True,
        "maximum_unique_classifier_fit_count": 162,
        "resume_policy": "hash_validated_phase_and_cell_checkpoints",
    }
    parsed = build_parser().parse_args(
        (
            "conditional-contrast-mmd-router-diagnostic",
            "--config",
            str(CONFIG),
            "--artifact-root",
            "/tmp/conditional-contrast-mmd",
        )
    )
    assert parsed.surface == "conditional-contrast-mmd-router-diagnostic"


def test_registry_and_catalog_fence_consumed_inputs_and_output() -> None:
    registry = _yaml(ROOT / "experiments/midogpp/registry.yaml")
    experiments = {
        str(row["experiment_id"]): row for row in registry["experiments"]
    }
    experiment = experiments[CONDITIONAL_PROFILE.experiment_id]
    assert tuple(experiment["input_artifact_ids"]) == CONDITIONAL_PROFILE.input_artifact_ids
    assert experiment["claim_scope"] == "diagnostic_only"
    assert experiment["status"] == "diagnostic"
    assert experiment["runner"]["argv"][3] == "cvae-diagnostics"
    assert (
        "midogpp_output_uniform_b_v2_consumed_validation_mmd_kmm_router_v1"
        not in experiment["input_artifact_ids"]
    )

    catalog = _yaml(ROOT / "experiments/midogpp/artifact_catalog.yaml")
    artifacts = {str(row["artifact_id"]): row for row in catalog["artifacts"]}
    for artifact_id in (
        CONDITIONAL_PROFILE.validation_cache_artifact_id,
        CONDITIONAL_PROFILE.validation_manifest_artifact_id,
    ):
        artifact = artifacts[artifact_id]
        semantics = artifact["semantic_identities"]
        assert semantics["fresh_evidence"] == "false"
        assert (
            semantics["authorized_consumer_experiment_ids"]
            == CONDITIONAL_PROFILE.experiment_id
        )
        assert artifact["may_feed_recipe_selection"] is False
        assert artifact["may_feed_deployable_selection"] is False

    pooled_cache = artifacts[
        "midogpp_stage90_mmd_kmm_router_validation_cache_v1"
    ]
    conditional_cache = artifacts[
        CONDITIONAL_PROFILE.validation_cache_artifact_id
    ]
    assert (
        conditional_cache["expected_file_hashes"]
        == pooled_cache["expected_file_hashes"]
    )
    pooled_manifest = artifacts[
        "midogpp_stage90_mmd_kmm_router_validation_manifest_v1"
    ]
    conditional_manifest = artifacts[
        CONDITIONAL_PROFILE.validation_manifest_artifact_id
    ]
    assert (
        conditional_manifest["expected_file_hashes"]
        == pooled_manifest["expected_file_hashes"]
    )

    output = artifacts[CONDITIONAL_PROFILE.output_artifact_id]
    assert output["claim_scope"] == "diagnostic_only"
    assert output["semantic_identities"]["promotion_eligible"] == "false"
    assert output["semantic_identities"]["may_feed_stage60"] == "false"
    assert output["semantic_identities"]["may_feed_stage70"] == "false"
    assert "routing_evidence" in output["forbidden_reuse"]
    assert "oracle_and_diagnostic_evidence" in output["forbidden_reuse"]
