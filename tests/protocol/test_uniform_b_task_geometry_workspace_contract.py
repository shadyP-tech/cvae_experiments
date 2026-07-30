from __future__ import annotations

from pathlib import Path

import yaml

from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "midogpp.cvae.uniform_b_geco_task_geometry_source_inner.v1"
OUTPUT_ID = "midogpp_output_cvae_uniform_b_geco_task_geometry_source_inner_v1"


def test_uniform_b_task_geometry_workspace_contract_is_non_consumable() -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    registry = yaml.safe_load(
        (ROOT / "experiments/midogpp/registry.yaml").read_text(encoding="utf-8")
    )
    experiments = {
        row["experiment_id"]: row for row in registry["experiments"]
    }
    experiment = experiments[EXPERIMENT_ID]
    assert experiment["input_artifact_ids"] == [
        "midogpp_dataset_contract_annotation_patch_v1",
        "midogpp_virchow2_uniform_b_canonical_train_cache_seed42",
    ]
    assert experiment["output_artifact_id"] == OUTPUT_ID
    consumers = [
        row["experiment_id"]
        for row in registry["experiments"]
        if OUTPUT_ID in row.get("input_artifact_ids", [])
    ]
    assert consumers == []


def test_output_catalog_forbids_all_downstream_reuse() -> None:
    catalog = yaml.safe_load(
        (ROOT / "experiments/midogpp/artifact_catalog.yaml").read_text(
            encoding="utf-8"
        )
    )
    artifacts = {row["artifact_id"]: row for row in catalog["artifacts"]}
    output = artifacts[OUTPUT_ID]
    assert output["claim_scope"] == "cvae_source_inner_study_only"
    assert output["may_feed_recipe_selection"] is False
    assert output["may_feed_deployable_selection"] is False
    assert {
        "expert_bank_evidence",
        "generation_evidence",
        "routing_evidence",
        "expert_selection_evidence",
        "nelbo_compatibility_evidence",
        "synthetic_downstream_utility_evidence",
    }.issubset(set(output["forbidden_reuse"]))
