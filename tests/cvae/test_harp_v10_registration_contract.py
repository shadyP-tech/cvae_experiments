from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.activation_paths import (
    reject_predecessor_path,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.activation_workspace import (
    validate_rendered_workspace,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.amendment_publisher import (
    _source_label_member_sha256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.identity import (
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.input_surfaces import (
    DEVELOPMENT_ROLE,
    SOURCE_LABEL_INDEX_SCHEMA,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.preparation_contracts import (
    CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
    EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER,
    EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v10.workspace_paths import (
    SOURCE_CROSSFIT_REQUIRED_OUTPUT_MEMBERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v10.yaml"
)
REGISTRY = REPOSITORY_ROOT / "experiments/midogpp/registry.yaml"
CATALOG = REPOSITORY_ROOT / "experiments/midogpp/artifact_catalog.yaml"


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _row(rows: object, identity: str, *, key: str) -> dict[str, object]:
    assert isinstance(rows, list)
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get(key) == identity
    ]
    assert len(matches) == 1
    return matches[0]


def test_v10_config_freezes_train_crossfit_full_test_and_workstation_geometry() -> None:
    config = load_config(CONFIG)
    protocol = config.protocol
    runtime = config.runtime

    assert protocol["source_development_row_count"] == 9_648
    assert protocol["source_development_case_count"] == 216
    assert protocol["target_evaluation_row_count"] == 9_928
    assert protocol["target_evaluation_case_count"] == 218
    assert protocol["consumed_test_development_cases_used"] is False
    assert protocol["source_fold_conditioned_physical_surface"] == "H_q_r"
    assert protocol["source_prediction_candidate_pool"] == (
        "C_minus_outer_H_and_heldout_q"
    )
    assert protocol["source_calibration_candidate_pool"] == (
        "C_minus_outer_H_heldout_q_and_current_query_r"
    )
    assert protocol["source_fold_prediction_context_count"] == 72
    assert protocol["source_fold_calibration_context_count"] == 504
    assert protocol["source_fold_classifier_task_count"] == 5_184
    assert protocol["source_fold_seed_cell_count"] == 42_120
    assert protocol["source_label_capability_center_sharded"] is True
    assert protocol["source_label_capability_shard_count"] == 9
    assert protocol["source_label_fold_workers_spawn_isolated"] is True
    assert (
        protocol[
            "heldout_q_label_shard_unauthorized_and_not_opened_by_typed_loader_in_own_H_q_worker"
        ]
        is True
    )
    assert protocol["cross_fold_model_or_prediction_state_shared"] is False
    assert protocol["global_source_label_open_order_claimed"] is False
    assert protocol[
        "pseudo_target_q_predictions_sealed_before_q_outcomes_joined_to_same_fold"
    ] is True
    assert protocol["six_source_calibration_base_total_per_class"] == 1_008
    assert protocol["six_source_calibration_final_total_per_class"] == 1_134
    assert protocol[
        "generic_quarter_max_weight_claimed_for_six_source_surface"
    ] is False
    assert protocol[
        "generic_min_six_effective_sources_claimed_for_six_source_surface"
    ] is False
    assert protocol["target_case_label_free_inference_features"] == (
        "own_case_embeddings_only"
    )
    assert protocol["separate_target_support_partition_used"] is False
    assert protocol["target_case_features_may_not_fit_or_calibrate_router"] is True

    assert runtime["gpu_devices"] == ["cuda:0", "cuda:1"]
    assert runtime["persistent_gpu_workers"] == 2
    assert runtime["classifier_workers"] == 4
    assert runtime["classifier_blas_threads_per_worker"] == 3
    assert runtime["science_workers"] == 4
    assert runtime["science_blas_threads_per_worker"] == 1


def test_v10_path_fence_rejects_every_predecessor_without_matching_v10() -> None:
    reject_predecessor_path(
        "datasets/midogpp/derived/features/virchow2/"
        "harp_source_train_full_test_cache_v10",
        label="current cache",
    )
    reject_predecessor_path(
        "src/midogpp_thesis/cvae/runtime/harp_v10_execution",
        label="current runtime",
    )
    for version in range(1, 10):
        with pytest.raises(ProtocolError, match="predecessor path"):
            reject_predecessor_path(
                f"artifacts/midogpp/fixed_bank_harp_router/v{version}",
                label="predecessor output",
            )
        with pytest.raises(ProtocolError, match="predecessor path"):
            reject_predecessor_path(
                f"src/midogpp_thesis/cvae/runtime/harp_v{version}_execution",
                label="predecessor runtime",
            )


def test_v10_catalog_and_registry_bind_center_shards_and_crossfit_outputs() -> None:
    registry = _yaml(REGISTRY)
    catalog = _yaml(CATALOG)
    experiment = _row(
        registry["experiments"], EXPERIMENT_ID, key="experiment_id"
    )
    assert experiment["status"] == "planned"
    assert experiment["input_artifact_ids"] == list(INPUT_ARTIFACT_IDS)

    development = _row(
        catalog["artifacts"], INPUT_ARTIFACT_IDS[3], key="artifact_id"
    )
    assert development["required_files"] == [
        "index.json",
        "by_center/center_0.csv",
        "by_center/center_1.csv",
        "by_center/center_2.csv",
        "by_center/center_3.csv",
        "by_center/center_5.csv",
        "by_center/center_6.csv",
        "by_center/center_7.csv",
        "by_center/center_8.csv",
        "by_center/center_9.csv",
    ]
    semantics = development["semantic_identities"]
    assert semantics["center_sharded_label_capability"] == "true"
    assert semantics["fold_fit_label_scope"] == "C_MINUS_H_MINUS_Q"
    assert semantics["fold_workers_spawn_isolated"] == "true"
    assert (
        semantics[
            "heldout_q_label_shard_unauthorized_and_not_opened_by_typed_loader_in_own_H_Q_worker"
        ]
        == "true"
    )
    assert semantics["cross_fold_model_or_prediction_state_shared"] == "false"
    assert semantics["global_source_label_open_order_claimed"] == "false"

    output = _row(catalog["artifacts"], OUTPUT_ARTIFACT_ID, key="artifact_id")
    assert set(SOURCE_CROSSFIT_REQUIRED_OUTPUT_MEMBERS).issubset(
        output["required_files"]
    )
    output_semantics = output["semantic_identities"]
    assert output_semantics["separate_target_support_partition_used"] == "false"
    assert (
        output_semantics[
            "heldout_q_label_shard_unauthorized_and_not_opened_by_typed_loader_in_own_fold_worker"
        ]
        == "true"
    )
    assert output_semantics[
        "source_q_predictions_sealed_before_same_fold_outcome_join"
    ] == "true"
    assert output_semantics["global_source_label_open_order_claimed"] == "false"
    assert output_semantics[
        "target_case_features_may_fit_or_calibrate_router"
    ] == "false"


def test_activation_projection_rejects_missing_crossfit_output_member() -> None:
    registry = deepcopy(_yaml(REGISTRY))
    catalog = deepcopy(_yaml(CATALOG))
    experiment = _row(
        registry["experiments"], EXPERIMENT_ID, key="experiment_id"
    )
    experiment["status"] = "diagnostic"
    for artifact_id in (*INPUT_ARTIFACT_IDS[2:], OUTPUT_ARTIFACT_ID):
        artifact = _row(catalog["artifacts"], artifact_id, key="artifact_id")
        semantics = artifact["semantic_identities"]
        semantics["execution_authorized"] = "true"
        semantics["consumed_test_reuse_authorized"] = "true"

    validate_rendered_workspace(registry, catalog)

    output = _row(catalog["artifacts"], OUTPUT_ARTIFACT_ID, key="artifact_id")
    output["required_files"].remove(SOURCE_CROSSFIT_REQUIRED_OUTPUT_MEMBERS[0])
    with pytest.raises(ProtocolError, match="cross-fit output inventory"):
        validate_rendered_workspace(registry, catalog)


def test_amendment_authenticates_every_center_sharded_label_member(tmp_path: Path) -> None:
    cache_hash = "a" * 64
    pre_manifest_hash = "b" * 64
    shards: list[dict[str, object]] = []
    for center in EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER:
        relative = f"by_center/center_{center}.csv"
        member = tmp_path / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(f"fixture-{center}\n", encoding="utf-8")
        shards.append(
            {
                "center": center,
                "relative_path": relative,
                "sha256": sha256_file(member),
                "row_count": EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER[center],
                "case_count": EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER[center],
                "ordered_key_hash": canonical_hash(
                    {"fixture_center": center}
                ),
            }
        )
    base: dict[str, object] = {
        "schema_version": SOURCE_LABEL_INDEX_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "artifact_role": "center_sharded_source_label_capability",
        "split_role": DEVELOPMENT_ROLE,
        "cache_index_hash": cache_hash,
        "pre_manifest_cache_content_sha256": pre_manifest_hash,
        "source_train_tensor_sha256": CANONICAL_SOURCE_TRAIN_TENSOR_SHA256,
        "shards": shards,
        "row_count": sum(EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER.values()),
        "case_count": sum(EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER.values()),
        "labels_stored_in_index": False,
        "capability_state": (
            "CENTER_SCOPED_OPEN_AFTER_FOLD_PHYSICAL_SURFACE_SEAL"
        ),
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
        "may_feed_stage60_or_stage70": False,
        "may_feed_another_experiment": False,
    }
    index = {**base, "index_hash": canonical_hash(base)}
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    inventory = _source_label_member_sha256(
        index_path,
        cache_index_hash=cache_hash,
        pre_manifest_cache_content_sha256=pre_manifest_hash,
    )
    assert set(inventory) == {
        "index.json",
        *(f"by_center/center_{center}.csv" for center in EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER),
    }

    (tmp_path / "by_center/center_0.csv").write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="shard bytes drifted"):
        _source_label_member_sha256(
            index_path,
            cache_index_hash=cache_hash,
            pre_manifest_cache_content_sha256=pre_manifest_hash,
        )
