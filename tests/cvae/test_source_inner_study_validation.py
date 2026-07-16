from __future__ import annotations

from types import SimpleNamespace
import copy

import pytest

from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    PriorStudyMetricV2,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.prior_validation import (
    _decision_metrics,
    _validate_metric_checkpoint_references as _validate_prior_checkpoint_references,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.fisher_validation import (
    _validate_fisher_checkpoint_binding,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.validation_common import (
    read_csv,
    selection_evidence_hash,
    study_implementation_lineage,
    validate_common_rows,
    validate_metric_grid,
    validate_workspace_provenance,
)
from midogpp_thesis.cvae.reporting import write_csv_rows, write_json
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)


def _evidence_hash(rows):
    return selection_evidence_hash(
        metric_rows=rows,
        paired_delta_rows=[],
        nested_reference_rows=[],
        nested_tuning_rows=[],
        sampler_rows=[],
        identity_rows=[],
        checkpoint_reuse_rows=[],
        initialization_pairing_rows=[],
        generation_budget_rows=[],
        checkpoint_index={"records": []},
        initialization_index={"records": []},
        feature_frame_index={"records": []},
        generation_budget_manifest={"n_records": 0},
        rng_rows=[],
        protocol_manifest={"protocol_hash": "protocol"},
        study_state_index={"records": []},
    )


def test_selection_evidence_hash_survives_csv_round_trip(tmp_path) -> None:
    rows = [
        {
            "arm": "E",
            "alpha": 0.0,
            "valid": True,
            "optional": None,
            "class_counts": [11, 7],
        }
    ]
    path = tmp_path / "rows.csv"
    write_csv_rows(path, rows)

    assert _evidence_hash(rows) == _evidence_hash(read_csv(path))


def test_study_implementation_lineage_is_mode_specific_and_hash_bound() -> None:
    prior = study_implementation_lineage(
        "learned_conditional_prior_source_inner_study"
    )
    fisher = study_implementation_lineage(
        "task_fisher_shrinkage_source_inner_study"
    )

    assert len(prior["lineage_hash"]) == 16
    assert len(fisher["lineage_hash"]) == 16
    assert prior["lineage_hash"] != fisher["lineage_hash"]
    assert "runtime_versions" in prior


def test_prior_metric_round_trip_preserves_mechanism_ineligibility() -> None:
    common = {
        "outer_target_center": "0",
        "inner_pseudo_target_center": "1",
        "training_seed": "17",
        "arm": "E",
        "bacc": "0.7",
        "preservation_ratio": "0.5",
        "valid": "true",
        "eligible": "false",
        "ineligibility_reason": "learned_prior_mechanism_ineligible",
    }
    rows = [
        {**common, "representation_role": "decode", "generation_seed": "-1"},
        {**common, "representation_role": "prior", "generation_seed": "17"},
        {**common, "representation_role": "posterior", "generation_seed": "17"},
    ]

    assert _decision_metrics(rows) == [
        PriorStudyMetricV2(
            outer_target_center="0",
            inner_pseudo_target_center="1",
            training_seed=17,
            generation_seed=17,
            arm="E",
            preservation_ratio=0.5,
            decode_bacc=0.7,
            posterior_bacc=0.7,
            valid=True,
            eligible=False,
            ineligibility_reason="learned_prior_mechanism_ineligible",
        )
    ]


def test_metric_grid_requires_exact_full_nested_seed_panel() -> None:
    config = SimpleNamespace(
        heldout_centers=MIDOGPP_ELIGIBLE_CENTERS,
        training_seeds=(17, 42, 101),
        generation_seeds=(17, 42, 101),
    )
    rows: list[dict[str, str]] = []
    for outer in MIDOGPP_ELIGIBLE_CENTERS:
        for inner in MIDOGPP_ELIGIBLE_CENTERS:
            if inner == outer:
                continue
            for training_seed in config.training_seeds:
                for arm in ("A", "C-diag", "E"):
                    rows.append(
                        {
                            "outer_target_center": outer,
                            "inner_pseudo_target_center": inner,
                            "training_seed": str(training_seed),
                            "generation_seed": "-1",
                            "arm": arm,
                            "representation_role": "decode",
                            "protocol_hash": "protocol",
                        }
                    )
                    for generation_seed in config.generation_seeds:
                        for role in ("prior", "posterior"):
                            rows.append(
                                {
                                    "outer_target_center": outer,
                                    "inner_pseudo_target_center": inner,
                                    "training_seed": str(training_seed),
                                    "generation_seed": str(generation_seed),
                                    "arm": arm,
                                    "representation_role": role,
                                    "protocol_hash": "protocol",
                                }
                            )

    validate_metric_grid(
        config,
        metric_rows=rows,
        axis_field="arm",
        axis_values=("A", "C-diag", "E"),
        protocol_hash="protocol",
    )
    with pytest.raises(ProtocolError, match="duplicate cell"):
        validate_metric_grid(
            config,
            metric_rows=[*rows, dict(rows[-1])],
            axis_field="arm",
            axis_values=("A", "C-diag", "E"),
            protocol_hash="protocol",
        )


def _provenance_file(path: str, digest: str) -> dict[str, object]:
    return {
        "path": path,
        "exists": True,
        "computed": {"sha256": digest},
        "expected": {"algorithm": "sha256", "digest": digest},
        "verification": "MATCH",
    }


def test_workspace_provenance_requires_present_hash_bound_inputs(tmp_path) -> None:
    manifest_hash = "a" * 64
    cache_hash = "b" * 64
    dataset_files = [
        _provenance_file(path, manifest_hash if path == "manifest.csv" else "c" * 64)
        for path in (
            "dataset_contract.json",
            "manifest.csv",
            "split_manifest.csv",
            "leakage_report.json",
            "path_relocation.json",
        )
    ]
    payload = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": "experiment",
        "stage": "20_cvae_preservation",
        "claim_scope": "cvae_source_inner_study_only",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            {
                "artifact_id": "midogpp_dataset_contract_annotation_patch_v1",
                "exists": True,
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {
                    "status": "EXPECTED_FILE_HASHES_MATCH",
                    "files": dataset_files,
                },
            },
            {
                "artifact_id": "midogpp_virchow2_xyxy_feature_cache_seed42",
                "exists": True,
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {
                    "status": "EXPECTED_FILE_HASHES_MATCH",
                    "files": [_provenance_file("embeddings/train.pt", cache_hash)],
                },
            },
        ],
    }
    path = tmp_path / "provenance" / "input_artifacts.json"
    write_json(path, payload)
    config = SimpleNamespace(
        heldout_centers=MIDOGPP_ELIGIBLE_CENTERS,
        training_seeds=(17, 42, 101),
        generation_seeds=(17, 42, 101),
    )
    protocol = {"manifest_hash": manifest_hash, "feature_cache_hash": cache_hash}

    validate_workspace_provenance(
        tmp_path,
        config,
        experiment_id="experiment",
        protocol=protocol,
    )

    tampered = copy.deepcopy(payload)
    tampered["input_artifacts"][0]["exists"] = False
    write_json(path, tampered)
    with pytest.raises(ProtocolError, match="missing"):
        validate_workspace_provenance(
            tmp_path,
            config,
            experiment_id="experiment",
            protocol=protocol,
        )


def test_common_metric_validation_recomputes_bounded_preservation_ratio() -> None:
    config = SimpleNamespace(
        heldout_centers=MIDOGPP_ELIGIBLE_CENTERS,
        training_seeds=(17, 42, 101),
        generation_seeds=(17, 42, 101),
        minimum_real_bacc=0.55,
    )
    row = {
        "schema_version": "midogpp_source_inner_study_metric_v2",
        "outer_target_center": "0",
        "inner_pseudo_target_center": "1",
        "fit_centers": '["2", "3", "5", "6", "7", "8", "9"]',
        "training_seed": "17",
        "generation_seed": "17",
        "representation_role": "prior",
        "bacc": "0.70",
        "macro_f1": "0.68",
        "real_reference_bacc": "0.75",
        "preservation_ratio": "0.8",
        "valid": "true",
        "status": "ok",
        "claim_scope": "cvae_source_inner_study_only",
        "selection_source": "fully_nested_source_inner",
        "target_eval_labels_used_for_selection": "false",
        "may_feed_model_recipe": "false",
        "may_feed_deployable_selection": "false",
        "routing_performed": "false",
        "composition_performed": "false",
    }
    validate_common_rows(
        config,
        metric_rows=[row],
        identity_rows=[{"status": "PASS"}],
    )

    invalid = {**row, "bacc": "2.0", "macro_f1": "-5", "preservation_ratio": "999"}
    with pytest.raises(ProtocolError, match="bounded score"):
        validate_common_rows(
            config,
            metric_rows=[invalid],
            identity_rows=[{"status": "PASS"}],
        )


def test_fisher_checkpoint_binds_alpha_specific_objective_context() -> None:
    states = {
        ("0", "1"): {
            "raw_fisher_state_hash": "raw-fisher",
            "derived_metrics": {
                "0.05": {"metric_state_hash": "derived-alpha-005"}
            },
        }
    }
    record = {
        "training_key": {
            "outer_target_center": "0",
            "inner_pseudo_target_center": "1",
            "alpha": 0.05,
            "raw_fisher_state_hash": "raw-fisher",
            "objective_context_hash": "derived-alpha-005",
        }
    }
    _validate_fisher_checkpoint_binding({"records": [record]}, states)

    mismatched = copy.deepcopy(record)
    mismatched["training_key"]["objective_context_hash"] = "wrong-derived-state"
    with pytest.raises(ProtocolError, match="derived-metric binding"):
        _validate_fisher_checkpoint_binding({"records": [mismatched]}, states)


def test_prior_metric_checkpoint_binding_rejects_cross_fold_relabeling() -> None:
    training_key = {
        "outer_target_center": "0",
        "inner_pseudo_target_center": "1",
        "training_seed": 17,
        "fit_centers": ["2", "3", "5", "6", "7", "8", "9"],
        "fit_row_hash": "fit-row",
        "frame_hash": "frame",
        "protocol_hash": "protocol",
        "model_family": "class_conditioned_cvae_v1",
        "prior_family": "standard_normal",
        "objective_id": "stochastic_isotropic_v1",
        "alpha": 0.0,
    }
    index = {
        "records": [
            {
                "training_key_hash": "key",
                "checkpoint_hash": "checkpoint",
                "training_key": training_key,
            }
        ]
    }
    row = {
        "training_key_hash": "key",
        "checkpoint_hash": "checkpoint",
        "outer_target_center": "0",
        "inner_pseudo_target_center": "1",
        "training_seed": "17",
        "fit_centers": '["2", "3", "5", "6", "7", "8", "9"]',
        "fit_row_hash": "fit-row",
        "frame_hash": "frame",
        "protocol_hash": "protocol",
        "model_family": "class_conditioned_cvae_v1",
        "prior_family": "standard_normal",
        "objective_id": "stochastic_isotropic_v1",
        "alpha": "0.0",
        "arm": "A",
    }
    _validate_prior_checkpoint_references([row], index)

    relabeled = {**row, "outer_target_center": "9", "inner_pseudo_target_center": "8"}
    with pytest.raises(ProtocolError, match="unpersisted checkpoint"):
        _validate_prior_checkpoint_references([relabeled], index)

    unbound = {**row, "training_key_hash": "none", "checkpoint_hash": "none"}
    with pytest.raises(ProtocolError, match="lacks its runtime checkpoint"):
        _validate_prior_checkpoint_references([unbound], index)
