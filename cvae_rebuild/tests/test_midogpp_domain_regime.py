import csv
import json
from pathlib import Path

import pytest
import yaml

from cvae_rebuild.cli import _load_config_for_validation
from cvae_rebuild.component_union_tailrisk_anchored_mass_bagged import (
    MIDOGPP_POSITIVE_UNION_TAILRISK_NAME,
    PRIMARY_POSITIVE_UNION_METHOD,
    POSITIVE_UNION_RULE_BETA050,
    _positive_union_rule_selection_manifest_rows,
    parse_source_inner_positive_union_config,
)
from cvae_rebuild.domain_regime import (
    MIDOGPP_DOMAIN_REGIME,
    load_midogpp_contract_info,
    validate_runtime_domain_coverage,
)
from cvae_rebuild.paired_dense_all4_reliability_confirmation import (
    DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME,
    PRIMARY_DENSE_ALL_SOURCES_METHOD,
    _alias_rows_for_output,
    _artifact_prefix,
    parse_paired_dense_all4_reliability_config,
)
from cvae_rebuild.protocol import ProtocolError


ELIGIBLE_MIDOGPP_IDS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def test_midogpp_contract_info_derives_eligible_ids_and_fingerprints(tmp_path: Path) -> None:
    artifact, _cache_report = _write_midogpp_contract_fixture(tmp_path)

    info = load_midogpp_contract_info(artifact)

    assert info.eligible_domain_ids == ELIGIBLE_MIDOGPP_IDS
    assert info.ineligible_domain_ids == ("4",)
    assert set(info.fingerprints) == {
        "dataset_contract.json",
        "domain_mapping.json",
        "domain_feasibility.csv",
        "manifest.csv",
    }
    assert all(len(value) == 64 for value in info.fingerprints.values())


def test_midogpp_dense_all_sources_config_validates_against_contract(tmp_path: Path) -> None:
    artifact, cache_report = _write_midogpp_contract_fixture(tmp_path)
    payload = _dense_all_sources_payload(tmp_path, artifact, cache_report)

    cfg = parse_paired_dense_all4_reliability_config(payload, base_dir=tmp_path)

    assert cfg.name == DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME
    assert cfg.domain_regime == MIDOGPP_DOMAIN_REGIME
    assert cfg.heldout_centers == ELIGIBLE_MIDOGPP_IDS
    assert cfg.primary_method == PRIMARY_DENSE_ALL_SOURCES_METHOD
    assert _artifact_prefix(cfg) == "dense_late_all_sources"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["run_matrix"]["heldout_centers"].append("4"), "eligible domains"),
        (lambda payload: payload["run_matrix"]["heldout_centers"].append("9"), "duplicate"),
        (lambda payload: payload["run_matrix"].__setitem__("strict_full_run_matrix", True), "strict_full_run_matrix"),
        (lambda payload: payload["run_matrix"].__setitem__("strict_available_seed_domain_coverage", False), "strict_available"),
    ],
)
def test_midogpp_dense_config_rejects_bad_domain_protocol(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    artifact, cache_report = _write_midogpp_contract_fixture(tmp_path)
    payload = _dense_all_sources_payload(tmp_path, artifact, cache_report)
    mutation(payload)

    with pytest.raises(ProtocolError, match=message):
        parse_paired_dense_all4_reliability_config(payload, base_dir=tmp_path)


def test_midogpp_dense_output_aliases_remove_all4_method_ids(tmp_path: Path) -> None:
    artifact, cache_report = _write_midogpp_contract_fixture(tmp_path)
    cfg = parse_paired_dense_all4_reliability_config(
        _dense_all_sources_payload(tmp_path, artifact, cache_report),
        base_dir=tmp_path,
    )

    rows = _alias_rows_for_output(
        cfg,
        [
            {
                "prior_method": "paired_equal_all4_geom",
                "pairing_group": "paired_equal_budget_all4_v1",
                "delta_vs_paired_equal_all4_center_equal_bacc": 0.1,
                "primary_verdict": "EQUAL_DENSE_ALL4_ROBUST_BASELINE",
            }
        ],
    )

    encoded = json.dumps(rows, sort_keys=True)
    assert "all4" not in encoded
    assert "dense_late_equal_all_sources_geom" in encoded


def test_validate_config_dispatches_midogpp_dense_all_sources(tmp_path: Path) -> None:
    artifact, cache_report = _write_midogpp_contract_fixture(tmp_path)
    path = tmp_path / "dense.yaml"
    path.write_text(yaml.safe_dump(_dense_all_sources_payload(tmp_path, artifact, cache_report)), encoding="utf-8")

    cfg = _load_config_for_validation(path)

    assert cfg.name == DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME


def test_midogpp_runtime_coverage_rejects_partial_seed_domain_cache(tmp_path: Path) -> None:
    artifact, _cache_report = _write_midogpp_contract_fixture(tmp_path)
    info = load_midogpp_contract_info(artifact)
    train_meta = [_cache_meta(center, 0) for center in info.eligible_domain_ids if center != "9"]
    test_meta = [_cache_meta(center, 0) for center in info.eligible_domain_ids]

    with pytest.raises(ProtocolError, match="missing_train"):
        validate_runtime_domain_coverage(
            domain_regime=MIDOGPP_DOMAIN_REGIME,
            eligible_domain_ids=info.eligible_domain_ids,
            experiment_seed=42,
            train_metadata=train_meta,
            test_metadata=test_meta,
        )


def test_midogpp_runtime_coverage_allows_unused_ineligible_cached_domain(tmp_path: Path) -> None:
    artifact, _cache_report = _write_midogpp_contract_fixture(tmp_path)
    info = load_midogpp_contract_info(artifact)
    train_meta = [_cache_meta(center, 0) for center in (*info.eligible_domain_ids, "4")]
    test_meta = [_cache_meta(center, 1) for center in (*info.eligible_domain_ids, "4")]

    rows = validate_runtime_domain_coverage(
        domain_regime=MIDOGPP_DOMAIN_REGIME,
        eligible_domain_ids=info.eligible_domain_ids,
        experiment_seed=42,
        train_metadata=train_meta,
        test_metadata=test_meta,
    )

    assert len(rows) == len(ELIGIBLE_MIDOGPP_IDS)
    assert {row["heldout_domain_id"] for row in rows} == set(ELIGIBLE_MIDOGPP_IDS)
    for row in rows:
        sources = json.loads(str(row["source_domain_ids"]))
        assert "4" not in sources
        assert row["heldout_domain_id"] not in sources
        assert row["expected_source_count"] == 8
        assert row["actual_source_count"] == 8
        assert row["domain_4_excluded"] is True
        assert json.loads(str(row["cache_extra_domain_ids"])) == ["4"]


def test_midogpp_positive_union_config_and_rule_manifest(tmp_path: Path) -> None:
    artifact, cache_report = _write_midogpp_contract_fixture(tmp_path)
    cfg = parse_source_inner_positive_union_config(
        _positive_union_payload(tmp_path, artifact, cache_report),
        base_dir=tmp_path,
    )
    source_pool_rows = validate_runtime_domain_coverage(
        domain_regime=MIDOGPP_DOMAIN_REGIME,
        eligible_domain_ids=ELIGIBLE_MIDOGPP_IDS,
        experiment_seed=42,
        train_metadata=[_cache_meta(center, 0) for center in ELIGIBLE_MIDOGPP_IDS],
        test_metadata=[_cache_meta(center, 1) for center in ELIGIBLE_MIDOGPP_IDS],
    )
    manifest_rows = _positive_union_rule_selection_manifest_rows(
        cfg,
        [
            {
                "experiment_seed": 42,
                "heldout_center": "5",
                "selected_rule": POSITIVE_UNION_RULE_BETA050,
                "selected_beta": 0.5,
                "source_inner_positive_count": 32,
                "source_inner_negative_count": 64,
            }
        ],
        source_pool_rows,
    )

    row = manifest_rows[0]
    assert cfg.name == MIDOGPP_POSITIVE_UNION_TAILRISK_NAME
    assert cfg.primary_method == PRIMARY_POSITIVE_UNION_METHOD
    assert row["selection_signal"] == "source_inner_only"
    assert row["target_labels_used_for_selection"] is False
    assert "selected_sources_by_class" not in row
    assert "5" not in json.loads(str(row["candidate_sources"]))
    assert len(json.loads(str(row["candidate_sources"]))) == 8


def _write_midogpp_contract_fixture(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "datasets/midogpp/artifacts/midogpp_annotation_patch_v1"
    artifact.mkdir(parents=True)
    axis = "tumor_type|lab_or_origin|scanner_model"
    manifest_rows = []
    row_id = 0
    for domain_id in (*ELIGIBLE_MIDOGPP_IDS, "4"):
        for split in ("train", "val", "test"):
            for label in ("0", "1"):
                manifest_rows.append(
                    {
                        "sample_id": f"s{row_id}",
                        "case_id": f"case_{domain_id}_{row_id}",
                        "image_path": f"patches/s{row_id}.jpg",
                        "label": label,
                        "split": split,
                        "domain_axis": axis,
                        "domain_name": f"domain_{domain_id}",
                        "domain_id": domain_id,
                        "center": domain_id,
                        "magnification": domain_id,
                    }
                )
                row_id += 1
    mapping = {
        "schema_version": "midogpp_domain_mapping_v1",
        "domain_axis": axis,
        "domain_name_to_id": {f"domain_{idx}": str(idx) for idx in range(10)},
        "domains": [
            {"domain_id": str(idx), "domain_name": f"domain_{idx}", "n_cases": 24, "n_rows": 6}
            for idx in range(10)
        ],
    }
    contract = {
        "schema_version": "midogpp_annotation_patch_dataset_contract_v1",
        "artifact_name": "midogpp_annotation_patch_v1",
        "status": "pass",
        "domain_policy": {"selected_domain_axis": axis},
    }
    feasibility_rows = [
        _feasibility_row(axis, domain_id, eligible=(domain_id != "4"))
        for domain_id in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    ]
    (artifact / "dataset_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (artifact / "domain_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    _write_csv(artifact / "manifest.csv", manifest_rows)
    _write_csv(artifact / "domain_feasibility.csv", feasibility_rows)

    cache_report = tmp_path / "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2/seed42/reports/cache_builder_report.json"
    cache_report.parent.mkdir(parents=True, exist_ok=True)
    split_counts = {"train": 20, "val": 20, "test": 20}
    cache_report.write_text(json.dumps({"split_counts": split_counts}), encoding="utf-8")
    return artifact, cache_report


def _dense_all_sources_payload(tmp_path: Path, artifact: Path, cache_report: Path) -> dict:
    return {
        "experiment": {
            "name": DENSE_LATE_ALL_SOURCES_MIDOGPP_NAME,
            "artifact_root": str(tmp_path / "cvae_rebuild/artifacts/midogpp/virchow2_cvae_dense_late_all_sources_midogpp_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2"),
            "repair_artifact_root": str(tmp_path / "repair"),
            "d1_2_artifact_root": None,
            "d1_4_artifact_root": None,
            "dataset_contract_artifact_root": str(artifact),
            "cache_report_path": str(cache_report),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "domain_regime": MIDOGPP_DOMAIN_REGIME,
            "strict_full_run_matrix": False,
            "strict_available_seed_domain_coverage": True,
            "experiment_seeds": [42],
            "heldout_centers": list(ELIGIBLE_MIDOGPP_IDS),
            "replicate_seeds": [17],
        },
        "generation": {"synthetic_per_class_total": 128, "min_per_source_per_class": 8},
        "dense_late_all_sources_reliability": {
            "primary_method": PRIMARY_DENSE_ALL_SOURCES_METHOD,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "heldout_excluded_source_local_reliability_dense_all_sources",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 5,
            "gmm_max_iter": 500,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "shrinkage_values": [0.25, 0.5],
            "primary_pooling": "weighted_geometric",
        },
        "classifier": _classifier_payload(),
    }


def _positive_union_payload(tmp_path: Path, artifact: Path, cache_report: Path) -> dict:
    return {
        "experiment": {
            "name": MIDOGPP_POSITIVE_UNION_TAILRISK_NAME,
            "artifact_root": str(tmp_path / "cvae_rebuild/artifacts/midogpp/virchow2_cvae_source_inner_class_conditional_positive_union_midogpp_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2"),
            "repair_artifact_root": str(tmp_path / "repair"),
            "paired_dense_artifact_root": None,
            "mass_bagged_artifact_root": None,
            "shrink050_artifact_root": None,
            "source_union_gmm_artifact_root": None,
            "balanced_gmm_artifact_root": None,
            "prior_tailrisk_artifact_root": None,
            "support_calibrated_artifact_root": None,
            "dataset_contract_artifact_root": str(artifact),
            "cache_report_path": str(cache_report),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "domain_regime": MIDOGPP_DOMAIN_REGIME,
            "strict_full_run_matrix": False,
            "strict_available_seed_domain_coverage": True,
            "experiment_seeds": [42],
            "heldout_centers": list(ELIGIBLE_MIDOGPP_IDS),
            "replicate_seeds": [17, 23, 31],
            "fresh_replicate_seeds": [101, 103, 107, 109, 113, 127],
        },
        "generation": {"synthetic_per_class_total": 128, "min_per_source_per_class": 8},
        "source_inner_class_conditional_positive_union": {
            "primary_method": PRIMARY_POSITIVE_UNION_METHOD,
            "primary_shrink_lambda": 0.5,
            "random_mass_bag_size": 11,
            "random_mass_bag_alpha": 4.0,
            "blend_alpha": 0.5,
            "matched_shuffled_reliability_null_permutations": 0,
            "panel_seed_groups": {
                "canonical": [17, 23, 31],
                "fresh_a": [101, 103, 107],
                "fresh_b": [109, 113, 127],
            },
            "candidate_pooling_rules": ["arithmetic_mean", "positive_union_beta025", "positive_union_beta050", "positive_union_beta100"],
            "positive_label": 1,
            "prediction_threshold": 0.5,
            "min_source_inner_positive_count": 5,
            "positive_union_eps": 1.0e-8,
            "source_inner_bacc_noninferiority_margin": 0.010,
            "source_inner_class0_recall_margin": 0.015,
            "source_inner_predicted_positive_rate_delta": 0.050,
            "beta100_class0_recall_margin": 0.005,
            "beta100_precision_margin": 0.010,
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 8,
            "source_weighting": "source_inner_class_conditional_positive_union",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "source_inner_selected_class_conditional_positive_union",
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "anchor_repro_tolerance": 1.0e-4,
            "primary_noninferiority_margin": 0.005,
            "weak_pass_noninferiority_margin": 0.010,
            "tailrisk_transfer_threshold": -0.010,
        },
        "classifier": _classifier_payload(),
    }


def _classifier_payload() -> dict:
    return {
        "type": "sklearn_logistic_regression",
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": "balanced",
        "classifier_seed": None,
    }


def _feasibility_row(axis: str, domain_id: str, *, eligible: bool) -> dict[str, str]:
    return {
        "domain_axis": axis,
        "domain_name": f"domain_{domain_id}",
        "domain_id_for_axis": domain_id,
        "total_rows": "6",
        "total_cases": "24" if eligible else "2",
        "train_cases": "12" if eligible else "1",
        "eval_cases": "12" if eligible else "1",
        "train_positives": "6",
        "train_negatives": "6",
        "eval_positives": "6",
        "eval_negatives": "6",
        "eligible": str(bool(eligible)),
        "ineligible_reasons": "" if eligible else "total_cases<20",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cache_meta(center: str, label: int) -> dict[str, object]:
    return {"center": center, "label": int(label), "sample_id": f"s_{center}_{label}"}
