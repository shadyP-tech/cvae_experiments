import csv
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

import numpy as np

from cvae_rebuild.config import parse_config
from cvae_rebuild.covariance_prior import (
    PRIMARY_COVARIANCE_METHOD,
    _stabilized_covariance_psd,
    parse_covariance_prior_config,
    run_covariance_prior_confirmation,
)
from cvae_rebuild.covariance_shrinkage import (
    PRIMARY_SHRINKAGE_METHOD,
    parse_covariance_shrinkage_config,
    run_covariance_shrinkage_stability,
)
from cvae_rebuild.covariance_viability import (
    parse_covariance_viability_config,
    run_covariance_prior_viability_audit,
)
from cvae_rebuild.decentralized_k16_gmm_prior import (
    PRIMARY_DECENTRALIZED_METHOD,
    parse_decentralized_k16_gmm_prior_config,
    run_decentralized_k16_gmm_prior,
)
from cvae_rebuild.decentralized_adaptive_gmm_prior import (
    PRIMARY_ADAPTIVE_METHOD,
    parse_decentralized_adaptive_gmm_prior_config,
    run_decentralized_adaptive_gmm_prior,
)
from cvae_rebuild.decentralized_component_union_prior import (
    PRIMARY_COMPONENT_UNION_METHOD,
    parse_decentralized_component_union_prior_config,
    run_decentralized_component_union_prior,
)
from cvae_rebuild.decentralized_reliability_weighted_gmm_prior import (
    PRIMARY_RELIABILITY_METHOD,
    SourceReliability,
    parse_decentralized_reliability_weighted_gmm_prior_config,
    run_decentralized_reliability_weighted_gmm_prior,
)
from cvae_rebuild.decentralized_reliability_top3_gmm_prior import (
    PRIMARY_RELIABILITY_TOP3_METHOD,
    parse_decentralized_reliability_top3_gmm_prior_config,
    run_decentralized_reliability_top3_gmm_prior,
)
from cvae_rebuild.decentralized_source_inner_transfer_top3_gmm_prior import (
    PRIMARY_SOURCE_INNER_TRANSFER_METHOD,
    parse_decentralized_source_inner_transfer_top3_gmm_prior_config,
    run_decentralized_source_inner_transfer_top3_gmm_prior,
)
from cvae_rebuild.decentralized_support_nelbo_reliability_gmm_prior import (
    PRIMARY_SUPPORT_RELIABILITY_METHOD,
    parse_decentralized_support_nelbo_reliability_gmm_prior_config,
    run_decentralized_support_nelbo_reliability_gmm_prior,
)
from cvae_rebuild.decentralized_support8_top3_tau05_gmm_prior import (
    PRIMARY_SUPPORT8_TOP3_TAU05_METHOD,
    parse_decentralized_support8_top3_tau05_gmm_prior_config,
    run_decentralized_support8_top3_tau05_gmm_prior,
)
from cvae_rebuild.paired_dense_all4_reliability_confirmation import (
    ROW_BUDGET_ONLY,
    ROW_EQUAL_ALL4,
    ROW_INVERSE,
    ROW_POOL_ONLY,
    ROW_RELIABILITY_ALL4_WEIGHTED,
    ROW_SHRINK025,
    ROW_SHRINK050,
    ROW_SHUFFLED,
    _heldout_excluded_reliability_transform,
    _inverse_rank_reversal_weights,
    _variant_plans,
    parse_paired_dense_all4_reliability_config,
    run_paired_dense_all4_reliability_confirmation,
)
from cvae_rebuild.generation import generate_reference_posterior
from cvae_rebuild.models import ClassConditionedCVAE
from cvae_rebuild.pipeline import run_real_cache_backed
from cvae_rebuild.preservation import (
    ROW_DECODE_MU,
    ROW_POSTERIOR,
    ROW_PRIOR,
    ROW_REAL_BUDGET,
    ROW_REAL_FULL,
    parse_preservation_config,
    run_preservation_diagnosis,
)
from cvae_rebuild.preservation_repair import (
    PRIMARY_VARIANT,
    _beta_for_epoch,
    _decision,
    _decision_rows,
    parse_repair_config,
    run_preservation_repair,
)
from cvae_rebuild.preservation_sampling import (
    ROW_DECODE_MU as SAMPLING_ROW_DECODE_MU,
    ROW_POSTERIOR as SAMPLING_ROW_POSTERIOR,
    ROW_PRIOR as SAMPLING_ROW_PRIOR,
    parse_sampling_config,
    run_preservation_sampling,
)
from cvae_rebuild.prior_calibration import (
    PRIMARY_PRIOR_METHOD,
    parse_prior_calibration_config,
    run_prior_calibration,
)
from cvae_rebuild.source_union_gmm_prior import (
    PRIMARY_GMM_METHOD,
    parse_source_union_gmm_prior_config,
    run_source_union_gmm_prior,
)
from cvae_rebuild.source_union_balanced_gmm_prior import (
    PRIMARY_BALANCED_METHOD,
    parse_source_union_balanced_gmm_prior_config,
    run_source_union_balanced_gmm_prior,
)
from cvae_rebuild.source_union_k24_gmm_prior import (
    PRIMARY_K24_GMM_METHOD,
    parse_source_union_k24_gmm_prior_config,
    run_source_union_k24_gmm_prior,
)
from cvae_rebuild.splits import stratified_source_train_val_split


def test_real_run_tiny_npz_cache_writes_protocol_artifacts(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_real_cache_backed(cfg)

    support_rows = list(csv.DictReader(open(root / "tables" / "support_nelbo_routing_scores.csv", newline="")))
    downstream_rows = list(csv.DictReader(open(root / "tables" / "all_expert_downstream_matrix.csv", newline="")))
    alignment_rows = list(csv.DictReader(open(root / "tables" / "routing_to_downstream_alignment.csv", newline="")))
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert support_rows
    assert downstream_rows
    assert alignment_rows
    assert leakage["status"] == "PASS"
    assert any(row["method"] == "support_nelbo_top2_geom" and row["status"] == "ok" for row in downstream_rows)
    assert any(row["method"] == "random_top2_geom" and row["status"] == "ok" for row in downstream_rows)
    assert any(
        row["method"] == "downstream_oracle_diagnostic_only"
        and row["selection_source"] == "diagnostic_only"
        for row in downstream_rows
    )
    assert {"oracle_gap_top1", "oracle_gap_top2", "mean_oracle_rank_of_selected_experts"}.issubset(
        alignment_rows[0]
    )


def test_real_run_records_mono_class_target_eval_as_ineligible(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42, mono_test_centers={"2"})

    root = run_real_cache_backed(cfg)

    support_rows = list(csv.DictReader(open(root / "tables" / "support_nelbo_routing_scores.csv", newline="")))
    downstream_rows = list(csv.DictReader(open(root / "tables" / "all_expert_downstream_matrix.csv", newline="")))
    alignment_rows = list(csv.DictReader(open(root / "tables" / "routing_to_downstream_alignment.csv", newline="")))
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    invalid_support = [row for row in support_rows if row["heldout_center"] == "2"]
    invalid_downstream = [row for row in downstream_rows if row["heldout_center"] == "2"]
    invalid_alignment = [row for row in alignment_rows if row["heldout_center"] == "2"]

    assert leakage["status"] == "PASS"
    assert invalid_support
    assert invalid_downstream
    assert invalid_alignment
    assert {row["eval_status"] for row in invalid_support} == {"ineligible"}
    assert {row["error_message"] for row in invalid_support} == {"mono_class_target_eval"}
    assert any(
        row["method"] == "support_nelbo_top2_geom"
        and row["status"] == "ineligible"
        and row["error_message"] == "mono_class_target_eval"
        for row in invalid_downstream
    )
    assert {row["status"] for row in invalid_alignment} == {"ineligible"}
    assert any(row["method"] == "support_nelbo_top2_geom" and row["status"] == "ok" for row in downstream_rows)


def test_source_train_val_split_uses_only_requested_source_center() -> None:
    metadata = []
    for center in ("0", "1"):
        for label in (0, 1):
            for idx in range(10):
                metadata.append(
                    {
                        "sample_id": f"c{center}_y{label}_{idx}",
                        "center": center,
                        "label": label,
                    }
                )
    split = stratified_source_train_val_split(metadata, center="1", experiment_seed=42)
    selected_ids = set(split.train_sample_ids).union(split.val_sample_ids)
    assert selected_ids
    assert all(sample_id.startswith("c1_") for sample_id in selected_ids)
    assert set(split.train_sample_ids).isdisjoint(split.val_sample_ids)


def test_reference_posterior_generation_is_torch_seed_deterministic() -> None:
    model = ClassConditionedCVAE(input_dim=3, hidden_dim=8, latent_dim=2, n_classes=2)
    refs = {
        0: np.random.default_rng(0).normal(size=(8, 3)),
        1: np.random.default_rng(1).normal(size=(8, 3)),
    }
    first = generate_reference_posterior(
        model=model,
        expert_id="1",
        source_embeddings_by_class=refs,
        budget_per_class=4,
        generation_seed=17,
    )
    second = generate_reference_posterior(
        model=model,
        expert_id="1",
        source_embeddings_by_class=refs,
        budget_per_class=4,
        generation_seed=17,
    )
    assert np.allclose(first.embeddings, second.embeddings)


def test_preservation_diagnosis_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    cfg = _tiny_preservation_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_preservation_diagnosis(cfg)

    downstream = list(csv.DictReader(open(root / "tables" / "preservation_downstream_matrix.csv", newline="")))
    gaps = list(csv.DictReader(open(root / "tables" / "preservation_gap_summary.csv", newline="")))
    sampling = list(csv.DictReader(open(root / "tables" / "reference_sampling_diagnostics.csv", newline="")))
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage["status"] == "PASS"
    assert len(downstream) == 100
    assert len([row for row in downstream if row["row_role"] == ROW_REAL_FULL]) == 20
    assert all(row["replicate_seed"] == "NA" for row in downstream if row["row_role"] == ROW_REAL_FULL)
    assert any(row["row_role"] == ROW_PRIOR and row["reference_sample_seed"] == "NA" for row in downstream)
    assert all(row["classifier_class_weight"] == "balanced" for row in downstream)
    assert gaps
    assert sampling


def test_preservation_diagnosis_marks_mono_class_target_eval_ineligible(tmp_path: Path) -> None:
    cfg = _tiny_preservation_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42, mono_test_centers={"2"})

    root = run_preservation_diagnosis(cfg)
    downstream = list(csv.DictReader(open(root / "tables" / "preservation_downstream_matrix.csv", newline="")))

    invalid = [row for row in downstream if row["heldout_center"] == "2"]
    valid = [row for row in downstream if row["heldout_center"] != "2"]
    assert len(invalid) == 20
    assert {row["status"] for row in invalid} == {"ineligible"}
    assert {row["error_message"] for row in invalid} == {"mono_class_target_eval"}
    assert any(row["status"] == "ok" for row in valid)


def test_preservation_gaps_use_paired_replicate_key_and_reference_hash(tmp_path: Path) -> None:
    cfg = _tiny_preservation_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_preservation_diagnosis(cfg)
    downstream = list(csv.DictReader(open(root / "tables" / "preservation_downstream_matrix.csv", newline="")))
    gaps = list(csv.DictReader(open(root / "tables" / "preservation_gap_summary.csv", newline="")))

    first_gap = gaps[0]
    key = {
        "experiment_seed": first_gap["experiment_seed"],
        "heldout_center": first_gap["heldout_center"],
        "expert_id": first_gap["expert_id"],
        "replicate_seed": first_gap["replicate_seed"],
    }
    paired = [
        row for row in downstream
        if all(row[field] == value for field, value in key.items())
    ]
    by_role = {row["row_role"]: row for row in paired}
    full = [
        row for row in downstream
        if row["experiment_seed"] == key["experiment_seed"]
        and row["heldout_center"] == key["heldout_center"]
        and row["expert_id"] == key["expert_id"]
        and row["row_role"] == ROW_REAL_FULL
    ]

    assert len(full) == 1
    assert {ROW_REAL_BUDGET, ROW_DECODE_MU, ROW_POSTERIOR, ROW_PRIOR}.issubset(by_role)
    assert by_role[ROW_REAL_BUDGET]["reference_ids_hash"] == by_role[ROW_DECODE_MU]["reference_ids_hash"]
    assert by_role[ROW_REAL_BUDGET]["reference_ids_hash"] == by_role[ROW_POSTERIOR]["reference_ids_hash"]
    assert by_role[ROW_PRIOR]["reference_sample_seed"] == "NA"
    expected_budget_gap = float(full[0]["bacc"]) - float(by_role[ROW_REAL_BUDGET]["bacc"])
    assert float(first_gap["budget_gap"]) == pytest.approx(expected_budget_gap)


def test_preservation_chance_adjusted_is_na_for_near_chance_real_budget(tmp_path: Path) -> None:
    cfg = _tiny_preservation_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_preservation_diagnosis(cfg)
    gaps_path = root / "tables" / "preservation_gap_summary.csv"
    rows = list(csv.DictReader(open(gaps_path, newline="")))

    # Force a focused check of the schema-level behavior with the produced rows:
    # rows at or below the guard must not carry a numeric preservation ratio.
    for row in rows:
        if float(row["real_source_budget_matched_bacc"]) <= 0.55:
            assert row["chance_adjusted_preservation"] == ""


def test_preservation_repair_tiny_cache_writes_protocol_artifacts(tmp_path: Path) -> None:
    cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_preservation_repair(cfg)

    expected = [
        "tables/feature_frame_ceiling_matrix.csv",
        "tables/decode_mu_repair_matrix.csv",
        "tables/repair_gap_summary.csv",
        "tables/source_pool_capacity_summary.csv",
        "tables/reconstruction_diagnostics.csv",
        "tables/source_probe_diagnostics.csv",
        "tables/training_loss_diagnostics.csv",
        "manifests/protocol_manifest.json",
        "manifests/expert_variant_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decode_mu_repair_matrix.csv", newline="")))
    gaps = list(csv.DictReader(open(root / "tables" / "repair_gap_summary.csv", newline="")))
    manifest = list(csv.DictReader(open(root / "manifests" / "expert_variant_manifest.csv", newline="")))

    assert leakage["status"] == "PASS"
    assert any(row["variant_id"] == PRIMARY_VARIANT and row["selection_source"] == "primary" for row in matrix)
    assert any(row["variant_id"] == "pca64_beta001_probe025" and row["selection_source"] == "diagnostic_only" for row in matrix)
    assert any(row["expert_pool_type"] == "source_union_excluding_target" for row in matrix)
    assert gaps
    assert all("2" not in row["source_scope"].split("|") for row in manifest if row["heldout_center"] == "2")


def test_preservation_repair_source_hashes_and_reference_strata_are_invariant(tmp_path: Path) -> None:
    cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_preservation_repair(cfg)
    gaps = list(csv.DictReader(open(root / "tables" / "repair_gap_summary.csv", newline="")))
    per_source = [row for row in gaps if row["expert_pool_type"] == "per_source"]
    grouped = {}
    for row in per_source:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["replicate_seed"])
        grouped.setdefault(key, []).append(row)
    assert grouped
    for rows in grouped.values():
        hashes = {row["source_budget_index_hash"] for row in rows}
        strata = {row["source_utility_stratum_reference"] for row in rows}
        assert len(hashes) == 1
        assert len(strata) == 1
        assert all(row["pca_compression_gap"] != "" for row in rows if row["variant_id"] != "current_pca200_beta1_reference")


def test_preservation_repair_decision_uses_only_primary_rows() -> None:
    rows = []
    for variant, selection, bacc in (
        ("current_pca200_beta1_reference", "reference_only", 0.55),
        ("pca64_beta001", "primary", 0.55),
        ("pca64_beta001_probe025", "diagnostic_only", 0.95),
        ("source_union_pca64_beta001_diagnostic", "diagnostic_only", 0.95),
    ):
        rows.append(
            {
                "experiment_seed": "42",
                "heldout_center": "0",
                "expert_id": "1" if "source_union" not in variant else "source_union_excluding_target",
                "expert_pool_type": "per_source" if "source_union" not in variant else "source_union_excluding_target",
                "variant_id": variant,
                "replicate_seed": "17",
                "source_utility_stratum_reference": "high",
                "selection_source": selection,
                "status": "ok",
                "cvae_decode_mu_bacc": str(bacc),
                "decoder_gap_vs_real_budget": "0.0",
                "pca_compression_gap": "0.0",
                "variant_real_budget_bacc": "0.9",
                "source_probe_train_acc": "0.95" if "probe" in variant else "",
                "source_probe_val_acc": "0.5" if "probe" in variant else "",
            }
        )
    cfg = _tiny_repair_config(Path("/tmp"))
    decision = _decision(rows, cfg, leakage_status="PASS")

    assert decision["primary_verdict"] == "REPAIR_FAIL"
    assert "PROBE_RESCUE" in decision["diagnostic_flags"]
    primary_rows = _decision_rows(rows, PRIMARY_VARIANT, "per_source")
    assert len(primary_rows) == 1
    assert primary_rows[0]["variant_id"] == PRIMARY_VARIANT


def test_preservation_repair_kl_warmup_reaches_beta_final(tmp_path: Path) -> None:
    cfg = _tiny_repair_config(tmp_path)
    variant = next(v for v in cfg.variants if v.variant_id == PRIMARY_VARIANT)

    assert _beta_for_epoch(variant, 1) == pytest.approx(variant.beta_final / variant.kl_warmup_epochs)
    assert _beta_for_epoch(variant, variant.kl_warmup_epochs) == pytest.approx(variant.beta_final)
    assert _beta_for_epoch(variant, variant.kl_warmup_epochs + 10) == pytest.approx(variant.beta_final)


def test_preservation_sampling_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)

    root = run_preservation_sampling(sampling_cfg)

    expected = [
        "tables/sampling_downstream_matrix.csv",
        "tables/sampling_gap_summary.csv",
        "tables/latent_distribution_diagnostics.csv",
        "tables/source_pool_sampling_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/sampling_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "sampling_downstream_matrix.csv", newline="")))
    gaps = list(csv.DictReader(open(root / "tables" / "sampling_gap_summary.csv", newline="")))
    manifest = list(csv.DictReader(open(root / "manifests" / "sampling_model_manifest.csv", newline="")))

    assert leakage["status"] == "PASS"
    assert any(row["row_role"] == SAMPLING_ROW_POSTERIOR and row["posterior_temperature"] == "1.0" for row in matrix)
    assert any(row["row_role"] == SAMPLING_ROW_PRIOR and row["prior_scale"] == "1.0" for row in matrix)
    assert any(row["row_role"] == "cvae_empirical_mu_sample_diagnostic" for row in matrix)
    assert any(row["expert_pool_type"] == "source_union_excluding_target" for row in matrix)
    assert gaps
    assert manifest
    assert all(row["variant_id"] in {"pca64_beta001", "source_union_pca64_beta001_diagnostic"} for row in matrix)


def test_preservation_sampling_requires_frozen_repair_reference(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    sampling_cfg = _tiny_sampling_config(tmp_path, tmp_path / "missing_repair")

    root = run_preservation_sampling(sampling_cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage["status"] == "FAIL"
    assert any("Missing frozen repair gap summary" in violation for violation in leakage["violations"])


def test_preservation_sampling_mono_class_target_eval_is_ineligible_not_protocol_fail(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42, mono_test_centers={"2"})
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)

    root = run_preservation_sampling(sampling_cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "sampling_downstream_matrix.csv", newline="")))

    invalid = [row for row in matrix if row["heldout_center"] == "2"]
    valid = [row for row in matrix if row["heldout_center"] != "2"]

    assert leakage["status"] == "PASS"
    assert invalid
    assert {row["status"] for row in invalid} == {"ineligible"}
    assert {row["error_message"] for row in invalid} == {"mono_class_target_eval"}
    assert any(row["status"] == "ok" for row in valid)


def test_preservation_sampling_source_hashes_and_budget_types(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)

    root = run_preservation_sampling(sampling_cfg)
    matrix = list(csv.DictReader(open(root / "tables" / "sampling_downstream_matrix.csv", newline="")))
    ok_rows = [row for row in matrix if row["status"] == "ok" and row["expert_pool_type"] == "per_source"]
    grouped = {}
    for row in ok_rows:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["replicate_seed"])
        grouped.setdefault(key, []).append(row)
    assert grouped
    first = next(rows for rows in grouped.values() if any(row["row_role"] == SAMPLING_ROW_PRIOR for row in rows))
    by_role = {}
    for row in first:
        if row["row_role"] in {ROW_REAL_BUDGET, SAMPLING_ROW_DECODE_MU, SAMPLING_ROW_POSTERIOR, SAMPLING_ROW_PRIOR}:
            by_role.setdefault(row["row_role"], []).append(row)

    real_hash = by_role[ROW_REAL_BUDGET][0]["source_budget_index_hash"]
    assert by_role[SAMPLING_ROW_DECODE_MU][0]["source_budget_index_hash"] == real_hash
    assert next(row for row in by_role[SAMPLING_ROW_POSTERIOR] if row["posterior_temperature"] == "1.0")["source_budget_index_hash"] == real_hash
    prior = next(row for row in by_role[SAMPLING_ROW_PRIOR] if row["prior_scale"] == "1.0")
    assert prior["source_budget_index_hash"] == "NA"
    assert prior["budget_match_type"] == "class_count_matched"
    assert by_role[ROW_REAL_BUDGET][0]["budget_match_type"] == "source_record_matched"


def test_preservation_sampling_config_rejects_noncanonical_values(tmp_path: Path) -> None:
    repair_root = tmp_path / "repair"
    payload = _tiny_sampling_payload(tmp_path, repair_root)
    payload["sampling"]["prior_scales_primary"] = [0.5]

    with pytest.raises(Exception, match="Primary prior scale"):
        parse_sampling_config(payload, base_dir=tmp_path)


def test_prior_calibration_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)

    root = run_prior_calibration(cfg)

    expected = [
        "tables/calibrated_prior_downstream_matrix.csv",
        "tables/calibrated_prior_gap_summary.csv",
        "tables/latent_prior_parameter_manifest.csv",
        "tables/latent_prior_diagnostics.csv",
        "tables/source_pool_prior_calibration_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/prior_calibration_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "calibrated_prior_downstream_matrix.csv", newline="")))
    gaps = list(csv.DictReader(open(root / "tables" / "calibrated_prior_gap_summary.csv", newline="")))
    manifest = list(csv.DictReader(open(root / "tables" / "latent_prior_parameter_manifest.csv", newline="")))

    assert leakage["status"] == "PASS"
    assert any(row["prior_method"] == PRIMARY_PRIOR_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "cvae_cc_diag_shrinkage_gaussian_prior_sample_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "cvae_standard_prior_sample_reference" for row in matrix)
    assert any(row["expert_pool_type"] == "source_union_excluding_target" for row in matrix)
    assert all(row["generated_features_hash"] for row in matrix if row["status"] == "ok")
    assert all(row["prediction_hash"] for row in matrix if row["status"] == "ok")
    assert gaps
    assert manifest


def test_prior_calibration_requires_sampling_reference(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    cfg = _tiny_prior_calibration_config(tmp_path, repair_root, tmp_path / "missing_sampling")

    root = run_prior_calibration(cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage["status"] == "FAIL"
    assert any("Missing sampling gap summary" in violation for violation in leakage["violations"])


def test_prior_calibration_config_rejects_noncanonical_primary(tmp_path: Path) -> None:
    payload = _tiny_prior_calibration_payload(tmp_path, tmp_path / "repair", tmp_path / "sampling")
    payload["prior_calibration"]["primary_method"] = "cvae_empirical_mu_codebook_prior_sample_diagnostic"

    with pytest.raises(Exception, match="primary_method"):
        parse_prior_calibration_config(payload, base_dir=tmp_path)


def test_prior_calibration_source_under_threshold_is_ineligible(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    payload = _tiny_prior_calibration_payload(tmp_path, repair_root, sampling_root)
    payload["prior_calibration"]["min_prior_fit_records_per_class"] = 100
    payload["prior_calibration"]["full_cov_min_records_per_class"] = 100
    cfg = parse_prior_calibration_config(payload, base_dir=tmp_path)

    root = run_prior_calibration(cfg)
    matrix = list(csv.DictReader(open(root / "tables" / "calibrated_prior_downstream_matrix.csv", newline="")))

    assert matrix
    assert any(row["status"] == "ineligible" and "insufficient_source_class_records" in row["error_message"] for row in matrix)


def test_covariance_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)

    root = run_covariance_prior_confirmation(cfg)

    expected = [
        "tables/covariance_prior_downstream_matrix.csv",
        "tables/covariance_prior_gap_summary.csv",
        "tables/covariance_prior_parameter_manifest.csv",
        "tables/covariance_fallback_audit.csv",
        "tables/covariance_prior_low_stratum_audit.csv",
        "tables/source_pool_covariance_prior_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/covariance_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "covariance_prior_downstream_matrix.csv", newline="")))
    params = list(csv.DictReader(open(root / "tables" / "covariance_prior_parameter_manifest.csv", newline="")))
    fallback = list(csv.DictReader(open(root / "tables" / "covariance_fallback_audit.csv", newline="")))
    summary = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert any(row["prior_method"] == PRIMARY_COVARIANCE_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "cvae_cc_diag_aggregate_prior_reference" for row in matrix)
    assert any(row["prior_method"] == "cvae_standard_prior_sample_reference" for row in matrix)
    assert any(row["expert_pool_type"] == "source_union_excluding_target" for row in matrix)
    assert all(row["generated_features_hash"] for row in matrix if row["status"] == "ok")
    assert all(row["prediction_hash"] for row in matrix if row["status"] == "ok")
    assert all("trace_before_shrinkage" in row for row in params)
    assert any(row["covariance_fallback_used"] == "True" for row in fallback)
    assert "PASS does not unlock routing directly." in summary


def test_covariance_prior_config_rejects_noncanonical_alpha(tmp_path: Path) -> None:
    payload = _tiny_covariance_prior_payload(tmp_path, tmp_path / "repair", tmp_path / "sampling", tmp_path / "prior")
    payload["covariance_prior"]["covariance_shrinkage_alpha"] = 0.2

    with pytest.raises(Exception, match="covariance_shrinkage_alpha"):
        parse_covariance_prior_config(payload, base_dir=tmp_path)


def test_covariance_prior_requires_imported_references(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    cfg = _tiny_covariance_prior_config(tmp_path, repair_root, tmp_path / "missing_sampling", tmp_path / "missing_prior")

    root = run_covariance_prior_confirmation(cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage["status"] == "FAIL"
    assert any("Missing sampling gap summary" in violation for violation in leakage["violations"])


def test_covariance_prior_formula_uses_aggregate_diag_and_deterministic_psd() -> None:
    mu = np.asarray([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]], dtype=float)
    post_var = np.asarray([[0.5, 0.2], [0.7, 0.3], [0.9, 0.4]], dtype=float)
    sigma_emp = np.cov(mu, rowvar=False, ddof=1) + np.diag(post_var.mean(axis=0))
    sigma_diag = np.diag(np.diag(sigma_emp))

    psd1, factor1, health1 = _stabilized_covariance_psd(sigma_emp, alpha=0.10, eigenvalue_floor=1.0e-4)
    psd2, factor2, health2 = _stabilized_covariance_psd(sigma_emp, alpha=0.10, eigenvalue_floor=1.0e-4)

    assert np.allclose(sigma_diag, np.diag(np.diag(np.cov(mu, rowvar=False, ddof=1) + np.diag(post_var.mean(axis=0)))))
    assert np.allclose(psd1, psd2)
    assert np.allclose(factor1, factor2)
    assert np.linalg.eigvalsh(psd1).min() >= 1.0e-4 - 1.0e-10
    assert health1 == health2


def test_covariance_viability_audit_tiny_artifact_writes_expected_outputs(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    cfg = _tiny_covariance_viability_config(tmp_path, cov_root)

    root = run_covariance_prior_viability_audit(cfg)

    expected = [
        "tables/conditional_viability_cells.csv",
        "tables/variant_real_stratum_summary.csv",
        "tables/original_9_cell_failure_audit.csv",
        "tables/center_seed_stability_summary.csv",
        "tables/fallback_viability_audit.csv",
        "tables/source_pool_viability_summary.csv",
        "manifests/protocol_manifest.json",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    conditional = list(csv.DictReader(open(root / "tables" / "conditional_viability_cells.csv", newline="")))
    strata = list(csv.DictReader(open(root / "tables" / "variant_real_stratum_summary.csv", newline="")))
    original = list(csv.DictReader(open(root / "tables" / "original_9_cell_failure_audit.csv", newline="")))
    summary = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert all(float(row["variant_real_budget_bacc"]) >= 0.80 for row in conditional)
    assert any(row["variant_real_stratum"] == "selection_denominator" for row in strata)
    assert original
    assert "This audit does not replace the original 9-cell covariance-prior verdict." in summary
    assert "does not evaluate routing" in summary


def test_covariance_viability_config_rejects_wrong_imported_artifact_name(tmp_path: Path) -> None:
    payload = _tiny_covariance_viability_payload(tmp_path, tmp_path / "wrong_artifact")

    with pytest.raises(Exception, match="covariance_confirmation_artifact_root"):
        parse_covariance_viability_config(payload, base_dir=tmp_path)


def test_covariance_viability_missing_artifact_is_protocol_fail(tmp_path: Path) -> None:
    cfg = _tiny_covariance_viability_config(tmp_path, tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1")

    root = run_covariance_prior_viability_audit(cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage["status"] == "FAIL"
    assert any("Missing covariance confirmation artifact files" in violation for violation in leakage["violations"])


def test_covariance_shrinkage_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    viability_cfg = _tiny_covariance_viability_config(tmp_path, cov_root)
    viability_root = run_covariance_prior_viability_audit(viability_cfg)
    cfg = _tiny_covariance_shrinkage_config(tmp_path, repair_root, sampling_root, prior_root, cov_root, viability_root)

    root = run_covariance_shrinkage_stability(cfg)

    expected = [
        "tables/shrinkage_prior_downstream_matrix.csv",
        "tables/shrinkage_prior_gap_summary.csv",
        "tables/shrinkage_alpha_comparison.csv",
        "tables/high_real_viability_summary.csv",
        "tables/original_9_stress_summary.csv",
        "tables/variant_real_stratum_summary.csv",
        "tables/covariance_health_by_alpha.csv",
        "tables/fallback_stability_audit.csv",
        "tables/source_pool_shrinkage_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/covariance_shrinkage_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "shrinkage_prior_downstream_matrix.csv", newline="")))
    health = list(csv.DictReader(open(root / "tables" / "covariance_health_by_alpha.csv", newline="")))
    summary = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert any(row["prior_method"] == PRIMARY_SHRINKAGE_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "cvae_cc_cov_diag_shrinkage050_prior_sample_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "cvae_cc_cov_diag_shrinkage090_prior_sample_diagnostic" for row in matrix)
    assert all("offdiag_frobenius_ratio" in row for row in matrix)
    assert all("trace_ratio_vs_diag" in row for row in matrix)
    assert health
    assert "does not evaluate routing" in summary


def test_covariance_shrinkage_alpha010_reference_uses_covariance_confirmation(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    viability_cfg = _tiny_covariance_viability_config(tmp_path, cov_root)
    viability_root = run_covariance_prior_viability_audit(viability_cfg)

    prior_gap_path = prior_root / "tables" / "calibrated_prior_gap_summary.csv"
    prior_rows = list(csv.DictReader(open(prior_gap_path, newline="")))
    fieldnames = list(prior_rows[0])
    for row in prior_rows:
        if row["row_role"] == "cvae_cc_full_cov_gaussian_prior_sample_diagnostic":
            row["calibrated_prior_bacc"] = "0.0"
            row["total_calibrated_prior_cvae_gap"] = "999.0"
    with prior_gap_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prior_rows)

    payload = _tiny_covariance_shrinkage_payload(tmp_path, repair_root, sampling_root, prior_root, cov_root, viability_root)
    payload["covariance_shrinkage"]["alpha010_repro_abs_tol_bacc"] = 1.0e-6
    cfg = parse_covariance_shrinkage_config(payload, base_dir=tmp_path)

    root = run_covariance_shrinkage_stability(cfg)

    matrix = list(csv.DictReader(open(root / "tables" / "shrinkage_prior_downstream_matrix.csv", newline="")))
    alpha010_rows = [
        row
        for row in matrix
        if row["row_role"] == "cvae_cc_cov_shrinkage010_prior_reference" and row["status"] == "ok"
    ]

    assert alpha010_rows
    assert any(float(row["imported_full_cov_diagnostic_bacc"]) != 0.0 for row in alpha010_rows)
    assert all(abs(float(row["bacc"]) - float(row["imported_full_cov_diagnostic_bacc"])) <= 1.0e-6 for row in alpha010_rows)


def test_covariance_shrinkage_config_rejects_noncanonical_primary_alpha(tmp_path: Path) -> None:
    payload = _tiny_covariance_shrinkage_payload(
        tmp_path,
        tmp_path / "repair",
        tmp_path / "sampling",
        tmp_path / "prior",
        tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1",
        tmp_path / "virchow2_cvae_covariance_prior_viability_audit_v1",
    )
    payload["covariance_shrinkage"]["primary_covariance_shrinkage_alpha"] = 0.50

    with pytest.raises(Exception, match="primary_covariance_shrinkage_alpha"):
        parse_covariance_shrinkage_config(payload, base_dir=tmp_path)


def test_covariance_shrinkage_missing_imports_is_protocol_fail(tmp_path: Path) -> None:
    cfg = _tiny_covariance_shrinkage_config(
        tmp_path,
        tmp_path / "repair",
        tmp_path / "sampling",
        tmp_path / "prior",
        tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1",
        tmp_path / "virchow2_cvae_covariance_prior_viability_audit_v1",
    )

    root = run_covariance_shrinkage_stability(cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))

    assert leakage["status"] == "FAIL"
    assert any("Missing imported shrinkage reference artifacts" in violation for violation in leakage["violations"])


def test_source_union_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    cfg = _tiny_source_union_gmm_config(tmp_path, repair_root, sampling_root, prior_root, cov_root)

    root = run_source_union_gmm_prior(cfg)

    expected = [
        "tables/gmm_prior_downstream_matrix.csv",
        "tables/gmm_prior_gap_summary.csv",
        "tables/source_union_gmm_summary.csv",
        "tables/per_source_gmm_diagnostic_summary.csv",
        "tables/gmm_component_diagnostics.csv",
        "tables/latent_mode_coverage_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/gmm_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "gmm_prior_downstream_matrix.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "source_union_gmm_summary.csv", newline="")))
    diagnostics = list(csv.DictReader(open(root / "tables" / "gmm_component_diagnostics.csv", newline="")))
    nn = list(csv.DictReader(open(root / "tables" / "nearest_neighbor_memorization_audit.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert any(row["prior_method"] == PRIMARY_GMM_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k8_shuffled_label_control_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "per_source_cc_diag_gmm_k8_prior_sample_diagnostic" for row in matrix)
    assert all(
        row["expert_pool_type"] == "source_union_excluding_target"
        for row in matrix
        if row["prior_method"] == PRIMARY_GMM_METHOD
    )
    assert all(row["generated_features_hash"] for row in matrix if row["status"] == "ok")
    assert all(row["prediction_hash"] for row in matrix if row["status"] == "ok")
    assert "paired_delta_vs_alpha010_ci95" in summary[0]
    assert diagnostics
    assert nn and {row["audit_interpretation"] for row in nn} == {"memorization_proximity_audit_only_not_formal_privacy"}
    assert "It does not evaluate metadata routing." in report
    assert "It does not evaluate decentralized per-source expert selection." in report
    assert "It does not provide formal differential privacy." in report


def test_source_union_gmm_prior_config_rejects_noncanonical_primary(tmp_path: Path) -> None:
    payload = _tiny_source_union_gmm_payload(
        tmp_path,
        tmp_path / "repair",
        tmp_path / "sampling",
        tmp_path / "prior",
        tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1",
    )
    payload["gmm_prior"]["primary_method"] = "source_union_cc_diag_gmm_k4_prior_sample_diagnostic"

    with pytest.raises(Exception, match="primary_method"):
        parse_source_union_gmm_prior_config(payload, base_dir=tmp_path)


def test_source_union_gmm_prior_marks_collapsed_fit_ineligible(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    payload = _tiny_source_union_gmm_payload(tmp_path, repair_root, sampling_root, prior_root, cov_root)
    payload["gmm_prior"]["min_effective_gmm_components"] = 99
    cfg = parse_source_union_gmm_prior_config(payload, base_dir=tmp_path)

    root = run_source_union_gmm_prior(cfg)
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "gmm_prior_downstream_matrix.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "source_union_gmm_summary.csv", newline="")))

    assert leakage["status"] == "PASS"
    assert any(row["status"] == "gmm_component_collapse" for row in matrix)
    assert summary[0]["primary_verdict"] == "GMM_FIT_INELIGIBLE"


def test_source_union_gmm_prior_mono_class_target_eval_is_not_fit_ineligible(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42, mono_test_centers={"1"})
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    cfg = _tiny_source_union_gmm_config(tmp_path, repair_root, sampling_root, prior_root, cov_root)

    root = run_source_union_gmm_prior(cfg)
    summary = list(csv.DictReader(open(root / "tables" / "source_union_gmm_summary.csv", newline="")))
    matrix = list(csv.DictReader(open(root / "tables" / "gmm_prior_downstream_matrix.csv", newline="")))

    assert summary[0]["primary_verdict"] != "GMM_FIT_INELIGIBLE"
    assert any(
        row["prior_method"] == PRIMARY_GMM_METHOD
        and row["heldout_center"] == "1"
        and row["status"] == "ineligible"
        and row["error_message"] == "mono_class_target_eval"
        for row in matrix
    )


def test_source_union_balanced_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    source_union_gmm_cfg = _tiny_source_union_gmm_config(tmp_path, repair_root, sampling_root, prior_root, cov_root)
    source_union_gmm_root = run_source_union_gmm_prior(source_union_gmm_cfg)
    cfg = _tiny_source_union_balanced_gmm_config(
        tmp_path,
        repair_root,
        sampling_root,
        prior_root,
        cov_root,
        source_union_gmm_root,
    )

    root = run_source_union_balanced_gmm_prior(cfg)

    expected = [
        "tables/balanced_gmm_downstream_matrix.csv",
        "tables/balanced_gmm_gap_summary.csv",
        "tables/source_union_balanced_gmm_summary.csv",
        "tables/source_center_balance_audit.csv",
        "tables/gmm_component_diagnostics.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_cell_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "tables/per_source_balanced_gmm_diagnostic_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/balanced_gmm_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "balanced_gmm_downstream_matrix.csv", newline="")))
    balance = list(csv.DictReader(open(root / "tables" / "source_center_balance_audit.csv", newline="")))
    coverage = list(csv.DictReader(open(root / "tables" / "generated_component_coverage_audit.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "source_union_balanced_gmm_summary.csv", newline="")))
    nn = list(csv.DictReader(open(root / "tables" / "nearest_neighbor_memorization_audit.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert any(row["prior_method"] == PRIMARY_BALANCED_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "source_union_center_balanced_cc_diag_gmm_k16_shuffled_label_control_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "per_source_center_balanced_cc_diag_gmm_k16_prior_sample_diagnostic" for row in matrix)
    assert all(
        row["heldout_center"] != row["source_center"]
        for row in balance
        if row["expert_pool_type"] == "source_union_excluding_target"
    )
    assert any(row["source_center_balance_strategy"] == "center_balanced" for row in matrix)
    assert all(row["generated_features_hash"] for row in matrix if row["status"] == "ok" and row["prior_method"] != "source_union_cc_diag_gmm_k16_prior_sample_reference")
    assert all(row["prediction_hash"] for row in matrix if row["status"] == "ok" and row["prior_method"] != "source_union_cc_diag_gmm_k16_prior_sample_reference")
    assert coverage
    assert "paired_delta_vs_vanilla_k16_ci95" in summary[0]
    assert nn and {row["audit_interpretation"] for row in nn} == {"memorization_proximity_audit_only_not_formal_privacy"}
    assert "It does not evaluate metadata routing." in report
    assert "It does not evaluate support-NELBO routing." in report
    assert "It does not evaluate decentralized per-source expert selection." in report
    assert "It does not provide formal differential privacy." in report


def test_source_union_balanced_gmm_prior_config_rejects_noncanonical_primary(tmp_path: Path) -> None:
    payload = _tiny_source_union_balanced_gmm_payload(
        tmp_path,
        tmp_path / "repair",
        tmp_path / "sampling",
        tmp_path / "prior",
        tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1",
        tmp_path / "virchow2_cvae_source_union_gmm_prior_v1",
    )
    payload["balanced_gmm_prior"]["primary_method"] = "source_union_center_balanced_cc_diag_gmm_k8_prior_sample_diagnostic"

    with pytest.raises(Exception, match="primary_method"):
        parse_source_union_balanced_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_k16_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_k16_gmm_payload(tmp_path)
    cfg = parse_decentralized_k16_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_k16_gmm_prior(cfg)

    expected = [
        "tables/decentralized_k16_downstream_matrix.csv",
        "tables/decentralized_k16_gap_summary.csv",
        "tables/decentralized_k16_summary.csv",
        "tables/exported_source_summary_manifest.csv",
        "tables/composed_prior_component_manifest.csv",
        "tables/source_summary_diagnostics.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_k16_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
        "summaries/source_0/class_0_k4_summary.npz",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_k16_downstream_matrix.csv", newline="")))
    summary_manifest_reader = csv.DictReader(open(root / "tables" / "exported_source_summary_manifest.csv", newline=""))
    summary_manifest = list(summary_manifest_reader)
    composition = list(csv.DictReader(open(root / "tables" / "composed_prior_component_manifest.csv", newline="")))
    diagnostics = list(csv.DictReader(open(root / "tables" / "source_summary_diagnostics.csv", newline="")))
    real_reference = list(csv.DictReader(open(root / "tables" / "real_feature_reference_matrix.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert "heldout_center" not in (summary_manifest_reader.fieldnames or [])
    assert summary_manifest
    assert diagnostics and all(row["heldout_center"] != row["source_center"] for row in composition)
    assert any(row["prior_method"] == PRIMARY_DECENTRALIZED_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_k4x4_cc_diag_gmm_k16_late_arith" for row in matrix)
    assert any(row["prior_method"] == "decentralized_k16_shuffled_summary_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_k16_shuffled_label_control" for row in matrix)
    assert any(row["prior_method"] == "real_source_embedding_classifier_dense_reference" and row["status"] == "ok" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k16_prior_sample_reference" and row["status"] == "missing_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_k4x4_cc_diag_gmm_k16_support_nelbo_weighted_geom_diagnostic" and row["status"] == "diagnostic_disabled" for row in matrix)
    assert all(row["heldout_center"] != row["source_center"] for row in composition)
    for key in {(row["experiment_seed"], row["heldout_center"], row["class_label"]) for row in composition}:
        total = sum(
            float(row["component_weight_after_equal_source_normalization"])
            for row in composition
            if (row["experiment_seed"], row["heldout_center"], row["class_label"]) == key
        )
        assert abs(total - 1.0) < 1.0e-6
    assert real_reference
    assert "not a target-specific compatibility-routing result" in report
    assert "not a formal differential privacy claim" in report


def test_decentralized_k16_gmm_prior_config_rejects_non_virchow2(tmp_path: Path) -> None:
    payload = _tiny_decentralized_k16_gmm_payload(tmp_path)
    payload["inputs"]["backbone"] = "dinov2"

    with pytest.raises(Exception, match="backbone=virchow2"):
        parse_decentralized_k16_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_k16_gmm_prior_ineligible_summary_does_not_fail_leakage(tmp_path: Path) -> None:
    payload = _tiny_decentralized_k16_gmm_payload(tmp_path)
    payload["decentralized_k16_prior"]["min_count_for_k4"] = 999
    cfg = parse_decentralized_k16_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_k16_gmm_prior(cfg)

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_k16_summary.csv", newline="")))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_k16_downstream_matrix.csv", newline="")))
    composition = list(csv.DictReader(open(root / "tables" / "composed_prior_component_manifest.csv", newline="")))

    assert leakage["status"] == "PASS"
    assert summary[0]["primary_verdict"] == "INELIGIBLE"
    assert any(
        row["prior_method"] == PRIMARY_DECENTRALIZED_METHOD
        and row["status"] == "ineligible_component_fit"
        for row in matrix
    )
    assert any(row["summary_status"] == "ineligible_component_fit" for row in composition)
    assert all(row["summary_error_message"] for row in composition if row["summary_status"] == "ineligible_component_fit")


def test_decentralized_adaptive_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_adaptive_gmm_payload(tmp_path)
    cfg = parse_decentralized_adaptive_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_adaptive_gmm_prior(cfg)

    expected = [
        "tables/decentralized_adaptive_downstream_matrix.csv",
        "tables/decentralized_adaptive_gap_summary.csv",
        "tables/decentralized_adaptive_summary.csv",
        "tables/exported_source_summary_manifest.csv",
        "tables/composed_prior_component_manifest.csv",
        "tables/source_summary_diagnostics.csv",
        "tables/adaptive_k_intervention_audit.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_adaptive_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
        "summaries/source_0/class_0_adaptive_largest_viable_summary.npz",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_adaptive_downstream_matrix.csv", newline="")))
    summary_manifest_reader = csv.DictReader(open(root / "tables" / "exported_source_summary_manifest.csv", newline=""))
    summary_manifest = list(summary_manifest_reader)
    composition = list(csv.DictReader(open(root / "tables" / "composed_prior_component_manifest.csv", newline="")))
    diagnostics = list(csv.DictReader(open(root / "tables" / "source_summary_diagnostics.csv", newline="")))
    adaptive_summary = list(csv.DictReader(open(root / "tables" / "decentralized_adaptive_summary.csv", newline="")))
    intervention = list(csv.DictReader(open(root / "tables" / "adaptive_k_intervention_audit.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert "heldout_center" not in (summary_manifest_reader.fieldnames or [])
    assert summary_manifest
    assert diagnostics and any(
        int(row["selected_k"]) < 4
        for row in diagnostics
        if row["selection_rule"] == "largest_viable" and row["status"] == "ok"
    )
    assert adaptive_summary[0]["adaptive_k_intervention_active"] == "True"
    assert intervention and {"selected_k", "component_count_after_composition", "sample_mass_assigned"}.issubset(intervention[0])
    assert any(row["prior_method"] == PRIMARY_ADAPTIVE_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_bic_selected_cc_diag_gmm_late_geom" for row in matrix)
    assert any(row["prior_method"] == "decentralized_adaptive_k_shuffled_summary_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_adaptive_k_shuffled_label_control" for row in matrix)
    assert any(row["prior_method"] == "real_source_embedding_classifier_dense_reference" and row["status"] == "ok" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k16_prior_sample_reference" and row["status"] == "missing_reference" for row in matrix)
    assert all(row["heldout_center"] != row["source_center"] for row in composition)
    for key in {(row["experiment_seed"], row["heldout_center"], row["class_label"]) for row in composition}:
        total = sum(
            float(row["component_weight_after_equal_source_normalization"])
            for row in composition
            if (row["experiment_seed"], row["heldout_center"], row["class_label"]) == key
        )
        assert abs(total - 1.0) < 1.0e-6
    assert "not a target-specific compatibility-routing result" in report
    assert "not a formal differential privacy claim" in report
    assert "Adaptive-K intervention active:" in report


def test_decentralized_adaptive_gmm_prior_rejects_non_virchow2(tmp_path: Path) -> None:
    payload = _tiny_decentralized_adaptive_gmm_payload(tmp_path)
    payload["inputs"]["backbone"] = "dinov2"

    with pytest.raises(Exception, match="backbone=virchow2"):
        parse_decentralized_adaptive_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_adaptive_gmm_prior_ineligible_only_when_k1_cannot_fit(tmp_path: Path) -> None:
    payload = _tiny_decentralized_adaptive_gmm_payload(tmp_path)
    payload["adaptive_gmm_prior"]["min_samples_per_component"] = 25
    cfg = parse_decentralized_adaptive_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_adaptive_gmm_prior(cfg)

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_adaptive_summary.csv", newline="")))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_adaptive_downstream_matrix.csv", newline="")))

    assert leakage["status"] == "PASS"
    assert summary[0]["primary_verdict"] == "INELIGIBLE"
    assert any(
        row["prior_method"] == PRIMARY_ADAPTIVE_METHOD
        and row["status"] == "ineligible_component_fit"
        for row in matrix
    )


def test_decentralized_reliability_weighted_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_reliability_weighted_gmm_payload(tmp_path)
    cfg = parse_decentralized_reliability_weighted_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_reliability_weighted_gmm_prior(cfg)

    expected = [
        "tables/decentralized_reliability_downstream_matrix.csv",
        "tables/decentralized_reliability_gap_summary.csv",
        "tables/decentralized_reliability_summary.csv",
        "tables/source_reliability_manifest.csv",
        "tables/reliability_weight_manifest.csv",
        "tables/source_reliability_rank_vs_target_utility.csv",
        "tables/centerwise_delta_summary.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_reliability_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_reliability_downstream_matrix.csv", newline="")))
    reliability_reader = csv.DictReader(open(root / "tables" / "source_reliability_manifest.csv", newline=""))
    reliability = list(reliability_reader)
    weights = list(csv.DictReader(open(root / "tables" / "reliability_weight_manifest.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_reliability_summary.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["fold_weight_manifest_excludes_heldout_center"] is True
    assert "heldout_center" not in (reliability_reader.fieldnames or [])
    assert reliability and weights
    assert any(row["prior_method"] == PRIMARY_RELIABILITY_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_equal_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_source_reliability_pool_only_geom" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_source_reliability_budget_only_geom" for row in matrix)
    assert any(row["prior_method"] == "decentralized_reliability_shuffled_summary_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_reliability_shuffled_label_control" for row in matrix)
    assert all(row["heldout_center"] != row["source_center"] for row in weights)
    for key in {(row["experiment_seed"], row["heldout_center"], row["replicate_seed"]) for row in weights}:
        subset = [
            row for row in weights
            if (row["experiment_seed"], row["heldout_center"], row["replicate_seed"]) == key
        ]
        assert abs(sum(float(row["normalized_reliability_weight"]) for row in subset) - 1.0) < 1.0e-6
        assert sum(int(row["synthetic_per_class_budget"]) for row in subset) == 128
        assert all(int(row["synthetic_per_class_budget"]) >= 8 for row in subset)
    assert "neutral_reliability_fallback_count" in summary[0]
    assert "mean_l1_distance_from_uniform" in summary[0]
    assert "not a target-specific compatibility-routing result" in report


def test_decentralized_reliability_weighted_gmm_prior_rejects_invalid_backbone(tmp_path: Path) -> None:
    payload = _tiny_decentralized_reliability_weighted_gmm_payload(tmp_path)
    payload["inputs"]["backbone"] = "dinov2"

    with pytest.raises(Exception, match="backbone=virchow2"):
        parse_decentralized_reliability_weighted_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_component_union_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_component_union_payload(tmp_path)
    cfg = parse_decentralized_component_union_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_component_union_prior(cfg)

    expected = [
        "tables/component_union_downstream_matrix.csv",
        "tables/component_union_gap_summary.csv",
        "tables/component_union_summary.csv",
        "tables/component_manifest.csv",
        "tables/prototype_manifest.csv",
        "tables/component_coverage_audit.csv",
        "tables/source_weight_manifest.csv",
        "tables/negative_control_summary.csv",
        "tables/source_ablation_audit.csv",
        "tables/paired_generation_audit.csv",
        "tables/source_summary_diagnostics.csv",
        "tables/source_reliability_manifest.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_component_union_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "component_union_downstream_matrix.csv", newline="")))
    component_manifest = list(csv.DictReader(open(root / "tables" / "component_manifest.csv", newline="")))
    weights = list(csv.DictReader(open(root / "tables" / "source_weight_manifest.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "component_union_summary.csv", newline="")))
    ablation = list(csv.DictReader(open(root / "tables" / "source_ablation_audit.csv", newline="")))
    paired = list(csv.DictReader(open(root / "tables" / "paired_generation_audit.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["fixed_all_source_inclusion"] is True
    assert protocol["tests_target_conditioned_routing"] is False
    assert any(row["prior_method"] == PRIMARY_COMPONENT_UNION_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_component_union_reliability_shrink025" for row in matrix)
    assert any(row["prior_method"] == "decentralized_component_union_reliability_shrink050" for row in matrix)
    assert any(row["prior_method"] == "decentralized_prototype_union_uniform" for row in matrix)
    assert any(row["prior_method"] == "decentralized_component_union_shuffled_summary_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_component_union_shuffled_label_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_equal_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_source_reliability_weighted_geom_reference" for row in matrix)
    assert any(row["row_scope"] == "fold_union_prior" for row in component_manifest)
    assert all(row["heldout_center"] != row["source_center"] for row in component_manifest if row["row_scope"] == "fold_union_prior")
    assert weights and all(row["heldout_center"] != row["source_center"] for row in weights)
    assert ablation and any(row["status"] == "not_applicable_target_source_excluded" for row in ablation)
    assert paired and {"paired_generation_invariant_key", "generated_features_hash"}.issubset(paired[0])
    assert "seed_cell_mean_bacc" in summary[0]
    assert "center_equal_mean_bacc" in summary[0]
    assert "does not test target-conditioned routing" in report
    assert "not a formal differential privacy claim" in report


def test_decentralized_component_union_prior_rejects_invalid_backbone(tmp_path: Path) -> None:
    payload = _tiny_decentralized_component_union_payload(tmp_path)
    payload["inputs"]["backbone"] = "dinov2"

    with pytest.raises(Exception, match="backbone=virchow2"):
        parse_decentralized_component_union_prior_config(payload, base_dir=tmp_path)


def test_paired_dense_all4_reliability_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_paired_dense_all4_reliability_payload(tmp_path)
    cfg = parse_paired_dense_all4_reliability_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_paired_dense_all4_reliability_confirmation(cfg)

    expected = [
        "tables/paired_dense_all4_downstream_matrix.csv",
        "tables/paired_dense_all4_gap_summary.csv",
        "tables/paired_dense_all4_center_summary.csv",
        "tables/paired_dense_all4_summary.csv",
        "tables/source_reliability_manifest.csv",
        "tables/reliability_weight_manifest.csv",
        "tables/realized_budget_table.csv",
        "tables/excluded_cell_report.csv",
        "tables/paired_generation_invariant_audit.csv",
        "tables/paired_delta_summary.csv",
        "tables/negative_control_summary.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/paired_dense_all4_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "paired_dense_all4_downstream_matrix.csv", newline="")))
    reliability = list(csv.DictReader(open(root / "tables" / "source_reliability_manifest.csv", newline="")))
    weights = list(csv.DictReader(open(root / "tables" / "reliability_weight_manifest.csv", newline="")))
    budgets = list(csv.DictReader(open(root / "tables" / "realized_budget_table.csv", newline="")))
    audit = list(csv.DictReader(open(root / "tables" / "paired_generation_invariant_audit.csv", newline="")))
    deltas = list(csv.DictReader(open(root / "tables" / "paired_delta_summary.csv", newline="")))
    gap = list(csv.DictReader(open(root / "tables" / "paired_dense_all4_gap_summary.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "paired_dense_all4_summary.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["target_center_excluded_from_reliability"] is True
    assert protocol["dense_all4_fixed_inclusion"] is True
    assert protocol["top_k_selection_enabled"] is False
    assert protocol["inverse_reliability_definition"] == "rank_reversal_matched_entropy"
    assert {ROW_EQUAL_ALL4, ROW_RELIABILITY_ALL4_WEIGHTED, ROW_POOL_ONLY, ROW_BUDGET_ONLY}.issubset(
        {row["prior_method"] for row in matrix}
    )
    assert {ROW_SHRINK025, ROW_SHRINK050, ROW_SHUFFLED, ROW_INVERSE}.issubset(
        {row["prior_method"] for row in matrix}
    )
    assert reliability and all(row["heldout_center"] != row["source_center"] for row in reliability)
    assert all(row["target_eval_labels_used_for_reliability"] == "False" for row in reliability)
    assert audit and {row["audit_status"] for row in audit} == {"PASS"}
    assert any(row["method"] == ROW_SHRINK050 for row in deltas)
    assert "seed_cell_mean_bacc" in gap[0]
    assert "center_equal_mean_bacc" in gap[0]
    assert "best_reliability_method" in summary[0]
    assert "not sparse expert selection" in protocol["claim_boundary"]
    assert "Do not claim sparse routing" in report

    for key in {
        (row["method"], row["experiment_seed"], row["heldout_center"], row["replicate_seed"])
        for row in weights
    }:
        subset = [
            row for row in weights
            if (row["method"], row["experiment_seed"], row["heldout_center"], row["replicate_seed"]) == key
        ]
        assert abs(sum(float(row["final_normalized_weight"]) for row in subset) - 1.0) < 1.0e-6
        assert sum(int(row["synthetic_per_class_budget"]) for row in subset) == 128
    assert budgets and all(int(row["budget_sum_per_class"]) == 128 for row in budgets)


def test_paired_dense_all4_reliability_weight_rules_are_locked(tmp_path: Path) -> None:
    cfg = parse_paired_dense_all4_reliability_config(
        _tiny_paired_dense_all4_reliability_payload(tmp_path),
        base_dir=tmp_path,
    )
    rels = {
        "1": SourceReliability(42, 17, "1", 0.90, 0.90, 0.80, "ok", "", 20, "g1", "p1"),
        "2": SourceReliability(42, 17, "2", 0.80, 0.80, 0.60, "ok", "", 20, "g2", "p2"),
        "3": SourceReliability(42, 17, "3", 0.70, 0.70, 0.40, "ok", "", 20, "g3", "p3"),
        "4": SourceReliability(42, 17, "4", 0.60, 0.60, 0.20, "ok", "", 20, "g4", "p4"),
    }
    transform = _heldout_excluded_reliability_transform(cfg, "0", ("1", "2", "3", "4"), rels)
    plans = _variant_plans(cfg, ("1", "2", "3", "4"), transform, experiment_seed=42, heldout_center="0", replicate_seed=17)

    assert plans[ROW_EQUAL_ALL4]["budgets"] == {"1": 32, "2": 32, "3": 32, "4": 32}
    assert sum(plans[ROW_RELIABILITY_ALL4_WEIGHTED]["budgets"].values()) == 128
    assert all(value >= 8 for value in plans[ROW_RELIABILITY_ALL4_WEIGHTED]["budgets"].values())
    assert plans[ROW_POOL_ONLY]["weights"] == plans[ROW_RELIABILITY_ALL4_WEIGHTED]["weights"]
    assert plans[ROW_POOL_ONLY]["budgets"] == plans[ROW_EQUAL_ALL4]["budgets"]
    assert plans[ROW_BUDGET_ONLY]["weights"] == plans[ROW_EQUAL_ALL4]["weights"]
    assert plans[ROW_BUDGET_ONLY]["budgets"] == plans[ROW_RELIABILITY_ALL4_WEIGHTED]["budgets"]

    normal = plans[ROW_RELIABILITY_ALL4_WEIGHTED]["weights"]
    shrink = plans[ROW_SHRINK050]["weights"]
    assert math.isclose(shrink["1"], 0.5 * normal["1"] + 0.5 * 0.25)
    inverse = _inverse_rank_reversal_weights(("1", "2", "3", "4"), normal, transform["imputed_scores"])
    assert sorted(inverse.values()) == sorted(normal.values())
    assert inverse["4"] == max(normal.values())


def test_paired_dense_all4_reliability_rejects_support_usage(tmp_path: Path) -> None:
    payload = _tiny_paired_dense_all4_reliability_payload(tmp_path)
    payload["run_matrix"]["support_size"] = 8

    with pytest.raises(Exception, match="target support"):
        parse_paired_dense_all4_reliability_config(payload, base_dir=tmp_path)


def test_decentralized_reliability_top3_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_reliability_top3_gmm_payload(tmp_path)
    cfg = parse_decentralized_reliability_top3_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_reliability_top3_gmm_prior(cfg)

    expected = [
        "tables/decentralized_reliability_top3_downstream_matrix.csv",
        "tables/decentralized_reliability_top3_gap_summary.csv",
        "tables/decentralized_reliability_top3_summary.csv",
        "tables/source_reliability_manifest.csv",
        "tables/reliability_top3_selection_manifest.csv",
        "tables/top3_selection_stability.csv",
        "tables/centerwise_delta_summary.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_reliability_top3_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_reliability_top3_downstream_matrix.csv", newline="")))
    selections = list(csv.DictReader(open(root / "tables" / "reliability_top3_selection_manifest.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_reliability_top3_summary.csv", newline="")))
    negative = list(csv.DictReader(open(root / "tables" / "negative_control_summary.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["target_support_features_for_selection"] is False
    assert protocol["support8_context_rows_decision_excluded"] is True
    assert any(row["prior_method"] == PRIMARY_RELIABILITY_TOP3_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_reliability_all4_weighted_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_equal_all4_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_reliability_top3_shuffled_reliability_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_reliability_top3_random_source_drop_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support8_d1_3_1_primary_context" for row in matrix)
    assert selections and all(row["heldout_center"] not in row["selected_sources"].split("|") for row in selections)
    for row in selections:
        budgets = json.loads(row["synthetic_per_class_budget_json"])
        assert len(budgets) == 3
        assert sum(int(value) for value in budgets.values()) == 128
        assert all(int(value) >= 8 for value in budgets.values())
    assert "shuffled_reliability_control_gap" in summary[0]
    assert "random_source_drop_control_gap" in summary[0]
    assert negative and "shuffled_reliability_control_gap" in negative[0]
    assert "reliability-based sparse composition, not target-conditioned routing" in report


def test_decentralized_reliability_top3_gmm_prior_rejects_support_usage(tmp_path: Path) -> None:
    payload = _tiny_decentralized_reliability_top3_gmm_payload(tmp_path)
    payload["run_matrix"]["support_size"] = 8

    with pytest.raises(Exception, match="must not configure or consume target support"):
        parse_decentralized_reliability_top3_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_reliability_top3_gmm_prior_rejects_invalid_top_k(tmp_path: Path) -> None:
    payload = _tiny_decentralized_reliability_top3_gmm_payload(tmp_path)
    payload["generation"]["top_k_sources"] = 2

    with pytest.raises(Exception, match="top_k_sources must be locked to 3"):
        parse_decentralized_reliability_top3_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_source_inner_transfer_top3_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_source_inner_transfer_top3_gmm_payload(tmp_path)
    cfg = parse_decentralized_source_inner_transfer_top3_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_source_inner_transfer_top3_gmm_prior(cfg)

    expected = [
        "tables/decentralized_source_inner_transfer_downstream_matrix.csv",
        "tables/decentralized_source_inner_transfer_gap_summary.csv",
        "tables/decentralized_source_inner_transfer_summary.csv",
        "tables/source_inner_transfer_matrix.csv",
        "tables/source_inner_subset_score_manifest.csv",
        "tables/source_inner_top3_selection_manifest.csv",
        "tables/source_drop_frequency_summary.csv",
        "tables/drop_one_subset_target_utility_matrix.csv",
        "tables/centerwise_delta_summary.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_source_inner_transfer_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_source_inner_transfer_downstream_matrix.csv", newline="")))
    transfer = list(csv.DictReader(open(root / "tables" / "source_inner_transfer_matrix.csv", newline="")))
    subsets = list(csv.DictReader(open(root / "tables" / "source_inner_subset_score_manifest.csv", newline="")))
    selections = list(csv.DictReader(open(root / "tables" / "source_inner_top3_selection_manifest.csv", newline="")))
    drops = list(csv.DictReader(open(root / "tables" / "source_drop_frequency_summary.csv", newline="")))
    target_subsets = list(csv.DictReader(open(root / "tables" / "drop_one_subset_target_utility_matrix.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_source_inner_transfer_summary.csv", newline="")))
    negative = list(csv.DictReader(open(root / "tables" / "negative_control_summary.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["heldout_target_rows_used_for_source_inner_scoring"] is False
    assert protocol["source_inner_uses_non_target_source_eval_rows"] is True
    assert protocol["method_comparison_uses_method_invariant_generation_seed"] is True
    assert any(row["prior_method"] == PRIMARY_SOURCE_INNER_TRANSFER_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_equal_all4_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_reliability_top3_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "exhaustive_drop_one_top3_mean_reference" for row in matrix)
    assert any(row["prior_method"] == "exhaustive_drop_one_top3_oracle_reference" for row in matrix)
    assert transfer and all(row["source_expert"] != row["pseudo_target_source"] for row in transfer)
    assert subsets and all(row["heldout_center"] not in row["selected_sources"].split("|") for row in subsets)
    assert selections and all(row["heldout_center"] not in row["selected_sources"].split("|") for row in selections)
    assert drops and {"dropped_source_target_utility_rank", "dropped_source_source_inner_score_rank"} <= set(drops[0])
    assert all(row["dropped_source_target_utility_rank"] not in {"", "nan"} for row in drops)
    assert target_subsets and len([row for row in target_subsets if row["heldout_center"] == "0"]) == 4
    equal_all4 = {
        (row["experiment_seed"], row["heldout_center"], row["replicate_seed"]): row
        for row in matrix
        if row["prior_method"] == "decentralized_equal_all4_geom_reference" and row["status"] == "ok"
    }
    top4 = {
        (row["experiment_seed"], row["heldout_center"], row["replicate_seed"]): row
        for row in matrix
        if row["prior_method"] == "decentralized_source_inner_transfer_top4_geom_diagnostic" and row["status"] == "ok"
    }
    assert equal_all4 and top4
    for key, equal_row in equal_all4.items():
        top4_row = top4[key]
        assert top4_row["generated_features_hash"] == equal_row["generated_features_hash"]
        assert float(top4_row["bacc"]) == float(equal_row["bacc"])
    assert "seed_equal_mean_bacc" in summary[0]
    assert "replicate_row_mean_bacc" in summary[0]
    assert "shuffled_score_control_gap" in negative[0]
    assert "Top-3 over four candidates is drop-one source selection" in report


def test_decentralized_source_inner_transfer_top3_rejects_support_usage(tmp_path: Path) -> None:
    payload = _tiny_decentralized_source_inner_transfer_top3_gmm_payload(tmp_path)
    payload["run_matrix"]["support_size"] = 8

    with pytest.raises(Exception, match="must not configure or consume target support"):
        parse_decentralized_source_inner_transfer_top3_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_source_inner_transfer_top3_rejects_invalid_top_k(tmp_path: Path) -> None:
    payload = _tiny_decentralized_source_inner_transfer_top3_gmm_payload(tmp_path)
    payload["generation"]["top_k_sources"] = 2

    with pytest.raises(Exception, match="top_k_sources must be locked to 3"):
        parse_decentralized_source_inner_transfer_top3_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_support_nelbo_reliability_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_support_nelbo_reliability_gmm_payload(tmp_path)
    cfg = parse_decentralized_support_nelbo_reliability_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_support_nelbo_reliability_gmm_prior(cfg)

    expected = [
        "tables/decentralized_support_nelbo_reliability_downstream_matrix.csv",
        "tables/decentralized_support_nelbo_reliability_gap_summary.csv",
        "tables/decentralized_support_nelbo_reliability_summary.csv",
        "tables/support_eval_split_manifest.csv",
        "tables/support_nelbo_scores.csv",
        "tables/support_nelbo_weight_manifest.csv",
        "tables/combined_weight_manifest.csv",
        "tables/support_nelbo_alignment_matrix.csv",
        "tables/source_reliability_manifest.csv",
        "tables/centerwise_delta_summary.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/negative_control_summary.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_support_nelbo_reliability_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_support_nelbo_reliability_downstream_matrix.csv", newline="")))
    splits = list(csv.DictReader(open(root / "tables" / "support_eval_split_manifest.csv", newline="")))
    support_weights = list(csv.DictReader(open(root / "tables" / "support_nelbo_weight_manifest.csv", newline="")))
    combined_weights = list(csv.DictReader(open(root / "tables" / "combined_weight_manifest.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_support_nelbo_reliability_summary.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["support_nelbo_uses_unlabeled_target_support_only"] is True
    assert protocol["decision_baselines_recomputed_on_support_excluded_eval_subset"] is True
    assert any(row["prior_method"] == PRIMARY_SUPPORT_RELIABILITY_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_source_reliability_weighted_geom_support_eval_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_equal_geom_support_eval_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_exported_adaptive_k_support_nelbo_uncalibrated_weighted_geom_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support_nelbo_tau05_x_reliability_geom_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support_nelbo_support_size64_x_reliability_geom_diagnostic" for row in matrix)
    assert splits and all(row["support_labels_used"] == "0" for row in splits)
    assert support_weights and combined_weights
    assert all(row["heldout_center"] != row["source_center"] for row in combined_weights)
    for key in {
        (row["experiment_seed"], row["heldout_center"], row["replicate_seed"], row["support_size"])
        for row in combined_weights
    }:
        subset = [
            row for row in combined_weights
            if (row["experiment_seed"], row["heldout_center"], row["replicate_seed"], row["support_size"]) == key
        ]
        assert abs(sum(float(row["combined_weight"]) for row in subset) - 1.0) < 1.0e-6
        assert sum(int(row["synthetic_per_class_budget"]) for row in subset) == 128
        assert all(int(row["synthetic_per_class_budget"]) >= 8 for row in subset)
    assert "spearman_support_nelbo_vs_downstream_utility" in summary[0]
    assert "support_size_diagnostic_num_valid_cells_json" in summary[0]
    assert "strongest_negative_control_method" in summary[0]
    assert "shuffled_support_control_gap" in summary[0]
    assert "target-conditioned support-NELBO compatibility-weighted composition" in report
    assert "Strongest negative control" in report


def test_decentralized_support_nelbo_reliability_gmm_prior_rejects_unaligned_support_seeds(tmp_path: Path) -> None:
    payload = _tiny_decentralized_support_nelbo_reliability_gmm_payload(tmp_path)
    payload["run_matrix"]["support_seeds"] = [18]

    with pytest.raises(Exception, match="support_seeds == replicate_seeds"):
        parse_decentralized_support_nelbo_reliability_gmm_prior_config(payload, base_dir=tmp_path)


def test_decentralized_support8_top3_tau05_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    payload = _tiny_decentralized_support8_top3_tau05_gmm_payload(tmp_path)
    cfg = parse_decentralized_support8_top3_tau05_gmm_prior_config(payload, base_dir=tmp_path)
    _write_tiny_cache(cfg.feature_cache_root, seed=42)

    root = run_decentralized_support8_top3_tau05_gmm_prior(cfg)

    expected = [
        "tables/decentralized_support8_top3_tau05_downstream_matrix.csv",
        "tables/decentralized_support8_top3_tau05_gap_summary.csv",
        "tables/decentralized_support8_top3_tau05_summary.csv",
        "tables/support_eval_split_manifest.csv",
        "tables/support_nelbo_scores.csv",
        "tables/support_nelbo_weight_manifest.csv",
        "tables/combined_weight_manifest.csv",
        "tables/top3_selection_stability.csv",
        "tables/support_nelbo_alignment_matrix.csv",
        "tables/source_reliability_manifest.csv",
        "tables/centerwise_delta_summary.csv",
        "tables/late_aggregation_matrix.csv",
        "tables/real_feature_reference_matrix.csv",
        "tables/negative_control_summary.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_source_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/decentralized_support8_top3_tau05_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "decentralized_support8_top3_tau05_downstream_matrix.csv", newline="")))
    splits = list(csv.DictReader(open(root / "tables" / "support_eval_split_manifest.csv", newline="")))
    combined_weights = list(csv.DictReader(open(root / "tables" / "combined_weight_manifest.csv", newline="")))
    stability = list(csv.DictReader(open(root / "tables" / "top3_selection_stability.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "decentralized_support8_top3_tau05_summary.csv", newline="")))
    negative = list(csv.DictReader(open(root / "tables" / "negative_control_summary.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert protocol["support_size"] == 8
    assert protocol["top_k_sources"] == 3
    assert protocol["support_nelbo_tau"] == 0.5
    assert protocol["support_nelbo_uses_unlabeled_target_support_only"] is True
    assert protocol["decision_baselines_recomputed_on_support8_excluded_eval_subset"] is True
    assert any(row["prior_method"] == PRIMARY_SUPPORT8_TOP3_TAU05_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support8_d1_2_reliability_all4_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support8_equal_all4_geom_reference" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support8_top3_tau05_support_nelbo_only_equal_budget_geom" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support8_shuffled_support_top3_tau05_control" for row in matrix)
    assert any(row["prior_method"] == "decentralized_support32_d1_3_primary_context" for row in matrix)
    assert splits and all(row["support_labels_used"] == "0" for row in splits)
    assert stability and "top3_selection_jaccard_across_support_seeds" in stability[0]
    assert combined_weights and all(row["heldout_center"] != row["source_center"] for row in combined_weights)
    for key in {
        (row["experiment_seed"], row["heldout_center"], row["replicate_seed"], row["support_size"])
        for row in combined_weights
    }:
        subset = [
            row for row in combined_weights
            if (row["experiment_seed"], row["heldout_center"], row["replicate_seed"], row["support_size"]) == key
        ]
        assert len(subset) == 3
        assert abs(sum(float(row["combined_weight"]) for row in subset) - 1.0) < 1.0e-6
        assert sum(int(row["synthetic_per_class_budget"]) for row in subset) == 128
        assert all(int(row["synthetic_per_class_budget"]) >= 8 for row in subset)
    assert "strongest_negative_control_gap" in summary[0]
    assert "shuffled_support_control_gap" in summary[0]
    assert "mean_top3_selection_jaccard" in summary[0]
    assert negative and negative[0]["strongest_negative_control_method"]
    assert "Support-size-8/top-3/tau-0.5 was predeclared" in report
    assert "not metadata routing" in report


def test_decentralized_support8_top3_tau05_gmm_prior_rejects_invalid_tau(tmp_path: Path) -> None:
    payload = _tiny_decentralized_support8_top3_tau05_gmm_payload(tmp_path)
    payload["support8_top3_tau05_gmm_prior"]["support_nelbo_tau"] = 1.0

    with pytest.raises(Exception, match="support_nelbo_tau must be locked to 0.5"):
        parse_decentralized_support8_top3_tau05_gmm_prior_config(payload, base_dir=tmp_path)


def test_source_union_k24_gmm_prior_tiny_cache_writes_expected_artifacts(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42)
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    source_union_gmm_cfg = _tiny_source_union_gmm_config(tmp_path, repair_root, sampling_root, prior_root, cov_root)
    source_union_gmm_root = run_source_union_gmm_prior(source_union_gmm_cfg)
    balanced_cfg = _tiny_source_union_balanced_gmm_config(
        tmp_path,
        repair_root,
        sampling_root,
        prior_root,
        cov_root,
        source_union_gmm_root,
    )
    balanced_root = run_source_union_balanced_gmm_prior(balanced_cfg)
    cfg = _tiny_source_union_k24_gmm_config(
        tmp_path,
        repair_root,
        sampling_root,
        prior_root,
        cov_root,
        source_union_gmm_root,
        balanced_root,
    )

    root = run_source_union_k24_gmm_prior(cfg)

    expected = [
        "tables/k24_gmm_downstream_matrix.csv",
        "tables/k24_gmm_gap_summary.csv",
        "tables/source_union_k24_gmm_summary.csv",
        "tables/gmm_component_diagnostics.csv",
        "tables/generated_component_coverage_audit.csv",
        "tables/weak_center_audit.csv",
        "tables/nearest_neighbor_memorization_audit.csv",
        "tables/negative_control_summary.csv",
        "tables/diagnostic_method_summary.csv",
        "manifests/protocol_manifest.json",
        "manifests/k24_gmm_prior_model_manifest.csv",
        "reports/leakage_report.json",
        "reports/decision_summary.md",
        "run_config_resolved.yaml",
    ]
    for rel in expected:
        assert (root / rel).exists()

    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    matrix = list(csv.DictReader(open(root / "tables" / "k24_gmm_downstream_matrix.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "source_union_k24_gmm_summary.csv", newline="")))
    coverage = list(csv.DictReader(open(root / "tables" / "generated_component_coverage_audit.csv", newline="")))
    nn = list(csv.DictReader(open(root / "tables" / "nearest_neighbor_memorization_audit.csv", newline="")))
    report = (root / "reports" / "decision_summary.md").read_text(encoding="utf-8")

    assert leakage["status"] == "PASS"
    assert any(row["prior_method"] == PRIMARY_K24_GMM_METHOD and row["selection_source"] == "primary" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k16_prior_sample_reference" for row in matrix)
    assert any(row["prior_method"] == "source_union_center_balanced_cc_diag_gmm_k16_prior_sample_reference" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k32_prior_sample_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k24_budget256_diagnostic" for row in matrix)
    assert any(row["prior_method"] == "source_union_cc_diag_gmm_k24_shuffled_label_control_diagnostic" for row in matrix)
    assert all(
        row["expert_pool_type"] == "source_union_excluding_target"
        for row in matrix
        if row["prior_method"] == PRIMARY_K24_GMM_METHOD
    )
    assert all(row["generated_features_hash"] for row in matrix if row["status"] == "ok" and row["fit_strategy"] != "reference")
    assert all(row["prediction_hash"] for row in matrix if row["status"] == "ok" and row["fit_strategy"] != "reference")
    assert "paired_delta_vs_center_balanced_k16_ci95" in summary[0]
    assert coverage and "component_mass_covered_by_generated_samples" in coverage[0]
    assert nn and {row["audit_interpretation"] for row in nn} == {"memorization_proximity_audit_only_not_formal_privacy"}
    assert "It does not evaluate metadata routing." in report
    assert "It does not evaluate support-NELBO routing." in report
    assert "It does not evaluate decentralized per-source expert selection." in report
    assert "It does not provide formal differential privacy." in report


def test_source_union_k24_gmm_prior_config_rejects_noncanonical_primary(tmp_path: Path) -> None:
    payload = _tiny_source_union_k24_gmm_payload(
        tmp_path,
        tmp_path / "repair",
        tmp_path / "sampling",
        tmp_path / "prior",
        tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1",
        tmp_path / "virchow2_cvae_source_union_gmm_prior_v1",
        tmp_path / "virchow2_cvae_source_union_center_balanced_gmm_prior_v1",
    )
    payload["k24_gmm_prior"]["primary_method"] = "source_union_cc_diag_gmm_k32_prior_sample_diagnostic"

    with pytest.raises(Exception, match="primary_method"):
        parse_source_union_k24_gmm_prior_config(payload, base_dir=tmp_path)


def test_source_union_k24_gmm_prior_mono_class_target_eval_is_target_eval_insufficient(tmp_path: Path) -> None:
    repair_cfg = _tiny_repair_config(tmp_path)
    _write_tiny_cache(repair_cfg.feature_cache_root, seed=42, mono_test_centers={"1"})
    repair_root = run_preservation_repair(repair_cfg)
    sampling_cfg = _tiny_sampling_config(tmp_path, repair_root)
    sampling_root = run_preservation_sampling(sampling_cfg)
    prior_cfg = _tiny_prior_calibration_config(tmp_path, repair_root, sampling_root)
    prior_root = run_prior_calibration(prior_cfg)
    cov_cfg = _tiny_covariance_prior_config(tmp_path, repair_root, sampling_root, prior_root)
    cov_root = run_covariance_prior_confirmation(cov_cfg)
    source_union_gmm_cfg = _tiny_source_union_gmm_config(tmp_path, repair_root, sampling_root, prior_root, cov_root)
    source_union_gmm_root = run_source_union_gmm_prior(source_union_gmm_cfg)
    balanced_cfg = _tiny_source_union_balanced_gmm_config(
        tmp_path,
        repair_root,
        sampling_root,
        prior_root,
        cov_root,
        source_union_gmm_root,
    )
    balanced_root = run_source_union_balanced_gmm_prior(balanced_cfg)
    cfg = _tiny_source_union_k24_gmm_config(
        tmp_path,
        repair_root,
        sampling_root,
        prior_root,
        cov_root,
        source_union_gmm_root,
        balanced_root,
    )

    root = run_source_union_k24_gmm_prior(cfg)
    matrix = list(csv.DictReader(open(root / "tables" / "k24_gmm_downstream_matrix.csv", newline="")))
    summary = list(csv.DictReader(open(root / "tables" / "source_union_k24_gmm_summary.csv", newline="")))

    assert summary[0]["primary_verdict"] != "GMM_FIT_INELIGIBLE"
    assert any(
        row["prior_method"] == PRIMARY_K24_GMM_METHOD
        and row["heldout_center"] == "1"
        and row["status"] == "ineligible"
        and row["error_message"] == "mono_class_target_eval"
        for row in matrix
    )


def _tiny_config(tmp_path: Path):
    return parse_config(
        {
            "experiment": {
                "name": "target_support32_calibrated_unlabeled_marginal_nelbo_top2_geom_virchow2_cvae_pca256_v1",
                "artifact_root": str(tmp_path / "artifacts"),
                "primary_method": "support_nelbo_top2_geom",
            },
            "inputs": {
                "feature_cache_root": str(tmp_path / "cache" / "virchow2"),
                "backbone": "virchow2",
            },
            "run_matrix": {
                "experiment_seeds": [42],
                "heldout_centers": ["0", "1", "2", "3", "4"],
                "support_size": 32,
                "support_seeds": [17],
                "generation_seeds": [17],
                "classifier_seeds": [17],
                "candidate_count_per_cell": 4,
            },
            "feature_frame": {"pca_dim": 256, "fit_scope": "per_expert_source_train"},
            "model": {
                "hidden_dim": 512,
                "latent_dim": 64,
                "num_hidden_layers": 2,
                "train_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "class_conditioning": "encoder_decoder_one_hot",
            },
            "routing": {
                "primary_score": "calibrated_marginal_support_nelbo",
                "support_sampler": "random_unlabeled_sample_ids",
            },
            "generation": {
                "mode": "class_stratified_reference_posterior",
                "synthetic_per_class_total": 128,
            },
            "downstream": {
                "classifier": "sklearn_logistic_regression",
                "aggregation": "geometric_probability_pooling",
                "eps": 1e-12,
            },
        },
        base_dir=tmp_path,
    )


def _tiny_preservation_config(tmp_path: Path):
    return parse_preservation_config(
        {
            "experiment": {
                "name": "virchow2_cvae_preservation_diagnosis_v1",
                "artifact_root": str(tmp_path / "preservation_artifacts"),
            },
            "inputs": {
                "feature_cache_root": str(tmp_path / "preservation_cache" / "virchow2"),
                "backbone": "virchow2",
            },
            "run_matrix": {
                "experiment_seeds": [42],
                "heldout_centers": ["0", "1", "2", "3", "4"],
                "replicate_seeds": [17],
                "candidate_count_per_cell": 4,
            },
            "feature_frame": {"pca_dim": 256, "fit_scope": "per_expert_source_train"},
            "model": {
                "hidden_dim": 512,
                "latent_dim": 64,
                "num_hidden_layers": 2,
                "train_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "class_conditioning": "encoder_decoder_one_hot",
            },
            "generation": {
                "synthetic_per_class_total": 128,
                "class_prior_for_generation": "uniform",
            },
            "classifier": {
                "type": "sklearn_logistic_regression",
                "solver": "lbfgs",
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "classifier_seed": None,
            },
        },
        base_dir=tmp_path,
    )


def _tiny_repair_config(tmp_path: Path):
    variants = [
        ("current_pca200_beta1_reference", "per_source", 256, 64, 1.0, 0.0, "legacy_sum_mse_kl", "reference_only", "adam"),
        ("pca64_beta001", "per_source", 64, 16, 0.001, 0.0, "normalized_repair", "primary", "adamw"),
        ("pca128_beta001", "per_source", 128, 32, 0.001, 0.0, "normalized_repair", "diagnostic_only", "adamw"),
        ("pca64_beta001_probe025", "per_source", 64, 16, 0.001, 0.25, "normalized_repair", "diagnostic_only", "adamw"),
        ("pca128_beta001_probe025", "per_source", 128, 32, 0.001, 0.25, "normalized_repair", "diagnostic_only", "adamw"),
        (
            "source_union_pca64_beta001_diagnostic",
            "source_union_excluding_target",
            64,
            16,
            0.001,
            0.0,
            "normalized_repair",
            "diagnostic_only",
            "adamw",
        ),
        (
            "source_union_pca64_beta001_probe025_diagnostic",
            "source_union_excluding_target",
            64,
            16,
            0.001,
            0.25,
            "normalized_repair",
            "diagnostic_only",
            "adamw",
        ),
    ]
    return parse_repair_config(
        {
            "experiment": {
                "name": "virchow2_cvae_preservation_repair_v1",
                "artifact_root": str(tmp_path / "repair_artifacts"),
                "primary_variant": "pca64_beta001",
                "min_decision_rows": 1,
            },
            "inputs": {
                "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
                "backbone": "virchow2",
            },
            "run_matrix": {
                "experiment_seeds": [42],
                "heldout_centers": ["0", "1", "2"],
                "replicate_seeds": [17],
            },
            "generation": {
                "synthetic_per_class_total": 128,
            },
            "classifier": {
                "type": "sklearn_logistic_regression",
                "solver": "lbfgs",
                "C": 1.0,
                "max_iter": 2000,
                "class_weight": "balanced",
                "classifier_seed": None,
            },
            "source_probe": {
                "type": "torch_linear_classifier",
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "epochs": 1,
                "batch_size": 16,
                "class_weight": "balanced",
                "early_stopping": False,
            },
            "variants": [
                {
                    "variant_id": variant_id,
                    "expert_pool_type": pool,
                    "requested_pca_dim": pca,
                    "hidden_dim": 512,
                    "latent_dim": latent,
                    "num_hidden_layers": 2,
                    "train_epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "optimizer": optimizer,
                    "weight_decay": 0.0 if optimizer == "adam" else 0.0001,
                    "gradient_clip_norm": 5.0,
                    "beta_final": beta,
                    "kl_warmup_epochs": 1 if variant_id == "current_pca200_beta1_reference" else 2,
                    "probe_ce_weight": probe,
                    "loss_style": loss_style,
                    "selection_source": selection,
                }
                for variant_id, pool, pca, latent, beta, probe, loss_style, selection, optimizer in variants
            ],
        },
        base_dir=tmp_path,
    )


def _tiny_sampling_config(tmp_path: Path, repair_root: Path):
    return parse_sampling_config(_tiny_sampling_payload(tmp_path, repair_root), base_dir=tmp_path)


def _tiny_sampling_payload(tmp_path: Path, repair_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_pca64_sampling_continuation_v1",
            "artifact_root": str(tmp_path / "sampling_artifacts"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 1,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "sampling": {
            "posterior_temperatures_primary": [1.0],
            "posterior_temperatures_diagnostic": [0.25, 0.5],
            "prior_scales_primary": [1.0],
            "prior_scales_diagnostic": [0.25, 0.5],
            "empirical_posterior_temperature": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_prior_calibration_config(tmp_path: Path, repair_root: Path, sampling_root: Path):
    return parse_prior_calibration_config(
        _tiny_prior_calibration_payload(tmp_path, repair_root, sampling_root),
        base_dir=tmp_path,
    )


def _tiny_prior_calibration_payload(tmp_path: Path, repair_root: Path, sampling_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_latent_prior_calibration_v1",
            "artifact_root": str(tmp_path / "prior_calibration_artifacts"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 9,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "prior_calibration": {
            "primary_method": "cvae_cc_diag_gaussian_prior_sample",
            "min_prior_fit_records_per_class": 8,
            "variance_floor": 1.0e-4,
            "variance_ddof": 0,
            "shrinkage_alphas": [0.25, 0.5],
            "standard_prior_repro_abs_tol_bacc": 1.0,
            "full_cov_min_records_per_class": 32,
            "full_cov_shrinkage_alpha": 0.1,
            "full_cov_eigenvalue_floor": 1.0e-4,
            "full_cov_fallback_if_singular": "diag",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_covariance_prior_config(tmp_path: Path, repair_root: Path, sampling_root: Path, prior_root: Path):
    return parse_covariance_prior_config(
        _tiny_covariance_prior_payload(tmp_path, repair_root, sampling_root, prior_root),
        base_dir=tmp_path,
    )


def _tiny_covariance_prior_payload(tmp_path: Path, repair_root: Path, sampling_root: Path, prior_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_covariance_prior_confirmation_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_covariance_prior_confirmation_v1"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 9,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "covariance_prior": {
            "primary_method": "cvae_cc_cov_shrinkage_prior_sample",
            "covariance_shrinkage_alpha": 0.10,
            "covariance_eigenvalue_floor": 1.0e-4,
            "full_cov_min_records_per_class": 32,
            "fallback_if_under_ranked": "diag",
            "standard_prior_repro_abs_tol_bacc": 1.0,
            "diag_prior_repro_abs_tol_bacc": 1.0,
            "full_cov_diagnostic_repro_abs_tol_bacc": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_covariance_viability_config(tmp_path: Path, covariance_root: Path):
    return parse_covariance_viability_config(
        _tiny_covariance_viability_payload(tmp_path, covariance_root),
        base_dir=tmp_path,
    )


def _tiny_covariance_viability_payload(tmp_path: Path, covariance_root: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_covariance_prior_viability_audit_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_covariance_prior_viability_audit_v1"),
        },
        "inputs": {
            "covariance_confirmation_artifact_root": str(covariance_root),
        },
        "viability_audit": {
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "min_viable_cells": 30,
            "min_viable_cells_per_center": 3,
            "min_viable_seeds_per_center": 2,
            "high_real_threshold": 0.80,
            "viable_real_threshold": 0.75,
            "borderline_real_threshold": 0.65,
            "global_center_equal_mean_bacc_min": 0.85,
            "mean_clipped_preservation_gap_max": 0.08,
            "mean_preservation_ratio_min": 0.92,
            "seed_std_max": 0.07,
            "delta_bacc_vs_standard_prior_min": 0.05,
            "delta_bacc_vs_diag_prior_min": 0.03,
            "covariance_beats_diag_cell_fraction_min": 0.70,
            "covariance_beats_diag_center_fraction_min": 0.75,
            "worst_delta_vs_diag_prior_min": -0.05,
            "min_cell_bacc_min": 0.60,
            "min_center_mean_bacc_min": 0.75,
        },
    }


def _tiny_covariance_shrinkage_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    viability_root: Path,
):
    return parse_covariance_shrinkage_config(
        _tiny_covariance_shrinkage_payload(tmp_path, repair_root, sampling_root, prior_root, covariance_root, viability_root),
        base_dir=tmp_path,
    )


def _tiny_covariance_shrinkage_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    viability_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_covariance_shrinkage_stability_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_covariance_shrinkage_stability_v1"),
            "primary_variant": "pca64_beta001",
            "min_decision_cells": 9,
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "covariance_viability_artifact_root": str(viability_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "covariance_shrinkage": {
            "primary_method": "cvae_cc_cov_diag_shrinkage075_prior_sample",
            "primary_covariance_shrinkage_alpha": 0.75,
            "diagnostic_covariance_shrinkage_alphas": [0.50, 0.90],
            "reference_covariance_shrinkage_alpha": 0.10,
            "diagonal_reference_alpha": 1.00,
            "covariance_eigenvalue_floor": 1.0e-4,
            "full_cov_min_records_per_class": 32,
            "fallback_if_under_ranked": "diag",
            "standard_prior_repro_abs_tol_bacc": 1.0,
            "diag_prior_repro_abs_tol_bacc": 1.0,
            "alpha010_repro_abs_tol_bacc": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_union_gmm_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
):
    return parse_source_union_gmm_prior_config(
        _tiny_source_union_gmm_payload(tmp_path, repair_root, sampling_root, prior_root, covariance_root),
        base_dir=tmp_path,
    )


def _tiny_source_union_gmm_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_union_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_union_gmm_prior_v1"),
            "primary_variant": "source_union_pca64_beta001_diagnostic",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "gmm_prior": {
            "primary_method": "source_union_cc_diag_gmm_k8_prior_sample",
            "gmm_components": 8,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 2,
            "gmm_max_iter": 200,
            "gmm_weight_floor": 0.01,
            "min_class_train_count": 8,
            "min_effective_gmm_components": 1,
            "posterior_noise_scale": 0.0,
            "diagnostic_gmm_components": [4, 16],
            "diagnostic_posterior_noise_scales": [0.25],
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_union_balanced_gmm_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
):
    return parse_source_union_balanced_gmm_prior_config(
        _tiny_source_union_balanced_gmm_payload(
            tmp_path,
            repair_root,
            sampling_root,
            prior_root,
            covariance_root,
            source_union_gmm_root,
        ),
        base_dir=tmp_path,
    )


def _tiny_source_union_balanced_gmm_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_union_center_balanced_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_union_center_balanced_gmm_prior_v1"),
            "primary_variant": "source_union_pca64_beta001_diagnostic",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "source_union_gmm_artifact_root": str(source_union_gmm_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "balanced_gmm_prior": {
            "primary_method": "source_union_center_balanced_cc_diag_gmm_k16_prior_sample",
            "gmm_components": 16,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 2,
            "gmm_max_iter": 200,
            "gmm_weight_floor": 0.005,
            "min_source_center_class_count": 8,
            "min_effective_gmm_components": 1,
            "balanced_fit_samples_per_center_class": 8,
            "max_center_class_replacement_rate": 1.0,
            "mean_center_class_replacement_rate": 1.0,
            "posterior_noise_scale": 0.0,
            "diagnostic_gmm_components": [8, 24],
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_k16_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_k16_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_k16_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "decentralized_k16_prior": {
            "primary_method": "decentralized_exported_k4x4_cc_diag_gmm_k16_late_geom",
            "local_gmm_components_per_source_class": 4,
            "composed_components_per_class": 16,
            "source_weighting": "equal_source_mass",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_count_for_k4": 8,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
        },
        "support_nelbo_diagnostic": {
            "enabled": False,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_adaptive_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_adaptive_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_adaptive_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "strict_d1_artifact_root": str(tmp_path / "missing_decentralized_k16"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "adaptive_gmm_prior": {
            "primary_method": "decentralized_exported_adaptive_k_cc_diag_gmm_late_geom",
            "bic_method": "decentralized_exported_bic_selected_cc_diag_gmm_late_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "equal_source_mass",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_reliability_weighted_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "reliability_weighted_gmm_prior": {
            "primary_method": "decentralized_exported_adaptive_k_source_reliability_weighted_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_local_reliability",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "weighted_geometric",
            "reliability_floor_score": 0.05,
            "softmax_tau": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_component_union_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_component_union_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_component_union_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_2_artifact_root": str(tmp_path / "missing_d1_2"),
            "source_union_gmm_artifact_root": str(tmp_path / "missing_source_union_gmm"),
            "balanced_gmm_artifact_root": str(tmp_path / "missing_balanced_gmm"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "budget_diagnostic_per_class_total": 256,
            "min_per_source_per_class": 8,
        },
        "component_union_prior": {
            "primary_method": "decentralized_component_union_uniform_gmm",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "uniform_source_component_union",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "variance_ceiling_multiplier": 16.0,
            "primary_pooling": "pooled_raw_logistic",
            "reliability_floor_score": 0.05,
            "shrink_lambdas": [0.25, 0.5],
            "prototype_candidate_counts_per_source_class": [4, 3, 2, 1],
            "prototype_min_samples_per_component": 12,
            "prototype_variance_floor": 1.0e-5,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_paired_dense_all4_reliability_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_paired_dense_all4_reliability_confirmation_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_paired_dense_all4_reliability_confirmation_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_2_artifact_root": str(tmp_path / "missing_d1_2_context"),
            "d1_4_artifact_root": str(tmp_path / "missing_d1_4_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "paired_dense_all4_reliability": {
            "primary_method": "paired_reliability_all4_shrink050_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "heldout_excluded_source_local_reliability_dense_all4",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 5,
            "gmm_max_iter": 500,
            "min_component_weight": 0.02,
            "variance_floor": 1.0e-5,
            "reliability_floor_score": 0.05,
            "reliability_epsilon": 1.0e-8,
            "shrinkage_values": [0.25, 0.50],
            "primary_pooling": "weighted_geometric",
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_reliability_top3_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_3_1_artifact_root": str(tmp_path / "missing_d1_3_1_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
            "top_k_sources": 3,
        },
        "reliability_top3_gmm_prior": {
            "primary_method": "decentralized_reliability_top3_geom_confirmation",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_local_reliability_top3",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
            "reliability_floor_score": 0.05,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_source_inner_transfer_top3_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
            "top_k_sources": 3,
        },
        "source_inner_transfer_top3_gmm_prior": {
            "primary_method": "decentralized_source_inner_transfer_top3_geom_confirmation",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "source_inner_transfer_top3",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "geometric",
            "reliability_floor_score": 0.05,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_support_nelbo_reliability_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "support_seeds": [17],
            "support_size": 32,
            "support_size_diagnostics": [8, 16, 64],
            "align_support_and_generation_seed": True,
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "support_nelbo_reliability_gmm_prior": {
            "primary_method": "decentralized_exported_adaptive_k_support_nelbo_x_reliability_weighted_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "support_nelbo_x_source_local_reliability",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "weighted_geometric",
            "reliability_floor_score": 0.05,
            "support_nelbo_tau": 1.0,
            "tau_diagnostics": [0.5, 2.0],
            "support_alpha": 1.0,
            "reliability_alpha": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_decentralized_support8_top3_tau05_gmm_payload(tmp_path: Path):
    return {
        "experiment": {
            "name": "virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1"),
            "primary_variant": "pca64_beta001",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(tmp_path / "virchow2_cvae_preservation_repair_v1"),
            "d1_3_artifact_root": str(tmp_path / "missing_d1_3_context"),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2", "3", "4"],
            "replicate_seeds": [17],
            "support_seeds": [17],
            "support_size": 8,
            "align_support_and_generation_seed": True,
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "min_per_source_per_class": 8,
        },
        "support8_top3_tau05_gmm_prior": {
            "primary_method": "decentralized_support8_top3_tau05_support_nelbo_x_reliability_geom",
            "candidate_components_per_source_class": [4, 3, 2, 1],
            "min_samples_per_component": 12,
            "source_weighting": "support_nelbo_x_source_local_reliability_top3",
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 1,
            "gmm_max_iter": 100,
            "min_component_weight": 0.001,
            "variance_floor": 1.0e-5,
            "primary_pooling": "weighted_geometric",
            "reliability_floor_score": 0.05,
            "support_nelbo_tau": 0.5,
            "top_k_sources": 3,
            "support_alpha": 1.0,
            "reliability_alpha": 1.0,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _tiny_source_union_k24_gmm_config(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
    balanced_gmm_root: Path,
):
    return parse_source_union_k24_gmm_prior_config(
        _tiny_source_union_k24_gmm_payload(
            tmp_path,
            repair_root,
            sampling_root,
            prior_root,
            covariance_root,
            source_union_gmm_root,
            balanced_gmm_root,
        ),
        base_dir=tmp_path,
    )


def _tiny_source_union_k24_gmm_payload(
    tmp_path: Path,
    repair_root: Path,
    sampling_root: Path,
    prior_root: Path,
    covariance_root: Path,
    source_union_gmm_root: Path,
    balanced_gmm_root: Path,
):
    return {
        "experiment": {
            "name": "virchow2_cvae_source_union_k24_gmm_prior_v1",
            "artifact_root": str(tmp_path / "virchow2_cvae_source_union_k24_gmm_prior_v1"),
            "primary_variant": "source_union_pca64_beta001_diagnostic",
        },
        "inputs": {
            "feature_cache_root": str(tmp_path / "repair_cache" / "virchow2"),
            "repair_artifact_root": str(repair_root),
            "sampling_artifact_root": str(sampling_root),
            "prior_calibration_artifact_root": str(prior_root),
            "covariance_confirmation_artifact_root": str(covariance_root),
            "source_union_gmm_artifact_root": str(source_union_gmm_root),
            "balanced_gmm_artifact_root": str(balanced_gmm_root),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "experiment_seeds": [42],
            "heldout_centers": ["0", "1", "2"],
            "replicate_seeds": [17],
        },
        "generation": {
            "synthetic_per_class_total": 128,
        },
        "k24_gmm_prior": {
            "primary_method": "source_union_cc_diag_gmm_k24_prior_sample",
            "gmm_components": 24,
            "gmm_covariance_type": "diag",
            "gmm_reg_covar": 1.0e-4,
            "gmm_n_init": 2,
            "gmm_max_iter": 200,
            "gmm_weight_floor": 0.005,
            "min_class_train_count": 24,
            "min_effective_gmm_components": 1,
            "min_train_count_per_effective_component": 1,
            "posterior_noise_scale": 0.0,
            "diagnostic_gmm_components": [20, 32],
            "budget256_synthetic_per_class_total": 256,
            "center_cap_samples_per_center_class": 8,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
    }


def _write_tiny_cache(root: Path, *, seed: int, mono_test_centers: set[str] | None = None) -> None:
    rng = np.random.default_rng(123)
    mono_test_centers = set(mono_test_centers or set())
    for split, per_class in (("train", 24), ("test", 26)):
        embeddings = []
        metadata = []
        for center_idx, center in enumerate(["0", "1", "2", "3", "4"]):
            for label in (0, 1):
                mean = np.array([center_idx * 0.5, label * 1.2, center_idx * 0.1, label * 0.3])
                for idx in range(per_class):
                    embeddings.append(mean + rng.normal(0.0, 0.05, size=4))
                    observed_label = 0 if split == "test" and center in mono_test_centers else label
                    metadata.append(
                        {
                            "sample_id": f"{split}_c{center}_y{observed_label}_{label}_{idx}",
                            "center": center,
                            "label": observed_label,
                            "split": split,
                        }
                    )
        path = root / f"seed{seed}" / "embeddings" / f"{split}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            embeddings=np.asarray(embeddings, dtype=float),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
