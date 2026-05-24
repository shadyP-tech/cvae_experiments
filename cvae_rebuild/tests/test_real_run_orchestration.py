import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("sklearn")

import numpy as np

from cvae_rebuild.config import parse_config
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


def _write_tiny_cache(root: Path, *, seed: int, mono_test_centers: set[str] | None = None) -> None:
    rng = np.random.default_rng(123)
    mono_test_centers = set(mono_test_centers or set())
    for split, per_class in (("train", 12), ("test", 26)):
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
