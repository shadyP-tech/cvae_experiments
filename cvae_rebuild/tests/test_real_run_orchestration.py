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
