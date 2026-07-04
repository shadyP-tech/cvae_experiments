from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sail.config import PipelineConfig, load_config  # noqa: E402
from sail.features import write_npz_cache  # noqa: E402
from sail.pipeline import run_pipeline  # noqa: E402


def test_imports_and_config_loading() -> None:
    config = load_config(ROOT / "configs" / "camelyon17_virchow2_legacy" / "sail_virchow2.yaml")
    assert config.primary_backbone == "virchow2"
    assert config.primary_k_values == (3, 5, 10)
    assert config.class_weight_grid == ("none", "balanced")


def test_cli_help_works() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "sail.cli", "--help"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0
    assert "SAIL source-only aggregation" in result.stdout


def test_small_synthetic_evaluation_path_and_selection_firewall(tmp_path: Path) -> None:
    _write_synthetic_caches(tmp_path)
    config = PipelineConfig(
        candidate_centers=("0", "1", "2"),
        experiment_seeds=(1,),
        support_sizes=(0,),
        support_seeds=(1,),
        representations=("raw",),
        c_grid=(1.0,),
        class_weight_grid=("none",),
        primary_k_values=(1,),
        aggregation_rules=("geometric",),
        cache_root="cache",
        cache_path_template="{cache_root}/{backbone}/seed{seed}/embeddings/{split}.npz",
        artifacts_root="artifacts/run",
    )
    result = run_pipeline(config=config, repo_root=tmp_path)

    assert (tmp_path / "artifacts" / "run" / "tables" / "dense_aggregation_matrix.csv").exists()
    leakage = json.loads((tmp_path / "artifacts" / "run" / "reports" / "leakage_report.json").read_text())
    assert leakage["status"] == "PASS"
    assert leakage["target_eval_labels_for_deployable_selection"] is False
    assert leakage["target_eval_labels_for_scoring_only"] is True

    with (tmp_path / "artifacts" / "run" / "tables" / "source_k_selection_matrix.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["selection_used_target_labels"] for row in rows} == {"false"}
    assert any(row["selected_by_source_inner_lodo"] == "true" for row in rows)

    with (tmp_path / "artifacts" / "run" / "tables" / "dense_aggregation_matrix.csv").open(newline="") as handle:
        dense_rows = list(csv.DictReader(handle))
    assert dense_rows
    assert {row["selection_used_target_labels"] for row in dense_rows} == {"false"}
    assert {row["fit_used_target_center"] for row in dense_rows} == {"false"}
    assert all(float(row["bacc"]) >= 0.5 for row in dense_rows if row["status"] == "ok")
    assert result.output_paths["protocol_manifest"].exists()


def _write_synthetic_caches(root: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    for split, per_class in (("train", 8), ("test", 5)):
        embeddings = []
        metadata = []
        for center in ("0", "1", "2"):
            center_shift = float(center) * 0.15
            for cls in (0, 1):
                for idx in range(per_class):
                    base = np.array([float(cls) * 2.0 + center_shift, float(cls) * -1.0 + center_shift])
                    embeddings.append(base + rng.normal(0.0, 0.05, size=2))
                    metadata.append(
                        {
                            "sample_id": f"{split}_c{center}_y{cls}_{idx}",
                            "center": center,
                            "label": cls,
                            "split": split,
                            "image_path": f"dummy/{split}_c{center}_y{cls}_{idx}.png",
                        }
                    )
        write_npz_cache(
            root / "cache" / "virchow2" / "seed1" / "embeddings" / f"{split}.npz",
            np.asarray(embeddings, dtype=float),
            metadata,
        )
