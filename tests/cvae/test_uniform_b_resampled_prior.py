from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from midogpp_thesis.cvae.preservation.cli import build_parser
from midogpp_thesis.cvae.preservation.uniform_b_resampled_prior.config import (
    load_uniform_b_resampled_prior_config,
)
from midogpp_thesis.cvae.preservation.uniform_b_resampled_prior.contracts import (
    P0,
    PQ,
    valid_outer_centers,
)
from midogpp_thesis.cvae.preservation.uniform_b_resampled_prior.execution import (
    SCORING_WORKERS_ENV,
    TRAINING_DEVICES_ENV,
    partition_panel_tasks,
    resolve_runtime_plan,
)
from midogpp_thesis.cvae.preservation.uniform_b_resampled_prior.runner import (
    run_uniform_b_resampled_prior_source_inner_study,
)
from midogpp_thesis.cvae.preservation.uniform_b_resampled_prior.validation import (
    validate_uniform_b_resampled_prior_bundle,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/20_cvae_preservation/configs"
    / "uniform_b_geco_posterior_resampled_prior_source_inner_v1.yaml"
)


def test_production_config_cli_and_score_counts_are_locked() -> None:
    config = load_uniform_b_resampled_prior_config(CONFIG)
    assert config.training_arm == "BG"
    assert config.priors == (P0, PQ)
    assert config.existing_checkpoint_input_allowed is False
    assert config.fresh_bg_training_required is True
    assert config.proposal_multiplier == 8
    parsed = build_parser().parse_args(
        [
            "source-inner-uniform-b-geco-posterior-resampled-prior",
            "--config",
            str(CONFIG),
        ]
    )
    assert parsed.surface == "source-inner-uniform-b-geco-posterior-resampled-prior"
    n = len(config.heldout_centers)
    unique = n * (n - 1) * len(config.training_seeds) * len(config.generation_seeds) * 2
    assert unique == 1296
    assert unique * (n - 2) == 9072


def test_outer_mapping_and_workstation_partition_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centers = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    assert valid_outer_centers(
        centers,
        source_center="0",
        inner_center="1",
    ) == ("2", "3", "5", "6", "7", "8", "9")
    tasks = tuple((center, seed) for center in centers for seed in (17, 42, 101))
    assert partition_panel_tasks(tasks, ("cuda:0", "cuda:1")) == {
        "cuda:0": tasks[0::2],
        "cuda:1": tasks[1::2],
    }
    config = replace(
        load_uniform_b_resampled_prior_config(CONFIG),
        heldout_centers=("0", "1", "2"),
        training_seeds=(17,),
        generation_seeds=(17,),
        device="cpu",
    )
    monkeypatch.setenv(SCORING_WORKERS_ENV, "8")
    monkeypatch.delenv(TRAINING_DEVICES_ENV, raising=False)
    before = config.contract_hash
    runtime = resolve_runtime_plan(config)
    assert runtime.scoring_workers == 8
    assert runtime.training_devices == ("cpu",)
    assert runtime.to_payload()["unique_score_reuse"] is True
    assert config.contract_hash == before


def test_tiny_runner_scores_unique_key_once_and_maps_without_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, cache, cache_hash = _tiny_uniform_b_cache(tmp_path)
    production = load_uniform_b_resampled_prior_config(CONFIG)
    config = replace(
        production,
        artifact_root=tmp_path / "artifact",
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_cache_hash=cache_hash,
        heldout_centers=("0", "1", "2"),
        training_seeds=(17,),
        generation_seeds=(17,),
        device="cpu",
        hidden_dim=16,
        latent_dim=4,
        batch_size=8,
        warmup_steps=1,
        total_steps=3,
        base_generation_per_class=2,
        proposal_multiplier=2,
        ratio_classifier_max_iter=200,
    )
    monkeypatch.setenv(SCORING_WORKERS_ENV, "2")
    monkeypatch.delenv(TRAINING_DEVICES_ENV, raising=False)
    root = run_uniform_b_resampled_prior_source_inner_study(config)
    report = validate_uniform_b_resampled_prior_bundle(root)
    assert report["status"] == "PASS"
    assert report["checkpoint_records"] == 3
    assert report["unique_score_rows"] == 12
    assert report["mapped_metric_rows"] == 12
    runtime = json.loads(
        (root / "reports/runtime_summary.json").read_text(encoding="utf-8")
    )
    assert runtime["classifier_fit_count"] == 12
    assert runtime["score_reuse_factor"] == 1.0
    checkpoint_index = json.loads(
        (root / "manifests/checkpoint_index.json").read_text(encoding="utf-8")
    )
    assert all(record["fresh_training"] for record in checkpoint_index["records"])
    assert all(not record["parent_checkpoint_used"] for record in checkpoint_index["records"])

    mapping_path = root / "manifests/score_reuse_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["records"][0]["mapped_outer_centers"] = []
    mapping_path.write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="exact legal outers"):
        validate_uniform_b_resampled_prior_bundle(root)


def _tiny_uniform_b_cache(tmp_path: Path) -> tuple[Path, Path, str]:
    data_root = tmp_path / "midogpp" / "virchow2"
    data_root.mkdir(parents=True)
    manifest = data_root / "manifest.csv"
    cache = data_root / "train.pt"
    rng = np.random.default_rng(31)
    embeddings = []
    metadata = []
    rows = []
    for center in ("0", "1", "2"):
        for case_index in range(16):
            label = case_index % 2
            for patch_index in range(8):
                sample_id = f"{center}-{case_index}-{patch_index}"
                case_id = f"{center}-case-{case_index}"
                image_path = f"/midogpp/{center}/{case_index}.png"
                row = {
                    "sample_id": sample_id,
                    "case_id": case_id,
                    "center": center,
                    "label": label,
                    "split": "train",
                    "image_path": image_path,
                }
                embeddings.append(
                    rng.normal(
                        loc=0.1 * int(center) + 0.2 * label,
                        scale=1.0,
                        size=3840,
                    )
                )
                metadata.append(dict(row))
                rows.append(row)
    rows.append(
        {
            "sample_id": "excluded-center-4-manifest-only",
            "case_id": "excluded-center-4-case",
            "center": "4",
            "label": 0,
            "split": "train",
            "image_path": "/midogpp/4/excluded.png",
        }
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "embeddings": torch.tensor(np.asarray(embeddings, dtype=np.float32)),
            "metadata": metadata,
            "feature_extractor": {
                "family": "Virchow2",
                "dataset": "MIDOG++",
                "variant": "uniform_b_resampled_prior_test",
            },
        },
        cache,
    )
    return manifest, cache, hashlib.sha256(cache.read_bytes()).hexdigest()
