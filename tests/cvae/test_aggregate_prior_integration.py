from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.preservation.aggregate_prior_study.config import (
    load_aggregate_prior_study_config,
)
from midogpp_thesis.cvae.preservation.aggregate_prior_study.runner import (
    run_aggregate_prior_source_inner_study,
)
from midogpp_thesis.cvae.preservation.aggregate_prior_study.validation import (
    validate_aggregate_prior_study_bundle,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


CONFIG = Path(
    "experiments/midogpp/stages/20_cvae_preservation/configs/"
    "aggregate_posterior_mixture_geco_source_inner_v3.yaml"
)


def test_small_independent_source_bundle_round_trip(tmp_path: Path) -> None:
    manifest, cache = _write_frame(tmp_path)
    root = tmp_path / "artifact"
    production = load_aggregate_prior_study_config(CONFIG)
    config = replace(
        production,
        artifact_root=root,
        manifest_path=manifest,
        feature_cache_path=cache,
        device="cpu",
        heldout_centers=("0", "1", "2", "3"),
        training_seeds=(17,),
        generation_seeds=(17,),
        expected_feature_dim=6,
        pca_dim=6,
        latent_dim=3,
        hidden_dim=8,
        warmup_epochs=1,
        continuation_epochs=2,
        batch_size=16,
        kl_warmup_epochs=1,
        refit_interval_epochs=1,
        final_stabilization_epochs=1,
        minimum_component_rows=2,
        minimum_component_cases=1,
        generation_per_class=8,
        min_inner_wins=1,
    )
    (root / "provenance").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("fixture: true\n", encoding="utf-8")
    (root / "provenance/input_artifacts.json").write_text(
        json.dumps({"fixture": True}),
        encoding="utf-8",
    )

    output = run_aggregate_prior_source_inner_study(
        config,
        artifact_root=root,
    )
    validation = validate_aggregate_prior_study_bundle(
        output,
        expected_config=config,
    )
    assert validation["status"] == "PASS"
    assert validation["may_feed_model_recipe"] is False
    assert validation["n_checkpoints"] == 16
    assert validation["n_metric_rows"] == 192
    with (root / "tables/source_expert_metrics.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        metrics = list(csv.DictReader(handle))
    assert {
        role: sum(row["representation_role"] == role for row in metrics)
        for role in ("prior", "posterior")
    } == {"prior": 96, "posterior": 96}
    assert len({row["evaluation_key_hash"] for row in metrics}) == len(metrics)
    child = json.loads(
        (root / "reports/child_decisions/seed17/0.json").read_text(
            encoding="utf-8"
        )
    )
    assert "prior_posterior_gap_reduction" in child
    assert "posterior_mean_bacc_delta_vs_sf" in child
    publication = json.loads(
        (root / "reports/publication_state.json").read_text(encoding="utf-8")
    )
    assert publication["status"] == "NON_CONSUMABLE_STUDY_COMPLETE"
    assert publication["stage30_recipe_ready"] is False
    publication["stage30_recipe_ready"] = True
    (root / "reports/publication_state.json").write_text(
        json.dumps(publication),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="consumable"):
        validate_aggregate_prior_study_bundle(
            output,
            expected_config=config,
        )


def _write_frame(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(21)
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    for center_index, center in enumerate(("0", "1", "2", "3")):
        for label in (0, 1):
            for local in range(16):
                sample_id = f"{center}-{label}-{local}"
                mode = local // 8
                vector = (
                    rng.normal(scale=0.18, size=6)
                    + center_index * 0.15
                    + label * np.asarray([0.8, -0.5, 0.3, 0.2, -0.1, 0.4])
                    + mode * np.asarray([1.5, -1.0, 0.6, 0.0, 0.3, -0.2])
                )
                row = {
                    "sample_id": sample_id,
                    "case_id": f"case-{sample_id}",
                    "center": center,
                    "label": label,
                    "split": "train",
                    "image_path": f"slide-{center}-{label}-{local}.tif",
                }
                rows.append(row)
                metadata.append(dict(row))
                embeddings.append(vector.astype(np.float32))
    manifest = root / "midogpp_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cache = root / "virchow2_midogpp_train.npz"
    np.savez(
        cache,
        embeddings=np.stack(embeddings),
        metadata_json=json.dumps(metadata),
        feature_extractor_json=json.dumps(
            {"backbone_type": "virchow2", "dataset": "midogpp"}
        ),
    )
    return manifest, cache
