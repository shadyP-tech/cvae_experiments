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
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.composition import (
    compose_generated_blocks,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.checkpoint_store import (
    TaskGeometryCheckpointStore,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.config import (
    load_uniform_b_task_geometry_config,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.contracts import (
    ARMS,
    COMPOSITION_MODES,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.generation import (
    GeneratedBlock,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.protocol import (
    candidate_pool_manifest,
    validate_candidate_pool,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.task_geometry import (
    fit_task_geometry,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.task_loss import (
    _TorchFold,
    task_terms,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.training import (
    train_source_panel,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.runner import (
    run_uniform_b_task_geometry_source_inner_study,
)
from midogpp_thesis.cvae.preservation.uniform_b_task_geometry.validation import (
    validate_uniform_b_task_geometry_bundle,
)
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/20_cvae_preservation/configs"
    / "uniform_b_geco_task_geometry_source_inner_v1.yaml"
)


def test_production_config_and_cli_are_locked() -> None:
    config = load_uniform_b_task_geometry_config(CONFIG)
    assert config.expected_feature_dim == 3840
    assert config.arms == ARMS
    assert config.composition_modes == COMPOSITION_MODES
    parsed = build_parser().parse_args(
        [
            "source-inner-uniform-b-geco-task-geometry",
            "--config",
            str(CONFIG),
        ]
    )
    assert parsed.surface == "source-inner-uniform-b-geco-task-geometry"


def test_candidate_pool_excludes_outer_and_inner() -> None:
    centers = ("0", "1", "2", "3")
    payload = candidate_pool_manifest(
        centers,
        outer_center="0",
        inner_center="1",
        base_per_class=8,
    )
    assert payload["legal_sources"] == ["2", "3"]
    validate_candidate_pool(payload, centers)


def test_all_composition_modes_have_exact_budget_controls() -> None:
    blocks = {
        source: _block(source, per_class=12) for source in ("2", "3", "4")
    }
    base = 4
    single = compose_generated_blocks(
        blocks,
        mode="single_base",
        base_per_class=base,
        shuffle_seed=1,
        selected_source="2",
    )
    matched = compose_generated_blocks(
        blocks,
        mode="single_budget_matched",
        base_per_class=base,
        shuffle_seed=1,
        selected_source="2",
    )
    equal = compose_generated_blocks(
        blocks,
        mode="union_equal_total",
        base_per_class=base,
        shuffle_seed=1,
    )
    expanded = compose_generated_blocks(
        blocks,
        mode="union_expanded",
        base_per_class=base,
        shuffle_seed=1,
    )
    assert len(single.labels) == 2 * base
    assert len(equal.labels) == 2 * base
    assert len(matched.labels) == 2 * 3 * base
    assert len(expanded.labels) == 2 * 3 * base
    assert sum(counts[0] for counts in equal.source_counts.values()) == base


def test_task_geometry_is_cross_fitted_and_task_loss_has_gradients() -> None:
    config = _small_config()
    x, y, cases, samples = _source_arrays()
    state = fit_task_geometry(
        x,
        y,
        cases,
        samples,
        source_center="2",
        source_row_hash="rows",
        frame_hash="frame",
        config=config,
        seed=17,
    )
    assert len(state.folds) == 3
    for fold in state.folds:
        assert fold.hessian_eigenvalues.min() >= config.hessian_eigenfloor
        assert fold.reference_per_class == config.reference_per_class
    generated = torch.tensor(x[:8], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    terms = task_terms(
        generated,
        labels,
        folds=tuple(_TorchFold(fold, device="cpu") for fold in state.folds),
        cdf_temperature=config.cdf_temperature,
    )
    total = sum(terms.values())
    total.backward()
    assert torch.isfinite(total)
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


def test_small_panel_has_exact_bg_branch_and_equal_steps(
    tmp_path: Path,
) -> None:
    config = _small_config()
    x, y, cases, samples = _source_arrays()
    geometry = fit_task_geometry(
        x,
        y,
        cases,
        samples,
        source_center="2",
        source_row_hash="rows",
        frame_hash="frame",
        config=config,
        seed=17,
    )
    panel = train_source_panel(
        x,
        y,
        cases,
        samples,
        geometry=geometry,
        config=config,
        source_center="2",
        training_seed=17,
        source_identity_hash="source",
        frame_hash="frame",
    )
    assert tuple(panel.arms) == ARMS
    assert {
        panel.arms[arm].branch_start_hash for arm in ("BG", "BM", "BT")
    } == {panel.task_branch_state_hash}
    assert {
        panel.arms[arm].state.completed_step for arm in ARMS
    } == {config.total_steps}
    store = TaskGeometryCheckpointStore(tmp_path, config)
    runtime = panel.arms["BT"]
    store.save(
        runtime,
        source_center="2",
        training_seed=17,
        frame_hash="frame",
        geometry_hash=geometry.state_hash,
    )
    restored = store.load(runtime.training_key_hash, device="cpu")
    assert restored is not None
    assert restored.state_hash == runtime.state.state_hash


def test_tiny_runner_writes_and_validates_non_consumable_bundle(
    tmp_path: Path,
) -> None:
    manifest, cache, cache_hash = _tiny_uniform_b_cache(tmp_path)
    config = replace(
        _small_config(),
        artifact_root=tmp_path / "artifacts",
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_cache_hash=cache_hash,
        base_generation_per_class=2,
    )
    root = run_uniform_b_task_geometry_source_inner_study(config)
    report = validate_uniform_b_task_geometry_bundle(root)
    assert report["status"] == "PASS"
    publication = (root / "reports/publication_state.json").read_text(
        encoding="utf-8"
    )
    assert "NON_CONSUMABLE_STUDY_COMPLETE" in publication
    assert '"may_feed_expert_bank": false' in publication
    publication_path = root / "reports/publication_state.json"
    tampered = json.loads(publication_path.read_text(encoding="utf-8"))
    tampered["may_feed_expert_bank"] = True
    publication_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="publication boundary"):
        validate_uniform_b_task_geometry_bundle(root)


def _small_config():
    production = load_uniform_b_task_geometry_config(CONFIG)
    return replace(
        production,
        artifact_root=Path("/tmp/uniform-b-test"),
        manifest_path=Path("/tmp/midogpp/manifest.csv"),
        feature_cache_path=Path("/tmp/midogpp/virchow2/train.pt"),
        heldout_centers=("0", "1", "2"),
        training_seeds=(17,),
        generation_seeds=(17,),
        device="cpu",
        hidden_dim=16,
        latent_dim=4,
        batch_size=8,
        warmup_steps=2,
        task_start_step=4,
        total_steps=6,
        crossfit_folds=3,
        nystrom_components=8,
        reference_per_class=4,
        base_generation_per_class=4,
    )


def _source_arrays():
    rng = np.random.default_rng(9)
    rows = []
    labels = []
    cases = []
    samples = []
    for case_index in range(18):
        label = case_index % 2
        for row_index in range(4):
            rows.append(
                rng.normal(loc=float(label), scale=0.5, size=128)
            )
            labels.append(label)
            cases.append(f"case-{case_index}")
            samples.append(f"case-{case_index}-row-{row_index}")
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        tuple(cases),
        tuple(samples),
    )


def _block(source: str, *, per_class: int) -> GeneratedBlock:
    rng = np.random.default_rng(int(source))
    embeddings = rng.normal(size=(2 * per_class, 3840)).astype(np.float32)
    labels = np.asarray([0] * per_class + [1] * per_class, dtype=np.int64)
    return GeneratedBlock(
        source_center=source,
        arm="BG",
        training_seed=17,
        generation_seed=17,
        embeddings=embeddings,
        labels=labels,
        per_class=per_class,
        checkpoint_hash=f"checkpoint-{source}",
        frame_hash=f"frame-{source}",
        stream_hash=f"stream-{source}",
    )


def _tiny_uniform_b_cache(tmp_path: Path) -> tuple[Path, Path, str]:
    data_root = tmp_path / "midogpp" / "virchow2"
    data_root.mkdir(parents=True)
    manifest = data_root / "manifest.csv"
    cache = data_root / "train.pt"
    rng = np.random.default_rng(31)
    embeddings = []
    metadata = []
    rows = []
    row_index = 0
    for center in ("0", "1", "2"):
        for case_index in range(16):
            label = case_index % 2
            for patch_index in range(8):
                sample_id = f"{center}-{case_index}-{patch_index}"
                case_id = f"{center}-case-{case_index}"
                image_path = f"/midogpp/{center}/{case_index}.png"
                embeddings.append(
                    rng.normal(
                        loc=0.1 * int(center) + 0.2 * label,
                        scale=1.0,
                        size=3840,
                    )
                )
                metadata.append(
                    {
                        "sample_id": sample_id,
                        "case_id": case_id,
                        "center": center,
                        "label": label,
                        "split": "train",
                        "image_path": image_path,
                    }
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "case_id": case_id,
                        "center": center,
                        "label": label,
                        "split": "train",
                        "image_path": image_path,
                    }
                )
                row_index += 1
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(
        {
            "embeddings": torch.tensor(
                np.asarray(embeddings, dtype=np.float32)
            ),
            "metadata": metadata,
            "feature_extractor": {
                "family": "Virchow2",
                "dataset": "MIDOG++",
                "variant": "uniform_b_test",
            },
        },
        cache,
    )
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    return manifest, cache, digest
