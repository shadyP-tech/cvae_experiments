from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sail.config import PipelineConfig  # noqa: E402
from sail.features import write_npz_cache  # noqa: E402
import sail.pipeline as pipeline  # noqa: E402


def test_source_inner_and_dense_reuse_identical_member_predictions(tmp_path: Path) -> None:
    _write_synthetic_caches(tmp_path)
    config = _config()
    provider = pipeline._CacheProvider(config=config, repo_root=tmp_path)

    row = pipeline.source_lodo_candidate_row(
        config=config,
        provider=provider,
        seed=1,
        heldout_center="0",
        source_centers=("1", "2"),
        representation="raw",
        c_value=1.0,
        class_weight="none",
    )
    assert row["status"] == "ok"
    assert provider.cache_misses == 2
    assert provider.cache_hits == 0

    pipeline.evaluate_dense_configs(
        config=config,
        provider=provider,
        seed=1,
        heldout_center="0",
        fit_centers=("2",),
        eval_center="1",
        eval_split="train",
        member_rows=[row],
        aggregation_rule="geometric",
    )
    assert provider.cache_misses == 2
    assert provider.cache_hits == 1


def test_aggregation_rules_reuse_member_predictions(tmp_path: Path) -> None:
    _write_synthetic_caches(tmp_path)
    config = _config()
    provider = pipeline._CacheProvider(config=config, repo_root=tmp_path)
    rows = [_member(0.1), _member(1.0)]

    pipeline.evaluate_dense_configs(
        config=config,
        provider=provider,
        seed=1,
        heldout_center="0",
        fit_centers=("2",),
        eval_center="1",
        eval_split="train",
        member_rows=rows,
        aggregation_rule="geometric",
    )
    assert provider.cache_misses == 2

    pipeline.evaluate_dense_configs(
        config=config,
        provider=provider,
        seed=1,
        heldout_center="0",
        fit_centers=("2",),
        eval_center="1",
        eval_split="train",
        member_rows=rows,
        aggregation_rule="arithmetic",
    )
    assert provider.cache_misses == 2
    assert provider.cache_hits == 2


def test_overlapping_topk_only_fits_new_members(tmp_path: Path) -> None:
    _write_synthetic_caches(tmp_path)
    config = _config()
    provider = pipeline._CacheProvider(config=config, repo_root=tmp_path)
    rows = [_member(0.01), _member(0.1), _member(1.0)]

    pipeline.evaluate_dense_configs(
        config=config,
        provider=provider,
        seed=1,
        heldout_center="0",
        fit_centers=("1", "2"),
        eval_center="0",
        eval_split="test",
        member_rows=rows[:1],
        aggregation_rule="geometric",
    )
    assert provider.cache_misses == 1

    pipeline.evaluate_dense_configs(
        config=config,
        provider=provider,
        seed=1,
        heldout_center="0",
        fit_centers=("1", "2"),
        eval_center="0",
        eval_split="test",
        member_rows=rows,
        aggregation_rule="geometric",
    )
    assert provider.cache_hits == 1
    assert provider.cache_misses == 3
    assert provider.cache_size == 3


def test_split_fields_and_cache_fingerprints_distinguish_keys(tmp_path: Path) -> None:
    _write_synthetic_caches(tmp_path)
    config = _config()
    provider = pipeline._CacheProvider(config=config, repo_root=tmp_path)

    assert provider.cache_fingerprint(1, "train") != provider.cache_fingerprint(1, "test")

    provider.member_prediction(
        config=config,
        seed=1,
        heldout_center="0",
        fit_centers=("2",),
        eval_center="1",
        eval_split="train",
        representation="raw",
        c_value=1.0,
        class_weight="none",
    )
    provider.member_prediction(
        config=config,
        seed=1,
        heldout_center="0",
        fit_centers=("2",),
        eval_center="1",
        eval_split="train",
        representation="raw",
        c_value=1.0,
        class_weight="none",
    )
    provider.member_prediction(
        config=config,
        seed=1,
        heldout_center="0",
        fit_centers=("1",),
        eval_center="2",
        eval_split="train",
        representation="raw",
        c_value=1.0,
        class_weight="none",
    )
    provider.member_prediction(
        config=config,
        seed=1,
        heldout_center="0",
        fit_centers=("1", "2"),
        eval_center="0",
        eval_split="test",
        representation="raw",
        c_value=1.0,
        class_weight="none",
    )

    assert provider.cache_hits == 1
    assert provider.cache_misses == 3
    assert provider.cache_size == 3
    eval_fingerprints = {key.eval_cache_fingerprint for key in provider._member_predictions}
    assert provider.cache_fingerprint(1, "train") in eval_fingerprints
    assert provider.cache_fingerprint(1, "test") in eval_fingerprints


def test_cached_probabilities_are_read_only_and_not_mutated(tmp_path: Path) -> None:
    import numpy as np

    _write_synthetic_caches(tmp_path)
    config = _config()
    provider = pipeline._CacheProvider(config=config, repo_root=tmp_path)
    bundle = provider.member_prediction(
        config=config,
        seed=1,
        heldout_center="0",
        fit_centers=("2",),
        eval_center="1",
        eval_split="train",
        representation="raw",
        c_value=1.0,
        class_weight="none",
    )

    assert not bundle.proba.flags.writeable
    before = bundle.proba.copy()
    pipeline.aggregate_member_predictions([bundle], aggregation_rule="geometric")
    np.testing.assert_array_equal(bundle.proba, before)
    with pytest.raises(ValueError):
        bundle.proba[0, 0] = 0.0


def test_cached_and_uncached_pipeline_outputs_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_synthetic_caches(tmp_path)
    config = _config(
        c_grid=(0.1, 1.0),
        primary_k_values=(1, 2),
        aggregation_rules=("geometric", "arithmetic"),
        artifacts_root="artifacts/cached",
    )
    cached = pipeline.run_pipeline(config=config, repo_root=tmp_path)
    cached_rows = _pipeline_rows(cached.output_paths)

    def uncached_member_prediction(
        self: object,
        *,
        config: PipelineConfig,
        seed: int,
        heldout_center: str,
        fit_centers: tuple[str, ...],
        eval_center: str,
        eval_split: str,
        representation: str,
        c_value: float,
        class_weight: str,
    ) -> pipeline.PredictionBundle:
        del heldout_center
        train_cache = self.train(int(seed))
        eval_cache = self.test(int(seed)) if str(eval_split) == "test" else train_cache
        return pipeline.fit_member_prediction(
            config=config,
            train_cache=train_cache,
            eval_cache=eval_cache,
            seed=int(seed),
            fit_centers=fit_centers,
            eval_center=str(eval_center),
            eval_split=str(eval_split),
            representation=str(representation),
            c_value=float(c_value),
            class_weight=pipeline.class_weight_label(class_weight),
        )

    monkeypatch.setattr(pipeline._CacheProvider, "member_prediction", uncached_member_prediction)
    uncached_config = replace(config, artifacts_root="artifacts/uncached")
    uncached = pipeline.run_pipeline(config=uncached_config, repo_root=tmp_path)

    assert cached.decision_labels == uncached.decision_labels
    assert cached_rows == _pipeline_rows(uncached.output_paths)


def _config(
    *,
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
    primary_k_values: tuple[int, ...] = (1, 3),
    aggregation_rules: tuple[str, ...] = ("geometric", "arithmetic"),
    artifacts_root: str = "artifacts/run",
) -> PipelineConfig:
    return PipelineConfig(
        candidate_centers=("0", "1", "2"),
        experiment_seeds=(1,),
        support_sizes=(0,),
        support_seeds=(1,),
        representations=("raw",),
        c_grid=c_grid,
        class_weight_grid=("none",),
        primary_k_values=primary_k_values,
        aggregation_rules=aggregation_rules,
        cache_root="cache",
        cache_path_template="{cache_root}/{backbone}/seed{seed}/embeddings/{split}.npz",
        artifacts_root=artifacts_root,
    )


def _member(c_value: float) -> dict[str, object]:
    return {
        "backbone_name": "virchow2",
        "representation": "raw",
        "C": float(c_value),
        "class_weight": "none",
    }


def _pipeline_rows(paths: dict[str, Path]) -> dict[str, list[dict[str, str]]]:
    return {
        name: _read_csv(path)
        for name, path in paths.items()
        if name in {"source_lodo_selection", "source_k_selection", "dense_aggregation"}
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


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
