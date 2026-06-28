"""MIDOG++ backend for exported source-summary artifacts.

This backend reads the ``exported_source_summary_manifest.csv`` produced by
``cvae_rebuild`` and target-evaluation feature caches, then exposes them through
the ``MidogppScoringBackend`` interface. Summary sampling is deterministic and
class-stratified. A decoder/transform callable can be injected when latent
summary samples need to be mapped into the downstream classifier feature frame.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..artifacts import FrozenProtocolSnapshot, stable_hash, write_frozen_snapshot
from ..protocol import ProtocolError
from ..schemas.midogpp import MIDOGPP_ELIGIBLE_CENTERS
from .midogpp_runner import MidogppCandidate, MidogppRunContext, MidogppScoringBackend, MidogppScoringResult

EmbeddingTransform = Callable[[MidogppCandidate, int, object], object]


@dataclass(frozen=True)
class MidogppFeatureCache:
    embeddings: object
    metadata: tuple[Mapping[str, object], ...]
    path: Path


@dataclass(frozen=True)
class SourceSummary:
    experiment_seed: int
    source_center: str
    class_label: int
    selection_rule: str
    summary_path: Path
    summary_hash: str
    expert_config_hash: str
    status: str


@dataclass(frozen=True)
class MidogppExternalBaselineScore:
    """Diagnostic score imported from an already-locked baseline artifact."""

    baseline_method: str
    experiment_seed: int
    heldout_center: str
    replicate_seed: int
    bacc: float
    macro_f1: float
    status: str
    error_message: str
    source_path: Path
    row_hash: str


@dataclass(frozen=True)
class MidogppSourceSummaryPreflight:
    """Readiness report for source-summary MIDOG++ phase-1 scoring."""

    summary_manifest: Path
    cache_paths: tuple[Path, ...]
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    missing_summary_keys: tuple[str, ...]
    cache_eval_counts: Mapping[str, int]
    cache_label_sets: Mapping[str, tuple[int, ...]]
    source_summary_dims: Mapping[str, int]
    source_summary_paths: Mapping[str, str]
    cache_embedding_dims: Mapping[str, int]
    status: str

    def to_report(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_source_summary_preflight_report_v1",
            "summary_manifest": str(self.summary_manifest),
            "cache_paths": [str(path) for path in self.cache_paths],
            "experiment_seeds": list(self.experiment_seeds),
            "heldout_centers": list(self.heldout_centers),
            "missing_summary_keys": list(self.missing_summary_keys),
            "cache_eval_counts": dict(self.cache_eval_counts),
            "cache_label_sets": {key: list(value) for key, value in self.cache_label_sets.items()},
            "source_summary_dims": dict(self.source_summary_dims),
            "source_summary_paths": dict(self.source_summary_paths),
            "cache_embedding_dims": dict(self.cache_embedding_dims),
            "status": self.status,
        }


@dataclass(frozen=True)
class MidogppBaselinePreflight:
    """Readiness report for imported locked baseline diagnostics."""

    baseline_matrix_paths: tuple[Path, ...]
    baseline_methods: tuple[str, ...]
    experiment_seed: int
    replicate_seed: int
    heldout_centers: tuple[str, ...]
    available_baseline_keys: tuple[str, ...]
    baseline_matrix_hashes: Mapping[str, str]
    baseline_row_hashes: Mapping[str, str]
    status: str

    def to_report(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_baseline_preflight_report_v1",
            "baseline_matrix_paths": [str(path) for path in self.baseline_matrix_paths],
            "baseline_methods": list(self.baseline_methods),
            "experiment_seed": self.experiment_seed,
            "replicate_seed": self.replicate_seed,
            "heldout_centers": list(self.heldout_centers),
            "available_baseline_keys": list(self.available_baseline_keys),
            "baseline_matrix_hashes": dict(self.baseline_matrix_hashes),
            "baseline_row_hashes": dict(self.baseline_row_hashes),
            "status": self.status,
        }


@dataclass(frozen=True)
class MidogppPhase1RunHashes:
    """Frozen run hashes required by MIDOG++ phase-1 row keys."""

    config_hash: str
    protocol_hash: str
    feature_frame_hash: str
    frozen_snapshot_path: Path
    snapshot: FrozenProtocolSnapshot
    summary_manifest_hash: str
    source_summary_file_hashes: Mapping[str, str]
    cache_file_hashes: Mapping[str, str]

    def to_report(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_phase1_run_hashes_v1",
            "config_hash": self.config_hash,
            "protocol_hash": self.protocol_hash,
            "feature_frame_hash": self.feature_frame_hash,
            "frozen_snapshot_path": str(self.frozen_snapshot_path),
            "summary_manifest_hash": self.summary_manifest_hash,
            "source_summary_file_hashes": dict(self.source_summary_file_hashes),
            "cache_file_hashes": dict(self.cache_file_hashes),
            "snapshot": self.snapshot.to_payload(),
        }


class SourceSummaryMidogppBackend(MidogppScoringBackend):
    """Backend using exported class-conditional source summary ``.npz`` files."""

    def __init__(
        self,
        *,
        summary_manifest: Path,
        test_cache_root: Path | None = None,
        test_cache_path: Path | None = None,
        transform: EmbeddingTransform | None = None,
        selection_rule: str = "largest_viable",
        baseline_matrix_paths: Sequence[Path] = (),
    ) -> None:
        if test_cache_root is None and test_cache_path is None:
            raise ProtocolError("MIDOG++ source-summary backend requires test_cache_root or test_cache_path.")
        self.summary_manifest = Path(summary_manifest)
        self.test_cache_root = Path(test_cache_root) if test_cache_root is not None else None
        self.test_cache_path = Path(test_cache_path) if test_cache_path is not None else None
        self.transform = transform or _identity_transform
        self.selection_rule = str(selection_rule)
        self._summaries = _load_source_summaries(self.summary_manifest, selection_rule=self.selection_rule)
        self._baseline_scores = _load_external_baseline_scores(baseline_matrix_paths)
        self._cache_by_seed: dict[int, MidogppFeatureCache] = {}

    def synthetic_train_batch(
        self,
        candidate: MidogppCandidate,
        *,
        context: MidogppRunContext,
    ) -> tuple[object, Sequence[int]]:
        class_chunks: list[object] = []
        labels: list[int] = []
        for class_label in (0, 1):
            summary = self._summary_for(
                experiment_seed=int(context.experiment_seed),
                source_center=str(candidate.candidate_source_center),
                class_label=int(class_label),
            )
            latent = _sample_summary_npz(
                summary.summary_path,
                n_samples=int(context.synthetic_per_class_total),
                seed=int(context.latent_sample_seed) + int(class_label),
            )
            class_chunks.append(self.transform(candidate, int(class_label), latent))
            labels.extend([int(class_label)] * int(context.synthetic_per_class_total))
        return _concat_rows(class_chunks), labels

    def target_eval_batch(
        self,
        *,
        context: MidogppRunContext,
    ) -> tuple[object, Sequence[int]]:
        cache = self._test_cache(int(context.experiment_seed))
        indices = [
            idx
            for idx, row in enumerate(cache.metadata)
            if str(_domain(row)) == str(context.heldout_center)
        ]
        if not indices:
            raise ProtocolError(f"No target eval rows found for heldout_center={context.heldout_center}.")
        embeddings = _take_rows(cache.embeddings, indices)
        labels = [int(_label(cache.metadata[idx])) for idx in indices]
        if sorted(set(labels)) != [0, 1]:
            raise ProtocolError(f"MIDOG++ target eval must contain binary labels 0/1, got {sorted(set(labels))}.")
        return embeddings, labels

    def method_baseline_score(
        self,
        baseline_method: str,
        *,
        context: MidogppRunContext,
        candidate_sources: Sequence[str],
    ) -> MidogppScoringResult | None:
        _ = candidate_sources
        key = (
            str(baseline_method),
            int(context.experiment_seed),
            str(context.heldout_center),
            int(context.replicate_seed),
        )
        score = self._baseline_scores.get(key)
        if score is None:
            return None
        return MidogppScoringResult(
            bacc=score.bacc,
            macro_f1=score.macro_f1,
            status=score.status,
            error_message=score.error_message,
            checkpoint_hash=f"baseline:{score.baseline_method}:{score.row_hash}",
        )

    def candidate_for_source(
        self,
        *,
        context: MidogppRunContext,
        source_center: str,
        candidate_method: str = "single_source_adaptive_k",
    ) -> MidogppCandidate:
        summaries = [
            self._summary_for(
                experiment_seed=int(context.experiment_seed),
                source_center=str(source_center),
                class_label=class_label,
            )
            for class_label in (0, 1)
        ]
        return MidogppCandidate(
            candidate_source_center=str(source_center),
            candidate_id=f"midogpp_source_{source_center}_{candidate_method}",
            candidate_method=candidate_method,
            checkpoint_hash="|".join(summary.summary_hash for summary in summaries),
        )

    def candidates_for_context(self, context: MidogppRunContext) -> tuple[MidogppCandidate, ...]:
        return tuple(
            self.candidate_for_source(context=context, source_center=center)
            for center in MIDOGPP_ELIGIBLE_CENTERS
            if str(center) != str(context.heldout_center)
            and self._has_source_pair(int(context.experiment_seed), str(center))
        )

    def _summary_for(self, *, experiment_seed: int, source_center: str, class_label: int) -> SourceSummary:
        key = (int(experiment_seed), str(source_center), int(class_label))
        try:
            return self._summaries[key]
        except KeyError as exc:
            raise ProtocolError(
                f"Missing MIDOG++ source summary for seed={experiment_seed}, "
                f"source={source_center}, class={class_label}."
            ) from exc

    def _has_source_pair(self, experiment_seed: int, source_center: str) -> bool:
        return all((int(experiment_seed), str(source_center), class_label) in self._summaries for class_label in (0, 1))

    def _test_cache(self, experiment_seed: int) -> MidogppFeatureCache:
        if int(experiment_seed) not in self._cache_by_seed:
            path = self.test_cache_path or _cache_path(self.test_cache_root, seed=int(experiment_seed), split="test")
            self._cache_by_seed[int(experiment_seed)] = load_midogpp_feature_cache(path)
        return self._cache_by_seed[int(experiment_seed)]


def preflight_midogpp_source_summary_inputs(
    *,
    summary_manifest: Path,
    experiment_seeds: Sequence[int],
    heldout_centers: Sequence[str],
    test_cache_root: Path | None = None,
    test_cache_path: Path | None = None,
    selection_rule: str = "largest_viable",
) -> MidogppSourceSummaryPreflight:
    """Validate real MIDOG++ phase-1 inputs before scoring starts."""

    if test_cache_root is None and test_cache_path is None:
        raise ProtocolError("MIDOG++ preflight requires test_cache_root or test_cache_path.")
    seeds = tuple(int(seed) for seed in experiment_seeds)
    heldouts = tuple(str(center) for center in heldout_centers)
    summaries = _load_source_summaries(Path(summary_manifest), selection_rule=str(selection_rule))
    missing: list[str] = []
    summary_dims: dict[str, int] = {}
    summary_paths: dict[str, str] = {}
    for seed in seeds:
        for heldout in heldouts:
            if heldout not in MIDOGPP_ELIGIBLE_CENTERS:
                raise ProtocolError(f"Unknown MIDOG++ heldout center: {heldout!r}")
            for source in MIDOGPP_ELIGIBLE_CENTERS:
                if source == heldout:
                    continue
                for cls in (0, 1):
                    key = (int(seed), str(source), int(cls))
                    summary = summaries.get(key)
                    if summary is None:
                        missing.append(f"seed={seed}|source={source}|class={cls}")
                    else:
                        summary_key = f"seed={seed}|source={source}|class={cls}"
                        summary_dims[summary_key] = _validate_summary_npz(summary.summary_path)
                        summary_paths[summary_key] = str(summary.summary_path)
    cache_paths: list[Path] = []
    eval_counts: dict[str, int] = {}
    label_sets: dict[str, tuple[int, ...]] = {}
    cache_dims: dict[str, int] = {}
    for seed in seeds:
        cache_path = Path(test_cache_path) if test_cache_path is not None else _cache_path(test_cache_root, seed=seed, split="test")
        cache_paths.append(cache_path)
        cache = load_midogpp_feature_cache(cache_path)
        cache_dim = _row_dim(cache.embeddings)
        cache_dims[f"seed={seed}|cache=test"] = cache_dim
        for heldout in heldouts:
            labels = [
                int(_label(row))
                for row in cache.metadata
                if str(_domain(row)) == str(heldout)
            ]
            key = f"seed={seed}|heldout={heldout}"
            eval_counts[key] = len(labels)
            label_sets[key] = tuple(sorted(set(labels)))
            if sorted(set(labels)) != [0, 1]:
                raise ProtocolError(f"MIDOG++ preflight requires binary target eval labels for {key}.")
    if missing:
        raise ProtocolError(f"MIDOG++ source summary coverage is incomplete: {missing[:10]}")
    unique_summary_dims = sorted(set(summary_dims.values()))
    unique_cache_dims = sorted(set(cache_dims.values()))
    if unique_summary_dims != unique_cache_dims:
        raise ProtocolError(
            "MIDOG++ identity source-summary backend requires sampled summary dimension to match "
            f"target cache embedding dimension; got source_summary_dims={unique_summary_dims}, "
            f"cache_embedding_dims={unique_cache_dims}. Provide an explicit decoder/transform backend."
        )
    return MidogppSourceSummaryPreflight(
        summary_manifest=Path(summary_manifest),
        cache_paths=tuple(cache_paths),
        experiment_seeds=seeds,
        heldout_centers=heldouts,
        missing_summary_keys=(),
        cache_eval_counts=eval_counts,
        cache_label_sets=label_sets,
        source_summary_dims=summary_dims,
        source_summary_paths=summary_paths,
        cache_embedding_dims=cache_dims,
        status="PASS",
    )


def preflight_midogpp_external_baselines(
    *,
    baseline_matrix_paths: Sequence[Path],
    baseline_methods: Sequence[str],
    experiment_seed: int,
    replicate_seed: int,
    heldout_centers: Sequence[str],
) -> MidogppBaselinePreflight:
    """Validate requested imported baseline diagnostics before phase-1 scoring."""

    paths = tuple(Path(path) for path in baseline_matrix_paths)
    methods = tuple(str(method) for method in baseline_methods)
    heldouts = tuple(str(center) for center in heldout_centers)
    if bool(paths) != bool(methods):
        raise ProtocolError("MIDOG++ baseline preflight requires both baseline_matrix_paths and baseline_methods.")
    scores = _load_external_baseline_scores(paths)
    missing: list[str] = []
    for heldout in heldouts:
        if heldout not in MIDOGPP_ELIGIBLE_CENTERS:
            raise ProtocolError(f"Unknown MIDOG++ heldout center for baseline preflight: {heldout!r}")
        for method in methods:
            key = (str(method), int(experiment_seed), str(heldout), int(replicate_seed))
            if key not in scores:
                missing.append(
                    f"method={method}|seed={int(experiment_seed)}|heldout={heldout}|replicate_seed={int(replicate_seed)}"
                )
    if missing:
        raise ProtocolError(f"MIDOG++ requested baseline diagnostics are incomplete: {missing[:10]}")
    available = tuple(
        f"method={method}|seed={seed}|heldout={heldout}|replicate_seed={replicate}"
        for method, seed, heldout, replicate in sorted(scores)
    )
    row_hashes = {
        f"method={method}|seed={seed}|heldout={heldout}|replicate_seed={replicate}": score.row_hash
        for (method, seed, heldout, replicate), score in sorted(scores.items())
    }
    return MidogppBaselinePreflight(
        baseline_matrix_paths=paths,
        baseline_methods=methods,
        experiment_seed=int(experiment_seed),
        replicate_seed=int(replicate_seed),
        heldout_centers=heldouts,
        available_baseline_keys=available,
        baseline_matrix_hashes={str(path): _file_hash(path) for path in paths},
        baseline_row_hashes=row_hashes,
        status="PASS",
    )


def build_midogpp_phase1_run_hashes(
    *,
    config_path: Path,
    summary_manifest: Path,
    preflight: MidogppSourceSummaryPreflight,
    heldout_centers: Sequence[str],
    experiment_seed: int,
    replicate_seed: int,
    synthetic_per_class_total: int,
    generation_seed: int,
    latent_sample_seed: int,
    classifier_seed: int,
    out_dir: Path,
) -> MidogppPhase1RunHashes:
    """Build and persist frozen MIDOG++ run hashes before scoring."""

    config = Path(config_path)
    config_hash = _file_hash(config)
    summary_manifest_hash = _file_hash(Path(summary_manifest))
    source_summary_file_hashes = {
        key: _file_hash(Path(path))
        for key, path in sorted(preflight.source_summary_paths.items())
    }
    cache_file_hashes = {
        str(path): _file_hash(path)
        for path in preflight.cache_paths
    }
    candidate_pool_hash = stable_hash(
        {
            "dataset": "midogpp",
            "heldout_centers": tuple(str(center) for center in heldout_centers),
            "eligible_centers": MIDOGPP_ELIGIBLE_CENTERS,
            "excluded_centers": ("4",),
            "summary_manifest": str(Path(summary_manifest)),
            "summary_manifest_hash": summary_manifest_hash,
            "source_summary_file_hashes": source_summary_file_hashes,
            "experiment_seed": int(experiment_seed),
            "selection_rule": "largest_viable",
        }
    )
    generation_config_hash = stable_hash(
        {
            "mode": "class_stratified_reference_posterior_resampling",
            "synthetic_per_class_total": int(synthetic_per_class_total),
            "generation_seed": int(generation_seed),
            "latent_sample_seed": int(latent_sample_seed),
            "source_summary_dims": dict(sorted(preflight.source_summary_dims.items())),
            "source_summary_file_hashes": source_summary_file_hashes,
        }
    )
    classifier_config_hash = stable_hash(
        {
            "family": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": None,
            "scaler_fit": "synthetic_train_only",
            "hyperparameter_tuning": "forbidden",
            "classifier_seed": int(classifier_seed),
        }
    )
    metric_config_hash = stable_hash({"primary": ("bacc", "macro_f1"), "chosen_before_target_eval": True})
    feature_frame_hash = stable_hash(
        {
            "cache_paths": tuple(str(path) for path in preflight.cache_paths),
            "cache_file_hashes": cache_file_hashes,
            "cache_embedding_dims": dict(sorted(preflight.cache_embedding_dims.items())),
            "heldout_centers": tuple(str(center) for center in heldout_centers),
        }
    )
    routing_config_hash = stable_hash(
        {
            "role": "diagnostic_only",
            "support_size": 0,
            "support_seed": "none",
            "support_set_id": "none",
            "replicate_seed": int(replicate_seed),
            "selection_used_target_labels": False,
            "support_labels_used": False,
        }
    )
    snapshot = FrozenProtocolSnapshot(
        candidate_pool_hash=candidate_pool_hash,
        generation_config_hash=generation_config_hash,
        classifier_config_hash=classifier_config_hash,
        metric_config_hash=metric_config_hash,
        feature_config_hash=feature_frame_hash,
        routing_config_hash=routing_config_hash,
    )
    snapshot_path = Path(out_dir) / "configs" / "frozen_protocol_snapshot.json"
    write_frozen_snapshot(snapshot_path, snapshot)
    return MidogppPhase1RunHashes(
        config_hash=config_hash,
        protocol_hash=snapshot.protocol_hash,
        feature_frame_hash=feature_frame_hash,
        frozen_snapshot_path=snapshot_path,
        snapshot=snapshot,
        summary_manifest_hash=summary_manifest_hash,
        source_summary_file_hashes=source_summary_file_hashes,
        cache_file_hashes=cache_file_hashes,
    )


def load_midogpp_feature_cache(path: Path) -> MidogppFeatureCache:
    """Load a MIDOG++ feature cache from ``.npz`` or Torch ``.pt``."""

    cache_path = Path(path)
    if cache_path.suffix == ".npz":
        return _load_npz_cache(cache_path)
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading .pt MIDOG++ feature caches requires torch.") from exc
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping) or "embeddings" not in payload:
        raise ProtocolError(f"Feature cache payload must contain embeddings: {cache_path}")
    metadata = tuple(_normalize_meta(row) for row in payload.get("metadata", ()))
    return MidogppFeatureCache(embeddings=payload["embeddings"], metadata=metadata, path=cache_path)


def _load_source_summaries(path: Path, *, selection_rule: str) -> dict[tuple[int, str, int], SourceSummary]:
    if not Path(path).exists():
        raise ProtocolError(f"MIDOG++ source summary manifest not found: {path}")
    summaries: dict[tuple[int, str, int], SourceSummary] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "ok")) != "ok":
                continue
            if str(row.get("selection_rule", "")) != str(selection_rule):
                continue
            summary_path = _resolve_summary_path(Path(path), str(row.get("summary_path", "")))
            summary = SourceSummary(
                experiment_seed=int(row["experiment_seed"]),
                source_center=str(row["source_center"]),
                class_label=int(row["class_label"]),
                selection_rule=str(row["selection_rule"]),
                summary_path=summary_path,
                summary_hash=str(row.get("summary_hash", "")),
                expert_config_hash=str(row.get("expert_config_hash", "")),
                status=str(row.get("status", "ok")),
            )
            key = (summary.experiment_seed, summary.source_center, summary.class_label)
            if key in summaries:
                raise ProtocolError(f"Duplicate MIDOG++ source summary manifest key: {key}")
            summaries[key] = summary
    if not summaries:
        raise ProtocolError(f"No usable MIDOG++ source summaries found in {path}")
    return summaries


def _file_hash(path: Path) -> str:
    if not Path(path).exists():
        raise ProtocolError(f"MIDOG++ hash input file not found: {path}")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _load_external_baseline_scores(
    paths: Sequence[Path],
) -> dict[tuple[str, int, str, int], MidogppExternalBaselineScore]:
    scores: dict[tuple[str, int, str, int], MidogppExternalBaselineScore] = {}
    for path in paths:
        source_path = Path(path)
        if not source_path.exists():
            raise ProtocolError(f"MIDOG++ baseline matrix not found: {source_path}")
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            for idx, row in enumerate(csv.DictReader(handle), start=2):
                score = _external_baseline_score_from_row(row, source_path=source_path, line_no=idx)
                key = (
                    score.baseline_method,
                    score.experiment_seed,
                    score.heldout_center,
                    score.replicate_seed,
                )
                if key in scores:
                    raise ProtocolError(f"Duplicate MIDOG++ external baseline score key {key}: {source_path}")
                scores[key] = score
    return scores


def _external_baseline_score_from_row(
    row: Mapping[str, object],
    *,
    source_path: Path,
    line_no: int,
) -> MidogppExternalBaselineScore:
    status = str(row.get("status", "ok") or "ok")
    if status != "ok":
        raise ProtocolError(f"MIDOG++ external baseline row must be status=ok at {source_path}:{line_no}")
    _assert_diagnostic_external_baseline(row, source_path=source_path, line_no=line_no)
    method = _external_baseline_method(row)
    bacc = _finite_float(row.get("bacc"), column="bacc", source_path=source_path, line_no=line_no)
    macro_f1 = _finite_float(row.get("macro_f1"), column="macro_f1", source_path=source_path, line_no=line_no)
    return MidogppExternalBaselineScore(
        baseline_method=method,
        experiment_seed=int(row["experiment_seed"]),
        heldout_center=str(row["heldout_center"]),
        replicate_seed=int(row.get("replicate_seed", 0) or 0),
        bacc=bacc,
        macro_f1=macro_f1,
        status=status,
        error_message=str(row.get("error_message", "") or ""),
        source_path=source_path,
        row_hash=_external_baseline_row_hash(row),
    )


def _assert_diagnostic_external_baseline(
    row: Mapping[str, object],
    *,
    source_path: Path,
    line_no: int,
) -> None:
    if "selection_used_target_labels" in row and _bool(row.get("selection_used_target_labels"), False):
        raise ProtocolError(f"MIDOG++ baseline row used target labels for selection at {source_path}:{line_no}")
    if "support_labels_used" in row and _bool(row.get("support_labels_used"), False):
        raise ProtocolError(f"MIDOG++ baseline row used support labels at {source_path}:{line_no}")
    if "target_eval_labels_used_for_scoring_only" in row and not _bool(
        row.get("target_eval_labels_used_for_scoring_only"),
        True,
    ):
        raise ProtocolError(f"MIDOG++ baseline row does not mark target labels as scoring-only at {source_path}:{line_no}")
    marker = str(row.get("selection_source", row.get("role", row.get("eligibility", "diagnostic_only"))) or "")
    if marker and marker not in {"diagnostic_only", "oracle_diagnostic"}:
        raise ProtocolError(f"MIDOG++ baseline row is not diagnostic-only at {source_path}:{line_no}: {marker}")


def _external_baseline_method(row: Mapping[str, object]) -> str:
    for column in ("candidate_method", "prior_method", "expert_id"):
        value = str(row.get(column, "") or "").strip()
        if value and value.lower() != "nan":
            return value
    raise ProtocolError(f"MIDOG++ external baseline row lacks a method identifier: {row}")


def _finite_float(value: object, *, column: str, source_path: Path, line_no: int) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ProtocolError(f"MIDOG++ baseline column {column} must be finite at {source_path}:{line_no}")
    return out


def _external_baseline_row_hash(row: Mapping[str, object]) -> str:
    payload = {
        "experiment_seed": str(row.get("experiment_seed", "")),
        "heldout_center": str(row.get("heldout_center", "")),
        "replicate_seed": str(row.get("replicate_seed", "")),
        "baseline_method": _external_baseline_method(row),
        "bacc": str(row.get("bacc", "")),
        "macro_f1": str(row.get("macro_f1", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bool(value: object, default: bool) -> bool:
    if value in {None, ""}:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sample_summary_npz(path: Path, *, n_samples: int, seed: int) -> object:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Sampling MIDOG++ source summaries requires numpy.") from exc
    if int(n_samples) <= 0:
        raise ProtocolError("synthetic_per_class_total must be positive.")
    payload = np.load(path, allow_pickle=False)
    weights = np.asarray(payload["weights"], dtype=float)
    means = np.asarray(payload["means"], dtype=float)
    diag_vars = np.asarray(payload["diag_vars"], dtype=float)
    if weights.ndim != 1 or means.ndim != 2 or diag_vars.shape != means.shape:
        raise ProtocolError(f"Malformed MIDOG++ source summary arrays: {path}")
    if means.shape[0] != weights.shape[0]:
        raise ProtocolError(f"Source summary component count mismatch: {path}")
    if not np.isfinite(weights).all() or not np.isfinite(means).all() or not np.isfinite(diag_vars).all():
        raise ProtocolError(f"Source summary contains non-finite values: {path}")
    if np.any(diag_vars < 0.0):
        raise ProtocolError(f"Source summary contains negative diagonal variance: {path}")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ProtocolError(f"Source summary weights must sum to a positive value: {path}")
    probs = weights / total
    rng = np.random.default_rng(int(seed))
    component_ids = rng.choice(np.arange(weights.shape[0]), size=int(n_samples), replace=True, p=probs)
    eps = rng.normal(size=(int(n_samples), means.shape[1]))
    return means[component_ids] + eps * np.sqrt(diag_vars[component_ids])


def _validate_summary_npz(path: Path) -> int:
    sampled = _sample_summary_npz(path, n_samples=1, seed=0)
    return _row_dim(sampled)


def _load_npz_cache(path: Path) -> MidogppFeatureCache:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Loading .npz MIDOG++ feature caches requires numpy.") from exc
    payload = np.load(path, allow_pickle=True)
    embeddings = payload["embeddings"]
    if "metadata_json" in payload:
        metadata_payload = json.loads(str(payload["metadata_json"].item()))
    else:
        metadata_payload = payload["metadata"].tolist()
    metadata = tuple(_normalize_meta(row) for row in metadata_payload)
    if int(embeddings.shape[0]) != len(metadata):
        raise ProtocolError(f"Embedding row count does not match metadata count: {path}")
    return MidogppFeatureCache(embeddings=embeddings, metadata=metadata, path=path)


def _normalize_meta(row: object) -> Mapping[str, object]:
    if not isinstance(row, Mapping):
        raise ProtocolError("MIDOG++ feature cache metadata rows must be mappings.")
    out = dict(row)
    if "center" not in out and "magnification" in out:
        out["center"] = out["magnification"]
    if "sample_id" not in out:
        out["sample_id"] = out.get("path", "")
    return out


def _resolve_summary_path(manifest_path: Path, raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = manifest_path.parent / path
        if candidate.exists():
            return candidate
    parts = path.parts
    if "summaries" in parts:
        suffix = Path(*parts[parts.index("summaries") :])
        for root in (manifest_path.parents[1], manifest_path.parents[2] if len(manifest_path.parents) > 2 else manifest_path.parent):
            candidate = root / suffix
            if candidate.exists():
                return candidate
    raise ProtocolError(f"MIDOG++ summary file not found: {raw}")


def _cache_path(root: Path | None, *, seed: int, split: str) -> Path:
    if root is None:
        raise ProtocolError("MIDOG++ test cache root is not configured.")
    base = Path(root) / f"seed{int(seed)}" / "embeddings"
    pt = base / f"{split}.pt"
    if pt.exists():
        return pt
    npz = base / f"{split}.npz"
    if npz.exists():
        return npz
    return pt


def _domain(row: Mapping[str, object]) -> str:
    for key in ("center", "magnification", "domain"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if key == "domain" and value.startswith("center_"):
            return value.split("_", 1)[1]
        return value.replace("x", "")
    raise ProtocolError(f"MIDOG++ metadata row lacks center/domain: {row}")


def _label(row: Mapping[str, object]) -> int:
    return int(row.get("label", 0))


def _concat_rows(chunks: Sequence[object]) -> object:
    try:
        import numpy as np  # type: ignore

        return np.concatenate([np.asarray(chunk, dtype=float) for chunk in chunks], axis=0)
    except ModuleNotFoundError:
        out: list[object] = []
        for chunk in chunks:
            out.extend(chunk)  # type: ignore[arg-type]
        return out


def _take_rows(values: object, indices: Sequence[int]) -> object:
    try:
        import numpy as np  # type: ignore

        return np.asarray(values)[list(indices)]
    except ModuleNotFoundError:
        return [values[int(idx)] for idx in indices]  # type: ignore[index]


def _row_dim(values: object) -> int:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(values)
        if arr.ndim != 2:
            raise ProtocolError(f"Expected 2D embeddings, got shape={arr.shape}.")
        return int(arr.shape[1])
    except ModuleNotFoundError:
        first = values[0]  # type: ignore[index]
        return len(first)  # type: ignore[arg-type]


def _identity_transform(candidate: MidogppCandidate, class_label: int, values: object) -> object:
    _ = candidate, class_label
    return values
