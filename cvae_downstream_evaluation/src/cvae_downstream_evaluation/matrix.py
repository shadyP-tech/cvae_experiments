"""Workstation matrix builder for downstream synthetic-only utility.

This module is intentionally a consumer of frozen support-run artifacts. It
loads existing embedding caches and CVAE expert checkpoints, generates
synthetic embeddings, trains the locked downstream classifier, and appends
rows to ``all_expert_downstream_matrix.csv``. It does not retrain experts,
regenerate foundation embeddings, or mutate routing artifacts.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .classifiers import ClassifierSpec, classifier_grid_hash
from .downstream import (
    CandidateDownstreamRow,
    balanced_accuracy,
    fit_locked_logistic_classifier,
    macro_f1,
    write_matrix_schema,
)
from .generation import allocate_equal_total_ensemble_budget
from .protocol import ArtifactSyncError, LockedV1Config, ProtocolError
from .routing import SupportSelectionUnit
from .schemas import (
    ALL_EXPERT_DOWNSTREAM_COLUMNS,
    ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY,
    ENSEMBLE_EXPERT_ID,
    MATRIX_SCHEMA_VERSION,
    METHOD_BASELINE_ROW_TYPE,
    NEGATIVE_CONTROL_GENERATION_MODE,
    PRIMARY_BUDGET_PER_CLASS,
    PRIMARY_GENERATION_MODE,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
)
from .schemas.classifier_tuning import SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS
from .source_inner_classifier_tuning import (
    SourceInnerClassifierFold,
    SourceInnerClassifierSelectionResult,
    select_classifier_spec_source_inner_lodo,
)
from .splits import assert_disjoint_ids
from .utility_matrix import assert_diagnostic_matrix_path, diagnostic_matrix_path


@dataclass(frozen=True)
class SupportRunArtifacts:
    experiment_seed: int
    run_dir: Path
    train_cache: Path
    test_cache: Path
    samples_manifest: Path
    expert_checkpoints_manifest: Path
    config_resolved: Path
    split_manifest: Path
    support_selection_path: Path


@dataclass(frozen=True)
class EmbeddingCache:
    embeddings: Any
    metadata: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class MatrixBuildLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None


def discover_support_run_artifacts(
    *,
    config: LockedV1Config,
    repo_root: Path,
) -> tuple[SupportRunArtifacts, ...]:
    """Discover the frozen support-run directories behind selection CSVs."""

    support_paths = sorted(Path(p) for p in glob.glob(str(repo_root / config.support_selection_glob)))
    if not support_paths:
        raise ProtocolError(f"No support selection artifacts matched: {config.support_selection_glob}")

    discovered: list[SupportRunArtifacts] = []
    for support_path in support_paths:
        run_dir = support_path.parent.parent
        config_resolved = run_dir / "config_resolved.yaml"
        seed = _experiment_seed_from_run(run_dir, config_resolved)
        artifact = SupportRunArtifacts(
            experiment_seed=seed,
            run_dir=run_dir,
            train_cache=run_dir / "embeddings" / "train.pt",
            test_cache=run_dir / "embeddings" / "test.pt",
            samples_manifest=run_dir / "manifests" / "samples.csv",
            expert_checkpoints_manifest=run_dir / "checkpoints" / "expert_checkpoints.json",
            config_resolved=config_resolved,
            split_manifest=run_dir / "reports" / "support_response_split_manifest.csv",
            support_selection_path=support_path,
        )
        _assert_support_artifact_files_exist(artifact)
        discovered.append(artifact)
    return tuple(sorted(discovered, key=lambda item: item.experiment_seed))


def materialize_downstream_manifests(
    *,
    artifacts: Sequence[SupportRunArtifacts],
    artifacts_root: Path,
) -> None:
    """Write downstream-local manifests derived from support-run artifacts."""

    manifest_dir = artifacts_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for artifact in artifacts:
        dimensions = _read_support_run_dimensions(artifact.config_resolved)
        checkpoints = _read_expert_checkpoint_manifest(artifact.expert_checkpoints_manifest)
        for expert_domain, checkpoint in sorted(checkpoints.items(), key=lambda item: int(item[0])):
            checkpoint_rows.append(
                {
                    "experiment_seed": artifact.experiment_seed,
                    "expert_domain": expert_domain,
                    "checkpoint_path": str(checkpoint),
                    "run_dir": str(artifact.run_dir),
                    "input_dim": dimensions["input_dim"],
                    "hidden_dim": dimensions["hidden_dim"],
                    "latent_dim": dimensions["latent_dim"],
                    "backend": "legacy_domain_cvae_identity_projection",
                }
            )
        for split, cache_path in (("train", artifact.train_cache), ("test", artifact.test_cache)):
            cache_rows.append(
                {
                    "experiment_seed": artifact.experiment_seed,
                    "split": split,
                    "cache_path": str(cache_path),
                    "samples_manifest": str(artifact.samples_manifest),
                    "run_dir": str(artifact.run_dir),
                }
            )
        provenance_rows.append(
            {
                "experiment_seed": artifact.experiment_seed,
                "run_dir": str(artifact.run_dir),
                "config_resolved_path": str(artifact.config_resolved),
                "expert_checkpoints_manifest": str(artifact.expert_checkpoints_manifest),
                "split_manifest_path": str(artifact.split_manifest),
                "samples_manifest": str(artifact.samples_manifest),
                "input_dim": dimensions["input_dim"],
                "hidden_dim": dimensions["hidden_dim"],
                "latent_dim": dimensions["latent_dim"],
                "feature_extractor_checkpoint": dimensions["feature_extractor_checkpoint"],
                "backend": "legacy_domain_cvae_identity_projection",
                "source_experts_frozen": "true",
            }
        )
        split_rows.extend(_read_split_manifest_rows(artifact))

    _write_dict_csv(manifest_dir / "expert_checkpoints.csv", checkpoint_rows)
    _write_dict_csv(manifest_dir / "embedding_cache_manifest.csv", cache_rows)
    _write_dict_csv(manifest_dir / "expert_provenance.csv", provenance_rows)
    _write_dict_csv(manifest_dir / "split_manifest.csv", split_rows)
    (manifest_dir / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "direct_support_nelbo_downstream_protocol_manifest_v1",
                "source": "frozen_support_run_artifacts",
                "experiment_seeds": [artifact.experiment_seed for artifact in artifacts],
                "source_experts_frozen": True,
                "target_expert_exclusion": "candidate experts are all centers except heldout center",
                "support_eval_exclusion": "matrix target eval pools exclude union of configured support sample_ids",
                "downstream_oracle_role": "diagnostic_only_single_expert_primary_rows",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_all_expert_downstream_matrix(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    limits: MatrixBuildLimits = MatrixBuildLimits(),
    output_path: Path | None = None,
    diagnostic_output: bool = False,
    source_inner_classifier_specs: Sequence[ClassifierSpec] | None = None,
    source_inner_classifier_tuning_path: Path | None = None,
) -> Path:
    """Build or resume an all-candidate downstream utility matrix."""

    if output_path is not None:
        matrix_path = Path(output_path)
    elif source_inner_classifier_specs is not None:
        grid_hash = classifier_grid_hash(source_inner_classifier_specs)
        matrix_path = artifacts_root / "tables" / f"source_inner_classifier_tuned_{grid_hash}_downstream_matrix.csv"
    elif diagnostic_output:
        matrix_path = diagnostic_matrix_path(artifacts_root)
    else:
        matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    if diagnostic_output or "diagnostic_downstream_utility" in matrix_path.name:
        assert_diagnostic_matrix_path(matrix_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_schema(matrix_path.with_suffix(".schema.json"))

    artifacts = discover_support_run_artifacts(config=config, repo_root=repo_root)
    artifacts = _limit_artifacts(artifacts, limits.experiment_seeds)
    completed = _read_completed_keys(matrix_path) if resume else set()

    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    units_by_seed = _units_by_seed(support_units)
    tuning_artifact_path = source_inner_classifier_tuning_path or (
        artifacts_root / "tables" / (
            f"source_inner_classifier_tuning_{classifier_grid_hash(source_inner_classifier_specs)}.csv"
            if source_inner_classifier_specs is not None
            else "source_inner_classifier_tuning.csv"
        )
    )

    for artifact in artifacts:
        seed_units = units_by_seed.get(int(artifact.experiment_seed), ())
        if not seed_units:
            raise ProtocolError(f"No support-selection units found for experiment_seed={artifact.experiment_seed}")

        samples = _read_samples_manifest(artifact.samples_manifest)
        train_records = _records_for_split(samples, "train")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(artifact.train_cache, train_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(artifact.test_cache, test_records, repo_root=repo_root)
        dimensions = _read_support_run_dimensions(artifact.config_resolved)
        bank = LegacyDomainCvaeExpertBank.from_artifact(
            artifact=artifact,
            dimensions=dimensions,
            repo_root=repo_root,
            device=device,
        )

        for heldout_center in selected_heldout_centers:
            heldout = str(heldout_center)
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            if heldout not in {str(c) for c in config.candidate_domains}:
                raise ProtocolError(f"Unknown heldout center requested: {heldout}")
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            if not target_pool.eval_indices:
                raise ProtocolError(
                    f"Target eval pool is empty after support exclusion for seed={artifact.experiment_seed}, "
                    f"heldout_center={heldout}."
                )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            label_values = tuple(sorted(set(target_labels).union({0, 1})))
            if label_values != (0, 1):
                raise ProtocolError(f"Locked v1 expects binary labels 0/1, got {label_values}")

            selected_specs_by_seed: dict[int, ClassifierSpec] = {}
            if source_inner_classifier_specs is not None:
                for classifier_seed in selected_classifier_seeds:
                    selection = _select_source_inner_classifier_spec(
                        experiment_seed=int(artifact.experiment_seed),
                        heldout_center=heldout,
                        classifier_seed=int(classifier_seed),
                        candidate_specs=source_inner_classifier_specs,
                        allowed_centers=candidates,
                        train_cache=train_cache,
                        test_cache=test_cache,
                    )
                    selected_specs_by_seed[int(classifier_seed)] = selection.selected_spec
                    _append_source_inner_classifier_tuning_rows(
                        tuning_artifact_path,
                        selection.to_artifact_rows(),
                    )

            for candidate in candidates:
                reference_pools = build_class_reference_pools(
                    train_cache=train_cache,
                    candidate_expert=candidate,
                    required_labels=label_values,
                )
                for generation_mode, budget in _single_expert_modes_and_budgets(config):
                    for generation_seed in selected_generation_seeds:
                        for classifier_seed in selected_classifier_seeds:
                            row = _single_expert_row(
                                experiment_seed=artifact.experiment_seed,
                                heldout_center=heldout,
                                candidate_expert=candidate,
                                generation_mode=generation_mode,
                                budget_per_class=int(budget),
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                                target_eval_pool=target_pool,
                                target_labels=target_labels,
                                label_values=label_values,
                                reference_pools=reference_pools,
                                train_cache=train_cache,
                                test_cache=test_cache,
                                bank=bank,
                                classifier_spec=selected_specs_by_seed.get(int(classifier_seed)),
                            )
                            if resume and row.primary_key() in completed:
                                continue
                            append_matrix_row(matrix_path, row)
                            completed.add(row.primary_key())

            for generation_seed in selected_generation_seeds:
                for classifier_seed in selected_classifier_seeds:
                    row = _ensemble_row(
                        experiment_seed=artifact.experiment_seed,
                        heldout_center=heldout,
                        candidate_experts=candidates,
                        generation_seed=int(generation_seed),
                        classifier_seed=int(classifier_seed),
                        target_eval_pool=target_pool,
                        target_labels=target_labels,
                        label_values=label_values,
                        train_cache=train_cache,
                        test_cache=test_cache,
                        bank=bank,
                        classifier_spec=selected_specs_by_seed.get(int(classifier_seed)),
                    )
                    if resume and row.primary_key() in completed:
                        continue
                    append_matrix_row(matrix_path, row)
                    completed.add(row.primary_key())

    return matrix_path


@dataclass(frozen=True)
class TargetEvalPool:
    eval_indices: tuple[int, ...]
    excluded_support_sample_ids: tuple[str, ...]
    target_eval_pool_id: str


def build_target_eval_pool(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_sizes: Sequence[int],
    support_seeds: Sequence[int],
) -> TargetEvalPool:
    """Exclude the union of configured target support samples by sample_id."""

    target_indices = tuple(
        idx for idx, row in enumerate(test_metadata) if str(_domain(row)) == str(heldout_center)
    )
    labels_by_index = {idx: _label(test_metadata[idx]) for idx in target_indices}
    support_ids: set[str] = set()
    for support_size in support_sizes:
        for support_seed in support_seeds:
            split = _make_support_eval_split(
                target_domain=int(heldout_center),
                target_indices=target_indices,
                labels_by_index=labels_by_index,
                support_size=int(support_size),
                sampling_policy="random",
                support_seed=int(support_seed),
            )
            support_ids.update(str(_sample_id(test_metadata[idx])) for idx in split.support_indices)
    eval_indices = tuple(
        idx for idx in target_indices if str(_sample_id(test_metadata[idx])) not in support_ids
    )
    assert_disjoint_ids(support_ids, (str(_sample_id(test_metadata[idx])) for idx in eval_indices))
    digest = hashlib.sha256("|".join(sorted(support_ids)).encode("utf-8")).hexdigest()[:12]
    return TargetEvalPool(
        eval_indices=eval_indices,
        excluded_support_sample_ids=tuple(sorted(support_ids)),
        target_eval_pool_id=f"target{heldout_center}_exclude_configured_support_union_{digest}",
    )


def build_class_reference_pools(
    *,
    train_cache: EmbeddingCache,
    candidate_expert: str,
    required_labels: Sequence[int],
) -> dict[int, Any]:
    pools: dict[int, Any] = {}
    for label in required_labels:
        idxs = [
            idx
            for idx, row in enumerate(train_cache.metadata)
            if str(_domain(row)) == str(candidate_expert) and _label(row) == int(label)
        ]
        pools[int(label)] = train_cache.embeddings[idxs] if idxs else None
    return pools


class LegacyDomainCvaeExpertBank:
    """Legacy per-domain CVAE backend with identity projection."""

    def __init__(
        self,
        *,
        expert_checkpoints: Mapping[str, Path],
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        repo_root: Path,
        device: str,
    ) -> None:
        torch, CVAEExpert, load_model_checkpoint = _torch_model_imports(repo_root)
        self.torch = torch
        self.device = _resolve_torch_device(torch, device)
        self.latent_dim = int(latent_dim)
        self.models: dict[int, Any] = {}
        for expert_domain, checkpoint in sorted(expert_checkpoints.items(), key=lambda item: int(item[0])):
            loaded = load_model_checkpoint(Path(checkpoint), map_location=self.device)
            model = CVAEExpert(int(input_dim), int(hidden_dim), int(latent_dim)).to(self.device)
            model.load_state_dict(loaded.model_state_dict)
            if int(getattr(model, "metadata_dim", 0)) != 0 or int(getattr(model, "class_condition_dim", 0)) != 0:
                raise ProtocolError(
                    "Legacy downstream backend only supports unconditioned domain CVAEs. "
                    "Hybrid/projected experts need an explicit matching projection backend."
                )
            model.eval()
            self.models[int(expert_domain)] = model

    @classmethod
    def from_artifact(
        cls,
        *,
        artifact: SupportRunArtifacts,
        dimensions: Mapping[str, object],
        repo_root: Path,
        device: str,
    ) -> "LegacyDomainCvaeExpertBank":
        return cls(
            expert_checkpoints=_read_expert_checkpoint_manifest(artifact.expert_checkpoints_manifest),
            input_dim=int(dimensions["input_dim"]),
            hidden_dim=int(dimensions["hidden_dim"]),
            latent_dim=int(dimensions["latent_dim"]),
            repo_root=repo_root,
            device=device,
        )

    def project(self, domain: int, x: Any) -> Any:
        _ = int(domain)
        return x.to(self.device)

    def generate_from_reference(self, domain: int, x_ref: Any, n_samples: int, seed: int) -> Any:
        torch = self.torch
        model = self.models[int(domain)]
        refs = x_ref.to(self.device)
        if int(refs.shape[0]) <= 0:
            raise ProtocolError(f"Empty reference pool for expert {domain}")
        idx_gen = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randint(int(refs.shape[0]), (int(n_samples),), generator=idx_gen, device="cpu").to(self.device)
        xb = refs[idx]
        gen = _torch_generator(torch, self.device, int(seed) + 104729)
        with torch.no_grad():
            mu, logvar = model.encode(xb)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape, generator=gen, device=self.device, dtype=std.dtype)
            z = mu + eps * std
            return model.decode(z).detach().cpu()

    def sample_prior(self, domain: int, n_samples: int, seed: int) -> Any:
        torch = self.torch
        model = self.models[int(domain)]
        gen = _torch_generator(torch, self.device, int(seed))
        with torch.no_grad():
            z = torch.randn((int(n_samples), self.latent_dim), generator=gen, device=self.device)
            return model.decode(z).detach().cpu()


def _single_expert_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    generation_mode: str,
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
    target_eval_pool: TargetEvalPool,
    target_labels: Sequence[int],
    label_values: Sequence[int],
    reference_pools: Mapping[int, Any],
    train_cache: EmbeddingCache,
    test_cache: EmbeddingCache,
    bank: LegacyDomainCvaeExpertBank,
    classifier_spec: ClassifierSpec | None = None,
) -> CandidateDownstreamRow:
    _ = train_cache
    n_synthetic = int(budget_per_class) * len(label_values)
    base = {
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "candidate_expert": candidate_expert,
        "generation_mode": generation_mode,
        "budget_per_class": int(budget_per_class),
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "row_type": SINGLE_EXPERT_ROW_TYPE,
        "n_synthetic_train": n_synthetic,
        "n_target_eval": len(target_eval_pool.eval_indices),
        "target_eval_pool_id": target_eval_pool.target_eval_pool_id,
        "candidate_experts_hash": SINGLE_EXPERT_HASH,
    }
    missing = [int(label) for label in label_values if reference_pools.get(int(label)) is None]
    if generation_mode == PRIMARY_GENERATION_MODE and missing:
        return CandidateDownstreamRow(
            **base,
            bacc=math.nan,
            macro_f1=math.nan,
            status="failed_empty_reference_pool",
            error_message=f"Empty source-train reference pool for labels {missing}",
        )
    try:
        synthetic_embeddings, synthetic_labels = _generate_synthetic(
            bank=bank,
            candidate_expert=candidate_expert,
            generation_mode=generation_mode,
            budget_per_class=budget_per_class,
            generation_seed=generation_seed,
            label_values=label_values,
            reference_pools=reference_pools,
        )
        target_embeddings = bank.project(
            int(candidate_expert),
            test_cache.embeddings[list(target_eval_pool.eval_indices)],
        ).detach().cpu()
        prediction = fit_locked_logistic_classifier(
            _to_numpy(synthetic_embeddings),
            synthetic_labels,
            _to_numpy(target_embeddings),
            target_labels,
            classifier_seed=classifier_seed,
            classifier_spec=classifier_spec,
        )
        return CandidateDownstreamRow(
            **base,
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
        )
    except Exception as exc:
        return CandidateDownstreamRow(
            **base,
            bacc=math.nan,
            macro_f1=math.nan,
            status=_failure_status(exc),
            error_message=str(exc),
        )


def _ensemble_row(
    *,
    experiment_seed: int,
    heldout_center: str,
    candidate_experts: Sequence[str],
    generation_seed: int,
    classifier_seed: int,
    target_eval_pool: TargetEvalPool,
    target_labels: Sequence[int],
    label_values: Sequence[int],
    train_cache: EmbeddingCache,
    test_cache: EmbeddingCache,
    bank: LegacyDomainCvaeExpertBank,
    classifier_spec: ClassifierSpec | None = None,
) -> CandidateDownstreamRow:
    candidate_hash = hash_candidate_experts(candidate_experts)
    base = {
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "candidate_expert": ENSEMBLE_EXPERT_ID,
        "generation_mode": PRIMARY_GENERATION_MODE,
        "budget_per_class": PRIMARY_BUDGET_PER_CLASS,
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "row_type": METHOD_BASELINE_ROW_TYPE,
        "n_synthetic_train": PRIMARY_BUDGET_PER_CLASS * len(label_values),
        "n_target_eval": len(target_eval_pool.eval_indices),
        "target_eval_pool_id": target_eval_pool.target_eval_pool_id,
        "candidate_experts_hash": candidate_hash,
    }
    try:
        import numpy as np  # type: ignore

        allocation = allocate_equal_total_ensemble_budget(
            total_per_class=PRIMARY_BUDGET_PER_CLASS,
            candidate_experts=candidate_experts,
        )
        probabilities = []
        for expert in sorted(str(v) for v in candidate_experts):
            reference_pools = build_class_reference_pools(
                train_cache=train_cache,
                candidate_expert=expert,
                required_labels=label_values,
            )
            missing = [int(label) for label in label_values if reference_pools.get(int(label)) is None]
            if missing:
                return CandidateDownstreamRow(
                    **base,
                    bacc=math.nan,
                    macro_f1=math.nan,
                    status="failed_empty_reference_pool",
                    error_message=f"Expert {expert} has empty source-train reference pool for labels {missing}",
                )
            synthetic_embeddings, synthetic_labels = _generate_synthetic(
                bank=bank,
                candidate_expert=expert,
                generation_mode=PRIMARY_GENERATION_MODE,
                budget_per_class=int(allocation[expert]),
                generation_seed=generation_seed,
                label_values=label_values,
                reference_pools=reference_pools,
            )
            target_embeddings = bank.project(
                int(expert),
                test_cache.embeddings[list(target_eval_pool.eval_indices)],
            ).detach().cpu()
            prediction = fit_locked_logistic_classifier(
                _to_numpy(synthetic_embeddings),
                synthetic_labels,
                _to_numpy(target_embeddings),
                target_labels,
                classifier_seed=classifier_seed,
                classifier_spec=classifier_spec,
            )
            if tuple(prediction.classes) != tuple(int(v) for v in label_values):
                raise ProtocolError(
                    f"Ensemble class-order mismatch for expert {expert}: "
                    f"{prediction.classes} != {tuple(label_values)}"
                )
            probabilities.append(np.asarray(prediction.probabilities, dtype=float))
        averaged = np.mean(np.stack(probabilities, axis=0), axis=0)
        bacc, macro, auroc, auprc = _score_probabilities(
            probabilities=averaged,
            class_order=tuple(int(v) for v in label_values),
            target_labels=target_labels,
        )
        return CandidateDownstreamRow(
            **base,
            bacc=bacc,
            macro_f1=macro,
            auroc=auroc,
            auprc=auprc,
        )
    except Exception as exc:
        return CandidateDownstreamRow(
            **base,
            bacc=math.nan,
            macro_f1=math.nan,
            status=_failure_status(exc),
            error_message=str(exc),
        )


def _generate_synthetic(
    *,
    bank: LegacyDomainCvaeExpertBank,
    candidate_expert: str,
    generation_mode: str,
    budget_per_class: int,
    generation_seed: int,
    label_values: Sequence[int],
    reference_pools: Mapping[int, Any],
) -> tuple[Any, list[int]]:
    torch = bank.torch
    if generation_mode == PRIMARY_GENERATION_MODE:
        chunks = []
        labels: list[int] = []
        for label in label_values:
            refs = reference_pools[int(label)]
            chunks.append(
                bank.generate_from_reference(
                    int(candidate_expert),
                    refs,
                    int(budget_per_class),
                    seed=int(generation_seed) + int(label),
                )
            )
            labels.extend([int(label)] * int(budget_per_class))
        return torch.cat(chunks, dim=0), labels
    if generation_mode == NEGATIVE_CONTROL_GENERATION_MODE:
        total = int(budget_per_class) * len(label_values)
        embeddings = bank.sample_prior(int(candidate_expert), total, int(generation_seed))
        labels = []
        for label in label_values:
            labels.extend([int(label)] * int(budget_per_class))
        return embeddings, labels
    raise ProtocolError(f"Unknown generation mode: {generation_mode}")


def _select_source_inner_classifier_spec(
    *,
    experiment_seed: int,
    heldout_center: str,
    classifier_seed: int,
    candidate_specs: Sequence[ClassifierSpec],
    allowed_centers: Sequence[str],
    train_cache: EmbeddingCache,
    test_cache: EmbeddingCache,
) -> SourceInnerClassifierSelectionResult:
    seeded_specs = tuple(
        ClassifierSpec(
            C=spec.C,
            penalty=spec.penalty,
            solver=spec.solver,
            max_iter=spec.max_iter,
            class_weight=spec.class_weight,
            random_state=int(classifier_seed),
            l1_ratio=spec.l1_ratio,
            threshold_policy=spec.threshold_policy,
            scaler_fit=spec.scaler_fit,
            family=spec.family,
        )
        for spec in candidate_specs
    )
    folds = _source_inner_classifier_folds(
        heldout_center=heldout_center,
        allowed_centers=allowed_centers,
        train_cache=train_cache,
        test_cache=test_cache,
    )
    return select_classifier_spec_source_inner_lodo(
        outer_target_center=heldout_center,
        folds=folds,
        candidate_specs=seeded_specs,
        experiment_seed=int(experiment_seed),
        classifier_seed=int(classifier_seed),
        selection_metric="bacc",
    )


def _source_inner_classifier_folds(
    *,
    heldout_center: str,
    allowed_centers: Sequence[str],
    train_cache: EmbeddingCache,
    test_cache: EmbeddingCache,
) -> tuple[SourceInnerClassifierFold, ...]:
    allowed = tuple(str(center) for center in allowed_centers)
    folds: list[SourceInnerClassifierFold] = []
    for pseudo_target in allowed:
        train_indices = [
            idx
            for idx, row in enumerate(train_cache.metadata)
            if _domain(row) in set(allowed) and _domain(row) != str(pseudo_target)
        ]
        validation_indices = [
            idx
            for idx, row in enumerate(test_cache.metadata)
            if _domain(row) == str(pseudo_target)
        ]
        if not train_indices:
            raise ProtocolError(
                f"No source-inner classifier training rows for heldout={heldout_center}, "
                f"pseudo_target={pseudo_target}."
            )
        if not validation_indices:
            raise ProtocolError(
                f"No source-inner classifier validation rows for heldout={heldout_center}, "
                f"pseudo_target={pseudo_target}."
            )
        train_labels = [_label(train_cache.metadata[idx]) for idx in train_indices]
        validation_labels = [_label(test_cache.metadata[idx]) for idx in validation_indices]
        if sorted(set(train_labels)) != [0, 1]:
            raise ProtocolError(
                f"Source-inner classifier training fold lacks binary labels for pseudo_target={pseudo_target}."
            )
        if sorted(set(validation_labels)) != [0, 1]:
            raise ProtocolError(
                f"Source-inner classifier validation fold lacks binary labels for pseudo_target={pseudo_target}."
            )
        folds.append(
            SourceInnerClassifierFold(
                pseudo_target_center=str(pseudo_target),
                train_centers=tuple(center for center in allowed if center != str(pseudo_target)),
                train_embeddings=_to_numpy(train_cache.embeddings[train_indices]),
                train_labels=train_labels,
                validation_embeddings=_to_numpy(test_cache.embeddings[validation_indices]),
                validation_labels=validation_labels,
            )
        )
    return tuple(folds)


def _append_source_inner_classifier_tuning_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SOURCE_INNER_CLASSIFIER_TUNING_COLUMNS})


def append_matrix_row(path: Path, row: CandidateDownstreamRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_matrix_schema(path.with_suffix(".schema.json"))
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ALL_EXPERT_DOWNSTREAM_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(row.to_csv_row())


def hash_candidate_experts(candidate_experts: Sequence[str]) -> str:
    payload = "|".join(sorted(str(v) for v in candidate_experts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _score_probabilities(
    *,
    probabilities: Any,
    class_order: Sequence[int],
    target_labels: Sequence[int],
) -> tuple[float, float, float, float]:
    import numpy as np  # type: ignore
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

    proba = np.asarray(probabilities, dtype=float)
    y_true = [int(v) for v in target_labels]
    pred_idx = np.argmax(proba, axis=1)
    y_pred = [int(class_order[int(i)]) for i in pred_idx.tolist()]
    auroc = math.nan
    auprc = math.nan
    if tuple(int(v) for v in class_order) == (0, 1) and proba.shape[1] == 2:
        try:
            auroc = float(roc_auc_score(y_true, proba[:, 1]))
        except ValueError:
            auroc = math.nan
        try:
            auprc = float(average_precision_score(y_true, proba[:, 1]))
        except ValueError:
            auprc = math.nan
    return balanced_accuracy(y_true, y_pred), macro_f1(y_true, y_pred), auroc, auprc


def _read_completed_keys(path: Path) -> set[tuple[object, ...]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY if field not in (reader.fieldnames or ())]
        if missing:
            raise ProtocolError(f"Existing matrix is missing primary-key columns: {missing}")
        return {
            tuple(_primary_key_value(field, row.get(field, "")) for field in ALL_EXPERT_DOWNSTREAM_PRIMARY_KEY)
            for row in reader
        }


def _primary_key_value(field: str, raw: object) -> object:
    if field in {"experiment_seed", "budget_per_class", "generation_seed", "classifier_seed"}:
        return int(raw)
    return str(raw)


def _single_expert_modes_and_budgets(config: LockedV1Config) -> tuple[tuple[str, int], ...]:
    pairs = [(PRIMARY_GENERATION_MODE, int(budget)) for budget in config.diagnostic_budgets_per_class]
    pairs.append((NEGATIVE_CONTROL_GENERATION_MODE, int(config.primary_budget_per_class)))
    return tuple(pairs)


def _limit_artifacts(
    artifacts: Sequence[SupportRunArtifacts],
    experiment_seeds: Sequence[int] | None,
) -> tuple[SupportRunArtifacts, ...]:
    if experiment_seeds is None:
        return tuple(artifacts)
    allowed = {int(seed) for seed in experiment_seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.experiment_seed) in allowed)


def _units_by_seed(units: Sequence[SupportSelectionUnit]) -> dict[int, tuple[SupportSelectionUnit, ...]]:
    grouped: dict[int, list[SupportSelectionUnit]] = {}
    for unit in units:
        grouped.setdefault(int(unit.experiment_seed), []).append(unit)
    return {seed: tuple(values) for seed, values in grouped.items()}


def _assert_support_artifact_files_exist(artifact: SupportRunArtifacts) -> None:
    missing = [
        path
        for path in (
            artifact.train_cache,
            artifact.test_cache,
            artifact.samples_manifest,
            artifact.expert_checkpoints_manifest,
            artifact.config_resolved,
            artifact.split_manifest,
            artifact.support_selection_path,
        )
        if not path.exists()
    ]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing)
        raise ArtifactSyncError(
            "Missing frozen support-run artifacts required for downstream matrix build:\n"
            f"{preview}"
        )


def _read_expert_checkpoint_manifest(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Expert checkpoint manifest must be a JSON object: {path}")
    out: dict[str, Path] = {}
    for raw_key, raw_value in payload.items():
        expert = str(_parse_expert_domain(raw_key))
        checkpoint = Path(str(raw_value))
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        elif not checkpoint.exists():
            local_checkpoint = path.parent / checkpoint.name
            if local_checkpoint.exists():
                checkpoint = local_checkpoint
        if not checkpoint.exists():
            raise ArtifactSyncError(f"Expert checkpoint for domain {expert} not found: {checkpoint}")
        out[expert] = checkpoint
    if not out:
        raise ProtocolError(f"No expert checkpoints found in {path}")
    return out


def _read_support_run_dimensions(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        features = loaded.get("features", {}) if isinstance(loaded, Mapping) else {}
        model = loaded.get("model", {}) if isinstance(loaded, Mapping) else {}
        return {
            "input_dim": int(features.get("embedding_dim")),
            "hidden_dim": int(model.get("hidden_dim")),
            "latent_dim": int(model.get("latent_dim")),
            "feature_extractor_checkpoint": str(features.get("feature_extractor_checkpoint", "")),
        }
    except Exception:
        return {
            "input_dim": _regex_int(text, r"embedding_dim:\s*(\d+)"),
            "hidden_dim": _regex_int(text, r"hidden_dim:\s*(\d+)"),
            "latent_dim": _regex_int(text, r"latent_dim:\s*(\d+)"),
            "feature_extractor_checkpoint": _regex_str(text, r"feature_extractor_checkpoint:\s*([^\n]+)"),
        }


def _read_samples_manifest(path: Path) -> tuple[Mapping[str, object], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ProtocolError(f"samples.csv is empty: {path}")
    for row in rows:
        if not str(row.get("sample_id", "")).strip():
            raise ProtocolError(f"samples.csv row missing stable sample_id in {path}")
    return tuple(rows)


def _read_split_manifest_rows(artifact: SupportRunArtifacts) -> list[dict[str, object]]:
    with artifact.split_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            enriched = {
                "experiment_seed": artifact.experiment_seed,
                "support_run_dir": str(artifact.run_dir),
                "support_selection_path": str(artifact.support_selection_path),
            }
            enriched.update(dict(row))
            rows.append(enriched)
    if not rows:
        raise ProtocolError(f"Support split manifest is empty: {artifact.split_manifest}")
    return rows


def _records_for_split(
    samples: Sequence[Mapping[str, object]],
    split: str,
) -> tuple[Mapping[str, object], ...]:
    rows = tuple(row for row in samples if str(row.get("split", "")).strip().lower() == split)
    if not rows:
        raise ProtocolError(f"No samples found for split={split!r}")
    return rows


def _load_embedding_cache(
    path: Path,
    split_records: Sequence[Mapping[str, object]],
    *,
    repo_root: Path,
) -> EmbeddingCache:
    safe_torch_load = _safe_torch_load_import(repo_root)
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "embeddings" not in payload:
        raise ProtocolError(f"Embedding cache has unexpected structure: {path}")
    embeddings = payload["embeddings"]
    metadata_raw = payload.get("metadata")
    metadata = [dict(row) for row in metadata_raw] if isinstance(metadata_raw, (list, tuple)) else []
    n = int(embeddings.shape[0])
    if metadata and len(metadata) != n:
        raise ProtocolError(f"Embedding metadata length mismatch in {path}: {len(metadata)} != {n}")
    if len(split_records) != n and not metadata:
        raise ProtocolError(
            f"samples.csv split row count does not match embedding cache {path}: "
            f"{len(split_records)} != {n}"
        )
    if not metadata:
        metadata = [dict(row) for row in split_records]
    elif len(split_records) == n:
        for idx, manifest_row in enumerate(split_records):
            for key, value in manifest_row.items():
                metadata[idx].setdefault(key, value)
    for idx, row in enumerate(metadata):
        if not str(row.get("sample_id", "")).strip() and idx < len(split_records):
            row["sample_id"] = split_records[idx].get("sample_id", "")
        if not str(row.get("sample_id", "")).strip():
            raise ProtocolError(f"Embedding cache metadata lacks sample_id at row {idx}: {path}")
    return EmbeddingCache(embeddings=embeddings, metadata=tuple(metadata))


def _experiment_seed_from_run(run_dir: Path, config_resolved: Path) -> int:
    if config_resolved.exists():
        text = config_resolved.read_text(encoding="utf-8")
        match = re.search(r"^seed:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        if match:
            return int(match.group(1))
    match = re.search(r"seed(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    raise ProtocolError(f"Could not infer experiment seed from support run: {run_dir}")


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolError(f"Refusing to write empty manifest: {path}")
    columns = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _safe_torch_load_import(repo_root: Path):
    cvae_testing_root = repo_root / "cvae_testing"
    if str(cvae_testing_root) not in sys.path:
        sys.path.insert(0, str(cvae_testing_root))
    from src.torch_utils import safe_torch_load  # type: ignore

    return safe_torch_load


def _torch_model_imports(repo_root: Path):
    cvae_testing_root = repo_root / "cvae_testing"
    if str(cvae_testing_root) not in sys.path:
        sys.path.insert(0, str(cvae_testing_root))
    import torch  # type: ignore
    from src.models.cvae_expert import CVAEExpert  # type: ignore
    from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

    return torch, CVAEExpert, load_model_checkpoint


def _resolve_torch_device(torch: Any, requested: str) -> Any:
    raw = str(requested or "auto").strip().lower()
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _torch_generator(torch: Any, device: Any, seed: int) -> Any:
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return torch.Generator(device="cpu").manual_seed(int(seed))


def _make_support_eval_split(**kwargs: Any) -> Any:
    try:
        from src.eval.evaluators.support_set_calibration import make_support_eval_split  # type: ignore

        return make_support_eval_split(**kwargs)
    except ModuleNotFoundError:
        return _local_make_support_eval_split(**kwargs)


@dataclass(frozen=True)
class _LocalSplit:
    support_indices: tuple[int, ...]


def _local_make_support_eval_split(
    *,
    target_domain: int,
    target_indices: Sequence[int],
    labels_by_index: Mapping[int, int],
    support_size: int,
    sampling_policy: str,
    support_seed: int,
) -> _LocalSplit:
    _ = labels_by_index
    if str(sampling_policy) != "random":
        raise ProtocolError("Local fallback support split supports only random sampling.")
    import random

    split_seed = int(support_seed) + int(target_domain) * 1009
    indices = sorted(int(i) for i in target_indices)
    rng = random.Random(split_seed)
    rng.shuffle(indices)
    return _LocalSplit(support_indices=tuple(indices[: int(support_size)]))


def _domain(row: Mapping[str, object]) -> str:
    for key in ("magnification", "center", "domain"):
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if key == "domain" and value.startswith("center_"):
            return value.split("_", 1)[1]
        return value.replace("x", "")
    raise ProtocolError(f"Metadata row lacks domain/center field: {row}")


def _label(row: Mapping[str, object]) -> int:
    return int(row.get("label", 0))


def _sample_id(row: Mapping[str, object]) -> str:
    value = str(row.get("sample_id", "")).strip()
    if not value:
        raise ProtocolError(f"Metadata row lacks sample_id: {row}")
    return value


def _parse_expert_domain(raw: object) -> int:
    text = str(raw).strip().lower()
    if text.startswith("expert_"):
        text = text[len("expert_") :]
    match = re.match(r"^(\d+)", text.replace("x", ""))
    if match is None:
        raise ProtocolError(f"Cannot parse expert domain from checkpoint key: {raw!r}")
    return int(match.group(1))


def _regex_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    if match is None:
        raise ProtocolError(f"Could not parse required integer using pattern: {pattern}")
    return int(match.group(1))


def _regex_str(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return str(match.group(1)).strip() if match else ""


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _failure_status(exc: Exception) -> str:
    if isinstance(exc, ArtifactSyncError):
        return "failed_missing_artifact"
    if "reference pool" in str(exc).lower():
        return "failed_empty_reference_pool"
    return "failed_metric_invalid"
