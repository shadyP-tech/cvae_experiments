"""C4.1 workstation runner helpers.

This module is the executable bridge for the heteroscedastic decoder
experiment. It retrains source-only PCA64 class-conditioned generators and
reuses locked support-router decisions only as immutable selected-expert IDs.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .c41_heteroscedastic import (
    GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    GENERATION_MODE_POSTERIOR_DECODER_NOISE,
    GENERATOR_FAMILY_HETEROSCEDASTIC,
    GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL,
    build_source_train_reference_pools,
    c41_routing_provenance_fields,
    decoder_logvar_diagnostics_by_class,
    fit_source_train_pca_projection,
    generate_posterior_sampled_embeddings,
    labels_from_metadata,
)
from .downstream import (
    CandidateDownstreamRow,
    assert_duplicate_utility_contexts_consistent,
    fit_locked_logistic_classifier,
)
from .matrix import (
    EmbeddingCache,
    MatrixBuildLimits,
    SupportRunArtifacts,
    TargetEvalPool,
    _domain,
    _label,
    _limit_artifacts,
    _load_embedding_cache,
    _read_completed_keys,
    _read_samples_manifest,
    _read_support_run_dimensions,
    _records_for_split,
    _resolve_torch_device,
    _safe_torch_load_import,
    _to_numpy,
    append_matrix_row,
    build_target_eval_pool,
    discover_support_run_artifacts,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import (
    SupportSelectionUnit,
    assert_target_excluded,
    parse_expert_scores_json,
    read_support_selection_units,
)
from .schemas import (
    BASELINE_ROUTING_FAMILY_USED,
    BASELINE_SELECTED_EXPERT_IDS_SOURCE,
    C41_DELTA_SUMMARY_COLUMNS,
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
    PRIMARY_BUDGET_PER_CLASS,
    ROUTING_ALIGNMENT_COLUMNS,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
    SUPPORT_NELBO_METHOD,
)


C41_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_v1"
PLAIN_BASELINE_RETRAINED = "retrained_in_c41"
DECISION_SUCCESS = "SUCCESS"
DECISION_NO_UTILITY_GAIN = "NO_UTILITY_GAIN"
DECISION_VARIANCE_INFLATION = "HETEROSCEDASTIC_VARIANCE_INFLATION"
DECISION_RANK_INSTABILITY = "HETEROSCEDASTIC_ORACLE_RANK_INSTABILITY"
DECISION_PROTOCOL_FAILURE = "PROTOCOL_FAILURE_SELECTED_EXPERT_CHANGED"

_FORBIDDEN_SUPPORT_COLUMNS = {
    "target_eval_labels",
    "target_evaluation_labels",
    "target_eval_metrics",
    "target_test_metrics",
    "target_evaluation_nelbo",
    "downstream_oracle_expert",
    "eval_nelbo_by_expert_json",
    "eval_rank_by_expert_json",
    "oracle_eval_nelbo",
    "oracle_expert",
    "candidate_oracle_expert",
    "candidate_oracle_nelbo",
    "oracle_nelbo",
    "oracle_gap",
    "oracle_gap_pct",
    "mean_oracle_gap_pct",
    "top1_oracle_hit",
}


@dataclass(frozen=True)
class C41RunArtifacts:
    support: SupportRunArtifacts
    val_cache: Path


@dataclass(frozen=True)
class C41TrainingProfile:
    name: str
    hidden_dim: int
    latent_dim: int
    lr: float
    epochs: int
    patience: int
    batch_size: int
    pca_components: int = 64

    @property
    def matches_locked_hparams(self) -> int:
        return int(self.name == "full")


@dataclass(frozen=True)
class GeneratorContext:
    family: str
    mode: str
    checkpoint_path: Path
    plain_baseline_source: str
    plain_baseline_artifact_path: str
    plain_baseline_training_profile: str
    plain_baseline_matches_locked_hparams: int


def safe_support_selection_units_from_paths(
    paths: Iterable[Path],
    *,
    strict_forbidden_columns: bool = True,
    methods: Sequence[str] | None = None,
) -> list[SupportSelectionUnit]:
    """Load support-router decisions through a narrow C4.1-safe boundary."""

    resolved = tuple(sorted(Path(path) for path in paths))
    if strict_forbidden_columns:
        _assert_no_forbidden_support_columns(resolved)
    allowed_methods = tuple(methods) if methods is not None else (SUPPORT_NELBO_METHOD,)
    return read_support_selection_units(resolved, methods=allowed_methods)


def c41_training_profile_from_config(
    config_path: Path,
    *,
    profile: str,
) -> C41TrainingProfile:
    raw = _read_training_config(config_path)
    hidden_dim = int(raw.get("hidden_dim", 256))
    latent_dim = int(raw.get("latent_dim", 16))
    lr = float(raw.get("learning_rate", raw.get("lr", 1.0e-3)))
    batch_size = int(raw.get("batch_size", 128))
    if str(profile) == "smoke":
        return C41TrainingProfile(
            name="smoke",
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            epochs=min(int(raw.get("epochs", 25)), 2),
            patience=1,
            batch_size=min(batch_size, 64),
        )
    if str(profile) != "full":
        raise ProtocolError(f"Unknown C4.1 training profile: {profile}")
    return C41TrainingProfile(
        name="full",
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=lr,
        epochs=int(raw.get("epochs", 25)),
        patience=int(raw.get("patience", 5)),
        batch_size=batch_size,
    )


def _profile_for_support_config(base: C41TrainingProfile, support_config_path: Path) -> C41TrainingProfile:
    """Resolve model/training hparams from the frozen support-run config."""

    raw = _read_training_config(support_config_path)
    if base.name == "smoke":
        return replace(
            base,
            hidden_dim=int(raw.get("hidden_dim", base.hidden_dim)),
            latent_dim=int(raw.get("latent_dim", base.latent_dim)),
            lr=float(raw.get("learning_rate", raw.get("lr", base.lr))),
        )
    return C41TrainingProfile(
        name=base.name,
        hidden_dim=int(raw.get("hidden_dim", base.hidden_dim)),
        latent_dim=int(raw.get("latent_dim", base.latent_dim)),
        lr=float(raw.get("learning_rate", raw.get("lr", base.lr))),
        epochs=int(raw.get("epochs", base.epochs)),
        patience=int(raw.get("patience", base.patience)),
        batch_size=int(raw.get("batch_size", base.batch_size)),
        pca_components=base.pca_components,
    )


def discover_c41_run_artifacts(
    *,
    config: LockedV1Config,
    repo_root: Path,
) -> tuple[C41RunArtifacts, ...]:
    discovered = []
    for artifact in discover_support_run_artifacts(config=config, repo_root=repo_root):
        val_cache = artifact.run_dir / "embeddings" / "val.pt"
        if not val_cache.exists():
            raise ProtocolError(f"C4.1 requires source-val embedding cache: {val_cache}")
        discovered.append(C41RunArtifacts(support=artifact, val_cache=val_cache))
    return tuple(discovered)


def build_c41_downstream_matrix(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    training_profile: C41TrainingProfile,
    limits: MatrixBuildLimits = MatrixBuildLimits(),
) -> Path:
    """Train C4.1 generators and write support-replicated utility rows."""

    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = _limit_c41_artifacts(
        discover_c41_run_artifacts(config=config, repo_root=repo_root),
        limits.experiment_seeds,
    )
    completed = _read_completed_keys(matrix_path) if resume else set()
    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    provenance_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for artifact in artifacts:
        support = artifact.support
        samples = _read_samples_manifest(support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        val_records = _records_for_split(samples, "val")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(support.train_cache, train_records, repo_root=repo_root)
        val_cache = _load_embedding_cache(artifact.val_cache, val_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(support.test_cache, test_records, repo_root=repo_root)
        artifact_profile = _profile_for_support_config(training_profile, support.config_resolved)

        for heldout_center in selected_heldout_centers:
            heldout = str(heldout_center)
            if heldout not in {str(c) for c in config.candidate_domains}:
                raise ProtocolError(f"Unknown heldout center requested: {heldout}")
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _support_conditions(
                support_units,
                experiment_seed=support.experiment_seed,
                heldout_center=heldout,
            )
            if not support_conditions:
                raise ProtocolError(
                    f"No locked support-selection conditions for seed={support.experiment_seed}, "
                    f"heldout_center={heldout}."
                )
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            label_values = tuple(sorted(set(target_labels).union({0, 1})))
            if label_values != (0, 1):
                raise ProtocolError(f"C4.1 expects binary labels 0/1, got {label_values}")

            for candidate in candidates:
                projection = _fit_or_load_projection(
                    artifacts_root=artifacts_root,
                    train_cache=train_cache,
                    source_domain=candidate,
                    seed=support.experiment_seed,
                    n_components=artifact_profile.pca_components,
                    resume=resume,
                )
                provenance_rows.append(
                    {
                        "experiment_seed": support.experiment_seed,
                        "heldout_center": heldout,
                        "candidate_expert": candidate,
                        **projection.provenance(),
                    }
                )
                train_projected_all = projection.transform(train_cache.embeddings)
                val_projected_all = projection.transform(val_cache.embeddings)
                candidate_train_idx = _indices_for_domain(train_cache.metadata, candidate)
                candidate_val_idx = _indices_for_domain(val_cache.metadata, candidate)
                if not candidate_train_idx or not candidate_val_idx:
                    raise ProtocolError(
                        f"C4.1 requires nonempty source train/val rows for candidate={candidate}."
                    )
                train_x = train_projected_all[candidate_train_idx]
                val_x = val_projected_all[candidate_val_idx]
                train_y = labels_from_metadata([train_cache.metadata[idx] for idx in candidate_train_idx])
                val_y = labels_from_metadata([val_cache.metadata[idx] for idx in candidate_val_idx])
                reference_pools = build_source_train_reference_pools(
                    train_projected_embeddings=train_projected_all,
                    train_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=label_values,
                )
                plain_ckpt = _train_c41_model(
                    repo_root=repo_root,
                    artifacts_root=artifacts_root,
                    experiment_seed=support.experiment_seed,
                    candidate_expert=candidate,
                    family=GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL,
                    train_x=train_x,
                    val_x=val_x,
                    train_y=train_y,
                    val_y=val_y,
                    profile=artifact_profile,
                    resume=resume,
                )
                hetero_ckpt = _train_c41_model(
                    repo_root=repo_root,
                    artifacts_root=artifacts_root,
                    experiment_seed=support.experiment_seed,
                    candidate_expert=candidate,
                    family=GENERATOR_FAMILY_HETEROSCEDASTIC,
                    train_x=train_x,
                    val_x=val_x,
                    train_y=train_y,
                    val_y=val_y,
                    profile=artifact_profile,
                    resume=resume,
                )
                contexts = (
                    GeneratorContext(
                        family=GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL,
                        mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
                        checkpoint_path=plain_ckpt,
                        plain_baseline_source=PLAIN_BASELINE_RETRAINED,
                        plain_baseline_artifact_path=str(plain_ckpt),
                        plain_baseline_training_profile=artifact_profile.name,
                        plain_baseline_matches_locked_hparams=artifact_profile.matches_locked_hparams,
                    ),
                    GeneratorContext(
                        family=GENERATOR_FAMILY_HETEROSCEDASTIC,
                        mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
                        checkpoint_path=hetero_ckpt,
                        plain_baseline_source=PLAIN_BASELINE_RETRAINED,
                        plain_baseline_artifact_path=str(plain_ckpt),
                        plain_baseline_training_profile=artifact_profile.name,
                        plain_baseline_matches_locked_hparams=artifact_profile.matches_locked_hparams,
                    ),
                    GeneratorContext(
                        family=GENERATOR_FAMILY_HETEROSCEDASTIC,
                        mode=GENERATION_MODE_POSTERIOR_DECODER_NOISE,
                        checkpoint_path=hetero_ckpt,
                        plain_baseline_source=PLAIN_BASELINE_RETRAINED,
                        plain_baseline_artifact_path=str(plain_ckpt),
                        plain_baseline_training_profile=artifact_profile.name,
                        plain_baseline_matches_locked_hparams=artifact_profile.matches_locked_hparams,
                    ),
                )
                for context in contexts:
                    model = _load_c41_model(repo_root, context.checkpoint_path, device=device)
                    if context.family == GENERATOR_FAMILY_HETEROSCEDASTIC:
                        diagnostic_rows.append(
                            {
                                "experiment_seed": support.experiment_seed,
                                "heldout_center": heldout,
                                "candidate_expert": candidate,
                                "generator_family": context.family,
                                "generation_mode": context.mode,
                                **decoder_logvar_diagnostics_by_class(model=model, reference_pools=reference_pools),
                            }
                        )
                    for generation_seed in selected_generation_seeds:
                        for classifier_seed in selected_classifier_seeds:
                            base_row, row_diagnostics = _score_c41_candidate(
                                model=model,
                                projection=projection,
                                context=context,
                                experiment_seed=support.experiment_seed,
                                heldout_center=heldout,
                                candidate_expert=candidate,
                                target_eval_pool=target_pool,
                                target_labels=target_labels,
                                test_cache=test_cache,
                                train_cache=train_cache,
                                label_values=label_values,
                                reference_pools=reference_pools,
                                budget_per_class=config.primary_budget_per_class,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                            )
                            diagnostic_rows.append(row_diagnostics)
                            for support_size, support_seed in support_conditions:
                                row = replace(base_row, support_size=int(support_size), support_seed=int(support_seed))
                                if resume and row.primary_key() in completed:
                                    continue
                                append_matrix_row(matrix_path, row)
                                completed.add(row.primary_key())

    _write_dict_csv(artifacts_root / "manifests" / "c41_generator_provenance.csv", provenance_rows)
    _write_dict_csv(artifacts_root / "tables" / "generator_distribution_diagnostics.csv", diagnostic_rows)
    return matrix_path


def build_c41_delta_summary_rows(
    *,
    alignment_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Build support-size/mode deltas against the retrained plain baseline."""

    support_rows = [row for row in alignment_rows if str(row.get("method")) == SUPPORT_NELBO_METHOD]
    if not support_rows:
        return []
    rows: list[dict[str, object]] = []
    generated_std_delta = _generated_std_delta_lookup(diagnostic_rows)
    groups = sorted(
        {
            (str(row["heldout_center"]), int(row["support_size"]))
            for row in support_rows
        }
    )
    for heldout, support_size in groups:
        base_subset = [
            row
            for row in support_rows
            if str(row["heldout_center"]) == heldout
            and int(row["support_size"]) == support_size
            and str(row["generator_family"]) == GENERATOR_FAMILY_PLAIN_CLASS_CONDITIONAL
            and str(row["generation_mode"]) == GENERATION_MODE_POSTERIOR_DECODER_MEAN
        ]
        if not base_subset:
            continue
        plain_oracle = _dedup_oracle_mean(base_subset, "oracle_bacc")
        plain_selected = _mean(base_subset, "selected_bacc")
        plain_gap = _mean(base_subset, "downstream_oracle_gap_bacc")
        plain_oracle_experts = {str(row["downstream_oracle_expert"]) for row in base_subset}
        for mode in (GENERATION_MODE_POSTERIOR_DECODER_MEAN, GENERATION_MODE_POSTERIOR_DECODER_NOISE):
            hetero_subset = [
                row
                for row in support_rows
                if str(row["heldout_center"]) == heldout
                and int(row["support_size"]) == support_size
                and str(row["generator_family"]) == GENERATOR_FAMILY_HETEROSCEDASTIC
                and str(row["generation_mode"]) == mode
            ]
            if not hetero_subset:
                continue
            hetero_oracle = _dedup_oracle_mean(hetero_subset, "oracle_bacc")
            hetero_selected = _mean(hetero_subset, "selected_bacc")
            hetero_gap = _mean(hetero_subset, "downstream_oracle_gap_bacc")
            selected_changed = int(_selected_expert_changed(base_subset, hetero_subset))
            oracle_changed = int(
                bool({str(row["downstream_oracle_expert"]) for row in hetero_subset}.difference(plain_oracle_experts))
            )
            oracle_delta = hetero_oracle - plain_oracle
            selected_delta = hetero_selected - plain_selected
            gap_delta = hetero_gap - plain_gap
            std_delta = generated_std_delta.get((heldout, mode), math.nan)
            decision = _c41_decision_label(
                oracle_delta=oracle_delta,
                selected_delta=selected_delta,
                selected_changed=selected_changed,
                oracle_changed=oracle_changed,
                std_delta=std_delta,
            )
            rows.append(
                {
                    "heldout_center": heldout,
                    "support_size": support_size,
                    "generation_mode": mode,
                    "oracle_bacc_plain": plain_oracle,
                    "oracle_bacc_hetero_mean": _mode_mean(support_rows, heldout, support_size, GENERATION_MODE_POSTERIOR_DECODER_MEAN, "oracle_bacc"),
                    "oracle_bacc_hetero_noise": _mode_mean(support_rows, heldout, support_size, GENERATION_MODE_POSTERIOR_DECODER_NOISE, "oracle_bacc"),
                    "selected_bacc_plain_router_plain_generator": plain_selected,
                    "selected_bacc_plain_router_hetero_mean_generator": _mode_mean(support_rows, heldout, support_size, GENERATION_MODE_POSTERIOR_DECODER_MEAN, "selected_bacc"),
                    "selected_bacc_plain_router_hetero_noise_generator": _mode_mean(support_rows, heldout, support_size, GENERATION_MODE_POSTERIOR_DECODER_NOISE, "selected_bacc"),
                    "oracle_bacc_delta_vs_plain_retrained": oracle_delta,
                    "selected_bacc_delta_vs_plain_retrained": selected_delta,
                    "oracle_gap_delta_vs_plain_retrained": gap_delta,
                    "generated_std_delta_vs_plain": std_delta,
                    "selected_expert_changed_across_modes": selected_changed,
                    "oracle_expert_changed_vs_plain": oracle_changed,
                    "decision_label": decision,
                }
            )
    return rows


def write_c41_delta_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, C41_DELTA_SUMMARY_COLUMNS, rows)


def assert_selected_expert_invariant(alignment_rows: Sequence[Mapping[str, object]]) -> None:
    grouped: dict[tuple[object, ...], set[str]] = {}
    for row in alignment_rows:
        if str(row.get("method")) != SUPPORT_NELBO_METHOD:
            continue
        key = (
            str(row["heldout_center"]),
            int(row["experiment_seed"]),
            int(row["support_size"]),
            int(row["support_seed"]),
            int(row["generation_seed"]),
            int(row["classifier_seed"]),
        )
        grouped.setdefault(key, set()).add(str(row["selected_expert"]))
    changed = {key: values for key, values in grouped.items() if len(values) > 1}
    if changed:
        raise ProtocolError(f"C4.1 selected expert changed across generator modes: {changed}")


def load_generator_diagnostics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _score_c41_candidate(
    *,
    model: Any,
    projection: Any,
    context: GeneratorContext,
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    target_eval_pool: TargetEvalPool,
    target_labels: Sequence[int],
    test_cache: EmbeddingCache,
    train_cache: EmbeddingCache,
    label_values: Sequence[int],
    reference_pools: Mapping[int, Any],
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
) -> tuple[CandidateDownstreamRow, dict[str, object]]:
    try:
        synthetic_chunks = []
        synthetic_labels = []
        sample_diagnostics: list[Mapping[str, float]] = []
        for label in label_values:
            generated = generate_posterior_sampled_embeddings(
                model=model,
                reference_pool=reference_pools[int(label)].to(next(model.parameters()).device),
                class_label=int(label),
                n_samples=int(budget_per_class),
                seed=int(generation_seed) + int(label),
                generation_mode=context.mode,
            )
            synthetic_chunks.append(generated.embeddings)
            synthetic_labels.extend(int(v) for v in generated.labels.tolist())
            sample_diagnostics.append(generated.diagnostics)
        torch = _torch_module()
        synthetic_embeddings = torch.cat(synthetic_chunks, dim=0)
        target_embeddings = projection.transform(test_cache.embeddings[list(target_eval_pool.eval_indices)])
        prediction = fit_locked_logistic_classifier(
            _to_numpy(synthetic_embeddings),
            synthetic_labels,
            _to_numpy(target_embeddings),
            target_labels,
            classifier_seed=classifier_seed,
        )
        diagnostics = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": context.family,
            "generation_mode": context.mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            **_aggregate_sample_diagnostics(sample_diagnostics),
            **_std_ratio_diagnostics(
                generated_pca=synthetic_embeddings,
                generated_dino=projection.inverse_transform(synthetic_embeddings),
                source_train_pca=projection.transform(train_cache.embeddings[_indices_for_domain(train_cache.metadata, candidate_expert)]),
                source_train_dino=train_cache.embeddings[_indices_for_domain(train_cache.metadata, candidate_expert)],
            ),
        }
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=context.family,
            generation_mode=context.mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=float(prediction.score.balanced_accuracy),
            macro_f1=float(prediction.score.macro_f1),
            auroc=float(prediction.score.secondary_metrics.get("auroc", math.nan)),
            auprc=float(prediction.score.secondary_metrics.get("auprc", math.nan)),
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=int(budget_per_class) * len(label_values),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            candidate_experts_hash=SINGLE_EXPERT_HASH,
            utility_depends_on_support=0,
            selection_depends_on_support=0,
            plain_baseline_source=context.plain_baseline_source,
            plain_baseline_artifact_path=context.plain_baseline_artifact_path,
            plain_baseline_training_profile=context.plain_baseline_training_profile,
            plain_baseline_matches_locked_hparams=context.plain_baseline_matches_locked_hparams,
            **c41_routing_provenance_fields(),
        )
        return row, diagnostics
    except Exception as exc:
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=context.family,
            generation_mode=context.mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=math.nan,
            macro_f1=math.nan,
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=int(budget_per_class) * len(label_values),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            status="failed_c41_candidate_scoring",
            error_message=str(exc),
            utility_depends_on_support=0,
            selection_depends_on_support=0,
            plain_baseline_source=context.plain_baseline_source,
            plain_baseline_artifact_path=context.plain_baseline_artifact_path,
            plain_baseline_training_profile=context.plain_baseline_training_profile,
            plain_baseline_matches_locked_hparams=context.plain_baseline_matches_locked_hparams,
            **c41_routing_provenance_fields(),
        )
        return row, {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": context.family,
            "generation_mode": context.mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            "status": row.status,
            "error_message": row.error_message,
        }


def _fit_or_load_projection(
    *,
    artifacts_root: Path,
    train_cache: EmbeddingCache,
    source_domain: str,
    seed: int,
    n_components: int,
    resume: bool,
) -> Any:
    torch = _torch_module()
    path = artifacts_root / "projections" / f"seed{int(seed)}" / f"expert_{source_domain}" / "pca64.pt"
    if resume and path.exists():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    projection = fit_source_train_pca_projection(
        train_embeddings=train_cache.embeddings,
        train_metadata=train_cache.metadata,
        source_domain=source_domain,
        seed=seed,
        n_components=n_components,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projection, path)
    return projection


def _train_c41_model(
    *,
    repo_root: Path,
    artifacts_root: Path,
    experiment_seed: int,
    candidate_expert: str,
    family: str,
    train_x: Any,
    val_x: Any,
    train_y: Any,
    val_y: Any,
    profile: C41TrainingProfile,
    resume: bool,
) -> Path:
    if str(repo_root / "cvae_testing") not in sys.path:
        sys.path.insert(0, str(repo_root / "cvae_testing"))
    from src.models.cvae_expert import (  # type: ignore
        DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        DECODER_LIKELIHOOD_MSE,
        RECON_LOSS_GAUSSIAN_NLL_DIAG,
        RECON_LOSS_MSE,
        REDUCTION_MEAN,
        REDUCTION_SUM,
    )
    from src.train.train_utils import run_training  # type: ignore

    model_slug = "heteroscedastic" if family == GENERATOR_FAMILY_HETEROSCEDASTIC else "plain"
    out_dir = artifacts_root / "checkpoints" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / model_slug
    ckpt = out_dir / f"{model_slug}_class_conditional_pca64.pt"
    if ckpt.exists() and not resume:
        raise ProtocolError(f"C4.1 checkpoint already exists; use --resume or a clean artifact root: {ckpt}")
    gaussian = family == GENERATOR_FAMILY_HETEROSCEDASTIC
    result = run_training(
        train_embeddings=train_x,
        val_embeddings=val_x,
        out_dir=out_dir,
        model_name=f"{model_slug}_class_conditional_pca64",
        input_dim=int(train_x.shape[1]),
        hidden_dim=profile.hidden_dim,
        latent_dim=profile.latent_dim,
        lr=profile.lr,
        epochs=profile.epochs,
        patience=profile.patience,
        batch_size=profile.batch_size,
        resume_from=ckpt if resume and ckpt.exists() else None,
        train_class_labels=train_y,
        val_class_labels=val_y,
        class_condition_dim=2,
        decoder_likelihood=DECODER_LIKELIHOOD_GAUSSIAN_DIAG if gaussian else DECODER_LIKELIHOOD_MSE,
        reconstruction_loss=RECON_LOSS_GAUSSIAN_NLL_DIAG if gaussian else RECON_LOSS_MSE,
        recon_reduction=REDUCTION_MEAN if gaussian else REDUCTION_SUM,
        kl_reduction=REDUCTION_MEAN if gaussian else REDUCTION_SUM,
        beta=1.0,
        checkpoint_metadata={
            "generator_family": family,
            "experiment_seed": int(experiment_seed),
            "candidate_expert": str(candidate_expert),
            "projection_family": "source_train_pca64",
            "plain_baseline_source": PLAIN_BASELINE_RETRAINED,
            "plain_baseline_training_profile": profile.name,
        },
    )
    return result.checkpoint_path


def _load_c41_model(repo_root: Path, checkpoint_path: Path, *, device: str) -> Any:
    if str(repo_root / "cvae_testing") not in sys.path:
        sys.path.insert(0, str(repo_root / "cvae_testing"))
    from src.models.cvae_expert import build_cvae_from_metadata  # type: ignore
    from src.train.checkpoint_provenance import load_model_checkpoint  # type: ignore

    torch = _torch_module()
    torch_device = _resolve_torch_device(torch, device)
    loaded = load_model_checkpoint(checkpoint_path, map_location=torch_device)
    model = build_cvae_from_metadata(loaded.checkpoint_metadata).to(torch_device)
    model.load_state_dict(loaded.model_state_dict)
    model.eval()
    return model


def _assert_no_forbidden_support_columns(paths: Sequence[Path]) -> None:
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = {str(name) for name in (reader.fieldnames or ())}
        forbidden = sorted(columns.intersection(_FORBIDDEN_SUPPORT_COLUMNS))
        if forbidden:
            raise ProtocolError(
                f"C4.1 support loader refuses target-eval/oracle-derived columns in {path}: {forbidden}"
            )


def _support_conditions(
    units: Sequence[SupportSelectionUnit],
    *,
    experiment_seed: int,
    heldout_center: str,
) -> tuple[tuple[int, int], ...]:
    conditions = {
        (int(unit.support_size), int(unit.support_seed))
        for unit in units
        if int(unit.experiment_seed) == int(experiment_seed)
        and str(unit.heldout_center) == str(heldout_center)
        and unit.method == SUPPORT_NELBO_METHOD
    }
    return tuple(sorted(conditions))


def _indices_for_domain(metadata: Sequence[Mapping[str, object]], domain: str) -> list[int]:
    return [idx for idx, row in enumerate(metadata) if str(_domain(row)) == str(domain)]


def _read_training_config(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, Mapping):
            model = loaded.get("model", {}) if isinstance(loaded.get("model"), Mapping) else {}
            training = loaded.get("training", {}) if isinstance(loaded.get("training"), Mapping) else {}
            merged = dict(model)
            merged.update(dict(training))
            return merged
    except Exception:
        pass
    return {
        "hidden_dim": _regex_int_default(text, r"hidden_dim:\s*(\d+)", 256),
        "latent_dim": _regex_int_default(text, r"latent_dim:\s*(\d+)", 16),
        "batch_size": _regex_int_default(text, r"batch_size:\s*(\d+)", 128),
        "epochs": _regex_int_default(text, r"epochs:\s*(\d+)", 25),
        "patience": _regex_int_default(text, r"patience:\s*(\d+)", 5),
        "learning_rate": _regex_float_default(text, r"learning_rate:\s*([0-9.eE+-]+)", 1.0e-3),
    }


def _regex_int_default(text: str, pattern: str, default: int) -> int:
    import re

    match = re.search(pattern, text)
    return int(match.group(1)) if match else int(default)


def _regex_float_default(text: str, pattern: str, default: float) -> float:
    import re

    match = re.search(pattern, text)
    return float(match.group(1)) if match else float(default)


def _limit_c41_artifacts(
    artifacts: Sequence[C41RunArtifacts],
    experiment_seeds: Sequence[int] | None,
) -> tuple[C41RunArtifacts, ...]:
    if experiment_seeds is None:
        return tuple(artifacts)
    allowed = {int(seed) for seed in experiment_seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.support.experiment_seed) in allowed)


def _aggregate_sample_diagnostics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {key: _nanmean(float(item[key]) for item in items if key in item) for key in keys}


def _std_ratio_diagnostics(
    *,
    generated_pca: Any,
    generated_dino: Any,
    source_train_pca: Any,
    source_train_dino: Any,
) -> dict[str, float]:
    return {
        "generated_pca_std_ratio": _std_ratio(generated_pca, source_train_pca),
        "generated_dino_std_ratio": _std_ratio(generated_dino, source_train_dino),
    }


def _std_ratio(generated: Any, reference: Any) -> float:
    gen = generated.detach().cpu().float()
    ref = reference.detach().cpu().float()
    gen_std = gen.std(dim=0, unbiased=False).mean()
    ref_std = ref.std(dim=0, unbiased=False).mean().clamp_min(1.0e-12)
    return float((gen_std / ref_std).item())


def _generated_std_delta_lookup(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if str(row.get("generator_family")) != GENERATOR_FAMILY_HETEROSCEDASTIC:
            continue
        if "generated_pca_std_ratio" not in row:
            continue
        key = (str(row.get("heldout_center")), str(row.get("generation_mode")))
        try:
            grouped.setdefault(key, []).append(float(row.get("generated_pca_std_ratio", math.nan)))
        except (TypeError, ValueError):
            continue
    return {key: _nanmean(values) - 1.0 for key, values in grouped.items()}


def _mode_mean(
    rows: Sequence[Mapping[str, object]],
    heldout: str,
    support_size: int,
    mode: str,
    metric: str,
) -> float:
    subset = [
        row
        for row in rows
        if str(row["heldout_center"]) == heldout
        and int(row["support_size"]) == int(support_size)
        and str(row["generator_family"]) == GENERATOR_FAMILY_HETEROSCEDASTIC
        and str(row["generation_mode"]) == mode
    ]
    if str(metric).startswith("oracle_"):
        return _dedup_oracle_mean(subset, metric)
    return _mean(subset, metric)


def _dedup_oracle_mean(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    by_context: dict[tuple[object, ...], float] = {}
    for row in rows:
        key = (
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            str(row["generator_family"]),
            str(row["generation_mode"]),
            int(row["generation_seed"]),
            int(row["classifier_seed"]),
        )
        by_context.setdefault(key, float(row[metric]))
    return _nanmean(by_context.values())


def _selected_expert_changed(
    plain_rows: Sequence[Mapping[str, object]],
    hetero_rows: Sequence[Mapping[str, object]],
) -> bool:
    plain = {
        (
            int(row["experiment_seed"]),
            int(row["support_seed"]),
            int(row["generation_seed"]),
            int(row["classifier_seed"]),
        ): str(row["selected_expert"])
        for row in plain_rows
    }
    for row in hetero_rows:
        key = (
            int(row["experiment_seed"]),
            int(row["support_seed"]),
            int(row["generation_seed"]),
            int(row["classifier_seed"]),
        )
        if key in plain and plain[key] != str(row["selected_expert"]):
            return True
    return False


def _c41_decision_label(
    *,
    oracle_delta: float,
    selected_delta: float,
    selected_changed: int,
    oracle_changed: int,
    std_delta: float,
) -> str:
    if selected_changed:
        return DECISION_PROTOCOL_FAILURE
    if float(oracle_delta) >= 0.02 and float(selected_delta) >= 0.0:
        return DECISION_SUCCESS
    if not math.isnan(float(std_delta)) and float(std_delta) > 0.5 and float(oracle_delta) < 0.02:
        return DECISION_VARIANCE_INFLATION
    if oracle_changed and float(oracle_delta) < 0.02:
        return DECISION_RANK_INSTABILITY
    return DECISION_NO_UTILITY_GAIN


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return _nanmean(float(row[key]) for row in rows)


def _nanmean(values: Iterable[float]) -> float:
    cleaned = [float(value) for value in values if not math.isnan(float(value))]
    return sum(cleaned) / float(len(cleaned)) if cleaned else math.nan


def _torch_module() -> Any:
    import torch  # type: ignore

    return torch


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    _write_csv(path, columns, rows)


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
