"""C4.2 workstation bridge for source-class latent GMM priors."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .c41_heteroscedastic import (
    GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    build_source_train_reference_pools,
    generate_posterior_sampled_embeddings,
    labels_from_metadata,
)
from .c41_workstation import (
    C41RunArtifacts,
    _load_c41_model,
    _support_conditions,
    _write_csv,
    _write_dict_csv,
    discover_c41_run_artifacts,
    safe_support_selection_units_from_paths,
)
from .c42_latent_gmm import (
    C42_LATENT_GMM_COMPONENTS_BY_MODE,
    C42_LATENT_GMM_GENERATION_MODES,
    SourceClassLatentDiagGMM,
    fit_source_class_latent_gmm,
    generate_latent_gmm_decoder_mean,
    generate_standard_prior_decoder_mean,
    generated_embedding_diagnostics,
)
from .downstream import CandidateDownstreamRow, fit_locked_logistic_classifier
from .matrix import (
    EmbeddingCache,
    MatrixBuildLimits,
    TargetEvalPool,
    _domain,
    _label,
    _load_embedding_cache,
    _read_completed_keys,
    _read_samples_manifest,
    _records_for_split,
    _to_numpy,
    append_matrix_row,
    build_target_eval_pool,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import SupportSelectionUnit
from .schemas import (
    C42_DELTA_SUMMARY_COLUMNS,
    C42_POSTERIOR_REPLAY_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
    SINGLE_EXPERT_HASH,
    SINGLE_EXPERT_ROW_TYPE,
    SUPPORT_NELBO_METHOD,
)


C42_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c42_latent_gmm_prior_v1"
C42_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
PLAIN_REPLAY_TOLERANCE = 1.0e-6

DECISION_PASS = "PASS_CANDIDATE"
DECISION_CEILING_ONLY = "LATENT_GMM_IMPROVES_GENERATOR_CEILING_BUT_ROUTING_STILL_LIMITS_UTILITY"
DECISION_UNDERDISPERSION = "LATENT_GMM_UNDERDISPERSION"
DECISION_OVERDISPERSION = "LATENT_GMM_OVERDISPERSION"
DECISION_PRIOR_DECODER_MISMATCH = "LATENT_PRIOR_DECODER_MISMATCH"
DECISION_MODE_COLLAPSE = "LATENT_GMM_MODE_COLLAPSE"
DECISION_RANK_INSTABILITY = "LATENT_GMM_ORACLE_RANK_INSTABILITY"
DECISION_NO_GAIN = "NO_UTILITY_GAIN"
DECISION_PROTOCOL_FAILURE = "PROTOCOL_FAILURE_SELECTED_EXPERT_CHANGED"
DECISION_REPLAY_MISMATCH = "PLAIN_REPLAY_MISMATCH"


@dataclass(frozen=True)
class C42PlainArtifacts:
    checkpoint_path: Path
    projection_path: Path


def build_c42_downstream_matrix(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    resume: bool,
    limits: MatrixBuildLimits = MatrixBuildLimits(),
    covariance_floor: float = 1.0e-4,
) -> Path:
    matrix_path = artifacts_root / "tables" / "all_expert_downstream_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_completed_keys(matrix_path) if resume else set()
    c41_artifacts = _limit_c41_artifacts(
        discover_c41_run_artifacts(config=config, repo_root=repo_root),
        limits.experiment_seeds,
    )
    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    diagnostic_rows: list[dict[str, object]] = []

    for artifact in c41_artifacts:
        support = artifact.support
        samples = _read_samples_manifest(support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(support.train_cache, train_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(support.test_cache, test_records, repo_root=repo_root)
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
                raise ProtocolError(f"No support conditions for seed={support.experiment_seed}, heldout={heldout}")
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            label_values = tuple(sorted(set(target_labels).union({0, 1})))
            if label_values != (0, 1):
                raise ProtocolError(f"C4.2 expects binary labels 0/1, got {label_values}")
            for candidate in candidates:
                plain = _plain_artifacts(c41_artifacts_root, support.experiment_seed, candidate)
                projection = _load_projection(plain.projection_path)
                model = _load_c41_model(repo_root, plain.checkpoint_path, device=device)
                train_projected_all = projection.transform(train_cache.embeddings)
                candidate_train_idx = _indices_for_domain(train_cache.metadata, candidate)
                if not candidate_train_idx:
                    raise ProtocolError(f"No source-train rows for candidate={candidate}")
                candidate_train_projected = train_projected_all[candidate_train_idx]
                candidate_train_labels = labels_from_metadata([train_cache.metadata[idx] for idx in candidate_train_idx])
                reference_pools = build_source_train_reference_pools(
                    train_projected_embeddings=train_projected_all,
                    train_metadata=train_cache.metadata,
                    source_domain=candidate,
                    label_values=label_values,
                )
                priors_by_mode = _fit_or_load_priors(
                    model=model,
                    artifacts_root=artifacts_root,
                    experiment_seed=support.experiment_seed,
                    candidate_expert=candidate,
                    train_projected=candidate_train_projected,
                    train_labels=candidate_train_labels,
                    label_values=label_values,
                    covariance_floor=covariance_floor,
                    resume=resume,
                )
                for generation_mode in (
                    C42_POSTERIOR_REPLAY_GENERATION_MODE,
                    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
                    *C42_LATENT_GMM_GENERATION_MODES,
                ):
                    for generation_seed in selected_generation_seeds:
                        for classifier_seed in selected_classifier_seeds:
                            base_row, diagnostics = _score_c42_candidate(
                                model=model,
                                projection=projection,
                                generation_mode=generation_mode,
                                priors=priors_by_mode.get(generation_mode, {}),
                                experiment_seed=support.experiment_seed,
                                heldout_center=heldout,
                                candidate_expert=candidate,
                                plain_checkpoint=plain.checkpoint_path,
                                target_eval_pool=target_pool,
                                target_labels=target_labels,
                                test_cache=test_cache,
                                source_train_projected=candidate_train_projected,
                                source_train_labels=candidate_train_labels,
                                label_values=label_values,
                                reference_pools=reference_pools,
                                budget_per_class=config.primary_budget_per_class,
                                generation_seed=int(generation_seed),
                                classifier_seed=int(classifier_seed),
                            )
                            diagnostic_rows.append(diagnostics)
                            for support_size, support_seed in support_conditions:
                                row = replace(base_row, support_size=int(support_size), support_seed=int(support_seed))
                                if resume and row.primary_key() in completed:
                                    continue
                                append_matrix_row(matrix_path, row)
                                completed.add(row.primary_key())
    _write_dict_csv(artifacts_root / "tables" / "latent_gmm_prior_diagnostics.csv", diagnostic_rows)
    return matrix_path


def build_c42_delta_summary_rows(
    *,
    c42_alignment_rows: Sequence[Mapping[str, object]],
    c41_alignment_rows: Sequence[Mapping[str, object]],
    diagnostics_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    c42_support = [row for row in c42_alignment_rows if str(row.get("method")) == SUPPORT_NELBO_METHOD]
    c41_plain = [
        row
        for row in c41_alignment_rows
        if str(row.get("method")) == SUPPORT_NELBO_METHOD
        and str(row.get("generator_family")) == PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY
        and str(row.get("generation_mode")) == GENERATION_MODE_POSTERIOR_DECODER_MEAN
    ]
    rows: list[dict[str, object]] = []
    groups = sorted({(str(row["heldout_center"]), int(row["support_size"])) for row in c42_support})
    for heldout, support_size in groups:
        plain_subset = _subset(c41_plain, heldout, support_size, GENERATION_MODE_POSTERIOR_DECODER_MEAN)
        posterior_replay = _subset(c42_support, heldout, support_size, C42_POSTERIOR_REPLAY_GENERATION_MODE)
        standard_replay = _subset(c42_support, heldout, support_size, C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE)
        if not plain_subset:
            continue
        plain_oracle = _dedup_oracle_mean(plain_subset, "oracle_bacc")
        plain_selected = _mean(plain_subset, "selected_bacc")
        plain_gap = _mean(plain_subset, "downstream_oracle_gap_bacc")
        posterior_delta = _mean(posterior_replay, "selected_bacc") - plain_selected if posterior_replay else math.nan
        replay_ok = int(not math.isnan(posterior_delta) and abs(posterior_delta) <= PLAIN_REPLAY_TOLERANCE)
        for mode in C42_LATENT_GMM_GENERATION_MODES:
            mode_subset = _subset(c42_support, heldout, support_size, mode)
            if not mode_subset:
                continue
            requested = C42_LATENT_GMM_COMPONENTS_BY_MODE[mode]
            effective = _mode_effective_components(diagnostics_rows, heldout, mode, requested)
            oracle_delta = _dedup_oracle_mean(mode_subset, "oracle_bacc") - plain_oracle
            selected_delta = _mean(mode_subset, "selected_bacc") - plain_selected
            gap_delta = _mean(mode_subset, "downstream_oracle_gap_bacc") - plain_gap
            selected_changed = int(_selected_expert_changed(plain_subset, mode_subset))
            oracle_changed = int(_oracle_expert_changed(plain_subset, mode_subset))
            stability = _oracle_top1_stability(mode_subset)
            decision = _decision_label(
                oracle_delta=oracle_delta,
                selected_delta=selected_delta,
                gap_delta=gap_delta,
                selected_changed=selected_changed,
                replay_ok=replay_ok,
                stability=stability,
                diagnostics=_mode_diagnostics(diagnostics_rows, heldout, mode),
            )
            rows.append(
                {
                    "heldout_center": heldout,
                    "support_size": support_size,
                    "generation_mode": mode,
                    "latent_gmm_components_requested": requested,
                    "latent_gmm_components_effective": effective,
                    "oracle_bacc_plain": plain_oracle,
                    "oracle_bacc_posterior_replay": _dedup_oracle_mean(posterior_replay, "oracle_bacc"),
                    "oracle_bacc_standard_prior_replay": _dedup_oracle_mean(standard_replay, "oracle_bacc"),
                    "oracle_bacc_latent_gmm": _dedup_oracle_mean(mode_subset, "oracle_bacc"),
                    "selected_bacc_locked_c41_router_plain_generator": plain_selected,
                    "selected_bacc_locked_c41_router_posterior_replay_generator": _mean(posterior_replay, "selected_bacc"),
                    "selected_bacc_locked_c41_router_standard_prior_replay_generator": _mean(standard_replay, "selected_bacc"),
                    "selected_bacc_locked_c41_router_latent_gmm_generator": _mean(mode_subset, "selected_bacc"),
                    "oracle_bacc_delta_vs_plain_retrained": oracle_delta,
                    "selected_bacc_delta_vs_plain_retrained": selected_delta,
                    "oracle_gap_delta_vs_plain_retrained": gap_delta,
                    "plain_replay_bacc_delta_vs_c41_stored": posterior_delta,
                    "plain_replay_matches_c41_within_tolerance": replay_ok,
                    "oracle_expert_changed_vs_plain": oracle_changed,
                    "oracle_top1_stability_across_generation_seeds": stability,
                    "selected_expert_changed_across_modes": selected_changed,
                    "decision_label": decision,
                }
            )
    return rows


def write_c42_delta_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(path, C42_DELTA_SUMMARY_COLUMNS, rows)


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _score_c42_candidate(
    *,
    model,
    projection,
    generation_mode: str,
    priors: Mapping[int, SourceClassLatentDiagGMM],
    experiment_seed: int,
    heldout_center: str,
    candidate_expert: str,
    plain_checkpoint: Path,
    target_eval_pool: TargetEvalPool,
    target_labels: Sequence[int],
    test_cache: EmbeddingCache,
    source_train_projected: torch.Tensor,
    source_train_labels: Sequence[int],
    label_values: Sequence[int],
    reference_pools: Mapping[int, torch.Tensor],
    budget_per_class: int,
    generation_seed: int,
    classifier_seed: int,
) -> tuple[CandidateDownstreamRow, dict[str, object]]:
    try:
        chunks = []
        labels: list[int] = []
        diagnostic_parts: list[Mapping[str, float]] = []
        for label in label_values:
            if generation_mode == C42_POSTERIOR_REPLAY_GENERATION_MODE:
                generated = generate_posterior_sampled_embeddings(
                    model=model,
                    reference_pool=reference_pools[int(label)].to(next(model.parameters()).device),
                    class_label=int(label),
                    n_samples=int(budget_per_class),
                    seed=int(generation_seed) + int(label),
                    generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
                )
            elif generation_mode == C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE:
                generated = generate_standard_prior_decoder_mean(
                    model=model,
                    class_label=int(label),
                    n_samples=int(budget_per_class),
                    seed=int(generation_seed) + int(label),
                )
            else:
                generated = generate_latent_gmm_decoder_mean(
                    model=model,
                    prior=priors[int(label)],
                    class_label=int(label),
                    n_samples=int(budget_per_class),
                    seed=int(generation_seed) + int(label),
                    generation_mode=generation_mode,
                )
            chunks.append(generated.embeddings)
            labels.extend(int(v) for v in generated.labels.tolist())
            diagnostic_parts.append(generated.diagnostics)
        synthetic_embeddings = torch.cat(chunks, dim=0)
        target_embeddings = projection.transform(test_cache.embeddings[list(target_eval_pool.eval_indices)])
        prediction = fit_locked_logistic_classifier(
            _to_numpy(synthetic_embeddings),
            labels,
            _to_numpy(target_embeddings),
            target_labels,
            classifier_seed=classifier_seed,
        )
        diagnostics = {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": LATENT_GMM_PRIOR_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "generation_seed": int(generation_seed),
            "classifier_seed": int(classifier_seed),
            **_mean_diagnostics(diagnostic_parts),
            **generated_embedding_diagnostics(
                synthetic_embeddings=synthetic_embeddings,
                synthetic_labels=labels,
                source_train_embeddings=source_train_projected,
                source_train_labels=source_train_labels,
            ),
        }
        row = CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
            generation_mode=generation_mode,
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
            plain_baseline_source="reused_c41_full_plain",
            plain_baseline_artifact_path=str(plain_checkpoint),
            plain_baseline_training_profile="full",
            plain_baseline_matches_locked_hparams=1,
        )
        return row, diagnostics
    except Exception as exc:
        return CandidateDownstreamRow(
            experiment_seed=int(experiment_seed),
            heldout_center=heldout_center,
            support_size=0,
            support_seed=0,
            candidate_expert=candidate_expert,
            generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
            generation_mode=generation_mode,
            budget_per_class=int(budget_per_class),
            generation_seed=int(generation_seed),
            classifier_seed=int(classifier_seed),
            bacc=math.nan,
            macro_f1=math.nan,
            row_type=SINGLE_EXPERT_ROW_TYPE,
            n_synthetic_train=int(budget_per_class) * len(label_values),
            n_target_eval=len(target_eval_pool.eval_indices),
            target_eval_pool_id=target_eval_pool.target_eval_pool_id,
            status="failed_c42_candidate_scoring",
            error_message=str(exc),
            plain_baseline_source="reused_c41_full_plain",
            plain_baseline_artifact_path=str(plain_checkpoint),
            plain_baseline_training_profile="full",
            plain_baseline_matches_locked_hparams=1,
        ), {
            "experiment_seed": int(experiment_seed),
            "heldout_center": heldout_center,
            "candidate_expert": candidate_expert,
            "generator_family": LATENT_GMM_PRIOR_GENERATOR_FAMILY,
            "generation_mode": generation_mode,
            "status": "failed_c42_candidate_scoring",
            "error_message": str(exc),
        }


def _fit_or_load_priors(
    *,
    model,
    artifacts_root: Path,
    experiment_seed: int,
    candidate_expert: str,
    train_projected: torch.Tensor,
    train_labels: torch.Tensor,
    label_values: Sequence[int],
    covariance_floor: float,
    resume: bool,
) -> dict[str, dict[int, SourceClassLatentDiagGMM]]:
    priors: dict[str, dict[int, SourceClassLatentDiagGMM]] = {}
    for mode, requested in C42_LATENT_GMM_COMPONENTS_BY_MODE.items():
        priors[mode] = {}
        for label in label_values:
            path = (
                artifacts_root
                / "latent_priors"
                / f"seed{int(experiment_seed)}"
                / f"expert_{candidate_expert}"
                / f"class_{int(label)}"
                / f"gmm_k{int(requested)}.pt"
            )
            if resume and path.exists():
                prior = SourceClassLatentDiagGMM.from_payload(_torch_load(path))
            else:
                prior = fit_source_class_latent_gmm(
                    model=model,
                    projected_embeddings=train_projected,
                    labels=train_labels,
                    experiment_seed=int(experiment_seed),
                    source_domain=candidate_expert,
                    class_label=int(label),
                    requested_components=int(requested),
                    fit_seed=int(experiment_seed) + int(label) + (1009 * int(requested)),
                    covariance_floor=float(covariance_floor),
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(prior.to_payload(), path)
            priors[mode][int(label)] = prior
    return priors


def _plain_artifacts(c41_root: Path, experiment_seed: int, candidate_expert: str) -> C42PlainArtifacts:
    checkpoint = (
        c41_root
        / "checkpoints"
        / f"seed{int(experiment_seed)}"
        / f"expert_{candidate_expert}"
        / "plain"
        / "plain_class_conditional_pca64.pt"
    )
    projection = c41_root / "projections" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / "pca64.pt"
    missing = [path for path in (checkpoint, projection) if not path.exists()]
    if missing:
        raise ProtocolError(f"Missing C4.1 plain artifacts required by C4.2: {missing}")
    return C42PlainArtifacts(checkpoint_path=checkpoint, projection_path=projection)


def _load_projection(path: Path):
    return _torch_load(path)


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _indices_for_domain(metadata: Sequence[Mapping[str, object]], domain: str) -> list[int]:
    return [idx for idx, row in enumerate(metadata) if str(_domain(row)) == str(domain)]


def _limit_c41_artifacts(
    artifacts: Sequence[C41RunArtifacts],
    experiment_seeds: Sequence[int] | None,
) -> tuple[C41RunArtifacts, ...]:
    if experiment_seeds is None:
        return tuple(artifacts)
    allowed = {int(seed) for seed in experiment_seeds}
    return tuple(artifact for artifact in artifacts if int(artifact.support.experiment_seed) in allowed)


def _subset(rows: Sequence[Mapping[str, object]], heldout: str, support_size: int, mode: str) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("heldout_center")) == heldout
        and int(row.get("support_size", 0)) == int(support_size)
        and str(row.get("generation_mode")) == mode
    ]


def _selected_expert_changed(
    plain_rows: Sequence[Mapping[str, object]],
    mode_rows: Sequence[Mapping[str, object]],
) -> bool:
    plain = {
        (int(row["experiment_seed"]), int(row["support_seed"]), int(row["generation_seed"]), int(row["classifier_seed"])): str(row["selected_expert"])
        for row in plain_rows
    }
    for row in mode_rows:
        key = (int(row["experiment_seed"]), int(row["support_seed"]), int(row["generation_seed"]), int(row["classifier_seed"]))
        if key in plain and plain[key] != str(row["selected_expert"]):
            return True
    return False


def _oracle_expert_changed(
    plain_rows: Sequence[Mapping[str, object]],
    mode_rows: Sequence[Mapping[str, object]],
) -> bool:
    plain = {
        (int(row["experiment_seed"]), int(row["generation_seed"]), int(row["classifier_seed"])): str(row["downstream_oracle_expert"])
        for row in plain_rows
    }
    for row in mode_rows:
        key = (int(row["experiment_seed"]), int(row["generation_seed"]), int(row["classifier_seed"]))
        if key in plain and plain[key] != str(row["downstream_oracle_expert"]):
            return True
    return False


def _oracle_top1_stability(rows: Sequence[Mapping[str, object]]) -> float:
    by_context: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        key = (int(row["experiment_seed"]), int(row["classifier_seed"]))
        by_context.setdefault(key, set()).add(str(row["downstream_oracle_expert"]))
    if not by_context:
        return math.nan
    stable = sum(1 for values in by_context.values() if len(values) == 1)
    return float(stable) / float(len(by_context))


def _decision_label(
    *,
    oracle_delta: float,
    selected_delta: float,
    gap_delta: float,
    selected_changed: int,
    replay_ok: int,
    stability: float,
    diagnostics: Mapping[str, float],
) -> str:
    if selected_changed:
        return DECISION_PROTOCOL_FAILURE
    if not replay_ok:
        return DECISION_REPLAY_MISMATCH
    if float(oracle_delta) >= 0.02 and float(selected_delta) >= 0.0 and float(gap_delta) <= 0.0 and float(stability) >= 0.67:
        return DECISION_PASS
    if float(oracle_delta) >= 0.02 and float(selected_delta) < 0.0:
        return DECISION_CEILING_ONLY
    cov_ratio = float(diagnostics.get("synthetic_pca64_cov_trace_ratio_to_source_train", math.nan))
    if not math.isnan(cov_ratio) and cov_ratio < 0.5 and float(oracle_delta) < 0.02:
        return DECISION_UNDERDISPERSION
    if not math.isnan(cov_ratio) and cov_ratio > 1.5 and float(oracle_delta) < 0.02:
        return DECISION_OVERDISPERSION
    if float(stability) < 0.67 and float(oracle_delta) < 0.02:
        return DECISION_RANK_INSTABILITY
    mismatch = float(diagnostics.get("decoder_output_norm_mean", 0.0)) > 10.0 * max(float(diagnostics.get("latent_mu_norm_mean", 1.0)), 1.0)
    if mismatch and float(oracle_delta) < 0.02:
        return DECISION_PRIOR_DECODER_MISMATCH
    return DECISION_NO_GAIN


def _mode_effective_components(
    rows: Sequence[Mapping[str, object]],
    heldout: str,
    mode: str,
    fallback: int,
) -> int:
    values = []
    for row in rows:
        if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode:
            try:
                values.append(int(float(row.get("effective_components", fallback))))
            except (TypeError, ValueError):
                pass
    return min(values) if values else int(fallback)


def _mode_diagnostics(rows: Sequence[Mapping[str, object]], heldout: str, mode: str) -> dict[str, float]:
    subset = [row for row in rows if str(row.get("heldout_center")) == heldout and str(row.get("generation_mode")) == mode]
    keys = sorted({key for row in subset for key in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = []
        for row in subset:
            try:
                value = float(row.get(key, math.nan))
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                vals.append(value)
        if vals:
            out[key] = sum(vals) / float(len(vals))
    return out


def _mean_diagnostics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item})
    out: dict[str, float] = {}
    for key in keys:
        vals = []
        for item in items:
            try:
                value = float(item.get(key, math.nan))
            except (TypeError, ValueError):
                continue
            if not math.isnan(value):
                vals.append(value)
        if vals:
            out[key] = sum(vals) / float(len(vals))
    return out


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isnan(value):
            vals.append(value)
    return sum(vals) / float(len(vals)) if vals else math.nan


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
    vals = [value for value in by_context.values() if not math.isnan(value)]
    return sum(vals) / float(len(vals)) if vals else math.nan
