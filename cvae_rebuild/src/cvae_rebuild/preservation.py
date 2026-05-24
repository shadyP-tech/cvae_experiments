from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from .experts import ExpertRuntime, label, to_numpy, train_seed_experts
from .features import default_cache_path, load_feature_cache, select_rows
from .metrics import nanmean, spearman
from .protocol import (
    ProtocolError,
    assert_candidate_pool,
    build_leakage_report,
)
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts


PRESERVATION_NAME = "virchow2_cvae_preservation_diagnosis_v1"
ROW_REAL_FULL = "real_source_full_balanced_classifier"
ROW_REAL_BUDGET = "real_source_budget_matched_classifier"
ROW_DECODE_MU = "cvae_decode_mu_budget_matched"
ROW_POSTERIOR = "cvae_reference_posterior_sample"
ROW_PRIOR = "cvae_prior_sample_budget_matched"
ROW_ROLES = (ROW_REAL_FULL, ROW_REAL_BUDGET, ROW_DECODE_MU, ROW_POSTERIOR, ROW_PRIOR)
CLASSIFIER_TYPE = "sklearn_logistic_regression"
CLASSIFIER_SOLVER = "lbfgs"
CLASSIFIER_C = 1.0
CLASSIFIER_MAX_ITER = 2000
CLASSIFIER_CLASS_WEIGHT = "balanced"
NA = "NA"


@dataclass(frozen=True)
class PreservationConfig:
    name: str
    artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    candidate_count_per_cell: int
    pca_dim: int
    hidden_dim: int
    latent_dim: int
    num_hidden_layers: int
    train_epochs: int
    batch_size: int
    learning_rate: float
    synthetic_per_class_total: int
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def expected_candidate_count(self) -> int:
        return int(self.candidate_count_per_cell)


@dataclass(frozen=True)
class ReferenceSample:
    embeddings: object
    labels: tuple[int, ...]
    reference_ids_hash: str
    diagnostics: tuple[dict[str, object], ...]


def load_preservation_config(path: str | Path) -> PreservationConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_preservation_config(data, base_dir=base_dir)


def parse_preservation_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> PreservationConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    frame = _mapping(data, "feature_frame")
    model = _mapping(data, "model")
    generation = _mapping(data, "generation")
    classifier = _mapping(data, "classifier")
    cfg = PreservationConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        candidate_count_per_cell=int(run["candidate_count_per_cell"]),
        pca_dim=int(frame["pca_dim"]),
        hidden_dim=int(model["hidden_dim"]),
        latent_dim=int(model["latent_dim"]),
        num_hidden_layers=int(model["num_hidden_layers"]),
        train_epochs=int(model.get("train_epochs", 25)),
        batch_size=int(model.get("batch_size", 128)),
        learning_rate=float(model.get("learning_rate", 1.0e-3)),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_preservation_config(cfg)
    return cfg


def validate_preservation_config(cfg: PreservationConfig) -> None:
    if cfg.name != PRESERVATION_NAME:
        raise ProtocolError(f"Preservation experiment name must be {PRESERVATION_NAME!r}.")
    if cfg.candidate_count_per_cell != 4:
        raise ProtocolError("candidate_count_per_cell must be locked to 4.")
    if cfg.pca_dim != 256:
        raise ProtocolError("pca_dim must be locked to 256.")
    if cfg.hidden_dim != 512 or cfg.latent_dim != 64 or cfg.num_hidden_layers != 2:
        raise ProtocolError("CVAE architecture must be locked to hidden=512, latent=64, two layers.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.classifier_type != CLASSIFIER_TYPE:
        raise ProtocolError(f"classifier.type must be {CLASSIFIER_TYPE!r}.")
    if cfg.classifier_solver != CLASSIFIER_SOLVER:
        raise ProtocolError(f"classifier.solver must be {CLASSIFIER_SOLVER!r}.")
    if cfg.classifier_c != CLASSIFIER_C:
        raise ProtocolError(f"classifier.C must be {CLASSIFIER_C}.")
    if cfg.classifier_max_iter != CLASSIFIER_MAX_ITER:
        raise ProtocolError(f"classifier.max_iter must be {CLASSIFIER_MAX_ITER}.")
    if cfg.classifier_class_weight != CLASSIFIER_CLASS_WEIGHT:
        raise ProtocolError(f"classifier.class_weight must be {CLASSIFIER_CLASS_WEIGHT!r}.")
    if cfg.classifier_seed is not None:
        raise ProtocolError("classifier_seed must be null for deterministic lbfgs logistic regression.")


def run_preservation_diagnosis(cfg: PreservationConfig, *, artifact_root: str | Path | None = None) -> Path:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Preservation diagnosis requires numpy and torch.") from exc

    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    downstream_rows: list[dict[str, object]] = []
    sampling_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    expert_manifest_rows: list[dict[str, object]] = []
    target_expert_excluded = True

    for experiment_seed in cfg.experiment_seeds:
        train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
        test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
        experts = train_seed_experts(cfg, train_cache=train_cache, experiment_seed=int(experiment_seed))
        for expert in experts.values():
            expert_manifest_rows.append(_expert_manifest_row(experiment_seed, expert, cfg))

        for heldout_center in cfg.heldout_centers:
            candidates = candidate_experts(cfg.heldout_centers, str(heldout_center))
            try:
                assert_candidate_pool(
                    heldout_center=str(heldout_center),
                    candidate_experts=candidates,
                    expected_count=cfg.expected_candidate_count,
                )
            except Exception:
                target_expert_excluded = False
                raise
            target_indices = _target_indices(test_cache.metadata, str(heldout_center))
            eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
            eval_labels = tuple(label(row) for row in eval_meta)
            eval_class_count = len(set(eval_labels))
            eval_error = "mono_class_target_eval" if eval_class_count < 2 else ""

            for expert_id in candidates:
                expert = experts[str(expert_id)]
                if eval_error:
                    downstream_rows.extend(
                        _ineligible_preservation_rows(
                            cfg=cfg,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            expert=expert,
                            n_target_eval=len(eval_labels),
                            error_message=eval_error,
                        )
                    )
                    continue

                eval_x = expert.frame.transform(to_numpy(eval_raw))
                full_bundle = fit_locked_logistic_classifier(
                    expert.source_train_embeddings,
                    expert.source_train_labels,
                    eval_x,
                    classifier_seed=cfg.classifier_seed,
                    expert_id=expert.expert_id,
                    class_weight=cfg.classifier_class_weight,
                )
                full_result = evaluate_probability_predictions(ROW_REAL_FULL, full_bundle.probabilities, eval_labels)
                downstream_rows.append(
                    _preservation_row(
                        cfg=cfg,
                        expert=expert,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        row_role=ROW_REAL_FULL,
                        generation_mode="real_source_full_balanced",
                        replicate_seed=NA,
                        reference_sample_seed=NA,
                        latent_sample_seed=NA,
                        reference_ids_hash=_hash_strings(expert.source_train_sample_ids),
                        generated_features_hash=_hash_array(expert.source_train_embeddings),
                        budget_per_class="all",
                        result=full_result,
                        n_target_eval=len(eval_labels),
                        source_utility_stratum="not_applicable",
                    )
                )

                for replicate_seed in cfg.replicate_seeds:
                    sample = _sample_source_refs(expert, cfg.synthetic_per_class_total, int(replicate_seed))
                    sampling_rows.extend(
                        _sampling_diagnostic_rows(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            replicate_seed=int(replicate_seed),
                            sample=sample,
                        )
                    )
                    budget_bundle = fit_locked_logistic_classifier(
                        sample.embeddings,
                        sample.labels,
                        eval_x,
                        classifier_seed=cfg.classifier_seed,
                        expert_id=expert.expert_id,
                        class_weight=cfg.classifier_class_weight,
                    )
                    budget_result = evaluate_probability_predictions(ROW_REAL_BUDGET, budget_bundle.probabilities, eval_labels)
                    stratum = _source_utility_stratum(budget_result.bacc)
                    downstream_rows.append(
                        _preservation_row(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            row_role=ROW_REAL_BUDGET,
                            generation_mode="real_source_budget_matched",
                            replicate_seed=int(replicate_seed),
                            reference_sample_seed=int(replicate_seed),
                            latent_sample_seed=NA,
                            reference_ids_hash=sample.reference_ids_hash,
                            generated_features_hash=_hash_array(sample.embeddings),
                            budget_per_class=cfg.synthetic_per_class_total,
                            result=budget_result,
                            n_target_eval=len(eval_labels),
                            source_utility_stratum=stratum,
                        )
                    )

                    decode_embeddings, recon = _decode_mu(expert, sample)
                    reconstruction_rows.append(
                        _reconstruction_row(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            row_role=ROW_DECODE_MU,
                            replicate_seed=int(replicate_seed),
                            diagnostics=recon,
                        )
                    )
                    decode_bundle = fit_locked_logistic_classifier(
                        decode_embeddings,
                        sample.labels,
                        eval_x,
                        classifier_seed=cfg.classifier_seed,
                        expert_id=expert.expert_id,
                        class_weight=cfg.classifier_class_weight,
                    )
                    decode_result = evaluate_probability_predictions(ROW_DECODE_MU, decode_bundle.probabilities, eval_labels)
                    downstream_rows.append(
                        _preservation_row(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            row_role=ROW_DECODE_MU,
                            generation_mode="cvae_decode_mu",
                            replicate_seed=int(replicate_seed),
                            reference_sample_seed=int(replicate_seed),
                            latent_sample_seed=NA,
                            reference_ids_hash=sample.reference_ids_hash,
                            generated_features_hash=_hash_array(decode_embeddings),
                            budget_per_class=cfg.synthetic_per_class_total,
                            result=decode_result,
                            n_target_eval=len(eval_labels),
                            source_utility_stratum=stratum,
                        )
                    )

                    posterior_embeddings, posterior_diag = _posterior_sample(expert, sample, int(replicate_seed))
                    reconstruction_rows.append(
                        _reconstruction_row(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            row_role=ROW_POSTERIOR,
                            replicate_seed=int(replicate_seed),
                            diagnostics=posterior_diag,
                        )
                    )
                    posterior_bundle = fit_locked_logistic_classifier(
                        posterior_embeddings,
                        sample.labels,
                        eval_x,
                        classifier_seed=cfg.classifier_seed,
                        expert_id=expert.expert_id,
                        class_weight=cfg.classifier_class_weight,
                    )
                    posterior_result = evaluate_probability_predictions(
                        ROW_POSTERIOR,
                        posterior_bundle.probabilities,
                        eval_labels,
                    )
                    downstream_rows.append(
                        _preservation_row(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            row_role=ROW_POSTERIOR,
                            generation_mode="cvae_reference_posterior_sample",
                            replicate_seed=int(replicate_seed),
                            reference_sample_seed=int(replicate_seed),
                            latent_sample_seed=int(replicate_seed),
                            reference_ids_hash=sample.reference_ids_hash,
                            generated_features_hash=_hash_array(posterior_embeddings),
                            budget_per_class=cfg.synthetic_per_class_total,
                            result=posterior_result,
                            n_target_eval=len(eval_labels),
                            source_utility_stratum=stratum,
                        )
                    )

                    prior_embeddings, prior_labels = _prior_sample(expert, cfg.synthetic_per_class_total, int(replicate_seed))
                    prior_bundle = fit_locked_logistic_classifier(
                        prior_embeddings,
                        prior_labels,
                        eval_x,
                        classifier_seed=cfg.classifier_seed,
                        expert_id=expert.expert_id,
                        class_weight=cfg.classifier_class_weight,
                    )
                    prior_result = evaluate_probability_predictions(ROW_PRIOR, prior_bundle.probabilities, eval_labels)
                    downstream_rows.append(
                        _preservation_row(
                            cfg=cfg,
                            expert=expert,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            row_role=ROW_PRIOR,
                            generation_mode="cvae_prior_sample",
                            replicate_seed=int(replicate_seed),
                            reference_sample_seed=NA,
                            latent_sample_seed=int(replicate_seed),
                            reference_ids_hash="",
                            generated_features_hash=_hash_array(prior_embeddings),
                            budget_per_class=cfg.synthetic_per_class_total,
                            result=prior_result,
                            n_target_eval=len(eval_labels),
                            source_utility_stratum=stratum,
                        )
                    )

    gap_rows = _gap_rows(downstream_rows)
    summary_rows, decision = _summary_rows(gap_rows)
    expert_summary_rows = _expert_summary_rows(gap_rows)
    observed_total = len(downstream_rows)
    observed_ineligible = sum(1 for row in downstream_rows if row.get("status") == "ineligible")
    expected_total = _expected_total_rows(cfg)
    expected_ineligible = observed_ineligible
    expected_eligible = expected_total - expected_ineligible

    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
    )
    write_json(root / "reports" / "leakage_report.json", {
        **report.to_json_dict(),
        "expected_total_rows": expected_total,
        "expected_ineligible_rows": expected_ineligible,
        "expected_eligible_rows": expected_eligible,
        "observed_total_rows": observed_total,
        "observed_ineligible_rows": observed_ineligible,
    })
    _write_protocol_manifest(root, cfg, expected_total, expected_ineligible, expected_eligible)
    _write_decision_summary(
        root,
        decision=decision,
        leakage_status=report.status,
        observed_total=observed_total,
        observed_ineligible=observed_ineligible,
        expected_total=expected_total,
        expected_ineligible=expected_ineligible,
        expected_eligible=expected_eligible,
    )
    write_csv_rows(root / "tables" / "preservation_downstream_matrix.csv", downstream_rows)
    write_csv_rows(root / "tables" / "preservation_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "cell_preservation_summary.csv", summary_rows)
    write_csv_rows(root / "tables" / "expert_preservation_summary.csv", expert_summary_rows)
    write_csv_rows(root / "tables" / "reconstruction_diagnostics.csv", reconstruction_rows)
    write_csv_rows(root / "tables" / "reference_sampling_diagnostics.csv", sampling_rows)
    write_csv_rows(root / "manifests" / "expert_manifest.csv", expert_manifest_rows)
    write_preservation_resolved_config(root / "run_config_resolved.yaml", cfg)
    return root


def _preservation_row(
    *,
    cfg: PreservationConfig,
    expert: ExpertRuntime,
    experiment_seed: int,
    heldout_center: str,
    row_role: str,
    generation_mode: str,
    replicate_seed: int | str,
    reference_sample_seed: int | str,
    latent_sample_seed: int | str,
    reference_ids_hash: str,
    generated_features_hash: str,
    budget_per_class: int | str,
    result: object,
    n_target_eval: int,
    source_utility_stratum: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": expert.expert_id,
        "row_role": row_role,
        "generation_mode": generation_mode,
        "replicate_seed": replicate_seed,
        "reference_sample_seed": reference_sample_seed,
        "latent_sample_seed": latent_sample_seed,
        "reference_ids_hash": reference_ids_hash,
        "generated_features_hash": generated_features_hash,
        "requested_pca_dim": cfg.pca_dim,
        "effective_pca_dim": expert.effective_dim,
        "budget_per_class": budget_per_class,
        "classifier_type": cfg.classifier_type,
        "classifier_class_weight": cfg.classifier_class_weight,
        "bacc": getattr(result, "bacc"),
        "macro_f1": getattr(result, "macro_f1"),
        "n_target_eval": int(n_target_eval),
        "source_utility_stratum": source_utility_stratum,
        "selection_source": "all_source_expert_evaluation",
        "status": "ok",
        "error_message": "",
    }


def _ineligible_preservation_rows(
    *,
    cfg: PreservationConfig,
    experiment_seed: int,
    heldout_center: str,
    expert: ExpertRuntime,
    n_target_eval: int,
    error_message: str,
) -> list[dict[str, object]]:
    rows = [
        _ineligible_row(
            cfg=cfg,
            expert=expert,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            row_role=ROW_REAL_FULL,
            generation_mode="real_source_full_balanced",
            replicate_seed=NA,
            reference_sample_seed=NA,
            latent_sample_seed=NA,
            reference_ids_hash="",
            budget_per_class="all",
            n_target_eval=n_target_eval,
            error_message=error_message,
        )
    ]
    for seed in cfg.replicate_seeds:
        for row_role, generation_mode, reference_seed, latent_seed in (
            (ROW_REAL_BUDGET, "real_source_budget_matched", int(seed), NA),
            (ROW_DECODE_MU, "cvae_decode_mu", int(seed), NA),
            (ROW_POSTERIOR, "cvae_reference_posterior_sample", int(seed), int(seed)),
            (ROW_PRIOR, "cvae_prior_sample", NA, int(seed)),
        ):
            rows.append(
                _ineligible_row(
                    cfg=cfg,
                    expert=expert,
                    experiment_seed=experiment_seed,
                    heldout_center=heldout_center,
                    row_role=row_role,
                    generation_mode=generation_mode,
                    replicate_seed=int(seed),
                    reference_sample_seed=reference_seed,
                    latent_sample_seed=latent_seed,
                    reference_ids_hash="",
                    budget_per_class=cfg.synthetic_per_class_total,
                    n_target_eval=n_target_eval,
                    error_message=error_message,
                )
            )
    return rows


def _ineligible_row(
    *,
    cfg: PreservationConfig,
    expert: ExpertRuntime,
    experiment_seed: int,
    heldout_center: str,
    row_role: str,
    generation_mode: str,
    replicate_seed: int | str,
    reference_sample_seed: int | str,
    latent_sample_seed: int | str,
    reference_ids_hash: str,
    budget_per_class: int | str,
    n_target_eval: int,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": expert.expert_id,
        "row_role": row_role,
        "generation_mode": generation_mode,
        "replicate_seed": replicate_seed,
        "reference_sample_seed": reference_sample_seed,
        "latent_sample_seed": latent_sample_seed,
        "reference_ids_hash": reference_ids_hash,
        "generated_features_hash": "",
        "requested_pca_dim": cfg.pca_dim,
        "effective_pca_dim": expert.effective_dim,
        "budget_per_class": budget_per_class,
        "classifier_type": cfg.classifier_type,
        "classifier_class_weight": cfg.classifier_class_weight,
        "bacc": "",
        "macro_f1": "",
        "n_target_eval": int(n_target_eval),
        "source_utility_stratum": "",
        "selection_source": "all_source_expert_evaluation",
        "status": "ineligible",
        "error_message": str(error_message),
    }


def _sample_source_refs(expert: ExpertRuntime, budget_per_class: int, seed: int) -> ReferenceSample:
    import numpy as np  # type: ignore

    x = np.asarray(expert.source_train_embeddings, dtype=float)
    y = np.asarray(expert.source_train_labels, dtype=int)
    ids = tuple(str(v) for v in expert.source_train_sample_ids)
    rng = np.random.default_rng(int(seed))
    chunks = []
    labels: list[int] = []
    all_ids: list[str] = []
    diagnostics = []
    for cls in (0, 1):
        pool = np.where(y == int(cls))[0]
        if pool.size == 0:
            raise ProtocolError(f"Expert {expert.expert_id} has no source refs for class {cls}.")
        chosen_offsets = rng.integers(0, pool.size, size=int(budget_per_class))
        chosen = pool[chosen_offsets]
        chosen_ids = [ids[int(idx)] for idx in chosen]
        chunks.append(x[chosen])
        labels.extend([int(cls)] * int(budget_per_class))
        all_ids.extend(f"{cls}:{sample_id}" for sample_id in chosen_ids)
        unique_count = len(set(chosen_ids))
        diagnostics.append(
            {
                "class_label": int(cls),
                "source_class_count": int(pool.size),
                "budget_per_class_requested": int(budget_per_class),
                "budget_per_class_effective": int(budget_per_class),
                "n_unique_refs_per_class": int(unique_count),
                "duplication_rate_per_class": 1.0 - (float(unique_count) / float(budget_per_class)),
                "reference_ids_hash": _hash_strings(chosen_ids),
            }
        )
    return ReferenceSample(
        embeddings=np.vstack(chunks),
        labels=tuple(labels),
        reference_ids_hash=_hash_strings(all_ids),
        diagnostics=tuple(diagnostics),
    )


def _decode_mu(expert: ExpertRuntime, sample: ReferenceSample) -> tuple[object, dict[str, float]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(sample.embeddings, dtype=np.float32)
    y_np = np.asarray(sample.labels, dtype=np.int64)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = expert.model.encode(x, y)
        decoded = expert.model.decode(mu, y).detach().cpu().numpy()
    return decoded, _reconstruction_diagnostics(x_np, decoded, mu, logvar)


def _posterior_sample(expert: ExpertRuntime, sample: ReferenceSample, seed: int) -> tuple[object, dict[str, float]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(sample.embeddings, dtype=np.float32)
    y_np = np.asarray(sample.labels, dtype=np.int64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = expert.model.encode(x, y)
        noise = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=mu.device)
        z = mu + (noise * torch.exp(0.5 * logvar))
        decoded = expert.model.decode(z, y).detach().cpu().numpy()
    return decoded, _reconstruction_diagnostics(x_np, decoded, mu, logvar)


def _prior_sample(expert: ExpertRuntime, budget_per_class: int, seed: int) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chunks = []
    labels: list[int] = []
    with torch.no_grad():
        for cls in (0, 1):
            y = torch.full((int(budget_per_class),), int(cls), dtype=torch.long)
            z = torch.randn(
                (int(budget_per_class), int(expert.model.latent_dim)),
                generator=generator,
                dtype=torch.float32,
            )
            decoded = expert.model.decode(z, y).detach().cpu().numpy()
            chunks.append(decoded)
            labels.extend([int(cls)] * int(budget_per_class))
    return np.vstack(chunks), tuple(labels)


def _reconstruction_diagnostics(original: object, generated: object, mu: object, logvar: object) -> dict[str, float]:
    import numpy as np  # type: ignore

    x = np.asarray(original, dtype=float)
    y = np.asarray(generated, dtype=float)
    mse = np.mean((x - y) ** 2, axis=1)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    cosine = np.divide(np.sum(x * y, axis=1), denom, out=np.zeros_like(denom), where=denom > 0.0)
    logvar_np = logvar.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    sigma = np.exp(0.5 * logvar_np)
    kl = -0.5 * np.sum(1.0 + logvar_np - (mu_np ** 2) - np.exp(logvar_np), axis=1)
    return {
        "recon_mse_mean": float(np.mean(mse)),
        "recon_cosine_mean": float(np.mean(cosine)),
        "posterior_sigma_mean": float(np.mean(sigma)),
        "posterior_sigma_p95": float(np.percentile(sigma, 95)),
        "latent_kl_mean": float(np.mean(kl)),
    }


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    full_by_key = {
        (row["experiment_seed"], row["heldout_center"], row["expert_id"]): row
        for row in ok_rows
        if row.get("row_role") == ROW_REAL_FULL
    }
    by_role = {
        (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["replicate_seed"], row["row_role"]): row
        for row in ok_rows
        if row.get("replicate_seed") != NA
    }
    out: list[dict[str, object]] = []
    for key_role, budget in by_role.items():
        experiment_seed, heldout_center, expert_id, replicate_seed, row_role = key_role
        if row_role != ROW_REAL_BUDGET:
            continue
        full = full_by_key.get((experiment_seed, heldout_center, expert_id))
        decode = by_role.get((experiment_seed, heldout_center, expert_id, replicate_seed, ROW_DECODE_MU))
        posterior = by_role.get((experiment_seed, heldout_center, expert_id, replicate_seed, ROW_POSTERIOR))
        prior = by_role.get((experiment_seed, heldout_center, expert_id, replicate_seed, ROW_PRIOR))
        if not (full and decode and posterior and prior):
            continue
        real_budget_bacc = _float(budget["bacc"])
        stratum = _source_utility_stratum(real_budget_bacc)
        chance_adjusted = (
            (_float(prior["bacc"]) - 0.5) / max(real_budget_bacc - 0.5, 1.0e-12)
            if real_budget_bacc > 0.55
            else math.nan
        )
        out.append(
            {
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "expert_id": expert_id,
                "replicate_seed": replicate_seed,
                "source_utility_stratum": stratum,
                "real_source_full_bacc": full["bacc"],
                "real_source_budget_matched_bacc": budget["bacc"],
                "cvae_decode_mu_bacc": decode["bacc"],
                "cvae_reference_posterior_bacc": posterior["bacc"],
                "cvae_prior_sample_bacc": prior["bacc"],
                "budget_gap": _float(full["bacc"]) - real_budget_bacc,
                "decoder_gap": real_budget_bacc - _float(decode["bacc"]),
                "posterior_gap": _float(decode["bacc"]) - _float(posterior["bacc"]),
                "prior_gap": _float(posterior["bacc"]) - _float(prior["bacc"]),
                "total_prior_cvae_gap": real_budget_bacc - _float(prior["bacc"]),
                "chance_adjusted_preservation": "" if math.isnan(chance_adjusted) else chance_adjusted,
                "real_source_full_macro_f1": full["macro_f1"],
                "real_source_budget_matched_macro_f1": budget["macro_f1"],
                "cvae_decode_mu_macro_f1": decode["macro_f1"],
                "cvae_reference_posterior_macro_f1": posterior["macro_f1"],
                "cvae_prior_sample_macro_f1": prior["macro_f1"],
                "macro_f1_budget_gap": _float(full["macro_f1"]) - _float(budget["macro_f1"]),
                "macro_f1_decoder_gap": _float(budget["macro_f1"]) - _float(decode["macro_f1"]),
                "macro_f1_posterior_gap": _float(decode["macro_f1"]) - _float(posterior["macro_f1"]),
                "macro_f1_prior_gap": _float(posterior["macro_f1"]) - _float(prior["macro_f1"]),
                "macro_f1_total_prior_cvae_gap": _float(budget["macro_f1"]) - _float(prior["macro_f1"]),
                "reference_ids_hash": budget["reference_ids_hash"],
                "status": "ok",
            }
        )
    return out


def _summary_rows(gap_rows: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_rows = list(gap_rows)
    informative = [row for row in all_rows if row.get("source_utility_stratum") in {"medium_real_utility", "high_real_utility"}]
    high = [row for row in all_rows if row.get("source_utility_stratum") == "high_real_utility"]
    rows.append(_summary_row("global_all_sources", all_rows))
    rows.append(_summary_row("global_medium_high_real_utility", informative))
    rows.append(_summary_row("global_high_real_utility_only", high))
    for center in sorted({str(row["heldout_center"]) for row in all_rows}):
        rows.append(_summary_row(f"heldout_center_{center}", [row for row in all_rows if str(row["heldout_center"]) == center]))
    for seed in sorted({str(row["experiment_seed"]) for row in all_rows}):
        rows.append(_summary_row(f"experiment_seed_{seed}", [row for row in all_rows if str(row["experiment_seed"]) == seed]))
    rows.extend(_best_oracle_summary_rows(all_rows))
    decision = _decision(informative, all_rows)
    return rows, decision


def _summary_row(summary_level: str, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fields = (
        "real_source_full_bacc",
        "real_source_budget_matched_bacc",
        "cvae_decode_mu_bacc",
        "cvae_reference_posterior_bacc",
        "cvae_prior_sample_bacc",
        "budget_gap",
        "decoder_gap",
        "posterior_gap",
        "prior_gap",
        "total_prior_cvae_gap",
        "chance_adjusted_preservation",
    )
    out: dict[str, object] = {"summary_level": summary_level, "n": len(rows)}
    for field in fields:
        out[f"mean_{field}"] = _mean_field(rows, field)
        out[f"median_{field}"] = _median_field(rows, field)
    out["seed_std_cvae_prior_sample_bacc"] = _seed_std(rows)
    return out


def _expert_summary_rows(gap_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    keys = sorted({(row["experiment_seed"], row["heldout_center"], row["expert_id"]) for row in gap_rows})
    for experiment_seed, heldout_center, expert_id in keys:
        subset = [
            row for row in gap_rows
            if row["experiment_seed"] == experiment_seed
            and row["heldout_center"] == heldout_center
            and row["expert_id"] == expert_id
        ]
        row = _summary_row("expert", subset)
        row.update({"experiment_seed": experiment_seed, "heldout_center": heldout_center, "expert_id": expert_id})
        out.append(row)
    return out


def _best_oracle_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    keys = sorted({(row["experiment_seed"], row["heldout_center"], row["replicate_seed"]) for row in rows})
    for experiment_seed, heldout_center, replicate_seed in keys:
        subset = [
            row for row in rows
            if row["experiment_seed"] == experiment_seed
            and row["heldout_center"] == heldout_center
            and row["replicate_seed"] == replicate_seed
        ]
        if not subset:
            continue
        best_real = max(subset, key=lambda row: (_float(row["real_source_budget_matched_bacc"]), str(row["expert_id"])))
        best_prior = max(subset, key=lambda row: (_float(row["cvae_prior_sample_bacc"]), str(row["expert_id"])))
        out.append(
            {
                "summary_level": "best_real_budget_source_oracle",
                "experiment_seed": experiment_seed,
                "heldout_center": heldout_center,
                "replicate_seed": replicate_seed,
                "best_real_budget_source_expert": best_real["expert_id"],
                "best_real_budget_source_bacc": best_real["real_source_budget_matched_bacc"],
                "best_cvae_prior_source_expert": best_prior["expert_id"],
                "best_cvae_prior_source_bacc": best_prior["cvae_prior_sample_bacc"],
                "oracle_preservation_gap": _float(best_real["real_source_budget_matched_bacc"]) - _float(best_prior["cvae_prior_sample_bacc"]),
                "best_source_expert_changed_by_cvae": int(best_real["expert_id"] != best_prior["expert_id"]),
                "spearman_real_budget_vs_cvae_prior_bacc": spearman(
                    [_float(row["real_source_budget_matched_bacc"]) for row in subset],
                    [_float(row["cvae_prior_sample_bacc"]) for row in subset],
                ),
                "top1_preservation_oracle_hit": float(best_real["expert_id"] == best_prior["expert_id"]),
                "selection_source": "diagnostic_only",
            }
        )
    return out


def _decision(informative: Sequence[Mapping[str, object]], all_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    medium_high_fraction = (
        float(len(informative)) / float(len(all_rows))
        if all_rows else 0.0
    )
    source_transfer_weak = _mean_field(all_rows, "real_source_budget_matched_bacc") < 0.65 or medium_high_fraction < 0.25
    prior_mean = _mean_field(informative, "cvae_prior_sample_bacc")
    cap = _mean_field(informative, "chance_adjusted_preservation")
    total_gap = _mean_field(informative, "total_prior_cvae_gap")
    seed_std = _seed_std(informative)
    real_budget_mean = _mean_field(informative, "real_source_budget_matched_bacc")
    outcome = "DIAGNOSTIC_MIXED"
    if source_transfer_weak:
        outcome = "SOURCE_TRANSFER_WEAK"
    elif prior_mean >= 0.80 and cap >= 0.80 and total_gap <= 0.05 and seed_std <= 0.03:
        outcome = "PRESERVATION_PASS"
    elif prior_mean >= 0.70 and cap >= 0.60 and total_gap <= 0.12:
        outcome = "PRESERVATION_PARTIAL"
    elif real_budget_mean >= 0.75 and prior_mean <= 0.60:
        outcome = "PRESERVATION_FAIL"

    bottlenecks = []
    if _mean_field(informative, "real_source_full_bacc") >= 0.80 and _mean_field(informative, "budget_gap") >= 0.08:
        bottlenecks.append("BUDGET_FAILURE")
    if real_budget_mean >= 0.80 and _mean_field(informative, "decoder_gap") >= 0.08:
        bottlenecks.append("DECODER_FAILURE")
    if _mean_field(informative, "cvae_decode_mu_bacc") >= 0.75 and _mean_field(informative, "posterior_gap") >= 0.05:
        bottlenecks.append("POSTERIOR_SAMPLING_FAILURE")
    if _mean_field(informative, "cvae_reference_posterior_bacc") >= 0.75 and _mean_field(informative, "prior_gap") >= 0.05:
        bottlenecks.append("PRIOR_SAMPLING_FAILURE")
    return {
        "preservation_outcome": outcome,
        "bottleneck_labels": "|".join(bottlenecks),
        "medium_high_fraction": medium_high_fraction,
        "seed_std": seed_std,
        "mean_cvae_prior_sample_bacc": prior_mean,
        "mean_total_prior_cvae_gap": total_gap,
        "mean_chance_adjusted_preservation": cap,
    }


def _source_utility_stratum(bacc: float) -> str:
    if float(bacc) >= 0.80:
        return "high_real_utility"
    if float(bacc) >= 0.65:
        return "medium_real_utility"
    return "low_real_utility"


def _expert_manifest_row(experiment_seed: int, expert: ExpertRuntime, cfg: PreservationConfig) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "expert_id": expert.expert_id,
        "checkpoint_path": "in_memory_not_serialized",
        "n_train": expert.n_train,
        "n_val": expert.n_val,
        "requested_pca_dim": cfg.pca_dim,
        "effective_pca_dim": expert.effective_dim,
        "n_train_for_pca": expert.n_train,
        "pca_explained_variance_ratio_sum": expert.frame.explained_variance_ratio_sum,
        "decoder_output_dim": int(expert.model.input_dim),
        "source_val_split_id": expert.source_val_split.split_id,
    }


def _sampling_diagnostic_rows(
    *,
    cfg: PreservationConfig,
    expert: ExpertRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    sample: ReferenceSample,
) -> list[dict[str, object]]:
    rows = []
    for diag in sample.diagnostics:
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": expert.expert_id,
                "replicate_seed": int(replicate_seed),
                "reference_sample_seed": int(replicate_seed),
                "latent_sample_seed": NA,
                "reference_ids_hash": sample.reference_ids_hash,
                "budget_per_class_requested": cfg.synthetic_per_class_total,
                **diag,
            }
        )
    return rows


def _reconstruction_row(
    *,
    cfg: PreservationConfig,
    expert: ExpertRuntime,
    experiment_seed: int,
    heldout_center: str,
    row_role: str,
    replicate_seed: int,
    diagnostics: Mapping[str, float],
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": expert.expert_id,
        "row_role": row_role,
        "replicate_seed": int(replicate_seed),
        "requested_pca_dim": cfg.pca_dim,
        "effective_pca_dim": expert.effective_dim,
        **diagnostics,
    }


def _write_protocol_manifest(
    root: Path,
    cfg: PreservationConfig,
    expected_total: int,
    expected_ineligible: int,
    expected_eligible: int,
) -> None:
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_preservation_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "preservation_diagnosis",
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": True,
            "oracle_role": "summary_diagnostic_only",
            "row_roles": list(ROW_ROLES),
            "expected_total_rows": int(expected_total),
            "expected_ineligible_rows": int(expected_ineligible),
            "expected_eligible_rows": int(expected_eligible),
        },
    )


def _write_decision_summary(
    root: Path,
    *,
    decision: Mapping[str, object],
    leakage_status: str,
    observed_total: int,
    observed_ineligible: int,
    expected_total: int,
    expected_ineligible: int,
    expected_eligible: int,
) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Preservation Diagnosis v1",
            "",
            "## Summary",
            "",
            f"- Preservation outcome: `{decision.get('preservation_outcome', 'DIAGNOSTIC_MIXED')}`",
            f"- Bottleneck labels: `{decision.get('bottleneck_labels', '')}`",
            f"- Mean CVAE prior BACC: {float(decision.get('mean_cvae_prior_sample_bacc', math.nan)):.4f}",
            f"- Mean total prior CVAE gap: {float(decision.get('mean_total_prior_cvae_gap', math.nan)):.4f}",
            f"- Mean chance-adjusted preservation: {float(decision.get('mean_chance_adjusted_preservation', math.nan)):.4f}",
            f"- Seed std: {float(decision.get('seed_std', math.nan)):.4f}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Row Counts",
            "",
            f"- Expected total rows before mono-class exclusion: {expected_total}",
            f"- Expected ineligible rows: {expected_ineligible}",
            f"- Expected eligible rows: {expected_eligible}",
            f"- Observed total rows: {observed_total}",
            f"- Observed ineligible rows: {observed_ineligible}",
            "",
            "## Claim Boundary",
            "",
            "This slice diagnoses CVAE preservation under fixed source-expert evaluation.",
            "It does not evaluate support-NELBO routing, metadata routing, or top-k composition.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_preservation_resolved_config(path: str | Path, cfg: PreservationConfig) -> None:
    Path(path).write_text(json.dumps(_resolved_config_dict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expected_total_rows(cfg: PreservationConfig) -> int:
    return (
        len(cfg.experiment_seeds)
        * len(cfg.heldout_centers)
        * int(cfg.candidate_count_per_cell)
        * (1 + (4 * len(cfg.replicate_seeds)))
    )


def _resolved_config_dict(cfg: PreservationConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "candidate_count_per_cell": cfg.candidate_count_per_cell,
        "pca_dim": cfg.pca_dim,
        "hidden_dim": cfg.hidden_dim,
        "latent_dim": cfg.latent_dim,
        "num_hidden_layers": cfg.num_hidden_layers,
        "train_epochs": cfg.train_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "classifier_type": cfg.classifier_type,
        "classifier_solver": cfg.classifier_solver,
        "classifier_c": cfg.classifier_c,
        "classifier_max_iter": cfg.classifier_max_iter,
        "classifier_class_weight": cfg.classifier_class_weight,
        "classifier_seed": cfg.classifier_seed,
    }


def _target_indices(metadata: Sequence[Mapping[str, object]], heldout_center: str) -> tuple[int, ...]:
    return tuple(
        idx for idx, row in enumerate(metadata)
        if str(row.get("center", row.get("magnification"))) == str(heldout_center)
    )


def _existing_cache_path(root: str | Path, *, seed: int, split: str) -> Path:
    pt_path = default_cache_path(root, seed=int(seed), split=str(split))
    if pt_path.exists():
        return pt_path
    npz_path = pt_path.with_suffix(".npz")
    if npz_path.exists():
        return npz_path
    return pt_path


def _hash_strings(values: Sequence[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _hash_array(value: object) -> str:
    import numpy as np  # type: ignore

    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", "NA"}])


def _median_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    vals = sorted(_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", "NA"})
    if not vals:
        return math.nan
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _seed_std(rows: Sequence[Mapping[str, object]]) -> float:
    import statistics

    by_seed: dict[str, list[float]] = {}
    for row in rows:
        by_seed.setdefault(str(row.get("experiment_seed", "")), []).append(_float(row.get("cvae_prior_sample_bacc", math.nan)))
    vals = [nanmean(values) for values in by_seed.values()]
    vals = [value for value in vals if math.isfinite(value)]
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def _float(value: object) -> float:
    if value in ("", NA, None):
        return math.nan
    return float(value)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ProtocolError("YAML config parsing requires PyYAML unless the file is JSON syntax.") from exc
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ProtocolError("Config root must be a mapping.")
        return data


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Config section {key!r} must be a mapping.")
    return value


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()
