from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from .features import load_feature_cache, select_rows
from .metrics import nanmean
from .preservation import _hash_array
from .preservation_repair import (
    NA,
    POOL_PER_SOURCE,
    POOL_SOURCE_UNION,
    PRIMARY_VARIANT,
    RepairConfig,
    RepairVariant,
    SourceData,
    SourceProbeConfig,
    VariantRuntime,
    _existing_cache_path,
    _float,
    _format_float,
    _hash_strings,
    _label,
    _load_mapping,
    _mapping,
    _path,
    _sample_source_positions,
    _source_data_for_centers,
    _stable_seed,
    _subset_rows,
    _target_indices,
    _to_numpy,
    _runtime_for_variant,
)
from .protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from .reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .splits import candidate_experts


SAMPLING_NAME = "virchow2_cvae_pca64_sampling_continuation_v1"
ROW_REAL_BUDGET = "real_source_budget_matched_classifier"
ROW_DECODE_MU = "cvae_decode_mu_budget_matched"
ROW_POSTERIOR = "cvae_posterior_sample_budget_matched"
ROW_EMPIRICAL_MU = "cvae_empirical_mu_sample_diagnostic"
ROW_EMPIRICAL_POSTERIOR = "cvae_empirical_posterior_sample_diagnostic"
ROW_PRIOR = "cvae_prior_sample_budget_matched"
ROW_ROLES = (
    ROW_REAL_BUDGET,
    ROW_DECODE_MU,
    ROW_POSTERIOR,
    ROW_EMPIRICAL_MU,
    ROW_EMPIRICAL_POSTERIOR,
    ROW_PRIOR,
)
UNION_VARIANT = "source_union_pca64_beta001_diagnostic"
PRIMARY_SELECTION = "primary"
DIAGNOSTIC_SELECTION = "diagnostic_only"


@dataclass(frozen=True)
class SamplingConfig:
    name: str
    artifact_root: Path
    repair_artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    min_decision_cells: int
    posterior_temperatures_primary: tuple[float, ...]
    posterior_temperatures_diagnostic: tuple[float, ...]
    prior_scales_primary: tuple[float, ...]
    prior_scales_diagnostic: tuple[float, ...]
    empirical_posterior_temperature: float
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None

    @property
    def posterior_temperatures(self) -> tuple[float, ...]:
        return self.posterior_temperatures_primary + self.posterior_temperatures_diagnostic

    @property
    def prior_scales(self) -> tuple[float, ...]:
        return self.prior_scales_primary + self.prior_scales_diagnostic


@dataclass(frozen=True)
class FrozenReference:
    reference_real_budget_bacc: float
    variant_real_budget_bacc: float
    source_utility_stratum_reference: str
    source_budget_index_hash: str


@dataclass(frozen=True)
class RuntimeSource:
    runtime: VariantRuntime
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_reused_from_repair: bool


def load_sampling_config(path: str | Path) -> SamplingConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_sampling_config(data, base_dir=base_dir)


def parse_sampling_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> SamplingConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    sampling = _mapping(data, "sampling")
    classifier = _mapping(data, "classifier")
    cfg = SamplingConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        repair_artifact_root=_path(base, str(inputs["repair_artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        min_decision_cells=int(experiment.get("min_decision_cells", 10)),
        posterior_temperatures_primary=tuple(float(v) for v in sampling["posterior_temperatures_primary"]),
        posterior_temperatures_diagnostic=tuple(float(v) for v in sampling["posterior_temperatures_diagnostic"]),
        prior_scales_primary=tuple(float(v) for v in sampling["prior_scales_primary"]),
        prior_scales_diagnostic=tuple(float(v) for v in sampling["prior_scales_diagnostic"]),
        empirical_posterior_temperature=float(sampling["empirical_posterior_temperature"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_sampling_config(cfg)
    return cfg


def validate_sampling_config(cfg: SamplingConfig) -> None:
    if cfg.name != SAMPLING_NAME:
        raise ProtocolError(f"Sampling experiment name must be {SAMPLING_NAME!r}.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.min_decision_cells < 1:
        raise ProtocolError("min_decision_cells must be positive.")
    if cfg.posterior_temperatures_primary != (1.0,):
        raise ProtocolError("Primary posterior temperature must be exactly [1.0].")
    if cfg.posterior_temperatures_diagnostic != (0.25, 0.5):
        raise ProtocolError("Diagnostic posterior temperatures must be exactly [0.25, 0.5].")
    if cfg.prior_scales_primary != (1.0,):
        raise ProtocolError("Primary prior scale must be exactly [1.0].")
    if cfg.prior_scales_diagnostic != (0.25, 0.5):
        raise ProtocolError("Diagnostic prior scales must be exactly [0.25, 0.5].")
    if not math.isclose(cfg.empirical_posterior_temperature, 1.0):
        raise ProtocolError("empirical_posterior_temperature must be 1.0.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")


def run_preservation_sampling(cfg: SamplingConfig, *, artifact_root: str | Path | None = None) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    downstream_rows: list[dict[str, object]] = []
    latent_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    protocol_violations: list[str] = []
    target_expert_excluded = True

    try:
        frozen = _load_frozen_repair_reference(cfg)
    except ProtocolError as exc:
        protocol_violations.append(str(exc))
        _write_sampling_artifacts(
            root,
            cfg,
            downstream_rows=[],
            gap_rows=[],
            latent_rows=[],
            manifest_rows=[],
            decision=_decision([], cfg, leakage_status="FAIL"),
            leakage_status="FAIL",
            protocol_violations=protocol_violations,
            target_expert_excluded=True,
        )
        return root

    repair_cfg = _repair_runtime_config(cfg, root)
    per_source_variant = _per_source_variant()
    union_variant = _union_variant()

    try:
        for experiment_seed in cfg.experiment_seeds:
            train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
            test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
            per_source_data = {
                center: _source_data_for_centers(train_cache, centers=(center,), experiment_seed=int(experiment_seed))
                for center in cfg.heldout_centers
            }
            per_source_runtime: dict[str, RuntimeSource] = {}
            for expert_id, source_data in per_source_data.items():
                per_source_runtime[str(expert_id)] = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(expert_id),
                    source_data=source_data,
                    variant=per_source_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, NA, per_source_runtime[str(expert_id)]))

            for heldout_center in cfg.heldout_centers:
                candidates = candidate_experts(cfg.heldout_centers, str(heldout_center))
                try:
                    assert_candidate_pool(
                        heldout_center=str(heldout_center),
                        candidate_experts=candidates,
                        expected_count=len(cfg.heldout_centers) - 1,
                    )
                except Exception:
                    target_expert_excluded = False
                    raise

                union_data = _source_data_for_centers(train_cache, centers=candidates, experiment_seed=int(experiment_seed))
                union_runtime = _runtime_source(
                    cfg,
                    repair_cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    expert_id=POOL_SOURCE_UNION,
                    source_data=union_data,
                    variant=union_variant,
                )
                manifest_rows.append(_manifest_row(experiment_seed, str(heldout_center), union_runtime))

                target_indices = _target_indices(test_cache.metadata, str(heldout_center))
                eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
                eval_labels = tuple(_label(row) for row in eval_meta)
                eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

                for expert_id in candidates:
                    runtime_source = per_source_runtime[str(expert_id)]
                    latent_rows.extend(_latent_rows(runtime_source.runtime, int(experiment_seed), str(heldout_center)))
                    downstream_rows.extend(
                        _evaluate_runtime(
                            cfg,
                            frozen,
                            runtime_source.runtime,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            eval_error=eval_error,
                        )
                    )
                latent_rows.extend(_latent_rows(union_runtime.runtime, int(experiment_seed), str(heldout_center)))
                downstream_rows.extend(
                    _evaluate_runtime(
                        cfg,
                        frozen,
                        union_runtime.runtime,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_error=eval_error,
                    )
                )
    except ProtocolError as exc:
        protocol_violations.append(str(exc))

    _augment_gap_fields(downstream_rows)
    gap_rows = _gap_rows(downstream_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    decision = _decision(gap_rows, cfg, leakage_status=leakage.status)
    _write_sampling_artifacts(
        root,
        cfg,
        downstream_rows=downstream_rows,
        gap_rows=gap_rows,
        latent_rows=latent_rows,
        manifest_rows=manifest_rows,
        decision=decision,
        leakage_status=leakage.status,
        protocol_violations=protocol_violations,
        target_expert_excluded=target_expert_excluded,
    )
    return root


def _evaluate_runtime(
    cfg: SamplingConfig,
    frozen: Mapping[tuple[object, ...], FrozenReference],
    runtime: VariantRuntime,
    *,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_error = "" if set(int(v) for v in runtime.source_train_labels) == {0, 1} else "mono_class_source_train"
    error = eval_error or source_error
    eval_x = None if error else runtime.frame.transform(_to_numpy(eval_raw))
    for seed in cfg.replicate_seeds:
        if error:
            ref = _empty_frozen_reference()
            rows.extend(_ineligible_rows(cfg, runtime, experiment_seed, heldout_center, int(seed), ref, error, len(eval_labels)))
            continue
        ref = _frozen_for_runtime(frozen, runtime, experiment_seed, heldout_center, int(seed))
        sample = _sample_source_positions(runtime, cfg.synthetic_per_class_total, int(seed))
        if ref.source_budget_index_hash and sample.source_budget_index_hash != ref.source_budget_index_hash:
            raise ProtocolError(
                f"Frozen repair source_budget_index_hash mismatch for seed={experiment_seed}, "
                f"heldout={heldout_center}, expert={runtime.expert_id}, replicate={seed}."
            )
        real_x = _subset_rows(runtime.source_train_embeddings, sample.positions)
        rows.append(
            _scored_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                ROW_REAL_BUDGET,
                "real_source_budget_matched",
                int(seed),
                int(seed),
                NA,
                sample.source_budget_index_hash,
                "",
                "source_record_matched",
                real_x,
                sample.labels,
                eval_x,
                eval_labels,
                ref,
                posterior_temperature=NA,
                prior_scale=NA,
                selection_source=runtime.variant.selection_source,
            )
        )
        decoded = _decode_mu(runtime, real_x, sample.labels)
        rows.append(
            _scored_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                ROW_DECODE_MU,
                "decode_mu",
                int(seed),
                int(seed),
                NA,
                sample.source_budget_index_hash,
                "",
                "source_record_matched",
                decoded,
                sample.labels,
                eval_x,
                eval_labels,
                ref,
                posterior_temperature=0.0,
                prior_scale=NA,
                selection_source=runtime.variant.selection_source,
            )
        )
        for temp in cfg.posterior_temperatures:
            posterior = _posterior_sample(runtime, real_x, sample.labels, seed=int(seed), temperature=float(temp))
            rows.append(
                _scored_row(
                    cfg,
                    runtime,
                    experiment_seed,
                    heldout_center,
                    ROW_POSTERIOR,
                    "paired_posterior",
                    int(seed),
                    int(seed),
                    int(seed),
                    sample.source_budget_index_hash,
                    "",
                    "source_record_matched",
                    posterior,
                    sample.labels,
                    eval_x,
                    eval_labels,
                    ref,
                    posterior_temperature=float(temp),
                    prior_scale=NA,
                    selection_source=_sampling_selection(runtime.variant.selection_source, primary=temp in cfg.posterior_temperatures_primary),
                )
            )
        empirical_mu_x, empirical_mu_labels, empirical_mu_hash = _empirical_mu_sample(
            runtime,
            cfg.synthetic_per_class_total,
            seed=_stable_seed(experiment_seed, heldout_center, runtime.expert_id, seed, "empirical_mu"),
        )
        rows.append(
            _scored_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                ROW_EMPIRICAL_MU,
                "empirical_mu",
                int(seed),
                NA,
                NA,
                NA,
                empirical_mu_hash,
                "empirical_latent_matched",
                empirical_mu_x,
                empirical_mu_labels,
                eval_x,
                eval_labels,
                ref,
                posterior_temperature=NA,
                prior_scale=NA,
                selection_source=DIAGNOSTIC_SELECTION,
            )
        )
        empirical_post_x, empirical_post_labels, empirical_post_hash = _empirical_posterior_sample(
            runtime,
            cfg.synthetic_per_class_total,
            seed=_stable_seed(experiment_seed, heldout_center, runtime.expert_id, seed, "empirical_posterior"),
            temperature=cfg.empirical_posterior_temperature,
        )
        rows.append(
            _scored_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                ROW_EMPIRICAL_POSTERIOR,
                "empirical_posterior",
                int(seed),
                NA,
                int(seed),
                NA,
                empirical_post_hash,
                "empirical_latent_matched",
                empirical_post_x,
                empirical_post_labels,
                eval_x,
                eval_labels,
                ref,
                posterior_temperature=cfg.empirical_posterior_temperature,
                prior_scale=NA,
                selection_source=DIAGNOSTIC_SELECTION,
            )
        )
        for scale in cfg.prior_scales:
            prior_x, prior_labels = _prior_sample(runtime, cfg.synthetic_per_class_total, seed=int(seed), prior_scale=float(scale))
            rows.append(
                _scored_row(
                    cfg,
                    runtime,
                    experiment_seed,
                    heldout_center,
                    ROW_PRIOR,
                    "standard_prior",
                    int(seed),
                    NA,
                    int(seed),
                    NA,
                    "",
                    "class_count_matched",
                    prior_x,
                    prior_labels,
                    eval_x,
                    eval_labels,
                    ref,
                    posterior_temperature=NA,
                    prior_scale=float(scale),
                    selection_source=_sampling_selection(runtime.variant.selection_source, primary=scale in cfg.prior_scales_primary),
                )
            )
    return rows


def _scored_row(
    cfg: SamplingConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    row_role: str,
    sampling_family: str,
    replicate_seed: int,
    reference_sample_seed: int | str,
    latent_sample_seed: int | str,
    source_budget_index_hash: str,
    empirical_latent_index_hash: str,
    budget_match_type: str,
    train_x: object,
    train_labels: Sequence[int],
    eval_x: object,
    eval_labels: Sequence[int],
    ref: FrozenReference,
    *,
    posterior_temperature: float | str,
    prior_scale: float | str,
    selection_source: str,
) -> dict[str, object]:
    bundle = fit_locked_logistic_classifier(
        train_x,
        train_labels,
        eval_x,
        classifier_seed=cfg.classifier_seed,
        expert_id=runtime.expert_id,
        class_weight=cfg.classifier_class_weight,
    )
    result = evaluate_probability_predictions(row_role, bundle.probabilities, eval_labels)
    return _sampling_row(
        cfg,
        runtime,
        experiment_seed,
        heldout_center,
        row_role,
        sampling_family,
        replicate_seed,
        reference_sample_seed,
        latent_sample_seed,
        source_budget_index_hash,
        empirical_latent_index_hash,
        budget_match_type,
        ref,
        bacc=result.bacc,
        macro_f1=result.macro_f1,
        generated_features_hash=_hash_array(train_x),
        posterior_temperature=posterior_temperature,
        prior_scale=prior_scale,
        selection_source=selection_source,
        status="ok",
        error_message="",
    )


def _sampling_row(
    cfg: SamplingConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    row_role: str,
    sampling_family: str,
    replicate_seed: int | str,
    reference_sample_seed: int | str,
    latent_sample_seed: int | str,
    source_budget_index_hash: str,
    empirical_latent_index_hash: str,
    budget_match_type: str,
    ref: FrozenReference,
    *,
    bacc: float | str,
    macro_f1: float | str,
    generated_features_hash: str,
    posterior_temperature: float | str,
    prior_scale: float | str,
    selection_source: str,
    status: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "row_role": row_role,
        "sampling_family": sampling_family,
        "posterior_temperature": posterior_temperature,
        "prior_scale": prior_scale,
        "replicate_seed": replicate_seed,
        "reference_sample_seed": reference_sample_seed,
        "latent_sample_seed": latent_sample_seed,
        "source_budget_index_hash": source_budget_index_hash,
        "empirical_latent_index_hash": empirical_latent_index_hash,
        "budget_match_type": budget_match_type,
        "generated_features_hash": generated_features_hash,
        "reference_real_budget_bacc": ref.reference_real_budget_bacc,
        "variant_real_budget_bacc": ref.variant_real_budget_bacc,
        "source_utility_stratum_reference": ref.source_utility_stratum_reference,
        "decoder_gap_vs_real_budget": NA,
        "posterior_gap": NA,
        "empirical_mu_gap": NA,
        "empirical_posterior_gap": NA,
        "prior_gap": NA,
        "decode_to_prior_gap": NA,
        "total_prior_cvae_gap": NA,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "selection_source": selection_source,
        "status": status,
        "error_message": error_message,
        "classifier_type": cfg.classifier_type,
        "classifier_class_weight": cfg.classifier_class_weight,
    }


def _ineligible_rows(
    cfg: SamplingConfig,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
    ref: FrozenReference,
    error_message: str,
    _n_target_eval: int,
) -> list[dict[str, object]]:
    rows = []
    rows.append(
        _sampling_row(
            cfg,
            runtime,
            experiment_seed,
            heldout_center,
            ROW_REAL_BUDGET,
            "real_source_budget_matched",
            replicate_seed,
            replicate_seed,
            NA,
            ref.source_budget_index_hash,
            "",
            "source_record_matched",
            ref,
            bacc="",
            macro_f1="",
            generated_features_hash="",
            posterior_temperature=NA,
            prior_scale=NA,
            selection_source=runtime.variant.selection_source,
            status="ineligible",
            error_message=error_message,
        )
    )
    for row_role, sampling_family in (
        (ROW_DECODE_MU, "decode_mu"),
        (ROW_EMPIRICAL_MU, "empirical_mu"),
        (ROW_EMPIRICAL_POSTERIOR, "empirical_posterior"),
    ):
        rows.append(
            _sampling_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                row_role,
                sampling_family,
                replicate_seed,
                replicate_seed if row_role == ROW_DECODE_MU else NA,
                replicate_seed if row_role == ROW_EMPIRICAL_POSTERIOR else NA,
                ref.source_budget_index_hash if row_role == ROW_DECODE_MU else NA,
                "" if row_role == ROW_DECODE_MU else "ineligible",
                "source_record_matched" if row_role == ROW_DECODE_MU else "empirical_latent_matched",
                ref,
                bacc="",
                macro_f1="",
                generated_features_hash="",
                posterior_temperature=0.0 if row_role == ROW_DECODE_MU else NA,
                prior_scale=NA,
                selection_source=runtime.variant.selection_source if row_role == ROW_DECODE_MU else DIAGNOSTIC_SELECTION,
                status="ineligible",
                error_message=error_message,
            )
        )
    for temp in cfg.posterior_temperatures:
        rows.append(
            _sampling_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                ROW_POSTERIOR,
                "paired_posterior",
                replicate_seed,
                replicate_seed,
                replicate_seed,
                ref.source_budget_index_hash,
                "",
                "source_record_matched",
                ref,
                bacc="",
                macro_f1="",
                generated_features_hash="",
                posterior_temperature=temp,
                prior_scale=NA,
                selection_source=_sampling_selection(runtime.variant.selection_source, primary=temp in cfg.posterior_temperatures_primary),
                status="ineligible",
                error_message=error_message,
            )
        )
    for scale in cfg.prior_scales:
        rows.append(
            _sampling_row(
                cfg,
                runtime,
                experiment_seed,
                heldout_center,
                ROW_PRIOR,
                "standard_prior",
                replicate_seed,
                NA,
                replicate_seed,
                NA,
                "",
                "class_count_matched",
                ref,
                bacc="",
                macro_f1="",
                generated_features_hash="",
                posterior_temperature=NA,
                prior_scale=scale,
                selection_source=_sampling_selection(runtime.variant.selection_source, primary=scale in cfg.prior_scales_primary),
                status="ineligible",
                error_message=error_message,
            )
        )
    return rows


def _decode_mu(runtime: VariantRuntime, x: object, labels: Sequence[int]) -> object:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
        mu, _logvar = runtime.model.encode(xt, yt)
        return runtime.model.decode(mu, yt).detach().cpu().numpy()


def _posterior_sample(runtime: VariantRuntime, x: object, labels: Sequence[int], *, seed: int, temperature: float) -> object:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
        mu, logvar = runtime.model.encode(xt, yt)
        noise = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=mu.device)
        z = mu + (float(temperature) * noise * torch.exp(0.5 * logvar))
        return runtime.model.decode(z, yt).detach().cpu().numpy()


def _empirical_mu_sample(runtime: VariantRuntime, budget_per_class: int, *, seed: int) -> tuple[object, tuple[int, ...], str]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    sample = _sample_source_positions(runtime, budget_per_class, int(seed))
    x = _subset_rows(runtime.source_train_embeddings, sample.positions)
    labels = sample.labels
    with torch.no_grad():
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long)
        mu, _logvar = runtime.model.encode(xt, yt)
        decoded = runtime.model.decode(mu, yt).detach().cpu().numpy()
    return decoded, labels, sample.source_budget_index_hash


def _empirical_posterior_sample(
    runtime: VariantRuntime,
    budget_per_class: int,
    *,
    seed: int,
    temperature: float,
) -> tuple[object, tuple[int, ...], str]:
    sample = _sample_source_positions(runtime, budget_per_class, int(seed))
    x = _subset_rows(runtime.source_train_embeddings, sample.positions)
    return _posterior_sample(runtime, x, sample.labels, seed=int(seed), temperature=float(temperature)), sample.labels, sample.source_budget_index_hash


def _prior_sample(runtime: VariantRuntime, budget_per_class: int, *, seed: int, prior_scale: float) -> tuple[object, tuple[int, ...]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chunks = []
    labels: list[int] = []
    with torch.no_grad():
        for cls in (0, 1):
            y = torch.full((int(budget_per_class),), int(cls), dtype=torch.long)
            z = float(prior_scale) * torch.randn(
                (int(budget_per_class), int(runtime.model.latent_dim)),
                generator=generator,
                dtype=torch.float32,
            )
            chunks.append(runtime.model.decode(z, y).detach().cpu().numpy())
            labels.extend([int(cls)] * int(budget_per_class))
    return np.vstack(chunks), tuple(labels)


def _latent_rows(runtime: VariantRuntime, experiment_seed: int, heldout_center: str) -> list[dict[str, object]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(runtime.source_train_embeddings, dtype=np.float32)
    y_np = np.asarray(runtime.source_train_labels, dtype=int)
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(x, y)
    mu_np = mu.detach().cpu().numpy()
    logvar_np = logvar.detach().cpu().numpy()
    sigma = np.exp(0.5 * logvar_np)
    kl = -0.5 * np.sum(1.0 + logvar_np - (mu_np ** 2) - np.exp(logvar_np), axis=1)
    rows = []
    class_centroids = {
        cls: mu_np[y_np == cls].mean(axis=0)
        for cls in (0, 1)
        if np.any(y_np == cls)
    }
    for cls in (0, 1):
        mask = y_np == cls
        if not np.any(mask):
            continue
        cls_mu = mu_np[mask]
        other = class_centroids.get(1 - cls)
        distances = np.linalg.norm(cls_mu - other, axis=1) if other is not None else np.asarray([math.nan])
        cov = np.cov(cls_mu, rowvar=False) if cls_mu.shape[0] > 1 else np.zeros((cls_mu.shape[1], cls_mu.shape[1]))
        eigvals = np.linalg.eigvalsh(np.atleast_2d(cov))
        eigvals = np.clip(eigvals, 0.0, None)
        total = float(eigvals.sum())
        effective_rank = 0.0 if total <= 0 else float((total ** 2) / max(float(np.sum(eigvals ** 2)), 1.0e-12))
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "class_label": int(cls),
                "n_source_train_class": int(mask.sum()),
                "n_encoded_records": int(mask.sum()),
                "mu_norm_mean": float(np.linalg.norm(cls_mu, axis=1).mean()),
                "mu_norm_std": float(np.linalg.norm(cls_mu, axis=1).std()),
                "sigma_mean": float(sigma[mask].mean()),
                "sigma_std": float(sigma[mask].std()),
                "kl_to_standard_normal_mean": float(kl[mask].mean()),
                "kl_to_standard_normal_std": float(kl[mask].std()),
                "aggregated_mu_mean_norm": float(np.linalg.norm(cls_mu.mean(axis=0))),
                "aggregated_mu_cov_trace": total,
                "aggregated_mu_effective_rank": effective_rank,
                "per_class_mu_distance_mean": float(np.nanmean(distances)),
                "per_class_mu_distance_min": float(np.nanmin(distances)),
            }
        )
    return rows


def _load_frozen_repair_reference(cfg: SamplingConfig) -> dict[tuple[object, ...], FrozenReference]:
    path = cfg.repair_artifact_root / "tables" / "repair_gap_summary.csv"
    if not path.exists():
        raise ProtocolError(f"Missing frozen repair gap summary: {path}")
    required = {
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "replicate_seed",
        "reference_real_budget_bacc",
        "variant_real_budget_bacc",
        "source_utility_stratum_reference",
        "source_budget_index_hash",
    }
    out: dict[tuple[object, ...], FrozenReference] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"Frozen repair summary is missing fields: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "ok":
                continue
            if row["variant_id"] not in {PRIMARY_VARIANT, UNION_VARIANT}:
                continue
            key = _frozen_key(
                row["experiment_seed"],
                row["heldout_center"],
                row["expert_id"],
                row["expert_pool_type"],
                row["variant_id"],
                row["replicate_seed"],
            )
            out[key] = FrozenReference(
                reference_real_budget_bacc=float(row["reference_real_budget_bacc"]),
                variant_real_budget_bacc=float(row["variant_real_budget_bacc"]),
                source_utility_stratum_reference=str(row["source_utility_stratum_reference"]),
                source_budget_index_hash=str(row["source_budget_index_hash"]),
            )
    if not out:
        raise ProtocolError("Frozen repair summary did not contain pca64 sampling reference rows.")
    return out


def _frozen_for_runtime(
    frozen: Mapping[tuple[object, ...], FrozenReference],
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    replicate_seed: int,
) -> FrozenReference:
    key = _frozen_key(
        experiment_seed,
        heldout_center,
        runtime.expert_id,
        runtime.variant.expert_pool_type,
        runtime.variant.variant_id,
        replicate_seed,
    )
    ref = frozen.get(key)
    if ref is None:
        raise ProtocolError(f"Missing frozen repair reference for {key}.")
    return ref


def _empty_frozen_reference() -> FrozenReference:
    return FrozenReference(
        reference_real_budget_bacc=math.nan,
        variant_real_budget_bacc=math.nan,
        source_utility_stratum_reference="",
        source_budget_index_hash=NA,
    )


def _frozen_key(
    experiment_seed: object,
    heldout_center: object,
    expert_id: object,
    expert_pool_type: object,
    variant_id: object,
    replicate_seed: object,
) -> tuple[object, ...]:
    return (
        str(experiment_seed),
        str(heldout_center),
        str(expert_id),
        str(expert_pool_type),
        str(variant_id),
        str(replicate_seed),
    )


def _runtime_source(
    cfg: SamplingConfig,
    repair_cfg: RepairConfig,
    *,
    root: Path,
    experiment_seed: int,
    heldout_center: str,
    expert_id: str,
    source_data: SourceData,
    variant: RepairVariant,
) -> RuntimeSource:
    checkpoint_name = f"seed{experiment_seed}_heldout{heldout_center}_expert{expert_id}_{variant.variant_id}.pkl"
    local_path = root / "checkpoints" / checkpoint_name
    repair_path = cfg.repair_artifact_root / "checkpoints" / checkpoint_name
    reused = False
    if local_path.exists():
        path = local_path
    elif repair_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repair_path, local_path)
        path = local_path
        reused = True
    else:
        runtime = _runtime_for_variant(
            repair_cfg,
            root=root,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            expert_id=expert_id,
            source_data=source_data,
            variant=variant,
        )
        path = local_path
        return RuntimeSource(runtime=runtime, checkpoint_path=path, checkpoint_sha256=_file_sha256(path), checkpoint_reused_from_repair=False)
    with path.open("rb") as f:
        runtime = pickle.load(f)
    _validate_runtime(runtime, variant)
    return RuntimeSource(runtime=runtime, checkpoint_path=path, checkpoint_sha256=_file_sha256(path), checkpoint_reused_from_repair=reused)


def _validate_runtime(runtime: VariantRuntime, variant: RepairVariant) -> None:
    if runtime.variant.variant_id != variant.variant_id:
        raise ProtocolError("Loaded repair checkpoint has wrong variant_id.")
    if runtime.variant.expert_pool_type != variant.expert_pool_type:
        raise ProtocolError("Loaded repair checkpoint has wrong expert_pool_type.")
    if runtime.variant.requested_pca_dim != variant.requested_pca_dim or runtime.variant.latent_dim != variant.latent_dim:
        raise ProtocolError("Loaded repair checkpoint has wrong PCA or latent dimension.")


def _repair_runtime_config(cfg: SamplingConfig, root: Path) -> RepairConfig:
    return RepairConfig(
        name="virchow2_cvae_preservation_repair_v1",
        artifact_root=root,
        feature_cache_root=cfg.feature_cache_root,
        experiment_seeds=cfg.experiment_seeds,
        heldout_centers=cfg.heldout_centers,
        replicate_seeds=cfg.replicate_seeds,
        synthetic_per_class_total=cfg.synthetic_per_class_total,
        primary_variant=cfg.primary_variant,
        min_decision_rows=cfg.min_decision_cells,
        variants=(_per_source_variant(), _union_variant()),
        source_probe=SourceProbeConfig(
            type="torch_linear_classifier",
            optimizer="adamw",
            learning_rate=0.001,
            weight_decay=0.0001,
            epochs=1,
            batch_size=128,
            class_weight="balanced",
            early_stopping=False,
        ),
        classifier_type=cfg.classifier_type,
        classifier_solver=cfg.classifier_solver,
        classifier_c=cfg.classifier_c,
        classifier_max_iter=cfg.classifier_max_iter,
        classifier_class_weight=cfg.classifier_class_weight,
        classifier_seed=cfg.classifier_seed,
    )


def _per_source_variant() -> RepairVariant:
    return RepairVariant(
        variant_id=PRIMARY_VARIANT,
        expert_pool_type=POOL_PER_SOURCE,
        requested_pca_dim=64,
        latent_dim=16,
        train_epochs=100,
        beta_final=0.001,
        kl_warmup_epochs=25,
        probe_ce_weight=0.0,
        loss_style="normalized_repair",
        selection_source=PRIMARY_SELECTION,
        optimizer="adamw",
        weight_decay=0.0001,
    )


def _union_variant() -> RepairVariant:
    return RepairVariant(
        variant_id=UNION_VARIANT,
        expert_pool_type=POOL_SOURCE_UNION,
        requested_pca_dim=64,
        latent_dim=16,
        train_epochs=100,
        beta_final=0.001,
        kl_warmup_epochs=25,
        probe_ce_weight=0.0,
        loss_style="normalized_repair",
        selection_source=DIAGNOSTIC_SELECTION,
        optimizer="adamw",
        weight_decay=0.0001,
    )


def _sampling_selection(base_selection: str, *, primary: bool) -> str:
    return base_selection if str(base_selection) == PRIMARY_SELECTION and primary else DIAGNOSTIC_SELECTION


def _augment_gap_fields(rows: list[dict[str, object]]) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in ok:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["expert_pool_type"], row["replicate_seed"])
        grouped.setdefault(key, []).append(row)
    for subset in grouped.values():
        by_role = {}
        posterior_by_temp = {}
        prior_by_scale = {}
        for row in subset:
            if row["row_role"] in {ROW_REAL_BUDGET, ROW_DECODE_MU, ROW_EMPIRICAL_MU, ROW_EMPIRICAL_POSTERIOR}:
                by_role[row["row_role"]] = row
            if row["row_role"] == ROW_POSTERIOR:
                posterior_by_temp[str(row["posterior_temperature"])] = row
            if row["row_role"] == ROW_PRIOR:
                prior_by_scale[str(row["prior_scale"])] = row
        real = by_role.get(ROW_REAL_BUDGET)
        decode = by_role.get(ROW_DECODE_MU)
        empirical_mu = by_role.get(ROW_EMPIRICAL_MU)
        empirical_post = by_role.get(ROW_EMPIRICAL_POSTERIOR)
        if not (real and decode and empirical_mu and empirical_post):
            continue
        decoder_gap = _float(real["bacc"]) - _float(decode["bacc"])
        empirical_mu_gap = _float(decode["bacc"]) - _float(empirical_mu["bacc"])
        empirical_post_gap = _float(decode["bacc"]) - _float(empirical_post["bacc"])
        for row in subset:
            row["decoder_gap_vs_real_budget"] = decoder_gap
            row["empirical_mu_gap"] = empirical_mu_gap
            row["empirical_posterior_gap"] = empirical_post_gap
        for posterior in posterior_by_temp.values():
            posterior_gap = _float(decode["bacc"]) - _float(posterior["bacc"])
            posterior["posterior_gap"] = posterior_gap
        for prior in prior_by_scale.values():
            prior_gap = _float(empirical_mu["bacc"]) - _float(prior["bacc"])
            decode_to_prior = _float(decode["bacc"]) - _float(prior["bacc"])
            total_prior = _float(real["bacc"]) - _float(prior["bacc"])
            prior["prior_gap"] = prior_gap
            prior["decode_to_prior_gap"] = decode_to_prior
            prior["total_prior_cvae_gap"] = total_prior


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    ok = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in ok:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["expert_pool_type"], row["replicate_seed"])
        grouped.setdefault(key, []).append(row)
    out = []
    for key, subset in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        by_role = {row["row_role"]: row for row in subset if row["row_role"] in {ROW_REAL_BUDGET, ROW_DECODE_MU, ROW_EMPIRICAL_MU, ROW_EMPIRICAL_POSTERIOR}}
        posterior_primary = _find_row(subset, ROW_POSTERIOR, "posterior_temperature", "1.0")
        prior_primary = _find_row(subset, ROW_PRIOR, "prior_scale", "1.0")
        if not ({ROW_REAL_BUDGET, ROW_DECODE_MU, ROW_EMPIRICAL_MU, ROW_EMPIRICAL_POSTERIOR}.issubset(by_role) and posterior_primary and prior_primary):
            continue
        real = by_role[ROW_REAL_BUDGET]
        decode = by_role[ROW_DECODE_MU]
        empirical_mu = by_role[ROW_EMPIRICAL_MU]
        empirical_post = by_role[ROW_EMPIRICAL_POSTERIOR]
        base_row = {
                "experiment_seed": key[0],
                "heldout_center": key[1],
                "expert_id": key[2],
                "expert_pool_type": key[3],
                "variant_id": real["variant_id"],
                "replicate_seed": key[4],
                "posterior_temperature": 1.0,
                "prior_scale": 1.0,
                "source_utility_stratum_reference": real["source_utility_stratum_reference"],
                "selection_source": real["selection_source"],
                "reference_real_budget_bacc": real["reference_real_budget_bacc"],
                "variant_real_budget_bacc": real["bacc"],
                "decode_mu_bacc": decode["bacc"],
                "posterior_bacc": posterior_primary["bacc"],
                "empirical_mu_bacc": empirical_mu["bacc"],
                "empirical_posterior_bacc": empirical_post["bacc"],
                "prior_bacc": prior_primary["bacc"],
                "decode_mu_macro_f1": decode["macro_f1"],
                "posterior_macro_f1": posterior_primary["macro_f1"],
                "prior_macro_f1": prior_primary["macro_f1"],
                "decoder_gap_vs_real_budget": _float(real["bacc"]) - _float(decode["bacc"]),
                "posterior_gap": _float(decode["bacc"]) - _float(posterior_primary["bacc"]),
                "empirical_mu_gap": _float(decode["bacc"]) - _float(empirical_mu["bacc"]),
                "empirical_posterior_gap": _float(decode["bacc"]) - _float(empirical_post["bacc"]),
                "prior_gap": _float(empirical_mu["bacc"]) - _float(prior_primary["bacc"]),
                "decode_to_prior_gap": _float(decode["bacc"]) - _float(prior_primary["bacc"]),
                "total_prior_cvae_gap": _float(real["bacc"]) - _float(prior_primary["bacc"]),
                "source_budget_index_hash": real["source_budget_index_hash"],
                "status": "ok",
            }
        out.append(base_row)
        for row in subset:
            if row.get("row_role") == ROW_POSTERIOR and str(row.get("posterior_temperature")) != "1.0":
                diagnostic = dict(base_row)
                diagnostic["posterior_temperature"] = row["posterior_temperature"]
                diagnostic["prior_scale"] = NA
                diagnostic["posterior_bacc"] = row["bacc"]
                diagnostic["posterior_macro_f1"] = row["macro_f1"]
                diagnostic["posterior_gap"] = _float(decode["bacc"]) - _float(row["bacc"])
                diagnostic["selection_source"] = DIAGNOSTIC_SELECTION
                out.append(diagnostic)
            if row.get("row_role") == ROW_PRIOR and str(row.get("prior_scale")) != "1.0":
                diagnostic = dict(base_row)
                diagnostic["posterior_temperature"] = NA
                diagnostic["prior_scale"] = row["prior_scale"]
                diagnostic["prior_bacc"] = row["bacc"]
                diagnostic["prior_macro_f1"] = row["macro_f1"]
                diagnostic["prior_gap"] = _float(empirical_mu["bacc"]) - _float(row["bacc"])
                diagnostic["decode_to_prior_gap"] = _float(decode["bacc"]) - _float(row["bacc"])
                diagnostic["total_prior_cvae_gap"] = _float(real["bacc"]) - _float(row["bacc"])
                diagnostic["selection_source"] = DIAGNOSTIC_SELECTION
                out.append(diagnostic)
    return out


def _find_row(rows: Sequence[Mapping[str, object]], row_role: str, field: str, value: str) -> Mapping[str, object] | None:
    for row in rows:
        if row.get("row_role") == row_role and str(row.get(field)) == str(value):
            return row
    return None


def _decision(rows: Sequence[Mapping[str, object]], cfg: SamplingConfig, *, leakage_status: str) -> dict[str, object]:
    primary = _decision_rows(rows, pool_type=POOL_PER_SOURCE)
    stats = _sampling_stats(primary)
    posterior_pass = _posterior_pass(stats)
    prior_pass = _prior_pass(stats)
    diagnostic_posterior = _diagnostic_posterior_works(rows, cfg)
    diagnostic_prior = _diagnostic_prior_works(rows, cfg)
    verdict = "SAMPLING_FAIL"
    if leakage_status != "PASS":
        verdict = "PROTOCOL_FAIL"
    elif int(stats["n_decision_cells"]) < int(cfg.min_decision_cells):
        verdict = "INSUFFICIENT_DECISION_ROWS"
    elif stats["mean_decode_mu_bacc"] < 0.80 or stats["mean_decoder_gap"] > 0.05:
        verdict = "DECODE_REPAIR_NOT_REPRODUCED"
    elif posterior_pass and prior_pass:
        verdict = "SAMPLING_PASS"
    elif posterior_pass and not prior_pass:
        verdict = "LATENT_PRIOR_MISMATCH"
    elif diagnostic_posterior or diagnostic_prior:
        verdict = "SAMPLING_PARTIAL"
    elif not posterior_pass and not diagnostic_posterior:
        verdict = "POSTERIOR_SAMPLING_FAIL"

    flags = []
    if stats["decode_mu_seed_std"] > 0.05:
        flags.append("DECODE_UNSTABLE")
    if _source_pool_passes(rows):
        flags.append("SOURCE_POOL_SAMPLING_STRONG")
    if _source_pool_posterior_passes(rows):
        flags.append("SOURCE_POOL_POSTERIOR_STRONG")
    if _source_pool_prior_passes(rows):
        flags.append("SOURCE_POOL_PRIOR_STRONG")
    if diagnostic_prior:
        flags.append("SCALED_PRIOR_RESCUE")
    if diagnostic_posterior:
        flags.append("LOW_TEMP_POSTERIOR_RESCUE")
    if _center_collapse(primary, "posterior_bacc"):
        flags.append("CENTER_COLLAPSE_POSTERIOR")
    if _center_collapse(primary, "prior_bacc"):
        flags.append("CENTER_COLLAPSE_PRIOR")
    return {"primary_verdict": verdict, "diagnostic_flags": "|".join(flags), **stats}


def _decision_rows(rows: Sequence[Mapping[str, object]], *, pool_type: str) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("variant_id") == (PRIMARY_VARIANT if pool_type == POOL_PER_SOURCE else UNION_VARIANT)
        and row.get("expert_pool_type") == pool_type
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and str(row.get("posterior_temperature")) == "1.0"
        and str(row.get("prior_scale")) == "1.0"
        and (pool_type != POOL_PER_SOURCE or row.get("selection_source") == PRIMARY_SELECTION)
    ]


def _sampling_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged(rows)
    by_seed: dict[str, list[Mapping[str, float | str]]] = {}
    centers = set()
    experts = set()
    for row in grouped:
        by_seed.setdefault(str(row["experiment_seed"]), []).append(row)
        centers.add(str(row["heldout_center"]))
        experts.add(str(row["expert_id"]))
    seed_decode = [_mean_field(values, "decode_mu_bacc") for values in by_seed.values()]
    seed_post = [_mean_field(values, "posterior_bacc") for values in by_seed.values()]
    seed_prior = [_mean_field(values, "prior_bacc") for values in by_seed.values()]
    return {
        "n_raw_decision_rows": len(rows),
        "n_decision_cells": len(grouped),
        "n_experiment_seeds": len(by_seed),
        "n_heldout_centers": len(centers),
        "n_experts": len(experts),
        "mean_decode_mu_bacc": _mean_field(grouped, "decode_mu_bacc"),
        "mean_posterior_bacc": _mean_field(grouped, "posterior_bacc"),
        "mean_prior_bacc": _mean_field(grouped, "prior_bacc"),
        "mean_decoder_gap": _mean_field(grouped, "decoder_gap_vs_real_budget"),
        "mean_posterior_gap": _mean_field(grouped, "posterior_gap"),
        "mean_total_prior_cvae_gap": _mean_field(grouped, "total_prior_cvae_gap"),
        "decode_mu_seed_std": _std(seed_decode),
        "posterior_seed_std": _std(seed_post),
        "prior_seed_std": _std(seed_prior),
        "per_center_decode_mu_bacc": json.dumps(_per_center_mean(grouped, "decode_mu_bacc"), sort_keys=True),
        "per_center_posterior_bacc": json.dumps(_per_center_mean(grouped, "posterior_bacc"), sort_keys=True),
        "per_center_prior_bacc": json.dumps(_per_center_mean(grouped, "prior_bacc"), sort_keys=True),
        "per_center_decoder_gap": json.dumps(_per_center_mean(grouped, "decoder_gap_vs_real_budget"), sort_keys=True),
        "per_center_posterior_gap": json.dumps(_per_center_mean(grouped, "posterior_gap"), sort_keys=True),
        "per_center_total_prior_cvae_gap": json.dumps(_per_center_mean(grouped, "total_prior_cvae_gap"), sort_keys=True),
        "per_seed_bacc": json.dumps({seed: nanmean([_float(row["prior_bacc"]) for row in values]) for seed, values in sorted(by_seed.items())}, sort_keys=True),
    }


def _replicate_averaged(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    out = []
    fields = (
        "decode_mu_bacc",
        "posterior_bacc",
        "prior_bacc",
        "decoder_gap_vs_real_budget",
        "posterior_gap",
        "total_prior_cvae_gap",
    )
    for (seed, center, expert), subset in groups.items():
        row = {"experiment_seed": seed, "heldout_center": center, "expert_id": expert}
        row.update({field: _mean_field(subset, field) for field in fields})
        out.append(row)
    return out


def _posterior_pass(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats["mean_posterior_bacc"]) >= 0.75
        and _float(stats["mean_posterior_gap"]) <= 0.05
        and _float(stats["posterior_seed_std"]) <= 0.07
    )


def _prior_pass(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats["mean_prior_bacc"]) >= 0.75
        and _float(stats["mean_total_prior_cvae_gap"]) <= 0.08
        and _float(stats["prior_seed_std"]) <= 0.07
    )


def _diagnostic_posterior_works(rows: Sequence[Mapping[str, object]], cfg: SamplingConfig) -> bool:
    for temp in cfg.posterior_temperatures_diagnostic:
        post_rows = _posterior_rows_for_temp(rows, temp)
        if post_rows and _posterior_pass(_sampling_stats(post_rows)):
            return True
    return False


def _diagnostic_prior_works(rows: Sequence[Mapping[str, object]], cfg: SamplingConfig) -> bool:
    for scale in cfg.prior_scales_diagnostic:
        prior_rows = _prior_rows_for_scale(rows, scale)
        if prior_rows and _prior_pass(_sampling_stats(prior_rows)):
            return True
    return False


def _posterior_rows_for_temp(rows: Sequence[Mapping[str, object]], temp: float) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if str(row.get("posterior_temperature")) == str(float(temp))
        and row.get("selection_source") == DIAGNOSTIC_SELECTION
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and row.get("status") == "ok"
    ]


def _prior_rows_for_scale(rows: Sequence[Mapping[str, object]], scale: float) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if str(row.get("prior_scale")) == str(float(scale))
        and row.get("selection_source") == DIAGNOSTIC_SELECTION
        and row.get("expert_pool_type") == POOL_PER_SOURCE
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and row.get("status") == "ok"
    ]


def _source_pool_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    stats = _sampling_stats(_decision_rows(rows, pool_type=POOL_SOURCE_UNION))
    return (
        stats["mean_decode_mu_bacc"] >= 0.80
        and stats["mean_decoder_gap"] <= 0.05
        and _posterior_pass(stats)
        and _prior_pass(stats)
    )


def _source_pool_posterior_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    return _posterior_pass(_sampling_stats(_decision_rows(rows, pool_type=POOL_SOURCE_UNION)))


def _source_pool_prior_passes(rows: Sequence[Mapping[str, object]]) -> bool:
    return _prior_pass(_sampling_stats(_decision_rows(rows, pool_type=POOL_SOURCE_UNION)))


def _center_collapse(rows: Sequence[Mapping[str, object]], field: str) -> bool:
    grouped = _replicate_averaged(rows)
    return any(value < 0.60 for value in _per_center_mean(grouped, field).values())


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", "NA"}])


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _per_center_mean(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row["heldout_center"]), []).append(_float(row[field]))
    return {center: nanmean(values) for center, values in sorted(groups.items())}


def _manifest_row(experiment_seed: int, heldout_center: str, runtime_source: RuntimeSource) -> dict[str, object]:
    runtime = runtime_source.runtime
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "source_scope": runtime.source_scope,
        "checkpoint_path": str(runtime_source.checkpoint_path),
        "checkpoint_sha256": runtime_source.checkpoint_sha256,
        "checkpoint_reused_from_repair": runtime_source.checkpoint_reused_from_repair,
        "model_config_hash": _hash_strings([json.dumps(runtime.variant.__dict__, sort_keys=True)]),
        "train_split_hash": _hash_strings(runtime.source_train_sample_ids),
        "pca_fit_hash": _hash_strings([str(runtime.frame.effective_dim), f"{runtime.frame.explained_variance_ratio_sum:.12f}"]),
        "scaler_hash": _hash_strings([str(runtime.frame.effective_dim)]),
        "requested_pca_dim": runtime.variant.requested_pca_dim,
        "effective_pca_dim": runtime.frame.effective_dim,
        "latent_dim": runtime.variant.latent_dim,
        "n_train": runtime.n_train,
        "n_val": runtime.n_val,
    }


def _source_pool_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    subset = _decision_rows(rows, pool_type=POOL_SOURCE_UNION)
    stats = _sampling_stats(subset)
    return [{
        "variant_id": UNION_VARIANT,
        "expert_pool_type": POOL_SOURCE_UNION,
        "selection_source": DIAGNOSTIC_SELECTION,
        **stats,
    }]


def _write_sampling_artifacts(
    root: Path,
    cfg: SamplingConfig,
    *,
    downstream_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    latent_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "sampling_downstream_matrix.csv", downstream_rows, columns=_matrix_columns())
    write_csv_rows(root / "tables" / "sampling_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "latent_distribution_diagnostics.csv", latent_rows)
    write_csv_rows(root / "tables" / "source_pool_sampling_summary.csv", _source_pool_summary_rows(gap_rows))
    write_csv_rows(root / "manifests" / "sampling_model_manifest.csv", manifest_rows)
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_preservation_sampling_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "preservation_sampling_continuation",
            "primary_variant": cfg.primary_variant,
            "row_roles": list(ROW_ROLES),
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "source_union_diagnostic_only": True,
            "claim_boundary": "sampling utility preservation only; no routing or formal privacy claim",
        },
    )
    _write_decision_summary(root, decision, leakage_status=leakage.status)
    write_json(root / "run_config_resolved.yaml", _resolved_config(cfg))


def _matrix_columns() -> tuple[str, ...]:
    return (
        "experiment_seed",
        "heldout_center",
        "expert_id",
        "expert_pool_type",
        "variant_id",
        "row_role",
        "sampling_family",
        "posterior_temperature",
        "prior_scale",
        "replicate_seed",
        "reference_sample_seed",
        "latent_sample_seed",
        "source_budget_index_hash",
        "empirical_latent_index_hash",
        "budget_match_type",
        "generated_features_hash",
        "reference_real_budget_bacc",
        "variant_real_budget_bacc",
        "source_utility_stratum_reference",
        "decoder_gap_vs_real_budget",
        "posterior_gap",
        "empirical_mu_gap",
        "empirical_posterior_gap",
        "prior_gap",
        "decode_to_prior_gap",
        "total_prior_cvae_gap",
        "bacc",
        "macro_f1",
        "selection_source",
        "status",
        "error_message",
        "classifier_type",
        "classifier_class_weight",
    )


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE PCA64 Sampling Continuation v1",
            "",
            "## Summary",
            "",
            f"- Primary variant: `{PRIMARY_VARIANT}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'SAMPLING_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Mean decode(mu) BACC: {_format_float(decision.get('mean_decode_mu_bacc'))}",
            f"- Mean posterior BACC: {_format_float(decision.get('mean_posterior_bacc'))}",
            f"- Mean prior BACC: {_format_float(decision.get('mean_prior_bacc'))}",
            f"- Mean posterior gap: {_format_float(decision.get('mean_posterior_gap'))}",
            f"- Mean total prior CVAE gap: {_format_float(decision.get('mean_total_prior_cvae_gap'))}",
            f"- Decision cells: {decision.get('n_decision_cells', 0)}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Per-Center Diagnostics",
            "",
            f"- Decode(mu) BACC: `{decision.get('per_center_decode_mu_bacc', '{}')}`",
            f"- Posterior BACC: `{decision.get('per_center_posterior_bacc', '{}')}`",
            f"- Prior BACC: `{decision.get('per_center_prior_bacc', '{}')}`",
            "",
            "## Claim Boundary",
            "",
            "This slice diagnoses sampled-feature downstream utility preservation.",
            "It does not evaluate routing, support-NELBO selection, metadata selection, top-k composition, or formal privacy.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _resolved_config(cfg: SamplingConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "min_decision_cells": cfg.min_decision_cells,
        "posterior_temperatures_primary": list(cfg.posterior_temperatures_primary),
        "posterior_temperatures_diagnostic": list(cfg.posterior_temperatures_diagnostic),
        "prior_scales_primary": list(cfg.prior_scales_primary),
        "prior_scales_diagnostic": list(cfg.prior_scales_diagnostic),
        "empirical_posterior_temperature": cfg.empirical_posterior_temperature,
        "classifier_type": cfg.classifier_type,
        "classifier_solver": cfg.classifier_solver,
        "classifier_c": cfg.classifier_c,
        "classifier_max_iter": cfg.classifier_max_iter,
        "classifier_class_weight": cfg.classifier_class_weight,
        "classifier_seed": cfg.classifier_seed,
    }


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
