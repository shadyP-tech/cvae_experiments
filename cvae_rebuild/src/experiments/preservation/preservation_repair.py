from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.downstream import evaluate_probability_predictions, fit_locked_logistic_classifier
from data.feature_frame import ExpertFeatureFrame, fit_expert_frame
from data.features import FeatureCache, default_cache_path, load_feature_cache, select_rows
from core.metrics import nanmean
from model.models import ClassConditionedCVAE, loss_for_batch
from core.protocol import ProtocolError, assert_candidate_pool, build_leakage_report
from core.reporting import prepare_artifact_dirs, write_csv_rows, write_json
from data.splits import candidate_experts, stratified_source_train_val_split


REPAIR_NAME = "virchow2_cvae_preservation_repair_v1"
PRIMARY_VARIANT = "pca64_beta001"
ROW_REAL_FULL = "real_source_full_balanced_classifier"
ROW_REAL_BUDGET = "real_source_budget_matched_classifier"
ROW_DECODE_MU = "cvae_decode_mu_budget_matched"
ROW_ROLES = (ROW_REAL_FULL, ROW_REAL_BUDGET, ROW_DECODE_MU)
POOL_PER_SOURCE = "per_source"
POOL_SOURCE_UNION = "source_union_excluding_target"
NA = "NA"


@dataclass(frozen=True)
class SourceProbeConfig:
    type: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    class_weight: str
    early_stopping: bool


@dataclass(frozen=True)
class RepairVariant:
    variant_id: str
    expert_pool_type: str
    requested_pca_dim: int
    latent_dim: int
    train_epochs: int
    beta_final: float
    kl_warmup_epochs: int
    probe_ce_weight: float
    loss_style: str
    selection_source: str
    hidden_dim: int = 512
    num_hidden_layers: int = 2
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    optimizer: str = "adamw"
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    activation: str = "relu"
    dropout: float = 0.0
    device: str = "cpu"


@dataclass(frozen=True)
class RepairConfig:
    name: str
    artifact_root: Path
    feature_cache_root: Path
    experiment_seeds: tuple[int, ...]
    heldout_centers: tuple[str, ...]
    replicate_seeds: tuple[int, ...]
    synthetic_per_class_total: int
    primary_variant: str
    min_decision_rows: int
    variants: tuple[RepairVariant, ...]
    source_probe: SourceProbeConfig
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None


@dataclass
class SourceData:
    raw_train: object
    raw_val: object
    train_labels: tuple[int, ...]
    val_labels: tuple[int, ...]
    train_sample_ids: tuple[str, ...]
    val_sample_ids: tuple[str, ...]
    train_centers: tuple[str, ...]
    val_centers: tuple[str, ...]
    source_scope: str


@dataclass
class SourceProbeRuntime:
    model: object | None
    train_acc: float | str
    val_acc: float | str
    best_val_acc: float | str
    train_loss: float | str
    val_loss: float | str
    epochs_trained: int | str


@dataclass
class VariantRuntime:
    variant: RepairVariant
    expert_id: str
    frame: ExpertFeatureFrame
    model: ClassConditionedCVAE
    probe: SourceProbeRuntime
    source_train_embeddings: object
    source_val_embeddings: object
    source_train_labels: tuple[int, ...]
    source_val_labels: tuple[int, ...]
    source_train_sample_ids: tuple[str, ...]
    source_train_centers: tuple[str, ...]
    source_scope: str
    n_train: int
    n_val: int
    training_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class SourceSample:
    positions: tuple[int, ...]
    labels: tuple[int, ...]
    source_budget_index_hash: str


def load_repair_config(path: str | Path) -> RepairConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = source.parents[2] if len(source.parents) >= 3 else source.parent
    return parse_repair_config(data, base_dir=base_dir)


def parse_repair_config(data: Mapping[str, Any], *, base_dir: str | Path = ".") -> RepairConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    run = _mapping(data, "run_matrix")
    generation = _mapping(data, "generation")
    classifier = _mapping(data, "classifier")
    probe = _mapping(data, "source_probe")
    variants_payload = data.get("variants")
    if not isinstance(variants_payload, Sequence) or isinstance(variants_payload, (str, bytes)):
        raise ProtocolError("variants must be a list of locked variant mappings.")
    cfg = RepairConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        feature_cache_root=_path(base, str(inputs["feature_cache_root"])),
        experiment_seeds=tuple(int(v) for v in run["experiment_seeds"]),
        heldout_centers=tuple(str(v) for v in run["heldout_centers"]),
        replicate_seeds=tuple(int(v) for v in run["replicate_seeds"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        primary_variant=str(experiment["primary_variant"]),
        min_decision_rows=int(experiment.get("min_decision_rows", 10)),
        variants=tuple(_parse_variant(v) for v in variants_payload),
        source_probe=SourceProbeConfig(
            type=str(probe["type"]),
            optimizer=str(probe["optimizer"]),
            learning_rate=float(probe["learning_rate"]),
            weight_decay=float(probe["weight_decay"]),
            epochs=int(probe["epochs"]),
            batch_size=int(probe["batch_size"]),
            class_weight=str(probe["class_weight"]),
            early_stopping=bool(probe["early_stopping"]),
        ),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    )
    validate_repair_config(cfg)
    return cfg


def validate_repair_config(cfg: RepairConfig) -> None:
    expected = {
        "current_pca200_beta1_reference",
        "pca64_beta001",
        "pca128_beta001",
        "pca64_beta001_probe025",
        "pca128_beta001_probe025",
        "source_union_pca64_beta001_diagnostic",
        "source_union_pca64_beta001_probe025_diagnostic",
    }
    ids = {variant.variant_id for variant in cfg.variants}
    if cfg.name != REPAIR_NAME:
        raise ProtocolError(f"Repair experiment name must be {REPAIR_NAME!r}.")
    if cfg.primary_variant != PRIMARY_VARIANT:
        raise ProtocolError(f"primary_variant must be {PRIMARY_VARIANT!r}.")
    if ids != expected:
        raise ProtocolError(f"Repair variants must be exactly {sorted(expected)!r}.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.min_decision_rows < 1:
        raise ProtocolError("min_decision_rows must be positive.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")
    if cfg.source_probe.type != "torch_linear_classifier" or cfg.source_probe.optimizer != "adamw":
        raise ProtocolError("source_probe must be a torch_linear_classifier trained with adamw.")
    if cfg.source_probe.class_weight != "balanced":
        raise ProtocolError("source_probe.class_weight must be balanced.")
    for variant in cfg.variants:
        _validate_variant(variant)


def run_preservation_repair(cfg: RepairConfig, *, artifact_root: str | Path | None = None) -> Path:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("Preservation repair requires numpy.") from exc

    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    downstream_rows: list[dict[str, object]] = []
    repair_gap_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    probe_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    target_expert_excluded = True

    per_source_variants = tuple(v for v in cfg.variants if v.expert_pool_type == POOL_PER_SOURCE)
    union_variants = tuple(v for v in cfg.variants if v.expert_pool_type == POOL_SOURCE_UNION)

    for experiment_seed in cfg.experiment_seeds:
        train_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="train"))
        test_cache = load_feature_cache(_existing_cache_path(cfg.feature_cache_root, seed=experiment_seed, split="test"))
        per_source_data = {
            center: _source_data_for_centers(train_cache, centers=(center,), experiment_seed=int(experiment_seed))
            for center in cfg.heldout_centers
        }
        per_source_runtime: dict[tuple[str, str], VariantRuntime] = {}
        for expert_id, source_data in per_source_data.items():
            for variant in per_source_variants:
                runtime = _runtime_for_variant(
                    cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=NA,
                    expert_id=str(expert_id),
                    source_data=source_data,
                    variant=variant,
                )
                per_source_runtime[(str(expert_id), variant.variant_id)] = runtime
                manifest_rows.append(_manifest_row(experiment_seed, NA, runtime))
                probe_rows.append(_probe_row(experiment_seed, NA, runtime))
                training_rows.extend(runtime.training_rows)

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

            union_runtime: dict[str, VariantRuntime] = {}
            union_data = _source_data_for_centers(train_cache, centers=candidates, experiment_seed=int(experiment_seed))
            for variant in union_variants:
                runtime = _runtime_for_variant(
                    cfg,
                    root=root,
                    experiment_seed=int(experiment_seed),
                    heldout_center=str(heldout_center),
                    expert_id=POOL_SOURCE_UNION,
                    source_data=union_data,
                    variant=variant,
                )
                union_runtime[variant.variant_id] = runtime
                manifest_rows.append(_manifest_row(experiment_seed, str(heldout_center), runtime))
                probe_rows.append(_probe_row(experiment_seed, str(heldout_center), runtime))
                training_rows.extend(runtime.training_rows)

            target_indices = _target_indices(test_cache.metadata, str(heldout_center))
            eval_raw, eval_meta = select_rows(test_cache.embeddings, test_cache.metadata, target_indices)
            eval_labels = tuple(_label(row) for row in eval_meta)
            eval_error = "mono_class_target_eval" if len(set(eval_labels)) < 2 else ""

            for expert_id in candidates:
                for variant in per_source_variants:
                    runtime = per_source_runtime[(str(expert_id), variant.variant_id)]
                    downstream_rows.extend(
                        _evaluate_runtime(
                            cfg,
                            runtime=runtime,
                            experiment_seed=int(experiment_seed),
                            heldout_center=str(heldout_center),
                            eval_raw=eval_raw,
                            eval_labels=eval_labels,
                            eval_error=eval_error,
                            reconstruction_rows=reconstruction_rows,
                        )
                    )
            for variant in union_variants:
                runtime = union_runtime[variant.variant_id]
                downstream_rows.extend(
                    _evaluate_runtime(
                        cfg,
                        runtime=runtime,
                        experiment_seed=int(experiment_seed),
                        heldout_center=str(heldout_center),
                        eval_raw=eval_raw,
                        eval_labels=eval_labels,
                        eval_error=eval_error,
                        reconstruction_rows=reconstruction_rows,
                    )
                )

    _augment_downstream_rows(downstream_rows)
    repair_gap_rows = _gap_rows(downstream_rows)
    _augment_source_pool_strata(repair_gap_rows)
    repair_decision = _decision(repair_gap_rows, cfg, leakage_status="PASS")
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
    )
    if leakage.status != "PASS":
        repair_decision = _decision(repair_gap_rows, cfg, leakage_status=leakage.status)

    write_csv_rows(root / "tables" / "feature_frame_ceiling_matrix.csv", _feature_ceiling_rows(downstream_rows))
    write_csv_rows(root / "tables" / "decode_mu_repair_matrix.csv", downstream_rows)
    write_csv_rows(root / "tables" / "repair_gap_summary.csv", repair_gap_rows)
    write_csv_rows(root / "tables" / "source_pool_capacity_summary.csv", _source_pool_summary_rows(repair_gap_rows))
    write_csv_rows(root / "tables" / "reconstruction_diagnostics.csv", reconstruction_rows)
    write_csv_rows(root / "tables" / "source_probe_diagnostics.csv", probe_rows)
    write_csv_rows(root / "tables" / "training_loss_diagnostics.csv", training_rows)
    write_csv_rows(root / "manifests" / "expert_variant_manifest.csv", manifest_rows)
    write_json(root / "reports" / "leakage_report.json", leakage.to_json_dict())
    _write_protocol_manifest(root, cfg)
    _write_decision_summary(root, repair_decision, leakage.status)
    _write_resolved_config(root / "run_config_resolved.yaml", cfg)
    return root


def _evaluate_runtime(
    cfg: RepairConfig,
    *,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    eval_raw: object,
    eval_labels: Sequence[int],
    eval_error: str,
    reconstruction_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if eval_error:
        rows.append(
            _repair_row(
                cfg,
                runtime=runtime,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                row_role=ROW_REAL_FULL,
                replicate_seed=NA,
                reference_sample_seed=NA,
                source_budget_index_hash="",
                bacc="",
                macro_f1="",
                n_target_eval=len(eval_labels),
                status="ineligible",
                error_message=eval_error,
            )
        )
        for seed in cfg.replicate_seeds:
            for row_role in (ROW_REAL_BUDGET, ROW_DECODE_MU):
                rows.append(
                    _repair_row(
                        cfg,
                        runtime=runtime,
                        experiment_seed=experiment_seed,
                        heldout_center=heldout_center,
                        row_role=row_role,
                        replicate_seed=int(seed),
                        reference_sample_seed=int(seed),
                        source_budget_index_hash="",
                        bacc="",
                        macro_f1="",
                        n_target_eval=len(eval_labels),
                        status="ineligible",
                        error_message=eval_error,
                    )
                )
        return rows

    eval_x = runtime.frame.transform(_to_numpy(eval_raw))
    full_bundle = fit_locked_logistic_classifier(
        runtime.source_train_embeddings,
        runtime.source_train_labels,
        eval_x,
        classifier_seed=cfg.classifier_seed,
        expert_id=runtime.expert_id,
        class_weight=cfg.classifier_class_weight,
    )
    full_result = evaluate_probability_predictions(ROW_REAL_FULL, full_bundle.probabilities, eval_labels)
    rows.append(
        _repair_row(
            cfg,
            runtime=runtime,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            row_role=ROW_REAL_FULL,
            replicate_seed=NA,
            reference_sample_seed=NA,
            source_budget_index_hash=_hash_strings(runtime.source_train_sample_ids),
            bacc=full_result.bacc,
            macro_f1=full_result.macro_f1,
            n_target_eval=len(eval_labels),
            status="ok",
            error_message="",
        )
    )
    for seed in cfg.replicate_seeds:
        sample = _sample_source_positions(runtime, cfg.synthetic_per_class_total, int(seed))
        real_x = _subset_rows(runtime.source_train_embeddings, sample.positions)
        budget_bundle = fit_locked_logistic_classifier(
            real_x,
            sample.labels,
            eval_x,
            classifier_seed=cfg.classifier_seed,
            expert_id=runtime.expert_id,
            class_weight=cfg.classifier_class_weight,
        )
        budget_result = evaluate_probability_predictions(ROW_REAL_BUDGET, budget_bundle.probabilities, eval_labels)
        rows.append(
            _repair_row(
                cfg,
                runtime=runtime,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                row_role=ROW_REAL_BUDGET,
                replicate_seed=int(seed),
                reference_sample_seed=int(seed),
                source_budget_index_hash=sample.source_budget_index_hash,
                bacc=budget_result.bacc,
                macro_f1=budget_result.macro_f1,
                n_target_eval=len(eval_labels),
                status="ok",
                error_message="",
            )
        )

        decoded, diagnostics = _decode_mu(runtime, real_x, sample.labels)
        reconstruction_rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": runtime.expert_id,
                "expert_pool_type": runtime.variant.expert_pool_type,
                "variant_id": runtime.variant.variant_id,
                "row_role": ROW_DECODE_MU,
                "replicate_seed": int(seed),
                "requested_pca_dim": runtime.variant.requested_pca_dim,
                "effective_pca_dim": runtime.frame.effective_dim,
                **diagnostics,
            }
        )
        decode_bundle = fit_locked_logistic_classifier(
            decoded,
            sample.labels,
            eval_x,
            classifier_seed=cfg.classifier_seed,
            expert_id=runtime.expert_id,
            class_weight=cfg.classifier_class_weight,
        )
        decode_result = evaluate_probability_predictions(ROW_DECODE_MU, decode_bundle.probabilities, eval_labels)
        rows.append(
            _repair_row(
                cfg,
                runtime=runtime,
                experiment_seed=experiment_seed,
                heldout_center=heldout_center,
                row_role=ROW_DECODE_MU,
                replicate_seed=int(seed),
                reference_sample_seed=int(seed),
                source_budget_index_hash=sample.source_budget_index_hash,
                bacc=decode_result.bacc,
                macro_f1=decode_result.macro_f1,
                n_target_eval=len(eval_labels),
                status="ok",
                error_message="",
            )
        )
    return rows


def _repair_row(
    cfg: RepairConfig,
    *,
    runtime: VariantRuntime,
    experiment_seed: int,
    heldout_center: str,
    row_role: str,
    replicate_seed: int | str,
    reference_sample_seed: int | str,
    source_budget_index_hash: str,
    bacc: float | str,
    macro_f1: float | str,
    n_target_eval: int,
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
        "replicate_seed": replicate_seed,
        "reference_sample_seed": reference_sample_seed,
        "source_budget_index_hash": source_budget_index_hash,
        "requested_pca_dim": runtime.variant.requested_pca_dim,
        "effective_pca_dim": runtime.frame.effective_dim,
        "latent_dim": runtime.variant.latent_dim,
        "beta_final": runtime.variant.beta_final,
        "kl_warmup_epochs": runtime.variant.kl_warmup_epochs,
        "probe_ce_weight": runtime.variant.probe_ce_weight,
        "reference_real_budget_bacc": "",
        "variant_real_budget_bacc": "",
        "pca_compression_gap": "",
        "source_probe_train_acc": runtime.probe.train_acc,
        "source_probe_val_acc": runtime.probe.val_acc,
        "bacc": bacc,
        "macro_f1": macro_f1,
        "decoder_gap_vs_real_budget": "",
        "source_utility_stratum_reference": "",
        "source_utility_stratum_variant": "",
        "selection_source": runtime.variant.selection_source,
        "status": status,
        "error_message": error_message,
        "classifier_type": cfg.classifier_type,
        "classifier_class_weight": cfg.classifier_class_weight,
        "n_target_eval": int(n_target_eval),
    }


def _runtime_for_variant(
    cfg: RepairConfig,
    *,
    root: Path,
    experiment_seed: int,
    heldout_center: str,
    expert_id: str,
    source_data: SourceData,
    variant: RepairVariant,
) -> VariantRuntime:
    checkpoint_path = (
        root
        / "checkpoints"
        / f"seed{experiment_seed}_heldout{heldout_center}_expert{expert_id}_{variant.variant_id}.pkl"
    )
    if checkpoint_path.exists():
        with checkpoint_path.open("rb") as f:
            return pickle.load(f)
    runtime = _train_variant_runtime(
        cfg,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        expert_id=expert_id,
        source_data=source_data,
        variant=variant,
    )
    with checkpoint_path.open("wb") as f:
        pickle.dump(runtime, f)
    return runtime


def _train_variant_runtime(
    cfg: RepairConfig,
    *,
    experiment_seed: int,
    heldout_center: str,
    expert_id: str,
    source_data: SourceData,
    variant: RepairVariant,
) -> VariantRuntime:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    torch.manual_seed(_stable_seed(experiment_seed, heldout_center, expert_id, variant.variant_id))
    frame = fit_expert_frame(
        expert_id=str(expert_id),
        source_train_embeddings=_to_numpy(source_data.raw_train),
        requested_dim=variant.requested_pca_dim,
    )
    train_x = np.asarray(frame.transform(_to_numpy(source_data.raw_train)), dtype=np.float32)
    val_x = np.asarray(frame.transform(_to_numpy(source_data.raw_val)), dtype=np.float32)
    train_y = tuple(int(v) for v in source_data.train_labels)
    val_y = tuple(int(v) for v in source_data.val_labels)
    if set(train_y) != {0, 1}:
        raise ProtocolError(f"Expert {expert_id} source train split must contain classes 0 and 1.")

    probe = _train_source_probe(cfg, variant, train_x, train_y, val_x, val_y, seed=experiment_seed)
    model = ClassConditionedCVAE(
        input_dim=int(train_x.shape[1]),
        hidden_dim=variant.hidden_dim,
        latent_dim=variant.latent_dim,
        n_classes=2,
        num_hidden_layers=variant.num_hidden_layers,
    )
    training_rows = _train_repair_cvae(
        model,
        variant=variant,
        probe=probe.model,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        experiment_seed=experiment_seed,
        heldout_center=heldout_center,
        expert_id=expert_id,
    )
    return VariantRuntime(
        variant=variant,
        expert_id=str(expert_id),
        frame=frame,
        model=model,
        probe=probe,
        source_train_embeddings=train_x,
        source_val_embeddings=val_x,
        source_train_labels=train_y,
        source_val_labels=val_y,
        source_train_sample_ids=source_data.train_sample_ids,
        source_train_centers=source_data.train_centers,
        source_scope=source_data.source_scope,
        n_train=int(train_x.shape[0]),
        n_val=int(val_x.shape[0]),
        training_rows=tuple(training_rows),
    )


def _train_repair_cvae(
    model: ClassConditionedCVAE,
    *,
    variant: RepairVariant,
    probe: object | None,
    train_x: object,
    train_y: Sequence[int],
    val_x: object,
    val_y: Sequence[int],
    experiment_seed: int,
    heldout_center: str,
    expert_id: str,
) -> list[dict[str, object]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from torch.nn.utils import clip_grad_norm_  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    x_train = torch.as_tensor(np.asarray(train_x, dtype=np.float32))
    y_train = torch.as_tensor(np.asarray(train_y, dtype=np.int64))
    x_val = torch.as_tensor(np.asarray(val_x, dtype=np.float32))
    y_val = torch.as_tensor(np.asarray(val_y, dtype=np.int64))
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(variant.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(_stable_seed(experiment_seed, expert_id, variant.variant_id, "loader")),
    )
    opt_cls = torch.optim.Adam if variant.optimizer == "adam" else torch.optim.AdamW
    opt = opt_cls(model.parameters(), lr=float(variant.learning_rate), weight_decay=float(variant.weight_decay))
    rows: list[dict[str, object]] = []
    for epoch in range(1, int(variant.train_epochs) + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            if variant.loss_style == "legacy_sum_mse_kl":
                loss = loss_for_batch(model, xb, yb).nelbo
            else:
                loss, _terms = _repair_loss(
                    model,
                    xb,
                    yb,
                    beta=_beta_for_epoch(variant, epoch),
                    probe=probe,
                    probe_ce_weight=variant.probe_ce_weight,
                )
            loss.backward()
            if variant.gradient_clip_norm > 0:
                clip_grad_norm_(model.parameters(), float(variant.gradient_clip_norm))
            opt.step()
        train_terms = _loss_terms(model, x_train, y_train, variant, probe=probe, epoch=epoch)
        val_terms = _loss_terms(model, x_val, y_val, variant, probe=probe, epoch=epoch)
        rows.append(
            {
                "experiment_seed": int(experiment_seed),
                "heldout_center": str(heldout_center),
                "expert_id": str(expert_id),
                "expert_pool_type": variant.expert_pool_type,
                "variant_id": variant.variant_id,
                "epoch": int(epoch),
                "beta": _beta_for_epoch(variant, epoch),
                "recon_mse_per_dim_train": train_terms["recon_mse_per_dim"],
                "recon_mse_per_dim_val": val_terms["recon_mse_per_dim"],
                "kl_per_latent_dim_train": train_terms["kl_per_latent_dim"],
                "kl_per_latent_dim_val": val_terms["kl_per_latent_dim"],
                "probe_ce_train": train_terms["probe_ce"],
                "probe_ce_val": val_terms["probe_ce"],
                "weighted_total_loss_train": train_terms["weighted_total_loss"],
                "weighted_total_loss_val": val_terms["weighted_total_loss"],
                "mu_norm_mean": val_terms["mu_norm_mean"],
                "logvar_mean": val_terms["logvar_mean"],
                "posterior_sigma_mean": val_terms["posterior_sigma_mean"],
            }
        )
    model.eval()
    return rows


def _repair_loss(
    model: ClassConditionedCVAE,
    x: object,
    y: object,
    *,
    beta: float,
    probe: object | None,
    probe_ce_weight: float,
) -> tuple[object, Mapping[str, object]]:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    mu, logvar = model.encode(x, y)
    decoded = model.decode(mu, y)
    recon = F.mse_loss(decoded, x, reduction="none").mean(dim=1).mean()
    kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / float(model.latent_dim)).mean()
    probe_ce = torch.zeros((), dtype=recon.dtype, device=recon.device)
    if probe is not None and float(probe_ce_weight) > 0.0:
        probe_ce = F.cross_entropy(probe(decoded), y.long())
    total = recon + (float(beta) * kl) + (float(probe_ce_weight) * probe_ce)
    return total, {"recon": recon, "kl": kl, "probe_ce": probe_ce}


def _loss_terms(
    model: ClassConditionedCVAE,
    x: object,
    y: object,
    variant: RepairVariant,
    *,
    probe: object | None,
    epoch: int,
) -> dict[str, float]:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x, y)
        decoded = model.decode(mu, y)
        recon = F.mse_loss(decoded, x, reduction="none").mean(dim=1).mean()
        kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1) / float(model.latent_dim)).mean()
        probe_ce = torch.zeros((), dtype=recon.dtype, device=recon.device)
        if probe is not None and float(variant.probe_ce_weight) > 0.0:
            probe_ce = F.cross_entropy(probe(decoded), y.long())
        beta = _beta_for_epoch(variant, epoch)
        total = recon + (float(beta) * kl) + (float(variant.probe_ce_weight) * probe_ce)
        sigma = torch.exp(0.5 * logvar)
    return {
        "recon_mse_per_dim": float(recon.detach().cpu()),
        "kl_per_latent_dim": float(kl.detach().cpu()),
        "probe_ce": float(probe_ce.detach().cpu()),
        "weighted_total_loss": float(total.detach().cpu()),
        "mu_norm_mean": float(torch.norm(mu, dim=1).mean().detach().cpu()),
        "logvar_mean": float(logvar.mean().detach().cpu()),
        "posterior_sigma_mean": float(sigma.mean().detach().cpu()),
    }


def _train_source_probe(
    cfg: RepairConfig,
    variant: RepairVariant,
    train_x: object,
    train_y: Sequence[int],
    val_x: object,
    val_y: Sequence[int],
    *,
    seed: int,
) -> SourceProbeRuntime:
    if float(variant.probe_ce_weight) <= 0.0:
        return SourceProbeRuntime(
            model=None,
            train_acc="",
            val_acc="",
            best_val_acc="",
            train_loss="",
            val_loss="",
            epochs_trained="",
        )
    import numpy as np  # type: ignore
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    torch.manual_seed(_stable_seed(seed, variant.variant_id, "source_probe"))
    x_train = torch.as_tensor(np.asarray(train_x, dtype=np.float32))
    y_train = torch.as_tensor(np.asarray(train_y, dtype=np.int64))
    x_val = torch.as_tensor(np.asarray(val_x, dtype=np.float32))
    y_val = torch.as_tensor(np.asarray(val_y, dtype=np.int64))
    model = nn.Linear(int(x_train.shape[1]), 2)
    weights = _class_weights(y_train)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=int(cfg.source_probe.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(_stable_seed(seed, variant.variant_id, "probe_loader")),
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.source_probe.learning_rate),
        weight_decay=float(cfg.source_probe.weight_decay),
    )
    train_loss = 0.0
    val_loss = 0.0
    best_val = 0.0
    for _epoch in range(int(cfg.source_probe.epochs)):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            loss.backward()
            opt.step()
        train_loss = float(F.cross_entropy(model(x_train), y_train, weight=weights).detach().cpu())
        val_loss = float(F.cross_entropy(model(x_val), y_val, weight=weights).detach().cpu())
        best_val = max(best_val, _accuracy(model, x_val, y_val))
    train_acc = _accuracy(model, x_train, y_train)
    val_acc = _accuracy(model, x_val, y_val)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return SourceProbeRuntime(
        model=model,
        train_acc=float(train_acc),
        val_acc=float(val_acc),
        best_val_acc=float(best_val),
        train_loss=float(train_loss),
        val_loss=float(val_loss),
        epochs_trained=int(cfg.source_probe.epochs),
    )


def _class_weights(y: object) -> object:
    import torch  # type: ignore

    counts = torch.bincount(y.long(), minlength=2).to(dtype=torch.float32)
    weights = counts.sum() / (2.0 * torch.clamp(counts, min=1.0))
    return weights


def _accuracy(model: object, x: object, y: object) -> float:
    import torch  # type: ignore

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)
        return float((pred == y).to(dtype=torch.float32).mean().detach().cpu())


def _decode_mu(runtime: VariantRuntime, x: object, labels: Sequence[int]) -> tuple[object, dict[str, float]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    with torch.no_grad():
        xt = torch.as_tensor(x_np, dtype=torch.float32)
        yt = torch.as_tensor(y_np, dtype=torch.long)
        mu, logvar = runtime.model.encode(xt, yt)
        decoded = runtime.model.decode(mu, yt).detach().cpu().numpy()
    return decoded, _reconstruction_diagnostics(x_np, decoded, mu, logvar)


def _source_data_for_centers(
    train_cache: FeatureCache,
    *,
    centers: Sequence[str],
    experiment_seed: int,
) -> SourceData:
    import numpy as np  # type: ignore

    raw_train = []
    raw_val = []
    train_labels: list[int] = []
    val_labels: list[int] = []
    train_ids: list[str] = []
    val_ids: list[str] = []
    train_centers: list[str] = []
    val_centers: list[str] = []
    for center in centers:
        split = stratified_source_train_val_split(
            train_cache.metadata,
            center=str(center),
            experiment_seed=int(experiment_seed),
        )
        source_train_raw, source_train_meta = select_rows(train_cache.embeddings, train_cache.metadata, split.train_indices)
        source_val_raw, source_val_meta = select_rows(train_cache.embeddings, train_cache.metadata, split.val_indices)
        raw_train.append(_to_numpy(source_train_raw))
        raw_val.append(_to_numpy(source_val_raw))
        train_labels.extend(_label(row) for row in source_train_meta)
        val_labels.extend(_label(row) for row in source_val_meta)
        train_ids.extend(str(v) for v in split.train_sample_ids)
        val_ids.extend(str(v) for v in split.val_sample_ids)
        train_centers.extend(str(row.get("center", center)) for row in source_train_meta)
        val_centers.extend(str(row.get("center", center)) for row in source_val_meta)
    return SourceData(
        raw_train=np.vstack(raw_train),
        raw_val=np.vstack(raw_val),
        train_labels=tuple(train_labels),
        val_labels=tuple(val_labels),
        train_sample_ids=tuple(train_ids),
        val_sample_ids=tuple(val_ids),
        train_centers=tuple(train_centers),
        val_centers=tuple(val_centers),
        source_scope="|".join(str(v) for v in centers),
    )


def _sample_source_positions(runtime: VariantRuntime, budget_per_class: int, seed: int) -> SourceSample:
    import numpy as np  # type: ignore

    labels_np = np.asarray(runtime.source_train_labels, dtype=int)
    rng = np.random.default_rng(int(seed))
    positions: list[int] = []
    labels: list[int] = []
    hash_parts: list[str] = []
    for cls in (0, 1):
        pool = np.where(labels_np == int(cls))[0]
        if pool.size == 0:
            raise ProtocolError(f"Expert {runtime.expert_id} has no source refs for class {cls}.")
        chosen_offsets = rng.integers(0, pool.size, size=int(budget_per_class))
        chosen = tuple(int(pool[int(offset)]) for offset in chosen_offsets)
        positions.extend(chosen)
        labels.extend([int(cls)] * int(budget_per_class))
        hash_parts.extend(f"{cls}:{runtime.source_train_sample_ids[pos]}" for pos in chosen)
    return SourceSample(
        positions=tuple(positions),
        labels=tuple(labels),
        source_budget_index_hash=_hash_strings(hash_parts),
    )


def _augment_downstream_rows(rows: list[dict[str, object]]) -> None:
    budget_rows = [
        row for row in rows
        if row.get("status") == "ok" and row.get("row_role") == ROW_REAL_BUDGET
    ]
    reference_by_key: dict[tuple[object, object, object, object], float] = {}
    variant_budget_by_key: dict[tuple[object, object, object, object, object, object], float] = {}
    union_reference_by_key: dict[tuple[object, object, object], float] = {}
    for row in budget_rows:
        key = (row["experiment_seed"], row["heldout_center"], row["expert_id"], row["replicate_seed"])
        if row.get("variant_id") == "current_pca200_beta1_reference" and row.get("expert_pool_type") == POOL_PER_SOURCE:
            reference_by_key[key] = _float(row["bacc"])
    for row in budget_rows:
        key = (
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["replicate_seed"],
            row["expert_pool_type"],
            row["variant_id"],
        )
        variant_budget_by_key[key] = _float(row["bacc"])
    for (seed, heldout, expert_id, replicate), value in reference_by_key.items():
        union_key = (seed, heldout, replicate)
        union_reference_by_key[union_key] = max(value, union_reference_by_key.get(union_key, -math.inf))

    for row in rows:
        if row.get("status") != "ok" or row.get("replicate_seed") == NA:
            continue
        if row.get("expert_pool_type") == POOL_SOURCE_UNION:
            ref = union_reference_by_key.get((row["experiment_seed"], row["heldout_center"], row["replicate_seed"]))
        else:
            ref = reference_by_key.get((row["experiment_seed"], row["heldout_center"], row["expert_id"], row["replicate_seed"]))
        variant = variant_budget_by_key.get(
            (
                row["experiment_seed"],
                row["heldout_center"],
                row["expert_id"],
                row["replicate_seed"],
                row["expert_pool_type"],
                row["variant_id"],
            )
        )
        if ref is None or not math.isfinite(ref) or variant is None:
            continue
        row["reference_real_budget_bacc"] = ref
        row["variant_real_budget_bacc"] = variant
        row["pca_compression_gap"] = ref - variant
        row["source_utility_stratum_reference"] = _source_utility_stratum(ref)
        row["source_utility_stratum_variant"] = _source_utility_stratum(variant)
        if row.get("row_role") == ROW_DECODE_MU:
            row["decoder_gap_vs_real_budget"] = variant - _float(row["bacc"])


def _gap_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    budgets = {
        _row_pair_key(row): row
        for row in rows
        if row.get("status") == "ok" and row.get("row_role") == ROW_REAL_BUDGET
    }
    decodes = {
        _row_pair_key(row): row
        for row in rows
        if row.get("status") == "ok" and row.get("row_role") == ROW_DECODE_MU
    }
    out: list[dict[str, object]] = []
    for key, budget in sorted(budgets.items(), key=lambda item: tuple(str(v) for v in item[0])):
        decode = decodes.get(key)
        if not decode:
            continue
        if budget["source_budget_index_hash"] != decode["source_budget_index_hash"]:
            raise ProtocolError("Real-budget and decode rows used different source samples.")
        variant_real = _float(budget["bacc"])
        decode_bacc = _float(decode["bacc"])
        out.append(
            {
                "experiment_seed": budget["experiment_seed"],
                "heldout_center": budget["heldout_center"],
                "expert_id": budget["expert_id"],
                "expert_pool_type": budget["expert_pool_type"],
                "variant_id": budget["variant_id"],
                "replicate_seed": budget["replicate_seed"],
                "source_budget_index_hash": budget["source_budget_index_hash"],
                "reference_real_budget_bacc": budget["reference_real_budget_bacc"],
                "variant_real_budget_bacc": variant_real,
                "cvae_decode_mu_bacc": decode_bacc,
                "real_source_budget_matched_macro_f1": budget["macro_f1"],
                "cvae_decode_mu_macro_f1": decode["macro_f1"],
                "pca_compression_gap": budget["pca_compression_gap"],
                "decoder_gap_vs_real_budget": variant_real - decode_bacc,
                "source_utility_stratum_reference": budget["source_utility_stratum_reference"],
                "source_utility_stratum_variant": budget["source_utility_stratum_variant"],
                "selection_source": budget["selection_source"],
                "status": "ok",
            }
        )
    _assert_cross_variant_hashes(out)
    return out


def _assert_cross_variant_hashes(rows: Sequence[Mapping[str, object]]) -> None:
    by_key: dict[tuple[object, object, object, object, object], set[str]] = {}
    for row in rows:
        key = (
            row["experiment_seed"],
            row["heldout_center"],
            row["expert_id"],
            row["expert_pool_type"],
            row["replicate_seed"],
        )
        by_key.setdefault(key, set()).add(str(row["source_budget_index_hash"]))
    for key, hashes in by_key.items():
        if len(hashes) > 1:
            raise ProtocolError(f"Cross-variant source-budget sample mismatch for {key}.")


def _augment_source_pool_strata(rows: list[dict[str, object]]) -> None:
    # Source-union rows inherit their reference stratum from the best per-source
    # current-reference budget for the same seed/heldout/replicate.
    for row in rows:
        if row.get("expert_pool_type") == POOL_SOURCE_UNION and not row.get("source_utility_stratum_reference"):
            row["source_utility_stratum_reference"] = _source_utility_stratum(_float(row["reference_real_budget_bacc"]))


def _decision(rows: Sequence[Mapping[str, object]], cfg: RepairConfig, *, leakage_status: str) -> dict[str, object]:
    primary = _decision_rows(rows, PRIMARY_VARIANT, POOL_PER_SOURCE)
    reference = _decision_rows(rows, "current_pca200_beta1_reference", POOL_PER_SOURCE)
    primary_stats = _repair_stats(primary)
    reference_stats = _repair_stats(reference)
    primary_stats["reference_decode_mu_bacc"] = reference_stats["mean_decode_mu_bacc"]
    primary_stats["decode_mu_improvement_vs_reference"] = (
        primary_stats["mean_decode_mu_bacc"] - reference_stats["mean_decode_mu_bacc"]
        if math.isfinite(primary_stats["mean_decode_mu_bacc"]) and math.isfinite(reference_stats["mean_decode_mu_bacc"])
        else math.nan
    )
    primary_verdict = "DIAGNOSTIC_MIXED"
    if leakage_status != "PASS":
        primary_verdict = "PROTOCOL_FAIL"
    elif int(primary_stats["n_decision_rows"]) < int(cfg.min_decision_rows):
        primary_verdict = "INSUFFICIENT_DECISION_ROWS"
    elif (
        primary_stats["mean_decoder_gap"] <= 0.05
        and primary_stats["mean_pca_compression_gap"] > 0.05
        and primary_stats["mean_variant_real_budget_bacc"] < 0.70
    ):
        primary_verdict = "PCA_COMPRESSION_FAIL"
    elif (
        primary_stats["mean_decode_mu_bacc"] >= 0.70
        and primary_stats["mean_decoder_gap"] <= 0.05
        and (
            primary_stats["mean_pca_compression_gap"] <= 0.05
            or primary_stats["mean_variant_real_budget_bacc"] >= 0.70
        )
        and primary_stats["seed_std_decode_mu_bacc"] <= 0.05
    ):
        primary_verdict = "REPAIR_PASS"
    elif (
        primary_stats["mean_decode_mu_bacc"] >= 0.65
        or primary_stats["decode_mu_improvement_vs_reference"] >= 0.10
    ):
        primary_verdict = "REPAIR_PARTIAL"
    elif (
        primary_stats["mean_decode_mu_bacc"] < 0.60
        and reference_stats["mean_decode_mu_bacc"] < 0.60
    ):
        primary_verdict = "REPAIR_FAIL"

    flags = []
    if _passes_repair(_repair_stats(_decision_rows(rows, "pca64_beta001_probe025", POOL_PER_SOURCE, include_diagnostic=True))) and primary_verdict != "REPAIR_PASS":
        flags.append("PROBE_RESCUE")
    if _passes_repair(_repair_stats(_decision_rows(rows, "source_union_pca64_beta001_diagnostic", POOL_SOURCE_UNION, include_diagnostic=True))) and primary_verdict != "REPAIR_PASS":
        flags.append("SOURCE_POOL_CAPACITY_ONLY")
    if _probe_overfit(rows):
        flags.append("PROBE_OVERFIT")
    return {
        "primary_verdict": primary_verdict,
        "diagnostic_flags": "|".join(flags),
        **primary_stats,
    }


def _decision_rows(
    rows: Sequence[Mapping[str, object]],
    variant_id: str,
    pool_type: str,
    *,
    include_diagnostic: bool = False,
) -> list[Mapping[str, object]]:
    return [
        row for row in rows
        if row.get("variant_id") == variant_id
        and row.get("expert_pool_type") == pool_type
        and row.get("status") == "ok"
        and row.get("source_utility_stratum_reference") in {"medium", "high"}
        and (include_diagnostic or row.get("selection_source") != "diagnostic_only")
    ]


def _repair_stats(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped = _replicate_averaged_rows(rows)
    cells: dict[tuple[str, str], list[Mapping[str, float]]] = {}
    by_seed: dict[str, list[float]] = {}
    centers = set()
    experts = set()
    for row in grouped:
        center = str(row["heldout_center"])
        expert = str(row["expert_id"])
        seed = str(row["experiment_seed"])
        centers.add(center)
        experts.add(expert)
        cells.setdefault((center, expert), []).append(row)
        by_seed.setdefault(seed, []).append(float(row["cvae_decode_mu_bacc"]))
    cell_means = [_mean_dicts(values) for values in cells.values()]
    seed_values = [nanmean(values) for values in by_seed.values()]
    return {
        "n_decision_rows": len(rows),
        "n_heldout_centers_covered": len(centers),
        "n_experts_covered": len(experts),
        "mean_decode_mu_bacc": _mean_field(cell_means, "cvae_decode_mu_bacc"),
        "mean_decoder_gap": _mean_field(cell_means, "decoder_gap_vs_real_budget"),
        "mean_pca_compression_gap": _mean_field(cell_means, "pca_compression_gap"),
        "mean_variant_real_budget_bacc": _mean_field(cell_means, "variant_real_budget_bacc"),
        "seed_std_decode_mu_bacc": _std(seed_values),
        "per_center_decode_mu_bacc": json.dumps(_per_center_mean(grouped, "cvae_decode_mu_bacc"), sort_keys=True),
        "per_center_decoder_gap": json.dumps(_per_center_mean(grouped, "decoder_gap_vs_real_budget"), sort_keys=True),
    }


def _passes_repair(stats: Mapping[str, object]) -> bool:
    return (
        _float(stats.get("n_decision_rows", 0)) >= 1
        and _float(stats["mean_decode_mu_bacc"]) >= 0.70
        and _float(stats["mean_decoder_gap"]) <= 0.05
        and (
            _float(stats["mean_pca_compression_gap"]) <= 0.05
            or _float(stats["mean_variant_real_budget_bacc"]) >= 0.70
        )
        and _float(stats["seed_std_decode_mu_bacc"]) <= 0.05
    )


def _probe_overfit(rows: Sequence[Mapping[str, object]]) -> bool:
    probe_rows = _decision_rows(rows, "pca64_beta001_probe025", POOL_PER_SOURCE, include_diagnostic=True)
    base_rows = _decision_rows(rows, "pca64_beta001", POOL_PER_SOURCE)
    if not probe_rows or not base_rows:
        return False
    probe_decode = _mean_field(probe_rows, "cvae_decode_mu_bacc")
    base_decode = _mean_field(base_rows, "cvae_decode_mu_bacc")
    # The gap needs source-probe diagnostics, which are recorded in matrix rows.
    train_acc = _mean_field(probe_rows, "source_probe_train_acc")
    val_acc = _mean_field(probe_rows, "source_probe_val_acc")
    return (
        train_acc >= 0.90
        and val_acc < 0.70
        and (train_acc - val_acc) >= 0.15
        and (probe_decode - base_decode) < 0.03
    )


def _feature_ceiling_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [row for row in rows if row.get("row_role") in {ROW_REAL_FULL, ROW_REAL_BUDGET}]


def _source_pool_summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for variant_id in sorted({str(row["variant_id"]) for row in rows if row.get("expert_pool_type") == POOL_SOURCE_UNION}):
        subset = [row for row in rows if row.get("variant_id") == variant_id and row.get("expert_pool_type") == POOL_SOURCE_UNION]
        out.append(
            {
                "variant_id": variant_id,
                "expert_pool_type": POOL_SOURCE_UNION,
                "n": len(subset),
                "mean_variant_real_budget_bacc": _mean_field(subset, "variant_real_budget_bacc"),
                "mean_cvae_decode_mu_bacc": _mean_field(subset, "cvae_decode_mu_bacc"),
                "mean_decoder_gap_vs_real_budget": _mean_field(subset, "decoder_gap_vs_real_budget"),
                "mean_pca_compression_gap": _mean_field(subset, "pca_compression_gap"),
                "selection_source": "diagnostic_only",
            }
        )
    return out


def _manifest_row(experiment_seed: int, heldout_center: str, runtime: VariantRuntime) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
        import sklearn  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError:
        np = sklearn = torch = None  # type: ignore
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "source_scope": runtime.source_scope,
        "checkpoint_path": "artifacts/checkpoints",
        "n_train": runtime.n_train,
        "n_val": runtime.n_val,
        "requested_pca_dim": runtime.variant.requested_pca_dim,
        "effective_pca_dim": runtime.frame.effective_dim,
        "latent_dim": runtime.variant.latent_dim,
        "beta_final": runtime.variant.beta_final,
        "kl_warmup_epochs": runtime.variant.kl_warmup_epochs,
        "probe_ce_weight": runtime.variant.probe_ce_weight,
        "source_probe_epochs_trained": runtime.probe.epochs_trained,
        "source_probe_best_val_acc": runtime.probe.best_val_acc,
        "source_probe_train_loss": runtime.probe.train_loss,
        "source_probe_val_loss": runtime.probe.val_loss,
        "optimizer": runtime.variant.optimizer,
        "learning_rate": runtime.variant.learning_rate,
        "weight_decay": runtime.variant.weight_decay,
        "batch_size": runtime.variant.batch_size,
        "gradient_clip_norm": runtime.variant.gradient_clip_norm,
        "activation": runtime.variant.activation,
        "dropout": runtime.variant.dropout,
        "torch_version": "" if torch is None else torch.__version__,
        "sklearn_version": "" if sklearn is None else sklearn.__version__,
        "numpy_version": "" if np is None else np.__version__,
        "cuda_available": False if torch is None else bool(torch.cuda.is_available()),
        "device": runtime.variant.device,
        "pca_explained_variance_ratio_sum": runtime.frame.explained_variance_ratio_sum,
    }


def _probe_row(experiment_seed: int, heldout_center: str, runtime: VariantRuntime) -> dict[str, object]:
    return {
        "experiment_seed": int(experiment_seed),
        "heldout_center": str(heldout_center),
        "expert_id": runtime.expert_id,
        "expert_pool_type": runtime.variant.expert_pool_type,
        "variant_id": runtime.variant.variant_id,
        "source_probe_train_acc": runtime.probe.train_acc,
        "source_probe_val_acc": runtime.probe.val_acc,
        "source_probe_best_val_acc": runtime.probe.best_val_acc,
        "source_probe_train_loss": runtime.probe.train_loss,
        "source_probe_val_loss": runtime.probe.val_loss,
        "source_probe_epochs_trained": runtime.probe.epochs_trained,
        "selection_source": runtime.variant.selection_source,
    }


def _write_protocol_manifest(root: Path, cfg: RepairConfig) -> None:
    write_json(
        root / "manifests" / "protocol_manifest.json",
        {
            "schema_version": "cvae_rebuild_preservation_repair_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "preservation_repair",
            "primary_variant": cfg.primary_variant,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": True,
            "secondary_variants_diagnostic_only": True,
            "source_union_diagnostic_only": True,
            "row_roles": list(ROW_ROLES),
            "variant_ids": [variant.variant_id for variant in cfg.variants],
            "claim_boundary": "downstream utility preservation only; no formal privacy claim",
        },
    )


def _write_decision_summary(root: Path, decision: Mapping[str, object], leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Virchow2-CVAE Preservation Repair v1",
            "",
            "## Summary",
            "",
            f"- Primary variant: `{PRIMARY_VARIANT}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'DIAGNOSTIC_MIXED')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Mean decode(mu) BACC: {_format_float(decision.get('mean_decode_mu_bacc'))}",
            f"- Mean decoder gap: {_format_float(decision.get('mean_decoder_gap'))}",
            f"- Mean PCA compression gap: {_format_float(decision.get('mean_pca_compression_gap'))}",
            f"- Seed std decode(mu) BACC: {_format_float(decision.get('seed_std_decode_mu_bacc'))}",
            f"- Decision rows: {decision.get('n_decision_rows', 0)}",
            f"- Held-out centers covered: {decision.get('n_heldout_centers_covered', 0)}",
            f"- Experts covered: {decision.get('n_experts_covered', 0)}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Per-Center Diagnostics",
            "",
            f"- Decode(mu) BACC: `{decision.get('per_center_decode_mu_bacc', '{}')}`",
            f"- Decoder gap: `{decision.get('per_center_decoder_gap', '{}')}`",
            "",
            "## Claim Boundary",
            "",
            "This slice diagnoses deterministic CVAE utility preservation.",
            "It does not evaluate posterior sampling, prior sampling, support-NELBO routing, metadata routing, top-k composition, or formal privacy.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_resolved_config(path: str | Path, cfg: RepairConfig) -> None:
    Path(path).write_text(json.dumps(_resolved_config(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolved_config(cfg: RepairConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "primary_variant": cfg.primary_variant,
        "min_decision_rows": cfg.min_decision_rows,
        "variants": [variant.__dict__ for variant in cfg.variants],
        "source_probe": cfg.source_probe.__dict__,
        "classifier_type": cfg.classifier_type,
        "classifier_solver": cfg.classifier_solver,
        "classifier_c": cfg.classifier_c,
        "classifier_max_iter": cfg.classifier_max_iter,
        "classifier_class_weight": cfg.classifier_class_weight,
        "classifier_seed": cfg.classifier_seed,
    }


def _parse_variant(value: object) -> RepairVariant:
    if not isinstance(value, Mapping):
        raise ProtocolError("Each repair variant must be a mapping.")
    return RepairVariant(
        variant_id=str(value["variant_id"]),
        expert_pool_type=str(value["expert_pool_type"]),
        requested_pca_dim=int(value["requested_pca_dim"]),
        latent_dim=int(value["latent_dim"]),
        train_epochs=int(value["train_epochs"]),
        beta_final=float(value["beta_final"]),
        kl_warmup_epochs=int(value["kl_warmup_epochs"]),
        probe_ce_weight=float(value["probe_ce_weight"]),
        loss_style=str(value["loss_style"]),
        selection_source=str(value["selection_source"]),
        hidden_dim=int(value.get("hidden_dim", 512)),
        num_hidden_layers=int(value.get("num_hidden_layers", 2)),
        batch_size=int(value.get("batch_size", 128)),
        learning_rate=float(value.get("learning_rate", 1.0e-3)),
        optimizer=str(value.get("optimizer", "adamw")),
        weight_decay=float(value.get("weight_decay", 1.0e-4)),
        gradient_clip_norm=float(value.get("gradient_clip_norm", 5.0)),
        activation=str(value.get("activation", "relu")),
        dropout=float(value.get("dropout", 0.0)),
        device=str(value.get("device", "cpu")),
    )


def _validate_variant(variant: RepairVariant) -> None:
    expected: dict[str, dict[str, object]] = {
        "current_pca200_beta1_reference": {
            "pool": POOL_PER_SOURCE,
            "pca": 256,
            "latent": 64,
            "beta": 1.0,
            "probe": 0.0,
            "loss": "legacy_sum_mse_kl",
            "selection": "reference_only",
            "optimizer": "adam",
        },
        "pca64_beta001": {
            "pool": POOL_PER_SOURCE,
            "pca": 64,
            "latent": 16,
            "beta": 0.001,
            "probe": 0.0,
            "loss": "normalized_repair",
            "selection": "primary",
            "optimizer": "adamw",
        },
        "pca128_beta001": {
            "pool": POOL_PER_SOURCE,
            "pca": 128,
            "latent": 32,
            "beta": 0.001,
            "probe": 0.0,
            "loss": "normalized_repair",
            "selection": "diagnostic_only",
            "optimizer": "adamw",
        },
        "pca64_beta001_probe025": {
            "pool": POOL_PER_SOURCE,
            "pca": 64,
            "latent": 16,
            "beta": 0.001,
            "probe": 0.25,
            "loss": "normalized_repair",
            "selection": "diagnostic_only",
            "optimizer": "adamw",
        },
        "pca128_beta001_probe025": {
            "pool": POOL_PER_SOURCE,
            "pca": 128,
            "latent": 32,
            "beta": 0.001,
            "probe": 0.25,
            "loss": "normalized_repair",
            "selection": "diagnostic_only",
            "optimizer": "adamw",
        },
        "source_union_pca64_beta001_diagnostic": {
            "pool": POOL_SOURCE_UNION,
            "pca": 64,
            "latent": 16,
            "beta": 0.001,
            "probe": 0.0,
            "loss": "normalized_repair",
            "selection": "diagnostic_only",
            "optimizer": "adamw",
        },
        "source_union_pca64_beta001_probe025_diagnostic": {
            "pool": POOL_SOURCE_UNION,
            "pca": 64,
            "latent": 16,
            "beta": 0.001,
            "probe": 0.25,
            "loss": "normalized_repair",
            "selection": "diagnostic_only",
            "optimizer": "adamw",
        },
    }
    spec = expected.get(variant.variant_id)
    if spec is None:
        raise ProtocolError(f"Unknown repair variant {variant.variant_id!r}.")
    if variant.expert_pool_type != spec["pool"]:
        raise ProtocolError(f"Variant {variant.variant_id} has wrong expert_pool_type.")
    if variant.requested_pca_dim != spec["pca"] or variant.latent_dim != spec["latent"]:
        raise ProtocolError(f"Variant {variant.variant_id} has wrong PCA/latent dimension.")
    if not math.isclose(variant.beta_final, float(spec["beta"])) or not math.isclose(variant.probe_ce_weight, float(spec["probe"])):
        raise ProtocolError(f"Variant {variant.variant_id} has wrong beta/probe weight.")
    if variant.loss_style != spec["loss"] or variant.selection_source != spec["selection"]:
        raise ProtocolError(f"Variant {variant.variant_id} has wrong loss or selection source.")
    if variant.optimizer != spec["optimizer"]:
        raise ProtocolError(f"Variant {variant.variant_id} has wrong optimizer.")
    if variant.num_hidden_layers != 2 or variant.hidden_dim != 512:
        raise ProtocolError("Repair variants must use hidden_dim=512 and two hidden layers.")
    if variant.train_epochs <= 0 or variant.batch_size <= 0 or variant.learning_rate <= 0:
        raise ProtocolError("Repair variant training settings must be positive.")
    if variant.loss_style == "normalized_repair" and variant.kl_warmup_epochs <= 0:
        raise ProtocolError("Normalized repair variants require positive kl_warmup_epochs.")


def _row_pair_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["experiment_seed"],
        row["heldout_center"],
        row["expert_id"],
        row["expert_pool_type"],
        row["variant_id"],
        row["replicate_seed"],
    )


def _replicate_averaged_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["experiment_seed"]), str(row["heldout_center"]), str(row["expert_id"])), []).append(row)
    out = []
    for (seed, center, expert), subset in groups.items():
        out.append(
            {
                "experiment_seed": seed,
                "heldout_center": center,
                "expert_id": expert,
                "cvae_decode_mu_bacc": _mean_field(subset, "cvae_decode_mu_bacc"),
                "decoder_gap_vs_real_budget": _mean_field(subset, "decoder_gap_vs_real_budget"),
                "pca_compression_gap": _mean_field(subset, "pca_compression_gap"),
                "variant_real_budget_bacc": _mean_field(subset, "variant_real_budget_bacc"),
            }
        )
    return out


def _mean_dicts(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    fields = ("cvae_decode_mu_bacc", "decoder_gap_vs_real_budget", "pca_compression_gap", "variant_real_budget_bacc")
    return {field: _mean_field(rows, field) for field in fields}


def _per_center_mean(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row["heldout_center"]), []).append(_float(row[field]))
    return {center: nanmean(values) for center, values in sorted(groups.items())}


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


def _beta_for_epoch(variant: RepairVariant, epoch: int) -> float:
    if variant.loss_style == "legacy_sum_mse_kl":
        return float(variant.beta_final)
    return float(variant.beta_final) * min(1.0, float(epoch) / float(variant.kl_warmup_epochs))


def _subset_rows(value: object, indices: Sequence[int]) -> object:
    import numpy as np  # type: ignore

    return np.asarray(value)[[int(idx) for idx in indices]]


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


def _source_utility_stratum(bacc: float) -> str:
    if float(bacc) >= 0.80:
        return "high"
    if float(bacc) >= 0.65:
        return "medium"
    return "low"


def _label(row: Mapping[str, object]) -> int:
    return int(float(str(row.get("label", 0))))


def _to_numpy(value: object) -> object:
    import numpy as np  # type: ignore

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _hash_strings(values: Sequence[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return nanmean([_float(row[field]) for row in rows if field in row and str(row.get(field, "")) not in {"", "NA"}])


def _std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if len(finite) < 2:
        return 0.0
    avg = sum(finite) / float(len(finite))
    return math.sqrt(sum((value - avg) ** 2 for value in finite) / float(len(finite)))


def _float(value: object) -> float:
    if value in ("", NA, None):
        return math.nan
    return float(value)


def _format_float(value: object) -> str:
    number = _float(value)
    return "nan" if math.isnan(number) else f"{number:.4f}"


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
