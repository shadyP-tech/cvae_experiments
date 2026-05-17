from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import build_checkpoint_metadata_from_cache
from src.train.checkpoint_provenance import load_model_checkpoint
from src.train.train_utils import run_training


def _indices_by_domain(payload: Dict[str, object], domain: int) -> list[int]:
    metadata = payload["metadata"]
    idxs = [i for i, m in enumerate(metadata) if int(m["magnification"]) == domain]
    return idxs


def _filter_by_domain(payload: Dict[str, object], domain: int):
    embeddings = payload["embeddings"]
    idxs = _indices_by_domain(payload, domain)
    if not idxs:
        return torch.empty((0, embeddings.shape[1]))
    return embeddings[idxs]


def build_label_one_hot(metadata: Sequence[dict], label_values: Sequence[int], label_field: str = "label") -> torch.Tensor:
    label_order = [int(v) for v in label_values]
    if not label_order:
        raise ValueError("label_values must be non-empty for label conditioning")
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    targets: list[int] = []
    for i, item in enumerate(metadata):
        if label_field not in item:
            raise ValueError(f"Missing label field '{label_field}' in metadata item at index {i}.")
        label = int(item[label_field])
        if label not in label_to_index:
            raise ValueError(
                f"Observed label '{label}' at index {i} is not present in configured label_values: {label_order}"
            )
        targets.append(label_to_index[label])
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return F.one_hot(target_tensor, num_classes=len(label_order)).to(dtype=torch.float32)


def train_domain_experts(
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    domains: list[int],
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    resume_from_dir: Path | None = None,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    label_conditioning_cfg: Dict[str, Any] | None = None,
    label_utility_cfg: Dict[str, Any] | None = None,
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")
    input_dim = int(train_payload["embeddings"].shape[1])
    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))
    metadata_dim = 0
    train_meta_all = None
    val_meta_all = None
    if conditioning_enabled:
        domain_order = resolve_domain_order(configured_domains or domains)
        train_meta_all = build_domain_one_hot(train_payload["metadata"], domain_order)
        val_meta_all = build_domain_one_hot(val_payload["metadata"], domain_order)
        metadata_dim = int(len(domain_order))
    label_cfg = label_conditioning_cfg or {}
    label_conditioning_enabled = bool(label_cfg.get("enabled", False))
    label_values = [int(v) for v in label_cfg.get("label_values", [])]
    label_field = str(label_cfg.get("label_field", "label"))
    class_condition_dim = 0
    train_y_all = None
    val_y_all = None
    if label_conditioning_enabled:
        if not label_values:
            raise ValueError("model.label_conditioning.label_values must be non-empty when enabled")
        train_y_all = build_label_one_hot(train_payload["metadata"], label_values, label_field=label_field)
        val_y_all = build_label_one_hot(val_payload["metadata"], label_values, label_field=label_field)
        class_condition_dim = int(len(label_values))
    family_d_cfg = label_utility_cfg or {}
    family_d_enabled = bool(family_d_cfg.get("enabled", False))
    if family_d_enabled and not label_conditioning_enabled:
        raise ValueError("Family D discriminative label-utility training requires model.label_conditioning.enabled=true")

    output: Dict[str, str] = {}
    provenance_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    for domain in domains:
        train_idxs = _indices_by_domain(train_payload, domain)
        val_idxs = _indices_by_domain(val_payload, domain)
        train_x = train_payload["embeddings"][train_idxs] if train_idxs else torch.empty((0, input_dim))
        val_x = val_payload["embeddings"][val_idxs] if val_idxs else torch.empty((0, input_dim))
        if train_x.numel() == 0 or val_x.numel() == 0:
            continue

        train_m = train_meta_all[train_idxs] if (conditioning_enabled and train_meta_all is not None) else None
        val_m = val_meta_all[val_idxs] if (conditioning_enabled and val_meta_all is not None) else None
        train_y = train_y_all[train_idxs] if (label_conditioning_enabled and train_y_all is not None) else None
        val_y = val_y_all[val_idxs] if (label_conditioning_enabled and val_y_all is not None) else None
        checkpoint_extra = {"model_name": f"expert_{domain}x", "expert_domain": int(domain)}
        if label_conditioning_enabled:
            checkpoint_extra.update(
                {
                    "expert_family": "family_c_label_conditioned_v1",
                    "condition_type": "class_label_one_hot",
                    "label_field": label_field,
                    "label_values": label_values,
                    "class_condition_dim": int(class_condition_dim),
                    "beta_kl_weight": 1.0,
                    "reconstruction_loss": "mse_sum",
                    "likelihood_variance_assumption": "unit",
                }
            )
        if family_d_enabled:
            checkpoint_extra.update(
                {
                    "expert_family": str(
                        family_d_cfg.get("expert_family", "family_d_discriminative_label_conditioned_v1")
                    ),
                    "condition_type": "class_label_one_hot",
                    "discriminative_training_enabled": 1,
                    "lambda_latent_cls": float(family_d_cfg.get("lambda_latent_cls", 0.0)),
                    "lambda_recon_cls": float(family_d_cfg.get("lambda_recon_cls", 0.0)),
                    "lambda_prior_cls": float(family_d_cfg.get("lambda_prior_cls", 0.0)),
                    "prior_samples_per_batch": str(family_d_cfg.get("prior_samples_per_batch", "same_batch_size")),
                    "early_stopping_metric": str(family_d_cfg.get("early_stopping_metric", "source_val_total_loss")),
                    "beta_kl_weight": 1.0,
                    "reconstruction_loss": "mse_sum",
                    "likelihood_variance_assumption": "unit",
                }
            )

        result = run_training(
            train_embeddings=train_x,
            val_embeddings=val_x,
            out_dir=out_dir,
            model_name=f"expert_{domain}x",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            lr=lr,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            resume_from=(resume_from_dir / f"expert_{domain}x.pt") if resume_from_dir is not None else None,
            train_metadata_vectors=train_m,
            val_metadata_vectors=val_m,
            train_class_condition_vectors=train_y,
            val_class_condition_vectors=val_y,
            metadata_dim=metadata_dim,
            class_condition_dim=class_condition_dim,
            metadata_constraint_cfg=metadata_constraint_cfg,
            label_utility_cfg=family_d_cfg if family_d_enabled else None,
            checkpoint_metadata=checkpoint_metadata
            or build_checkpoint_metadata_from_cache(
                train_payload,
                extra=checkpoint_extra,
            ),
        )
        output[f"{domain}x"] = str(result.checkpoint_path)
        if family_d_enabled:
            provenance_rows.append(
                _family_d_provenance_row(
                    domain=domain,
                    checkpoint_path=result.checkpoint_path,
                    checkpoint_extra=checkpoint_extra,
                    train_payload=train_payload,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    label_values=label_values,
                )
            )
            history_rows.extend(_family_d_history_rows(domain=domain, history=result.history))
            audit_rows.append(
                {
                    "expert_domain": int(domain),
                    "expert_family": checkpoint_extra["expert_family"],
                    "expert_training_split": "source_train",
                    "expert_validation_split": "source_val",
                    "target_labels_used_for_training": 0,
                    "target_eval_labels_used_for_training": 0,
                    "target_oracle_used_for_selection": 0,
                    "early_stopping_metric": checkpoint_extra["early_stopping_metric"],
                    "source_train_domain_only": int(all(int(train_payload["metadata"][i]["magnification"]) == int(domain) for i in train_idxs)),
                    "source_val_domain_only": int(all(int(val_payload["metadata"][i]["magnification"]) == int(domain) for i in val_idxs)),
                    "available": 1,
                }
            )
            diagnostics_rows.append(
                _family_d_source_val_diagnostics_row(
                    domain=domain,
                    checkpoint_path=result.checkpoint_path,
                    train_x=train_x,
                    val_x=val_x,
                    train_y=train_y,
                    val_y=val_y,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim,
                    class_condition_dim=class_condition_dim,
                    label_utility_cfg=family_d_cfg,
                    history=result.history,
                    label_values=label_values,
                )
            )

    with (out_dir / "expert_checkpoints.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    if family_d_enabled:
        reports_dir = out_dir.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(reports_dir / "family_d_checkpoint_provenance.csv", provenance_rows)
        _write_csv(reports_dir / "family_d_training_history.csv", history_rows)
        _write_csv(reports_dir / "family_d_training_protocol_audit.csv", audit_rows)
        _write_csv(reports_dir / "family_d_source_val_diagnostics.csv", diagnostics_rows)
    return output


def _family_d_provenance_row(
    *,
    domain: int,
    checkpoint_path: Path,
    checkpoint_extra: Mapping[str, object],
    train_payload: Dict[str, object],
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    label_values: Sequence[int],
) -> dict[str, object]:
    return {
        "expert_domain": int(domain),
        "checkpoint_path": str(checkpoint_path),
        "model_name": f"expert_{domain}x",
        "expert_family": checkpoint_extra.get("expert_family", ""),
        "condition_type": checkpoint_extra.get("condition_type", ""),
        "discriminative_training_enabled": int(checkpoint_extra.get("discriminative_training_enabled", 0)),
        "label_field": checkpoint_extra.get("label_field", "label"),
        "label_values_json": json.dumps([int(v) for v in label_values]),
        "class_condition_dim": int(len(label_values)),
        "embedding_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "feature_extractor_name": _payload_value(train_payload, "feature_extractor_name", "dinov2_vitb14"),
        "feature_extractor_checkpoint": _payload_value(train_payload, "feature_extractor_checkpoint", "facebook/dinov2-base"),
        "beta_kl_weight": float(checkpoint_extra.get("beta_kl_weight", 1.0)),
        "reconstruction_loss": checkpoint_extra.get("reconstruction_loss", "mse_sum"),
        "likelihood_variance_assumption": checkpoint_extra.get("likelihood_variance_assumption", "unit"),
        "lambda_latent_cls": float(checkpoint_extra.get("lambda_latent_cls", 0.0)),
        "lambda_recon_cls": float(checkpoint_extra.get("lambda_recon_cls", 0.0)),
        "lambda_prior_cls": float(checkpoint_extra.get("lambda_prior_cls", 0.0)),
        "prior_samples_per_batch": checkpoint_extra.get("prior_samples_per_batch", "same_batch_size"),
        "early_stopping_metric": checkpoint_extra.get("early_stopping_metric", "source_val_total_loss"),
    }


def _payload_value(payload: Dict[str, object], key: str, default: object) -> object:
    if key in payload:
        return payload[key]
    feature_cfg = payload.get("feature_extractor")
    if isinstance(feature_cfg, dict) and key in feature_cfg:
        return feature_cfg[key]
    return default


def _family_d_history_rows(*, domain: int, history: Dict[str, list[float]]) -> list[dict[str, object]]:
    n_epochs = len(history.get("train", []))
    rows: list[dict[str, object]] = []
    for epoch in range(n_epochs):
        row: dict[str, object] = {"expert_domain": int(domain), "epoch": int(epoch)}
        for key, values in history.items():
            if len(values) == n_epochs:
                row[key] = float(values[epoch])
        rows.append(row)
    return rows


def _family_d_source_val_diagnostics_row(
    *,
    domain: int,
    checkpoint_path: Path,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_y: torch.Tensor | None,
    val_y: torch.Tensor | None,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    class_condition_dim: int,
    label_utility_cfg: Dict[str, Any],
    history: Dict[str, list[float]],
    label_values: Sequence[int],
) -> dict[str, object]:
    if train_y is None or val_y is None:
        raise ValueError("Family D diagnostics require class-condition vectors")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        class_condition_dim=class_condition_dim,
        label_utility_cfg=label_utility_cfg,
    ).to(device)
    model.load_state_dict(load_model_checkpoint(checkpoint_path, map_location=device).model_state_dict)
    model.eval()
    with torch.no_grad():
        x_val = val_x.to(device)
        y_val = val_y.to(device)
        recon, mu, logvar = model(x_val, y=y_val)
        recon_terms, kl_terms = elbo_components(recon, x_val, mu, logvar)
        latent_logits = model.label_utility_latent_logits(mu)
        recon_logits = model.label_utility_decoded_logits(recon)
        z_prior = torch.randn((int(y_val.shape[0]), int(latent_dim)), dtype=x_val.dtype, device=device)
        prior_decoded = model.decode(z_prior, y=y_val)
        prior_logits = model.label_utility_decoded_logits(prior_decoded)

    generated_x, generated_y = _generate_family_d_prior_samples(
        model=model,
        label_values=label_values,
        class_condition_dim=class_condition_dim,
        latent_dim=latent_dim,
        budget_per_class=128,
        device=device,
        seed=173,
    )
    val_labels = val_y.argmax(dim=1).detach().cpu()
    train_labels = train_y.argmax(dim=1).detach().cpu()
    external = _external_generated_utility(
        synthetic_x=generated_x,
        synthetic_y=generated_y,
        train_x=train_x.detach().cpu(),
        train_y=train_labels,
        val_x=val_x.detach().cpu(),
        val_y=val_labels,
    )
    real_source_norms = torch.linalg.norm(val_x.detach().cpu(), dim=1)
    generated_norms = torch.linalg.norm(generated_x, dim=1)
    relative = _relative_family_c_nelbo(domain, float(recon_terms.mean().item() + kl_terms.mean().item()), label_utility_cfg)
    return {
        "expert_domain": int(domain),
        "source_val_total_loss": _last(history, "val_total_loss"),
        "source_val_nelbo": float((recon_terms + kl_terms).mean().item()),
        "source_val_reconstruction_mse": float(recon_terms.mean().item()),
        "source_val_kl": float(kl_terms.mean().item()),
        "source_val_latent_cls_acc": float(model.label_utility_accuracy(latent_logits, y_val).item()),
        "source_val_recon_cls_acc": float(model.label_utility_accuracy(recon_logits, y_val).item()),
        "source_val_internal_prior_head_acc": float(model.label_utility_accuracy(prior_logits, y_val).item()),
        "source_val_nelbo_relative_worsening_vs_family_c": relative,
        "best_epoch": _last(history, "best_epoch"),
        "best_source_val_total_loss": _last(history, "best_source_val_total_loss"),
        "source_val_nelbo_at_best_epoch": _last(history, "source_val_nelbo_at_best_epoch"),
        "source_val_nelbo_best_epoch": _last(history, "source_val_nelbo_best_epoch"),
        "source_val_total_loss_at_nelbo_best_epoch": _last(history, "source_val_total_loss_at_nelbo_best_epoch"),
        **external,
        "generated_norm_mean": float(generated_norms.mean().item()) if generated_norms.numel() else math.nan,
        "generated_norm_std": float(generated_norms.std(unbiased=False).item()) if generated_norms.numel() else math.nan,
        "real_source_norm_mean": float(real_source_norms.mean().item()) if real_source_norms.numel() else math.nan,
        "real_source_norm_std": float(real_source_norms.std(unbiased=False).item()) if real_source_norms.numel() else math.nan,
        "generated_nan_count": int(torch.isnan(generated_x).sum().item()),
        "generated_inf_count": int(torch.isinf(generated_x).sum().item()),
    }


def _generate_family_d_prior_samples(
    *,
    model: CVAEExpert,
    label_values: Sequence[int],
    class_condition_dim: int,
    latent_dim: int,
    budget_per_class: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for label in label_values:
        y = torch.zeros((int(budget_per_class), int(class_condition_dim)), dtype=torch.float32, device=device)
        y[:, int(label)] = 1.0
        z = torch.randn((int(budget_per_class), int(latent_dim)), generator=generator, dtype=torch.float32).to(device)
        with torch.no_grad():
            decoded = model.decode(z, y=y)
        chunks.append(decoded.detach().cpu())
        labels.append(torch.full((int(budget_per_class),), int(label), dtype=torch.long))
    return torch.cat(chunks, dim=0), torch.cat(labels, dim=0)


def _external_generated_utility(
    *,
    synthetic_x: torch.Tensor,
    synthetic_y: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import balanced_accuracy_score, f1_score  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError:
        return {
            "synthetic_train_to_real_source_val_bacc": math.nan,
            "synthetic_train_to_real_source_val_macro_f1": math.nan,
            "real_source_val_accuracy_from_real_train": math.nan,
            "synthetic_to_real_gap": math.nan,
            "generated_class_centroid_distance": math.nan,
            "real_class_centroid_distance": math.nan,
            "centroid_distance_ratio": math.nan,
        }

    x_syn = synthetic_x.detach().cpu().numpy()
    y_syn = synthetic_y.detach().cpu().numpy()
    x_train = train_x.detach().cpu().numpy()
    y_train = train_y.detach().cpu().numpy()
    x_val = val_x.detach().cpu().numpy()
    y_val = val_y.detach().cpu().numpy()

    def fit_eval(x_fit, y_fit) -> tuple[float, float]:
        scaler = StandardScaler()
        x_fit_scaled = scaler.fit_transform(x_fit)
        x_val_scaled = scaler.transform(x_val)
        clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, class_weight=None)
        clf.fit(x_fit_scaled, y_fit)
        pred = clf.predict(x_val_scaled)
        return float(balanced_accuracy_score(y_val, pred)), float(f1_score(y_val, pred, average="macro"))

    syn_bacc, syn_f1 = fit_eval(x_syn, y_syn)
    real_bacc, _ = fit_eval(x_train, y_train)
    generated_centroid_distance = _centroid_distance(x_syn, y_syn)
    real_centroid_distance = _centroid_distance(x_val, y_val)
    return {
        "synthetic_train_to_real_source_val_bacc": syn_bacc,
        "synthetic_train_to_real_source_val_macro_f1": syn_f1,
        "real_source_val_accuracy_from_real_train": real_bacc,
        "synthetic_to_real_gap": real_bacc - syn_bacc,
        "generated_class_centroid_distance": generated_centroid_distance,
        "real_class_centroid_distance": real_centroid_distance,
        "centroid_distance_ratio": (
            generated_centroid_distance / real_centroid_distance
            if real_centroid_distance and not math.isnan(real_centroid_distance)
            else math.nan
        ),
    }


def _centroid_distance(x: object, y: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=int)
    if not np.any(labels == 0) or not np.any(labels == 1):
        return math.nan
    return float(np.linalg.norm(arr[labels == 0].mean(axis=0) - arr[labels == 1].mean(axis=0)))


def _relative_family_c_nelbo(domain: int, nelbo: float, cfg: Dict[str, Any]) -> float:
    raw = cfg.get("family_c_reference_source_val_nelbo_by_expert", {}) or {}
    if not isinstance(raw, dict):
        return math.nan
    ref = raw.get(str(domain), raw.get(int(domain)))
    if ref is None:
        return math.nan
    ref_float = float(ref)
    if ref_float == 0.0:
        return math.nan
    return (float(nelbo) - ref_float) / abs(ref_float)


def _last(history: Dict[str, list[float]], key: str) -> float:
    values = history.get(key, [])
    return float(values[-1]) if values else math.nan


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
