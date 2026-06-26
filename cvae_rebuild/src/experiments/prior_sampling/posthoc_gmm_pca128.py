from __future__ import annotations

import json
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from downstream.evaluation import evaluate_probability_predictions, fit_locked_logistic_classifier
from latent.posterior_latents import build_posterior_latent_rows, split_fit_eval_latents
from models import ClassConditionedCVAE
from preservation_repair import _load_mapping, _mapping, _path
from priors.gmm import fit_class_conditional_gaussian_prior, fit_class_conditional_gmm_prior
from protocol import ProtocolError, build_leakage_report
from reporting import prepare_artifact_dirs, write_json, write_protocol_finalization


PCA128_POSTHOC_GMM_NAME = "pca128_posthoc_gmm_prior_v1"
PCA128_POSTHOC_GMM_CLAIM_BOUNDARY = (
    "post-hoc latent-prior sampling utility feasibility only; because "
    "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING was emitted, success supports better latent-prior "
    "sampling utility, not proven controllable class-conditional generation; no learned-prior, routing, "
    "or formal privacy claim"
)


@dataclass(frozen=True)
class Pca128PosthocGmmConfig:
    name: str
    artifact_root: Path
    posterior_latents_path: Path
    frozen_checkpoint_path: Path
    target_eval_path: Path
    reference_metrics_path: Path
    pca_dim: int
    fit_split: str
    eval_split: str
    forbid_eval_encoding: bool
    prior_type: str
    gmm_components: int
    covariance_type: str
    reg_covar: float
    n_init: int
    max_iter: int
    min_class_fit_count: int
    synthetic_per_class_total: int
    generation_seed: int
    classifier_type: str
    classifier_solver: str
    classifier_c: float
    classifier_max_iter: int
    classifier_class_weight: str
    classifier_seed: int | None
    reference_real_key: str
    reference_decode_mu_key: str


def load_pca128_posthoc_gmm_config(path: str | Path) -> Pca128PosthocGmmConfig:
    source = Path(path).resolve()
    data = _load_mapping(source)
    base_dir = _repo_root_for_config(source)
    return parse_pca128_posthoc_gmm_config(data, base_dir=base_dir)


def parse_pca128_posthoc_gmm_config(
    data: Mapping[str, Any],
    *,
    base_dir: str | Path = ".",
) -> Pca128PosthocGmmConfig:
    base = Path(base_dir)
    experiment = _mapping(data, "experiment")
    inputs = _mapping(data, "inputs")
    protocol = _mapping(data, "protocol")
    prior = _mapping(data, "prior")
    generation = _mapping(data, "generation")
    classifier = _mapping(data, "classifier")
    references = _mapping(data, "references")
    cfg = Pca128PosthocGmmConfig(
        name=str(experiment["name"]),
        artifact_root=_path(base, str(experiment["artifact_root"])),
        posterior_latents_path=_path(base, str(inputs["posterior_latents_path"])),
        frozen_checkpoint_path=_path(base, str(inputs["frozen_checkpoint_path"])),
        target_eval_path=_path(base, str(inputs["target_eval_path"])),
        reference_metrics_path=_path(base, str(inputs["reference_metrics_path"])),
        pca_dim=int(protocol["pca_dim"]),
        fit_split=str(protocol["fit_split"]),
        eval_split=str(protocol["eval_split"]),
        forbid_eval_encoding=bool(protocol["forbid_eval_encoding"]),
        prior_type=str(prior["type"]),
        gmm_components=int(prior["gmm_components"]),
        covariance_type=str(prior["covariance_type"]),
        reg_covar=float(prior["reg_covar"]),
        n_init=int(prior["n_init"]),
        max_iter=int(prior["max_iter"]),
        min_class_fit_count=int(prior["min_class_fit_count"]),
        synthetic_per_class_total=int(generation["synthetic_per_class_total"]),
        generation_seed=int(generation["generation_seed"]),
        classifier_type=str(classifier["type"]),
        classifier_solver=str(classifier["solver"]),
        classifier_c=float(classifier["C"]),
        classifier_max_iter=int(classifier["max_iter"]),
        classifier_class_weight=str(classifier["class_weight"]),
        classifier_seed=None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
        reference_real_key=str(references["real_pca128_reference_key"]),
        reference_decode_mu_key=str(references["decode_mu_key"]),
    )
    validate_pca128_posthoc_gmm_config(cfg)
    return cfg


def validate_pca128_posthoc_gmm_config(cfg: Pca128PosthocGmmConfig) -> None:
    if cfg.name != PCA128_POSTHOC_GMM_NAME:
        raise ProtocolError(f"Experiment name must be {PCA128_POSTHOC_GMM_NAME!r}.")
    if cfg.pca_dim != 128:
        raise ProtocolError("pca_dim must be locked to 128.")
    if cfg.fit_split == cfg.eval_split:
        raise ProtocolError("fit_split and eval_split must be different.")
    if cfg.fit_split != "fit" or cfg.eval_split != "eval":
        raise ProtocolError("pca128 post-hoc prior audit must use fit/eval split names.")
    if not cfg.forbid_eval_encoding:
        raise ProtocolError("forbid_eval_encoding must be true.")
    if cfg.prior_type not in {"class_conditional_gmm", "class_conditional_gaussian"}:
        raise ProtocolError("prior.type must be class_conditional_gmm or class_conditional_gaussian.")
    if cfg.gmm_components < 1 or cfg.reg_covar <= 0.0 or cfg.n_init < 1 or cfg.max_iter < 1:
        raise ProtocolError("GMM components, regularization, n_init, and max_iter must be positive.")
    if cfg.min_class_fit_count < 2:
        raise ProtocolError("min_class_fit_count must be at least 2.")
    if cfg.synthetic_per_class_total != 128:
        raise ProtocolError("synthetic_per_class_total must be locked to 128.")
    if cfg.classifier_type != "sklearn_logistic_regression":
        raise ProtocolError("classifier.type must be sklearn_logistic_regression.")
    if cfg.classifier_solver != "lbfgs" or cfg.classifier_c != 1.0 or cfg.classifier_max_iter != 2000:
        raise ProtocolError("Classifier solver/C/max_iter must remain locked.")
    if cfg.classifier_class_weight != "balanced" or cfg.classifier_seed is not None:
        raise ProtocolError("Classifier must use class_weight=balanced and classifier_seed=null.")
    if cfg.reference_real_key != "real_pca128_reference" or cfg.reference_decode_mu_key != "decode_mu":
        raise ProtocolError("Reference comparison keys must be real_pca128_reference and decode_mu.")


def run_pca128_posthoc_gmm_prior(
    cfg: Pca128PosthocGmmConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    root = prepare_artifact_dirs(Path(artifact_root) if artifact_root is not None else cfg.artifact_root)
    protocol_violations: list[str] = []
    caught_exception: Exception | None = None
    try:
        rows = _load_posterior_latent_npz(cfg.posterior_latents_path)
        fit_rows, eval_rows = split_fit_eval_latents(rows, fit_split=cfg.fit_split, eval_split=cfg.eval_split)
        if cfg.prior_type == "class_conditional_gmm":
            prior = fit_class_conditional_gmm_prior(
                fit_rows.latents,
                fit_rows.labels,
                n_components=cfg.gmm_components,
                covariance_type=cfg.covariance_type,
                reg_covar=cfg.reg_covar,
                random_state=cfg.generation_seed,
                n_init=cfg.n_init,
                max_iter=cfg.max_iter,
                min_class_count=cfg.min_class_fit_count,
            )
        else:
            prior = fit_class_conditional_gaussian_prior(
                fit_rows.latents,
                fit_rows.labels,
                min_class_count=cfg.min_class_fit_count,
            )
        reference_metrics = _load_reference_metrics(cfg.reference_metrics_path)
        missing_execution_inputs = [
            str(path)
            for path in (cfg.frozen_checkpoint_path, cfg.target_eval_path)
            if not path.exists()
        ]
        summary = {
            "status": "READY_FOR_DECODING" if missing_execution_inputs else "COMPLETE",
            "prior_type": prior.prior_type,
            "classes": list(prior.classes),
            "latent_dim": prior.latent_dim,
            "fit_rows": int(fit_rows.latents.shape[0]),
            "eval_rows_held_out": int(eval_rows.latents.shape[0]),
            "eval_rows_encoded_for_generation": False,
            "reference_real_pca128_bacc": reference_metrics.get(cfg.reference_real_key),
            "reference_decode_mu_bacc": reference_metrics.get(cfg.reference_decode_mu_key),
            "missing_execution_inputs": missing_execution_inputs,
            "claim_boundary": PCA128_POSTHOC_GMM_CLAIM_BOUNDARY,
            "input_hashes": _existing_input_hashes(cfg),
        }
        if not missing_execution_inputs:
            synthetic_labels = [0] * cfg.synthetic_per_class_total + [1] * cfg.synthetic_per_class_total
            sampled_z = prior.sample(labels=synthetic_labels, random_state=cfg.generation_seed)
            model = _load_frozen_cvae_model(cfg.frozen_checkpoint_path)
            synthetic_embeddings = _decode_latent_samples(model, sampled_z, synthetic_labels)
            target_eval = _load_target_eval_npz(cfg.target_eval_path)
            _assert_target_eval_disjoint_from_fit(target_eval["row_ids"], fit_rows.row_ids)
            predictions = fit_locked_logistic_classifier(
                synthetic_embeddings,
                synthetic_labels,
                target_eval["embeddings"],
                classifier_seed=cfg.classifier_seed,
                expert_id="pca128_posthoc_gmm_prior",
                class_weight=cfg.classifier_class_weight,
            )
            result = evaluate_probability_predictions(
                "pca128_posthoc_gmm_prior",
                predictions.probabilities,
                target_eval["labels"],
                classes=predictions.classes,
            )
            summary.update(
                {
                    "generated_embedding_shape": list(synthetic_embeddings.shape),
                    "downstream_bacc": result.bacc,
                    "downstream_macro_f1": result.macro_f1,
                    "n_target_eval": result.n_target_eval,
                    "target_eval_row_ids_present": True,
                    "target_eval_fit_row_overlap": 0,
                }
            )
        write_json(root / "reports" / "pca128_posthoc_gmm_prior_summary.json", summary)
    except Exception as exc:
        protocol_violations.append(str(exc))
        caught_exception = exc
    leakage_report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
        extra_violations=tuple(protocol_violations),
    ).to_json_dict()
    protocol_manifest = {
            "schema_version": "pca128_posthoc_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "fit_posterior_latents_on_fit_rows_only": True,
            "eval_rows_encoded_for_generation": False,
            "decode_with_frozen_pca128_cvae": True,
            "downstream_classifier_settings_locked": True,
            "compares_against": [cfg.reference_real_key, cfg.reference_decode_mu_key],
            "claim_boundary": PCA128_POSTHOC_GMM_CLAIM_BOUNDARY,
            "latent_class_signal_warning": "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING",
            "learned_prior_added": False,
            "input_hashes": _existing_input_hashes(cfg),
            "protocol_violations": protocol_violations,
    }
    write_protocol_finalization(
        root,
        leakage_report=leakage_report,
        protocol_manifest=protocol_manifest,
        resolved_config={"experiment_name": cfg.name},
    )
    if caught_exception is not None:
        if isinstance(caught_exception, ProtocolError):
            raise caught_exception
        raise ProtocolError("pca128 post-hoc GMM prior audit failed; see leakage_report.json") from caught_exception
    return root


def _load_posterior_latent_npz(path: Path):
    import numpy as np

    payload = np.load(path, allow_pickle=False)
    return build_posterior_latent_rows(
        latents=payload["latents"],
        labels=payload["labels"],
        row_ids=[str(value) for value in payload["row_ids"]],
        split_names=[str(value) for value in payload["split_names"]],
    )


def _repo_root_for_config(source: Path) -> Path:
    for parent in source.parents:
        if parent.name == "cvae_rebuild":
            return parent.parent
    return source.parent


def _load_reference_metrics(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ProtocolError("Reference metrics must be a JSON object.")
    return payload


def _load_frozen_cvae_model(path: Path) -> ClassConditionedCVAE:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    model = getattr(payload, "model", payload)
    if isinstance(model, ClassConditionedCVAE):
        model.eval()
        return model
    if isinstance(payload, Mapping):
        state_dict = payload.get("model_state_dict") or payload.get("state_dict")
        if state_dict is None:
            raise ProtocolError("Frozen checkpoint mapping must contain model_state_dict or state_dict.")
        model_cfg = payload.get("model_config", payload)
        model = ClassConditionedCVAE(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            latent_dim=int(model_cfg["latent_dim"]),
            n_classes=int(model_cfg.get("n_classes", 2)),
            num_hidden_layers=int(model_cfg.get("num_hidden_layers", 2)),
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model
    raise ProtocolError("Frozen checkpoint must contain a ClassConditionedCVAE model.")


def _decode_latent_samples(
    model: ClassConditionedCVAE,
    latents: np.ndarray,
    labels: Sequence[int],
) -> np.ndarray:
    with torch.no_grad():
        z = torch.as_tensor(np.asarray(latents, dtype=np.float32))
        y = torch.as_tensor(np.asarray(labels, dtype=np.int64))
        decoded = model.decode(z, y).detach().cpu().numpy()
    return np.asarray(decoded, dtype=np.float32)


def _load_target_eval_npz(path: Path) -> Mapping[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if embeddings.ndim != 2 or labels.shape != (embeddings.shape[0],):
        raise ProtocolError("target_eval_path must contain 2D embeddings and matching labels.")
    if "row_ids" not in payload:
        raise ProtocolError("target_eval_path must include row_ids for fit/eval leakage checks.")
    row_ids = np.asarray([str(value) for value in payload["row_ids"]])
    if row_ids.shape != (embeddings.shape[0],):
        raise ProtocolError("target_eval_path row_ids must match target embedding row count.")
    if len(set(row_ids.tolist())) != len(row_ids):
        raise ProtocolError("target_eval_path row_ids must be unique.")
    return {"embeddings": embeddings, "labels": labels, "row_ids": row_ids}


def _assert_target_eval_disjoint_from_fit(target_eval_row_ids: Sequence[str], fit_row_ids: Sequence[str]) -> None:
    overlap = set(str(value) for value in target_eval_row_ids).intersection(str(value) for value in fit_row_ids)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ProtocolError(f"Target eval row_ids overlap prior-fit rows: {preview}")


def _existing_input_hashes(cfg: Pca128PosthocGmmConfig) -> dict[str, str]:
    paths = {
        "posterior_latents_path": cfg.posterior_latents_path,
        "frozen_checkpoint_path": cfg.frozen_checkpoint_path,
        "target_eval_path": cfg.target_eval_path,
        "reference_metrics_path": cfg.reference_metrics_path,
    }
    return {name: _sha256_file(path) for name, path in paths.items() if path.exists()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
