from pathlib import Path
import json
import pickle

import numpy as np
import pytest
import yaml

from experiments.prior_sampling.posthoc_gmm_pca128 import (
    PCA128_POSTHOC_GMM_CLAIM_BOUNDARY,
    PCA128_POSTHOC_GMM_NAME,
    load_pca128_posthoc_gmm_config,
    parse_pca128_posthoc_gmm_config,
    run_pca128_posthoc_gmm_prior,
)
from cli_registry import load_config_for_validation
from downstream.evaluation import DownstreamResult, PredictionBundle
from latent.posterior_latents import build_posterior_latent_rows, split_fit_eval_latents
from models import ClassConditionedCVAE
from protocol import ProtocolError


def _valid_payload(tmp_path: Path) -> dict:
    return {
        "experiment": {
            "name": PCA128_POSTHOC_GMM_NAME,
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "inputs": {
            "posterior_latents_path": str(tmp_path / "posterior_latents.npz"),
            "frozen_checkpoint_path": str(tmp_path / "frozen_pca128_cvae.pt"),
            "target_eval_path": str(tmp_path / "target_eval_embeddings.npz"),
            "reference_metrics_path": str(tmp_path / "reference_metrics.json"),
        },
        "protocol": {
            "pca_dim": 128,
            "fit_split": "fit",
            "eval_split": "eval",
            "forbid_eval_encoding": True,
        },
        "prior": {
            "type": "class_conditional_gmm",
            "gmm_components": 8,
            "covariance_type": "diag",
            "reg_covar": 1.0e-6,
            "n_init": 3,
            "max_iter": 300,
            "min_class_fit_count": 16,
        },
        "generation": {
            "synthetic_per_class_total": 128,
            "generation_seed": 17,
        },
        "classifier": {
            "type": "sklearn_logistic_regression",
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 2000,
            "class_weight": "balanced",
            "classifier_seed": None,
        },
        "references": {
            "real_pca128_reference_key": "real_pca128_reference",
            "decode_mu_key": "decode_mu",
        },
    }


def test_fit_eval_latent_split_keeps_rows_disjoint() -> None:
    rows = build_posterior_latent_rows(
        latents=np.arange(12, dtype=np.float32).reshape(6, 2),
        labels=[0, 0, 1, 1, 0, 1],
        row_ids=["fit-0", "fit-1", "fit-2", "eval-0", "eval-1", "eval-2"],
        split_names=["fit", "fit", "fit", "eval", "eval", "eval"],
    )

    fit_rows, eval_rows = split_fit_eval_latents(rows)

    assert fit_rows.row_ids == ("fit-0", "fit-1", "fit-2")
    assert eval_rows.row_ids == ("eval-0", "eval-1", "eval-2")
    assert set(fit_rows.row_ids).isdisjoint(eval_rows.row_ids)


def test_pca128_config_validation_enforces_protocol_boundary(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    cfg = parse_pca128_posthoc_gmm_config(payload, base_dir=tmp_path)

    assert cfg.pca_dim == 128
    assert cfg.forbid_eval_encoding is True
    assert cfg.classifier_seed is None

    payload["protocol"]["forbid_eval_encoding"] = False
    with pytest.raises(ProtocolError, match="forbid_eval_encoding"):
        parse_pca128_posthoc_gmm_config(payload, base_dir=tmp_path)


def test_pca128_config_file_loads() -> None:
    cfg = load_pca128_posthoc_gmm_config("configs/prior_sampling/pca128_posthoc_gmm_prior_v1.yaml")

    assert cfg.name == PCA128_POSTHOC_GMM_NAME
    assert cfg.reference_real_key == "real_pca128_reference"
    assert cfg.reference_decode_mu_key == "decode_mu"
    assert "cvae_rebuild/cvae_rebuild" not in str(cfg.posterior_latents_path)
    assert cfg.posterior_latents_path.match("*/cvae_rebuild/artifacts/pca128_posthoc_gmm_prior_v1/inputs/posterior_latents_fit_eval.npz")


def test_pca128_config_rejects_eval_split_as_fit_split(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    payload["protocol"]["fit_split"] = "eval"

    with pytest.raises(ProtocolError, match="fit_split and eval_split"):
        parse_pca128_posthoc_gmm_config(payload, base_dir=tmp_path)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("protocol", "pca_dim", 64, "pca_dim"),
        ("prior", "type", "learned_prior", "prior.type"),
        ("prior", "gmm_components", 0, "GMM components"),
        ("generation", "synthetic_per_class_total", 256, "synthetic_per_class_total"),
        ("classifier", "solver", "liblinear", "Classifier solver"),
        ("classifier", "C", 0.5, "Classifier solver"),
        ("classifier", "max_iter", 100, "Classifier solver"),
        ("classifier", "class_weight", None, "Classifier must use class_weight"),
        ("classifier", "classifier_seed", 7, "Classifier must use class_weight"),
        ("references", "decode_mu_key", "posterior_decode_mu", "Reference comparison keys"),
    ],
)
def test_pca128_config_validation_locks_protocol_fields(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    match: str,
) -> None:
    payload = _valid_payload(tmp_path)
    payload[section][key] = value

    with pytest.raises(ProtocolError, match=match):
        parse_pca128_posthoc_gmm_config(payload, base_dir=tmp_path)


def test_pca128_validate_config_routes_by_experiment_name(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    config_path = tmp_path / "pca128_posthoc_gmm_prior_v1.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = load_config_for_validation(config_path)

    assert cfg.name == PCA128_POSTHOC_GMM_NAME


def test_pca128_posthoc_prior_fit_receives_fit_latents_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _valid_payload(tmp_path)
    payload["prior"]["min_class_fit_count"] = 2
    cfg = parse_pca128_posthoc_gmm_config(payload, base_dir=tmp_path)
    np.savez(
        cfg.posterior_latents_path,
        latents=np.array(
            [
                [-1.0, -1.0],
                [-1.2, -0.9],
                [1.0, 1.0],
                [1.2, 0.9],
                [-99.0, -99.0],
                [99.0, 99.0],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1, 0, 1], dtype=np.int64),
        row_ids=np.array(["fit0", "fit1", "fit2", "fit3", "eval0", "eval1"]),
        split_names=np.array(["fit", "fit", "fit", "fit", "eval", "eval"]),
    )
    cfg.reference_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reference_metrics_path.write_text(
        json.dumps({"real_pca128_reference": 0.64, "decode_mu": 0.64}),
        encoding="utf-8",
    )
    captured = {}

    class _Prior:
        prior_type = "class_conditional_gmm"
        classes = (0, 1)
        latent_dim = 2

    def _fit_spy(latents, labels, **kwargs):
        captured["latents"] = np.asarray(latents)
        captured["labels"] = np.asarray(labels)
        return _Prior()

    monkeypatch.setattr("experiments.prior_sampling.posthoc_gmm_pca128.fit_class_conditional_gmm_prior", _fit_spy)

    root = run_pca128_posthoc_gmm_prior(cfg)
    summary = json.loads((root / "reports" / "pca128_posthoc_gmm_prior_summary.json").read_text(encoding="utf-8"))

    assert summary["status"] == "READY_FOR_DECODING"
    assert captured["latents"].shape == (4, 2)
    assert not np.any(np.abs(captured["latents"]) == 99.0)
    assert captured["labels"].tolist() == [0, 0, 1, 1]


def test_pca128_posthoc_gmm_run_decodes_without_eval_latents_for_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_payload(tmp_path)
    payload["prior"]["gmm_components"] = 1
    payload["prior"]["min_class_fit_count"] = 2
    cfg = parse_pca128_posthoc_gmm_config(payload, base_dir=tmp_path)
    np.savez(
        cfg.posterior_latents_path,
        latents=np.array(
            [
                [-1.0, -1.0],
                [-1.2, -0.9],
                [1.0, 1.0],
                [1.2, 0.9],
                [-9.0, -9.0],
                [9.0, 9.0],
            ],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1, 0, 1], dtype=np.int64),
        row_ids=np.array(["fit0", "fit1", "fit2", "fit3", "eval0", "eval1"]),
        split_names=np.array(["fit", "fit", "fit", "fit", "eval", "eval"]),
    )
    cfg.reference_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.reference_metrics_path.write_text(
        json.dumps({"real_pca128_reference": 0.64, "decode_mu": 0.64}),
        encoding="utf-8",
    )
    cfg.target_eval_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.target_eval_path,
        embeddings=np.array([[-1.0] * 128, [1.0] * 128, [-0.8] * 128, [0.8] * 128], dtype=np.float32),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
        row_ids=np.array(["eval0", "eval1", "eval2", "eval3"]),
    )
    cfg.frozen_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.frozen_checkpoint_path.open("wb") as handle:
        pickle.dump(ClassConditionedCVAE(input_dim=128, hidden_dim=512, latent_dim=2, n_classes=2), handle)
    captured = {}

    def _classifier_spy(
        synthetic_embeddings,
        synthetic_labels,
        target_embeddings,
        *,
        classifier_seed,
        expert_id,
        class_weight=None,
    ):
        captured["synthetic_shape"] = np.asarray(synthetic_embeddings).shape
        captured["synthetic_labels"] = list(synthetic_labels)
        captured["target_shape"] = np.asarray(target_embeddings).shape
        captured["classifier_seed"] = classifier_seed
        captured["class_weight"] = class_weight
        return PredictionBundle(
            expert_id=expert_id,
            probabilities=((0.9, 0.1), (0.1, 0.9), (0.8, 0.2), (0.2, 0.8)),
            classes=(0, 1),
        )

    monkeypatch.setattr(
        "experiments.prior_sampling.posthoc_gmm_pca128.fit_locked_logistic_classifier",
        _classifier_spy,
    )
    monkeypatch.setattr(
        "experiments.prior_sampling.posthoc_gmm_pca128.evaluate_probability_predictions",
        lambda method, probabilities, target_labels, *, classes: DownstreamResult(
            method=method,
            bacc=1.0,
            macro_f1=1.0,
            n_target_eval=len(target_labels),
        ),
    )

    root = run_pca128_posthoc_gmm_prior(cfg)
    summary = json.loads((root / "reports" / "pca128_posthoc_gmm_prior_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "COMPLETE"
    assert summary["eval_rows_encoded_for_generation"] is False
    assert summary["generated_embedding_shape"] == [256, 128]
    assert summary["claim_boundary"] == PCA128_POSTHOC_GMM_CLAIM_BOUNDARY
    assert "downstream_bacc" in summary
    assert manifest["fit_posterior_latents_on_fit_rows_only"] is True
    assert manifest["claim_boundary"] == PCA128_POSTHOC_GMM_CLAIM_BOUNDARY
    assert manifest["latent_class_signal_warning"] == "LATENT_CLASS_SIGNAL_DOMINATES_CONDITION_WARNING"
    assert manifest["learned_prior_added"] is False
    assert summary["target_eval_row_ids_present"] is True
    assert summary["target_eval_fit_row_overlap"] == 0
    assert captured["synthetic_shape"] == (256, 128)
    assert captured["synthetic_labels"] == [0] * 128 + [1] * 128
    assert captured["target_shape"] == (4, 128)
    assert captured["classifier_seed"] is None
    assert captured["class_weight"] == "balanced"
