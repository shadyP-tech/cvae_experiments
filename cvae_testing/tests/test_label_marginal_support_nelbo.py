from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.load_config import load_config
from src.config.schema import validate_config
from src.eval.evaluators.label_marginal_support_nelbo import (
    evaluate_label_marginal_support_nelbo,
    label_marginal_nelbo_proxy,
)
from src.eval.metrics import spearman_corr
from src.models.cvae_expert import CVAEExpert
from src.train.checkpoint_provenance import wrap_model_state_dict


FAMILY_C_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "camelyon17"
    / "camelyon17_label_marginal_support_nelbo_v1.yaml"
)


def _family_c_cfg() -> dict:
    return yaml.safe_load(FAMILY_C_CONFIG.read_text(encoding="utf-8"))


def test_label_conditioned_cvae_requires_y_and_legacy_calls_remain_compatible() -> None:
    x = torch.randn(3, 5)
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    conditioned = CVAEExpert(input_dim=5, hidden_dim=7, latent_dim=2, class_condition_dim=2)
    with pytest.raises(ValueError, match="Class-condition tensor is required"):
        conditioned(x)

    recon, mu, logvar = conditioned(x, y=y)
    assert recon.shape == x.shape
    assert mu.shape == (3, 2)
    assert logvar.shape == (3, 2)

    unconditioned = CVAEExpert(input_dim=5, hidden_dim=7, latent_dim=2)
    recon_u, mu_u, logvar_u = unconditioned(x)
    assert recon_u.shape == x.shape
    assert mu_u.shape == (3, 2)
    assert logvar_u.shape == (3, 2)


def test_label_marginal_nelbo_proxy_matches_hand_logsumexp() -> None:
    class_nelbo = torch.tensor([[2.0, 4.0], [3.5, 1.0]], dtype=torch.float32)
    class_prior = torch.tensor([0.25, 0.75], dtype=torch.float32)

    expected = -torch.logsumexp(torch.log(class_prior).reshape(1, -1) - class_nelbo, dim=1)

    actual = label_marginal_nelbo_proxy(class_nelbo, class_prior)
    assert torch.allclose(actual, expected)


def test_metric_signs_are_positive_when_support_and_eval_rankings_match() -> None:
    support_scores = [1.0, 2.0, 3.0]
    eval_scores = [1.5, 2.5, 3.5]

    assert spearman_corr(support_scores, eval_scores) > 0.99
    assert spearman_corr([-v for v in support_scores], [-v for v in eval_scores]) > 0.99


def test_family_c_config_validates_protocol_locks() -> None:
    cfg = load_config(FAMILY_C_CONFIG)

    assert cfg["experiment"]["name"] == "camelyon17_label_marginal_support_nelbo_v1"
    assert cfg["model"]["label_conditioning"] == {
        "enabled": True,
        "label_field": "label",
        "label_values": [0, 1],
    }
    assert cfg["learned_utility"]["label_marginal_support_nelbo"]["primary_prior"] == "balanced"


def test_family_c_rejects_target_priors_true_label_routing_and_class_balanced_support() -> None:
    target_prior = _family_c_cfg()
    target_prior["learned_utility"]["label_marginal_support_nelbo"][
        "primary_prior"
    ] = "target_support_empirical"
    with pytest.raises(ValueError, match="forbids target empirical"):
        validate_config(target_prior)

    true_label_prior = _family_c_cfg()
    true_label_prior["learned_utility"]["label_marginal_support_nelbo"][
        "sensitivity_priors"
    ] = ["target_true_label_prior"]
    with pytest.raises(ValueError, match="forbids target empirical"):
        validate_config(true_label_prior)

    class_balanced = _family_c_cfg()
    class_balanced["learned_utility"]["label_marginal_support_nelbo"][
        "sampling_policies"
    ] = ["class_balanced"]
    with pytest.raises(ValueError, match="class-balanced support sampling is forbidden"):
        validate_config(class_balanced)


def test_family_c_rejects_non_binary_label_conditioning_and_metadata_conditioning() -> None:
    bad_labels = _family_c_cfg()
    bad_labels["model"]["label_conditioning"]["label_values"] = [0, 1, 2]
    with pytest.raises(ValueError, match=r"exactly \[0, 1\]"):
        validate_config(bad_labels)

    metadata_conditioned = _family_c_cfg()
    metadata_conditioned["model"]["conditioning"]["enabled"] = True
    with pytest.raises(ValueError, match="label conditioning separate"):
        validate_config(metadata_conditioned)

    metadata_constraint = _family_c_cfg()
    metadata_constraint["model"]["metadata_constraint"]["enabled"] = True
    with pytest.raises(ValueError, match="label conditioning separate"):
        validate_config(metadata_constraint)


def test_family_c_evaluator_marks_missing_family_a_imports_unavailable(tmp_path: Path) -> None:
    torch.manual_seed(7)
    input_dim = 5
    hidden_dim = 7
    latent_dim = 2
    domains = [0, 1, 2]

    train_metadata = [
        {"magnification": domain, "label": idx % 2}
        for domain in domains
        for idx in range(4)
    ]
    test_metadata = [
        {"magnification": domain, "label": idx % 2}
        for domain in domains
        for idx in range(5)
    ]
    train_cache = tmp_path / "train.pt"
    test_cache = tmp_path / "test.pt"
    torch.save({"embeddings": torch.randn(len(train_metadata), input_dim), "metadata": train_metadata}, train_cache)
    torch.save({"embeddings": torch.randn(len(test_metadata), input_dim), "metadata": test_metadata}, test_cache)

    expert_checkpoints: dict[str, str] = {}
    checkpoint_metadata = {
        "expert_family": "family_c_label_conditioned_v1",
        "condition_type": "class_label_one_hot",
        "label_values": [0, 1],
        "class_condition_dim": 2,
        "feature_extractor_name": "dinov2_vitb14",
        "feature_extractor_checkpoint": "facebook/dinov2-base",
        "feature_extractor_layer": "final_norm_cls",
        "embedding_pooling": "cls_token",
        "embedding_dim": input_dim,
        "beta_kl_weight": 1.0,
        "reconstruction_loss": "mse_sum",
        "likelihood_variance_assumption": "unit",
    }
    for domain in domains:
        model = CVAEExpert(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            class_condition_dim=2,
        )
        checkpoint = tmp_path / f"expert_{domain}x.pt"
        torch.save(wrap_model_state_dict(model.state_dict(), checkpoint_metadata), checkpoint)
        expert_checkpoints[f"{domain}x"] = str(checkpoint)

    summary = evaluate_label_marginal_support_nelbo(
        train_cache=train_cache,
        test_cache=test_cache,
        expert_checkpoints=expert_checkpoints,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        strategy="categorical_exact",
        tau=1.0,
        seed=42,
        learned_cfg={
            "label_marginal_support_nelbo": {
                "label_conditioning": {
                    "label_field": "label",
                    "label_values": [0, 1],
                },
                "primary_prior": "balanced",
                "sensitivity_priors": ["source_global_laplace"],
                "laplace_alpha": 1.0,
                "support_sizes": [2],
                "support_seeds": [17],
                "sampling_policies": ["random"],
            }
        },
        reports_dir=tmp_path / "reports",
        batch_size=4,
        family_a_selection_path=tmp_path / "missing_family_a.csv",
    )

    assert summary["status"] == "DIAGNOSTIC_ONLY"
    assert "family_a_direct_support_nelbo_selection" in summary["metrics_by_method"]
    assert summary["metrics_by_method"]["family_a_direct_support_nelbo_selection"]["n_rows"] == 0.0

    import_rows_path = tmp_path / "reports" / "family_c_imported_selection_baselines.csv"
    with import_rows_path.open("r", encoding="utf-8") as f:
        imported_rows = list(csv.DictReader(f))
    assert imported_rows
    assert {row["available"] for row in imported_rows} == {"0"}

    audit_path = tmp_path / "reports" / "label_marginal_protocol_audit.csv"
    with audit_path.open("r", encoding="utf-8") as f:
        audit_rows = list(csv.DictReader(f))
    assert audit_rows
    assert {row["support_labels_used_for_routing"] for row in audit_rows} == {"0"}
    assert {row["routing_uses_eval_score"] for row in audit_rows} == {"0"}
    assert {row["target_expert_excluded"] for row in audit_rows} == {"1"}
    assert {row["support_eval_disjoint"] for row in audit_rows} == {"1"}
