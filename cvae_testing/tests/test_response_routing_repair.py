from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_compatibility_loqdo_decision_table import _aggregate_methods
from scripts.build_response_routing_artifact_suite import CSV_SCHEMAS, build_artifact_suite
from src.eval.evaluators.response_indirect import compute_response_features
from src.eval.feature_regimes import (
    FEATURE_REGISTRY,
    build_feature_matrix,
    feature_schema_hash,
    get_feature_regime,
    shuffle_response_feature_rows,
)
from src.models.cvae_expert import CVAEExpert
from src.train.checkpoint_provenance import load_model_checkpoint, wrap_model_state_dict


def _rows() -> list[dict]:
    return [
        {
            "query_id": 1,
            "expert_id": 10,
            "query_domain": 40,
            "expert_domain": 100,
            "metadata_distance": 0.1,
            "embedding_distance": 0.2,
            "query_domain_value": 0.0,
            "expert_domain_value": 0.2,
            "abs_domain_diff": 0.2,
            "is_exact_domain_match": 0.0,
            "oracle_utility": 0.5,
            "oracle_nelbo": 1.0,
            "response_posterior_mu_mean": 0.1,
            "response_posterior_mu_std": 0.2,
            "response_recon_repeat_variance_q75": 0.3,
            "response_decode_repeat_variance_max": 0.4,
            "response_nelbo_mean": 9.0,
            "response_recon_mean": 8.0,
            "response_kl_mean": 7.0,
            "response_feature_stream_name": "response_feature",
            "response_target_stream_name": "target_oracle",
            "response_feature_seed": 12345,
        },
        {
            "query_id": 2,
            "expert_id": 20,
            "query_domain": 40,
            "expert_domain": 200,
            "metadata_distance": 0.4,
            "embedding_distance": 0.5,
            "query_domain_value": 0.0,
            "expert_domain_value": 0.5,
            "abs_domain_diff": 0.5,
            "is_exact_domain_match": 0.0,
            "oracle_utility": 0.7,
            "oracle_nelbo": 0.8,
            "response_posterior_mu_mean": 0.4,
            "response_posterior_mu_std": 0.5,
            "response_recon_repeat_variance_q75": 0.6,
            "response_decode_repeat_variance_max": 0.7,
            "response_nelbo_mean": 9.1,
            "response_recon_mean": 8.1,
            "response_kl_mean": 7.1,
            "response_feature_stream_name": "response_feature",
            "response_target_stream_name": "target_oracle",
            "response_feature_seed": 67890,
        },
    ]


def test_feature_regime_registry_statuses() -> None:
    expected = {
        "static_metadata": (1, 0, 0),
        "static_embedding": (1, 0, 0),
        "static_combined": (1, 0, 0),
        "response_indirect": (1, 0, 0),
        "static_response_indirect": (1, 0, 0),
        "response_indirect_shuffled": (0, 0, 1),
        "response_target_adjacent_diagnostic": (0, 1, 0),
        "response_oracle_diagnostic": (0, 1, 0),
    }
    assert set(FEATURE_REGISTRY) == set(expected)
    for name, statuses in expected.items():
        regime = get_feature_regime(name)
        assert (
            int(regime.adoption_eligible),
            int(regime.diagnostic_only),
            int(regime.control_only),
        ) == statuses
    with pytest.raises(ValueError):
        get_feature_regime("unknown")


def test_response_indirect_blocks_utility_and_identity_terms_but_allows_variance_features() -> None:
    regime = get_feature_regime("response_indirect")
    result = build_feature_matrix(_rows(), regime=regime, expert_domains=[100, 200])
    assert "response_recon_repeat_variance_q75" in result.feature_names
    assert "response_decode_repeat_variance_max" in result.feature_names
    assert "response_nelbo_mean" not in result.feature_names
    assert "response_recon_mean" not in result.feature_names
    assert "response_kl_mean" not in result.feature_names
    assert "response_feature_stream_name" not in result.feature_names
    assert "response_target_stream_name" not in result.feature_names
    assert "response_feature_seed" not in result.feature_names
    assert set(result.blocked_feature_terms) >= {"nelbo", "recon_mean", "kl_mean"}

    forced = build_feature_matrix(
        _rows(),
        regime=regime,
        expert_domains=[100, 200],
        feature_names=["query_id", "expert_id", "oracle_utility"],
    )
    assert forced.feature_names == []
    assert set(forced.blocked_feature_terms) >= {"query_id", "expert_id", "oracle_"}


def test_feature_audit_zero_variance_and_hash_order() -> None:
    regime = get_feature_regime("response_indirect")
    rows = _rows()
    for row in rows:
        row["response_posterior_constant"] = 1.0
    first = build_feature_matrix(rows, regime=regime, expert_domains=[100, 200])
    second = build_feature_matrix(rows, regime=regime, expert_domains=[100, 200])
    assert first.feature_names == second.feature_names
    assert first.feature_schema_hash == second.feature_schema_hash
    assert "response_posterior_constant" in first.dropped_zero_variance
    assert feature_schema_hash(regime.name, ["response_a", "response_b"]) != feature_schema_hash(
        regime.name,
        ["response_b", "response_a"],
    )


def test_fold_local_shuffled_control_is_deterministic_and_preserves_multiset() -> None:
    rows = [
        {"row": i, "response_posterior_a": float(i), "response_posterior_b": float(i + 10)}
        for i in range(5)
    ]
    shuffled_a = shuffle_response_feature_rows(
        rows,
        dataset="breakhis",
        seed=42,
        fold_id="fold-1",
        split_id="train",
        regime_name="response_indirect_shuffled",
    )
    shuffled_b = shuffle_response_feature_rows(
        rows,
        dataset="breakhis",
        seed=42,
        fold_id="fold-1",
        split_id="train",
        regime_name="response_indirect_shuffled",
    )
    shuffled_c = shuffle_response_feature_rows(
        rows,
        dataset="breakhis",
        seed=42,
        fold_id="fold-2",
        split_id="train",
        regime_name="response_indirect_shuffled",
    )
    assert shuffled_a == shuffled_b
    assert shuffled_a != rows
    assert shuffled_a != shuffled_c
    assert sorted((r["response_posterior_a"], r["response_posterior_b"]) for r in shuffled_a) == sorted(
        (r["response_posterior_a"], r["response_posterior_b"]) for r in rows
    )
    assert [r["row"] for r in shuffled_a] == [r["row"] for r in rows]


def _raw_decision_rows(method: str, regime: str, *, perfect: bool = False) -> list[dict]:
    base = {
        "dataset_name": "breakhis",
        "backbone_type": "dinov2_vitb14",
        "run_id": "run1",
        "variant": "B",
        "heldout_query_domain": "40",
        "feature_set": "static_metadata",
        "feature_regime": "static_metadata",
        "method": "metadata_routing",
        "probe_feature_mode": "off",
        "interaction_feature_mode": "off",
        "disentanglement_arm": "baseline",
        "top1_agreement_with_best_expert": "0",
        "spearman_similarity_vs_neg_nelbo": "0",
        "metadata_to_oracle_gap": "1.0",
        "normalized_metadata_to_oracle_gap": "1.0",
        "calibration_error_bin10": "1.0",
        "top1_margin": "0",
    }
    cand = dict(base)
    cand.update(
        {
            "feature_regime": regime,
            "feature_set": regime,
            "method": method,
            "disentanglement_arm": regime,
            "top1_agreement_with_best_expert": "1" if perfect else "0.5",
            "spearman_similarity_vs_neg_nelbo": "1" if perfect else "0.5",
            "metadata_to_oracle_gap": "0",
            "normalized_metadata_to_oracle_gap": "0",
            "calibration_error_bin10": "0",
            "feature_names": "oracle_utility" if "diagnostic" in regime else "response_posterior_mu_mean",
        }
    )
    return [base, cand]


def test_decision_vetoes_diagnostic_and_control_even_with_perfect_metrics() -> None:
    rows, _ = _aggregate_methods(
        _raw_decision_rows("oracle_eval_mean_cheat", "response_oracle_diagnostic", perfect=True),
        uplift_reference_method="metadata_routing",
        min_improving_runs=1,
        strong={"spearman_uplift_min": 0, "top1_uplift_min": 0, "oracle_gap_reduction_min": 0, "normalized_oracle_gap_reduction_min": 0},
        weak={"spearman_uplift_min": 0, "top1_uplift_min": 0, "oracle_gap_reduction_min": 0, "normalized_oracle_gap_reduction_min": 0},
        instability_std_threshold=999,
        instability_sign_inconsistency_min_count=99,
        max_calibration_error_mean=999,
        calibration_reduction_min=0,
    )
    diagnostic = [r for r in rows if "response_oracle_diagnostic" in r["method_key"]][0]
    assert int(diagnostic["adoption_gate_pass_proxy"]) == 0
    assert "diagnostic_or_target_derived_features" in diagnostic["veto_reason"]

    rows, _ = _aggregate_methods(
        _raw_decision_rows("linear_regression", "response_indirect_shuffled", perfect=True),
        uplift_reference_method="metadata_routing",
        min_improving_runs=1,
        strong={"spearman_uplift_min": 0, "top1_uplift_min": 0, "oracle_gap_reduction_min": 0, "normalized_oracle_gap_reduction_min": 0},
        weak={"spearman_uplift_min": 0, "top1_uplift_min": 0, "oracle_gap_reduction_min": 0, "normalized_oracle_gap_reduction_min": 0},
        instability_std_threshold=999,
        instability_sign_inconsistency_min_count=99,
        max_calibration_error_mean=999,
        calibration_reduction_min=0,
    )
    control = [r for r in rows if "response_indirect_shuffled" in r["method_key"]][0]
    assert int(control["adoption_gate_pass_proxy"]) == 0
    assert "control_only" in control["veto_reason"]


def test_empty_safe_artifact_suite(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    manifest = build_artifact_suite(out, scope="development", input_csv=tmp_path / "missing.csv")
    for filename in list(CSV_SCHEMAS) + ["decision_summary.json", "failure_mode_summary.md", "artifact_manifest.json"]:
        assert (out / filename).exists()
    for filename, headers in CSV_SCHEMAS.items():
        with (out / filename).open("r", encoding="utf-8", newline="") as f:
            assert next(csv.reader(f)) == headers
    assert json.loads((out / "decision_summary.json").read_text())["status"] == "no_data"
    assert "no benchmark rows available" in (out / "failure_mode_summary.md").read_text()
    assert "artifact_manifest.json" in manifest["created_files"]


def test_checkpoint_loader_accepts_legacy_and_wrapped_and_rejects_malformed(tmp_path: Path) -> None:
    state = {"weight": torch.ones(1)}
    raw_path = tmp_path / "raw.pt"
    wrapped_path = tmp_path / "wrapped.pt"
    malformed_path = tmp_path / "bad.pt"
    torch.save(state, raw_path)
    torch.save(wrap_model_state_dict(state, {"feature_extractor_name": "dinov2_vitb14"}), wrapped_path)
    torch.save({"checkpoint_metadata": {}}, malformed_path)

    raw = load_model_checkpoint(raw_path)
    wrapped = load_model_checkpoint(wrapped_path)
    assert raw.legacy_format is True
    assert wrapped.legacy_format is False
    assert wrapped.checkpoint_metadata["feature_extractor_name"] == "dinov2_vitb14"
    with pytest.raises(ValueError):
        load_model_checkpoint(malformed_path)


class _FakeBank:
    def __init__(self) -> None:
        self.model = CVAEExpert(input_dim=3, hidden_dim=4, latent_dim=2)

    def project(self, expert_domain: int, x: torch.Tensor) -> torch.Tensor:
        return x

    def domain_cvae(self, expert_domain: int) -> CVAEExpert:
        return self.model


def test_response_feature_extraction_emits_variance_schema() -> None:
    torch.manual_seed(1)
    features = compute_response_features(
        bank=_FakeBank(),
        expert_domain=1,
        x_cpu=torch.randn(4, 3),
        support_idxs=[0, 1, 2],
        device=torch.device("cpu"),
        n_repeats=4,
        repeat_seed_base=123,
        include_residual_shape_features=True,
    )
    assert "response_posterior_mu_q75" in features
    assert "response_decode_repeat_variance_q75" in features
    assert "response_decode_repeat_variance_max" in features
    assert "response_recon_repeat_variance_q75" in features
    assert "response_residual_abs_q75" in features
    assert "response_kl_mean" not in features
