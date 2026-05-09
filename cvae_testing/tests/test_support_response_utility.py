from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.support_response_routing import (
    ProtocolError,
    RiskConstrainedResponseConfig,
    SupportResponseConfig,
    SupportUtilityConfig,
    audit_support_response_features,
    evaluate_support_response_routing_from_arrays,
)
from src.eval.evaluators.support_set_calibration import make_support_eval_split


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fixture() -> tuple[np.ndarray, list[dict], np.ndarray, list[int]]:
    domains = [0, 1, 2, 3, 4]
    metadata: list[dict] = []
    embeddings: list[list[float]] = []
    for domain in domains:
        for offset in range(6):
            metadata.append(
                {
                    "sample_id": f"{domain}-{offset}",
                    "magnification": int(domain),
                    "label": int(offset % 2),
                }
            )
            embeddings.append([float(domain), float(offset % 2), float(offset) / 10.0])

    preferred = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}
    nelbo = np.zeros((len(metadata), len(domains)), dtype=np.float64)
    for row_idx, row in enumerate(metadata):
        q = int(row["magnification"])
        for col, expert in enumerate(domains):
            rank = (int(expert) - int(preferred[q])) % len(domains)
            nelbo[row_idx, col] = 1.0 + rank + (row_idx % 3) * 0.01
    return np.asarray(embeddings, dtype=np.float64), metadata, nelbo, domains


def _support_cfg() -> SupportResponseConfig:
    return SupportResponseConfig(
        enabled=True,
        support_sizes=(2,),
        support_seeds=(17,),
        sampling_policies=("random",),
        feature_regimes=("static_response_indirect", "response_indirect_shuffled"),
        primary_feature_regime="static_response_indirect",
        ranker="linear_pairwise_ridge",
        ridge_l2=1.0e-3,
        num_response_repeats=2,
        tie_policy="stable_expert_index",
        domain_level_aggregation=True,
        source_leave_pseudo_domain_out_diagnostic=True,
    )


def _risk_support_cfg() -> SupportResponseConfig:
    cfg = _support_cfg()
    return SupportResponseConfig(
        enabled=cfg.enabled,
        support_sizes=cfg.support_sizes,
        support_seeds=cfg.support_seeds,
        sampling_policies=cfg.sampling_policies,
        feature_regimes=cfg.feature_regimes,
        primary_feature_regime=cfg.primary_feature_regime,
        ranker=cfg.ranker,
        ridge_l2=cfg.ridge_l2,
        num_response_repeats=cfg.num_response_repeats,
        tie_policy=cfg.tie_policy,
        domain_level_aggregation=cfg.domain_level_aggregation,
        source_leave_pseudo_domain_out_diagnostic=cfg.source_leave_pseudo_domain_out_diagnostic,
        risk_constrained=RiskConstrainedResponseConfig(
            enabled=True,
            margin_thresholds=(0.0, 0.25),
            support_regret_thresholds=(0.0, 10.0),
            top1_tolerance=0.02,
            spearman_tolerance=0.05,
            focus_query_domain=3,
            focus_expert=4,
        ),
    )


def _support_utility_cfg(*, sampling_policies: tuple[str, ...] = ("random",)) -> SupportResponseConfig:
    cfg = _support_cfg()
    return SupportResponseConfig(
        enabled=cfg.enabled,
        support_sizes=cfg.support_sizes,
        support_seeds=cfg.support_seeds,
        sampling_policies=sampling_policies,
        feature_regimes=cfg.feature_regimes,
        primary_feature_regime=cfg.primary_feature_regime,
        ranker=cfg.ranker,
        ridge_l2=cfg.ridge_l2,
        num_response_repeats=cfg.num_response_repeats,
        tie_policy=cfg.tie_policy,
        domain_level_aggregation=cfg.domain_level_aggregation,
        source_leave_pseudo_domain_out_diagnostic=cfg.source_leave_pseudo_domain_out_diagnostic,
        support_utility=SupportUtilityConfig(
            enabled=True,
            alpha_grid=(0.0, 0.5, 1.0),
            alpha_selection_policy="source_inner_gap_min_with_non_regression",
            require_unlabeled_support=True,
        ),
    )


def _run(tmp_path: Path) -> dict:
    embeddings, metadata, nelbo, expert_domains = _fixture()
    domain_by_index = {idx: int(row["magnification"]) for idx, row in enumerate(metadata)}
    preferred = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}

    def response_feature_fn(support_indices, expert_domain: int, split_id: str):
        assert support_indices
        query_domain = domain_by_index[int(support_indices[0])]
        assert all(domain_by_index[int(i)] == query_domain for i in support_indices)
        rank = (int(expert_domain) - int(preferred[query_domain])) % len(expert_domains)
        return {
            "response_posterior_mu_mean": float(rank),
            "response_decode_repeat_variance_mean": float(rank) / 10.0,
        }

    return evaluate_support_response_routing_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        seed=11,
        dataset_name="camelyon17",
        strategy="categorical_exact",
        tau=1.0,
        support_cfg=_support_cfg(),
        reports_dir=tmp_path,
        response_feature_fn=response_feature_fn,
        data_cfg={
            "dataset_domain_semantics": "camelyon17_center",
            "legacy_domain_field_alias": "magnification",
        },
    )


def _run_risk(tmp_path: Path) -> dict:
    embeddings, metadata, nelbo, expert_domains = _fixture()
    domain_by_index = {idx: int(row["magnification"]) for idx, row in enumerate(metadata)}
    preferred = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}

    def response_feature_fn(support_indices, expert_domain: int, split_id: str):
        assert support_indices
        query_domain = domain_by_index[int(support_indices[0])]
        rank = (int(expert_domain) - int(preferred[query_domain])) % len(expert_domains)
        return {
            "response_posterior_mu_mean": float(rank),
            "response_decode_repeat_variance_mean": float(rank) / 10.0,
        }

    return evaluate_support_response_routing_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        seed=11,
        dataset_name="camelyon17",
        strategy="categorical_exact",
        tau=1.0,
        support_cfg=_risk_support_cfg(),
        reports_dir=tmp_path,
        response_feature_fn=response_feature_fn,
        data_cfg={
            "dataset_domain_semantics": "camelyon17_center",
            "legacy_domain_field_alias": "magnification",
        },
    )


def _run_support_utility(tmp_path: Path, *, sampling_policies: tuple[str, ...] = ("random",)) -> dict:
    embeddings, metadata, nelbo, expert_domains = _fixture()
    domain_by_index = {idx: int(row["magnification"]) for idx, row in enumerate(metadata)}
    preferred = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}

    def response_feature_fn(support_indices, expert_domain: int, split_id: str):
        assert support_indices
        query_domain = domain_by_index[int(support_indices[0])]
        rank = (int(expert_domain) - int(preferred[query_domain])) % len(expert_domains)
        return {
            "response_posterior_mu_mean": float(rank),
            "response_decode_repeat_variance_mean": float(rank) / 10.0,
        }

    return evaluate_support_response_routing_from_arrays(
        embeddings=embeddings,
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        seed=11,
        dataset_name="camelyon17",
        strategy="categorical_exact",
        tau=1.0,
        support_cfg=_support_utility_cfg(sampling_policies=sampling_policies),
        reports_dir=tmp_path,
        response_feature_fn=response_feature_fn,
        data_cfg={
            "dataset_domain_semantics": "camelyon17_center",
            "legacy_domain_field_alias": "magnification",
        },
    )


def test_candidate_specific_pairwise_labels_and_exclusions(tmp_path: Path) -> None:
    embeddings, metadata, nelbo, expert_domains = _fixture()
    _ = embeddings
    labels_by_index = {idx: int(row["label"]) for idx, row in enumerate(metadata)}
    _run(tmp_path)

    rows = [
        row
        for row in _read_csv(tmp_path / "support_response_pair_predictions.csv")
        if row["method"] == "support_response_pairwise_static_response_indirect"
    ]
    assert rows
    assert len(rows) == 5 * 4 * 3
    for row in rows:
        outer = int(row["outer_target_domain"])
        query = int(row["pseudo_query_domain"])
        better = int(row["better_candidate_expert"])
        worse = int(row["worse_candidate_expert"])
        assert row["comparison_scope"] == "within_pseudo_query_domain"
        assert better != query
        assert worse != query
        assert better != outer
        assert worse != outer

        query_indices = [idx for idx, meta in enumerate(metadata) if int(meta["magnification"]) == query]
        split = make_support_eval_split(
            target_domain=query,
            target_indices=query_indices,
            labels_by_index=labels_by_index,
            support_size=2,
            sampling_policy="random",
            support_seed=17,
        )
        better_mean = float(np.mean(nelbo[np.asarray(split.eval_indices), expert_domains.index(better)]))
        worse_mean = float(np.mean(nelbo[np.asarray(split.eval_indices), expert_domains.index(worse)]))
        assert abs(float(row["better_label_nelbo"]) - better_mean) < 1e-12
        assert abs(float(row["worse_label_nelbo"]) - worse_mean) < 1e-12
        assert better_mean < worse_mean


def test_support_response_artifacts_encode_protocol_controls_and_score_direction(tmp_path: Path) -> None:
    results = _run(tmp_path)
    assert results["protocol_lock"]["score_direction"] == "predicted_score_is_predicted_mean_nelbo_lower_is_better"
    assert results["metrics_by_method"]["support_metadata_routing"]["n_samples_micro"] == 5.0
    assert results["metrics_by_method"]["support_metadata_routing"]["n_query_domains_macro"] == 5.0

    sample_rows = _read_csv(tmp_path / "support_response_sample_selections.csv")
    methods = {row["method"] for row in sample_rows}
    assert "source_global_prior_routing" in methods
    assert "expert_id_only_pairwise" in methods
    assert "support_candidate_oracle" in methods
    assert "support_response_pairwise_static_response_indirect" in methods
    assert "support_response_pairwise_response_indirect_shuffled" in methods

    for row in sample_rows:
        candidates = {int(v) for v in row["candidate_experts"].split("|") if v}
        assert int(row["selected_expert"]) in candidates
        assert int(row["candidate_oracle_expert"]) in candidates
        assert int(row["fold_query_domain"]) not in candidates
        assert row["score_direction"] == "lower_predicted_score_is_higher_compatibility"
        assert row["dataset_domain_semantics"] == "camelyon17_center"
        assert row["storage_field"] == "magnification"
        assert row["target_support_data_location"] == "target_local"
        assert row["raw_target_images_exported"] == "False"
        assert row["target_embeddings_exported"] == "False"
        score_map = {int(k): float(v) for k, v in json.loads(row["predicted_score_by_expert_json"]).items()}
        selected = int(row["selected_expert"])
        assert selected == sorted(score_map, key=lambda expert: (score_map[expert], expert))[0]
        if row["method"] == "support_candidate_oracle":
            assert int(row["adoption_eligible"]) == 0
            assert int(row["diagnostic_only"]) == 1
            assert int(row["routing_uses_eval_nelbo"]) == 1
        if row["method"] == "expert_id_only_pairwise":
            assert int(row["adoption_eligible"]) == 0
            assert row["method_role"] == "control"

    audit_rows = _read_csv(tmp_path / "support_response_feature_audit.csv")
    assert audit_rows
    assert {row["scaler_fit_scope"] for row in audit_rows} == {"source_training_pairs_only"}
    for artifact_name in [
        "support_response_domain_breakdown.csv",
        "support_response_method_summary.csv",
    ]:
        artifact_rows = _read_csv(tmp_path / artifact_name)
        assert artifact_rows
        assert {row["dataset_domain_semantics"] for row in artifact_rows} == {"camelyon17_center"}
        assert {row["storage_field"] for row in artifact_rows} == {"magnification"}


def test_risk_constrained_response_writes_frozen_threshold_and_audits(tmp_path: Path) -> None:
    results = _run_risk(tmp_path)

    assert "risk_constrained_response_routing" in results["metrics_by_method"]
    risk_metrics = results["metrics_by_method"]["risk_constrained_response_routing"]
    assert risk_metrics["method_role"] == "learned"
    assert risk_metrics["adoption_eligible"] == 1.0
    assert risk_metrics["routing_uses_eval_nelbo"] == 0.0

    threshold_rows = _read_csv(tmp_path / "risk_constrained_selected_thresholds.csv")
    assert len(threshold_rows) == 5
    for row in threshold_rows:
        assert row["selection_source"] == "source_inner_only"
        assert row["created_before_target_eval_scoring"] == "1"
        assert int(row["num_source_inner_units"]) > 0
        assert row["method"] == "risk_constrained_response_routing"

    risk_rows = _read_csv(tmp_path / "risk_constrained_sample_selections.csv")
    assert len(risk_rows) == 5
    for row in risk_rows:
        candidates = {int(v) for v in row["candidate_experts"].split("|") if v}
        assert int(row["selected_expert"]) in candidates
        assert int(row["fold_query_domain"]) not in candidates
        assert row["policy_name"] == "metadata_anchored_response_routing_with_support_regret_gate"
        assert row["threshold_selection_policy"] == "source_inner_only"
        assert row["risk_gate_source"] == "support_nelbo_regret_vs_metadata_anchor"

    override_rows = _read_csv(tmp_path / "risk_constrained_override_audit.csv")
    expert4_rows = _read_csv(tmp_path / "risk_constrained_expert4_audit.csv")
    assert len(override_rows) == len(risk_rows)
    assert len(expert4_rows) == len(risk_rows)


def test_support_utility_conservative_writes_alpha_and_unlabeled_protocol_fields(tmp_path: Path) -> None:
    results = _run_support_utility(tmp_path)

    assert "support_set_nelbo_top1" in results["metrics_by_method"]
    assert "support_set_nelbo_conservative" in results["metrics_by_method"]
    conservative_metrics = results["metrics_by_method"]["support_set_nelbo_conservative"]
    assert conservative_metrics["adoption_eligible"] == 1.0
    assert conservative_metrics["routing_uses_eval_nelbo"] == 0.0
    assert "high_regret_selection_rate" in conservative_metrics
    assert results["protocol_lock"]["support_estimated_utility"]["alpha_grid"] == [0.0, 0.5, 1.0]
    assert (
        results["protocol_lock"]["support_estimated_utility"]["alpha_selection_policy"]
        == "source_inner_gap_min_with_non_regression"
    )

    hyper_rows = _read_csv(tmp_path / "support_utility_selected_hyperparams.csv")
    assert len(hyper_rows) == 5
    for row in hyper_rows:
        assert row["method"] == "support_set_nelbo_conservative"
        assert json.loads(row["alpha_grid"]) == [0.0, 0.5, 1.0]
        assert row["alpha_selection_policy"] == "source_inner_gap_min_with_non_regression"
        assert row["selection_source"] == "source_inner_only"
        assert row["selected_before_target_eval_scoring"] == "1"
        assert int(row["n_aggregation_units"]) > 0
        assert abs(float(row["top1_tolerance_abs"]) - (1.0 / int(row["n_aggregation_units"]))) < 1e-12

    sample_rows = [
        row
        for row in _read_csv(tmp_path / "support_response_sample_selections.csv")
        if row["method"] == "support_set_nelbo_conservative"
    ]
    assert len(sample_rows) == 5
    for row in sample_rows:
        assert row["support_labels_used_for_routing"] == "0"
        assert int(row["support_n"]) == 2
        assert float(row["stderr_support_nelbo"]) >= 0.0
        assert "conservative_support_score" in row
        assert row["bottom_half_selection"] in {"0", "1"}
        assert row["high_regret_selection"] in {"0", "1"}
        assert row["catastrophic_mistake"] in {"0", "1"}


def test_support_utility_blocks_label_dependent_support_sampling(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="requires unlabeled support routing"):
        _run_support_utility(tmp_path, sampling_policies=("class_balanced",))


def test_source_global_prior_excludes_target_and_self_utility(tmp_path: Path) -> None:
    _embeddings, metadata, nelbo, expert_domains = _fixture()
    _run(tmp_path)
    labels_by_index = {idx: int(row["label"]) for idx, row in enumerate(metadata)}

    rows = [
        row
        for row in _read_csv(tmp_path / "support_response_sample_selections.csv")
        if row["method"] == "source_global_prior_routing"
    ]
    assert rows
    for row in rows:
        outer = int(row["target_domain"])
        predicted = {int(k): float(v) for k, v in json.loads(row["predicted_score_by_expert_json"]).items()}
        for expert, score in predicted.items():
            vals = []
            for pseudo_query in sorted(set(expert_domains) - {outer, expert}):
                query_indices = [
                    idx for idx, meta in enumerate(metadata) if int(meta["magnification"]) == int(pseudo_query)
                ]
                split = make_support_eval_split(
                    target_domain=int(pseudo_query),
                    target_indices=query_indices,
                    labels_by_index=labels_by_index,
                    support_size=2,
                    sampling_policy="random",
                    support_seed=17,
                )
                vals.append(float(np.mean(nelbo[np.asarray(split.eval_indices), expert_domains.index(expert)])))
            assert abs(score - float(np.mean(vals))) < 1e-12


def test_response_feature_audit_blocks_direct_utility_identity_and_eval_terms() -> None:
    rows = [
        {
            "candidate_expert": 2,
            "response_posterior_mu_mean": 0.1,
            "response_nelbo_mean": 1.0,
            "response_recon_mean": 2.0,
            "response_kl_mean": 3.0,
            "oracle_rank": 1.0,
            "target_eval_stat": 4.0,
            "expert_id": 2,
        },
        {
            "candidate_expert": 3,
            "response_posterior_mu_mean": 0.2,
            "response_nelbo_mean": 1.1,
            "response_recon_mean": 2.1,
            "response_kl_mean": 3.1,
            "oracle_rank": 2.0,
            "target_eval_stat": 4.1,
            "expert_id": 3,
        },
    ]
    audit = audit_support_response_features(
        rows,
        regime="response_indirect",
        feature_names=[
            "response_posterior_mu_mean",
            "response_nelbo_mean",
            "response_recon_mean",
            "response_kl_mean",
            "oracle_rank",
            "target_eval_stat",
            "expert_id",
        ],
    )
    assert audit.feature_names == ["response_posterior_mu_mean"]
    assert set(audit.blocked_features) == {
        "response_nelbo_mean",
        "response_recon_mean",
        "response_kl_mean",
        "oracle_rank",
        "target_eval_stat",
        "expert_id",
    }
    assert {"nelbo", "oracle", "target", "eval", "expert_id"} <= set(audit.blocked_feature_terms)

    expert_only = audit_support_response_features(
        rows,
        regime="expert_id_only",
        allow_candidate_identity=True,
    )
    assert expert_only.feature_names == ["expert_onehot_2", "expert_onehot_3"]
