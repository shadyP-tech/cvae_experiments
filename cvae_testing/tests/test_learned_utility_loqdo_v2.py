from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators import learned_utility as lu
from src.eval.evaluators.learned_utility_pairs import _build_fold_training_pair_features
from src.eval.evaluators.learned_utility_protocol import (
    ProtocolError,
    _aggregate_metrics_from_sample_rows,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fake_scored_payload():
    expert_domains = [40, 100, 200]
    sample_domains = np.asarray([40, 40, 100, 100, 200, 200], dtype=np.int64)
    embeddings = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
        ],
        dtype=np.float64,
    )
    nelbo = np.asarray(
        [
            [0.1, 0.5, 0.8],
            [0.1, 0.4, 0.9],
            [0.6, 0.1, 0.7],
            [0.8, 0.1, 0.5],
            [0.9, 0.4, 0.1],
            [0.8, 0.3, 0.1],
        ],
        dtype=np.float64,
    )
    metadata = [
        {"magnification": int(domain), "sample_id": f"s{i}"}
        for i, domain in enumerate(sample_domains.tolist())
    ]
    return embeddings, sample_domains, nelbo, expert_domains, metadata


def _learned_cfg() -> dict:
    return {
        "predictors": ["linear_regressor"],
        "pair_features": {"include_metadata_features": True},
        "scoring": {"pair_batch_size": 2},
        "hybrid_scoring": {
            "enabled": True,
            "alphas": [0.0, 1.0],
            "normalization_primary": "per_query_zscore",
            "normalization_sensitivity": "per_query_minmax",
            "run_sensitivity": False,
            "tie_policy": "stable_expert_index",
        },
        "compatibility_research": {
            "floors": {"random_rank_floor": False, "random_score_floor": False},
            "permutation_tests": {
                "expert_label_permutation": False,
                "metadata_permutation": False,
                "repeats": 1,
            },
            "diagnostics": {"save_distribution_plots": False},
            "gate": {"uplift_reference_method": "metadata_routing"},
        },
    }


def test_fold_training_pairs_exclude_outer_target_and_query_self_expert() -> None:
    embeddings = np.arange(5 * 2, dtype=np.float64).reshape(5, 2)
    sample_domains = np.asarray([40, 100, 100, 200, 400], dtype=np.int64)
    x, q, e, s = _build_fold_training_pair_features(
        sample_embeddings=embeddings,
        sample_domains=sample_domains,
        train_indices=np.asarray([1, 2, 3, 4], dtype=np.int64),
        expert_domains=[40, 100, 200, 400],
        outer_heldout_domain=40,
        include_metadata_features=True,
    )

    assert x.shape[0] == q.shape[0] == e.shape[0] == s.shape[0]
    assert all(int(expert) != 40 for expert in e.tolist())
    assert all(int(expert) != int(query) for query, expert in zip(q.tolist(), e.tolist()))
    assert set(e[q == 100].tolist()) == {200, 400}
    assert set(e[q == 200].tolist()) == {100, 400}
    assert set(e[q == 400].tolist()) == {100, 200}


def test_learned_utility_v2_candidate_oracle_and_artifact_invariants(tmp_path, monkeypatch) -> None:
    def fake_score(**kwargs):
        _ = kwargs
        return _fake_scored_payload()

    monkeypatch.setattr(lu, "_score_experts_batched", fake_score)

    results = lu.evaluate_learned_utility_loqdo(
        test_cache=tmp_path / "unused.pt",
        expert_checkpoints={"expert_40": "unused", "expert_100": "unused", "expert_200": "unused"},
        hidden_dim=4,
        latent_dim=2,
        strategy="categorical_exact",
        tau=1.0,
        seed=7,
        learned_cfg=_learned_cfg(),
        reports_dir=tmp_path,
    )

    assert results["protocol_contract"]["protocol_version"] == "learned_utility_loqdo_candidate_exclusion_v2"
    assert results["protocol_contract"]["metrics_comparable_to_previous_protocol"] is False
    assert results["protocol_contract"]["previous_protocol_invalidated_by_target_candidate_leakage"] is True
    assert "oracle_routing" not in results["metrics_by_method"]
    assert "candidate_oracle_routing" in results["metrics_by_method"]
    oracle_metrics = results["metrics_by_method"]["candidate_oracle_routing"]
    assert oracle_metrics["protocol_version"] == "learned_utility_loqdo_candidate_exclusion_v2"
    assert oracle_metrics["method_role"] == "diagnostic"
    assert oracle_metrics["adoption_eligible"] == 0.0
    assert oracle_metrics["diagnostic_only"] == 1.0
    assert results["metrics_by_method"]["latent_wasserstein_routing"]["diagnostic_only"] == 1.0
    assert results["metrics_by_method"]["latent_wasserstein_routing"]["method_role"] == "diagnostic"
    assert results["metrics_by_method"]["hybrid_alpha_0.0"]["diagnostic_only"] == 1.0
    assert results["metrics_by_method"]["hybrid_alpha_0.0"]["method_role"] == "diagnostic"
    assert results["artifacts"]["method_summary"] == "learned_utility_method_summary.csv"

    sample_rows = _read_csv(tmp_path / "learned_utility_sample_selections.csv")
    assert sample_rows
    assert all(row["method"] != "oracle_routing" for row in sample_rows)

    method_summary_rows = _read_csv(tmp_path / "learned_utility_method_summary.csv")
    assert method_summary_rows
    summary_by_method = {row["method"]: row for row in method_summary_rows}
    oracle_summary = summary_by_method["candidate_oracle_routing"]
    assert oracle_summary["protocol_version"] == "learned_utility_loqdo_candidate_exclusion_v2"
    assert oracle_summary["method_role"] == "diagnostic"
    assert int(oracle_summary["adoption_eligible"]) == 0
    assert int(oracle_summary["diagnostic_only"]) == 1
    assert int(summary_by_method["latent_wasserstein_routing"]["diagnostic_only"]) == 1
    assert int(summary_by_method["hybrid_alpha_0.0"]["diagnostic_only"]) == 1

    row_40 = [
        row
        for row in sample_rows
        if row["method"] == "candidate_oracle_routing"
        and int(row["query_domain"]) == 40
        and int(row["sample_index"]) == 0
    ][0]
    assert int(row_40["global_oracle_expert"]) == 40
    assert int(row_40["global_oracle_excluded_by_policy"]) == 1
    assert int(row_40["candidate_oracle_expert"]) == 100
    assert int(row_40["selected_expert"]) == 100

    for row in sample_rows:
        candidates = {int(v) for v in row["candidate_experts"].split("|")}
        assert int(row["fold_query_domain"]) not in candidates
        assert int(row["target_expert_excluded"]) == 1
        assert int(row["n_candidate_experts"]) == len(candidates)
        assert int(row["selected_expert"]) in candidates
        assert int(row["candidate_oracle_expert"]) in candidates
        if int(row["adoption_eligible"]) == 1:
            assert int(row["diagnostic_only"]) == 0
            assert int(row["routing_uses_eval_nelbo"]) == 0
            assert int(row["routing_uses_eval_domain_statistics"]) == 0

    pair_rows = _read_csv(tmp_path / "learned_utility_pair_predictions.csv")
    assert len(pair_rows) == 12
    for fold_domain in [40, 100, 200]:
        fold_rows = [r for r in pair_rows if int(r["fold_query_domain"]) == fold_domain]
        assert len(fold_rows) == 4
        assert all(int(r["expert_domain"]) != fold_domain for r in fold_rows)


def test_metric_aggregation_hard_fails_on_invalid_candidate_rows() -> None:
    valid = {
        "method": "metadata_routing",
        "query_domain": 40,
        "fold_query_domain": 40,
        "candidate_experts": "100|200",
        "selected_expert": 100,
        "candidate_oracle_expert": 100,
        "adoption_eligible": 1,
        "diagnostic_only": 0,
        "routing_uses_eval_nelbo": 0,
        "routing_uses_eval_domain_statistics": 0,
        "top1_oracle_hit": 1,
        "selected_rank": 1.0,
        "oracle_gap": 0.0,
        "oracle_gap_pct": 0.0,
        "spearman": 1.0,
        "pairwise_auc": 1.0,
        "selected_nelbo": 0.5,
        "candidate_oracle_nelbo": 0.5,
    }
    invalid = dict(valid)
    invalid["selected_expert"] = 40

    try:
        _aggregate_metrics_from_sample_rows([invalid])
    except ProtocolError as exc:
        assert "selected_expert" in str(exc)
    else:
        raise AssertionError("Expected ProtocolError for selected expert outside candidate pool")

    metrics = _aggregate_metrics_from_sample_rows([valid])
    assert metrics["metadata_routing"]["n_samples_micro"] == 1.0
    assert metrics["metadata_routing"]["n_query_domains_macro"] == 1.0


def test_delta_gate_guard_failure_demotes_method_summary_even_when_first_row_selected() -> None:
    base = {
        "method": "pairwise_tournament_delta_gated_sparse_mix_v1",
        "query_domain": 40,
        "fold_query_domain": 40,
        "candidate_experts": "100|200",
        "selected_expert": 100,
        "candidate_oracle_expert": 100,
        "adoption_eligible": 1,
        "diagnostic_only": 0,
        "routing_uses_eval_nelbo": 0,
        "routing_uses_eval_domain_statistics": 0,
        "top1_oracle_hit": 1,
        "selected_rank": 1.0,
        "oracle_gap": 0.0,
        "oracle_gap_pct": 0.0,
        "spearman": 1.0,
        "pairwise_auc": 1.0,
        "selected_nelbo": 0.5,
        "candidate_oracle_nelbo": 0.5,
    }
    selected = {
        **base,
        "delta_gate_selection_status": "selected",
        "delta_gate_diagnostic_only_reason": "",
    }
    failed = {
        **base,
        "query_domain": 100,
        "fold_query_domain": 100,
        "candidate_experts": "40|200",
        "selected_expert": 40,
        "candidate_oracle_expert": 40,
        "delta_gate_selection_status": "failed_guards_noop",
        "delta_gate_diagnostic_only_reason": "activation_rate_too_high",
    }

    metrics = _aggregate_metrics_from_sample_rows([selected, failed])
    delta = metrics["pairwise_tournament_delta_gated_sparse_mix_v1"]

    assert delta["method_role"] == "diagnostic"
    assert delta["adoption_eligible"] == 0.0
    assert delta["diagnostic_only"] == 1.0
    assert delta["delta_gate_source_inner_guard_pass"] == 0.0
    assert delta["diagnostic_only_reason"] == "activation_rate_too_high"
