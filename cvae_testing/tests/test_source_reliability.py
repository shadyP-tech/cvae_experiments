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

from src.eval.evaluators.learned_utility_config import (  # noqa: E402
    SourceReliabilityConfig,
    _parse_learned_utility_config,
)
from src.eval.evaluators.learned_utility_protocol import (  # noqa: E402
    FoldCandidateSet,
    ProtocolError,
    _method_protocol,
)
from src.eval.evaluators.learned_utility_selection import _selection_metrics  # noqa: E402
from src.config.load_config import load_config  # noqa: E402
from src.eval.evaluators import source_reliability as sr  # noqa: E402


def _cfg(**overrides) -> SourceReliabilityConfig:
    values = {
        "enabled": True,
        "primary_method": sr.PRIMARY_METHOD,
        "fallback_method": sr.FALLBACK_METHOD,
        "candidate_methods": ("pairwise_ranker_ae_only", "pairwise_ranker_ae_combined"),
        "group_key_candidates": ("patient_id", "slide_id", "case_id"),
        "pseudo_domain_strategy": "per_parent_group_embedding_kmeans",
        "n_pseudo_domains_per_source": 2,
        "min_pseudo_domains_per_source": 2,
        "min_groups_per_pseudo_domain": 1,
        "min_samples_per_pseudo_domain": 2,
        "min_candidate_pool_size": 2,
        "pca_dim": 2,
        "kmeans_iterations": 5,
        "aggregation_unit": "parent_domain_x_pseudo_domain_macro",
        "min_source_inner_units": 2,
        "min_parent_domains": 1,
        "min_units_per_parent_for_gain_share": 1,
        "max_top1_drop_abs": 0.02,
        "max_spearman_drop_abs": 0.03,
        "max_gap_pct_degradation": 1.0,
        "max_worst_unit_gap_degradation": 2.0,
        "min_gap_reduction_vs_fallback": 0.0,
        "min_positive_unit_rate": 0.60,
        "min_positive_parent_rate": 0.50,
        "max_positive_gain_share": 0.60,
        "require_parent_holdout_guard": True,
    }
    values.update(overrides)
    return SourceReliabilityConfig(**values)


def _payload():
    expert_domains = [0, 1, 2, 3, 4]
    rows = []
    domains = []
    metadata = []
    for domain in expert_domains:
        for group in range(4):
            for item in range(2):
                rows.append([float(domain), float(group), float(item), float(domain * 10 + group)])
                domains.append(domain)
                metadata.append(
                    {
                        "patient_id": f"patient_{domain}_{group}",
                        "slide_id": f"slide_{domain}_{group}",
                        "case_id": f"case_{domain}_{group}",
                    }
                )
    embeddings = np.asarray(rows, dtype=np.float64)
    sample_domains = np.asarray(domains, dtype=np.int64)
    true_nelbo = np.asarray(
        [[10.0 - float(domain) for domain in expert_domains] for _ in range(len(sample_domains))],
        dtype=np.float64,
    )
    ae_z = np.asarray(
        [[float(domain) for domain in expert_domains] for _ in range(len(sample_domains))],
        dtype=np.float64,
    )
    heldout = 0
    train_idx = np.where(sample_domains != heldout)[0].astype(np.int64)
    test_idx = np.where(sample_domains == heldout)[0].astype(np.int64)
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=heldout, expert_domains=expert_domains)
    true_eval = fold.slice_nelbo(true_nelbo, test_idx)
    global_eval = true_nelbo[test_idx]
    return {
        "embeddings": embeddings,
        "sample_domains": sample_domains,
        "metadata": metadata,
        "true_nelbo": true_nelbo,
        "ae_z": ae_z,
        "expert_domains": expert_domains,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "fold": fold,
        "true_eval": true_eval,
        "global_eval": global_eval,
    }


def _candidate_rows(payload, *, method: str, oracle_like: bool) -> list[dict]:
    fold = payload["fold"]
    test_idx = payload["test_idx"]
    if oracle_like:
        score_matrix = payload["true_eval"].copy()
    else:
        score_matrix = payload["ae_z"][test_idx][:, list(fold.candidate_col_indices)]
    _metrics, rows = _selection_metrics(
        method=method,
        query_domains=payload["sample_domains"][test_idx],
        expert_domains=fold.candidate_expert_domains,
        score_matrix=score_matrix,
        true_nelbo_matrix=payload["true_eval"],
        fold=fold,
        global_true_nelbo_matrix=payload["global_eval"],
        global_expert_domains=payload["expert_domains"],
        tie_policy="stable_expert_index",
    )
    for row in rows:
        row["sample_index"] = int(test_idx[int(row["sample_index"])])
    return rows


def _clean_candidate_rows(payload) -> list[dict]:
    return [
        *_candidate_rows(payload, method="pairwise_ranker_ae_only", oracle_like=False),
        *_candidate_rows(payload, method="pairwise_ranker_ae_combined", oracle_like=True),
    ]


def _run(payload, cfg=None, learned_rows=None):
    return sr.run_source_reliability_for_fold(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        metadata=payload["metadata"],
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        train_idx=payload["train_idx"],
        test_idx=payload["test_idx"],
        fold=payload["fold"],
        true_eval=payload["true_eval"],
        global_eval=payload["global_eval"],
        ae_zscore_matrix=payload["ae_z"],
        learned_sample_rows=_clean_candidate_rows(payload) if learned_rows is None else learned_rows,
        pairwise_cfg={"epochs": 1, "hidden_dim": 4, "batch_size": 64, "device": "cpu"},
        cfg=_cfg() if cfg is None else cfg,
        seed=7,
        tie_policy="stable_expert_index",
    )


def test_source_reliability_config_parses_defaults() -> None:
    cfg = _parse_learned_utility_config({"source_reliability": {"enabled": True}}).source_reliability
    assert cfg.enabled is True
    assert cfg.primary_method == sr.PRIMARY_METHOD
    assert cfg.fallback_method == sr.FALLBACK_METHOD
    assert cfg.candidate_methods == ("pairwise_ranker_ae_only", "pairwise_ranker_ae_combined")
    assert cfg.aggregation_unit == "parent_domain_x_pseudo_domain_macro"
    assert cfg.require_parent_holdout_guard is True


def test_source_reliability_dataset_configs_validate() -> None:
    for dataset in ["breakhis", "camelyon17"]:
        path = PROJECT_ROOT / "configs" / "experiments" / dataset / "learned_utility_source_reliability_v1.yaml"
        cfg = load_config(path)
        source = cfg["learned_utility"]["source_reliability"]
        assert cfg["experiment"]["name"] == "learned_utility_source_reliability_v1"
        assert source["enabled"] is True
        assert source["primary_method"] == sr.PRIMARY_METHOD
        assert source["fallback_method"] == sr.FALLBACK_METHOD
        assert source["candidate_methods"] == ["pairwise_ranker_ae_only", "pairwise_ranker_ae_combined"]


def test_source_reliability_method_protocol_flags_are_adoption_clean() -> None:
    protocol = _method_protocol(sr.PRIMARY_METHOD)
    assert protocol.adoption_eligible == 1
    assert protocol.diagnostic_only == 0
    assert protocol.routing_uses_eval_nelbo == 0
    assert protocol.routing_uses_eval_domain_statistics == 0


def test_source_reliability_pseudo_domains_use_source_only_rows() -> None:
    payload = _payload()
    pseudo = sr.build_source_pseudo_domains(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        metadata=payload["metadata"],
        source_indices=payload["train_idx"],
        heldout_domain=0,
        cfg=_cfg(),
    )
    assert pseudo.status == "available"
    assert pseudo.pseudo_rows
    assert all(int(row["source_only"]) == 1 for row in pseudo.pseudo_rows)
    assert all(int(row["parent_domain"]) != 0 for row in pseudo.pseudo_rows)
    with pytest.raises(ProtocolError, match="held-out target rows"):
        sr.build_source_pseudo_domains(
            embeddings=payload["embeddings"],
            sample_domains=payload["sample_domains"],
            metadata=payload["metadata"],
            source_indices=np.arange(len(payload["sample_domains"]), dtype=np.int64),
            heldout_domain=0,
            cfg=_cfg(),
        )


def test_source_reliability_pseudo_domains_constructed_per_parent_domain() -> None:
    payload = _payload()
    pseudo = sr.build_source_pseudo_domains(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        metadata=payload["metadata"],
        source_indices=payload["train_idx"],
        heldout_domain=0,
        cfg=_cfg(),
    )
    for (_parent, _pseudo), indices in pseudo.unit_indices.items():
        assert len(set(payload["sample_domains"][indices].tolist())) == 1


def test_source_reliability_group_key_candidates_use_first_available_key() -> None:
    payload = _payload()
    metadata = [dict(row, patient_id="") for row in payload["metadata"]]
    pseudo = sr.build_source_pseudo_domains(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        metadata=metadata,
        source_indices=payload["train_idx"],
        heldout_domain=0,
        cfg=_cfg(),
    )
    assert pseudo.group_key == "slide_id"


def test_source_reliability_missing_group_key_falls_back_needs_evidence() -> None:
    payload = _payload()
    metadata = [{key: "" for key in ("patient_id", "slide_id", "case_id")} for _ in payload["metadata"]]
    output = sr.run_source_reliability_for_fold(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        metadata=metadata,
        true_nelbo=payload["true_nelbo"],
        expert_domains=payload["expert_domains"],
        train_idx=payload["train_idx"],
        test_idx=payload["test_idx"],
        fold=payload["fold"],
        true_eval=payload["true_eval"],
        global_eval=payload["global_eval"],
        ae_zscore_matrix=payload["ae_z"],
        learned_sample_rows=_clean_candidate_rows(payload),
        pairwise_cfg={"epochs": 1, "hidden_dim": 4, "batch_size": 64, "device": "cpu"},
        cfg=_cfg(),
        seed=7,
        tie_policy="stable_expert_index",
    )
    assert output.selected_method_rows[0]["selection_status"] == sr.SELECTION_INSUFFICIENT


def test_source_reliability_patient_groups_not_split() -> None:
    payload = _payload()
    pseudo = sr.build_source_pseudo_domains(
        embeddings=payload["embeddings"],
        sample_domains=payload["sample_domains"],
        metadata=payload["metadata"],
        source_indices=payload["train_idx"],
        heldout_domain=0,
        cfg=_cfg(),
    )
    assignment: dict[str, str] = {}
    for row in pseudo.pseudo_rows:
        previous = assignment.setdefault(str(row["group_id"]), str(row["pseudo_domain_id"]))
        assert previous == str(row["pseudo_domain_id"])


def test_source_reliability_candidate_pool_excludes_target_and_parent_domain() -> None:
    expert_domains = [0, 1, 2, 3, 4]
    fold = FoldCandidateSet.for_heldout_domain(heldout_domain=0, expert_domains=expert_domains, excluded_domains=[2])
    assert 0 not in fold.candidate_expert_domains
    assert 2 not in fold.candidate_expert_domains


def test_source_reliability_candidate_pool_too_small_falls_back() -> None:
    payload = _payload()
    output = _run(payload, cfg=_cfg(min_candidate_pool_size=4))
    assert output.selected_method_rows[0]["selection_status"] == sr.SELECTION_POOL_TOO_SMALL


def test_source_reliability_source_inner_training_excludes_pseudo_validation_unit(monkeypatch) -> None:
    payload = _payload()
    seen_train: list[np.ndarray] = []

    def fake_fit(**kwargs):
        seen_train.append(np.asarray(kwargs["train_idx"], dtype=np.int64))
        fold = kwargs["fold"]
        return fold.slice_nelbo(kwargs["true_nelbo"], kwargs["val_idx"])

    monkeypatch.setattr(sr, "_fit_predict_pairwise_unit", fake_fit)
    _run(payload, cfg=_cfg(min_positive_unit_rate=0.0, min_positive_parent_rate=0.0, max_positive_gain_share=1.0))
    assert seen_train
    for train_idx in seen_train:
        assert not np.any(payload["sample_domains"][train_idx] == 0)


def test_source_reliability_parent_holdout_guard_excludes_full_parent_domain(monkeypatch) -> None:
    payload = _payload()
    seen: list[tuple[int, np.ndarray]] = []

    def fake_fit(**kwargs):
        seen.append((int(kwargs["parent_domain"]), np.asarray(kwargs["train_idx"], dtype=np.int64)))
        return kwargs["fold"].slice_nelbo(kwargs["true_nelbo"], kwargs["val_idx"])

    monkeypatch.setattr(sr, "_fit_predict_pairwise_unit", fake_fit)
    _run(payload, cfg=_cfg(min_positive_unit_rate=0.0, min_positive_parent_rate=0.0, max_positive_gain_share=1.0))
    assert seen
    for parent_domain, train_idx in seen:
        assert not np.any(payload["sample_domains"][train_idx] == parent_domain)


def test_source_reliability_rejects_non_protocol_clean_candidate_rows() -> None:
    payload = _payload()
    rows = _candidate_rows(payload, method="pairwise_ranker_ae_combined", oracle_like=True)
    rows[0]["routing_uses_eval_nelbo"] = 1
    clean, reason = sr._candidate_rows_clean(rows, fold=payload["fold"])
    assert clean is False
    assert reason == "candidate_method_routing_uses_eval_nelbo"


def test_source_reliability_selection_uses_no_target_nelbo(monkeypatch) -> None:
    payload = _payload()

    def fake_fit(**kwargs):
        return kwargs["fold"].slice_nelbo(kwargs["true_nelbo"], kwargs["val_idx"])

    monkeypatch.setattr(sr, "_fit_predict_pairwise_unit", fake_fit)
    output = _run(payload, cfg=_cfg(min_positive_unit_rate=0.0, min_positive_parent_rate=0.0, max_positive_gain_share=1.0))
    assert all(int(row["heldout_target_nelbo_used_for_selection"]) == 0 for row in output.selected_method_rows)
    assert all(int(row.get("heldout_target_nelbo_used_for_selection", 0)) == 0 for row in output.policy_audit_rows)


def test_source_reliability_gain_share_gate_rejects_dominated_signal() -> None:
    rows = [
        {"candidate_method": "m", "parent_domain": 1, "gap_pct_reduction_vs_fallback": 10.0, "material_degradation_vs_fallback": 0, "gap_pct_degradation_vs_fallback": -10.0, "top1_delta_vs_fallback": 0.0, "spearman_delta_vs_fallback": 0.0},
        {"candidate_method": "m", "parent_domain": 2, "gap_pct_reduction_vs_fallback": 0.1, "material_degradation_vs_fallback": 0, "gap_pct_degradation_vs_fallback": -0.1, "top1_delta_vs_fallback": 0.0, "spearman_delta_vs_fallback": 0.0},
    ]
    metric = sr._source_inner_metrics_for_candidate(unit_rows=rows, method="m", cfg=_cfg(max_positive_gain_share=0.60))
    assert metric["passes_reliability_gates"] == 0


def test_source_reliability_worst_unit_degradation_gate() -> None:
    rows = [
        {"candidate_method": "m", "parent_domain": 1, "gap_pct_reduction_vs_fallback": 1.0, "material_degradation_vs_fallback": 0, "gap_pct_degradation_vs_fallback": -1.0, "top1_delta_vs_fallback": 0.0, "spearman_delta_vs_fallback": 0.0},
        {"candidate_method": "m", "parent_domain": 2, "gap_pct_reduction_vs_fallback": -3.0, "material_degradation_vs_fallback": 0, "gap_pct_degradation_vs_fallback": 3.0, "top1_delta_vs_fallback": 0.0, "spearman_delta_vs_fallback": 0.0},
    ]
    metric = sr._source_inner_metrics_for_candidate(unit_rows=rows, method="m", cfg=_cfg(max_worst_unit_gap_degradation=2.0))
    assert metric["passes_reliability_gates"] == 0


def test_source_reliability_fallback_exactly_matches_ae_argmin() -> None:
    payload = _payload()
    output = _run(payload, cfg=_cfg(min_candidate_pool_size=4))
    fallback = sr._fallback_rows(
        sample_domains=payload["sample_domains"],
        expert_domains=payload["expert_domains"],
        test_idx=payload["test_idx"],
        fold=payload["fold"],
        true_eval=payload["true_eval"],
        global_eval=payload["global_eval"],
        ae_zscore_matrix=payload["ae_z"],
        tie_policy="stable_expert_index",
        selection_status=sr.SELECTION_POOL_TOO_SMALL,
        selected_source_method=sr.FALLBACK_METHOD,
    )
    assert [r["selected_expert"] for r in output.sample_rows] == [r["selected_expert"] for r in fallback]
    assert [r["selected_nelbo"] for r in output.sample_rows] == [r["selected_nelbo"] for r in fallback]
    assert [r["oracle_gap"] for r in output.sample_rows] == [r["oracle_gap"] for r in fallback]


def test_source_reliability_selected_rows_match_chosen_candidate_method(monkeypatch) -> None:
    payload = _payload()

    def fake_fit(**kwargs):
        if kwargs["method"] == "pairwise_ranker_ae_combined":
            return kwargs["fold"].slice_nelbo(kwargs["true_nelbo"], kwargs["val_idx"])
        return payload["ae_z"][kwargs["val_idx"]][:, list(kwargs["fold"].candidate_col_indices)]

    monkeypatch.setattr(sr, "_fit_predict_pairwise_unit", fake_fit)
    output = _run(payload, cfg=_cfg(min_positive_unit_rate=0.0, min_positive_parent_rate=0.0, max_positive_gain_share=1.0))
    assert output.selected_method_rows[0]["selection_status"] == sr.SELECTION_SELECTED
    assert output.selected_method_rows[0]["selected_method_by_outer_domain"] == "pairwise_ranker_ae_combined"
    candidate = _candidate_rows(payload, method="pairwise_ranker_ae_combined", oracle_like=True)
    assert [r["selected_expert"] for r in output.sample_rows] == [r["selected_expert"] for r in candidate]


def test_source_reliability_reports_fallback_reason() -> None:
    payload = _payload()
    output = _run(payload, cfg=_cfg(min_candidate_pool_size=4))
    assert output.selected_method_rows[0]["selection_status"] == sr.SELECTION_POOL_TOO_SMALL


def test_source_reliability_reports_predicted_vs_realized_gain() -> None:
    payload = _payload()
    output = _run(payload, cfg=_cfg(min_candidate_pool_size=4))
    assert output.predicted_vs_realized_rows
    assert "source_inner_predicted_gain" in output.predicted_vs_realized_rows[0]
    assert "heldout_realized_gain" in output.predicted_vs_realized_rows[0]


def test_source_reliability_reports_selected_method_by_outer_domain() -> None:
    payload = _payload()
    output = _run(payload, cfg=_cfg(min_candidate_pool_size=4))
    assert output.selected_method_rows[0]["selected_method_by_outer_domain"] == sr.FALLBACK_METHOD


def test_source_reliability_reports_dataset_fallback_rates(tmp_path: Path) -> None:
    artifacts = sr.write_source_reliability_artifacts(
        reports_dir=tmp_path,
        pseudo_domain_rows=[],
        source_inner_unit_rows=[],
        candidate_metric_rows=[],
        parent_guard_rows=[],
        selection_policy_rows=[{"selection_status": sr.SELECTION_POOL_TOO_SMALL}],
        policy_audit_rows=[],
        predicted_vs_realized_rows=[],
        selected_method_rows=[{"selection_status": sr.SELECTION_POOL_TOO_SMALL}],
    )
    summary = json.loads((tmp_path / "source_reliability_provenance.json").read_text(encoding="utf-8"))
    assert artifacts["source_reliability_dataset_selection_summary"] == "source_reliability_dataset_selection_summary.csv"
    assert summary["fallback_rate_by_dataset"] == 1.0
    assert summary["candidate_selection_rate_by_dataset"] == 0.0


def test_source_reliability_reports_required_artifacts(tmp_path: Path) -> None:
    artifacts = sr.write_source_reliability_artifacts(
        reports_dir=tmp_path,
        pseudo_domain_rows=[{"a": 1}],
        source_inner_unit_rows=[{"a": 1}],
        candidate_metric_rows=[{"a": 1}],
        parent_guard_rows=[{"a": 1}],
        selection_policy_rows=[{"selection_status": sr.SELECTION_SELECTED}],
        policy_audit_rows=[{"a": 1}],
        predicted_vs_realized_rows=[{"source_inner_predicted_gain": 1.0, "heldout_realized_gain": 1.0, "selection_correct": 1}],
        selected_method_rows=[{"selection_status": sr.SELECTION_SELECTED}],
    )
    expected = {
        "source_reliability_pseudo_domains",
        "source_reliability_source_inner_units",
        "source_reliability_candidate_metrics",
        "source_reliability_parent_guard",
        "source_reliability_selection_policy",
        "source_reliability_policy_audit",
        "source_reliability_predicted_vs_realized",
        "source_reliability_selected_method_by_outer_domain",
        "source_reliability_dataset_selection_summary",
        "source_reliability_provenance",
    }
    assert expected.issubset(set(artifacts))
    for filename in artifacts.values():
        assert (tmp_path / filename).exists()
    with (tmp_path / "source_reliability_dataset_selection_summary.csv").open("r", encoding="utf-8") as f:
        assert list(csv.DictReader(f))[0]["candidate_selection_rate_by_dataset"] == "1.0"


def test_source_reliability_tiny_capped_smoke_run(monkeypatch) -> None:
    payload = _payload()

    def fake_fit(**kwargs):
        return kwargs["fold"].slice_nelbo(kwargs["true_nelbo"], kwargs["val_idx"])

    monkeypatch.setattr(sr, "_fit_predict_pairwise_unit", fake_fit)
    output = _run(payload, cfg=_cfg(min_positive_unit_rate=0.0, min_positive_parent_rate=0.0, max_positive_gain_share=1.0))
    assert len(output.sample_rows) == int(payload["test_idx"].shape[0])
    assert output.policy_audit_rows
