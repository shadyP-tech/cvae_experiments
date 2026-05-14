from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.evaluators.learned_utility_config import SourceUtilityTransferConfig
from src.eval.evaluators.source_utility_transfer import (
    DIRECT_METHOD,
    RANDOM_CONTROL_METHOD,
    SAFE_METHOD,
    SHUFFLED_CONTROL_METHOD,
    evaluate_source_utility_transfer,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], dict[int, int]]:
    expert_domains = [0, 1, 2, 3, 4]
    domain_to_idx = {int(domain): idx for idx, domain in enumerate(expert_domains)}
    sample_domains = np.asarray(
        [domain for domain in expert_domains for _ in range(6)],
        dtype=np.int64,
    )
    domain_utility = np.asarray(
        [
            [0.10, 0.72, 0.31, 0.58, 0.46],
            [0.54, 0.10, 0.77, 0.29, 0.48],
            [0.44, 0.68, 0.10, 0.36, 0.26],
            [0.22, 0.59, 0.43, 0.10, 0.64],
            [0.61, 0.24, 0.49, 0.57, 0.10],
        ],
        dtype=np.float64,
    )
    true_nelbo = np.empty((sample_domains.shape[0], len(expert_domains)), dtype=np.float64)
    for sample_idx, domain in enumerate(sample_domains.tolist()):
        noise = 0.01 * ((sample_idx % 3) - 1)
        true_nelbo[sample_idx] = domain_utility[int(domain)] + noise

    metadata_similarity = np.empty_like(true_nelbo)
    for sample_idx, domain in enumerate(sample_domains.tolist()):
        for expert in expert_domains:
            metadata_similarity[sample_idx, int(expert)] = 1.0 / (1.0 + abs(int(domain) - int(expert)))
    return true_nelbo, sample_domains, metadata_similarity, expert_domains, domain_to_idx


def _cfg() -> SourceUtilityTransferConfig:
    return SourceUtilityTransferConfig(
        enabled=True,
        variants=("metadata_only",),
        query_unit="minibag",
        minibag_size=3,
        minibags_per_domain=2,
        minibag_seeds=(17,),
        ranker="linear_pairwise_ridge",
        ridge_l2=1.0e-3,
        normalized_margin_thresholds=(0.0, 0.5, float("inf")),
        fallback_method="metadata_routing",
        random_control_seeds=(101, 102),
        enable_shuffled_profile_control=True,
    )


def test_source_utility_transfer_writes_clustered_artifacts_and_protocol_flags(tmp_path: Path) -> None:
    true_nelbo, sample_domains, metadata_similarity, expert_domains, domain_to_idx = _fixture()
    outputs = evaluate_source_utility_transfer(
        true_nelbo=true_nelbo,
        sample_domains=sample_domains,
        metadata_similarity=metadata_similarity,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        seed=42,
        cfg=_cfg(),
        reports_dir=tmp_path,
    )

    expected_artifacts = {
        "source_utility_transfer_sample_selections",
        "source_utility_transfer_pair_predictions",
        "source_utility_transfer_feature_audit",
        "source_utility_transfer_threshold_audit",
        "source_utility_transfer_override_diagnostics",
        "source_utility_transfer_domain_breakdown",
        "source_utility_transfer_clustered_metrics",
        "source_utility_transfer_random_matched_control",
        "source_utility_transfer_negative_controls",
    }
    assert set(outputs.artifacts) == expected_artifacts
    for filename in outputs.artifacts.values():
        assert (tmp_path / filename).exists()

    methods = {str(row["method"]) for row in outputs.sample_rows}
    assert {DIRECT_METHOD, SAFE_METHOD, RANDOM_CONTROL_METHOD, SHUFFLED_CONTROL_METHOD} <= methods

    safe_rows = [row for row in outputs.sample_rows if row["method"] == SAFE_METHOD]
    direct_rows = [row for row in outputs.sample_rows if row["method"] == DIRECT_METHOD]
    assert safe_rows and direct_rows
    assert {int(row["adoption_eligible"]) for row in safe_rows} == {1}
    assert {int(row["diagnostic_only"]) for row in safe_rows} == {0}
    assert {int(row["strict_source_only"]) for row in safe_rows} == {1}
    assert {int(row["adoption_eligible"]) for row in direct_rows} == {0}
    assert {int(row["diagnostic_only"]) for row in direct_rows} == {1}

    clustered = _read_csv(tmp_path / outputs.artifacts["source_utility_transfer_clustered_metrics"])
    clustered_methods = {row["method"] for row in clustered}
    assert {"metadata_routing", DIRECT_METHOD, SAFE_METHOD, RANDOM_CONTROL_METHOD} <= clustered_methods
    assert all(int(float(row["n_effective_domains"])) == 5 for row in clustered)


def test_source_utility_transfer_audits_nested_scope_and_pseudo_query_exclusion(tmp_path: Path) -> None:
    true_nelbo, sample_domains, metadata_similarity, expert_domains, domain_to_idx = _fixture()
    outputs = evaluate_source_utility_transfer(
        true_nelbo=true_nelbo,
        sample_domains=sample_domains,
        metadata_similarity=metadata_similarity,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        seed=42,
        cfg=_cfg(),
        reports_dir=tmp_path,
    )

    feature_rows = _read_csv(tmp_path / outputs.artifacts["source_utility_transfer_feature_audit"])
    assert feature_rows
    for row in feature_rows:
        assert int(row["forbidden_target_eval"]) == 0
        assert int(row["strict_source_only"]) == 1
        profile_domains = {int(v) for v in row["profile_domains"].split("|") if v}
        assert int(row["outer_target_domain"]) not in profile_domains
        if row["split_role"] != "target":
            assert int(row["query_domain"]) not in profile_domains
            assert row["feature_source_scope"].startswith(row["split_role"])
        else:
            assert row["feature_source_scope"] == "target_metadata|source_utility_profile"
        assert "forbidden_target_eval" not in row["feature_source_scope"]

    threshold_rows = _read_csv(tmp_path / outputs.artifacts["source_utility_transfer_threshold_audit"])
    assert threshold_rows
    assert {row["selection_source"] for row in threshold_rows} == {"nested_source_inner_domain_aggregate"}
    selected_by_outer: dict[int, int] = {}
    for row in threshold_rows:
        outer = int(row["outer_target_domain"])
        selected_by_outer[outer] = selected_by_outer.get(outer, 0) + int(row["selected_threshold"])
        assert int(row["n_inner_validation_domains"]) == 4
    assert selected_by_outer == {domain: 1 for domain in expert_domains}


def test_random_control_matches_safe_override_coverage_per_outer_domain(tmp_path: Path) -> None:
    true_nelbo, sample_domains, metadata_similarity, expert_domains, domain_to_idx = _fixture()
    outputs = evaluate_source_utility_transfer(
        true_nelbo=true_nelbo,
        sample_domains=sample_domains,
        metadata_similarity=metadata_similarity,
        expert_domains=expert_domains,
        domain_to_idx=domain_to_idx,
        seed=42,
        cfg=_cfg(),
        reports_dir=tmp_path,
    )

    safe_by_domain = {
        int(row["query_domain"]): int(row["accepted_override"])
        for row in outputs.sample_rows
        if row["method"] == SAFE_METHOD
    }
    random_rows = [row for row in outputs.random_control_rows if row["method"] == RANDOM_CONTROL_METHOD]
    assert random_rows
    for row in random_rows:
        assert int(row["accepted_override"]) == safe_by_domain[int(row["query_domain"])]
        assert int(row["strict_source_only"]) == 1
        assert int(row["adoption_eligible"]) == 0
