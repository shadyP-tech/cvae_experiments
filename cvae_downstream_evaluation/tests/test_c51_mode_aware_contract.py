from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c51_mode_aware import (  # noqa: E402
    PRIMARY_SELECTOR,
    SELECTOR_SEED_SELECTED,
    _add_rank_scores,
    _assert_selector_score_rows_safe,
    _select_candidate,
    _support_condition_bandwidths,
    _unlabeled_support_split,
    build_c51_alignment_rows,
)
from cvae_downstream_evaluation.downstream import CandidateDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
)


def test_c51_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c51_mode_aware_support_selector.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c41-artifacts-root" in result.stdout
    assert "--build-reports-only" in result.stdout


def test_unlabeled_support_split_uses_no_support_labels() -> None:
    metadata = tuple({"center": 0, "sample_id": f"s{i}", "label": i % 2} for i in range(10))

    split = _unlabeled_support_split(
        test_metadata=metadata,
        target_indices=tuple(range(10)),
        heldout_center="0",
        support_size=4,
        support_seed=17,
    )

    assert split.support_size_actual == 4
    assert split.support_labels_used == 0
    assert split.support_eval_split_id == "target0_seed17_random_k4"


def test_bandwidth_and_rank_scores_are_deterministic() -> None:
    support = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    synthetic = torch.tensor([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]])

    first = _support_condition_bandwidths(support, [synthetic])
    second = _support_condition_bandwidths(support, [synthetic])

    assert first == second
    assert first[0] < first[1] < first[2]

    rows = [
        _score_row("1", 17, energy=0.1, mmd=0.2, mean=0.1, cov=0.1, pair=0.1),
        _score_row("1", 23, energy=0.3, mmd=0.4, mean=0.3, cov=0.3, pair=0.3),
        _score_row("2", 17, energy=0.2, mmd=0.3, mean=0.2, cov=0.2, pair=0.2),
    ]
    _add_rank_scores(rows, "dino")
    _add_rank_scores(rows, "pca")

    assert rows[0]["rankmean_dino_score"] < rows[2]["rankmean_dino_score"] < rows[1]["rankmean_dino_score"]


def test_seed_marginal_and_seed_selected_choose_different_granularity() -> None:
    rows = [
        _score_row("1", 17, energy=1.0, mmd=1.0, mean=1.0, cov=1.0, pair=1.0),
        _score_row("1", 23, energy=100.0, mmd=100.0, mean=100.0, cov=100.0, pair=100.0),
        _score_row("1", 31, energy=101.0, mmd=101.0, mean=101.0, cov=101.0, pair=101.0),
        _score_row("2", 17, energy=40.0, mmd=40.0, mean=40.0, cov=40.0, pair=40.0),
        _score_row("2", 23, energy=41.0, mmd=41.0, mean=41.0, cov=41.0, pair=41.0),
        _score_row("2", 31, energy=42.0, mmd=42.0, mean=42.0, cov=42.0, pair=42.0),
    ]
    _add_rank_scores(rows, "dino")
    _add_rank_scores(rows, "pca")

    seed_selected = _select_candidate(rows, SELECTOR_SEED_SELECTED)
    seed_marginal = _select_candidate(rows, PRIMARY_SELECTOR)

    assert seed_selected["candidate_expert"] == "1"
    assert int(seed_selected["generation_seed"]) == 17
    assert seed_marginal["candidate_expert"] == "2"


def test_selector_score_guard_rejects_downstream_utility_columns() -> None:
    try:
        _assert_selector_score_rows_safe([{"selected_bacc": 0.9, "rankmean_dino_score": 1.0}])
    except ProtocolError:
        pass
    else:
        raise AssertionError("C5.1 selector guard accepted downstream utility as selector input")


def test_c51_alignment_joins_selected_candidate_to_all_classifier_seeds() -> None:
    score_rows = [
        _score_row("1", 17, energy=0.1, mmd=0.1, mean=0.1, cov=0.1, pair=0.1),
        _score_row("1", 23, energy=0.2, mmd=0.2, mean=0.2, cov=0.2, pair=0.2),
        _score_row("2", 17, energy=1.0, mmd=1.0, mean=1.0, cov=1.0, pair=1.0),
        _score_row("2", 23, energy=1.1, mmd=1.1, mean=1.1, cov=1.1, pair=1.1),
    ]
    for prefix in ("dino", "pca"):
        _add_rank_scores(score_rows, prefix)
    downstream = [
        _candidate_row("1", 17, 17, 0.70),
        _candidate_row("1", 17, 23, 0.72),
        _candidate_row("1", 23, 17, 0.74),
        _candidate_row("1", 23, 23, 0.76),
        _candidate_row("2", 17, 17, 0.60),
        _candidate_row("2", 17, 23, 0.61),
        _candidate_row("2", 23, 17, 0.62),
        _candidate_row("2", 23, 23, 0.63),
    ]

    alignment = build_c51_alignment_rows(score_rows, downstream)
    primary = [row for row in alignment if row["selector_name"] == PRIMARY_SELECTOR][0]
    seed_selected = [row for row in alignment if row["selector_name"] == SELECTOR_SEED_SELECTED][0]

    assert primary["selected_expert"] == "1"
    assert primary["selected_generation_seed"] == ""
    assert round(float(primary["selected_bacc_mean"]), 2) == 0.73
    assert seed_selected["selected_generation_seed"] == 17
    assert round(float(seed_selected["selected_bacc_mean"]), 2) == 0.71


def _score_row(
    expert: str,
    generation_seed: int,
    *,
    energy: float,
    mmd: float,
    mean: float,
    cov: float,
    pair: float,
) -> dict[str, object]:
    return {
        "experiment_seed": 42,
        "heldout_center": "0",
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": "target0_seed17_random_k4",
        "candidate_expert": expert,
        "generator_family": HETEROSCEDASTIC_GENERATOR_FAMILY,
        "generation_mode": POSTERIOR_DECODER_MEAN_GENERATION_MODE,
        "mode_label": "hetero_mean",
        "generation_seed": generation_seed,
        "support_nelbo": float(expert),
        "support_nelbo_rank": int(expert),
        "metadata_selected_expert": "1",
        "energy_distance_dino": energy,
        "rbf_mmd_dino_sigma05": mmd,
        "rbf_mmd_dino_sigma10": mmd,
        "rbf_mmd_dino_sigma20": mmd,
        "mean_l2_dino": mean,
        "abs_log_cov_trace_ratio_dino": cov,
        "abs_log_pairwise_distance_ratio_dino": pair,
        "energy_distance_pca": energy,
        "rbf_mmd_pca_sigma05": mmd,
        "rbf_mmd_pca_sigma10": mmd,
        "rbf_mmd_pca_sigma20": mmd,
        "mean_l2_pca": mean,
        "abs_log_cov_trace_ratio_pca": cov,
        "abs_log_pairwise_distance_ratio_pca": pair,
    }


def _candidate_row(expert: str, generation_seed: int, classifier_seed: int, bacc: float) -> CandidateDownstreamRow:
    return CandidateDownstreamRow(
        experiment_seed=42,
        heldout_center="0",
        support_size=4,
        support_seed=17,
        candidate_expert=expert,
        generator_family=HETEROSCEDASTIC_GENERATOR_FAMILY,
        generation_mode=POSTERIOR_DECODER_MEAN_GENERATION_MODE,
        budget_per_class=128,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        bacc=bacc,
        macro_f1=bacc,
    )
