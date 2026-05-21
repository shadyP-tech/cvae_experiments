from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c52_utility_ranker import (  # noqa: E402
    NOISE_GENERATION_MODE,
    PRIMARY_SELECTOR,
    SELECTOR_RIDGE_NO_EXPERT,
    assert_prejoin_rows_safe,
    build_c52_prejoin_predictions,
    build_c52_router_examples,
    build_c52_selected_routes_prejoin,
    build_c52_utility_join,
)
from cvae_downstream_evaluation.downstream import CandidateDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
)


MEAN_MODE = "posterior_sample_decoder_mean"


def test_c52_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c52_utility_rank_router.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c51-artifacts-root" in result.stdout
    assert "--limit-support-sizes" in result.stdout


def test_c52_examples_normalize_support_nelbo_and_exclude_noise() -> None:
    score_rows = [
        _score_row("0", "1", MEAN_MODE, 17, support_nelbo=2.0),
        _score_row("0", "1", NOISE_GENERATION_MODE, 17, support_nelbo=2.0),
        _score_row("0", "2", MEAN_MODE, 17, support_nelbo=4.0),
    ]
    downstream = [
        _candidate_row("0", "1", MEAN_MODE, 17, 17, 0.70),
        _candidate_row("0", "2", MEAN_MODE, 17, 17, 0.60),
    ]

    examples = build_c52_router_examples(score_rows, downstream)
    noise = [row for row in examples if row["generation_mode"] == NOISE_GENERATION_MODE][0]
    best = [row for row in examples if row["candidate_expert"] == "1" and row["generation_mode"] == MEAN_MODE][0]

    assert noise["primary_candidate_eligible"] == 0
    assert best["primary_candidate_eligible"] == 1
    assert best["support_nelbo_delta_to_best_within_unit"] == 0.0
    assert best["utility_label_bacc"] == 0.70


def test_c52_prejoin_guard_rejects_current_heldout_utility_injection() -> None:
    assert_prejoin_rows_safe(
        [
            {
                "predicted_utility_score": 0.7,
                "current_heldout_utility_visible_before_selection": 0,
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
            }
        ]
    )
    try:
        assert_prejoin_rows_safe([{"current_heldout_bacc": 0.9, "predicted_utility_score": 0.7}])
    except ProtocolError:
        pass
    else:
        raise AssertionError("C5.2 pre-join guard accepted current-heldout BACC leakage")


def test_c52_loco_predictions_exclude_current_center_and_include_no_expert_ablation() -> None:
    examples = []
    for center, label_shift in (("0", 0.0), ("1", 0.05), ("2", 0.10)):
        examples.append(_example(center, "1", MEAN_MODE, support_rank=1, bacc=0.72 + label_shift))
        examples.append(_example(center, "2", MEAN_MODE, support_rank=2, bacc=0.62 + label_shift))

    prejoin, audit = build_c52_prejoin_predictions(examples)

    assert prejoin
    assert {row["selector_name"] for row in prejoin}.issuperset({PRIMARY_SELECTOR, SELECTOR_RIDGE_NO_EXPERT})
    assert all(int(row["current_heldout_utility_visible_before_selection"]) == 0 for row in prejoin)
    assert all(int(row["current_heldout_center_in_training"]) == 0 for row in audit)
    assert all(str(row["generation_mode"]) != NOISE_GENERATION_MODE for row in prejoin)


def test_c52_selection_joins_utility_after_route_is_frozen() -> None:
    examples = []
    for center, label_shift in (("0", 0.0), ("1", 0.05), ("2", 0.10)):
        examples.append(_example(center, "1", MEAN_MODE, support_rank=1, bacc=0.72 + label_shift))
        examples.append(_example(center, "2", MEAN_MODE, support_rank=2, bacc=0.62 + label_shift))
    prejoin, _audit = build_c52_prejoin_predictions(examples)
    selected = build_c52_selected_routes_prejoin(prejoin)

    assert selected
    assert all("selected_bacc_mean" not in row for row in selected)

    joined = build_c52_utility_join(selected, prejoin, examples, downstream_hash="abc.def")
    primary = [row for row in joined if row["selector_name"] == PRIMARY_SELECTOR]

    assert primary
    assert all(float(row["selected_bacc_mean"]) >= 0.62 for row in primary)
    assert all(row["protocol_status"] == "pass" for row in primary)
    assert all(float(row["candidate_scores_available_before_selection"]) == 2 for row in primary)


def _score_row(
    heldout: str,
    expert: str,
    mode: str,
    generation_seed: int,
    *,
    support_nelbo: float,
) -> dict[str, object]:
    family = HETEROSCEDASTIC_GENERATOR_FAMILY if mode == NOISE_GENERATION_MODE else PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY
    return {
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": f"target{heldout}_seed17_random_k4",
        "candidate_expert": expert,
        "generator_family": family,
        "generation_mode": mode,
        "mode_label": mode,
        "generation_seed": generation_seed,
        "support_nelbo": support_nelbo,
        "support_nelbo_rank": int(float(support_nelbo)),
        "metadata_selected_expert": "1",
        "source_global_selected_expert": "2",
        "rankmean_dino_score": support_nelbo,
        "zsum_dino_score": support_nelbo,
        "rankmean_pca_score": support_nelbo,
        "rank_energy_dino": support_nelbo,
        "rank_rbf_mmd_dino": support_nelbo,
        "rank_mean_l2_dino": support_nelbo,
        "rank_cov_trace_dino": support_nelbo,
        "rank_pairwise_dino": support_nelbo,
    }


def _candidate_row(
    heldout: str,
    expert: str,
    mode: str,
    generation_seed: int,
    classifier_seed: int,
    bacc: float,
) -> CandidateDownstreamRow:
    family = HETEROSCEDASTIC_GENERATOR_FAMILY if mode == NOISE_GENERATION_MODE else PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY
    return CandidateDownstreamRow(
        experiment_seed=42,
        heldout_center=heldout,
        support_size=4,
        support_seed=17,
        candidate_expert=expert,
        generator_family=family,
        generation_mode=mode,
        budget_per_class=128,
        generation_seed=generation_seed,
        classifier_seed=classifier_seed,
        bacc=bacc,
        macro_f1=bacc,
    )


def _example(center: str, expert: str, mode: str, *, support_rank: int, bacc: float) -> dict[str, object]:
    return {
        "experiment_seed": 42,
        "heldout_center": center,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": f"target{center}_seed17_random_k4",
        "candidate_expert": expert,
        "generator_family": PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
        "generation_mode": mode,
        "mode_label": mode,
        "primary_candidate_eligible": int(mode != NOISE_GENERATION_MODE),
        "diagnostic_only": int(mode == NOISE_GENERATION_MODE),
        "generation_seed_count": 2,
        "support_nelbo_mean": float(support_rank),
        "support_nelbo_rank_within_unit": support_rank,
        "support_nelbo_z_within_unit": float(support_rank - 1),
        "support_nelbo_delta_to_best_within_unit": float(support_rank - 1),
        "metadata_match": int(expert == "1"),
        "source_global_match": int(expert == "2"),
        "rankmean_dino_median": float(support_rank),
        "rankmean_dino_mean": float(support_rank),
        "rankmean_dino_std": 0.0,
        "rankmean_dino_iqr": 0.0,
        "zsum_dino_median": float(support_rank),
        "rankmean_pca_median": float(support_rank),
        "rank_energy_dino_median": float(support_rank),
        "rank_rbf_mmd_dino_median": float(support_rank),
        "rank_mean_l2_dino_median": float(support_rank),
        "rank_cov_trace_dino_median": float(support_rank),
        "rank_pairwise_dino_median": float(support_rank),
        "utility_label_bacc": bacc,
        "utility_label_bacc_std": 0.01,
        "utility_label_bacc_min": bacc - 0.01,
        "utility_label_bacc_max": bacc + 0.01,
        "utility_label_ge_080_rate": float(bacc >= 0.80),
        "utility_label_macro_f1": bacc,
        "utility_label_generation_seed_count": 2,
        "utility_label_classifier_seed_count": 2,
    }
