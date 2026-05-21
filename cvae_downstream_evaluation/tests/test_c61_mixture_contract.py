from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c61_mixture import (  # noqa: E402
    C42_LATENT_GMM_K1_GENERATION_MODE,
    C42_LATENT_GMM_K2_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    MIN_COMPONENT_BUDGET_PER_CLASS,
    POLICY_C52_TOPK_EQUAL,
    POLICY_C52_TOPK_RANK_SOFTMAX,
    POLICY_FIXED_SAFE,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    SELECTOR_RIDGE_NO_EXPERT,
    assert_c61_prejoin_rows_safe,
    allocate_weighted_budget_per_class,
    build_c61_mixture_components,
    clip_and_normalize_weights,
    mixture_entropy_diagnostics,
    select_safe_mode_prefix,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    LATENT_GMM_PRIOR_GENERATOR_FAMILY,
)


def test_c61_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c61_cvae_mixture_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c52-artifacts-root" in result.stdout
    assert "--limit-support-sizes" in result.stdout


def test_safe_mode_prefix_preserves_min_component_budget() -> None:
    assert select_safe_mode_prefix(128, 4) == ("hetero_mean", "gmm_k1", "gmm_k2", "standard_prior")
    assert select_safe_mode_prefix(64, 4) == ("hetero_mean", "gmm_k1")
    try:
        select_safe_mode_prefix(16, 4)
    except ProtocolError:
        pass
    else:
        raise AssertionError("C6.1 accepted a fragmented mixture below the minimum component budget")


def test_weighted_budget_allocation_is_deterministic_and_conserved() -> None:
    weights = {("2", "b"): 0.25, ("1", "a"): 0.50, ("3", "c"): 0.25}
    first = allocate_weighted_budget_per_class(total_per_class=10, weights=weights)
    second = allocate_weighted_budget_per_class(total_per_class=10, weights=weights)

    assert first == second
    assert sum(first.values()) == 10
    assert first[("1", "a")] == 5


def test_rank_softmax_weight_clipping_stays_within_bounds() -> None:
    clipped = clip_and_normalize_weights(
        {"a": 100.0, "b": 1.0, "c": 0.1},
        min_weight=0.05,
        max_weight=0.80,
    )

    assert round(sum(clipped.values()), 12) == 1.0
    assert all(0.05 <= value <= 0.80 for value in clipped.values())


def test_fixed_safe_components_exclude_heldout_noise_and_gmm_k4() -> None:
    components = build_c61_mixture_components(
        policy=POLICY_FIXED_SAFE,
        candidates=("0", "1", "2", "3"),
        total_budget_per_class=128,
        experiment_seed=42,
        heldout_center="0",
        support_size=4,
        support_seed=17,
    )

    assert components
    assert all(component.source_expert != "0" for component in components)
    assert {component.bank.mode_label for component in components} == {
        "hetero_mean",
        "gmm_k1",
        "gmm_k2",
        "standard_prior",
    }
    assert min(component.allocated_budget_per_class for component in components) >= MIN_COMPONENT_BUDGET_PER_CLASS
    assert "gmm_k4" not in {component.bank.mode_label for component in components}
    assert "hetero_noise" not in {component.bank.mode_label for component in components}


def test_c52_topk_components_use_no_expert_prejoin_ranks_and_safe_modes_only() -> None:
    rows = [
        _prejoin("1", POSTERIOR_DECODER_MEAN_GENERATION_MODE, "hetero_mean", HETEROSCEDASTIC_GENERATOR_FAMILY, 2),
        _prejoin("2", C42_LATENT_GMM_K1_GENERATION_MODE, "gmm_k1", LATENT_GMM_PRIOR_GENERATOR_FAMILY, 1),
        _prejoin("3", C42_LATENT_GMM_K2_GENERATION_MODE, "gmm_k2", LATENT_GMM_PRIOR_GENERATOR_FAMILY, 3),
        _prejoin("4", C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE, "standard_prior", LATENT_GMM_PRIOR_GENERATOR_FAMILY, 4),
        _prejoin("0", C42_LATENT_GMM_K1_GENERATION_MODE, "gmm_k1", LATENT_GMM_PRIOR_GENERATOR_FAMILY, 0),
        _prejoin("1", "latent_gmm_k4_decoder_mean", "gmm_k4", LATENT_GMM_PRIOR_GENERATOR_FAMILY, 0),
        {**_prejoin("1", C42_LATENT_GMM_K1_GENERATION_MODE, "gmm_k1", LATENT_GMM_PRIOR_GENERATOR_FAMILY, 0), "selector_name": "other_selector"},
    ]

    equal = build_c61_mixture_components(
        policy=POLICY_C52_TOPK_EQUAL,
        candidates=("0", "1", "2", "3", "4"),
        total_budget_per_class=128,
        experiment_seed=42,
        heldout_center="0",
        support_size=4,
        support_seed=17,
        c52_prejoin_scores=rows,
    )
    weighted = build_c61_mixture_components(
        policy=POLICY_C52_TOPK_RANK_SOFTMAX,
        candidates=("0", "1", "2", "3", "4"),
        total_budget_per_class=128,
        experiment_seed=42,
        heldout_center="0",
        support_size=4,
        support_seed=17,
        c52_prejoin_scores=rows,
    )

    assert sorted((component.source_expert, component.bank.mode_label) for component in equal) == [
        ("1", "hetero_mean"),
        ("2", "gmm_k1"),
        ("3", "gmm_k2"),
    ]
    assert sum(component.allocated_budget_per_class for component in equal) == 128
    assert sum(component.allocated_budget_per_class for component in weighted) == 128
    assert max(component.desired_weight for component in weighted) <= 0.40


def test_mixture_entropy_diagnostics_distinguish_sparse_and_broad_mixtures() -> None:
    broad = build_c61_mixture_components(
        policy=POLICY_FIXED_SAFE,
        candidates=("0", "1", "2", "3", "4"),
        total_budget_per_class=128,
        experiment_seed=42,
        heldout_center="0",
        support_size=4,
        support_seed=17,
    )
    entropy = mixture_entropy_diagnostics(broad)

    assert entropy["num_components"] == 16.0
    assert entropy["effective_num_components"] > 10.0
    assert entropy["component_budget_min"] >= MIN_COMPONENT_BUDGET_PER_CLASS


def test_c61_prejoin_guard_rejects_utility_columns() -> None:
    assert_c61_prejoin_rows_safe([{"component_key": "expert_1::hetero_mean"}])
    try:
        assert_c61_prejoin_rows_safe([{"component_key": "x", "selected_bacc": 0.9}])
    except ProtocolError:
        pass
    else:
        raise AssertionError("C6.1 pre-join component guard accepted downstream utility")


def _prejoin(
    expert: str,
    mode: str,
    mode_label: str,
    family: str,
    rank: int,
) -> dict[str, object]:
    return {
        "selector_name": SELECTOR_RIDGE_NO_EXPERT,
        "experiment_seed": 42,
        "heldout_center": "0",
        "support_size": 4,
        "support_seed": 17,
        "candidate_expert": expert,
        "generator_family": family,
        "generation_mode": mode,
        "mode_label": mode_label,
        "predicted_utility_score": 1.0 / (rank + 1),
        "predicted_rank_within_unit": rank,
    }
