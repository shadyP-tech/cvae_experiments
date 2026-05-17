from pathlib import Path
import math
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.family_c import FamilyCDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.family_c2 import (  # noqa: E402
    FAMILY_C2_PRIMARY_GENERATION_MODE,
    FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
    FittedLatentPrior,
    _generation_manifest_row_c2,
    apply_support_coral,
    assert_family_c2_config_text,
    build_c2_generation_mode_comparison_rows,
    fitted_prior_classifier_cache_key,
    fit_diagonal_latent_prior_from_arrays,
    load_family_c2_downstream_config,
    sample_fitted_latent_prior_embeddings,
)
from cvae_downstream_evaluation.schemas import SINGLE_EXPERT_ROW_TYPE  # noqa: E402


def test_family_c2_config_validates() -> None:
    config_path = ROOT / "configs" / "experiments" / "family_c2_fitted_latent_prior_downstream_v1.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert_family_c2_config_text(text)
    config = load_family_c2_downstream_config(config_path)
    assert config.min_source_train_per_class_for_prior == 16
    assert config.label_values == (0, 1)


def test_fitted_prior_variance_formula_and_clipping() -> None:
    np = pytest.importorskip("numpy")
    mu = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    logvar = np.log(np.asarray([[0.01, 100.0], [0.01, 100.0]], dtype=float))
    fitted = fit_diagonal_latent_prior_from_arrays(
        mu,
        logvar,
        min_count=2,
        var_clip_min=0.1,
        var_clip_max=25.0,
    )
    assert fitted["available"] == 1
    assert np.allclose(fitted["mean"], [1.0, 2.0])
    assert np.allclose(fitted["var"], [1.01, 25.0])
    assert fitted["num_var_clipped_high"] == 1
    assert fitted["num_var_clipped_low"] == 0


def test_fitted_prior_min_count_marks_unavailable() -> None:
    np = pytest.importorskip("numpy")
    fitted = fit_diagonal_latent_prior_from_arrays(
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        min_count=2,
        var_clip_min=0.0001,
        var_clip_max=25.0,
    )
    assert fitted["available"] == 0
    assert fitted["n_source_train"] == 1


def test_fitted_prior_sampling_is_deterministic_and_labels_match_conditions() -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    backend = _FakeBackend(torch.device("cpu"))
    priors = {
        ("2", 0): FittedLatentPrior("2", 0, np.zeros(2), np.ones(2), 20, 1, {}, {}),
        ("2", 1): FittedLatentPrior("2", 1, np.ones(2), np.ones(2), 20, 1, {}, {}),
    }
    a = sample_fitted_latent_prior_embeddings(
        backend,
        priors,
        expert_domain=2,
        generation_seed=17,
        budget_per_class=3,
        label_values=(0, 1),
    )
    b = sample_fitted_latent_prior_embeddings(
        backend,
        priors,
        expert_domain=2,
        generation_seed=17,
        budget_per_class=3,
        label_values=(0, 1),
    )
    ax = np.concatenate([np.asarray(chunk) for chunk in a.embeddings], axis=0)
    bx = np.concatenate([np.asarray(chunk) for chunk in b.embeddings], axis=0)
    assert np.allclose(ax, bx)
    assert list(a.labels) == [0, 0, 0, 1, 1, 1]


def test_generated_embedding_sanity_fields_catch_nan_and_inf() -> None:
    np = pytest.importorskip("numpy")
    from cvae_downstream_evaluation.generation import SyntheticBatch

    batch = SyntheticBatch(
        expert_domain="1",
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
        projection_frame="test",
        embeddings=np.asarray([[0.0, math.nan], [math.inf, 1.0]]),
        labels=np.asarray([0, 1]),
    )
    row = _generation_manifest_row_c2("0", "1", 17, batch, real_x=np.asarray([[1.0, 1.0]]))
    assert row["generated_nan_count"] == 1
    assert row["generated_inf_count"] == 1


def test_support_coral_uses_support_only_flags_and_marks_unstable_unavailable() -> None:
    np = pytest.importorskip("numpy")
    generated = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=float)
    support = np.asarray([[10.0, 10.0], [11.0, 10.0]], dtype=float)
    aligned, audit = apply_support_coral(
        generated,
        support,
        eps=0.001,
        max_condition_number=0.0,
    )
    assert aligned.shape == generated.shape
    assert audit["target_support_labels_used_for_coral"] == 0
    assert audit["target_eval_embeddings_used_for_coral"] == 0
    assert audit["coral_transform_finite"] == 1
    assert audit["available"] == 0


def test_coral_cache_key_includes_support_split_and_primary_key_excludes_it() -> None:
    primary_a = fitted_prior_classifier_cache_key(
        heldout_center="0",
        candidate_expert="1",
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
        support_size=4,
        support_seed=17,
        support_eval_split_id="a",
    )
    primary_b = fitted_prior_classifier_cache_key(
        heldout_center="0",
        candidate_expert="1",
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
        support_size=8,
        support_seed=23,
        support_eval_split_id="b",
    )
    coral_a = fitted_prior_classifier_cache_key(
        heldout_center="0",
        candidate_expert="1",
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
        support_size=4,
        support_seed=17,
        support_eval_split_id="a",
    )
    coral_b = fitted_prior_classifier_cache_key(
        heldout_center="0",
        candidate_expert="1",
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=FAMILY_C2_SUPPORT_CORAL_GENERATION_MODE,
        support_size=8,
        support_seed=23,
        support_eval_split_id="b",
    )
    assert primary_a == primary_b
    assert coral_a != coral_b


def test_generation_mode_comparison_pairs_same_selected_expert_and_split() -> None:
    c2_row = _row(
        heldout_center="0",
        candidate_expert="4",
        bacc=0.80,
        macro_f1=0.70,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    c2_oracle = _row(
        heldout_center="0",
        candidate_expert="1",
        bacc=0.85,
        macro_f1=0.75,
        generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    comparison = build_c2_generation_mode_comparison_rows(
        standard_alignment_rows=[
            {
                "heldout_center": "0",
                "method": "family_c_source_transfer_downstream_prior",
                "selected_expert": "4",
                "generation_seed": "17",
                "classifier_seed": "17",
                "budget_per_class": "128",
                "support_size": "4",
                "support_seed": "17",
                "support_eval_split_id": "target0_seed17_random_k4",
                "selected_bacc": "0.70",
                "selected_macro_f1": "0.60",
                "oracle_bacc": "0.75",
                "available": "1",
            }
        ],
        standard_rows=[],
        c2_rows=[c2_row, c2_oracle],
        c2_generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE,
    )
    assert len(comparison) == 1
    row = comparison[0]
    assert row["selected_expert"] == "4"
    assert math.isclose(float(row["delta_bacc_fitted_minus_standard"]), 0.10)
    assert math.isclose(float(row["oracle_bacc_fitted_prior"]), 0.85)


class _FakeModel:
    def decode(self, z, y=None):
        if y is None:
            raise ValueError("y required")
        return z + y[:, : z.shape[1]]


class _FakeBackend:
    def __init__(self, device):
        self.models = {2: _FakeModel()}
        self.device = device
        self.class_condition_dim = 2
        self.latent_dim = 2


def _row(
    *,
    heldout_center: str,
    candidate_expert: str,
    bacc: float,
    macro_f1: float,
    generation_mode: str,
) -> FamilyCDownstreamRow:
    return FamilyCDownstreamRow(
        heldout_center=heldout_center,
        candidate_expert=candidate_expert,
        generation_seed=17,
        classifier_seed=17,
        budget_per_class=128,
        generation_mode=generation_mode,
        support_size=4,
        support_seed=17,
        support_eval_split_id=f"target{heldout_center}_seed17_random_k4",
        eval_n=100,
        eval_class_counts='{"0": 50, "1": 50}',
        target_eval_n_class0=50,
        target_eval_n_class1=50,
        target_eval_min_class_count=50,
        metric_valid_bacc=1,
        metric_valid_macro_f1=1,
        bacc=bacc,
        macro_f1=macro_f1,
        row_type=SINGLE_EXPERT_ROW_TYPE,
    )
