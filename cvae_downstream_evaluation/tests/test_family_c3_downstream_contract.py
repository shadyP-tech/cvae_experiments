from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.family_c import FamilyCDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.family_c3 import (  # noqa: E402
    FAMILY_C3_BOOTSTRAP_MU_MODE,
    FAMILY_C3_BOOTSTRAP_T1_MODE,
    FAMILY_C3_GMM_MODE,
    FAMILY_C3_SOURCE_TRANSFER_METHOD,
    PosteriorBank,
    _duplicate_rate,
    assert_family_c3_config_text,
    build_c3_fixed_expert_generation_mode_comparison_rows,
    build_c3_selected_policy_comparison_rows,
    build_c3_source_transfer_sampler_prior_audit_rows,
    build_c3_source_transfer_selection_alignment_rows,
    fit_gmm_prior_from_mu,
    fit_posterior_bank_from_arrays,
    load_family_c3_downstream_config,
    sample_posterior_bootstrap_embeddings,
    valid_gmm_k_candidates,
)
from cvae_downstream_evaluation.family_c2 import FAMILY_C2_PRIMARY_GENERATION_MODE  # noqa: E402
from cvae_downstream_evaluation.schemas import SINGLE_EXPERT_ROW_TYPE  # noqa: E402


def test_family_c3_config_validates() -> None:
    config_path = ROOT / "configs" / "experiments" / "family_c3_rich_latent_sampler_downstream_v1.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert_family_c3_config_text(text)
    config = load_family_c3_downstream_config(config_path)
    assert config.generation_modes == (
        FAMILY_C3_BOOTSTRAP_MU_MODE,
        FAMILY_C3_BOOTSTRAP_T1_MODE,
        FAMILY_C3_GMM_MODE,
    )
    assert config.mode_tie_break_order[0] == FAMILY_C3_BOOTSTRAP_MU_MODE


def test_posterior_bank_min_count_marks_unavailable() -> None:
    np = pytest.importorskip("numpy")
    fitted = fit_posterior_bank_from_arrays(
        np.zeros((1, 2)),
        np.zeros((1, 2)),
        min_count=2,
    )
    assert fitted["available"] == 0
    assert fitted["n_source_train"] == 1


def test_bootstrap_mu_sampling_is_deterministic_and_uses_means_exactly() -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    backend = _FakeBackend(torch.device("cpu"))
    banks = {
        ("2", 0): PosteriorBank("2", 0, np.asarray([[0.0, 1.0], [2.0, 3.0]]), np.ones((2, 2)), 2, 1, {}, {}),
        ("2", 1): PosteriorBank("2", 1, np.asarray([[4.0, 5.0], [6.0, 7.0]]), np.ones((2, 2)), 2, 1, {}, {}),
    }
    a, a_stats = sample_posterior_bootstrap_embeddings(
        backend,
        banks,
        expert_domain=2,
        generation_seed=17,
        budget_per_class=2,
        label_values=(0, 1),
        temperature=0.0,
        generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE,
    )
    b, _ = sample_posterior_bootstrap_embeddings(
        backend,
        banks,
        expert_domain=2,
        generation_seed=17,
        budget_per_class=2,
        label_values=(0, 1),
        temperature=0.0,
        generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE,
    )
    ax = np.concatenate([np.asarray(chunk) for chunk in a.embeddings], axis=0)
    bx = np.concatenate([np.asarray(chunk) for chunk in b.embeddings], axis=0)
    assert np.allclose(ax, bx)
    assert list(a.labels) == [0, 0, 1, 1]
    assert a_stats["latent_sample_norm_mean"] > 0
    assert backend.last_z_values
    for z in backend.last_z_values:
        assert any(np.allclose(z, mu) for bank in banks.values() for mu in np.asarray(bank.mu))


def test_gmm_valid_k_candidates_precede_bic_selection() -> None:
    assert valid_gmm_k_candidates(31, (1, 2, 4)) == ()
    assert valid_gmm_k_candidates(32, (1, 2, 4)) == (1, 2, 4)


def test_gmm_fit_reports_convergence_and_component_weight() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    mu = np.concatenate(
        [
            rng.normal(loc=-2.0, scale=0.1, size=(40, 2)),
            rng.normal(loc=2.0, scale=0.1, size=(40, 2)),
        ],
        axis=0,
    )
    fitted = fit_gmm_prior_from_mu(
        mu,
        k_candidates=(1, 2, 4),
        reg_covar=1e-4,
        min_component_weight=1e-6,
        random_state=0,
    )
    assert fitted["available"] == 1
    assert fitted["diagnostics"]["gmm_converged"] == 1
    assert fitted["diagnostics"]["gmm_min_component_weight"] > 0
    assert fitted["diagnostics"]["gmm_selected_k"] in {1, 2, 4}


def test_duplicate_rate_uses_nearest_neighbor_eps() -> None:
    np = pytest.importorskip("numpy")
    x = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=float)
    assert _duplicate_rate(x, eps=1e-6) == pytest.approx(2 / 3)


def test_source_transfer_sampler_prior_excludes_target_and_self_rows() -> None:
    rows = []
    for heldout in ["0", "1", "2", "3", "4"]:
        for candidate in ["0", "1", "2", "3", "4"]:
            if candidate == heldout:
                continue
            rows.append(_row(heldout_center=heldout, candidate_expert=candidate, bacc=0.5 + 0.05 * int(candidate)))
    audit = build_c3_source_transfer_sampler_prior_audit_rows(
        downstream_rows=rows,
        generation_modes=(FAMILY_C3_BOOTSTRAP_MU_MODE,),
        mode_tie_break_order=(FAMILY_C3_BOOTSTRAP_MU_MODE,),
    )
    selected_rows = [r for r in audit if r["heldout_center"] == "0" and r["candidate_expert"] == r["selected_expert"]]
    assert selected_rows
    for row in audit:
        assert row["target_heldout_rows_used_for_sampler_prior"] == 0
        assert row["target_mode_oracle_used_for_selection"] == 0
        assert row["target_expert_oracle_used_for_selection"] == 0
        assert row["n_source_centers_used"] == 3


def test_alignment_uses_all_mode_oracle_without_oracle_for_selection() -> None:
    rows = [
        _row(heldout_center="0", candidate_expert="1", bacc=0.7, generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE),
        _row(heldout_center="0", candidate_expert="2", bacc=0.8, generation_mode=FAMILY_C3_BOOTSTRAP_T1_MODE),
        _row(heldout_center="1", candidate_expert="2", bacc=0.9, generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE),
        _row(heldout_center="2", candidate_expert="1", bacc=0.9, generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE),
        _row(heldout_center="3", candidate_expert="1", bacc=0.9, generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE),
        _row(heldout_center="4", candidate_expert="1", bacc=0.9, generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE),
    ]
    audit = build_c3_source_transfer_sampler_prior_audit_rows(
        downstream_rows=rows,
        generation_modes=(FAMILY_C3_BOOTSTRAP_MU_MODE, FAMILY_C3_BOOTSTRAP_T1_MODE),
        mode_tie_break_order=(FAMILY_C3_BOOTSTRAP_MU_MODE, FAMILY_C3_BOOTSTRAP_T1_MODE),
        min_required_source_centers=1,
    )
    alignment = build_c3_source_transfer_selection_alignment_rows(
        source_transfer_audit_rows=audit,
        downstream_rows=rows,
    )
    row0 = next(r for r in alignment if r["heldout_center"] == "0")
    assert row0["method"] == FAMILY_C3_SOURCE_TRANSFER_METHOD
    assert row0["downstream_oracle_expert"] == "2"
    assert row0["downstream_oracle_generation_mode"] == FAMILY_C3_BOOTSTRAP_T1_MODE


def test_comparisons_keep_fixed_expert_and_selected_policy_semantics() -> None:
    c2_row = _row(candidate_expert="1", bacc=0.7, generation_mode=FAMILY_C2_PRIMARY_GENERATION_MODE)
    c3_row = _row(candidate_expert="1", bacc=0.8, generation_mode=FAMILY_C3_BOOTSTRAP_MU_MODE)
    fixed = build_c3_fixed_expert_generation_mode_comparison_rows(c2_rows=[c2_row], c3_rows=[c3_row])
    assert fixed[0]["delta_bacc_c3_minus_c2"] == pytest.approx(0.1)

    selected = build_c3_selected_policy_comparison_rows(
        c2_alignment_rows=[
            {
                "heldout_center": "0",
                "method": "family_c_source_transfer_downstream_prior",
                "selected_expert": "1",
                "generation_mode": FAMILY_C2_PRIMARY_GENERATION_MODE,
                "generation_seed": "17",
                "classifier_seed": "17",
                "support_size": "4",
                "support_seed": "17",
                "support_eval_split_id": "split",
                "selected_bacc": "0.7",
                "downstream_oracle_gap_bacc": "0.1",
            }
        ],
        c3_alignment_rows=[
            {
                "heldout_center": "0",
                "method": FAMILY_C3_SOURCE_TRANSFER_METHOD,
                "selected_expert": "2",
                "selected_generation_mode": FAMILY_C3_BOOTSTRAP_MU_MODE,
                "generation_seed": 17,
                "classifier_seed": 17,
                "support_size": 4,
                "support_seed": 17,
                "support_eval_split_id": "split",
                "selected_bacc": 0.8,
                "downstream_oracle_gap_bacc": 0.0,
            }
        ],
    )
    assert selected[0]["c2_selected_expert"] == "1"
    assert selected[0]["c3_selected_expert"] == "2"


def _row(
    *,
    heldout_center: str = "0",
    candidate_expert: str = "1",
    bacc: float = 0.7,
    macro_f1: float = 0.6,
    generation_mode: str = FAMILY_C3_BOOTSTRAP_MU_MODE,
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
        support_eval_split_id="split",
        eval_n=10,
        eval_class_counts="0:5|1:5",
        target_eval_n_class0=5,
        target_eval_n_class1=5,
        target_eval_min_class_count=5,
        metric_valid_bacc=1,
        metric_valid_macro_f1=1,
        bacc=bacc,
        macro_f1=macro_f1,
        row_type=SINGLE_EXPERT_ROW_TYPE,
    )


class _FakeBackend:
    def __init__(self, device: object) -> None:
        self.device = device
        self.class_condition_dim = 2
        self.models = {2: _FakeModel(self)}
        self.last_z_values = []


class _FakeModel:
    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend

    def decode(self, z: object, *, y: object) -> object:
        _ = y
        arr = z.detach().cpu().numpy()
        self.backend.last_z_values.extend([row.copy() for row in arr])
        return z
