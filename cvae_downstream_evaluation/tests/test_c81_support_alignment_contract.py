from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.c81_support_alignment import (  # noqa: E402
    ALIGN_DIAG_ALPHA05,
    ALIGN_IDENTITY,
    ALIGN_MEAN_ONLY,
    ALIGNMENT_COLUMNS,
    MATRIX_COLUMNS,
    build_unlabeled_support_eval_pools,
    fit_alignment_transform,
    hash_member_bank,
)


def test_c81_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_c81_support_alignment_geometric_ensemble.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--c63-artifacts-root" in result.stdout
    assert "--enable-full-coral-diagnostic" in result.stdout


def test_c81_diagvar_alignment_uses_explicit_shrinkage_formula() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    synthetic = np.asarray([[float(i), float(i * 2)] for i in range(8)], dtype=float)
    support = np.asarray([[10.0 + float(i), 20.0 + float(i * 3)] for i in range(8)], dtype=float)
    transform, bootstrap = fit_alignment_transform(
        synthetic,
        support,
        policy=ALIGN_DIAG_ALPHA05,
        alpha=0.5,
        bootstrap_seed=1,
    )

    mu_s = synthetic.mean(axis=0)
    mu_t = support.mean(axis=0)
    var_s = synthetic.var(axis=0)
    var_t = support.var(axis=0)
    expected_mu = 0.5 * mu_s + 0.5 * mu_t
    expected_var = 0.5 * var_s + 0.5 * var_t
    expected_scale = np.clip(np.sqrt(expected_var / np.maximum(var_s, 1.0e-4)), 0.5, 2.0)

    assert transform.policy_applied == ALIGN_DIAG_ALPHA05
    assert np.allclose(transform.mu_a, expected_mu)
    assert np.allclose(transform.scale, expected_scale)
    assert set(bootstrap) == {
        "support_bootstrap_mean_shift_std",
        "support_bootstrap_scale_std",
        "support_bootstrap_scale_std_mean",
        "support_bootstrap_unstable_flag",
    }


def test_c81_small_support_falls_back_to_mean_only() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    synthetic = np.asarray([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dtype=float)
    support = np.asarray([[10.0, 10.0], [12.0, 14.0]], dtype=float)
    transform, _bootstrap = fit_alignment_transform(
        synthetic,
        support,
        policy=ALIGN_DIAG_ALPHA05,
        alpha=0.5,
        bootstrap_seed=1,
    )

    assert transform.policy_applied == ALIGN_MEAN_ONLY
    assert transform.fallback_policy == ALIGN_MEAN_ONLY
    assert transform.fallback_trigger == "n_support_lt_min_support_for_diagvar"


def test_c81_identity_alignment_is_noop() -> None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    synthetic = np.asarray([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]], dtype=float)
    support = np.asarray([[10.0, 10.0], [12.0, 14.0], [14.0, 18.0]], dtype=float)
    transform, _bootstrap = fit_alignment_transform(
        synthetic,
        support,
        policy=ALIGN_IDENTITY,
        alpha=0.5,
        bootstrap_seed=1,
    )

    assert transform.policy_applied == ALIGN_IDENTITY
    assert np.allclose(transform.mu_a, synthetic.mean(axis=0))
    assert np.allclose(transform.scale, 1.0)


def test_c81_support_eval_pool_is_disjoint_and_unlabeled() -> None:
    metadata = []
    for idx in range(20):
        metadata.append(
            {
                "sample_id": f"s{idx}",
                "center": "1",
                "label": idx % 2,
            }
        )

    pools = build_unlabeled_support_eval_pools(
        test_metadata=metadata,
        heldout_center="1",
        support_size=4,
        support_seed=17,
        support_eval_split_id="split",
        union_eval_indices=(),
        union_target_eval_pool_id="eval",
    )

    assert len(pools.support_indices) == 4
    assert set(pools.support_sample_ids).isdisjoint(set(pools.eval_sample_ids))
    assert pools.target_eval_pool_id == "eval"


def test_c81_member_bank_hash_uses_member_inventory_and_class_order() -> None:
    class _Bank:
        mode_label = "hetero_mean"
        generation_mode = "posterior_sample_decoder_mean"
        generator_family = "family"

    class _Spec:
        source_expert = "1"
        bank = _Bank()
        generation_seed = 17
        allocated_budget_per_class = 8
        weight = 1.0

        @property
        def member_key(self) -> str:
            return "expert_1::hetero_mean::seed_17"

    assert hash_member_bank([_Spec()]) == hash_member_bank([_Spec()])


def test_c81_audit_columns_express_train_only_alignment() -> None:
    assert "alignment_applied_to_synthetic_train_only" in MATRIX_COLUMNS
    assert "target_eval_features_transformed" in MATRIX_COLUMNS
    assert "target_eval_features_used_for_alignment" in MATRIX_COLUMNS
    assert "support_bootstrap_unstable_flag" in ALIGNMENT_COLUMNS
