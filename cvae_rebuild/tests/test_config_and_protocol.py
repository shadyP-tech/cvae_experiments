from pathlib import Path

import pytest
import yaml

from cvae_rebuild.config import load_config
from cvae_rebuild.decentralized_adaptive_gmm_prior import load_decentralized_adaptive_gmm_prior_config
from cvae_rebuild.decentralized_component_union_prior import (
    load_decentralized_component_union_prior_config,
    parse_decentralized_component_union_prior_config,
)
from cvae_rebuild.component_union_mass_bagged import (
    PRIMARY_BAG_MEMBERS,
    load_mass_bagged_component_union_config,
    parse_mass_bagged_component_union_config,
)
from cvae_rebuild.component_union_tailrisk_anchored_mass_bagged import (
    MULTIPANEL_PANEL_SEEDS,
    MULTIPANEL_TAILRISK_NAME,
    PRIMARY_MULTIPANEL_TAILRISK_METHOD,
    PRIMARY_TAILRISK_METHOD,
    load_multipanel_tailrisk_component_union_config,
    load_tailrisk_anchored_component_union_config,
    parse_multipanel_tailrisk_component_union_config,
    parse_tailrisk_anchored_component_union_config,
)
from cvae_rebuild.dense_reliability_tailshield_random_mass_bag import (
    PRIMARY_DENSE_TAILSHIELD_METHOD,
    load_dense_tailshield_random_mass_bag_config,
    parse_dense_tailshield_random_mass_bag_config,
)
from cvae_rebuild.source_inner_harmful_source_suppression import (
    PRIMARY_HARMFUL_SUPPRESSION_METHOD,
    load_harmful_source_suppression_config,
    parse_harmful_source_suppression_config,
)
from cvae_rebuild.target_support_regime_risk_gated_component_union import (
    PRIMARY_RISK_GATED_METHOD,
    load_target_support_regime_risk_gate_config,
    parse_target_support_regime_risk_gate_config,
)
from cvae_rebuild.labeled_support_random_vs_dense_policy_calibration import (
    PRIMARY_LABELED_SUPPORT_POLICY_METHOD,
    load_labeled_support_policy_calibration_config,
    nested_labeled_support_eval_splits,
    parse_labeled_support_policy_calibration_config,
)
from cvae_rebuild.decentralized_pruned_adaptive_equal_all4_prior import load_pruned_adaptive_equal_all4_config
from cvae_rebuild.decentralized_k16_gmm_prior import load_decentralized_k16_gmm_prior_config
from cvae_rebuild.decentralized_reliability_weighted_gmm_prior import (
    load_decentralized_reliability_weighted_gmm_prior_config,
)
from cvae_rebuild.decentralized_reliability_top3_gmm_prior import (
    load_decentralized_reliability_top3_gmm_prior_config,
)
from cvae_rebuild.decentralized_source_inner_transfer_top3_gmm_prior import (
    load_decentralized_source_inner_transfer_top3_gmm_prior_config,
)
from cvae_rebuild.decentralized_support_nelbo_reliability_gmm_prior import (
    load_decentralized_support_nelbo_reliability_gmm_prior_config,
)
from cvae_rebuild.decentralized_support8_top3_tau05_gmm_prior import (
    load_decentralized_support8_top3_tau05_gmm_prior_config,
)
from cvae_rebuild.support_calibrated_component_union_prior import (
    PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD,
    _constrained_weighted_budgets,
    _matched_shuffled_support_plan,
    _support_shrink_plan,
    load_support_calibrated_component_union_config,
    nested_unlabeled_support_eval_splits,
    parse_support_calibrated_component_union_config,
)
from cvae_rebuild.support_nelbo import SupportScore
from cvae_rebuild.paired_dense_all4_reliability_confirmation import (
    load_paired_dense_all4_reliability_config,
)
from cvae_rebuild.paired_component_coverage_audit import (
    load_paired_component_coverage_audit_config,
)
from cvae_rebuild.source_inner_validated_dense_component_hybrid import (
    load_source_inner_validated_hybrid_config,
    parse_source_inner_validated_hybrid_config,
)
from cvae_rebuild.pipeline import run_artifact_contract_smoke, run_synthetic_smoke
from cvae_rebuild.preservation import load_preservation_config
from cvae_rebuild.preservation_repair import load_repair_config
from cvae_rebuild.protocol import (
    ProtocolError,
    assert_candidate_pool,
    assert_support_labels_unused,
    build_leakage_report,
    split_budget,
)
from cvae_rebuild.reporting import REQUIRED_OUTPUTS


def test_locked_config_loads() -> None:
    cfg = load_config("cvae_rebuild/configs/target_support32_virchow2_cvae_top2_v1.yaml")
    assert cfg.primary_method == "support_nelbo_top2_geom"
    assert cfg.support_size == 32
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/target_support32_virchow2_cvae_top2_v1")


def test_locked_preservation_config_loads() -> None:
    cfg = load_preservation_config("cvae_rebuild/configs/virchow2_cvae_preservation_diagnosis_v1.yaml")
    assert cfg.name == "virchow2_cvae_preservation_diagnosis_v1"
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.classifier_class_weight == "balanced"
    assert cfg.classifier_seed is None
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/virchow2_cvae_preservation_diagnosis_v1")


def test_locked_preservation_repair_config_loads() -> None:
    cfg = load_repair_config("cvae_rebuild/configs/virchow2_cvae_preservation_repair_v1.yaml")
    assert cfg.name == "virchow2_cvae_preservation_repair_v1"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.min_decision_rows == 10
    assert {variant.variant_id for variant in cfg.variants} == {
        "current_pca200_beta1_reference",
        "pca64_beta001",
        "pca128_beta001",
        "pca64_beta001_probe025",
        "pca128_beta001_probe025",
        "source_union_pca64_beta001_diagnostic",
        "source_union_pca64_beta001_probe025_diagnostic",
    }
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/virchow2_cvae_preservation_repair_v1")


def test_locked_decentralized_k16_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_k16_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_k16_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_k16_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_exported_k4x4_cc_diag_gmm_k16_late_geom"
    assert cfg.local_gmm_components_per_source_class == 4
    assert cfg.composed_components_per_class == 16
    assert cfg.min_count_for_k4 == 48
    assert cfg.source_weighting == "equal_source_mass"
    assert cfg.support_nelbo_enabled is False


def test_locked_decentralized_adaptive_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_adaptive_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_adaptive_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_adaptive_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_exported_adaptive_k_cc_diag_gmm_late_geom"
    assert cfg.bic_method == "decentralized_exported_bic_selected_cc_diag_gmm_late_geom"
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.min_samples_per_component == 12
    assert cfg.source_weighting == "equal_source_mass"


def test_locked_decentralized_reliability_weighted_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_reliability_weighted_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_exported_adaptive_k_source_reliability_weighted_geom"
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.min_samples_per_component == 12
    assert cfg.source_weighting == "source_local_reliability"
    assert cfg.min_per_source_per_class == 8
    assert cfg.primary_pooling == "weighted_geometric"


def test_locked_decentralized_component_union_prior_config_loads() -> None:
    cfg = load_decentralized_component_union_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_component_union_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_component_union_uniform_gmm"
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.min_samples_per_component == 12
    assert cfg.source_weighting == "uniform_source_component_union"
    assert cfg.primary_pooling == "pooled_raw_logistic"
    assert cfg.shrink_lambdas == (0.25, 0.5)
    assert cfg.budget_diagnostic_per_class_total == 256


def test_locked_decentralized_component_union_shrink025_v2_config_loads() -> None:
    cfg = load_decentralized_component_union_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_component_union_reliability_shrink025_v2"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_component_union_reliability_shrink025"
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.budget_diagnostic_per_class_total is None
    assert cfg.matched_shuffled_reliability_null_permutations == 20


def test_locked_decentralized_component_union_shrink050_confirmation_config_loads() -> None:
    cfg = load_decentralized_component_union_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_component_union_reliability_shrink050"
    assert cfg.primary_shrink_lambda == 0.5
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.fresh_replicate_seeds == (101, 103, 107)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.budget_diagnostic_per_class_total is None
    assert cfg.matched_shuffled_reliability_null_permutations == 20
    assert cfg.random_mass_bag_control_size == 11


def test_locked_decentralized_component_union_shrink050_rejects_changed_lambda() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["component_union_prior"]["primary_shrink_lambda"] = 0.25

    with pytest.raises(Exception, match="primary_shrink_lambda=0.50"):
        parse_decentralized_component_union_prior_config(payload, base_dir=Path("."))


def test_locked_decentralized_component_union_shrink050_rejects_changed_fresh_seed_grid() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["fresh_replicate_seeds"] = [101]

    with pytest.raises(Exception, match="fresh_replicate_seeds=\\[101, 103, 107\\]"):
        parse_decentralized_component_union_prior_config(payload, base_dir=Path("."))


def test_locked_decentralized_component_union_shrink025_v2_rejects_changed_seed_grid() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["experiment_seeds"] = [42]

    with pytest.raises(Exception, match="experiment_seeds=\\[42, 43, 44\\]"):
        parse_decentralized_component_union_prior_config(payload, base_dir=Path("."))


def test_locked_mass_bagged_component_union_config_loads() -> None:
    cfg = load_mass_bagged_component_union_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_mass_bagged_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_component_union_mass_bagged_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_component_union_mass_uncertainty_bagged_v1"
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.primary_bag_members == PRIMARY_BAG_MEMBERS
    assert cfg.control_bag_size == 11
    assert all("shuffled" not in member for member in cfg.primary_bag_members)
    assert cfg.primary_pooling == "arithmetic_probability_ensemble"


def test_mass_bagged_component_union_rejects_shuffled_primary_member() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_decentralized_component_union_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["mass_bagged_component_union"]["primary_bag_members"][0] = "shuffled_reliability_shrink025_perm000"

    with pytest.raises(Exception, match="must not contain shuffled"):
        parse_mass_bagged_component_union_config(payload, base_dir=Path("."))


def test_locked_tailrisk_anchored_component_union_config_loads() -> None:
    cfg = load_tailrisk_anchored_component_union_config(
        "cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_TAILRISK_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.fresh_replicate_seeds == (101, 103, 107)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.blend_alpha == 0.5
    assert cfg.primary_shrink_lambda == 0.5
    assert cfg.matched_shuffled_reliability_null_permutations == 20


def test_tailrisk_anchored_component_union_rejects_changed_blend_alpha() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_anchored_component_union"]["blend_alpha"] = 0.25

    with pytest.raises(Exception, match="blend_alpha"):
        parse_tailrisk_anchored_component_union_config(payload, base_dir=Path("."))


def test_tailrisk_anchored_component_union_rejects_changed_random_bag_size() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_anchored_component_union"]["random_mass_bag_size"] = 3

    with pytest.raises(Exception, match="random_mass_bag_size=11"):
        parse_tailrisk_anchored_component_union_config(payload, base_dir=Path("."))


def test_locked_multipanel_tailrisk_component_union_config_loads() -> None:
    cfg = load_multipanel_tailrisk_component_union_config(
        "cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml"
    )
    assert cfg.name == MULTIPANEL_TAILRISK_NAME
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_MULTIPANEL_TAILRISK_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.fresh_replicate_seeds == (101, 103, 107, 109, 113, 127)
    assert cfg.panel_seed_groups == MULTIPANEL_PANEL_SEEDS
    assert cfg.all_panel_seeds == (17, 23, 31, 101, 103, 107, 109, 113, 127)
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.blend_alpha == 0.5
    assert cfg.primary_pooling == "seed_blend_then_equal_probability_pool"
    assert cfg.matched_shuffled_reliability_null_permutations == 0


def test_multipanel_tailrisk_rejects_changed_blend_alpha() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_multipanel_component_union"]["blend_alpha"] = 0.25

    with pytest.raises(Exception, match="blend_alpha"):
        parse_multipanel_tailrisk_component_union_config(payload, base_dir=Path("."))


def test_multipanel_tailrisk_rejects_undeclared_panel_seed() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_multipanel_component_union"]["panel_seed_groups"]["fresh_b"][-1] = 131
    payload["run_matrix"]["fresh_replicate_seeds"][-1] = 131

    with pytest.raises(Exception, match="panel_seed_groups"):
        parse_multipanel_tailrisk_component_union_config(payload, base_dir=Path("."))


def test_multipanel_tailrisk_rejects_target_support_field() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["inputs"]["support_calibrated_artifact_root"] = "cvae_rebuild/artifacts/not_allowed"

    with pytest.raises(Exception, match="support_calibrated_artifact_root"):
        parse_multipanel_tailrisk_component_union_config(payload, base_dir=Path("."))


def test_locked_dense_tailshield_random_mass_bag_config_loads() -> None:
    cfg = load_dense_tailshield_random_mass_bag_config(
        "cvae_rebuild/configs/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_DENSE_TAILSHIELD_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.fresh_replicate_seeds == (101, 103, 107)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.dense_blend_alpha == 0.25
    assert cfg.bag_blend_alpha == 0.75
    assert cfg.alpha_curve_dense_values == (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    assert cfg.nontrivial_rescue_threshold == 0.02


def test_dense_tailshield_random_mass_bag_rejects_changed_dense_alpha() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["dense_tailshield_random_mass_bag"]["dense_blend_alpha"] = 0.5

    with pytest.raises(Exception, match="dense_blend_alpha"):
        parse_dense_tailshield_random_mass_bag_config(payload, base_dir=Path("."))


def test_dense_tailshield_random_mass_bag_rejects_changed_alpha_curve() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["dense_tailshield_random_mass_bag"]["alpha_curve_dense_values"] = [0.0, 0.25, 1.0]

    with pytest.raises(Exception, match="alpha_curve_dense_values"):
        parse_dense_tailshield_random_mass_bag_config(payload, base_dir=Path("."))


def test_locked_harmful_source_suppression_config_loads() -> None:
    cfg = load_harmful_source_suppression_config(
        "cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_HARMFUL_SUPPRESSION_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.fresh_replicate_seeds == (101, 103, 107)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.random_mass_bag_size == 11
    assert cfg.dirichlet_total_concentration == 16.0
    assert cfg.min_harmfulness_observations == 6
    assert cfg.suppression_rate_low == 0.05
    assert cfg.suppression_rate_high == 0.80


def test_harmful_source_suppression_rejects_changed_bag_size() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["harmful_source_suppression"]["random_mass_bag_size"] = 3

    with pytest.raises(Exception, match="random_mass_bag_size=11"):
        parse_harmful_source_suppression_config(payload, base_dir=Path("."))


def test_harmful_source_suppression_rejects_target_support_config() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["support_size"] = 8

    with pytest.raises(Exception, match="must not configure or consume target support"):
        parse_harmful_source_suppression_config(payload, base_dir=Path("."))


def test_locked_target_support_regime_risk_gate_config_loads() -> None:
    cfg = load_target_support_regime_risk_gate_config(
        "cvae_rebuild/configs/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_target_support32_regime_risk_gated_component_union_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_RISK_GATED_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.support_seeds == (17, 23, 31)
    assert cfg.support_size == 32
    assert cfg.support_size_diagnostics == (8, 16)
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.risk_low_threshold == 0.60
    assert cfg.risk_high_threshold == 0.75
    assert cfg.gate_c == 0.25
    assert cfg.skip_nearest_neighbor_audit is True


def test_target_support_regime_risk_gate_rejects_changed_support_size() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["support_size"] = 16

    with pytest.raises(Exception, match="support_size"):
        parse_target_support_regime_risk_gate_config(payload, base_dir=Path("."))


def test_target_support_regime_risk_gate_rejects_changed_thresholds() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["target_support_regime_risk_gate"]["risk_low_threshold"] = 0.5

    with pytest.raises(Exception, match="risk_low_threshold"):
        parse_target_support_regime_risk_gate_config(payload, base_dir=Path("."))


def test_target_support_regime_risk_gate_rejects_enabled_nn_audit() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["memory"]["skip_nearest_neighbor_audit"] = False

    with pytest.raises(Exception, match="skip nearest-neighbor audit"):
        parse_target_support_regime_risk_gate_config(payload, base_dir=Path("."))


def test_locked_labeled_support_policy_calibration_config_loads() -> None:
    cfg = load_labeled_support_policy_calibration_config(
        "cvae_rebuild/configs/virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_LABELED_SUPPORT_POLICY_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.support_seeds == (17, 23, 31)
    assert cfg.primary_labeled_support_size == 16
    assert cfg.diagnostic_labeled_support_sizes == (8, 32)
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.primary_switch_quantum == 0.0625
    assert cfg.support_quantum_by_size == {8: 0.125, 16: 0.0625, 32: 0.03125}
    assert cfg.skip_nearest_neighbor_audit is True


def test_labeled_support_policy_calibration_rejects_changed_switch_quantum() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["labeled_support_policy_calibration"]["primary_switch_quantum"] = 0.025

    with pytest.raises(Exception, match="primary_switch_quantum"):
        parse_labeled_support_policy_calibration_config(payload, base_dir=Path("."))


def test_nested_labeled_support_splits_are_class_balanced_and_disjoint() -> None:
    metadata = []
    for label in (0, 1):
        for idx in range(20):
            metadata.append({"sample_id": f"c3_y{label}_{idx}", "center": "3", "label": label})
    splits = nested_labeled_support_eval_splits(
        metadata,
        heldout_center="3",
        support_seed=17,
        support_sizes=(8, 16, 32),
        max_support_size=32,
    )
    by_key = {(split.support_size, split.eval_mode): split for split in splits}
    assert len(by_key[(8, "primary_style")].support_indices) == 8
    assert len(by_key[(16, "primary_style")].support_indices) == 16
    assert len(by_key[(32, "primary_style")].support_indices) == 32
    assert set(by_key[(8, "primary_style")].support_indices).issubset(by_key[(16, "primary_style")].support_indices)
    assert set(by_key[(16, "primary_style")].support_indices).issubset(by_key[(32, "primary_style")].support_indices)
    for split in splits:
        assert split.support_labels.count(0) == split.support_labels.count(1)
        assert set(split.support_sample_ids).isdisjoint(split.eval_sample_ids)
        assert split.support_labels_used is True


def test_locked_source_inner_validated_dense_component_hybrid_config_loads() -> None:
    cfg = load_source_inner_validated_hybrid_config(
        "cvae_rebuild/configs/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_source_inner_validated_dense_component_hybrid_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "source_inner_validated_dense_component_binary_gate"
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.component_shrink_lambda == 0.25
    assert cfg.matched_shuffled_gate_null_permutations == 20


def test_source_inner_validated_dense_component_hybrid_rejects_changed_lambda() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_validated_dense_component_hybrid"]["component_shrink_lambda"] = 0.5

    with pytest.raises(Exception, match="component_shrink_lambda"):
        parse_source_inner_validated_hybrid_config(payload, base_dir=Path("."))


def test_locked_pruned_adaptive_equal_all4_config_loads() -> None:
    cfg = load_pruned_adaptive_equal_all4_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_pruned_adaptive_equal_all4_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_pruned_adaptive_equal_all4_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_pruned_adaptive_k_equal_all4_late_geom"
    assert cfg.unpruned_fixed_k == 4
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.min_samples_per_component == 12
    assert cfg.source_weighting == "equal_source_mass"
    assert cfg.primary_pooling == "geometric"
    assert cfg.synthetic_per_class_total == 128


def test_locked_decentralized_reliability_top3_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_reliability_top3_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_reliability_top3_geom_confirmation"
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.source_weighting == "source_local_reliability_top3"
    assert cfg.top_k_sources == 3
    assert cfg.primary_pooling == "geometric"


def test_locked_decentralized_source_inner_transfer_top3_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_source_inner_transfer_top3_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_source_inner_transfer_top3_geom_confirmation"
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.source_weighting == "source_inner_transfer_top3"
    assert cfg.top_k_sources == 3
    assert cfg.primary_pooling == "geometric"


def test_locked_decentralized_support_nelbo_reliability_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_support_nelbo_reliability_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_exported_adaptive_k_support_nelbo_x_reliability_weighted_geom"
    assert cfg.support_size == 32
    assert cfg.support_size_diagnostics == (8, 16, 64)
    assert cfg.support_seeds == cfg.replicate_seeds
    assert cfg.source_weighting == "support_nelbo_x_source_local_reliability"
    assert cfg.support_nelbo_tau == 1.0
    assert cfg.tau_diagnostics == (0.5, 2.0)


def test_locked_decentralized_support8_top3_tau05_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_support8_top3_tau05_gmm_prior_config(
        "cvae_rebuild/configs/virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "decentralized_support8_top3_tau05_support_nelbo_x_reliability_geom"
    assert cfg.support_size == 8
    assert cfg.support_seeds == cfg.replicate_seeds
    assert cfg.source_weighting == "support_nelbo_x_source_local_reliability_top3"
    assert cfg.support_nelbo_tau == 0.5
    assert cfg.top_k_sources == 3


def test_locked_support_calibrated_component_union_config_loads() -> None:
    cfg = load_support_calibrated_component_union_config(
        "cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_support8_calibrated_component_union_prior_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD
    assert cfg.support_size == 8
    assert cfg.support_size_diagnostics == (16, 32)
    assert cfg.nested_support_max_size == 32
    assert cfg.support_seeds == cfg.replicate_seeds
    assert cfg.support_nelbo_tau == 1.0
    assert cfg.support_shrink_lambda == 0.5
    assert cfg.matched_shuffled_support_null_permutations == 20
    assert cfg.random_mass_bag_control_size == 11
    assert cfg.synthetic_per_class_total == 128


def test_support_calibrated_component_union_rejects_changed_support_size() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["support_size"] = 32

    with pytest.raises(Exception, match="support_size"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_support_calibrated_component_union_rejects_strict_changed_null_count() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["support_calibrated_component_union_prior"]["matched_shuffled_support_null_permutations"] = 2

    with pytest.raises(Exception, match="matched_shuffled_support_null_permutations"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_support_calibrated_component_union_rejects_strict_changed_random_bag_size() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["support_calibrated_component_union_prior"]["random_mass_bag_control_size"] = 3

    with pytest.raises(Exception, match="random_mass_bag_control_size"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_support_calibrated_component_union_rejects_strict_changed_budget() -> None:
    path = Path("cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["generation"]["synthetic_per_class_total"] = 64

    with pytest.raises(Exception, match="synthetic_per_class_total"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_nested_support_sets_are_ordered_and_eval_disjoint() -> None:
    metadata = [
        {"sample_id": f"c0_{idx}", "center": "0", "label": idx % 2}
        for idx in range(80)
    ]
    metadata += [
        {"sample_id": f"c1_{idx}", "center": "1", "label": idx % 2}
        for idx in range(10)
    ]
    splits = nested_unlabeled_support_eval_splits(
        metadata,
        heldout_center="0",
        support_seed=17,
        support_sizes=(8, 16, 32),
        max_support_size=32,
    )
    by_key = {(split.support_size, split.eval_mode): split for split in splits}
    s8 = set(by_key[(8, "primary_style")].support_sample_ids)
    s16 = set(by_key[(16, "primary_style")].support_sample_ids)
    s32 = set(by_key[(32, "primary_style")].support_sample_ids)
    assert s8 < s16 < s32
    assert set(by_key[(8, "primary_style")].support_sample_ids).isdisjoint(
        by_key[(8, "primary_style")].eval_sample_ids
    )
    assert set(by_key[(8, "fixed_support32")].eval_sample_ids) == set(
        by_key[(32, "fixed_support32")].eval_sample_ids
    )
    assert by_key[(8, "primary_style")].support_labels_used is False


def test_support_shrink_plan_preserves_mass_rule_and_floor_nonbinding() -> None:
    cfg = load_support_calibrated_component_union_config(
        "cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml"
    )
    scores = [
        SupportScore(42, "0", 17, 8, "1", 10.0, 0.0),
        SupportScore(42, "0", 17, 8, "2", 10.0, 1.0),
        SupportScore(42, "0", 17, 8, "3", 10.0, 2.0),
        SupportScore(42, "0", 17, 8, "4", 10.0, 3.0),
    ]
    plan = _support_shrink_plan(cfg, ("1", "2", "3", "4"), scores, total=128)
    assert sum(plan["weights"].values()) == pytest.approx(1.0)
    assert sum(plan["budgets"].values()) == 128
    assert plan["floor_binding_count"] == 0
    for source in ("1", "2", "3", "4"):
        expected = 0.5 * 0.25 + 0.5 * plan["support_weights"][source]
        assert plan["weights"][source] == pytest.approx(expected)


def test_support_shuffled_null_preserves_score_multiset() -> None:
    cfg = load_support_calibrated_component_union_config(
        "cvae_rebuild/configs/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml"
    )
    scores = [
        SupportScore(42, "0", 17, 8, "1", 10.0, 0.0),
        SupportScore(42, "0", 17, 8, "2", 11.0, 1.0),
        SupportScore(42, "0", 17, 8, "3", 12.0, 2.0),
        SupportScore(42, "0", 17, 8, "4", 13.0, 3.0),
    ]
    plan = _matched_shuffled_support_plan(
        cfg,
        ("1", "2", "3", "4"),
        scores,
        experiment_seed=42,
        heldout_center="0",
        support_seed=17,
        permutation_id=0,
        total=128,
    )
    assert sorted(plan["calibrated_support_nelbo"].values()) == [0.0, 1.0, 2.0, 3.0]
    assert sum(plan["budgets"].values()) == 128
    assert plan["control_permutation_id"] == 0


def test_constrained_budget_floor_binding_is_reported() -> None:
    budgets, bindings = _constrained_weighted_budgets(
        128,
        ("1", "2", "3", "4"),
        {"1": 0.01, "2": 0.10, "3": 0.20, "4": 0.69},
        8,
    )
    assert sum(budgets.values()) == 128
    assert bindings["1"] is True
    assert all(value >= 8 for value in budgets.values())


def test_locked_paired_dense_all4_reliability_config_loads() -> None:
    cfg = load_paired_dense_all4_reliability_config(
        "cvae_rebuild/configs/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_paired_dense_all4_reliability_confirmation_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "paired_reliability_all4_shrink050_geom"
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.gmm_n_init == 5
    assert cfg.gmm_max_iter == 500
    assert cfg.min_component_weight == 0.02
    assert cfg.shrinkage_values == (0.25, 0.5)


def test_locked_paired_component_coverage_audit_config_loads() -> None:
    cfg = load_paired_component_coverage_audit_config(
        "cvae_rebuild/configs/virchow2_cvae_paired_component_coverage_audit_v1.yaml"
    )
    assert cfg.name == "virchow2_cvae_paired_component_coverage_audit_v1"
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == "paired_reliability_all4_weighted_component_stratified128_geom"
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.synthetic_per_class_total == 128
    assert cfg.diagnostic_synthetic_per_class_total == 256
    assert cfg.component_sampling_rules == ("multinomial", "stratified_largest_remainder")
    assert cfg.candidate_components_per_source_class == (4, 3, 2, 1)
    assert cfg.gmm_n_init == 5
    assert cfg.gmm_max_iter == 500
    assert cfg.min_component_weight == 0.02


def test_budget_splits_are_rank_order_deterministic() -> None:
    assert split_budget(128, ["1"]) == {"1": 128}
    assert split_budget(128, ["1", "2"]) == {"1": 64, "2": 64}
    assert split_budget(128, ["1", "2", "3"]) == {"1": 43, "2": 43, "3": 42}
    assert split_budget(128, ["1", "2", "3", "4"]) == {"1": 32, "2": 32, "3": 32, "4": 32}


def test_candidate_pool_excludes_target() -> None:
    assert_candidate_pool(heldout_center="0", candidate_experts=["1", "2", "3", "4"])
    try:
        assert_candidate_pool(heldout_center="0", candidate_experts=["0", "2", "3", "4"])
    except ProtocolError:
        pass
    else:
        raise AssertionError("target expert inclusion was not rejected")


def test_support_label_use_fails_leakage_report() -> None:
    try:
        assert_support_labels_unused(True)
    except ProtocolError:
        pass
    else:
        raise AssertionError("support label use was not rejected")
    report = build_leakage_report(
        target_support_labels_for_selection=True,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
    )
    assert report.status == "FAIL"
    assert "target_support_labels_for_selection" in report.violations


def test_smoke_artifacts_write_contract(tmp_path: Path) -> None:
    cfg = load_config("cvae_rebuild/configs/target_support32_virchow2_cvae_top2_v1.yaml")
    root = run_artifact_contract_smoke(cfg, artifact_root=tmp_path / "artifacts")
    missing = [rel for rel in REQUIRED_OUTPUTS if not (root / rel).exists()]
    assert not missing


def test_synthetic_smoke_runs_mini_cvae_and_writes_nonempty_tables(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    cfg = load_config("cvae_rebuild/configs/target_support32_virchow2_cvae_top2_v1.yaml")
    root = run_synthetic_smoke(cfg, artifact_root=tmp_path / "synthetic_smoke")
    assert (root / "tables" / "support_nelbo_routing_scores.csv").read_text(encoding="utf-8").count("\n") > 1
    assert "support_nelbo_top2_geom" in (root / "tables" / "all_expert_downstream_matrix.csv").read_text(
        encoding="utf-8"
    )
