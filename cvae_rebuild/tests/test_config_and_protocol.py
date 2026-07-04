import json
from pathlib import Path

import pytest
import yaml

from config import load_config, resolved_config_dict
from decentralized_adaptive_gmm_prior import load_decentralized_adaptive_gmm_prior_config
from decentralized_component_union_prior import (
    load_decentralized_component_union_prior_config,
    parse_decentralized_component_union_prior_config,
)
from component_union_mass_bagged import (
    PRIMARY_BAG_MEMBERS,
    load_mass_bagged_component_union_config,
    parse_mass_bagged_component_union_config,
)
from component_union_tailrisk_anchored_mass_bagged import (
    FIXED_BETA050_CONFIRMATION_EXPERIMENT_SEEDS,
    FIXED_BETA050_DEVELOPMENT_EXPERIMENT_SEEDS,
    FIXED_BETA050_POSITIVE_UNION_NAME,
    HARM_GATED_DEVELOPMENT_EXPERIMENT_SEEDS,
    HARM_GATED_POSITIVE_UNION_NAME,
    HARM_GATED_PRIMARY_SELECTABLE_RULES,
    HARM_GATED_REQUESTED_EXPERIMENT_SEEDS,
    HARM_GATED_RESERVE_EXPERIMENT_SEEDS,
    MULTIPANEL_PANEL_SEEDS,
    MULTIPANEL_TAILRISK_NAME,
    POSITIVE_UNION_RULES,
    POSITIVE_UNION_TAILRISK_NAME,
    POSITIVE_UNION_RULE_BETA050,
    PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD,
    PRIMARY_HARM_GATED_POSITIVE_UNION_METHOD,
    PRIMARY_POSITIVE_UNION_METHOD,
    PRIMARY_MULTIPANEL_TAILRISK_METHOD,
    PRIMARY_TAILRISK_METHOD,
    load_fixed_beta050_positive_union_config,
    load_harm_gated_positive_union_config,
    load_multipanel_tailrisk_component_union_config,
    load_source_inner_positive_union_config,
    load_tailrisk_anchored_component_union_config,
    parse_fixed_beta050_positive_union_config,
    parse_harm_gated_positive_union_config,
    parse_multipanel_tailrisk_component_union_config,
    parse_source_inner_positive_union_config,
    parse_tailrisk_anchored_component_union_config,
)
from dense_reliability_tailshield_random_mass_bag import (
    PRIMARY_DENSE_TAILSHIELD_METHOD,
    load_dense_tailshield_random_mass_bag_config,
    parse_dense_tailshield_random_mass_bag_config,
)
from source_inner_harmful_source_suppression import (
    PRIMARY_HARMFUL_SUPPRESSION_METHOD,
    load_harmful_source_suppression_config,
    parse_harmful_source_suppression_config,
)
from target_support_regime_risk_gated_component_union import (
    PRIMARY_RISK_GATED_METHOD,
    load_target_support_regime_risk_gate_config,
    parse_target_support_regime_risk_gate_config,
)
from labeled_support_random_vs_dense_policy_calibration import (
    PRIMARY_LABELED_SUPPORT_POLICY_METHOD,
    load_labeled_support_policy_calibration_config,
    nested_labeled_support_eval_splits,
    parse_labeled_support_policy_calibration_config,
)
from decentralized_pruned_adaptive_equal_all4_prior import load_pruned_adaptive_equal_all4_config
from decentralized_k16_gmm_prior import load_decentralized_k16_gmm_prior_config
from decentralized_reliability_weighted_gmm_prior import (
    load_decentralized_reliability_weighted_gmm_prior_config,
)
from decentralized_reliability_top3_gmm_prior import (
    load_decentralized_reliability_top3_gmm_prior_config,
)
from decentralized_source_inner_transfer_top3_gmm_prior import (
    load_decentralized_source_inner_transfer_top3_gmm_prior_config,
)
from decentralized_support_nelbo_reliability_gmm_prior import (
    load_decentralized_support_nelbo_reliability_gmm_prior_config,
)
from decentralized_support8_top3_tau05_gmm_prior import (
    load_decentralized_support8_top3_tau05_gmm_prior_config,
)
from support_calibrated_component_union_prior import (
    PRIMARY_SUPPORT_CALIBRATED_COMPONENT_UNION_METHOD,
    _constrained_weighted_budgets,
    _matched_shuffled_support_plan,
    _support_shrink_plan,
    load_support_calibrated_component_union_config,
    nested_unlabeled_support_eval_splits,
    parse_support_calibrated_component_union_config,
)
from support_nelbo import SupportScore
from paired_dense_all4_reliability_confirmation import (
    load_paired_dense_all4_reliability_config,
)
from paired_component_coverage_audit import (
    load_paired_component_coverage_audit_config,
)
from source_inner_validated_dense_component_hybrid import (
    load_source_inner_validated_hybrid_config,
    parse_source_inner_validated_hybrid_config,
)
from pipeline import run_artifact_contract_smoke, run_synthetic_smoke
from preservation import load_preservation_config
from preservation_repair import load_repair_config
from protocol import (
    ProtocolError,
    assert_oracle_diagnostic_only,
    assert_candidate_pool,
    assert_support_labels_unused,
    build_leakage_report,
    split_budget,
)
from reporting import (
    REQUIRED_OUTPUTS,
    protocol_manifest_payload,
    validate_imported_artifacts,
    write_csv_rows,
    write_json,
    write_protocol_finalization,
)
from support_split_rows import labeled_support_split_rows, unlabeled_support_split_rows


def test_locked_config_loads() -> None:
    cfg = load_config("configs/camelyon17_virchow2_legacy/target_support32_virchow2_cvae_top2_v1.yaml")
    assert cfg.primary_method == "support_nelbo_top2_geom"
    assert cfg.support_size == 32
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/camelyon17_virchow2_legacy/target_support32_virchow2_cvae_top2_v1")


def test_locked_preservation_config_loads() -> None:
    cfg = load_preservation_config("configs/camelyon17_virchow2_legacy/virchow2_cvae_preservation_diagnosis_v1.yaml")
    assert cfg.name == "virchow2_cvae_preservation_diagnosis_v1"
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.classifier_class_weight == "balanced"
    assert cfg.classifier_seed is None
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/camelyon17_virchow2_legacy/virchow2_cvae_preservation_diagnosis_v1")


def test_locked_preservation_repair_config_loads() -> None:
    cfg = load_repair_config("configs/camelyon17_virchow2_legacy/virchow2_cvae_preservation_repair_v1.yaml")
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
    assert cfg.artifact_root.as_posix().endswith("cvae_rebuild/artifacts/camelyon17_virchow2_legacy/virchow2_cvae_preservation_repair_v1")


def test_locked_decentralized_k16_gmm_prior_config_loads() -> None:
    cfg = load_decentralized_k16_gmm_prior_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_k16_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_adaptive_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["component_union_prior"]["primary_shrink_lambda"] = 0.25

    with pytest.raises(Exception, match="primary_shrink_lambda=0.50"):
        parse_decentralized_component_union_prior_config(payload, base_dir=Path("."))


def test_locked_decentralized_component_union_shrink050_rejects_changed_fresh_seed_grid() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_reliability_shrink050_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["fresh_replicate_seeds"] = [101]

    with pytest.raises(Exception, match="fresh_replicate_seeds=\\[101, 103, 107\\]"):
        parse_decentralized_component_union_prior_config(payload, base_dir=Path("."))


def test_locked_decentralized_component_union_shrink025_v2_rejects_changed_seed_grid() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_reliability_shrink025_v2.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["experiment_seeds"] = [42]

    with pytest.raises(Exception, match="experiment_seeds=\\[42, 43, 44\\]"):
        parse_decentralized_component_union_prior_config(payload, base_dir=Path("."))


def test_locked_mass_bagged_component_union_config_loads() -> None:
    cfg = load_mass_bagged_component_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_mass_bagged_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_component_union_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["mass_bagged_component_union"]["primary_bag_members"][0] = "shuffled_reliability_shrink025_perm000"

    with pytest.raises(Exception, match="must not contain shuffled"):
        parse_mass_bagged_component_union_config(payload, base_dir=Path("."))


def test_locked_tailrisk_anchored_component_union_config_loads() -> None:
    cfg = load_tailrisk_anchored_component_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_anchored_component_union"]["blend_alpha"] = 0.25

    with pytest.raises(Exception, match="blend_alpha"):
        parse_tailrisk_anchored_component_union_config(payload, base_dir=Path("."))


def test_tailrisk_anchored_component_union_rejects_changed_random_bag_size() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_anchored_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_anchored_component_union"]["random_mass_bag_size"] = 3

    with pytest.raises(Exception, match="random_mass_bag_size=11"):
        parse_tailrisk_anchored_component_union_config(payload, base_dir=Path("."))


def test_locked_multipanel_tailrisk_component_union_config_loads() -> None:
    cfg = load_multipanel_tailrisk_component_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_multipanel_component_union"]["blend_alpha"] = 0.25

    with pytest.raises(Exception, match="blend_alpha"):
        parse_multipanel_tailrisk_component_union_config(payload, base_dir=Path("."))


def test_multipanel_tailrisk_rejects_undeclared_panel_seed() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["tailrisk_multipanel_component_union"]["panel_seed_groups"]["fresh_b"][-1] = 131
    payload["run_matrix"]["fresh_replicate_seeds"][-1] = 131

    with pytest.raises(Exception, match="panel_seed_groups"):
        parse_multipanel_tailrisk_component_union_config(payload, base_dir=Path("."))


def test_multipanel_tailrisk_rejects_target_support_field() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_component_union_tailrisk_multipanel_mass_bagged_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["inputs"]["support_calibrated_artifact_root"] = "cvae_rebuild/artifacts/not_allowed"

    with pytest.raises(Exception, match="support_calibrated_artifact_root"):
        parse_multipanel_tailrisk_component_union_config(payload, base_dir=Path("."))


def test_locked_source_inner_positive_union_config_loads() -> None:
    cfg = load_source_inner_positive_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_class_conditional_positive_union_v1.yaml"
    )
    assert cfg.name == POSITIVE_UNION_TAILRISK_NAME
    assert cfg.backbone == "virchow2"
    assert cfg.primary_variant == "pca64_beta001"
    assert cfg.primary_method == PRIMARY_POSITIVE_UNION_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (42, 43, 44)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.replicate_seeds == (17, 23, 31)
    assert cfg.fresh_replicate_seeds == (101, 103, 107, 109, 113, 127)
    assert cfg.panel_seed_groups == MULTIPANEL_PANEL_SEEDS
    assert cfg.all_panel_seeds == (17, 23, 31, 101, 103, 107, 109, 113, 127)
    assert cfg.candidate_pooling_rules == POSITIVE_UNION_RULES
    assert cfg.positive_label == 1
    assert cfg.prediction_threshold == 0.5
    assert cfg.min_source_inner_positive_count == 5
    assert cfg.positive_union_eps == 1.0e-8
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.blend_alpha == 0.5
    assert cfg.primary_pooling == "source_inner_selected_class_conditional_positive_union"


def test_source_inner_positive_union_rejects_changed_beta_grid() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_class_conditional_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_class_conditional_positive_union"]["candidate_pooling_rules"][-1] = "positive_union_beta075"

    with pytest.raises(Exception, match="candidate_pooling_rules"):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))


def test_source_inner_positive_union_rejects_changed_min_positive_count() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_class_conditional_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_class_conditional_positive_union"]["min_source_inner_positive_count"] = 3

    with pytest.raises(Exception, match="min_source_inner_positive_count"):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("source_inner_class_conditional_positive_union", "blend_alpha", 0.25, "blend_alpha"),
        ("source_inner_class_conditional_positive_union", "random_mass_bag_alpha", 2.0, "random_mass_bag_alpha"),
        ("source_inner_class_conditional_positive_union", "positive_label", 0, "positive_label"),
        ("source_inner_class_conditional_positive_union", "prediction_threshold", 0.25, "prediction_threshold"),
        ("source_inner_class_conditional_positive_union", "positive_union_eps", 1.0e-6, "positive_union_eps"),
    ],
)
def test_source_inner_positive_union_rejects_changed_locked_fields(section: str, key: str, value: object, message: str) -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_class_conditional_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload[section][key] = value

    with pytest.raises(Exception, match=message):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))


def test_source_inner_positive_union_rejects_undeclared_panel_seed_and_bag_size_in_strict_run() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_class_conditional_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_class_conditional_positive_union"]["panel_seed_groups"]["fresh_b"][-1] = 131
    payload["run_matrix"]["fresh_replicate_seeds"][-1] = 131
    with pytest.raises(Exception, match="panel_seed_groups"):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_class_conditional_positive_union"]["random_mass_bag_size"] = 9
    with pytest.raises(Exception, match="random_mass_bag_size"):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))


def test_source_inner_positive_union_rejects_target_support_and_target_selection() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_class_conditional_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["inputs"]["support_calibrated_artifact_root"] = "cvae_rebuild/artifacts/not_allowed"

    with pytest.raises(Exception, match="support_calibrated_artifact_root"):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_class_conditional_positive_union"]["target_eval_metric_selection"] = True
    with pytest.raises(Exception, match="target_eval_metric_selection"):
        parse_source_inner_positive_union_config(payload, base_dir=Path("."))


def test_locked_fixed_beta050_positive_union_config_loads() -> None:
    cfg = load_fixed_beta050_positive_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_fixed_beta050_positive_union_confirmation_v1.yaml"
    )
    assert cfg.name == FIXED_BETA050_POSITIVE_UNION_NAME
    assert cfg.primary_method == PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == FIXED_BETA050_CONFIRMATION_EXPERIMENT_SEEDS
    assert cfg.development_experiment_seeds == FIXED_BETA050_DEVELOPMENT_EXPERIMENT_SEEDS
    assert set(cfg.experiment_seeds).isdisjoint(cfg.development_experiment_seeds)
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.panel_seed_groups == MULTIPANEL_PANEL_SEEDS
    assert cfg.candidate_pooling_rules == POSITIVE_UNION_RULES
    assert cfg.fixed_pooling_rule == POSITIVE_UNION_RULE_BETA050
    assert cfg.fixed_beta == 0.5
    assert cfg.primary_pooling == "fixed_global_positive_union_beta050"
    assert cfg.positive_label == 1
    assert cfg.prediction_threshold == 0.5
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.blend_alpha == 0.5
    assert cfg.rare_positive_count_threshold == 10
    assert cfg.rare_positive_prevalence_threshold == 0.05


def test_fixed_beta050_positive_union_rejects_changed_beta_and_seed_overlap() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_fixed_beta050_positive_union_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["fixed_beta050_positive_union_confirmation"]["fixed_beta"] = 0.25
    with pytest.raises(Exception, match="fixed_beta"):
        parse_fixed_beta050_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["experiment_seeds"][0] = 42
    payload["fixed_beta050_positive_union_confirmation"]["primary_confirmation_experiment_seeds"][0] = 42
    with pytest.raises(Exception, match="must not overlap"):
        parse_fixed_beta050_positive_union_config(payload, base_dir=Path("."))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("blend_alpha", 0.25, "blend_alpha"),
        ("random_mass_bag_alpha", 2.0, "random_mass_bag_alpha"),
        ("positive_label", 0, "positive_label"),
        ("prediction_threshold", 0.25, "prediction_threshold"),
        ("fixed_pooling_rule", "positive_union_beta025", "fixed_pooling_rule"),
        ("rare_positive_count_threshold", 9, "rare_positive_count_threshold"),
    ],
)
def test_fixed_beta050_positive_union_rejects_changed_locked_fields(key: str, value: object, message: str) -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_fixed_beta050_positive_union_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["fixed_beta050_positive_union_confirmation"][key] = value
    with pytest.raises(Exception, match=message):
        parse_fixed_beta050_positive_union_config(payload, base_dir=Path("."))


def test_fixed_beta050_positive_union_rejects_target_support_and_target_selection() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_fixed_beta050_positive_union_confirmation_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["inputs"]["support_calibrated_artifact_root"] = "cvae_rebuild/artifacts/not_allowed"
    with pytest.raises(Exception, match="support_calibrated_artifact_root"):
        parse_fixed_beta050_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["fixed_beta050_positive_union_confirmation"]["target_eval_metric_selection"] = True
    with pytest.raises(Exception, match="target_eval_metric_selection"):
        parse_fixed_beta050_positive_union_config(payload, base_dir=Path("."))


def test_locked_harm_gated_positive_union_config_loads() -> None:
    cfg = load_harm_gated_positive_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_harm_gated_positive_union_v1.yaml"
    )
    assert cfg.name == HARM_GATED_POSITIVE_UNION_NAME
    assert cfg.primary_method == PRIMARY_HARM_GATED_POSITIVE_UNION_METHOD
    assert cfg.strict_full_run_matrix is True
    assert cfg.experiment_seeds == (*HARM_GATED_REQUESTED_EXPERIMENT_SEEDS, *HARM_GATED_RESERVE_EXPERIMENT_SEEDS)
    assert cfg.development_experiment_seeds == HARM_GATED_DEVELOPMENT_EXPERIMENT_SEEDS
    assert cfg.primary_requested_experiment_seeds == HARM_GATED_REQUESTED_EXPERIMENT_SEEDS
    assert cfg.reserve_experiment_seeds == HARM_GATED_RESERVE_EXPERIMENT_SEEDS
    assert cfg.heldout_centers == ("0", "1", "2", "3", "4")
    assert cfg.panel_seed_groups == MULTIPANEL_PANEL_SEEDS
    assert cfg.candidate_pooling_rules == POSITIVE_UNION_RULES
    assert cfg.primary_selectable_rules == HARM_GATED_PRIMARY_SELECTABLE_RULES
    assert cfg.beta100_primary_selectable is False
    assert cfg.positive_label == 1
    assert cfg.prediction_threshold == 0.5
    assert cfg.min_source_inner_positive_count == 5
    assert cfg.beta050_min_source_inner_positive_count == 10
    assert cfg.random_mass_bag_size == 11
    assert cfg.random_mass_bag_alpha == 4.0
    assert cfg.blend_alpha == 0.5
    assert cfg.selector_thresholds_frozen_before_primary is True
    assert cfg.selector_threshold_source == "retrospective_development_only"
    assert cfg.selector_thresholds_may_be_changed_after_primary is False
    assert cfg.cell_level_reserve_stitching_allowed is False
    assert cfg.skip_nearest_neighbor_audit is True


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("blend_alpha", 0.25, "blend_alpha"),
        ("random_mass_bag_alpha", 2.0, "random_mass_bag_alpha"),
        ("random_mass_bag_size", 9, "random_mass_bag_size"),
        ("positive_label", 0, "positive_label"),
        ("prediction_threshold", 0.25, "prediction_threshold"),
        ("beta050_min_source_inner_positive_count", 9, "beta050_min_source_inner_positive_count"),
        ("harm_gate_bacc_noninferiority_margin", 0.010, "harm_gate_bacc_noninferiority_margin"),
        ("beta025_predicted_positive_rate_delta", 0.050, "beta025_predicted_positive_rate_delta"),
        ("beta050_precision_margin", 0.010, "beta050_precision_margin"),
    ],
)
def test_harm_gated_positive_union_rejects_changed_locked_fields(key: str, value: object, message: str) -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_harm_gated_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_harm_gated_positive_union"][key] = value
    with pytest.raises(Exception, match=message):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))


def test_harm_gated_positive_union_rejects_beta100_primary_target_support_and_seed_overlap() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_harm_gated_positive_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_harm_gated_positive_union"]["beta100_primary_selectable"] = True
    with pytest.raises(Exception, match="beta100_primary_selectable"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_harm_gated_positive_union"]["primary_selectable_rules"].append("positive_union_beta100")
    with pytest.raises(Exception, match="primary_selectable_rules"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["inputs"]["support_calibrated_artifact_root"] = "cvae_rebuild/artifacts/not_allowed"
    with pytest.raises(Exception, match="support_calibrated_artifact_root"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_harm_gated_positive_union"]["target_threshold_selection"] = True
    with pytest.raises(Exception, match="target_threshold_selection"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_harm_gated_positive_union"]["reserve_experiment_seeds"][0] = 50
    payload["run_matrix"]["experiment_seeds"][-2] = 50
    with pytest.raises(Exception, match="must not overlap"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_harm_gated_positive_union"]["cell_level_reserve_stitching_allowed"] = True
    with pytest.raises(Exception, match="cell_level_reserve_stitching_allowed"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["memory"]["skip_nearest_neighbor_audit"] = False
    with pytest.raises(Exception, match="skip nearest-neighbor audit"):
        parse_harm_gated_positive_union_config(payload, base_dir=Path("."))


def test_locked_dense_tailshield_random_mass_bag_config_loads() -> None:
    cfg = load_dense_tailshield_random_mass_bag_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["dense_tailshield_random_mass_bag"]["dense_blend_alpha"] = 0.5

    with pytest.raises(Exception, match="dense_blend_alpha"):
        parse_dense_tailshield_random_mass_bag_config(payload, base_dir=Path("."))


def test_dense_tailshield_random_mass_bag_rejects_changed_alpha_curve() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_dense_reliability_tailshield_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["dense_tailshield_random_mass_bag"]["alpha_curve_dense_values"] = [0.0, 0.25, 1.0]

    with pytest.raises(Exception, match="alpha_curve_dense_values"):
        parse_dense_tailshield_random_mass_bag_config(payload, base_dir=Path("."))


def test_locked_harmful_source_suppression_config_loads() -> None:
    cfg = load_harmful_source_suppression_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["harmful_source_suppression"]["random_mass_bag_size"] = 3

    with pytest.raises(Exception, match="random_mass_bag_size=11"):
        parse_harmful_source_suppression_config(payload, base_dir=Path("."))


def test_harmful_source_suppression_rejects_target_support_config() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_harmful_source_suppression_random_mass_bag_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["support_size"] = 8

    with pytest.raises(Exception, match="must not configure or consume target support"):
        parse_harmful_source_suppression_config(payload, base_dir=Path("."))


def test_locked_target_support_regime_risk_gate_config_loads() -> None:
    cfg = load_target_support_regime_risk_gate_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["support_size"] = 16

    with pytest.raises(Exception, match="support_size"):
        parse_target_support_regime_risk_gate_config(payload, base_dir=Path("."))


def test_target_support_regime_risk_gate_rejects_changed_thresholds() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["target_support_regime_risk_gate"]["risk_low_threshold"] = 0.5

    with pytest.raises(Exception, match="risk_low_threshold"):
        parse_target_support_regime_risk_gate_config(payload, base_dir=Path("."))


def test_target_support_regime_risk_gate_rejects_enabled_nn_audit() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_target_support32_regime_risk_gated_component_union_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["memory"]["skip_nearest_neighbor_audit"] = False

    with pytest.raises(Exception, match="skip nearest-neighbor audit"):
        parse_target_support_regime_risk_gate_config(payload, base_dir=Path("."))


def test_locked_labeled_support_policy_calibration_config_loads() -> None:
    cfg = load_labeled_support_policy_calibration_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_labeled_support16_random_vs_dense_policy_calibration_v1.yaml")
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
    rows = labeled_support_split_rows(splits, experiment_seed=45, support_seed=17, scope="target")
    primary_row = next(row for row in rows if row["support_size"] == 8 and row["eval_mode"] == "primary_style")
    assert primary_row["experiment_seed"] == 45
    assert primary_row["support_seed"] == 17
    assert primary_row["split_scope"] == "target"
    assert primary_row["support_labels_used"] == 1
    assert primary_row["support_count_class0"] == 4
    assert primary_row["support_count_class1"] == 4
    assert primary_row["class_balanced_support"] == 1
    assert primary_row["support_eval_disjoint"] == 1
    assert primary_row["size_specific_eval_exclusion"] == 1


def test_locked_source_inner_validated_dense_component_hybrid_config_loads() -> None:
    cfg = load_source_inner_validated_hybrid_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_source_inner_validated_dense_component_hybrid_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["source_inner_validated_dense_component_hybrid"]["component_shrink_lambda"] = 0.5

    with pytest.raises(Exception, match="component_shrink_lambda"):
        parse_source_inner_validated_hybrid_config(payload, base_dir=Path("."))


def test_locked_pruned_adaptive_equal_all4_config_loads() -> None:
    cfg = load_pruned_adaptive_equal_all4_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_pruned_adaptive_equal_all4_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_support_nelbo_reliability_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml"
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
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["run_matrix"]["support_size"] = 32

    with pytest.raises(Exception, match="support_size"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_support_calibrated_component_union_rejects_strict_changed_null_count() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["support_calibrated_component_union_prior"]["matched_shuffled_support_null_permutations"] = 2

    with pytest.raises(Exception, match="matched_shuffled_support_null_permutations"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_support_calibrated_component_union_rejects_strict_changed_random_bag_size() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["support_calibrated_component_union_prior"]["random_mass_bag_control_size"] = 3

    with pytest.raises(Exception, match="random_mass_bag_control_size"):
        parse_support_calibrated_component_union_config(payload, base_dir=Path("."))


def test_support_calibrated_component_union_rejects_strict_changed_budget() -> None:
    path = Path("configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml")
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
    rows = unlabeled_support_split_rows(splits, experiment_seed=42, replicate_seed=17)
    primary_row = next(row for row in rows if row["support_size"] == 8 and row["eval_mode"] == "primary_style")
    fixed_row = next(row for row in rows if row["support_size"] == 8 and row["eval_mode"] == "fixed_support32")
    assert primary_row["experiment_seed"] == 42
    assert primary_row["replicate_seed"] == 17
    assert primary_row["support_labels_used"] == 0
    assert primary_row["support_size_actual"] == 8
    assert primary_row["nested_support_diagnostics"] == 1
    assert fixed_row["fixed_eval_support_size_diagnostics"] == 1


def test_support_shrink_plan_preserves_mass_rule_and_floor_nonbinding() -> None:
    cfg = load_support_calibrated_component_union_config(
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_support8_calibrated_component_union_prior_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1.yaml"
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
        "configs/camelyon17_virchow2_legacy/virchow2_cvae_paired_component_coverage_audit_v1.yaml"
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


def test_oracle_rows_must_be_diagnostic_only_for_method_schema() -> None:
    assert_oracle_diagnostic_only(
        [{"method": "downstream_oracle_diagnostic_only", "selection_source": "diagnostic_only"}]
    )
    with pytest.raises(ProtocolError, match="Downstream oracle rows"):
        assert_oracle_diagnostic_only(
            [{"method": "downstream_oracle_diagnostic_only", "selection_source": "primary"}]
        )


def test_oracle_rows_must_be_diagnostic_only_for_prior_method_schema() -> None:
    assert_oracle_diagnostic_only(
        [{"prior_method": "exhaustive_drop_one_top3_oracle_reference", "selection_source": "diagnostic_only"}]
    )
    with pytest.raises(ProtocolError, match="Downstream oracle rows"):
        assert_oracle_diagnostic_only(
            [{"prior_method": "exhaustive_drop_one_top3_oracle_reference", "selection_source": "primary"}]
        )


def test_reporting_finalizer_writes_exact_default_protocol_payloads(tmp_path: Path) -> None:
    cfg = load_config("configs/camelyon17_virchow2_legacy/target_support32_virchow2_cvae_top2_v1.yaml")
    report = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=True,
        oracle_rows_diagnostic_only=True,
    )
    root = tmp_path / "finalized"

    write_protocol_finalization(
        root,
        leakage_report=report.to_json_dict(),
        protocol_manifest=protocol_manifest_payload(cfg),
        resolved_config=resolved_config_dict(cfg),
    )

    written_files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    assert written_files == [
        "manifests/protocol_manifest.json",
        "reports/leakage_report.json",
        "run_config_resolved.yaml",
    ]
    assert json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8")) == {
        "candidate_count_per_cell": 4,
        "experiment_name": "target_support32_calibrated_unlabeled_marginal_nelbo_top2_geom_virchow2_cvae_pca256_v1",
        "oracle_role": "diagnostic_only",
        "primary_method": "support_nelbo_top2_geom",
        "schema_version": "cvae_rebuild_protocol_manifest_v1",
        "support_labels_for_selection": False,
        "support_size": 32,
        "target_eval_labels_for_scoring_only": True,
    }
    assert json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8")) == {
        "oracle_rows_diagnostic_only": True,
        "schema_version": "cvae_rebuild_leakage_report_v1",
        "status": "PASS",
        "target_eval_labels_for_scoring_only": True,
        "target_expert_excluded": True,
        "target_support_labels_for_selection": False,
        "violations": [],
    }
    assert json.loads((root / "run_config_resolved.yaml").read_text(encoding="utf-8")) == resolved_config_dict(cfg)


def test_reporting_finalizer_preserves_explicit_leakage_report_payload(tmp_path: Path) -> None:
    cfg = load_config("configs/camelyon17_virchow2_legacy/target_support32_virchow2_cvae_top2_v1.yaml")
    report = build_leakage_report(
        target_support_labels_for_selection=True,
        target_eval_labels_for_scoring_only=False,
        target_expert_excluded=False,
        oracle_rows_diagnostic_only=False,
        extra_violations=("sentinel_violation",),
    )
    root = tmp_path / "failed_finalized"

    write_protocol_finalization(
        root,
        leakage_report=report.to_json_dict(),
        protocol_manifest=protocol_manifest_payload(cfg),
        resolved_config=resolved_config_dict(cfg),
    )

    assert report.status == "FAIL"
    assert json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8")) == report.to_json_dict()


def test_validate_imported_artifacts_accepts_pass_leakage_reports(tmp_path: Path) -> None:
    leakage = tmp_path / "reports" / "leakage_report.json"
    table = tmp_path / "tables" / "summary.csv"
    write_json(leakage, {"status": "PASS"})
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("method,status\nprimary,ok\n", encoding="utf-8")

    validate_imported_artifacts(
        (leakage, table),
        missing_message="Missing imported source-union GMM reference artifacts: {missing}",
    )


def test_validate_imported_artifacts_rejects_missing_required_path_with_context_message(tmp_path: Path) -> None:
    missing = tmp_path / "tables" / "missing.csv"

    with pytest.raises(
        ProtocolError,
        match=r"Missing imported source-union GMM reference artifacts: .*missing\.csv",
    ):
        validate_imported_artifacts(
            (missing,),
            missing_message="Missing imported source-union GMM reference artifacts: {missing}",
        )


def test_validate_imported_artifacts_rejects_missing_k24_path_with_context_message(tmp_path: Path) -> None:
    missing = tmp_path / "tables" / "balanced_gmm_gap_summary.csv"

    with pytest.raises(
        ProtocolError,
        match=r"Missing imported K24 GMM reference artifacts: .*balanced_gmm_gap_summary\.csv",
    ):
        validate_imported_artifacts(
            (missing,),
            missing_message="Missing imported K24 GMM reference artifacts: {missing}",
        )


def test_validate_imported_artifacts_rejects_non_pass_leakage_report_with_legacy_message(tmp_path: Path) -> None:
    leakage = tmp_path / "reports" / "leakage_report.json"
    write_json(leakage, {"status": "FAIL"})

    with pytest.raises(ProtocolError, match=rf"Imported leakage report is not PASS: {leakage}"):
        validate_imported_artifacts(
            (leakage,),
            missing_message="Missing imported source-union GMM reference artifacts: {missing}",
        )


def test_write_csv_rows_preserves_union_column_order_and_empty_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "tables" / "union.csv"

    write_csv_rows(path, [{"a": 1, "b": 2}, {"c": 3, "a": 4}])

    assert path.read_text(encoding="utf-8") == "a,b,c\n1,2,\n4,,3\n"


def test_write_csv_rows_respects_explicit_columns(tmp_path: Path) -> None:
    path = tmp_path / "tables" / "explicit.csv"

    write_csv_rows(
        path,
        [{"method": "support_nelbo_top2_geom", "status": "ok", "extra": "ignored"}],
        columns=("method", "status"),
    )

    assert path.read_text(encoding="utf-8") == "method,status\nsupport_nelbo_top2_geom,ok\n"


def test_write_json_uses_sorted_keys_and_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "payload.json"

    write_json(path, {"z": 1, "a": {"b": True}})

    assert path.read_text(encoding="utf-8") == '{\n  "a": {\n    "b": true\n  },\n  "z": 1\n}\n'


def test_smoke_artifacts_write_contract(tmp_path: Path) -> None:
    cfg = load_config("configs/camelyon17_virchow2_legacy/target_support32_virchow2_cvae_top2_v1.yaml")
    root = run_artifact_contract_smoke(cfg, artifact_root=tmp_path / "artifacts")
    missing = [rel for rel in REQUIRED_OUTPUTS if not (root / rel).exists()]
    assert not missing


def test_synthetic_smoke_runs_mini_cvae_and_writes_nonempty_tables(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    cfg = load_config("configs/camelyon17_virchow2_legacy/target_support32_virchow2_cvae_top2_v1.yaml")
    root = run_synthetic_smoke(cfg, artifact_root=tmp_path / "synthetic_smoke")
    assert (root / "tables" / "support_nelbo_routing_scores.csv").read_text(encoding="utf-8").count("\n") > 1
    assert "support_nelbo_top2_geom" in (root / "tables" / "all_expert_downstream_matrix.csv").read_text(
        encoding="utf-8"
    )
