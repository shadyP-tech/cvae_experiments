from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.cli import build_parser
from src.config.load_config import load_config
from src.config.schema import validate_config
from src.experiments.registry import EXPERIMENT_REGISTRY, create_experiment


def test_quarantined_modes_are_not_registered() -> None:
    assert "legacy_routed_cvae" not in EXPERIMENT_REGISTRY
    assert "latent_compatibility" not in EXPERIMENT_REGISTRY

    for mode in ["legacy_routed_cvae", "latent_compatibility"]:
        with pytest.raises(ValueError, match="quarantined"):
            create_experiment(mode)


def test_cli_default_points_to_protocol_safe_config() -> None:
    args = build_parser().parse_args([])
    assert args.config == Path("configs/experiments/breakhis/learned_utility_routing_v1.yaml")

    cfg = load_config(PROJECT_ROOT / args.config)
    assert cfg["experiment"]["mode"] == "learned_utility_routing"


def test_active_experiment_configs_have_supported_modes() -> None:
    supported = {"hybrid_ablation", "learned_utility_routing"}
    for path in (PROJECT_ROOT / "configs" / "experiments").rglob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        mode = str((cfg.get("experiment") or {}).get("mode", "")).strip()
        assert mode, f"{path} must declare experiment.mode"
        assert mode in supported, f"{path} has unsupported experiment.mode={mode}"


def test_support_estimated_utility_v2_config_is_unlabeled_and_grid_locked() -> None:
    path = PROJECT_ROOT / "configs" / "experiments" / "camelyon17" / "camelyon17_support_estimated_utility_routing_v2.yaml"
    cfg = load_config(path)
    support_cfg = cfg["learned_utility"]["support_response_routing"]
    utility_cfg = support_cfg["support_utility"]

    assert cfg["experiment"]["name"] == "camelyon17_support_estimated_utility_routing_v2"
    assert support_cfg["sampling_policies"] == ["random"]
    assert utility_cfg["enabled"] is True
    assert utility_cfg["alpha_grid"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert utility_cfg["alpha_selection_policy"] == "source_inner_gap_min_with_non_regression"
    assert utility_cfg["require_unlabeled_support"] is True

    invalid = yaml.safe_load(path.read_text(encoding="utf-8"))
    invalid["learned_utility"]["support_response_routing"]["sampling_policies"] = ["class_balanced"]
    with pytest.raises(ValueError, match="random support sampling only"):
        validate_config(invalid)


def test_breakhis_support_estimated_utility_config_is_protocol_locked() -> None:
    path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "breakhis"
        / "breakhis_support_estimated_utility_routing_v1.yaml"
    )
    cfg = load_config(path)
    validate_config(cfg)

    assert cfg["experiment"]["name"] == "breakhis_support_estimated_utility_routing_v1"
    assert cfg["data"]["dataset_type"] == "breakhis"
    assert cfg["data"]["dataset_domain_semantics"] == "breakhis_magnification"
    assert cfg["data"]["magnifications"] == [40, 100, 200, 400]
    assert cfg["data"]["require_patient_ids"] is True
    assert cfg["features"]["backbone_type"] == "dinov2_vitb14"

    support_cfg = cfg["learned_utility"]["support_response_routing"]
    assert support_cfg["support_sizes"] == [4, 8, 16, 32]
    assert support_cfg["support_seeds"] == [17, 23, 31]
    assert support_cfg["sampling_policies"] == ["random"]
    assert support_cfg["support_utility"]["require_unlabeled_support"] is True
    assert support_cfg["random_floor"] == {
        "enabled": True,
        "adoption_eligible": False,
        "diagnostic_only": True,
        "report_only": True,
    }


def test_breakhis_support_estimated_utility_config_rejects_protocol_drift() -> None:
    path = (
        PROJECT_ROOT
        / "configs"
        / "experiments"
        / "breakhis"
        / "breakhis_support_estimated_utility_routing_v1.yaml"
    )
    invalid_domains = yaml.safe_load(path.read_text(encoding="utf-8"))
    invalid_domains["data"]["magnifications"] = [40, 100, 400]
    with pytest.raises(ValueError, match=r"exactly \[40, 100, 200, 400\]"):
        validate_config(invalid_domains)

    missing_patient_gate = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing_patient_gate["data"]["require_patient_ids"] = False
    with pytest.raises(ValueError, match="require_patient_ids must be true"):
        validate_config(missing_patient_gate)

    class_balanced = yaml.safe_load(path.read_text(encoding="utf-8"))
    class_balanced["learned_utility"]["support_response_routing"]["sampling_policies"] = ["class_balanced"]
    with pytest.raises(ValueError, match=r"exactly \['random'\]"):
        validate_config(class_balanced)

    random_floor_disabled = yaml.safe_load(path.read_text(encoding="utf-8"))
    random_floor_disabled["learned_utility"]["support_response_routing"]["random_floor"]["enabled"] = False
    with pytest.raises(ValueError, match="random_floor.enabled must be true"):
        validate_config(random_floor_disabled)

    random_floor_adoptable = yaml.safe_load(path.read_text(encoding="utf-8"))
    random_floor_adoptable["learned_utility"]["support_response_routing"]["random_floor"][
        "adoption_eligible"
    ] = True
    with pytest.raises(ValueError, match="must not be adoption eligible"):
        validate_config(random_floor_adoptable)

    assert 3 * 4 * 4 * 3 == 144
    assert 144 * 2 == 288
    assert 3 * 4 * 3 * 3 * sum([4, 8, 16, 32]) == 6480


def test_ae_first_routing_configs_are_protocol_locked() -> None:
    for dataset in ["breakhis", "camelyon17"]:
        path = PROJECT_ROOT / "configs" / "experiments" / dataset / "learned_utility_ae_first_routing_v1.yaml"
        cfg = load_config(path)
        validate_config(cfg)
        ae_first = cfg["learned_utility"]["autoencoder_proxy"]["ae_first_routing"]
        assert cfg["experiment"]["name"] == "learned_utility_ae_first_routing_v1"
        assert ae_first["primary_method"] == "ae_first_margin_gated_v1"
        assert ae_first["fallback_baseline"] == "source_prior_fallback"
        assert ae_first["margin_thresholds"][-1] == "__inf__"
        assert ae_first["metadata_auxiliary_features"] is True


def test_ae_utility_calibrator_configs_are_protocol_locked() -> None:
    for dataset in ["breakhis", "camelyon17"]:
        path = PROJECT_ROOT / "configs" / "experiments" / dataset / "learned_utility_ae_utility_calibrator_v1.yaml"
        cfg = load_config(path)
        validate_config(cfg)
        utility = cfg["learned_utility"]["autoencoder_proxy"]["utility_calibrator"]
        assert cfg["experiment"]["name"] == "learned_utility_ae_utility_calibrator_v1"
        assert utility["primary_method"] == "ae_utility_calibrated_safe_override_v1"
        assert utility["model_types"] == ["ridge_delta"]
        assert utility["primary_model_type"] == "ridge_delta"
        assert utility["fallback_policy"] == "ae_argmin_zscore"
        assert utility["feature_sets_primary"] == ["ae_core", "ae_quality"]
        assert "ae_metadata" in utility["feature_sets_diagnostic"]
        assert utility["delta_thresholds"][-1] == "__inf__"


def test_ae_utility_calibrator_v2_configs_are_protocol_locked() -> None:
    for dataset in ["breakhis", "camelyon17"]:
        path = PROJECT_ROOT / "configs" / "experiments" / dataset / "learned_utility_ae_utility_calibrator_v2.yaml"
        cfg = load_config(path)
        validate_config(cfg)
        utility = cfg["learned_utility"]["autoencoder_proxy"]["utility_calibrator"]
        assert cfg["experiment"]["name"] == "learned_utility_ae_utility_calibrator_v2"
        assert utility["primary_method"] == "ae_utility_calibrated_consensus_safe_override_v2"
        assert utility["model_types"] == ["ridge_delta_consensus"]
        assert utility["primary_model_type"] == "ridge_delta_consensus"
        assert utility["fallback_policy"] == "ae_argmin_zscore"
        assert utility["feature_sets_primary"] == ["ae_consensus_core", "ae_consensus_quality"]
        assert not any("metadata" in name for name in utility["feature_sets_primary"])
        assert "ae_metadata_consensus" in utility["feature_sets_diagnostic"]
        assert utility["delta_thresholds"][-1] == "__inf__"
        assert utility["ensemble_strategy"] == "source_domain_leave_one_plus_full"


def test_ae_utility_calibrator_harm_veto_v13_config_is_protocol_locked() -> None:
    path = PROJECT_ROOT / "configs" / "experiments" / "camelyon17" / "learned_utility_ae_utility_calibrator_harm_veto_v13.yaml"
    cfg = load_config(path)
    validate_config(cfg)
    utility = cfg["learned_utility"]["autoencoder_proxy"]["utility_calibrator"]
    harm_veto = utility["harm_veto"]
    assert cfg["experiment"]["name"] == "learned_utility_ae_utility_calibrator_harm_veto_v13"
    assert utility["primary_method"] == "ae_utility_calibrated_v1_harm_veto_safe_override_v13"
    assert utility["selection_mode"] == "v1_harm_veto_v13"
    assert utility["model_types"] == ["ridge_delta"]
    assert utility["primary_model_type"] == "ridge_delta"
    assert utility["fallback_policy"] == "ae_argmin_zscore"
    assert utility["feature_sets_primary"] == ["ae_core", "ae_quality"]
    assert utility["feature_sets_diagnostic"] == []
    assert harm_veto["veto_score_model"] == "logistic_harm_score"
    assert harm_veto["veto_thresholds"][-1] == "__inf__"


def test_quarantined_entrypoints_fail_fast() -> None:
    checks = [
        ([sys.executable, "scripts/run_learned_compatibility_loqdo.py"], "target expert"),
        (["bash", "scripts/run_learned_compatibility_breakhis_seed_sweep.sh"], "legacy LOQDO"),
        (["bash", "scripts/run_learned_compatibility_camelyon17_seed_sweep.sh"], "legacy LOQDO"),
        (["bash", "scripts/run_legacy_conditioning_seed_sweep.sh"], "legacy routed-CVAE"),
        (["bash", "scripts/run_metadata_aux_constraint_seed_sweep.sh"], "legacy routed mode"),
        (["bash", "scripts/run_metadata_conditional_prior_seed_sweep.sh"], "legacy routed mode"),
    ]
    for cmd, expected in checks:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
        assert "BLOCKED" in result.stderr
        assert expected in result.stderr


def test_thesis_facing_manifests_do_not_reference_quarantined_artifacts() -> None:
    forbidden_terms = [
        "quarantined",
        "learned_compatibility_loqdo",
        "response_dev",
        "routed_cvae_v1_seed_sweep",
        "legacy_std_v1",
    ]
    manifest_paths = sorted((PROJECT_ROOT / "results" / "comparison_tables").glob("*manifest*.txt"))
    assert manifest_paths
    for path in manifest_paths:
        text = path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in text, f"{path} references forbidden artifact term: {term}"
