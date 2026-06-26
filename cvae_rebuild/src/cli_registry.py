from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from component_union_mass_bagged import (
    load_mass_bagged_component_union_config,
    run_mass_bagged_component_union,
)
from component_union_tailrisk_anchored_mass_bagged import (
    load_fixed_beta050_positive_union_config,
    load_harm_gated_positive_union_config,
    load_multipanel_tailrisk_component_union_config,
    load_source_inner_positive_union_config,
    load_tailrisk_anchored_component_union_config,
    run_fixed_beta050_positive_union,
    run_harm_gated_positive_union,
    run_multipanel_tailrisk_component_union,
    run_source_inner_positive_union,
    run_tailrisk_anchored_component_union,
)
from config import load_config
from covariance_prior import load_covariance_prior_config, run_covariance_prior_confirmation
from covariance_shrinkage import load_covariance_shrinkage_config, run_covariance_shrinkage_stability
from covariance_viability import load_covariance_viability_config, run_covariance_prior_viability_audit
from decentralized_adaptive_gmm_prior import (
    load_decentralized_adaptive_gmm_prior_config,
    run_decentralized_adaptive_gmm_prior,
)
from decentralized_component_union_prior import (
    load_decentralized_component_union_prior_config,
    run_decentralized_component_union_prior,
)
from decentralized_k16_gmm_prior import (
    load_decentralized_k16_gmm_prior_config,
    run_decentralized_k16_gmm_prior,
)
from decentralized_pruned_adaptive_equal_all4_prior import (
    load_pruned_adaptive_equal_all4_config,
    run_pruned_adaptive_equal_all4_confirmation,
)
from decentralized_reliability_top3_gmm_prior import (
    load_decentralized_reliability_top3_gmm_prior_config,
    run_decentralized_reliability_top3_gmm_prior,
)
from decentralized_reliability_weighted_gmm_prior import (
    load_decentralized_reliability_weighted_gmm_prior_config,
    run_decentralized_reliability_weighted_gmm_prior,
)
from decentralized_source_inner_transfer_top3_gmm_prior import (
    load_decentralized_source_inner_transfer_top3_gmm_prior_config,
    run_decentralized_source_inner_transfer_top3_gmm_prior,
)
from decentralized_support8_top3_tau05_gmm_prior import (
    load_decentralized_support8_top3_tau05_gmm_prior_config,
    run_decentralized_support8_top3_tau05_gmm_prior,
)
from decentralized_support_nelbo_reliability_gmm_prior import (
    load_decentralized_support_nelbo_reliability_gmm_prior_config,
    run_decentralized_support_nelbo_reliability_gmm_prior,
)
from dense_reliability_tailshield_random_mass_bag import (
    load_dense_tailshield_random_mass_bag_config,
    run_dense_reliability_tailshield_random_mass_bag,
)
from labeled_support_random_vs_dense_policy_calibration import (
    load_labeled_support_policy_calibration_config,
    run_labeled_support_policy_calibration,
)
from experiments.prior_sampling.posthoc_gmm_pca128 import (
    PCA128_POSTHOC_GMM_NAME,
    load_pca128_posthoc_gmm_config,
    run_pca128_posthoc_gmm_prior,
)
from experiments.support_selection.midogpp_support_nelbo_routing import (
    load_midogpp_support_nelbo_routing_config,
    run_midogpp_support_nelbo_routing,
)
from paired_component_coverage_audit import (
    load_paired_component_coverage_audit_config,
    run_paired_component_coverage_audit,
)
from paired_dense_all4_reliability_confirmation import (
    load_dense_late_all_sources_reliability_config,
    load_paired_dense_all4_reliability_config,
    run_dense_late_all_sources_reliability,
    run_paired_dense_all4_reliability_confirmation,
)
from preservation import load_preservation_config, run_preservation_diagnosis
from preservation_repair import load_repair_config, run_preservation_repair
from preservation_sampling import load_sampling_config, run_preservation_sampling
from prior_calibration import load_prior_calibration_config, run_prior_calibration
from source_inner_harmful_source_suppression import (
    load_harmful_source_suppression_config,
    run_harmful_source_suppression,
)
from source_inner_validated_dense_component_hybrid import (
    load_source_inner_validated_hybrid_config,
    run_source_inner_validated_dense_component_hybrid,
)
from source_union_balanced_gmm_prior import (
    load_source_union_balanced_gmm_prior_config,
    run_source_union_balanced_gmm_prior,
)
from source_union_gmm_prior import load_source_union_gmm_prior_config, run_source_union_gmm_prior
from source_union_k24_gmm_prior import load_source_union_k24_gmm_prior_config, run_source_union_k24_gmm_prior
from support_calibrated_component_union_prior import (
    load_support_calibrated_component_union_config,
    run_support_calibrated_component_union_prior,
)
from target_support_regime_risk_gated_component_union import (
    load_target_support_regime_risk_gate_config,
    run_target_support_regime_risk_gated_component_union,
)

ConfigLoader = Callable[[str | Path], object]
CommandRunner = Callable[[Any], Path]


@dataclass(frozen=True)
class DiagnosisCommand:
    command: str
    help: str
    load_config: ConfigLoader
    run: CommandRunner
    validation_keys: tuple[str, ...] = ()


DIAGNOSIS_COMMANDS: tuple[DiagnosisCommand, ...] = (
    DiagnosisCommand(
        "diagnose-preservation",
        "Run the Virchow2-CVAE preservation diagnosis.",
        load_preservation_config,
        run_preservation_diagnosis,
    ),
    DiagnosisCommand(
        "diagnose-preservation-repair",
        "Run the Virchow2-CVAE preservation repair diagnosis.",
        load_repair_config,
        run_preservation_repair,
    ),
    DiagnosisCommand(
        "diagnose-preservation-sampling",
        "Run the Virchow2-CVAE PCA64 sampling continuation.",
        load_sampling_config,
        run_preservation_sampling,
    ),
    DiagnosisCommand(
        "diagnose-latent-prior-calibration",
        "Run the Virchow2-CVAE latent prior calibration diagnostic.",
        load_prior_calibration_config,
        run_prior_calibration,
    ),
    DiagnosisCommand(
        "diagnose-covariance-prior-confirmation",
        "Run the Virchow2-CVAE covariance prior confirmation diagnostic.",
        load_covariance_prior_config,
        run_covariance_prior_confirmation,
    ),
    DiagnosisCommand(
        "diagnose-covariance-prior-viability",
        "Run the Virchow2-CVAE covariance prior viability audit.",
        load_covariance_viability_config,
        run_covariance_prior_viability_audit,
    ),
    DiagnosisCommand(
        "diagnose-covariance-shrinkage-stability",
        "Run the Virchow2-CVAE covariance shrinkage stability diagnostic.",
        load_covariance_shrinkage_config,
        run_covariance_shrinkage_stability,
    ),
    DiagnosisCommand(
        "diagnose-source-union-gmm-prior",
        "Run the Virchow2-CVAE source-union GMM prior diagnostic.",
        load_source_union_gmm_prior_config,
        run_source_union_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-source-union-balanced-gmm-prior",
        "Run the Virchow2-CVAE source-union center-balanced GMM prior diagnostic.",
        load_source_union_balanced_gmm_prior_config,
        run_source_union_balanced_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-source-union-k24-gmm-prior",
        "Run the Virchow2-CVAE source-union K24 GMM prior locked follow-up.",
        load_source_union_k24_gmm_prior_config,
        run_source_union_k24_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-k16-gmm-prior",
        "Run the Virchow2-CVAE decentralized K16 summary-composition preservation test.",
        load_decentralized_k16_gmm_prior_config,
        run_decentralized_k16_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-adaptive-gmm-prior",
        "Run the Virchow2-CVAE adaptive source-local latent summary preservation test.",
        load_decentralized_adaptive_gmm_prior_config,
        run_decentralized_adaptive_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-component-union-prior",
        "Run the Virchow2-CVAE decentralized component-level prior composition audit.",
        load_decentralized_component_union_prior_config,
        run_decentralized_component_union_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-component-union-reliability-shrink050",
        "Run the Virchow2-CVAE component-union reliability shrink050 confirmation audit.",
        load_decentralized_component_union_prior_config,
        run_decentralized_component_union_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-component-union-mass-bagged",
        "Run the Virchow2-CVAE mass-uncertainty bagged component-union audit.",
        load_mass_bagged_component_union_config,
        run_mass_bagged_component_union,
    ),
    DiagnosisCommand(
        "diagnose-component-union-tailrisk-anchored-mass-bagged",
        "Run the Virchow2-CVAE tail-risk anchored mass-bagged component-union audit.",
        load_tailrisk_anchored_component_union_config,
        run_tailrisk_anchored_component_union,
    ),
    DiagnosisCommand(
        "diagnose-component-union-tailrisk-multipanel-mass-bagged",
        "Run the Virchow2-CVAE multipanel tail-risk mass-bag stabilization audit.",
        load_multipanel_tailrisk_component_union_config,
        run_multipanel_tailrisk_component_union,
    ),
    DiagnosisCommand(
        "diagnose-source-inner-class-conditional-positive-union",
        "Run the Virchow2-CVAE source-inner class-conditional positive-union audit.",
        load_source_inner_positive_union_config,
        run_source_inner_positive_union,
        validation_keys=("source_inner_class_conditional_positive_union",),
    ),
    DiagnosisCommand(
        "diagnose-fixed-beta050-positive-union-confirmation",
        "Run the Virchow2-CVAE fixed beta050 positive-union fresh-seed confirmation.",
        load_fixed_beta050_positive_union_config,
        run_fixed_beta050_positive_union,
        validation_keys=("fixed_beta050_positive_union_confirmation",),
    ),
    DiagnosisCommand(
        "diagnose-source-inner-harm-gated-positive-union",
        "Run the Virchow2-CVAE source-inner harm-gated positive-union confirmation.",
        load_harm_gated_positive_union_config,
        run_harm_gated_positive_union,
        validation_keys=("source_inner_harm_gated_positive_union",),
    ),
    DiagnosisCommand(
        "diagnose-dense-reliability-tailshield-random-mass-bag",
        "Run the dense reliability tail-shield over random mass-bag component-union audit.",
        load_dense_tailshield_random_mass_bag_config,
        run_dense_reliability_tailshield_random_mass_bag,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-pruned-adaptive-equal-all4-prior",
        "Run the pruned adaptive equal-all4 late-geometric confirmation test.",
        load_pruned_adaptive_equal_all4_config,
        run_pruned_adaptive_equal_all4_confirmation,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-reliability-weighted-gmm-prior",
        "Run the Virchow2-CVAE source-local reliability-weighted decentralized composition test.",
        load_decentralized_reliability_weighted_gmm_prior_config,
        run_decentralized_reliability_weighted_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-reliability-top3-gmm-prior",
        "Run the locked D1.4 source-local reliability top-3 decentralized composition test.",
        load_decentralized_reliability_top3_gmm_prior_config,
        run_decentralized_reliability_top3_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-source-inner-transfer-top3-gmm-prior",
        "Run the locked D1.5 source-inner off-diagonal transfer drop-one confirmation test.",
        load_decentralized_source_inner_transfer_top3_gmm_prior_config,
        run_decentralized_source_inner_transfer_top3_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-support-nelbo-reliability-gmm-prior",
        "Run the Virchow2-CVAE support-NELBO x reliability decentralized composition test.",
        load_decentralized_support_nelbo_reliability_gmm_prior_config,
        run_decentralized_support_nelbo_reliability_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-decentralized-support8-top3-tau05-gmm-prior",
        "Run the locked D1.3.1 support-size-8 top-3 tau-0.5 confirmation test.",
        load_decentralized_support8_top3_tau05_gmm_prior_config,
        run_decentralized_support8_top3_tau05_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-support-calibrated-component-union-prior",
        "Run the support-calibrated component-union prior audit.",
        load_support_calibrated_component_union_config,
        run_support_calibrated_component_union_prior,
    ),
    DiagnosisCommand(
        "diagnose-paired-dense-all4-reliability",
        "Run the paired dense-all4 source-only reliability confirmation audit.",
        load_paired_dense_all4_reliability_config,
        run_paired_dense_all4_reliability_confirmation,
        validation_keys=("paired_dense_all4_reliability",),
    ),
    DiagnosisCommand(
        "diagnose-dense-late-all-sources-reliability",
        "Run the MIDOG++ dense-late all-source reliability pilot.",
        load_dense_late_all_sources_reliability_config,
        run_dense_late_all_sources_reliability,
        validation_keys=("dense_late_all_sources_reliability",),
    ),
    DiagnosisCommand(
        "diagnose-paired-component-coverage-audit",
        "Run the paired dense-all4 component coverage sampling-fidelity audit.",
        load_paired_component_coverage_audit_config,
        run_paired_component_coverage_audit,
    ),
    DiagnosisCommand(
        "diagnose-source-inner-validated-dense-component-hybrid",
        "Run the source-inner validated dense/component binary-gate confirmation audit.",
        load_source_inner_validated_hybrid_config,
        run_source_inner_validated_dense_component_hybrid,
    ),
    DiagnosisCommand(
        "diagnose-source-inner-harmful-source-suppression-random-mass-bag",
        "Run the source-inner harmful-source suppression over random mass-bag component-union audit.",
        load_harmful_source_suppression_config,
        run_harmful_source_suppression,
    ),
    DiagnosisCommand(
        "diagnose-target-support-regime-risk-gated-component-union",
        "Run the target-support regime-risk gated component-policy audit.",
        load_target_support_regime_risk_gate_config,
        run_target_support_regime_risk_gated_component_union,
    ),
    DiagnosisCommand(
        "diagnose-labeled-support-random-vs-dense-policy-calibration",
        "Run the Tier 2 labeled-support random-vs-dense policy calibration audit.",
        load_labeled_support_policy_calibration_config,
        run_labeled_support_policy_calibration,
    ),
    DiagnosisCommand(
        "diagnose-pca128-posthoc-gmm-prior",
        "Run the pca128 post-hoc class-conditional GMM prior feasibility audit.",
        load_pca128_posthoc_gmm_config,
        run_pca128_posthoc_gmm_prior,
    ),
    DiagnosisCommand(
        "diagnose-midogpp-support-nelbo-routing",
        "Run the MIDOG++ routing-stage support-NELBO selection surface.",
        load_midogpp_support_nelbo_routing_config,
        run_midogpp_support_nelbo_routing,
        validation_keys=("midogpp_support_nelbo_routing",),
    ),
)

COMMANDS_BY_NAME: Mapping[str, DiagnosisCommand] = {
    command.command: command for command in DIAGNOSIS_COMMANDS
}

VALIDATION_COMMANDS_BY_KEY: Mapping[str, DiagnosisCommand] = {
    key: command
    for command in DIAGNOSIS_COMMANDS
    for key in command.validation_keys
}


def load_config_for_validation(path: str | Path) -> object:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    keys = set(str(key) for key in data)
    for key in sorted(keys):
        command = VALIDATION_COMMANDS_BY_KEY.get(key)
        if command is not None:
            return command.load_config(source)
    experiment = data.get("experiment")
    if isinstance(experiment, Mapping) and str(experiment.get("name")) == PCA128_POSTHOC_GMM_NAME:
        return COMMANDS_BY_NAME["diagnose-pca128-posthoc-gmm-prior"].load_config(source)
    return load_config(source)
