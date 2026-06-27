from cli_registry import COMMANDS_BY_NAME, DIAGNOSIS_COMMANDS, VALIDATION_COMMANDS_BY_KEY
from component_union_tailrisk_anchored_mass_bagged import (
    load_fixed_beta050_positive_union_config,
    load_harm_gated_positive_union_config,
    load_source_inner_positive_union_config,
    load_tailrisk_anchored_component_union_config,
    run_fixed_beta050_positive_union,
    run_harm_gated_positive_union,
    run_source_inner_positive_union,
    run_tailrisk_anchored_component_union,
)
from decentralized_component_union_prior import (
    load_decentralized_component_union_prior_config,
    run_decentralized_component_union_prior,
)
from labeled_support_random_vs_dense_policy_calibration import (
    load_labeled_support_policy_calibration_config,
    run_labeled_support_policy_calibration,
)
from midogpp_condition_audit import load_midogpp_condition_audit_config, run_midogpp_condition_audit
from midogpp_preservation_gate import load_midogpp_preservation_gate_config, run_midogpp_preservation_gate
from midogpp_preservation_sanity import load_midogpp_preservation_sanity_config, run_midogpp_preservation_sanity
from experiments.prior_sampling.posthoc_gmm_pca128 import (
    load_pca128_posthoc_gmm_config,
    run_pca128_posthoc_gmm_prior,
)
from experiments.support_selection.midogpp_support_nelbo_routing import (
    load_midogpp_support_nelbo_routing_config,
    run_midogpp_support_nelbo_routing,
    scaffold_midogpp_support_nelbo_routing_inputs,
)
from support_calibrated_component_union_prior import (
    load_support_calibrated_component_union_config,
    run_support_calibrated_component_union_prior,
)
from target_support_regime_risk_gated_component_union import (
    load_target_support_regime_risk_gate_config,
    run_target_support_regime_risk_gated_component_union,
)


EXPECTED_DIAGNOSIS_COMMANDS = {
    "diagnose-preservation",
    "diagnose-preservation-repair",
    "diagnose-preservation-sampling",
    "diagnose-latent-prior-calibration",
    "diagnose-covariance-prior-confirmation",
    "diagnose-covariance-prior-viability",
    "diagnose-covariance-shrinkage-stability",
    "diagnose-source-union-gmm-prior",
    "diagnose-source-union-balanced-gmm-prior",
    "diagnose-source-union-k24-gmm-prior",
    "diagnose-decentralized-k16-gmm-prior",
    "diagnose-decentralized-adaptive-gmm-prior",
    "diagnose-decentralized-component-union-prior",
    "diagnose-decentralized-component-union-reliability-shrink050",
    "diagnose-decentralized-component-union-mass-bagged",
    "diagnose-component-union-tailrisk-anchored-mass-bagged",
    "diagnose-component-union-tailrisk-multipanel-mass-bagged",
    "diagnose-source-inner-class-conditional-positive-union",
    "diagnose-fixed-beta050-positive-union-confirmation",
    "diagnose-source-inner-harm-gated-positive-union",
    "diagnose-dense-reliability-tailshield-random-mass-bag",
    "diagnose-decentralized-pruned-adaptive-equal-all4-prior",
    "diagnose-decentralized-reliability-weighted-gmm-prior",
    "diagnose-decentralized-reliability-top3-gmm-prior",
    "diagnose-decentralized-source-inner-transfer-top3-gmm-prior",
    "diagnose-decentralized-support-nelbo-reliability-gmm-prior",
    "diagnose-decentralized-support8-top3-tau05-gmm-prior",
    "diagnose-support-calibrated-component-union-prior",
    "diagnose-paired-dense-all4-reliability",
    "diagnose-dense-late-all-sources-reliability",
    "diagnose-paired-component-coverage-audit",
    "diagnose-source-inner-validated-dense-component-hybrid",
    "diagnose-source-inner-harmful-source-suppression-random-mass-bag",
    "diagnose-target-support-regime-risk-gated-component-union",
    "diagnose-labeled-support-random-vs-dense-policy-calibration",
    "diagnose-midogpp-preservation-sanity",
    "diagnose-midogpp-preservation-condition-audit",
    "diagnose-midogpp-preservation-gate-pca128",
    "diagnose-pca128-posthoc-gmm-prior",
    "diagnose-midogpp-support-nelbo-routing",
    "scaffold-midogpp-support-nelbo-routing-inputs",
}


def test_cli_registry_command_names_are_unique_and_complete() -> None:
    names = [command.command for command in DIAGNOSIS_COMMANDS]

    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_DIAGNOSIS_COMMANDS
    assert set(COMMANDS_BY_NAME) == EXPECTED_DIAGNOSIS_COMMANDS


def test_cli_registry_keeps_component_union_confirmation_alias_explicit() -> None:
    primary = COMMANDS_BY_NAME["diagnose-decentralized-component-union-prior"]
    shrink050 = COMMANDS_BY_NAME["diagnose-decentralized-component-union-reliability-shrink050"]

    assert primary.load_config is load_decentralized_component_union_prior_config
    assert primary.run is run_decentralized_component_union_prior
    assert shrink050.load_config is load_decentralized_component_union_prior_config
    assert shrink050.run is run_decentralized_component_union_prior


def test_cli_registry_keeps_source_only_positive_union_commands_separate() -> None:
    tailrisk = COMMANDS_BY_NAME["diagnose-component-union-tailrisk-anchored-mass-bagged"]
    positive_union = COMMANDS_BY_NAME["diagnose-source-inner-class-conditional-positive-union"]
    fixed_beta050 = COMMANDS_BY_NAME["diagnose-fixed-beta050-positive-union-confirmation"]
    harm_gated = COMMANDS_BY_NAME["diagnose-source-inner-harm-gated-positive-union"]

    assert tailrisk.load_config is load_tailrisk_anchored_component_union_config
    assert tailrisk.run is run_tailrisk_anchored_component_union
    assert positive_union.load_config is load_source_inner_positive_union_config
    assert positive_union.run is run_source_inner_positive_union
    assert fixed_beta050.load_config is load_fixed_beta050_positive_union_config
    assert fixed_beta050.run is run_fixed_beta050_positive_union
    assert harm_gated.load_config is load_harm_gated_positive_union_config
    assert harm_gated.run is run_harm_gated_positive_union


def test_cli_registry_validation_keys_are_explicit_for_non_default_loaders() -> None:
    assert VALIDATION_COMMANDS_BY_KEY["source_inner_class_conditional_positive_union"] is COMMANDS_BY_NAME[
        "diagnose-source-inner-class-conditional-positive-union"
    ]
    assert VALIDATION_COMMANDS_BY_KEY["fixed_beta050_positive_union_confirmation"] is COMMANDS_BY_NAME[
        "diagnose-fixed-beta050-positive-union-confirmation"
    ]
    assert VALIDATION_COMMANDS_BY_KEY["source_inner_harm_gated_positive_union"] is COMMANDS_BY_NAME[
        "diagnose-source-inner-harm-gated-positive-union"
    ]
    assert VALIDATION_COMMANDS_BY_KEY["paired_dense_all4_reliability"] is COMMANDS_BY_NAME[
        "diagnose-paired-dense-all4-reliability"
    ]
    assert VALIDATION_COMMANDS_BY_KEY["dense_late_all_sources_reliability"] is COMMANDS_BY_NAME[
        "diagnose-dense-late-all-sources-reliability"
    ]


def test_cli_registry_keeps_support_regimes_separate_from_source_only() -> None:
    support_calibrated = COMMANDS_BY_NAME["diagnose-support-calibrated-component-union-prior"]
    target_support = COMMANDS_BY_NAME["diagnose-target-support-regime-risk-gated-component-union"]
    labeled_support = COMMANDS_BY_NAME["diagnose-labeled-support-random-vs-dense-policy-calibration"]

    assert support_calibrated.load_config is load_support_calibrated_component_union_config
    assert support_calibrated.run is run_support_calibrated_component_union_prior
    assert target_support.load_config is load_target_support_regime_risk_gate_config
    assert target_support.run is run_target_support_regime_risk_gated_component_union
    assert labeled_support.load_config is load_labeled_support_policy_calibration_config
    assert labeled_support.run is run_labeled_support_policy_calibration


def test_cli_registry_wires_pca128_posthoc_gmm_audit() -> None:
    command = COMMANDS_BY_NAME["diagnose-pca128-posthoc-gmm-prior"]

    assert command.load_config is load_pca128_posthoc_gmm_config
    assert command.run is run_pca128_posthoc_gmm_prior


def test_cli_registry_wires_midogpp_support_nelbo_routing() -> None:
    command = COMMANDS_BY_NAME["diagnose-midogpp-support-nelbo-routing"]
    scaffold = COMMANDS_BY_NAME["scaffold-midogpp-support-nelbo-routing-inputs"]

    assert command.load_config is load_midogpp_support_nelbo_routing_config
    assert command.run is run_midogpp_support_nelbo_routing
    assert scaffold.load_config is load_midogpp_support_nelbo_routing_config
    assert scaffold.run is scaffold_midogpp_support_nelbo_routing_inputs
    assert VALIDATION_COMMANDS_BY_KEY["midogpp_support_nelbo_routing"] is command


def test_cli_registry_wires_midogpp_preservation_diagnostics() -> None:
    sanity = COMMANDS_BY_NAME["diagnose-midogpp-preservation-sanity"]
    condition = COMMANDS_BY_NAME["diagnose-midogpp-preservation-condition-audit"]
    gate = COMMANDS_BY_NAME["diagnose-midogpp-preservation-gate-pca128"]

    assert sanity.load_config is load_midogpp_preservation_sanity_config
    assert sanity.run is run_midogpp_preservation_sanity
    assert condition.load_config is load_midogpp_condition_audit_config
    assert condition.run is run_midogpp_condition_audit
    assert gate.load_config is load_midogpp_preservation_gate_config
    assert gate.run is run_midogpp_preservation_gate
