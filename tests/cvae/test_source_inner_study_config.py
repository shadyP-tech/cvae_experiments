from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from midogpp_thesis.cvae.preservation.source_inner_studies.config import (
    FISHER_SHRINKAGE_STUDY_NAME,
    LEARNED_PRIOR_STUDY_NAME,
    LearnedConditionalPriorStudyConfig,
    TaskFisherShrinkageStudyConfig,
    decision_contract_hash,
    load_source_inner_study_config,
    study_contract_payload,
    study_contract_hash,
)
from midogpp_thesis.cvae.preservation.source_inner_studies.contracts import (
    FISHER_SHRINKAGE_MODE,
    LEARNED_PRIOR_MODE,
    LEARNED_PRIOR_MODEL_FAMILY,
    LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
    STANDARD_MODEL_FAMILY,
    STANDARD_NORMAL_PRIOR,
    StudyTrainingKey,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


def test_prior_config_is_exact_v2_source_inner_contract(tmp_path: Path) -> None:
    path = _write(tmp_path, "prior.yaml", _prior_yaml())
    config = load_source_inner_study_config(path, expected_mode=LEARNED_PRIOR_MODE)

    assert isinstance(config, LearnedConditionalPriorStudyConfig)
    assert config.name == LEARNED_PRIOR_STUDY_NAME
    assert config.heldout_centers == ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    assert config.pca_dim == 128
    assert config.training_seeds == (17, 42, 101)
    assert config.generation_seeds == (17, 42, 101)
    assert config.minimum_real_bacc == 0.55
    assert config.prior_gradient_clip_norm == 5.0
    config_payload = study_contract_payload(config)
    assert "learned_prior_diagnostics" in config_payload
    assert config_payload["learned_prior_diagnostics"]["active_unit_threshold"] == 0.01
    assert len(study_contract_hash(config)) == 16
    assert len(decision_contract_hash(config)) == 16


def test_study_contract_binds_execution_device() -> None:
    config = _load_text_config(_prior_yaml())
    assert isinstance(config, LearnedConditionalPriorStudyConfig)
    assert study_contract_hash(config) != study_contract_hash(
        replace(config, device="cuda:0")
    )


def test_fisher_config_builds_literal_zero_and_paired_nonzero_variants(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "fisher.yaml", _fisher_yaml())
    config = load_source_inner_study_config(path, expected_mode=FISHER_SHRINKAGE_MODE)

    assert isinstance(config, TaskFisherShrinkageStudyConfig)
    assert config.name == FISHER_SHRINKAGE_STUDY_NAME
    zero = config.training_variant(
        model_family=STANDARD_MODEL_FAMILY,
        prior_family=STANDARD_NORMAL_PRIOR,
    )
    mild = config.training_variant(
        model_family=STANDARD_MODEL_FAMILY,
        prior_family=STANDARD_NORMAL_PRIOR,
        alpha=0.05,
        raw_fisher_state_hash="raw-fisher",
        objective_context_hash="derived-alpha-005",
    )
    assert zero.objective_id == "stochastic_isotropic_v1"
    assert zero.raw_fisher_state_hash == "none"
    assert mild.objective_id == "stochastic_task_fisher_v1"
    assert zero.hash != mild.hash
    assert zero.arm_neutral_pairing_payload() == mild.arm_neutral_pairing_payload()
    with pytest.raises(ProtocolError, match="outside the panel"):
        config.training_variant(
            model_family=STANDARD_MODEL_FAMILY,
            prior_family=STANDARD_NORMAL_PRIOR,
            alpha=1.0,
            raw_fisher_state_hash="raw-fisher",
            objective_context_hash="derived-alpha-100",
        )


def test_training_key_binds_full_identity_but_pairs_only_the_study_axis() -> None:
    prior_config = _load_text_config(_prior_yaml())
    assert isinstance(prior_config, LearnedConditionalPriorStudyConfig)
    baseline = prior_config.training_variant(
        model_family=STANDARD_MODEL_FAMILY,
        prior_family=STANDARD_NORMAL_PRIOR,
    )
    learned = prior_config.training_variant(
        model_family=LEARNED_PRIOR_MODEL_FAMILY,
        prior_family=LEARNED_CONDITIONAL_DIAGONAL_PRIOR,
    )
    key = _key(baseline)
    learned_key = replace(key, variant=learned)

    assert key.hash != learned_key.hash
    assert key.arm_neutral_pairing_hash == learned_key.arm_neutral_pairing_hash
    assert replace(key, training_seed=42).arm_neutral_pairing_hash != key.arm_neutral_pairing_hash
    payload = key.to_payload()
    assert {
        "study_id",
        "study_version",
        "outer_target_center",
        "inner_pseudo_target_center",
        "fit_row_hash",
        "frame_hash",
        "feature_cache_hash",
        "manifest_hash",
        "protocol_hash",
        "training_seed",
        "model_family",
        "prior_family",
        "alpha",
        "raw_fisher_state_hash",
        "objective_context_hash",
    }.issubset(payload)


@pytest.mark.parametrize(
    "old,new",
    [
        ("training_seeds: [17, 42, 101]", "training_seeds: [17, 42]"),
        ("pca_dim: 128", "pca_dim: 64"),
        ("prior_gradient_clip_norm: 5.0", "prior_gradient_clip_norm: 1.0"),
        ("minimum_real_bacc: 0.55", "minimum_real_bacc: 0.50"),
        ("target_evaluation_data_used: false", "target_evaluation_data_used: true"),
        ("may_feed_recipe_selection: false", "may_feed_recipe_selection: true"),
        ("family: stochastic_isotropic_v1", "family: stochastic_task_fisher_v1"),
    ],
)
def test_prior_config_rejects_protocol_drift(
    tmp_path: Path, old: str, new: str
) -> None:
    path = _write(tmp_path, "drift.yaml", _prior_yaml().replace(old, new))
    with pytest.raises(ProtocolError):
        load_source_inner_study_config(path)


def test_loader_rejects_wrong_expected_mode(tmp_path: Path) -> None:
    path = _write(tmp_path, "prior.yaml", _prior_yaml())
    with pytest.raises(ProtocolError, match="Expected source-inner study mode"):
        load_source_inner_study_config(path, expected_mode=FISHER_SHRINKAGE_MODE)


def _key(variant: object) -> StudyTrainingKey:
    return StudyTrainingKey(
        study_id=LEARNED_PRIOR_STUDY_NAME,
        study_version="v2",
        outer_target_center="0",
        inner_pseudo_target_center="1",
        fit_centers=("2", "3", "5", "6", "7", "8", "9"),
        fit_row_hash="fit-rows",
        frame_hash="pca128-frame",
        feature_cache_hash="virchow2-cache",
        manifest_hash="dataset-manifest",
        protocol_hash="protocol",
        training_seed=17,
        variant=variant,  # type: ignore[arg-type]
    )


def _load_text_config(text: str) -> object:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        return load_source_inner_study_config(
            _write(Path(directory), "config.yaml", text)
        )


def _write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def _common_yaml(*, name: str, mode: str) -> str:
    allowed = (
        "fully nested source-inner learned-prior study evidence only"
        if mode == LEARNED_PRIOR_MODE
        else "fully nested source-inner Task-Fisher shrinkage study evidence only"
    )
    return f"""
experiment:
  name: {name}
  mode: {mode}
  study_version: v2
  artifact_root: artifacts/out
  code_version: source_inner_studies_v2
inputs:
  manifest_path: inputs/manifest.csv
  feature_cache_path: inputs/train.pt
run:
  heldout_centers: all
  expected_feature_dim: 2560
  training_seeds: [17, 42, 101]
  generation_seeds: [17, 42, 101]
  device: cpu
model:
  pca_dim: 128
  latent_dim: 32
  hidden_dim: 512
  num_hidden_layers: 2
  train_epochs: 100
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0001
  beta_final: 0.001
  kl_warmup_epochs: 25
  gradient_clip_norm: 5.0
generation:
  budget_policy: source_empirical_class_counts_from_y_fit
claim_boundary:
  allowed: {allowed}
  forbidden: recipe adoption, outer-target preservation, expert-bank input, generation evidence, routing, expert selection, NELBO compatibility, or downstream utility
  target_evaluation_data_used: false
  may_change_existing_consensus_locks: false
  may_feed_recipe_selection: false
  may_feed_deployable_selection: false
"""


def _prior_yaml() -> str:
    return _common_yaml(name=LEARNED_PRIOR_STUDY_NAME, mode=LEARNED_PRIOR_MODE) + """
objective:
  family: stochastic_isotropic_v1
  fixed_across_arms: true
prior:
  arms: [A, C-diag, E]
  standard_family: standard_normal
  ex_post_family: class_conditional_diagonal_total_moment
  learned_family: learned_class_conditional_diagonal_gaussian
  logvar_parameterization: bounded_tanh
  logvar_bound: 6.0
  initialization: exact_standard_normal
  optimizer_learning_rate_multiplier: 1.0
  optimizer_weight_decay: 0.0
  prior_gradient_clip_norm: 5.0
decisions:
  e_vs_a_min_mean_delta: 0.05
  e_vs_c_min_mean_delta: 0.01
  min_inner_wins: 6
  safety_max_bacc_regression: 0.01
  minimum_real_bacc: 0.55
"""


def _fisher_yaml() -> str:
    return _common_yaml(name=FISHER_SHRINKAGE_STUDY_NAME, mode=FISHER_SHRINKAGE_MODE) + """
prior:
  family: standard_normal
  fixed_across_alphas: true
objective:
  alphas: [0.0, 0.05, 0.10, 0.25]
  raw_fisher_fit_scope: shared_per_outer_inner
  alpha_zero_policy: literal_isotropic_metric_none
decisions:
  fisher_min_mean_delta: 0.01
  min_inner_wins: 6
  tie_margin: 0.01
  safety_max_bacc_regression: 0.01
  minimum_real_bacc: 0.55
"""
