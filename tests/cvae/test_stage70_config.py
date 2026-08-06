from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.frozen_policy_downstream.config import (
    load_frozen_policy_downstream_config,
)
from midogpp_thesis.cvae.protocol import ProtocolError


_INPUT_ARTIFACT_IDS = [
    "midogpp_output_uniform_b_v2_descriptive_test_final_authorization_v1",
    "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42",
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
    "midogpp_output_uniform_b_v2_generation_lock_v1",
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1",
    "midogpp_output_uniform_b_v2_metadata_tie_union_policy_lock_v1",
    "midogpp_output_uniform_b_v2_utility_regret_policy_lock_v1",
    "midogpp_frozen_policy_test_scoring_manifest_v1",
]


def _valid_payload(tmp_path: Path) -> dict[str, object]:
    artifacts = tmp_path / "workspace" / "artifacts"
    return {
        "experiment": {
            "name": "uniform_b_v2_descriptive_frozen_policy_comparison_v1",
            "artifact_root": str(artifacts / "stage70"),
            "claim_scope": (
                "descriptive_frozen_policy_comparison_on_previously_consumed_test"
            ),
        },
        "inputs": {
            "final_authorization_root": str(artifacts / "authorization"),
            "bank_root": str(artifacts / "bank"),
            "generation_lock_root": str(artifacts / "generation"),
            "equal_union_policy_root": str(artifacts / "equal"),
            "metadata_policy_root": str(artifacts / "metadata"),
            "utility_policy_root": str(artifacts / "utility"),
            "target_cache_root": str(artifacts / "target_cache"),
            "scoring_manifest_path": str(artifacts / "scoring" / "manifest.csv"),
            "artifact_ids": list(_INPUT_ARTIFACT_IDS),
        },
        "protocol": {
            "authorized_consumer_experiment_id": (
                "midogpp.frozen_policy_downstream."
                "uniform_b_v2_descriptive_frozen_policy_comparison.v1"
            ),
            "eligible_centers": ["0", "1", "2", "3", "5", "6", "7", "8", "9"],
            "training_seeds": [17, 42, 101],
            "generation_seeds": [17, 42, 101],
            "synthetic_samples_per_class": 1024,
            "evaluation_split": (
                "test_previously_consumed_for_representation_adoption"
            ),
            "predictions_persisted_before_labels_opened": True,
            "final_authorization_hash": "a" * 16,
            "dataset_contract_hash": "b" * 64,
            "target_cache_content_hash": "c" * 64,
            "target_row_order_hash": "d" * 64,
            "scoring_manifest_sha256": "e" * 64,
            "representation_id": "annotation_jpeg_fixed_center_b_v3",
            "backbone_identity_hash": "f" * 16,
            "device": "cuda",
        },
        "classifier": {
            "family": "sklearn_logistic_regression",
            "C": 0.01,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 3000,
            "class_weight": None,
            "random_state": 23,
            "l1_ratio": None,
            "threshold_policy": "predict",
            "scaler_fit": "synthetic_train_only",
        },
        "bootstrap": {"seed": 42, "valid_replicates": 2000, "max_attempts": 20000},
        "claim_boundary": {
            "descriptive_comparison_only": True,
            "previously_consumed_test": True,
            "fresh_confirmatory_evidence": False,
            "routing_policy_promotion_allowed": False,
            "deployment_claim_allowed": False,
        },
    }


def _write(tmp_path: Path, payload: object, *, name: str = "config.resolved.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _section(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload[name]
    assert isinstance(value, dict)
    return value


def test_loads_closed_world_workspace_resolved_config_and_verifies_contract_hash(
    tmp_path: Path,
) -> None:
    payload = _valid_payload(tmp_path)
    expected_hash = stable_hash(payload)
    _section(payload, "protocol")["config_contract_hash"] = expected_hash

    config = load_frozen_policy_downstream_config(_write(tmp_path, payload))

    assert config.source_path == (tmp_path / "config.resolved.yaml").resolve()
    assert config.artifact_root == tmp_path / "workspace" / "artifacts" / "stage70"
    assert config.final_authorization_hash == "a" * 16
    assert config.dataset_contract_hash == "b" * 64
    assert config.backbone_identity_hash == "f" * 16
    assert config.device == "cuda"
    assert config.bootstrap_valid_replicates == 2000
    assert config.bootstrap_max_attempts == 20000
    assert config.contract_hash == expected_hash


def test_contract_hash_is_computed_from_workspace_resolved_payload_when_omitted(
    tmp_path: Path,
) -> None:
    payload = _valid_payload(tmp_path)

    config = load_frozen_policy_downstream_config(_write(tmp_path, payload))

    assert config.contract_hash == stable_hash(payload)


@pytest.mark.parametrize(
    ("section_name", "missing_key"),
    [
        ("top-level", "bootstrap"),
        ("experiment", "artifact_root"),
        ("inputs", "bank_root"),
        ("protocol", "dataset_contract_hash"),
        ("classifier", "max_iter"),
        ("bootstrap", "valid_replicates"),
        ("claim_boundary", "descriptive_comparison_only"),
    ],
)
@pytest.mark.parametrize("drift", ["missing", "extra"])
def test_rejects_missing_or_extra_keys_in_every_mapping(
    tmp_path: Path,
    section_name: str,
    missing_key: str,
    drift: str,
) -> None:
    payload = _valid_payload(tmp_path)
    target = payload if section_name == "top-level" else _section(payload, section_name)
    if drift == "missing":
        target.pop(missing_key)
    else:
        target["unexpected"] = "closed-world violation"

    with pytest.raises(ProtocolError, match="keys drifted"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


def test_hash_bound_registry_config_loads_after_authorization_artifacts_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    planned = (
        repo_root
        / "experiments/midogpp/stages/70_frozen_policy_downstream/configs"
        / "uniform_b_v2_descriptive_frozen_policy_comparison_v1.yaml"
    )

    config = load_frozen_policy_downstream_config(planned)

    assert config.final_authorization_hash == "a344cd66fc88daae"
    assert config.target_cache_content_hash == (
        "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
    )
    assert config.target_row_order_hash == (
        "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
    )


@pytest.mark.parametrize(
    ("field", "valid_length"),
    [
        ("final_authorization_hash", 16),
        ("dataset_contract_hash", 64),
        ("target_cache_content_hash", 64),
        ("target_row_order_hash", 64),
        ("scoring_manifest_sha256", 64),
        ("backbone_identity_hash", 16),
    ],
)
@pytest.mark.parametrize("malformation", ["uppercase", "wrong_length"])
def test_rejects_malformed_hash_identity(
    tmp_path: Path,
    field: str,
    valid_length: int,
    malformation: str,
) -> None:
    payload = _valid_payload(tmp_path)
    protocol = _section(payload, "protocol")
    protocol[field] = (
        "A" * valid_length if malformation == "uppercase" else "a" * (valid_length - 1)
    )

    with pytest.raises(ProtocolError, match="lowercase hexadecimal digest"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


def test_rejects_mismatched_or_malformed_config_contract_hash(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    _section(payload, "protocol")["config_contract_hash"] = "0" * 16
    with pytest.raises(ProtocolError, match="contract hash drifted"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))

    payload = _valid_payload(tmp_path)
    _section(payload, "protocol")["config_contract_hash"] = "A" * 16
    with pytest.raises(ProtocolError, match="lowercase hexadecimal digest"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("representation_id", "", "representation id"),
        ("device", "mps", "cpu.*cuda"),
    ],
)
def test_rejects_invalid_representation_or_device(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    payload = _valid_payload(tmp_path)
    _section(payload, "protocol")[field] = value

    with pytest.raises(ProtocolError, match=message):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


@pytest.mark.parametrize("field", ["descriptive_comparison_only", "previously_consumed_test"])
def test_rejects_drift_in_consumed_descriptive_claim_booleans(
    tmp_path: Path,
    field: str,
) -> None:
    payload = _valid_payload(tmp_path)
    _section(payload, "claim_boundary")[field] = False

    with pytest.raises(ProtocolError, match="claim boundary drifted"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("section_name", "field", "value", "message"),
    [
        ("classifier", "C", 0, "classifier C must be positive"),
        ("classifier", "max_iter", 0, "classifier max_iter must be positive"),
        ("bootstrap", "valid_replicates", 0, "valid_replicates must be positive"),
        ("bootstrap", "max_attempts", 0, "max_attempts must be positive"),
        ("bootstrap", "seed", -1, "seed must be non-negative"),
    ],
)
def test_rejects_nonpositive_classifier_or_bootstrap_limits(
    tmp_path: Path,
    section_name: str,
    field: str,
    value: int,
    message: str,
) -> None:
    payload = _valid_payload(tmp_path)
    _section(payload, section_name)[field] = value

    with pytest.raises(ProtocolError, match=message):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


def test_rejects_bootstrap_attempt_budget_below_valid_replicates(
    tmp_path: Path,
) -> None:
    payload = _valid_payload(tmp_path)
    _section(payload, "bootstrap")["max_attempts"] = 1999

    with pytest.raises(ProtocolError, match="cover all valid_replicates"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))


def test_rejects_input_artifact_identity_drift(tmp_path: Path) -> None:
    payload = _valid_payload(tmp_path)
    inputs = _section(payload, "inputs")
    artifact_ids = deepcopy(_INPUT_ARTIFACT_IDS)
    artifact_ids[-1] = "unexpected_scoring_manifest"
    inputs["artifact_ids"] = artifact_ids

    with pytest.raises(ProtocolError, match="artifact identities drifted"):
        load_frozen_policy_downstream_config(_write(tmp_path, payload))
