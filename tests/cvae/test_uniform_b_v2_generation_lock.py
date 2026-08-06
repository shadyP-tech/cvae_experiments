from __future__ import annotations

import inspect
import json
import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from midogpp_thesis.cli import COMMANDS, main as root_cli_main
from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae import generation as generation_module
from midogpp_thesis.cvae.generation.cli import main as generation_cli_main
from midogpp_thesis.cvae.generation.config import load_generation_lock_config
from midogpp_thesis.cvae.generation.contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_BANK_INDEX_SHA256,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_CONTENT_HASH,
    EXPECTED_CONTENT_INDEX_SHA256,
    EXPECTED_CONTROL_LOCK_HASH,
    EXPECTED_CONTROL_LOCK_SHA256,
    GENERATION_SEEDS,
    SOURCE_BUDGET_PER_CLASS,
    SOURCE_STREAM_NAMESPACE,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    GenerationLock,
)
from midogpp_thesis.cvae.generation.generation import (
    derived_generation_seed,
    equal_union_replicate_plan,
    generate_source_block,
    source_generation_plan,
)
from midogpp_thesis.cvae.generation.validation import validate_generation_provenance
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.validation import (
    REQUIRED_FILES as PROMOTED_BANK_REQUIRED_FILES,
)
from midogpp_thesis.cvae.generation_samplers import standard_normal_sampler
from midogpp_thesis.cvae.protocol import ProtocolError


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/40_prior_and_generation/configs"
    / "uniform_b_v2_generation_lock_v1.yaml"
)


def _generation_lock_payload() -> dict[str, object]:
    expert_locks = [
        {
            "source_center": center,
            "training_seed": training_seed,
            "expert_lock_hash": stable_hash(
                {
                    "source_center": center,
                    "training_seed": training_seed,
                }
            ),
        }
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
        "claim_scope": CLAIM_SCOPE,
        "bank": {
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "expert_locks": expert_locks,
            "candidate_sources_by_target": {
                target: [center for center in CENTERS if center != target]
                for target in CENTERS
            },
        },
        "generation": {
            "training_seeds": list(TRAINING_SEEDS),
            "generation_seeds": list(GENERATION_SEEDS),
            "source_stream_namespace": SOURCE_STREAM_NAMESPACE,
            "max_source_block_per_class": TOTAL_PER_CLASS,
            "equal_union_source_budget_per_class": SOURCE_BUDGET_PER_CLASS,
            "total_per_class": TOTAL_PER_CLASS,
        },
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    return payload


def _generation_lock() -> GenerationLock:
    return GenerationLock(_generation_lock_payload())


def _provenance_fixture(tmp_path: Path) -> tuple[object, Path, Path, dict[str, object]]:
    bank_root = tmp_path / "bank"
    output_root = tmp_path / "output"
    output_root.joinpath("provenance").mkdir(parents=True)
    config = replace(
        load_generation_lock_config(CONFIG),
        bank_root=bank_root,
        artifact_root=output_root,
    )
    required = set(PROMOTED_BANK_REQUIRED_FILES) | {"reports/validation_report.json"}
    files = []
    for relative in sorted(required):
        member = bank_root / relative
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text(relative + "\n", encoding="utf-8")
        digest = hashlib.sha256(member.read_bytes()).hexdigest()
        files.append(
            {
                "path": relative,
                "resolved_path": str(member.resolve()),
                "exists": True,
                "expected": None,
                "size_bytes": member.stat().st_size,
                "computed": {"sha256": digest},
                "verification": "RECORDED_NO_EXPECTATION",
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": "midogpp.prior_and_generation.uniform_b_v2_generation_lock.v1",
        "stage": "40_prior_and_generation",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            {
                "artifact_id": "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
                "resolved_path": str(bank_root.resolve()),
                "stage": "30_expert_bank",
                "evidence_label": "ROUTING_AUTHORIZED_AFTER_VALIDATION",
                "claim_scope": "expert_bank_construction_only",
                "semantic_identities": {},
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {
                    "status": "HASHES_RECORDED_NO_EXPECTATIONS",
                    "default_recording_algorithm": "sha256",
                    "files": files,
                },
                "exists": True,
            }
        ],
    }
    provenance_path = output_root / "provenance/input_artifacts.json"
    provenance_path.write_text(json.dumps(manifest), encoding="utf-8")
    return config, output_root, provenance_path, manifest


def test_config_locks_upstream_identity_classifier_and_claim_firewall() -> None:
    config = load_generation_lock_config(CONFIG)

    assert config.expected_bank_lock_hash == EXPECTED_BANK_LOCK_HASH
    assert config.expected_control_lock_hash == EXPECTED_CONTROL_LOCK_HASH
    assert config.expected_bank_index_sha256 == EXPECTED_BANK_INDEX_SHA256
    assert config.expected_control_sha256 == EXPECTED_CONTROL_LOCK_SHA256
    assert config.expected_content_index_sha256 == EXPECTED_CONTENT_INDEX_SHA256
    assert config.expected_content_hash == EXPECTED_CONTENT_HASH
    assert config.centers == CENTERS
    assert config.training_seeds == TRAINING_SEEDS
    assert config.generation_seeds == GENERATION_SEEDS
    assert config.generation_contract["total_per_class"] == 1024
    assert config.generation_contract["source_budget_per_class"] == 128
    assert config.generation_contract["source_budgets_split_across_seeds"] is False
    assert config.generation_contract["target_expert_excluded"] is True
    assert config.generation_contract["target_conditioned_source_weighting"] is False
    assert config.generation_contract["no_expert_selection"] is True
    assert config.generation_contract["no_seed_selection"] is True
    assert config.classifier.to_payload() == {
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
    }
    assert config.claim_boundary["claim_scope"] == CLAIM_SCOPE
    assert config.claim_boundary["may_feed_deployable_selection"] is True
    assert config.claim_boundary["source_only_frozen_state"] is True
    for forbidden in (
        "target_data_used",
        "target_support_used",
        "target_labels_used",
        "target_evaluation_labels_used",
        "routing_evidence_computed",
        "routing_quality_claimed",
        "nelbo_computed",
        "expert_selection_performed",
        "source_weighting_learned",
        "classifier_fit_performed",
        "downstream_utility_computed",
        "stage20_bacc_reused_as_stage40_result",
        "eight_source_control_scored",
    ):
        assert config.claim_boundary[forbidden] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("classifier", "C", 1.0),
        ("claim_boundary", "target_support_used", True),
        ("generation_contract", "no_seed_selection", False),
    ),
)
def test_config_rejects_classifier_firewall_or_selection_drift(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][key] = value
    candidate = tmp_path / "generation_lock.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="drifted"):
        load_generation_lock_config(candidate)


def test_derived_generation_seed_is_stable_and_target_independent() -> None:
    kwargs = {
        "namespace": SOURCE_STREAM_NAMESPACE,
        "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expert_lock_hash": "expert-a",
        "generation_seed": 17,
        "class_label": 0,
    }
    baseline = derived_generation_seed(**kwargs)

    assert baseline == derived_generation_seed(**kwargs)
    assert "target_center" not in inspect.signature(derived_generation_seed).parameters
    variants = {
        baseline,
        derived_generation_seed(**{**kwargs, "class_label": 1}),
        derived_generation_seed(**{**kwargs, "expert_lock_hash": "expert-b"}),
        derived_generation_seed(**{**kwargs, "generation_seed": 42}),
    }
    assert len(variants) == 4


def test_source_generation_plan_covers_exact_three_by_three_per_source() -> None:
    rows = source_generation_plan(_generation_lock())

    assert len(rows) == 81
    assert len({row.stream_id for row in rows}) == 81
    for center in CENTERS:
        center_rows = [row for row in rows if row.source_center == center]
        assert len(center_rows) == 9
        assert {
            (row.training_seed, row.generation_seed) for row in center_rows
        } == set(product(TRAINING_SEEDS, GENERATION_SEEDS))
        assert all(row.max_samples_per_class == 1024 for row in center_rows)
        assert all(row.equal_union_prefix_per_class == 128 for row in center_rows)


def test_equal_union_plan_has_exact_target_exclusion_and_fixed_budget() -> None:
    rows = equal_union_replicate_plan(_generation_lock())

    assert len(rows) == 81
    assert len({row.replicate_id for row in rows}) == 81
    for target in CENTERS:
        target_rows = [row for row in rows if row.target_center == target]
        assert len(target_rows) == 9
        assert {
            (row.training_seed, row.generation_seed) for row in target_rows
        } == set(product(TRAINING_SEEDS, GENERATION_SEEDS))
        for row in target_rows:
            assert target not in row.candidate_source_centers
            assert set(row.candidate_source_centers) == set(CENTERS).difference({target})
            assert len(row.candidate_source_centers) == 8
            assert len(row.source_stream_ids) == 8
            assert row.source_budget_per_class == 128
            assert row.total_per_class == 1024
            assert row.source_budget_per_class * len(row.source_stream_ids) == row.total_per_class


def test_generation_lock_rejects_semantic_tampering() -> None:
    payload = _generation_lock().to_payload()
    payload["generation"]["total_per_class"] = 2048

    with pytest.raises(ProtocolError, match="hash drifted"):
        GenerationLock(payload)


def test_source_generation_is_prefix_invariant_and_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def eval(self) -> FakeModel:
            return self

        def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.cat([z] * 64, dim=1) + y[:, None].to(z.dtype) * 0.25

    class FakeFrame:
        def __init__(self, source_frame: object, *, device: str) -> None:
            del source_frame, device

        def inverse_transform(self, projected: torch.Tensor) -> torch.Tensor:
            return projected.repeat(1, 15)

    monkeypatch.setattr(generation_module.generation, "TorchOptimizedFrame", FakeFrame)
    key = source_generation_plan(_generation_lock())[0]
    expert = SimpleNamespace(
        source_center=key.source_center,
        training_seed=key.training_seed,
        expert_lock_hash=key.expert_lock_hash,
        model=FakeModel(),
        source_frame=object(),
        sampler=standard_normal_sampler(latent_dim=4),
    )

    short = generate_source_block(expert, key, per_class=4, device="cpu")
    long = generate_source_block(expert, key, per_class=9, device="cpu")
    repeated = generate_source_block(expert, key, per_class=4, device="cpu")

    np.testing.assert_array_equal(short.embeddings[:4], long.embeddings[:4])
    np.testing.assert_array_equal(short.embeddings[4:], long.embeddings[9:13])
    np.testing.assert_array_equal(short.labels, np.array([0] * 4 + [1] * 4))
    np.testing.assert_array_equal(short.embeddings, repeated.embeddings)
    np.testing.assert_array_equal(short.labels, repeated.labels)
    assert short.output_sha256 == repeated.output_sha256


def test_generation_cli_and_top_level_command_are_registered(capsys: pytest.CaptureFixture[str]) -> None:
    assert COMMANDS["cvae-generation"][0] == "midogpp_thesis.cvae.generation.cli:main"

    with pytest.raises(SystemExit) as direct:
        generation_cli_main(["--help"])
    assert direct.value.code == 0
    assert "uniform-b-v2-generation-lock" in capsys.readouterr().out

    with pytest.raises(SystemExit) as routed:
        root_cli_main(["cvae-generation", "--help"])
    assert routed.value.code == 0
    assert "uniform-b-v2-generation-lock" in capsys.readouterr().out


def test_generation_cli_rejects_noncanonical_output_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midogpp_thesis.cvae.generation import config as config_module
    from midogpp_thesis.cvae.generation import runner as runner_module
    from midogpp_thesis.cvae.generation import workspace_binding as binding_module

    canonical = tmp_path / "canonical"
    alternate = tmp_path / "alternate"
    fake_config = SimpleNamespace(artifact_root=canonical)
    monkeypatch.setattr(config_module, "load_generation_lock_config", lambda _: fake_config)
    monkeypatch.setattr(binding_module, "validate_production_workspace_binding", lambda _: None)
    monkeypatch.setattr(
        runner_module,
        "run_generation_lock",
        lambda *args, **kwargs: pytest.fail("runner accepted a noncanonical output"),
    )

    with pytest.raises(ProtocolError, match="canonical workspace path"):
        generation_cli_main(
            [
                "uniform-b-v2-generation-lock",
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(alternate),
            ]
        )


def test_generation_provenance_accepts_only_the_canonical_singleton_bank(
    tmp_path: Path,
) -> None:
    config, output_root, _, _ = _provenance_fixture(tmp_path)

    validate_generation_provenance(output_root, config=config)  # type: ignore[arg-type]


@pytest.mark.parametrize("tamper", ("wrong_input", "target_eval", "path_escape"))
def test_generation_provenance_rejects_identity_and_path_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    config, output_root, provenance_path, manifest = _provenance_fixture(tmp_path)
    row = manifest["input_artifacts"][0]  # type: ignore[index]
    if tamper == "wrong_input":
        row["artifact_id"] = "wrong_bank"  # type: ignore[index]
    elif tamper == "target_eval":
        manifest["selection_used_target_eval_artifacts"] = True
    else:
        row["resolved_path"] = str((tmp_path / "escaped").resolve())  # type: ignore[index]
    provenance_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProtocolError, match="provenance|bank"):
        validate_generation_provenance(output_root, config=config)  # type: ignore[arg-type]
