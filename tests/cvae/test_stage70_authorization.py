from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.cvae.frozen_policy_downstream.authorization import (
    ArtifactBinding,
    AuthorizationValidationInputs,
    CacheBinding,
    FinalAuthorizationConfig,
    PolicyBinding,
    ReservationConfig,
    load_final_authorization_config,
    load_reservation_config,
    read_final_authorization_token,
    run_final_prediction_authorization,
    run_target_evaluation_reservation,
    validate_final_prediction_authorization,
    validate_target_evaluation_reservation,
)
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.config import (
    CACHE_ARTIFACT_ID,
    CACHE_EXPERIMENT_ID,
    CANONICAL_CACHE_RELATIVE_ROOT,
)
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.contracts import (
    FINAL_DESCRIPTIVE_STATUS,
    FRESH_CONFIRMATORY_STATUS,
    RESERVATION_DESCRIPTIVE_STATUS,
    validate_consumption_ledger,
)
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.workspace_binding import (
    COMMON_INPUT_IDS,
    DATASET_CONTRACT_ARTIFACT_ID,
    FINAL_INPUT_IDS,
    RESERVATION_INPUT_IDS,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_SCORING_MANIFEST_ARTIFACT_ID,
)
from midogpp_thesis.cvae.frozen_policy_downstream.contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    CONTROL_ARM,
    METADATA_ARM,
    POLICY_ARMS,
    MaterializationAssignment,
    PolicyReplicate,
    UTILITY_ARM,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    EXPECTED_TEST_ROWS_BY_CENTER,
)


@dataclass(frozen=True)
class _FakeGenerationLock:
    generation_lock_hash: str = "d" * 16


def test_consumption_ledger_accepts_published_hyphenated_reuse_key() -> None:
    validate_consumption_ledger(
        {
            "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
            "split": "test",
            "row_count": 9928,
            "observed_centers": 9,
            "may_be_reused_as_fresh_representation_selection_evidence": False,
            "may_be_reused_for_descriptive_locked-model_scoring": True,
        }
    )


def test_consumption_ledger_rejects_conflicting_reuse_aliases() -> None:
    with pytest.raises(ProtocolError, match="conflicting descriptive-use aliases"):
        validate_consumption_ledger(
            {
                "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
                "split": "test",
                "row_count": 9928,
                "observed_centers": 9,
                "may_be_reused_as_fresh_representation_selection_evidence": False,
                "may_be_reused_for_descriptive_locked_model_scoring": False,
                "may_be_reused_for_descriptive_locked-model_scoring": True,
            }
        )


def test_reservation_hashes_manifest_without_persisting_outcomes_or_paths(
    tmp_path: Path,
) -> None:
    config, projected, inputs = _reservation_fixture(tmp_path)

    root = run_target_evaluation_reservation(
        config,
        validation_inputs=inputs,
        projector=_projector(projected),
    )

    decision = _json(root / "reports/authorization_decision.json")
    assert decision["fresh_confirmatory_status"] == FRESH_CONFIRMATORY_STATUS
    assert decision["descriptive_status"] == RESERVATION_DESCRIPTIVE_STATUS
    assert decision["cache_extraction_allowed"] is True
    assert decision["prediction_allowed"] is False
    assert decision["metric_scoring_allowed"] is False
    identity_header = (root / "tables/target_identity.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert identity_header == (
        "evaluation_row_id,contract_row_index,target_center,split"
    )
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "SECRET_OUTCOME_VALUE" not in artifact_text
    assert "SECRET_SAMPLE_ID" not in artifact_text
    assert "/secret/image/path" not in artifact_text
    assert not root.with_name(f".{root.name}.staging").exists()

    checks = validate_target_evaluation_reservation(
        root,
        config=config,
        validation_inputs=inputs,
        projected_reservation=projected,
    )
    assert checks["status"] == "PASS"
    assert checks["row_count"] == 9928
    assert checks["target_labels_opened"] is False


def test_reservation_validator_rejects_tamper_and_unknown_members(
    tmp_path: Path,
) -> None:
    config, projected, inputs = _reservation_fixture(tmp_path)
    root = run_target_evaluation_reservation(
        config,
        validation_inputs=inputs,
        projector=_projector(projected),
    )
    decision_path = root / "reports/authorization_decision.json"
    decision = _json(decision_path)
    decision["descriptive_status"] = "FRESH_CONFIRMATORY_EVIDENCE"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(ProtocolError, match="decision drifted"):
        validate_target_evaluation_reservation(
            root,
            config=config,
            validation_inputs=inputs,
            projected_reservation=projected,
        )

    other_config, other_projected, other_inputs = _reservation_fixture(
        tmp_path / "other"
    )
    other_root = run_target_evaluation_reservation(
        other_config,
        validation_inputs=other_inputs,
        projector=_projector(other_projected),
    )
    (other_root / "predictions.csv").write_text("forbidden\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="closed-world coverage"):
        validate_target_evaluation_reservation(
            other_root,
            config=other_config,
            validation_inputs=other_inputs,
            projected_reservation=other_projected,
        )


def test_final_authorization_freezes_exact_factorial_and_is_prediction_only(
    tmp_path: Path,
) -> None:
    reservation_config, projected, inputs = _reservation_fixture(tmp_path)
    reservation_root = run_target_evaluation_reservation(
        reservation_config,
        validation_inputs=inputs,
        projector=_projector(projected),
    )
    cache = _cache_binding(projected, reservation_config)
    config = _final_config(
        tmp_path,
        reservation_root=reservation_root,
        reservation_config=reservation_config,
    )

    root = run_final_prediction_authorization(
        config,
        validation_inputs=inputs,
        projected_reservation=projected,
        cache_binding=cache,
    )

    plan = _json(root / "manifests/evaluation_plan.json")
    decision = _json(root / "reports/authorization_decision.json")
    token = read_final_authorization_token(root)
    assert len(plan["records"]) == 9 * 3 * 3 * 3 == 243
    assert {
        (row["policy_id"], row["target_center"], row["training_seed"], row["generation_seed"])
        for row in plan["records"]
    } == {
        (policy, center, training_seed, generation_seed)
        for policy in POLICY_ARMS
        for center in EXPECTED_TEST_ROWS_BY_CENTER
        for training_seed in (17, 42, 101)
        for generation_seed in (17, 42, 101)
    }
    assert all(row["synthetic_rows_per_class"] == 1024 for row in plan["records"])
    assert all(row["target_expert_excluded"] is True for row in plan["records"])
    assert all(
        assignment["source_center"] != row["target_center"]
        for row in plan["records"]
        for assignment in row["assignments"]
    )
    assert decision["fresh_confirmatory_status"] == FRESH_CONFIRMATORY_STATUS
    assert decision["descriptive_status"] == FINAL_DESCRIPTIVE_STATUS
    assert decision["prediction_allowed"] is True
    assert decision["label_access_allowed"] is False
    assert decision["metric_scoring_allowed"] is False
    assert token.to_payload()["authorized_consumer_experiment_id"] == (
        AUTHORIZED_CONSUMER_EXPERIMENT_ID
    )

    checks = validate_final_prediction_authorization(
        root,
        config=config,
        validation_inputs=inputs,
        projected_reservation=projected,
        cache_binding=cache,
    )
    assert checks["status"] == "PASS"
    assert checks["evaluation_plan_rows"] == 243
    assert checks["prediction_performed"] is False


def test_final_authorization_rejects_cache_drift_and_token_phase_tamper(
    tmp_path: Path,
) -> None:
    reservation_config, projected, inputs = _reservation_fixture(tmp_path)
    reservation_root = run_target_evaluation_reservation(
        reservation_config,
        validation_inputs=inputs,
        projector=_projector(projected),
    )
    config = _final_config(
        tmp_path,
        reservation_root=reservation_root,
        reservation_config=reservation_config,
    )
    bad_cache = _cache_binding(projected, reservation_config, reservation_id="wrong")
    with pytest.raises(ProtocolError, match="cache/reservation binding"):
        run_final_prediction_authorization(
            config,
            validation_inputs=inputs,
            projected_reservation=projected,
            cache_binding=bad_cache,
        )
    assert not config.artifact_root.exists()

    cache = _cache_binding(projected, reservation_config)
    root = run_final_prediction_authorization(
        config,
        validation_inputs=inputs,
        projected_reservation=projected,
        cache_binding=cache,
    )
    token_path = root / "manifests/authorization_token.json"
    token_payload = _json(token_path)
    token_payload["phase"] = "METRIC_SCORING_AUTHORIZATION"
    token_path.write_text(json.dumps(token_payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="token hash drifted"):
        validate_final_prediction_authorization(
            root,
            config=config,
            validation_inputs=inputs,
            projected_reservation=projected,
            cache_binding=cache,
        )


def test_loaded_configs_fail_closed_on_test_injection(tmp_path: Path) -> None:
    reservation, _projected, _inputs = _reservation_fixture(tmp_path)
    production_reservation = replace(
        reservation,
        prospective_cache_root=(Path.cwd() / CANONICAL_CACHE_RELATIVE_ROOT).resolve(),
        production_workspace_binding=True,
        allow_test_validation_injection=False,
    )
    reservation_path = tmp_path / "reservation.yaml"
    reservation_path.write_text(
        yaml.safe_dump(production_reservation.to_payload(), sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_reservation_config(reservation_path)
    assert loaded.contract_hash == production_reservation.contract_hash
    assert loaded.prospective_cache_root == (
        Path.cwd() / CANONICAL_CACHE_RELATIVE_ROOT
    ).resolve()

    final = replace(
        _final_config(
            tmp_path,
            reservation_root=reservation.artifact_root,
            reservation_config=reservation,
        ),
        production_workspace_binding=True,
        allow_test_validation_injection=False,
    )
    final_path = tmp_path / "final.yaml"
    final_path.write_text(yaml.safe_dump(final.to_payload(), sort_keys=False), encoding="utf-8")
    loaded_final = load_final_authorization_config(final_path)
    assert loaded_final.contract_hash == final.contract_hash

    payload = production_reservation.to_payload()
    payload["execution"]["production_workspace_binding"] = False
    payload["execution"]["allow_test_validation_injection"] = True
    reservation_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="must require production workspace"):
        load_reservation_config(reservation_path)


def test_workspace_prepared_roots_preserve_config_and_provenance_bytes(
    tmp_path: Path,
) -> None:
    reservation, projected, inputs = _reservation_fixture(tmp_path)
    reservation_config_bytes, reservation_provenance_bytes = _prepare_workspace_root(
        reservation
    )
    reservation_root = run_target_evaluation_reservation(
        reservation,
        validation_inputs=inputs,
        projector=_projector(projected),
    )
    assert (reservation_root / "config.resolved.yaml").read_bytes() == (
        reservation_config_bytes
    )
    assert (reservation_root / "provenance/input_artifacts.json").read_bytes() == (
        reservation_provenance_bytes
    )
    assert (reservation_root / "manifests/input_binding.json").is_file()

    final = _final_config(
        tmp_path,
        reservation_root=reservation_root,
        reservation_config=reservation,
    )
    final_config_bytes, final_provenance_bytes = _prepare_workspace_root(final)
    cache = _cache_binding(projected, reservation)
    final_root = run_final_prediction_authorization(
        final,
        validation_inputs=inputs,
        projected_reservation=projected,
        cache_binding=cache,
    )
    assert (final_root / "config.resolved.yaml").read_bytes() == final_config_bytes
    assert (final_root / "provenance/input_artifacts.json").read_bytes() == (
        final_provenance_bytes
    )
    assert (final_root / "manifests/input_binding.json").is_file()
    assert validate_final_prediction_authorization(
        final_root,
        config=final,
        validation_inputs=inputs,
        projected_reservation=projected,
        cache_binding=cache,
    )["status"] == "PASS"


def _reservation_fixture(
    root: Path,
) -> tuple[ReservationConfig, object, AuthorizationValidationInputs]:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "scoring_manifest.csv"
    manifest.write_bytes(
        b"\xffSECRET_OUTCOME_VALUE,SECRET_SAMPLE_ID,/secret/image/path\n"
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cache_root = root / "cache"
    config = ReservationConfig(
        artifact_root=root / "reservation",
        canonical_reference_root=root / "canonical_reference",
        bank_root=root / "bank",
        generation_lock_root=root / "generation",
        equal_union_policy_root=root / "equal_policy",
        metadata_policy_root=root / "metadata_policy",
        utility_policy_root=root / "utility_policy",
        scoring_manifest_path=manifest,
        test_consumption_ledger_path=(
            root / "canonical_reference/reports/test_consumption_ledger.json"
        ),
        prospective_cache_root=cache_root,
        expected_scoring_manifest_sha256=manifest_sha,
        expected_cache_extractor_protocol_hash="e" * 64,
        production_workspace_binding=False,
        allow_test_validation_injection=True,
    )
    projected = _projected_reservation(manifest_sha)
    return config, projected, _validation_inputs()


def _final_config(
    root: Path,
    *,
    reservation_root: Path,
    reservation_config: ReservationConfig,
) -> FinalAuthorizationConfig:
    return FinalAuthorizationConfig(
        artifact_root=root / "final_authorization",
        reservation_root=reservation_root,
        cache_root=reservation_config.prospective_cache_root,
        canonical_reference_root=reservation_config.canonical_reference_root,
        bank_root=reservation_config.bank_root,
        generation_lock_root=reservation_config.generation_lock_root,
        equal_union_policy_root=reservation_config.equal_union_policy_root,
        metadata_policy_root=reservation_config.metadata_policy_root,
        utility_policy_root=reservation_config.utility_policy_root,
        scoring_manifest_path=reservation_config.scoring_manifest_path,
        expected_scoring_manifest_sha256=(
            reservation_config.expected_scoring_manifest_sha256
        ),
        expected_cache_extractor_protocol_hash=(
            reservation_config.expected_cache_extractor_protocol_hash
        ),
        production_workspace_binding=False,
        allow_test_validation_injection=True,
    )


def _projected_reservation(manifest_sha: str) -> object:
    rows = []
    index = 0
    for center, count in EXPECTED_TEST_ROWS_BY_CENTER.items():
        for _ in range(count):
            rows.append(
                SimpleNamespace(
                    evaluation_row_id=f"eval_{index:08d}",
                    contract_row_index=index,
                    case_id=f"case-{index // 2}",
                    center=center,
                    split="test",
                )
            )
            index += 1
    return SimpleNamespace(
        manifest_sha256=manifest_sha,
        reservation_id="reservation_" + "1" * 64,
        protocol_hash="2" * 64,
        rows=tuple(rows),
        rows_by_center=dict(EXPECTED_TEST_ROWS_BY_CENTER),
    )


def _projector(projected: object):
    def project(path: Path, *, expected_manifest_sha256: str) -> object:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_manifest_sha256
        return projected

    return project


def _validation_inputs() -> AuthorizationValidationInputs:
    lock_hashes = {
        CONTROL_ARM: "1" * 16,
        METADATA_ARM: "2" * 16,
        UTILITY_ARM: "3" * 16,
    }
    plan_hashes = {
        CONTROL_ARM: "4" * 16,
        METADATA_ARM: "5" * 16,
        UTILITY_ARM: "6" * 16,
    }
    assignment_hashes = {
        CONTROL_ARM: "7" * 16,
        METADATA_ARM: "8" * 16,
        UTILITY_ARM: "9" * 16,
    }
    artifact_ids = {
        CONTROL_ARM: "test_equal_policy",
        METADATA_ARM: "test_metadata_policy",
        UTILITY_ARM: "test_utility_policy",
    }
    bindings = tuple(
        PolicyBinding(
            policy_id=policy,
            policy_artifact_id=artifact_ids[policy],
            policy_lock_hash=lock_hashes[policy],
            policy_plan_hash=plan_hashes[policy],
            assignment_table_hash=assignment_hashes[policy],
            assignment_table_sha256=(str(index + 1) * 64),
            assignment_count=81,
        )
        for index, policy in enumerate(POLICY_ARMS)
    )
    replicates = []
    centers = tuple(EXPECTED_TEST_ROWS_BY_CENTER)
    for policy in POLICY_ARMS:
        for target in centers:
            source = next(center for center in centers if center != target)
            for training_seed in (17, 42, 101):
                for generation_seed in (17, 42, 101):
                    assignment = MaterializationAssignment(
                        assignment_id=(
                            f"assignment-{policy}-{target}-{training_seed}-{generation_seed}"
                        ),
                        policy_id=policy,
                        target_center=target,
                        training_seed=training_seed,
                        generation_seed=generation_seed,
                        source_center=source,
                        source_stream_id=(
                            f"stream-{source}-{training_seed}-{generation_seed}"
                        ),
                        source_ordinal=0,
                        source_budget_per_class=1024,
                        prior_method="test_frozen_prior",
                        selection_source="test_frozen_policy",
                    )
                    replicates.append(
                        PolicyReplicate(
                            policy_id=policy,
                            policy_lock_hash=lock_hashes[policy],
                            policy_plan_hash=plan_hashes[policy],
                            assignment_table_hash=assignment_hashes[policy],
                            replicate_id=(
                                f"replicate-{target}-{training_seed}-{generation_seed}"
                            ),
                            target_center=target,
                            training_seed=training_seed,
                            generation_seed=generation_seed,
                            assignments=(assignment,),
                            class_shuffle_seed_by_label={"0": 11, "1": 13},
                        )
                    )
    artifact = lambda artifact_id, marker: ArtifactBinding(
        artifact_id=artifact_id,
        content_index_sha256=marker * 64,
        semantic_hashes={"lock_hash": marker * 16},
        validator=f"validate_{artifact_id}",
    )
    return AuthorizationValidationInputs(
        consumption_ledger={
            "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
            "split": "test",
            "row_count": 9928,
            "observed_centers": 9,
            "may_be_reused_as_fresh_representation_selection_evidence": False,
            "may_be_reused_for_descriptive_locked_model_scoring": True,
        },
        canonical_reference=artifact("test_reference", "a"),
        bank=artifact("test_bank", "b"),
        generation=artifact("test_generation", "c"),
        policies=bindings,
        generation_lock=_FakeGenerationLock(),
        policy_replicates=tuple(replicates),
        classifier_spec={
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
            "config_hash": "d" * 16,
            "scaler_family": "sklearn.preprocessing.StandardScaler",
            "fit_in_stage_40": False,
        },
    )


def _cache_binding(
    projected: object,
    config: ReservationConfig,
    *,
    reservation_id: str | None = None,
) -> CacheBinding:
    return CacheBinding(
        artifact_id=CACHE_ARTIFACT_ID,
        manifest_sha256=config.expected_scoring_manifest_sha256,
        target_evaluation_reservation_id=(
            reservation_id or projected.reservation_id
        ),
        target_evaluation_reservation_protocol_hash=projected.protocol_hash,
        cache_extractor_protocol_hash=(
            config.expected_cache_extractor_protocol_hash
        ),
        row_count=9928,
        rows_by_center=dict(EXPECTED_TEST_ROWS_BY_CENTER),
        row_order_hash="a" * 64,
        shard_sha256_by_center={
            center: f"{index:x}" * 64
            for index, center in enumerate(EXPECTED_TEST_ROWS_BY_CENTER, start=1)
        },
        content_hash="f" * 64,
        purpose="descriptive_frozen_policy_comparison_on_previously_consumed_test",
        fresh_evidence=False,
    )


def _prepare_workspace_root(
    config: ReservationConfig | FinalAuthorizationConfig,
) -> tuple[bytes, bytes]:
    root = config.artifact_root
    (root / "provenance").mkdir(parents=True)
    for directory in ("manifests", "reports", "tables"):
        (root / directory).mkdir()
    config_bytes = yaml.safe_dump(config.to_payload(), sort_keys=False).encode("utf-8")
    (root / "config.resolved.yaml").write_bytes(config_bytes)

    common_roots = dict(
        zip(
            COMMON_INPUT_IDS,
            (
                config.canonical_reference_root,
                config.bank_root,
                config.generation_lock_root,
                config.equal_union_policy_root,
                config.metadata_policy_root,
                config.utility_policy_root,
            ),
            strict=True,
        )
    )
    if isinstance(config, ReservationConfig):
        expected_ids = RESERVATION_INPUT_IDS
        roots = {
            **common_roots,
            DATASET_CONTRACT_ARTIFACT_ID: config.scoring_manifest_path.parent,
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: (
                config.test_consumption_ledger_path.parent.parent
            ),
        }
    else:
        expected_ids = FINAL_INPUT_IDS
        roots = {
            **common_roots,
            TEST_SCORING_MANIFEST_ARTIFACT_ID: config.scoring_manifest_path.parent,
            config.reservation_artifact_id: config.reservation_root,
            config.cache_artifact_id: config.cache_root,
        }
    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": config.experiment_id,
        "stage": "70_frozen_policy_downstream",
        "claim_scope": config.claim_scope,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            {
                "artifact_id": artifact_id,
                "resolved_path": str(roots[artifact_id]),
                "stage": "test_upstream",
                "evidence_label": "TEST_VALIDATED_INPUT",
                "claim_scope": "test_protocol_fixture",
                "semantic_identities": {},
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {
                    "status": "NO_PROVENANCE_FILES_DECLARED",
                    "default_recording_algorithm": "sha256",
                    "files": [],
                },
                "exists": True,
            }
            for artifact_id in sorted(expected_ids)
        ],
        "repository_revision": "a" * 40,
        "repository_dirty": True,
        "repository_status_hash": "b" * 64,
    }
    provenance_bytes = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (root / "provenance/input_artifacts.json").write_bytes(provenance_bytes)
    return config_bytes, provenance_bytes


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
