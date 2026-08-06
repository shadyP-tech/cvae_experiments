from __future__ import annotations

from collections.abc import Callable
import hashlib
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cli import COMMANDS, main as root_cli_main
from midogpp_thesis.cvae.generation.contracts import (
    SOURCE_STREAM_NAMESPACE,
    GenerationLock,
)
from midogpp_thesis.cvae.generation.validation import (
    REQUIRED_FILES as GENERATION_REQUIRED_FILES,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.validation import (
    REQUIRED_FILES as BANK_REQUIRED_FILES,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.cli import main as routing_cli_main
from midogpp_thesis.cvae.routing.config import (
    UniformBV2EqualUnionPolicyConfig,
    load_equal_union_policy_config,
)
from midogpp_thesis.cvae.routing.contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_REPLICATE_COUNT,
    GENERATION_SEEDS,
    POLICY_FAMILY,
    SOURCE_BUDGET_PER_CLASS,
    SOURCES_PER_TARGET,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    EqualUnionPolicyLock,
)
from midogpp_thesis.cvae.routing.policy import assignment_rows, build_policy_plan
from midogpp_thesis.cvae.routing.bundle import REQUIRED_FILES as POLICY_REQUIRED_FILES
from midogpp_thesis.cvae.routing.runner import run_equal_union_policy_lock
from midogpp_thesis.cvae.routing.validation import (
    validate_equal_union_policy_bundle,
    validate_policy_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_equal_union_policy_lock_v1.yaml"
)


def _generation_lock_payload() -> dict[str, object]:
    expert_locks = [
        {
            "source_center": center,
            "training_seed": training_seed,
            "expert_lock_hash": stable_hash(
                {"source_center": center, "training_seed": training_seed}
            ),
        }
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
        "claim_scope": "generation_settings_and_frame_lock",
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


def _provenance_fixture(
    tmp_path: Path,
) -> tuple[
    UniformBV2EqualUnionPolicyConfig,
    Path,
    Path,
    dict[str, object],
    dict[str, Path],
]:
    bank_root = tmp_path / "bank"
    generation_root = tmp_path / "generation"
    output_root = tmp_path / "output"
    output_root.joinpath("provenance").mkdir(parents=True)
    resolved_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved_payload["experiment"]["artifact_root"] = str(output_root)
    resolved_payload["inputs"]["bank_root"] = str(bank_root)
    resolved_payload["inputs"]["generation_lock_root"] = str(generation_root)
    resolved_config = output_root / "config.resolved.yaml"
    resolved_config.write_text(
        yaml.safe_dump(resolved_payload, sort_keys=False),
        encoding="utf-8",
    )
    config = load_equal_union_policy_config(resolved_config)
    specs = (
        (
            "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1",
            bank_root,
            "30_expert_bank",
            "expert_bank_construction_only",
            "ROUTING_AUTHORIZED_AFTER_VALIDATION",
            set(BANK_REQUIRED_FILES) | {"reports/validation_report.json"},
            {},
        ),
        (
            "midogpp_output_uniform_b_v2_generation_lock_v1",
            generation_root,
            "40_prior_and_generation",
            "generation_settings_and_frame_lock",
            "GENERATION_SETTINGS_LOCKED_AFTER_VALIDATION",
            set(GENERATION_REQUIRED_FILES) | {"reports/validation_report.json"},
            {
                "generation_lock_contract": "midogpp_uniform_b_v2_generation_lock_v1",
                "generation_lock_hash": "34e551425710362e",
                "expert_bank_lock_hash": "9972a41dcd4814cd",
                "equal_union_control_lock_hash": "cddbcc3b3343fe38",
            },
        ),
    )
    input_rows: list[dict[str, object]] = []
    members: dict[str, Path] = {}
    for (
        artifact_id,
        artifact_root,
        stage,
        scope,
        evidence,
        required,
        semantic_identities,
    ) in specs:
        files = []
        for relative in sorted(required):
            member = artifact_root / relative
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_text(f"{artifact_id}:{relative}\n", encoding="utf-8")
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
            members[f"{artifact_id}:{relative}"] = member
        input_rows.append(
            {
                "artifact_id": artifact_id,
                "resolved_path": str(artifact_root.resolve()),
                "stage": stage,
                "claim_scope": scope,
                "evidence_label": evidence,
                "semantic_identities": semantic_identities,
                "semantic_identities_are_file_hashes": False,
                "exists": True,
                "file_integrity": {
                    "status": "HASHES_RECORDED_NO_EXPECTATIONS",
                    "default_recording_algorithm": "sha256",
                    "files": files,
                },
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": (
            "midogpp.routing_and_composition."
            "uniform_b_v2_equal_union_policy_lock.v1"
        ),
        "stage": "60_routing_and_composition",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": input_rows,
        "repository_revision": "0" * 40,
        "repository_dirty": True,
        "repository_status_hash": "1" * 64,
    }
    path = output_root / "provenance/input_artifacts.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return config, output_root, path, manifest, members


def test_policy_config_locks_the_complete_nonadaptive_control() -> None:
    config = load_equal_union_policy_config(CONFIG)
    policy = config.policy_contract

    assert config.centers == CENTERS
    assert config.training_seeds == TRAINING_SEEDS
    assert config.generation_seeds == GENERATION_SEEDS
    assert config.expected_bank_lock_hash == EXPECTED_BANK_LOCK_HASH
    assert config.expected_generation_lock_hash == EXPECTED_GENERATION_LOCK_HASH
    assert policy["family"] == POLICY_FAMILY
    assert policy["seed_pairing"] == "cartesian_product"
    assert policy["sources_per_target"] == SOURCES_PER_TARGET == 8
    assert policy["source_budget_per_class"] == SOURCE_BUDGET_PER_CLASS == 128
    assert policy["total_per_class"] == TOTAL_PER_CLASS == 1024
    assert policy["expected_replicate_count"] == EXPECTED_REPLICATE_COUNT == 81
    assert policy["expected_assignments_per_replicate"] == 8
    assert policy["expected_assignment_count"] == EXPECTED_ASSIGNMENT_COUNT == 648
    assert policy["all_eligible_sources_retained"] is True
    assert policy["target_expert_excluded"] is True
    assert policy["no_expert_selection"] is True
    assert policy["no_seed_selection"] is True
    assert policy["no_source_ranking"] is True
    assert policy["no_learned_source_weighting"] is True
    assert policy["target_identity_role"] == (
        "fold_identity_candidate_exclusion_and_label_blind_shuffle_seeding_only"
    )
    candidates = policy["candidate_sources_by_target"]
    for target in CENTERS:
        assert len(candidates[target]) == 8
        assert target not in candidates[target]
        assert set(candidates[target]) == set(CENTERS).difference({target})


def test_policy_config_accepts_workspace_resolved_canonical_paths(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    artifact_root = tmp_path / "artifacts/midogpp/60_routing_and_composition/control/v1"
    bank_root = tmp_path / "artifacts/midogpp/30_expert_bank/bank"
    generation_root = tmp_path / "artifacts/midogpp/40_prior_and_generation/lock/v1"
    payload["experiment"]["artifact_root"] = str(artifact_root)
    payload["inputs"]["bank_root"] = str(bank_root)
    payload["inputs"]["generation_lock_root"] = str(generation_root)
    resolved = tmp_path / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    config = load_equal_union_policy_config(resolved)

    assert config.artifact_root == artifact_root
    assert config.bank_root == bank_root
    assert config.generation_lock_root == generation_root


def test_policy_config_freezes_exact_composition_and_future_evaluation_pairing() -> None:
    config = load_equal_union_policy_config(CONFIG)
    execution = config.composition_execution
    future = config.future_evaluation_contract

    assert execution["source_slice"] == "first_n_from_generation_lock_stream"
    assert execution["source_prefix_per_class"] == 128
    assert execution["source_concatenation_order"] == (
        "candidate_sources_by_target_order"
    )
    assert execution["class_composition_order"] == [0, 1]
    assert execution["shuffle_scope"] == "independently_within_class_after_union"
    assert execution["shuffle_algorithm"] == "numpy_generator_pcg64_permutation"
    assert execution["shuffle_seed_source"] == (
        "generation_lock_equal_union_replicate_plan"
    )
    assert execution["shuffle_seed_field"] == "class_shuffle_seed_by_label"
    assert execution["shuffle_applied_once_per_class"] is True
    assert execution["final_class_concatenation_order"] == [0, 1]
    assert future["authorization"] == (
        "separate_stage70_target_eval_artifact_required"
    )
    assert future["evaluation_stage"] == "70_frozen_policy_downstream"
    assert future["future_target_evaluation_rows"] == (
        "all_authorized_target_rows_paired_across_policies"
    )
    assert future["row_filtering"] == "none"
    assert future["row_subsampling"] == "none"
    assert future["labels_access"] == "metrics_only_after_predictions"
    assert future["support_rows_used"] is False
    assert future["evaluation_occurs_in_stage60"] is False
    assert future["target_identity_role"] == (
        "fold_membership_candidate_exclusion_and_label_blind_shuffle_seeding_only"
    )
    assert config.execution["target_dataset_access_allowed"] is False
    assert config.execution["support_set_access_allowed"] is False
    assert config.execution["classifier_fit_allowed"] is False
    assert config.execution["metric_computation_allowed"] is False
    for key in (
        "target_data_used",
        "target_support_used",
        "target_labels_used",
        "target_evaluation_labels_used",
        "target_metadata_used",
        "classifier_fit_performed",
        "bacc_computed",
        "macro_f1_computed",
        "downstream_utility_computed",
        "stage20_scores_reused",
    ):
        assert config.claim_boundary[key] is False
    assert config.claim_boundary[
        "target_identity_used_for_fold_candidate_exclusion_and_label_blind_shuffle_only"
    ] is True


def test_policy_plan_has_exact_81_replicates_and_648_fixed_assignments() -> None:
    config = load_equal_union_policy_config(CONFIG)
    lock = _generation_lock()
    replicates = build_policy_plan(lock, config)
    assignments = assignment_rows(lock, config)

    assert len(replicates) == 81
    assert len({row.replicate_id for row in replicates}) == 81
    assert len(assignments) == 648
    assert len({row.assignment_id for row in assignments}) == 648
    observed_replicates = {
        (row.target_center, row.training_seed, row.generation_seed)
        for row in replicates
    }
    assert observed_replicates == set(
        product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS)
    )
    for row in replicates:
        expected_sources = tuple(center for center in CENTERS if center != row.target_center)
        assert row.candidate_source_centers == expected_sources
        assert len(row.candidate_source_centers) == 8
        assert len(row.source_stream_ids) == 8
        assert len(row.assignment_ids) == 8
        assert row.target_center not in row.candidate_source_centers
        assert row.source_budget_per_class == 128
        assert row.total_per_class == 1024
        assert row.source_budget_per_class * len(row.assignment_ids) == row.total_per_class
        payload = row.to_payload()
        assert payload["all_eligible_sources_retained"] is True
        assert payload["expert_selection_performed"] is False
        assert payload["seed_selection_performed"] is False
        assert payload["source_ranking_performed"] is False
        assert payload["source_weighting_learned"] is False

    assignments_by_replicate = {
        row.replicate_id: [
            item for item in assignments if item.replicate_id == row.replicate_id
        ]
        for row in replicates
    }
    for replicate in replicates:
        rows = assignments_by_replicate[replicate.replicate_id]
        assert [row.source_center for row in rows] == list(
            replicate.candidate_source_centers
        )
        assert [row.source_ordinal for row in rows] == list(range(8))
        assert all(row.target_center != row.source_center for row in rows)
        assert all(row.source_budget_per_class == 128 for row in rows)
        for row in rows:
            payload = row.to_payload()
            assert payload["selection_rank"] is None
            assert payload["selection_score"] is None
            assert payload["learned_weight"] is None
            assert payload["target_expert"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("inputs", "target_data_root", "/forbidden/target"),
        ("inputs", "target_support_artifact_id", "forbidden_support"),
        ("inputs", "target_labels_path", "/forbidden/labels.csv"),
        ("inputs", "target_evaluation_artifact_id", "forbidden_evaluation"),
        ("inputs", "stage20_artifact_id", "forbidden_stage20"),
        ("classifier", "family", "sklearn_logistic_regression"),
        ("claim_boundary", "bacc_computed", True),
    ),
)
def test_config_rejects_forbidden_target_classifier_metric_or_stage20_fields(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if section not in payload:
        payload[section] = {}
    payload[section][key] = value
    candidate = tmp_path / "forbidden.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="drifted"):
        load_equal_union_policy_config(candidate)


@pytest.mark.parametrize(
    ("field", "mutator"),
    (
        (
            "candidate_sources_by_target",
            lambda payload: payload["policy_contract"][
                "candidate_sources_by_target"
            ]["0"].append("0"),
        ),
        (
            "source_budget_per_class",
            lambda payload: payload["policy_contract"].__setitem__(
                "source_budget_per_class", 127
            ),
        ),
        (
            "total_per_class",
            lambda payload: payload["policy_contract"].__setitem__(
                "total_per_class", 2048
            ),
        ),
        (
            "training_seeds",
            lambda payload: payload["policy_contract"].__setitem__(
                "training_seeds", [17, 42]
            ),
        ),
        (
            "generation_seeds",
            lambda payload: payload["policy_contract"].__setitem__(
                "generation_seeds", [17, 42]
            ),
        ),
    ),
)
def test_config_rejects_candidate_budget_and_seed_semantic_drift(
    tmp_path: Path,
    field: str,
    mutator: Callable[[dict[str, object]], object],
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutator(payload)
    candidate = tmp_path / f"drifted-{field}.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="drifted"):
        load_equal_union_policy_config(candidate)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("composition_execution", "source_slice", "last_n"),
        ("composition_execution", "source_prefix_per_class", 64),
        ("composition_execution", "shuffle_algorithm", "global_rng"),
        ("composition_execution", "final_class_concatenation_order", [1, 0]),
        ("future_evaluation_contract", "row_filtering", "policy_specific"),
        (
            "future_evaluation_contract",
            "future_target_evaluation_rows",
            "policy_specific_rows",
        ),
        ("future_evaluation_contract", "labels_access", "before_prediction"),
    ),
)
def test_config_rejects_composition_or_future_evaluation_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][field] = value
    candidate = tmp_path / f"drifted-{section}-{field}.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="drifted"):
        load_equal_union_policy_config(candidate)


def test_policy_lock_rejects_byte_level_payload_tampering() -> None:
    unhashed = {
        "schema_version": "midogpp_uniform_b_v2_equal_union_policy_lock_v1",
        "claim_scope": CLAIM_SCOPE,
        "upstreams": {
            "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        },
        "policy": {"family": POLICY_FAMILY},
    }
    payload = {**unhashed, "policy_lock_hash": stable_hash(unhashed)}
    lock = EqualUnionPolicyLock(payload)
    tampered = lock.to_payload()
    tampered["policy"]["family"] = "target_conditioned_weighting"  # type: ignore[index]

    with pytest.raises(ProtocolError, match="hash drifted"):
        EqualUnionPolicyLock(tampered)


@pytest.mark.parametrize("tamper", ("target_candidate", "budget", "training_seed"))
def test_policy_plan_rejects_rehashed_generation_semantic_drift(tamper: str) -> None:
    payload = _generation_lock_payload()
    if tamper == "target_candidate":
        payload["bank"]["candidate_sources_by_target"]["0"].append("0")  # type: ignore[index]
    elif tamper == "budget":
        payload["generation"]["total_per_class"] = 2048  # type: ignore[index]
    else:
        payload["generation"]["training_seeds"] = [17, 42]  # type: ignore[index]
    payload["generation_lock_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "generation_lock_hash"}
    )
    semantically_drifted = GenerationLock(payload)

    with pytest.raises(
        ProtocolError,
        match="81 replicates|coverage|pool or budget|target expert",
    ):
        build_policy_plan(semantically_drifted)


def test_policy_provenance_accepts_only_the_two_frozen_upstreams(tmp_path: Path) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)

    validate_policy_provenance(output_root, config=config)


def test_policy_provenance_rejects_upstream_byte_drift(tmp_path: Path) -> None:
    config, output_root, _, _, members = _provenance_fixture(tmp_path)
    member = members[
        "midogpp_output_uniform_b_v2_generation_lock_v1:"
        "manifests/generation_lock.json"
    ]
    member.write_text("drifted bytes\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="input member drifted"):
        validate_policy_provenance(output_root, config=config)


@pytest.mark.parametrize(
    "tamper",
    (
        "target_eval",
        "stage20_input",
        "extra_metric",
        "row_metric",
        "computed_metric",
        "missing_git",
    ),
)
def test_policy_provenance_rejects_forbidden_semantic_inputs(
    tmp_path: Path,
    tamper: str,
) -> None:
    config, output_root, path, manifest, _ = _provenance_fixture(tmp_path)
    if tamper == "target_eval":
        manifest["selection_used_target_eval_artifacts"] = True
    elif tamper == "stage20_input":
        manifest["input_artifacts"][0]["artifact_id"] = (  # type: ignore[index]
            "midogpp_output_cvae_uniform_b_geco_aggregate_prior_union_source_inner_v2"
        )
    elif tamper == "extra_metric":
        manifest["bacc"] = 0.99
    elif tamper == "row_metric":
        manifest["input_artifacts"][0]["target_utility"] = 0.99  # type: ignore[index]
    elif tamper == "computed_metric":
        manifest["input_artifacts"][0]["file_integrity"]["files"][0][  # type: ignore[index]
            "bacc"
        ] = 0.99
    else:
        manifest.pop("repository_status_hash")
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ProtocolError,
        match="provenance|only Stage 30 and Stage 40|fields drifted",
    ):
        validate_policy_provenance(output_root, config=config)


def _stub_validated_generation_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> GenerationLock:
    from midogpp_thesis.cvae.routing import runner as runner_module
    from midogpp_thesis.cvae.routing import validation as validation_module

    lock = _generation_lock()
    monkeypatch.setattr(runner_module, "load_validated_inputs", lambda _: lock)
    monkeypatch.setattr(validation_module, "load_validated_inputs", lambda _: lock)
    return lock


def test_runner_materializes_a_closed_world_bundle_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    lock = _stub_validated_generation_lock(monkeypatch)

    assert run_equal_union_policy_lock(config) == output_root
    assert run_equal_union_policy_lock(config) == output_root
    actual = {
        member.relative_to(output_root).as_posix()
        for member in output_root.rglob("*")
        if member.is_file()
    }
    assert actual == set(POLICY_REQUIRED_FILES)
    checks = validate_equal_union_policy_bundle(
        output_root,
        config=config,
        _validated_generation_lock=lock,
    )
    assert checks["status"] == "PASS"
    assert checks["target_replicate_count"] == 81
    assert checks["assignment_count"] == 648


def test_runner_rejects_a_stale_or_target_derived_extra_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    _stub_validated_generation_lock(monkeypatch)
    stale = output_root / "reports/stage70_bacc.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"bacc": 0.99}\n', encoding="utf-8")

    with pytest.raises(ProtocolError, match="unexpected files"):
        run_equal_union_policy_lock(config)


def test_runner_recovers_from_an_allowed_failed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    _stub_validated_generation_lock(monkeypatch)
    state = output_root / "reports/run_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_uniform_b_v2_equal_union_policy_run_state_v1",
                "status": "FAILED",
                "claim_scope": CLAIM_SCOPE,
            }
        ),
        encoding="utf-8",
    )

    run_equal_union_policy_lock(config)

    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "COMPLETE"


@pytest.mark.parametrize("tamper", ("validation_metric", "assignment_column"))
def test_bundle_validator_rejects_stage70_or_selection_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    lock = _stub_validated_generation_lock(monkeypatch)
    run_equal_union_policy_lock(config)
    if tamper == "validation_metric":
        path = output_root / "reports/validation_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["bacc"] = 0.99
        path.write_text(json.dumps(payload), encoding="utf-8")
        match = "validation report fields drifted"
    else:
        path = output_root / "tables/policy_assignments.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] += ",target_utility"
        lines[1:] = [f"{line},0.99" for line in lines[1:]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        match = "assignment table columns drifted"

    with pytest.raises(ProtocolError, match=match):
        validate_equal_union_policy_bundle(
            output_root,
            config=config,
            _validated_generation_lock=lock,
        )


def test_declared_seed_cartesian_product_contains_every_seed_without_selection() -> None:
    expected = set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))

    assert len(expected) == 81
    assert {training for _, training, _ in expected} == set(TRAINING_SEEDS)
    assert {generation for _, _, generation in expected} == set(GENERATION_SEEDS)


def test_routing_cli_and_top_level_command_are_registered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert COMMANDS["cvae-routing"][0] == "midogpp_thesis.cvae.routing.cli:main"

    with pytest.raises(SystemExit) as direct:
        routing_cli_main(["--help"])
    assert direct.value.code == 0
    assert "uniform-b-v2-equal-union-policy-lock" in capsys.readouterr().out

    with pytest.raises(SystemExit) as routed:
        root_cli_main(["cvae-routing", "--help"])
    assert routed.value.code == 0
    assert "uniform-b-v2-equal-union-policy-lock" in capsys.readouterr().out


def test_routing_cli_rejects_noncanonical_output_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midogpp_thesis.cvae.routing import config as config_module
    from midogpp_thesis.cvae.routing import runner as runner_module
    from midogpp_thesis.cvae.routing import workspace_binding as binding_module

    canonical = tmp_path / "canonical"
    alternate = tmp_path / "alternate"
    fake_config = SimpleNamespace(artifact_root=canonical)
    monkeypatch.setattr(
        config_module,
        "load_equal_union_policy_config",
        lambda _: fake_config,
    )
    monkeypatch.setattr(
        binding_module,
        "validate_production_workspace_binding",
        lambda _: None,
    )
    monkeypatch.setattr(
        runner_module,
        "run_equal_union_policy_lock",
        lambda *args, **kwargs: pytest.fail("runner accepted a noncanonical output"),
    )

    with pytest.raises(ProtocolError, match="canonical workspace path"):
        routing_cli_main(
            [
                "uniform-b-v2-equal-union-policy-lock",
                "--config",
                str(CONFIG),
                "--artifact-root",
                str(alternate),
            ]
        )
