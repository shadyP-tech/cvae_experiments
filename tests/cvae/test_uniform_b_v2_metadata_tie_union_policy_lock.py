from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.generation import equal_union_replicate_plan
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
from midogpp_thesis.cvae.routing.bundle import (
    REQUIRED_FILES as EQUAL_UNION_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.contracts import (
    EXPECTED_CONFIG_CONTRACT_HASH as EXPECTED_EQUAL_UNION_CONFIG_CONTRACT_HASH,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.bundle import (
    REQUIRED_FILES as COMPATIBILITY_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.contracts import (
    OUTPUT_SEMANTIC_IDENTITIES as COMPATIBILITY_SEMANTIC_IDENTITIES,
    CompatibilityScore,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.profiles import (
    derive_metadata_profiles,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.scoring import (
    derive_compatibility_scores,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.bundle import (
    REQUIRED_FILES as POLICY_REQUIRED_FILES,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.config import (
    UniformBV2MetadataTieUnionPolicyConfig,
    load_metadata_tie_union_policy_config,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.contracts import (
    CENTERS,
    CLAIM_SCOPE,
    COMPATIBILITY_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPECTED_ASSIGNMENT_COUNT,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_COMPATIBILITY_LOCK_HASH,
    EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH,
    EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_REPLICATE_COUNT,
    EXPECTED_SELECTION_COUNT,
    EXPERIMENT_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    MetadataTieUnionPolicyLock,
    SELECTED_SOURCES_BY_TARGET,
    SOURCE_BUDGET_BY_TIE_COUNT,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.inputs import (
    ValidatedTieUnionInputs,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.policy import (
    assignment_rows,
    build_policy_lock,
    build_policy_plan,
    build_policy_selections,
    read_policy_lock,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.runner import (
    run_metadata_tie_union_policy_lock,
)
from midogpp_thesis.cvae.routing.metadata_tie_union.validation import (
    validate_metadata_tie_union_policy_bundle,
    validate_policy_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_metadata_tie_union_policy_lock_v1.yaml"
)
DOMAIN_MAPPING = (
    ROOT / "datasets/midogpp/contract/annotation_patch_v1/domain_mapping.json"
)

CANONICAL_EXPERT_LOCK_HASHES = {
    ("0", 17): "ce2354bb48ddab52",
    ("0", 42): "20d9824d7dd6b64a",
    ("0", 101): "845da15bad36d8f4",
    ("1", 17): "b0ba12474280c063",
    ("1", 42): "e5226410dae5f2e8",
    ("1", 101): "390c2148a99f9ecf",
    ("2", 17): "e132ae0828905e7f",
    ("2", 42): "7529d6571fc566af",
    ("2", 101): "873daf87fada6fd3",
    ("3", 17): "9ec99dd63a1b7d08",
    ("3", 42): "c852d0c770d947e0",
    ("3", 101): "962bd7126b3392a2",
    ("5", 17): "6d644705b147968e",
    ("5", 42): "61e675e3b662ba44",
    ("5", 101): "57c4512335c66997",
    ("6", 17): "67546ee1fa793fea",
    ("6", 42): "8b32bc91c8362f43",
    ("6", 101): "fb723a52eaed8af2",
    ("7", 17): "f9843c7940b75231",
    ("7", 42): "82a15a34ba935e1c",
    ("7", 101): "fa4644339aec2d27",
    ("8", 17): "ade76df579ef5a45",
    ("8", 42): "7615693e72b915ad",
    ("8", 101): "f770fd0ed3eb87b7",
    ("9", 17): "be00fe033145ce1f",
    ("9", 42): "f11129a3d9683259",
    ("9", 101): "01b30862fd3eb4ac",
}


class _CanonicalGenerationLock:
    generation_lock_hash = EXPECTED_GENERATION_LOCK_HASH
    bank_lock_hash = EXPECTED_BANK_LOCK_HASH

    def __init__(self) -> None:
        self._payload: dict[str, object] = {
            "schema_version": "midogpp_uniform_b_v2_generation_lock_v1",
            "claim_scope": "generation_settings_and_frame_lock",
            "bank": {
                "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
                "expert_locks": [
                    {
                        "source_center": center,
                        "training_seed": training_seed,
                        "expert_lock_hash": CANONICAL_EXPERT_LOCK_HASHES[
                            (center, training_seed)
                        ],
                    }
                    for center in CENTERS
                    for training_seed in TRAINING_SEEDS
                ],
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
                "equal_union_source_budget_per_class": 128,
                "total_per_class": TOTAL_PER_CLASS,
            },
            "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        }

    def to_payload(self) -> dict[str, object]:
        return deepcopy(self._payload)


def _canonical_generation_lock() -> _CanonicalGenerationLock:
    return _CanonicalGenerationLock()


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
            "equal_union_source_budget_per_class": 128,
            "total_per_class": TOTAL_PER_CLASS,
        },
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    return payload


def _generation_lock() -> GenerationLock:
    return GenerationLock(_generation_lock_payload())


def _scores() -> tuple[CompatibilityScore, ...]:
    return derive_compatibility_scores(derive_metadata_profiles(DOMAIN_MAPPING))


def _validated_inputs() -> ValidatedTieUnionInputs:
    return ValidatedTieUnionInputs(
        generation_lock=_canonical_generation_lock(),  # type: ignore[arg-type]
        equal_union_policy_lock=SimpleNamespace(
            policy_lock_hash=EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH
        ),
        compatibility_lock=SimpleNamespace(
            compatibility_lock_hash=EXPECTED_COMPATIBILITY_LOCK_HASH
        ),
        compatibility_scores=_scores(),
    )


def _canonical_policy_lock() -> MetadataTieUnionPolicyLock:
    config = load_metadata_tie_union_policy_config(CONFIG)
    inputs = _validated_inputs()
    return build_policy_lock(
        config,
        inputs.generation_lock,
        inputs.equal_union_policy_lock,
        inputs.compatibility_lock,
        inputs.compatibility_scores,
    )


def _input_semantics(
    config: UniformBV2MetadataTieUnionPolicyConfig,
) -> dict[str, dict[str, object]]:
    return {
        EXPERT_BANK_ARTIFACT_ID: {},
        GENERATION_LOCK_ARTIFACT_ID: {
            "generation_lock_contract": "midogpp_uniform_b_v2_generation_lock_v1",
            "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
            "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
            "equal_union_control_lock_hash": "cddbcc3b3343fe38",
        },
        EQUAL_UNION_POLICY_ARTIFACT_ID: {
            "policy_lock_contract": "midogpp_uniform_b_v2_equal_union_policy_lock_v1",
            "config_contract_hash": EXPECTED_EQUAL_UNION_CONFIG_CONTRACT_HASH,
            "policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
            "policy_plan_hash": EXPECTED_EQUAL_UNION_POLICY_PLAN_HASH,
            "assignment_table_hash": EXPECTED_EQUAL_UNION_ASSIGNMENT_TABLE_HASH,
            "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
            "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        },
        COMPATIBILITY_ARTIFACT_ID: dict(COMPATIBILITY_SEMANTIC_IDENTITIES),
    }


def _provenance_fixture(
    tmp_path: Path,
) -> tuple[
    UniformBV2MetadataTieUnionPolicyConfig,
    Path,
    Path,
    dict[str, object],
    dict[str, Path],
]:
    roots = {
        EXPERT_BANK_ARTIFACT_ID: tmp_path / "bank",
        GENERATION_LOCK_ARTIFACT_ID: tmp_path / "generation",
        EQUAL_UNION_POLICY_ARTIFACT_ID: tmp_path / "equal-union",
        COMPATIBILITY_ARTIFACT_ID: tmp_path / "compatibility",
    }
    output_root = tmp_path / "output"
    output_root.joinpath("provenance").mkdir(parents=True)
    resolved_payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved_payload["experiment"]["artifact_root"] = str(output_root)
    resolved_payload["inputs"]["bank_root"] = str(roots[EXPERT_BANK_ARTIFACT_ID])
    resolved_payload["inputs"]["generation_lock_root"] = str(
        roots[GENERATION_LOCK_ARTIFACT_ID]
    )
    resolved_payload["inputs"]["equal_union_policy_root"] = str(
        roots[EQUAL_UNION_POLICY_ARTIFACT_ID]
    )
    resolved_payload["inputs"]["metadata_compatibility_root"] = str(
        roots[COMPATIBILITY_ARTIFACT_ID]
    )
    resolved_config = output_root / "config.resolved.yaml"
    resolved_config.write_text(
        yaml.safe_dump(resolved_payload, sort_keys=False), encoding="utf-8"
    )
    config = load_metadata_tie_union_policy_config(resolved_config)
    specs = {
        EXPERT_BANK_ARTIFACT_ID: (
            "30_expert_bank",
            "expert_bank_construction_only",
            "ROUTING_AUTHORIZED_AFTER_VALIDATION",
            set(BANK_REQUIRED_FILES) | {"reports/validation_report.json"},
        ),
        GENERATION_LOCK_ARTIFACT_ID: (
            "40_prior_and_generation",
            "generation_settings_and_frame_lock",
            "GENERATION_SETTINGS_LOCKED_AFTER_VALIDATION",
            set(GENERATION_REQUIRED_FILES) | {"reports/validation_report.json"},
        ),
        EQUAL_UNION_POLICY_ARTIFACT_ID: (
            "60_routing_and_composition",
            "routing_and_composition",
            "ROUTING_POLICY_FROZEN_AFTER_VALIDATION",
            set(EQUAL_UNION_REQUIRED_FILES),
        ),
        COMPATIBILITY_ARTIFACT_ID: (
            "60_routing_and_composition",
            "routing_compatibility_only",
            "ROUTING_COMPATIBILITY_PROXY_FROZEN_AFTER_VALIDATION",
            set(COMPATIBILITY_REQUIRED_FILES),
        ),
    }
    semantics = _input_semantics(config)
    input_rows: list[dict[str, object]] = []
    members: dict[str, Path] = {}
    for artifact_id in (
        EXPERT_BANK_ARTIFACT_ID,
        GENERATION_LOCK_ARTIFACT_ID,
        EQUAL_UNION_POLICY_ARTIFACT_ID,
        COMPATIBILITY_ARTIFACT_ID,
    ):
        stage, scope, evidence, required = specs[artifact_id]
        artifact_root = roots[artifact_id]
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
                "semantic_identities": semantics[artifact_id],
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
        "experiment_id": EXPERIMENT_ID,
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


def test_config_and_proxy_selection_lock_all_expected_maximum_ties() -> None:
    config = load_metadata_tie_union_policy_config(CONFIG)
    selections = build_policy_selections(_scores(), config)

    assert len(selections) == EXPECTED_SELECTION_COUNT == 9
    assert config.expected_compatibility_lock_hash == EXPECTED_COMPATIBILITY_LOCK_HASH
    assert (
        config.expected_compatibility_score_table_hash
        == EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH
    )
    assert {
        row.target_center: row.selected_source_centers for row in selections
    } == dict(SELECTED_SOURCES_BY_TARGET)
    for row in selections:
        assert row.selected_source_centers == tuple(
            source
            for source, score in zip(
                row.candidate_source_centers,
                row.candidate_exact_match_scores,
                strict=True,
            )
            if score == row.maximum_exact_match_score
        )
        assert row.source_budget_per_class == SOURCE_BUDGET_BY_TIE_COUNT[row.tie_count]
        assert row.source_budget_per_class * row.tie_count == TOTAL_PER_CLASS
        assert row.source_budget_per_class in {1024, 512, 256}
        payload = row.to_payload()
        assert payload["canonical_candidate_order_role"] == (
            "ordering_only_never_tie_break"
        )
        assert payload["tie_break_applied"] is False
        assert payload["metadata_proxy_only"] is True


def test_policy_plan_pairs_exact_stage40_streams_shuffles_and_seed_lattice() -> None:
    config = load_metadata_tie_union_policy_config(CONFIG)
    generation_lock = _generation_lock()
    controls = equal_union_replicate_plan(generation_lock)
    plan = build_policy_plan(generation_lock, _scores(), config)

    assert len(plan) == EXPECTED_REPLICATE_COUNT == 81
    assert {
        (row.target_center, row.training_seed, row.generation_seed) for row in plan
    } == set(product(CENTERS, TRAINING_SEEDS, GENERATION_SEEDS))
    for observed, control in zip(plan, controls, strict=True):
        assert observed.replicate_id == control.replicate_id
        assert observed.class_shuffle_seed_by_label == control.class_shuffle_seed_by_label
        stream_by_source = dict(
            zip(control.candidate_source_centers, control.source_stream_ids, strict=True)
        )
        assert observed.selected_source_stream_ids == tuple(
            stream_by_source[source] for source in observed.selected_source_centers
        )
        assert observed.source_budget_per_class * observed.tie_count == 1024
        assert observed.target_center not in observed.selected_source_centers
        assert observed.to_payload()["seed_selection_performed"] is False


def test_assignments_are_exact_prefixes_with_complete_153_row_coverage() -> None:
    config = load_metadata_tie_union_policy_config(CONFIG)
    rows = assignment_rows(_generation_lock(), _scores(), config)

    assert len(rows) == EXPECTED_ASSIGNMENT_COUNT == 153
    assert len({row.assignment_id for row in rows}) == 153
    assert {row.training_seed for row in rows} == set(TRAINING_SEEDS)
    assert {row.generation_seed for row in rows} == set(GENERATION_SEEDS)
    for row in rows:
        payload = row.to_payload()
        assert payload["source_prefix_start_per_class"] == 0
        assert payload["source_prefix_stop_per_class"] == row.source_budget_per_class
        assert row.source_budget_per_class <= 1024
        assert payload["selection_rank"] is None
        assert payload["tie_break_applied"] is False
        assert payload["learned_weight"] is None
        assert payload["target_expert"] is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    (
        ("inputs", "stage50_artifact_id", "forbidden"),
        ("inputs", "target_labels_path", "/forbidden/labels.csv"),
        ("policy_contract", "total_per_class", 2048),
        ("policy_contract", "no_seed_selection", False),
        ("composition_execution", "shuffle_seed_reused_exactly", False),
        ("claim_boundary", "routing_quality_claimed", True),
    ),
)
def test_config_rejects_budget_seed_target_or_claim_drift(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[section][key] = value
    candidate = tmp_path / "drifted.yaml"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="drifted"):
        load_metadata_tie_union_policy_config(candidate)


def test_selection_rejects_dropping_a_maximum_tie_even_if_rows_are_complete() -> None:
    scores = list(_scores())
    index = next(
        idx
        for idx, row in enumerate(scores)
        if row.target_center == "7" and row.source_center == "9"
    )
    row = scores[index]
    scores[index] = replace(
        row,
        scanner_model_exact_match=0,
        exact_match_count=row.exact_match_count - row.scanner_model_exact_match,
    )

    with pytest.raises(ProtocolError, match="maximum-tie selection drifted"):
        build_policy_selections(tuple(scores))


def test_policy_rejects_rehashed_stage40_source_budget_drift() -> None:
    payload = _generation_lock_payload()
    payload["generation"]["max_source_block_per_class"] = 512  # type: ignore[index]
    payload["generation_lock_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "generation_lock_hash"}
    )

    with pytest.raises(ProtocolError, match="Stage-40 source budget drifted"):
        build_policy_plan(GenerationLock(payload), _scores())


def test_policy_lock_rejects_byte_level_tampering() -> None:
    tampered = _canonical_policy_lock().to_payload()
    tampered["policy"]["all_maximum_ties_retained"] = False  # type: ignore[index]

    with pytest.raises(ProtocolError, match="hash drifted"):
        MetadataTieUnionPolicyLock(tampered)


def test_policy_lock_defensively_owns_caller_payload() -> None:
    caller_payload = _canonical_policy_lock().to_payload()
    lock = MetadataTieUnionPolicyLock(caller_payload)

    caller_payload["policy"]["family"] = "caller_mutation"  # type: ignore[index]
    caller_payload["upstreams"]["generation_lock_hash"] = "caller_mutation"  # type: ignore[index]

    observed = lock.to_payload()
    assert observed["policy"]["family"] != "caller_mutation"  # type: ignore[index]
    assert (
        observed["upstreams"]["generation_lock_hash"]  # type: ignore[index]
        == EXPECTED_GENERATION_LOCK_HASH
    )


def test_policy_lock_reader_rejects_rehashed_semantic_drift(tmp_path: Path) -> None:
    payload = _canonical_policy_lock().to_payload()
    payload["policy"]["family"] = "rehashed_but_noncanonical"  # type: ignore[index]
    payload["policy_lock_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "policy_lock_hash"}
    )
    path = tmp_path / "policy_lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProtocolError, match="semantic identity|semantics drifted"):
        read_policy_lock(path)


def test_policy_provenance_accepts_only_four_frozen_upstreams(tmp_path: Path) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)

    validate_policy_provenance(output_root, config=config)


def test_policy_provenance_rejects_input_byte_drift(tmp_path: Path) -> None:
    config, output_root, _, _, members = _provenance_fixture(tmp_path)
    member = members[
        f"{COMPATIBILITY_ARTIFACT_ID}:tables/compatibility_scores.csv"
    ]
    member.write_text("drifted bytes\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="input member drifted"):
        validate_policy_provenance(output_root, config=config)


@pytest.mark.parametrize("tamper", ("target_eval", "extra_input", "metric_field"))
def test_policy_provenance_rejects_target_eval_extra_input_or_metric(
    tmp_path: Path,
    tamper: str,
) -> None:
    config, output_root, path, manifest, _ = _provenance_fixture(tmp_path)
    if tamper == "target_eval":
        manifest["selection_used_target_eval_artifacts"] = True
    elif tamper == "extra_input":
        manifest["input_artifacts"].append(  # type: ignore[union-attr]
            {"artifact_id": "midogpp_stage50_forbidden"}
        )
    else:
        manifest["bacc"] = 0.99
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProtocolError, match="provenance|four frozen inputs|fields drifted"):
        validate_policy_provenance(output_root, config=config)


@pytest.mark.parametrize("tamper", ("invented_status", "expectation_status_mismatch"))
def test_policy_provenance_rejects_invalid_or_inconsistent_integrity_status(
    tmp_path: Path,
    tamper: str,
) -> None:
    config, output_root, path, manifest, _ = _provenance_fixture(tmp_path)
    integrity = manifest["input_artifacts"][0]["file_integrity"]  # type: ignore[index]
    if tamper == "invented_status":
        integrity["status"] = "PASS"
    else:
        first = integrity["files"][0]
        first["expected"] = {
            "algorithm": "sha256",
            "digest": first["computed"]["sha256"],
        }
        first["verification"] = "MATCH"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ProtocolError, match="integrity|status drifted"):
        validate_policy_provenance(output_root, config=config)


def _stub_validated_inputs(monkeypatch: pytest.MonkeyPatch) -> ValidatedTieUnionInputs:
    from midogpp_thesis.cvae.routing.metadata_tie_union import runner as runner_module
    from midogpp_thesis.cvae.routing.metadata_tie_union import validation as validation_module

    inputs = _validated_inputs()
    monkeypatch.setattr(runner_module, "load_validated_inputs", lambda _: inputs)
    monkeypatch.setattr(validation_module, "load_validated_inputs", lambda _: inputs)
    return inputs


def test_runner_materializes_closed_world_bundle_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    inputs = _stub_validated_inputs(monkeypatch)

    assert run_metadata_tie_union_policy_lock(config) == output_root
    assert run_metadata_tie_union_policy_lock(config) == output_root
    actual = {
        member.relative_to(output_root).as_posix()
        for member in output_root.rglob("*")
        if member.is_file()
    }
    assert actual == set(POLICY_REQUIRED_FILES)
    checks = validate_metadata_tie_union_policy_bundle(
        output_root,
        config=config,
        _validated_inputs=inputs,
    )
    assert checks["status"] == "PASS"
    assert checks["selection_count"] == 9
    assert checks["target_replicate_count"] == 81
    assert checks["assignment_count"] == 153


@pytest.mark.parametrize(
    "tamper",
    ("missing_validation_report", "rehashed_policy_lock", "unexpected_metric_file"),
)
def test_runner_downgrades_invalid_complete_bundle_to_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    _stub_validated_inputs(monkeypatch)
    run_metadata_tie_union_policy_lock(config)
    if tamper == "missing_validation_report":
        (output_root / "reports/validation_report.json").unlink()
    elif tamper == "rehashed_policy_lock":
        lock_path = output_root / "manifests/policy_lock.json"
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        payload["policy"]["family"] = "rehashed_but_noncanonical"
        payload["policy_lock_hash"] = stable_hash(
            {
                key: value
                for key, value in payload.items()
                if key != "policy_lock_hash"
            }
        )
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        (output_root / "reports/stale_bacc.json").write_text(
            '{"bacc": 0.99}\n', encoding="utf-8"
        )

    with pytest.raises(ProtocolError):
        run_metadata_tie_union_policy_lock(config)

    state = json.loads(
        (output_root / "reports/run_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "FAILED"


def test_runner_rejects_stale_target_metric_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    _stub_validated_inputs(monkeypatch)
    stale = output_root / "reports/stage70_bacc.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text('{"bacc": 0.99}\n', encoding="utf-8")

    with pytest.raises(ProtocolError, match="unexpected files"):
        run_metadata_tie_union_policy_lock(config)


def test_runner_records_failed_state_on_reconstructive_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midogpp_thesis.cvae.routing.metadata_tie_union import runner as runner_module

    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    _stub_validated_inputs(monkeypatch)

    def fail(*args: object, **kwargs: object) -> object:
        raise ProtocolError("injected reconstructive failure")

    monkeypatch.setattr(runner_module, "build_policy_plan_payload", fail)
    with pytest.raises(ProtocolError, match="injected reconstructive failure"):
        run_metadata_tie_union_policy_lock(config)
    state = json.loads(
        (output_root / "reports/run_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "FAILED"


@pytest.mark.parametrize("tamper", ("selection", "assignment", "protocol_metric"))
def test_reconstructive_validator_rejects_table_or_claim_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config, output_root, _, _, _ = _provenance_fixture(tmp_path)
    inputs = _stub_validated_inputs(monkeypatch)
    run_metadata_tie_union_policy_lock(config)
    if tamper == "selection":
        path = output_root / "tables/policy_selections.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace("['5']", "['5', '6']")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        match = "selection table content drifted|content member drifted"
    elif tamper == "assignment":
        path = output_root / "tables/policy_assignments.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] += ",target_utility"
        lines[1:] = [f"{line},0.99" for line in lines[1:]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        match = "assignment table columns drifted|content member drifted"
    else:
        path = output_root / "manifests/protocol_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["bacc"] = 0.99
        payload["protocol_hash"] = stable_hash(
            {key: value for key, value in payload.items() if key != "protocol_hash"}
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        match = "protocol fields drifted|content member drifted"

    with pytest.raises(ProtocolError, match=match):
        validate_metadata_tie_union_policy_bundle(
            output_root,
            config=config,
            _validated_inputs=inputs,
        )
