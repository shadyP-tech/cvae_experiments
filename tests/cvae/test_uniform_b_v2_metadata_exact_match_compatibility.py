from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.metadata_compatibility.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.routing.metadata_compatibility.config import (
    MetadataCompatibilityConfig,
    load_metadata_compatibility_config,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.contracts import (
    CLAIM_SCOPE,
    DOMAIN_MAPPING_SHA256,
    ELIGIBLE_CENTERS,
    EXPECTED_COMPATIBILITY_LOCK_HASH,
    EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPECTED_METADATA_PROFILE_LOCK_HASH,
    EXPECTED_METADATA_PROFILE_TABLE_HASH,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_ID,
    MetadataCompatibilityLock,
    MetadataProfile,
    candidate_sources,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.locks import (
    read_compatibility_lock,
    read_metadata_profile_lock,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.profiles import (
    derive_metadata_profiles,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.runner import (
    run_metadata_compatibility_lock,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.scoring import (
    compatibility_score_table_hash,
    derive_compatibility_scores,
    metadata_profile_table_hash,
    score_profile_values,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.table_io import (
    read_compatibility_scores_table,
    read_metadata_profiles_table,
)
from midogpp_thesis.cvae.routing.metadata_compatibility.validation import (
    validate_metadata_compatibility_bundle,
    validate_metadata_compatibility_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_metadata_exact_match_compatibility_v1.yaml"
)
DOMAIN_MAPPING = ROOT / "datasets/midogpp/contract/annotation_patch_v1/domain_mapping.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[MetadataCompatibilityConfig, Path, Path]:
    input_root = tmp_path / "metadata_input"
    input_root.mkdir(parents=True)
    mapping = input_root / "domain_mapping.json"
    shutil.copyfile(DOMAIN_MAPPING, mapping)
    output = tmp_path / "output"
    (output / "provenance").mkdir(parents=True)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["experiment"]["artifact_root"] = str(output)
    payload["inputs"]["metadata_mapping_path"] = str(mapping)
    resolved = output / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_metadata_compatibility_config(resolved)

    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": EXPERIMENT_ID,
        "stage": "60_routing_and_composition",
        "claim_scope": CLAIM_SCOPE,
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            {
                "artifact_id": INPUT_ARTIFACT_ID,
                "resolved_path": str(input_root.resolve()),
                "stage": "dataset_contract",
                "evidence_label": "ROUTING_METADATA_INPUT_AUTHORIZED",
                "claim_scope": "dataset_contract_and_split_provenance",
                "semantic_identities": {
                    "routing_metadata_source": "midogpp_domain_mapping_v1",
                    "domain_axis": "tumor_type|lab_or_origin|scanner_model",
                    "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
                },
                "semantic_identities_are_file_hashes": False,
                "file_integrity": {
                    "status": "EXPECTED_FILE_HASHES_MATCH",
                    "default_recording_algorithm": "sha256",
                    "files": [
                        {
                            "path": "domain_mapping.json",
                            "resolved_path": str(mapping.resolve()),
                            "exists": True,
                            "expected": {
                                "algorithm": "sha256",
                                "digest": DOMAIN_MAPPING_SHA256,
                            },
                            "size_bytes": mapping.stat().st_size,
                            "computed": {"sha256": DOMAIN_MAPPING_SHA256},
                            "verification": "MATCH",
                        }
                    ],
                },
                "exists": True,
            }
        ],
        "repository_revision": "0" * 40,
        "repository_dirty": True,
        "repository_status_hash": "1" * 64,
    }
    (output / "provenance/input_artifacts.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return config, output, mapping


def test_config_and_semantic_hashes_are_frozen() -> None:
    config = load_metadata_compatibility_config(CONFIG)
    assert config.experiment_id == EXPERIMENT_ID
    assert config.contract_hash == EXPECTED_CONFIG_CONTRACT_HASH == "89191838fbb3f1c8"
    assert config.ordered_axes == ("tumor_type", "lab_or_origin", "scanner_model")
    assert config.eligible_centers == ELIGIBLE_CENTERS
    assert config.profile_contract["parsed_input_fields"] == [
        "domain_axis",
        "domain_name_to_id",
    ]
    assert config.profile_contract["center_4_profile_emitted"] is False
    assert config.compatibility_contract["scorer_inputs"] == (
        "metadata_profile_values_only"
    )
    assert config.compatibility_contract["ranking_performed"] is False
    assert config.claim_boundary["metadata_score_is_proxy_only"] is True
    assert config.claim_boundary["true_utility_computed"] is False


def test_exact_profiles_and_all_72_ordered_target_excluded_scores() -> None:
    profiles = derive_metadata_profiles(DOMAIN_MAPPING)
    assert {center: profile.values for center, profile in profiles.items()} == {
        "0": ("canine cutaneous mast cell tumor", "FU Berlin", "Aperio CS2"),
        "1": ("canine lung cancer", "VMU Vienna", "3D Histech"),
        "2": ("canine lymphoma", "VMU Vienna", "3D Histech"),
        "3": ("canine soft tissue sarcoma", "AMC New York", "3D Histech"),
        "5": ("human breast cancer", "UMC Utrecht", "Aperio CS2"),
        "6": ("human breast cancer", "UMC Utrecht", "Hamamatsu S360"),
        "7": ("human breast cancer", "UMC Utrecht", "Hamamatsu XR"),
        "8": ("human melanoma", "UMC Utrecht", "Hamamatsu XR"),
        "9": ("human neuroendocrine tumor", "UMC Utrecht", "Hamamatsu XR"),
    }
    assert "4" not in profiles
    assert metadata_profile_table_hash(profiles) == EXPECTED_METADATA_PROFILE_TABLE_HASH
    assert EXPECTED_METADATA_PROFILE_TABLE_HASH == "eee8dececd62bef8"

    rows = derive_compatibility_scores(profiles)
    assert len(rows) == 72
    assert tuple((row.target_center, row.source_center) for row in rows) == tuple(
        (target, source)
        for target in ELIGIBLE_CENTERS
        for source in candidate_sources(target)
    )
    assert all(row.target_center != row.source_center for row in rows)
    assert all("4" not in (row.target_center, row.source_center) for row in rows)
    assert {score: sum(row.exact_match_count == score for row in rows) for score in range(4)} == {
        0: 44,
        1: 14,
        2: 14,
        3: 0,
    }
    keyed = {(row.target_center, row.source_center): row for row in rows}
    assert keyed[("1", "2")].exact_match_count == 2
    assert keyed[("1", "3")].exact_match_count == 1
    assert keyed[("5", "6")].exact_match_count == 2
    assert keyed[("7", "8")].exact_match_count == 2
    assert keyed[("0", "1")].exact_match_count == 0
    assert compatibility_score_table_hash(rows) == EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH
    assert EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH == "aec9e0b5b09a1fe5"


def test_scorer_accepts_profile_values_only_and_never_ids() -> None:
    assert tuple(MetadataProfile.__dataclass_fields__) == (
        "tumor_type",
        "lab_or_origin",
        "scanner_model",
    )
    target = MetadataProfile("tumor", "lab", "scanner")
    source = MetadataProfile("tumor", "other lab", "scanner")
    assert score_profile_values(target, source) == (1, 0, 1, 2)
    with pytest.raises(ProtocolError, match="values only, never IDs"):
        score_profile_values("0", "1")  # type: ignore[arg-type]


def test_direct_runner_builds_closed_world_valid_idempotent_bundle(tmp_path: Path) -> None:
    config, output, _ = _fixture(tmp_path)
    assert run_metadata_compatibility_lock(config) == output
    assert {
        member.relative_to(output).as_posix()
        for member in output.rglob("*")
        if member.is_file()
    } == set(REQUIRED_FILES)
    assert validate_metadata_compatibility_bundle(output, config=config)["status"] == "PASS"
    assert read_metadata_profiles_table(output) == derive_metadata_profiles(
        config.metadata_mapping_path
    )
    scores = read_compatibility_scores_table(output)
    lock = read_compatibility_lock(output)
    assert len(scores) == 72
    assert lock.metadata_profile_lock_hash == EXPECTED_METADATA_PROFILE_LOCK_HASH
    assert EXPECTED_METADATA_PROFILE_LOCK_HASH == "de23d1c8de734503"
    assert lock.compatibility_lock_hash == EXPECTED_COMPATIBILITY_LOCK_HASH
    assert EXPECTED_COMPATIBILITY_LOCK_HASH == "4b46b3d157b07781"

    before = {relative: _sha256(output / relative) for relative in REQUIRED_FILES}
    assert run_metadata_compatibility_lock(config) == output
    after = {relative: _sha256(output / relative) for relative in REQUIRED_FILES}
    assert after == before


def test_low_level_runner_explicit_root_is_a_cli_independent_test_override(
    tmp_path: Path,
) -> None:
    config, configured_output, _ = _fixture(tmp_path)
    override = tmp_path / "explicit_runner_override"
    (override / "provenance").mkdir(parents=True)
    shutil.copyfile(
        configured_output / "config.resolved.yaml",
        override / "config.resolved.yaml",
    )
    shutil.copyfile(
        configured_output / "provenance/input_artifacts.json",
        override / "provenance/input_artifacts.json",
    )
    assert override != config.artifact_root
    assert run_metadata_compatibility_lock(config, artifact_root=override) == override
    assert validate_metadata_compatibility_bundle(override, config=config)["status"] == "PASS"


@pytest.mark.parametrize(
    "failure_mode",
    ["validation_drift", "missing_validation_report", "unexpected_file"],
)
def test_complete_bundle_preflight_or_validation_failure_marks_failed(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    config, output, _ = _fixture(tmp_path)
    run_metadata_compatibility_lock(config)
    state_path = output / "reports/run_state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "COMPLETE"

    if failure_mode == "validation_drift":
        report = output / "reports/validation_report.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["status"] = "FAIL"
        report.write_text(json.dumps(payload), encoding="utf-8")
    elif failure_mode == "missing_validation_report":
        (output / "reports/validation_report.json").unlink()
    else:
        (output / "reports/undeclared.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ProtocolError):
        run_metadata_compatibility_lock(config)
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_run_state_v1",
        "status": "FAILED",
        "claim_scope": CLAIM_SCOPE,
    }


def test_lock_object_deep_copies_and_freezes_caller_owned_payload(tmp_path: Path) -> None:
    config, output, _ = _fixture(tmp_path)
    run_metadata_compatibility_lock(config)
    raw = json.loads(
        (output / "manifests/compatibility_lock.json").read_text(encoding="utf-8")
    )
    lock = MetadataCompatibilityLock(raw)
    expected = lock.to_payload()

    raw["ordered_axes"][0] = "mutated_axis"
    raw["component_weights"]["tumor_type"] = 99
    raw["experiment_id"] = "mutated.experiment"
    assert lock.to_payload() == expected

    exported = lock.to_payload()
    exported["ordered_axes"][0] = "mutated_export"
    exported["component_weights"]["tumor_type"] = 77
    assert lock.to_payload() == expected
    with pytest.raises(TypeError):
        lock._payload["experiment_id"] = "blocked"  # type: ignore[index]
    with pytest.raises(TypeError):
        lock._payload["component_weights"]["tumor_type"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "midogpp.routing_compatibility.tampered.v1"),
        ("claim_scope", "routing_and_composition"),
        ("scoring_family", "weighted_metadata_similarity"),
        ("metadata_profile_table_hash", "0" * 16),
    ],
)
def test_compatibility_lock_reader_rejects_rehashed_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, output, _ = _fixture(tmp_path)
    run_metadata_compatibility_lock(config)
    lock_path = output / "manifests/compatibility_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload[field] = value
    payload["compatibility_lock_hash"] = stable_hash(
        {key: item for key, item in payload.items() if key != "compatibility_lock_hash"}
    )
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="semantic identity drifted"):
        read_compatibility_lock(lock_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", "midogpp.routing_compatibility.tampered.v1"),
        ("claim_scope", "routing_and_composition"),
        ("metadata_profile_table_hash", "f" * 16),
    ],
)
def test_metadata_profile_lock_reader_rejects_rehashed_semantic_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, output, _ = _fixture(tmp_path)
    run_metadata_compatibility_lock(config)
    lock_path = output / "manifests/metadata_profile_lock.json"
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload[field] = value
    payload["metadata_profile_lock_hash"] = stable_hash(
        {key: item for key, item in payload.items() if key != "metadata_profile_lock_hash"}
    )
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolError, match="semantic identity drifted"):
        read_metadata_profile_lock(lock_path)


def test_provenance_is_narrow_hash_pinned_and_rejects_extra_semantics(
    tmp_path: Path,
) -> None:
    config, output, _ = _fixture(tmp_path)
    run_metadata_compatibility_lock(config)
    validate_metadata_compatibility_provenance(output, config=config)
    provenance_path = output / "provenance/input_artifacts.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert [row["artifact_id"] for row in provenance["input_artifacts"]] == [
        INPUT_ARTIFACT_ID
    ]
    file_row = provenance["input_artifacts"][0]["file_integrity"]["files"][0]
    assert file_row["expected"]["digest"] == DOMAIN_MAPPING_SHA256
    assert file_row["verification"] == "MATCH"

    provenance["input_artifacts"][0]["target_utility"] = 0.99
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ProtocolError, match="provenance fields drifted"):
        validate_metadata_compatibility_bundle(output, config=config)


@pytest.mark.parametrize(
    ("relative", "mutate"),
    [
        (
            "tables/metadata_profiles.csv",
            lambda text: text.replace("FU Berlin", "target label", 1),
        ),
        (
            "tables/compatibility_scores.csv",
            lambda text: text.replace(",0,0,0,0,0,3,True,True", ",1,0,0,1,0,3,True,True", 1),
        ),
        (
            "manifests/compatibility_lock.json",
            lambda text: text.replace('"selection_performed": false', '"selection_performed": true', 1),
        ),
        (
            "reports/compatibility_decision.json",
            lambda text: text.replace('"ranking_performed": false', '"ranking_performed": true', 1),
        ),
        (
            "reports/validation_report.json",
            lambda text: text.replace('"status": "PASS"', '"status": "FAIL"', 1),
        ),
    ],
)
def test_validator_rejects_semantic_and_table_tampering(
    tmp_path: Path,
    relative: str,
    mutate: object,
) -> None:
    config, output, _ = _fixture(tmp_path)
    run_metadata_compatibility_lock(config)
    member = output / relative
    original = member.read_text(encoding="utf-8")
    changed = mutate(original)  # type: ignore[operator]
    assert changed != original
    member.write_text(changed, encoding="utf-8")
    with pytest.raises(ProtocolError):
        validate_metadata_compatibility_bundle(output, config=config)


def test_closed_world_rejection_and_fail_closed_state(tmp_path: Path) -> None:
    config, output, mapping = _fixture(tmp_path)
    mapping.write_text(mapping.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="SHA-256 drifted"):
        run_metadata_compatibility_lock(config)
    failed = json.loads((output / "reports/run_state.json").read_text(encoding="utf-8"))
    assert failed["status"] == "FAILED"

    config, output, _ = _fixture(tmp_path / "closed")
    run_metadata_compatibility_lock(config)
    (output / "reports/undeclared_utility.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="unexpected files"):
        validate_metadata_compatibility_bundle(output, config=config)
    with pytest.raises(ProtocolError, match="unexpected files"):
        run_metadata_compatibility_lock(config)


def test_config_rejects_id_based_scoring_and_extra_sections(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["compatibility_contract"]["scorer_inputs"] = "center_ids"
    tampered = tmp_path / "tampered.yaml"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="compatibility contract drifted"):
        load_metadata_compatibility_config(tampered)

    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["oracle"] = {"nelbo": True}
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="top-level config keys drifted"):
        load_metadata_compatibility_config(tampered)
