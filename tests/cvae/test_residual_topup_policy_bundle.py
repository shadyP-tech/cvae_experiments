from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256
from midogpp_thesis.cvae.routing.residual_topup_policy import (
    PROXY_ENERGY_SEMANTICS,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.actions import (
    GLOBAL_POLICY_ACTION_KIND,
    PERMUTATION_POLICY_ACTION_KIND,
    SINGLE_SOURCE_POLICY_ACTION_KIND,
    SUPPORT_POLICY_ACTION_KIND,
    build_frozen_action_library,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.routing.residual_topup_policy.config import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    load_residual_topup_policy_lock_config,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.io import (
    ATTESTATION_SCHEMA_VERSION,
    PROXY_SCORE_COLUMNS,
    load_validated_fresh_proxy_inputs,
)
from midogpp_thesis.cvae.routing.residual_topup_policy import io as io_module
from midogpp_thesis.cvae.routing.residual_topup_policy.runner import (
    build_policy_products,
    run_residual_topup_policy_lock,
)
from midogpp_thesis.cvae.routing.residual_topup_policy.validation import (
    validate_residual_topup_policy_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1.yaml"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "generation_seeds": list(TRAINING_SEEDS),
            "source_stream_namespace": "uniform_b_v2_source_stream_v1",
            "max_source_block_per_class": 1024,
            "equal_union_source_budget_per_class": 128,
            "total_per_class": 1024,
        },
    }
    payload["generation_lock_hash"] = stable_hash(payload)
    return payload


def _proxy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    index_by_center = {center: index for index, center in enumerate(CENTERS)}
    for target in CENTERS:
        for query in CENTERS:
            if query == target:
                continue
            for source in CENTERS:
                if source in {target, query}:
                    continue
                for seed_index, seed in enumerate(TRAINING_SEEDS):
                    rows.append(
                        {
                            "outer_target": target,
                            "query_role": "global_pseudoquery",
                            "query_center": query,
                            "case_id": f"pq-{query}",
                            "candidate_source": source,
                            "training_seed": seed,
                            "proxy_energy": (
                                index_by_center[source] * 10.0 + seed_index * 0.1
                            ),
                            "labels_consumed": "false",
                            "evaluation_overlap": "false",
                            "source_expert_updated": "false",
                            "proxy_energy_semantics": PROXY_ENERGY_SEMANTICS,
                        }
                    )
        for source in CENTERS:
            if source == target:
                continue
            for seed_index, seed in enumerate(TRAINING_SEEDS):
                rows.append(
                    {
                        "outer_target": target,
                        "query_role": "target_support",
                        "query_center": target,
                        "case_id": f"support-{target}",
                        "candidate_source": source,
                        "training_seed": seed,
                        "proxy_energy": (
                            (8 - index_by_center[source]) * 10.0 + seed_index * 0.1
                        ),
                        "labels_consumed": "false",
                        "evaluation_overlap": "false",
                        "source_expert_updated": "false",
                        "proxy_energy_semantics": PROXY_ENERGY_SEMANTICS,
                    }
                )
    return rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path):
    output = tmp_path / "output"
    bank = tmp_path / "bank"
    generation = tmp_path / "generation"
    equal = tmp_path / "equal"
    proxy = tmp_path / "proxy"
    for root in (output, bank, generation, equal, proxy):
        root.mkdir(parents=True, exist_ok=True)

    bank_lock_path = bank / "manifests/expert_bank_index.json"
    generation_lock_path = generation / "manifests/generation_lock.json"
    equal_lock_path = equal / "manifests/policy_lock.json"
    _write_json(
        bank_lock_path,
        {"schema_version": "fixture_bank_v1", "bank_lock_hash": EXPECTED_BANK_LOCK_HASH},
    )
    _write_json(generation_lock_path, _generation_lock_payload())
    _write_json(
        equal_lock_path,
        {
            "schema_version": "fixture_equal_union_lock_v1",
            "policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
        },
    )

    score_path = proxy / "tables/proxy_scores.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_rows = _proxy_rows()
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROXY_SCORE_COLUMNS))
        writer.writeheader()
        writer.writerows(proxy_rows)
    input_hashes = {
        "expert_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "equal_union_policy_lock_hash": EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
        "expert_bank_index_sha256": _sha256(bank_lock_path),
        "generation_lock_sha256": _sha256(generation_lock_path),
        "equal_union_policy_lock_sha256": _sha256(equal_lock_path),
        "proxy_score_table_sha256": _sha256(score_path),
    }
    attestation: dict[str, object] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "artifact_id": INPUT_ARTIFACT_IDS[-1],
        "authorized_consumer_experiment_id": EXPERIMENT_ID,
        "dataset_family": "MIDOG++",
        "feature_backbone": "Virchow2",
        "representation_id": "annotation_jpeg_fixed_center_b_v3",
        "reservation_id": "fresh-reservation-fixture-v1",
        "proxy_surface_hash": stable_hash({"fixture": "proxy-surface"}),
        "query_shard_hashes": {
            **{
                f"{target}::global_pseudoquery::{query}": stable_hash(
                    {
                        "fixture": "query-shard",
                        "outer_target": target,
                        "query_role": "global_pseudoquery",
                        "query_center": query,
                    }
                )
                for target in CENTERS
                for query in CENTERS
                if query != target
            },
            **{
                f"{target}::target_support::{target}": stable_hash(
                    {
                        "fixture": "query-shard",
                        "outer_target": target,
                        "query_role": "target_support",
                        "query_center": target,
                    }
                )
                for target in CENTERS
            },
        },
        "fresh_surface": True,
        "previously_consumed": False,
        "consumed_stage70_used": False,
        "consumed_stage90_used": False,
        "labels_present": False,
        "labels_consumed": False,
        "evaluation_labels_opened": False,
        "target_evaluation_used": False,
        "source_experts_updated": False,
        "pseudoquery_support_case_overlap_count": 0,
        "pseudoquery_evaluation_case_overlap_count": 0,
        "support_evaluation_case_overlap_count": 0,
        "pseudoquery_case_ids_by_center": {
            center: [f"pq-{center}"] for center in CENTERS
        },
        "support_case_ids_by_target": {
            center: [f"support-{center}"] for center in CENTERS
        },
        "evaluation_case_ids_by_target": {
            center: [f"evaluation-{center}"] for center in CENTERS
        },
        "proxy_score_row_count": len(proxy_rows),
        "input_hashes": input_hashes,
    }
    attestation["attestation_hash"] = canonical_sha256(attestation)
    attestation_path = proxy / "manifests/fresh_surface_attestation.json"
    _write_json(attestation_path, attestation)

    resolved = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    resolved["experiment"]["artifact_root"] = str(output)
    resolved["inputs"].update(
        {
            "expert_bank_root": str(bank),
            "generation_lock_root": str(generation),
            "equal_union_policy_root": str(equal),
            "proxy_surface_root": str(proxy),
            "proxy_score_table_path": str(score_path),
            "proxy_attestation_path": str(attestation_path),
        }
    )
    resolved_path = output / "config.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    config = load_residual_topup_policy_lock_config(resolved_path)

    provenance_rows = []
    for artifact_id, artifact_root in zip(
        INPUT_ARTIFACT_IDS,
        (bank, generation, equal, proxy),
        strict=True,
    ):
        row: dict[str, object] = {
            "artifact_id": artifact_id,
            "resolved_path": str(artifact_root.resolve()),
            "exists": True,
        }
        if artifact_id == INPUT_ARTIFACT_IDS[-1]:
            row["file_integrity"] = {
                "status": "HASHES_RECORDED_NO_EXPECTATIONS",
                "default_recording_algorithm": "sha256",
                "files": [
                    {
                        "path": "tables/proxy_scores.csv",
                        "resolved_path": str(score_path.resolve()),
                        "exists": True,
                        "computed": {"sha256": _sha256(score_path)},
                    },
                    {
                        "path": "manifests/fresh_surface_attestation.json",
                        "resolved_path": str(attestation_path.resolve()),
                        "exists": True,
                        "computed": {"sha256": _sha256(attestation_path)},
                    },
                ],
            }
        provenance_rows.append(row)
    _write_json(
        output / "provenance/input_artifacts.json",
        {
            "schema_version": "midogpp_input_artifacts_v2",
            "dataset_id": "midogpp",
            "experiment_id": EXPERIMENT_ID,
            "stage": "60_routing_and_composition",
            "claim_scope": "routing_and_composition",
            "selection_used_target_eval_artifacts": False,
            "input_artifacts": provenance_rows,
        },
    )
    return config, output, score_path, attestation_path


def _stub_upstream_lock_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        io_module,
        "read_generation_lock",
        lambda _path: SimpleNamespace(
            generation_lock_hash=EXPECTED_GENERATION_LOCK_HASH
        ),
    )
    monkeypatch.setattr(
        io_module,
        "read_equal_union_policy_lock",
        lambda _path: SimpleNamespace(
            policy_lock_hash=EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH
        ),
    )


def test_fresh_inputs_build_complete_distinct_action_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_upstream_lock_readers(monkeypatch)
    config, _, _, _ = _fixture(tmp_path)
    inputs = load_validated_fresh_proxy_inputs(config)
    products = build_policy_products(config, inputs)
    library = products.action_library

    assert library.action_count == 9 * 13 == 117
    for target in CENTERS:
        actions = library.actions_by_target[target]
        assert tuple(action.policy_id for action in actions[:5]) == ("B", "U", "G", "S", "P")
        assert actions[2].action_kind == GLOBAL_POLICY_ACTION_KIND
        assert actions[3].action_kind == SUPPORT_POLICY_ACTION_KIND
        assert actions[4].action_kind == PERMUTATION_POLICY_ACTION_KIND
        assert all(
            action.action_kind == SINGLE_SOURCE_POLICY_ACTION_KIND
            for action in actions[5:]
        )
        assert len({action.action_hash for action in actions}) == 13
        assert sum(actions[0].topup_counts_by_source.values()) == 0
        assert all(
            sum(action.topup_counts_by_source.values()) == 128
            for action in actions[1:]
        )
        assert all(
            sum(action.final_counts_by_class[label].values())
            == action.final_total_per_class
            for action in actions
            for label in (0, 1)
        )


def test_runner_writes_closed_world_lock_and_validator_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_upstream_lock_readers(monkeypatch)
    config, output, _, _ = _fixture(tmp_path)
    run_residual_topup_policy_lock(config)
    assert {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } == set(REQUIRED_FILES)
    checks = validate_residual_topup_policy_bundle(output, config=config)
    assert checks["status"] == "PASS"
    assert checks["action_count"] == 117
    lock = json.loads((output / "manifests/policy_lock.json").read_text())
    assert lock["policy_frozen_before_stage70"] is True
    assert lock["labels_consumed"] is False
    assert lock["target_evaluation_used"] is False
    assert lock["source_experts_updated"] is False
    assert set(lock["actions_by_target"]) == set(CENTERS)

    action_library = output / "manifests/action_library.json"
    payload = json.loads(action_library.read_text())
    payload["actions_by_target"]["0"][0]["final_total_per_class"] = 999
    _write_json(action_library, payload)
    with pytest.raises(ProtocolError, match="action library drifted"):
        validate_residual_topup_policy_bundle(output, config=config)


def test_planned_config_fails_closed_while_fresh_files_are_absent() -> None:
    config = load_residual_topup_policy_lock_config(CONFIG)
    with pytest.raises(ProtocolError, match="planned artifact remains blocked"):
        load_validated_fresh_proxy_inputs(config)


def test_attestation_or_literal_false_drift_fails_closed(tmp_path: Path) -> None:
    config, _, score_path, attestation_path = _fixture(tmp_path)
    rows = list(csv.DictReader(score_path.open(encoding="utf-8")))
    rows[0]["labels_consumed"] = "False"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROXY_SCORE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ProtocolError, match="literal false"):
        load_validated_fresh_proxy_inputs(config)

    config, _, _, attestation_path = _fixture(tmp_path / "second")
    attestation = json.loads(attestation_path.read_text())
    attestation["fresh_surface"] = False
    unhashed = {key: value for key, value in attestation.items() if key != "attestation_hash"}
    attestation["attestation_hash"] = canonical_sha256(unhashed)
    _write_json(attestation_path, attestation)
    with pytest.raises(ProtocolError, match="attestation protocol failed"):
        load_validated_fresh_proxy_inputs(config)
