from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh import (
    BASE_ACTION_ID,
    CENTERS,
    GLOBAL_ACTION_ID,
    PERMUTATION_ACTION_ID,
    SUPPORT_ACTION_ID,
    UNIFORM_ACTION_ID,
    FrozenActionPayload,
    PredictionCell,
    build_evaluation_plan,
    legal_sources,
    load_residual_topup_fresh_config,
    run_residual_topup_fresh,
    tail_action_id,
    validate_prediction_seal,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.config import (
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.label_access import (
    SCORING_COLUMNS,
    SCORING_SCHEMA,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.runner import (
    FreshRunnerDependencies,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.target_cache import (
    CACHE_CONTENT_SCHEMA,
    CACHE_PROTOCOL_SCHEMA,
    ROW_COLUMNS,
    ROW_INDEX_MEMBER,
    ROW_SCHEMA,
    load_fresh_target_surface,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.workspace_binding import (
    validate_residual_topup_fresh_workspace_binding,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.residual_topup.hashing import canonical_sha256


CONFIG = Path(
    "experiments/midogpp/stages/70_frozen_policy_downstream/configs/"
    "uniform_b_v2_residual_topup_b_u_g_s_fresh_v1.yaml"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _with_short_hash(payload: dict[str, object], key: str) -> dict[str, object]:
    return {**payload, key: stable_hash(payload)}


def _write_target_fixture(tmp_path: Path):
    base = load_residual_topup_fresh_config(CONFIG)
    policy_root = tmp_path / "policy"
    cache_root = tmp_path / "target-cache"
    reservation_path = tmp_path / "reservation/manifests/reservation.json"
    scoring_path = tmp_path / "scoring/manifest.csv"
    support = {
        center: [f"support::{center}"] for center in CENTERS
    }
    evaluation = {
        center: [f"evaluation::{center}::0", f"evaluation::{center}::1"]
        for center in CENTERS
    }

    scoring_path.parent.mkdir(parents=True, exist_ok=True)
    with scoring_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORING_COLUMNS)
        writer.writeheader()
        # Rows are filled after reservation/cache hashes are known; target
        # admission hashes bytes and deliberately does not parse them.
        writer.writerow({column: "sealed" for column in SCORING_COLUMNS})
    scoring_sha = _sha(scoring_path)

    reservation_unhashed = {
        "schema_version": "midogpp_residual_topup_fresh_target_reservation_v1",
        "artifact_id": "midogpp_residual_topup_fresh_target_reservation_v1",
        "status": "COMPLETE",
        "dataset_family": "MIDOG++",
        "centers": list(CENTERS),
        "reservation_id": "fresh-reservation-v1",
        "split_role": "fresh_unconsumed_case_disjoint_target_evaluation",
        "reservation_frozen_before_cache_extraction": True,
        "fresh_unconsumed_surface": True,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
        "support_evaluation_case_disjoint": True,
        "labels_opened": False,
        "scoring_manifest_artifact_id": (
            "midogpp_residual_topup_fresh_target_manifest_v1"
        ),
        "scoring_manifest_sha256": scoring_sha,
        "support_case_ids_by_center": support,
        "evaluation_case_ids_by_center": evaluation,
    }
    reservation = _with_short_hash(reservation_unhashed, "reservation_hash")
    _write_json(reservation_path, reservation)

    policy_unhashed = {
        "schema_version": "midogpp_residual_topup_b_u_g_s_policy_lock_v1",
        "fresh_surface_reservation_id": "fresh-reservation-v1",
        "support_case_ids_by_target": support,
        "evaluation_case_ids_by_target": evaluation,
        "policy_frozen_before_stage70": True,
    }
    policy = {
        **policy_unhashed,
        "policy_lock_hash": canonical_sha256(policy_unhashed),
    }
    _write_json(policy_root / "manifests/policy_lock.json", policy)
    _write_json(
        policy_root / "reports/run_state.json",
        {"status": "COMPLETE"},
    )
    _write_json(
        policy_root / "reports/validation_report.json",
        {
            "status": "PASS",
            "validator": "validate_residual_topup_policy_bundle",
            "checks": {
                "status": "PASS",
                "policy_lock_hash": policy["policy_lock_hash"],
                "labels_consumed": False,
                "target_evaluation_used": False,
                "source_experts_updated": False,
            },
        },
    )

    row_path = cache_root / ROW_INDEX_MEMBER
    row_path.parent.mkdir(parents=True, exist_ok=True)
    row_rows: list[dict[str, object]] = []
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_COLUMNS)
        writer.writeheader()
        for center in CENTERS:
            member = f"embeddings/by_center/center_{center}.npy"
            array_path = cache_root / member
            array_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(array_path, np.zeros((2, 3840), dtype=np.float32), allow_pickle=False)
            for index, case_id in enumerate(evaluation[center]):
                row = {
                    "schema_version": ROW_SCHEMA,
                    "row_id": f"row::{center}::{index}",
                    "center": center,
                    "case_id": case_id,
                    "center_row_index": index,
                    "embedding_file": member,
                }
                row_rows.append(row)
                writer.writerow(row)

    members = [
        {"path": ROW_INDEX_MEMBER, "sha256": _sha(row_path)},
        *(
            {
                "path": f"embeddings/by_center/center_{center}.npy",
                "sha256": _sha(
                    cache_root / f"embeddings/by_center/center_{center}.npy"
                ),
            }
            for center in CENTERS
        ),
    ]
    content_unhashed = {
        "schema_version": CACHE_CONTENT_SCHEMA,
        "artifact_id": "midogpp_residual_topup_fresh_target_cache_v1",
        "status": "COMPLETE",
        "labels_persisted": False,
        "files": members,
    }
    content = _with_short_hash(content_unhashed, "content_hash")
    _write_json(cache_root / "manifests/content_index.json", content)
    normalized_rows = [
        {
            **row,
            "center_row_index": int(row["center_row_index"]),
        }
        for row in row_rows
    ]
    protocol_unhashed = {
        "schema_version": CACHE_PROTOCOL_SCHEMA,
        "artifact_id": "midogpp_residual_topup_fresh_target_cache_v1",
        "status": "COMPLETE",
        "dataset_family": "MIDOG++",
        "representation_id": "annotation_jpeg_fixed_center_b_v3",
        "feature_backbone": "Virchow2",
        "feature_dim": 3840,
        "reservation_artifact_id": (
            "midogpp_residual_topup_fresh_target_reservation_v1"
        ),
        "reservation_hash": reservation["reservation_hash"],
        "policy_artifact_id": (
            "midogpp_output_uniform_b_v2_residual_topup_b_u_g_s_policy_lock_v1"
        ),
        "policy_lock_hash": policy["policy_lock_hash"],
        "policy_lock_frozen_before_target_cache_extraction": True,
        "scoring_manifest_artifact_id": (
            "midogpp_residual_topup_fresh_target_manifest_v1"
        ),
        "scoring_manifest_sha256": scoring_sha,
        "cache_content_hash": content["content_hash"],
        "row_identity_hash": stable_hash(normalized_rows),
        "labels_persisted": False,
        "fresh_unconsumed_surface": True,
        "consumed_test_used": False,
        "consumed_validation_used": False,
        "consumed_stage90_used": False,
        "reservation_frozen_before_cache_extraction": True,
    }
    protocol = _with_short_hash(protocol_unhashed, "cache_protocol_hash")
    _write_json(cache_root / "manifests/cache_protocol.json", protocol)
    config = replace(
        base,
        artifact_root=tmp_path / "output",
        policy_root=policy_root,
        fresh_target_cache_root=cache_root,
        fresh_reservation_path=reservation_path,
        fresh_scoring_manifest_path=scoring_path,
    )
    return config, reservation, policy, protocol


def _rehash_reservation(path: Path, payload: dict[str, object]) -> None:
    unhashed = {key: value for key, value in payload.items() if key != "reservation_hash"}
    _write_json(path, {**unhashed, "reservation_hash": stable_hash(unhashed)})


def test_target_surface_enforces_global_case_disjointness(tmp_path: Path) -> None:
    config, reservation, _, _ = _write_target_fixture(tmp_path)
    surface = load_fresh_target_surface(config)
    assert surface.reservation.reservation_id == "fresh-reservation-v1"
    assert all(not frame.embeddings.flags.writeable for frame in surface.frames_by_center.values())

    duplicate = json.loads(json.dumps(reservation))
    duplicate["support_case_ids_by_center"][CENTERS[1]][0] = duplicate[
        "support_case_ids_by_center"
    ][CENTERS[0]][0]
    _rehash_reservation(config.fresh_reservation_path, duplicate)
    with pytest.raises(ProtocolError, match="globally unique and disjoint"):
        load_fresh_target_surface(config)


def test_target_surface_rejects_policy_grid_or_cache_hash_drift_before_rows(
    tmp_path: Path,
) -> None:
    config, _, policy, protocol = _write_target_fixture(tmp_path)
    drifted_policy = json.loads(json.dumps(policy))
    drifted_policy["evaluation_case_ids_by_target"][CENTERS[0]][0] = "other-case"
    unhashed = {
        key: value for key, value in drifted_policy.items() if key != "policy_lock_hash"
    }
    drifted_policy["policy_lock_hash"] = canonical_sha256(unhashed)
    _write_json(config.policy_root / "manifests/policy_lock.json", drifted_policy)
    validation = json.loads(
        (config.policy_root / "reports/validation_report.json").read_text()
    )
    validation["checks"]["policy_lock_hash"] = drifted_policy["policy_lock_hash"]
    _write_json(config.policy_root / "reports/validation_report.json", validation)
    # If row parsing happened first this malformed file would produce a row-index error.
    (config.fresh_target_cache_root / ROW_INDEX_MEMBER).write_text("bad\n")
    with pytest.raises(ProtocolError, match="case grids drifted"):
        load_fresh_target_surface(config)

    config, _, _, protocol = _write_target_fixture(tmp_path / "cache-drift")
    drifted_protocol = dict(protocol)
    drifted_protocol["policy_lock_hash"] = "0" * 64
    unhashed_protocol = {
        key: value
        for key, value in drifted_protocol.items()
        if key != "cache_protocol_hash"
    }
    drifted_protocol["cache_protocol_hash"] = stable_hash(unhashed_protocol)
    _write_json(
        config.fresh_target_cache_root / "manifests/cache_protocol.json",
        drifted_protocol,
    )
    with pytest.raises(ProtocolError, match="target-cache protocol drifted"):
        load_fresh_target_surface(config)


def _actions_by_target() -> dict[str, dict[str, FrozenActionPayload]]:
    output: dict[str, dict[str, FrozenActionPayload]] = {}
    for target in CENTERS:
        sources = legal_sources(target)
        ranks = {
            source: index / float(len(sources) - 1)
            for index, source in enumerate(sources)
        }
        permutation = {
            source: sources[(index + 1) % len(sources)]
            for index, source in enumerate(sources)
        }
        support = {source: 128 for source in sources}
        support[sources[0]] += 128
        permuted = {permutation[source]: support[source] for source in sources}

        def action(
            action_id: str,
            counts: dict[str, int],
            *,
            action_ranks: dict[str, float] | None = None,
            action_permutation: dict[str, str] | None = None,
        ) -> FrozenActionPayload:
            return FrozenActionPayload(
                target_center=target,
                action_id=action_id,
                source_counts_by_class={0: counts, 1: counts},
                action_hash=f"hash::{target}::{action_id}",
                mean_normalized_midrank_by_source=action_ranks or {},
                source_identity_permutation=action_permutation or {},
            )

        actions = {
            BASE_ACTION_ID: action(
                BASE_ACTION_ID, {source: 128 for source in sources}
            ),
            UNIFORM_ACTION_ID: action(
                UNIFORM_ACTION_ID, {source: 144 for source in sources}
            ),
            GLOBAL_ACTION_ID: action(
                GLOBAL_ACTION_ID,
                {source: 128 + (128 if source == sources[-1] else 0) for source in sources},
                action_ranks={source: 1.0 - ranks[source] for source in sources},
            ),
            SUPPORT_ACTION_ID: action(
                SUPPORT_ACTION_ID, support, action_ranks=ranks
            ),
            PERMUTATION_ACTION_ID: action(
                PERMUTATION_ACTION_ID,
                permuted,
                action_ranks={permutation[source]: ranks[source] for source in sources},
                action_permutation=permutation,
            ),
        }
        for source in sources:
            counts = {candidate: 128 for candidate in sources}
            counts[source] += 128
            actions[tail_action_id(source)] = action(tail_action_id(source), counts)
        output[target] = actions
    return output


def test_runner_issues_complete_seal_before_injected_label_opener(
    tmp_path: Path,
) -> None:
    config = replace(
        load_residual_topup_fresh_config(CONFIG),
        artifact_root=tmp_path / "output",
    )
    rows = {
        target: (f"row::{target}::0", f"row::{target}::1")
        for target in CENTERS
    }
    actions = _actions_by_target()
    plan = build_evaluation_plan(actions, evaluation_row_ids_by_target=rows)
    predictions = tuple(
        PredictionCell(
            target_center=cell.target_center,
            training_seed=cell.training_seed,
            generation_seed=cell.generation_seed,
            action_id=cell.action_id,
            action_hash=cell.action_hash,
            evaluation_row_ids=rows[cell.target_center],
            probabilities=np.asarray([0.25, 0.75]),
        )
        for cell in plan.cells
    )
    events: list[str] = []

    class StopAfterSeal(RuntimeError):
        pass

    def labels(_target: object, capability: object) -> dict[str, int]:
        events.append("labels")
        assert validate_prediction_seal(capability).prediction_cell_count == 1053
        raise StopAfterSeal

    deps = FreshRunnerDependencies(
        require_inputs=lambda _config: events.append("inputs"),
        validate_workspace=lambda _config: events.append("workspace"),
        run_preflight=lambda *_args, **_kwargs: events.append("preflight") or {"status": "PASS"},
        load_policy=lambda _config: events.append("policy") or SimpleNamespace(actions_by_target=actions),
        load_target=lambda _config: events.append("target") or SimpleNamespace(evaluation_row_ids_by_target=rows),
        load_generation=lambda _config: events.append("generation") or SimpleNamespace(generation_lock_hash="generation"),
        materialize_source=lambda *_args, **_kwargs: events.append("source") or object(),
        materialize_prediction=lambda *_args, **_kwargs: events.append("prediction") or SimpleNamespace(predictions=predictions),
        open_labels=labels,
    )
    with pytest.raises(StopAfterSeal):
        run_residual_topup_fresh(config, dependencies=deps)
    assert events == [
        "inputs",
        "workspace",
        "preflight",
        "policy",
        "target",
        "generation",
        "source",
        "prediction",
        "labels",
    ]


def test_runner_fails_immediately_for_current_absent_fresh_artifacts() -> None:
    config = load_residual_topup_fresh_config(CONFIG)
    workspace_called = False

    def workspace(_config: object) -> None:
        nonlocal workspace_called
        workspace_called = True

    with pytest.raises(ProtocolError, match="Fresh Stage-70 is blocked"):
        run_residual_topup_fresh(
            config,
            dependencies=FreshRunnerDependencies(validate_workspace=workspace),
        )
    assert workspace_called is False


def test_planned_registry_status_is_not_a_production_launch(monkeypatch) -> None:
    config = load_residual_topup_fresh_config(CONFIG)
    experiment = SimpleNamespace(
        status="planned",
        stage="70_frozen_policy_downstream",
        claim_scope="synthetic_downstream_utility",
        output_artifact_id=OUTPUT_ARTIFACT_ID,
        input_artifact_ids=INPUT_ARTIFACT_IDS,
    )
    output = SimpleNamespace(
        stage="70_frozen_policy_downstream",
        claim_scope="synthetic_downstream_utility",
        may_feed_deployable_selection=False,
        may_feed_recipe_selection=False,
    )
    fake = SimpleNamespace(
        validate=lambda: None,
        get_experiment=lambda _experiment_id: experiment,
        artifacts={OUTPUT_ARTIFACT_ID: output},
        stages={
            "70_frozen_policy_downstream": {
                "allowed_claim_scopes": ("synthetic_downstream_utility",)
            }
        },
    )
    import midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.workspace_binding as binding

    monkeypatch.setattr(binding.MidogppWorkspace, "load", lambda: fake)
    with pytest.raises(ProtocolError, match="registry binding drifted"):
        validate_residual_topup_fresh_workspace_binding(config)
