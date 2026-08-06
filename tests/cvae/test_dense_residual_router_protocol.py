from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.dense_residual_router.bundle import (
    REQUIRED_FILES,
    assert_non_adoptive_payload,
    publication_decision_payload,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.config import (
    canonical_claim_boundary_payload,
    load_dense_residual_diagnostic_config,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.contracts import (
    ACTION_IDS,
    ACTION_LIBRARY_HASH,
    CENTERS,
    CONTROL_ACTION_ID,
    EXPERIMENT_ID,
    ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID,
    PUBLICATION_STATUS,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    ValidationRowIdentity,
    action_library,
    development_queries,
    legal_sources,
    row_identity_hash,
    target_sources,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.label_access import (
    open_development_labels,
    open_target_labels,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.seals import (
    AllActionTargetPredictionSeal,
    PredictionCellSeal,
    build_all_action_target_prediction_seal,
    build_development_prediction_seal,
    build_diagnostic_decision_seal,
    build_target_prediction_seal,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


CONFIG_PATH = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_validation_dense_residual_router_v1.yaml"
)


def test_config_import_does_not_load_runner_or_execution_modules() -> None:
    code = (
        "import sys; "
        "from midogpp_thesis.cvae.diagnostics.dense_residual_router.config "
        "import DenseResidualDiagnosticConfig; "
        "blocked=('midogpp_thesis.cvae.routing.runner', "
        "'midogpp_thesis.cvae.diagnostics.dense_residual_router.runner', "
        "'midogpp_thesis.cvae.diagnostics.dense_residual_router.execution'); "
        "print([name for name in blocked if name in sys.modules])"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ".:src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "[]"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    *,
    ordinal: int,
    manifest_index: int,
    center: str,
    label: int,
    role: str = "evaluation",
) -> ValidationRowIdentity:
    return ValidationRowIdentity(
        row_ordinal=ordinal,
        manifest_row_index=manifest_index,
        sample_id=f"sample-{center}-{role}-{label}",
        case_id=f"case-{center}-{role}-{label}",
        center=center,
        split="val",
        partition_role=role,
    )


def _prediction_cell(
    *,
    phase: str,
    outer_target: str,
    query_center: str,
    action_id: str,
    arm_role: str,
    training_seed: int,
    generation_seed: int,
    rows: tuple[ValidationRowIdentity, ...],
) -> PredictionCellSeal:
    key = (
        f"{phase}-{outer_target}-{query_center}-{action_id}-{arm_role}-"
        f"{training_seed}-{generation_seed}"
    )
    return PredictionCellSeal(
        phase=phase,
        outer_target=outer_target,
        query_center=query_center,
        action_id=action_id,
        arm_role=arm_role,
        candidate_sources=(
            legal_sources(outer_target=outer_target, query_center=query_center)
            if phase == "development"
            else target_sources(outer_target)
        ),
        training_seed=training_seed,
        generation_seed=generation_seed,
        evaluation_row_ids=tuple(row.sample_id for row in rows),
        evaluation_row_identity_hash=row_identity_hash(rows),
        prediction_sha256=_sha(f"prediction-{key}"),
        probability_sha256=_sha(f"probability-{key}"),
        composition_hash=_sha(f"composition-{key}")[:16],
        classifier_config_hash=_sha("classifier")[:16],
    )


def _development_seal(
    rows_by_query: dict[str, tuple[ValidationRowIdentity, ...]],
    *,
    outer_target: str = "0",
    manifest_sha256: str | None = None,
    prediction_index_sha256: str | None = None,
    prediction_arrays_sha256: str | None = None,
):
    cells = tuple(
        _prediction_cell(
            phase="development",
            outer_target=outer_target,
            query_center=query,
            action_id=action,
            arm_role="development_action",
            training_seed=training_seed,
            generation_seed=generation_seed,
            rows=rows_by_query[query],
        )
        for action in ACTION_IDS
        for query in development_queries(outer_target)
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    return build_development_prediction_seal(
        outer_target=outer_target,
        config_contract_hash=_sha("config")[:16],
        support_partition_lock_hash=_sha("partition")[:16],
        validation_cache_binding_hash=_sha("cache")[:16],
        validation_manifest_sha256=manifest_sha256 or _sha("manifest-placeholder"),
        prediction_index_sha256=(
            prediction_index_sha256 or _sha("development-index")
        ),
        prediction_arrays_sha256=(
            prediction_arrays_sha256 or _sha("development-arrays")
        ),
        evaluation_rows_by_query=rows_by_query,
        cells=cells,
    )


def _target_seal(
    rows: tuple[ValidationRowIdentity, ...],
    *,
    manifest_sha256: str,
    outer_target: str = "0",
    selected_action_id: str = "rho_0.25",
    prediction_index_sha256: str | None = None,
    prediction_arrays_sha256: str | None = None,
):
    decision = build_diagnostic_decision_seal(
        outer_target=outer_target,
        config_contract_hash=_sha("config")[:16],
        development_prediction_seal_hash=_sha("development-seal")[:16],
        development_label_vector_hash=_sha("development-labels")[:16],
        development_metrics_sha256=_sha("development-metrics"),
        action_summaries_sha256=_sha("action-summaries"),
        selected_action_id=selected_action_id,
        selected_mean_paired_bacc_delta_vs_control=0.01,
        fallback_applied=False,
    )
    cells = tuple(
        _prediction_cell(
            phase="target",
            outer_target=outer_target,
            query_center=outer_target,
            action_id=(
                selected_action_id if arm_role == "selected" else CONTROL_ACTION_ID
            ),
            arm_role=arm_role,
            training_seed=training_seed,
            generation_seed=generation_seed,
            rows=rows,
        )
        for arm_role in ("selected", "control")
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    return build_target_prediction_seal(
        outer_target=outer_target,
        config_contract_hash=_sha("config")[:16],
        diagnostic_decision_hash=decision.decision_hash,
        selected_action_id=selected_action_id,
        validation_cache_binding_hash=_sha("cache")[:16],
        validation_manifest_sha256=manifest_sha256,
        prediction_index_sha256=prediction_index_sha256 or _sha("target-index"),
        prediction_arrays_sha256=prediction_arrays_sha256 or _sha("target-arrays"),
        evaluation_rows=rows,
        cells=cells,
    )


def _all_action_target_seal(
    rows_by_target: dict[str, tuple[ValidationRowIdentity, ...]],
    *,
    manifest_sha256: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
) -> AllActionTargetPredictionSeal:
    arms = tuple(("selected", action) for action in ACTION_IDS) + (
        ("control", CONTROL_ACTION_ID),
    )
    cells = tuple(
        _prediction_cell(
            phase="target",
            outer_target=target,
            query_center=target,
            action_id=action_id,
            arm_role=arm_role,
            training_seed=training_seed,
            generation_seed=generation_seed,
            rows=rows_by_target[target],
        )
        for target in CENTERS
        for arm_role, action_id in arms
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    return build_all_action_target_prediction_seal(
        config_contract_hash=_sha("config")[:16],
        support_partition_lock_hash=_sha("partition")[:16],
        compatibility_index_hash=_sha("compatibility")[:16],
        validation_cache_binding_hash=_sha("cache")[:16],
        validation_manifest_sha256=manifest_sha256,
        prediction_index_sha256=prediction_index_sha256,
        prediction_arrays_sha256=prediction_arrays_sha256,
        evaluation_rows_by_target=rows_by_target,
        cells=cells,
    )


def test_config_freezes_consumed_diagnostic_aliases_and_nonadoptive_scope() -> None:
    config = load_dense_residual_diagnostic_config(CONFIG_PATH)

    assert config.expert_bank_artifact_id.endswith("routing_authorized_expert_bank_v1")
    assert config.validation_cache_artifact_id == VALIDATION_CACHE_ARTIFACT_ID
    assert config.validation_manifest_artifact_id == VALIDATION_MANIFEST_ARTIFACT_ID
    assert config.input_artifact_ids[2:] == (
        VALIDATION_CACHE_ARTIFACT_ID,
        VALIDATION_MANIFEST_ARTIFACT_ID,
    )
    assert config.compatibility["target_support_labels_used"] is False
    assert config.compatibility["query_rows_consumed"] == (
        "support_partition_rows_only"
    )
    assert config.compatibility["query_evaluation_embeddings_consumed"] is False
    assert config.compatibility["exact_nelbo_claimed"] is False
    assert config.router["rhos"] == [0.0, 0.25, 0.5]
    assert config.router["minimum_effective_source_count"] == 6.0
    assert config.router["minimum_integer_allocation_per_source"] == 1
    assert config.selection["upper_quartile_cvar_definition"] == (
        "mean_of_largest_ceil_25_percent_regrets"
    )
    assert config.claim_boundary == canonical_claim_boundary_payload()
    assert config.claim_boundary["publication_status"] == PUBLICATION_STATUS


def test_config_rejects_original_stage60_validation_alias(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["inputs"][
        "validation_cache_artifact_id"
    ] = ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ProtocolError, match="rejects original Stage-60"):
        load_dense_residual_diagnostic_config(changed)

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["inputs"]["validation_cache_root"] = (
        f"artifact://{ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID}"
    )
    changed_root = tmp_path / "changed-root.yaml"
    changed_root.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="exact diagnostic alias"):
        load_dense_residual_diagnostic_config(changed_root)


def test_action_library_includes_exact_uniform_and_dense_constraints() -> None:
    actions = action_library()

    assert tuple(action.action_id for action in actions) == ACTION_IDS
    assert actions[0].exact_equal_union is True
    assert [action.rho for action in actions] == [0.0, 0.25, 0.5]
    assert all(action.max_source_weight == 0.25 for action in actions)
    assert all(action.min_effective_source_count == 6.0 for action in actions)
    assert all(action.minimum_integer_allocation_per_source == 1 for action in actions)
    assert ACTION_LIBRARY_HASH == stable_hash(
        [action.to_payload() for action in actions]
    )


def test_development_seal_requires_every_action_query_and_seed_cell() -> None:
    rows_by_query = {
        query: (
            _row(ordinal=index * 2, manifest_index=index * 2, center=query, label=0),
            _row(
                ordinal=index * 2 + 1,
                manifest_index=index * 2 + 1,
                center=query,
                label=1,
            ),
        )
        for index, query in enumerate(development_queries("0"))
    }
    complete = _development_seal(rows_by_query)
    assert complete.cell_count == 3 * 8 * 9

    with pytest.raises(ProtocolError, match="complete all-action coverage"):
        build_development_prediction_seal(
            outer_target="0",
            config_contract_hash=_sha("config")[:16],
            support_partition_lock_hash=_sha("partition")[:16],
            validation_cache_binding_hash=_sha("cache")[:16],
            validation_manifest_sha256=_sha("manifest-placeholder"),
            prediction_index_sha256=_sha("development-index"),
            prediction_arrays_sha256=_sha("development-arrays"),
            evaluation_rows_by_query=rows_by_query,
            cells=complete.cells[:-1],
        )


def test_global_target_seal_requires_every_target_action_and_seed_cell() -> None:
    rows_by_target = {
        target: (
            _row(
                ordinal=index,
                manifest_index=index,
                center=target,
                label=index % 2,
            ),
        )
        for index, target in enumerate(CENTERS)
    }
    complete = _all_action_target_seal(
        rows_by_target,
        manifest_sha256=_sha("manifest"),
        prediction_index_sha256=_sha("target-index"),
        prediction_arrays_sha256=_sha("target-arrays"),
    )
    assert complete.cell_count == 9 * 4 * 9
    assert complete.status == (
        "SEALED_ALL_TARGET_ACTION_PREDICTIONS_BEFORE_ANY_LABEL_ACCESS"
    )

    with pytest.raises(ProtocolError, match="complete all-target-action coverage"):
        build_all_action_target_prediction_seal(
            config_contract_hash=_sha("config")[:16],
            support_partition_lock_hash=_sha("partition")[:16],
            compatibility_index_hash=_sha("compatibility")[:16],
            validation_cache_binding_hash=_sha("cache")[:16],
            validation_manifest_sha256=_sha("manifest"),
            prediction_index_sha256=_sha("target-index"),
            prediction_arrays_sha256=_sha("target-arrays"),
            evaluation_rows_by_target=rows_by_target,
            cells=complete.cells[:-1],
        )


def test_label_capabilities_skip_support_and_nonrequested_rows_before_label(
    tmp_path: Path,
) -> None:
    lines = ["sample_id,case_id,center,split,label"]
    evaluation_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    ordinal = 0
    manifest_index = 0
    for center in CENTERS:
        support = _row(
            ordinal=ordinal,
            manifest_index=manifest_index,
            center=center,
            label=9,
            role="support",
        )
        lines.append(
            f"{support.sample_id},{support.case_id},{center},val,NOT_A_LABEL"
        )
        ordinal += 1
        manifest_index += 1
        rows: list[ValidationRowIdentity] = []
        for label in (0, 1):
            row = _row(
                ordinal=ordinal,
                manifest_index=manifest_index,
                center=center,
                label=label,
            )
            rows.append(row)
            lines.append(f"{row.sample_id},{row.case_id},{center},val,{label}")
            ordinal += 1
            manifest_index += 1
        evaluation_by_center[center] = tuple(rows)
    lines.extend(
        [
            "train-bad,case-train,0,train,NOT_A_LABEL",
            "test-bad,case-test,0,test,NOT_A_LABEL",
            "excluded-bad,case-4,4,val,NOT_A_LABEL",
        ]
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    development_index = tmp_path / "development_index.csv"
    development_arrays = tmp_path / "development_predictions.npz"
    target_index = tmp_path / "target_index.csv"
    target_arrays = tmp_path / "target_predictions.npz"
    development_index.write_bytes(b"sealed development index\n")
    development_arrays.write_bytes(b"sealed development arrays\n")
    target_index.write_bytes(b"sealed target index\n")
    target_arrays.write_bytes(b"sealed target arrays\n")

    rows_by_query = {
        query: evaluation_by_center[query] for query in development_queries("0")
    }
    development = _development_seal(
        rows_by_query,
        manifest_sha256=digest,
        prediction_index_sha256=hashlib.sha256(
            development_index.read_bytes()
        ).hexdigest(),
        prediction_arrays_sha256=hashlib.sha256(
            development_arrays.read_bytes()
        ).hexdigest(),
    )
    all_action_target = _all_action_target_seal(
        evaluation_by_center,
        manifest_sha256=digest,
        prediction_index_sha256=hashlib.sha256(
            target_index.read_bytes()
        ).hexdigest(),
        prediction_arrays_sha256=hashlib.sha256(
            target_arrays.read_bytes()
        ).hexdigest(),
    )
    all_action_target_path = tmp_path / "all_action_target_seal.json"
    all_action_target_path.write_text(
        json.dumps(all_action_target.to_payload(), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolError, match="all-action prediction seal"):
        open_development_labels(  # type: ignore[arg-type]
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=None,
            all_action_target_seal=all_action_target,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )

    with pytest.raises(ProtocolError, match="pre-label all-target-action"):
        open_development_labels(  # type: ignore[arg-type]
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=None,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )
    with pytest.raises(ProtocolError, match="pre-label all-target-action"):
        open_development_labels(  # type: ignore[arg-type]
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=development,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )

    development_labels = open_development_labels(
        manifest,
        tuple(row for rows in rows_by_query.values() for row in rows),
        seal=development,
        all_action_target_seal=all_action_target,
        all_action_target_seal_path=all_action_target_path,
        prediction_index_path=development_index,
        prediction_arrays_path=development_arrays,
        target_prediction_index_path=target_index,
        target_prediction_arrays_path=target_arrays,
        expected_manifest_sha256=digest,
    )
    assert development_labels.labels == (0, 1) * 8
    assert all(row.center != "0" for row in development_labels.rows)

    with pytest.raises(ProtocolError, match="not durably persisted"):
        open_development_labels(
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=all_action_target,
            all_action_target_seal_path=tmp_path / "missing_target_seal.json",
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )
    all_action_target_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="differs from its capability"):
        open_development_labels(
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=all_action_target,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )
    all_action_target_path.write_text(
        json.dumps(all_action_target.to_payload(), sort_keys=True),
        encoding="utf-8",
    )

    target_arrays.write_bytes(b"tampered target arrays\n")
    with pytest.raises(ProtocolError, match="persisted prediction bytes drifted"):
        open_development_labels(
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=all_action_target,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )
    target_arrays.write_bytes(b"sealed target arrays\n")

    original_global_hash = all_action_target.seal_hash
    object.__setattr__(all_action_target, "seal_hash", _sha("tampered-global"))
    with pytest.raises(ProtocolError, match="seal hash drifted"):
        open_development_labels(
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=all_action_target,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )
    object.__setattr__(all_action_target, "seal_hash", original_global_hash)

    development_arrays.write_bytes(b"tampered development arrays\n")
    with pytest.raises(ProtocolError, match="persisted prediction bytes drifted"):
        open_development_labels(
            manifest,
            tuple(row for rows in rows_by_query.values() for row in rows),
            seal=development,
            all_action_target_seal=all_action_target,
            all_action_target_seal_path=all_action_target_path,
            prediction_index_path=development_index,
            prediction_arrays_path=development_arrays,
            target_prediction_index_path=target_index,
            target_prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )

    target_rows = evaluation_by_center["0"]
    target = _target_seal(
        target_rows,
        manifest_sha256=digest,
        prediction_index_sha256=hashlib.sha256(target_index.read_bytes()).hexdigest(),
        prediction_arrays_sha256=hashlib.sha256(target_arrays.read_bytes()).hexdigest(),
    )
    with pytest.raises(ProtocolError, match="selected-plus-control"):
        open_target_labels(  # type: ignore[arg-type]
            manifest,
            target_rows,
            seal=development,
            prediction_index_path=target_index,
            prediction_arrays_path=target_arrays,
            expected_manifest_sha256=digest,
        )
    target_labels = open_target_labels(
        manifest,
        target_rows,
        seal=target,
        prediction_index_path=target_index,
        prediction_arrays_path=target_arrays,
        expected_manifest_sha256=digest,
    )
    assert target_labels.labels == (0, 1)
    assert target_labels.phase == "target"


def test_bundle_is_closed_world_and_rejects_adoptive_claim_mutation() -> None:
    required = set(REQUIRED_FILES)
    assert "arrays/development_predictions.npz" in required
    assert "arrays/target_predictions.npz" in required
    assert "manifests/development_prediction_seals.json" in required
    assert "manifests/diagnostic_decision_seals.json" in required
    assert "manifests/target_prediction_seals.json" in required
    assert "reports/validation_report.json" in required

    decision = publication_decision_payload(
        descriptive_summary_hash=_sha("descriptive-summary")[:16]
    )
    assert decision["decision"] == PUBLICATION_STATUS
    assert decision["fresh_evidence"] is False
    assert decision["may_feed_stage60"] is False
    assert decision["may_feed_stage70"] is False
    assert decision["may_feed_deployable_selection"] is False
    assert_non_adoptive_payload(decision)

    changed = dict(decision)
    changed["may_feed_deployable_selection"] = True
    with pytest.raises(ProtocolError, match="adoptive or fresh-evidence"):
        assert_non_adoptive_payload(changed)


def test_public_experiment_identity_is_stage90_oracle_diagnostic() -> None:
    assert EXPERIMENT_ID == (
        "midogpp.oracle."
        "uniform_b_v2_consumed_validation_dense_residual_router.v1"
    )
