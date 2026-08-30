from __future__ import annotations

import ast
from dataclasses import replace
import csv
import hashlib
import inspect
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from midogpp_thesis.cvae.diagnostics.cli import build_parser
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import authorization
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import (
    physical_menu as physical_menu_module,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1 import preparation
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.config import (
    INPUT_ARTIFACT_IDS,
    load_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.input_surfaces import (
    DEVELOPMENT_ROLE,
    EVALUATION_ROLE,
    HarpCacheRow,
    HarpConsumedCacheIndex,
    _read_label_manifest,
    load_cache_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.modeling import (
    select_and_route,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.physical_menu import (
    EXPECTED_ACTION_COUNT,
    EXPECTED_CELL_COUNT,
    EXPECTED_CLASSIFIER_TASK_COUNT,
    _load_authoritative_inputs,
    _persist_lineage_receipt,
    build_physical_plan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.runner import (
    _write_content_index,
    dry_run_harp_stage90,
    inspect_harp_stage90,
    run_harp_stage90,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.terminal_diagnostics import (
    build_terminal_action_diagnostics,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_model import (
    LAMBDA_GRID,
    HarpActionScore,
    HarpSupportCell,
    HarpTargetAction,
)
from midogpp_thesis.cvae.routing.harp_portfolio import (
    HarpPolicyConfig,
    select_harp_portfolio,
)
from midogpp_thesis.cvae.routing.harp_replay.evaluation import (
    _case_equal_balanced_accuracy,
)
from midogpp_thesis.cvae.runtime.harp_probability_menu import (
    EXACT_NINE_SEED_PAIRS,
    TARGET_SURFACE,
    HarpPredictionCell,
    HarpRouteDecision,
    build_all_development_actions,
    build_all_target_actions,
    route_harp_probability_vector,
    seal_harp_prediction_menu,
)
from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = (
    ROOT
    / "src/midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v1"
)
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/90_oracles_and_diagnostics/configs"
    / "uniform_b_v2_consumed_test_fixed_bank_harp_router_v1.yaml"
)


def _cache(rows: tuple[HarpCacheRow, ...], root: Path) -> HarpConsumedCacheIndex:
    return HarpConsumedCacheIndex(
        root=root,
        rows=rows,
        shards={},
        member_sha256={},
        content_sha256="a" * 64,
        cache_hash="b" * 64,
    )


def _all_role_rows(role: str) -> tuple[HarpCacheRow, ...]:
    rows: list[HarpCacheRow] = []
    for center in CENTERS:
        for ordinal in range(2):
            rows.append(
                HarpCacheRow(
                    center=center,
                    case_id=f"case-{center}",
                    sample_id=f"sample-{center}-{ordinal}",
                    split_role=role,
                    split_row_index=ordinal,
                    embedding_file="unused.npy",
                    embedding_row_index=ordinal,
                )
            )
    return tuple(rows)


def _write_role_manifest(
    path: Path, rows: tuple[HarpCacheRow, ...], *, role: str, extra_role: str | None = None
) -> str:
    lines = ["center,case_id,sample_id,label,split_role"]
    for row in rows:
        label = int(row.split_row_index % 2)
        lines.append(
            f"{row.center},{row.case_id},{row.sample_id},{label},{role}"
        )
    if extra_role is not None:
        lines.append(f"0,foreign-case,foreign-sample,0,{extra_role}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_physical_plan_is_complete_exact_nine_inventory() -> None:
    plan = build_physical_plan()
    assert plan["action_count"] == EXPECTED_ACTION_COUNT == 738
    assert plan["exact_nine_cell_count"] == EXPECTED_CELL_COUNT == 6642
    assert plan["classifier_task_count"] == EXPECTED_CLASSIFIER_TASK_COUNT == 729
    assert plan["development_action_count"] == 648
    assert plan["target_action_count"] == 90
    assert plan["matched_budget_reference_action"] == "U"
    assert plan["operational_fallback_action"] == "B"
    assert len(build_all_development_actions()) + len(build_all_target_actions()) == 738


def test_outer_H_query_candidate_and_model_donor_fences_are_explicit() -> None:
    actions = build_all_development_actions()
    assert len(actions) == 648
    for action in actions:
        assert action.outer_target_id != action.query_center_id
        assert action.outer_target_id not in action.source_order
        assert action.query_center_id not in action.source_order
        if action.selected_source_id is not None:
            assert action.selected_source_id in action.source_order
            assert action.selected_source_id not in {
                action.outer_target_id,
                action.query_center_id,
            }
    model_source = (PACKAGE / "modeling.py").read_text(encoding="utf-8")
    assert "heldout_donor_id in audit.training_query_ids" in model_source
    assert "heldout_donor_id in audit.training_source_ids" in model_source
    assert "donor in model.training_query_ids" in model_source
    assert "donor in model.training_source_ids" in model_source


def test_mixed_patch_labels_within_midog_case_are_accepted(tmp_path: Path) -> None:
    rows = _all_role_rows(DEVELOPMENT_ROLE)
    cache = _cache(rows, tmp_path)
    manifest = tmp_path / "development.csv"
    digest = _write_role_manifest(manifest, rows, role=DEVELOPMENT_ROLE)
    opened = _read_label_manifest(
        manifest,
        expected_sha256=digest,
        expected_role=DEVELOPMENT_ROLE,
        cache=cache,
    )
    assert len(opened) == 2 * len(CENTERS)
    assert {
        label for center, case, _sample, label in opened if center == "0" and case == "case-0"
    } == {0, 1}


def test_role_scoped_capability_rejects_file_containing_other_role(tmp_path: Path) -> None:
    rows = _all_role_rows(DEVELOPMENT_ROLE)
    cache = _cache(rows, tmp_path)
    manifest = tmp_path / "mixed.csv"
    digest = _write_role_manifest(
        manifest,
        rows,
        role=DEVELOPMENT_ROLE,
        extra_role=EVALUATION_ROLE,
    )
    with pytest.raises(ProtocolError, match="another split role"):
        _read_label_manifest(
            manifest,
            expected_sha256=digest,
            expected_role=DEVELOPMENT_ROLE,
            cache=cache,
        )


def test_case_equal_bacc_uses_per_case_class_denominators() -> None:
    truth = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int64)
    probability = np.asarray([0.1, 0.8, 0.2, 0.8, 0.1, 0.9], dtype=np.float64)
    cases = np.asarray(["A", "A", "A", "B", "B", "B"], dtype=str)
    # class 0: mean(case A=1, case B=1/2)=3/4;
    # class 1: mean(case A=1/2, case B=1)=3/4.
    assert _case_equal_balanced_accuracy(truth, probability, cases) == pytest.approx(0.75)


def test_exact_b_fallback_preserves_the_same_bytes_object() -> None:
    predictive_reference = struct.pack("<d", 0.37)
    exact_b_fallback = struct.pack("<d", 0.29)
    scores = []
    for source in ("0", "1"):
        for lam in LAMBDA_GRID:
            action = HarpTargetAction(
                outer_target_id="H",
                target_query_id="H",
                candidate_source_id=source,
                case_id="case",
                sample_id="sample",
                lambda_value=lam,
                direction="ALL_MARGINS",
                feature_names=("margin", "seed_dispersion"),
                feature_values=(0.1, 0.02),
                baseline_probability_bytes=predictive_reference,
                operational_fallback_probability_bytes=exact_b_fallback,
                expert_probability=0.8,
                ensemble_size=9,
                ensemble_receipt_hash="e" * 64,
                prediction_seal_hash="f" * 64,
            )
            scores.append(
                HarpActionScore(
                    action,
                    (-0.1,) * 4,
                    (0.1,) * 4,
                    (0.1,) * 4,
                    (0.1,) * 4,
                    HarpSupportCell(source, lam, "ALL_MARGINS", 4, 16, (0, 1)),
                    ("0", "1", "2", "3"),
                )
            )
    decision = select_harp_portfolio(scores)[0]
    assert not decision.routed
    assert decision.output_probability_bytes == exact_b_fallback
    assert decision.output_probability_bytes != predictive_reference
    assert decision.output_probability_bytes is decision.baseline_probability_bytes


def test_inspection_and_planned_dry_run_are_mutation_free(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    destination = tmp_path / "must-not-exist"
    inspection = inspect_harp_stage90(config)
    dry = dry_run_harp_stage90(config, artifact_root=destination)
    assert inspection["status"] == "PLANNED_NEEDS_NEW_EXECUTION_AMENDMENT"
    assert inspection["physical_plan"]["exact_nine_cell_count"] == 6642
    assert dry["status"] == "NEEDS_EXECUTION_AMENDMENT"
    assert inspection["filesystem_mutations"] == dry["filesystem_mutations"] == 0
    assert not destination.exists()


def test_terminal_action_diagnostics_score_full_sealed_matrix_without_feedback() -> None:
    actions = build_all_target_actions()
    cells = []
    for action in actions:
        if action.is_exact_b:
            probability = np.asarray([0.20, 0.80], dtype=np.float32)
        elif action.is_uniform_topup:
            probability = np.asarray([0.30, 0.70], dtype=np.float32)
        else:
            probability = np.asarray([0.10, 0.90], dtype=np.float32)
        for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS:
            cells.append(
                HarpPredictionCell(
                    action=action,
                    training_seed=training_seed,
                    generation_seed=generation_seed,
                    row_ids=(f"row-{action.outer_target_id}-0", f"row-{action.outer_target_id}-1"),
                    case_ids=(f"case-{action.outer_target_id}",) * 2,
                    probabilities=probability,
                    bank_hash="a" * 64,
                    generation_lock_hash="b" * 64,
                    source_cache_hash="c" * 64,
                    frame_hash="d" * 64,
                    classifier_hash="e" * 64,
                    composition_hash=action.action_hash,
                    scaler_state_hash="f" * 64,
                )
            )
    menu = seal_harp_prediction_menu(actions, cells)
    vectors = []
    physical_vectors = []
    truth = {}
    for center in CENTERS:
        source = next(value for value in CENTERS if value != center)
        decisions = (
            HarpRouteDecision(
                surface_kind=TARGET_SURFACE,
                outer_target_id=center,
                query_center_id=center,
                row_id=f"row-{center}-0",
                case_id=f"case-{center}",
                eligible=False,
                selected_source_id=None,
                lambda_value=0.0,
                direction="NO_DISAGREEMENT",
                decision_reason="exact_b_fallback",
                policy_hash="1" * 64,
                prediction_menu_seal_hash=menu.seal_hash,
            ),
            HarpRouteDecision(
                surface_kind=TARGET_SURFACE,
                outer_target_id=center,
                query_center_id=center,
                row_id=f"row-{center}-1",
                case_id=f"case-{center}",
                eligible=True,
                selected_source_id=source,
                lambda_value=0.5,
                direction="ALL_MARGINS",
                decision_reason="conservative_action_admitted",
                policy_hash="1" * 64,
                prediction_menu_seal_hash=menu.seal_hash,
            ),
        )
        vectors.append(route_harp_probability_vector(menu, decisions))
        physical_vectors.append(
            route_harp_probability_vector(
                menu,
                (
                    decisions[0],
                    replace(decisions[1], lambda_value=1.0),
                ),
            )
        )
        truth[(center, f"case-{center}", f"row-{center}-0")] = 0
        truth[(center, f"case-{center}", f"row-{center}-1")] = 1
    diagnostics = build_terminal_action_diagnostics(
        menu,
        vectors,
        physical_vectors,
        truth,
        prelabel_bundle_hash="2" * 64,
        physical_reference_preserving_surface_hash="3" * 64,
    )
    assert diagnostics["action_matrix_row_count"] == 9 * 34 == 306
    assert len(diagnostics["center_diagnostics"]) == 9
    assert all(row["action_count"] == 34 for row in diagnostics["center_diagnostics"])
    assert all(
        row["physical_matched_action_count"] == 9
        for row in diagnostics["center_diagnostics"]
    )
    assert diagnostics["labels_available_to_policy"] is False
    assert diagnostics["policy_or_threshold_update_emitted"] is False


def test_missing_authority_fails_before_output_creation(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    destination = tmp_path / "output"
    with pytest.raises(ProtocolError, match="new HARP-specific"):
        run_harp_stage90(config, artifact_root=destination)
    assert not destination.exists()


def _authorized_config(tmp_path: Path):
    base = load_config(CONFIG)
    expected_hashes = {
        **dict(base.expected_hashes),
        "test_cache_content_sha256": "a" * 64,
        "development_manifest_sha256": "b" * 64,
        "evaluation_manifest_sha256": "c" * 64,
        "parent_ledger_sha256": "d" * 64,
    }
    binding_config = replace(
        base,
        execution_authorized=True,
        expected_hashes=expected_hashes,
    )
    amendment = authorization.canonical_execution_amendment_payload(binding_config)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return replace(
        base,
        artifact_root=str(tmp_path / "prepared-output"),
        execution_authorized=True,
        input_locations={**dict(base.input_locations), "execution_amendment_path": str(path)},
        expected_hashes={**expected_hashes, "execution_amendment_sha256": digest},
    )


def test_exhausted_authority_fails_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _authorized_config(tmp_path)
    lease = tmp_path / "already-consumed-lease"
    lease.mkdir()
    monkeypatch.setattr(authorization, "lease_path", lambda *_args: lease)
    destination = Path(config.artifact_root)
    with pytest.raises(ProtocolError, match="authorization is exhausted"):
        run_harp_stage90(config, artifact_root=destination)
    assert not destination.exists()


def test_authority_is_bound_to_exact_prepared_input_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _authorized_config(tmp_path)
    drifted = replace(
        config,
        expected_hashes={
            **dict(config.expected_hashes),
            "development_manifest_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        authorization,
        "lease_path",
        lambda *_args: tmp_path / "unused-lease",
    )
    with pytest.raises(ProtocolError, match="failed authentication"):
        authorization.load_authorization(drifted)


def test_missing_cache_and_tampered_authoritative_lineage_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = SimpleNamespace(
        resolved_path=lambda _role: tmp_path / "missing",
        expected_hashes={"test_cache_content_sha256": "a" * 64},
    )
    with pytest.raises(ProtocolError):
        load_cache_index(missing)  # type: ignore[arg-type]

    bank_root = tmp_path / "bank"
    generation_root = tmp_path / "generation"
    for root in (bank_root, generation_root):
        (root / "reports").mkdir(parents=True)
        (root / "manifests").mkdir(parents=True)
        (root / "reports/run_state.json").write_text(
            json.dumps({"status": "COMPLETE"}), encoding="utf-8"
        )
        (root / "reports/validation_report.json").write_text(
            json.dumps({"status": "PASS"}), encoding="utf-8"
        )
    bank_path = bank_root / "manifests/expert_bank_index.json"
    bank_path.write_text(json.dumps({"bank_lock_hash": "1" * 16}), encoding="utf-8")
    generation_path = generation_root / "manifests/generation_lock.json"
    generation_path.write_text("{}", encoding="utf-8")
    bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    spec = ClassifierSpec()
    classifier = {
        **spec.to_payload(),
        "config_hash": spec.config_hash,
        "scaler_family": "sklearn.preprocessing.StandardScaler",
        "fit_in_stage_40": False,
    }

    class _Lock:
        bank_lock_hash = "1" * 16
        generation_lock_hash = "2" * 16

        def to_payload(self):
            return {
                "bank": {"bank_index_sha256": bank_sha},
                "classifier": classifier,
            }

    monkeypatch.setattr(
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v1.physical_menu.read_generation_lock",
        lambda _path: _Lock(),
    )
    fake_config = SimpleNamespace(
        resolved_path=lambda role: {
            "expert_bank_root": bank_root,
            "generation_lock_root": generation_root,
        }[role],
        expected_hashes={
            "expert_bank_lock_hash": "1" * 16,
            "generation_lock_hash": "2" * 16,
        },
    )
    cache = _cache(
        tuple(
            HarpCacheRow(
                center=center,
                case_id=f"case-{role}-{center}",
                sample_id=f"sample-{role}-{center}",
                split_role=role,
                split_row_index=0,
                embedding_file="unused.npy",
                embedding_row_index=0,
            )
            for role in (DEVELOPMENT_ROLE, EVALUATION_ROLE)
            for center in CENTERS
        ),
        tmp_path,
    )
    bank_path.write_text(
        json.dumps({"bank_lock_hash": "1" * 16, "tampered": True}),
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="generation lineage drifted"):
        _load_authoritative_inputs(fake_config, cache)  # type: ignore[arg-type]

    bank_path.write_text(json.dumps({"bank_lock_hash": "1" * 16}), encoding="utf-8")
    fake_config.expected_hashes["generation_lock_hash"] = "3" * 16
    with pytest.raises(ProtocolError, match="generation lineage drifted"):
        _load_authoritative_inputs(fake_config, cache)  # type: ignore[arg-type]


def test_persisted_source_lock_tamper_fails_lineage_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "workstation/source_streams/manifests"
    source_root.mkdir(parents=True)
    source_lock = source_root / "frozen_source_stream_lock.json"
    source_index = source_root / "frozen_source_stream_index.json"
    source_lock.write_text('{"status":"sealed"}\n', encoding="utf-8")
    source_index.write_text('{"rows":[]}\n', encoding="utf-8")
    cache = SimpleNamespace(
        lock_hash="a" * 64,
        lock_payload={
            "source_array_sha256": "b" * 64,
            "source_stream_index_sha256": hashlib.sha256(
                source_index.read_bytes()
            ).hexdigest(),
            "config_contract_hash": "c" * 64,
            "generation_lock_hash": "d" * 64,
        },
        records=(),
    )

    def tampering_reload(*_args, **_kwargs):
        source_lock.write_text('{"status":"tampered"}\n', encoding="utf-8")
        return SimpleNamespace(lock_payload=cache.lock_payload, records=cache.records)

    monkeypatch.setattr(
        physical_menu_module, "load_frozen_source_streams", tampering_reload
    )
    with pytest.raises(ProtocolError, match="lineage receipt"):
        _persist_lineage_receipt(
            tmp_path,
            SimpleNamespace(to_payload=lambda: {}),
            cache,  # type: ignore[arg-type]
            SimpleNamespace(source_content_hash="e" * 64, receipt_hash="f" * 64),
        )


def test_stage90_rejects_an_incomplete_legal_candidate_universe() -> None:
    banks = tuple(SimpleNamespace(outer_target_id=center) for center in CENTERS)
    incomplete = (
        SimpleNamespace(
            outer_target_id=CENTERS[0],
            candidate_source_id=CENTERS[1],
            case_id="case",
            sample_id="sample",
            lambda_value=LAMBDA_GRID[0],
        ),
    )
    with pytest.raises(ProtocolError, match="complete legal candidate universe"):
        select_and_route(  # type: ignore[arg-type]
            SimpleNamespace(),
            banks,  # type: ignore[arg-type]
            incomplete,  # type: ignore[arg-type]
            policy=HarpPolicyConfig(),
            fitted_policy_hash="a" * 64,
        )


def test_source_menu_and_two_route_reconstructions_precede_label_capabilities() -> None:
    source = inspect.getsource(run_harp_stage90)
    assert source.index("materialize_physical_harp_menu") < source.index(
        "load_development_labels"
    )
    assert source.index("_fsync_tree(root)") < source.index(
        "load_development_labels"
    )
    assert source.count("_validate_route_reconstruction(") == 2
    assert source.index("_validate_route_reconstruction(") < source.index(
        "load_evaluation_truth"
    )
    assert source.index("prelabel_route_bundle.json") < source.index(
        "load_evaluation_truth"
    )
    last_prelabel_fsync = source.rfind(
        "_fsync_tree(root)", 0, source.index("load_evaluation_truth")
    )
    assert source.index("prelabel_route_bundle.json") < last_prelabel_fsync
    assert last_prelabel_fsync < source.index("load_evaluation_truth")


def test_final_commit_follows_authorization_and_content_index(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir(parents=True)
    (tmp_path / "manifests").mkdir(parents=True)
    (tmp_path / "reports/validation_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "reports/run_state.json").write_text("{}", encoding="utf-8")
    _write_content_index(tmp_path)
    content = json.loads(
        (tmp_path / "manifests/content_index.json").read_text(encoding="utf-8")
    )
    members = {row["path"] for row in content["members"]}
    assert "reports/validation_report.json" in members
    assert "reports/run_state.json" not in members
    assert content["run_state_excluded_as_final_commit"] is True
    source = inspect.getsource(run_harp_stage90)
    success = source[source.index("# The global lease is finalized") : source.index("return str(root)")]
    assert success.index("finalize_authorization") < success.index("_write_content_index")
    assert success.index("_write_content_index") < success.index('root / "reports/run_state.json"')


def test_cli_registry_and_catalog_are_planned_terminal_and_nonfeeding() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["fixed-bank-harp-router-v1", "--config", str(CONFIG), "--inspect-plan"]
    )
    assert args.surface == "fixed-bank-harp-router-v1"
    registry = yaml.safe_load(
        (ROOT / "experiments/midogpp/registry.yaml").read_text(encoding="utf-8")
    )
    catalog = yaml.safe_load(
        (ROOT / "experiments/midogpp/artifact_catalog.yaml").read_text(encoding="utf-8")
    )
    experiment = next(
        row for row in registry["experiments"] if row["experiment_id"] == EXPERIMENT_ID
    )
    assert experiment["status"] == "planned"
    assert tuple(experiment["input_artifact_ids"]) == INPUT_ARTIFACT_IDS
    assert "fixed-bank-harp-router-v1" in experiment["runner"]["argv"]
    output = next(
        row
        for row in catalog["artifacts"]
        if row["artifact_id"] == experiment["output_artifact_id"]
    )
    identities = output["semantic_identities"]
    assert output["availability"] == "planned"
    assert identities["publication_status"] == PUBLICATION_STATUS
    assert identities["terminal_decision"] == TERMINAL_DECISION
    assert identities["fresh_evidence"] == "false"
    assert identities["may_feed_stage60"] == "false"
    assert identities["may_feed_stage70"] == "false"
    assert identities["may_feed_another_experiment"] == "false"


def _tiny_preparation_fixture(tmp_path: Path):
    raw_rows: list[dict[str, str]] = []
    row_specs: list[tuple[str, str, int, int]] = []
    contract_index = 0
    for center in CENTERS:
        center_index = 0
        for case_ordinal in range(4):
            case = f"case-{center}-{case_ordinal}"
            for label in (0, 1):
                raw_rows.append(
                    {
                        "case_id": case,
                        "center": center,
                        "split": "test",
                        "label": str(label),
                    }
                )
                row_specs.append((center, case, contract_index, center_index))
                contract_index += 1
                center_index += 1
    manifest = tmp_path / "canonical.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("case_id", "center", "split", "label")
        )
        writer.writeheader()
        writer.writerows(raw_rows)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows_by_center: dict[str, tuple[preparation.CanonicalFrameRow, ...]] = {}
    embeddings_by_center: dict[str, np.ndarray] = {}
    for center in CENTERS:
        rows = tuple(
            preparation.CanonicalFrameRow(
                center=center,
                case_id=case,
                sample_id=preparation._evaluation_row_id(manifest_sha, global_index),
                contract_row_index=global_index,
                center_row_index=center_index,
            )
            for center_, case, global_index, center_index in row_specs
            if center_ == center
        )
        rows_by_center[center] = rows
        values = np.arange(
            len(rows) * 3840, dtype=np.float32
        ).reshape(len(rows), 3840)
        embeddings_by_center[center] = values + float(int(center))
    frame = preparation.CanonicalLabelBlindFrame(
        rows_by_center=rows_by_center,
        embeddings_by_center=embeddings_by_center,
        cache_content_hash="c" * 64,
        row_order_hash="d" * 64,
        source_member_sha256={},
    )
    ledger = tmp_path / "parent_ledger.json"
    ledger.write_text("{}\n", encoding="utf-8")
    ledger_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    return frame, manifest, manifest_sha, ledger, ledger_sha


def test_preparation_is_label_blind_whole_case_and_manifest_open_is_after_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, manifest, manifest_sha, ledger, ledger_sha = _tiny_preparation_fixture(
        tmp_path
    )
    monkeypatch.setattr(preparation, "EXPECTED_ROW_COUNT", 72)
    events: list[str] = []

    def load_frame(_root: Path):
        events.append("label_blind_cache_loaded")
        return frame

    original_validate = preparation._independently_validate_label_blind_barrier
    original_publish = preparation._publish_role_pure_manifests

    def validate_barrier(root: Path, *, expected_partition_hash: str):
        result = original_validate(
            root, expected_partition_hash=expected_partition_hash
        )
        events.append("label_free_barrier_validated")
        return result

    def publish_manifest(path: Path, **kwargs):
        cache_root = kwargs["cache"].root
        assert events == [
            "label_blind_cache_loaded",
            "label_free_barrier_validated",
        ]
        assert (cache_root / preparation.CASE_PARTITION).is_file()
        assert (cache_root / preparation.LABEL_FREE_BARRIER).is_file()
        assert (cache_root / preparation.LABEL_FREE_CONTENT_INDEX).is_file()
        assert not (cache_root / preparation.PREPARATION_RECEIPT).exists()
        events.append("canonical_scoring_manifest_opened")
        return original_publish(path, **kwargs)

    monkeypatch.setattr(preparation, "load_canonical_label_blind_cache", load_frame)
    monkeypatch.setattr(
        preparation, "_independently_validate_label_blind_barrier", validate_barrier
    )
    monkeypatch.setattr(preparation, "_publish_role_pure_manifests", publish_manifest)
    cache_root = tmp_path / "prepared-cache"
    development = tmp_path / "development" / "manifest.csv"
    evaluation = tmp_path / "evaluation" / "manifest.csv"
    prepared = preparation.prepare_harp_consumed_test_inputs(
        canonical_cache_root=tmp_path / "unused-canonical-cache",
        canonical_manifest_path=manifest,
        parent_ledger_path=ledger,
        cache_root=cache_root,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        expected_manifest_sha256=manifest_sha,
        expected_parent_ledger_sha256=ledger_sha,
    )
    assert events[-1] == "canonical_scoring_manifest_opened"
    assert prepared.cache_content_sha256 == json.loads(
        (cache_root / "manifests/content_index.json").read_text(encoding="utf-8")
    )["content_index_hash"]
    partition = json.loads(
        (cache_root / preparation.CASE_PARTITION).read_text(encoding="utf-8")
    )
    assignments = {
        (row["center"], row["case_id"]): row["split_role"]
        for row in partition["assignments"]
    }
    assert assignments == preparation.deterministic_case_partition(
        {center: tuple(reversed(frame.rows_by_center[center])) for center in CENTERS}
    )
    development_cases = {
        key for key, role in assignments.items() if role == DEVELOPMENT_ROLE
    }
    evaluation_cases = {
        key for key, role in assignments.items() if role == EVALUATION_ROLE
    }
    assert development_cases.isdisjoint(evaluation_cases)
    receipt = json.loads(
        (cache_root / preparation.PREPARATION_RECEIPT).read_text(encoding="utf-8")
    )
    assert receipt["cache_fsynced_and_independently_validated_before_manifest_open"]
    assert receipt["mixed_patch_labels_within_case_supported"]
    assert receipt["execution_amendment_created"] is False
    assert receipt["execution_authorized"] is False
    assert not any("amendment" in path.name for path in tmp_path.rglob("*"))


def test_preparation_outputs_are_role_pure_and_final_cache_is_closed_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame, manifest, manifest_sha, ledger, ledger_sha = _tiny_preparation_fixture(
        tmp_path
    )
    monkeypatch.setattr(preparation, "EXPECTED_ROW_COUNT", 72)
    monkeypatch.setattr(
        preparation, "load_canonical_label_blind_cache", lambda _root: frame
    )
    cache_root = tmp_path / "cache"
    development = tmp_path / "dev.csv"
    evaluation = tmp_path / "eval.csv"
    prepared = preparation.prepare_harp_consumed_test_inputs(
        canonical_cache_root=tmp_path / "unused",
        canonical_manifest_path=manifest,
        parent_ledger_path=ledger,
        cache_root=cache_root,
        development_manifest_path=development,
        evaluation_manifest_path=evaluation,
        expected_manifest_sha256=manifest_sha,
        expected_parent_ledger_sha256=ledger_sha,
    )
    config = SimpleNamespace(
        resolved_path=lambda role: cache_root if role == "test_cache_root" else None,
        expected_hashes={"test_cache_content_sha256": prepared.cache_content_sha256},
    )
    cache = load_cache_index(config)  # type: ignore[arg-type]
    dev = _read_label_manifest(
        development,
        expected_sha256=prepared.development_manifest_sha256,
        expected_role=DEVELOPMENT_ROLE,
        cache=cache,
    )
    evl = _read_label_manifest(
        evaluation,
        expected_sha256=prepared.evaluation_manifest_sha256,
        expected_role=EVALUATION_ROLE,
        cache=cache,
    )
    assert {row.split_role for row in cache.rows} == {
        DEVELOPMENT_ROLE,
        EVALUATION_ROLE,
    }
    assert {label for *_key, label in dev} == {0, 1}
    assert {label for *_key, label in evl} == {0, 1}
    (cache_root / "unindexed.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        load_cache_index(config)  # type: ignore[arg-type]


def test_preparation_cli_and_harp_dry_run_parse_independently() -> None:
    parser = build_parser()
    prepared = parser.parse_args(
        [
            "prepare-fixed-bank-harp-router-v1-inputs",
            "--canonical-cache-root", "/canonical-cache",
            "--canonical-manifest", "/canonical.csv",
            "--parent-ledger", "/ledger.json",
            "--cache-root", "/prepared-cache",
            "--development-manifest", "/development.csv",
            "--evaluation-manifest", "/evaluation.csv",
        ]
    )
    assert prepared.surface == "prepare-fixed-bank-harp-router-v1-inputs"
    assert not hasattr(prepared, "dry_run")
    dry_run = parser.parse_args(
        ["fixed-bank-harp-router-v1", "--config", str(CONFIG), "--dry-run"]
    )
    assert dry_run.dry_run is True


def test_package_has_no_stage60_stage70_sceptre_or_old_utility_dependency() -> None:
    imported_modules: list[str] = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module.lower())
            elif isinstance(node, ast.Import):
                imported_modules.extend(alias.name.lower() for alias in node.names)
    assert not any("sceptre" in module for module in imported_modules)
    assert not any("harp_stage60" in module for module in imported_modules)
    assert not any("harp_fresh" in module for module in imported_modules)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rendered_inputs = json.dumps(config["inputs"], sort_keys=True).lower()
    assert "probability_menu" not in rendered_inputs
    assert "source_inner_candidate_utility" not in rendered_inputs
    assert "sceptre" not in rendered_inputs
