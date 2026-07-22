from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from midogpp_thesis.real_features.classifier_reference.artifacts import (
    stable_hash,
    write_csv_rows,
    write_json,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment import (
    ConditionalLogitAlignmentConfig,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment import (
    runner as cla_runner,
    selection as cla_selection,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.artifacts import (
    build_leakage_report,
    build_runtime_summary,
    write_content_index,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.estimator import (
    AlignmentFitResult,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.reporting import (
    build_decision_summary,
    render_decision_report,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.schema import (
    AlignmentArtifactTables,
    CLA_COMPLETE_REQUIRED_OUTPUTS,
    CLA_REQUIRED_OUTPUTS,
    CLA_RUNNER_REQUIRED_OUTPUTS,
    TABLE_COLUMNS,
    TABLE_PATHS,
    table_bundle_hash,
    table_hashes,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.validation import (
    assert_conditional_logit_alignment_artifacts,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


@pytest.fixture(scope="module")
def partial_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    base = tmp_path_factory.mktemp("midogpp_virchow2_cla")
    manifest, cache = _write_fixture(base)
    patch = pytest.MonkeyPatch()
    for key in cla_runner.THREAD_ENVIRONMENT_KEYS:
        patch.setenv(key, "1")
    patch.setattr(cla_selection, "fit_prepared_conditional_logit", _fast_fit)
    patch.setattr(cla_runner, "fit_prepared_conditional_logit", _fast_fit)
    try:
        root = cla_runner.run_conditional_logit_alignment(
            ConditionalLogitAlignmentConfig(
                name="conditional_logit_alignment_v1",
                artifact_root=base / "artifact",
                manifest_path=manifest,
                feature_cache_path=cache,
                heldout_centers=("0",),
                expected_feature_dim=4,
                allow_partial_test_coverage=True,
            )
        )
    finally:
        patch.undo()
    return root


def test_partial_runner_writes_and_independently_validates_14_file_bundle(
    partial_bundle: Path,
) -> None:
    assert_conditional_logit_alignment_artifacts(partial_bundle)
    files = {
        path.relative_to(partial_bundle).as_posix()
        for path in partial_bundle.rglob("*")
        if path.is_file()
    }
    assert files == set(CLA_RUNNER_REQUIRED_OUTPUTS)
    assert len(files) == 14
    assert len(_read_csv(partial_bundle / TABLE_PATHS["source_inner_fold_scores"])) == 21
    assert len(_read_csv(partial_bundle / TABLE_PATHS["source_inner_gamma_summary"])) == 7
    assert len(_read_csv(partial_bundle / TABLE_PATHS["outer_results"])) == 2
    assert len(_read_csv(partial_bundle / TABLE_PATHS["conditional_frame_audit"])) == 4
    # Every source-inner candidate ties, so gamma zero is selected and each
    # outer semantic pair shares exactly one physical solver fit.
    assert len(_read_csv(partial_bundle / TABLE_PATHS["solver_audit"])) == 22


def test_complete_and_partial_file_contracts_are_distinct() -> None:
    assert CLA_REQUIRED_OUTPUTS == CLA_COMPLETE_REQUIRED_OUTPUTS
    assert len(CLA_COMPLETE_REQUIRED_OUTPUTS) == 16
    assert len(CLA_RUNNER_REQUIRED_OUTPUTS) == 14
    assert set(CLA_COMPLETE_REQUIRED_OUTPUTS).difference(CLA_RUNNER_REQUIRED_OUTPUTS) == {
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
    }


def test_complete_bundle_rejects_missing_workspace_snapshots(
    partial_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle, tmp_path)
    protocol_path = root / "manifests/protocol_manifest.json"
    protocol = _read_json(protocol_path)
    protocol["coverage_mode"] = "complete"
    write_json(protocol_path, protocol)

    with pytest.raises(ProtocolError, match="config.resolved.yaml"):
        assert_conditional_logit_alignment_artifacts(root)


@pytest.mark.parametrize(
    ("table_name", "mutate", "marker"),
    (
        (
            "source_inner_fold_scores",
            lambda row: row.__setitem__("inner_bacc", "0.5"),
            "gamma summary",
        ),
        (
            "outer_predictions",
            lambda row: (
                row.__setitem__("y_pred", "1" if row["y_pred"] == "0" else "0"),
                row.__setitem__("prob_pos", "0.9" if row["y_pred"] == "1" else "0.1"),
            ),
            "heldout_bacc",
        ),
        (
            "conditional_frame_audit",
            lambda row: row.__setitem__("fit_eval_case_overlap_count", "1"),
            "fit_eval_case_overlap_count",
        ),
        (
            "solver_audit",
            lambda row: row.__setitem__("backend", "unreviewed_backend"),
            "backend",
        ),
    ),
)
def test_coordinated_table_tampering_is_rejected_semantically(
    partial_bundle: Path,
    tmp_path: Path,
    table_name: str,
    mutate: object,
    marker: str,
) -> None:
    root = _copy_bundle(partial_bundle, tmp_path)
    path = root / TABLE_PATHS[table_name]
    rows = _read_csv(path)
    mutate(rows[0])  # type: ignore[operator]
    _write_csv(path, rows, TABLE_COLUMNS[table_name])
    _rebind_hash_chain(root)

    with pytest.raises(ProtocolError, match=marker):
        assert_conditional_logit_alignment_artifacts(root)


def test_decision_hash_tampering_is_rejected(
    partial_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle, tmp_path)
    path = root / "reports/decision_summary.json"
    payload = _read_json(path)
    payload["decision"] = "PASS_DIAGNOSTIC_ONLY"
    write_json(path, payload)

    with pytest.raises(ProtocolError, match="decision hash"):
        assert_conditional_logit_alignment_artifacts(root)


def test_content_index_rejects_stale_extra_file(
    partial_bundle: Path,
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle, tmp_path)
    (root / "reports/stale.txt").write_text("stale\n", encoding="utf-8")
    write_content_index(root)

    with pytest.raises(ProtocolError, match="stale extra"):
        assert_conditional_logit_alignment_artifacts(root)


def _fast_fit(prepared: object, gamma: float, optimizer: object = None) -> AlignmentFitResult:
    del optimizer
    fold = prepared.fold_data
    y = np.asarray(fold.eval_labels, dtype=int)
    positive = np.where(y == 1, 0.9, 0.1)
    fit_identity = stable_hash(
        {
            "method": "conditional_logit_alignment",
            "training_frame_hash": fold.training_frame_hash,
            "fit_row_hash": fold.fit_row_hash,
            "scaler_state_hash": prepared.scaler_state_hash,
            "factor_hash": prepared.penalty_operator.factor_hash,
            "classifier_config_hash": prepared.classifier_spec.config_hash,
            "gamma": float(gamma),
        }
    )
    return AlignmentFitResult(
        gamma=float(gamma),
        predictions=y.copy(),
        probabilities=np.column_stack((1.0 - positive, positive)),
        coefficients=np.zeros(prepared.standardized.fit_embeddings.shape[1]),
        intercept=0.0,
        classes=(0, 1),
        n_iter=(1,),
        converged=True,
        backend="sklearn_lbfgs" if float(gamma) == 0.0 else "scipy_lbfgsb",
        optimizer_success=True,
        optimizer_status=0,
        optimizer_message="synthetic_test_fit",
        n_function_evaluations=0 if float(gamma) == 0.0 else 1,
        n_gradient_evaluations=0 if float(gamma) == 0.0 else 1,
        objective=0.5,
        mean_log_loss=0.5,
        l2_penalty=0.0,
        alignment_penalty=0.0,
        unscaled_alignment_value=0.0,
        gradient_inf_norm=0.0,
        classifier_config_hash=prepared.classifier_spec.config_hash,
        scaler_state_hash=prepared.scaler_state_hash,
        penalty_operator_hash=prepared.penalty_operator.factor_hash,
        fit_identity=fit_identity,
    )


def _write_fixture(base: Path) -> tuple[Path, Path]:
    manifest = base / "midogpp_manifest.csv"
    cache = base / "virchow2_train.npz"
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    for center_index, center in enumerate(("0", "1", "2", "3")):
        for label in (0, 1):
            for item in range(5):
                sample_id = f"c{center}_y{label}_{item}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "case_id": f"case_{sample_id}",
                        "image_path": f"/images/{sample_id}.png",
                        "center": center,
                        "label": label,
                        "split": "train",
                    }
                )
                embeddings.append(
                    np.asarray(
                        (
                            1.5 * label + 0.2 * center_index,
                            -0.7 * label + 0.3 * center_index,
                            0.15 * label * (center_index + 1),
                            0.1 * center_index,
                        )
                    )
                    + rng.normal(0.0, 0.08, size=4)
                )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        cache,
        embeddings=np.stack(embeddings),
        metadata_json=np.asarray(json.dumps(rows)),
        feature_extractor_json=np.asarray(
            json.dumps({"name": "Virchow2", "dataset": "MIDOGpp"})
        ),
    )
    return manifest, cache


def _rebind_hash_chain(root: Path) -> None:
    raw_tables = {
        name: _read_csv(root / relative) for name, relative in TABLE_PATHS.items()
    }
    tables = AlignmentArtifactTables.from_mapping(raw_tables)
    protocol_path = root / "manifests/protocol_manifest.json"
    protocol = _read_json(protocol_path)
    protocol.pop("protocol_hash", None)
    protocol["table_hashes"] = table_hashes(tables)
    protocol["table_bundle_hash"] = table_bundle_hash(tables)
    protocol_hash = stable_hash(protocol)
    protocol["protocol_hash"] = protocol_hash
    write_json(protocol_path, protocol)
    for name, rows in raw_tables.items():
        for row in rows:
            row["protocol_hash"] = protocol_hash
        _write_csv(root / TABLE_PATHS[name], rows, TABLE_COLUMNS[name])

    rebound_tables = AlignmentArtifactTables.from_mapping(raw_tables)
    leakage = build_leakage_report(protocol, rebound_tables.conditional_frame_audit)
    write_json(root / "reports/leakage_provenance_report.json", leakage)
    decision = build_decision_summary(
        rebound_tables.outer_results,
        rebound_tables.outer_comparison,
        rebound_tables.source_inner_gamma_summary,
        design_hash=str(protocol["design_hash"]),
        table_bundle_hash=str(protocol["table_bundle_hash"]),
        protocol_hash=protocol_hash,
    )
    write_json(root / "reports/decision_summary.json", decision)
    (root / "reports/decision_report.md").write_text(
        render_decision_report(decision), encoding="utf-8"
    )
    old_runtime = _read_json(root / "reports/runtime_summary.json")
    runtime = build_runtime_summary(
        protocol,
        rebound_tables,
        elapsed_seconds=float(old_runtime["elapsed_seconds"]),
    )
    write_json(root / "reports/runtime_summary.json", runtime)
    write_content_index(root)


def _copy_bundle(source: Path, tmp_path: Path) -> Path:
    root = tmp_path / "artifact"
    shutil.copytree(source, root)
    return root


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> None:
    write_csv_rows(path, rows, columns)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
