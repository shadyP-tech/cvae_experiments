from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from midogpp_thesis.real_features.classifier_reference import (
    fixed_c_risk_diagnostic as fixed_c_runner,
)
from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash
from midogpp_thesis.real_features.classifier_reference.fixed_c_risk_diagnostic import (
    FixedCRiskDiagnosticConfig,
    run_fixed_c_risk_diagnostic,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    load_midogpp_real_feature_frame,
)
from midogpp_thesis.real_features.classifier_reference.schemas.fixed_c_risk_diagnostic import (
    FIXED_C_RISK_REQUIRED_OUTPUTS,
    RISK_POLICY_IDS,
    assert_fixed_c_risk_artifacts,
    fixed_c_risk_bundle_hash,
    render_diagnostic_report,
)
from midogpp_thesis.real_features.classifier_reference.schemas.midogpp import (
    MIDOGPP_ELIGIBLE_CENTERS,
)


TABLE_PATHS = {
    "results": "tables/fixed_c_risk_results.csv",
    "predictions": "tables/fixed_c_risk_predictions.csv",
    "audits": "tables/fixed_c_risk_weight_audit.csv",
    "paired": "tables/fixed_c_risk_paired_comparison.csv",
}


@pytest.fixture(scope="module")
def partial_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, RealFeatureFrame]:
    workspace = tmp_path_factory.mktemp("fixed_c_risk_validation")
    manifest, cache = _write_fixture(
        workspace / "midogpp_fixture",
        centers=("0", "1", "2"),
        feature_dim=4,
        rows_per_center=12,
    )
    root = run_fixed_c_risk_diagnostic(
        FixedCRiskDiagnosticConfig(
            name="fixed_c_risk_diagnostic_v1",
            artifact_root=workspace / "artifact",
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=("0", "1", "2"),
            expected_feature_dim=4,
            allow_partial_test_coverage=True,
        )
    )
    frame = load_midogpp_real_feature_frame(
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_dim=4,
    )
    return root, frame


def test_partial_bundle_validates_through_public_facade(
    partial_bundle: tuple[Path, RealFeatureFrame],
) -> None:
    root, frame = partial_bundle

    assert_fixed_c_risk_artifacts(root, already_loaded_frame=frame)

    generated = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert generated == set(FIXED_C_RISK_REQUIRED_OUTPUTS)
    assert len(_read_csv(root / TABLE_PATHS["results"])) == 12
    assert len(_read_csv(root / TABLE_PATHS["paired"])) == 3
    summary = _read_json(root / "reports/diagnostic_summary.json")
    assert summary["status"] == "COMPLETE_DIAGNOSTIC_ONLY"


@pytest.mark.parametrize("identity", ("sample_id", "case_id"))
def test_bound_input_identity_overlap_is_rejected(
    partial_bundle: tuple[Path, RealFeatureFrame],
    identity: str,
) -> None:
    root, frame = partial_bundle
    source_index = next(
        index for index, row in enumerate(frame.rows) if row.center == "1"
    )
    target_index = next(
        index for index, row in enumerate(frame.rows) if row.center == "0"
    )
    rows = list(frame.rows)
    rows[target_index] = replace(
        rows[target_index],
        **{identity: getattr(rows[source_index], identity)},
    )
    changed_frame = replace(frame, rows=tuple(rows))

    _assert_semantic_rejection(
        root,
        "overlap",
        already_loaded_frame=changed_frame,
    )


def test_runner_rejects_cross_center_case_overlap_before_classifier_fit(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frame = partial_bundle
    source_index = next(
        index for index, row in enumerate(frame.rows) if row.center == "1"
    )
    target_index = next(
        index for index, row in enumerate(frame.rows) if row.center == "0"
    )
    rows = list(frame.rows)
    rows[target_index] = replace(
        rows[target_index],
        case_id=rows[source_index].case_id,
    )
    changed_frame = replace(frame, rows=tuple(rows))
    fit_call_count = 0

    def unexpected_fit(*_args: object, **_kwargs: object) -> object:
        nonlocal fit_call_count
        fit_call_count += 1
        raise AssertionError("classifier fitting must not start after identity overlap")

    monkeypatch.setattr(
        fixed_c_runner,
        "load_midogpp_real_feature_frame",
        lambda **_kwargs: changed_frame,
    )
    monkeypatch.setattr(
        fixed_c_runner,
        "fit_logistic_classifier",
        unexpected_fit,
    )

    with pytest.raises(ProtocolError) as caught:
        run_fixed_c_risk_diagnostic(
            FixedCRiskDiagnosticConfig(
                name="fixed_c_risk_diagnostic_v1",
                artifact_root=tmp_path / "artifact",
                manifest_path=frame.manifest_path,
                feature_cache_path=frame.feature_cache_path,
                heldout_centers=("0", "1", "2"),
                expected_feature_dim=4,
                allow_partial_test_coverage=True,
            )
        )

    message = str(caught.value).lower()
    assert "case" in message
    assert "overlap" in message
    assert fit_call_count == 0


def test_probability_out_of_range_is_rejected_after_full_rebind(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS["predictions"]
    rows = _read_csv(path)
    rows[0]["prob_pos"] = "1.000001"
    _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, "prediction")


def test_duplicate_prediction_sample_is_rejected_after_full_rebind(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS["predictions"]
    rows = _read_csv(path)
    matching = [
        index
        for index, row in enumerate(rows)
        if row["heldout_center"] == "0"
        and row["risk_policy_id"] == "global_class"
    ]
    rows[matching[1]]["sample_id"] = rows[matching[0]]["sample_id"]
    _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, "sample_id")


def test_weight_group_formula_drift_is_rejected_after_full_rebind(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS["audits"]
    rows = _read_csv(path)
    row = _row_for(rows, heldout="0", policy="global_class")
    weights = json.loads(row["group_weights"])
    key = sorted(weights)[0]
    weights[key] = float(weights[key]) + 0.125
    row["group_weights"] = _json(weights)
    _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, "formula")


def test_consistently_rebound_weight_vector_drift_is_rejected_from_inputs(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    replacement_hash = "f" * 16
    for table_name in ("results", "predictions", "audits"):
        path = root / TABLE_PATHS[table_name]
        rows = _read_csv(path)
        for row in rows:
            if (
                row["heldout_center"] == "0"
                and row["risk_policy_id"] == "global_class"
            ):
                row["weight_vector_hash"] = replacement_hash
        _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, "weight-vector")


@pytest.mark.parametrize(
    ("field", "replacement", "marker"),
    (
        ("heldout_bacc", "0.123456789", "bacc"),
        ("heldout_macro_f1", "0.123456789", "macro-f1"),
    ),
)
def test_metric_drift_is_rejected_after_full_rebind(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
    field: str,
    replacement: str,
    marker: str,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS["results"]
    rows = _read_csv(path)
    _row_for(rows, heldout="0", policy="global_class")[field] = replacement
    _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, marker)


@pytest.mark.parametrize(
    ("field", "replacement", "marker"),
    (
        ("experiment_seed", "43", "experiment_seed"),
        ("classifier_seed", "24", "classifier_seed"),
        ("n_iter", "[5000]", "n_iter"),
        ("n_iter", "[5001]", "n_iter"),
    ),
)
def test_seed_and_iteration_drift_is_rejected_after_full_rebind(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
    field: str,
    replacement: str,
    marker: str,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS["results"]
    rows = _read_csv(path)
    _row_for(rows, heldout="0", policy="global_class")[field] = replacement
    _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, marker)


@pytest.mark.parametrize(
    ("table_name", "field", "replacement", "marker"),
    (
        ("results", "row_role", "deployable_result", "row_role"),
        ("audits", "claim_scope", "routing_and_composition", "claim_scope"),
        ("predictions", "support_labels_used", "true", "support_labels_used"),
        ("paired", "oracle_eligible", "true", "oracle_eligible"),
        ("results", "diagnostic_only", "false", "diagnostic_only"),
        ("results", "adoption_eligible", "true", "adoption_eligible"),
    ),
)
def test_row_and_claim_boundary_drift_is_rejected_after_full_rebind(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
    table_name: str,
    field: str,
    replacement: str,
    marker: str,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS[table_name]
    rows = _read_csv(path)
    rows[0][field] = replacement
    _write_csv(path, rows)
    _rebind_content_bundle(root)

    _assert_semantic_rejection(root, marker)


def test_frozen_snapshot_drift_is_rejected(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / "manifests/frozen_protocol_snapshot.json"
    payload = _read_json(path)
    payload["classifier_config_hash"] = "0" * 16
    _write_json(path, payload)

    _assert_semantic_rejection(root, "frozen")


def test_table_row_must_remain_bound_to_protocol(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / TABLE_PATHS["results"]
    rows = _read_csv(path)
    rows[0]["protocol_hash"] = "0" * 16
    _write_csv(path, rows)

    # protocol_hash is deliberately excluded from the table content digest.
    _assert_semantic_rejection(root, "protocol_hash")


@pytest.mark.parametrize(
    "relative_path",
    (
        "reports/leakage_provenance_report.json",
        "reports/diagnostic_summary.json",
        "reports/runtime_summary.json",
    ),
)
def test_json_report_must_remain_bound_to_protocol_and_bundle(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
    relative_path: str,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / relative_path
    payload = _read_json(path)
    payload["bundle_hash"] = "0" * 16
    _write_json(path, payload)

    _assert_semantic_rejection(root, "bound")


def test_rendered_report_must_remain_bound_to_summary(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / "reports/diagnostic_report.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    _assert_semantic_rejection(root, "report")


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("n_fits", 0),
        ("used_for_selection", True),
        ("diagnostic_only", False),
    ),
)
def test_runtime_semantics_are_revalidated(
    partial_bundle: tuple[Path, RealFeatureFrame],
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    root = _copy_bundle(partial_bundle[0], tmp_path)
    path = root / "reports/runtime_summary.json"
    payload = _read_json(path)
    payload[field] = replacement
    _write_json(path, payload)

    _assert_semantic_rejection(root, "runtime")


def test_complete_bundle_validates_portably_with_current_input_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, cache = _write_fixture(
        tmp_path / "midogpp_complete_fixture",
        centers=MIDOGPP_ELIGIBLE_CENTERS,
        feature_dim=2560,
        rows_per_center=4,
    )
    root = tmp_path / "complete_artifact"
    _write_complete_workspace_bindings(root, manifest=manifest, cache=cache)

    monkeypatch.setattr(
        fixed_c_runner,
        "fit_logistic_classifier",
        _fast_logistic_fit,
    )
    monkeypatch.setattr(
        fixed_c_runner,
        "assert_fixed_c_risk_artifacts",
        lambda *_args, **_kwargs: None,
    )
    root = run_fixed_c_risk_diagnostic(
        FixedCRiskDiagnosticConfig(
            name="fixed_c_risk_diagnostic_v1",
            artifact_root=root,
            manifest_path=manifest,
            feature_cache_path=cache,
            heldout_centers=MIDOGPP_ELIGIBLE_CENTERS,
        )
    )
    frame = load_midogpp_real_feature_frame(
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_dim=2560,
    )

    from midogpp_thesis.real_features.classifier_reference import (
        fixed_c_risk_protocol_validation,
    )

    monkeypatch.setattr(
        fixed_c_risk_protocol_validation,
        "resolve_current_fixed_c_risk_input_paths",
        lambda: (manifest, cache),
    )

    assert_fixed_c_risk_artifacts(root, already_loaded_frame=frame)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    assert protocol["coverage_mode"] == "complete"
    assert protocol["expected_fit_count"] == 36


def _assert_semantic_rejection(
    root: Path,
    marker: str,
    *,
    already_loaded_frame: RealFeatureFrame | None = None,
) -> None:
    with pytest.raises(ProtocolError) as caught:
        assert_fixed_c_risk_artifacts(
            root,
            already_loaded_frame=already_loaded_frame,
        )
    message = str(caught.value).lower()
    assert "content bundle hash mismatch" not in message
    assert marker.lower() in message


def _copy_bundle(source: Path, tmp_path: Path) -> Path:
    target = tmp_path / "changed_bundle"
    shutil.copytree(source, target)
    return target


def _rebind_content_bundle(root: Path) -> None:
    tables = {
        name: _read_csv(root / relative)
        for name, relative in TABLE_PATHS.items()
    }
    bundle_hash = fixed_c_risk_bundle_hash(
        tables["results"],
        tables["predictions"],
        tables["audits"],
        tables["paired"],
    )
    protocol_path = root / "manifests/protocol_manifest.json"
    protocol = _read_json(protocol_path)
    protocol.pop("protocol_hash", None)
    protocol["bundle_hash"] = bundle_hash
    protocol_hash = stable_hash(protocol)
    protocol["protocol_hash"] = protocol_hash
    _write_json(protocol_path, protocol)

    for name, relative in TABLE_PATHS.items():
        for row in tables[name]:
            row["protocol_hash"] = protocol_hash
        _write_csv(root / relative, tables[name])

    for relative in (
        "reports/leakage_provenance_report.json",
        "reports/diagnostic_summary.json",
        "reports/runtime_summary.json",
    ):
        path = root / relative
        payload = _read_json(path)
        payload["protocol_hash"] = protocol_hash
        payload["bundle_hash"] = bundle_hash
        _write_json(path, payload)

    summary = _read_json(root / "reports/diagnostic_summary.json")
    (root / "reports/diagnostic_report.md").write_text(
        render_diagnostic_report(summary),
        encoding="utf-8",
    )


def _write_fixture(
    root: Path,
    *,
    centers: tuple[str, ...],
    feature_dim: int,
    rows_per_center: int,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    manifest = root / "midogpp_manifest.csv"
    cache = root / "virchow2_midogpp_train.npz"
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    embeddings: list[np.ndarray] = []
    rng = np.random.default_rng(1701)
    index = 0
    for center_index, center in enumerate(centers):
        for local in range(rows_per_center):
            label = local % 2
            sample_id = f"s{index}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "case_id": f"case{index}",
                    "label": label,
                    "split": "train",
                    "center": center,
                }
            )
            metadata.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "center": center,
                    "split": "train",
                }
            )
            vector = rng.normal(scale=0.05, size=feature_dim)
            vector[0] = (-2.0 if label == 0 else 2.0) + 0.01 * center_index
            embeddings.append(vector)
            index += 1
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        cache,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        metadata_json=json.dumps(metadata),
        feature_extractor_json=json.dumps(
            {"backbone_type": "virchow2", "dataset": "midogpp"}
        ),
    )
    return manifest, cache


def _write_complete_workspace_bindings(
    root: Path,
    *,
    manifest: Path,
    cache: Path,
) -> None:
    source = Path(
        "experiments/midogpp/stages/10_real_feature_reference/configs/"
        "fixed_c_risk_diagnostic_v1.yaml"
    )
    resolved = yaml.safe_load(source.read_text(encoding="utf-8"))
    resolved["experiment"]["artifact_root"] = str(root)
    resolved["inputs"] = {
        "manifest_path": str(manifest),
        "feature_cache_path": str(cache),
    }
    root.mkdir(parents=True)
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )

    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": "midogpp.real_feature.fixed_c_risk_diagnostic.v1",
        "stage": "10_real_feature_reference",
        "claim_scope": "real_feature_transfer_only",
        "selection_used_target_eval_artifacts": False,
        "input_artifacts": [
            _provenance_row(
                artifact_id="midogpp_dataset_contract_annotation_patch_v1",
                relative_path="manifest.csv",
                digest=_sha256(manifest),
            ),
            _provenance_row(
                artifact_id="midogpp_virchow2_xyxy_feature_cache_seed42",
                relative_path="embeddings/train.pt",
                digest=_sha256(cache),
            ),
        ],
    }
    _write_json(root / "provenance/input_artifacts.json", provenance)


def _provenance_row(
    *,
    artifact_id: str,
    relative_path: str,
    digest: str,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "exists": True,
        "semantic_identities_are_file_hashes": False,
        "file_integrity": {
            "files": [
                {
                    "path": relative_path,
                    "exists": True,
                    "computed": {"sha256": digest},
                }
            ]
        },
    }


def _fast_logistic_fit(
    x_train: np.ndarray,
    _y_train: tuple[int, ...],
    x_eval: np.ndarray,
    *,
    spec: object,
    sample_weight: object,
) -> SimpleNamespace:
    del sample_weight
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(x_eval)[:, 0]))
    predictions = (probabilities >= 0.5).astype(int)
    return SimpleNamespace(
        predictions=predictions,
        probabilities=np.column_stack((1.0 - probabilities, probabilities)),
        converged=True,
        n_iter=(1,),
        scaler_state_hash=stable_hash(
            {
                "shape": list(np.asarray(x_train).shape),
                "mean": float(np.asarray(x_train)[:, 0].mean()),
                "classifier": getattr(spec, "config_hash", ""),
            }
        ),
    )


def _row_for(
    rows: list[dict[str, str]],
    *,
    heldout: str,
    policy: str,
) -> dict[str, str]:
    return next(
        row
        for row in rows
        if row["heldout_center"] == heldout and row["risk_policy_id"] == policy
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
