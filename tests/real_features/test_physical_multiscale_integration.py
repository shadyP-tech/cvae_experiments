from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from midogpp_thesis.data.features.cache_io import write_center_shard
from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from midogpp_thesis.real_features.classifier_reference.downstream import (
    balanced_accuracy,
    macro_f1,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.config import (
    BootstrapConfig,
    GateConfig,
    PhysicalMultiscalePilotConfig,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.runner import (
    run_physical_multiscale_center_pooling_pilot,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling import (
    runner as pilot_runner,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.profiles import (
    ANNOTATION_LOCAL_PROFILE_V2,
    CENTER_POOLING_PROFILE_V1,
    CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    PhysicalMultiscaleProfile,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.input_lineage import (
    _center_grouped_pooling_keys,
    compute_input_hashes,
)
from midogpp_thesis.real_features.classifier_reference.physical_multiscale_center_pooling.validation import (
    validate_physical_multiscale_pilot_bundle,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


def test_v3_input_lineage_matches_center_sharded_pooling_order() -> None:
    contract_rows = [
        {"sample_id": "center1-first", "center": "1"},
        {"sample_id": "center0-first", "center": "0"},
        {"sample_id": "center1-second", "center": "1"},
        {"sample_id": "center0-second", "center": "0"},
    ]

    assert _center_grouped_pooling_keys(
        contract_rows,
        center_order=("0", "1"),
    ) == [
        ("center0-first", 28.0),
        ("center0-first", 56.0),
        ("center0-first", 112.0),
        ("center0-second", 28.0),
        ("center0-second", 56.0),
        ("center0-second", 112.0),
        ("center1-first", 28.0),
        ("center1-first", 56.0),
        ("center1-first", 112.0),
        ("center1-second", 28.0),
        ("center1-second", 56.0),
        ("center1-second", 112.0),
    ]


def test_physical_multiscale_runner_validates_full_bundle_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    config = _write_pilot_fixture(tmp_path)

    root = run_physical_multiscale_center_pooling_pilot(config)

    validation = validate_physical_multiscale_pilot_bundle(root, config=config)
    assert validation == {
        "status": "PASS",
        "selector_cells": 18,
        "candidate_summaries": 9,
        "decision_locks": 3,
        "outer_results": 6,
        "posthoc_rows": 6,
    }
    protocol = _json(root / "manifests" / "protocol_manifest.json")
    report = _json(root / "reports" / "validation_report.json")
    assert protocol["status"] == "PASS"
    assert report["authoritative_bundle_verdict"] is True
    assert report["checks"] == validation
    replay = _csv(root / "tables" / "canonical_a_replay.csv")
    assert len(replay) == 3
    assert all(row["predictions_exact"] == "True" for row in replay)

    posthoc_path = root / "tables" / "posthoc_candidate_isolation.csv"
    posthoc = _csv(posthoc_path)
    posthoc[0]["may_feed_selection"] = "True"
    _write_csv(posthoc_path, posthoc)
    with pytest.raises(ProtocolError, match="Posthoc candidate isolation"):
        validate_physical_multiscale_pilot_bundle(root, config=config)


def test_annotation_local_v2_uses_same_locked_stage10_engine(
    tmp_path: Path,
) -> None:
    config = _write_pilot_fixture(
        tmp_path,
        profile=ANNOTATION_LOCAL_PROFILE_V2,
    )

    root = run_physical_multiscale_center_pooling_pilot(config)
    validation = validate_physical_multiscale_pilot_bundle(root, config=config)

    assert validation["status"] == "PASS"
    frozen = _json(root / "manifests" / "frozen_protocol_snapshot.json")
    assert frozen["profile_id"] == ANNOTATION_LOCAL_PROFILE_V2.profile_id
    assert frozen["representations"] == dict(
        ANNOTATION_LOCAL_PROFILE_V2.representation_dims
    )
    decisions = _csv(root / "tables" / "representation_decisions.csv")
    assert {
        row["selected_representation"] for row in decisions
    }.issubset(set(ANNOTATION_LOCAL_PROFILE_V2.representation_order))


def test_v3_binds_atomic_parent_and_rejects_parent_or_child_tampering(
    tmp_path: Path,
) -> None:
    config = _write_pilot_fixture(
        tmp_path,
        profile=CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    )

    hashes = compute_input_hashes(config)
    assert {
        "cache_bundle_manifest",
        "cache_bundle_pooling_audit",
        "cache_bundle_content_index",
        "cache_bundle_report",
    }.issubset(hashes)
    root = run_physical_multiscale_center_pooling_pilot(config)
    report_text = (root / "reports" / "decision_report.md").read_text(
        encoding="utf-8"
    )
    protocol = _json(root / "manifests" / "protocol_manifest.json")
    assert report_text.startswith(
        "# MIDOG++ Clipped-Bbox Annotation-Local Pooling Pilot v3"
    )
    assert protocol["performs_expert_aggregation"] is False
    assert "performs_aggregation" not in protocol
    assert protocol["claim_role"] == (
        "complete_deterministic_representation_plus_classifier_"
        "pipeline_diagnostic"
    )

    child_report_path = config.b_cache_root / "reports" / "cache_builder_report.json"
    child_report = _json(child_report_path)
    child_report["profile_id"] = "mixed-lineage"
    _write_json(child_report_path, child_report)
    with pytest.raises(ValueError, match="child lineage is mixed"):
        compute_input_hashes(config)


def test_v3_rejects_self_consistent_parent_firewall_rewrite(
    tmp_path: Path,
) -> None:
    config = _write_pilot_fixture(
        tmp_path,
        profile=CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    )
    assert config.cache_bundle_root is not None
    manifest_path = (
        config.cache_bundle_root / "manifests" / "bundle_manifest.json"
    )
    report_path = config.cache_bundle_root / "reports" / "cache_bundle_report.json"
    manifest = _json(manifest_path)
    report = _json(report_path)
    manifest["may_feed_deployable_selection"] = True
    report["may_feed_deployable_selection"] = True
    _write_json(manifest_path, manifest)
    _write_json(report_path, report)
    _refresh_content_index(config.cache_bundle_root)

    with pytest.raises(ValueError, match="parent lineage drifted"):
        compute_input_hashes(config)


@pytest.mark.parametrize(
    ("relative_path", "key", "value"),
    (
        (
            "b_3840/reports/cache_builder_report.json",
            "schema_version",
            "mixed_or_legacy_schema",
        ),
        (
            "c_11520/reports/cache_builder_report.json",
            "pooling",
            "unsafe_or_mixed_pooling",
        ),
        (
            "b_3840/reports/cache_builder_report.json",
            "may_feed_deployable_selection",
            True,
        ),
        (
            "c_11520/manifests/row_alignment.json",
            "schema_version",
            "mixed_or_legacy_alignment",
        ),
        (
            "b_3840/manifests/row_alignment.json",
            "unexpected_semantic_field",
            "unsafe",
        ),
        (
            "b_3840/reports/cache_builder_report.json",
            "input_decoder",
            "pyvips_raw_tiff",
        ),
    ),
)
def test_v3_rejects_self_consistent_child_lineage_rewrite(
    tmp_path: Path,
    relative_path: str,
    key: str,
    value: object,
) -> None:
    config = _write_pilot_fixture(
        tmp_path,
        profile=CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    )
    assert config.cache_bundle_root is not None
    path = config.cache_bundle_root / relative_path
    payload = _json(path)
    payload[key] = value
    _write_json(path, payload)
    _refresh_content_index(config.cache_bundle_root)

    with pytest.raises(ValueError, match="child lineage is mixed"):
        compute_input_hashes(config)


def test_v3_production_failure_quarantines_partial_stage10_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_pilot_fixture(
        tmp_path,
        profile=CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3,
    )
    final_root = tmp_path / "production" / "v3" / "seed42"
    config = replace(
        fixture,
        artifact_root=final_root,
        allow_partial_test_coverage=False,
    )
    _write_workspace_prepared_root(final_root)

    def fail_after_partial(
        inner: PhysicalMultiscalePilotConfig,
        *,
        production_binding_validated: bool = False,
    ) -> Path:
        assert production_binding_validated is True
        inner.artifact_root.mkdir(parents=True, exist_ok=True)
        (inner.artifact_root / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected v3 failure")

    monkeypatch.setattr(
        pilot_runner,
        "_run_physical_multiscale_pilot_in_place",
        fail_after_partial,
    )
    monkeypatch.setattr(
        pilot_runner,
        "validate_production_workspace_binding",
        lambda _config: None,
    )
    with pytest.raises(RuntimeError, match="injected v3 failure"):
        run_physical_multiscale_center_pooling_pilot(config)

    assert not final_root.exists()
    assert not final_root.with_name(".seed42.staging").exists()
    quarantines = tuple(final_root.parent.glob(".seed42.quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "partial.txt").read_text(encoding="utf-8") == "partial"
    assert (quarantines[0] / "config.resolved.yaml").is_file()
    assert (
        quarantines[0] / "provenance" / "input_artifacts.json"
    ).is_file()


def _write_pilot_fixture(
    tmp_path: Path,
    *,
    profile: PhysicalMultiscaleProfile = CENTER_POOLING_PROFILE_V1,
) -> PhysicalMultiscalePilotConfig:
    centers = ("0", "1", "2")
    spec = ClassifierSpec(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=500,
        class_weight=None,
        random_state=23,
        threshold_policy="predict",
    )
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    (artifact_root / "config.resolved.yaml").write_text(
        f"experiment:\n  name: {profile.experiment_name}\n",
        encoding="utf-8",
    )
    (artifact_root / "provenance").mkdir()
    _write_json(
        artifact_root / "provenance" / "input_artifacts.json",
        {"status": "TEST_FIXTURE", "allow_partial_test_coverage": True},
    )
    base_manifest = tmp_path / "manifest.csv"
    contract_root = tmp_path / "contract"
    cache_bundle_root = (
        tmp_path / "bundle"
        if profile is CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3
        else None
    )
    b_root = (
        cache_bundle_root / "b_3840"
        if cache_bundle_root is not None
        else tmp_path / "b"
    )
    c_root = (
        cache_bundle_root / "c_11520"
        if cache_bundle_root is not None
        else tmp_path / "c"
    )
    reference_root = tmp_path / "reference"
    a_cache = tmp_path / "canonical_a.pt"
    rows: list[dict[str, object]] = []
    a_by_center: dict[str, np.ndarray] = {}
    labels_by_center: dict[str, np.ndarray] = {}
    metadata_by_center: dict[str, list[dict[str, object]]] = {}
    rng = np.random.default_rng(42)
    for center_index, center in enumerate(centers):
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        informative = np.stack(
            (
                labels * 2.5 + rng.normal(0.0, 0.05, len(labels)),
                labels * 1.5 + center_index * 0.02,
                (1 - labels) * 1.2,
                np.linspace(-0.2, 0.2, len(labels)),
            ),
            axis=1,
        )
        a = np.zeros((len(labels), 2560), dtype=np.float32)
        a[:, :4] = informative
        b = np.zeros((len(labels), 3840), dtype=np.float32)
        b[:, :2560] = a
        c = np.zeros((len(labels), 11520), dtype=np.float32)
        c[:, :2560] = a
        metadata = []
        for index, label in enumerate(labels.tolist()):
            sample_id = f"center_{center}_sample_{index}"
            row = {
                "sample_id": sample_id,
                "case_id": f"case_{sample_id}",
                "image_path": f"{sample_id}.png",
                "label": label,
                "split": "train",
                "center": center,
                "contract_row_index": len(rows),
            }
            rows.append(row)
            metadata.append(dict(row))
        write_center_shard(
            b_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.from_numpy(b),
            canonical_a_embeddings=torch.from_numpy(a),
            metadata=metadata,
            feature_extractor={"representation_id": profile.representation_order[1]},
        )
        write_center_shard(
            c_root / "embeddings" / "by_center" / f"center_{center}.pt",
            embeddings=torch.from_numpy(c),
            metadata=metadata,
            feature_extractor={"representation_id": profile.representation_order[2]},
        )
        a_by_center[center] = a
        labels_by_center[center] = labels
        metadata_by_center[center] = metadata
    _write_csv(base_manifest, rows)
    torch.save(
        {
            "embeddings": torch.from_numpy(np.concatenate(tuple(a_by_center.values()))),
            "metadata": rows,
        },
        a_cache,
    )
    canonical_sha256 = _sha256(a_cache)
    contract_root.mkdir()
    contract_payload: dict[str, object] = {
        "status": "PASS",
        "row_count": len(rows),
        "contract_hash": "fixture",
    }
    if profile is CLIPPED_BBOX_ANNOTATION_LOCAL_PROFILE_V3:
        contract_payload.update(
            {
                "canonical_cache_sha256": canonical_sha256,
                "fov_um": [28.0, 56.0, 112.0],
                "eligible_centers": list(centers),
                "geometry_policy": {
                    "annotation_anchor_policy_id": (
                        "continuous_half_open_bbox_image_intersection_centroid_v1"
                    )
                },
                "claim_firewall": {
                    "feature_extraction_stochastic": False,
                    "geometry_uses_labels": False,
                    "geometry_uses_center_identity": False,
                    "may_feed_recipe_selection": False,
                    "may_feed_deployable_selection": False,
                    "uses_likelihood": False,
                    "uses_nelbo": False,
                    "uses_latent_prior": False,
                    "uses_posterior": False,
                    "uses_mixture_model": False,
                    "uses_experts": False,
                    "performs_expert_aggregation": False,
                    "uses_generative_sampling": False,
                },
            }
        )
    _write_json(
        contract_root / "physical_multiscale_contract.json",
        contract_payload,
    )
    _write_csv(
        contract_root / "physical_multiscale_manifest.csv",
        [
            {
                **row,
                "row_index": index,
                "raw_tiff_path": f"raw/{row['sample_id']}.tif",
            }
            for index, row in enumerate(rows)
        ],
    )
    _write_csv(
        contract_root / "resolution_audit.csv",
        [
            {
                "sample_id": row["sample_id"],
                "center": row["center"],
                "status": "PASS",
            }
            for row in rows
        ],
    )
    for root, representation_id, dimension in (
        (b_root, profile.representation_order[1], 3840),
        (c_root, profile.representation_order[2], 11520),
    ):
        (root / "manifests").mkdir(parents=True)
        (root / "reports").mkdir()
        _write_json(
            root / "manifests" / "row_alignment.json",
            {
                "schema_version": "midogpp_physical_multiscale_cache_alignment_v3",
                "status": "PASS",
                "profile_id": profile.profile_id,
                "row_count": len(rows),
                "sample_id_order_hash": stable_hash(
                    [row["sample_id"] for row in rows]
                ),
                "eligible_centers": list(centers),
                "physical_contract_hash": "fixture",
                "canonical_a_cache_sha256": canonical_sha256,
                "annotation_anchor_policy_id": (
                    "continuous_half_open_bbox_image_intersection_centroid_v1"
                ),
                "center_4_present": False,
            },
        )
        _write_json(
            root / "reports" / "cache_builder_report.json",
            {
                "schema_version": "midogpp_physical_multiscale_cache_builder_v3",
                "status": "PASS",
                "profile_id": profile.profile_id,
                "representation_id": representation_id,
                "feature_dim": dimension,
                "row_count": len(rows),
                "sample_id_order_hash": stable_hash(
                    [row["sample_id"] for row in rows]
                ),
                "physical_contract_hash": "fixture",
                "canonical_a_cache_sha256": canonical_sha256,
                "annotation_anchor_policy_id": (
                    "continuous_half_open_bbox_image_intersection_centroid_v1"
                ),
                "model_ref": "fixture-model",
                "model_revision": "fixture-revision",
                "model_identity": {
                    "model_ref": "fixture-model",
                    "requested_revision": "fixture-revision",
                },
                "runtime_identity": {"fixture": True},
                "input_decoder": (
                    "pillow_jpeg"
                    if root == b_root
                    else "pyvips_raw_tiff"
                ),
                "preprocessing_spatial_identity": {"fixture": True},
                "pooling": (
                    "fixed_center_rows6to9_cols6to9"
                    if root == b_root
                    else "annotation_local_start_clamp_floor_16p_minus2_window4"
                ),
                "bridge": {"status": "PASS"},
            },
        )
    if cache_bundle_root is not None:
        _write_atomic_bundle_fixture(
            cache_bundle_root,
            profile=profile,
            sample_ids=[str(row["sample_id"]) for row in rows],
            canonical_cache_sha256=canonical_sha256,
        )
    _write_reference(
        reference_root,
        centers=centers,
        spec=spec,
        a_by_center=a_by_center,
        labels_by_center=labels_by_center,
        metadata_by_center=metadata_by_center,
    )
    return PhysicalMultiscalePilotConfig(
        name=profile.experiment_name,
        artifact_root=artifact_root,
        base_manifest_path=base_manifest,
        physical_contract_root=contract_root,
        canonical_a_cache_path=a_cache,
        b_cache_root=b_root,
        c_cache_root=c_root,
        canonical_reference_root=reference_root,
        heldout_centers=centers,
        classifier_specs=(spec,),
        classifier_seed=23,
        experiment_seed=42,
        gate=GateConfig(),
        bootstrap=BootstrapConfig(
            seed=42,
            valid_replicates=20,
            max_attempts=500,
        ),
        expected_selector_cells=18,
        expected_candidate_summaries=9,
        allow_partial_test_coverage=True,
        profile=profile,
        cache_bundle_root=cache_bundle_root,
    )


def _write_reference(
    root: Path,
    *,
    centers: tuple[str, ...],
    spec: ClassifierSpec,
    a_by_center: dict[str, np.ndarray],
    labels_by_center: dict[str, np.ndarray],
    metadata_by_center: dict[str, list[dict[str, object]]],
) -> None:
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for heldout in centers:
        sources = tuple(center for center in centers if center != heldout)
        train = np.concatenate(tuple(a_by_center[center] for center in sources))
        train_labels = np.concatenate(tuple(labels_by_center[center] for center in sources))
        target = a_by_center[heldout]
        target_labels = labels_by_center[heldout]
        fitted = fit_logistic_classifier(
            train,
            train_labels,
            target,
            spec=spec,
        )
        predictions = [int(value) for value in fitted.predictions.tolist()]
        result_rows.append(
            {
                "heldout_center": heldout,
                "selected_classifier_config_hash": spec.config_hash,
                "selected_classifier_spec": json.dumps(spec.to_payload(), sort_keys=True),
                "heldout_bacc": balanced_accuracy(
                    target_labels.tolist(), predictions
                ),
                "heldout_macro_f1": macro_f1(
                    target_labels.tolist(), predictions
                ),
            }
        )
        prediction_rows.extend(
            {
                "heldout_center": heldout,
                "sample_id": metadata["sample_id"],
                "y_pred": prediction,
            }
            for metadata, prediction in zip(
                metadata_by_center[heldout], predictions, strict=True
            )
        )
    (root / "manifests").mkdir(parents=True)
    (root / "tables").mkdir()
    _write_json(root / "manifests" / "protocol_manifest.json", {"status": "PASS"})
    _write_csv(root / "tables" / "classifier_tuned_source_results.csv", result_rows)
    _write_csv(root / "tables" / "classifier_tuned_predictions.csv", prediction_rows)


def _write_atomic_bundle_fixture(
    root: Path,
    *,
    profile: PhysicalMultiscaleProfile,
    sample_ids: list[str],
    canonical_cache_sha256: str,
) -> None:
    (root / "manifests").mkdir(parents=True)
    (root / "reports").mkdir()
    pooling_rows = [
        {"sample_id": sample_id, "fov_um": fov}
        for sample_id in sample_ids
        for fov in (28.0, 56.0, 112.0)
    ]
    _write_csv(root / "manifests" / "pooling_audit.csv", pooling_rows)
    manifest = {
        "schema_version": "midogpp_physical_multiscale_cache_bundle_v3",
        "status": "PASS",
        "profile_id": profile.profile_id,
        "annotation_anchor_policy_id": (
            "continuous_half_open_bbox_image_intersection_centroid_v1"
        ),
        "physical_contract_hash": "fixture",
        "canonical_a_cache_sha256": canonical_cache_sha256,
        "row_count": len(sample_ids),
        "sample_id_order_hash": stable_hash(sample_ids),
        "representations": [
            {
                "representation_id": profile.representation_order[1],
                "relative_root": "b_3840",
                "feature_dim": 3840,
            },
            {
                "representation_id": profile.representation_order[2],
                "relative_root": "c_11520",
                "feature_dim": 11520,
            },
        ],
        "c_scale_order_um": [28.0, 56.0, 112.0],
        "representation_c_combination": "feature_concatenation_not_mixture",
        "patch_pool": "uniform_arithmetic_mean_16_tokens",
        "annotation_jpeg_decoder": "pillow",
        "raw_tiff_slide_reader_backend": "pyvips",
        "feature_extraction_stochastic": False,
        "may_feed_recipe_selection": False,
        "may_feed_deployable_selection": False,
        "uses_likelihood": False,
        "uses_nelbo": False,
        "uses_mixture_model": False,
        "uses_experts": False,
        "performs_expert_aggregation": False,
        "uses_generative_sampling": False,
    }
    _write_json(root / "manifests" / "bundle_manifest.json", manifest)
    _write_json(
        root / "reports" / "cache_bundle_report.json",
        {
            **manifest,
            "schema_version": "midogpp_physical_multiscale_cache_bundle_report_v3",
            "pooling_audit_row_count": len(pooling_rows),
            "pooling_audit_hash": stable_hash(
                [
                    {str(key): str(value) for key, value in row.items()}
                    for row in pooling_rows
                ]
            ),
            "model_identity": {
                "model_ref": "fixture-model",
                "requested_revision": "fixture-revision",
            },
            "runtime_identity": {"fixture": True},
            "preprocessing_spatial_identity": {"fixture": True},
            "bridge": {"status": "PASS"},
        },
    )
    _refresh_content_index(root)


def _refresh_content_index(root: Path) -> None:
    _write_json(
        root / "manifests" / "content_index.json",
        {
            "schema_version": "midogpp_physical_multiscale_content_index_v3",
            "status": "PASS",
            "annotation_anchor_policy_id": (
                "continuous_half_open_bbox_image_intersection_centroid_v1"
            ),
            "physical_contract_hash": "fixture",
            "files": {
                str(path.relative_to(root)): _sha256(path)
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.name != "content_index.json"
            },
        },
    )


def _write_workspace_prepared_root(root: Path) -> None:
    for relative in ("manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "config.resolved.yaml").write_text(
        "experiment:\n  artifact_root: prepared\n",
        encoding="utf-8",
    )
    _write_json(
        root / "provenance" / "input_artifacts.json",
        {"status": "PREPARED"},
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]] | list[dict[str, str]]) -> None:
    if not rows:
        raise AssertionError("Test fixture cannot write an empty CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
