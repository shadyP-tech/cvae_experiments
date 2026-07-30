from pathlib import Path
import csv
import json

import numpy as np

from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.schemas.midogpp_real_feature_classifier import (
    assert_midogpp_real_feature_artifacts,
)
from midogpp_thesis.real_features.classifier_reference.source_inner_classifier_tuning import (
    ClassifierFoldScore,
    SourceInnerClassifierFold,
)
from midogpp_thesis.real_features.classifier_reference.midogpp_real_feature_classifier import (
    build_source_inner_folds,
    run_midogpp_real_feature_source_inner_classifier_tuning,
)
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    load_midogpp_real_feature_frame,
)
from midogpp_thesis.real_features.classifier_reference.cli import build_parser


def test_midogpp_real_feature_runner_writes_protocol_clean_artifacts(tmp_path: Path) -> None:
    manifest, cache = _write_midogpp_real_feature_fixture(tmp_path)
    out_dir = tmp_path / "out"
    specs = (
        ClassifierSpec(C=0.1, random_state=23),
        ClassifierSpec(C=1.0, random_state=23),
    )

    paths = run_midogpp_real_feature_source_inner_classifier_tuning(
        manifest_path=manifest,
        feature_cache_path=cache,
        out_dir=out_dir,
        candidate_specs=specs,
        heldout_centers=("0",),
        experiment_seed=42,
        classifier_seed=23,
        expected_feature_dim=4,
    )

    assert paths.results.exists()
    assert_midogpp_real_feature_artifacts(out_dir)
    tuning_rows = _read_csv(paths.source_inner_tuning)
    selected = [row for row in tuning_rows if row["selected_by_source_inner_lodo"] == "true"]
    assert len(selected) == 1
    audit = json.loads(selected[0]["source_inner_fold_audit"])
    for pseudo_target, fold in audit.items():
        assert "0" not in fold["train_centers"]
        assert pseudo_target not in fold["train_centers"]
    result_rows = _read_csv(paths.results)
    assert {row["method"] for row in result_rows} == {
        "source_inner_tuned_fixed_0_5",
        "source_inner_tuned_source_inner_threshold",
        "default_untuned_fixed_0_5",
        "default_untuned_source_inner_threshold",
    }
    for row in result_rows:
        assert "0" not in json.loads(row["train_centers"])
        assert row["target_eval_labels_used_for_scoring_only"] == "true"
        assert row["selection_used_target_labels"] == "false"
        assert row["fit_used_target_center"] == "false"
        assert row["claim_scope"] == "real_feature_transfer_only"
        assert row["target_eval_labels_used_for_threshold"] == "False" or row["target_eval_labels_used_for_threshold"] == "false"
        assert row["oracle_rows_used_for_threshold"] == "False" or row["oracle_rows_used_for_threshold"] == "false"
        assert row["threshold_value"] != ""
    prediction_rows = _read_csv(paths.predictions)
    assert {row["center"] for row in prediction_rows} == {"0"}
    for row in prediction_rows:
        expected = int(float(row["prob_pos"]) >= float(row["threshold_value"]))
        assert int(row["y_pred"]) == expected
    protocol = json.loads(paths.protocol_manifest.read_text(encoding="utf-8"))
    assert protocol["is_router"] is False
    assert protocol["generated_embeddings_used"] is False
    assert protocol["source_summary_manifest_used"] is False


def test_midogpp_real_feature_runner_selection_ignores_hypothetical_target_metric(tmp_path: Path) -> None:
    manifest, cache = _write_midogpp_real_feature_fixture(tmp_path)
    low_c = ClassifierSpec(C=0.1, random_state=23)
    high_c = ClassifierSpec(C=10.0, random_state=23)

    def factory(heldout: str):
        assert heldout == "0"
        return _score_low_c

    paths = run_midogpp_real_feature_source_inner_classifier_tuning(
        manifest_path=manifest,
        feature_cache_path=cache,
        out_dir=tmp_path / "out",
        candidate_specs=(high_c, low_c),
        heldout_centers=("0",),
        experiment_seed=42,
        classifier_seed=23,
        expected_feature_dim=4,
        score_fn_factory=factory,
    )

    target_bacc_by_hash = {
        low_c.config_hash: 0.60,
        high_c.config_hash: 0.99,
    }
    assert target_bacc_by_hash[high_c.config_hash] > target_bacc_by_hash[low_c.config_hash]
    selected = [row for row in _read_csv(paths.source_inner_tuning) if row["selected_by_source_inner_lodo"] == "true"]
    assert selected[0]["selected_classifier_config_hash"] == low_c.config_hash


def test_midogpp_real_feature_cache_alignment_rejects_label_mismatch(tmp_path: Path) -> None:
    manifest, cache = _write_midogpp_real_feature_fixture(tmp_path, bad_cache_label=True)

    try:
        load_midogpp_real_feature_frame(
            manifest_path=manifest,
            feature_cache_path=cache,
            expected_feature_dim=4,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("cache/manifest label mismatch was accepted")


def test_midogpp_real_feature_folds_require_both_classes(tmp_path: Path) -> None:
    manifest, cache = _write_midogpp_real_feature_fixture(tmp_path, center2_single_class=True)
    frame = load_midogpp_real_feature_frame(
        manifest_path=manifest,
        feature_cache_path=cache,
        expected_feature_dim=4,
    )

    try:
        build_source_inner_folds(frame, outer_target_center="0")
    except ProtocolError:
        pass
    else:
        raise AssertionError("source-inner fold with one-class train split was accepted")


def test_midogpp_real_feature_cli_excludes_forbidden_backend_inputs() -> None:
    parser = build_parser()
    tune_parser = parser._subparsers._group_actions[0].choices["tune"]
    option_strings = {
        option
        for action in tune_parser._actions
        for option in action.option_strings
    }

    assert "--manifest" in option_strings
    assert "--feature-cache" in option_strings
    assert "--summary-manifest" not in option_strings
    assert "--generation-seed" not in option_strings
    assert "--latent-sample-seed" not in option_strings
    assert "--synthetic-per-class-total" not in option_strings
    assert "--checkpoint" not in option_strings


def test_real_feature_cli_help_exposes_registered_diagnostics() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {
        "tune",
        "matched-reference",
        "fixed-c-risk-diagnostic",
        "conditional-logit-alignment",
        "physical-multiscale-center-pooling-pilot",
        "physical-multiscale-annotation-local-pooling-pilot",
        "uniform-b-v3-replay",
        "build-uniform-b-v3-test-cache",
        "uniform-b-v3-confirmation",
        "build-uniform-b-canonical-train-cache",
        "uniform-b-canonical-reference",
        "uniform-b-nystroem-nonlinear-probe",
        "uniform-b-robust-interaction-probe",
        "uniform-b-sens-spec-constrained-nystroem-probe",
    }


def _write_midogpp_real_feature_fixture(
    tmp_path: Path,
    *,
    bad_cache_label: bool = False,
    center2_single_class: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "midogpp"
    root.mkdir()
    manifest = root / "manifest.csv"
    cache = root / "virchow2_train.npz"
    rows = []
    embeddings = []
    for center in ("0", "1", "2"):
        labels = (0, 0) if center == "2" and center2_single_class else (0, 1)
        for idx, label in enumerate(labels):
            sample_id = f"center{center}_sample{idx}_y{label}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "case_id": sample_id,
                    "image_path": f"{sample_id}.png",
                    "label": str(label),
                    "split": "train",
                    "center": center,
                }
            )
            base = float(int(center) * 4)
            embeddings.append([base + label, base + (label * 2), float(label), 1.0])
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "case_id", "image_path", "label", "split", "center"),
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata = [dict(row) for row in rows]
    if bad_cache_label:
        metadata[0]["label"] = "1" if metadata[0]["label"] == "0" else "0"
    np.savez(
        cache,
        embeddings=np.asarray(embeddings, dtype=float),
        metadata_json=json.dumps(metadata, sort_keys=True),
        feature_extractor_json=json.dumps(
            {
                "dataset": "midogpp",
                "backbone_type": "virchow2",
                "cache_builder": "test_fixture",
            },
            sort_keys=True,
        ),
    )
    return manifest, cache


def _score_low_c(spec: ClassifierSpec, fold: SourceInnerClassifierFold) -> ClassifierFoldScore:
    del fold
    if spec.C == 0.1:
        return ClassifierFoldScore(bacc=0.90, macro_f1=0.88, n_iter=(10,))
    return ClassifierFoldScore(bacc=0.70, macro_f1=0.68, n_iter=(10,))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
