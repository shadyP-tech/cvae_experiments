from pathlib import Path
import csv
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cvae_downstream_evaluation.midogpp_real_feature_recipe_classifier import (  # noqa: E402
    run_midogpp_real_feature_recipe_tuning,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.real_feature_recipes import ModelSpec, PreprocessingSpec, RecipeSpec  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp_real_feature_recipe import (  # noqa: E402
    assert_midogpp_real_feature_recipe_artifacts,
)
from run_midogpp_real_feature_recipe_tuning import build_parser  # noqa: E402


def test_midogpp_real_feature_recipe_runner_writes_protocol_clean_artifacts(tmp_path: Path) -> None:
    manifest, cache = _write_midogpp_real_feature_fixture(tmp_path)
    recipes = (
        _recipe("logistic", c=0.1),
        _recipe("logistic", c=1.0),
        _recipe("linear_svm", c=0.1),
    )
    baseline = (_recipe("logistic", c=0.1), _recipe("logistic", c=1.0))

    paths = run_midogpp_real_feature_recipe_tuning(
        manifest_path=manifest,
        feature_cache_path=cache,
        out_dir=tmp_path / "out",
        candidate_recipes=recipes,
        logistic_baseline_recipes=baseline,
        heldout_centers=("0",),
        experiment_seed=42,
        classifier_seed=23,
        expected_feature_dim=4,
    )

    assert paths.results.exists()
    assert_midogpp_real_feature_recipe_artifacts(tmp_path / "out")
    tuning_rows = _read_csv(paths.source_inner_tuning)
    selected = [row for row in tuning_rows if row["selected_by_source_inner_lodo"] == "true" and row["row_role"] == "selection_candidate"]
    assert len(selected) == 1
    audit = json.loads(selected[0]["source_inner_fold_audit"])
    for pseudo_target, fold in audit.items():
        assert "0" not in fold["train_centers"]
        assert pseudo_target not in fold["train_centers"]
    result_rows = _read_csv(paths.results)
    assert {row["method"] for row in result_rows} == {
        "source_inner_recipe_selected_predict",
        "locked_logistic_baseline_predict",
    }
    for row in result_rows:
        assert "0" not in json.loads(row["train_centers"])
        assert row["target_eval_labels_used_for_scoring_only"] == "true"
        assert row["selection_used_target_labels"] == "false"
        assert row["fit_used_target_center"] == "false"
        assert row["claim_scope"] == "real_feature_transfer_only"
        assert "threshold_policy" not in row
    protocol = json.loads(paths.protocol_manifest.read_text(encoding="utf-8"))
    assert protocol["is_router"] is False
    assert protocol["generated_embeddings_used"] is False
    assert protocol["source_summary_manifest_used"] is False
    leakage = json.loads(paths.leakage_report.read_text(encoding="utf-8"))
    assert leakage["fit_scope_rows"]
    for row in leakage["fit_scope_rows"]:
        assert row["target_center_excluded_from_fit"] is True


def test_recipe_artifact_validator_rejects_svm_probability_rows(tmp_path: Path) -> None:
    manifest, cache = _write_midogpp_real_feature_fixture(tmp_path)
    recipes = (_recipe("linear_svm", c=0.1), _recipe("logistic", c=0.1))
    paths = run_midogpp_real_feature_recipe_tuning(
        manifest_path=manifest,
        feature_cache_path=cache,
        out_dir=tmp_path / "out",
        candidate_recipes=recipes,
        logistic_baseline_recipes=(_recipe("logistic", c=0.1),),
        heldout_centers=("0",),
        experiment_seed=42,
        classifier_seed=23,
        expected_feature_dim=4,
    )
    rows = _read_csv(paths.predictions)
    rows[0]["recipe_family"] = "linear_svm"
    rows[0]["score_kind"] = "decision_function"
    rows[0]["score_pos"] = "0.5"
    with paths.predictions.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    try:
        assert_midogpp_real_feature_recipe_artifacts(tmp_path / "out")
    except ProtocolError:
        pass
    else:
        raise AssertionError("validator accepted SVM probability rows")


def test_midogpp_real_feature_recipe_cli_has_no_threshold_or_calibration_flags() -> None:
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--manifest" in option_strings
    assert "--feature-cache" in option_strings
    assert "--threshold-policy" not in option_strings
    assert "--classifier-c-grid" not in option_strings
    assert "--checkpoint" not in option_strings


def _recipe(family: str, *, c: float = 1.0) -> RecipeSpec:
    preprocessing = PreprocessingSpec(kind="standardize", random_state=23)
    if family == "logistic":
        model = ModelSpec(family="logistic", C=float(c), solver="lbfgs", max_iter=2000, random_state=23)
    else:
        model = ModelSpec(family="linear_svm", C=float(c), max_iter=10000, random_state=23)
    return RecipeSpec(preprocessing=preprocessing, model=model)


def _write_midogpp_real_feature_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "midogpp"
    root.mkdir()
    manifest = root / "manifest.csv"
    cache = root / "virchow2_train.npz"
    rows = []
    embeddings = []
    for center in ("0", "1", "2"):
        for idx, label in enumerate((0, 1)):
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
    np.savez(
        cache,
        embeddings=np.asarray(embeddings, dtype=float),
        metadata_json=json.dumps(rows, sort_keys=True),
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
