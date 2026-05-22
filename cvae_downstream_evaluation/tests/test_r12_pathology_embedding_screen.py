from dataclasses import replace
import csv
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.pathology_embedding_screen import (  # noqa: E402
    ELIGIBILITY_AUDIT_ONLY,
    ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC,
    ELIGIBILITY_NON_DEPLOYABLE,
    LABEL_090_AUDIT,
    LABEL_090_NOT_SUPPORTED,
    LABEL_090_SOURCE_SELECTED,
    LABEL_CACHE_INCOMPLETE,
    LABEL_CLASS_BALANCE,
    LABEL_REBUILD_ELIGIBLE,
    R12RunLimits,
    ROW_POSTHOC_BEST,
    ROW_SOURCE_INNER_SELECTED,
    ROW_TARGET_TRAIN,
    assert_r12_config_text,
    build_center_summary_rows,
    compute_r12_decision_labels,
    default_r12_config,
    eligibility_for_row_role,
    eval_class_warning,
    load_r12_config,
    run_r12_pathology_embedding_screen,
    select_source_inner_lodo_candidate,
    validate_cache_manifest_alignment,
)


def test_r12_config_loads_locked_template() -> None:
    config = load_r12_config(ROOT / "configs" / "experiments" / "r12_pathology_embedding_screen.yaml")
    assert config.backbones == ("uni", "virchow2", "conch", "ctranspath", "phikon", "plip")
    assert config.representations == ("raw", "PCA64", "PCA128", "PCA256")
    assert config.c_grid == (0.01, 0.1, 1.0, 10.0)


def test_r12_config_rejects_diagnostics_as_selector() -> None:
    text = (ROOT / "configs" / "experiments" / "r12_pathology_embedding_screen.yaml").read_text(
        encoding="utf-8"
    )
    try:
        assert_r12_config_text(text.replace("diagnostics_used_for_selection: false", "diagnostics_used_for_selection: true"))
    except Exception as exc:
        assert "diagnostics_used_for_selection" in str(exc)
    else:
        raise AssertionError("diagnostics selector leakage was not rejected")


def test_cache_manifest_alignment_accepts_reorderable_match() -> None:
    manifest = [
        {"sample_id": "a", "split": "train", "center": "0", "label": "1"},
        {"sample_id": "b", "split": "train", "center": "1", "label": "0"},
    ]
    cache = [
        {"sample_id": "b", "split": "train", "center": "1", "label": "0"},
        {"sample_id": "a", "split": "train", "center": "0", "label": "1"},
    ]
    match, order = validate_cache_manifest_alignment(cache, manifest)
    assert match == "reorderable_match"
    assert order is False


def test_cache_manifest_alignment_rejects_label_mismatch() -> None:
    manifest = [{"sample_id": "a", "split": "test", "center": "0", "label": "1"}]
    cache = [{"sample_id": "a", "split": "test", "center": "0", "label": "0"}]
    match, _ = validate_cache_manifest_alignment(cache, manifest)
    assert match == "label_mismatch"


def test_eligibility_mapping_for_r12_row_roles() -> None:
    assert eligibility_for_row_role(ROW_SOURCE_INNER_SELECTED) == ELIGIBILITY_DEPLOYABLE_DIAGNOSTIC
    assert eligibility_for_row_role(ROW_TARGET_TRAIN) == ELIGIBILITY_NON_DEPLOYABLE
    assert eligibility_for_row_role(ROW_POSTHOC_BEST) == ELIGIBILITY_AUDIT_ONLY


def test_eval_class_warning_distinguishes_absent_and_sparse_classes() -> None:
    assert eval_class_warning(0, 20) == ("single_class_eval", False)
    assert eval_class_warning(3, 20) == ("low_minority_eval_class_count", True)
    assert eval_class_warning(5, 20) == ("", True)


def test_source_inner_lodo_selector_uses_source_metric_not_target_fields() -> None:
    config = default_r12_config()
    selected = select_source_inner_lodo_candidate(
        [
            {
                "row_id": "raw",
                "representation": "raw",
                "C": 1.0,
                "source_inner_lodo_mean_bacc": 0.70,
                "target_bacc_if_someone_added_it": 0.99,
            },
            {
                "row_id": "pca64",
                "representation": "PCA64",
                "C": 0.1,
                "source_inner_lodo_mean_bacc": 0.80,
                "target_bacc_if_someone_added_it": 0.50,
            },
        ],
        config=config,
    )
    assert selected["row_id"] == "pca64"


def test_center_summary_uses_source_selected_backbone_not_target_best() -> None:
    config = default_r12_config()
    selection_rows = [
        _selection_row("a", source_bacc=0.90),
        _selection_row("b", source_bacc=0.70),
    ]
    real_rows = [
        _real_row("a", bacc=0.60),
        _real_row("b", bacc=0.99),
    ]
    rows = build_center_summary_rows(
        config=config,
        real_rows=real_rows,
        selection_rows=selection_rows,
        z11_reference={"0": 0.80},
    )
    center = next(row for row in rows if row["heldout_center"] == "0")
    assert center["best_source_selected_backbone"] == "a"
    assert center["best_source_selected_target_eval_bacc"] == 0.60


def test_decision_labels_separate_posthoc_and_source_selected_090() -> None:
    config = default_r12_config()
    center_rows = [
        _center("0", posthoc=0.91, selected=0.90, z11=0.80),
        _center("1", posthoc=0.90, selected=0.89, z11=0.80),
        _center("2", posthoc=0.92, selected=0.91, z11=0.80),
        _center("3", posthoc=0.89, selected=0.90, z11=0.80),
        _center("4", posthoc=0.91, selected=0.90, z11=0.80),
        _center("__mean__", posthoc=0.906, selected=0.90, z11=0.80, delta=0.10),
    ]
    labels = compute_r12_decision_labels(
        config=config,
        fingerprint_rows=[{"cache_status": "ok"}],
        real_rows=[],
        center_rows=center_rows,
        ranking_rows=[],
    )
    assert LABEL_090_AUDIT in labels
    assert LABEL_090_SOURCE_SELECTED in labels
    assert LABEL_REBUILD_ELIGIBLE in labels


def test_decision_labels_mark_not_supported_and_class_balance_caveat() -> None:
    config = default_r12_config()
    labels = compute_r12_decision_labels(
        config=config,
        fingerprint_rows=[{"cache_status": "missing_not_failed"}],
        real_rows=[{"eval_class_warning": "single_class_eval"}],
        center_rows=[
            _center("0", posthoc=0.70, selected=0.69, z11=0.80, delta=-0.11),
            _center("__mean__", posthoc=0.70, selected=0.69, z11=0.80, delta=-0.11),
        ],
        ranking_rows=[],
    )
    assert LABEL_CACHE_INCOMPLETE in labels
    assert LABEL_090_NOT_SUPPORTED in labels
    assert LABEL_CLASS_BALANCE in labels


def test_missing_cache_run_writes_all_r12_outputs(tmp_path: Path) -> None:
    config = replace(
        default_r12_config(),
        experiment_seeds=(42,),
        candidate_centers=("0", "1"),
        backbones=("phikon",),
        artifacts_root="artifacts",
    )
    config = replace(
        config,
        z11_config=replace(
            config.z11_config,
            experiment_seeds=(42,),
            candidate_centers=("0", "1"),
            expected_support_run_root="support_runs",
            expected_support_run_dir_pattern="seed{seed}",
        ),
    )
    result = run_r12_pathology_embedding_screen(
        config=config,
        repo_root=tmp_path,
        limits=R12RunLimits(representations=("raw", "PCA64")),
    )
    assert LABEL_CACHE_INCOMPLETE in result.decision_labels
    for path in result.output_paths.values():
        assert path.exists(), path

    with result.output_paths["fingerprint"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["cache_status"] for row in rows} == {"missing_not_failed"}

    manifest = result.output_paths["protocol_manifest"].read_text(encoding="utf-8")
    assert '"diagnostics_used_for_selection": false' in manifest
    assert '"source_inner_lodo_selected_rows": "deployable_diagnostic"' in manifest


def _center(
    center: str,
    *,
    posthoc: float,
    selected: float,
    z11: float,
    delta: float | None = None,
) -> dict[str, object]:
    selected_delta = selected - z11 if delta is None else delta
    return {
        "heldout_center": center,
        "best_posthoc_target_eval_bacc": posthoc,
        "best_source_selected_target_eval_bacc": selected,
        "z11_reference_bacc": z11,
        "delta_vs_z11_pca64": selected_delta,
        "weak_center_repaired": str(selected >= 0.85).lower() if center != "__mean__" else "true",
        "weak_center_persists": str(selected < 0.85).lower() if center != "__mean__" else "false",
    }


def _selection_row(backbone: str, *, source_bacc: float) -> dict[str, object]:
    return {
        "row_id": backbone,
        "experiment_seed": 42,
        "backbone_name": backbone,
        "heldout_center": "0",
        "representation": "PCA64",
        "C": 1.0,
        "source_inner_lodo_mean_bacc": source_bacc,
        "selected_by_source_inner_lodo": "true",
        "status": "ok",
    }


def _real_row(backbone: str, *, bacc: float) -> dict[str, object]:
    return {
        "row_id": f"real_{backbone}",
        "experiment_seed": 42,
        "backbone_name": backbone,
        "heldout_center": "0",
        "row_role": ROW_SOURCE_INNER_SELECTED,
        "representation": "PCA64",
        "C": 1.0,
        "bacc": bacc,
        "z11_reference_bacc": 0.80,
        "eval_class_warning": "",
        "status": "ok",
    }
