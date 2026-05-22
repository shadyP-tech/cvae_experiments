from dataclasses import replace
import csv
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.ceiling_audit import (  # noqa: E402
    ELIGIBILITY_AUDIT_ONLY,
    ELIGIBILITY_NON_DEPLOYABLE,
    LABEL_FEASIBLE,
    LABEL_IDENTITY_INCOMPLETE,
    LABEL_IDENTITY_PASS,
    LABEL_NOT_SUPPORTED,
    LABEL_PCA_BOTTLENECK,
    LABEL_PCA_NO_GAIN,
    LABEL_SOURCE_TRANSFER,
    LABEL_SYNTHETIC_MISSING,
    Z11RunLimits,
    compute_decision_labels,
    default_z11_config,
    eligibility_for_train_scope,
    load_z11_config,
    pca_dim_warning,
    preservation_ratio,
    run_z11_ceiling_audit,
)


def test_z11_config_loads_locked_template() -> None:
    config = load_z11_config(
        ROOT / "configs" / "experiments" / "z11_current_setup_ceiling_audit.yaml"
    )
    assert config.candidate_centers == ("0", "1", "2", "3", "4")
    assert "PCA64_reconstruction" in config.representations
    assert config.preservation_ratio_min == 0.70


def test_preservation_ratio_uses_bacc_headroom_and_guards_chance() -> None:
    assert preservation_ratio(0.90, 0.78) == (0.78 - 0.50) / (0.90 - 0.50)
    assert math.isnan(preservation_ratio(0.50, 0.80))
    assert math.isnan(preservation_ratio(0.49, 0.80))


def test_eligibility_mapping_blocks_target_train_rows() -> None:
    assert eligibility_for_train_scope("source_only") == ELIGIBILITY_AUDIT_ONLY
    assert eligibility_for_train_scope("target_train_diagnostic") == ELIGIBILITY_NON_DEPLOYABLE


def test_pca_dim_warning_marks_low_sample_ratio_without_failing() -> None:
    assert pca_dim_warning(700, 256, multiplier=3) == "low_sample_to_dimension_ratio"
    assert pca_dim_warning(800, 256, multiplier=3) == ""
    assert pca_dim_warning(4, None, multiplier=3) == ""


def test_decision_labels_separate_feasibility_source_transfer_and_pca() -> None:
    config = default_z11_config()
    labels = compute_decision_labels(
        config=config,
        fingerprint_rows=[{"fingerprint_status": "ok"}],
        real_rows=[],
        pca_rows=[
            {
                "candidate_representation": "PCA128",
                "delta_bacc": 0.03,
            }
        ],
        center_summary_rows=[
            {
                "heldout_center": "0",
                "best_posthoc_source_only_bacc": 0.91,
                "source_transfer_bottleneck": "true",
                "weak_center_bottleneck": "false",
            },
            {
                "heldout_center": "1",
                "best_posthoc_source_only_bacc": 0.90,
                "source_transfer_bottleneck": "false",
                "weak_center_bottleneck": "false",
            },
            {
                "heldout_center": "__mean__",
                "best_posthoc_source_only_bacc": 0.905,
            },
        ],
        synthetic_rows=[
            {
                "heldout_center": "__mean__",
                "evidence_status": "missing",
            }
        ],
    )
    assert LABEL_IDENTITY_PASS in labels
    assert LABEL_FEASIBLE in labels
    assert LABEL_SOURCE_TRANSFER in labels
    assert LABEL_PCA_BOTTLENECK in labels
    assert LABEL_SYNTHETIC_MISSING in labels


def test_decision_labels_use_not_supported_and_pca_no_gain() -> None:
    config = default_z11_config()
    labels = compute_decision_labels(
        config=config,
        fingerprint_rows=[{"fingerprint_status": "missing_not_failed"}],
        real_rows=[],
        pca_rows=[
            {
                "candidate_representation": "PCA128",
                "delta_bacc": 0.001,
            }
        ],
        center_summary_rows=[
            {
                "heldout_center": "0",
                "best_posthoc_source_only_bacc": 0.84,
                "source_transfer_bottleneck": "false",
                "weak_center_bottleneck": "true",
            },
            {
                "heldout_center": "__mean__",
                "best_posthoc_source_only_bacc": 0.84,
            },
        ],
        synthetic_rows=[],
    )
    assert LABEL_IDENTITY_INCOMPLETE in labels
    assert LABEL_NOT_SUPPORTED in labels
    assert LABEL_PCA_NO_GAIN in labels


def test_missing_artifact_run_writes_all_z11_outputs(tmp_path: Path) -> None:
    config = replace(
        default_z11_config(),
        experiment_seeds=(42,),
        candidate_centers=("0", "1"),
        artifacts_root="artifacts",
    )
    result = run_z11_ceiling_audit(
        config=config,
        repo_root=tmp_path,
        limits=Z11RunLimits(representations=("raw", "PCA64")),
    )
    assert LABEL_IDENTITY_INCOMPLETE in result.decision_labels
    assert LABEL_SYNTHETIC_MISSING in result.decision_labels
    for path in result.output_paths.values():
        assert path.exists(), path

    fingerprint_path = result.output_paths["fingerprint"]
    with fingerprint_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["fingerprint_status"] for row in rows} == {"missing_not_failed"}

    with result.output_paths["real_feature"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "eligibility" in (reader.fieldnames or ())
        assert "uses_target_eval_labels_for_training" in (reader.fieldnames or ())
