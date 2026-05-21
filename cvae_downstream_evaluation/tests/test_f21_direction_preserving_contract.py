from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "cvae_testing"))

from cvae_downstream_evaluation.downstream import CandidateDownstreamRow  # noqa: E402
from cvae_downstream_evaluation.f1_source_anchored import AnchoredResidualCVAE, build_source_anchor_index  # noqa: E402
from cvae_downstream_evaluation.f21_direction_preserving import (  # noqa: E402
    F21_GENERATOR_FAMILY,
    F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE,
    F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE,
    STATUS_DIRECTION_BANK_INVALID,
    build_f21_direction_bank,
    build_f21_routing_alignment_rows,
    generate_f21_direction_preserving_embeddings,
)
from cvae_downstream_evaluation.f2_calibrated_residual import ResidualCalibration  # noqa: E402
from cvae_downstream_evaluation.routing import SupportSelectionUnit  # noqa: E402
from cvae_downstream_evaluation.schemas import PRIMARY_BUDGET_PER_CLASS, SINGLE_EXPERT_ROW_TYPE, SUPPORT_NELBO_METHOD  # noqa: E402


def test_f21_runner_cli_help_loads() -> None:
    script = ROOT / "scripts" / "run_f21_direction_preserving_downstream.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--f2-artifacts-root" in result.stdout
    assert "direction-preserving" in result.stdout


def test_f21_direction_bank_marks_invalid_under_default_threshold() -> None:
    bank = build_f21_direction_bank(
        anchor_index=_anchor_index(),
        label_values=(0, 1),
        min_valid_directions=64,
    )

    assert not bank.valid_for((0, 1))
    assert "invalid_direction_bank_classes" in bank.invalid_reason_for((0, 1))
    assert {row["direction_reference_split"] for row in bank.rows} == {"source_train"}
    assert {row["direction_anchor_split"] for row in bank.rows} == {"source_train"}


def test_f21_generation_is_seeded_source_train_only_and_self_excluded() -> None:
    torch.manual_seed(17)
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    anchor_index = _anchor_index()
    bank = build_f21_direction_bank(anchor_index=anchor_index, label_values=(0, 1), min_valid_directions=4)
    calibration = ResidualCalibration(class_scales={0: 1.0, 1: 1.0}, global_scale=1.0, rows=())

    first = generate_f21_direction_preserving_embeddings(
        model=model,
        anchor_index=anchor_index,
        direction_bank=bank,
        calibration=calibration,
        class_label=0,
        n_samples=6,
        seed=23,
        generation_mode=F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
    )
    second = generate_f21_direction_preserving_embeddings(
        model=model,
        anchor_index=anchor_index,
        direction_bank=bank,
        calibration=calibration,
        class_label=0,
        n_samples=6,
        seed=23,
        generation_mode=F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE,
        experiment_seed=42,
        heldout_center="0",
        candidate_expert="1",
    )

    assert torch.allclose(first.embeddings, second.embeddings)
    assert len(first.provenance_rows) == 6
    assert {row["anchor_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["direction_reference_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["direction_anchor_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["residual_reference_split"] for row in first.provenance_rows} == {"source_train"}
    assert {row["calibration_split"] for row in first.provenance_rows} == {"source_val"}
    assert not any("x_target" in key for row in first.provenance_rows for key in row)
    for row in first.provenance_rows:
        assert row["synthetic_anchor_id"] != row["direction_reference_sample_id"]
        assert row["synthetic_anchor_id"] != row["direction_anchor_id"]
        assert row["synthetic_anchor_id"] != row["residual_reference_sample_id"]
        assert row["direction_reference_sample_id"] != row["residual_reference_sample_id"]
    assert "residual_direction_cosine_real_vs_synthetic" in first.diagnostics
    assert "fraction_same_direction_and_anchor_pair_reused" in first.diagnostics


def test_f21_empirical_magnitude_control_is_deterministic() -> None:
    model = AnchoredResidualCVAE(input_dim=2, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    anchor_index = _anchor_index()
    bank = build_f21_direction_bank(anchor_index=anchor_index, label_values=(0, 1), min_valid_directions=4)
    calibration = ResidualCalibration(class_scales={0: 2.5, 1: 2.5}, global_scale=2.5, rows=())

    first = generate_f21_direction_preserving_embeddings(
        model=model,
        anchor_index=anchor_index,
        direction_bank=bank,
        calibration=calibration,
        class_label=1,
        n_samples=4,
        seed=41,
        generation_mode=F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE,
    )
    second = generate_f21_direction_preserving_embeddings(
        model=model,
        anchor_index=anchor_index,
        direction_bank=bank,
        calibration=calibration,
        class_label=1,
        n_samples=4,
        seed=41,
        generation_mode=F21_MODE_EMP_DIRECTION_EMP_MAGNITUDE,
    )

    assert torch.allclose(first.embeddings, second.embeddings)
    assert abs(first.diagnostics["residual_magnitude_ratio_to_real"] - 1.0) < 1.0e-6


def test_f21_invalid_selected_expert_alignment_emits_protocol_label() -> None:
    rows = [
        _candidate("1", F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE, status=STATUS_DIRECTION_BANK_INVALID, bacc=float("nan")),
        _candidate("2", F21_MODE_EMP_DIRECTION_CVAE_MAGNITUDE, status="ok", bacc=0.75),
    ]
    unit = SupportSelectionUnit(
        heldout_center="0",
        experiment_seed=42,
        support_size=16,
        support_seed=17,
        method=SUPPORT_NELBO_METHOD,
        selected_expert="1",
        candidate_experts=("1", "2"),
        support_nelbo_by_expert={"1": 1.0, "2": 2.0},
        target_expert_excluded=True,
        support_eval_split_id="split",
    )

    alignment = build_f21_routing_alignment_rows(selections=[unit], downstream_rows=rows)

    assert alignment[0]["selected_status"] == STATUS_DIRECTION_BANK_INVALID
    assert alignment[0]["decision_label"] == STATUS_DIRECTION_BANK_INVALID
    assert alignment[0]["downstream_oracle_expert"] == "2"


def _anchor_index():
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.1],
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.0],
            [0.2, 1.1],
            [1.0, 1.0],
            [1.1, 1.1],
        ],
        dtype=torch.float32,
    )
    metadata = tuple(
        {"center": "1", "label": 0 if idx < 5 else 1, "sample_id": f"s{idx}"}
        for idx in range(10)
    )
    return build_source_anchor_index(
        source_projected_embeddings=embeddings,
        source_metadata=metadata,
        source_domain="1",
        label_values=(0, 1),
        neighbor_k=3,
    )


def _candidate(expert: str, mode: str, *, status: str, bacc: float) -> CandidateDownstreamRow:
    return CandidateDownstreamRow(
        experiment_seed=42,
        heldout_center="0",
        support_size=16,
        support_seed=17,
        candidate_expert=expert,
        generator_family=F21_GENERATOR_FAMILY,
        generation_mode=mode,
        budget_per_class=PRIMARY_BUDGET_PER_CLASS,
        generation_seed=17,
        classifier_seed=17,
        bacc=bacc,
        macro_f1=bacc,
        row_type=SINGLE_EXPERT_ROW_TYPE,
        status=status,
        error_message="" if status == "ok" else status,
    )
