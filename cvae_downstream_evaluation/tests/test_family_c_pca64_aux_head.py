from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "cvae_testing"))

from cvae_downstream_evaluation.family_c_pca64_conditional import (  # noqa: E402
    FAMILY_C_PCA64_AUX_NAME,
    PCA64_AUX_CVAE_MODE,
    PCA64_AUX_FEATURE_SPACE,
    PCA64_AUX_RAW_SELECTOR,
    assert_family_c_pca64_aux_head_config_text,
    assert_pca64_cc_checkpoint_metadata,
    build_family_c_pca64_cc_alignment_rows,
    classify_family_c_pca64_cc_decision,
    default_family_c_pca64_aux_head_config,
    load_family_c_pca64_aux_head_config,
    _decoded_space_probe_diagnostics,
    _metadata_constraint_cfg,
    _pca64_cc_cvae_components,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_aux_head_config_exists_and_loads_locked_identity() -> None:
    path = ROOT / "configs" / "experiments" / "family_c_pca64_class_conditional_aux_head_cvae_downstream_v1.yaml"
    text = path.read_text(encoding="utf-8")
    assert_family_c_pca64_aux_head_config_text(text)
    config = load_family_c_pca64_aux_head_config(path)
    assert config.experiment_name == FAMILY_C_PCA64_AUX_NAME
    assert config.feature_space == PCA64_AUX_FEATURE_SPACE
    assert config.cvae_mode == PCA64_AUX_CVAE_MODE
    assert config.raw_selector == PCA64_AUX_RAW_SELECTOR
    assert config.metadata_constraint_enabled is True
    assert config.metadata_constraint_variant == "aux_head"
    assert config.metadata_constraint_aux_weight == 1.0


def test_aux_head_checkpoint_metadata_rejects_non_aux_or_wrong_weight() -> None:
    expected = {
        "input_dim": 64,
        "feature_space": PCA64_AUX_FEATURE_SPACE,
        "conditioning": "class_label_one_hot",
        "metadata_dim": 2,
        "metadata_constraint_enabled": 1,
        "metadata_constraint_variant": "aux_head",
        "metadata_constraint_use_mu": 1,
        "metadata_constraint_aux_weight": 1.0,
        "pca_artifact_id": "pca",
        "scaler_artifact_id": "scaler",
        "source_center": "2",
        "experiment_seed": 42,
    }
    assert_pca64_cc_checkpoint_metadata(expected, expected=expected)

    non_aux = dict(expected)
    non_aux["metadata_constraint_enabled"] = 0
    try:
        assert_pca64_cc_checkpoint_metadata(non_aux, expected=expected)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Non-aux-head checkpoint metadata was accepted")

    wrong_weight = dict(expected)
    wrong_weight["metadata_constraint_aux_weight"] = 0.25
    try:
        assert_pca64_cc_checkpoint_metadata(wrong_weight, expected=expected)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Wrong aux-head loss weight was accepted")


def test_aux_head_loss_includes_ce_and_changes_when_labels_change() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from src.models.cvae_expert import CVAEExpert

    config = default_family_c_pca64_aux_head_config()
    model = CVAEExpert(
        64,
        8,
        2,
        metadata_dim=2,
        metadata_constraint_cfg=_metadata_constraint_cfg(config),
        aux_metadata_dim=2,
    )
    for param in model.parameters():
        param.data.zero_()
    with torch.no_grad():
        model.fc_mu.bias[0] = 3.0
        model.metadata_aux_head.weight[0, 0] = 1.0
        model.metadata_aux_head.weight[1, 0] = -1.0
    x = torch.zeros((4, 64), dtype=torch.float32)
    labels0 = torch.zeros((4,), dtype=torch.long)
    labels1 = torch.ones((4,), dtype=torch.long)

    comp0 = _pca64_cc_cvae_components(model, x, labels0, kl_beta=1.0, aux_weight=1.0, torch=torch)
    comp1 = _pca64_cc_cvae_components(model, x, labels1, kl_beta=1.0, aux_weight=1.0, torch=torch)

    assert comp0["aux_ce_mean"].detach().item() < comp1["aux_ce_mean"].detach().item()
    assert comp0["total_loss"].detach().item() < comp1["total_loss"].detach().item()
    assert comp0["aux_loss_fraction"].detach().item() > 0.0
    assert tuple(model.metadata_constraint_logits(torch.zeros((2, 2)), torch.zeros((2, 2))).shape) == (2, 2)


def test_decoded_space_probe_diagnostics_are_finite_for_separable_classes() -> None:
    import numpy as np

    x0 = np.zeros((20, 4), dtype=float)
    x1 = np.ones((20, 4), dtype=float) * 5.0
    generated = np.vstack([x0, x1])
    labels = [0] * 20 + [1] * 20
    source = generated + 0.1
    diag = _decoded_space_probe_diagnostics(
        generated=generated,
        generated_labels=labels,
        source=source,
        source_labels=labels,
        seed=17,
    )
    assert math.isfinite(diag["generated_class_linear_probe_bacc"])
    assert math.isfinite(diag["source_val_reconstruction_probe_bacc"])
    assert math.isfinite(diag["generated_vs_source_probe_gap"])


def test_aux_alignment_and_decision_use_aux_selectors_and_baseline_delta() -> None:
    config = default_family_c_pca64_aux_head_config()
    rows = [
        _matrix_row("0", "1", support=10.0, bacc=0.78, probe=0.80),
        _matrix_row("0", "2", support=2.0, bacc=0.83, probe=0.84),
    ]
    alignment = build_family_c_pca64_cc_alignment_rows(rows=rows, candidate_domains=("0", "1", "2"), config=config)
    raw = [row for row in alignment if row["selector"] == PCA64_AUX_RAW_SELECTOR][0]
    assert raw["selected_expert"] == "2"
    assert raw["downstream_oracle_expert"] == "2"

    summary = classify_family_c_pca64_cc_decision(rows=rows, alignment_rows=alignment, config=config)
    assert summary["decision_classification"] == "AUX_HEAD_GENERATOR_SOLVED"
    assert summary["metrics"]["delta_vs_pca64_class_conditional_oracle_bacc"] > 0.02


def _matrix_row(heldout: str, expert: str, *, support: float, bacc: float, probe: float) -> dict[str, object]:
    return {
        "schema_version": "family_c_pca64_class_conditional_aux_head_cvae_downstream_v1",
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": "target0_seed17_random_k4",
        "candidate_expert": expert,
        "generation_mode": PCA64_AUX_CVAE_MODE,
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "bacc": bacc,
        "macro_f1": bacc,
        "row_type": "single_expert_pca64_class_conditional_aux_head_cvae",
        "support_nelbo_raw": support,
        "support_nelbo_source_prior": support,
        "support_nelbo_global_source_prior": support,
        "target_eval_nelbo_unlabeled": support,
        "target_eval_has_all_classes": 1,
        "generated_class_linear_probe_bacc": probe,
        "available": 1,
        "status": "ok",
    }
