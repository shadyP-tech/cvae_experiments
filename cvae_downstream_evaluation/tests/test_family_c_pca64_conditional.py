from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "cvae_testing"))

from cvae_downstream_evaluation.family_c_pca64 import (  # noqa: E402
    FamilyCPca64BuildLimits,
    fit_source_train_pca64_preprocessor,
)
from cvae_downstream_evaluation.family_c_pca64_conditional import (  # noqa: E402
    FAMILY_C_PCA64_CC_NAME,
    PCA64_CC_CVAE_MODE,
    PCA64_CC_FEATURE_SPACE,
    PCA64_CC_PCA_DIM,
    PCA64_CC_RAW_SELECTOR,
    Pca64ClassConditionalExpert,
    assert_family_c_pca64_cc_config_text,
    assert_pca64_cc_checkpoint_metadata,
    build_family_c_pca64_cc_alignment_rows,
    class_one_hot,
    classify_family_c_pca64_cc_decision,
    default_family_c_pca64_cc_config,
    score_label_marginal_nelbo,
)
from cvae_downstream_evaluation.matrix import EmbeddingCache  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_family_c_pca64_cc_config_exists_and_is_locked() -> None:
    path = ROOT / "configs" / "experiments" / "family_c_pca64_class_conditional_cvae_downstream_v1.yaml"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert_family_c_pca64_cc_config_text(text)
    config = default_family_c_pca64_cc_config()
    assert config.experiment_name == FAMILY_C_PCA64_CC_NAME
    assert config.pca_dim == 64
    assert config.budget_per_class == 128


def test_pca_scaler_fit_only_matching_source_train_center_rows() -> None:
    cache = _cache(n_per_center=70, dim=68)
    prep = fit_source_train_pca64_preprocessor(
        experiment_seed=42,
        source_center="1",
        train_cache=cache,
        feature_extractor="dino",
        split_id="source_train_seed42",
        pca_dim=64,
        target_center="0",
    )
    assert prep.source_center == "1"
    assert prep.n_fit_samples == 70
    assert prep.embedding_dim == 68

    try:
        fit_source_train_pca64_preprocessor(
            experiment_seed=42,
            source_center="1",
            train_cache=cache,
            feature_extractor="dino",
            split_id="source_train_seed42",
            pca_dim=64,
            target_center="1",
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("PCA fit accepted target-center leakage")


def test_class_one_hot_and_checkpoint_metadata_reject_unconditioned_or_768d() -> None:
    one_hot = class_one_hot([0, 1, 0])
    assert one_hot.shape == (3, 2)
    assert one_hot.tolist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    try:
        class_one_hot([2])
    except ProtocolError:
        pass
    else:
        raise AssertionError("Invalid class label was accepted")

    expected = {
        "input_dim": 64,
        "feature_space": PCA64_CC_FEATURE_SPACE,
        "conditioning": "class_label_one_hot",
        "metadata_dim": 2,
        "pca_artifact_id": "pca",
        "scaler_artifact_id": "scaler",
        "source_center": "2",
        "experiment_seed": 42,
    }
    assert_pca64_cc_checkpoint_metadata(expected, expected=expected)
    legacy = dict(expected)
    legacy["input_dim"] = 768
    try:
        assert_pca64_cc_checkpoint_metadata(legacy, expected=expected)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Legacy 768-D checkpoint metadata was accepted")
    unconditioned = dict(expected)
    unconditioned["conditioning"] = "none"
    try:
        assert_pca64_cc_checkpoint_metadata(unconditioned, expected=expected)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Unconditioned PCA64 checkpoint metadata was accepted")


def test_cvae_decoder_conditioning_changes_decoder_path_output() -> None:
    import pytest

    torch = pytest.importorskip("torch")
    from src.models.cvae_expert import CVAEExpert

    model = CVAEExpert(64, 8, 2, metadata_dim=2, aux_metadata_dim=2)
    for param in model.parameters():
        param.data.zero_()
    with torch.no_grad():
        model.dec1.weight[0, 2] = 1.0
        model.dec1.weight[1, 3] = 1.0
        model.dec2.weight[0, 0] = 1.0
        model.dec2.weight[1, 1] = 1.0
    z = torch.zeros((1, 2), dtype=torch.float32)
    out0 = model.decode(z, m=torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    out1 = model.decode(z, m=torch.tensor([[0.0, 1.0]], dtype=torch.float32))
    assert not torch.allclose(out0, out1)
    assert float(out0.detach()[0, 0]) == 1.0
    assert float(out1.detach()[0, 1]) == 1.0


def test_uniform_label_marginal_nelbo_evaluates_both_classes_without_labels() -> None:
    import pytest

    torch = pytest.importorskip("torch")

    class DummyModel:
        def __init__(self) -> None:
            self.seen: list[tuple[float, float]] = []

        def eval(self) -> None:
            return None

        def encode(self, x, m=None):
            assert m is not None
            self.seen.append(tuple(float(v) for v in m[0].detach().cpu().tolist()))
            return torch.zeros((x.shape[0], 2), device=x.device), torch.zeros((x.shape[0], 2), device=x.device)

        def decode(self, z, m=None):
            assert m is not None
            return m[:, :1].repeat(1, 64)

    model = DummyModel()
    x = torch.zeros((4, 64), dtype=torch.float32)
    score = score_label_marginal_nelbo(
        model,
        x,
        class_prior={0: 0.5, 1: 0.5},
        torch=torch,
        device=torch.device("cpu"),
        kl_beta=1.0,
    )
    assert score.n_samples == 4
    assert math.isfinite(score.total)
    assert model.seen == [(1.0, 0.0), (0.0, 1.0)]


def test_headline_alignment_excludes_single_class_target_eval_rows() -> None:
    rows = [
        _matrix_row("0", "1", support=10.0, target_nelbo=10.0, bacc=0.60, has_all_classes=1),
        _matrix_row("0", "2", support=1.0, target_nelbo=1.0, bacc=0.99, has_all_classes=0),
    ]
    alignment = build_family_c_pca64_cc_alignment_rows(rows=rows, candidate_domains=("0", "1", "2"))
    raw = [row for row in alignment if row["selector"] == PCA64_CC_RAW_SELECTOR][0]
    assert raw["selected_expert"] == "1"
    assert raw["downstream_oracle_expert"] == "1"


def test_decision_logic_uses_fixed_gain_and_routing_thresholds() -> None:
    config = default_family_c_pca64_cc_config()
    rows = [
        {
            "generation_mode": "pca64_real_reconstruction_upper",
            "status": "ok",
            "target_eval_has_all_classes": "1",
            "experiment_seed": 42,
            "heldout_center": "0",
            "support_size": 4,
            "support_seed": 17,
            "support_eval_split_id": "target0_seed17_random_k4",
            "generation_seed": 17,
            "classifier_seed": 17,
            "candidate_expert": "1",
            "bacc": 0.86,
        }
    ]
    no_gain = [
        {
            "selector": PCA64_CC_RAW_SELECTOR,
            "heldout_center": "0",
            "selected_bacc": 0.61,
            "selected_macro_f1": 0.61,
            "downstream_oracle_bacc": 0.62,
            "oracle_gap_bacc": 0.01,
            "top1_oracle_hit": 1,
            "oracle_agreement": 1,
            "available": 1,
        }
    ]
    summary = classify_family_c_pca64_cc_decision(rows=rows, alignment_rows=no_gain, config=config)
    assert summary["decision_classification"] == "NO_MEANINGFUL_GAIN"

    routing = [
        {
            "selector": PCA64_CC_RAW_SELECTOR,
            "heldout_center": "0",
            "selected_bacc": 0.70,
            "selected_macro_f1": 0.70,
            "downstream_oracle_bacc": 0.85,
            "oracle_gap_bacc": 0.15,
            "top1_oracle_hit": 0,
            "oracle_agreement": 0,
            "available": 1,
        }
    ]
    summary = classify_family_c_pca64_cc_decision(rows=rows, alignment_rows=routing, config=config)
    assert summary["decision_classification"] == "ROUTING_BOTTLENECK"


def test_class_conditional_resume_limits_use_existing_build_limits_type() -> None:
    limits = FamilyCPca64BuildLimits(
        experiment_seeds=(42,),
        heldout_centers=("0",),
        support_sizes=(4,),
        support_seeds=(17,),
        generation_seeds=(17,),
        classifier_seeds=(17,),
    )
    assert limits.experiment_seeds == (42,)
    assert limits.heldout_centers == ("0",)


def _cache(*, n_per_center: int, dim: int) -> EmbeddingCache:
    import numpy as np

    rows = []
    values = []
    for center in ("0", "1"):
        for idx in range(n_per_center):
            rows.append(
                {
                    "sample_id": f"c{center}_{idx}",
                    "center": center,
                    "label": str(idx % 2),
                }
            )
            base = float(center) * 10.0
            values.append(base + np.linspace(0.0, 1.0, dim) + (idx * 0.001))
    return EmbeddingCache(embeddings=np.asarray(values, dtype=float), metadata=tuple(rows))


def _matrix_row(
    heldout: str,
    expert: str,
    *,
    support: float,
    target_nelbo: float,
    bacc: float,
    has_all_classes: int,
) -> dict[str, object]:
    return {
        "schema_version": "family_c_pca64_class_conditional_cvae_downstream_v1",
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": "target0_seed17_random_k4",
        "candidate_expert": expert,
        "generation_mode": PCA64_CC_CVAE_MODE,
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "bacc": bacc,
        "macro_f1": bacc,
        "row_type": "single_expert_pca64_class_conditional_cvae",
        "support_nelbo_raw": support,
        "support_nelbo_source_prior": support,
        "support_nelbo_global_source_prior": support,
        "target_eval_nelbo_unlabeled": target_nelbo,
        "target_eval_has_all_classes": has_all_classes,
        "available": 1,
        "status": "ok",
    }
