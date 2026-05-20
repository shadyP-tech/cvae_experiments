from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.family_c_pca64 import (  # noqa: E402
    FAMILY_C_PCA64_NAME,
    PCA64_CHECKPOINT_FEATURE_SPACE,
    PCA64_RAW_SELECTOR,
    Pca64CvaeExpert,
    Pca64Preprocessor,
    _audit_row,
    assert_family_c_pca64_config_text,
    assert_pca64_checkpoint_metadata,
    build_family_c_pca64_alignment_rows,
    classify_family_c_pca64_decision,
    default_family_c_pca64_config,
    fit_source_train_pca64_preprocessor,
    inverse_transform_pca64,
    preprocessing_artifact_key,
    score_unlabeled_nelbo,
    source_normalized_nelbo,
    transform_pca64,
)
from cvae_downstream_evaluation.matrix import EmbeddingCache  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_family_c_pca64_config_exists_and_is_locked() -> None:
    path = ROOT / "configs" / "experiments" / "family_c_pca64_standardized_cvae_downstream_v1.yaml"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert_family_c_pca64_config_text(text)
    config = default_family_c_pca64_config()
    assert config.experiment_name == FAMILY_C_PCA64_NAME
    assert config.pca_dim == 64
    assert config.budget_per_class == 128


def test_pca_scaler_fits_only_matching_source_train_center_rows() -> None:
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


def test_pca_scaler_artifact_key_separates_center_seed_and_split() -> None:
    base = preprocessing_artifact_key(
        experiment_seed=42,
        source_center="0",
        feature_extractor="dino",
        split_id="source_train_seed42",
        pca_dim=64,
        standardized=True,
    )
    changed_center = preprocessing_artifact_key(
        experiment_seed=42,
        source_center="1",
        feature_extractor="dino",
        split_id="source_train_seed42",
        pca_dim=64,
        standardized=True,
    )
    changed_seed = preprocessing_artifact_key(
        experiment_seed=43,
        source_center="0",
        feature_extractor="dino",
        split_id="source_train_seed43",
        pca_dim=64,
        standardized=True,
    )
    assert len({base, changed_center, changed_seed}) == 3


def test_pca64_checkpoint_metadata_rejects_legacy_768d_checkpoint() -> None:
    expected = {
        "input_dim": 64,
        "feature_space": PCA64_CHECKPOINT_FEATURE_SPACE,
        "pca_artifact_id": "pca",
        "scaler_artifact_id": "scaler",
        "source_center": "2",
        "experiment_seed": 42,
    }
    assert_pca64_checkpoint_metadata(expected, expected=expected)
    legacy = dict(expected)
    legacy["input_dim"] = 768
    try:
        assert_pca64_checkpoint_metadata(legacy, expected=expected)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Legacy 768-D checkpoint metadata was accepted")


def test_transform_inverse_returns_original_dino_dimension() -> None:
    import numpy as np

    cache = _cache(n_per_center=70, dim=70)
    prep = fit_source_train_pca64_preprocessor(
        experiment_seed=42,
        source_center="0",
        train_cache=cache,
        feature_extractor="dino",
        split_id="source_train_seed42",
        pca_dim=64,
    )
    original = cache.embeddings[:5]
    z = transform_pca64(prep, original)
    reconstructed = inverse_transform_pca64(prep, z)
    assert z.shape == (5, 64)
    assert reconstructed.shape == (5, 70)
    assert np.isfinite(reconstructed).all()


def test_source_normalized_nelbo_rejects_tiny_std() -> None:
    value, available = source_normalized_nelbo(3.0, mean=1.0, std=2.0, eps=1e-8)
    assert available == 1
    assert value == 1.0
    value, available = source_normalized_nelbo(3.0, mean=1.0, std=1e-12, eps=1e-8)
    assert available == 0
    assert math.isnan(value)


def test_score_unlabeled_nelbo_uses_pca64_shape_and_mean_reductions() -> None:
    import torch

    class DummyModel:
        def eval(self) -> None:
            return None

        def encode(self, x):
            return torch.zeros((x.shape[0], 2), device=x.device), torch.zeros((x.shape[0], 2), device=x.device)

        def decode(self, z):
            return torch.zeros((z.shape[0], 64), device=z.device)

    x = torch.ones((3, 64), dtype=torch.float32)
    score = score_unlabeled_nelbo(DummyModel(), x, torch=torch, device=torch.device("cpu"), kl_beta=1.0)
    assert score.n_samples == 3
    assert score.recon == 1.0
    assert score.kl == 0.0
    assert score.total == 1.0

    try:
        score_unlabeled_nelbo(DummyModel(), torch.ones((3, 63)), torch=torch, device=torch.device("cpu"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("NELBO scorer accepted non-PCA64 inputs")


def test_raw_selector_and_oracles_are_reported_separately() -> None:
    rows = [
        _matrix_row("0", "1", support=10.0, target_nelbo=20.0, bacc=0.60),
        _matrix_row("0", "2", support=11.0, target_nelbo=5.0, bacc=0.90),
    ]
    alignment = build_family_c_pca64_alignment_rows(rows=rows, candidate_domains=("0", "1", "2"))
    raw = [row for row in alignment if row["selector"] == PCA64_RAW_SELECTOR][0]
    assert raw["selected_expert"] == "1"
    assert raw["downstream_oracle_expert"] == "2"
    assert raw["density_oracle_expert"] == "2"
    assert raw["oracle_agreement"] == 1
    assert raw["top1_oracle_hit"] == 0


def test_protocol_audit_carries_pca_cvae_classifier_lineage() -> None:
    prep = Pca64Preprocessor(
        experiment_seed=42,
        source_center="1",
        feature_extractor="dino",
        split_id="source_train_seed42",
        pca_dim=64,
        pca=object(),
        scaler=object(),
        pca_artifact_id="pca1",
        scaler_artifact_id="scaler1",
        artifact_key="prep1",
        pca_explained_variance_ratio_sum=0.9,
        pca_coord_mean_before_scaling=0.0,
        pca_coord_std_before_scaling=1.0,
        n_fit_samples=70,
        embedding_dim=768,
    )
    expert = Pca64CvaeExpert(
        experiment_seed=42,
        source_center="1",
        model=object(),
        preprocessor=prep,
        source_train_nelbo_mean=1.0,
        source_train_nelbo_std=0.5,
        checkpoint_path=Path("checkpoint.pt"),
        input_dim=64,
        hidden_dim=128,
        latent_dim=16,
        kl_beta=1.0,
    )
    row = _matrix_row("0", "1", support=10.0, target_nelbo=20.0, bacc=0.60)
    audit = _audit_row(
        row=row,
        expert=expert,
        target_expert_excluded=1,
        support_eval_disjoint=1,
        checkpoint_feature_space=PCA64_CHECKPOINT_FEATURE_SPACE,
    )
    assert audit["support_eval_split_id"] == row["support_eval_split_id"]
    assert audit["generation_seed"] == row["generation_seed"]
    assert audit["classifier_seed"] == row["classifier_seed"]
    assert audit["preprocessing_artifact_key"] == "prep1"
    assert audit["pca_artifact_id"] == "pca1"
    assert audit["scaler_artifact_id"] == "scaler1"
    assert audit["cvae_input_dim"] == 64
    assert audit["generated_embedding_dim"] == 768
    assert audit["lineage_key"]


def test_decision_logic_distinguishes_modeling_from_routing_bottleneck() -> None:
    align = [
        {
            "selector": PCA64_RAW_SELECTOR,
            "heldout_center": "0",
            "selected_bacc": 0.70,
            "downstream_oracle_bacc": 0.85,
            "oracle_gap_bacc": 0.15,
            "available": 1,
        }
    ]
    summary = classify_family_c_pca64_decision(rows=[], alignment_rows=align)
    assert summary["decision_classification"] == "PCA64_CVAE_ORACLE_STRONG_ROUTING_BOTTLENECK"


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


def _matrix_row(heldout: str, expert: str, *, support: float, target_nelbo: float, bacc: float) -> dict[str, object]:
    return {
        "schema_version": "family_c_pca64_standardized_cvae_downstream_v1",
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": "target0_seed17_random_k4",
        "candidate_expert": expert,
        "generation_mode": "family_c_pca64_standardized_cvae_reference_posterior_resampling",
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "bacc": bacc,
        "macro_f1": bacc,
        "row_type": "single_expert_pca64_cvae",
        "support_nelbo_raw": support,
        "support_nelbo_source_normalized": support,
        "target_eval_nelbo_unlabeled": target_nelbo,
        "available": 1,
        "status": "ok",
    }
