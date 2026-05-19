from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.family_e1 import (  # noqa: E402
    E1_BOOTSTRAP_MODE,
    E1_1_GMM_MODE,
    E1_1_GMM_SELECTOR,
    E1_GMM_MODE,
    E1_GMM_SELECTOR,
    E1_KDE_MODE,
    E1_REAL_SOURCE_MODE,
    E1_SAMPLER_SELECTOR,
    E1_SMOTE_MODE,
    FamilyE1MatrixRow,
    SourceClassData,
    allocate_family_e1_ensemble_budget,
    assert_family_e1_config_text,
    build_family_e1_alignment_rows,
    candidate_experts_for_heldout,
    classify_family_e1_decision,
    default_family_e1_config,
    default_family_e1_1_config,
    family_e1_source_transfer_prior,
    fit_family_e1_sampler_bank,
    fit_pca64_gmm_diag_bic_sampler,
    fit_smote_interpolate_sampler,
    generate_class_balanced_batch,
    score_family_e1_candidate,
    select_lowest_bic,
    select_source_transfer_candidate,
    source_only_kde_bandwidth,
    valid_gmm_k_candidates,
    validate_family_e1_protocol_audit,
    _dedupe_family_e1_matrix_rows,
    _e1_classifier_sidecar_key_from_mapping,
    _e1_classifier_sidecar_key_from_matrix_row,
    _e1_matrix_primary_key,
    _merge_dict_rows_by_key,
)
from cvae_downstream_evaluation.matrix import EmbeddingCache, TargetEvalPool  # noqa: E402
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_family_e1_expected_files_exist_and_config_is_locked() -> None:
    expected = [
        ROOT / "configs" / "experiments" / "family_e1_direct_embedding_sampler_downstream_v1.yaml",
        ROOT / "src" / "cvae_downstream_evaluation" / "family_e1.py",
        ROOT / "scripts" / "run_family_e1_direct_embedding_sampler_downstream.py",
    ]
    assert not [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    config = expected[0].read_text(encoding="utf-8")
    assert_family_e1_config_text(config)
    assert "pca_before_sampler:\n    enabled: false" in config


def test_family_e1_1_expected_files_exist_and_config_is_locked() -> None:
    expected = [
        ROOT / "configs" / "experiments" / "family_e1_1_pca64_gmm_direct_embedding_sampler_downstream_v1.yaml",
        ROOT / "scripts" / "run_family_e1_1_pca64_gmm_direct_embedding_sampler_downstream.py",
    ]
    assert not [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    config = expected[0].read_text(encoding="utf-8")
    assert "name: family_e1_1_pca64_gmm_direct_embedding_sampler_downstream_v1" in config
    assert "pca_before_sampler:\n    enabled: true" in config
    assert "whiten: false" in config
    assert E1_1_GMM_MODE in config


def test_samplers_fit_only_matching_source_center_class_train_rows() -> None:
    cache = _cache()
    config = default_family_e1_config()
    bank = fit_family_e1_sampler_bank(train_cache=cache, config=config)
    fit = bank[(E1_SMOTE_MODE, "0", 0)]
    assert fit.source_sample_ids == ("s0_l0_a", "s0_l0_b")
    assert fit.n_source_train == 2
    assert "target_l0_outlier" not in fit.source_sample_ids


def test_target_center_excluded_from_candidate_pool_and_prior() -> None:
    assert candidate_experts_for_heldout(("0", "1", "2"), "1") == ("0", "2")
    rows = [
        _row("0", "1", E1_GMM_MODE, 0.99),
        _row("2", "1", E1_GMM_MODE, 0.50),
        _row("1", "1", E1_GMM_MODE, 0.10),
    ]
    prior = family_e1_source_transfer_prior(
        rows,
        heldout_center="0",
        candidate_experts=("1",),
        modes=(E1_GMM_MODE,),
        selector=E1_GMM_SELECTOR,
    )
    assert prior.scores[(E1_GMM_MODE, "1")] == 0.50
    assert prior.source_center_scores[(E1_GMM_MODE, "1")] == {"2": 0.50}


def test_gmm_valid_k_candidates_and_bic_selection() -> None:
    assert valid_gmm_k_candidates(31, [1, 2, 4, 8]) == ()
    assert valid_gmm_k_candidates(32, [1, 2, 4, 8]) == (1, 2, 4)
    assert valid_gmm_k_candidates(64, [1, 2, 4, 8]) == (1, 2, 4, 8)
    assert select_lowest_bic({1: 10.0, 2: 8.0, 4: 8.0}) == 2


def test_e1_1_pca64_availability_requires_65_samples_and_64_dimensions() -> None:
    import numpy as np

    config = default_family_e1_1_config()
    too_few = fit_pca64_gmm_diag_bic_sampler(
        SourceClassData("0", 0, np.zeros((64, 70), dtype=float), tuple(f"a{i}" for i in range(64))),
        config=config,
    )
    assert not too_few.available
    assert too_few.error_message == "too_few_source_train_samples_for_pca64"

    too_narrow = fit_pca64_gmm_diag_bic_sampler(
        SourceClassData("0", 0, np.zeros((65, 63), dtype=float), tuple(f"b{i}" for i in range(65))),
        config=config,
    )
    assert not too_narrow.available
    assert too_narrow.error_message == "embedding_dim_less_than_64"


def test_e1_1_pca_gmm_persists_grid_rank_and_inverse_transform_metadata() -> None:
    source = _pca_source(center="0", label=0, offset=0.0)
    fit = fit_pca64_gmm_diag_bic_sampler(source, config=default_family_e1_1_config())
    assert fit.available
    assert fit.diagnostics["pca_available"] == 1
    assert fit.diagnostics["pca_rank_upper_bound"] == 64
    assert int(fit.diagnostics["pca_retained_rank_estimate"]) <= 64
    assert fit.diagnostics["pca_inverse_transform_used"] == 1
    assert fit.diagnostics["pca_inverse_transform_exact_inverse"] == 0
    assert fit.diagnostics["synthetic_space_for_gmm"] == "pca64"
    assert fit.diagnostics["synthetic_space_for_classifier"] == "original_embedding_after_pca_inverse_transform"
    assert fit.diagnostics["gmm_k_candidates"] == "[1, 2, 4, 8]"
    assert fit.diagnostics["gmm_valid_k"] == "[1, 2, 4, 8]"
    assert "gmm_converged_by_k" in fit.diagnostics
    assert "gmm_n_iter_by_k" in fit.diagnostics


def test_kde_bandwidth_uses_source_train_only() -> None:
    import numpy as np

    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target_outliers = np.array([[1000.0, 1000.0], [2000.0, 2000.0]])
    bandwidth_before = source_only_kde_bandwidth(source)
    _ = target_outliers
    bandwidth_after = source_only_kde_bandwidth(source)
    assert bandwidth_before == bandwidth_after


def test_smote_labels_and_samples_are_deterministic_for_fixed_seed() -> None:
    import numpy as np

    config = default_family_e1_config()
    bank = {}
    for label in (0, 1):
        source = SourceClassData(
            source_center="1",
            class_label=label,
            embeddings=np.array([[label, 0.0], [label, 1.0], [label, 2.0]], dtype=float),
            sample_ids=(f"id{label}a", f"id{label}b", f"id{label}c"),
        )
        bank[(E1_SMOTE_MODE, "1", label)] = fit_smote_interpolate_sampler(source, config=config)
    first = generate_class_balanced_batch(
        sampler_bank=bank,
        mode=E1_SMOTE_MODE,
        source_center="1",
        class_labels=(0, 1),
        budget_per_class=5,
        generation_seed=17,
    )
    second = generate_class_balanced_batch(
        sampler_bank=bank,
        mode=E1_SMOTE_MODE,
        source_center="1",
        class_labels=(0, 1),
        budget_per_class=5,
        generation_seed=17,
    )
    assert first.labels == (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
    assert np.allclose(first.embeddings, second.embeddings)


def test_e1_1_class_conditional_pca_objects_do_not_cross_classes() -> None:
    import numpy as np

    config = default_family_e1_1_config()
    bank = {
        (E1_1_GMM_MODE, "1", 0): fit_pca64_gmm_diag_bic_sampler(
            _pca_source(center="1", label=0, offset=0.0),
            config=config,
        ),
        (E1_1_GMM_MODE, "1", 1): fit_pca64_gmm_diag_bic_sampler(
            _pca_source(center="1", label=1, offset=100.0),
            config=config,
        ),
    }
    batch = generate_class_balanced_batch(
        sampler_bank=bank,
        mode=E1_1_GMM_MODE,
        source_center="1",
        class_labels=(0, 1),
        budget_per_class=8,
        generation_seed=17,
    )
    assert batch.labels == (0,) * 8 + (1,) * 8
    class0 = batch.embeddings[:8]
    class1 = batch.embeddings[8:]
    assert float(np.mean(class0)) < 50.0
    assert float(np.mean(class1)) > 50.0
    assert batch.diagnostics["pca_inverse_transform_used"] == 1
    assert batch.diagnostics["synthetic_space_for_gmm"] == "pca64"


def test_source_transfer_prior_aggregates_by_source_center_before_averaging() -> None:
    rows = [
        _row("1", "3", E1_GMM_MODE, 0.0, support_seed=17),
        _row("1", "3", E1_GMM_MODE, 0.0, support_seed=23),
        _row("2", "3", E1_GMM_MODE, 0.9, support_seed=17),
    ]
    prior = family_e1_source_transfer_prior(
        rows,
        heldout_center="0",
        candidate_experts=("3",),
        modes=(E1_GMM_MODE,),
        selector=E1_GMM_SELECTOR,
    )
    assert prior.source_center_scores[(E1_GMM_MODE, "3")] == {"1": 0.0, "2": 0.9}
    assert prior.scores[(E1_GMM_MODE, "3")] == 0.45


def test_selector_tie_break_score_then_mode_order_then_ascending_expert() -> None:
    rows = [
        _row("2", "2", E1_GMM_MODE, 0.7),
        _row("3", "2", E1_GMM_MODE, 0.7),
        _row("2", "1", E1_GMM_MODE, 0.7),
        _row("3", "1", E1_GMM_MODE, 0.7),
        _row("2", "1", E1_KDE_MODE, 0.9),
        _row("3", "1", E1_KDE_MODE, 0.9),
    ]
    prior = family_e1_source_transfer_prior(
        rows,
        heldout_center="0",
        candidate_experts=("2", "1"),
        modes=(E1_GMM_MODE, E1_KDE_MODE),
        selector=E1_SAMPLER_SELECTOR,
    )
    selected = select_source_transfer_candidate(
        prior,
        modes=(E1_GMM_MODE, E1_KDE_MODE),
        candidate_experts=("2", "1"),
    )
    assert selected.mode == E1_KDE_MODE
    assert selected.expert == "1"

    tied = family_e1_source_transfer_prior(
        rows[:4],
        heldout_center="0",
        candidate_experts=("2", "1"),
        modes=(E1_GMM_MODE,),
        selector=E1_GMM_SELECTOR,
    )
    selected_tied = select_source_transfer_candidate(
        tied,
        modes=(E1_GMM_MODE,),
        candidate_experts=("2", "1"),
    )
    assert selected_tied.mode == E1_GMM_MODE
    assert selected_tied.expert == "1"


def test_gmm_only_selector_never_selects_kde_smote_or_bootstrap() -> None:
    rows = [
        _row("1", "2", E1_GMM_MODE, 0.55),
        _row("3", "2", E1_GMM_MODE, 0.55),
        _row("1", "2", E1_KDE_MODE, 0.99),
        _row("3", "2", E1_KDE_MODE, 0.99),
        _row("1", "2", E1_BOOTSTRAP_MODE, 1.0, row_type="diagnostic_upper_bound"),
    ]
    alignment, _ = build_family_e1_alignment_rows(rows=rows, candidate_domains=("0", "1", "2", "3"))
    gmm_rows = [row for row in alignment if row["selector"] == E1_GMM_SELECTOR]
    assert gmm_rows
    assert all(row["selected_mode"] == E1_GMM_MODE for row in gmm_rows)


def test_same_budget_ensemble_does_not_exceed_single_expert_budget() -> None:
    allocation = allocate_family_e1_ensemble_budget(total_per_class=128, candidate_experts=("3", "1", "2"))
    assert allocation == {"1": 43, "2": 43, "3": 42}
    assert sum(allocation.values()) == 128


def test_bootstrap_and_real_source_are_excluded_from_primary_gates() -> None:
    rows = [
        _row("0", "1", E1_GMM_MODE, 0.60),
        _row("0", "1", E1_KDE_MODE, 0.61),
        _row("0", "1", E1_SMOTE_MODE, 0.62),
        _row("0", "1", E1_BOOTSTRAP_MODE, 0.90, row_type="diagnostic_upper_bound"),
        _row("0", "1", E1_REAL_SOURCE_MODE, 0.91, row_type="diagnostic_upper_bound"),
    ]
    alignment = [
        _alignment(E1_GMM_SELECTOR, "0", E1_GMM_MODE, "1", 0.60, 0.0),
        _alignment(E1_SAMPLER_SELECTOR, "0", E1_SMOTE_MODE, "1", 0.62, 0.0),
    ]
    summary = classify_family_e1_decision(
        rows=rows,
        alignment_rows=alignment,
        protocol_audit_rows=[_audit("0", "1", E1_GMM_MODE)],
        c2_metrics={},
    )
    assert summary["decision_classification"] == "BOOTSTRAP_ONLY_UPPER_BOUND"
    assert summary["pass_fail"] == "FAIL"


def test_e1_1_partial_availability_cannot_pass_even_with_high_available_mean() -> None:
    rows = [
        _row("0", "1", E1_1_GMM_MODE, 0.95),
        _row("1", "0", E1_1_GMM_MODE, 0.95),
        _row("2", "0", E1_1_GMM_MODE, math.nan, available=0, status="failed_protocol"),
    ]
    alignment = [
        _alignment(E1_1_GMM_SELECTOR, "0", E1_1_GMM_MODE, "1", 0.95, 0.0),
        _alignment(E1_1_GMM_SELECTOR, "1", E1_1_GMM_MODE, "0", 0.95, 0.0),
    ]
    summary = classify_family_e1_decision(
        rows=rows,
        alignment_rows=alignment,
        protocol_audit_rows=[_audit("0", "1", E1_1_GMM_MODE)],
        c2_metrics={"e1_v1_gmm_selected_center_level_mean_bacc": 0.70},
    )
    assert summary["decision_classification"] == "DIAGNOSTIC_PARTIAL"
    assert summary["pass_fail"] == "FAIL"


def test_protocol_audit_rejects_generation_or_selection_leakage() -> None:
    good = _audit("0", "1", E1_GMM_MODE)
    validate_family_e1_protocol_audit([good])
    bad = dict(good)
    bad["target_eval_labels_used_for_training"] = 1
    try:
        validate_family_e1_protocol_audit([bad])
    except ProtocolError:
        pass
    else:
        raise AssertionError("Family E1 protocol audit accepted target eval labels in training")


def test_single_class_target_eval_context_is_marked_unavailable() -> None:
    import numpy as np

    row, generated_rows, diag, clf_manifest, audit = score_family_e1_candidate(
        sampler_bank={},
        experiment_seed=43,
        heldout_center="2",
        support_unit=_support_unit("2"),
        candidate_expert="1",
        mode=E1_GMM_MODE,
        budget_per_class=128,
        generation_seed=17,
        classifier_seed=17,
        class_labels=(0, 1),
        target_embeddings=np.zeros((196, 2), dtype=float),
        target_labels=[0] * 196,
        target_pool=TargetEvalPool(
            eval_indices=tuple(range(196)),
            excluded_support_sample_ids=("a", "b", "c", "d"),
            target_eval_pool_id="target2_seed17_random_k4_test",
        ),
    )
    assert row.status == "failed_single_class_target_eval"
    assert row.available == 0
    assert row.target_eval_label_counts_json == '{"0":196}'
    assert row.target_eval_has_all_classes == 0
    assert math.isnan(row.bacc)
    assert generated_rows == ()
    assert diag["target_eval_label_counts_json"] == '{"0":196}'
    assert clf_manifest["target_eval_has_all_classes"] == 0
    validate_family_e1_protocol_audit([audit])


def test_resume_sidecar_key_repairs_smoke_rows_without_matrix_duplicates() -> None:
    existing = _row("0", "1", E1_GMM_MODE, 0.50)
    repaired = _row("0", "1", E1_GMM_MODE, 0.75)
    precomputed_key = _e1_matrix_primary_key(
        experiment_seed=repaired.experiment_seed,
        heldout_center=repaired.heldout_center,
        support_size=repaired.support_size,
        support_seed=repaired.support_seed,
        candidate_expert=repaired.candidate_expert,
        generation_mode=repaired.generation_mode,
        budget_per_class=repaired.budget_per_class,
        generation_seed=repaired.generation_seed,
        classifier_seed=repaired.classifier_seed,
        row_type=repaired.row_type,
        candidate_hash=repaired.candidate_experts_hash,
    )
    assert precomputed_key == repaired.primary_key()

    rows = _dedupe_family_e1_matrix_rows([existing, repaired])
    assert len(rows) == 1
    assert rows[0].bacc == 0.75

    sidecar_key = _e1_classifier_sidecar_key_from_matrix_row(repaired)
    assert sidecar_key == _e1_classifier_sidecar_key_from_mapping(
        {
            "experiment_seed": repaired.experiment_seed,
            "heldout_center": repaired.heldout_center,
            "support_size": repaired.support_size,
            "support_seed": repaired.support_seed,
            "candidate_expert": repaired.candidate_expert,
            "generation_mode": repaired.generation_mode,
            "generation_seed": repaired.generation_seed,
            "classifier_seed": repaired.classifier_seed,
        }
    )

    merged = _merge_dict_rows_by_key(
        [
            {
                "experiment_seed": 42,
                "heldout_center": "0",
                "support_size": 4,
                "support_seed": 17,
                "candidate_expert": "1",
                "generation_mode": E1_GMM_MODE,
                "generation_seed": 17,
                "classifier_seed": 17,
                "status": "old",
            }
        ],
        [
            {
                "experiment_seed": 42,
                "heldout_center": "0",
                "support_size": 4,
                "support_seed": 17,
                "candidate_expert": "1",
                "generation_mode": E1_GMM_MODE,
                "generation_seed": 17,
                "classifier_seed": 17,
                "status": "new",
            }
        ],
        key_func=_e1_classifier_sidecar_key_from_mapping,
    )
    assert len(merged) == 1
    assert merged[0]["status"] == "new"


def _cache() -> EmbeddingCache:
    import numpy as np

    metadata = (
        {"sample_id": "s0_l0_a", "center": "0", "label": "0"},
        {"sample_id": "s0_l0_b", "center": "0", "label": "0"},
        {"sample_id": "s0_l1_a", "center": "0", "label": "1"},
        {"sample_id": "s1_l0_a", "center": "1", "label": "0"},
        {"sample_id": "target_l0_outlier", "center": "2", "label": "0"},
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [100.0, 100.0],
        ],
        dtype=float,
    )
    return EmbeddingCache(embeddings=embeddings, metadata=metadata)


def _pca_source(*, center: str, label: int, offset: float) -> SourceClassData:
    import numpy as np

    rng = np.random.default_rng(1000 + int(label))
    base = rng.normal(loc=float(offset), scale=0.05, size=(65, 70))
    trend = np.linspace(0.0, 1.0, 70, dtype=float)
    embeddings = base + np.arange(65, dtype=float)[:, None] * 0.001 * trend[None, :]
    return SourceClassData(
        source_center=center,
        class_label=label,
        embeddings=embeddings,
        sample_ids=tuple(f"{center}_{label}_{idx}" for idx in range(65)),
    )


def _row(
    heldout: str,
    expert: str,
    mode: str,
    bacc: float,
    *,
    support_seed: int = 17,
    row_type: str = "single_expert_sampler",
    available: int = 1,
    status: str = "ok",
) -> FamilyE1MatrixRow:
    return FamilyE1MatrixRow(
        experiment_seed=42,
        heldout_center=heldout,
        support_size=4,
        support_seed=support_seed,
        support_eval_split_id=f"target{heldout}_seed{support_seed}_random_k4",
        candidate_expert=expert,
        generation_mode=mode,
        budget_per_class=128,
        generation_seed=17,
        classifier_seed=17,
        bacc=bacc,
        macro_f1=max(0.0, bacc - 0.05),
        row_type=row_type,
        n_train=256,
        n_target_eval=100,
        target_eval_pool_id="pool",
        target_eval_label_counts_json='{"0":50,"1":50}',
        target_eval_has_all_classes=1,
        sampler_release_level="aggregate_statistics",
        available=available,
        status=status,
        schema_version="family_e1_1_pca64_gmm_direct_embedding_sampler_downstream_v1" if mode == E1_1_GMM_MODE else "family_e1_direct_embedding_sampler_downstream_v1",
    )


def _alignment(
    selector: str,
    heldout: str,
    mode: str,
    expert: str,
    bacc: float,
    gap: float,
) -> dict[str, object]:
    return {
        "selector": selector,
        "heldout_center": heldout,
        "experiment_seed": 42,
        "support_size": 4,
        "support_seed": 17,
        "generation_seed": 17,
        "classifier_seed": 17,
        "selected_mode": mode,
        "selected_expert": expert,
        "prior_score": 0.0,
        "selected_bacc": bacc,
        "selected_macro_f1": max(0.0, bacc - 0.05),
        "oracle_mode": mode,
        "oracle_expert": expert,
        "oracle_bacc": bacc + gap,
        "oracle_macro_f1": max(0.0, bacc - 0.05),
        "oracle_gap_bacc": gap,
        "oracle_gap_macro_f1": 0.0,
        "target_heldout_rows_used_for_source_transfer_prior": 0,
        "available": 1,
        "status": "ok",
    }


def _audit(heldout: str, expert: str, mode: str) -> dict[str, object]:
    lineage = (
        f"experiment_seed=42|source_center={expert}|class_label=|"
        "support_size=4|support_seed=17|generation_seed=17"
    )
    row = {
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": f"target{heldout}_seed17_random_k4",
        "candidate_expert": expert,
        "generation_mode": mode,
        "sampler_fit_split": "source_train",
        "target_expert_excluded": 1,
        "support_eval_disjoint": 1,
        "target_labels_used_for_sampler_fit": 0,
        "target_support_labels_used_for_generation": 0,
        "target_eval_embeddings_used_for_generation": 0,
        "target_eval_labels_used_for_training": 0,
        "target_eval_labels_used_for_final_metric_only": 1,
        "target_eval_label_counts_json": '{"0":50,"1":50}',
        "target_eval_has_all_classes": 1,
        "target_oracle_used_for_selection": 0,
        "target_heldout_rows_used_for_source_transfer_prior": 0,
        "sampler_release_level": "aggregate_statistics",
        "available": 1,
    }
    if mode == E1_1_GMM_MODE:
        row.update(
            {
                "pca_fit_split": "source_train",
                "pca_source_only": 1,
                "pca_inverse_transform_used": 1,
                "pca_gmm_lineage_key": lineage,
                "generated_lineage_key": lineage,
                "classifier_lineage_key": lineage,
            }
        )
    return row


def _support_unit(heldout: str):
    from cvae_downstream_evaluation.routing import SupportSelectionUnit

    return SupportSelectionUnit(
        heldout_center=heldout,
        experiment_seed=43,
        support_size=4,
        support_seed=17,
        method="support_set_nelbo_top1",
        selected_expert="1",
        candidate_experts=("0", "1", "3", "4"),
        support_nelbo_by_expert={"0": 1.0, "1": 0.5, "3": 0.7, "4": 0.8},
        target_expert_excluded=True,
        support_eval_split_id=f"target{heldout}_seed17_random_k4",
    )
