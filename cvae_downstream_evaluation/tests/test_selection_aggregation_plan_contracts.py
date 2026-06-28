from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.artifacts import (  # noqa: E402
    FrozenProtocolSnapshot,
    assert_frozen_snapshot_exists,
    assert_selection_lineage,
    write_frozen_snapshot,
)
from cvae_downstream_evaluation.baselines import (  # noqa: E402
    assert_deployable_candidate_pool,
    assert_oracle_rows_diagnostic_only,
)
from cvae_downstream_evaluation.compatibility import (  # noqa: E402
    CompatibilityPrediction,
    assert_source_inner_training_labels,
    select_top1,
    softmax_weights,
    topk_uniform,
)
from cvae_downstream_evaluation.compatibility.select_candidates import (  # noqa: E402
    build_top1_selection_rows,
)
from cvae_downstream_evaluation.compatibility.diagnostics import estimator_diagnostics  # noqa: E402
from cvae_downstream_evaluation.compatibility.estimators import (  # noqa: E402
    load_estimator,
    predict_rows,
    save_estimator,
)
from cvae_downstream_evaluation.compatibility.train_source_inner import (  # noqa: E402
    train_linear_utility_estimator,
)
from cvae_downstream_evaluation.compatibility.pipeline import (  # noqa: E402
    LearnedUtilityPipelineInputs,
    run_learned_utility_pipeline,
)
from cvae_downstream_evaluation.compatibility.legacy_adapters import normalize_c52_legacy_artifacts  # noqa: E402
from cvae_downstream_evaluation.compatibility.legacy_adapters import (  # noqa: E402
    discover_c52_contexts,
    run_c52_legacy_batch,
)
from cvae_downstream_evaluation.downstream import (  # noqa: E402
    CandidateDownstreamRow,
    write_candidate_downstream_matrix,
)
from cvae_downstream_evaluation.features import (  # noqa: E402
    assert_allowed_feature_table,
    deployable_feature_columns,
)
from cvae_downstream_evaluation.features.feature_table_builder import (  # noqa: E402
    build_allowed_feature_table_from_artifacts,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.reports.leakage_report import build_leakage_report  # noqa: E402
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    SELECTION_ELIGIBLE,
)
from cvae_downstream_evaluation.utility_matrix import (  # noqa: E402
    assert_diagnostic_matrix_path,
    assert_selection_does_not_read_matrix,
)
from cvae_downstream_evaluation.reports.rank_metrics import (  # noqa: E402
    build_learned_utility_alignment_rows,
)


def test_candidate_pool_excludes_heldout_target_for_deployable_rows() -> None:
    rows = [
        _candidate("c1", source_domain="1", eligibility=SELECTION_ELIGIBLE),
        _candidate("c0_oracle", source_domain="0", eligibility=DIAGNOSTIC_ONLY, role="oracle_reference"),
    ]
    assert_deployable_candidate_pool(heldout_target="0", candidate_rows=rows)
    assert_oracle_rows_diagnostic_only(rows)

    leaked = rows + [_candidate("leak", source_domain="0", eligibility=SELECTION_ELIGIBLE)]
    try:
        assert_deployable_candidate_pool(heldout_target="0", candidate_rows=leaked)
    except ProtocolError:
        pass
    else:
        raise AssertionError("deployable target expert leakage was not rejected")


def test_feature_table_rejects_target_identity_and_downstream_metrics() -> None:
    good = _feature_row(candidate_id="c1") | {
        "feature_source": "target_support_only",
        "support_nelbo": 12.3,
    }
    assert_allowed_feature_table([good])
    assert deployable_feature_columns(good.keys()) == ("support_nelbo",)

    bad = good | {"target_domain_id": "0", "bacc": 0.91}
    try:
        assert_allowed_feature_table([bad])
    except ProtocolError:
        pass
    else:
        raise AssertionError("target identity/downstream utility feature leak was not rejected")


def test_estimator_training_requires_source_inner_pseudo_target_labels() -> None:
    assert_source_inner_training_labels(
        [{"fold_role": "source_inner_pseudo_target", "source_inner_heldout_bacc": 0.74}]
    )
    try:
        assert_source_inner_training_labels(
            [{"fold_role": "source_inner_pseudo_target", "target_bacc": 0.99}]
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("real held-out target downstream label was not rejected")


def test_linear_source_inner_estimator_trains_saves_loads_and_predicts(tmp_path: Path) -> None:
    train_rows = [
        {
            "fold_role": "source_inner_pseudo_target",
            "support_nelbo": 4.0,
            "source_inner_candidate_stability": 0.1,
            "source_inner_heldout_bacc": 0.60,
        },
        {
            "fold_role": "source_inner_pseudo_target",
            "support_nelbo": 3.0,
            "source_inner_candidate_stability": 0.3,
            "source_inner_heldout_bacc": 0.70,
        },
        {
            "fold_role": "source_inner_pseudo_target",
            "support_nelbo": 2.0,
            "source_inner_candidate_stability": 0.5,
            "source_inner_heldout_bacc": 0.82,
        },
    ]
    estimator = train_linear_utility_estimator(
        train_rows,
        feature_columns=("support_nelbo", "source_inner_candidate_stability"),
        ridge_lambda=1e-3,
    )
    low = estimator.predict_one({"support_nelbo": 4.0, "source_inner_candidate_stability": 0.1})
    high = estimator.predict_one({"support_nelbo": 2.0, "source_inner_candidate_stability": 0.5})
    assert high > low

    model_path = tmp_path / "model.json"
    save_estimator(model_path, estimator)
    loaded = load_estimator(model_path)
    assert loaded.feature_columns == estimator.feature_columns
    predicted = predict_rows(loaded, train_rows)
    diagnostics = estimator_diagnostics(predicted)
    assert diagnostics["n_rows"] == 3.0
    assert diagnostics["spearman_predicted_vs_observed"] > 0.9

    try:
        train_linear_utility_estimator(
            [train_rows[0] | {"target_bacc": 0.99}],
            feature_columns=("support_nelbo",),
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("linear estimator accepted real target downstream label")


def test_diagnostic_matrix_firewall() -> None:
    assert_diagnostic_matrix_path(Path("matrices/diagnostic_downstream_utility.csv"))
    try:
        assert_selection_does_not_read_matrix(Path("matrices/diagnostic_downstream_utility.csv"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("selection access to diagnostic matrix was not rejected")

    try:
        assert_diagnostic_matrix_path(Path("features/diagnostic_downstream_utility.csv"))
    except ProtocolError:
        pass
    else:
        raise AssertionError("diagnostic matrix under feature path was not rejected")


def test_frozen_snapshot_and_lineage(tmp_path: Path) -> None:
    snapshot = FrozenProtocolSnapshot(
        candidate_pool_hash="pool",
        generation_config_hash="gen",
        classifier_config_hash="clf",
        metric_config_hash="metric",
        feature_config_hash="feat",
        routing_config_hash="route",
    )
    path = tmp_path / "configs" / "frozen_protocol_snapshot.json"
    write_frozen_snapshot(path, snapshot)
    assert_frozen_snapshot_exists(path)

    feature = _feature_row(candidate_id="c1")
    selection = feature | {"predicted_primary_utility": 0.8}
    candidate = _candidate("c1", source_domain="1", eligibility=SELECTION_ELIGIBLE)
    assert_selection_lineage(selection_rows=[selection], feature_rows=[feature], candidate_rows=[candidate])

    duplicate_features = [feature, feature | {"support_nelbo": 3.0}]
    try:
        assert_selection_lineage(
            selection_rows=[selection],
            feature_rows=duplicate_features,
            candidate_rows=[candidate],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("duplicate feature lineage was not rejected")


def test_compatibility_selection_and_aggregation_rules_are_deterministic() -> None:
    predictions = [
        CompatibilityPrediction("c1", predicted_primary_utility=0.7, support_nelbo=10.0, source_inner_stability=0.5),
        CompatibilityPrediction("c2", predicted_primary_utility=0.8, support_nelbo=12.0, source_inner_stability=0.2),
        CompatibilityPrediction("c3", predicted_primary_utility=0.8, support_nelbo=9.0, source_inner_stability=0.4),
    ]
    assert select_top1(predictions).candidate_id == "c3"
    assert topk_uniform(predictions, k=2) == {"c3": 0.5, "c2": 0.5}

    weights = softmax_weights(predictions, tau=0.5)
    assert set(weights) == {"c1", "c2", "c3"}
    assert math.isclose(sum(weights.values()), 1.0)

    fallback = softmax_weights(
        [CompatibilityPrediction("c1", math.nan, support_nelbo=2.0, source_inner_stability=0.0)],
        tau=1.0,
    )
    assert fallback == {"c1": 1.0}


def test_top1_selection_uses_allowed_features_before_report_time_utility_join() -> None:
    rows = [
        _feature_row(candidate_id="c1", expert_checkpoint_id="expert_1")
        | {
            "feature_source": "target_support_only",
            "predicted_primary_utility": 0.71,
            "support_nelbo": 5.0,
            "source_inner_stability": 0.3,
        },
        _feature_row(candidate_id="c2", expert_checkpoint_id="expert_2")
        | {
            "feature_source": "source_inner_only",
            "predicted_primary_utility": 0.81,
            "support_nelbo": 6.0,
            "source_inner_stability": 0.2,
        },
    ]
    selections = build_top1_selection_rows(rows)
    assert len(selections) == 1
    assert selections[0]["candidate_id"] == "c2"
    assert "selected_bacc" not in selections[0]


def test_learned_utility_alignment_joins_downstream_matrix_only_for_reporting() -> None:
    selection = _feature_row(candidate_id="c2", expert_checkpoint_id="expert_2") | {
        "method": "learned_downstream_utility_top1",
        "predicted_primary_utility": 0.81,
        "support_nelbo": 6.0,
        "source_inner_stability": 0.2,
        "selection_rank": 1,
        "aggregation_weight": 1.0,
    }
    downstream = [
        _downstream_row(candidate_expert="expert_1", bacc=0.88, macro_f1=0.80),
        _downstream_row(candidate_expert="expert_2", bacc=0.76, macro_f1=0.73),
    ]
    aligned = build_learned_utility_alignment_rows(selection_rows=[selection], downstream_rows=downstream)
    assert len(aligned) == 1
    assert aligned[0]["selected_bacc"] == 0.76
    assert aligned[0]["oracle_bacc"] == 0.88
    assert aligned[0]["top1_downstream_oracle_hit"] == 0
    assert aligned[0]["downstream_oracle_gap_bacc"] > 0


def test_artifact_smoke_path_builds_features_selections_alignment_and_leakage_report() -> None:
    candidates = [
        _candidate_manifest_row("c1", source_domain="1", expert_checkpoint_id="expert_1"),
        _candidate_manifest_row("c2", source_domain="2", expert_checkpoint_id="expert_2"),
    ]
    support_features = [
        _feature_artifact_row("c1") | {"support_nelbo": 6.0},
        _feature_artifact_row("c2", expert_checkpoint_id="expert_2") | {"support_nelbo": 5.0},
    ]
    source_inner = [
        _feature_artifact_row("c1") | {
            "predicted_primary_utility": 0.70,
            "source_inner_stability": 0.3,
        },
        _feature_artifact_row("c2", expert_checkpoint_id="expert_2") | {
            "predicted_primary_utility": 0.83,
            "source_inner_stability": 0.5,
        },
    ]
    feature_rows = build_allowed_feature_table_from_artifacts(
        candidate_rows=candidates,
        support_feature_rows=support_features,
        source_inner_rows=source_inner,
    )
    estimator = train_linear_utility_estimator(
        [
            {
                "fold_role": "source_inner_pseudo_target",
                "support_nelbo": 6.0,
                "source_inner_stability": 0.3,
                "source_inner_heldout_bacc": 0.70,
            },
            {
                "fold_role": "source_inner_pseudo_target",
                "support_nelbo": 5.0,
                "source_inner_stability": 0.5,
                "source_inner_heldout_bacc": 0.83,
            },
        ],
        feature_columns=("support_nelbo", "source_inner_stability"),
        ridge_lambda=1e-3,
    )
    feature_rows = tuple(predict_rows(estimator, feature_rows))
    selections = build_top1_selection_rows(feature_rows)
    assert selections[0]["candidate_id"] == "c2"

    second_seed_rows = [dict(row) | {"experiment_seed": 1, "candidate_id": f"{row['candidate_id']}_seed1"} for row in feature_rows]
    multi_seed = build_top1_selection_rows(tuple(feature_rows) + tuple(second_seed_rows))
    assert len(multi_seed) == 2

    downstream = [
        _downstream_row(candidate_expert="expert_1", bacc=0.80, macro_f1=0.78),
        _downstream_row(candidate_expert="expert_2", bacc=0.86, macro_f1=0.82),
    ]
    aligned = build_learned_utility_alignment_rows(selection_rows=selections, downstream_rows=downstream)
    assert aligned[0]["top1_downstream_oracle_hit"] == 1

    report = build_leakage_report(
        candidate_rows=candidates,
        feature_rows=feature_rows,
        selection_rows=selections,
        frozen_generation=True,
        frozen_classifier=True,
    )
    assert report["support_eval_overlap"] is False
    assert report["selection_read_downstream_matrix"] is False


def test_learned_utility_pipeline_writes_expected_artifacts(tmp_path: Path) -> None:
    candidates_path = tmp_path / "inputs" / "candidates.csv"
    support_path = tmp_path / "inputs" / "support.csv"
    source_inner_features_path = tmp_path / "inputs" / "source_inner_features.csv"
    training_path = tmp_path / "inputs" / "source_inner_training.csv"
    matrix_path = tmp_path / "tables" / "diagnostic_downstream_utility.csv"

    _write_rows(
        candidates_path,
        [
            _candidate_manifest_row("c1", source_domain="1", expert_checkpoint_id="expert_1"),
            _candidate_manifest_row("c2", source_domain="2", expert_checkpoint_id="expert_2"),
        ],
    )
    _write_rows(
        support_path,
        [
            _feature_artifact_row("c1") | {"support_nelbo": 6.0},
            _feature_artifact_row("c2", expert_checkpoint_id="expert_2") | {"support_nelbo": 5.0},
        ],
    )
    _write_rows(
        source_inner_features_path,
        [
            _feature_artifact_row("c1") | {"source_inner_stability": 0.3},
            _feature_artifact_row("c2", expert_checkpoint_id="expert_2") | {"source_inner_stability": 0.5},
        ],
    )
    _write_rows(
        training_path,
        [
            {
                "fold_role": "source_inner_pseudo_target",
                "support_nelbo": 6.0,
                "source_inner_stability": 0.3,
                "source_inner_heldout_bacc": 0.70,
            },
            {
                "fold_role": "source_inner_pseudo_target",
                "support_nelbo": 5.0,
                "source_inner_stability": 0.5,
                "source_inner_heldout_bacc": 0.84,
            },
            {
                "fold_role": "source_inner_pseudo_target",
                "support_nelbo": 7.0,
                "source_inner_stability": 0.1,
                "source_inner_heldout_bacc": 0.60,
            },
        ],
    )
    write_candidate_downstream_matrix(
        matrix_path,
        [
            _downstream_row(candidate_expert="expert_1", bacc=0.78, macro_f1=0.75),
            _downstream_row(candidate_expert="expert_2", bacc=0.87, macro_f1=0.84),
        ],
    )

    outputs = run_learned_utility_pipeline(
        LearnedUtilityPipelineInputs(
            candidates=candidates_path,
            source_inner_training=training_path,
            diagnostic_matrix=matrix_path,
            out_dir=tmp_path / "run",
            feature_columns=("support_nelbo", "source_inner_stability"),
            support_features=support_path,
            source_inner_features=source_inner_features_path,
            ridge_lambda=1e-3,
        )
    )

    for path in outputs.__dict__.values():
        assert Path(path).exists()
    selections = Path(outputs.selections).read_text(encoding="utf-8")
    assert "c2" in selections
    baseline_alignment = Path(outputs.baseline_alignment).read_text(encoding="utf-8")
    assert "metadata_top1" in baseline_alignment
    assert "support_nelbo_top1" in baseline_alignment
    assert "random_expert" in baseline_alignment
    leakage = Path(outputs.leakage_report).read_text(encoding="utf-8")
    assert '"selection_read_downstream_matrix": false' in leakage


def test_c52_legacy_adapter_excludes_current_target_from_training(tmp_path: Path) -> None:
    examples = tmp_path / "legacy" / "router_training.csv"
    matrix = tmp_path / "legacy" / "downstream.csv"
    _write_rows(
        examples,
        [
            _legacy_example("0", "1", utility=0.90),
            _legacy_example("1", "0", utility=0.65),
            _legacy_example("2", "0", utility=0.70),
        ],
    )
    _write_rows(
        matrix,
        [
            _legacy_downstream("0", "1", bacc=0.80),
            _legacy_downstream("0", "2", bacc=0.75),
        ],
    )
    paths = normalize_c52_legacy_artifacts(
        router_training_examples=examples,
        downstream_matrix=matrix,
        target_domain="0",
        out_dir=tmp_path / "normalized",
        support_size=None,
        support_seed=None,
    )
    training_text = Path(paths["source_inner_training"]).read_text(encoding="utf-8")
    assert "0.90" not in training_text
    assert "0.65" in training_text
    assert Path(paths["diagnostic_matrix"]).name == "diagnostic_downstream_utility.csv"


def test_c52_legacy_batch_discovers_contexts_and_writes_summaries(tmp_path: Path) -> None:
    examples = tmp_path / "legacy" / "router_training.csv"
    matrix = tmp_path / "legacy" / "downstream.csv"
    _write_rows(
        examples,
        [
            _legacy_example("0", "1", utility=0.90),
            _legacy_example("0", "2", utility=0.80),
            _legacy_example("1", "0", utility=0.65),
            _legacy_example("1", "2", utility=0.72),
            _legacy_example("2", "0", utility=0.70),
            _legacy_example("2", "1", utility=0.74),
        ],
    )
    _write_rows(
        matrix,
        [
            _legacy_downstream("0", "1", bacc=0.80),
            _legacy_downstream("0", "2", bacc=0.75),
            _legacy_downstream("1", "0", bacc=0.66),
            _legacy_downstream("1", "2", bacc=0.77),
        ],
    )
    contexts = discover_c52_contexts(router_training_examples=examples, downstream_matrix=matrix)
    assert contexts == (("0", 4, 17), ("1", 4, 17))

    outputs = run_c52_legacy_batch(
        router_training_examples=examples,
        downstream_matrix=matrix,
        out_dir=tmp_path / "batch",
        feature_columns=("support_nelbo", "source_inner_stability"),
        target_domains=("0", "1"),
        support_sizes=(4,),
        support_seeds=(17,),
        ridge_lambda=1e-3,
    )
    assert Path(outputs["alignment_summary"]).exists()
    assert Path(outputs["baseline_summary"]).exists()
    assert Path(outputs["leakage_summary"]).exists()
    assert Path(outputs["manifest"]).exists()


def _candidate(
    candidate_id: str,
    *,
    source_domain: str,
    eligibility: str,
    role: str = "",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_domain": source_domain,
        "expert_checkpoint_id": f"expert_{source_domain}",
        "expert_checkpoint_hash": "ckpt_hash",
        "eligibility": eligibility,
        "role": role,
    }


def _candidate_manifest_row(
    candidate_id: str,
    *,
    source_domain: str,
    expert_checkpoint_id: str,
) -> dict[str, object]:
    return {
        "fold_id": "fold0",
        "experiment_seed": 0,
        "target_domain": "0",
        "support_split_id": "support0",
        "eval_split_id": "eval0",
        "candidate_id": candidate_id,
        "source_domain": source_domain,
        "expert_checkpoint_id": expert_checkpoint_id,
        "expert_checkpoint_hash": "ckpt_hash",
        "generation_mode": "class_stratified_reference_posterior_resampling",
        "generation_seed": 17,
        "classifier_seed": 17,
        "config_hash": "config_hash",
        "protocol_hash": "protocol_hash",
        "eligibility": SELECTION_ELIGIBLE,
    }


def _feature_row(candidate_id: str, *, expert_checkpoint_id: str = "expert_1") -> dict[str, object]:
    return {
        "fold_id": "fold0",
        "experiment_seed": 0,
        "target_domain": "0",
        "support_split_id": "support0",
        "eval_split_id": "eval0",
        "candidate_id": candidate_id,
        "expert_checkpoint_id": expert_checkpoint_id,
        "expert_checkpoint_hash": "ckpt_hash",
        "generation_mode": "class_stratified_reference_posterior_resampling",
        "generation_seed": 17,
        "classifier_seed": 17,
        "config_hash": "config_hash",
        "protocol_hash": "protocol_hash",
        "eligibility": SELECTION_ELIGIBLE,
    }


def _feature_artifact_row(candidate_id: str, *, expert_checkpoint_id: str = "expert_1") -> dict[str, object]:
    return _feature_row(candidate_id, expert_checkpoint_id=expert_checkpoint_id)


def _downstream_row(candidate_expert: str, bacc: float, macro_f1: float) -> CandidateDownstreamRow:
    return CandidateDownstreamRow(
        experiment_seed=0,
        heldout_center="0",
        candidate_expert=candidate_expert,
        generation_mode="class_stratified_reference_posterior_resampling",
        budget_per_class=128,
        generation_seed=17,
        classifier_seed=17,
        bacc=bacc,
        macro_f1=macro_f1,
        row_type="single_expert",
        status="ok",
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _legacy_example(heldout: str, candidate: str, *, utility: float) -> dict[str, object]:
    return {
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "support_eval_split_id": f"target{heldout}_support",
        "candidate_expert": candidate,
        "generator_family": "legacy_family",
        "generation_mode": "class_stratified_reference_posterior_resampling",
        "mode_label": "legacy_mode",
        "primary_candidate_eligible": 1,
        "support_nelbo_mean": 5.0 + float(candidate),
        "support_nelbo_rank_within_unit": int(candidate) + 1,
        "support_nelbo_z_within_unit": 0.1,
        "metadata_match": 0,
        "utility_label_bacc": utility,
        "utility_label_bacc_std": 0.02,
        "utility_label_ge_080_rate": 0.5,
    }


def _legacy_downstream(heldout: str, candidate: str, *, bacc: float) -> dict[str, object]:
    return {
        "schema_version": "all_expert_downstream_matrix_v3",
        "experiment_seed": 42,
        "heldout_center": heldout,
        "support_size": 4,
        "support_seed": 17,
        "candidate_expert": candidate,
        "generator_family": "legacy_family",
        "generation_mode": "class_stratified_reference_posterior_resampling",
        "budget_per_class": 128,
        "generation_seed": 17,
        "classifier_seed": 17,
        "bacc": bacc,
        "macro_f1": bacc - 0.05,
        "auroc": bacc,
        "auprc": bacc,
        "row_type": "single_expert",
        "n_synthetic_train": 256,
        "n_target_eval": 100,
        "target_eval_pool_id": f"target{heldout}_eval",
        "candidate_experts_hash": "__single_expert__",
        "status": "ok",
        "error_message": "",
    }
