from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.downstream import (  # noqa: E402
    CandidateDownstreamRow,
    assert_matrix_schema,
    compute_single_expert_oracles,
    read_candidate_downstream_matrix,
    validate_candidate_downstream_matrix,
    write_candidate_downstream_matrix,
)
from cvae_downstream_evaluation.generation import (  # noqa: E402
    allocate_equal_total_ensemble_budget,
)
from cvae_downstream_evaluation.protocol import (  # noqa: E402
    ProtocolError,
    assert_locked_v1_config_text,
    load_locked_v1_config,
)
from cvae_downstream_evaluation.reporting import build_routing_alignment_rows  # noqa: E402
from cvae_downstream_evaluation.source_global_gated import (  # noqa: E402
    build_source_global_gated_alignment_rows,
    derive_source_global_gated_units,
    gated_method_name,
    source_global_gated_comparison_rows,
)
from cvae_downstream_evaluation.matrix import (  # noqa: E402
    build_target_eval_pool,
    hash_candidate_experts,
)
from cvae_downstream_evaluation.routing import (  # noqa: E402
    SupportSelectionUnit,
    add_deterministic_random_units,
)
from cvae_downstream_evaluation.schemas import (  # noqa: E402
    ENSEMBLE_METHOD,
    METADATA_METHOD,
    PRIMARY_GENERATION_MODE,
    RANDOM_METHOD,
    SOURCE_GLOBAL_METHOD,
    SUPPORT_NELBO_METHOD,
    NEGATIVE_CONTROL_GENERATION_MODE,
)
from cvae_downstream_evaluation.utility_matrix import (  # noqa: E402
    diagnostic_matrix_path,
    read_diagnostic_downstream_matrix,
    write_diagnostic_downstream_matrix,
)


def test_expected_skeleton_files_exist() -> None:
    expected = [
        ROOT / "README.md",
        ROOT / "docs" / "protocol.md",
        ROOT / "docs" / "thesis_alignment.md",
        ROOT / "docs" / "implementation_order.md",
        ROOT / "configs" / "experiments" / "direct_support_nelbo_selected_synthetic_downstream_v1.yaml",
        ROOT / "configs" / "experiments" / "downstream_compatibility_v1.yaml",
        ROOT / "src" / "cvae_downstream_evaluation" / "protocol.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "routing.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "generation.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "downstream.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "matrix.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "fidelity.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "reporting.py",
        ROOT / "src" / "cvae_downstream_evaluation" / "source_global_gated.py",
        ROOT / "scripts" / "build_source_global_gated_router_report.py",
        ROOT / "scripts" / "build_learned_utility_selection_report.py",
        ROOT / "scripts" / "build_allowed_feature_table.py",
        ROOT / "scripts" / "build_selection_leakage_report.py",
        ROOT / "scripts" / "train_source_inner_utility_estimator.py",
        ROOT / "scripts" / "predict_learned_utility_features.py",
        ROOT / "scripts" / "run_learned_utility_pipeline.py",
        ROOT / "scripts" / "normalize_c52_legacy_artifacts.py",
        ROOT / "scripts" / "run_c52_legacy_learned_utility_batch.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    assert not missing


def test_protocol_docs_preserve_claim_boundary() -> None:
    protocol = (ROOT / "docs" / "protocol.md").read_text(encoding="utf-8")
    assert "Lower support NELBO proves better generative quality" in protocol
    assert "Forbidden" in protocol
    assert "target evaluation labels" in protocol


def test_config_declares_second_stage_outputs() -> None:
    config = (
        ROOT
        / "configs"
        / "experiments"
        / "direct_support_nelbo_selected_synthetic_downstream_v1.yaml"
    ).read_text(encoding="utf-8")
    assert "all_expert_downstream_matrix.csv" in config
    assert "single_expert_downstream_oracle_diagnostic_only" in config
    assert "support_eval_separation: required" in config
    assert "support_size_stratified_downstream_summary.csv" in config


def test_locked_config_rejects_stale_template_fields() -> None:
    config = (
        ROOT
        / "configs"
        / "experiments"
        / "direct_support_nelbo_selected_synthetic_downstream_v1.yaml"
    ).read_text(encoding="utf-8")
    assert_locked_v1_config_text(config)
    try:
        assert_locked_v1_config_text(config + "\ngeneration:\n  decoder_sampling: conditional_cvae_decoder\n")
    except ProtocolError:
        pass
    else:
        raise AssertionError("stale conditional-generation wording was not rejected")


def test_locked_config_loads_without_yaml_dependency() -> None:
    config = load_locked_v1_config(
        ROOT
        / "configs"
        / "experiments"
        / "direct_support_nelbo_selected_synthetic_downstream_v1.yaml"
    )
    assert config.dataset_name == "camelyon17"
    assert config.support_seeds == (17, 23, 31)
    assert config.generation_seeds == (17, 23, 31)


def test_deterministic_random_units_are_support_unit_aligned() -> None:
    unit = _unit(SUPPORT_NELBO_METHOD, "2")
    generated = add_deterministic_random_units([unit])
    random_units = [row for row in generated if row.method == RANDOM_METHOD]
    assert len(random_units) == 1
    assert random_units[0].heldout_center == unit.heldout_center
    assert random_units[0].support_eval_split_id == unit.support_eval_split_id
    assert random_units[0].selected_expert in unit.candidate_experts


def test_single_expert_oracle_excludes_method_baselines() -> None:
    rows = [
        _downstream("1", 0.70, "single_expert"),
        _downstream("2", 0.80, "single_expert"),
        _downstream("__ensemble__", 0.95, "method_baseline"),
    ]
    oracle = compute_single_expert_oracles(rows)
    winner = next(iter(oracle.values()))
    assert winner.expert == "2"


def test_single_expert_oracle_excludes_failed_and_negative_control_rows() -> None:
    rows = [
        _downstream("1", 0.70, "single_expert"),
        _downstream("2", 0.95, "single_expert", status="failed_empty_reference_pool"),
        _downstream(
            "3",
            0.99,
            "single_expert",
            generation_mode=NEGATIVE_CONTROL_GENERATION_MODE,
        ),
    ]
    oracle = compute_single_expert_oracles(rows)
    winner = next(iter(oracle.values()))
    assert winner.expert == "1"


def test_candidate_matrix_rejects_duplicate_rows() -> None:
    row = _downstream("1", 0.70, "single_expert")
    try:
        validate_candidate_downstream_matrix([row, row])
    except ProtocolError:
        pass
    else:
        raise AssertionError("duplicate downstream rows were not rejected")


def test_candidate_matrix_key_includes_experiment_seed_and_candidate_hash() -> None:
    row_seed42 = _downstream("1", 0.70, "single_expert", experiment_seed=42)
    row_seed43 = _downstream("1", 0.70, "single_expert", experiment_seed=43)
    row_hash_a = _downstream("__ensemble__", 0.80, "method_baseline", candidate_hash="abc")
    row_hash_b = _downstream("__ensemble__", 0.80, "method_baseline", candidate_hash="def")
    validate_candidate_downstream_matrix([row_seed42, row_seed43, row_hash_a, row_hash_b])


def test_matrix_schema_sidecar_is_required_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "all_expert_downstream_matrix.csv"
    write_candidate_downstream_matrix(path, [_downstream("1", 0.70, "single_expert")])
    assert_matrix_schema(path)
    (tmp_path / "all_expert_downstream_matrix.schema.json").write_text(
        '{"schema_version":"stale"}\n',
        encoding="utf-8",
    )
    try:
        assert_matrix_schema(path)
    except ProtocolError:
        pass
    else:
        raise AssertionError("stale downstream matrix schema was not rejected")


def test_diagnostic_downstream_matrix_uses_same_schema_with_quarantined_name(tmp_path: Path) -> None:
    path = diagnostic_matrix_path(tmp_path)
    write_diagnostic_downstream_matrix(path, [_downstream("1", 0.70, "single_expert")])
    assert path.name == "diagnostic_downstream_utility.csv"
    assert_matrix_schema(path)
    assert read_diagnostic_downstream_matrix(path)[0].candidate_expert == "1"
    assert read_candidate_downstream_matrix(path)[0].candidate_expert == "1"

    bad_path = tmp_path / "tables" / "all_expert_downstream_matrix.csv"
    try:
        write_diagnostic_downstream_matrix(bad_path, [_downstream("1", 0.70, "single_expert")])
    except ProtocolError:
        pass
    else:
        raise AssertionError("diagnostic writer accepted a non-quarantined matrix name")


def test_late_ensemble_budget_is_equal_total() -> None:
    assert allocate_equal_total_ensemble_budget(total_per_class=128, candidate_experts=["3", "1", "2"]) == {
        "1": 43,
        "2": 43,
        "3": 42,
    }


def test_candidate_expert_hash_is_order_stable() -> None:
    assert hash_candidate_experts(["3", "1", "2"]) == hash_candidate_experts(["2", "3", "1"])


def test_target_eval_pool_excludes_support_by_sample_id() -> None:
    records = [
        {"sample_id": f"c0_{idx}", "magnification": "0", "label": str(idx % 2)}
        for idx in range(12)
    ]
    records += [
        {"sample_id": f"c1_{idx}", "magnification": "1", "label": str(idx % 2)}
        for idx in range(12)
    ]
    pool = build_target_eval_pool(
        test_metadata=records,
        heldout_center="0",
        support_sizes=(4, 8),
        support_seeds=(17, 23),
    )
    eval_ids = {records[idx]["sample_id"] for idx in pool.eval_indices}
    assert set(pool.excluded_support_sample_ids).isdisjoint(eval_ids)
    assert all(str(records[idx]["magnification"]) == "0" for idx in pool.eval_indices)


def test_alignment_uses_spearman_only_for_ranked_methods() -> None:
    selections = [
        _unit(SUPPORT_NELBO_METHOD, "2"),
        _unit(METADATA_METHOD, "1"),
        _unit(SOURCE_GLOBAL_METHOD, "1"),
    ]
    downstream = [
        _downstream("1", 0.70, "single_expert"),
        _downstream("2", 0.80, "single_expert"),
    ]
    rows = build_routing_alignment_rows(selections=selections, downstream_rows=downstream)
    by_method = {str(row["method"]): row for row in rows}
    assert by_method[SUPPORT_NELBO_METHOD]["downstream_oracle_gap_bacc"] == 0.0
    assert by_method[SUPPORT_NELBO_METHOD]["spearman_neg_nelbo_vs_bacc"] > 0
    assert math.isnan(float(by_method[METADATA_METHOD]["spearman_neg_nelbo_vs_bacc"]))
    assert by_method[SUPPORT_NELBO_METHOD]["delta_vs_metadata"] > 0


def test_source_global_gate_switches_on_sufficient_gain() -> None:
    units = [
        _unit(SUPPORT_NELBO_METHOD, "2"),
        _unit(SOURCE_GLOBAL_METHOD, "1"),
    ]
    gated = derive_source_global_gated_units(units, taus=(0.10,))
    assert len(gated) == 1
    assert gated[0].method == gated_method_name(0.10)
    assert gated[0].selected_expert == "2"
    assert gated[0].eligible_switch
    assert gated[0].switched_from_global
    assert not gated[0].same_as_global


def test_source_global_gate_falls_back_below_threshold_and_zero_range() -> None:
    weak_support = _custom_unit(
        SUPPORT_NELBO_METHOD,
        "2",
        candidates=("1", "2", "3"),
        scores={"1": 10.0, "2": 9.95, "3": 20.0},
    )
    weak_global = _custom_unit(
        SOURCE_GLOBAL_METHOD,
        "1",
        candidates=("1", "2", "3"),
        scores={"1": 10.0, "2": 9.95, "3": 20.0},
    )
    weak_gated = derive_source_global_gated_units([weak_support, weak_global], taus=(0.10,))
    assert weak_gated[0].selected_expert == "1"
    assert weak_gated[0].eligible_switch
    assert not weak_gated[0].switched_from_global

    flat_support = _custom_unit(
        SUPPORT_NELBO_METHOD,
        "2",
        scores={"1": 10.0, "2": 10.0},
    )
    flat_global = _custom_unit(
        SOURCE_GLOBAL_METHOD,
        "1",
        scores={"1": 10.0, "2": 10.0},
    )
    flat_gated = derive_source_global_gated_units([flat_support, flat_global], taus=(0.0,))
    assert flat_gated[0].score_range == 0.0
    assert flat_gated[0].normalized_gain_vs_global == 0.0
    assert flat_gated[0].selected_expert == "1"


def test_source_global_gate_requires_global_expert_in_support_scores() -> None:
    support = _custom_unit(
        SUPPORT_NELBO_METHOD,
        "2",
        candidates=("1", "2", "3"),
        scores={"1": 10.0, "2": 5.0},
    )
    source_global = _custom_unit(
        SOURCE_GLOBAL_METHOD,
        "3",
        candidates=("1", "2", "3"),
        scores={"1": 10.0, "2": 5.0},
    )
    try:
        derive_source_global_gated_units([support, source_global], taus=(0.10,))
    except ProtocolError:
        pass
    else:
        raise AssertionError("missing source-global support score was not rejected")


def test_source_global_gated_alignment_and_oracle_gap_deltas_are_paired() -> None:
    selections = [
        _unit(SUPPORT_NELBO_METHOD, "2"),
        _unit(METADATA_METHOD, "1"),
        _unit(SOURCE_GLOBAL_METHOD, "1"),
    ]
    downstream = [
        _downstream("1", 0.70, "single_expert"),
        _downstream("2", 0.80, "single_expert"),
    ]
    gated_units = derive_source_global_gated_units(selections, taus=(0.10,))
    gated_rows = build_source_global_gated_alignment_rows(
        gated_units=gated_units,
        downstream_rows=downstream,
    )
    assert len(gated_rows) == 1
    assert gated_rows[0]["selected_expert"] == "2"
    assert gated_rows[0]["selected_bacc"] == 0.80

    baseline_rows = build_routing_alignment_rows(selections=selections, downstream_rows=downstream)
    comparison = source_global_gated_comparison_rows(
        gated_alignment_rows=gated_rows,
        baseline_alignment_rows=baseline_rows,
    )
    assert len(comparison) == 1
    row = comparison[0]
    assert row["method"] == gated_method_name(0.10)
    assert row["mean_delta_bacc_vs_source_global"] > 0
    assert row["mean_delta_oracle_gap_vs_source_global"] > 0
    assert row["mean_delta_bacc_vs_support_nelbo"] == 0.0


def _unit(method: str, selected: str) -> SupportSelectionUnit:
    return SupportSelectionUnit(
        heldout_center="0",
        experiment_seed=42,
        support_size=16,
        support_seed=17,
        method=method,
        selected_expert=selected,
        candidate_experts=("1", "2"),
        support_nelbo_by_expert={"1": 10.0, "2": 5.0},
        target_expert_excluded=True,
        support_eval_split_id="target0_seed17_random_k16",
    )


def _custom_unit(
    method: str,
    selected: str,
    *,
    candidates: tuple[str, ...] = ("1", "2"),
    scores: dict[str, float] | None = None,
) -> SupportSelectionUnit:
    return SupportSelectionUnit(
        heldout_center="0",
        experiment_seed=42,
        support_size=16,
        support_seed=17,
        method=method,
        selected_expert=selected,
        candidate_experts=candidates,
        support_nelbo_by_expert=scores or {"1": 10.0, "2": 5.0},
        target_expert_excluded=True,
        support_eval_split_id="target0_seed17_random_k16",
    )


def _downstream(
    expert: str,
    bacc: float,
    row_type: str,
    *,
    experiment_seed: int = 42,
    status: str = "ok",
    generation_mode: str = PRIMARY_GENERATION_MODE,
    candidate_hash: str = "__single_expert__",
) -> CandidateDownstreamRow:
    return CandidateDownstreamRow(
        experiment_seed=experiment_seed,
        heldout_center="0",
        candidate_expert=expert,
        generation_mode=generation_mode,
        budget_per_class=128,
        generation_seed=17,
        classifier_seed=17,
        bacc=bacc,
        macro_f1=bacc - 0.05,
        row_type=row_type,
        n_synthetic_train=256,
        n_target_eval=100,
        target_eval_pool_id="target0_exclude_configured_support_union_test",
        candidate_experts_hash=candidate_hash,
        status=status,
    )
