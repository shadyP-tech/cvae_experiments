from pathlib import Path
import math
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.artifacts.midogpp_phase2 import (  # noqa: E402
    create_phase2_artifact_root,
    default_phase2_artifact_root,
    phase2_validation_payload,
    write_phase2_csv,
    write_locked_phase2_support_eval_split,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402
from cvae_downstream_evaluation.schemas import SELECTION_ELIGIBLE  # noqa: E402
from cvae_downstream_evaluation.schemas.midogpp_phase2 import (  # noqa: E402
    PHASE2_ROOT_NAME,
    PHASE2_SCORE_FUNCTIONAL_ID,
    assert_no_stale_score_semantics,
    assert_phase2_artifact_contract,
    assert_phase2_candidate_manifest,
    assert_phase2_routing_firewall,
    assert_phase2_score_config,
    assert_phase2_snapshot,
    assert_phase2_split_manifests,
    build_locked_phase2_support_eval_split,
    build_phase2_candidate_manifest,
    class_prior_hash,
    log_mixture_marginal_nelbo_reference,
    phase2_artifact_root,
    prior_weighted_expected_conditional_nelbo,
)


def test_phase2_artifact_root_scaffold_is_separate_from_phase1(tmp_path: Path) -> None:
    root = phase2_artifact_root(tmp_path)
    paths = create_phase2_artifact_root(root)

    assert root.name == PHASE2_ROOT_NAME
    assert set(paths) == {"configs", "manifests", "tables", "reports"}
    assert_phase2_artifact_contract(root)
    assert default_phase2_artifact_root(tmp_path) == root

    try:
        create_phase2_artifact_root(tmp_path / "midogpp" / "phase1_virchow2_late_import_seed42")
    except ProtocolError:
        pass
    else:
        raise AssertionError("phase-1 diagnostic root was accepted as phase-2")


def test_phase2_candidate_builder_is_quarantine_aware() -> None:
    rows = [
        _source_row("0"),
        _source_row("1"),
        _source_row("4"),
        _source_row("5"),
        _source_row("99"),
    ]

    candidates = build_phase2_candidate_manifest(rows, heldout_center="0")

    assert [row["candidate_source_center"] for row in candidates] == ["1", "5"]
    assert all(row["eligibility"] == SELECTION_ELIGIBLE for row in candidates)
    assert all(row["score_formula_id"] == PHASE2_SCORE_FUNCTIONAL_ID for row in candidates)
    assert_phase2_candidate_manifest(candidates, heldout_center="0")

    leaked = candidates + [
        _candidate_row(
            heldout_center="0",
            source_center="4",
            candidate_id="leaked_center4",
        )
    ]
    try:
        assert_phase2_candidate_manifest(leaked, heldout_center="0")
    except ProtocolError:
        pass
    else:
        raise AssertionError("selection-eligible center 4 candidate was not rejected")


def test_phase2_score_is_expected_conditional_nelbo_not_log_mixture() -> None:
    prior = {"0": 0.5, "1": 0.5}
    nelbo = {"0": 2.0, "1": 8.0}

    expected = prior_weighted_expected_conditional_nelbo(nelbo, prior, class_order=("0", "1"))
    marginal = log_mixture_marginal_nelbo_reference(nelbo, prior, class_order=("0", "1"))

    assert math.isclose(expected, 5.0)
    assert not math.isclose(expected, marginal)

    prior_hash = class_prior_hash(prior, class_order=("0", "1"))
    assert_phase2_score_config(
        {
            "score_formula_id": PHASE2_SCORE_FUNCTIONAL_ID,
            "class_order": ("0", "1"),
            "class_prior_values": prior,
            "class_prior_value_hash": prior_hash,
        }
    )

    try:
        assert_phase2_score_config(
            {
                "score_formula_id": "marginal_unlabeled",
                "class_order": ("0", "1"),
                "class_prior_values": prior,
                "class_prior_value_hash": prior_hash,
            }
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("stale marginal score id was not rejected")


def test_phase2_stale_score_semantics_are_rejected() -> None:
    assert_no_stale_score_semantics("score_formula_id: prior_weighted_expected_conditional_nelbo_v1")
    for stale in ("marginal_unlabeled", "calibrated_marginal_support_nelbo", "posterior"):
        try:
            assert_no_stale_score_semantics(f"score_formula_id: {stale}")
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"stale token {stale!r} was accepted")


def test_phase2_split_validation_checks_all_available_id_families() -> None:
    support = [
        {
            "sample_id": "s1",
            "patient_id": "p1",
            "slide_id": "slide1",
            "center": "0",
            "label_availability": "withheld_from_routing",
        }
    ]
    evaluation = [
        {
            "sample_id": "e1",
            "patient_id": "p2",
            "slide_id": "slide2",
            "center": "0",
            "class_label": 1,
        }
    ]

    report = assert_phase2_split_manifests(support_rows=support, eval_rows=evaluation)
    checks = report["checks"]
    assert checks["sample_id"]["status"] == "disjoint"
    assert checks["case_id"]["status"] == "unavailable"

    try:
        assert_phase2_split_manifests(
            support_rows=[support[0] | {"sample_id": "shared"}],
            eval_rows=[evaluation[0] | {"sample_id": "shared"}],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("support/eval sample overlap was not rejected")

    try:
        assert_phase2_split_manifests(
            support_rows=[support[0] | {"label": 1}],
            eval_rows=evaluation,
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("support label exposure was not rejected")


def test_phase2_locked_splitter_writes_label_free_support_manifests(tmp_path: Path) -> None:
    rows = [
        _target_sample("s1", patient_id="p1", slide_id="slide1", label=0),
        _target_sample("s2", patient_id="p2", slide_id="slide2", label=1),
        _target_sample("s3", patient_id="p3", slide_id="slide3", label=0),
        _target_sample("s4", patient_id="p4", slide_id="slide4", label=1),
    ]

    support, evaluation = build_locked_phase2_support_eval_split(
        rows,
        heldout_center="0",
        support_size=2,
        support_seed=42,
    )

    assert support
    assert evaluation
    assert all(row["split_role"] == "support" for row in support)
    assert all("label" not in row for row in support)
    assert any("label" in row for row in evaluation)
    assert_phase2_split_manifests(support_rows=support, eval_rows=evaluation)

    root = tmp_path / "midogpp" / PHASE2_ROOT_NAME
    create_phase2_artifact_root(root)
    support_path, eval_path = write_locked_phase2_support_eval_split(
        root,
        rows,
        heldout_center="0",
        support_size=2,
        support_seed=42,
    )
    assert support_path.name == "support_sets.csv"
    assert eval_path.name == "eval_sets.csv"
    support_header = support_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    eval_header = eval_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "label" not in support_header
    assert "label_availability" in support_header
    assert "label" in eval_header


def test_phase2_locked_splitter_accepts_scanner_domain_labels() -> None:
    rows = [
        _target_sample("s1", patient_id="p1", slide_id="slide1", label=0) | {"domain": "Hamamatsu XR"},
        _target_sample("s2", patient_id="p2", slide_id="slide2", label=1) | {"domain": "Hamamatsu XR"},
        _target_sample("s3", patient_id="p3", slide_id="slide3", label=0) | {"domain": "Hamamatsu XR"},
        _target_sample("s4", patient_id="p4", slide_id="slide4", label=1) | {"domain": "Hamamatsu XR"},
        _target_sample("s5", patient_id="p5", slide_id="slide5", label=0) | {"domain": "Aperio ScanScope CS"},
    ]

    support, evaluation = build_locked_phase2_support_eval_split(
        rows,
        heldout_center="Hamamatsu XR",
        support_size=2,
        support_seed=42,
        center_column="domain",
    )

    assert support
    assert evaluation
    assert {row["domain"] for row in support}.union(row["domain"] for row in evaluation) == {"Hamamatsu XR"}


def test_phase2_routing_firewall_blocks_diagnostic_and_phase1_inputs() -> None:
    assert_phase2_routing_firewall(
        input_paths=[Path("tables/support_score_matrix.csv")],
        input_rows=[{"candidate_id": "c1", "support_score": 4.2}],
    )

    forbidden_paths = [
        Path("tables/diagnostic_downstream_utility.csv"),
        Path("tables/diagnostic_eval_nelbo_matrix.csv"),
        Path("phase1_virchow2_late_import_seed42/tables/anything.csv"),
    ]
    for path in forbidden_paths:
        try:
            assert_phase2_routing_firewall(input_paths=[path])
        except ProtocolError:
            pass
        else:
            raise AssertionError(f"diagnostic path {path} was accepted by routing firewall")

    try:
        assert_phase2_routing_firewall(input_rows=[{"candidate_id": "c1", "bacc": 0.9}])
    except ProtocolError:
        pass
    else:
        raise AssertionError("target-eval metric column was not rejected")


def test_phase2_snapshot_binds_score_prior_and_routing_decision() -> None:
    prior = {"0": 0.5, "1": 0.5}
    prior_hash = class_prior_hash(prior, class_order=("0", "1"))
    candidate = _candidate_row(
        heldout_center="0",
        source_center="1",
        candidate_id="c1",
        class_prior_value_hash=prior_hash,
    )
    snapshot = {
        "candidate_pool_hash": "pool",
        "support_split_hash": "support",
        "eval_split_hash": "eval",
        "checkpoint_cache_hash": "ckpt",
        "generation_config_hash": "gen",
        "classifier_config_hash": "clf",
        "metric_config_hash": "metric",
        "feature_whitelist_hash": "features",
        "routing_rule": "argmin_support_score",
        "score_formula_id": PHASE2_SCORE_FUNCTIONAL_ID,
        "class_prior_value_hash": prior_hash,
        "score_direction": "lower_is_better",
        "support_aggregation": "mean_over_support_samples",
        "tie_breaker": "stable_candidate_id",
        "protocol_hash": "protocol",
    }

    assert_phase2_snapshot(snapshot, candidate_rows=[candidate])

    try:
        assert_phase2_snapshot(snapshot | {"score_direction": "higher_is_better"}, candidate_rows=[candidate])
    except ProtocolError:
        pass
    else:
        raise AssertionError("wrong score direction was accepted")


def test_phase2_artifact_writer_rejects_forbidden_matrix_name(tmp_path: Path) -> None:
    allowed = tmp_path / "tables" / "support_score_matrix.csv"
    write_phase2_csv(allowed, [{"candidate_id": "c1", "support_score": 1.0}])
    assert allowed.exists()

    try:
        write_phase2_csv(
            tmp_path / "tables" / "target_support_downstream_matrix.csv",
            [{"candidate_id": "c1"}],
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("forbidden target_support_downstream_matrix.csv was written")

    payload = phase2_validation_payload(
        artifacts_root=tmp_path,
        status="PASS",
        checks={"schema": "ok"},
    )
    assert payload["schema_version"] == "midogpp_phase2_validation_report_v1"


def _source_row(source_center: str) -> dict[str, object]:
    prior = {"0": 0.5, "1": 0.5}
    return {
        "source_center": source_center,
        "candidate_id": f"source_{source_center}",
        "checkpoint_path": f"ckpt/{source_center}.pt",
        "checkpoint_hash": f"ckpt-{source_center}",
        "checkpoint_provenance_hash": f"prov-{source_center}",
        "feature_frame_hash": f"frame-{source_center}",
        "checkpoint_seed": 42,
        "generation_mode": "class_balanced",
        "generation_class_prior_policy": "uniform_generation_policy",
        "synthetic_budget": 128,
        "generation_seed": 42,
        "classifier_seed": 42,
        "class_order": ("0", "1"),
        "class_prior_values": prior,
        "class_prior_rule": "uniform",
        "scorer_implementation_hash": "score-hash",
        "config_hash": "config",
        "protocol_hash": "protocol",
    }


def _target_sample(sample_id: str, *, patient_id: str, slide_id: str, label: int) -> dict[str, object]:
    return {
        "center": "0",
        "sample_id": sample_id,
        "patient_id": patient_id,
        "slide_id": slide_id,
        "case_id": f"case-{sample_id}",
        "group_id": patient_id,
        "embedding_path": f"emb/{sample_id}.npy",
        "label": label,
    }


def _candidate_row(
    *,
    heldout_center: str,
    source_center: str,
    candidate_id: str,
    class_prior_value_hash: str | None = None,
) -> dict[str, object]:
    prior = {"0": 0.5, "1": 0.5}
    return {
        "schema_version": "midogpp_phase2_target_support_adaptation_v1",
        "heldout_center": heldout_center,
        "candidate_source_center": source_center,
        "candidate_id": candidate_id,
        "checkpoint_path": "ckpt.pt",
        "checkpoint_hash": "ckpt",
        "checkpoint_provenance_hash": "prov",
        "feature_frame_hash": "frame",
        "checkpoint_seed": 42,
        "generation_mode": "class_balanced",
        "generation_class_prior_policy": "uniform_generation_policy",
        "synthetic_budget": 128,
        "generation_seed": 42,
        "classifier_seed": 42,
        "class_prior_rule": "uniform",
        "class_prior_value_hash": class_prior_value_hash
        or class_prior_hash(prior, class_order=("0", "1")),
        "class_order": "0|1",
        "class_prior_values": prior,
        "score_formula_id": PHASE2_SCORE_FUNCTIONAL_ID,
        "deterministic_scoring": True,
        "calibration_source": "source_only_or_uniform",
        "scorer_implementation_hash": "score-hash",
        "config_hash": "config",
        "protocol_hash": "protocol",
        "row_role": "selection_candidate",
        "eligibility": SELECTION_ELIGIBLE,
        "support_labels_used": False,
    }
