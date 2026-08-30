from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from dataclasses import dataclass, replace
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_action_surface import (
    ACTION_FEATURE_NAMES,
    ACTION_LAMBDAS,
    HarpProbabilityRow,
    build_action_feature_surface,
    build_directional_response_surface,
    build_disagreement_rows,
    build_probability_ensemble_surface,
    build_probability_surface,
)
from midogpp_thesis.cvae.routing.harp_action_model import (
    HarpTargetAction,
    fit_harp_action_model_bank,
    model_bank_collection_payload,
    score_harp_actions,
    training_observations_from_surfaces,
)
from midogpp_thesis.cvae.routing.harp_portfolio import (
    HarpPolicyConfig,
    select_harp_portfolio,
)
from midogpp_thesis.cvae.routing.harp_protocol import (
    HarpSourceLabelCapability,
    HarpSourceLabelRow,
    build_durable_prediction_seal,
)
from midogpp_thesis.cvae.routing.harp_stage60 import load_harp_stage60_config
from midogpp_thesis.cvae.routing.harp_stage60.config import HarpInputReadiness
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.cvae.runtime.harp_probability_menu import EXACT_NINE_SEED_PAIRS
from midogpp_thesis.real_features.classifier_reference.classifiers import ClassifierSpec


UPSTREAM_SEAL = "1" * 64
SEEDS = tuple(f"seed-{index}" for index in range(9))


def _probability_row(
    *,
    seed: str = "seed-0",
    case: str = "B-case-1",
    sample: str = "B-sample-1",
    baseline: float = 0.4,
    expert: float = 0.8,
    outer: str = "A",
    query: str = "B",
    source: str = "C",
    donor: str | None = "D",
) -> HarpProbabilityRow:
    return HarpProbabilityRow(
        outer_target=outer,
        pseudo_query=query,
        candidate_source=source,
        inner_donor=donor,
        case_id=case,
        sample_id=sample,
        seed_id=seed,
        baseline_probability=baseline,
        expert_probability=expert,
        prediction_seal_hash=UPSTREAM_SEAL,
    )


def _exact_nine_rows() -> tuple[HarpProbabilityRow, ...]:
    return tuple(
        _probability_row(
            seed=seed,
            case=f"B-case-{truth}",
            sample=f"B-sample-{truth}",
            baseline=0.4,
            expert=0.8,
        )
        for truth in (0, 1)
        for seed in SEEDS
    )


@pytest.mark.parametrize(
    "overrides",
    (
        {"query": "A"},
        {"source": "A"},
        {"source": "B"},
        {"donor": "A"},
        {"donor": "B"},
        {"donor": "C"},
    ),
)
def test_probability_contract_rejects_poison_roles_before_feature_build(
    overrides: dict[str, str],
) -> None:
    with pytest.raises(ProtocolError):
        _probability_row(**overrides)


def test_seed_cells_are_nonfeeding_and_exact_nine_precedes_case_features() -> None:
    rows = _exact_nine_rows()
    surface_a = build_probability_surface(rows)
    surface_b = build_probability_surface(tuple(reversed(rows)))
    assert surface_a.surface_hash == surface_b.surface_hash
    assert all(row.model_feeding is False for row in surface_a.rows)
    with pytest.raises(ProtocolError, match="never seed cells"):
        build_action_feature_surface(surface_a)  # type: ignore[arg-type]

    ensembles = build_probability_ensemble_surface(
        surface_a, expected_seed_ids=SEEDS
    )
    assert len(ensembles.rows) == 2
    assert all(row.seed_count == 9 and row.model_feeding for row in ensembles.rows)
    assert all(len(row.row_key) == 6 for row in ensembles.rows)
    features = build_action_feature_surface(ensembles)
    assert len(features.rows) == 2 * len(ACTION_LAMBDAS)
    assert all(
        row.action_probability == row.expert_probability
        for row in features.rows
        if row.action_lambda == 1.0
    )
    assert "seed_dispersion" in ACTION_FEATURE_NAMES
    assert all(row.seed_count == 9 and len(row.row_key) == 7 for row in features.rows)
    disagreements = build_disagreement_rows(features)
    assert {row.direction for row in disagreements} == {"D01"}


def test_incomplete_seed_inventory_cannot_reach_model_surface() -> None:
    incomplete = build_probability_surface(_exact_nine_rows()[:-1])
    with pytest.raises(ProtocolError, match="exact sealed nine-seed"):
        build_probability_ensemble_surface(incomplete, expected_seed_ids=SEEDS)


def test_source_response_math_uses_independent_case_denominators(
    tmp_path: Path,
) -> None:
    seed_surface = build_probability_surface(_exact_nine_rows())
    ensembles = build_probability_ensemble_surface(
        seed_surface, expected_seed_ids=SEEDS
    )
    features = build_action_feature_surface(ensembles)
    label_rows = tuple(
        HarpSourceLabelRow(
            center=center,
            case_id=f"{center}-case-{truth}",
            sample_id=f"{center}-sample-{truth}",
            label=truth,
        )
        for center in ("A", "B", "C", "D")
        for truth in (0, 1)
    )
    artifact = tmp_path / "predictions.bin"
    artifact.write_bytes(b"probability artifact")
    seal = build_durable_prediction_seal(
        probability_surface_hash=seed_surface.surface_hash,
        upstream_prediction_seal_hash=UPSTREAM_SEAL,
        prediction_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        prediction_row_count=len(seed_surface.rows),
    )
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal.to_payload()), encoding="utf-8")
    opened = HarpSourceLabelCapability(
        centers=("A", "B", "C", "D"),
        seal=seal,
        seal_path=seal_path,
        prediction_artifact_path=artifact,
        label_loader=lambda: label_rows,
    ).open()
    responses = build_directional_response_surface(features, opened)
    lambda_one = {row.case_id: row for row in responses.rows if row.action_lambda == 1.0}
    positive = lambda_one["B-case-1"]
    negative = lambda_one["B-case-0"]
    assert positive.truth_class == 1 and positive.weighted_correctness_surrogate == 1.0
    assert negative.truth_class == 0 and negative.weighted_correctness_surrogate == -1.0
    assert positive.brier_delta == pytest.approx((0.8 - 1) ** 2 - (0.4 - 1) ** 2)
    assert negative.brier_delta == pytest.approx(0.8**2 - 0.4**2)
    assert positive.log_loss_delta == pytest.approx(-math.log(0.8) + math.log(0.4))
    assert negative.log_loss_delta == pytest.approx(-math.log(0.2) + math.log(0.6))
    assert positive.denominator_receipt_hash == positive.class_prior_receipt_hash
    assert all(receipt.positive_case_count == receipt.negative_case_count == 1 for receipt in responses.receipts)
    assert all(row.seed_count == 9 and len(row.row_key) == 7 for row in responses.rows)


def test_outer_h_label_perturbation_cannot_change_h_bank_or_route_bytes(
    tmp_path: Path,
) -> None:
    donors = ("0", "1", "2", "3", "4")
    seed_rows = tuple(
        _probability_row(
            seed=seed,
            case=f"{query}-case-{truth}",
            sample=f"{query}-sample-{truth}",
            baseline=0.60 if truth == 0 else 0.40,
            expert=0.35 if truth == 0 else 0.75,
            outer="H",
            query=query,
            source=source,
            donor=None,
        )
        for query in donors
        for source in donors
        if source != query
        for truth in (0, 1)
        for seed in SEEDS
    )
    probability_surface = build_probability_surface(seed_rows)
    features = build_action_feature_surface(
        build_probability_ensemble_surface(
            probability_surface, expected_seed_ids=SEEDS
        )
    )
    artifact = tmp_path / "outer-scoped-probabilities.bin"
    artifact.write_bytes(b"outer-scoped probability artifact")
    seal = build_durable_prediction_seal(
        probability_surface_hash=probability_surface.surface_hash,
        upstream_prediction_seal_hash=UPSTREAM_SEAL,
        prediction_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        prediction_row_count=len(seed_rows),
    )
    seal_path = tmp_path / "outer-scoped-seal.json"
    seal_path.write_text(json.dumps(seal.to_payload()), encoding="utf-8")

    def opened(*, perturb_outer: bool):
        rows = tuple(
            HarpSourceLabelRow(
                center=center,
                case_id=f"{center}-case-{truth}",
                sample_id=f"{center}-sample-{truth}",
                label=(1 - truth if perturb_outer and center == "H" else truth),
            )
            for center in (*donors, "H")
            for truth in (0, 1)
        )
        return HarpSourceLabelCapability(
            centers=(*donors, "H"),
            seal=seal,
            seal_path=seal_path,
            prediction_artifact_path=artifact,
            label_loader=lambda: rows,
        ).open()

    original_labels = opened(perturb_outer=False)
    perturbed_labels = opened(perturb_outer=True)
    assert original_labels.label_surface_hash != perturbed_labels.label_surface_hash
    assert (
        original_labels.scope_for_outer_target("H").label_surface_hash
        == perturbed_labels.scope_for_outer_target("H").label_surface_hash
    )
    original = build_directional_response_surface(features, original_labels)
    perturbed = build_directional_response_surface(features, perturbed_labels)
    assert original == perturbed

    original_observations = training_observations_from_surfaces(features, original)
    perturbed_observations = training_observations_from_surfaces(features, perturbed)
    assert original_observations == perturbed_observations
    original_bank = fit_harp_action_model_bank(
        original_observations, outer_target_id="H", alphas=(0.1,)
    )
    perturbed_bank = fit_harp_action_model_bank(
        perturbed_observations, outer_target_id="H", alphas=(0.1,)
    )
    assert model_bank_collection_payload(
        (original_bank,)
    ) == model_bank_collection_payload((perturbed_bank,))

    action_rows = tuple(
        row
        for row in features.rows
        if row.pseudo_query == "0"
        and row.candidate_source == "1"
        and row.case_id == "0-case-1"
    )
    target_actions = tuple(
        HarpTargetAction(
            outer_target_id="H",
            target_query_id="H",
            candidate_source_id="1",
            case_id="target-case",
            sample_id="target-sample",
            lambda_value=row.action_lambda,
            direction=row.direction,
            feature_names=row.feature_names,
            feature_values=row.feature_values,
            baseline_probability_bytes=struct.pack(
                "<d", row.baseline_probability
            ),
            operational_fallback_probability_bytes=struct.pack("<d", 0.25),
            expert_probability=row.expert_probability,
            ensemble_size=9,
            ensemble_receipt_hash=row.ensemble_receipt_hash,
            prediction_seal_hash=row.prediction_seal_hash,
        )
        for row in action_rows
    )
    policy = HarpPolicyConfig(
        gain_threshold=-1.0e6,
        brier_noninferiority_margin=1.0e6,
        log_loss_noninferiority_margin=1.0e6,
        min_donor_count=1,
        min_paired_case_count=1,
        max_leverage=1.0e12,
    )
    original_decisions = select_harp_portfolio(
        score_harp_actions(original_bank, target_actions), config=policy
    )
    perturbed_decisions = select_harp_portfolio(
        score_harp_actions(perturbed_bank, target_actions), config=policy
    )
    assert original_decisions == perturbed_decisions
    assert tuple(row.output_probability_bytes for row in original_decisions) == tuple(
        row.output_probability_bytes for row in perturbed_decisions
    )


def test_response_rejects_extra_label_outside_sealed_query_frame(tmp_path: Path) -> None:
    seed_surface = build_probability_surface(_exact_nine_rows())
    features = build_action_feature_surface(
        build_probability_ensemble_surface(seed_surface, expected_seed_ids=SEEDS)
    )
    labels = tuple(
        HarpSourceLabelRow(
            center=center,
            case_id=f"{center}-case-{truth}",
            sample_id=f"{center}-sample-{truth}",
            label=truth,
        )
        for center in ("A", "B", "C", "D")
        for truth in (0, 1)
    ) + (
        HarpSourceLabelRow(
            center="B",
            case_id="B-unreserved-case",
            sample_id="B-unreserved-sample",
            label=1,
        ),
    )
    artifact = tmp_path / "sealed-probabilities.bin"
    artifact.write_bytes(b"sealed")
    seal = build_durable_prediction_seal(
        probability_surface_hash=seed_surface.surface_hash,
        upstream_prediction_seal_hash=UPSTREAM_SEAL,
        prediction_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        prediction_row_count=len(seed_surface.rows),
    )
    seal_path = tmp_path / "sealed-probabilities.json"
    seal_path.write_text(json.dumps(seal.to_payload()), encoding="utf-8")
    opened = HarpSourceLabelCapability(
        centers=("A", "B", "C", "D"),
        seal=seal,
        seal_path=seal_path,
        prediction_artifact_path=artifact,
        label_loader=lambda: labels,
    ).open()
    with pytest.raises(ProtocolError, match="differ from the sealed development menu"):
        build_directional_response_surface(features, opened)


def test_two_samples_in_one_case_remain_two_equal_case_bound_rows() -> None:
    rows = tuple(
        _probability_row(
            seed=seed,
            case="B-case-1",
            sample=sample,
            baseline=0.35 if sample.endswith("a") else 0.45,
            expert=0.75 if sample.endswith("a") else 0.85,
        )
        for sample in ("B-sample-1a", "B-sample-1b")
        for seed in SEEDS
    )
    ensembles = build_probability_ensemble_surface(
        build_probability_surface(rows), expected_seed_ids=SEEDS
    )
    assert len(ensembles.rows) == 2
    assert {row.sample_id for row in ensembles.rows} == {
        "B-sample-1a",
        "B-sample-1b",
    }
    assert len({row.case_aggregation_receipt_hash for row in ensembles.rows}) == 1
    assert ensembles.rows[0].baseline_probability != ensembles.rows[1].baseline_probability


@pytest.mark.parametrize("positive_copies", (1, 2))
def test_mixed_label_case_surrogate_matches_case_equal_bacc_delta(
    tmp_path: Path, positive_copies: int
) -> None:
    samples = (("B-neg", 0, 0.6, 0.4),) + tuple(
        (f"B-pos-{index}", 1, 0.4, 0.6) for index in range(positive_copies)
    )
    seed_rows = tuple(
        _probability_row(
            seed=seed,
            case="B-mixed-case",
            sample=sample,
            baseline=baseline,
            expert=expert,
        )
        for sample, _, baseline, expert in samples
        for seed in SEEDS
    )
    seed_surface = build_probability_surface(seed_rows)
    features = build_action_feature_surface(
        build_probability_ensemble_surface(seed_surface, expected_seed_ids=SEEDS)
    )
    query_labels = tuple(
        HarpSourceLabelRow(
            center="B",
            case_id="B-mixed-case",
            sample_id=sample,
            label=truth,
        )
        for sample, truth, _, _ in samples
    )
    other_labels = tuple(
        HarpSourceLabelRow(
            center=center,
            case_id=f"{center}-case-{truth}",
            sample_id=f"{center}-sample-{truth}",
            label=truth,
        )
        for center in ("A", "C", "D")
        for truth in (0, 1)
    )
    artifact = tmp_path / f"predictions-{positive_copies}.bin"
    artifact.write_bytes(b"probability artifact")
    seal = build_durable_prediction_seal(
        probability_surface_hash=seed_surface.surface_hash,
        upstream_prediction_seal_hash=UPSTREAM_SEAL,
        prediction_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        prediction_row_count=len(seed_rows),
    )
    seal_path = tmp_path / f"seal-{positive_copies}.json"
    seal_path.write_text(json.dumps(seal.to_payload()), encoding="utf-8")
    labels = HarpSourceLabelCapability(
        centers=("A", "B", "C", "D"),
        seal=seal,
        seal_path=seal_path,
        prediction_artifact_path=artifact,
        label_loader=lambda: (*other_labels, *query_labels),
    ).open()
    responses = build_directional_response_surface(features, labels)
    selected = tuple(row for row in responses.rows if row.action_lambda == 1.0)
    # HARP ridge assigns each row 1/n_samples(case); the response multiplier
    # therefore integrates to the direct 0.5*(recall_0_delta+recall_1_delta).
    weighted_case_mean = sum(row.weighted_correctness_surrogate for row in selected) / len(selected)
    truth = [truth for _, truth, _, _ in samples]
    baseline_prediction = [int(baseline >= 0.5) for _, _, baseline, _ in samples]
    action_prediction = [int(expert >= 0.5) for _, _, _, expert in samples]
    direct_delta = 0.5 * sum(
        (
            sum(action_prediction[i] == label for i, value in enumerate(truth) if value == label)
            - sum(baseline_prediction[i] == label for i, value in enumerate(truth) if value == label)
        )
        / sum(value == label for value in truth)
        for label in (0, 1)
    )
    assert weighted_case_mean == pytest.approx(direct_delta)
    receipt = responses.receipts[0]
    assert receipt.case_sample_counts == (("B-mixed-case", len(samples)),)
    assert receipt.case_class_sample_counts == (
        ("B-mixed-case", 0, 1),
        ("B-mixed-case", 1, positive_copies),
    )


def test_default_workstation_producer_reconstructs_complete_target_menu_below_gpu_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real default producer while replacing only GPU/fit primitives."""

    from midogpp_thesis.cvae.routing.harp_action_surface import workstation_runtime
    from midogpp_thesis.cvae.routing.harp_action_surface.workstation_producer import (
        materialize_harp_probability_menu,
    )

    root = Path(__file__).resolve().parents[2]
    base = load_harp_stage60_config(
        root
        / "experiments/midogpp/stages/60_routing_and_composition/configs"
        / "uniform_b_v2_harp_target_support_surface_v1.yaml"
    )
    input_paths = {key: tmp_path / key for key in base.contract.input_path_keys}
    cache_root = input_paths["target_support_cache_root"]
    (cache_root / "manifests").mkdir(parents=True)
    (cache_root / "tables").mkdir(parents=True)
    index_path = cache_root / "manifests/cache_index.json"
    row_path = cache_root / "tables/row_index.csv"
    index_path.write_text(
        json.dumps({"schema_version": "midogpp_harp_label_blind_frame_cache_v1"}),
        encoding="utf-8",
    )
    row_path.write_text("row_id\nplaceholder\n", encoding="utf-8")
    (cache_root / "manifests/content_index.json").write_text(
        json.dumps(
            {
                "members": {
                    "manifests/cache_index.json": sha256_file(index_path),
                    "tables/row_index.csv": sha256_file(row_path),
                }
            }
        ),
        encoding="utf-8",
    )
    config = replace(
        base,
        artifact_root=tmp_path / "output",
        input_paths=input_paths,
        protocol={**dict(base.protocol), "input_status": "ready"},
    )
    readiness = HarpInputReadiness(
        surface=config.contract.surface,
        experiment_id=config.experiment_id,
        input_binding_sha256="1" * 64,
        reservation_sha256="2" * 64,
        cache_binding_sha256="3" * 64,
        manifest_sha256="4" * 64,
        attestation_sha256="5" * 64,
    )
    classifier = ClassifierSpec()
    generation_lock = SimpleNamespace(
        bank_lock_hash="a" * 16,
        generation_lock_hash="b" * 16,
    )
    monkeypatch.setattr(
        workstation_runtime,
        "_load_authoritative_generation_inputs",
        lambda _: (
            generation_lock,
            "6" * 64,
            "7" * 64,
            generation_lock.bank_lock_hash,
            generation_lock.generation_lock_hash,
            classifier,
            "8" * 64,
        ),
    )
    frame_path = config.artifact_root / "test-frame.npy"
    frame_path.parent.mkdir(parents=True)
    np.save(frame_path, np.zeros((9, 3840), dtype=np.float32), allow_pickle=False)
    centers = tuple(str(value) for value in config.protocol["center_universe"])
    frame = workstation_runtime._FrameCache(
        array_path=frame_path,
        rows_by_center={center: (f"{center}-sample",) for center in centers},
        cases_by_center={center: (f"{center}-case",) for center in centers},
        offsets_by_center={center: (index, index + 1) for index, center in enumerate(centers)},
        frame_hash_by_center={center: hashlib.sha256(center.encode()).hexdigest() for center in centers},
        cache_binding_hash=readiness.cache_binding_sha256,
        receipt_hash="9" * 64,
    )
    monkeypatch.setattr(
        workstation_runtime, "_load_and_stage_frame_cache", lambda *_: frame
    )

    @dataclass(frozen=True)
    class _Record:
        block_ordinal: int
        source_center: str
        training_seed: int
        generation_seed: int
        stream_id: str
        expert_lock_hash: str
        rows_per_class: int
        output_sha256: str

        def to_payload(self) -> dict[str, object]:
            return {
                "block_ordinal": self.block_ordinal,
                "source_center": self.source_center,
                "training_seed": self.training_seed,
                "generation_seed": self.generation_seed,
                "stream_id": self.stream_id,
                "expert_lock_hash": self.expert_lock_hash,
                "rows_per_class": self.rows_per_class,
                "output_sha256": self.output_sha256,
            }

    records = tuple(
        _Record(
            block_ordinal=ordinal,
            source_center=center,
            training_seed=train,
            generation_seed=generation,
            stream_id=f"{center}-{train}-{generation}",
                expert_lock_hash="e" * 16,
            rows_per_class=270,
            output_sha256=hashlib.sha256(
                f"{center}-{train}-{generation}".encode()
            ).hexdigest(),
        )
        for ordinal, (center, train, generation) in enumerate(
            (pair for center in centers for pair in ((center, 17, 17), (center, 17, 42), (center, 17, 101), (center, 42, 17), (center, 42, 42), (center, 42, 101), (center, 101, 17), (center, 101, 42), (center, 101, 101)))
        )
    )

    def _fake_source_cache(*args, **kwargs):
        del args
        source_root = Path(kwargs["root"])
        (source_root / "manifests").mkdir(parents=True, exist_ok=True)
        (source_root / "manifests/frozen_source_stream_lock.json").write_text(
            "{}", encoding="utf-8"
        )
        (source_root / "manifests/frozen_source_stream_index.json").write_text(
            "{}", encoding="utf-8"
        )
        source_array = source_root / "source.npy"
        np.save(source_array, np.zeros((1,), dtype=np.float32), allow_pickle=False)
        return SimpleNamespace(
            records=records,
            source_array_path=source_array,
            lock_hash="c" * 16,
            lock_payload={
                "source_stream_index_hash": "d" * 16,
                "source_array_sha256": sha256_file(source_array),
            },
        )

    monkeypatch.setattr(
        workstation_runtime, "materialize_frozen_source_streams", _fake_source_cache
    )

    def _fake_classifier_phase(tasks):
        completed = {}
        for ordinal, task in enumerate(tasks):
            values = np.asarray(
                [
                    [
                        0.4
                        if action["action_id"] == "B"
                        else 0.5
                        if action["action_id"] == "U"
                        else 0.6
                    ]
                    for action in task["actions"]
                ],
                dtype=np.float32,
            )
            path = Path(str(task["checkpoint_npz_path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, probabilities=values)
            completed[ordinal] = {
                "actions": [
                    {
                        "action_hash": action["action_hash"],
                        "composition_hash": "f" * 16,
                        "scaler_state_hash": "0" * 16,
                    }
                    for action in task["actions"]
                ]
            }
        return completed

    monkeypatch.setattr(
        workstation_runtime, "_execute_classifier_tasks", _fake_classifier_phase
    )
    menu = materialize_harp_probability_menu(config, readiness)
    assert len(menu.actions) == 90
    assert len(menu.cells) == 90 * len(EXACT_NINE_SEED_PAIRS)
    assert (config.artifact_root / workstation_runtime.LINEAGE_RECEIPT_MEMBER).is_file()
    assert all(len(cell.row_ids) == 1 for cell in menu.cells)
