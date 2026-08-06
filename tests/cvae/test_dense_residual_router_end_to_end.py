from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.dense_residual_router import config as config_module
from midogpp_thesis.cvae.diagnostics.dense_residual_router import execution
from midogpp_thesis.cvae.diagnostics.dense_residual_router import runner
from midogpp_thesis.cvae.diagnostics.dense_residual_router import validation
from midogpp_thesis.cvae.diagnostics.dense_residual_router.bundle import REQUIRED_FILES
from midogpp_thesis.cvae.diagnostics.dense_residual_router.config import (
    load_dense_residual_diagnostic_config,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    SUPPORT_CASE_COUNT,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    TRAINING_SEEDS,
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.dense_residual_router.execution import (
    CompatibilitySurface,
    LabelFreeValidationFrame,
)
from midogpp_thesis.cvae.routing.dense_residual_soft_router import (
    deterministic_case_partitions,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_validation_dense_residual_router_v1.yaml"
)


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_tiny_manifest_and_frame(
    manifest_path: Path,
) -> tuple[str, LabelFreeValidationFrame]:
    rows: list[ValidationRowIdentity] = []
    rows_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    manifest_rows: list[dict[str, object]] = []
    ordinal = 0

    for center in CENTERS:
        center_rows = tuple(
            ValidationRowIdentity(
                row_ordinal=ordinal + local_index,
                manifest_row_index=ordinal + local_index,
                sample_id=f"sample-{center}-{local_index}",
                case_id=f"case-{center}-{local_index}",
                center=center,
                split="val",
                partition_role="evaluation",
            )
            for local_index in range(4)
        )
        partition = deterministic_case_partitions(
            [row.sample_id for row in center_rows],
            [row.case_id for row in center_rows],
            target_center=center,
            support_case_count=SUPPORT_CASE_COUNT,
            namespace=SUPPORT_PARTITION_NAMESPACE,
            split_seed=SUPPORT_SPLIT_SEED,
        )
        evaluation_labels = dict(zip(partition.evaluation_indices, (0, 1), strict=True))
        for local_index, row in enumerate(center_rows):
            manifest_rows.append(
                {
                    "sample_id": row.sample_id,
                    "case_id": row.case_id,
                    "center": center,
                    "split": "val",
                    "label": evaluation_labels.get(local_index, 0),
                }
            )
        rows.extend(center_rows)
        rows_by_center[center] = center_rows
        ordinal += len(center_rows)

    manifest_path.parent.mkdir(parents=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "case_id", "center", "split", "label"),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    frame = LabelFreeValidationFrame(
        embeddings=np.zeros((len(rows), 3840), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding={
            "schema_version": "midogpp_dense_residual_smoke_cache_binding_v1",
            "row_count": len(rows),
            "center_count": len(CENTERS),
            "labels_persisted": False,
            "manifest_opened": False,
        },
    )
    return digest, frame


def _compatibility_surface(
    _config: object,
    _frame: LabelFreeValidationFrame,
    partitions: object,
) -> CompatibilitySurface:
    center_index = {center: index for index, center in enumerate(CENTERS)}
    calibrated = {
        query: {
            source: (center_index[source] - center_index[query]) / 10.0
            for source in CENTERS
            if source != query
        }
        for query in CENTERS
    }
    case_rows: list[Mapping[str, object]] = []
    for source in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for query in CENTERS:
                for row in partitions.support_rows_by_center[query]:
                    energy = (
                        1.0
                        + center_index[source] / 10.0
                        + center_index[query] / 100.0
                        + training_seed / 10_000.0
                    )
                    case_rows.append(
                        {
                            "schema_version": (
                                "midogpp_dense_residual_compatibility_case_energy_v1"
                            ),
                            "source_center": source,
                            "training_seed": training_seed,
                            "query_center": query,
                            "case_id": row.case_id,
                            "query_partition_role": "support",
                            "row_count": 1,
                            "marginal_variational_energy": energy,
                            "class_0_energy": energy,
                            "class_1_energy": energy + 0.01,
                            "class_0_common_reconstruction_mse": energy / 2.0,
                            "class_1_common_reconstruction_mse": energy / 2.0,
                            "class_0_normalized_ps_kl": energy / 2.0,
                            "class_1_normalized_ps_kl": energy / 2.0 + 0.01,
                            "class_prior_json": _compact_json([0.5, 0.5]),
                            "labels_used": False,
                            "exact_nelbo_claimed": False,
                        }
                    )
    score_rows: list[Mapping[str, object]] = []
    for query in CENTERS:
        for source in CENTERS:
            if source == query:
                continue
            score = calibrated[query][source]
            score_rows.append(
                {
                    "schema_version": "midogpp_dense_residual_compatibility_score_v1",
                    "query_center": query,
                    "source_center": source,
                    "training_seed_17_z": score,
                    "training_seed_42_z": score,
                    "training_seed_101_z": score,
                    "mean_calibrated_energy_z": score,
                    "query_support_case_count": SUPPORT_CASE_COUNT,
                    "replica_aggregation": (
                        "arithmetic_mean_all_three_no_seed_selection"
                    ),
                    "legal_development_outer_targets_json": _compact_json(
                        [
                            center
                            for center in CENTERS
                            if center not in {query, source}
                        ]
                    ),
                    "legal_target_candidate": True,
                    "query_support_labels_used": False,
                    "exact_nelbo_claimed": False,
                }
            )
    return CompatibilitySurface(
        calibrated_energy_by_query=calibrated,
        case_rows=tuple(case_rows),
        score_rows=tuple(score_rows),
    )


def test_runner_to_bundle_to_validator_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_root = tmp_path / "dense-residual-smoke"
    bank_root = tmp_path / "expert-bank"
    generation_root = tmp_path / "generation-lock"
    cache_root = tmp_path / "validation-cache"
    manifest_path = tmp_path / "validation-manifest" / "manifest.csv"
    for path in (artifact_root, bank_root, generation_root, cache_root):
        path.mkdir(parents=True)

    manifest_sha256, frame = _write_tiny_manifest_and_frame(manifest_path)
    monkeypatch.setattr(
        config_module,
        "EXPECTED_MANIFEST_SHA256",
        manifest_sha256,
    )
    config_payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config_payload["experiment"]["artifact_root"] = str(artifact_root)
    config_payload["inputs"].update(
        {
            "expert_bank_root": str(bank_root),
            "generation_lock_root": str(generation_root),
            "validation_cache_root": str(cache_root),
            "validation_manifest_path": str(manifest_path),
            "expected_manifest_sha256": manifest_sha256,
        }
    )
    resolved_config_path = artifact_root / "config.resolved.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    config = load_dense_residual_diagnostic_config(resolved_config_path)

    paths_by_artifact_id = {
        config.expert_bank_artifact_id: bank_root,
        config.generation_lock_artifact_id: generation_root,
        config.validation_cache_artifact_id: cache_root,
        config.validation_manifest_artifact_id: manifest_path.parent,
    }
    provenance = {
        "schema_version": "midogpp_input_artifacts_v2",
        "dataset_id": "midogpp",
        "experiment_id": (
            "midogpp.oracle."
            "uniform_b_v2_consumed_validation_dense_residual_router.v1"
        ),
        "stage": "90_oracles_and_diagnostics",
        "claim_scope": "diagnostic_only",
        "input_artifacts": [
            {
                "artifact_id": artifact_id,
                "resolved_path": str(paths_by_artifact_id[artifact_id]),
                "exists": True,
                "semantic_identities": {},
                "file_integrity": {},
            }
            for artifact_id in INPUT_ARTIFACT_IDS
        ],
    }
    provenance_path = artifact_root / "provenance/input_artifacts.json"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    generation_lock = SimpleNamespace(
        generation_lock_hash=config.expected_generation_lock_hash,
        bank_lock_hash=config.expected_bank_lock_hash,
    )
    source_plan = tuple(
        SimpleNamespace(
            source_center=source,
            training_seed=training_seed,
            generation_seed=generation_seed,
            stream_id=(
                f"smoke-{source}-{training_seed}-{generation_seed}"
            ),
        )
        for source in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    counters = {"generation": 0, "composition": 0, "fit": 0}

    def fake_generate_blocks(
        _config: object,
        _key_map: object,
        *,
        training_seed: int,
        generation_seed: int,
    ) -> dict[str, object]:
        counters["generation"] += 1
        return {
            source: SimpleNamespace(
                source_center=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
            )
            for source in CENTERS
        }

    def fake_compose(
        source_blocks: Mapping[str, object],
        allocation_per_class: Mapping[str, int],
        *,
        shuffle_seed_by_class: Mapping[str | int, int],
        total_per_class: int,
    ) -> object:
        counters["composition"] += 1
        assert set(source_blocks) == set(allocation_per_class)
        assert sum(allocation_per_class.values()) == total_per_class
        return SimpleNamespace(
            embeddings=np.zeros((2, 4), dtype=np.float32),
            labels=np.asarray([0, 1], dtype=np.uint8),
            composition_hash=stable_hash(
                {
                    "sources": sorted(source_blocks),
                    "allocation": dict(allocation_per_class),
                    "shuffle_seeds": dict(shuffle_seed_by_class),
                    "total_per_class": total_per_class,
                }
            ),
        )

    def fake_fit(
        run_config: object,
        _train_embeddings: np.ndarray,
        _train_labels: np.ndarray,
        eval_embeddings: np.ndarray,
    ) -> dict[str, object]:
        counters["fit"] += 1
        assert len(eval_embeddings) == 2
        return {
            "predictions": np.asarray([0, 1], dtype=np.uint8),
            "probabilities": np.asarray([0.1, 0.9], dtype=np.float32),
            "classes": (0, 1),
            "n_iter": (1,),
            "converged": True,
            "classifier_config_hash": run_config.classifier.config_hash,
            "scaler_state_hash": stable_hash("dense-residual-smoke-scaler"),
        }

    for module in (runner, validation):
        monkeypatch.setattr(module, "load_label_free_validation_frame", lambda _config: frame)
        monkeypatch.setattr(module, "compute_compatibility_surface", _compatibility_surface)
        monkeypatch.setattr(
            module,
            "_load_validated_generation_lock",
            lambda _config: generation_lock,
        )
    monkeypatch.setattr(execution, "source_generation_plan", lambda _lock: source_plan)
    monkeypatch.setattr(validation, "source_generation_plan", lambda _lock: source_plan)
    monkeypatch.setattr(execution, "_generate_seed_cell_blocks", fake_generate_blocks)
    monkeypatch.setattr(execution, "compose_prefix_blocks", fake_compose)
    monkeypatch.setattr(execution, "_fit_classifier", fake_fit)

    output = runner.run_dense_residual_router_diagnostic(config)

    assert output == artifact_root
    assert json.loads((output / "reports/run_state.json").read_text())["status"] == (
        "COMPLETE"
    )
    validation_report = json.loads(
        (output / "reports/validation_report.json").read_text()
    )
    assert validation_report["status"] == "PASS"
    assert validation_report["checks"]["status"] == "PASS"
    target_seal = json.loads(
        (output / "manifests/all_action_target_prediction_seal.json").read_text()
    )
    assert target_seal["status"] == (
        "SEALED_ALL_TARGET_ACTION_PREDICTIONS_BEFORE_ANY_LABEL_ACCESS"
    )
    assert target_seal["cell_count"] == 324
    assert len(target_seal["cells"]) == 324
    assert {
        member.relative_to(output).as_posix()
        for member in output.rglob("*")
        if member.is_file()
    } == set(REQUIRED_FILES)
    assert counters == {
        "generation": 18,
        "composition": 2187,
        "fit": 2187,
    }

    checks = validation.validate_dense_residual_router_bundle(output, config=config)
    assert checks["status"] == "PASS"
    assert checks["development_prediction_cell_count"] == 1944
    assert checks["target_prediction_cell_count"] == 324

    completed_counts = dict(counters)
    assert runner.run_dense_residual_router_diagnostic(config) == output
    assert counters == completed_counts
