import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cli import _load_config_for_validation
from domain_regime import MIDOGPP_DOMAIN_REGIME
from experiments.support_selection.midogpp_support_nelbo_routing import (
    MIDOGPP_SUPPORT_NELBO_ROUTING_NAME,
    parse_midogpp_support_nelbo_routing_config,
    run_midogpp_support_nelbo_routing,
)
from protocol import ProtocolError
from test_midogpp_domain_regime import ELIGIBLE_MIDOGPP_IDS, _write_midogpp_contract_fixture


def test_midogpp_support_nelbo_routing_writes_protocol_artifacts(tmp_path: Path) -> None:
    payload = _payload_with_inputs(tmp_path)
    cfg = parse_midogpp_support_nelbo_routing_config(payload, base_dir=tmp_path)

    root = run_midogpp_support_nelbo_routing(cfg)

    decisions = _read_csv(root / "tables" / "routing_decisions.csv")
    scores = _read_csv(root / "tables" / "support_nelbo_scores.csv")
    eval_rows = _read_csv(root / "tables" / "all_expert_eval_nelbo_matrix.csv")
    alignment = _read_csv(root / "tables" / "routing_to_eval_nelbo_alignment.csv")
    leakage = json.loads((root / "reports" / "leakage_report.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "manifests" / "protocol_manifest.json").read_text(encoding="utf-8"))

    assert cfg.name == MIDOGPP_SUPPORT_NELBO_ROUTING_NAME
    assert len(decisions) == len(ELIGIBLE_MIDOGPP_IDS)
    assert len(scores) == len(ELIGIBLE_MIDOGPP_IDS) * 8
    assert len(eval_rows) == len(ELIGIBLE_MIDOGPP_IDS) * 8
    assert len(alignment) == len(ELIGIBLE_MIDOGPP_IDS)
    assert leakage["status"] == "PASS"
    assert protocol["downstream_bacc_macro_f1_scope"] == "out_of_scope_v1"
    for row in decisions:
        candidates = json.loads(row["candidate_experts"])
        assert row["heldout_center"] not in candidates
        assert "4" not in candidates
        assert int(row["eligible_expert_count"]) == 8
        assert row["support_labels_used"] == "False"
        assert row["routing_uses_eval_nelbo"] == "0"
        assert row["adoption_eligible"] == "True"
        assert row["oracle_eligible"] == "False"
        assert row["decision_materialized_before_eval"] == "True"
    for row in eval_rows:
        assert row["selection_source"] == "diagnostic_only"
        assert row["adoption_eligible"] == "False"
        assert row["oracle_eligible"] == "True"
        assert row["routing_uses_eval_nelbo"] == "1"


def test_midogpp_support_nelbo_routing_rejects_strict_full_matrix(tmp_path: Path) -> None:
    payload = _payload_with_inputs(tmp_path)
    payload["run_matrix"]["strict_full_run_matrix"] = True

    with pytest.raises(ProtocolError, match="strict_full_run_matrix"):
        parse_midogpp_support_nelbo_routing_config(payload, base_dir=tmp_path)


def test_midogpp_support_nelbo_routing_rejects_target_candidate(tmp_path: Path) -> None:
    payload = _payload_with_inputs(tmp_path)
    support_path = Path(payload["inputs"]["support_nelbo_scores_path"])
    rows = _read_csv(support_path)
    rows[0]["expert_id"] = rows[0]["heldout_center"]
    _write_csv(support_path, rows)
    cfg = parse_midogpp_support_nelbo_routing_config(payload, base_dir=tmp_path)

    with pytest.raises(ProtocolError, match="candidate pool"):
        run_midogpp_support_nelbo_routing(cfg)


def test_midogpp_support_nelbo_routing_validation_dispatch(tmp_path: Path) -> None:
    payload = _payload_with_inputs(tmp_path)
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = _load_config_for_validation(path)

    assert cfg.name == MIDOGPP_SUPPORT_NELBO_ROUTING_NAME


def _payload_with_inputs(tmp_path: Path) -> dict:
    artifact, cache_report = _write_midogpp_contract_fixture(tmp_path)
    feature_cache_root = tmp_path / "sail/artifacts/pathology_embeddings_midogpp_annotation_patch_v1/virchow2"
    _write_feature_cache(feature_cache_root)
    inputs_root = tmp_path / "cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1/inputs"
    inputs_root.mkdir(parents=True)
    expert_manifest = inputs_root / "source_expert_manifest.csv"
    support_scores = inputs_root / "frozen_support_nelbo_scores.csv"
    eval_matrix = inputs_root / "frozen_eval_nelbo_matrix.csv"
    _write_expert_manifest(expert_manifest)
    _write_support_scores(support_scores)
    _write_eval_matrix(eval_matrix)
    return {
        "experiment": {
            "name": MIDOGPP_SUPPORT_NELBO_ROUTING_NAME,
            "artifact_root": str(tmp_path / "cvae_rebuild/artifacts/midogpp/support_nelbo_routing_v1"),
            "primary_method": "support_nelbo_top1_marginal_unlabeled",
        },
        "inputs": {
            "feature_cache_root": str(feature_cache_root),
            "dataset_contract_artifact_root": str(artifact),
            "cache_report_path": str(cache_report),
            "source_expert_manifest_path": str(expert_manifest),
            "support_nelbo_scores_path": str(support_scores),
            "eval_nelbo_matrix_path": str(eval_matrix),
            "backbone": "virchow2",
        },
        "run_matrix": {
            "domain_regime": MIDOGPP_DOMAIN_REGIME,
            "strict_full_run_matrix": False,
            "strict_available_seed_domain_coverage": True,
            "experiment_seeds": [42],
            "heldout_centers": list(ELIGIBLE_MIDOGPP_IDS),
            "support_size": 2,
            "support_seeds": [101],
        },
        "routing": {
            "primary_score": "calibrated_marginal_support_nelbo",
            "support_sampler": "random_unlabeled_sample_ids",
            "selection_rule": "min_calibrated_support_nelbo",
            "tie_breaking": "expert_id_ascending",
            "weighting_policy": "none",
            "softmax_tau": None,
            "top_k": None,
            "weight_aggregation_target": "none",
        },
        "scoring": {
            "nelbo_target": "marginal_unlabeled",
            "class_prior_source": "uniform",
            "calibration_source": "source_validation_only",
            "scorer_config_hash": "fixture_hash",
            "feature_frame_policy": "expert_local_source_only",
        },
        "midogpp_support_nelbo_routing": {"enabled": True},
    }


def _write_feature_cache(root: Path) -> None:
    cache_path = root / "seed42/embeddings/test.npz"
    cache_path.parent.mkdir(parents=True)
    metadata = []
    embeddings = []
    row_id = 0
    for center in ELIGIBLE_MIDOGPP_IDS:
        for idx in range(4):
            metadata.append(
                {
                    "sample_id": f"sample_{center}_{idx}",
                    "case_id": f"case_{center}_{idx}",
                    "center": center,
                    "label": idx % 2,
                }
            )
            embeddings.append([float(row_id), float(idx)])
            row_id += 1
    np.savez(cache_path, embeddings=np.asarray(embeddings), metadata_json=json.dumps(metadata))


def _write_expert_manifest(path: Path) -> None:
    rows = [
        {
            "experiment_seed": 42,
            "expert_id": center,
            "source_domain_id": center,
            "checkpoint_path": f"experts/{center}.pt",
            "checkpoint_hash": f"hash_{center}",
            "source_only": True,
            "frozen": True,
            "feature_frame_hash": f"frame_{center}",
        }
        for center in ELIGIBLE_MIDOGPP_IDS
    ]
    _write_csv(path, rows)


def _write_support_scores(path: Path) -> None:
    rows = []
    for heldout in ELIGIBLE_MIDOGPP_IDS:
        for expert in ELIGIBLE_MIDOGPP_IDS:
            if expert == heldout:
                continue
            value = float(int(expert)) + 0.1
            rows.append(
                {
                    "experiment_seed": 42,
                    "heldout_center": heldout,
                    "support_seed": 101,
                    "support_size": 2,
                    "expert_id": expert,
                    "raw_support_nelbo": value,
                    "calibrated_support_nelbo": value,
                    "support_n": 2,
                    "support_se": 0.01,
                }
            )
    _write_csv(path, rows)


def _write_eval_matrix(path: Path) -> None:
    rows = []
    for heldout in ELIGIBLE_MIDOGPP_IDS:
        for expert in ELIGIBLE_MIDOGPP_IDS:
            if expert == heldout:
                continue
            rows.append(
                {
                    "experiment_seed": 42,
                    "heldout_center": heldout,
                    "expert_id": expert,
                    "eval_mean_nelbo": float(int(expert)) + 0.2,
                    "eval_n": 2,
                }
            )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
