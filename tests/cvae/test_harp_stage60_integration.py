from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.harp_protocol.hashing import canonical_hash
from midogpp_thesis.cvae.routing.harp_stage60 import (
    ACTION_SURFACE,
    load_harp_stage60_config,
)
from midogpp_thesis.cvae.routing.harp_stage60.runner import (
    HarpBuiltProduct,
    HarpDurablePrelabelSeal,
    HarpRunReceipt,
    run_harp_stage60_surface,
)
from midogpp_thesis.workspace.runtime import MidogppWorkspace


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "experiments/midogpp/stages/60_routing_and_composition/configs"
    / "uniform_b_v2_harp_action_surface_v1.yaml"
)


def test_harp_configs_are_path_independent_and_closed_schema(tmp_path: Path) -> None:
    first = load_harp_stage60_config(CONFIG)
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["experiment"]["artifact_root"] = str(tmp_path / "other-output")
    for key in payload["inputs"]["paths"]:
        payload["inputs"]["paths"][key] = str(tmp_path / key)
    moved = tmp_path / "config.yaml"
    moved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    second = load_harp_stage60_config(moved)
    assert first.contract_hash == second.contract_hash

    payload["unexpected"] = True
    moved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="top-level schema"):
        load_harp_stage60_config(moved)


def test_workspace_registers_harp_as_planned_and_nonrunnable() -> None:
    workspace = MidogppWorkspace.load(ROOT)
    workspace.validate()
    for experiment_id in (
        "midogpp.routing_and_composition.uniform_b_v2_harp_action_surface.v1",
        "midogpp.routing_and_composition.uniform_b_v2_harp_target_support_surface.v1",
        "midogpp.routing_and_composition.uniform_b_v2_harp_policy_lock.v1",
    ):
        experiment = workspace.get_experiment(experiment_id)
        assert experiment.status == "planned"
        assert experiment.runnable is False
        assert all("sceptre" not in value.lower() for value in experiment.input_artifact_ids)
        assert all("stage90" not in value.lower() for value in experiment.input_artifact_ids)


def test_planned_readiness_rejects_before_adapter_or_output_mutation(tmp_path: Path) -> None:
    config = replace(load_harp_stage60_config(CONFIG), artifact_root=tmp_path / "output")
    adapter = _RecordingAdapter(tmp_path)
    with pytest.raises(ProtocolError, match="remains planned"):
        run_harp_stage60_surface(
            config,
            adapter=adapter,
            workspace_validator=lambda value: adapter.events.append("workspace"),
        )
    assert adapter.events == ["workspace"]
    assert not config.artifact_root.exists()


def test_action_runner_opens_source_labels_only_after_durable_global_seal(
    tmp_path: Path,
) -> None:
    base = load_harp_stage60_config(CONFIG)
    reservation = tmp_path / "reservation"
    attestation = reservation / ACTION_SURFACE.readiness_member
    paths = {
        key: tmp_path / key for key in ACTION_SURFACE.input_path_keys
    }
    paths["readiness_attestation_path"] = attestation
    config = replace(
        base,
        artifact_root=tmp_path / "output",
        input_paths=paths,
        protocol={**dict(base.protocol), "input_status": "ready"},
    )
    attestation.parent.mkdir(parents=True)
    hashes = {
        "input_binding_sha256": "1" * 64,
        "reservation_sha256": "2" * 64,
        "cache_binding_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
    }
    attestation.write_text(
        json.dumps(
            {
                "schema_version": "midogpp_harp_input_readiness_v1",
                "status": "READY",
                "surface": ACTION_SURFACE.surface,
                "experiment_id": ACTION_SURFACE.experiment_id,
                "input_artifact_ids": list(ACTION_SURFACE.input_artifact_ids),
                "dataset_family": "MIDOG++",
                "whole_case_disjoint": True,
                "outer_target_excluded_before_transform": True,
                "target_support_labels_used": False,
                "target_evaluation_labels_used": False,
                "stage50_artifacts_used": False,
                "stage90_artifacts_used": False,
                "consumed_test_rows_used": False,
                **hashes,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    adapter = _RecordingAdapter(tmp_path)
    receipt = run_harp_stage60_surface(
        config,
        adapter=adapter,
        workspace_validator=lambda value: adapter.events.append("workspace"),
    )
    assert receipt.status == "COMPLETE"
    assert adapter.events == [
        "workspace",
        "preflight",
        "materialize",
        "open-labels",
        "build",
        "persist",
        "validate",
    ]


class _RecordingAdapter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.events: list[str] = []

    def validate_completed_bundle(self, config):
        self.events.append("validate")
        return HarpRunReceipt(
            surface=config.contract.surface,
            artifact_root=config.artifact_root,
            product_hash="a" * 64,
            validation_hash="b" * 64,
        )

    def preflight(self, config, readiness):
        self.events.append("preflight")

    def materialize_and_seal_label_free_menu(self, config, readiness):
        self.events.append("materialize")
        seal_path = config.artifact_root / "manifests/global_prediction_seal.json"
        seal_path.parent.mkdir(parents=True)
        unhashed = {
            "schema_version": "midogpp_harp_durable_prelabel_seal_v1",
            "status": "SEALED_COMPLETE_LABEL_FREE_MENU",
            "surface": config.contract.surface,
            "probability_menu_hash": "c" * 64,
            "row_identity_hash": "d" * 64,
            "target_support_labels_used": False,
            "target_evaluation_labels_used": False,
            "source_development_labels_opened": False,
        }
        seal_hash = canonical_hash(unhashed)
        seal_path.write_text(
            json.dumps({**unhashed, "seal_hash": seal_hash}, sort_keys=True),
            encoding="utf-8",
        )
        return HarpDurablePrelabelSeal(
            surface=config.contract.surface,
            seal_path=seal_path,
            seal_hash=seal_hash,
            probability_menu_hash="c" * 64,
            row_identity_hash="d" * 64,
        )

    def open_source_development_labels(self, config, seal):
        assert Path(seal.seal_path).is_file()
        seal.verify_durable()
        self.events.append("open-labels")
        return object()

    def build_product(self, config, seal, source_development_labels):
        assert source_development_labels is not None
        self.events.append("build")
        payload = {
            "schema_version": "test_harp_product_v1",
            "seal_hash": seal.seal_hash,
        }
        return HarpBuiltProduct(
            surface=config.contract.surface,
            payload=payload,
            product_hash=canonical_hash(payload),
            source_development_labels_used_for_scoring_only=True,
        )

    def persist_product(self, config, seal, product):
        self.events.append("persist")
        state = config.artifact_root / "reports/run_state.json"
        state.parent.mkdir(parents=True)
        state.write_text('{"status":"COMPLETE"}', encoding="utf-8")
        return config.artifact_root
