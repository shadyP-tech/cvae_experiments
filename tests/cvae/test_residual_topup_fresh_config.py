from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import yaml

from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.config import (
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    MINIMUM_PHYSICAL_RAM_BYTES,
    OUTPUT_ARTIFACT_ID,
    SOURCE_BLOCK_PER_CLASS,
    canonical_runtime_payload,
    load_residual_topup_fresh_config,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh import (
    workstation,
)
from midogpp_thesis.cvae.protocol import ProtocolError


REQUIRED_ENVIRONMENT = workstation.REQUIRED_ENVIRONMENT
WorkstationProbes = workstation.WorkstationProbes
WorkstationSnapshot = workstation.WorkstationSnapshot
publish_validated_scratch_file = workstation.publish_validated_scratch_file
run_workstation_preflight = workstation.run_workstation_preflight
validate_workstation_snapshot = workstation.validate_workstation_snapshot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "experiments/midogpp/stages/70_frozen_policy_downstream/configs"
    / "uniform_b_v2_residual_topup_b_u_g_s_fresh_v1.yaml"
)


def _payload() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _gpu_rows(*, free_mib: int = 22000, name: str = "NVIDIA RTX A5000"):
    return (
        {
            "index": 0,
            "name": name,
            "memory_total_mib": 24564,
            "memory_free_mib": free_mib,
        },
        {
            "index": 1,
            "name": name,
            "memory_total_mib": 24564,
            "memory_free_mib": free_mib,
        },
    )


def _snapshot() -> WorkstationSnapshot:
    return WorkstationSnapshot(
        available_cpu_count=24,
        physical_ram_bytes=125 * 1024**3,
        artifact_disk_free_bytes=20 * 1024**3,
        gpu_rows=_gpu_rows(),
        spawn_available=True,
        parent_cuda_context_initialized=False,
    )


def _probes(*, scratch_writable: bool = True) -> WorkstationProbes:
    return WorkstationProbes(
        available_cpu_count=lambda: 24,
        physical_ram_bytes=lambda: 125 * 1024**3,
        disk_free_bytes=lambda _path: 20 * 1024**3,
        gpu_rows=_gpu_rows,
        spawn_available=lambda: True,
        parent_cuda_context_initialized=lambda: False,
        directory_writable=lambda path: (
            scratch_writable if path == Path("/data/local") else True
        ),
        atomic_replace_supported=lambda path: (
            scratch_writable if path == Path("/data/local") else True
        ),
    )


def test_loads_the_exact_planned_fresh_stage70_contract() -> None:
    config = load_residual_topup_fresh_config(CONFIG_PATH)
    assert config.experiment_id == EXPERIMENT_ID
    assert config.output_artifact_id == OUTPUT_ARTIFACT_ID
    assert config.input_artifact_ids == INPUT_ARTIFACT_IDS
    assert config.protocol["target_support_evaluation_case_disjoint"] is True
    assert config.evaluation["primary_endpoint"] == (
        "all_nine_seed_probability_ensemble_bacc"
    )
    assert config.classifier.C == 0.01
    assert config.runtime["source_block_per_class"] == SOURCE_BLOCK_PER_CLASS
    assert config.claim_boundary["current_checkout_has_eligible_fresh_surface"] is False
    assert len(config.contract_hash) == 16


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("protocol", "evaluation_labels_available_to_prediction", True),
        ("evaluation", "primary_endpoint", "seed_cell_mean_bacc"),
        ("classifier", "C", 1.0),
        ("runtime", "source_block_per_class", 255),
        ("claim_boundary", "consumed_stage90_used", True),
    ],
)
def test_config_fails_closed_on_scientific_or_runtime_drift(
    tmp_path: Path,
    section: str,
    key: str,
    replacement: object,
) -> None:
    payload = _payload()
    nested = payload[section]
    assert isinstance(nested, dict)
    nested[key] = replacement
    with pytest.raises(ProtocolError, match="drifted"):
        load_residual_topup_fresh_config(_write_config(tmp_path, payload))


def test_config_rejects_placeholders_and_wrong_artifact_uris(tmp_path: Path) -> None:
    placeholder = _payload()
    placeholder_inputs = placeholder["inputs"]
    assert isinstance(placeholder_inputs, dict)
    placeholder_inputs["fresh_target_cache_root"] = "PENDING"
    with pytest.raises(ProtocolError, match="placeholder"):
        load_residual_topup_fresh_config(_write_config(tmp_path, placeholder))

    wrong_uri = _payload()
    wrong_inputs = wrong_uri["inputs"]
    assert isinstance(wrong_inputs, dict)
    wrong_inputs["policy_root"] = "artifact://wrong-policy"
    with pytest.raises(ProtocolError, match="location"):
        load_residual_topup_fresh_config(_write_config(tmp_path, wrong_uri))


def test_snapshot_validation_accepts_only_the_frozen_workstation() -> None:
    report = validate_workstation_snapshot(
        _snapshot(), runtime=canonical_runtime_payload()
    )
    assert report["available_cpu_count"] == 24
    assert report["parent_cuda_context_initialized"] is False
    assert [row["index"] for row in report["gpus"]] == [0, 1]


@pytest.mark.parametrize(
    ("snapshot", "message"),
    [
        (replace(_snapshot(), available_cpu_count=11), "fewer than 12 CPUs"),
        (
            replace(
                _snapshot(),
                physical_ram_bytes=MINIMUM_PHYSICAL_RAM_BYTES - 1,
            ),
            "less than 100 GiB RAM",
        ),
        (replace(_snapshot(), spawn_available=False), "requires multiprocessing spawn"),
        (
            replace(_snapshot(), parent_cuda_context_initialized=True),
            "parent CUDA context",
        ),
        (
            replace(_snapshot(), gpu_rows=_gpu_rows(free_mib=17999)),
            "less than 18 GiB free",
        ),
        (
            replace(_snapshot(), gpu_rows=_gpu_rows(name="NVIDIA RTX 4090")),
            "not an RTX A5000",
        ),
    ],
)
def test_snapshot_validation_fails_closed(snapshot, message: str) -> None:
    with pytest.raises(ProtocolError, match=message):
        validate_workstation_snapshot(
            snapshot,
            runtime=canonical_runtime_payload(),
        )


def test_preflight_enforces_environment_schedule_and_explicit_scratch(
    tmp_path: Path,
) -> None:
    report = run_workstation_preflight(
        tmp_path,
        runtime=canonical_runtime_payload(),
        probes=_probes(scratch_writable=False),
        environment=REQUIRED_ENVIRONMENT,
    )
    assert report["optional_local_scratch_enabled"] is False
    assert report["scratch_authoritative"] is False
    assert report["generation_workers_per_device"] == 1
    assert report["source_block_per_class"] == 256
    assert report["classifier_workers"] == CLASSIFIER_WORKERS
    assert report["classifier_threads_per_worker"] == CLASSIFIER_THREADS_PER_WORKER
    assert report["tf32_enabled"] is False
    assert report["amp_enabled"] is False
    assert report["gpu_and_cpu_phases_disjoint"] is True

    with pytest.raises(ProtocolError, match="not writable"):
        run_workstation_preflight(
            tmp_path,
            runtime=canonical_runtime_payload(),
            probes=_probes(scratch_writable=False),
            environment=REQUIRED_ENVIRONMENT,
            enable_optional_local_scratch=True,
        )

    scratch_report = run_workstation_preflight(
        tmp_path,
        runtime=canonical_runtime_payload(),
        probes=_probes(scratch_writable=True),
        environment=REQUIRED_ENVIRONMENT,
        enable_optional_local_scratch=True,
    )
    assert scratch_report["optional_local_scratch_root"] == "/data/local"
    assert scratch_report["canonical_publication_required"] is True
    assert scratch_report["scratch_authoritative"] is False

    drifted_environment = dict(REQUIRED_ENVIRONMENT)
    drifted_environment["OMP_NUM_THREADS"] = "3"
    with pytest.raises(ProtocolError, match="environment"):
        run_workstation_preflight(
            tmp_path,
            runtime=canonical_runtime_payload(),
            probes=_probes(),
            environment=drifted_environment,
        )


def test_scratch_publication_validates_then_atomically_copies_to_canonical(
    tmp_path: Path,
) -> None:
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    scratch_file = scratch_root / "prediction.npy"
    content = b"sealed fresh prediction bytes\n"
    scratch_file.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    canonical_file = tmp_path / "canonical" / "prediction.npy"

    published = publish_validated_scratch_file(
        scratch_file,
        canonical_file,
        expected_sha256=digest,
        scratch_root=scratch_root,
    )
    assert published == canonical_file.resolve()
    assert published.read_bytes() == content
    assert not published.is_relative_to(scratch_root.resolve())
    assert publish_validated_scratch_file(
        scratch_file,
        canonical_file,
        expected_sha256=digest,
        scratch_root=scratch_root,
    ) == published

    with pytest.raises(ProtocolError, match="hash mismatched"):
        publish_validated_scratch_file(
            scratch_file,
            tmp_path / "canonical" / "bad.npy",
            expected_sha256="0" * 64,
            scratch_root=scratch_root,
        )
