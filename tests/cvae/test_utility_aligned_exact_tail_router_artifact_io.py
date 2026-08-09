from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router import (
    validation,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.artifact_io import (
    json_ready,
    persist_or_validate_json,
    read_json,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.input_contracts import (
    LabelFreeValidationFrame,
    ValidationRowIdentity,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.partitions import (
    build_case_fold_surface,
    build_fixed_partition_surface,
)
from midogpp_thesis.cvae.diagnostics.utility_aligned_exact_tail_router.runner_persistence import (
    persist_initial_surfaces,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_initial_persistence_detaches_real_frozen_locks_and_resumes(
    tmp_path: Path,
) -> None:
    frame = _label_free_44_case_frame()
    partitions = build_fixed_partition_surface(
        frame, config_contract_hash="contract"
    )
    case_folds = build_case_fold_surface(
        partitions, config_contract_hash="contract"
    )
    config = SimpleNamespace(contract_hash="contract", input_artifact_ids=("bank",))
    arguments = {
        "config": config,
        "provenance": {"bank": {"artifact_id": "bank"}},
        "frame": frame,
        "firewall": MappingProxyType(
            {
                "status": "PASS",
                "workspace_binding": MappingProxyType({"status": "PASS"}),
            }
        ),
        "partitions": partitions,
        "case_folds": case_folds,
    }

    persist_initial_surfaces(tmp_path, **arguments)
    persist_initial_surfaces(tmp_path, **arguments)

    assert read_json(tmp_path / "manifests/support_partition_lock.json") == json_ready(
        partitions.lock_payload
    )
    assert read_json(tmp_path / "manifests/case_fold_lock.json") == json_ready(
        case_folds.lock_payload
    )
    assert len(case_folds.folds) == 26
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_nested_frozen_json_resume_rejects_drift_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    payload = MappingProxyType(
        {
            "nested": MappingProxyType({"value": 1}),
            "sequence": ("a", "b"),
        }
    )
    persist_or_validate_json(path, payload)
    original = path.read_bytes()
    persist_or_validate_json(path, payload)

    with pytest.raises(ProtocolError, match="resumed JSON drifted"):
        persist_or_validate_json(
            path,
            MappingProxyType({"nested": MappingProxyType({"value": 2}), "sequence": ("a", "b")}),
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    (
        {"nested": {1: "non-string-key"}},
        {"value": float("nan")},
        {"value": object()},
    ),
)
def test_json_boundary_rejects_noncanonical_values_before_publication(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    path = tmp_path / "invalid.json"
    with pytest.raises(ProtocolError, match="Utility-aligned JSON"):
        persist_or_validate_json(path, payload)
    assert not path.exists()
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_independent_json_assertion_normalizes_then_detects_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lock.json"
    payload = MappingProxyType({"nested": MappingProxyType({"value": 1})})
    persist_or_validate_json(path, payload)
    validation._assert_json(path, payload)

    path.write_text(json.dumps({"nested": {"value": 2}}) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="derived JSON drifted"):
        validation._assert_json(path, payload)


def _label_free_44_case_frame() -> LabelFreeValidationFrame:
    rows: list[ValidationRowIdentity] = []
    rows_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    ordinal = 0
    for center_index, center in enumerate(CENTERS):
        center_rows: list[ValidationRowIdentity] = []
        case_count = 4 if center_index == 0 else 5
        for case_index in range(case_count):
            row = ValidationRowIdentity(
                row_ordinal=ordinal,
                manifest_row_index=ordinal,
                sample_id=f"sample::{center}::{case_index}",
                case_id=f"case::{center}::{case_index}",
                center=center,
            )
            rows.append(row)
            center_rows.append(row)
            ordinal += 1
        rows_by_center[center] = tuple(center_rows)
    return LabelFreeValidationFrame(
        embeddings=np.zeros((len(rows), 3_840), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding={"schema_version": "test", "labels_persisted": False},
    )
