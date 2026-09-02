from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

from midogpp_thesis.cvae.runtime.artifact_io import read_json
from midogpp_thesis.cvae.runtime.harp_v8_execution.contracts import (
    ActionKind,
    PrelabelRouteSet,
    RoutedCase,
)
from midogpp_thesis.cvae.runtime.harp_v8_execution.stores import (
    read_prelabel_routes,
    write_prelabel_routes,
)
from midogpp_thesis.cvae.runtime.harp_v8_execution import gpu_surface, json_payloads


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_prelabel_store_canonicalizes_production_tuple_payload(tmp_path: Path) -> None:
    baseline = np.asarray((0.2, 0.8), dtype=np.float32)
    case = RoutedCase(
        outer_target_id="H",
        case_id="case-0",
        sample_ids=("s0", "s1"),
        selected_kind=ActionKind.B,
        selected_source_id=None,
        reason="EXACT_B_ACTION_CERTIFICATE_FAILED",
        baseline_probabilities=baseline,
        uniform_probabilities=np.asarray((0.6, 0.4), dtype=np.float32),
        selected_probabilities=baseline.copy(),
        routed_probabilities=baseline.copy(),
        decision_payload={
            "deployed_action": "CERTIFIED_EXACT_TOP1_PHYSICAL_OR_EXACT_B",
            # Production policy code deliberately uses immutable tuples.
            "failed_gates": ("ACTION_BRIER_UCB_FAILED", "ACTION_HARM_UCB_FAILED"),
        },
    )
    routes = PrelabelRouteSet(
        cases=(case,),
        policy_hash=SHA_A,
        model_hash=SHA_B,
        target_action_hash=SHA_C,
    )

    receipt = write_prelabel_routes(tmp_path / "routes", routes)
    restored = read_prelabel_routes(receipt.root)
    manifest = read_json(receipt.manifest_path)

    assert restored.route_hash == routes.route_hash
    assert restored.cases[0].decision_hash == case.decision_hash
    assert manifest["cases"][0]["decision_payload"]["failed_gates"] == [
        "ACTION_BRIER_UCB_FAILED",
        "ACTION_HARM_UCB_FAILED",
    ]


def test_nested_mappingproxy_payload_is_normalized_only_at_json_boundary(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": "harp-v8-compatibility-test",
        "support_binding": MappingProxyType(
            {
                "center": "0",
                "case_ids": ("case-a", "case-b"),
                "labels_present": False,
            }
        ),
        "replicas": (MappingProxyType({"source_center": "1", "seed": 17}),),
    }
    path = tmp_path / "compatibility.json"
    normalized = json_payloads.plain_json_mapping(payload)

    gpu_surface._persist_or_validate_json(path, payload)
    first = path.read_bytes()
    gpu_surface._persist_or_validate_json(path, payload)

    assert read_json(path) == normalized
    assert path.read_bytes() == first
    assert normalized["support_binding"]["case_ids"] == ["case-a", "case-b"]
