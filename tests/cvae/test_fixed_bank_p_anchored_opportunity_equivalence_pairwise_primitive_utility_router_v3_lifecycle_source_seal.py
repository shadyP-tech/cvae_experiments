from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.authorization_contract import (
    authorization_amendment_bytes,
    validate_authorization_amendment_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.config import (
    ResolvedV3ConfigBundle,
    build_authorization_ready_config,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.inputs import (
    ResolvedDirectInput,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    DIRECT_INPUT_ARTIFACT_IDS,
    DIRECT_INPUT_ROLES,
    EXPECTED_INPUT_KINDS,
    INPUT_RELATIVE_MEMBERS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.lifecycle_source_seal import (
    build_lifecycle_source_seal,
    validate_lifecycle_source_seal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.protocol import (
    frozen_protocol_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3 import (
    run_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_supervision import (
    SourceTrainingSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.terminal import authority
from midogpp_thesis.cvae.protocol import ProtocolError


SOURCE_RECEIPT = "1" * 64


def _lifecycle_tree(root: Path, *, marker: str) -> Path:
    entrypoint = root / "src/midogpp_thesis/oe_ppur_v3.py"
    preparation = (
        root
        / "src/midogpp_thesis/cvae/diagnostics/oe_ppur_v3_preparation"
    )
    preparation.mkdir(parents=True)
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text(f'MARKER = "{marker}"\n', encoding="utf-8")
    (preparation / "__init__.py").write_text(
        f'MARKER = "{marker}"\n', encoding="utf-8"
    )
    return root


def test_lifecycle_seal_rebuild_detects_executable_tree_drift(
    tmp_path: Path,
) -> None:
    root = _lifecycle_tree(tmp_path / "repository", marker="before")
    receipt = build_lifecycle_source_seal(root)

    assert receipt.lifecycle_source_sha256 == (
        receipt.lifecycle_source_seal_sha256
    )
    assert receipt.member_count == 2
    assert validate_lifecycle_source_seal(receipt) == receipt

    (root / "src/midogpp_thesis/oe_ppur_v3.py").write_text(
        'MARKER = "after"\n', encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="lifecycle source bytes drifted"):
        validate_lifecycle_source_seal(receipt)


def test_amendment_validation_requires_the_exact_live_lifecycle_hash() -> None:
    protocol_hash = str(frozen_protocol_payload()["protocol_hash"])
    first = "2" * 64
    second = "3" * 64
    raw = authorization_amendment_bytes(
        source_contract_hash=SOURCE_RECEIPT,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=first,
    )
    payload = json.loads(raw)

    assert payload["lifecycle_source_seal_sha256"] == first
    with pytest.raises(ProtocolError, match="amendment drifted"):
        validate_authorization_amendment_payload(
            payload,
            source_contract_hash=SOURCE_RECEIPT,
            protocol_hash=protocol_hash,
            lifecycle_source_seal_sha256=second,
        )


def test_terminal_authority_rejects_lifecycle_drift_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = build_lifecycle_source_seal(
        _lifecycle_tree(tmp_path / "first-repository", marker="first")
    )
    second = build_lifecycle_source_seal(
        _lifecycle_tree(tmp_path / "second-repository", marker="second")
    )
    protocol_hash = str(frozen_protocol_payload()["protocol_hash"])
    amendment_raw = authorization_amendment_bytes(
        source_contract_hash=SOURCE_RECEIPT,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=first.lifecycle_source_sha256,
    )
    parent_raw = (
        json.dumps(
            {
                "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
                "split": "test",
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    parent = tmp_path / "inputs/parent/reports/test_consumption_ledger.json"
    amendment = tmp_path / "inputs/amendment" / INPUT_RELATIVE_MEMBERS[6]
    parent.parent.mkdir(parents=True)
    amendment.parent.mkdir(parents=True)
    parent.write_bytes(parent_raw)
    amendment.write_bytes(amendment_raw)
    monkeypatch.setattr(
        authority,
        "EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256",
        hashlib.sha256(parent_raw).hexdigest(),
    )
    config = build_authorization_ready_config(
        source_supervision_content_sha256=SOURCE_RECEIPT,
        source_supervision_row_order_sha256="4" * 64,
        source_supervision_producer_seal_sha256="5" * 64,
        source_supervision_recomputation_receipt_sha256="6" * 64,
        authorization_amendment_sha256=hashlib.sha256(amendment_raw).hexdigest(),
    )
    locations = []
    for index, (role, artifact_id, kind, relative) in enumerate(
        zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            EXPECTED_INPUT_KINDS,
            INPUT_RELATIVE_MEMBERS,
            strict=True,
        )
    ):
        if index == 5:
            path = parent
        elif index == 6:
            path = amendment
        else:
            root = tmp_path / f"inputs/input-{index}"
            path = root / relative if relative else root
        locations.append(ResolvedDirectInput(role, artifact_id, kind, path))
    bundle = ResolvedV3ConfigBundle(
        config=config,
        source_path=tmp_path / "output/config.resolved.yaml",
        artifact_root=tmp_path / "output",
        input_bindings=tuple(locations),
    )

    monkeypatch.setattr(
        authority,
        "build_lifecycle_source_seal",
        lambda: first,
    )
    assert authority.validate_resolved_terminal_authority(
        bundle,
        source_training_surface_receipt_hash=SOURCE_RECEIPT,
    ) == first

    monkeypatch.setattr(
        authority,
        "build_lifecycle_source_seal",
        lambda: second,
    )
    with pytest.raises(ProtocolError, match="amendment drifted"):
        authority.validate_resolved_terminal_authority(
            bundle,
            source_training_surface_receipt_hash=SOURCE_RECEIPT,
        )


def test_launch_admission_stops_on_lifecycle_drift_before_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    scratch = tmp_path / "scratch"
    config = build_authorization_ready_config(
        source_supervision_content_sha256=SOURCE_RECEIPT,
        source_supervision_row_order_sha256="4" * 64,
        source_supervision_producer_seal_sha256="5" * 64,
        source_supervision_recomputation_receipt_sha256="6" * 64,
        authorization_amendment_sha256="7" * 64,
    )
    bundle = ResolvedV3ConfigBundle(
        config=config,
        source_path=artifact / "config.resolved.yaml",
        artifact_root=artifact,
        input_bindings=tuple(
            ResolvedDirectInput(
                role,
                artifact_id,
                kind,
                (
                    tmp_path / f"input-{index}" / relative_member
                    if relative_member
                    else tmp_path / f"input-{index}"
                ),
            )
            for index, (role, artifact_id, kind, relative_member) in enumerate(
                zip(
                    DIRECT_INPUT_ROLES,
                    DIRECT_INPUT_ARTIFACT_IDS,
                    EXPECTED_INPUT_KINDS,
                    INPUT_RELATIVE_MEMBERS,
                    strict=True,
                )
            )
        ),
    )
    surface = object.__new__(SourceTrainingSurface)
    object.__setattr__(
        surface,
        "receipt",
        SimpleNamespace(
            receipt_hash=SOURCE_RECEIPT,
            row_order_sha256="4" * 64,
            contract=SimpleNamespace(producer_source_seal_sha256="5" * 64),
            compiler_recomputation_receipt_sha256="6" * 64,
        ),
    )
    monkeypatch.setattr(
        run_admission,
        "validate_source_seal",
        lambda _value: SimpleNamespace(),
    )
    monkeypatch.setattr(
        run_admission,
        "validate_launch_roots",
        lambda _artifact, _scratch: (artifact, scratch),
    )
    monkeypatch.setattr(
        run_admission,
        "assert_canonical_output_root",
        lambda _artifact: None,
    )
    monkeypatch.setattr(
        run_admission,
        "_validate_input_paths",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_admission,
        "validate_live_producer_seal_binding",
        lambda **_kwargs: None,
    )

    def fail_lifecycle(*_args: object, **_kwargs: object) -> object:
        raise ProtocolError("OE-PPUR v3 lifecycle source bytes drifted.")

    monkeypatch.setattr(
        run_admission,
        "validate_resolved_terminal_authority",
        fail_lifecycle,
    )
    monkeypatch.setattr(
        run_admission,
        "validate_workspace_input_provenance",
        lambda *_args, **_kwargs: pytest.fail(
            "workspace provenance must remain closed after lifecycle drift"
        ),
    )

    with pytest.raises(ProtocolError, match="lifecycle source bytes drifted"):
        run_admission.admit_seven_input_execution(
            bundle,
            artifact_root=artifact,
            scratch_root=scratch,
            source_seal=object(),  # type: ignore[arg-type]
            source_surface=surface,
        )
