from __future__ import annotations

from pathlib import Path

import pytest

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
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.run_admission import (
    _validate_input_paths,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _bundle(
    *,
    artifact_root: Path,
    input_parent: Path,
) -> ResolvedV3ConfigBundle:
    config = build_authorization_ready_config(
        source_supervision_content_sha256="1" * 64,
        source_supervision_row_order_sha256="2" * 64,
        source_supervision_producer_seal_sha256="3" * 64,
        source_supervision_recomputation_receipt_sha256="4" * 64,
        authorization_amendment_sha256="5" * 64,
    )
    bindings = []
    for index, (role, artifact_id, kind, relative_member) in enumerate(
        zip(
            DIRECT_INPUT_ROLES,
            DIRECT_INPUT_ARTIFACT_IDS,
            EXPECTED_INPUT_KINDS,
            INPUT_RELATIVE_MEMBERS,
            strict=True,
        )
    ):
        scope = input_parent / f"input-{index}"
        path = scope / relative_member if relative_member else scope
        if kind == "directory":
            path.mkdir(parents=True, exist_ok=False)
        else:
            path.parent.mkdir(parents=True, exist_ok=False)
            path.write_text("{}\n", encoding="utf-8")
        bindings.append(ResolvedDirectInput(role, artifact_id, kind, path))
    return ResolvedV3ConfigBundle(
        config=config,
        source_path=artifact_root / "config.resolved.yaml",
        artifact_root=artifact_root,
        input_bindings=tuple(bindings),
    )


def test_file_backed_input_scope_rejects_scratch_descendant(
    tmp_path: Path,
) -> None:
    bundle = _bundle(
        artifact_root=tmp_path / "output",
        input_parent=tmp_path / "inputs",
    )
    manifest_scope = bundle.input_bindings[4].path.parent

    with pytest.raises(ProtocolError, match="overlaps the scratch root"):
        _validate_input_paths(
            bundle,
            artifact_root=bundle.artifact_root,
            scratch_root=manifest_scope / "future-run-scratch",
        )


def test_direct_inputs_reject_scratch_ancestor(tmp_path: Path) -> None:
    input_parent = tmp_path / "future-scratch"
    bundle = _bundle(
        artifact_root=tmp_path / "output",
        input_parent=input_parent,
    )

    with pytest.raises(ProtocolError, match="overlaps the scratch root"):
        _validate_input_paths(
            bundle,
            artifact_root=bundle.artifact_root,
            scratch_root=input_parent,
        )


def test_direct_inputs_reject_output_descendant(tmp_path: Path) -> None:
    input_parent = tmp_path / "inputs"
    bundle = _bundle(
        artifact_root=input_parent / "input-0/output",
        input_parent=input_parent,
    )

    with pytest.raises(ProtocolError, match="overlaps the output root"):
        _validate_input_paths(
            bundle,
            artifact_root=bundle.artifact_root,
            scratch_root=tmp_path / "scratch",
        )


def test_direct_inputs_reject_output_ancestor(tmp_path: Path) -> None:
    artifact_root = tmp_path / "output"
    bundle = _bundle(
        artifact_root=artifact_root,
        input_parent=artifact_root / "inputs",
    )

    with pytest.raises(ProtocolError, match="overlaps the output root"):
        _validate_input_paths(
            bundle,
            artifact_root=bundle.artifact_root,
            scratch_root=tmp_path / "scratch",
        )
