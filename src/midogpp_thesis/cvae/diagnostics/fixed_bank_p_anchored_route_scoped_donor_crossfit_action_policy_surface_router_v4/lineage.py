"""Path-independent six-input lineage assembly for P-DCAPS v4."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    SixInputBinding,
    binding_from_mappings,
)
from .identity import DIRECT_INPUT_ROLES
from .experiment_contracts import INPUT_ARTIFACT_IDS
from .identity import canonical_hash, require_sha256
from .workspace_manifest import validate_workspace_manifest_header


def build_six_input_binding(
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
) -> SixInputBinding:
    """Bind the six direct artifacts without persisting resolved paths."""

    artifact_ids = tuple(getattr(config, "input_artifact_ids", ()))
    protocol = getattr(config, "protocol", None)
    if (
        artifact_ids != INPUT_ARTIFACT_IDS
        or not isinstance(protocol, Mapping)
        or set(provenance) != set(INPUT_ARTIFACT_IDS)
    ):
        raise ProtocolError("P-DCAPS v4 six-input lineage inventory drifted.")
    protocol_hash = require_sha256(
        protocol.get("protocol_hash"), "v4 protocol contract"
    )
    ids_by_role = dict(zip(DIRECT_INPUT_ROLES, artifact_ids, strict=True))
    hashes_by_role: dict[str, str] = {}
    for role, artifact_id in ids_by_role.items():
        row = provenance[artifact_id]
        semantic = row.get("semantic_identities")
        integrity = row.get("file_integrity")
        if not isinstance(semantic, Mapping) or not isinstance(integrity, Mapping):
            raise ProtocolError("P-DCAPS v4 input provenance row drifted.")
        hashes_by_role[role] = canonical_hash(
            {
                "schema_version": "pdcaps_v4_direct_input_content_identity_v1",
                "artifact_id": artifact_id,
                "semantic_identities": dict(semantic),
                "file_integrity": dict(integrity),
            }
        )
    return binding_from_mappings(
        ids_by_role,
        hashes_by_role,
        protocol_hash=protocol_hash,
    )


def reconstruct_persisted_six_input_binding(
    root: Path,
    config: object,
) -> SixInputBinding:
    """Reconstruct the path-free binding from indexed provenance bytes."""

    payload = read_json(Path(root) / "provenance/input_artifacts.json")
    rows = payload.get("input_artifacts")
    validate_workspace_manifest_header(payload)
    if (
        not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
        or tuple(str(row.get("artifact_id")) for row in rows)
        != tuple(sorted(INPUT_ARTIFACT_IDS))
    ):
        raise ProtocolError("P-DCAPS v4 indexed provenance drifted.")
    by_id = {str(row["artifact_id"]): row for row in rows}
    return build_six_input_binding(config, by_id)


__all__ = (
    "build_six_input_binding",
    "reconstruct_persisted_six_input_binding",
)
