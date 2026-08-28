"""Compact, reconstructive receipt for OE-PPUR v3 direct input number three."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.action_compiler import (
    canonical_compiler_receipt,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.hashing import (
    canonical_hash,
    require_sha256,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.identity import (
    SOURCE_SUPERVISION_ARTIFACT_ID,
    SOURCE_SUPERVISION_REQUIRED_MEMBERS,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.hashing import (
    read_json,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.contracts import (
    SourceTrainingSurface,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_bundle.parsing import (
    parse_source_training_bundle,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_production.held_actions import (
    canonical_held_action_library,
)
from ..fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.source_seal import (
    build_source_seal,
)
from ...protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class SourceArtifactReceipt:
    content_sha256: str
    row_order_sha256: str
    producer_source_seal_sha256: str
    compiler_recomputation_receipt_sha256: str
    probability_matrix_sha256: str
    source_outcome_sha256: str
    surface_sha256: str
    exact_member_hashes: tuple[tuple[str, str], ...]
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "content_sha256",
            "row_order_sha256",
            "producer_source_seal_sha256",
            "compiler_recomputation_receipt_sha256",
            "probability_matrix_sha256",
            "source_outcome_sha256",
            "surface_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name.replace("_", " ")),
            )
        members = tuple(
            (str(member), require_sha256(digest, f"source member {member}"))
            for member, digest in self.exact_member_hashes
        )
        if tuple(member for member, _digest in members) != tuple(
            SOURCE_SUPERVISION_REQUIRED_MEMBERS
        ):
            raise ProtocolError("OE-PPUR v3 compact source receipt inventory drifted.")
        object.__setattr__(self, "exact_member_hashes", members)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_source_artifact_receipt_v1",
            "artifact_id": SOURCE_SUPERVISION_ARTIFACT_ID,
            "content_sha256": self.content_sha256,
            "row_order_sha256": self.row_order_sha256,
            "producer_source_seal_sha256": self.producer_source_seal_sha256,
            "compiler_recomputation_receipt_sha256": (
                self.compiler_recomputation_receipt_sha256
            ),
            "probability_matrix_sha256": self.probability_matrix_sha256,
            "source_outcome_sha256": self.source_outcome_sha256,
            "surface_sha256": self.surface_sha256,
            "exact_member_hashes": [
                {"member": member, "sha256": digest}
                for member, digest in self.exact_member_hashes
            ],
            "source_rows_only": True,
            "target_rows_present": False,
            "target_labels_used": False,
            "execution_authorized": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def validate_materialized_source_artifact(
    root: str | Path,
) -> SourceArtifactReceipt:
    """Reopen all six members and reconstruct compiler lineage from bytes."""

    surface = load_materialized_source_surface(root)
    receipt = surface.receipt
    return SourceArtifactReceipt(
        content_sha256=receipt.receipt_hash,
        row_order_sha256=receipt.row_order_sha256,
        producer_source_seal_sha256=receipt.contract.producer_source_seal_sha256,
        compiler_recomputation_receipt_sha256=(
            receipt.compiler_recomputation_receipt_sha256
        ),
        probability_matrix_sha256=receipt.probability_matrix_sha256,
        source_outcome_sha256=receipt.source_outcome_sha256,
        surface_sha256=surface.surface_hash,
        exact_member_hashes=receipt.member_hashes,
    )


def load_materialized_source_surface(root: str | Path) -> SourceTrainingSurface:
    """Return the typed, live-seal-bound source surface for admission."""

    path = _safe_exact_source_root(root)
    manifest = read_json(path / "manifests/source_training_surface.json")
    recomputation = require_sha256(
        manifest.get("producer_compiler_recomputation_receipt_sha256"),
        "source compiler recomputation candidate",
    )
    live_seal = build_source_seal()
    if manifest.get("producer_source_seal_sha256") != live_seal.combined_source_sha256:
        raise ProtocolError("OE-PPUR v3 materialized source seal is not live.")
    compiler = canonical_compiler_receipt()
    library = canonical_held_action_library()
    return parse_source_training_bundle(
        path,
        compiler=compiler,
        expected_producer_source_seal_sha256=live_seal.combined_source_sha256,
        expected_compiler_recomputation_receipt_sha256=recomputation,
        expected_held_action_library_sha256=library.library_hash,
        expected_held_mass_policy_receipt_sha256=library.mass_policy.receipt_hash,
    )


def _safe_exact_source_root(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path == Path(path.anchor):
        raise ProtocolError("OE-PPUR v3 materialized source root is unsafe.")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 materialized source root is absent.") from exc
    if root != path or root.is_symlink() or not root.is_dir():
        raise ProtocolError("OE-PPUR v3 materialized source root drifted.")
    observed: dict[str, Path] = {}
    for member in root.rglob("*"):
        if member.is_symlink():
            raise ProtocolError("OE-PPUR v3 materialized source tree has a symlink.")
        if member.is_file():
            observed[member.relative_to(root).as_posix()] = member
    if tuple(sorted(observed)) != tuple(sorted(SOURCE_SUPERVISION_REQUIRED_MEMBERS)):
        raise ProtocolError("OE-PPUR v3 materialized source inventory drifted.")
    return root


__all__ = (
    "SourceArtifactReceipt",
    "load_materialized_source_surface",
    "validate_materialized_source_artifact",
)
