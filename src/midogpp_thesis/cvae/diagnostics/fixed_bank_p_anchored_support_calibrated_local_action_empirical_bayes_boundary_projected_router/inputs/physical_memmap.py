"""Bind physical cell probabilities to exact read-only memmap case slices."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np

from ..execution.memmaps import open_readonly_memmap
from ..execution.physical_bank import PhysicalBankReceipt
from ..hashing import canonical_hash
from ..physical.endpoint_surface import (
    PhysicalCellSurface,
    _issue_physical_cell_surface,
)
from ..physical.library import PhysicalCellIdentity
from ..protocol import ProtocolError
from ..route_identity import RouteScopeWitness


def load_physical_cell_surface(
    physical_bank: PhysicalBankReceipt,
    *,
    identity: PhysicalCellIdentity,
    route_witness: RouteScopeWitness,
    ordered_sample_keys: Sequence[tuple[str, str, str]],
) -> PhysicalCellSurface:
    """Load one exact case slice and issue the only legal surface contract."""

    if (
        not isinstance(physical_bank, PhysicalBankReceipt)
        or not isinstance(identity, PhysicalCellIdentity)
        or not isinstance(route_witness, RouteScopeWitness)
    ):
        raise ProtocolError("SCALE-BP physical memmap input type drifted.")
    binding = route_witness.evaluation_binding
    keys = tuple(tuple(str(value) for value in key) for key in ordered_sample_keys)
    expected_key_hash = canonical_hash(
        {
            "schema_version": "scale_bp_case_sample_keys_v1",
            "keys": keys,
        }
    )
    inventory = route_witness.identity_inventory.case_inventory
    reference = physical_bank.reference_for(identity)
    if (
        physical_bank.route_identity_inventory_hash
        != route_witness.identity_inventory.inventory_hash
        or physical_bank.dataset_case_inventory_hash != inventory.inventory_hash
        or physical_bank.population_key_hash
        != route_witness.identity_inventory.population_key_hash
        or physical_bank.manifest_hash != inventory.manifest_hash
        or reference.semantic_role != "physical_probabilities"
        or reference.dtype != "float32"
        or reference.shape != (physical_bank.population_row_count,)
        or reference.cache_content_hash != inventory.cache_content_hash
        or reference.row_order_hash != inventory.row_order_hash
        or identity.target_center != route_witness.target_center
        or len(keys) != binding.row_count
        or keys != tuple(sorted(set(keys)))
        or any(key[:2] != (binding.center, binding.case_id) for key in keys)
        or expected_key_hash != binding.sample_key_hash
    ):
        raise ProtocolError("SCALE-BP physical memmap case lineage drifted.")
    start = 0
    found = False
    for candidate in route_witness.identity_inventory.case_bindings:
        if candidate.binding_hash == binding.binding_hash:
            found = True
            break
        start += candidate.row_count
    if not found or start + binding.row_count > physical_bank.population_row_count:
        raise ProtocolError("SCALE-BP physical memmap case offset drifted.")
    values = open_readonly_memmap(
        reference,
        physical_bank=physical_bank,
        physical_identity=identity,
    )
    case_values = np.ascontiguousarray(
        values[start : start + binding.row_count],
        dtype=np.float32,
    )
    slice_hash = hashlib.sha256(case_values.tobytes(order="C")).hexdigest()
    return _issue_physical_cell_surface(
        identity=identity,
        case_id=binding.case_id,
        cache_content_hash=inventory.cache_content_hash,
        row_order_hash=inventory.row_order_hash,
        probabilities=tuple(float(value) for value in case_values),
        physical_bank_receipt_hash=physical_bank.receipt_hash,
        memmap_reference_hash=reference.reference_hash,
        memmap_slice_sha256=slice_hash,
        memmap_row_index_hash=binding.sample_key_hash,
    )


__all__ = ("load_physical_cell_surface",)
