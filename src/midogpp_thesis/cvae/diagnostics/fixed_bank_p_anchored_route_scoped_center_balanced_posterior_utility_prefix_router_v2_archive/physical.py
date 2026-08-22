"""Internal hash-graph validation for the persisted label-free surface."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from .contracts import CONFIG_CONTRACT_HASH
from .hashing import canonical_hash, short_hash, without


def validate_label_free_physical_graph(root: Path) -> dict[str, str]:
    source_index = read_json(root / "manifests/frozen_source_stream_index.json")
    source_lock = read_json(root / "manifests/frozen_source_stream_lock.json")
    prediction_index = read_json(
        root / "manifests/fixed_bank_a1_prediction_index.json"
    )
    prediction = read_json(root / "manifests/fixed_bank_a1_prediction_seal.json")
    probability_index = read_json(root / "tables/exact_nine_probability_index.json")
    physical = read_json(root / "manifests/physical_surface_seal.json")
    rows = probability_index.get("rows")
    if (
        source_index.get("source_stream_index_hash")
        != short_hash(without(source_index, "source_stream_index_hash"))
        or source_index.get("stream_count") != 81
        or source_index.get("labels_consumed") is not False
        or source_lock.get("source_stream_lock_hash")
        != short_hash(without(source_lock, "source_stream_lock_hash"))
        or source_lock.get("config_contract_hash") != CONFIG_CONTRACT_HASH
        or source_lock.get("source_array_sha256")
        != sha256_file(root / "arrays/frozen_source_streams.npy")
        or source_lock.get("source_stream_index_sha256")
        != sha256_file(root / "manifests/frozen_source_stream_index.json")
        or source_lock.get("source_stream_index_hash")
        != source_index.get("source_stream_index_hash")
        or source_lock.get("stream_count") != 81
        or source_lock.get("labels_consumed") is not False
        or source_lock.get("source_experts_updated") is not False
        or prediction.get("global_prediction_seal_hash")
        != short_hash(without(prediction, "global_prediction_seal_hash"))
        or prediction.get("config_contract_hash") != CONFIG_CONTRACT_HASH
        or prediction.get("source_stream_lock_hash")
        != source_lock.get("source_stream_lock_hash")
        or prediction.get("arrays_sha256")
        != sha256_file(root / "arrays/fixed_bank_a1_action_probabilities.npz")
        or prediction.get("index_sha256")
        != sha256_file(root / "manifests/fixed_bank_a1_prediction_index.json")
        or prediction.get("cell_count") != 810
        or prediction.get("task_count") != 81
        or prediction.get("labels_opened") is not False
        or prediction.get("target_expert_used") is not False
        or any(
            prediction_index.get(key) != prediction.get(key)
            for key in (
                "config_contract_hash",
                "partition_hash",
                "source_stream_lock_hash",
                "action_library_hash",
                "target_cache_binding_hash",
                "store_hash",
            )
        )
        or probability_index.get("schema_version")
        != "fixed_bank_cbpupr_exact_nine_probability_index_v1"
        or not isinstance(rows, list)
        or probability_index.get("row_count") != 90
        or len(rows) != 90
        or physical.get("schema_version")
        != "fixed_bank_cbpupr_physical_surface_seal_v1"
        or physical.get("source_stream_lock_hash")
        != source_lock.get("source_stream_lock_hash")
        or physical.get("global_prediction_seal_hash")
        != prediction.get("global_prediction_seal_hash")
        or physical.get("probability_store_hash") != prediction.get("store_hash")
        or physical.get("probability_index_hash") != canonical_hash(rows)
        or physical.get("target_probability_cell_count") != 810
        or physical.get("labels_used") is not False
        or physical.get("physical_surface_seal_hash")
        != canonical_hash(without(physical, "physical_surface_seal_hash"))
    ):
        raise ProtocolError("CBPUPR v2 preterminal physical hash graph drifted.")
    return {
        "physical_surface_hash": str(physical["surface_hash"]),
        "physical_surface_seal_hash": str(physical["physical_surface_seal_hash"]),
        "global_prediction_seal_hash": str(
            prediction["global_prediction_seal_hash"]
        ),
        "source_stream_lock_hash": str(source_lock["source_stream_lock_hash"]),
    }


__all__ = ("validate_label_free_physical_graph",)
