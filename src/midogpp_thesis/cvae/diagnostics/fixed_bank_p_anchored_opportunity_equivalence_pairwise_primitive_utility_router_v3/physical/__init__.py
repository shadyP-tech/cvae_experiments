"""Successor-owned physical probability contracts for OE-PPUR v3."""

from .actions import (
    B_ACTION_ID,
    PhysicalActionSpec,
    U_ACTION_ID,
    a1_action_id,
    action_library_by_target,
    actions_for_target,
    candidate_sources,
)
from .cpu_pool import execute_prediction_tasks_one_thread
from .compiled_matrix import (
    CompiledProbabilityMatrix,
    assemble_compiled_probability_matrix,
)
from .cache_loader import load_label_free_test_frame
from .frame import LabelFreeTestFrame, TestRowIdentity
from .prediction_runtime import (
    MaterializedPhysicalInputs,
    PHASE_ORDER,
    materialize_physical_inputs,
    physical_partition_hash,
)
from .runtime_config import PhysicalRuntimeConfig, physical_runtime_payload
from .surfaces import build_final_compiled_surface
from .upstream import ValidatedUpstreamInputs, load_validated_upstream_inputs

__all__ = (
    "B_ACTION_ID",
    "PhysicalActionSpec",
    "U_ACTION_ID",
    "a1_action_id",
    "action_library_by_target",
    "actions_for_target",
    "candidate_sources",
    "execute_prediction_tasks_one_thread",
    "CompiledProbabilityMatrix",
    "assemble_compiled_probability_matrix",
    "LabelFreeTestFrame",
    "load_label_free_test_frame",
    "MaterializedPhysicalInputs",
    "PHASE_ORDER",
    "materialize_physical_inputs",
    "PhysicalRuntimeConfig",
    "physical_runtime_payload",
    "physical_partition_hash",
    "build_final_compiled_surface",
    "ValidatedUpstreamInputs",
    "load_validated_upstream_inputs",
    "TestRowIdentity",
)
