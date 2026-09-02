"""Independent durable projection of the pre-label HARP v10 effective menu."""

from __future__ import annotations

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from .compatibility_adapter import compatibility_state_from_artifact
from .contracts import ArtifactValue
from .production_validation import require_sha256


_ARRAY_NAMES = (
    "effective_action_features",
    "effective_menu_baselines",
    "effective_menu_baseline_offsets",
    "effective_action_probabilities",
    "effective_action_probability_offsets",
)


def build_effective_menu_artifact(compatibility: ArtifactValue) -> ArtifactValue:
    """Split the shared source/target menu seal into its own auditable store."""

    state = compatibility_state_from_artifact(compatibility)
    menus = state.effective_menus
    raw_menus = compatibility.manifest.get("effective_menus")
    raw_actions = compatibility.manifest.get("effective_actions")
    if not isinstance(raw_menus, list) or not isinstance(raw_actions, list):
        raise ProtocolError("HARP v10 compatibility lacks its effective-menu projection.")
    arrays = {}
    for name in _ARRAY_NAMES:
        if name not in compatibility.arrays:
            raise ProtocolError("HARP v10 effective-menu durable array is absent.")
        arrays[name] = compatibility.arrays[name]
    body = {
        "schema_version": "midogpp_harp_v10_effective_menu_store_v1",
        "compatibility_hash": require_sha256(
            compatibility.manifest.get("compatibility_hash"),
            role="compatibility hash",
        ),
        "effective_menus": raw_menus,
        "effective_actions": raw_actions,
        "effective_menu_count": len(menus),
        "effective_action_count": sum(len(menu.actions) for menu in menus),
        "active_case_count": sum(bool(menu.actions) for menu in menus),
        "source_context_count": len(
            {(row.outer_target_id, row.query_center_id) for row in menus if row.query_center_id != row.outer_target_id}
        ),
        "target_context_count": len(
            {(row.outer_target_id, row.query_center_id) for row in menus if row.query_center_id == row.outer_target_id}
        ),
        "filter_inputs": "LABEL_FREE_ONLY",
        "shared_source_target_implementation": True,
        "directions_retained": ["D01", "D10"],
        "all_margins_excluded": True,
        "exact_b_noops_removed": True,
        "exact_byte_duplicates_deduplicated_with_aliases": True,
        "development_labels_used": False,
        "evaluation_labels_used": False,
    }
    return ArtifactValue(
        state=menus,
        manifest={**body, "effective_menu_hash": canonical_hash(body)},
        arrays=arrays,
    )


__all__ = ("build_effective_menu_artifact",)
