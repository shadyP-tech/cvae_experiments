"""Frozen routing and composition contracts for CVAE expert banks.

Public facade attributes are loaded lazily so importing a nested routing
contract does not also load policy runners, validators, and data caches.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "UniformBV2EqualUnionPolicyConfig": (".config", "UniformBV2EqualUnionPolicyConfig"),
    "load_equal_union_policy_config": (".config", "load_equal_union_policy_config"),
    "EqualUnionPolicyLock": (".contracts", "EqualUnionPolicyLock"),
    "PolicyAssignment": (".contracts", "PolicyAssignment"),
    "PolicyReplicate": (".contracts", "PolicyReplicate"),
    "assignment_rows": (".policy", "assignment_rows"),
    "assignment_table_hash": (".policy", "assignment_table_hash"),
    "build_policy_lock": (".policy", "build_policy_lock"),
    "build_policy_plan": (".policy", "build_policy_plan"),
    "build_policy_plan_payload": (".policy", "build_policy_plan_payload"),
    "read_policy_lock": (".policy", "read_policy_lock"),
    "run_equal_union_policy_lock": (".runner", "run_equal_union_policy_lock"),
    "validate_equal_union_policy_bundle": (
        ".validation",
        "validate_equal_union_policy_bundle",
    ),
    "validate_policy_provenance": (".validation", "validate_policy_provenance"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module fallback
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = tuple(_EXPORTS)
