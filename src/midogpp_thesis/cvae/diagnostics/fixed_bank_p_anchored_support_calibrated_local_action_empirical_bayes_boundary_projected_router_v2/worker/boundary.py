"""Strict primitive payload parsing and scientific-firewall validation."""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
from types import MappingProxyType

from ..execution.dtos import OuterCenterTask
from ..hashing import canonical_hash, require_sha256
from ..identity import CENTERS, GovernanceError
from ..label_capabilities import WorkerLabelDelegation, WorkerSupportScope
from ..label_identity import LabelIdentityFrame
from ..physical.contracts import ACTION_IDS
from ..physical_memmaps import MappedPhysicalStore
from ..posterior.local import LOCAL_FOLD_COUNT
from ..routing.admission import AdmissionThresholds
from ..routing.selection import SafetyThresholds
from .contracts import (
    ParsedTaskPayload,
    SCIENTIFIC_SECTION_NAMES,
    ScienceSettings,
    TASK_PAYLOAD_SCHEMA,
)


EXPECTED_PHYSICAL_ROLES = tuple(
    f"physical_probability_center_{center}" for center in CENTERS
)
_TASK_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "artifact_root",
        "physical_index_path",
        "physical_index_hash",
        "label_identity_index_path",
        "label_identity_hash",
        "manifest_path",
        "manifest_sha256",
        "delegation_seed",
        "scientific_contracts",
    }
)
_DELEGATION_SEED_KEYS = frozenset(
    {
        "parent_journal_id",
        "run_identity_hash",
        "task_id",
        "donor_identity_hash",
        "route_scopes",
    }
)


def parse_task_payload(task: OuterCenterTask) -> ParsedTaskPayload:
    """Parse only the sealed primitive boundary accepted by a spawned worker."""

    payload = task.primitive_payload()
    if (
        set(payload) != _TASK_PAYLOAD_KEYS
        or payload.get("schema_version") != TASK_PAYLOAD_SCHEMA
    ):
        raise GovernanceError("SCALE-BP v2 worker science payload schema drifted.")
    artifact_root = _absolute_path(payload.get("artifact_root"), "artifact root")
    physical_index_path = _absolute_path(
        payload.get("physical_index_path"), "physical index path"
    )
    identity_index_path = _absolute_path(
        payload.get("label_identity_index_path"), "label identity index path"
    )
    manifest_path = _absolute_path(payload.get("manifest_path"), "manifest path")
    if (
        artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or physical_index_path.is_symlink()
        or not physical_index_path.is_file()
        or identity_index_path.is_symlink()
        or not identity_index_path.is_file()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise GovernanceError("SCALE-BP v2 worker science path binding drifted.")
    physical_index_hash = require_sha256(
        payload.get("physical_index_hash"), "worker physical index hash"
    )
    identity_hash = require_sha256(
        payload.get("label_identity_hash"), "worker label identity hash"
    )
    manifest_sha256 = require_sha256(
        payload.get("manifest_sha256"), "worker manifest hash"
    )

    seed = _mapping(payload.get("delegation_seed"), "delegation seed")
    if set(seed) != _DELEGATION_SEED_KEYS:
        raise GovernanceError("SCALE-BP v2 worker delegation seed drifted.")
    route_scopes_raw = seed.get("route_scopes")
    if not isinstance(route_scopes_raw, list) or not all(
        isinstance(row, Mapping) for row in route_scopes_raw
    ):
        raise GovernanceError("SCALE-BP v2 worker route scope seed drifted.")
    route_scopes = tuple(
        WorkerSupportScope.from_payload(row) for row in route_scopes_raw
    )
    delegation = WorkerLabelDelegation(
        parent_journal_id=str(seed.get("parent_journal_id", "")),
        run_identity_hash=str(seed.get("run_identity_hash", "")),
        task_id=str(seed.get("task_id", "")),
        task_hash=task.task_hash,
        outer_center=task.target_center,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        donor_identity_hash=str(seed.get("donor_identity_hash", "")),
        route_scopes=route_scopes,
    )

    scientific_raw = _mapping(
        payload.get("scientific_contracts"), "scientific contracts"
    )
    if set(scientific_raw) != set(SCIENTIFIC_SECTION_NAMES):
        raise GovernanceError(
            "SCALE-BP v2 worker scientific section inventory drifted."
        )
    scientific = {
        name: MappingProxyType(
            dict(_mapping(scientific_raw[name], f"scientific section {name}"))
        )
        for name in SCIENTIFIC_SECTION_NAMES
    }
    validate_scientific_firewall(task, scientific)
    settings = science_settings(scientific)
    return ParsedTaskPayload(
        artifact_root,
        physical_index_path,
        physical_index_hash,
        identity_index_path,
        identity_hash,
        manifest_path,
        manifest_sha256,
        delegation,
        MappingProxyType(scientific),
        settings,
    )


def validate_worker_inventory(
    task: OuterCenterTask,
    store: MappedPhysicalStore,
    identity: LabelIdentityFrame,
    parsed: ParsedTaskPayload,
) -> None:
    """Bind physical rows, label identities, route scopes, and task identity."""

    physical_cases = store.case_ids(task.target_center)
    identity_cases = tuple(
        dict.fromkeys(
            row.case_id for row in identity.rows_by_center[task.target_center]
        )
    )
    route_scope_cases = tuple(
        scope.held_case_id for scope in parsed.delegation.route_scopes
    )
    if (
        tuple(sorted(physical_cases)) != task.case_ids
        or tuple(sorted(identity_cases)) != task.case_ids
        or route_scope_cases != task.case_ids
        or parsed.delegation.outer_center != task.target_center
        or parsed.delegation.task_hash != task.task_hash
        or parsed.delegation.manifest_path != str(parsed.manifest_path)
        or parsed.delegation.manifest_sha256 != parsed.manifest_sha256
    ):
        raise GovernanceError("SCALE-BP v2 worker case/delegation inventory drifted.")


def validate_scientific_firewall(
    task: OuterCenterTask,
    scientific: Mapping[str, Mapping[str, object]],
) -> None:
    """Reject any task that weakens the frozen v2 scientific contract."""

    geometry = scientific["action_geometry"]
    support = scientific["support_folds"]
    if (
        geometry.get("anchor") != "P"
        or tuple(geometry.get("direct_actions", ACTION_IDS)) != ACTION_IDS
        or geometry.get("boundary_projection_primary") is not True
        or geometry.get("full_endpoint_primary") is not False
        or int(support.get("fold_count", -1)) != LOCAL_FOLD_COUNT
        or support.get("held_case_excluded") is not True
        or task.support_fold_ids != tuple(range(LOCAL_FOLD_COUNT))
        or scientific["donor_prior"].get("outer_center_excluded") is not True
        or scientific["donor_prior"].get("equal_center_weighting") is not True
        or scientific["local_residual"].get("route_local_only") is not True
        or scientific["local_residual"].get("updates_global_state") is not False
        or scientific["empirical_bayes"].get("weight_bounded_0_1") is not True
        or scientific["uncertainty"].get("descriptive_only") is not True
        or scientific["selection"].get("direct_case_action_selection") is not True
        or scientific["selection"].get("exact_p_fallback") is not True
        or scientific["admission"].get("abort_before_terminal_on_failure")
        is not True
        or scientific["controls"].get("controls_may_authorize_primary") is not False
    ):
        raise GovernanceError("SCALE-BP v2 worker scientific firewall drifted.")


def science_settings(
    scientific: Mapping[str, Mapping[str, object]],
) -> ScienceSettings:
    """Decode immutable numeric settings after firewall validation."""

    donor = scientific["donor_prior"]
    local = scientific["local_residual"]
    uncertainty = scientific["uncertainty"]
    selection = scientific["selection"]
    admission = scientific["admission"]
    donor_ridge = _finite_positive(donor.get("ridge_alpha", 1.0), "donor ridge")
    local_ridge = _finite_positive(local.get("ridge_alpha", 1.0), "local ridge")
    maximum_abs = _finite_positive(
        donor.get("maximum_abs_standardized_feature", 4.0),
        "maximum standardized feature",
    )
    minimum_centers = _integer(
        donor.get("minimum_independent_centers", 6),
        "minimum independent centers",
        minimum=3,
    )
    base_multiplier = _finite_positive(
        uncertainty.get("base_multiplier", 1.2815515655446004),
        "uncertainty base multiplier",
    )
    safety = SafetyThresholds(
        minimum_bacc_lower=_finite(
            selection.get("minimum_bacc_lower", 0.0), "minimum BACC lower"
        ),
        maximum_brier_upper=_finite(
            selection.get("maximum_brier_upper", 0.0), "maximum Brier upper"
        ),
        maximum_log_upper=_finite(
            selection.get("maximum_log_upper", 0.0), "maximum log upper"
        ),
        tie_tolerance=_finite_nonnegative(
            selection.get("tie_tolerance", 1.0e-12), "selection tie tolerance"
        ),
    )
    admission_thresholds = AdmissionThresholds(
        minimum_opportunity_cases=_integer(
            admission.get("minimum_opportunity_cases", 24),
            "minimum opportunity cases",
            minimum=1,
        ),
        minimum_represented_centers=_integer(
            admission.get("minimum_represented_centers", 6),
            "minimum represented centers",
            minimum=1,
        ),
        minimum_within_case_spearman=_finite(
            admission.get("minimum_within_case_spearman", 0.0),
            "minimum within-case Spearman",
        ),
        maximum_normalized_oracle_gap=_finite_nonnegative(
            admission.get("maximum_normalized_oracle_gap", 1.0),
            "maximum normalized oracle gap",
        ),
        maximum_harmful_selected_policy_count=_integer(
            admission.get("maximum_harmful_selected_policy_count", 0),
            "maximum harmful selected count",
            minimum=0,
        ),
    )
    return ScienceSettings(
        donor_ridge,
        local_ridge,
        maximum_abs,
        minimum_centers,
        base_multiplier,
        safety,
        admission_thresholds,
        canonical_hash(
            {name: dict(scientific[name]) for name in SCIENTIFIC_SECTION_NAMES}
        ),
    )


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GovernanceError(f"SCALE-BP v2 {role} is not a mapping.")
    return {str(key): item for key, item in value.items()}


def _absolute_path(value: object, role: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise GovernanceError(f"SCALE-BP v2 {role} is not absolute.")
    return path


def _finite(value: object, role: str) -> float:
    if isinstance(value, bool):
        raise GovernanceError(f"SCALE-BP v2 {role} is not numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GovernanceError(f"SCALE-BP v2 {role} is not numeric.") from exc
    if not math.isfinite(number):
        raise GovernanceError(f"SCALE-BP v2 {role} is nonfinite.")
    return number


def _finite_positive(value: object, role: str) -> float:
    number = _finite(value, role)
    if number <= 0.0:
        raise GovernanceError(f"SCALE-BP v2 {role} is not positive.")
    return number


def _finite_nonnegative(value: object, role: str) -> float:
    number = _finite(value, role)
    if number < 0.0:
        raise GovernanceError(f"SCALE-BP v2 {role} is negative.")
    return number


def _integer(value: object, role: str, *, minimum: int) -> int:
    if isinstance(value, bool):
        raise GovernanceError(f"SCALE-BP v2 {role} is not an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise GovernanceError(f"SCALE-BP v2 {role} is not an integer.") from exc
    if number < minimum or number != value:
        raise GovernanceError(f"SCALE-BP v2 {role} is outside its contract.")
    return number


__all__ = (
    "EXPECTED_PHYSICAL_ROLES",
    "parse_task_payload",
    "science_settings",
    "validate_scientific_firewall",
    "validate_worker_inventory",
)
