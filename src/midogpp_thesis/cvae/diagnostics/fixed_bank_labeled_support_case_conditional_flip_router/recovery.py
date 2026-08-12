"""Exact, capability-typed recovery for one registered diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Mapping

from ...protocol import ProtocolError
from .bundle import REQUIRED_FILES


RUN_STATE_SCHEMA = "fixed_bank_labeled_support_flip_run_state_v1"
RECOVERABLE_FAILURE_CLASSES = frozenset({"MemoryError", "OSError", "RuntimeError"})
_COMPLETED_LOCAL_SOURCE_INVENTORY_ERROR = (
    "Flip-router local source inventory drifted: missing=[], extras=[]."
)
_LEGACY_TERMINAL_HEADER_MEMBER = "tables/terminal_case_confusions.csv"

_BASE = frozenset({
    "config.resolved.yaml", "provenance/input_artifacts.json", "reports/run_state.json",
})
_PRELABEL_PHASES = (
    ("ADMISSION", ("reports/workstation_preflight.json",)),
    ("INITIAL_SURFACES", (
        "manifests/protocol_manifest.json", "manifests/three_role_partition.json",
        "tables/three_role_partitions.csv", "tables/action_library.csv",
        "manifests/action_library.json",
    )),
    ("SOURCE_GENERATION", (
        "arrays/frozen_source_streams.npy", "manifests/frozen_source_stream_index.json",
        "manifests/frozen_source_stream_lock.json",
    )),
    ("PREDICTION_MATERIALIZATION", (
        "arrays/fixed_bank_a1_action_probabilities.npz",
        "manifests/fixed_bank_a1_prediction_index.json",
        "manifests/fixed_bank_a1_prediction_seal.json",
    )),
    ("PRELABEL_SEALING", (
        "tables/seed_probability_rows.csv", "tables/aggregated_probability_rows.csv",
        "tables/case_action_features.csv", "manifests/sealed_probability_surface.json",
        "manifests/prelabel_feature_seal.json",
    )),
    ("FOLD_PLAN_SEALING", ("manifests/fold_plan_seals.json",)),
)
_DONOR_SEQUENCE = (
    "tables/donor_contribution_targets.csv", "tables/model_fits.csv",
    "manifests/donor_model_seals.json", "manifests/permutation_provenance_seal.json",
)
_DONOR_GROUP = frozenset(_DONOR_SEQUENCE)
_DECISION_SEQUENCE = (
    "tables/static_source_selections.csv", "tables/directional_calibrations.csv",
    "manifests/static_selection_seals.json", "manifests/calibration_seals.json",
    "tables/method_decisions.csv", "manifests/all_method_decisions_seal.json",
)
_DECISION_GROUP = frozenset(_DECISION_SEQUENCE)
_TERMINAL_SEQUENCE = (
    "tables/terminal_case_confusions.csv", "tables/terminal_center_metrics.csv",
    "tables/terminal_contrasts.csv", "tables/router_identification_metrics.csv",
    "tables/permutation_metrics.csv", "manifests/sealed_terminal_evaluation.json",
    "reports/label_capability_report.json", "reports/leakage_report.json",
    "reports/publication_decision.json", "reports/runtime_summary.json",
)
_TERMINAL_GROUP = frozenset(_TERMINAL_SEQUENCE)
_TERMINAL_CHECKPOINT = "checkpoints/terminal_evaluation/sealed_result.json"
_FINAL_INDEX = frozenset({"manifests/content_index.json"})
_FINAL_VALIDATION = frozenset({"reports/validation_report.json"})
_PRELABEL_INDEX = {phase: index for index, (phase, _) in enumerate(_PRELABEL_PHASES)}
_PRELABEL_COMPLETE = _BASE | frozenset().union(*(frozenset(group) for _, group in _PRELABEL_PHASES))
_POSTLABEL_PREFIX = _PRELABEL_COMPLETE | _DONOR_GROUP | _DECISION_GROUP
_COMPLETE_FILES = frozenset(REQUIRED_FILES)


@dataclass(frozen=True)
class RecoveryCapability:
    mode: str
    state_phase: str
    resume_phase: str | None
    labels_may_be_reopened_for_validation: bool
    labels_may_be_opened_for_deterministic_policy_construction: bool
    labels_may_update_frozen_policy_contract: bool
    validation_only: bool

    def __post_init__(self) -> None:
        if self.mode not in {
            "PRELABEL_REPLAY",
            "LABEL_AWARE_REPLAY",
            "TERMINAL_FINALIZATION",
            "COMPLETE_REVALIDATION",
        }:
            raise ProtocolError("Flip-router recovery capability mode drifted.")
        if self.labels_may_update_frozen_policy_contract:
            raise ProtocolError("Recovery cannot update the frozen policy contract.")
        if self.mode in {"PRELABEL_REPLAY", "LABEL_AWARE_REPLAY"}:
            if (
                self.validation_only
                or self.labels_may_be_reopened_for_validation
                or not self.labels_may_be_opened_for_deterministic_policy_construction
            ):
                raise ProtocolError("Deterministic replay capability is malformed.")
        elif (
            not self.validation_only
            or not self.labels_may_be_reopened_for_validation
            or self.labels_may_be_opened_for_deterministic_policy_construction
        ):
            raise ProtocolError("Validation-only recovery capability is malformed.")


def recovery_capability(root: Path) -> RecoveryCapability | None:
    """Recognize only exact prelabel replay or validation-only boundaries."""

    state_path = root / "reports/run_state.json"
    if not root.exists() or not state_path.exists():
        return None
    state = _load_state(state_path)
    observed = _inventory(root)
    status = str(state["status"])
    phase = str(state["phase"])
    if status == "COMPLETE":
        _require_inventory(observed, _COMPLETE_FILES, boundary="COMPLETE")
        return RecoveryCapability(
            "COMPLETE_REVALIDATION", phase, None, True, False, False, True
        )
    if phase in _PRELABEL_INDEX:
        _validate_retry_state(state)
        if _is_registered_protocol_retry(state):
            _require_inventory(
                observed,
                _prelabel_prior(phase),
                boundary="COMPLETED_LOCAL_SOURCE_INVENTORY_DEFECT",
            )
        else:
            _validate_prelabel_inventory(observed, phase=phase)
        return RecoveryCapability(
            "PRELABEL_REPLAY", phase, phase, False, True, False, False
        )
    if phase == "DONOR_MODEL_FITTING":
        _validate_retry_state(state)
        _require_phase_prefix(
            observed,
            prior=_PRELABEL_COMPLETE,
            sequence=_DONOR_SEQUENCE,
            boundary=phase,
        )
        return RecoveryCapability(
            "LABEL_AWARE_REPLAY", phase, phase, False, True, False, False
        )
    if phase == "FOLD_DECISION_SEALING":
        _validate_retry_state(state)
        _require_phase_prefix(
            observed,
            prior=_PRELABEL_COMPLETE | _DONOR_GROUP,
            sequence=_DECISION_SEQUENCE,
            boundary=phase,
        )
        return RecoveryCapability(
            "LABEL_AWARE_REPLAY", phase, phase, False, True, False, False
        )
    if phase == "TERMINAL_EVALUATION":
        _validate_retry_state(state)
        exact_without_checkpoint = _POSTLABEL_PREFIX | _TERMINAL_GROUP
        prefixes = {
            _POSTLABEL_PREFIX
            | {_TERMINAL_CHECKPOINT}
            | frozenset(_TERMINAL_SEQUENCE[:length])
            for length in range(len(_TERMINAL_SEQUENCE) + 1)
        }
        if observed != exact_without_checkpoint and observed not in prefixes:
            raise ProtocolError("Flip-router sealed terminal recovery inventory drifted.")
        return RecoveryCapability(
            "TERMINAL_FINALIZATION", phase, "FINALIZATION", True, False, False, True
        )
    if phase == "FINALIZATION":
        legacy_header_retry = _is_registered_terminal_header_retry(
            state, root=root
        )
        if not legacy_header_retry:
            _validate_retry_state(state)
        elif observed != (
            _POSTLABEL_PREFIX | _TERMINAL_GROUP | _FINAL_INDEX
        ):
            raise ProtocolError(
                "Flip-router LEGACY_TERMINAL_HEADER inventory drifted."
            )
        allowed = (
            _POSTLABEL_PREFIX | _TERMINAL_GROUP,
            _POSTLABEL_PREFIX | _TERMINAL_GROUP | _FINAL_INDEX,
            _POSTLABEL_PREFIX | _TERMINAL_GROUP | _FINAL_INDEX | _FINAL_VALIDATION,
        )
        if observed not in allowed:
            raise ProtocolError("Flip-router finalization recovery inventory drifted.")
        return RecoveryCapability(
            "TERMINAL_FINALIZATION", phase, "FINALIZATION", True, False, False, True
        )
    raise ProtocolError(
        "Flip-router recovery refuses post-label replay without an exact terminal seal."
    )


def detect_registered_flip_router_recovery(root: Path) -> bool:
    return recovery_capability(root) is not None


def _validate_prelabel_inventory(observed: frozenset[str], *, phase: str) -> None:
    ordinal = _PRELABEL_INDEX[phase]
    prior = _prelabel_prior(phase)
    sequence = _PRELABEL_PHASES[ordinal][1]
    current = frozenset(sequence)
    checkpoints = frozenset(member for member in observed if _checkpoint_member(member))
    durable = observed - checkpoints
    allowed = {
        prior | frozenset(sequence[:length])
        for length in range(len(sequence) + 1)
    }
    if durable not in allowed:
        raise ProtocolError(
            "Flip-router prelabel recovery is not an exact phase boundary: "
            f"phase={phase}, missing={sorted(prior - durable)}, "
            f"unexpected={sorted(durable - prior - current)}, "
            f"partial_current={sorted((durable & current))}."
        )
    _validate_checkpoint_pairs(checkpoints, phase=phase)


def _prelabel_prior(phase: str) -> frozenset[str]:
    ordinal = _PRELABEL_INDEX[phase]
    return _BASE | frozenset().union(*(
        frozenset(group) for _, group in _PRELABEL_PHASES[:ordinal]
    ))


def _require_phase_prefix(
    observed: frozenset[str],
    *,
    prior: frozenset[str],
    sequence: tuple[str, ...],
    boundary: str,
) -> None:
    if any(_checkpoint_member(member) for member in observed):
        raise ProtocolError(
            "Flip-router label-aware recovery found a stale compute checkpoint."
        )
    allowed = {
        prior | frozenset(sequence[:length])
        for length in range(len(sequence) + 1)
    }
    if observed not in allowed:
        raise ProtocolError(
            f"Flip-router {boundary} recovery inventory is not an exact phase prefix."
        )


def _validate_retry_state(state: Mapping[str, object]) -> None:
    status = state["status"]
    if status == "RUNNING":
        if "error" in state or "error_class" in state:
            raise ProtocolError("RUNNING flip-router recovery state carries an error.")
        return
    if _is_registered_protocol_retry(state):
        return
    if (
        status != "FAILED"
        or state.get("error_class") not in RECOVERABLE_FAILURE_CLASSES
        or not isinstance(state.get("error"), str)
        or not str(state["error"])
    ):
        raise ProtocolError("Flip-router FAILED state is not a registered retry class.")


def _is_registered_protocol_retry(state: Mapping[str, object]) -> bool:
    """Admit only the one historical, pre-label source-inventory defect."""

    expected = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "FAILED",
        "phase": "SOURCE_GENERATION",
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
        "error": _COMPLETED_LOCAL_SOURCE_INVENTORY_ERROR,
        "error_class": "ProtocolError",
    }
    return dict(state) == expected


def _is_registered_terminal_header_retry(
    state: Mapping[str, object], *, root: Path
) -> bool:
    """Admit the one post-terminal, serialization-only header failure."""

    expected = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "FAILED",
        "phase": "FINALIZATION",
        "terminal_consumed_test_diagnostic_only": True,
        "automatic_resume_requires_hash_validation": True,
        "error": (
            "Flip-router persisted table header drifted: "
            f"{root / _LEGACY_TERMINAL_HEADER_MEMBER}."
        ),
        "error_class": "ProtocolError",
    }
    return dict(state) == expected


def _load_state(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("Flip-router recovery run state is unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Flip-router recovery run state is unreadable.") from exc
    if not isinstance(value, Mapping):
        raise ProtocolError("Flip-router recovery run state is malformed.")
    status = value.get("status")
    if (
        value.get("schema_version") != RUN_STATE_SCHEMA
        or status not in {"RUNNING", "FAILED", "COMPLETE"}
        or value.get("terminal_consumed_test_diagnostic_only") is not True
        or value.get("automatic_resume_requires_hash_validation") is not True
        or not isinstance(value.get("phase"), str)
        or (status == "COMPLETE" and (
            value.get("phase") != "COMPLETE" or "error" in value or "error_class" in value
        ))
    ):
        raise ProtocolError("Flip-router recovery run state drifted.")
    return value


def _inventory(root: Path) -> frozenset[str]:
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("Flip-router recovery root is unsafe.")
    members: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            if (base / name).is_symlink():
                raise ProtocolError("Flip-router recovery inventory contains a symlink.")
        for name in files:
            relative = (base / name).relative_to(root).as_posix()
            if relative != ".run.lock":
                match = re.fullmatch(
                    r"(?P<member>.+)\.[1-9][0-9]*\.tmp", relative
                )
                if match and _owned_atomic_member(match.group("member")):
                    # Exact package-owned atomic remnants are removed by the
                    # runner only after this recovery boundary is validated.
                    continue
                members.add(relative)
    return frozenset(members)


def _owned_atomic_member(member: str) -> bool:
    return (
        member in REQUIRED_FILES
        or member == _TERMINAL_CHECKPOINT
        or _checkpoint_member(member)
    )


def _checkpoint_member(member: str) -> bool:
    centers = r"(?:0|1|2|3|5|6|7|8|9)"
    seeds = r"(?:17|42|101)"
    return any(re.fullmatch(pattern, member) is not None for pattern in (
        rf"checkpoints/frozen_source_streams/source_{centers}_train_{seeds}\.(?:json|npy)",
        r"checkpoints/fixed_bank_a1_action_predictions/(?:target_scratch\.json|target_embeddings\.npy)",
        rf"checkpoints/fixed_bank_a1_action_predictions/tasks/target_{centers}_train_{seeds}_generation_{seeds}\.(?:json|npz)",
    ))


def _validate_checkpoint_pairs(members: frozenset[str], *, phase: str) -> None:
    if members and phase not in {"SOURCE_GENERATION", "PREDICTION_MATERIALIZATION"}:
        raise ProtocolError("Flip-router checkpoints survived outside their owning phase.")
    scratch_pair = frozenset({
        "checkpoints/fixed_bank_a1_action_predictions/target_scratch.json",
        "checkpoints/fixed_bank_a1_action_predictions/target_embeddings.npy",
    })
    if members & scratch_pair and members & scratch_pair != scratch_pair:
        raise ProtocolError("Flip-router recovery found a partial target scratch pair.")
    for member in members - scratch_pair:
        pair = None
        if member.endswith(".json"):
            pair = member[:-5] + (".npy" if "frozen_source_streams" in member else ".npz")
        elif member.endswith(".npy") or member.endswith(".npz"):
            pair = member.rsplit(".", 1)[0] + ".json"
        if pair is None or pair not in members:
            raise ProtocolError("Flip-router recovery found a partial checkpoint pair.")
    source = {member for member in members if member.startswith("checkpoints/frozen_source_streams/")}
    prediction = set(members) - source
    if source and phase != "SOURCE_GENERATION":
        raise ProtocolError("Flip-router source checkpoints survived their phase.")
    if prediction and phase != "PREDICTION_MATERIALIZATION":
        raise ProtocolError("Flip-router prediction checkpoints survived their phase.")


def _require_inventory(observed: frozenset[str], expected: frozenset[str], *, boundary: str) -> None:
    if observed != expected:
        raise ProtocolError(
            f"Flip-router {boundary} inventory drifted: "
            f"missing={sorted(expected - observed)}, extras={sorted(observed - expected)}."
        )


__all__ = (
    "RECOVERABLE_FAILURE_CLASSES", "RUN_STATE_SCHEMA", "RecoveryCapability",
    "detect_registered_flip_router_recovery", "recovery_capability",
)
