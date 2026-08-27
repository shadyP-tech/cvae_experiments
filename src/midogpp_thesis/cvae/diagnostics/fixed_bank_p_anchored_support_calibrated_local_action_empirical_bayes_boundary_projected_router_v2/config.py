"""Strict standalone configuration for executable SCALE-BP v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .experiment_contracts import (
    LEDGER_AMENDMENT_FILENAME,
    validate_exact_input_fence,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    CANONICAL_SCRATCH_ROOT,
    CLAIM_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXECUTION_REVISION,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_DIRECT_INPUT_COUNT,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    GovernanceError,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .protocol import (
    frozen_protocol_payload,
    terminal_claim_firewall_payload,
    validate_protocol_payload,
    validate_terminal_claim_firewall,
)
from .scientific_contracts import (
    SCIENTIFIC_SECTION_NAMES,
    canonical_scientific_contracts_payload,
)
from .workstation import canonical_workstation_payload


CONFIG_SCHEMA = "scale_bp_v2_executable_config_v1"
SOURCE_BINDING_SCHEMA = "scale_bp_v2_source_binding_v1"
RESUME_POLICY = "SINGLE_USE_NO_CROSS_RUN_RECOVERY_ATOMIC_PHASE_CHUNKS"
CONFIG_TOP_LEVEL = frozenset(
    {
        "experiment",
        "inputs",
        "source",
        "protocol",
        *SCIENTIFIC_SECTION_NAMES,
        "classifier",
        "runtime",
        "claim_boundary",
    }
)
INPUT_LOCATION_KEYS = (
    "expert_bank_root",
    "generation_lock_root",
    "test_cache_root",
    "test_manifest_path",
    "test_consumption_ledger_path",
    "ledger_amendment_path",
)


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    family: str
    C: float
    penalty: str
    solver: str
    max_iter: int
    class_weight: str | None
    random_state: int
    l1_ratio: float | None
    threshold_policy: str
    scaler_fit: str

    def to_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "l1_ratio": self.l1_ratio,
            "threshold_policy": self.threshold_policy,
            "scaler_fit": self.scaler_fit,
        }


CANONICAL_CLASSIFIER = ClassifierConfig(
    family="sklearn_logistic_regression",
    C=0.01,
    penalty="l2",
    solver="lbfgs",
    max_iter=3_000,
    class_weight=None,
    random_state=23,
    l1_ratio=None,
    threshold_policy="predict",
    scaler_fit="synthetic_train_only",
)


@dataclass(frozen=True, slots=True)
class ScaleBPV2Config:
    source_path: Path
    artifact_root: Path
    scratch_root: Path
    expert_bank_root: Path
    generation_lock_root: Path
    test_cache_root: Path
    test_manifest_path: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    protocol: Mapping[str, object]
    scientific_contracts: Mapping[str, Mapping[str, object]]
    classifier: ClassifierConfig
    runtime: Mapping[str, object]
    claim_boundary: Mapping[str, object]
    contract_hash: str
    expected_authorization_amendment_sha256: str
    expected_source_snapshot_manifest_sha256: str
    expected_source_snapshot_tree_sha256: str
    expected_source_snapshot_member_count: int

    experiment_id: str = EXPERIMENT_ID
    output_artifact_id: str = OUTPUT_ARTIFACT_ID
    direct_input_artifact_ids: tuple[str, ...] = DIRECT_INPUT_ARTIFACT_IDS
    input_artifact_ids: tuple[str, ...] = DIRECT_INPUT_ARTIFACT_IDS
    execution_authorized: bool = True
    consumed_test_reuse_authorized: bool = True

    @property
    def config_hash(self) -> str:
        return self.contract_hash

    @property
    def parent_ledger_path(self) -> Path:
        return self.test_consumption_ledger_path

    @property
    def authorization_amendment_path(self) -> Path:
        return self.ledger_amendment_path

    @property
    def expected_ledger_amendment_sha256(self) -> str:
        return self.expected_authorization_amendment_sha256

    @property
    def source_snapshot_manifest_sha256(self) -> str:
        return self.expected_source_snapshot_manifest_sha256

    @property
    def source_snapshot_tree_sha256(self) -> str:
        return self.expected_source_snapshot_tree_sha256

    @property
    def source_snapshot_member_count(self) -> int:
        return self.expected_source_snapshot_member_count

    @property
    def expected_bank_lock_hash(self) -> str:
        return EXPECTED_BANK_LOCK_HASH

    @property
    def expected_generation_lock_hash(self) -> str:
        return EXPECTED_GENERATION_LOCK_HASH

    @property
    def expected_test_cache_content_hash(self) -> str:
        return EXPECTED_TEST_CACHE_CONTENT_HASH

    @property
    def expected_test_cache_row_order_hash(self) -> str:
        return EXPECTED_TEST_CACHE_ROW_ORDER_HASH

    @property
    def expected_manifest_sha256(self) -> str:
        return EXPECTED_TEST_MANIFEST_SHA256

    @property
    def expected_parent_ledger_sha256(self) -> str:
        return EXPECTED_PARENT_LEDGER_SHA256

    def scientific(self, section: str) -> Mapping[str, object]:
        try:
            return self.scientific_contracts[section]
        except KeyError as exc:
            raise GovernanceError(
                f"SCALE-BP v2 scientific section is absent: {section}."
            ) from exc

    @property
    def action_geometry(self) -> Mapping[str, object]:
        return self.scientific("action_geometry")

    @property
    def support_folds(self) -> Mapping[str, object]:
        return self.scientific("support_folds")

    @property
    def influence(self) -> Mapping[str, object]:
        return self.scientific("influence")

    @property
    def donor_prior(self) -> Mapping[str, object]:
        return self.scientific("donor_prior")

    @property
    def local_residual(self) -> Mapping[str, object]:
        return self.scientific("local_residual")

    @property
    def empirical_bayes(self) -> Mapping[str, object]:
        return self.scientific("empirical_bayes")

    @property
    def uncertainty(self) -> Mapping[str, object]:
        return self.scientific("uncertainty")

    @property
    def selection(self) -> Mapping[str, object]:
        return self.scientific("selection")

    @property
    def admission(self) -> Mapping[str, object]:
        return self.scientific("admission")

    @property
    def controls(self) -> Mapping[str, object]:
        return self.scientific("controls")


def fixed_experiment_payload() -> dict[str, object]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "id": EXPERIMENT_ID,
        "name": EXPERIMENT_NAME,
        "claim_scope": CLAIM_SCOPE,
        "publication_status": PUBLICATION_STATUS,
        "execution_revision": EXECUTION_REVISION,
        "execution_authorized": True,
        "implementation_request_alone_authorizes_execution": False,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "single_use_execution_identity": True,
        "consumed_test_reuse_authorized": True,
    }


def fixed_inputs_payload() -> dict[str, object]:
    return {
        "expert_bank_artifact_id": EXPERT_BANK_ARTIFACT_ID,
        "generation_lock_artifact_id": GENERATION_LOCK_ARTIFACT_ID,
        "test_cache_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "test_manifest_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "test_consumption_ledger_artifact_id": (
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID
        ),
        "ledger_amendment_artifact_id": AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "direct_input_count": EXPECTED_DIRECT_INPUT_COUNT,
        "all_direct_input_artifact_ids_unique": True,
        "expected_bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "expected_generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expected_test_cache_semantic_id": EXPECTED_TEST_CACHE_SEMANTIC_ID,
        "expected_test_cache_representation_id": (
            EXPECTED_TEST_CACHE_REPRESENTATION_ID
        ),
        "expected_test_cache_content_hash": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "expected_test_cache_row_order_hash": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "expected_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "expected_test_consumption_ledger_sha256": (
            EXPECTED_PARENT_LEDGER_SHA256
        ),
        "ledger_amendment_registered_consumer_experiment_id": EXPERIMENT_ID,
        "ledger_amendment_execution_authorized": True,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_scratch_or_checkpoints_used": False,
        "historical_manifests_used": False,
    }


def load_config(path: str | Path) -> ScaleBPV2Config:
    """Load a sealed v2 config and resolve every concrete filesystem path."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise GovernanceError("Cannot read SCALE-BP v2 config.") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(CONFIG_TOP_LEVEL):
        raise GovernanceError("SCALE-BP v2 top-level config drifted.")
    _reject_pending(raw)

    experiment = _section(raw, "experiment")
    fixed_experiment = fixed_experiment_payload()
    if set(experiment) != {*fixed_experiment, "artifact_root"} or any(
        experiment.get(key) != value for key, value in fixed_experiment.items()
    ):
        raise GovernanceError("SCALE-BP v2 execution identity drifted.")
    artifact_text = _text(experiment["artifact_root"], "artifact root")
    artifact_root = _resolve_output_root(source_path, artifact_text)

    inputs = _section(raw, "inputs")
    fixed_inputs = fixed_inputs_payload()
    if set(inputs) != {
        *fixed_inputs,
        *INPUT_LOCATION_KEYS,
        "expected_ledger_amendment_sha256",
    } or any(inputs.get(key) != value for key, value in fixed_inputs.items()):
        raise GovernanceError("SCALE-BP v2 exact-six input contract drifted.")
    amendment_hash = require_sha256(
        inputs["expected_ledger_amendment_sha256"], "authorization amendment hash"
    )
    _validate_input_locations(inputs)
    locations = {
        key: _resolve_location(source_path.parent, _text(inputs[key], key))
        for key in INPUT_LOCATION_KEYS
    }

    source = _section(raw, "source")
    if set(source) != {
        "schema_version",
        "source_snapshot_manifest_sha256",
        "source_snapshot_tree_sha256",
        "source_snapshot_member_count",
        "closed_world_source_fence_required",
        "predecessor_source_snapshot_used",
    } or source.get("schema_version") != SOURCE_BINDING_SCHEMA or source.get(
        "closed_world_source_fence_required"
    ) is not True or source.get("predecessor_source_snapshot_used") is not False:
        raise GovernanceError("SCALE-BP v2 source binding drifted.")
    source_manifest_hash = require_sha256(
        source["source_snapshot_manifest_sha256"], "source snapshot manifest hash"
    )
    source_tree_hash = require_sha256(
        source["source_snapshot_tree_sha256"], "source snapshot tree hash"
    )
    source_member_count = source["source_snapshot_member_count"]
    if (
        isinstance(source_member_count, bool)
        or not isinstance(source_member_count, int)
        or source_member_count <= 0
    ):
        raise GovernanceError("SCALE-BP v2 source member count drifted.")

    protocol = _section(raw, "protocol")
    validate_protocol_payload(protocol)
    claims = _section(raw, "claim_boundary")
    validate_terminal_claim_firewall(claims)
    classifier = _parse_classifier(_section(raw, "classifier"))
    if classifier != CANONICAL_CLASSIFIER:
        raise GovernanceError("SCALE-BP v2 classifier contract drifted.")

    scientific = {
        name: _section(raw, name) for name in SCIENTIFIC_SECTION_NAMES
    }
    _validate_scientific_contracts(scientific)

    runtime = _section(raw, "runtime")
    scratch_text = _text(runtime.get("scratch_root"), "scratch root")
    scratch_root = _resolve_absolute_scratch(scratch_text)
    _validate_runtime(runtime, scratch_root)

    path_independent = {
        "experiment": fixed_experiment,
        "inputs": {
            **fixed_inputs,
            "expected_ledger_amendment_sha256": amendment_hash,
        },
        "source": source,
        "protocol": protocol,
        **scientific,
        "classifier": classifier.to_payload(),
        "runtime": canonical_workstation_payload(),
        "claim_boundary": claims,
    }
    return ScaleBPV2Config(
        source_path=source_path,
        artifact_root=artifact_root,
        scratch_root=scratch_root,
        expert_bank_root=locations["expert_bank_root"],
        generation_lock_root=locations["generation_lock_root"],
        test_cache_root=locations["test_cache_root"],
        test_manifest_path=locations["test_manifest_path"],
        test_consumption_ledger_path=locations[
            "test_consumption_ledger_path"
        ],
        ledger_amendment_path=locations["ledger_amendment_path"],
        protocol=protocol,
        scientific_contracts=scientific,
        classifier=classifier,
        runtime=runtime,
        claim_boundary=claims,
        contract_hash=canonical_hash(path_independent),
        expected_authorization_amendment_sha256=amendment_hash,
        expected_source_snapshot_manifest_sha256=source_manifest_hash,
        expected_source_snapshot_tree_sha256=source_tree_hash,
        expected_source_snapshot_member_count=int(source_member_count),
    )


def _validate_scientific_contracts(
    contracts: Mapping[str, Mapping[str, object]],
) -> None:
    if {
        name: dict(values) for name, values in contracts.items()
    } != canonical_scientific_contracts_payload():
        raise GovernanceError("SCALE-BP v2 scientific firewall drifted.")


def _validate_runtime(runtime: Mapping[str, object], scratch_root: Path) -> None:
    plan = canonical_workstation_payload()
    extra = {"scratch_root", "scratch_preference", "resume_policy"}
    if set(runtime) != {*plan, *extra} or any(
        runtime.get(key) != value for key, value in plan.items()
    ):
        raise GovernanceError("SCALE-BP v2 workstation runtime drifted.")
    if (
        runtime.get("scratch_root") != str(scratch_root)
        or runtime.get("scratch_preference")
        != [str(scratch_root), "artifact_parent"]
        or runtime.get("resume_policy") != RESUME_POLICY
    ):
        raise GovernanceError("SCALE-BP v2 scratch/recovery contract drifted.")


def _parse_classifier(raw: Mapping[str, object]) -> ClassifierConfig:
    try:
        parsed = ClassifierConfig(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(
                None if raw["class_weight"] is None else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(None if raw["l1_ratio"] is None else float(raw["l1_ratio"])),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GovernanceError("SCALE-BP v2 classifier is malformed.") from exc
    if set(raw) != set(parsed.to_payload()):
        raise GovernanceError("SCALE-BP v2 classifier fields drifted.")
    return parsed


def _validate_input_locations(inputs: Mapping[str, object]) -> None:
    roles = (
        ("expert_bank_root", EXPERT_BANK_ARTIFACT_ID, ""),
        ("generation_lock_root", GENERATION_LOCK_ARTIFACT_ID, ""),
        ("test_cache_root", TEST_CACHE_ARTIFACT_ID, ""),
        ("test_manifest_path", TEST_MANIFEST_ARTIFACT_ID, "manifest.csv"),
        (
            "test_consumption_ledger_path",
            TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
            "reports/test_consumption_ledger.json",
        ),
        (
            "ledger_amendment_path",
            AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
            LEDGER_AMENDMENT_FILENAME,
        ),
    )
    resolved_values: list[str] = []
    for key, artifact_id, member in roles:
        value = _text(inputs[key], key)
        if value.startswith("artifact://"):
            expected = f"artifact://{artifact_id}" + (f"/{member}" if member else "")
            if value != expected:
                raise GovernanceError(
                    f"SCALE-BP v2 direct input URI drifted: {key}."
                )
        else:
            path = Path(value)
            if not path.is_absolute() or path == Path(path.anchor):
                raise GovernanceError(
                    f"SCALE-BP v2 resolved direct input is unsafe: {key}."
                )
            resolved_values.append(value)
    if resolved_values:
        if len(resolved_values) != len(roles):
            raise GovernanceError(
                "SCALE-BP v2 direct inputs mix artifact URIs and resolved paths."
            )
        validate_exact_input_fence(
            DIRECT_INPUT_ARTIFACT_IDS,
            resolved_paths=resolved_values,
        )


def _resolve_output_root(source: Path, value: str) -> Path:
    if value.startswith("output://"):
        if value != f"output://{OUTPUT_ARTIFACT_ID}":
            raise GovernanceError("SCALE-BP v2 output artifact identity drifted.")
        return (_workspace_root(source) / CANONICAL_OUTPUT_RELATIVE_ROOT).resolve()
    return _resolve_location(source.parent, value)


def _resolve_absolute_scratch(value: str) -> Path:
    if value != CANONICAL_SCRATCH_ROOT:
        raise GovernanceError("SCALE-BP v2 scratch root drifted.")
    return Path(value).resolve(strict=False)


def _resolve_location(base: Path, value: str) -> Path:
    if value.startswith(("artifact://", "output://")):
        return Path(value)
    path = Path(value)
    return path.resolve(strict=False) if path.is_absolute() else (base / path).resolve()


def _workspace_root(source: Path) -> Path:
    candidates = (source.parent, *source.parents, Path.cwd(), *Path.cwd().parents)
    for candidate in candidates:
        if (candidate / "src" / "midogpp_thesis").is_dir():
            return candidate.resolve()
    raise GovernanceError("SCALE-BP v2 cannot resolve its workspace root.")


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise GovernanceError(f"SCALE-BP v2 section is absent: {name}.")
    return dict(value)


def _text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value:
        raise GovernanceError(f"SCALE-BP v2 {role} must be nonempty text.")
    return value


def _reject_pending(value: object) -> None:
    if isinstance(value, str) and (
        "__PENDING" in value or value.startswith("pending://")
    ):
        raise GovernanceError("SCALE-BP v2 config contains a pending value.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_pending(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_pending(item)


load_scale_bp_v2_config = load_config


__all__ = (
    "CANONICAL_CLASSIFIER",
    "CONFIG_SCHEMA",
    "CONFIG_TOP_LEVEL",
    "ClassifierConfig",
    "INPUT_LOCATION_KEYS",
    "RESUME_POLICY",
    "SCIENTIFIC_SECTION_NAMES",
    "SOURCE_BINDING_SCHEMA",
    "ScaleBPV2Config",
    "fixed_experiment_payload",
    "fixed_inputs_payload",
    "load_config",
    "load_scale_bp_v2_config",
)
