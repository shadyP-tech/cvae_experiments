"""New, single-use execution authority for terminal HARP sensitivity v1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from types import MappingProxyType

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash, require_sha256
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from .config import HarpStage90Config, INPUT_ARTIFACT_IDS
from .identity import (
    AUTHORIZATION_SCOPE,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    authorization_input_binding_payload,
)
from .source_seal import source_snapshot_identity


_LEASE_DIRECTORY = (
    ".authorization_lease__midogpp_oracle_uniform_b_v2_consumed_test_"
    "fixed_bank_harp_router_v1"
)
EXECUTION_AMENDMENT_ARTIFACT_ID = INPUT_ARTIFACT_IDS[-1]
EXECUTION_AMENDMENT_FILENAME = "harp_stage90_execution_amendment_v1.json"
EXECUTION_AMENDMENT_SCHEMA = "midogpp_harp_stage90_execution_amendment_v2"
SCIENTIFIC_CONTRACT_SCHEMA = "midogpp_harp_stage90_scientific_contract_v1"
AUTHORIZATION_BASIS = (
    "explicit_user_authorization_2026_08_30_for_harp_v1_terminal_"
    "consumed_test_diagnostic"
)
AUTHORIZATION_DATE = "2026-08-30"
WORKSPACE_REGISTRATION_CONTRACT_SCHEMA = (
    "midogpp_harp_stage90_workspace_registration_execution_contract_v1"
)
WORKSPACE_REGISTRY_RELATIVE_PATH = "experiments/midogpp/registry.yaml"
WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH = (
    "experiments/midogpp/artifact_catalog.yaml"
)
WORKSPACE_CONFIG_RELATIVE_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router_v1.yaml"
)
WORKSPACE_PREPARATION_AUTHORITY_GATE = (
    "harp_v1_consumed_test_execution_amendment_v1"
)
WORKSPACE_OUTPUT_CANONICAL_PATH = (
    "artifacts/midogpp/90_oracles_and_diagnostics/"
    "uniform_b_v2_consumed_test_fixed_bank_harp_router/v1"
)
WORKSPACE_RUNNER_ARGV = (
    "{python}",
    "-m",
    "midogpp_thesis",
    "cvae-diagnostics",
    "fixed-bank-harp-router-v1",
    "--config",
    "{resolved_config}",
    "--artifact-root",
    f"output://{OUTPUT_ARTIFACT_ID}",
)
WORKSPACE_RUNNER_ENV = MappingProxyType(
    {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
        "OPENBLAS_NUM_THREADS": "3",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
)


_ACTION_SEMANTICS = {
    "routing_estimand": (
        "frozen_predictive_probability_ensemble_over_frozen_generative_"
        "expert_actions"
    ),
    "matched_budget_reference_action": "U",
    "utility_deltas_reference_action": "U",
    "lambda_semantics": (
        "post_classifier_predictive_probability_ensemble_not_generated_distribution"
    ),
    "physical_expert_routing_primary_lambda": 1.0,
    "exact_b_role": "byte_identical_abstention_and_operational_baseline",
}


@dataclass(frozen=True, slots=True)
class HarpAuthorization:
    amendment_path: Path
    amendment_sha256: str
    amendment_hash: str
    input_binding_hash: str
    scientific_contract_hash: str
    workspace_registration_execution_contract_hash: str
    source_snapshot_schema: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    source_snapshot_member_count: int


@dataclass(frozen=True, slots=True)
class HarpExecutionAmendment:
    """Typed, fully reconstructed HARP execution authority."""

    payload: Mapping[str, object]
    amendment_hash: str
    input_binding_hash: str
    scientific_contract_hash: str
    workspace_registration_execution_contract_hash: str
    source_snapshot_schema: str
    source_snapshot_manifest_sha256: str
    source_snapshot_tree_sha256: str
    source_snapshot_member_count: int


@dataclass(frozen=True, slots=True)
class HarpAuthorizationLease:
    root: Path
    lease_hash: str
    process_id: int


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def lease_path(repo_root: Path | None = None) -> Path:
    repository = repository_root() if repo_root is None else Path(repo_root).resolve()
    return (
        repository
        / "artifacts/midogpp/90_oracles_and_diagnostics"
        / _LEASE_DIRECTORY
    )


def authorization_input_binding(config: HarpStage90Config) -> dict[str, object]:
    """Reconstruct the exact path-independent scientific input binding."""

    if not isinstance(config, HarpStage90Config):
        raise ProtocolError("HARP Stage-90 input binding requires a typed config.")
    roles = (
        "expert_bank_lock_hash",
        "generation_lock_hash",
        "test_cache_content_sha256",
        "development_manifest_sha256",
        "evaluation_manifest_sha256",
        "parent_ledger_sha256",
    )
    values = {role: config.expected_hashes.get(role) for role in roles}
    if any(type(value) is not str or not value for value in values.values()):
        raise ProtocolError("HARP Stage-90 amendment input binding is incomplete.")
    return authorization_input_binding_payload(**values)  # type: ignore[arg-type]


def workspace_registration_execution_contract() -> dict[str, object]:
    """Return the exact executable HARP registry projection.

    Registry notes and claim-scope rationales are deliberately outside this
    projection; every field that can change what the workstation executes is
    closed and hash-bound.
    """

    body: dict[str, object] = {
        "schema_version": WORKSPACE_REGISTRATION_CONTRACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "stage": "90_oracles_and_diagnostics",
        "status": "diagnostic",
        "claim_scope": "diagnostic_only",
        "config_path": WORKSPACE_CONFIG_RELATIVE_PATH,
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "output_canonical_path": WORKSPACE_OUTPUT_CANONICAL_PATH,
        "input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "preparation_authority_gate": WORKSPACE_PREPARATION_AUTHORITY_GATE,
        "run_recovery_strategy": None,
        "runner_argv": list(WORKSPACE_RUNNER_ARGV),
        "runner_environment": dict(WORKSPACE_RUNNER_ENV),
    }
    return {
        **body,
        "workspace_registration_execution_contract_hash": canonical_hash(body),
    }


def validate_workspace_registration_execution_projection(
    projection: Mapping[str, object] | None,
) -> str:
    """Validate the frozen workspace entry that is about to execute HARP.

    This is deliberately a pure, path-free comparison.  Both workspace
    validation and the pre-render authority gate call this function so the
    diagnostic view and the execution boundary cannot drift apart.
    """

    if not isinstance(projection, Mapping):
        raise ProtocolError(
            "HARP v1 in-memory workspace registration projection is absent."
        )
    expected = workspace_registration_execution_contract()
    observed = {
        "schema_version": expected["schema_version"],
        **dict(projection),
        "workspace_registration_execution_contract_hash": expected[
            "workspace_registration_execution_contract_hash"
        ],
    }
    if canonical_bytes(observed) != canonical_bytes(expected):
        raise ProtocolError(
            "HARP v1 in-memory workspace registration execution contract drifted."
        )
    return str(expected["workspace_registration_execution_contract_hash"])


def scientific_contract_payload(config: HarpStage90Config) -> dict[str, object]:
    """Return the non-circular scientific contract and its hash.

    Paths, the config hash, and the execution-amendment digest are deliberately
    absent.  Therefore the resulting identity can be embedded in the amendment
    whose file digest is later written into the executable config.
    """

    if not isinstance(config, HarpStage90Config):
        raise ProtocolError("HARP Stage-90 scientific contract requires typed config.")
    claim_boundary = {
        key: value
        for key, value in config.claim_boundary.items()
        if key
        not in {
            "claim_boundary_hash",
            "execution_authorized",
            "implementation_authorizes_execution",
        }
    }
    body: dict[str, object] = {
        "schema_version": SCIENTIFIC_CONTRACT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "workspace_registration_execution_contract": (
            workspace_registration_execution_contract()
        ),
        "direct_input_artifact_ids": list(config.input_artifact_ids),
        "protocol": dict(config.protocol),
        "model": {
            "schema_version": "midogpp_harp_stage90_model_v1",
            "alpha_grid": list(config.alpha_grid),
            "policy": asdict(config.policy),
            "action_semantics": dict(_ACTION_SEMANTICS),
        },
        "runtime": dict(config.runtime),
        "claim_boundary": claim_boundary,
    }
    return {**body, "scientific_contract_hash": canonical_hash(body)}


def canonical_execution_amendment_payload(
    config: HarpStage90Config,
    *,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Construct the only valid HARP v1 terminal execution amendment."""

    if not isinstance(config, HarpStage90Config) or not config.execution_authorized:
        raise ProtocolError(
            "HARP Stage-90 canonical amendment requires an authorized typed config."
        )
    input_binding = authorization_input_binding(config)
    scientific_contract = scientific_contract_payload(config)
    workspace_registration = workspace_registration_execution_contract()
    source_identity = dict(source_snapshot_identity(repo_root))
    body: dict[str, object] = {
        "schema_version": EXECUTION_AMENDMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_date": AUTHORIZATION_DATE,
        "execution_authorized": True,
        "authorization_is_separate_from_implementation_request": True,
        "implementation_request_alone_authorizes_execution": False,
        "source_code_or_registration_alone_authorizes_execution": False,
        "single_use": True,
        "authorization_exhausted": False,
        "consumed_test_reuse": True,
        "direct_input_artifact_ids": list(config.input_artifact_ids),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "predecessor_authority_reused": False,
        "predecessor_output_or_policy_used": False,
        "old_aggregate_utility_surface_used": False,
        "output_deletion_restores_authority": False,
        "authorized_input_binding": input_binding,
        "scientific_contract_hash": scientific_contract[
            "scientific_contract_hash"
        ],
        "workspace_registration_execution_contract_hash": (
            workspace_registration[
                "workspace_registration_execution_contract_hash"
            ]
        ),
        "source_snapshot_identity": source_identity,
    }
    return {**body, "amendment_hash": canonical_hash(body)}


def validate_execution_amendment_payload(
    value: object,
    config: HarpStage90Config,
    *,
    repo_root: Path | None = None,
) -> HarpExecutionAmendment:
    """Validate amendment semantics once for the runner and workspace gate."""

    if not isinstance(value, Mapping):
        raise ProtocolError("HARP Stage-90 execution amendment must be an object.")
    expected = canonical_execution_amendment_payload(config, repo_root=repo_root)
    try:
        observed_bytes = canonical_bytes(value)
        expected_bytes = canonical_bytes(expected)
    except ProtocolError as exc:
        raise ProtocolError(
            "HARP Stage-90 execution amendment is not canonically typed."
        ) from exc
    if observed_bytes != expected_bytes:
        raise ProtocolError("HARP Stage-90 execution amendment failed authentication.")

    binding = expected["authorized_input_binding"]
    source = expected["source_snapshot_identity"]
    if not isinstance(binding, Mapping) or not isinstance(source, Mapping):
        raise ProtocolError("HARP Stage-90 execution amendment reconstruction failed.")
    amendment_hash = require_sha256(expected["amendment_hash"], name="amendment_hash")
    input_binding_hash = require_sha256(
        binding.get("input_binding_hash"), name="input_binding_hash"
    )
    scientific_contract_hash = require_sha256(
        expected["scientific_contract_hash"], name="scientific_contract_hash"
    )
    registration_contract_hash = require_sha256(
        expected["workspace_registration_execution_contract_hash"],
        name="workspace_registration_execution_contract_hash",
    )
    manifest_hash = require_sha256(
        source.get("source_snapshot_manifest_sha256"),
        name="source_snapshot_manifest_sha256",
    )
    tree_hash = require_sha256(
        source.get("source_snapshot_tree_sha256"),
        name="source_snapshot_tree_sha256",
    )
    member_count = source.get("source_snapshot_member_count")
    snapshot_schema = source.get("source_snapshot_schema")
    if type(snapshot_schema) is not str or not snapshot_schema:
        raise ProtocolError("HARP Stage-90 source snapshot schema is invalid.")
    if type(member_count) is not int or member_count < 1:
        raise ProtocolError("HARP Stage-90 source snapshot member count is invalid.")
    return HarpExecutionAmendment(
        payload=MappingProxyType(dict(expected)),
        amendment_hash=amendment_hash,
        input_binding_hash=input_binding_hash,
        scientific_contract_hash=scientific_contract_hash,
        workspace_registration_execution_contract_hash=(
            registration_contract_hash
        ),
        source_snapshot_schema=snapshot_schema,
        source_snapshot_manifest_sha256=manifest_hash,
        source_snapshot_tree_sha256=tree_hash,
        source_snapshot_member_count=member_count,
    )


def load_authorization(
    config: HarpStage90Config,
    *,
    repo_root: Path | None = None,
) -> HarpAuthorization:
    """Authenticate the new HARP-only amendment without mutating state."""

    if not isinstance(config, HarpStage90Config) or not config.execution_authorized:
        raise ProtocolError(
            "HARP Stage-90 execution is not authorized; a new HARP-specific "
            "single-use amendment is required."
        )
    expected = config.expected_execution_amendment_sha256
    if expected is None:
        raise ProtocolError("HARP Stage-90 execution amendment hash is absent.")
    path = config.resolved_path("execution_amendment_path")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise ProtocolError("HARP Stage-90 execution amendment bytes are absent or drifted.")
    amendment = validate_execution_amendment_payload(
        read_json(path),
        config,
        repo_root=repo_root,
    )
    lease = lease_path(repo_root)
    if lease.exists() or lease.is_symlink():
        raise ProtocolError("HARP Stage-90 single-use authorization is exhausted.")
    if not lease.parent.is_dir() or lease.parent.is_symlink():
        raise ProtocolError("HARP Stage-90 authorization lease parent is unsafe.")
    return HarpAuthorization(
        amendment_path=path,
        amendment_sha256=expected,
        amendment_hash=amendment.amendment_hash,
        input_binding_hash=amendment.input_binding_hash,
        scientific_contract_hash=amendment.scientific_contract_hash,
        workspace_registration_execution_contract_hash=(
            amendment.workspace_registration_execution_contract_hash
        ),
        source_snapshot_schema=amendment.source_snapshot_schema,
        source_snapshot_manifest_sha256=(
            amendment.source_snapshot_manifest_sha256
        ),
        source_snapshot_tree_sha256=amendment.source_snapshot_tree_sha256,
        source_snapshot_member_count=amendment.source_snapshot_member_count,
    )


def _input_binding(config: HarpStage90Config) -> dict[str, object]:
    """Compatibility alias for older tests and preparation receipts."""

    return authorization_input_binding(config)


def claim_authorization(
    authorization: HarpAuthorization, *, admission_hash: str
) -> HarpAuthorizationLease:
    """Atomically consume authority; the lease directory is the first mutation."""

    if not isinstance(authorization, HarpAuthorization):
        raise ProtocolError("HARP Stage-90 lease lacks typed authorization.")
    root = lease_path()
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ProtocolError("HARP Stage-90 single-use authorization is exhausted.") from exc
    except OSError as exc:
        raise ProtocolError("Cannot claim HARP Stage-90 authorization lease.") from exc
    payload = {
        "schema_version": "midogpp_harp_stage90_authorization_lease_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "CLAIMED_IN_PROGRESS",
        "process_id": os.getpid(),
        "admission_hash": admission_hash,
        "amendment_sha256": authorization.amendment_sha256,
        "amendment_hash": authorization.amendment_hash,
        "input_binding_hash": authorization.input_binding_hash,
        "scientific_contract_hash": authorization.scientific_contract_hash,
        "workspace_registration_execution_contract_hash": (
            authorization.workspace_registration_execution_contract_hash
        ),
        "source_snapshot_schema": authorization.source_snapshot_schema,
        "source_snapshot_manifest_sha256": (
            authorization.source_snapshot_manifest_sha256
        ),
        "source_snapshot_tree_sha256": (
            authorization.source_snapshot_tree_sha256
        ),
        "source_snapshot_member_count": authorization.source_snapshot_member_count,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_exhausted": True,
        "recovery_allowed": False,
        "output_deletion_restores_authority": False,
    }
    sealed = {**payload, "lease_hash": canonical_hash(payload)}
    atomic_json(root / "lease.json", sealed)
    return HarpAuthorizationLease(root, str(sealed["lease_hash"]), os.getpid())


def finalize_authorization(
    lease: HarpAuthorizationLease, *, status: str, error: str | None = None
) -> HarpAuthorizationLease:
    if status not in {"COMPLETE_EXHAUSTED", "FAILED_EXHAUSTED"}:
        raise ProtocolError("HARP Stage-90 lease final status is invalid.")
    raw = read_json(lease.root / "lease.json")
    if raw.get("lease_hash") != lease.lease_hash or raw.get("process_id") != os.getpid():
        raise ProtocolError("HARP Stage-90 authorization lease was replaced.")
    base = {key: value for key, value in raw.items() if key not in {"lease_hash", "status"}}
    base.update(
        {
            "status": status,
            "predecessor_lease_hash": lease.lease_hash,
            "error": None if error is None else error[:2000],
        }
    )
    final_hash = canonical_hash(base)
    atomic_json(lease.root / "lease.json", {**base, "lease_hash": final_hash})
    return HarpAuthorizationLease(lease.root, final_hash, lease.process_id)


__all__ = (
    "AUTHORIZATION_BASIS",
    "AUTHORIZATION_DATE",
    "EXECUTION_AMENDMENT_ARTIFACT_ID",
    "EXECUTION_AMENDMENT_FILENAME",
    "EXECUTION_AMENDMENT_SCHEMA",
    "HarpAuthorization",
    "HarpAuthorizationLease",
    "HarpExecutionAmendment",
    "SCIENTIFIC_CONTRACT_SCHEMA",
    "WORKSPACE_CONFIG_RELATIVE_PATH",
    "WORKSPACE_ARTIFACT_CATALOG_RELATIVE_PATH",
    "WORKSPACE_OUTPUT_CANONICAL_PATH",
    "WORKSPACE_PREPARATION_AUTHORITY_GATE",
    "WORKSPACE_REGISTRATION_CONTRACT_SCHEMA",
    "WORKSPACE_REGISTRY_RELATIVE_PATH",
    "WORKSPACE_RUNNER_ARGV",
    "WORKSPACE_RUNNER_ENV",
    "authorization_input_binding",
    "canonical_execution_amendment_payload",
    "claim_authorization",
    "finalize_authorization",
    "lease_path",
    "load_authorization",
    "scientific_contract_payload",
    "validate_execution_amendment_payload",
    "validate_workspace_registration_execution_projection",
    "workspace_registration_execution_contract",
)
