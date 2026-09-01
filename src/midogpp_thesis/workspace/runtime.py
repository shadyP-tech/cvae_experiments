"""Canonical MIDOG++ experiment and artifact workspace runtime.

This module is intentionally orchestration-only. Active reusable MIDOG++ logic
lives elsewhere in the canonical ``midogpp_thesis`` package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .recovery import (
    RecoveryContractError,
    SnapshotBytesGuard,
    detect_registered_exact_recovery,
    registered_recovery_state_status,
    registration_errors,
    validate_preserved_snapshots,
)
from .preparation_authority import (
    AuthorityMember,
    HARP_V1_EXECUTION_AMENDMENT_GATE,
    HARP_V2_EXECUTION_AMENDMENT_GATE,
    HARP_V3_EXECUTION_AMENDMENT_GATE,
    HARP_V3_RUN_CONFIRMATION_TOKEN,
    PreparationAuthorityError,
    PreparationAuthorityReceipt,
    enforce_preparation_authority,
    expected_workspace_registration_contract_hash,
    preparation_authority_registration_error,
    validate_preparation_authority_extra_args,
    validate_preparation_authority_gate_id,
    validate_preparation_authority_registration_projection,
)


ARTIFACT_URI_RE = re.compile(r"^(artifact|output)://([^/]+)(?:/(.*))?$")
RUNNABLE_STATUSES = {"active", "diagnostic"}
BLOCKED_EVIDENCE_LABELS = {"REJECTED"}
ALLOWED_FILE_HASH_ALGORITHMS = {"sha256", "sha512", "blake2b"}
REPOSITORY_MARKER = Path("experiments/midogpp/registry.yaml")
HARP_EXECUTION_AMENDMENT_GATES = frozenset(
    {
        HARP_V1_EXECUTION_AMENDMENT_GATE,
        HARP_V2_EXECUTION_AMENDMENT_GATE,
        HARP_V3_EXECUTION_AMENDMENT_GATE,
    }
)


class WorkspaceError(ValueError):
    """Raised when a workspace definition or requested run is unsafe."""


@dataclass(frozen=True)
class FileHashExpectation:
    algorithm: str
    digest: str


@dataclass(frozen=True)
class ArtifactEntry:
    artifact_id: str
    stage: str
    physical_path: str | None
    canonical_path: str | None
    migration: str
    availability: str
    evidence_label: str
    claim_scope: str
    semantic_identities: Mapping[str, str]
    required_files: tuple[str, ...]
    authoritative_files: tuple[str, ...]
    expected_file_hashes: Mapping[str, FileHashExpectation]
    forbidden_reuse: tuple[str, ...]
    may_feed_recipe_selection: bool | None
    may_feed_deployable_selection: bool | None

    @property
    def provenance_files(self) -> tuple[str, ...]:
        """Files whose bytes must be hashed when the artifact is prepared."""

        return tuple(
            dict.fromkeys(
                (*self.required_files, *self.authoritative_files, *self.expected_file_hashes)
            )
        )


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id: str
    stage: str
    status: str
    claim_scope: str
    output_artifact_id: str
    config_path: str | None
    runner_argv: tuple[str, ...]
    runner_env: Mapping[str, str]
    run_recovery_strategy: str | None
    input_artifact_ids: tuple[str, ...]
    input_claim_scope_exceptions: Mapping[str, str]
    notes: tuple[str, ...]
    preparation_authority_gate: str | None = None

    @property
    def runnable(self) -> bool:
        return self.status in RUNNABLE_STATUSES


@dataclass(frozen=True)
class PreparedRun:
    experiment: ExperimentEntry
    artifact_root: Path
    resolved_config_path: Path
    input_manifest_path: Path
    argv: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class _RenderedRun:
    prepared: PreparedRun
    resolved_config_content: str
    input_manifest_content: str
    input_manifest: Mapping[str, Any]


class MidogppWorkspace:
    """Load, validate, resolve, and launch canonical MIDOG++ experiments."""

    def __init__(
        self,
        *,
        repo_root: Path,
        registry: Mapping[str, Any],
        catalog: Mapping[str, Any],
        workspace: Mapping[str, Any],
        protocol_defaults: Mapping[str, Any],
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.registry_payload = registry
        self.catalog_payload = catalog
        self.workspace_payload = workspace
        self.protocol_defaults_payload = protocol_defaults
        self.stages = self._parse_stages(registry)
        self.artifacts = self._parse_artifacts(catalog)
        self.experiments = self._parse_experiments(registry)

    @classmethod
    def load(cls, repo_root: str | Path | None = None) -> "MidogppWorkspace":
        root = _discover_repo_root(repo_root)
        base = root / "experiments" / "midogpp"
        return cls(
            repo_root=root,
            registry=_read_yaml(base / "registry.yaml"),
            catalog=_read_yaml(base / "artifact_catalog.yaml"),
            workspace=_read_yaml(base / "shared" / "workspace.yaml"),
            protocol_defaults=_read_yaml(base / "shared" / "protocol_defaults.yaml"),
        )

    def validate(self) -> None:
        errors: list[str] = []
        workspace_roots = self.workspace_payload.get("roots", {})
        expected_root = str(workspace_roots.get("canonical_artifacts", ""))
        if expected_root != "artifacts/midogpp":
            errors.append("workspace canonical_artifacts must remain artifacts/midogpp")
        expected_derived_root = str(workspace_roots.get("derived_features", ""))
        if expected_derived_root != "datasets/midogpp/derived/features":
            errors.append(
                "workspace derived_features must remain datasets/midogpp/derived/features"
            )

        catalog_policy = self.catalog_payload.get("catalog_policy", {})
        if not isinstance(catalog_policy, Mapping):
            errors.append("artifact catalog policy must be a mapping")
            catalog_policy = {}
        expected_catalog_roots = {
            "new_run_outputs_belong_under": "artifacts/midogpp",
            "new_dataset_contracts_belong_under": "datasets/midogpp/contract",
            "new_derived_feature_caches_belong_under": "datasets/midogpp/derived/features",
        }
        if "new_outputs_belong_under" in catalog_policy:
            errors.append(
                "artifact catalog policy must distinguish run outputs from reusable inputs"
            )
        for policy_key, required_root in expected_catalog_roots.items():
            actual_root = str(catalog_policy.get(policy_key, ""))
            if actual_root != required_root:
                errors.append(
                    f"artifact catalog {policy_key} must remain {required_root}"
                )
        if not self.stages:
            errors.append("registry must define stages")
        if not self.experiments:
            errors.append("registry must define at least one experiment")

        known_reuse_purposes: set[str] = set()
        for stage_id, stage in self.stages.items():
            allowed_claim_scopes = self._stage_allowed_claim_scopes(stage_id)
            allowed_input_scopes = self._stage_string_values(stage_id, "allowed_input_claim_scopes")
            reuse_purposes = self._stage_string_values(stage_id, "input_reuse_purposes")
            forbidden_upstream = self._stage_string_values(stage_id, "forbidden_upstream")
            known_reuse_purposes.update(reuse_purposes)
            if not allowed_claim_scopes:
                errors.append(f"{stage_id}: stage must declare allowed_claim_scopes")
            if not allowed_input_scopes:
                errors.append(f"{stage_id}: stage must declare allowed_input_claim_scopes")
            if not reuse_purposes:
                errors.append(f"{stage_id}: stage must declare input_reuse_purposes")
            for forbidden_stage in forbidden_upstream:
                if forbidden_stage not in self.stages:
                    errors.append(
                        f"{stage_id}: forbidden_upstream references unknown stage {forbidden_stage}"
                    )
            if stage_id == "60_routing_and_composition" and not self._stage_bool(
                stage_id, "performs_deployable_selection", default=False
            ):
                errors.append(
                    "60_routing_and_composition: performs_deployable_selection must be true"
                )

        for artifact in self.artifacts.values():
            resolved_canonical = (
                None
                if artifact.canonical_path is None
                else self._repo_path(artifact.canonical_path).resolve()
            )
            if artifact.migration == "canonical_output":
                if not artifact.canonical_path:
                    errors.append(f"{artifact.artifact_id}: canonical output lacks canonical_path")
                else:
                    canonical_root = (
                        self.repo_root / expected_catalog_roots["new_run_outputs_belong_under"]
                    ).resolve()
                    if (
                        resolved_canonical == canonical_root
                        or not resolved_canonical.is_relative_to(canonical_root)
                    ):
                        errors.append(
                            f"{artifact.artifact_id}: canonical output escapes artifacts/midogpp"
                        )
            if artifact.stage == "dataset_contract" and resolved_canonical is not None:
                contract_root = (
                    self.repo_root / expected_catalog_roots["new_dataset_contracts_belong_under"]
                ).resolve()
                if resolved_canonical == contract_root or not resolved_canonical.is_relative_to(
                    contract_root
                ):
                    errors.append(
                        f"{artifact.artifact_id}: canonical dataset contract escapes "
                        "datasets/midogpp/contract"
                    )
            if artifact.stage == "derived_features" and resolved_canonical is not None:
                derived_root = (
                    self.repo_root
                    / expected_catalog_roots["new_derived_feature_caches_belong_under"]
                ).resolve()
                if resolved_canonical == derived_root or not resolved_canonical.is_relative_to(
                    derived_root
                ):
                    errors.append(
                        f"{artifact.artifact_id}: canonical derived feature escapes "
                        "datasets/midogpp/derived/features"
                    )
            if artifact.evidence_label in BLOCKED_EVIDENCE_LABELS and artifact.may_feed_deployable_selection:
                errors.append(f"{artifact.artifact_id}: rejected artifact may not feed selection")
            for relative in artifact.provenance_files:
                try:
                    self._safe_member(
                        self.repo_root / ".midogpp-artifact-validation-root",
                        relative,
                        f"artifact://{artifact.artifact_id}/{relative}",
                    )
                except WorkspaceError as exc:
                    errors.append(str(exc))
            unknown_reuse = set(artifact.forbidden_reuse).difference(known_reuse_purposes)
            if unknown_reuse:
                errors.append(
                    f"{artifact.artifact_id}: forbidden_reuse contains unknown purposes "
                    f"{sorted(unknown_reuse)}"
                )

        for experiment in self.experiments.values():
            if experiment.stage not in self.stages:
                errors.append(f"{experiment.experiment_id}: unknown stage {experiment.stage}")
                continue
            allowed_claim_scopes = set(self._stage_allowed_claim_scopes(experiment.stage))
            allowed_input_scopes = set(
                self._stage_string_values(experiment.stage, "allowed_input_claim_scopes")
            )
            reuse_purposes = set(
                self._stage_string_values(experiment.stage, "input_reuse_purposes")
            )
            forbidden_upstream = set(
                self._stage_string_values(experiment.stage, "forbidden_upstream")
            )
            deployable_selection = self._stage_bool(
                experiment.stage, "performs_deployable_selection", default=False
            )
            if experiment.claim_scope not in allowed_claim_scopes:
                errors.append(
                    f"{experiment.experiment_id}: claim_scope {experiment.claim_scope!r} is not "
                    f"allowed by stage {experiment.stage}"
                )
            for artifact_id, rationale in experiment.input_claim_scope_exceptions.items():
                if artifact_id not in experiment.input_artifact_ids:
                    errors.append(
                        f"{experiment.experiment_id}: claim-scope exception references undeclared "
                        f"input {artifact_id}"
                    )
                if not rationale.strip():
                    errors.append(
                        f"{experiment.experiment_id}: claim-scope exception for {artifact_id} "
                        "requires a non-empty rationale"
                    )
            output = self.artifacts.get(experiment.output_artifact_id)
            if output is None:
                errors.append(f"{experiment.experiment_id}: missing output artifact {experiment.output_artifact_id}")
            elif output.migration != "canonical_output":
                errors.append(f"{experiment.experiment_id}: output must use migration=canonical_output")
            elif output.stage != experiment.stage:
                errors.append(f"{experiment.experiment_id}: output stage does not match experiment stage")
            elif output.claim_scope != experiment.claim_scope:
                errors.append(
                    f"{experiment.experiment_id}: output claim_scope {output.claim_scope!r} does "
                    f"not match experiment claim_scope {experiment.claim_scope!r}"
                )
            for artifact_id in experiment.input_artifact_ids:
                artifact = self.artifacts.get(artifact_id)
                if artifact is None:
                    errors.append(f"{experiment.experiment_id}: unknown input artifact {artifact_id}")
                    continue
                if experiment.runnable and artifact.evidence_label in BLOCKED_EVIDENCE_LABELS:
                    errors.append(f"{experiment.experiment_id}: runnable experiment uses rejected {artifact_id}")
                if artifact.stage in forbidden_upstream:
                    errors.append(
                        f"{experiment.experiment_id}: consumes forbidden upstream stage "
                        f"{artifact.stage} via {artifact_id}"
                    )
                if deployable_selection and artifact.may_feed_deployable_selection is not True:
                    errors.append(
                        f"{experiment.experiment_id}: deployable selection requires "
                        f"may_feed_deployable_selection=true for {artifact_id}; got "
                        f"{artifact.may_feed_deployable_selection!r}"
                    )
                authorized_consumers = artifact.semantic_identities.get(
                    "authorized_consumer_experiment_ids"
                )
                registered_consumers = artifact.semantic_identities.get(
                    "registered_consumer_experiment_ids"
                )
                authorized_consumer_ids = {
                    value.strip()
                    for value in (authorized_consumers or "").split("|")
                    if value.strip()
                }
                registered_consumer_ids = {
                    value.strip()
                    for value in (registered_consumers or "").split("|")
                    if value.strip()
                }
                if authorized_consumers and registered_consumers:
                    errors.append(
                        f"{experiment.experiment_id}: input {artifact_id} mixes "
                        "authorized and resolution-only registered consumer fences"
                    )
                if authorized_consumers:
                    if experiment.experiment_id not in authorized_consumer_ids:
                        errors.append(
                            f"{experiment.experiment_id}: input {artifact_id} is fenced to "
                            f"authorized consumers {sorted(authorized_consumer_ids)}"
                        )
                if registered_consumers:
                    if experiment.experiment_id not in registered_consumer_ids:
                        errors.append(
                            f"{experiment.experiment_id}: input {artifact_id} is fenced to "
                            "resolution-only registered consumers "
                            f"{sorted(registered_consumer_ids)}"
                        )
                forbidden_reuse = reuse_purposes.intersection(artifact.forbidden_reuse)
                claim_scope_rationale = experiment.input_claim_scope_exceptions.get(
                    artifact_id, ""
                )
                permits_planned_resolution_only_reuse = (
                    registered_consumer_ids == {experiment.experiment_id}
                    and experiment.status == "planned"
                    and not experiment.runnable
                    and artifact.semantic_identities.get(
                        "consumer_resolution_fence_only"
                    )
                    == "true"
                    and artifact.semantic_identities.get("execution_authorized")
                    == "false"
                    and artifact.semantic_identities.get(
                        "consumed_test_reuse_authorized"
                    )
                    == "false"
                )
                if registered_consumers and not permits_planned_resolution_only_reuse:
                    errors.append(
                        f"{experiment.experiment_id}: input {artifact_id} registered-consumer "
                        "fence is valid only for a planned, non-runnable resolution-only "
                        "experiment"
                    )
                permits_single_consumer_oracle_reuse = (
                    forbidden_reuse == {"oracle_and_diagnostic_evidence"}
                    and (
                        authorized_consumer_ids == {experiment.experiment_id}
                        or permits_planned_resolution_only_reuse
                    )
                    and bool(claim_scope_rationale.strip())
                )
                if forbidden_reuse and not permits_single_consumer_oracle_reuse:
                    errors.append(
                        f"{experiment.experiment_id}: {artifact_id} forbids reuse as "
                        f"{sorted(forbidden_reuse)}"
                    )
                if (
                    artifact.claim_scope not in allowed_input_scopes
                    and artifact_id not in experiment.input_claim_scope_exceptions
                ):
                    errors.append(
                        f"{experiment.experiment_id}: input {artifact_id} claim_scope "
                        f"{artifact.claim_scope!r} is incompatible with stage {experiment.stage}; "
                        "add an artifact-specific input_claim_scope_exceptions rationale only after "
                        "protocol review"
                    )
            if experiment.config_path:
                raw_config = Path(experiment.config_path)
                config = self._repo_path(experiment.config_path).resolve()
                if not config.is_file():
                    errors.append(f"{experiment.experiment_id}: missing config {experiment.config_path}")
                elif raw_config.is_absolute() or not config.is_relative_to(self.repo_root):
                    errors.append(f"{experiment.experiment_id}: config is outside repository")
            if not experiment.runner_argv:
                errors.append(f"{experiment.experiment_id}: runner argv is empty")
            authority_registration_error = preparation_authority_registration_error(
                experiment.preparation_authority_gate,
                experiment_id=experiment.experiment_id,
            )
            if authority_registration_error is not None:
                errors.append(authority_registration_error)
            output_canonical_path = None if output is None else output.canonical_path
            if (
                experiment.runnable
                and experiment.preparation_authority_gate
                in HARP_EXECUTION_AMENDMENT_GATES
                and output is not None
            ):
                try:
                    validate_preparation_authority_registration_projection(
                        experiment.preparation_authority_gate,
                        self._preparation_authority_registration_projection(
                            experiment
                        ),
                    )
                except (PreparationAuthorityError, WorkspaceError) as exc:
                    errors.append(f"{experiment.experiment_id}: {exc}")
            errors.extend(
                registration_errors(
                    experiment.run_recovery_strategy,
                    experiment_id=experiment.experiment_id,
                    stage=experiment.stage,
                    status=experiment.status,
                    claim_scope=experiment.claim_scope,
                    config_path=experiment.config_path,
                    output_artifact_id=experiment.output_artifact_id,
                    output_canonical_path=output_canonical_path,
                    input_artifact_ids=experiment.input_artifact_ids,
                    runner_argv=experiment.runner_argv,
                    runner_env=experiment.runner_env,
                )
            )

        if errors:
            raise WorkspaceError("Invalid MIDOG++ workspace:\n- " + "\n- ".join(errors))

    def resolve_artifact(
        self,
        artifact_id: str,
        *,
        for_output: bool = False,
        require_exists: bool = True,
    ) -> Path:
        try:
            artifact = self.artifacts[artifact_id]
        except KeyError as exc:
            raise WorkspaceError(f"Unknown MIDOG++ artifact_id: {artifact_id}") from exc
        if for_output:
            if artifact.migration != "canonical_output" or not artifact.canonical_path:
                raise WorkspaceError(f"Artifact is not a canonical output: {artifact_id}")
            path = self._repo_path(artifact.canonical_path).resolve()
            canonical_root = (self.repo_root / "artifacts" / "midogpp").resolve()
            if path == canonical_root or not path.is_relative_to(canonical_root):
                raise WorkspaceError(f"Canonical output escapes artifacts/midogpp: {artifact_id}")
            return path

        candidates = self._artifact_candidates(artifact)
        for candidate in candidates:
            if candidate.exists() and all(
                self._safe_member(
                    candidate,
                    relative,
                    f"artifact://{artifact.artifact_id}/{relative}",
                ).is_file()
                for relative in artifact.provenance_files
            ):
                return candidate
        if require_exists:
            rendered = ", ".join(str(path) for path in candidates) or "<no paths declared>"
            raise WorkspaceError(f"Artifact {artifact_id} is unavailable; checked: {rendered}")
        if candidates:
            return candidates[0]
        raise WorkspaceError(f"Artifact {artifact_id} has no resolvable path")

    def resolve_value(
        self,
        value: Any,
        *,
        require_inputs: bool,
        used_inputs: set[str] | None = None,
    ) -> Any:
        if isinstance(value, Mapping):
            return {
                key: self.resolve_value(item, require_inputs=require_inputs, used_inputs=used_inputs)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.resolve_value(item, require_inputs=require_inputs, used_inputs=used_inputs) for item in value]
        if not isinstance(value, str):
            return value
        match = ARTIFACT_URI_RE.match(value)
        if match is None:
            return value
        mode, artifact_id, suffix = match.groups()
        if mode == "artifact":
            if used_inputs is not None:
                used_inputs.add(artifact_id)
            if suffix:
                artifact = self.artifacts.get(artifact_id)
                if artifact is None:
                    raise WorkspaceError(f"Unknown MIDOG++ artifact_id: {artifact_id}")
                members = [self._safe_member(root, suffix, value) for root in self._artifact_candidates(artifact)]
                for member in members:
                    if member.exists():
                        return str(member)
                if require_inputs:
                    checked = ", ".join(str(member) for member in members) or "<no paths declared>"
                    raise WorkspaceError(f"Artifact member {value} is unavailable; checked: {checked}")
                if members:
                    return str(members[0])
                raise WorkspaceError(f"Artifact {artifact_id} has no resolvable path")
            root = self.resolve_artifact(artifact_id, require_exists=require_inputs)
        else:
            root = self.resolve_artifact(artifact_id, for_output=True, require_exists=False)
        if suffix:
            return str(self._safe_member(root, suffix, value))
        return str(root.resolve())

    def prepare(
        self,
        experiment_id: str,
        *,
        require_inputs: bool = True,
        force: bool = False,
        _run_admitted: bool = False,
    ) -> PreparedRun:
        self.validate()
        experiment = self.get_experiment(experiment_id)
        if not experiment.runnable:
            raise WorkspaceError(
                f"Experiment {experiment_id} is status={experiment.status!r} and cannot be launched"
            )
        try:
            validate_preparation_authority_extra_args(
                experiment.preparation_authority_gate,
                (),
                force=force,
                preparation_only=_run_admitted,
            )
        except PreparationAuthorityError as exc:
            raise WorkspaceError(f"{experiment_id}: {exc}") from exc
        authority_receipt = self._enforce_preparation_authority(
            experiment,
        )
        rendered = self._render_run(
            experiment_id,
            require_inputs=require_inputs,
            validate_workspace=False,
            include_all_declared_inputs=False,
            authority_receipt=authority_receipt,
        )
        prepared = rendered.prepared
        for relative in ("manifests", "provenance", "reports", "tables"):
            (prepared.artifact_root / relative).mkdir(parents=True, exist_ok=True)
        _write_checked(
            prepared.resolved_config_path,
            rendered.resolved_config_content,
            force=force,
        )
        _write_checked(
            prepared.input_manifest_path,
            rendered.input_manifest_content,
            force=force,
        )
        return prepared

    def _enforce_preparation_authority(
        self,
        experiment: ExperimentEntry,
    ) -> PreparationAuthorityReceipt | None:
        if experiment.preparation_authority_gate is None:
            return None
        try:
            return enforce_preparation_authority(
                experiment.preparation_authority_gate,
                repo_root=self.repo_root,
                experiment_id=experiment.experiment_id,
                config_path=experiment.config_path,
                input_artifact_ids=experiment.input_artifact_ids,
                registration_projection=(
                    self._preparation_authority_registration_projection(
                        experiment
                    )
                ),
                resolve_authority_member=self._resolve_preparation_authority_member,
            )
        except PreparationAuthorityError as exc:
            raise WorkspaceError(
                f"{experiment.experiment_id}: preparation authority rejected: {exc}"
            ) from exc

    def validate_preparation_authority(
        self,
        experiment_id: str,
    ) -> PreparationAuthorityReceipt | None:
        """Validate one registered authority through the real catalog resolver.

        This is the public, read-only counterpart of the pre-render gate.  It
        lets an administrative transaction authenticate its committed
        registry/catalog/config projection without fabricating an authority
        member or preparing an output snapshot.
        """

        self.validate()
        experiment = self.get_experiment(experiment_id)
        if not experiment.runnable:
            raise WorkspaceError(
                f"Experiment {experiment_id} is status={experiment.status!r} "
                "and has no executable preparation authority"
            )
        return self._enforce_preparation_authority(experiment)

    def _preparation_authority_registration_projection(
        self,
        experiment: ExperimentEntry,
    ) -> Mapping[str, object] | None:
        """Project the frozen entry and catalog target that will execute."""

        if (
            experiment.preparation_authority_gate
            not in HARP_EXECUTION_AMENDMENT_GATES
        ):
            return None
        output = self.artifacts.get(experiment.output_artifact_id)
        if output is None:
            raise WorkspaceError(
                f"{experiment.experiment_id}: HARP output artifact is absent"
            )
        return {
            "experiment_id": experiment.experiment_id,
            "stage": experiment.stage,
            "status": experiment.status,
            "claim_scope": experiment.claim_scope,
            "config_path": experiment.config_path,
            "output_artifact_id": experiment.output_artifact_id,
            "output_canonical_path": output.canonical_path,
            "input_artifact_ids": list(experiment.input_artifact_ids),
            "preparation_authority_gate": experiment.preparation_authority_gate,
            "run_recovery_strategy": experiment.run_recovery_strategy,
            "runner_argv": list(experiment.runner_argv),
            "runner_environment": dict(experiment.runner_env),
        }

    def _resolve_preparation_authority_member(
        self,
        artifact_id: str,
        relative: str,
    ) -> AuthorityMember:
        """Resolve one catalog-pinned authority file, never a scientific input."""

        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            raise WorkspaceError(f"Unknown MIDOG++ authority artifact_id: {artifact_id}")
        if artifact.provenance_files != (relative,):
            raise WorkspaceError(
                f"Authority artifact {artifact_id} must expose only {relative!r}"
            )
        expectation = artifact.expected_file_hashes.get(relative)
        if expectation is None or expectation.algorithm != "sha256":
            raise WorkspaceError(
                f"Authority artifact {artifact_id} lacks an exact sha256 expectation"
            )
        raw_root_value = artifact.canonical_path or artifact.physical_path
        if raw_root_value is None:
            raise WorkspaceError(
                f"Authority artifact {artifact_id} has no resolvable path"
            )
        raw_root = self._repo_path(raw_root_value)
        if raw_root.is_symlink():
            raise WorkspaceError(
                f"Authority artifact {artifact_id} root may not be a symlink"
            )
        root = raw_root.resolve()
        member = self._safe_member(
            root,
            relative,
            f"artifact://{artifact_id}/{relative}",
        )
        if member.is_symlink() or not member.is_file():
            raise WorkspaceError(
                f"Authority artifact {artifact_id} member is absent or unsafe: {member}"
            )
        return AuthorityMember(path=member, expected_sha256=expectation.digest)

    def _render_run(
        self,
        experiment_id: str,
        *,
        require_inputs: bool,
        validate_workspace: bool,
        include_all_declared_inputs: bool,
        authority_receipt: PreparationAuthorityReceipt | None = None,
    ) -> _RenderedRun:
        """Resolve and serialize a run without creating or changing any file."""

        if validate_workspace:
            self.validate()
        experiment = self.get_experiment(experiment_id)
        if not experiment.runnable:
            raise WorkspaceError(
                f"Experiment {experiment_id} is status={experiment.status!r} and cannot be launched"
            )
        if (
            authority_receipt is None
            and experiment.preparation_authority_gate is not None
        ):
            # Private provenance-replay callers cannot bypass the same gate.
            # Public prepare/run paths already carry a receipt into this method.
            if not validate_workspace:
                self.validate()
            authority_receipt = self._enforce_preparation_authority(experiment)
        self._verify_preparation_authority_receipt(
            experiment,
            receipt=authority_receipt,
        )
        artifact_root = self.resolve_artifact(
            experiment.output_artifact_id,
            for_output=True,
            require_exists=False,
        )

        used_inputs: set[str] = set()
        resolved_config_path = artifact_root / "config.resolved.yaml"
        resolved_payload: Any | None = None
        if experiment.config_path is not None:
            config_payload = _read_yaml(self.repo_root / experiment.config_path)
            resolved_payload = self.resolve_value(
                config_payload,
                require_inputs=require_inputs,
                used_inputs=used_inputs,
            )

        token_context = {
            "{python}": sys.executable,
            "{repo}": str(self.repo_root),
            "{resolved_config}": str(resolved_config_path),
        }
        argv: list[str] = []
        for raw in experiment.runner_argv:
            value = token_context.get(raw, raw)
            value = self.resolve_value(value, require_inputs=require_inputs, used_inputs=used_inputs)
            if value != "":
                argv.append(str(value))
        env = {
            key: str(self.resolve_value(value, require_inputs=require_inputs, used_inputs=used_inputs))
            for key, value in experiment.runner_env.items()
        }
        undeclared = used_inputs.difference(experiment.input_artifact_ids)
        if undeclared:
            raise WorkspaceError(
                f"{experiment_id}: config or runner uses undeclared input artifacts: {sorted(undeclared)}"
            )
        manifest_artifact_ids = (
            set(experiment.input_artifact_ids)
            if include_all_declared_inputs
            else used_inputs or set(experiment.input_artifact_ids)
        )
        input_manifest = self._input_manifest(
            experiment,
            manifest_artifact_ids,
            require_inputs,
        )
        if resolved_payload is None:
            resolved_payload = {
                "schema_version": "midogpp_resolved_command_v1",
                "experiment": {
                    "experiment_id": experiment.experiment_id,
                    "stage": experiment.stage,
                    "claim_scope": experiment.claim_scope,
                    "artifact_root": str(artifact_root),
                },
                "inputs": {"artifact_ids": list(experiment.input_artifact_ids)},
                "runner": {"environment": env, "argv": argv},
            }
        input_manifest_path = artifact_root / "provenance" / "input_artifacts.json"
        return _RenderedRun(
            prepared=PreparedRun(
                experiment=experiment,
                artifact_root=artifact_root,
                resolved_config_path=resolved_config_path,
                input_manifest_path=input_manifest_path,
                argv=tuple(argv),
                env=env,
            ),
            resolved_config_content=yaml.safe_dump(resolved_payload, sort_keys=False),
            input_manifest_content=json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
            input_manifest=input_manifest,
        )

    def _verify_preparation_authority_receipt(
        self,
        experiment: ExperimentEntry,
        *,
        receipt: PreparationAuthorityReceipt | None,
    ) -> None:
        gate_id = experiment.preparation_authority_gate
        if gate_id is None:
            if receipt is not None:
                raise WorkspaceError("Unexpected workspace preparation authority receipt")
            return
        if receipt is None:
            raise WorkspaceError(
                f"{experiment.experiment_id}: pre-render preparation authority receipt is required"
            )
        expected_config = (
            None
            if experiment.config_path is None
            else self._repo_path(experiment.config_path).resolve()
        )
        if (
            receipt.gate_id != gate_id
            or receipt.experiment_id != experiment.experiment_id
            or expected_config is None
            or receipt.config_path != expected_config
            or receipt.config_path.is_symlink()
            or receipt.authority_path.is_symlink()
            or not receipt.config_path.is_file()
            or not receipt.authority_path.is_file()
            or _hash_file(receipt.config_path, "sha256") != receipt.config_sha256
            or _hash_file(receipt.authority_path, "sha256")
            != receipt.authority_sha256
        ):
            raise WorkspaceError(
                f"{experiment.experiment_id}: pre-render preparation authority bytes changed"
            )
        if gate_id in HARP_EXECUTION_AMENDMENT_GATES:
            try:
                expected_registration_hash = (
                    expected_workspace_registration_contract_hash(gate_id)
                )
            except PreparationAuthorityError as exc:
                raise WorkspaceError(str(exc)) from exc
            if (
                type(receipt.workspace_registration_contract_hash) is not str
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    receipt.workspace_registration_contract_hash,
                )
                is None
                or receipt.workspace_registration_contract_hash
                != expected_registration_hash
                or receipt.registry_path is None
                or receipt.registry_sha256 is None
                or receipt.artifact_catalog_path is None
                or receipt.artifact_catalog_sha256 is None
                or receipt.registry_path.is_symlink()
                or receipt.artifact_catalog_path.is_symlink()
                or not receipt.registry_path.is_file()
                or not receipt.artifact_catalog_path.is_file()
                or _hash_file(receipt.registry_path, "sha256")
                != receipt.registry_sha256
                or _hash_file(receipt.artifact_catalog_path, "sha256")
                != receipt.artifact_catalog_sha256
            ):
                raise WorkspaceError(
                    f"{experiment.experiment_id}: HARP workspace registration "
                    "authority bytes changed"
                )

    def run(self, experiment_id: str, *, force: bool = False, extra_args: Sequence[str] = ()) -> int:
        self.validate()
        experiment = self.get_experiment(experiment_id)
        if not experiment.runnable:
            raise WorkspaceError(
                f"Experiment {experiment_id} is status={experiment.status!r} and cannot be launched"
            )
        try:
            extra_args = validate_preparation_authority_extra_args(
                experiment.preparation_authority_gate,
                extra_args,
                force=force,
            )
        except PreparationAuthorityError as exc:
            raise WorkspaceError(f"{experiment_id}: {exc}") from exc
        authority_receipt = self._enforce_preparation_authority(
            experiment,
        )
        artifact_root = self.resolve_artifact(
            experiment.output_artifact_id,
            for_output=True,
            require_exists=False,
        )
        strategy = experiment.run_recovery_strategy
        try:
            recovery_state_status = (
                registered_recovery_state_status(strategy, artifact_root)
                if strategy is not None
                else None
            )
        except RecoveryContractError as exc:
            raise WorkspaceError(str(exc)) from exc
        recovery_detected = strategy is not None and detect_registered_exact_recovery(
            strategy, artifact_root
        )
        if recovery_detected:
            if force:
                raise WorkspaceError(
                    "Registered exact-existing-snapshot recovery rejects --force."
                )
            if extra_args:
                raise WorkspaceError(
                    "Registered exact-existing-snapshot recovery rejects extra runner arguments."
                )
            rendered = self._render_run(
                experiment_id,
                require_inputs=True,
                validate_workspace=False,
                include_all_declared_inputs=True,
                authority_receipt=authority_receipt,
            )
            prepared = rendered.prepared
            try:
                guard = SnapshotBytesGuard.capture(
                    prepared.resolved_config_path,
                    prepared.input_manifest_path,
                )
                validate_preserved_snapshots(
                    guard,
                    current_resolved_config_bytes=rendered.resolved_config_content.encode(
                        "utf-8"
                    ),
                    current_input_manifest=rendered.input_manifest,
                )
                guard.assert_unchanged()
            except RecoveryContractError as exc:
                raise WorkspaceError(str(exc)) from exc
            return self._execute(prepared, extra_args=(), recovery_guard=guard)

        if strategy is not None and recovery_state_status in {"FAILED", "RUNNING"}:
            raise WorkspaceError(
                "Registered exact-existing-snapshot recovery state is unrecognized; "
                f"refusing normal preparation for strategy {strategy!r}."
            )

        prepared = self.prepare(
            experiment_id,
            require_inputs=True,
            force=force,
            _run_admitted=True,
        )
        return self._execute(prepared, extra_args=extra_args)

    def _execute(
        self,
        prepared: PreparedRun,
        *,
        extra_args: Sequence[str],
        recovery_guard: SnapshotBytesGuard | None = None,
    ) -> int:
        env = os.environ.copy()
        env.update(prepared.env)
        try:
            if recovery_guard is not None:
                self._assert_recovery_snapshots_unchanged(recovery_guard)
            completed = subprocess.run(
                [*prepared.argv, *extra_args],
                cwd=self.repo_root,
                env=env,
                check=False,
            )
        except BaseException:
            if recovery_guard is not None:
                self._assert_recovery_snapshots_unchanged(recovery_guard)
            raise
        if recovery_guard is not None:
            self._assert_recovery_snapshots_unchanged(recovery_guard)
        return int(completed.returncode)

    @staticmethod
    def _assert_recovery_snapshots_unchanged(guard: SnapshotBytesGuard) -> None:
        try:
            guard.assert_unchanged()
        except RecoveryContractError as exc:
            raise WorkspaceError(str(exc)) from exc

    def get_experiment(self, experiment_id: str) -> ExperimentEntry:
        try:
            return self.experiments[experiment_id]
        except KeyError as exc:
            raise WorkspaceError(f"Unknown MIDOG++ experiment_id: {experiment_id}") from exc

    def central_command(self, experiment_id: str) -> str:
        experiment = self.get_experiment(experiment_id)
        argv = [
            sys.executable,
            "-m",
            "midogpp_thesis",
            "workspace",
            "run",
            experiment_id,
        ]
        if experiment.preparation_authority_gate == HARP_V3_EXECUTION_AMENDMENT_GATE:
            argv.extend(("--", "--confirm", HARP_V3_RUN_CONFIRMATION_TOKEN))
        return shlex.join(argv)

    def _input_manifest(
        self,
        experiment: ExperimentEntry,
        artifact_ids: set[str],
        require_inputs: bool,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for artifact_id in sorted(artifact_ids):
            artifact = self.artifacts[artifact_id]
            resolved = self.resolve_artifact(artifact_id, require_exists=require_inputs)
            file_integrity = self._artifact_file_integrity(
                artifact,
                resolved,
                require_inputs=require_inputs,
            )
            rows.append(
                {
                    "artifact_id": artifact_id,
                    "resolved_path": str(resolved),
                    "stage": artifact.stage,
                    "evidence_label": artifact.evidence_label,
                    "claim_scope": artifact.claim_scope,
                    "semantic_identities": dict(artifact.semantic_identities),
                    "semantic_identities_are_file_hashes": False,
                    "file_integrity": file_integrity,
                    "exists": resolved.exists(),
                }
            )
        return {
            "schema_version": "midogpp_input_artifacts_v2",
            "dataset_id": "midogpp",
            "experiment_id": experiment.experiment_id,
            "stage": experiment.stage,
            "claim_scope": experiment.claim_scope,
            "selection_used_target_eval_artifacts": False,
            "input_artifacts": rows,
            **_git_state(self.repo_root),
        }

    def _artifact_file_integrity(
        self,
        artifact: ArtifactEntry,
        root: Path,
        *,
        require_inputs: bool,
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        missing = False
        has_expectations = bool(artifact.expected_file_hashes)
        for relative in artifact.provenance_files:
            member = self._safe_member(
                root,
                relative,
                f"artifact://{artifact.artifact_id}/{relative}",
            )
            expectation = artifact.expected_file_hashes.get(relative)
            row: dict[str, Any] = {
                "path": relative,
                "resolved_path": str(member),
                "exists": member.exists(),
                "expected": (
                    None
                    if expectation is None
                    else {"algorithm": expectation.algorithm, "digest": expectation.digest}
                ),
            }
            if not member.exists():
                missing = True
                row.update({"size_bytes": None, "computed": {}, "verification": "MISSING"})
                files.append(row)
                if require_inputs:
                    raise WorkspaceError(
                        f"Artifact {artifact.artifact_id} provenance file is missing: {member}"
                    )
                continue
            if not member.is_file():
                raise WorkspaceError(
                    f"Artifact {artifact.artifact_id} provenance member is not a file: {member}"
                )
            algorithms = {"sha256"}
            if expectation is not None:
                algorithms.add(expectation.algorithm)
            computed = {algorithm: _hash_file(member, algorithm) for algorithm in sorted(algorithms)}
            verification = "RECORDED_NO_EXPECTATION"
            if expectation is not None:
                actual = computed[expectation.algorithm]
                if actual != expectation.digest:
                    raise WorkspaceError(
                        f"Artifact {artifact.artifact_id} file hash mismatch for {relative}: "
                        f"expected {expectation.algorithm}:{expectation.digest}, got "
                        f"{expectation.algorithm}:{actual}"
                    )
                verification = "MATCH"
            row.update(
                {
                    "size_bytes": member.stat().st_size,
                    "computed": computed,
                    "verification": verification,
                }
            )
            files.append(row)
        if missing:
            status = "MISSING_ALLOWED"
        elif has_expectations:
            status = "EXPECTED_FILE_HASHES_MATCH"
        elif files:
            status = "HASHES_RECORDED_NO_EXPECTATIONS"
        else:
            status = "NO_PROVENANCE_FILES_DECLARED"
        return {
            "status": status,
            "default_recording_algorithm": "sha256",
            "files": files,
        }

    def _repo_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    def _artifact_candidates(self, artifact: ArtifactEntry) -> list[Path]:
        candidates: list[Path] = []
        if artifact.canonical_path:
            candidates.append(self._repo_path(artifact.canonical_path).resolve())
            return candidates
        if artifact.physical_path:
            physical = self._repo_path(artifact.physical_path).resolve()
            if physical not in candidates:
                candidates.append(physical)
        return candidates

    def _stage_allowed_claim_scopes(self, stage_id: str) -> tuple[str, ...]:
        stage = self.stages[stage_id]
        if "allowed_claim_scopes" in stage:
            return _string_sequence(
                stage.get("allowed_claim_scopes"),
                label=f"{stage_id}.allowed_claim_scopes",
            )
        singular = stage.get("allowed_claim_scope")
        return () if singular in (None, "") else (str(singular),)

    def _stage_string_values(self, stage_id: str, field: str) -> tuple[str, ...]:
        return _string_sequence(
            self.stages[stage_id].get(field, ()),
            label=f"{stage_id}.{field}",
        )

    def _stage_bool(self, stage_id: str, field: str, *, default: bool) -> bool:
        value = self.stages[stage_id].get(field, default)
        if not isinstance(value, bool):
            raise WorkspaceError(f"{stage_id}.{field} must be a boolean")
        return value

    @staticmethod
    def _safe_member(root: Path, suffix: str, uri: str) -> Path:
        resolved_root = root.resolve()
        resolved = (resolved_root / suffix).resolve()
        if not resolved.is_relative_to(resolved_root):
            raise WorkspaceError(f"Artifact URI escapes its root: {uri}")
        return resolved

    @staticmethod
    def _parse_stages(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        stages: dict[str, Mapping[str, Any]] = {}
        for raw in registry.get("stages", ()):
            if not isinstance(raw, Mapping):
                raise WorkspaceError("registry stages must be mappings")
            directory = str(raw.get("directory", ""))
            stage = Path(directory).name
            if not stage:
                raise WorkspaceError("registry stage lacks directory")
            if stage in stages:
                raise WorkspaceError(f"duplicate registry stage: {stage}")
            stages[stage] = raw
        return stages

    @staticmethod
    def _parse_artifacts(catalog: Mapping[str, Any]) -> dict[str, ArtifactEntry]:
        parsed: dict[str, ArtifactEntry] = {}
        for raw in catalog.get("artifacts", ()):
            if not isinstance(raw, Mapping):
                raise WorkspaceError("catalog artifacts must be mappings")
            artifact_id = str(raw.get("artifact_id", ""))
            if not artifact_id or artifact_id in parsed:
                raise WorkspaceError(f"missing or duplicate artifact_id: {artifact_id!r}")
            if "semantic_identities" in raw and "declared_hashes" in raw:
                raise WorkspaceError(
                    f"{artifact_id}: use semantic_identities, not both semantic_identities and "
                    "legacy declared_hashes"
                )
            identities = raw.get("semantic_identities", raw.get("declared_hashes", {})) or {}
            if not isinstance(identities, Mapping):
                raise WorkspaceError(f"{artifact_id}: semantic_identities must be a mapping")
            required_files = _string_sequence(
                raw.get("required_files", ()),
                label=f"{artifact_id}.required_files",
            )
            authoritative_files = _string_sequence(
                raw.get("authoritative_files", ()),
                label=f"{artifact_id}.authoritative_files",
            )
            forbidden_reuse = _string_sequence(
                raw.get("forbidden_reuse", ()),
                label=f"{artifact_id}.forbidden_reuse",
            )
            expected_file_hashes = _parse_expected_file_hashes(
                raw.get("expected_file_hashes", {}),
                artifact_id=artifact_id,
            )
            may_feed_selection = raw.get("may_feed_deployable_selection")
            if may_feed_selection is not None and not isinstance(may_feed_selection, bool):
                raise WorkspaceError(
                    f"{artifact_id}: may_feed_deployable_selection must be a boolean or null"
                )
            may_feed_recipe_selection = raw.get("may_feed_recipe_selection")
            if may_feed_recipe_selection is not None and not isinstance(
                may_feed_recipe_selection, bool
            ):
                raise WorkspaceError(
                    f"{artifact_id}: may_feed_recipe_selection must be a boolean or null"
                )
            parsed[artifact_id] = ArtifactEntry(
                artifact_id=artifact_id,
                stage=str(raw.get("stage", "")),
                physical_path=None if raw.get("physical_path") in (None, "") else str(raw["physical_path"]),
                canonical_path=None if raw.get("canonical_path") in (None, "") else str(raw["canonical_path"]),
                migration=str(raw.get("migration", "pointer_only")),
                availability=str(raw.get("availability", "unknown")),
                evidence_label=str(raw.get("evidence_label", "TODO_VERIFY_ARTIFACT")),
                claim_scope=str(raw.get("claim_scope", "none")),
                semantic_identities={str(key): str(value) for key, value in identities.items()},
                required_files=required_files,
                authoritative_files=authoritative_files,
                expected_file_hashes=expected_file_hashes,
                forbidden_reuse=forbidden_reuse,
                may_feed_recipe_selection=may_feed_recipe_selection,
                may_feed_deployable_selection=may_feed_selection,
            )
        return parsed

    @staticmethod
    def _parse_experiments(registry: Mapping[str, Any]) -> dict[str, ExperimentEntry]:
        parsed: dict[str, ExperimentEntry] = {}
        for raw in registry.get("experiments", ()):
            if not isinstance(raw, Mapping):
                raise WorkspaceError("registry experiments must be mappings")
            experiment_id = str(raw.get("experiment_id", ""))
            if not experiment_id or experiment_id in parsed:
                raise WorkspaceError(f"missing or duplicate experiment_id: {experiment_id!r}")
            runner = raw.get("runner", {}) or {}
            if not isinstance(runner, Mapping):
                raise WorkspaceError(f"{experiment_id}: runner must be a mapping")
            environment = runner.get("environment", {}) or {}
            if not isinstance(environment, Mapping):
                raise WorkspaceError(f"{experiment_id}: runner.environment must be a mapping")
            raw_recovery_strategy = runner.get("run_recovery_strategy")
            if raw_recovery_strategy is not None and (
                not isinstance(raw_recovery_strategy, str)
                or not raw_recovery_strategy.strip()
            ):
                raise WorkspaceError(
                    f"{experiment_id}: runner.run_recovery_strategy must be a non-empty string"
                )
            try:
                preparation_authority_gate = validate_preparation_authority_gate_id(
                    runner.get("preparation_authority_gate")
                )
            except PreparationAuthorityError as exc:
                raise WorkspaceError(f"{experiment_id}: {exc}") from exc
            exceptions = raw.get("input_claim_scope_exceptions", {}) or {}
            if not isinstance(exceptions, Mapping):
                raise WorkspaceError(
                    f"{experiment_id}: input_claim_scope_exceptions must map artifact IDs to "
                    "protocol-review rationales"
                )
            parsed[experiment_id] = ExperimentEntry(
                experiment_id=experiment_id,
                stage=str(raw.get("stage", "")),
                status=str(raw.get("status", "blocked")),
                claim_scope=str(raw.get("claim_scope", "none")),
                output_artifact_id=str(raw.get("output_artifact_id", "")),
                config_path=None if raw.get("config_path") in (None, "") else str(raw["config_path"]),
                runner_argv=tuple(str(value) for value in runner.get("argv", ())),
                runner_env={str(key): str(value) for key, value in environment.items()},
                preparation_authority_gate=preparation_authority_gate,
                run_recovery_strategy=raw_recovery_strategy,
                input_artifact_ids=tuple(str(value) for value in raw.get("input_artifact_ids", ())),
                input_claim_scope_exceptions={
                    str(key): str(value) for key, value in exceptions.items()
                },
                notes=tuple(str(value) for value in raw.get("notes", ())),
            )
        return parsed


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WorkspaceError(f"{label} must be a sequence")
    rendered = tuple(str(item) for item in value)
    if any(not item for item in rendered):
        raise WorkspaceError(f"{label} may not contain empty values")
    if len(rendered) != len(set(rendered)):
        raise WorkspaceError(f"{label} may not contain duplicates")
    return rendered


def _parse_expected_file_hashes(
    value: Any,
    *,
    artifact_id: str,
) -> dict[str, FileHashExpectation]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping):
        raise WorkspaceError(f"{artifact_id}: expected_file_hashes must be a mapping")
    parsed: dict[str, FileHashExpectation] = {}
    for raw_path, raw_spec in value.items():
        relative = str(raw_path)
        if not relative:
            raise WorkspaceError(f"{artifact_id}: expected_file_hashes contains an empty path")
        if not isinstance(raw_spec, Mapping):
            raise WorkspaceError(
                f"{artifact_id}: expected_file_hashes[{relative!r}] must declare algorithm and digest"
            )
        algorithm_value = raw_spec.get("algorithm")
        digest_value = raw_spec.get("digest")
        if not isinstance(algorithm_value, str) or not isinstance(digest_value, str):
            raise WorkspaceError(
                f"{artifact_id}: expected_file_hashes[{relative!r}] algorithm and digest "
                "must be strings"
            )
        algorithm = algorithm_value.lower()
        digest = digest_value.lower()
        if algorithm not in ALLOWED_FILE_HASH_ALGORITHMS:
            raise WorkspaceError(
                f"{artifact_id}: unsupported file hash algorithm {algorithm!r} for {relative}; "
                f"allowed: {sorted(ALLOWED_FILE_HASH_ALGORITHMS)}"
            )
        expected_hex_length = hashlib.new(algorithm).digest_size * 2
        if len(digest) != expected_hex_length or re.fullmatch(r"[0-9a-f]+", digest) is None:
            raise WorkspaceError(
                f"{artifact_id}: invalid {algorithm} digest for {relative}; expected "
                f"{expected_hex_length} lowercase hexadecimal characters"
            )
        parsed[relative] = FileHashExpectation(algorithm=algorithm, digest=digest)
    return parsed


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise WorkspaceError(f"Missing MIDOG++ workspace file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise WorkspaceError(f"Workspace YAML must be a mapping: {path}")
    return payload


def _discover_repo_root(explicit: str | Path | None) -> Path:
    """Locate the checkout that owns the declarative MIDOG++ workspace.

    The installed package contains orchestration code, while registry and
    protocol YAML remain repository-owned experiment definitions. An explicit
    root wins; otherwise ``MIDOGPP_REPO_ROOT``, the current working directory,
    and source-checkout ancestors are checked in that order.
    """

    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        marker = root / REPOSITORY_MARKER
        if not marker.is_file():
            raise WorkspaceError(f"MIDOG++ repository root lacks {REPOSITORY_MARKER}: {root}")
        return root

    candidates: list[Path] = []
    environment_root = os.environ.get("MIDOGPP_REPO_ROOT")
    if environment_root:
        candidates.append(Path(environment_root).expanduser())
    candidates.extend((Path.cwd(), *Path.cwd().parents))
    module_path = Path(__file__).resolve()
    candidates.extend(module_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / REPOSITORY_MARKER).is_file():
            return root
    raise WorkspaceError(
        "Could not locate the MIDOG++ repository. Run inside the checkout, pass repo_root, "
        "or set MIDOGPP_REPO_ROOT."
    )


def _write_checked(path: Path, content: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content and not force:
        raise WorkspaceError(f"Refusing to overwrite changed run snapshot without --force: {path}")
    path.write_text(content, encoding="utf-8")


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "repository_revision": "unknown",
            "repository_dirty": None,
            "repository_status_hash": "unknown",
        }
    status_text = status.stdout
    return {
        "repository_revision": revision.stdout.strip() or "unknown",
        "repository_dirty": bool(status_text.strip()),
        "repository_status_hash": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
    }
