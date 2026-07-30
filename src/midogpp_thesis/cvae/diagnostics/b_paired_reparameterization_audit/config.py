"""Frozen configuration for the canonical-B paired reparameterization audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.protocol import ProtocolError


CONFIG_SCHEMA = "midogpp_b_paired_reparameterization_audit_config_v1"
SNAPSHOT_BUILD_CONFIG_SCHEMA = "midogpp_b_paired_reparameterization_snapshot_build_v1"
STAGE = "90_oracles_and_diagnostics"
EVIDENCE_LABEL = "AUDIT_ONLY"
CLAIM_SCOPE = "diagnostic_only"
SNAPSHOT_ARTIFACT_ID = (
    "midogpp_stage90_uniform_b_paired_reparameterization_snapshot_v1"
)
SNAPSHOT_INPUT_URI = f"artifact://{SNAPSHOT_ARTIFACT_ID}"
CONTRACT_ARTIFACT_ID = "midogpp_dataset_contract_annotation_patch_v1"
FEATURE_CACHE_ARTIFACT_ID = (
    "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
)

AUDIT_CENTERS = ("2", "5", "6", "9")
INITIALIZATION_SEEDS = (17, 42, 101)
LEGACY_CANDIDATE = "legacy_v2_seed_specific_one_epsilon"
FIXED_ONE_EPSILON_CANDIDATE = "fold_fixed_one_epsilon"
FIXED_ANTITHETIC_CANDIDATE = "fold_fixed_antithetic"
AUDIT_CANDIDATES = (
    LEGACY_CANDIDATE,
    FIXED_ONE_EPSILON_CANDIDATE,
    FIXED_ANTITHETIC_CANDIDATE,
)
CONTROLLED_CANDIDATES = (
    FIXED_ONE_EPSILON_CANDIDATE,
    FIXED_ANTITHETIC_CANDIDATE,
)

CLAIM_FIREWALL_FIELDS = (
    "may_export_recipe_lock",
    "may_feed_stage20",
    "may_feed_expert_bank",
    "may_feed_generation",
    "may_feed_routing",
    "may_feed_composition",
    "may_feed_downstream",
    "may_feed_deployable_selection",
    "may_tune_or_select",
    "may_support_thesis_claim",
)


@dataclass(frozen=True)
class FrozenBRecipe:
    """Exact legacy-v2 canonical-B block frame and fixed-step training recipe."""

    expected_b_dim: int = 3840
    global_dim: int = 2560
    local_dim: int = 1280
    block_global_pca_dim: int = 96
    block_local_pca_dim: int = 32
    output_dim: int = 128
    pca_fit_scope: str = "source_fit_only"
    pca_svd_solver: str = "randomized"
    pca_random_state: int = 0
    pca_n_oversamples: int = 10
    pca_iterated_power: int = 4
    pca_whiten: bool = False
    post_fit_reweighting: bool = False
    validation_fraction: float = 0.20
    case_split_seed: int = 2718
    optimizer_steps: int = 1000
    batch_size: int = 128
    hidden_dim: int = 512
    latent_dim: int = 32
    num_hidden_layers: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    beta_final: float = 0.001
    kl_warmup_steps: int = 250
    gradient_clip_norm: float = 5.0
    objective: str = "stochastic_isotropic_beta_objective_step_normalized_v1"
    optimizer: str = "AdamW"
    batch_policy: str = "class_then_case_then_row_balanced_with_replacement"
    batch_class_quota: str = "exact_half_per_binary_class"
    batch_case_sampling: str = "uniform_with_replacement"
    batch_row_sampling: str = "uniform_with_replacement_within_case"
    batch_permutation: str = "deterministic_per_step"

    def __post_init__(self) -> None:
        fields = type(self).__dataclass_fields__
        if any(getattr(self, name) != field.default for name, field in fields.items()):
            raise ProtocolError(
                "Stage-90 audit recipe must exactly replay legacy-v2 B PCA96+32."
            )

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_b_paired_audit_frozen_recipe_v1",
            **asdict(self),
        }


@dataclass(frozen=True)
class ClaimFirewall:
    """All audit promotion and claim paths are permanently disabled."""

    may_export_recipe_lock: bool = False
    may_feed_stage20: bool = False
    may_feed_expert_bank: bool = False
    may_feed_generation: bool = False
    may_feed_routing: bool = False
    may_feed_composition: bool = False
    may_feed_downstream: bool = False
    may_feed_deployable_selection: bool = False
    may_tune_or_select: bool = False
    may_support_thesis_claim: bool = False

    def __post_init__(self) -> None:
        if any(asdict(self).values()):
            raise ProtocolError("Every Stage-90 audit claim-firewall flag must be false.")

    def to_payload(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionThresholds:
    """Predeclared diagnostic gates; none can promote a recipe or thesis claim."""

    mean_preservation_min: float = 0.80
    minimum_seed_mean_preservation: float = 0.75
    maximum_seed_mean_preservation_range: float = 0.05
    maximum_within_center_class_direction_seed_range: float = 0.15
    mean_bacc_delta_vs_fixed_min: float = -0.01
    mean_preservation_delta_vs_fixed_min: float = -0.02

    def __post_init__(self) -> None:
        fields = type(self).__dataclass_fields__
        if any(
            float(getattr(self, name)) != float(field.default)
            for name, field in fields.items()
        ):
            raise ProtocolError("Stage-90 audit decision thresholds must remain frozen.")

    def to_payload(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalLineage:
    predecessor_experiment: str
    predecessor_protocol_hash: str
    predecessor_bundle_hashes: Mapping[str, str]
    predecessor_root_provenance_only: str
    historical_paths_read: bool

    def __post_init__(self) -> None:
        if not self.predecessor_experiment or not self.predecessor_root_provenance_only:
            raise ProtocolError("Historical lineage identities must be explicit.")
        _validate_semantic_hash(
            self.predecessor_protocol_hash, "predecessor_protocol_hash"
        )
        if self.historical_paths_read:
            raise ProtocolError("Historical provenance paths may never be read.")
        if len(self.predecessor_bundle_hashes) != 9:
            raise ProtocolError("Historical lineage requires exactly nine bundle digests.")
        for digest in self.predecessor_bundle_hashes.values():
            _validate_sha256(str(digest), "historical bundle digest")

    def to_payload(self) -> dict[str, object]:
        return {
            "predecessor_experiment": self.predecessor_experiment,
            "predecessor_protocol_hash": self.predecessor_protocol_hash,
            "predecessor_bundle_hashes": dict(self.predecessor_bundle_hashes),
            "predecessor_root_provenance_only": self.predecessor_root_provenance_only,
            "historical_paths_read": self.historical_paths_read,
        }


@dataclass(frozen=True)
class LegacyExpectation:
    center: str
    training_seed: int
    device: str
    training_key_hash: str
    checkpoint_hash: str
    initialization_hash: str
    schedule_hash: str
    posterior_stream_hash: str
    fit_row_hash: str
    eval_row_hash: str
    frame_hash: str
    expected_decode_prediction_sha256: str
    expected_decode_metric: Mapping[str, int | float]

    def __post_init__(self) -> None:
        if self.center not in AUDIT_CENTERS or self.training_seed not in INITIALIZATION_SEEDS:
            raise ProtocolError("Legacy expectation coordinate is outside the audit.")
        if self.device not in {"cuda:0", "cuda:1"}:
            raise ProtocolError("Legacy replay device must be cuda:0 or cuda:1.")
        for value in (
            self.training_key_hash,
            self.schedule_hash,
            self.posterior_stream_hash,
            self.fit_row_hash,
            self.eval_row_hash,
            self.frame_hash,
        ):
            _validate_semantic_hash(value, "legacy semantic hash")
        for value in (
            self.checkpoint_hash,
            self.initialization_hash,
            self.expected_decode_prediction_sha256,
        ):
            _validate_sha256(value, "legacy SHA-256")
        required = {
            "bacc",
            "positive_recall",
            "specificity",
            "fn",
            "fp",
            "tn",
            "tp",
        }
        if set(self.expected_decode_metric) != required:
            raise ProtocolError("Legacy decode metric must contain the exact metric subset.")
        if not all(
            math.isfinite(float(self.expected_decode_metric[key]))
            for key in ("bacc", "positive_recall", "specificity")
        ):
            raise ProtocolError("Legacy decode metric rates must be finite.")
        if any(int(self.expected_decode_metric[key]) < 0 for key in ("fn", "fp", "tn", "tp")):
            raise ProtocolError("Legacy decode metric counts must be nonnegative.")

    @property
    def expected_decode_metric_sha256(self) -> str:
        return _canonical_sha256(dict(self.expected_decode_metric))

    def to_payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "training_seed": self.training_seed,
            "device": self.device,
            "training_key_hash": self.training_key_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "initialization_hash": self.initialization_hash,
            "schedule_hash": self.schedule_hash,
            "posterior_stream_hash": self.posterior_stream_hash,
            "fit_row_hash": self.fit_row_hash,
            "eval_row_hash": self.eval_row_hash,
            "frame_hash": self.frame_hash,
            "expected_decode_prediction_sha256": self.expected_decode_prediction_sha256,
            "expected_decode_metric": dict(self.expected_decode_metric),
            "expected_decode_metric_sha256": self.expected_decode_metric_sha256,
        }


@dataclass(frozen=True)
class SnapshotBuildConfig:
    """Only canonical contract/cache inputs plus inert historical digest strings."""

    name: str
    code_version: str
    artifact_root: str
    manifest_path: str
    b_feature_cache_path: str
    devices: tuple[str, ...]
    cpu_threads_per_worker: int
    historical_lineage: HistoricalLineage
    legacy_expectations: tuple[LegacyExpectation, ...]
    centers: tuple[str, ...] = AUDIT_CENTERS
    initialization_seeds: tuple[int, ...] = INITIALIZATION_SEEDS
    recipe: FrozenBRecipe = FrozenBRecipe()
    claim_firewall: ClaimFirewall = ClaimFirewall()

    def __post_init__(self) -> None:
        if not self.name or not self.code_version:
            raise ProtocolError("Snapshot-build name and code_version must be explicit.")
        _validate_output_location(self.artifact_root)
        _validate_artifact_file(
            self.manifest_path, CONTRACT_ARTIFACT_ID, "manifest.csv", "manifest_path"
        )
        _validate_artifact_file(
            self.b_feature_cache_path,
            FEATURE_CACHE_ARTIFACT_ID,
            "embeddings/train.pt",
            "b_feature_cache_path",
        )
        if (
            self.centers != AUDIT_CENTERS
            or self.initialization_seeds != INITIALIZATION_SEEDS
        ):
            raise ProtocolError("Snapshot-build center/seed panel drifted.")
        if self.devices != ("cuda:0", "cuda:1") or self.cpu_threads_per_worker != 1:
            raise ProtocolError("Snapshot-build device/thread policy drifted.")
        coordinates = [
            (record.center, record.training_seed) for record in self.legacy_expectations
        ]
        expected = {
            (center, seed)
            for center in AUDIT_CENTERS
            for seed in INITIALIZATION_SEEDS
        }
        if len(coordinates) != 12 or set(coordinates) != expected:
            raise ProtocolError("Snapshot builder requires exactly 12 legacy expectations.")
        if self.recipe != FrozenBRecipe() or self.claim_firewall != ClaimFirewall():
            raise ProtocolError("Snapshot-build protocol or claim identity drifted.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": SNAPSHOT_BUILD_CONFIG_SCHEMA,
            "name": self.name,
            "code_version": self.code_version,
            "artifact_root": self.artifact_root,
            "inputs": {
                "manifest_path": self.manifest_path,
                "b_feature_cache_path": self.b_feature_cache_path,
            },
            "run": {
                "centers": list(self.centers),
                "initialization_seeds": list(self.initialization_seeds),
                "devices": list(self.devices),
                "cpu_threads_per_worker": self.cpu_threads_per_worker,
            },
            "recipe": self.recipe.to_payload(),
            "historical_lineage": self.historical_lineage.to_payload(),
            "legacy_expectations": [
                record.to_payload() for record in self.legacy_expectations
            ],
            "claim_firewall": self.claim_firewall.to_payload(),
        }


@dataclass(frozen=True)
class AuditConfig:
    """Validated, immutable settings consumed by the snapshot builder and runner."""

    name: str
    code_version: str
    snapshot_artifact_id: str
    snapshot_root: str
    centers: tuple[str, ...] = AUDIT_CENTERS
    initialization_seeds: tuple[int, ...] = INITIALIZATION_SEEDS
    candidates: tuple[str, ...] = AUDIT_CANDIDATES
    stage: str = STAGE
    evidence_label: str = EVIDENCE_LABEL
    claim_scope: str = CLAIM_SCOPE
    recipe: FrozenBRecipe = FrozenBRecipe()
    decision_thresholds: DecisionThresholds = DecisionThresholds()
    claim_firewall: ClaimFirewall = ClaimFirewall()

    def __post_init__(self) -> None:
        if not self.name or not self.code_version:
            raise ProtocolError("Audit name and code_version must be explicit.")
        if self.snapshot_artifact_id != SNAPSHOT_ARTIFACT_ID:
            raise ProtocolError(
                "Audit input must retain the canonical Stage-90 snapshot artifact ID."
            )
        _validate_artifact_location(
            self.snapshot_root,
            artifact_id=SNAPSHOT_ARTIFACT_ID,
            field="snapshot_root",
        )
        if self.centers != AUDIT_CENTERS:
            raise ProtocolError("Audit centers are frozen to 2,5,6,9.")
        if self.initialization_seeds != INITIALIZATION_SEEDS:
            raise ProtocolError("Audit initialization seeds are frozen to 17,42,101.")
        if self.candidates != AUDIT_CANDIDATES:
            raise ProtocolError("Audit candidates or their canonical order drifted.")
        if (
            self.stage != STAGE
            or self.evidence_label != EVIDENCE_LABEL
            or self.claim_scope != CLAIM_SCOPE
        ):
            raise ProtocolError("Stage-90 AUDIT_ONLY claim identity drifted.")
        if (
            self.recipe != FrozenBRecipe()
            or self.decision_thresholds != DecisionThresholds()
            or self.claim_firewall != ClaimFirewall()
        ):
            raise ProtocolError("Audit recipe or claim firewall drifted.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_SCHEMA,
            "name": self.name,
            "code_version": self.code_version,
            "snapshot_artifact_id": self.snapshot_artifact_id,
            "snapshot_root": self.snapshot_root,
            "stage": self.stage,
            "evidence_label": self.evidence_label,
            "claim_scope": self.claim_scope,
            "centers": list(self.centers),
            "initialization_seeds": list(self.initialization_seeds),
            "candidates": list(self.candidates),
            "recipe": self.recipe.to_payload(),
            "decision_thresholds": self.decision_thresholds.to_payload(),
            "claim_firewall": self.claim_firewall.to_payload(),
        }


def audit_config_from_mapping(payload: Mapping[str, object]) -> AuditConfig:
    """Build a fail-closed config from a resolved JSON/YAML-like mapping."""

    if str(payload.get("schema_version", "")) != CONFIG_SCHEMA:
        raise ProtocolError(f"Audit config schema must be {CONFIG_SCHEMA!r}.")
    recipe = _recipe_from_mapping(_mapping(payload, "recipe"))
    firewall_payload = _mapping(payload, "claim_firewall")
    if set(firewall_payload) != set(CLAIM_FIREWALL_FIELDS):
        raise ProtocolError(
            "Audit claim_firewall must contain exactly the frozen false flags."
        )
    try:
        thresholds_payload = _mapping(payload, "decision_thresholds")
        if set(thresholds_payload) != set(DecisionThresholds.__dataclass_fields__):
            raise ProtocolError("Audit decision thresholds must be explicit and exact.")
        firewall = ClaimFirewall(
            **{key: _strict_bool(firewall_payload[key], key) for key in CLAIM_FIREWALL_FIELDS}
        )
        return AuditConfig(
            name=str(payload.get("name", "")),
            code_version=str(payload.get("code_version", "")),
            snapshot_artifact_id=str(payload.get("snapshot_artifact_id", "")),
            snapshot_root=str(
                payload.get("snapshot_root", payload.get("snapshot_input_uri", ""))
            ),
            centers=tuple(str(value) for value in payload.get("centers", ())),
            initialization_seeds=tuple(
                int(value) for value in payload.get("initialization_seeds", ())
            ),
            candidates=tuple(str(value) for value in payload.get("candidates", ())),
            stage=str(payload.get("stage", "")),
            evidence_label=str(payload.get("evidence_label", "")),
            claim_scope=str(payload.get("claim_scope", "")),
            recipe=recipe,
            decision_thresholds=DecisionThresholds(
                **{key: float(value) for key, value in thresholds_payload.items()}
            ),
            claim_firewall=firewall,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Audit config fields are malformed.") from exc


def load_audit_config(path: str | Path) -> AuditConfig:
    """Load a resolved YAML/JSON config without importing a training framework."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("YAML audit configs require PyYAML.") from exc
            payload = yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise ProtocolError(f"Cannot read Stage-90 audit config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError("Audit config root must be a mapping.")
    return audit_config_from_mapping(payload)


def snapshot_build_config_from_mapping(
    payload: Mapping[str, object],
) -> SnapshotBuildConfig:
    """Build the portable snapshot config from a resolved YAML-like mapping."""

    if str(payload.get("schema_version", "")) != SNAPSHOT_BUILD_CONFIG_SCHEMA:
        raise ProtocolError(
            f"Snapshot-build schema must be {SNAPSHOT_BUILD_CONFIG_SCHEMA!r}."
        )
    recipe = _recipe_from_mapping(_mapping(payload, "recipe"))
    firewall_payload = _mapping(payload, "claim_firewall")
    if set(firewall_payload) != set(CLAIM_FIREWALL_FIELDS):
        raise ProtocolError(
            "Snapshot-build claim_firewall must contain exactly the frozen false flags."
        )
    inputs = _mapping(payload, "inputs")
    run = _mapping(payload, "run")
    lineage = _mapping(payload, "historical_lineage")
    records_value = payload.get("legacy_expectations")
    if not isinstance(records_value, list):
        raise ProtocolError("legacy_expectations must be a list.")
    try:
        records = tuple(
            LegacyExpectation(
                center=str(record.get("center", "")),
                training_seed=int(record.get("training_seed", -1)),
                device=str(record.get("device", "")),
                training_key_hash=str(record.get("training_key_hash", "")),
                checkpoint_hash=str(record.get("checkpoint_hash", "")),
                initialization_hash=str(record.get("initialization_hash", "")),
                schedule_hash=str(record.get("schedule_hash", "")),
                posterior_stream_hash=str(record.get("posterior_stream_hash", "")),
                fit_row_hash=str(record.get("fit_row_hash", "")),
                eval_row_hash=str(record.get("eval_row_hash", "")),
                frame_hash=str(record.get("frame_hash", "")),
                expected_decode_prediction_sha256=str(
                    record.get("expected_decode_prediction_sha256", "")
                ),
                expected_decode_metric=dict(
                    _mapping(record, "expected_decode_metric")
                ),
            )
            for record in records_value
            if isinstance(record, Mapping)
        )
        return SnapshotBuildConfig(
            name=str(payload.get("name", "")),
            code_version=str(payload.get("code_version", "")),
            artifact_root=str(payload.get("artifact_root", "")),
            manifest_path=str(inputs.get("manifest_path", "")),
            b_feature_cache_path=str(inputs.get("b_feature_cache_path", "")),
            centers=tuple(str(value) for value in run.get("centers", ())),
            initialization_seeds=tuple(
                int(value) for value in run.get("initialization_seeds", ())
            ),
            devices=tuple(str(value) for value in run.get("devices", ())),
            cpu_threads_per_worker=int(run.get("cpu_threads_per_worker", -1)),
            historical_lineage=HistoricalLineage(
                predecessor_experiment=str(
                    lineage.get("predecessor_experiment", "")
                ),
                predecessor_protocol_hash=str(
                    lineage.get("predecessor_protocol_hash", "")
                ),
                predecessor_bundle_hashes={
                    str(key): str(value)
                    for key, value in _mapping(
                        lineage, "predecessor_bundle_hashes"
                    ).items()
                },
                predecessor_root_provenance_only=str(
                    lineage.get("predecessor_root_provenance_only", "")
                ),
                historical_paths_read=_strict_bool(
                    lineage.get("historical_paths_read"), "historical_paths_read"
                ),
            ),
            legacy_expectations=records,
            recipe=recipe,
            claim_firewall=ClaimFirewall(
                **{
                    key: _strict_bool(firewall_payload[key], key)
                    for key in CLAIM_FIREWALL_FIELDS
                }
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Snapshot-build config fields are malformed.") from exc


def load_snapshot_build_config(path: str | Path) -> SnapshotBuildConfig:
    """Load a resolved YAML/JSON snapshot-build config."""

    payload = _load_mapping_file(path, label="Stage-90 snapshot-build config")
    return snapshot_build_config_from_mapping(payload)


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Audit config requires mapping section {key!r}.")
    return value


def _strict_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"Audit claim-firewall field {key!r} must be boolean.")
    return value


def _recipe_from_mapping(payload: Mapping[str, object]) -> FrozenBRecipe:
    values = dict(payload)
    if values.pop("schema_version", None) != "midogpp_b_paired_audit_frozen_recipe_v1":
        raise ProtocolError("Audit frozen recipe schema is missing or invalid.")
    if set(values) != set(FrozenBRecipe.__dataclass_fields__):
        raise ProtocolError("Audit frozen recipe fields must be explicit and exact.")
    try:
        return FrozenBRecipe(**values)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Audit recipe fields are malformed.") from exc


def _validate_artifact_location(value: str, *, artifact_id: str, field: str) -> None:
    location = str(value)
    if not location:
        raise ProtocolError(f"Audit {field} must be explicit.")
    if location.startswith("artifact://"):
        if location.rstrip("/") != f"artifact://{artifact_id}":
            raise ProtocolError(f"Audit {field} references the wrong artifact ID.")
        return
    if not Path(location).is_absolute():
        raise ProtocolError(f"Resolved audit {field} must be an absolute path.")


def _validate_artifact_file(
    value: str, artifact_id: str, relative_path: str, field: str
) -> None:
    location = str(value)
    expected = f"artifact://{artifact_id}/{relative_path}"
    if location.startswith("artifact://"):
        if location != expected:
            raise ProtocolError(f"Snapshot-build {field} references the wrong input.")
        return
    if not Path(location).is_absolute():
        raise ProtocolError(f"Resolved snapshot-build {field} must be absolute.")


def _validate_output_location(value: str) -> None:
    location = str(value)
    if location.startswith("output://"):
        if location != f"output://{SNAPSHOT_ARTIFACT_ID}":
            raise ProtocolError("Snapshot-build artifact_root references the wrong output.")
        return
    if not Path(location).is_absolute():
        raise ProtocolError("Resolved snapshot-build artifact_root must be absolute.")


def _nullable_string(value: object) -> str | None:
    return None if value is None else str(value)


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError(f"{label} must be a lowercase full SHA-256 digest.")


def _validate_semantic_hash(value: str, label: str) -> None:
    if len(value) != 16 or any(character not in "0123456789abcdef" for character in value):
        raise ProtocolError(f"{label} must be a canonical 16-hex semantic hash.")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_mapping_file(path: str | Path, *, label: str) -> Mapping[str, object]:
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("YAML audit configs require PyYAML.") from exc
            payload = yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise ProtocolError(f"Cannot read {label}: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"{label} root must be a mapping.")
    return payload
