"""Build the portable, hash-promoted input snapshot for the Stage-90 B audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.block_frame import fit_pilot_frame
from midogpp_thesis.cvae.case_split import deterministic_case_holdout
from midogpp_thesis.cvae.fixed_step_training import StepTrainingSpec
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.schedules import (
    BalancedSchedule,
    build_balanced_schedule,
    build_fold_fixed_schedule,
)
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    load_midogpp_real_feature_frame,
)

from .artifacts import file_sha256, write_json
from .config import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    LEGACY_CANDIDATE,
    LegacyExpectation,
    SnapshotBuildConfig,
    load_snapshot_build_config,
)
from .entrypoint import (
    SNAPSHOT_CANONICAL_RELATIVE,
    SNAPSHOT_EXPERIMENT_ID,
    assert_workspace_prepared_entrypoint,
)
from .protocol import build_key_record
from .run_lock import exclusive_artifact_lock
from .snapshot import (
    HASH_PROMOTED,
    ArrayBinding,
    CenterPreparedData,
    ContentEntry,
    PreparedPartition,
    build_snapshot,
    snapshot_manifest_hash,
    validate_snapshot,
)
from .snapshot_io import (
    PREPARED_ARRAY_NAMES,
    canonical_mapping_hash,
    save_array,
    save_prepared_bundle,
    save_schedule,
)
from .trace import trace_content_hash


_BLOCK_ARM = "b_block_pca96_32"


def build_snapshot_from_config(
    config_path: str | Path,
    *,
    artifact_root: Path,
) -> Path:
    """Materialize the only input artifact accepted by the paired audit."""

    config = load_snapshot_build_config(config_path)
    root = Path(artifact_root).resolve()
    assert_workspace_prepared_entrypoint(
        resolved_config_path=config_path,
        artifact_root=root,
        experiment_id=SNAPSHOT_EXPERIMENT_ID,
        canonical_relative=SNAPSHOT_CANONICAL_RELATIVE,
        input_artifact_ids=(
            "midogpp_dataset_contract_annotation_patch_v1",
            "midogpp_virchow2_uniform_b_canonical_train_cache_seed42",
        ),
        expected_input_members={
            "midogpp_dataset_contract_annotation_patch_v1": config.manifest_path,
            "midogpp_virchow2_uniform_b_canonical_train_cache_seed42": (
                config.b_feature_cache_path
            ),
        },
    )
    _assert_output_root(config, root)
    with exclusive_artifact_lock(root, purpose="stage90_snapshot"):
        return _build_snapshot(config=config, root=root)


def _build_snapshot(*, config: SnapshotBuildConfig, root: Path) -> Path:
    _ensure_snapshot_directories(root)
    started = perf_counter()
    _write_run_state(root, status="BUILDING", completed_centers=0)

    frame = load_midogpp_real_feature_frame(
        manifest_path=Path(config.manifest_path),
        feature_cache_path=Path(config.b_feature_cache_path),
        expected_feature_dim=config.recipe.expected_b_dim,
        allow_excluded_center_omission=True,
    )
    expectations = {
        (item.center, item.training_seed): item
        for item in config.legacy_expectations
    }
    prepared_centers: list[CenterPreparedData] = []
    content_entries: list[ContentEntry] = []
    prepared_arrays: dict[str, Mapping[str, object]] = {}
    frame_hashes: dict[str, str] = {}
    legacy_schedules: dict[tuple[str, int], _StoredSchedule] = {}
    controlled_schedules: dict[str, _StoredSchedule] = {}
    legacy_traces: dict[tuple[str, int], _StoredTrace] = {}
    controlled_traces: dict[str, _StoredTrace] = {}

    for completed, center in enumerate(config.centers, start=1):
        (
            prepared,
            arrays,
            frame_hash,
            center_entries,
        ) = _prepare_center(
            root=root,
            center=center,
            source_frame=frame,
            config=config,
            expectations=expectations,
        )
        prepared_centers.append(prepared)
        prepared_arrays[center] = arrays
        frame_hashes[center] = frame_hash
        content_entries.extend(center_entries)

        for seed in config.initialization_seeds:
            expected = expectations[(center, seed)]
            stored = _build_legacy_schedule(
                root=root,
                center=center,
                training_seed=seed,
                arrays=arrays,
                protocol_hash=config.historical_lineage.predecessor_protocol_hash,
                expected=expected,
                config=config,
                frame_hash=frame_hash,
            )
            legacy_schedules[(center, seed)] = stored
            content_entries.append(stored.entry)
            trace = _build_legacy_trace(
                root=root,
                center=center,
                training_seed=seed,
                schedule=stored.schedule,
                expected=expected,
                protocol_hash=config.historical_lineage.predecessor_protocol_hash,
                recipe=config.recipe,
            )
            legacy_traces[(center, seed)] = trace
            content_entries.append(trace.entry)

        controlled = _build_controlled_schedule(
            root=root,
            center=center,
            arrays=arrays,
            fit_row_hash=expectations[(center, config.initialization_seeds[0])].fit_row_hash,
            recipe_hash=config.recipe.hash,
            code_version=config.code_version,
            recipe=config.recipe,
        )
        controlled_schedules[center] = controlled
        content_entries.append(controlled.entry)
        controlled_trace = _build_controlled_trace(
            root=root,
            center=center,
            frame_hash=frame_hash,
            schedule=controlled,
            config=config,
        )
        controlled_traces[center] = controlled_trace
        content_entries.append(controlled_trace.entry)
        _write_run_state(root, status="BUILDING", completed_centers=completed)

    historical_paths = (
        config.historical_lineage.predecessor_root_provenance_only,
    )
    manifest_hash = snapshot_manifest_hash(
        config_hash=config.hash,
        protocol_hash=config.historical_lineage.predecessor_protocol_hash,
        dataset_id="midogpp",
        feature_frame="canonical_b_block_pca96_32_v1",
        domain_axis="center",
        prepared_centers=prepared_centers,
        historical_paths=historical_paths,
    )
    keys = []
    for center in config.centers:
        prepared = next(item for item in prepared_centers if item.center == center)
        for seed in config.initialization_seeds:
            expected = expectations[(center, seed)]
            for candidate in AUDIT_CANDIDATES:
                if candidate == LEGACY_CANDIDATE:
                    schedule = legacy_schedules[(center, seed)]
                    trace = legacy_traces[(center, seed)]
                else:
                    schedule = controlled_schedules[center]
                    trace = controlled_traces[center]
                keys.append(
                    build_key_record(
                        center=center,
                        initialization_seed=seed,
                        execution_device=expected.device,
                        candidate=candidate,
                        prepared_relpath=prepared.prepared_bundle.relative_path,
                        prepared_sha256=prepared.prepared_bundle.sha256,
                        prepared_content_hash=prepared.prepared_bundle.content_hash,
                        schedule_relpath=schedule.entry.relative_path,
                        schedule_sha256=schedule.entry.sha256,
                        schedule_content_hash=schedule.entry.content_hash,
                        epsilon_trace_relpath=trace.entry.relative_path,
                        epsilon_trace_sha256=trace.entry.sha256,
                        epsilon_trace_content_hash=trace.entry.content_hash,
                        snapshot_manifest_hash=manifest_hash,
                        legacy_expected_checkpoint_hash=(
                            expected.checkpoint_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_expected_prediction_hash=(
                            expected.expected_decode_prediction_sha256
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_expected_metric_hash=(
                            expected.expected_decode_metric_sha256
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_expected_initialization_hash=(
                            expected.initialization_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_historical_training_key_hash=(
                            expected.training_key_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_historical_schedule_hash=(
                            expected.schedule_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_historical_posterior_stream_hash=(
                            expected.posterior_stream_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_historical_frame_hash=(
                            expected.frame_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_historical_fit_row_hash=(
                            expected.fit_row_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_historical_eval_row_hash=(
                            expected.eval_row_hash
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                        legacy_expected_decode_metric=(
                            expected.expected_decode_metric
                            if candidate == LEGACY_CANDIDATE
                            else None
                        ),
                    )
                )
    snapshot = build_snapshot(
        publication_state=HASH_PROMOTED,
        config_hash=config.hash,
        protocol_hash=config.historical_lineage.predecessor_protocol_hash,
        dataset_id="midogpp",
        feature_frame="canonical_b_block_pca96_32_v1",
        domain_axis="center",
        prepared_centers=prepared_centers,
        keys=keys,
        content_index=content_entries,
        historical_paths=historical_paths,
    )
    validate_snapshot(snapshot, artifact_root=root, require_hash_promoted=True)
    write_json(root / "manifests/snapshot_manifest.json", snapshot.to_payload())
    write_json(
        root / "manifests/key_inventory.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_key_inventory_v1",
            "snapshot_hash": snapshot.snapshot_hash,
            "manifest_hash": snapshot.manifest_hash,
            "key_inventory_hash": snapshot.key_inventory_hash,
            "record_count": len(snapshot.keys),
            "records": [record.to_payload() for record in snapshot.keys],
        },
    )
    write_json(root / "manifests/content_index.json", snapshot.content_index_payload())
    write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_snapshot_leakage_v1",
            "status": "PASS",
            "fit_eval_case_disjoint_by_center": True,
            "fit_labels_role": "source_fit_only",
            "eval_labels_role": "final_diagnostic_scoring_only",
            "eval_labels_used_for_training_or_selection": False,
            "historical_paths_read": False,
            "historical_paths_are_provenance_strings_only": True,
            "selection_used_target_eval_artifacts": False,
            "claim_scope": "diagnostic_only",
            **config.claim_firewall.to_payload(),
        },
    )
    validation_report = {
        "schema_version": "midogpp_b_paired_reparameterization_snapshot_validation_v1",
        "status": "PASS",
        "publication_state": snapshot.publication_state,
        "snapshot_hash": snapshot.snapshot_hash,
        "manifest_hash": snapshot.manifest_hash,
        "key_inventory_hash": snapshot.key_inventory_hash,
        "key_count": len(snapshot.keys),
        "legacy_key_count": sum(record.is_legacy for record in snapshot.keys),
        "controlled_key_count": sum(not record.is_legacy for record in snapshot.keys),
        "controlled_pair_count": len(
            {record.pair_id for record in snapshot.keys if not record.is_legacy}
        ),
        "center_count": len(snapshot.prepared_centers),
        "content_file_count": len(snapshot.content_index),
        "historical_paths_read": False,
        "claim_scope": "diagnostic_only",
        **config.claim_firewall.to_payload(),
    }
    write_json(root / "reports/validation_report.json", validation_report)
    _write_run_state(
        root,
        status="COMPLETE",
        completed_centers=len(AUDIT_CENTERS),
        snapshot_hash=snapshot.snapshot_hash,
        wall_seconds=perf_counter() - started,
    )
    return root


class _StoredSchedule:
    def __init__(
        self,
        *,
        schedule: BalancedSchedule,
        seed: int,
        entry: ContentEntry,
    ) -> None:
        self.schedule = schedule
        self.seed = int(seed)
        self.entry = entry


class _StoredTrace:
    def __init__(self, *, entry: ContentEntry) -> None:
        self.entry = entry


def _prepare_center(
    *,
    root: Path,
    center: str,
    source_frame: object,
    config: SnapshotBuildConfig,
    expectations: Mapping[tuple[str, int], LegacyExpectation],
) -> tuple[CenterPreparedData, Mapping[str, object], str, list[ContentEntry]]:
    import numpy as np

    rows = source_frame.rows
    center_indices = [
        index for index, row in enumerate(rows) if str(row.center) == str(center)
    ]
    if not center_indices:
        raise ProtocolError(f"Canonical-B cache has no rows for center {center}.")
    labels = [rows[index].label for index in center_indices]
    cases = [rows[index].case_id for index in center_indices]
    holdout = deterministic_case_holdout(
        cases,
        labels,
        validation_fraction=config.recipe.validation_fraction,
        seed=config.recipe.case_split_seed,
    )
    absolute_fit = [center_indices[index] for index in holdout.fit_indices]
    absolute_eval = [center_indices[index] for index in holdout.eval_indices]
    fit_sample_ids = [rows[index].sample_id for index in absolute_fit]
    eval_sample_ids = [rows[index].sample_id for index in absolute_eval]
    legacy_fit_hash = stable_hash(fit_sample_ids)
    legacy_eval_hash = stable_hash(eval_sample_ids)
    for seed in config.initialization_seeds:
        expected = expectations[(center, seed)]
        if (
            expected.fit_row_hash != legacy_fit_hash
            or expected.eval_row_hash != legacy_eval_hash
        ):
            raise ProtocolError(
                f"Canonical snapshot row inventory does not replay legacy center {center}."
            )
    source_embeddings = np.asarray(source_frame.embeddings)
    fitted_frame = fit_pilot_frame(
        _BLOCK_ARM,
        source_embeddings[absolute_fit],
        fit_sample_hash=legacy_fit_hash,
    )
    for seed in config.initialization_seeds:
        if expectations[(center, seed)].frame_hash != fitted_frame.state_hash:
            raise ProtocolError(
                f"Canonical snapshot PCA frame does not replay legacy center {center}."
            )
    arrays: dict[str, object] = {
        "x_fit": fitted_frame.transform(source_embeddings[absolute_fit]),
        "y_fit": np.asarray([rows[index].label for index in absolute_fit], dtype="<i8"),
        "case_fit": np.asarray(
            [rows[index].case_id for index in absolute_fit], dtype=str
        ),
        "sample_fit": np.asarray(fit_sample_ids, dtype=str),
        "x_eval": fitted_frame.transform(source_embeddings[absolute_eval]),
        "y_eval": np.asarray([rows[index].label for index in absolute_eval], dtype="<i8"),
        "case_eval": np.asarray(
            [rows[index].case_id for index in absolute_eval], dtype=str
        ),
        "sample_eval": np.asarray(eval_sample_ids, dtype=str),
    }
    bundle_path = root / "prepared" / center / "arrays.npz"
    bundle_content_hash = save_prepared_bundle(bundle_path, arrays)
    bundle_entry = _content_entry(
        root,
        bundle_path,
        content_hash=bundle_content_hash,
        role="prepared_center_bundle",
    )
    entries = [bundle_entry]
    bindings: dict[str, ArrayBinding] = {}
    roles = {
        "x_fit": "source_fit_features",
        "y_fit": "source_fit_only",
        "case_fit": "source_fit_case_ids",
        "sample_fit": "source_fit_sample_ids",
        "x_eval": "diagnostic_eval_features",
        "y_eval": "final_diagnostic_scoring_only",
        "case_eval": "diagnostic_eval_case_ids",
        "sample_eval": "diagnostic_eval_sample_ids",
    }
    for name in PREPARED_ARRAY_NAMES:
        path = root / "prepared" / center / f"{name}.npy"
        content_hash = save_array(path, arrays[name])
        entry = _content_entry(
            root,
            path,
            content_hash=content_hash,
            role=roles[name],
        )
        entries.append(entry)
        value = np.asarray(arrays[name])
        bindings[name] = ArrayBinding(
            relative_path=entry.relative_path,
            sha256=entry.sha256,
            content_hash=entry.content_hash,
            dtype=value.dtype.str,
            shape=tuple(int(item) for item in value.shape),
            role=roles[name],
        )
    fit = _partition(
        arrays=arrays,
        bindings=bindings,
        suffix="fit",
        cases=holdout.fit_cases,
    )
    evaluation = _partition(
        arrays=arrays,
        bindings=bindings,
        suffix="eval",
        cases=holdout.eval_cases,
    )
    return (
        CenterPreparedData(
            center=center,
            prepared_bundle=bundle_entry,
            fit=fit,
            evaluation=evaluation,
        ),
        arrays,
        fitted_frame.state_hash,
        entries,
    )


def _partition(
    *,
    arrays: Mapping[str, object],
    bindings: Mapping[str, ArrayBinding],
    suffix: str,
    cases: Sequence[str],
) -> PreparedPartition:
    import numpy as np

    samples = [str(value) for value in np.asarray(arrays[f"sample_{suffix}"]).tolist()]
    row_cases = [str(value) for value in np.asarray(arrays[f"case_{suffix}"]).tolist()]
    labels = [int(value) for value in np.asarray(arrays[f"y_{suffix}"]).tolist()]
    row_hash = canonical_mapping_hash(
        {
            "schema_version": "midogpp_b_prepared_row_inventory_v1",
            "rows": [
                {"sample_id": sample, "case_id": case, "label": label}
                for sample, case, label in zip(samples, row_cases, labels, strict=True)
            ],
        }
    )
    return PreparedPartition(
        features=bindings[f"x_{suffix}"],
        sample_ids=bindings[f"sample_{suffix}"],
        case_ids=bindings[f"case_{suffix}"],
        class_labels=bindings[f"y_{suffix}"],
        case_id_inventory=tuple(str(value) for value in cases),
        row_inventory_hash=row_hash,
        sample_id_inventory_hash=canonical_mapping_hash(
            {"schema_version": "midogpp_sample_inventory_v1", "sample_ids": samples}
        ),
        case_id_inventory_hash=canonical_mapping_hash(
            {
                "schema_version": "midogpp_case_inventory_v1",
                "case_ids": [str(value) for value in cases],
            }
        ),
        row_count=len(samples),
        sample_count=len(samples),
        case_count=len(cases),
    )


def _build_legacy_schedule(
    *,
    root: Path,
    center: str,
    training_seed: int,
    arrays: Mapping[str, object],
    protocol_hash: str,
    expected: LegacyExpectation,
    config: SnapshotBuildConfig,
    frame_hash: str,
) -> _StoredSchedule:
    seed = _derived_seed(protocol_hash, center, training_seed, "case_class_schedule")
    schedule = build_balanced_schedule(
        arrays["y_fit"],
        arrays["case_fit"],
        arrays["sample_fit"],
        steps=config.recipe.optimizer_steps,
        batch_size=config.recipe.batch_size,
        seed=seed,
    )
    if schedule.stream_hash != expected.schedule_hash:
        raise ProtocolError(
            f"Legacy schedule replay failed for center={center}, seed={training_seed}."
        )
    pairing_key = _legacy_pairing_key(
        protocol_hash=protocol_hash,
        center=center,
        training_seed=training_seed,
        schedule_hash=schedule.stream_hash,
    )
    spec = _training_spec(config)
    historical_training_key = stable_hash(
        {
            "schema_version": "midogpp_b_adaptation_training_key_v1",
            "protocol_hash": protocol_hash,
            "center": center,
            "arm": _BLOCK_ARM,
            "training_seed": training_seed,
            "fit_row_hash": expected.fit_row_hash,
            "frame_hash": frame_hash,
            "schedule_hash": schedule.stream_hash,
            "training_spec_hash": spec.hash,
        }
    )
    if historical_training_key != expected.training_key_hash:
        raise ProtocolError(
            f"Legacy training key does not replay for center={center}, seed={training_seed}."
        )
    posterior_hash = _posterior_stream_hash(
        pairing_key=pairing_key,
        schedule_hash=schedule.stream_hash,
        optimizer_steps=config.recipe.optimizer_steps,
    )
    if posterior_hash != expected.posterior_stream_hash:
        raise ProtocolError(
            f"Legacy posterior stream does not replay for center={center}, seed={training_seed}."
        )
    path = root / "schedules" / "legacy" / f"center_{center}_seed_{training_seed}.npz"
    from .snapshot_io import save_schedule

    content_hash = save_schedule(path, schedule, seed=seed)
    return _StoredSchedule(
        schedule=schedule,
        seed=seed,
        entry=_content_entry(
            root,
            path,
            content_hash=content_hash,
            role="legacy_seed_specific_schedule",
        ),
    )


def _build_controlled_schedule(
    *,
    root: Path,
    center: str,
    arrays: Mapping[str, object],
    fit_row_hash: str,
    recipe_hash: str,
    code_version: str,
    recipe: object,
) -> _StoredSchedule:
    recipe_version = f"{code_version}:{recipe_hash}"
    schedule = build_fold_fixed_schedule(
        arrays["y_fit"],
        arrays["case_fit"],
        arrays["sample_fit"],
        steps=recipe.optimizer_steps,
        batch_size=recipe.batch_size,
        center=center,
        fit_row_hash=fit_row_hash,
        recipe_version=recipe_version,
    )
    key = stable_hash(
        {
            "schema_version": "midogpp_b_fold_fixed_schedule_v1",
            "center": center,
            "fit_row_hash": fit_row_hash,
            "recipe_version": recipe_version,
        }
    )
    seed = int(key[:16], 16) % (2**31 - 1)
    path = root / "schedules" / "controlled" / f"center_{center}.npz"
    content_hash = save_schedule(path, schedule, seed=seed)
    return _StoredSchedule(
        schedule=schedule,
        seed=seed,
        entry=_content_entry(
            root,
            path,
            content_hash=content_hash,
            role="fold_fixed_schedule",
        ),
    )


def _build_legacy_trace(
    *,
    root: Path,
    center: str,
    training_seed: int,
    schedule: BalancedSchedule,
    expected: LegacyExpectation,
    protocol_hash: str,
    recipe: object,
) -> _StoredTrace:
    pairing_key = _legacy_pairing_key(
        protocol_hash=protocol_hash,
        center=center,
        training_seed=training_seed,
        schedule_hash=schedule.stream_hash,
    )
    values = _legacy_cuda_trace(
        pairing_key=pairing_key,
        device=expected.device,
        optimizer_steps=recipe.optimizer_steps,
        batch_size=recipe.batch_size,
        latent_dim=recipe.latent_dim,
    )
    return _store_trace(
        root=root,
        relative=Path("traces")
        / "legacy"
        / f"center_{center}_seed_{training_seed}.npy",
        values=values,
        role="legacy_seed_specific_epsilon_trace",
    )


def _build_controlled_trace(
    *,
    root: Path,
    center: str,
    frame_hash: str,
    schedule: _StoredSchedule,
    config: SnapshotBuildConfig,
) -> _StoredTrace:
    import numpy as np

    seed_key = stable_hash(
        {
            "schema_version": "midogpp_b_fold_fixed_epsilon_trace_v1",
            "center": center,
            "frame_hash": frame_hash,
            "schedule_content_hash": schedule.entry.content_hash,
            "recipe_hash": config.recipe.hash,
        }
    )
    seed = int(seed_key[:16], 16) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(
        (
            config.recipe.optimizer_steps,
            config.recipe.batch_size,
            config.recipe.latent_dim,
        )
    ).astype("<f4")
    return _store_trace(
        root=root,
        relative=Path("traces") / "controlled" / f"center_{center}.npy",
        values=values,
        role="fold_fixed_epsilon_trace",
    )


def _store_trace(
    *,
    root: Path,
    relative: Path,
    values: object,
    role: str,
) -> _StoredTrace:
    import numpy as np

    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, np.asarray(values, dtype="<f4", order="C"), allow_pickle=False)
    temporary.replace(path)
    content_hash = trace_content_hash(values)
    return _StoredTrace(
        entry=_content_entry(root, path, content_hash=content_hash, role=role)
    )


def _legacy_cuda_trace(
    *,
    pairing_key: str,
    device: str,
    optimizer_steps: int,
    batch_size: int,
    latent_dim: int,
) -> object:
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise ProtocolError(
            "Building exact legacy epsilon traces requires the workstation CUDA GPUs."
        )
    try:
        torch.cuda.set_device(device)
    except (RuntimeError, ValueError) as exc:
        raise ProtocolError(f"Legacy replay device is unavailable: {device}") from exc
    values = np.empty(
        (optimizer_steps, batch_size, latent_dim),
        dtype="<f4",
    )
    for step in range(1, optimizer_steps + 1):
        posterior_seed = _derived_seed(pairing_key, step, "posterior")
        generator = torch.Generator(device=device).manual_seed(posterior_seed)
        epsilon = torch.randn(
            (batch_size, latent_dim),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        values[step - 1] = epsilon.detach().cpu().numpy()
    return values


def _legacy_pairing_key(
    *,
    protocol_hash: str,
    center: str,
    training_seed: int,
    schedule_hash: str,
) -> str:
    return stable_hash(
        {
            "protocol_hash": protocol_hash,
            "center": center,
            "training_seed": training_seed,
            "schedule_hash": schedule_hash,
            "paired_across_arms": True,
        }
    )


def _posterior_stream_hash(
    *,
    pairing_key: str,
    schedule_hash: str,
    optimizer_steps: int,
) -> str:
    return stable_hash(
        {
            "posterior_stream_key": pairing_key,
            "posterior_seeds": [
                _derived_seed(pairing_key, step, "posterior")
                for step in range(1, optimizer_steps + 1)
            ],
            "schedule_hash": schedule_hash,
        }
    )


def _training_spec(config: SnapshotBuildConfig) -> StepTrainingSpec:
    recipe = config.recipe
    return StepTrainingSpec(
        optimizer_steps=recipe.optimizer_steps,
        batch_size=recipe.batch_size,
        hidden_dim=recipe.hidden_dim,
        latent_dim=recipe.latent_dim,
        learning_rate=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        beta_final=recipe.beta_final,
        kl_warmup_steps=recipe.kl_warmup_steps,
        gradient_clip_norm=recipe.gradient_clip_norm,
    )


def _content_entry(
    root: Path,
    path: Path,
    *,
    content_hash: str,
    role: str,
) -> ContentEntry:
    return ContentEntry(
        relative_path=str(path.relative_to(root)),
        sha256=file_sha256(path),
        content_hash=content_hash,
        size_bytes=int(path.stat().st_size),
        role=role,
    )


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _assert_output_root(config: SnapshotBuildConfig, root: Path) -> None:
    configured = str(config.artifact_root)
    if configured.startswith("output://"):
        raise ProtocolError(
            "Snapshot execution requires a workspace-resolved absolute artifact_root."
        )
    if Path(configured).resolve() != root:
        raise ProtocolError("Snapshot CLI artifact root differs from resolved config.")


def _ensure_snapshot_directories(root: Path) -> None:
    for relative in (
        "manifests",
        "prepared",
        "provenance",
        "reports",
        "schedules/controlled",
        "schedules/legacy",
        "traces/controlled",
        "traces/legacy",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _write_run_state(
    root: Path,
    *,
    status: str,
    completed_centers: int,
    snapshot_hash: str = "",
    wall_seconds: float = 0.0,
) -> None:
    write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_b_paired_reparameterization_snapshot_state_v1",
            "status": status,
            "completed_centers": int(completed_centers),
            "expected_centers": len(AUDIT_CENTERS),
            "snapshot_hash": snapshot_hash,
            "wall_seconds": float(wall_seconds),
            "historical_paths_read": False,
            "claim_scope": "diagnostic_only",
            "may_export_recipe_lock": False,
            "may_feed_stage20": False,
            "may_feed_expert_bank": False,
            "may_feed_generation": False,
            "may_feed_routing": False,
            "may_feed_composition": False,
            "may_feed_downstream": False,
            "may_feed_deployable_selection": False,
            "may_tune_or_select": False,
            "may_support_thesis_claim": False,
        },
    )


__all__ = ("build_snapshot_from_config",)
