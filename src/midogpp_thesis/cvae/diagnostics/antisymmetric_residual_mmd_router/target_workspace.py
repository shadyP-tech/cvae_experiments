"""Reusable target-excluded kernel workspace for all case-crossfit folds.

Each target job fits the source-only scaler, class-prior model, and shared
Nyström map exactly once.  All target rows are transformed in one GPU pass;
fold planners may then take label-free row subsets without refitting state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.mmd_kmm_mixture import (
    FrozenNystroemFeatureMap,
    MMDKMMProtocol,
    PriorControlConfig,
    SourceKernelReplica,
    TargetSupportKernelFeatures,
    TransformedKernelFeatures,
    prepare_source_only_responsibilities,
)
from .contracts import (
    COMMON_FEATURE_DIM,
    COMMON_FRAME_HASH,
    GENERATION_SEEDS,
    KERNEL_BATCH_ROWS,
    MAX_SOURCE_PREFIX_PER_CLASS,
    NYSTROEM_COMPONENTS,
    NYSTROEM_GAMMA,
    NYSTROEM_RANDOM_STATE,
    PRIOR_CLASSIFIER,
    ROUTER_PREFIX_PER_CLASS,
    TRAINING_SEEDS,
    candidate_sources,
)


@dataclass(frozen=True)
class TargetKernelWorkspace:
    """Source-only fitted state reused across one target's crossfit folds."""

    target_center: str
    target_sample_ids: tuple[str, ...]
    target_case_ids: tuple[str, ...]
    target_kernel_features: np.ndarray
    raw_prior_probabilities: np.ndarray
    source_replicas: tuple[SourceKernelReplica, ...]
    preprocessing_hash: str
    candidate_pool_fit_hash: str
    kernel_map_hash: str
    prior_model_hash: str
    prior_fit_pool_hash: str
    gpu_probe_max_abs_error: float
    state_arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        features = np.asarray(self.target_kernel_features, dtype=np.float64)
        probabilities = np.asarray(self.raw_prior_probabilities, dtype=np.float64)
        if (
            self.target_center not in {"0", "1", "2", "3", "5", "6", "7", "8", "9"}
            or len(self.target_sample_ids) != len(features)
            or len(self.target_case_ids) != len(features)
            or len(set(self.target_sample_ids)) != len(self.target_sample_ids)
            or features.ndim != 2
            or features.shape[1] != NYSTROEM_COMPONENTS
            or probabilities.shape != (len(features), 2)
            or not np.isfinite(features).all()
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
            or len(self.source_replicas)
            != len(candidate_sources(self.target_center))
            * len(TRAINING_SEEDS)
            * len(GENERATION_SEEDS)
            * 2
            or not all(
                str(value)
                for value in (
                    self.preprocessing_hash,
                    self.candidate_pool_fit_hash,
                    self.kernel_map_hash,
                    self.prior_model_hash,
                    self.prior_fit_pool_hash,
                )
            )
            or not np.isfinite(float(self.gpu_probe_max_abs_error))
            or float(self.gpu_probe_max_abs_error) > 5.0e-4
        ):
            raise ProtocolError("Antisymmetric target kernel workspace is malformed.")

    def target_support(
        self,
        protocol: MMDKMMProtocol,
        support_sample_ids: Sequence[str],
        *,
        prior_control: PriorControlConfig,
    ) -> TargetSupportKernelFeatures:
        """Build one label-free fold support object from cached target rows."""

        if (
            protocol.target_center != self.target_center
            or protocol.candidate_sources != candidate_sources(self.target_center)
        ):
            raise ProtocolError("Crossfit support crossed its target workspace.")
        ordinal = {sample_id: index for index, sample_id in enumerate(self.target_sample_ids)}
        try:
            indices = np.asarray([ordinal[str(value)] for value in support_sample_ids], dtype=np.int64)
        except KeyError as exc:
            raise ProtocolError("Crossfit support contains an unknown target row.") from exc
        if len(indices) == 0 or len(set(indices.tolist())) != len(indices):
            raise ProtocolError("Crossfit support row selection is empty or duplicated.")
        cases = tuple(self.target_case_ids[int(index)] for index in indices)
        if set(cases) != set(protocol.support_case_ids):
            raise ProtocolError("Crossfit support cases differ from the fold protocol.")
        features = TransformedKernelFeatures(
            values=self.target_kernel_features[indices],
            common_frame_hash=COMMON_FRAME_HASH,
            preprocessing_hash=self.preprocessing_hash,
            candidate_pool_fit_hash=self.candidate_pool_fit_hash,
            kernel_map_hash=self.kernel_map_hash,
        )
        prior = prepare_source_only_responsibilities(
            self.raw_prior_probabilities[indices],
            protocol=protocol,
            prior_model_hash=self.prior_model_hash,
            prior_fit_pool_hash=self.prior_fit_pool_hash,
            config=prior_control,
        )
        return TargetSupportKernelFeatures(
            target_center=self.target_center,
            case_ids=cases,
            kernel_features=features,
            prior_prediction=prior,
            support_labels_used=False,
            evaluation_embeddings_used=True,
            cross_fitted_transductive_support=True,
            cohort_evaluation_embeddings_used=True,
            heldout_evaluation_embeddings_used=False,
        )


def build_target_kernel_workspace(task: Mapping[str, object]) -> TargetKernelWorkspace:
    """Fit and transform one target-excluded workspace in a GPU worker."""

    target = str(task["target_center"])
    candidates = candidate_sources(target)
    source_array = np.load(Path(str(task["source_array_path"])), mmap_mode="r")
    index_rows = tuple(task["source_index_rows"])
    block_index = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])):
        int(row["block_ordinal"])
        for row in index_rows
    }
    index_by_key = {
        (str(row["source_center"]), int(row["training_seed"]), int(row["generation_seed"])): row
        for row in index_rows
    }
    pool_rows: list[np.ndarray] = []
    pool_labels: list[np.ndarray] = []
    replica_slices: list[tuple[str, int, int, int, int, int]] = []
    block_hashes: list[str] = []
    cursor = 0
    for source, training_seed, generation_seed in product(
        candidates, TRAINING_SEEDS, GENERATION_SEEDS
    ):
        key = (source, training_seed, generation_seed)
        try:
            block = source_array[block_index[key]]
            block_hashes.append(str(index_by_key[key]["output_sha256"]))
        except KeyError as exc:
            raise ProtocolError("Antisymmetric source block grid is incomplete.") from exc
        for label, start in ((0, 0), (1, MAX_SOURCE_PREFIX_PER_CLASS)):
            values = np.ascontiguousarray(
                block[start : start + ROUTER_PREFIX_PER_CLASS], dtype=np.float64
            )
            pool_rows.append(values)
            pool_labels.append(np.full(len(values), label, dtype=np.int64))
            stop = cursor + len(values)
            replica_slices.append(
                (source, training_seed, generation_seed, label, cursor, stop)
            )
            cursor = stop
    pool = np.ascontiguousarray(np.concatenate(pool_rows), dtype=np.float64)
    labels = np.concatenate(pool_labels)
    expected_rows = (
        len(candidates)
        * len(TRAINING_SEEDS)
        * len(GENERATION_SEEDS)
        * 2
        * ROUTER_PREFIX_PER_CLASS
    )
    if pool.shape != (expected_rows, COMMON_FEATURE_DIM):
        raise ProtocolError("Antisymmetric source-only fit-pool geometry drifted.")
    target_embeddings = np.asarray(task["target_embeddings"], dtype=np.float64)
    target_sample_ids = tuple(str(value) for value in task["target_sample_ids"])
    target_case_ids = tuple(str(value) for value in task["target_case_ids"])
    if (
        target_embeddings.shape != (len(target_sample_ids), COMMON_FEATURE_DIM)
        or len(target_case_ids) != len(target_sample_ids)
        or not np.isfinite(target_embeddings).all()
    ):
        raise ProtocolError("Antisymmetric target workspace rows do not align.")

    candidate_pool_fit_hash = stable_hash(
        {
            "target_center": target,
            "candidate_sources": list(candidates),
            "training_seeds": list(TRAINING_SEEDS),
            "generation_seeds": list(GENERATION_SEEDS),
            "prefix_per_class": ROUTER_PREFIX_PER_CLASS,
            "source_block_hashes": block_hashes,
            "fit_role": "target_excluded_source_pool_only",
        }
    )
    try:
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError(
            "Antisymmetric target workspace requires scikit-learn and threadpoolctl."
        ) from exc
    with threadpool_limits(limits=int(task["classifier_threads"])):
        scaler = StandardScaler().fit(pool)
        scaled_pool = scaler.transform(pool)
        scaled_target = scaler.transform(target_embeddings)
        prior = LogisticRegression(**PRIOR_CLASSIFIER.to_sklearn_kwargs()).fit(
            scaled_pool, labels
        )
        raw_probabilities = np.asarray(prior.predict_proba(scaled_target), dtype=np.float64)
        nystroem = Nystroem(
            kernel="rbf",
            gamma=NYSTROEM_GAMMA,
            n_components=NYSTROEM_COMPONENTS,
            random_state=NYSTROEM_RANDOM_STATE,
        ).fit(scaled_pool)
    if (
        tuple(int(value) for value in prior.classes_) != (0, 1)
        or int(np.max(prior.n_iter_)) >= PRIOR_CLASSIFIER.max_iter
    ):
        raise ProtocolError("Antisymmetric source-only prior did not converge.")
    preprocessing_hash = stable_hash(
        {
            "candidate_pool_fit_hash": candidate_pool_fit_hash,
            "mean_sha256": _sha256_array(np.asarray(scaler.mean_)),
            "var_sha256": _sha256_array(np.asarray(scaler.var_)),
            "scale_sha256": _sha256_array(np.asarray(scaler.scale_)),
            "fit_role": "target_excluded_candidate_pool_generated_common_frame",
        }
    )
    feature_map = FrozenNystroemFeatureMap(
        components=np.asarray(nystroem.components_, dtype=np.float64),
        normalization=np.asarray(nystroem.normalization_, dtype=np.float64),
        gamma=NYSTROEM_GAMMA,
        common_frame_hash=COMMON_FRAME_HASH,
        preprocessing_hash=preprocessing_hash,
        candidate_pool_fit_hash=candidate_pool_fit_hash,
        random_state=NYSTROEM_RANDOM_STATE,
    )
    transformed_pool, pool_error = _transform_nystroem_batched(
        scaled_pool,
        feature_map,
        device=str(task["device"]),
        batch_rows=KERNEL_BATCH_ROWS,
    )
    transformed_target, target_error = _transform_nystroem_batched(
        scaled_target,
        feature_map,
        device=str(task["device"]),
        batch_rows=KERNEL_BATCH_ROWS,
    )
    prior_fit_pool_hash = stable_hash(
        {
            "candidate_pool_fit_hash": candidate_pool_fit_hash,
            "balance": "equal_source_seed_class_prefix",
            "row_count": len(pool),
            "labels_sha256": _sha256_array(labels),
        }
    )
    prior_model_hash = stable_hash(
        {
            "classifier": PRIOR_CLASSIFIER.to_payload(),
            "preprocessing_hash": preprocessing_hash,
            "prior_fit_pool_hash": prior_fit_pool_hash,
            "coefficient_sha256": _sha256_array(np.asarray(prior.coef_)),
            "intercept_sha256": _sha256_array(np.asarray(prior.intercept_)),
            "classes": [int(value) for value in prior.classes_],
            "n_iter": [int(value) for value in prior.n_iter_],
        }
    )
    replicas: list[SourceKernelReplica] = []
    for source, training_seed, generation_seed, label, start, stop in replica_slices:
        replicas.append(
            SourceKernelReplica(
                source_center=source,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
                kernel_features=TransformedKernelFeatures(
                    values=transformed_pool[start:stop],
                    common_frame_hash=COMMON_FRAME_HASH,
                    preprocessing_hash=preprocessing_hash,
                    candidate_pool_fit_hash=candidate_pool_fit_hash,
                    kernel_map_hash=feature_map.kernel_map_hash,
                ),
            )
        )
    state = {
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scaler_var": np.asarray(scaler.var_, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "kernel_components": np.asarray(feature_map.components, dtype=np.float64),
        "kernel_normalization": np.asarray(feature_map.normalization, dtype=np.float64),
        "prior_coef": np.asarray(prior.coef_, dtype=np.float64),
        "prior_intercept": np.asarray(prior.intercept_, dtype=np.float64),
    }
    return TargetKernelWorkspace(
        target_center=target,
        target_sample_ids=target_sample_ids,
        target_case_ids=target_case_ids,
        target_kernel_features=transformed_target,
        raw_prior_probabilities=raw_probabilities,
        source_replicas=tuple(replicas),
        preprocessing_hash=preprocessing_hash,
        candidate_pool_fit_hash=candidate_pool_fit_hash,
        kernel_map_hash=feature_map.kernel_map_hash,
        prior_model_hash=prior_model_hash,
        prior_fit_pool_hash=prior_fit_pool_hash,
        gpu_probe_max_abs_error=max(pool_error, target_error),
        state_arrays=state,
    )


def _transform_nystroem_batched(
    values: np.ndarray,
    feature_map: FrozenNystroemFeatureMap,
    *,
    device: str,
    batch_rows: int,
) -> tuple[np.ndarray, float]:
    try:
        import torch
        from sklearn.metrics.pairwise import rbf_kernel
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError(
            "Antisymmetric GPU kernel transform requires torch and scikit-learn."
        ) from exc
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    components = torch.as_tensor(
        np.array(feature_map.components, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    normalization = torch.as_tensor(
        np.array(feature_map.normalization, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )
    component_norm = torch.sum(components * components, dim=1)[None, :]
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_rows)):
            batch = torch.as_tensor(
                values[start : start + int(batch_rows)],
                dtype=torch.float32,
                device=device,
            )
            distances = (
                torch.sum(batch * batch, dim=1)[:, None]
                + component_norm
                - 2.0 * (batch @ components.T)
            )
            kernel = torch.exp(
                -float(feature_map.gamma) * torch.clamp(distances, min=0.0)
            )
            outputs.append(
                (kernel @ normalization.T).cpu().numpy().astype(np.float64, copy=False)
            )
    output = np.ascontiguousarray(np.concatenate(outputs), dtype=np.float64)
    probe_count = min(3, len(values))
    expected = rbf_kernel(
        np.asarray(values[:probe_count], dtype=np.float64),
        np.asarray(feature_map.components, dtype=np.float64),
        gamma=float(feature_map.gamma),
    ) @ np.asarray(feature_map.normalization, dtype=np.float64).T
    error = float(np.max(np.abs(output[:probe_count] - expected)))
    if (
        output.shape != (len(values), len(feature_map.components))
        or not np.isfinite(output).all()
        or error > 5.0e-4
    ):
        raise ProtocolError("Antisymmetric GPU Nyström transform failed its CPU probe.")
    return output, error


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = ("TargetKernelWorkspace", "build_target_kernel_workspace")
