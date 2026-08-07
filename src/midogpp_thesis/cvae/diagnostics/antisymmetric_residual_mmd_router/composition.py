"""Class-specific source composition and downstream classifier fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import fit_logistic_classifier
from ...protocol import ProtocolError
from ._prediction_common import compact_json, integer, is_hash, sha256_array
from .contracts import (
    CONTROL_ARM,
    MAX_SOURCE_PREFIX_PER_CLASS,
    ROUTED_ARM,
    TOTAL_PER_CLASS,
    candidate_sources,
)


@dataclass(frozen=True)
class ClassSpecificComposition:
    embeddings: np.ndarray
    labels: np.ndarray
    composition_hash: str


def compose_class_specific_prefix_blocks(
    blocks: Mapping[str, object],
    allocations_by_class: Mapping[str, object],
    *,
    shuffle_seed_by_class: Mapping[str, object],
) -> ClassSpecificComposition:
    """Compose fixed source prefixes and deterministically shuffle by class."""

    candidates = tuple(sorted(str(source) for source in blocks))
    allocations = normalize_allocations_by_class(
        allocations_by_class, candidates=candidates
    )
    seeds = {
        str(class_label): integer(
            shuffle_seed_by_class[str(class_label)], "class shuffle seed"
        )
        for class_label in (0, 1)
    }
    class_arrays: list[np.ndarray] = []
    prefix_hashes: dict[str, str] = {}
    post_shuffle_hashes: dict[str, str] = {}
    feature_dim: int | None = None
    for class_label in (0, 1):
        prefixes: list[np.ndarray] = []
        for source in candidates:
            block = blocks[source]
            embeddings = np.asarray(
                getattr(block, "embeddings", None), dtype=np.float32
            )
            labels = np.asarray(getattr(block, "labels", None), dtype=np.int64)
            if (
                embeddings.ndim != 2
                or labels.shape != (len(embeddings),)
                or not np.isfinite(embeddings).all()
                or getattr(getattr(block, "key", None), "source_center", None)
                != source
            ):
                raise ProtocolError("Antisymmetric source block is malformed.")
            feature_dim = embeddings.shape[1] if feature_dim is None else feature_dim
            if embeddings.shape[1] != feature_dim:
                raise ProtocolError(
                    "Antisymmetric source feature dimensions drifted."
                )
            count = allocations[str(class_label)][source]
            indices = np.flatnonzero(labels == class_label)[:count]
            if len(indices) != count:
                raise ProtocolError("Antisymmetric source prefix is too short.")
            prefix = np.ascontiguousarray(embeddings[indices], dtype=np.float32)
            prefixes.append(prefix)
            prefix_hashes[f"{source}:{class_label}"] = sha256_array(prefix)
        unshuffled = np.ascontiguousarray(
            np.concatenate(prefixes), dtype=np.float32
        )
        if unshuffled.shape != (TOTAL_PER_CLASS, feature_dim):
            raise ProtocolError("Antisymmetric class composition total drifted.")
        permutation = np.random.default_rng(seeds[str(class_label)]).permutation(
            TOTAL_PER_CLASS
        )
        shuffled = np.ascontiguousarray(unshuffled[permutation], dtype=np.float32)
        class_arrays.append(shuffled)
        post_shuffle_hashes[str(class_label)] = sha256_array(shuffled)
    embeddings = np.ascontiguousarray(np.concatenate(class_arrays), dtype=np.float32)
    labels = np.concatenate(
        (
            np.zeros(TOTAL_PER_CLASS, dtype=np.int64),
            np.ones(TOTAL_PER_CLASS, dtype=np.int64),
        )
    )
    payload = {
        "schema_version": "midogpp_antisymmetric_class_specific_composition_v1",
        "source_order": list(candidates),
        "allocations_by_class": allocations,
        "shuffle_seed_by_class": seeds,
        "prefix_sha256_by_source_class": prefix_hashes,
        "post_shuffle_sha256_by_class": post_shuffle_hashes,
        "output_embeddings_sha256": sha256_array(embeddings),
        "output_labels_sha256": sha256_array(labels),
        "total_per_class": TOTAL_PER_CLASS,
    }
    return ClassSpecificComposition(embeddings, labels, stable_hash(payload))


def fit_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    evaluation_embeddings: np.ndarray,
    *,
    classifier: object,
    threads: int,
) -> dict[str, object]:
    """Fit the frozen classifier reference under the fixed thread budget."""

    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Antisymmetric prediction requires threadpoolctl."
        ) from exc
    if threads != 3:
        raise ProtocolError(
            "Antisymmetric classifier workers require three threads."
        )
    with threadpool_limits(limits=threads):
        fitted = fit_logistic_classifier(
            train_embeddings,
            train_labels,
            evaluation_embeddings,
            spec=classifier,
        )
    predictions = np.asarray(fitted.predictions, dtype=np.uint8)
    probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
    if (
        tuple(int(value) for value in fitted.classes) != (0, 1)
        or predictions.shape != (len(evaluation_embeddings),)
        or probabilities.shape != (len(evaluation_embeddings), 2)
        or not np.isfinite(probabilities).all()
        or not np.allclose(
            probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=0.0
        )
        or not fitted.converged
        or fitted.classifier_config_hash != classifier.config_hash
    ):
        raise ProtocolError("Antisymmetric downstream classifier fit drifted.")
    return {
        "predictions": predictions,
        "probabilities": probabilities[:, 1].astype(np.float32, copy=False),
        "classifier_config_hash": fitted.classifier_config_hash,
        "scaler_state_hash": fitted.scaler_state_hash,
        "n_iter": tuple(int(value) for value in fitted.n_iter),
        "converged": bool(fitted.converged),
    }


def fit_metadata(fitted: Mapping[str, object]) -> dict[str, object]:
    return {
        "classifier_config_hash": fitted["classifier_config_hash"],
        "scaler_state_hash": fitted["scaler_state_hash"],
        "classifier_n_iter_json": compact_json(list(fitted["n_iter"])),
        "classifier_converged": fitted["converged"],
    }


def validate_plan(plan: object, *, fold: object) -> None:
    """Validate both frozen composition arms for one cross-fit fold."""

    if not isinstance(plan, Mapping):
        raise ProtocolError(
            f"Antisymmetric plan is missing for fold {fold.fold_id}."
        )
    candidates = candidate_sources(fold.target_center)
    control = source_float_mapping(plan.get("control_weights"), candidates)
    class_0 = source_float_mapping(plan.get("class_0_weights"), candidates)
    class_1 = source_float_mapping(plan.get("class_1_weights"), candidates)
    control_allocations = normalize_allocations_by_class(
        plan.get("control_allocations_by_class"), candidates=candidates
    )
    routed_allocations = normalize_allocations_by_class(
        plan.get("routed_allocations_by_class"), candidates=candidates
    )
    if (
        plan.get("fold_id") != fold.fold_id
        or plan.get("target_center") != fold.target_center
        or tuple(str(value) for value in plan.get("candidate_sources", ()))
        != candidates
        or not is_hash(plan.get("plan_hash"))
        or any(
            abs(sum(values.values()) - 1.0) > 1e-10
            for values in (control, class_0, class_1)
        )
        or control_allocations["0"] != control_allocations["1"]
    ):
        raise ProtocolError("Antisymmetric fold plan failed its frozen contract.")
    # Keep this local variable live as a complete parse of the routed arm.
    _ = routed_allocations


def arm_plan_payload(
    plan: Mapping[str, object],
    arm: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Return normalized class weights and integer allocations for one arm."""

    candidates = tuple(str(value) for value in plan["candidate_sources"])
    if arm == CONTROL_ARM:
        control = source_float_mapping(plan["control_weights"], candidates)
        weights = {"0": control, "1": dict(control)}
        allocations = normalize_allocations_by_class(
            plan["control_allocations_by_class"], candidates=candidates
        )
    elif arm == ROUTED_ARM:
        weights = {
            "0": source_float_mapping(plan["class_0_weights"], candidates),
            "1": source_float_mapping(plan["class_1_weights"], candidates),
        }
        allocations = normalize_allocations_by_class(
            plan["routed_allocations_by_class"], candidates=candidates
        )
    else:
        raise ProtocolError("Antisymmetric arm is unknown.")
    return weights, allocations


def normalize_allocations_by_class(
    value: object,
    *,
    candidates: Sequence[str],
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping):
        raise ProtocolError(
            "Antisymmetric class-specific allocations are malformed."
        )
    normalized: dict[str, dict[str, int]] = {}
    for class_label in (0, 1):
        raw = value.get(str(class_label), value.get(class_label))
        if not isinstance(raw, Mapping):
            raise ProtocolError("Antisymmetric class allocation is missing.")
        parsed: dict[str, int] = {}
        for source in candidates:
            if source not in raw:
                raise ProtocolError("Antisymmetric source allocation is missing.")
            parsed[source] = integer(raw[source], "source allocation")
        if (
            set(str(key) for key in raw) != set(candidates)
            or any(
                count <= 0 or count > MAX_SOURCE_PREFIX_PER_CLASS
                for count in parsed.values()
            )
            or sum(parsed.values()) != TOTAL_PER_CLASS
        ):
            raise ProtocolError(
                "Antisymmetric class allocation geometry drifted."
            )
        normalized[str(class_label)] = parsed
    return normalized


def source_float_mapping(
    value: object,
    candidates: Sequence[str],
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(str(key) for key in value) != set(
        candidates
    ):
        raise ProtocolError("Antisymmetric source weights are malformed.")
    parsed = {source: float(value[source]) for source in candidates}
    if any(
        not np.isfinite(weight) or weight < 0.0 for weight in parsed.values()
    ):
        raise ProtocolError("Antisymmetric source weights are invalid.")
    return parsed


__all__ = (
    "ClassSpecificComposition",
    "arm_plan_payload",
    "compose_class_specific_prefix_blocks",
    "fit_classifier",
    "fit_metadata",
    "normalize_allocations_by_class",
    "source_float_mapping",
    "validate_plan",
)
