"""Typed source-inner and target feature-surface reconstruction."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from ...routing.utility_aligned import (
    EnsembleCandidateFeatureRow,
    GlobalSourceControl,
    build_case_bootstrap_plan,
    build_ensemble_feature_surface,
    cyclically_permute_target_scalar,
)
from ...routing.utility_aligned.ensemble_feature_contracts import (
    GLOBAL_SOURCE_CONTROL_SEMANTICS,
)
from .artifact_io import read_json
from .contracts import CENTERS, candidate_sources, inner_candidate_sources
from .features import (
    SourceInnerFeatureSurfaces,
    build_source_inner_feature_surface_set,
)
from .validation_science_common import (
    boolean,
    floating,
    integer,
    json_value,
    mapping,
    mapping_field,
    nullable_text,
    read_csv,
    require_fields,
    require_payload_hash,
)
from .validation_science_contracts import (
    FEATURE_FIELDS,
    FeatureScienceValidation,
    ScientificPartitionContext,
)


def validate_feature_science(
    root: str | Path, partitions: ScientificPartitionContext
) -> FeatureScienceValidation:
    base = Path(root)
    source_rows = tuple(
        feature_row(raw)
        for raw in read_csv(base / "tables/source_inner_feature_rows.csv")
    )
    target_rows = tuple(
        feature_row(raw)
        for raw in read_csv(base / "tables/target_feature_rows.csv")
    )
    expected_source_keys = {
        (outer, query, source)
        for outer in CENTERS
        for query in candidate_sources(outer)
        for source in inner_candidate_sources(outer, query)
    }
    expected_target_keys = {
        (target, target, source)
        for target in CENTERS
        for source in candidate_sources(target)
    }
    if (
        len(source_rows) != 504
        or {row.row_key for row in source_rows} != expected_source_keys
        or len(target_rows) != 72
        or {row.row_key for row in target_rows} != expected_target_keys
    ):
        raise ProtocolError("Candidate feature key geometry drifted.")
    for row in (*source_rows, *target_rows):
        if (
            row.support_partition_hash
            != partitions.support_feature_hash_by_center[row.query_id]
            or row.support_case_count != 8
        ):
            raise ProtocolError("Candidate feature partition binding drifted.")

    manifest = read_json(base / "manifests/feature_surface_set.json")
    source_payloads = mapping_field(manifest, "source_surfaces_by_target")
    control_payloads = mapping_field(
        manifest, "global_source_controls_by_target"
    )
    row_hashes_by_target = mapping_field(manifest, "m1_row_hashes_by_target")
    source_surfaces: dict[str, SourceInnerFeatureSurfaces] = {}
    for target in CENTERS:
        control = global_control(mapping(control_payloads[target], "control"))
        selected = tuple(
            sorted(
                (row for row in source_rows if row.outer_target_id == target),
                key=lambda row: row.row_key,
            )
        )
        expected_control_values = {
            source: float(
                np.mean(
                    np.asarray(
                        [
                            row.feature_mean_by_name["metadata_similarity"]
                            for row in selected
                            if row.candidate_source == source
                        ],
                        dtype=np.float64,
                    ),
                    dtype=np.float64,
                )
            )
            for source in candidate_sources(target)
        }
        expected_seed_row_hash = canonical_sha256(
            [seed_hash for row in selected for seed_hash in row.seed_row_hashes]
        )
        if (
            any(
                not math.isclose(
                    control.value_by_source[source],
                    expected_control_values[source],
                    rel_tol=8.0 * np.finfo(np.float64).eps,
                    abs_tol=8.0 * np.finfo(np.float64).eps,
                )
                for source in candidate_sources(target)
            )
            or control.input_row_hashes_hash != expected_seed_row_hash
        ):
            raise ProtocolError(
                "Global source control cannot be replayed from feature rows."
            )
        if row_hashes_by_target.get(target) != [row.row_hash for row in selected]:
            raise ProtocolError("Source feature row-hash lineage drifted.")
        m0_rows = tuple(without_target_scalar(row) for row in selected)
        kwargs = {
            "global_source_control_by_source": control.value_by_source,
            "global_source_control_semantics": control.semantics,
            "global_source_control_provenance_hash": control.provenance_hash,
        }
        m0 = build_ensemble_feature_surface(m0_rows, **kwargs)
        m1 = build_ensemble_feature_surface(selected, **kwargs)
        permutation = cyclically_permute_target_scalar(
            m1, permutation_seed=90_902_026
        )
        wrapper = mapping(source_payloads[target], "source surface")
        surface = SourceInnerFeatureSurfaces(
            outer_target_id=target,
            global_source_control=control,
            m0=m0,
            m1=m1,
            permutation=permutation,
            feature_input_seal_hash=str(wrapper.get("feature_input_seal_hash", "")),
            surface_hash=str(wrapper.get("surface_hash", "")),
        )
        if wrapper != surface.to_payload():
            raise ProtocolError("Source feature surface payload drifted.")
        source_surfaces[target] = surface
    source_set = build_source_inner_feature_surface_set(source_surfaces)
    base_source_manifest = source_set.to_payload()
    if any(
        manifest.get(key) != value
        for key, value in base_source_manifest.items()
    ):
        raise ProtocolError("Source feature surface-set hash drifted.")

    plans = read_json(base / "manifests/target_policy_plans.json")
    feature_payloads = mapping_field(plans, "target_features_by_target")
    production_payloads = mapping_field(
        plans, "target_feature_productions_by_target"
    )
    target_hashes = mapping_field(plans, "target_point_row_hashes_by_target")
    target_feature_hash_by_target: dict[str, str] = {}
    for target in CENTERS:
        selected = tuple(
            sorted(
                (row for row in target_rows if row.outer_target_id == target),
                key=lambda row: row.row_key,
            )
        )
        if target_hashes.get(target) != [row.row_hash for row in selected]:
            raise ProtocolError("Target feature row-hash lineage drifted.")
        control = source_surfaces[target].global_source_control
        point = build_ensemble_feature_surface(
            selected,
            global_source_control_by_source=control.value_by_source,
            global_source_control_semantics=control.semantics,
            global_source_control_provenance_hash=control.provenance_hash,
        )
        production = mapping(production_payloads[target], "target production")
        require_payload_hash(production, "production_hash", "target production")
        if (
            production.get("target_id") != target
            or production.get("point_surface_hash") != point.surface_hash
            or production.get("labels_used") is not False
            or production.get("utility_responses_used") is not False
            or int(production.get("bootstrap_replicate_count", -1)) != 32
        ):
            raise ProtocolError("Target feature production drifted.")
        wrapper = mapping(feature_payloads[target], "target feature")
        require_payload_hash(wrapper, "feature_hash", "target feature")
        bootstrap = build_case_bootstrap_plan(
            target_id=target,
            support_case_ids=partitions.support_case_ids_by_center[target],
        )
        if (
            wrapper.get("target_id") != target
            or wrapper.get("source_feature_surface_hash")
            != source_surfaces[target].surface_hash
            or wrapper.get("support_partition_lock_hash")
            != partitions.support_partition_lock_hash
            or wrapper.get("case_bootstrap_plan_hash") != bootstrap.plan_hash
            or wrapper.get("target_feature_production_hash")
            != production.get("production_hash")
            or int(wrapper.get("support_case_count", -1)) != 8
            or int(wrapper.get("bootstrap_replicate_count", -1)) != 32
            or wrapper.get("labels_used") is not False
            or wrapper.get("utility_responses_used") is not False
        ):
            raise ProtocolError("Target feature wrapper drifted.")
        target_feature_hash_by_target[target] = str(wrapper["feature_hash"])
    return FeatureScienceValidation(
        source_feature_count=len(source_rows),
        target_feature_count=len(target_rows),
        source_surface_set_hash=source_set.surface_set_hash,
        source_surface_hash_by_target=MappingProxyType(
            {target: source_surfaces[target].surface_hash for target in CENTERS}
        ),
        target_feature_hash_by_target=MappingProxyType(
            target_feature_hash_by_target
        ),
        source_surface_set=source_set,
    )


def feature_row(raw: Mapping[str, str]) -> EnsembleCandidateFeatureRow:
    require_fields(raw, FEATURE_FIELDS, "candidate feature")
    payload: dict[str, object] = dict(raw)
    for name in ("candidate_source_count", "support_case_count", "seed_pair_count"):
        payload[name] = integer(raw[name], name)
    payload["seed_row_hashes"] = json_value(
        raw["seed_row_hashes"], "seed_row_hashes", list
    )
    for name in ("feature_mean_by_name", "feature_seed_standard_deviation_by_name"):
        payload[name] = json_value(raw[name], name, dict)
    for name in (
        "target_local_scalar", "target_local_scalar_seed_standard_deviation"
    ):
        payload[name] = None if raw[name] == "" else floating(raw[name], name)
    for name in (
        "target_local_scalar_name", "target_local_scalar_semantics",
        "target_local_scalar_provenance_hash",
    ):
        payload[name] = nullable_text(raw[name])
    payload["seed_rows_are_independent_observations"] = boolean(
        raw["seed_rows_are_independent_observations"],
        "seed_rows_are_independent_observations",
    )
    persisted_hash = str(payload.pop("row_hash"))
    if (
        payload.pop("seed_pair_count") != 9
        or payload.pop("seed_rows_are_independent_observations") is not False
    ):
        raise ProtocolError("Candidate feature technical-seed semantics drifted.")
    row = EnsembleCandidateFeatureRow(
        role=str(payload["role"]),
        outer_target_id=str(payload["outer_target_id"]),
        query_id=str(payload["query_id"]),
        candidate_source=str(payload["candidate_source"]),
        candidate_source_count=int(payload["candidate_source_count"]),
        support_partition_hash=str(payload["support_partition_hash"]),
        support_case_count=int(payload["support_case_count"]),
        seed_row_hashes=tuple(str(value) for value in payload["seed_row_hashes"]),
        feature_mean_by_name={
            str(key): float(value)
            for key, value in mapping(
                payload["feature_mean_by_name"], "feature means"
            ).items()
        },
        feature_seed_standard_deviation_by_name={
            str(key): float(value)
            for key, value in mapping(
                payload["feature_seed_standard_deviation_by_name"],
                "feature spread",
            ).items()
        },
        target_local_scalar=payload["target_local_scalar"],
        target_local_scalar_name=payload["target_local_scalar_name"],
        target_local_scalar_semantics=payload["target_local_scalar_semantics"],
        target_local_scalar_seed_standard_deviation=(
            payload["target_local_scalar_seed_standard_deviation"]
        ),
        target_local_scalar_provenance_hash=(
            payload["target_local_scalar_provenance_hash"]
        ),
    )
    if persisted_hash != row.row_hash or {
        **row.to_payload(), "row_hash": persisted_hash
    } != {
        **payload,
        "seed_pair_count": 9,
        "seed_rows_are_independent_observations": False,
        "row_hash": persisted_hash,
    }:
        raise ProtocolError("Candidate feature row hash drifted.")
    return row


def without_target_scalar(
    row: EnsembleCandidateFeatureRow,
) -> EnsembleCandidateFeatureRow:
    return EnsembleCandidateFeatureRow(
        role=row.role,
        outer_target_id=row.outer_target_id,
        query_id=row.query_id,
        candidate_source=row.candidate_source,
        candidate_source_count=row.candidate_source_count,
        support_partition_hash=row.support_partition_hash,
        support_case_count=row.support_case_count,
        seed_row_hashes=row.seed_row_hashes,
        feature_mean_by_name=row.feature_mean_by_name,
        feature_seed_standard_deviation_by_name=(
            row.feature_seed_standard_deviation_by_name
        ),
        target_local_scalar=None,
        target_local_scalar_name=None,
        target_local_scalar_semantics=None,
        target_local_scalar_seed_standard_deviation=None,
        target_local_scalar_provenance_hash=None,
    )


def global_control(payload: Mapping[str, object]) -> GlobalSourceControl:
    require_payload_hash(payload, "provenance_hash", "global source control")
    if (
        payload.get("schema_version")
        != "midogpp_utility_aligned_global_source_control_v1"
        or payload.get("labels_used") is not False
        or payload.get("utility_responses_used") is not False
        or payload.get("semantics") != GLOBAL_SOURCE_CONTROL_SEMANTICS
    ):
        raise ProtocolError("Global source-control semantics drifted.")
    values = mapping_field(payload, "value_by_source")
    control = GlobalSourceControl(
        outer_target_id=str(payload["outer_target_id"]),
        value_by_source={str(key): float(value) for key, value in values.items()},
        source_inner_seed_row_count=int(payload["source_inner_seed_row_count"]),
        input_row_hashes_hash=str(payload["input_row_hashes_hash"]),
        provenance_hash=str(payload["provenance_hash"]),
        semantics=str(payload["semantics"]),
    )
    if payload != control.to_payload():
        raise ProtocolError("Global source-control payload drifted.")
    return control


__all__ = (
    "feature_row", "global_control", "validate_feature_science",
    "without_target_scalar",
)
