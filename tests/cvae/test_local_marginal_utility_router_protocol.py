from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.bundle import (
    REQUIRED_FILES,
    assert_non_adoptive_payload,
    publication_decision_payload,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.config import (
    load_local_marginal_utility_router_config,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.contracts import (
    BOOST_ARM_ROLE,
    CENTERS,
    CONTROL_ACTION_ID,
    CONTROL_ARM_ROLE,
    EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    EXPECTED_MARGINAL_UTILITY_ROW_COUNT,
    ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID,
    PERTURBATION_LIBRARY_HASH,
    PUBLICATION_STATUS,
    TRAINING_SEEDS,
    ValidationRowIdentity,
    perturbation_library_for,
    row_identity_hash,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.label_access import (
    open_globally_sealed_development_labels,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.prediction_io import (
    PredictionAccumulator,
)
from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.seals import (
    PredictionCellSeal,
    build_global_development_prediction_seal,
    expected_prediction_keys,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.local_marginal_utility import (
    build_perturbation_library,
)


CONFIG_PATH = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v2_consumed_validation_local_marginal_utility_router_v1.yaml"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_rows() -> dict[str, tuple[ValidationRowIdentity, ...]]:
    rows: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    manifest_index = 1  # row zero is an intentionally unopened support row
    ordinal = 0
    for center in CENTERS:
        center_rows = []
        for label in (0, 1):
            center_rows.append(
                ValidationRowIdentity(
                    row_ordinal=ordinal,
                    manifest_row_index=manifest_index,
                    sample_id=f"sample-{center}-{label}",
                    case_id=f"case-{center}-{label}",
                    center=center,
                    partition_role="evaluation",
                )
            )
            manifest_index += 1
            ordinal += 1
        rows[center] = tuple(center_rows)
    return rows


def _cells(
    rows_by_query: dict[str, tuple[ValidationRowIdentity, ...]],
) -> tuple[PredictionCellSeal, ...]:
    libraries = {
        (outer, query): {
            action.action_id: action
            for action in perturbation_library_for(
                outer_target=outer,
                query_center=query,
            )
        }
        for outer in CENTERS
        for query in CENTERS
        if query != outer
    }
    cells = []
    for outer, query, action_id, training_seed, generation_seed in expected_prediction_keys():
        action = libraries[(outer, query)][action_id]
        key = f"{outer}-{query}-{action_id}-{training_seed}-{generation_seed}"
        cells.append(
            PredictionCellSeal(
                outer_target=outer,
                query_center=query,
                action_id=action_id,
                arm_role=(
                    CONTROL_ARM_ROLE
                    if action_id == CONTROL_ACTION_ID
                    else BOOST_ARM_ROLE
                ),
                boosted_source=action.boosted_source,
                candidate_sources=action.candidate_sources,
                training_seed=training_seed,
                generation_seed=generation_seed,
                evaluation_row_ids=tuple(
                    row.sample_id for row in rows_by_query[query]
                ),
                evaluation_row_identity_hash=row_identity_hash(rows_by_query[query]),
                perturbation_hash=stable_hash(action.to_payload()),
                prediction_sha256=_sha(f"prediction-{key}"),
                probability_sha256=_sha(f"probability-{key}"),
                composition_hash=_sha(f"composition-{key}")[:16],
                classifier_config_hash=_sha("classifier")[:16],
            )
        )
    return tuple(cells)


def test_config_import_does_not_load_runner_or_execution_modules() -> None:
    code = (
        "import sys; "
        "from midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.config "
        "import LocalMarginalUtilityRouterConfig; "
        "blocked=('midogpp_thesis.cvae.routing.runner', "
        "'midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.runner', "
        "'midogpp_thesis.cvae.diagnostics.local_marginal_utility_router.execution'); "
        "print([name for name in blocked if name in sys.modules])"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ".:src"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "[]"


def test_config_is_consumed_stage90_and_rejects_stage60_alias(tmp_path: Path) -> None:
    config = load_local_marginal_utility_router_config(CONFIG_PATH)
    assert config.claim_boundary["publication_status"] == PUBLICATION_STATUS
    assert config.claim_boundary["may_feed_stage60"] is False
    assert config.claim_boundary["may_feed_stage70"] is False
    assert config.protocol["target_H_labels_used_for_target_plan"] is False
    assert config.model["outer_fold_domain_exclusion"] == (
        "heldout_domain_excluded_from_both_query_center_and_source_roles"
    )
    assert config.model["target_prediction_covariance"] == (
        "parameter_covariance_plus_residual_variance_conservative"
    )
    assert config.model["geometry_transfer_status"] == (
        "extrapolative_unscored_diagnostic_only"
    )
    assert config.runtime["expected_development_classifier_fit_count"] == 5184
    assert config.runtime["expected_marginal_utility_row_count"] == 4536

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["inputs"]["validation_cache_artifact_id"] = (
        ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID
    )
    payload["inputs"]["validation_cache_root"] = (
        f"artifact://{ORIGINAL_STAGE60_VALIDATION_CACHE_ARTIFACT_ID}"
    )
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="rejects original Stage-60 aliases"):
        load_local_marginal_utility_router_config(mutated)


def test_diagnostic_perturbations_are_derived_from_math_core() -> None:
    actions = perturbation_library_for(outer_target="0", query_center="1")
    core = build_perturbation_library(
        actions[0].candidate_sources,
        total_per_class=1008,
    )
    assert tuple(action.action_id for action in actions) == tuple(
        action.action_id for action in core
    )
    assert actions[0].allocations == {source: 144 for source in actions[0].candidate_sources}
    for diagnostic, mathematical in zip(actions[1:], core[1:], strict=True):
        assert dict(diagnostic.weights) == dict(mathematical.weights)
        assert dict(diagnostic.allocations) == dict(
            mathematical.allocations_per_class
        )
        assert diagnostic.effective_source_count == pytest.approx(6.4)
        assert max(diagnostic.weights.values()) == 0.25
    assert len(expected_prediction_keys()) == EXPECTED_DEVELOPMENT_CLASSIFIER_FIT_COUNT
    assert EXPECTED_MARGINAL_UTILITY_ROW_COUNT == 4536
    assert len(PERTURBATION_LIBRARY_HASH) in {16, 64}


def test_global_seal_rejects_one_missing_prediction_cell() -> None:
    rows = _evaluation_rows()
    cells = _cells(rows)
    with pytest.raises(ProtocolError, match="lacks complete cell coverage"):
        build_global_development_prediction_seal(
            config_contract_hash=_sha("config")[:16],
            support_partition_lock_hash=_sha("partition")[:16],
            compatibility_index_hash=_sha("compatibility")[:16],
            validation_cache_binding_hash=_sha("cache")[:16],
            validation_manifest_sha256=_sha("manifest"),
            prediction_index_sha256=_sha("index"),
            prediction_arrays_sha256=_sha("arrays"),
            evaluation_rows_by_query=rows,
            cells=cells[:-1],
        )


def test_labels_open_only_after_durable_global_seal_and_skip_support(
    tmp_path: Path,
) -> None:
    rows = _evaluation_rows()
    manifest = tmp_path / "manifest.csv"
    lines = ["sample_id,case_id,center,split,label"]
    # If the implementation reads this unrequested support label, it fails.
    lines.append("support-0,support-case-0,0,val,NOT_BINARY")
    for center in CENTERS:
        for label in (0, 1):
            lines.append(
                f"sample-{center}-{label},case-{center}-{label},{center},val,{label}"
            )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    index_path = tmp_path / "development_prediction_index.csv"
    array_path = tmp_path / "development_predictions.npz"
    index_path.write_text("sealed-index\n", encoding="utf-8")
    array_path.write_bytes(b"sealed-arrays")
    seal = build_global_development_prediction_seal(
        config_contract_hash=_sha("config")[:16],
        support_partition_lock_hash=_sha("partition")[:16],
        compatibility_index_hash=_sha("compatibility")[:16],
        validation_cache_binding_hash=_sha("cache")[:16],
        validation_manifest_sha256=_sha_file(manifest),
        prediction_index_sha256=_sha_file(index_path),
        prediction_arrays_sha256=_sha_file(array_path),
        evaluation_rows_by_query=rows,
        cells=_cells(rows),
    )
    seal_path = tmp_path / "global_seal.json"
    seal_path.write_text(
        json.dumps(seal.to_payload(), sort_keys=True),
        encoding="utf-8",
    )
    opened = open_globally_sealed_development_labels(
        manifest,
        rows,
        seal=seal,
        seal_path=seal_path,
        prediction_index_path=index_path,
        prediction_arrays_path=array_path,
        expected_manifest_sha256=_sha_file(manifest),
    )
    assert tuple(opened) == CENTERS
    assert all(vector.labels == (0, 1) for vector in opened.values())

    index_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="persisted prediction bytes drifted"):
        open_globally_sealed_development_labels(
            manifest,
            rows,
            seal=seal,
            seal_path=seal_path,
            prediction_index_path=index_path,
            prediction_arrays_path=array_path,
            expected_manifest_sha256=_sha_file(manifest),
        )


def test_prediction_metadata_and_bundle_firewall_are_closed() -> None:
    accumulator = PredictionAccumulator()
    with pytest.raises(ProtocolError, match="attempted to persist labels"):
        accumulator.append(
            predictions=np.asarray([0, 1], dtype=np.uint8),
            probabilities=np.asarray([0.2, 0.8], dtype=np.float32),
            metadata={"labels": [0, 1]},
        )

    decision = publication_decision_payload(descriptive_summary_hash=_sha("summary"))
    assert_non_adoptive_payload(decision)
    mutated = dict(decision)
    mutated["may_feed_stage60"] = True
    with pytest.raises(ProtocolError, match="adoptive or fresh-evidence"):
        assert_non_adoptive_payload(mutated)
    assert "tables/model_fits.csv" in REQUIRED_FILES
    assert "tables/target_metrics.csv" not in REQUIRED_FILES
    assert "arrays/target_predictions.npz" not in REQUIRED_FILES
