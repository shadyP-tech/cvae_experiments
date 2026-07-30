"""Exact workspace binding for the constrained Nyström diagnostic."""

from pathlib import Path

from midogpp_thesis.workspace.runtime import MidogppWorkspace

from ..protocol import ProtocolError
from .config import (
    BOUNDED_SHRINKAGE_EXPERIMENT_ID,
    BOUNDED_SHRINKAGE_EXPERIMENT_NAME,
    BOUNDED_SHRINKAGE_OUTPUT_ID,
    CACHE_ID,
    CANONICAL_ID,
    DATASET_ID,
    EXPERIMENT_ID,
    NONLINEAR_ID,
    OUTPUT_ID,
    ROBUST_ID,
    SOURCE_INNER_REPLAY_ID,
    ConstrainedNystroemConfig,
)


INPUT_IDS = (DATASET_ID, CACHE_ID, CANONICAL_ID, NONLINEAR_ID, ROBUST_ID)
BOUNDED_SHRINKAGE_INPUT_IDS = (
    *INPUT_IDS,
    SOURCE_INNER_REPLAY_ID,
)


def validate_production_workspace_binding(
    config: ConstrainedNystroemConfig,
) -> None:
    workspace = MidogppWorkspace.load()
    workspace.validate()
    bounded = config.name == BOUNDED_SHRINKAGE_EXPERIMENT_NAME
    experiment_id = BOUNDED_SHRINKAGE_EXPERIMENT_ID if bounded else EXPERIMENT_ID
    output_id = BOUNDED_SHRINKAGE_OUTPUT_ID if bounded else OUTPUT_ID
    input_ids = BOUNDED_SHRINKAGE_INPUT_IDS if bounded else INPUT_IDS
    experiment = workspace.get_experiment(experiment_id)
    if (
        experiment.status != "diagnostic"
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != output_id
        or experiment.input_artifact_ids != input_ids
    ):
        raise ProtocolError("Constrained-Nyström workspace binding drifted.")
    expected = {
        "root": workspace.resolve_artifact(output_id, for_output=True, require_exists=False),
        "manifest": workspace.resolve_artifact(DATASET_ID, require_exists=True)
        / "manifest.csv",
        "cache": workspace.resolve_artifact(CACHE_ID, require_exists=True)
        / "embeddings/train.pt",
        "canonical": workspace.resolve_artifact(CANONICAL_ID, require_exists=True),
        "nonlinear": workspace.resolve_artifact(NONLINEAR_ID, require_exists=True),
        "robust_lineage": workspace.resolve_artifact(
            ROBUST_ID, require_exists=True
        ),
    }
    if bounded:
        expected["source_inner_replay"] = workspace.resolve_artifact(
            SOURCE_INNER_REPLAY_ID, require_exists=True
        )
    actual = {
        "root": config.artifact_root,
        "manifest": config.manifest_path,
        "cache": config.feature_cache_path,
        "canonical": config.canonical_root,
        "nonlinear": config.nonlinear_root,
        "robust_lineage": config.robust_lineage_root,
        "source_inner_replay": config.source_inner_replay_root,
    }
    for key, value in expected.items():
        if actual[key].resolve() != Path(value).resolve():
            raise ProtocolError(f"Constrained-Nyström input binding drifted: {key}.")
