from __future__ import annotations

import pickle
from pathlib import Path

from midogpp_thesis.cvae.frozen_policy_downstream import cli
from types import MappingProxyType

from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh import execution
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.bundle import (
    ORACLE_COLUMNS,
    REQUIRED_FILES,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.execution import (
    EXPECTED_PREDICTION_TASK_COUNT,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh.source_cache import (
    EXPECTED_EXPERT_TASK_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    GENERATION_DEVICES,
)
from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh import (
    prediction_tasks,
)


def test_closed_world_static_bundle_contract_is_exact() -> None:
    assert REQUIRED_FILES == (
        "config.resolved.yaml",
        "provenance/input_artifacts.json",
        "manifests/protocol_manifest.json",
        "manifests/policy_binding.json",
        "manifests/evaluation_plan.json",
        "manifests/prediction_seal.json",
        "manifests/content_index.json",
        "checkpoints/source/source_cache.json",
        "checkpoints/predictions/prediction_cache.json",
        "tables/prediction_index.csv",
        "tables/seed_cell_metrics.csv",
        "tables/ensemble_metrics.csv",
        "tables/center_contrasts.csv",
        "tables/contrast_inference.csv",
        "tables/oracle_diagnostics.csv",
        "reports/label_access_report.json",
        "reports/leakage_report.json",
        "reports/publication_decision.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    )
    assert "source_center" not in ORACLE_COLUMNS
    assert "oracle_action" not in ORACLE_COLUMNS
    assert "utility_by_source" not in ORACLE_COLUMNS


def test_workstation_schedule_geometry_is_frozen() -> None:
    assert GENERATION_DEVICES == ("cuda:0", "cuda:1")
    assert EXPECTED_EXPERT_TASK_COUNT == 27
    assert EXPECTED_SOURCE_BLOCK_COUNT == 81
    assert EXPECTED_PREDICTION_TASK_COUNT == 81


def test_fresh_execution_layer_has_no_consumed_stage70_or_stage90_dependency() -> None:
    package = Path(
        "src/midogpp_thesis/cvae/frozen_policy_downstream/residual_topup_fresh"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )
    assert "diagnostics.residual_topup_router" not in source
    assert "diagnostics.dense_residual_router" not in source
    assert "frozen_policy_downstream.runner" not in source
    assert "frozen_policy_downstream.validation" not in source
    assert "frozen_policy_downstream.scoring" not in source
    assert "policy_update_emitted\": True" not in source
    assert "oracle_action_exported\": True" not in source


def test_fresh_cli_dispatch_forwards_local_scratch_opt_in(
    monkeypatch,
    capsys,
) -> None:
    from midogpp_thesis.cvae.frozen_policy_downstream.residual_topup_fresh import (
        config as config_module,
        runner as runner_module,
    )

    sentinel = object()
    observed: dict[str, object] = {}

    def run(config: object, *, enable_optional_local_scratch: bool) -> Path:
        observed["config"] = config
        observed["scratch"] = enable_optional_local_scratch
        return Path("/canonical/fresh-output")

    monkeypatch.setattr(
        config_module,
        "load_residual_topup_fresh_config",
        lambda path: observed.setdefault("path", path) and sentinel,
    )
    monkeypatch.setattr(runner_module, "run_residual_topup_fresh", run)
    assert (
        cli.main(
            [
                "evaluate-residual-topup-fresh",
                "--config",
                "resolved.yaml",
                "--enable-local-scratch",
            ]
        )
        == 0
    )
    assert observed == {
        "path": "resolved.yaml",
        "config": sentinel,
        "scratch": True,
    }
    assert capsys.readouterr().out.strip() == "/canonical/fresh-output"


def test_execution_facade_preserves_the_stable_public_api() -> None:
    assert execution.__all__ == (
        "ACTION_LIBRARY_SCHEMA",
        "ACTION_SCHEMA",
        "EXPECTED_PREDICTION_TASK_COUNT",
        "FrozenPolicySurface",
        "PREDICTION_CACHE_SCHEMA",
        "PREDICTION_INDEX_COLUMNS",
        "PredictionCache",
        "PredictionTaskExecutor",
        "PredictionTaskRecord",
        "PredictionTaskSpec",
        "execute_prediction_task",
        "load_frozen_policy_actions",
        "load_prediction_cache",
        "materialize_prediction_cache",
        "write_prediction_index",
    )


def test_spawn_dispatch_converts_read_only_payload_to_picklable_dict(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, values, *, chunksize):
            dispatched = tuple(values)
            observed["function"] = function
            observed["values"] = dispatched
            observed["chunksize"] = chunksize
            pickle.dumps(dispatched)

    class FakeContext:
        def Pool(self, *, processes):
            observed["processes"] = processes
            return FakePool()

    def fake_get_context(method):
        observed["method"] = method
        return FakeContext()

    monkeypatch.setattr(prediction_tasks.mp, "get_context", fake_get_context)
    task = execution.PredictionTaskSpec(
        MappingProxyType({"task_id": "picklable-task"})
    )

    prediction_tasks.spawn_prediction_tasks((task,))

    assert observed["method"] == "spawn"
    assert observed["processes"] == 4
    assert observed["chunksize"] == 1
    assert observed["values"] == ({"task_id": "picklable-task"},)
