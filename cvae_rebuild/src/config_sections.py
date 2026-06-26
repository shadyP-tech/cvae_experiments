from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from config import _mapping
from preservation_repair import _path


@dataclass(frozen=True)
class ExperimentConfigSections:
    experiment: Mapping[str, Any]
    inputs: Mapping[str, Any]
    run_matrix: Mapping[str, Any]
    generation: Mapping[str, Any]
    classifier: Mapping[str, Any]


def experiment_config_sections(data: Mapping[str, Any]) -> ExperimentConfigSections:
    return ExperimentConfigSections(
        experiment=_mapping(data, "experiment"),
        inputs=_mapping(data, "inputs"),
        run_matrix=_mapping(data, "run_matrix"),
        generation=_mapping(data, "generation"),
        classifier=_mapping(data, "classifier"),
    )


def config_base_dir_for_path(path: str | Path) -> Path:
    source = Path(path).resolve()
    return source.parents[2] if len(source.parents) >= 3 else source.parent


def optional_config_path(base: Path, value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return _path(base, str(value))


def classifier_config_fields(classifier: Mapping[str, Any]) -> dict[str, object]:
    return {
        "classifier_type": str(classifier["type"]),
        "classifier_solver": str(classifier["solver"]),
        "classifier_c": float(classifier["C"]),
        "classifier_max_iter": int(classifier["max_iter"]),
        "classifier_class_weight": str(classifier["class_weight"]),
        "classifier_seed": None if classifier.get("classifier_seed") is None else int(classifier["classifier_seed"]),
    }
