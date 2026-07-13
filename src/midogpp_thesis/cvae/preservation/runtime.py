"""Small preservation runtime identities and assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ...real_features.classifier_reference.artifacts import stable_hash


@dataclass(frozen=True)
class SamplerFitKey:
    checkpoint_hash: str
    source_row_hash: str
    class_label: int
    sampler_rule: str

    @property
    def hash(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class GenerationKey:
    source_state_hash: str
    generation_seed: int
    class_count_vector: tuple[int, int]
    representation_role: str

    @property
    def hash(self) -> str:
        return stable_hash(self.__dict__)


@dataclass(frozen=True)
class EvaluationKey:
    generated_artifact_hash: str
    frozen_classifier_spec_hash: str
    eval_center: str
    eval_row_hash: str
    metric_schema_version: str
    protocol_hash: str

    @property
    def hash(self) -> str:
        return stable_hash(self.__dict__)
