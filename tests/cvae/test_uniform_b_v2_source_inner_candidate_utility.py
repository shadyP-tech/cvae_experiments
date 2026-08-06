from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.source_inner_utility.bundle import (
    read_prediction_arrays,
    write_prediction_arrays,
)
from midogpp_thesis.cvae.routing.source_inner_utility.cache_inputs import (
    ScoringLabels,
    UnlabeledValidationFrame,
    open_scoring_labels,
    read_manifest_evaluation_index,
)
from midogpp_thesis.cvae.routing.source_inner_utility.config import (
    load_source_inner_utility_config,
)
from midogpp_thesis.cvae.routing.source_inner_utility.contracts import (
    CENTERS,
    CLAIM_SCOPE,
    EXPERIMENT_ID,
    EXPECTED_CONFIG_CONTRACT_HASH,
    OUTPUT_ARTIFACT_ID,
    OUTPUT_SEMANTIC_IDENTITIES,
    POLICY_CONSUMPTION_LOCK_HASH,
    TRAINING_SEEDS,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    EvaluationRow,
    SourceIdentity,
    policy_consumption_lock_payload,
)
from midogpp_thesis.cvae.routing.source_inner_utility.runner import (
    run_source_inner_candidate_utility,
)
from midogpp_thesis.cvae.routing.source_inner_utility.scoring import (
    CASE_CONFUSION_COLUMNS,
    FIT_COLUMNS,
    UTILITY_COLUMNS,
    reconstruct_metrics_from_case_confusions,
    generated_block_sha256,
    run_label_free_prediction_pass,
    score_prediction_pass,
)
from midogpp_thesis.cvae.routing.source_inner_utility import workspace_binding
from midogpp_thesis.cvae.routing.source_inner_utility.workspace_binding import (
    INPUT_IDS,
    validate_production_workspace_binding,
)


CONFIG_PATH = Path(
    "experiments/midogpp/stages/60_routing_and_composition/configs/"
    "uniform_b_v2_source_inner_candidate_utility_v1.yaml"
)
GENERATION_SEEDS = (17, 42, 101)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frame() -> tuple[UnlabeledValidationFrame, np.ndarray]:
    rows: list[EvaluationRow] = []
    labels: list[int] = []
    embeddings: list[list[float]] = []
    ordinal = 0
    for center_ordinal, center in enumerate(CENTERS):
        for label in (0, 1):
            rows.append(
                EvaluationRow(
                    row_ordinal=ordinal,
                    manifest_row_index=100 + ordinal,
                    sample_id=f"sample-{center}-{label}",
                    case_id=f"case-{center}",
                    center=center,
                    split="val",
                    cache_shard_path=f"embeddings/by_center/center_{center}.pt",
                    cache_row_index=label,
                )
            )
            labels.append(label)
            embeddings.append(
                [float(label), float(center_ordinal), float(label + 1), -1.0]
            )
            ordinal += 1
    return (
        UnlabeledValidationFrame(
            embeddings=np.asarray(embeddings, dtype=np.float32),
            rows=tuple(rows),
            cache_binding={"labels_present": False},
        ),
        np.asarray(labels, dtype=np.uint8),
    )


def _identities() -> dict[tuple[str, int], SourceIdentity]:
    result: dict[tuple[str, int], SourceIdentity] = {}
    for center in CENTERS:
        for training_seed in TRAINING_SEEDS:
            prefix = f"{center}-{training_seed}"
            result[(center, training_seed)] = SourceIdentity(
                source_center=center,
                training_seed=training_seed,
                expert_lock_hash=_sha(f"expert-{prefix}"),
                checkpoint_hash=_sha(f"checkpoint-contract-{prefix}"),
                checkpoint_file_sha256=_sha(f"checkpoint-file-{prefix}"),
                frame_hash=_sha(f"frame-contract-{prefix}"),
                frame_file_sha256=_sha(f"frame-file-{prefix}"),
                sampler_state_hash=_sha(f"sampler-contract-{prefix}"),
                sampler_file_sha256=_sha(f"sampler-file-{prefix}"),
            )
    return result


def _generation_keys(
    identities: dict[tuple[str, int], SourceIdentity],
) -> tuple[SourceGenerationKey, ...]:
    return tuple(
        SourceGenerationKey(
            source_center=center,
            training_seed=training_seed,
            generation_seed=generation_seed,
            expert_lock_hash=identities[(center, training_seed)].expert_lock_hash,
            stream_id=_sha(f"stream-{center}-{training_seed}-{generation_seed}")[:16],
            class_seed_by_label={"0": generation_seed, "1": generation_seed + 1},
            max_samples_per_class=1024,
            equal_union_prefix_per_class=128,
        )
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )


@dataclass
class _Expert:
    source_center: str
    training_seed: int
    expert_lock_hash: str
    checkpoint_hash: str


def test_config_freezes_narrow_manifest_alias_and_consumption_rule() -> None:
    config = load_source_inner_utility_config(CONFIG_PATH)

    assert config.contract_hash == EXPECTED_CONFIG_CONTRACT_HASH
    assert config.validation_manifest_artifact_id == VALIDATION_MANIFEST_ARTIFACT_ID
    assert str(config.manifest_path).endswith(
        f"{VALIDATION_MANIFEST_ARTIFACT_ID}/manifest.csv"
    )
    assert config.policy_consumption_lock == policy_consumption_lock_payload()
    assert POLICY_CONSUMPTION_LOCK_HASH == "6c18c72a017403a7"
    assert config.claim_boundary["policy_selection_performed"] is False
    assert config.claim_boundary["outer_target_instantiated"] is False
    assert config.generation_device == "cuda:0"
    assert config.classifier_device == "cpu"
    assert config.threads_per_fit == 1


def test_workspace_binding_requires_four_selection_authorized_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_source_inner_utility_config(CONFIG_PATH)
    roots = {artifact_id: tmp_path / artifact_id for artifact_id in INPUT_IDS}
    output_root = tmp_path / OUTPUT_ARTIFACT_ID
    configured = replace(
        config,
        artifact_root=output_root,
        bank_root=roots[INPUT_IDS[0]],
        generation_lock_root=roots[INPUT_IDS[1]],
        validation_cache_root=roots[INPUT_IDS[2]],
        manifest_path=roots[INPUT_IDS[3]] / "manifest.csv",
    )
    artifacts = {
        artifact_id: SimpleNamespace(may_feed_deployable_selection=True)
        for artifact_id in INPUT_IDS
    }
    artifacts[OUTPUT_ARTIFACT_ID] = SimpleNamespace(
        claim_scope=CLAIM_SCOPE,
        may_feed_deployable_selection=True,
        semantic_identities=OUTPUT_SEMANTIC_IDENTITIES,
    )

    class _Workspace:
        stages = {
            "60_routing_and_composition": {
                "allowed_claim_scopes": [CLAIM_SCOPE],
                "allowed_input_claim_scopes": [CLAIM_SCOPE],
            }
        }

        def __init__(self) -> None:
            self.artifacts = artifacts

        def validate(self) -> None:
            return None

        def get_experiment(self, experiment_id: str) -> SimpleNamespace:
            assert experiment_id == EXPERIMENT_ID
            return SimpleNamespace(
                status="active",
                stage="60_routing_and_composition",
                claim_scope=CLAIM_SCOPE,
                output_artifact_id=OUTPUT_ARTIFACT_ID,
                input_artifact_ids=INPUT_IDS,
            )

        def resolve_artifact(
            self,
            artifact_id: str,
            *,
            for_output: bool = False,
            require_exists: bool = True,
        ) -> Path:
            del require_exists
            if for_output:
                assert artifact_id == OUTPUT_ARTIFACT_ID
                return output_root
            return roots[artifact_id]

    fake = _Workspace()
    monkeypatch.setattr(
        workspace_binding,
        "MidogppWorkspace",
        SimpleNamespace(load=lambda: fake),
    )
    validate_production_workspace_binding(configured)

    artifacts[VALIDATION_MANIFEST_ARTIFACT_ID] = SimpleNamespace(
        may_feed_deployable_selection=False
    )
    with pytest.raises(ProtocolError, match="workspace binding drifted"):
        validate_production_workspace_binding(configured)


def test_manifest_index_is_label_blind_until_explicit_scoring_join(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    eligible_lines = [
        f"s-{center}-{label},c-{center},{center},val,{label}"
        for center in CENTERS
        for label in (0, 1)
    ]
    manifest.write_text(
        "sample_id,case_id,center,split,label\n"
        + "\n".join(eligible_lines)
        + "\ntrain-bad,ct,0,train,NOT_A_LABEL\n"
        + "test-bad,cz,0,test,NOT_A_LABEL\n"
        + "excluded-bad,c4,4,val,NOT_A_LABEL\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identities = read_manifest_evaluation_index(
        manifest,
        expected_sha256=digest,
        expected_rows=18,
        expected_cases=9,
    )
    evaluation_rows = tuple(
        EvaluationRow(
            row_ordinal=ordinal,
            manifest_row_index=row.manifest_row_index,
            sample_id=row.sample_id,
            case_id=row.case_id,
            center=row.center,
            split=row.split,
                cache_shard_path=f"embeddings/by_center/center_{row.center}.pt",
                cache_row_index=ordinal % 2,
        )
        for ordinal, row in enumerate(identities)
    )

    labels = open_scoring_labels(
        manifest,
        evaluation_rows,
        expected_sha256=digest,
    )

    assert not hasattr(identities[0], "label")
    assert labels.labels.tolist() == [0, 1] * len(CENTERS)
    assert labels.consumed_split == "val"


def test_label_free_full_prediction_surface_and_scoring_geometry(tmp_path: Path) -> None:
    config = load_source_inner_utility_config(CONFIG_PATH)
    frame, labels = _frame()
    identities = _identities()
    generation_keys = _generation_keys(identities)
    loads: list[tuple[str, int]] = []
    generated: list[tuple[str, int, int]] = []
    fitted: list[int] = []

    def load_expert(
        _root: str | Path,
        *,
        source_center: str,
        training_seed: int,
        device: str,
    ) -> _Expert:
        assert device == "cpu"
        loads.append((source_center, training_seed))
        identity = identities[(source_center, training_seed)]
        return _Expert(
            source_center=source_center,
            training_seed=training_seed,
            expert_lock_hash=identity.expert_lock_hash,
            checkpoint_hash=identity.checkpoint_hash,
        )

    def generate_block(
        _expert: _Expert,
        key: SourceGenerationKey,
        *,
        per_class: int,
        device: str,
    ) -> SimpleNamespace:
        assert per_class == 2
        assert device == "cpu"
        generated.append((key.source_center, key.training_seed, key.generation_seed))
        embeddings = np.asarray(
            [
                [0.0, 0.0, 1.0, -1.0],
                [0.0, 0.1, 1.0, -1.0],
                [1.0, 0.0, 2.0, -1.0],
                [1.0, 0.1, 2.0, -1.0],
            ],
            dtype=np.float32,
        )
        # ``generate_source_block`` hashes the frozen int64 label array.  The
        # utility adapter may cast a copy for sklearn only after verifying
        # provenance against these original bytes.
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        return SimpleNamespace(
            key=key,
            embeddings=embeddings,
            labels=labels,
            output_sha256=generated_block_sha256(embeddings, labels),
        )

    def fit_classifier(
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        eval_embeddings: np.ndarray,
        *,
        spec: object,
    ) -> SimpleNamespace:
        assert train_embeddings.shape == (4, 4)
        assert train_labels.tolist() == [0, 0, 1, 1]
        assert eval_embeddings.shape == (18, 4)
        fitted.append(len(fitted))
        prediction = (eval_embeddings[:, 0] >= 0.5).astype(np.uint8)
        positive = np.where(prediction == 1, 0.9, 0.1)
        return SimpleNamespace(
            predictions=prediction,
            probabilities=np.stack((1.0 - positive, positive), axis=1),
            classes=(0, 1),
            n_iter=(7,),
            converged=True,
            classifier_config_hash=getattr(spec, "config_hash"),
            scaler_state_hash=_sha(f"scaler-{len(fitted)}")[:16],
        )

    prediction_pass = run_label_free_prediction_pass(
        frame,
        bank_root=tmp_path,
        classifier_spec=config.classifier,
        generation_keys=generation_keys,
        source_identities=identities,
        per_class=2,
        device="cpu",
        threads_per_fit=1,
        expert_loader=load_expert,
        block_generator=generate_block,
        classifier_fitter=fit_classifier,
    )

    assert len(loads) == 27
    assert len(set(loads)) == 27
    assert len(generated) == 81
    assert len(fitted) == 81
    assert prediction_pass.y_pred.shape == (81, 18)
    assert prediction_pass.prob_pos.shape == (81, 18)
    assert all(set(row) == set(FIT_COLUMNS) for row in prediction_pass.fit_rows)
    assert all(row["all_eval_row_count"] == 18 for row in prediction_pass.fit_rows)
    assert all(
        row["eval_labels_available_to_fit_or_predict"] is False
        for row in prediction_pass.fit_rows
    )

    prediction_path = tmp_path / "candidate_predictions.npz"
    write_prediction_arrays(prediction_path, prediction_pass)
    persisted_prediction, persisted_probability = read_prediction_arrays(prediction_path)
    assert np.array_equal(persisted_prediction, prediction_pass.y_pred)
    assert np.array_equal(persisted_probability, prediction_pass.prob_pos)
    with np.load(prediction_path, allow_pickle=False) as payload:
        assert set(payload.files) == {"y_pred", "prob_pos"}

    utility_rows, case_rows = score_prediction_pass(
        prediction_pass,
        ScoringLabels(
            labels=labels,
            evaluation_order_hash=prediction_pass.evaluation_order_hash,
            manifest_sha256=_sha("synthetic-manifest"),
        ),
    )

    assert len(utility_rows) == 648
    assert len(case_rows) == 648
    assert all(set(row) == set(UTILITY_COLUMNS) for row in utility_rows)
    assert all(set(row) == set(CASE_CONFUSION_COLUMNS) for row in case_rows)
    assert all(
        row["pseudo_target_center"] != row["candidate_source_center"]
        for row in utility_rows
    )
    assert all(row["outer_target_instantiated"] is False for row in utility_rows)
    assert all(row["policy_selection_performed"] is False for row in utility_rows)
    assert all(row["bacc"] == 1.0 and row["macro_f1"] == 1.0 for row in utility_rows)
    first_id = str(utility_rows[0]["utility_row_id"])
    first_cases = [row for row in case_rows if row["utility_row_id"] == first_id]
    assert reconstruct_metrics_from_case_confusions(first_cases) == (1.0, 1.0)


def test_generated_block_hash_preserves_frozen_int64_label_dtype() -> None:
    embeddings = np.asarray(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    frozen_labels = np.asarray([0, 1], dtype=np.int64)

    frozen_hash = generated_block_sha256(embeddings, frozen_labels)
    classifier_copy_hash = generated_block_sha256(
        embeddings,
        frozen_labels.astype(np.uint8),
    )

    assert frozen_hash != classifier_copy_hash
    assert len(frozen_hash) == 64


def test_prediction_array_reader_rejects_persisted_labels(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        y_pred=np.zeros((1, 1), dtype=np.uint8),
        prob_pos=np.zeros((1, 1), dtype=np.float32),
        y_true=np.zeros((1,), dtype=np.uint8),
    )

    with pytest.raises(ProtocolError, match="keys drifted"):
        read_prediction_arrays(path)


def test_stale_complete_artifact_is_demoted_to_failed(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    (root / "reports").mkdir(parents=True)
    (root / "config.resolved.yaml").write_text("{}\n", encoding="utf-8")
    (root / "provenance").mkdir()
    (root / "provenance/input_artifacts.json").write_text("{}\n", encoding="utf-8")
    state_path = root / "reports/run_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "midogpp_uniform_b_v2_source_inner_utility_run_state_v1"
                ),
                "experiment_id": (
                    "midogpp.routing_and_composition."
                    "uniform_b_v2_source_inner_candidate_utility.v1"
                ),
                "claim_scope": "routing_and_composition",
                "status": "COMPLETE",
                "selection_performed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolError, match="incomplete"):
        run_source_inner_candidate_utility(
            SimpleNamespace(artifact_root=root),  # type: ignore[arg-type]
            artifact_root=root,
        )

    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "FAILED"
