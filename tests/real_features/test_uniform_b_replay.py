from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.uniform_b_replay.artifacts import (
    paired_case_bootstrap,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_replay.config import (
    BootstrapConfig,
    CANONICAL_A,
    UNIFORM_B,
    load_uniform_b_replay_config,
)
from midogpp_thesis.real_features.classifier_reference.uniform_b_replay.frames import (
    UniformBShardedStore,
)


CONFIG = Path(
    "experiments/midogpp/stages/90_oracles_and_diagnostics/configs/"
    "uniform_b_v3_retrospective_replay_v1.yaml"
)


def test_uniform_b_config_freezes_retrospective_claim_boundary() -> None:
    config = load_uniform_b_replay_config(CONFIG)

    assert config.name == "uniform_b_v3_retrospective_replay_v1"
    assert config.heldout_centers == ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    assert config.bootstrap == BootstrapConfig(seed=42, valid_replicates=2000, max_attempts=20000)


def test_uniform_b_config_rejects_claim_promotion(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace(
        "adoption_eligible: false", "adoption_eligible: true"
    )
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ProtocolError, match="claim boundary"):
        load_uniform_b_replay_config(path)


def test_uniform_b_store_loads_only_a_and_b(tmp_path: Path) -> None:
    root = tmp_path / "b"
    for center, offset in (("0", 0), ("1", 4)):
        path = root / "embeddings/by_center" / f"center_{center}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = [
            {
                "sample_id": f"{center}-{index}",
                "case_id": f"case-{center}",
                "label": index % 2,
                "center": center,
                "contract_row_index": offset + index,
            }
            for index in range(4)
        ]
        torch.save(
            {
                "canonical_a_embeddings": torch.as_tensor(
                    np.zeros((4, 2560), dtype=np.float32)
                ),
                "embeddings": torch.as_tensor(
                    np.zeros((4, 3840), dtype=np.float32)
                ),
                "metadata": metadata,
            },
            path,
        )
    (root / "c_11520").mkdir()
    store = UniformBShardedStore(root)

    source = store.source_frame(heldout="0", eligible_centers=("0", "1"))
    target = store.target_frame("0")

    assert set(source.embeddings) == {CANONICAL_A, UNIFORM_B}
    assert set(target.centers) == {"0"}
    assert store.access_log == [("source_outer_0", "1"), ("target_outer_0", "0")]


def test_uniform_b_bootstrap_is_paired_and_marks_retrospective_scope() -> None:
    rows = []
    for role, predictions in (("canonical_a", (0, 1, 0, 1)), ("uniform_b", (0, 1, 1, 1))):
        for index, (label, prediction) in enumerate(zip((0, 1, 1, 0), predictions)):
            rows.append(
                {
                    "heldout_center": "0",
                    "role": role,
                    "case_id": f"case-{index // 2}",
                    "label": label,
                    "prediction": prediction,
                }
            )
    result = paired_case_bootstrap(
        rows,
        config=BootstrapConfig(seed=42, valid_replicates=20, max_attempts=200),
    )

    assert result["status"] == "PASS"
    assert result["valid_replicates"] == 20
    assert result["covers_representation_choice_uncertainty"] is False
    assert result["covers_new_center_uncertainty"] is False
