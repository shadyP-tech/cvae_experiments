"""Train-case-only PCA frames shared by canonical-B CVAE workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..common.hashing import stable_hash
from .protocol import ProtocolError


@dataclass(frozen=True)
class PCAState:
    start: int
    stop: int
    output_dim: int
    scaler_mean: object
    scaler_scale: object
    pca_mean: object
    pca_components: object
    explained_variance: object
    explained_variance_ratio_sum: float

    def to_payload(self) -> dict[str, object]:
        def convert(value: object) -> object:
            return value.tolist() if hasattr(value, "tolist") else value

        return {
            "start": self.start,
            "stop": self.stop,
            "output_dim": self.output_dim,
            "scaler_mean": convert(self.scaler_mean),
            "scaler_scale": convert(self.scaler_scale),
            "pca_mean": convert(self.pca_mean),
            "pca_components": convert(self.pca_components),
            "explained_variance": convert(self.explained_variance),
            "explained_variance_ratio_sum": self.explained_variance_ratio_sum,
        }


@dataclass(frozen=True)
class PilotFeatureFrame:
    arm: str
    input_dim: int
    output_dim: int
    blocks: tuple[PCAState, ...]
    fit_sample_hash: str

    @property
    def state_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_b_adaptation_feature_frame_v1",
            "arm": self.arm,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "fit_sample_hash": self.fit_sample_hash,
            "blocks": [block.to_payload() for block in self.blocks],
        }

    def transform(self, embeddings: Sequence[Sequence[float]]) -> object:
        import numpy as np

        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] < self.input_dim:
            raise ProtocolError("Pilot frame transform input has the wrong dimension.")
        outputs = []
        for block in self.blocks:
            raw = x[:, block.start : block.stop]
            scaled = (raw - np.asarray(block.scaler_mean)) / np.asarray(
                block.scaler_scale
            )
            outputs.append(
                (scaled - np.asarray(block.pca_mean))
                @ np.asarray(block.pca_components).T
            )
        transformed = np.concatenate(outputs, axis=1).astype(np.float32)
        if transformed.shape[1] != self.output_dim:
            raise ProtocolError("Pilot frame produced an unexpected output dimension.")
        return transformed

    def inverse_transform(self, projected: Sequence[Sequence[float]]) -> object:
        import numpy as np

        z = np.asarray(projected, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.output_dim:
            raise ProtocolError("Pilot frame inverse input has the wrong dimension.")
        raw_blocks = []
        cursor = 0
        for block in self.blocks:
            block_z = z[:, cursor : cursor + block.output_dim]
            scaled = (
                block_z @ np.asarray(block.pca_components)
                + np.asarray(block.pca_mean)
            )
            raw_blocks.append(
                scaled * np.asarray(block.scaler_scale)
                + np.asarray(block.scaler_mean)
            )
            cursor += block.output_dim
        return np.concatenate(raw_blocks, axis=1).astype(np.float32)


def fit_pilot_frame(
    arm: str,
    fit_embeddings: Sequence[Sequence[float]],
    *,
    fit_sample_hash: str,
) -> PilotFeatureFrame:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(fit_embeddings, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3840 or len(x) < 128:
        raise ProtocolError("Pilot PCA requires at least 128 canonical-B fit rows.")
    layouts = {
        "a_global_pca128": ((0, 2560, 128),),
        "b_joint_pca128": ((0, 3840, 128),),
        "b_block_pca96_32": ((0, 2560, 96), (2560, 3840, 32)),
        "b_block_pca192_64": ((0, 2560, 192), (2560, 3840, 64)),
    }
    if arm not in layouts:
        raise ProtocolError(f"Unknown pilot arm: {arm!r}")
    required_rows = max(output_dim for _, _, output_dim in layouts[arm])
    if len(x) < required_rows:
        raise ProtocolError(
            f"Pilot PCA arm {arm!r} requires at least {required_rows} fit rows."
        )
    blocks = []
    for start, stop, output_dim in layouts[arm]:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x[:, start:stop])
        pca = PCA(n_components=output_dim, svd_solver="randomized", random_state=0)
        pca.fit(scaled)
        blocks.append(
            PCAState(
                start=start,
                stop=stop,
                output_dim=output_dim,
                scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
                pca_mean=np.asarray(pca.mean_, dtype=np.float64),
                pca_components=np.asarray(pca.components_, dtype=np.float64),
                explained_variance=np.asarray(
                    pca.explained_variance_, dtype=np.float64
                ),
                explained_variance_ratio_sum=float(
                    pca.explained_variance_ratio_.sum()
                ),
            )
        )
    return PilotFeatureFrame(
        arm=arm,
        input_dim=2560 if arm == "a_global_pca128" else 3840,
        output_dim=sum(block.output_dim for block in blocks),
        blocks=tuple(blocks),
        fit_sample_hash=str(fit_sample_hash),
    )


def bridge_a_prefix(
    b_embeddings: object,
    a_embeddings: object,
    *,
    minimum_cosine: float = 0.99999,
    maximum_relative_l2: float = 0.001,
) -> dict[str, object]:
    import numpy as np

    b = np.asarray(b_embeddings, dtype=np.float64)
    a = np.asarray(a_embeddings, dtype=np.float64)
    if b.ndim != 2 or a.ndim != 2 or b.shape[0] != a.shape[0]:
        raise ProtocolError("A/B bridge arrays are not row aligned.")
    if b.shape[1] != 3840 or a.shape[1] != 2560:
        raise ProtocolError("A/B bridge dimensions must be 3840 and 2560.")
    prefix = b[:, :2560]
    dot = np.sum(prefix * a, axis=1)
    denom = np.linalg.norm(prefix, axis=1) * np.linalg.norm(a, axis=1)
    cosine = dot / np.maximum(denom, 1e-12)
    relative = np.linalg.norm(prefix - a, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1), 1e-12
    )
    result = {
        "schema_version": "midogpp_uniform_b_a_prefix_bridge_v1",
        "n_rows": int(len(a)),
        "minimum_cosine": float(cosine.min()),
        "maximum_relative_l2": float(relative.max()),
        "required_minimum_cosine": float(minimum_cosine),
        "required_maximum_relative_l2": float(maximum_relative_l2),
    }
    result["status"] = (
        "PASS"
        if result["minimum_cosine"] >= minimum_cosine
        and result["maximum_relative_l2"] <= maximum_relative_l2
        else "FAIL"
    )
    if result["status"] != "PASS":
        raise ProtocolError("Canonical-B prefix failed the canonical-A numerical bridge.")
    return result


__all__ = (
    "PCAState",
    "PilotFeatureFrame",
    "bridge_a_prefix",
    "fit_pilot_frame",
)
