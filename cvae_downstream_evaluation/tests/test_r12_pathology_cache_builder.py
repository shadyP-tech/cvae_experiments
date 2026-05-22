from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvae_downstream_evaluation.pathology_cache_builder import (  # noqa: E402
    CacheBuildRequest,
    assert_cache_payload,
    build_r12_pathology_embedding_cache,
    canonical_cache_metadata,
    parse_csv_list,
    read_manifest_rows_by_split,
    resolve_manifest_image_path,
)
from cvae_downstream_evaluation.protocol import ProtocolError  # noqa: E402


def test_manifest_reader_preserves_split_rows_and_resolves_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manifest = (
        repo
        / "cvae_testing"
        / "outputs"
        / "camelyon17"
        / "run"
        / "seed42"
        / "manifests"
        / "samples.csv"
    )
    manifest.parent.mkdir(parents=True)
    image = repo / "data" / "patch.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-used")
    manifest.write_text(
        "sample_id,image_path,label,label_name,magnification,domain,patient_id,split\n"
        "s1,data/patch.png,1,tumor,0,center_0,p1,train\n"
        "s2,data/patch.png,0,normal,1,center_1,p2,test\n",
        encoding="utf-8",
    )

    rows = read_manifest_rows_by_split(manifest, splits=("train", "test"))
    assert rows["train"][0]["sample_id"] == "s1"
    assert rows["train"][0]["image_path"] == str(image)
    assert rows["test"][0]["center"] == "1"


def test_canonical_cache_metadata_adds_center_and_numeric_label() -> None:
    row = {
        "sample_id": "s1",
        "image_path": "/tmp/img.png",
        "label": "1",
        "magnification": "4",
        "split": "train",
    }
    meta = canonical_cache_metadata(row, split="train")
    assert meta["label"] == 1
    assert meta["center"] == "4"
    assert meta["split"] == "train"


def test_assert_cache_payload_rejects_missing_metadata_key() -> None:
    torch = __import__("torch")
    payload = {
        "embeddings": torch.zeros((1, 2)),
        "metadata": [{"sample_id": "s1", "image_path": "/tmp/img.png", "label": 1, "split": "train"}],
    }
    try:
        assert_cache_payload(payload, expected_rows=1, split="train")
    except ProtocolError as exc:
        assert "magnification" in str(exc)
    else:
        raise AssertionError("cache payload without magnification was accepted")


def test_parse_csv_list_uses_default_and_strips_values() -> None:
    assert parse_csv_list(None, default=("train", "test")) == ("train", "test")
    assert parse_csv_list(" train, test ", default=("x",)) == ("train", "test")


def test_cache_builder_dry_run_reports_expected_paths(tmp_path: Path) -> None:
    support_run = tmp_path / "support" / "seed42"
    manifest = support_run / "manifests" / "samples.csv"
    manifest.parent.mkdir(parents=True)
    image = tmp_path / "image.png"
    image.write_bytes(b"not-used")
    manifest.write_text(
        "sample_id,image_path,label,label_name,magnification,domain,patient_id,split\n"
        f"s1,{image},1,tumor,0,center_0,p1,train\n"
        f"s2,{image},0,normal,0,center_0,p2,val\n"
        f"s3,{image},1,tumor,0,center_0,p3,test\n",
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    request = CacheBuildRequest(
        backbone_name="phikon",
        model_dir=model_dir,
        experiment_seed=42,
        support_run_dir=support_run,
        output_root=tmp_path / "pathology_embeddings",
        dry_run=True,
    )

    result = build_r12_pathology_embedding_cache(request)
    assert result.status == "dry_run_passed"
    assert result.split_counts == {"train": 1, "val": 1, "test": 1}
    assert result.output_paths["train"].name == "train.pt"
    assert "phikon" in str(result.output_paths["train"])


def test_relative_manifest_path_requires_repo_marker(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    manifest = repo / "cvae_testing" / "outputs" / "x" / "manifests" / "samples.csv"
    manifest.parent.mkdir(parents=True)
    resolved = resolve_manifest_image_path(manifest, "data/a.png")
    assert resolved == repo / "data" / "a.png"
