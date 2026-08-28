"""Frozen identities for the OE-PPUR v3 source-supervision bundle."""

from __future__ import annotations

from ..identity import SOURCE_SUPERVISION_REQUIRED_MEMBERS


SOURCE_CACHE_ARTIFACT_ID = "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
SOURCE_SPLIT = "train"
SOURCE_REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
SOURCE_FEATURE_DIM = 3_840
DERIVED_FEATURE_DIM = 6

RAW_SOURCE_ROW_COUNT = 9_648
RAW_SOURCE_CASE_COUNT = 216
HELD_POOL_BLOCK_COUNT = 72
LOGICAL_SOURCE_ROW_COUNT = 77_184
LOGICAL_SOURCE_CASE_GROUP_COUNT = 1_728
PROBABILITY_COLUMN_COUNT = 7
PROBABILITY_DTYPE = "<f4"

SOURCE_CACHE_FILE_HASHES = (
    ("embeddings/train.pt", "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"),
    ("manifests/frozen_cache_protocol.json", "a4faf27a427cfb424789e5592048aa748a057f37124566d46b8b6c557e2bfe69"),
    ("manifests/content_index.json", "307991668f11454da69e3798feb23a2e899e1a00c2ee5132b031e7f7fb9ab82e"),
    ("reports/cache_builder_report.json", "3e3c40449196dc6db9fe0ab982defa86afb1094e3d958e944875396bc363b0ec"),
    ("reports/validation_report.json", "e8b69f557ea92ac8e70a20e504150aba1c947f2b47f735b34e3ca7147efcf6b7"),
)

SOURCE_SUPERVISION_MEMBERS = tuple(SOURCE_SUPERVISION_REQUIRED_MEMBERS)
INDEXED_MEMBERS = SOURCE_SUPERVISION_MEMBERS[:4]
SOURCE_ROW_COLUMNS = (
    "matrix_row_index",
    "outer_target_center",
    "query_center",
    "source_cache_row_index",
    "source_row_id",
    "case_id",
    "split",
    "outcome",
)

# Each block proves which exact label-free arrays were fed to the compiler.
BASE_PROTECTED_IDS = ("B", "U")
BASE_CANDIDATE_PREFIX = "A1::source="


__all__ = tuple(name for name in globals() if name.isupper())
