"""Resolve immutable contract paths after the repository ownership migration."""

from __future__ import annotations

from pathlib import Path


ORIGINAL_ANNOTATION_PATCH_ROOT = Path(
    "datasets/midogpp/artifacts/midogpp_annotation_patch_v1"
)
CANONICAL_ANNOTATION_PATCH_ROOT = Path(
    "datasets/midogpp/contract/annotation_patch_v1"
)


def resolve_contract_path(repo_root: Path, raw_path: str | Path) -> Path:
    """Resolve a frozen repo-relative path without rewriting its hashed manifest.

    The annotation-patch v1 manifest is immutable thesis evidence. Its image
    path strings record the original ownership root, while the bytes now live
    under the canonical contract tree. Only this exact audited prefix is
    relocated; arbitrary missing paths remain missing.
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path
    root = Path(repo_root).resolve()
    direct = (root / path).resolve()
    if direct.exists():
        return direct
    try:
        relative = path.relative_to(ORIGINAL_ANNOTATION_PATCH_ROOT)
    except ValueError:
        return direct
    return (root / CANONICAL_ANNOTATION_PATCH_ROOT / relative).resolve()
