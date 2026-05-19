"""Entrypoint for Family E1.1 PCA-64 GMM downstream diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_family_e1_direct_embedding_sampler_downstream import main  # noqa: E402


if __name__ == "__main__":
    main()
