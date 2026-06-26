"""Virchow2-CVAE rebuild package.

The package keeps heavy ML dependencies out of top-level imports so protocol
and artifact tests can run in minimal environments.
"""

from config import RebuildConfig, load_config
from protocol import ProtocolError

__all__ = ["ProtocolError", "RebuildConfig", "load_config"]
