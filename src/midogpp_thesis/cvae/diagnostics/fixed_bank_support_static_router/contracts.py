"""Compatibility import surface for immutable core products."""

from .actions import ActionSpec
from .products import *  # noqa: F403
from .products import __all__ as _PRODUCT_EXPORTS

__all__ = ("ActionSpec", *_PRODUCT_EXPORTS)
