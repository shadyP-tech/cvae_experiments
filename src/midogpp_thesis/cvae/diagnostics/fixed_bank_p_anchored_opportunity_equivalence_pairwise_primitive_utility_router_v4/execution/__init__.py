"""Dependency-light execution namespace for OE-PPUR v4.

Concrete contracts are imported from their defining submodules.  Keeping this
initializer empty prevents configuration, admission, and service modules from
gaining circular import edges.
"""

__all__: tuple[str, ...] = ()
