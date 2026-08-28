"""External preparation commands for the sealed OE-PPUR v3 successor.

The initializer deliberately imports nothing.  The source executable must
establish CUDA and BLAS variables before importing NumPy, torch, scikit-learn,
or the sealed producer modules.
"""

__all__: tuple[str, ...] = ()
