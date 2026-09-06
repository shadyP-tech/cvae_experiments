"""Permitted own-case predictors; identities never enter a design matrix."""
import numpy as np
from scipy.special import logit

from ....protocol import ProtocolError
from ..contracts import decode_probability_hex

PATCH_DIMENSION = 3840
PATCH_SCHEMA = 'harp_v21_canonical_virchow2_3840_float32'
VARIANTS = ('baseline', 'calibrated_baseline', 'embedding_residual')
CLIP = 1e-6


def sketch_virchow2(features):
    """Compatibility name: validate and retain every canonical coordinate."""
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != PATCH_DIMENSION or not np.isfinite(x).all():
        raise ProtocolError('HARP v21 requires canonical finite Virchow2_3840 rows.')
    result = np.array(x, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(result).all():
        raise ProtocolError('HARP v21 Virchow2_3840 conversion overflowed.')
    result.setflags(write=False)
    return result


def patch_array(menu):
    x = np.asarray(menu.patch_features, dtype=np.float32)
    if x.shape != (len(menu.sample_ids), PATCH_DIMENSION) or not np.isfinite(x).all():
        raise ProtocolError('HARP v21 requires sample-aligned, sealed patch features.')
    return x


def baseline(menu):
    return np.asarray(decode_probability_hex(menu.baseline_probability_hex), dtype=float)


def raw_design(menu, variant):
    if variant not in VARIANTS:
        raise ProtocolError('HARP v21 correction evidence variant is not predeclared.')
    x = patch_array(menu)  # Validate even baseline controls before labels open.
    b = np.clip(baseline(menu), CLIP, 1-CLIP)
    logits = logit(b)[:, None]
    return np.column_stack((logits, x)) if variant == 'embedding_residual' else logits


def standardized(raw, means, scales, variant):
    # Coordinate-wise scaling preserves sparse discriminative information.
    # Ridge is fixed on standardized coefficients, without width dilution.
    return (raw - np.asarray(means)) / np.asarray(scales)


def case_design(menu):
    b = baseline(menu)
    return np.asarray([1., b.mean(), (b >= .5).mean(), b.std(),
                       np.mean(np.abs(b-.5)), np.log1p(len(b))/10.])


def full_scope_case_weights(menus):
    from collections import Counter
    counts = Counter(m.center_id for m in menus)
    return np.asarray([1./(len(counts)*counts[m.center_id]) for m in menus])
