"""Private training targets for the frozen source-population estimand."""
from collections import Counter
import numpy as np
from ....protocol import ProtocolError


def correction_targets(menus, labels):
    """Return mass totals and within-case targets; caller must not persist them.

    T_ijk=(n_c/S_ck)*1[Y_ij=k]/N_ik. Missing-class contributions
    are zero. Source center identities determine responses, never predictors.
    """
    counts = Counter(m.center_id for m in menus)
    support = {(c, k): sum(m.center_id == c and np.any(y == k)
                           for m, y in zip(menus, labels, strict=True))
               for c in counts for k in (0, 1)}
    if any(n == 0 for n in support.values()):
        raise ProtocolError('HARP v21 correction scope requires both classes in each center.')
    totals = np.zeros((len(menus), 2))
    normalized = []
    source_weights = np.zeros_like(totals)
    for i, (menu, y) in enumerate(zip(menus, labels, strict=True)):
        if y.shape != (len(menu.sample_ids),) or not np.isin(y, (0, 1)).all():
            raise ProtocolError('HARP v21 correction labels are not aligned binary rows.')
        target = np.zeros((len(y), 2))
        for k in (0, 1):
            v = counts[menu.center_id] / support[(menu.center_id, k)]
            present = y == k
            source_weights[i, k] = v
            if np.any(present):
                target[present, k] = 1. / present.sum()
                totals[i, k] = v
        normalized.append(target)
    return totals, tuple(normalized), source_weights
