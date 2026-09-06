"""Label-independent fit/calibration partition for an unchanged complete selector."""
from .splitting import center_stratified_folds


def proposer_calibration_partition(case_keys):
    """Reserve approximately one third of cases; never refit after calibration.

    Center-stratified round robin keeps the split deterministic even for the
    tiny construction fixtures. Production scopes have multiple cases per center.
    """
    folds = center_stratified_folds(case_keys, fold_count=3,
                                   namespace="HARP_V21_FROZEN_SELECTOR_CALIBRATION")
    calibration = folds[0]
    fitting = tuple(sorted(key for fold in folds[1:] for key in fold))
    return fitting, calibration
