from __future__ import annotations

from typing import Any, Sequence

from preservation_repair import _hash_strings


def unlabeled_support_split_rows(
    splits: Sequence[Any],
    experiment_seed: int,
    replicate_seed: int,
) -> list[dict[str, object]]:
    return [
        {
            "experiment_seed": int(experiment_seed),
            "replicate_seed": int(replicate_seed),
            "heldout_center": split.heldout_center,
            "support_seed": split.support_seed,
            "support_size": split.support_size,
            "eval_mode": split.eval_mode,
            "support_eval_split_id": split.support_eval_split_id,
            "parent_support32_split_id": split.parent_support32_split_id,
            "support_labels_used": int(split.support_labels_used),
            "support_size_actual": len(split.support_indices),
            "n_target_eval": len(split.eval_indices),
            "support_sample_id_hash": _hash_strings(split.support_sample_ids),
            "eval_sample_id_hash": _hash_strings(split.eval_sample_ids),
            "nested_support_diagnostics": 1,
            "fixed_eval_support_size_diagnostics": int(split.eval_mode == "fixed_support32"),
        }
        for split in splits
    ]


def scoped_unlabeled_support_split_rows(
    splits: Sequence[Any],
    experiment_seed: int,
    support_seed: int,
    scope: str,
) -> list[dict[str, object]]:
    rows = unlabeled_support_split_rows(splits, experiment_seed, support_seed)
    for row in rows:
        row["split_scope"] = scope
    return rows


def labeled_support_split_rows(
    splits: Sequence[Any],
    experiment_seed: int,
    support_seed: int,
    scope: str,
) -> list[dict[str, object]]:
    return [
        {
            "experiment_seed": int(experiment_seed),
            "support_seed": int(support_seed),
            "heldout_center": split.heldout_center,
            "support_size": split.support_size,
            "eval_mode": split.eval_mode,
            "split_scope": scope,
            "support_eval_split_id": split.support_eval_split_id,
            "parent_support32_split_id": split.parent_support32_split_id,
            "support_labels_used": int(split.support_labels_used),
            "support_size_actual": len(split.support_indices),
            "support_count_class0": split.support_labels.count(0),
            "support_count_class1": split.support_labels.count(1),
            "class_balanced_support": int(split.class_balanced_support),
            "n_target_eval": len(split.eval_indices),
            "support_sample_id_hash": _hash_strings(split.support_sample_ids),
            "eval_sample_id_hash": _hash_strings(split.eval_sample_ids),
            "support_eval_disjoint": 1,
            "size_specific_eval_exclusion": int(split.eval_mode == "primary_style"),
            "common_eval_excluding_support32": int(split.eval_mode == "fixed_support32"),
        }
        for split in splits
    ]
