from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch


def resolve_domain_order(configured_domains: Iterable[int]) -> List[int]:
    """Return a deterministic domain order sourced from config values only."""
    order: List[int] = []
    seen = set()
    for raw in configured_domains:
        domain = int(raw)
        if domain in seen:
            continue
        seen.add(domain)
        order.append(domain)
    if not order:
        raise ValueError("Domain order cannot be empty; expected non-empty data.magnifications in config.")
    return order


def domain_to_index_map(domain_order: Sequence[int]) -> Dict[int, int]:
    return {int(domain): idx for idx, domain in enumerate(domain_order)}


def build_domain_one_hot(metadata: List[dict], domain_order: Sequence[int], domain_key: str = "magnification") -> torch.Tensor:
    """Build one-hot metadata vectors using only the configured domain field.

    This function intentionally reads only `domain_key` to avoid accidental leakage
    from labels, split identifiers, or patient-specific fields.
    """
    forbidden = {"label", "label_name", "split", "patient_id"}
    if str(domain_key).strip().lower() in forbidden:
        raise ValueError("Domain one-hot encoding must not use label/split/patient fields.")

    mapping = domain_to_index_map(domain_order)
    n = len(metadata)
    k = len(mapping)
    one_hot = torch.zeros((n, k), dtype=torch.float32)
    for i, item in enumerate(metadata):
        if domain_key not in item:
            raise ValueError(f"Missing domain key '{domain_key}' in metadata item at index {i}.")
        domain = int(item[domain_key])
        if domain not in mapping:
            raise ValueError(
                f"Observed domain '{domain}' at index {i} is not present in configured domain order: {list(domain_order)}"
            )
        one_hot[i, mapping[domain]] = 1.0
    return one_hot
