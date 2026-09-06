"""Frozen donor ordering; independent of correction estimation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SupportActionOutcome,
    SupportCaseClassProfile,
    canonical_text,
)
from .hashing import canonical_hash

from .ranker_numerics import _DIRECTIONS
from .ranker_features import FittedFeatureTransform, fit_feature_transform
from .pairwise_ranker import PairwiseRanker, build_pairwise_comparisons, fit_pairwise_ranker


@dataclass(frozen=True, slots=True)
class CaseModelPrediction:
    menu_hash: str
    d01_ranked_action_ids: tuple[str, ...]
    d10_ranked_action_ids: tuple[str, ...]
    model_hash: str

    def public_payload(self) -> dict[str, object]:
        return {"menu_hash":self.menu_hash,"d01_ranked_action_ids":list(self.d01_ranked_action_ids),
                "d10_ranked_action_ids":list(self.d10_ranked_action_ids),"model_hash":self.model_hash,
                "proposal_only":True,"labels_consumed":False}


@dataclass(frozen=True, slots=True)
class ProposalModel:
    transform: FittedFeatureTransform
    ranker: PairwiseRanker
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.transform.transform_hash != self.ranker.transform_hash
            or self.transform.training_case_keys != self.ranker.training_case_keys):
            raise ProtocolError("HARP v21 proposal ranker/transform role binding drifted.")
        object.__setattr__(self,"model_hash",canonical_hash({
            "schema_version":"case_conditional_proposal_model_v21",
            "transform_hash":self.transform.transform_hash,"ranker_hash":self.ranker.ranker_hash,
            "opportunity_heads_used":False}))

    @property
    def training_case_keys(self) -> tuple[tuple[str,str], ...]:
        return self.transform.training_case_keys

    def predict_menu(self, menu: LabelFreeCaseMenu) -> CaseModelPrediction:
        if (menu.center_id,menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v21 honest proposal prediction includes a fitted case.")
        coefficients = np.asarray(self.ranker.coefficients,dtype=np.float64)
        rankings = []
        for direction in _DIRECTIONS:
            rows = menu.actions_for(direction)
            if not rows:
                rankings.append(())
                continue
            matrix = np.stack([self.transform.action_vector(a) for a in rows])
            scores = matrix @ coefficients
            order = sorted(range(len(rows)),key=lambda i:(-float(scores[i]),rows[i].arm_id,rows[i].donor_id))
            rankings.append(tuple(rows[i].arm_id for i in order))
        return CaseModelPrediction(menu.menu_hash,rankings[0],rankings[1],self.model_hash)

    def public_payload(self) -> dict[str, object]:
        return {"schema_version":"case_conditional_proposal_model_v21",
                "transform":self.transform.public_payload(),"ranker":self.ranker.public_payload(),
                "model_hash":self.model_hash,"training_case_keys":[list(k) for k in self.training_case_keys],
                "opportunity_heads_used":False}


def fit_proposal_model(menus: Sequence[LabelFreeCaseMenu], profiles: Sequence[SupportCaseClassProfile],
                       outcomes: Sequence[SupportActionOutcome], *, maximum_numeric_features: int = 20
                       ) -> ProposalModel:
    rows = tuple(menus)
    keys = tuple(sorted((m.center_id,m.case_id) for m in rows))
    if (keys != tuple(sorted((p.center_id,p.case_id) for p in profiles))
        or len(keys)!=len(set(keys)) or not keys):
        raise ProtocolError("HARP v21 proposal fitting role inventories differ.")
    menu_by_key = {(m.center_id,m.case_id):m for m in rows}
    expected = {a.action_hash for m in rows for a in m.actions}
    if ({o.action.action_hash for o in outcomes} != expected
        or len(outcomes)!=len(expected)
        or any(o.menu_hash != menu_by_key[(o.action.center_id,o.action.case_id)].menu_hash for o in outcomes)):
        raise ProtocolError("HARP v21 primitive outcomes drifted from fitted menu inventory.")
    from .aligned_metrics import ClassSupportNormalizer
    from dataclasses import replace
    norm = ClassSupportNormalizer.fit(profiles)
    if any(o.class_0_gain is None and o.class_1_gain is None for o in outcomes):
        raise ProtocolError("HARP v21 primitive training requires raw classwise recall deltas.")
    normalized = tuple(replace(o,bacc_gain=norm.contribution(o.action.center_id,o.class_0_gain,o.class_1_gain),
                               normalization_hash=norm.normalization_hash) for o in outcomes)
    transform = fit_feature_transform(rows,maximum_numeric_features=maximum_numeric_features)
    comparisons = build_pairwise_comparisons(rows,normalized,transform=transform)
    return ProposalModel(transform,fit_pairwise_ranker(comparisons,alpha=1.0,transform=transform))
