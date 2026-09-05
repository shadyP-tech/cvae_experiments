"""Shared action-conditioned outcome estimates for honest, executed composites.

The regression targets are signed. Each case gets one center-balanced weight,
shared across its candidate rows; action configurations are not extra subjects.
The probability heads and pessimistic score are model estimates, not calibrated
per-case guarantees. Only the complete outer-held policy can be admitted.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Sequence
import numpy as np
from ...protocol import ProtocolError
from .contracts import CompositeKind, LabelFreeCaseMenu, SoftTopKComposite, decode_probability_hex
from .hashing import canonical_hash
from .modeling import FittedFeatureTransform, _case_weights, _sigmoid, _solve_logistic_ridge, _solve_ridge, fit_feature_transform
from .truth import CompositeOutcome

KINDS = (CompositeKind.B, CompositeKind.U_FULL, CompositeKind.D01_ONLY,
         CompositeKind.D10_ONLY, CompositeKind.BOTH)
_DESCRIPTOR_NAMES = ("log_k","lambda","baseline_mean","baseline_positive_fraction",
                     "mean_probability_delta","mean_absolute_delta","rms_delta","maximum_absolute_delta",
                     "hard_change_fraction","d01_mean_delta","d10_mean_delta",
                     "d01_hard_change_fraction","d10_hard_change_fraction","mean_margin_change")


def _descriptor(menu: LabelFreeCaseMenu, composite: SoftTopKComposite,
                transform: FittedFeatureTransform, *,
                baseline_array: np.ndarray | None = None,
                numeric_cache: dict[str, np.ndarray] | None = None) -> np.ndarray:
    if (composite.menu_hash != menu.menu_hash or composite.sample_ids != menu.sample_ids
        or composite.baseline_probability_hex != menu.baseline_probability_hex):
        raise ProtocolError("HARP v18 action feature extraction crossed a sealed menu.")
    base = (np.asarray(decode_probability_hex(menu.baseline_probability_hex),dtype=np.float64)
            if baseline_array is None else baseline_array)
    selected = np.asarray(decode_probability_hex(composite.probability_hex),dtype=np.float64)
    shift = selected-base
    positive = base >= .5
    hard_change = (selected>=.5) != positive
    d01,d10 = ~positive,positive
    selected_actions = tuple(menu.action_for(a) for a in (*composite.d01_action_ids,*composite.d10_action_ids))
    if composite.kind is CompositeKind.U_FULL:
        selected_actions = (menu.full_action,)
    # Outcome descriptors contain selected action statistics and probability changes,
    # never center/case identity, availability labels, or outcome columns.
    def action_numeric(action):
        if numeric_cache is None:
            return transform.numeric(action)
        if action.arm_id not in numeric_cache:
            numeric_cache[action.arm_id] = transform.numeric(action)
        return numeric_cache[action.arm_id]
    numeric = np.mean(np.stack([action_numeric(a) for a in selected_actions]),axis=0) if selected_actions else np.zeros(len(transform.feature_names))
    def branch_numeric(ids):
        return (np.mean(np.stack([action_numeric(menu.action_for(arm)) for arm in ids]),axis=0)
                if ids else np.zeros(len(transform.feature_names)))
    # Explicit case-by-branch interactions: a shared compatibility proxy may
    # favor D01 on one case and D10 on another. BOTH retains each branch's own
    # selected features instead of averaging away that disagreement.
    d01_numeric = branch_numeric(composite.d01_action_ids)
    d10_numeric = branch_numeric(composite.d10_action_ids)
    mean_on = lambda values,mask: float(np.mean(values[mask])) if np.any(mask) else 0.
    return np.asarray([1.,*numeric,*d01_numeric,*d10_numeric,*(float(composite.kind is k) for k in KINDS),
        math.log1p(composite.k or 0),float(composite.mixing_lambda or 0),float(np.mean(base)),float(np.mean(positive)),
        float(np.mean(shift)),float(np.mean(np.abs(shift))),float(np.sqrt(np.mean(shift**2))),float(np.max(np.abs(shift))),
        float(np.mean(hard_change)),mean_on(shift,d01),mean_on(shift,d10),
        mean_on(hard_change,d01),mean_on(hard_change,d10),float(np.mean(np.abs(selected-.5)-np.abs(base-.5)))],dtype=np.float64)


def _descriptor_matrix(menu: LabelFreeCaseMenu, composites: Sequence[SoftTopKComposite],
                       transform: FittedFeatureTransform) -> np.ndarray:
    # Decode baseline and standardize each donor once per case, shared by all K/lambda actions.
    baseline = np.asarray(decode_probability_hex(menu.baseline_probability_hex),dtype=np.float64)
    numeric_cache: dict[str,np.ndarray] = {}
    return np.stack([_descriptor(menu,c,transform,baseline_array=baseline,numeric_cache=numeric_cache)
                     for c in composites])


@dataclass(frozen=True,slots=True)
class ActionOutcomePrediction:
    composite_hash: str
    predicted_gain: float
    predicted_harm: float
    predicted_brier_delta: float
    predicted_logloss_delta: float
    safe_positive_probability: float
    predicted_class_0_gain: float
    predicted_class_1_gain: float
    approximate_gain_lower_score: float

    def public_payload(self) -> dict[str,object]:
        return {**{name:getattr(self,name) for name in self.__dataclass_fields__},
                "per_action_safety_guarantee":False,"lower_score_is_model_based":True}


@dataclass(frozen=True,slots=True)
class ActionOutcomeModel:
    transform: FittedFeatureTransform
    design_names: tuple[str,...]
    descriptor_means: tuple[float,...]
    descriptor_scales: tuple[float,...]
    continuous_coefficients: tuple[tuple[float,...],...]
    harm_coefficients: tuple[float,...]
    safe_positive_coefficients: tuple[float,...]
    training_composite_hashes: tuple[str,...]
    training_outcome_hashes: tuple[str,...]
    row_weights: tuple[float,...]
    residual_rmse: float
    ridge_alpha: float = 1.0
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        d=len(self.design_names)
        if (self.ridge_alpha != 1.0 or len(self.descriptor_means)!=d-1
            or len(self.descriptor_scales)!=d-1
            or any(s<=0 or not math.isfinite(s) for s in self.descriptor_scales)
            or len(self.continuous_coefficients)!=5
            or any(len(c)!=d for c in self.continuous_coefficients)
            or len(self.harm_coefficients)!=d or len(self.safe_positive_coefficients)!=d
            or len(self.training_composite_hashes)!=len(self.row_weights)
            or len(self.training_outcome_hashes)!=len(self.row_weights)
            or not self.row_weights or any(w<=0 or not math.isfinite(w) for w in self.row_weights)
            or not math.isfinite(self.residual_rmse) or self.residual_rmse<0
            or any(not math.isfinite(v) for cs in (*self.continuous_coefficients,self.harm_coefficients,self.safe_positive_coefficients) for v in cs)):
            raise ProtocolError("HARP v18 shared action-outcome model is malformed.")
        object.__setattr__(self,"model_hash",canonical_hash(self._payload()))

    @property
    def training_case_keys(self) -> tuple[tuple[str,str],...]:
        return self.transform.training_case_keys

    def _payload(self) -> dict[str,object]:
        return {"schema_version":"case_conditional_action_outcome_model_v18",
            "transform":self.transform.public_payload(),"design_names":self.design_names,
            "descriptor_means":self.descriptor_means,"descriptor_scales":self.descriptor_scales,
            "continuous_coefficients":self.continuous_coefficients,
            "continuous_targets":["signed_aligned_gain","brier_delta","logloss_delta","class_0_recall_delta","class_1_recall_delta"],
            "harm_coefficients":self.harm_coefficients,"safe_positive_coefficients":self.safe_positive_coefficients,
            "training_composite_hashes":self.training_composite_hashes,"training_outcome_hashes":self.training_outcome_hashes,
            "row_weights":self.row_weights,"residual_rmse":self.residual_rmse,"ridge_alpha":self.ridge_alpha,
            "one_weight_per_case_shared_across_candidates":True,"negative_examples_retained":True,
            "selected_numeric_feature_blocks":["shared","D01_gated","D10_gated"],
            "safe_positive_event":"gain>0 and Brier_delta<=0 and logloss_delta<=0",
            "per_action_safety_guarantee":False,"lower_score_is_model_based":True}

    def public_payload(self) -> dict[str,object]:
        return {**self._payload(),"model_hash":self.model_hash}

    def predict_composites(self,menu:LabelFreeCaseMenu,composites:Sequence[SoftTopKComposite]
                           ) -> tuple[ActionOutcomePrediction,...]:
        rows=tuple(composites)
        if (menu.center_id,menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v18 honest action prediction includes a fitted case.")
        if not rows:
            return ()
        matrix=_descriptor_matrix(menu,rows,self.transform)
        matrix[:,1:]=(matrix[:,1:]-np.asarray(self.descriptor_means))/np.asarray(self.descriptor_scales)
        continuous=matrix @ np.asarray(self.continuous_coefficients).T
        harms=_sigmoid(matrix @ np.asarray(self.harm_coefficients))
        safe=_sigmoid(matrix @ np.asarray(self.safe_positive_coefficients))
        result=[]
        for i,composite in enumerate(rows):
            if composite.kind is CompositeKind.B:
                result.append(ActionOutcomePrediction(composite.composite_hash,0.,0.,0.,0.,0.,0.,0.,0.))
            else:
                gain,brier,logloss,g0,g1=map(float,continuous[i])
                result.append(ActionOutcomePrediction(composite.composite_hash,gain,float(harms[i]),brier,logloss,
                    float(safe[i]),g0,g1,gain-self.residual_rmse))
        return tuple(result)


def fit_action_outcome_model(menus:Sequence[LabelFreeCaseMenu],composites:Sequence[SoftTopKComposite],
                             outcomes:Sequence[CompositeOutcome],*,maximum_numeric_features:int=20
                             ) -> ActionOutcomeModel:
    menu_rows=tuple(menus)
    by_case={(m.center_id,m.case_id):m for m in menu_rows}
    rows=tuple(composites)
    truth=tuple(outcomes)
    if (not rows or len(by_case)!=len(menu_rows) or len(rows)!=len(truth)
        or { (c.center_id,c.case_id) for c in rows } != set(by_case)
        or len({c.composite_hash for c in rows})!=len(rows)
        or any(c.composite_hash!=o.composite.composite_hash or o.normalization_hash is None
               for c,o in zip(rows,truth,strict=True))):
        raise ProtocolError("HARP v18 action fitting needs exact scope-aligned honest composite outcomes.")
    # Outcomes must all have been normalized jointly over this fitting universe.
    # Class presence is an aggregate scoring field, never a candidate feature.
    from .aligned_metrics import ClassSupportNormalizer
    from .contracts import SupportCaseClassProfile
    by_case_outcome={}
    for o in truth:
        key=(o.composite.center_id,o.composite.case_id)
        present=(o.class_0_gain is not None,o.class_1_gain is not None)
        if key in by_case_outcome and by_case_outcome[key]!=present:
            raise ProtocolError("HARP v18 candidate outcomes disagree on case class support.")
        by_case_outcome[key]=present
    profiles=tuple(SupportCaseClassProfile(c,k,int(p0)+int(p1),int(p0),int(p1),0,0)
                   for (c,k),(p0,p1) in sorted(by_case_outcome.items()))
    normalizer=ClassSupportNormalizer.fit(profiles)
    if any(o.normalization_hash!=normalizer.normalization_hash or abs(o.bacc_gain-normalizer.contribution(
            o.composite.center_id,o.class_0_gain,o.class_1_gain))>1e-12 for o in truth):
        raise ProtocolError("HARP v18 action outcome normalization crossed its exact fitting scope.")
    transform=fit_feature_transform(menu_rows,maximum_numeric_features=maximum_numeric_features)
    names=("intercept",*(f"selected::{n}" for n in transform.feature_names),
           *(f"selected_D01::{n}" for n in transform.feature_names),
           *(f"selected_D10::{n}" for n in transform.feature_names),
           *(f"kind::{k.value}" for k in KINDS),*_DESCRIPTOR_NAMES)
    grouped_indices: dict[tuple[str,str],list[int]] = {}
    for index,c in enumerate(rows):
        grouped_indices.setdefault((c.center_id,c.case_id),[]).append(index)
    raw = np.empty((len(rows),len(names)),dtype=np.float64)
    for key,indices in grouped_indices.items():
        raw[indices] = _descriptor_matrix(by_case[key],tuple(rows[i] for i in indices),transform)
    counts=Counter((c.center_id,c.case_id) for c in rows)
    case_weights=_case_weights(tuple(by_case))
    weights=np.asarray([case_weights[(c.center_id,c.case_id)]/counts[(c.center_id,c.case_id)] for c in rows])
    means=np.sum(weights[:,None]*raw[:,1:],axis=0)/sum(weights)
    scales=np.sqrt(np.sum(weights[:,None]*(raw[:,1:]-means)**2,axis=0)/sum(weights))
    scales[scales<=math.sqrt(np.finfo(float).eps)]=1.
    matrix=raw.copy(); matrix[:,1:]=(raw[:,1:]-means)/scales
    y=np.asarray([[o.bacc_gain,o.brier_delta,o.log_loss_delta,
                   0. if o.class_0_gain is None else o.class_0_gain,
                   0. if o.class_1_gain is None else o.class_1_gain] for o in truth])
    coeff=[]
    for j in range(5):
        w=weights.copy()
        if j in (3,4):
            w*=np.asarray([getattr(o,f"class_{j-3}_gain") is not None for o in truth])
        coeff.append(_solve_ridge(matrix,y[:,j],w,alpha=1.,penalize_intercept=False))
    harmed=np.asarray([float(o.harmed) for o in truth])
    safe=np.asarray([float(o.safe_positive) for o in truth])
    harm_coeff=_solve_logistic_ridge(matrix,harmed,weights,alpha=1.,penalize_intercept=False)
    safe_coeff=_solve_logistic_ridge(matrix,safe,weights,alpha=1.,penalize_intercept=False)
    rmse=math.sqrt(float(np.dot(weights,(y[:,0]-matrix @ coeff[0])**2)/sum(weights)))
    return ActionOutcomeModel(transform,names,tuple(map(float,means)),tuple(map(float,scales)),
        tuple(tuple(map(float,c)) for c in coeff),tuple(map(float,harm_coeff)),tuple(map(float,safe_coeff)),
        tuple(c.composite_hash for c in rows),tuple(o.outcome_hash for o in truth),tuple(map(float,weights)),rmse)


__all__=("ActionOutcomePrediction","ActionOutcomeModel","fit_action_outcome_model")
