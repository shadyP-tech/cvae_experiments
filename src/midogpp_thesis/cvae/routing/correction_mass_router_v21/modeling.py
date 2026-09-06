"""Public donor-proposal facade for the modular HARP v21 ranker."""
from .ranker_numerics import _case_weights, _sigmoid, _solve_ridge, _solve_logistic_ridge
from .ranker_features import FittedFeatureTransform, fit_feature_transform
from .pairwise_ranker import PairwiseComparison, PairwiseRanker, build_pairwise_comparisons, fit_pairwise_ranker
from .proposal_model import CaseModelPrediction, ProposalModel, fit_proposal_model

PooledScienceModel = ProposalModel
fit_pooled_science_model = fit_proposal_model
__all__ = ("FittedFeatureTransform", "PairwiseComparison", "PairwiseRanker", "ProposalModel",
           "PooledScienceModel", "CaseModelPrediction", "fit_feature_transform",
           "build_pairwise_comparisons", "fit_pairwise_ranker", "fit_proposal_model",
           "fit_pooled_science_model")
