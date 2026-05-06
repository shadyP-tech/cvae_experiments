from src.eval.evaluators.expert_matrix import compute_expert_domain_matrix
from src.eval.evaluators.latent_usage import evaluate_expert_latent_usage
from src.eval.evaluators.domain_query_oracle_gap import (
    aggregate_domain_query_oracle_gap_rows,
    evaluate_domain_query_oracle_gap_for_run,
    evaluate_domain_query_oracle_gap_from_arrays,
)
from src.eval.evaluators.hybrid import (
    compute_hybrid_matrices_and_routing,
    evaluate_downstream_utility,
    evaluate_global_baselines,
)
from src.eval.evaluators.latent_compatibility import (
    compute_distance_matrices,
    compute_domain_gaussian_stats,
    compute_metric_utility_correlation,
    distance_to_similarity,
    evaluate_routing_alignment,
    load_embeddings_with_domains,
    verify_similarity_matrix,
)
from src.eval.evaluators.routing import evaluate_routing

__all__ = [
    "compute_expert_domain_matrix",
    "aggregate_domain_query_oracle_gap_rows",
    "evaluate_domain_query_oracle_gap_for_run",
    "evaluate_domain_query_oracle_gap_from_arrays",
    "evaluate_expert_latent_usage",
    "evaluate_routing",
    "compute_hybrid_matrices_and_routing",
    "evaluate_downstream_utility",
    "evaluate_global_baselines",
    "load_embeddings_with_domains",
    "compute_domain_gaussian_stats",
    "compute_distance_matrices",
    "distance_to_similarity",
    "verify_similarity_matrix",
    "evaluate_routing_alignment",
    "compute_metric_utility_correlation",
]
