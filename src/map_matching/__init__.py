"""GPS map matching with a hidden Markov model (RT03)."""

from map_matching.geometry import Candidate, candidates_for_point, edge_length, project_onto_segment  # noqa: F401
from map_matching.hmm import (  # noqa: F401
    emission_logprob,
    mean_error_m,
    nearest_edge_baseline,
    route_distance,
    transition_logprob,
    viterbi_match,
)

__version__ = "0.1.0"
