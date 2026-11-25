"""Hidden Markov map matching (Newson & Krumm style) with Viterbi decoding.

States are candidate points on road edges. Emission: Gaussian on the distance from the GPS fix to the candidate.
Transition: exponential penalty on |route distance - great-circle distance| between consecutive fixes.
"""

import math

import networkx as nx
import numpy as np

from map_matching.geometry import Candidate, candidates_for_point, edge_length


def emission_logprob(dist: float, sigma: float) -> float:
    """log N(dist; 0, sigma^2)."""
    return -0.5 * (dist / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


def transition_logprob(route_dist: float, gc_dist: float, beta: float) -> float:
    """log of the exponential density on |route - great-circle| distance. Infinite route distance gives -inf."""
    if math.isinf(route_dist):
        return -math.inf
    return -abs(route_dist - gc_dist) / beta - math.log(beta)


def route_distance(graph: nx.DiGraph, a: Candidate, b: Candidate) -> float:
    """Network distance from candidate a to candidate b along directed edges."""
    if (a.u, a.v) == (b.u, b.v) and b.t >= a.t:
        return (b.t - a.t) * edge_length(graph, a.u, a.v)
    try:
        middle = nx.shortest_path_length(graph, a.v, b.u, weight=lambda u, v, d: edge_length(graph, u, v))
    except nx.NetworkXNoPath:
        return math.inf
    return (1 - a.t) * edge_length(graph, a.u, a.v) + middle + b.t * edge_length(graph, b.u, b.v)


def nearest_edge_baseline(graph: nx.DiGraph, points: np.ndarray, radius: float) -> list[Candidate]:
    """Baseline: snap each fix independently to its nearest candidate (no transitions)."""
    matched = []
    for px, py in points:
        cands = candidates_for_point(graph, px, py, radius)
        if not cands:
            raise ValueError(f"no road within {radius} m of ({px}, {py})")
        matched.append(min(cands, key=lambda c: c.dist))
    return matched


def viterbi_match(graph: nx.DiGraph, points: np.ndarray, sigma: float = 20.0, beta: float = 10.0, radius: float = 60.0) -> list[Candidate]:
    """Most likely sequence of candidates for the observed fixes (Viterbi in log space)."""
    layers = []
    for px, py in points:
        cands = candidates_for_point(graph, px, py, radius)
        if not cands:
            raise ValueError(f"no road within {radius} m of ({px}, {py})")
        layers.append(cands)

    scores = [np.array([emission_logprob(c.dist, sigma) for c in layers[0]])]
    back = []
    for k in range(1, len(points)):
        gc = float(np.hypot(*(points[k] - points[k - 1])))
        prev_scores = scores[-1]
        new_scores = np.full(len(layers[k]), -math.inf)
        new_back = np.zeros(len(layers[k]), dtype=int)
        for j, cj in enumerate(layers[k]):
            em = emission_logprob(cj.dist, sigma)
            for i, ci in enumerate(layers[k - 1]):
                if math.isinf(prev_scores[i]):
                    continue
                total = prev_scores[i] + transition_logprob(route_distance(graph, ci, cj), gc, beta) + em
                if total > new_scores[j]:
                    new_scores[j] = total
                    new_back[j] = i
        if np.all(np.isinf(new_scores)):
            raise ValueError(f"no feasible route into fix {k}")
        scores.append(new_scores)
        back.append(new_back)

    state = int(np.argmax(scores[-1]))
    path = [state]
    for k in range(len(points) - 1, 0, -1):
        state = int(back[k - 1][state])
        path.append(state)
    path.reverse()
    return [layers[k][s] for k, s in enumerate(path)]


def mean_error_m(matched: list[Candidate], truth_xy: np.ndarray) -> float:
    """Mean distance in metres between the matched road points and the ground-truth positions."""
    matched_xy = np.array([[c.x, c.y] for c in matched])
    return float(np.mean(np.hypot(*(matched_xy - truth_xy).T)))
