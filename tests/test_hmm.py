import math

import networkx as nx
import numpy as np
import pytest

from map_matching import (
    candidates_for_point,
    emission_logprob,
    nearest_edge_baseline,
    route_distance,
    transition_logprob,
    viterbi_match,
)


def _two_roads():
    """Two parallel 200 m roads, y=0 (A-B-C) and y=30 (D-E-F), joined at both ends by connectors. Edges are two-way."""
    coords = {"A": (0, 0), "B": (100, 0), "C": (200, 0), "D": (0, 30), "E": (100, 30), "F": (200, 30)}
    g = nx.DiGraph()
    for name, (x, y) in coords.items():
        g.add_node(name, x=float(x), y=float(y))
    for u, v in [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("A", "D"), ("C", "F")]:
        g.add_edge(u, v)
        g.add_edge(v, u)
    return g


def test_emission_peaks_at_zero_distance_and_matches_gaussian_formula():
    assert emission_logprob(0.0, 20.0) == pytest.approx(-math.log(20.0 * math.sqrt(2 * math.pi)))
    assert emission_logprob(5.0, 20.0) > emission_logprob(15.0, 20.0)


def test_transition_is_best_when_route_matches_straight_line_distance():
    best = transition_logprob(100.0, 100.0, beta=10.0)
    assert best > transition_logprob(130.0, 100.0, beta=10.0)
    assert transition_logprob(math.inf, 100.0, beta=10.0) == -math.inf


def test_route_distance_equals_dijkstra_length_on_the_graph():
    g = _two_roads()
    a = candidates_for_point(g, 50.0, 0.0, radius=1.0)[0]  # on A-B at fraction 0.5
    b = candidates_for_point(g, 150.0, 30.0, radius=1.0)[0]  # on E-F at fraction 0.5
    # Route from (50, 0) to (150, 30): leave A-B at B and reach E through the graph (B->A->D->E, length 230).
    dijkstra_nodes = nx.shortest_path_length(g, "B", "E", weight=lambda u, v, d: math.dist(
        (g.nodes[u]["x"], g.nodes[u]["y"]), (g.nodes[v]["x"], g.nodes[v]["y"])))
    assert dijkstra_nodes == pytest.approx(230.0)
    expected = 50.0 + dijkstra_nodes + 50.0
    assert route_distance(g, a, b) == pytest.approx(expected)


def test_viterbi_keeps_the_consistent_road_where_nearest_snap_jumps():
    g = _two_roads()
    # The fix at (100, 17) is 13 m from the upper road and 17 m from the lower road, so nearest snapping jumps.
    fixes = np.array([[10.0, 2.0], [50.0, 4.0], [100.0, 17.0], [150.0, 4.0], [190.0, 2.0]])

    nearest = nearest_edge_baseline(g, fixes, radius=60.0)
    assert {nearest[2].u, nearest[2].v} & {"D", "E", "F"}

    matched = viterbi_match(g, fixes, sigma=20.0, beta=10.0, radius=60.0)
    assert {matched[2].u, matched[2].v} <= {"A", "B", "C"}
    assert all({c.u, c.v} & {"A", "B", "C"} for c in matched)


def test_candidates_outside_radius_are_dropped():
    g = _two_roads()
    cands = candidates_for_point(g, 50.0, 500.0, radius=60.0)
    assert cands == []
