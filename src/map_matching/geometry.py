"""Geometry helpers: project GPS points onto road-graph edges in a metric CRS (metres)."""

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Candidate:
    """A point on edge (u, v): fraction t along the edge, and the distance from the GPS fix to that point."""

    u: object
    v: object
    t: float
    dist: float
    x: float
    y: float


def project_onto_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float):
    """Closest point to (px, py) on segment A-B, clamped to the segment. Returns (t, x, y, distance)."""
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0:
        t = 0.0
    else:
        t = float(np.clip(((px - ax) * dx + (py - ay) * dy) / seg_len2, 0.0, 1.0))
    x, y = ax + t * dx, ay + t * dy
    return t, x, y, float(np.hypot(px - x, py - y))


def edge_length(graph: nx.DiGraph, u, v) -> float:
    """Edge length in metres from the 'length' attribute, falling back to the straight-line distance."""
    data = graph.edges[u, v]
    if "length" in data:
        return float(data["length"])
    return float(np.hypot(graph.nodes[v]["x"] - graph.nodes[u]["x"], graph.nodes[v]["y"] - graph.nodes[u]["y"]))


def candidates_for_point(graph: nx.DiGraph, px: float, py: float, radius: float) -> list[Candidate]:
    """All edges whose projected point lies within `radius` metres of the GPS fix."""
    found = []
    for u, v in graph.edges:
        a, b = graph.nodes[u], graph.nodes[v]
        t, x, y, d = project_onto_segment(px, py, a["x"], a["y"], b["x"], b["y"])
        if d <= radius:
            found.append(Candidate(u, v, t, d, x, y))
    return found
