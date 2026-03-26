"""
models.py - Data structures for the synthetic map-matching simulation.
Defines the road network graph, road types, and Pydantic response models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
import numpy as np


# ---------------------------------------------------------------------------
# Road network constants
# ---------------------------------------------------------------------------

class RoadType:
    TOLLWAY = "tollway"    # speed limit 120 km/h
    MAIN    = "main"       # speed limit  60 km/h
    SUB     = "sub"        # speed limit  30 km/h

SPEED_LIMITS: Dict[str, float] = {
    RoadType.TOLLWAY: 120.0,
    RoadType.MAIN:     60.0,
    RoadType.SUB:      30.0,
}

# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class Node:
    node_id: int
    x: float
    y: float


@dataclass
class Edge:
    edge_id: int
    node_a: int
    node_b: int
    road_type: str
    road_name: str

    # derived – filled by MapGraph after construction
    ax: float = 0.0
    ay: float = 0.0
    bx: float = 0.0
    by: float = 0.0
    length: float = 0.0          # Euclidean length in grid units

    def midpoint(self) -> Tuple[float, float]:
        return ((self.ax + self.bx) / 2, (self.ay + self.by) / 2)

    def closest_point_and_dist(self, px: float, py: float) -> Tuple[float, float, float]:
        """
        Returns (cx, cy, distance) – the closest point on the segment to (px, py).
        Uses parametric projection: t = dot(P-A, B-A) / |B-A|^2
        """
        dx, dy = self.bx - self.ax, self.by - self.ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0:          # degenerate edge (point)
            return self.ax, self.ay, np.hypot(px - self.ax, py - self.ay)
        t = max(0.0, min(1.0, ((px - self.ax) * dx + (py - self.ay) * dy) / len_sq))
        cx = self.ax + t * dx
        cy = self.ay + t * dy
        return cx, cy, np.hypot(px - cx, py - cy)


# ---------------------------------------------------------------------------
# The synthetic 1000×1000 map
# ---------------------------------------------------------------------------

class MapGraph:
    """
    10 roads on a 1000×1000 Cartesian grid.

    Road layout (deliberately intersecting to create HMM ambiguity):
      Tollways (T):  long diagonal / horizontal corridors
      Main roads (M): mid-length roads connecting clusters
      Sub roads (S):  short local connectors

    Intersection nodes are shared so connectivity is explicit.
    """

    def __init__(self):
        # -- Nodes (intersections) -----------------------------------------
        # Grid layout on 1000×1000 canvas – named by compass position.
        #   NW(0)---N(1)---NE(2)
        #    |       |       |
        #    W(3)---C(4)---E(5)
        #    |               |
        #   SW(6)   S(7)   SE(8)
        #
        #  id : (x,   y)
        raw_nodes = {
            0: ( 200,  800),   # NW
            1: ( 500,  800),   # N
            2: ( 800,  800),   # NE
            3: ( 200,  500),   # W
            4: ( 500,  500),   # Centre (major hub)
            5: ( 800,  500),   # E
            6: ( 200,  200),   # SW
            7: ( 500,  200),   # S
            8: ( 800,  200),   # SE
        }
        self.nodes: Dict[int, Node] = {
            nid: Node(nid, x, y) for nid, (x, y) in raw_nodes.items()
        }

        # -- Edges (road segments) -----------------------------------------
        # Layout ensures the Centre node (4) is shared by 6 roads, creating
        # rich HMM ambiguity.  Outer nodes are shared by 2–3 roads each.
        #
        # (edge_id, node_a, node_b, road_type, road_name)
        raw_edges = [
            # Tollways – T1, T2, T3  (high-speed corridors through centre)
            ( 0,  3,  4, RoadType.TOLLWAY, "T1-West"),        # W→Centre
            ( 1,  4,  5, RoadType.TOLLWAY, "T2-East"),        # Centre→E
            ( 2,  1,  4, RoadType.TOLLWAY, "T3-North"),       # N→Centre

            # Main roads – M1..M4
            ( 3,  4,  7, RoadType.MAIN,    "M1-South"),       # Centre→S
            ( 4,  0,  4, RoadType.MAIN,    "M2-NW-Diagonal"), # NW→Centre (diagonal)
            ( 5,  4,  8, RoadType.MAIN,    "M3-SE-Diagonal"), # Centre→SE (diagonal)
            ( 6,  2,  5, RoadType.MAIN,    "M4-NE-Vertical"), # NE→E

            # Sub roads – S1..S3  (local connectors, creates outer-loop ambiguity)
            ( 7,  0,  3, RoadType.SUB,     "S1-Left"),        # NW→W
            ( 8,  1,  2, RoadType.SUB,     "S2-Top"),         # N→NE
            ( 9,  6,  3, RoadType.SUB,     "S3-SW-Spur"),     # SW→W
        ]
        self.edges: Dict[int, Edge] = {}
        for eid, na, nb, rtype, rname in raw_edges:
            e = Edge(eid, na, nb, rtype, rname)
            na_node = self.nodes[na]
            nb_node = self.nodes[nb]
            e.ax, e.ay = na_node.x, na_node.y
            e.bx, e.by = nb_node.x, nb_node.y
            e.length   = np.hypot(e.bx - e.ax, e.by - e.ay)
            self.edges[eid] = e

        # Build adjacency for HMM transition probs
        self._build_adjacency()

    def _build_adjacency(self):
        """
        Two edges are *adjacent* if they share a node endpoint.
        Stored as set of edge_id pairs for quick lookup.
        """
        # node_id → list of edge_ids incident on that node
        node_to_edges: Dict[int, List[int]] = {n: [] for n in self.nodes}
        for eid, e in self.edges.items():
            node_to_edges[e.node_a].append(eid)
            node_to_edges[e.node_b].append(eid)

        self.adjacent: Dict[int, set] = {eid: set() for eid in self.edges}
        for node_edges in node_to_edges.values():
            for i in range(len(node_edges)):
                for j in range(len(node_edges)):
                    if i != j:
                        self.adjacent[node_edges[i]].add(node_edges[j])

    def num_edges(self) -> int:
        return len(self.edges)


# ---------------------------------------------------------------------------
# Pydantic response schemas (used by FastAPI)
# ---------------------------------------------------------------------------

class NodeSchema(BaseModel):
    node_id: int
    x: float
    y: float

class EdgeSchema(BaseModel):
    edge_id: int
    node_a: int
    node_b: int
    road_type: str
    road_name: str
    ax: float
    ay: float
    bx: float
    by: float
    length: float

class MapResponse(BaseModel):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]

class VehicleResponse(BaseModel):
    vehicle_id: str
    raw_x: float
    raw_y: float
    smooth_x: float
    smooth_y: float
    heading_deg: float          # degrees, 0=East, CCW positive
    matched_edge_id: Optional[int]
    matched_road_name: Optional[str]
    speed_kmh: float

class TrafficSegment(BaseModel):
    edge_id: int
    road_name: str
    road_type: str
    avg_speed_kmh: float
    color: str                  # "green" | "yellow" | "red"
    vehicle_count: int

class TrafficResponse(BaseModel):
    segments: List[TrafficSegment]
    timestamp: float
