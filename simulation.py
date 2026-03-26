"""
simulation.py - Synthetic traffic generator and per-tick simulation engine.

Each vehicle follows a random walk along the road network:
  1. Choose a starting edge + parametric position t ∈ [0,1]
  2. Travel at a speed drawn from the edge's speed-limit band
  3. At endpoints pick a random connected edge (turn)
  4. Inject Gaussian GPS noise → raw observation
  5. Feed raw obs through Kalman filter → smoothed position + velocity
  6. Feed smoothed obs window through HMM → matched edge
"""

from __future__ import annotations
import math
import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Deque
from collections import deque

from models import MapGraph, Edge, RoadType, SPEED_LIMITS
from algorithms import KalmanFilter2D, HMMMapMatcher, OFF_ROAD_ID

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
NUM_VEHICLES      = 8
GPS_NOISE_STD     = 15.0    # standard deviation of Gaussian GPS noise (grid units)
DT                = 1.0     # simulation tick in seconds
SPEED_SCALE       = 0.1     # grid-units per second per km/h unit
                             # (120 km/h * 0.1 = 12 grid units/s)
MOVING_AVG_WINDOW = 10      # samples for per-road moving average speed
OBS_WINDOW        = 12      # HMM decoding window length (matches HMMMapMatcher)

# Traffic colour thresholds (km/h)
GREEN_THRESH  = 60.0
YELLOW_THRESH = 20.0


# ---------------------------------------------------------------------------
# Vehicle state
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    vehicle_id: str
    edge_id: int             # current edge
    t: float                 # parametric position along edge [0,1], direction
    direction: int           # +1 (A→B) or -1 (B→A)
    speed_kmh: float         # current target speed

    # running state
    raw_x:    float = 0.0
    raw_y:    float = 0.0
    smooth_x: float = 0.0
    smooth_y: float = 0.0
    vx:       float = 0.0
    vy:       float = 0.0
    heading_deg: float = 0.0
    matched_edge_id: int = OFF_ROAD_ID

    kalman: KalmanFilter2D  = field(default_factory=KalmanFilter2D)
    obs_window: Deque = field(default_factory=lambda: deque(maxlen=OBS_WINDOW))


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------

class SimulationEngine:

    def __init__(self):
        self.graph   = MapGraph()
        self.matcher = HMMMapMatcher(self.graph, emission_sigma=GPS_NOISE_STD * 1.5)
        self.vehicles: Dict[str, Vehicle] = {}

        # Per-edge circular buffer of recent speeds (km/h) for moving average
        self.speed_history: Dict[int, Deque[float]] = {
            eid: deque(maxlen=MOVING_AVG_WINDOW) for eid in self.graph.edges
        }

        self._spawn_vehicles(NUM_VEHICLES)

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def _spawn_vehicles(self, n: int):
        edge_ids = list(self.graph.edges.keys())
        for i in range(n):
            vid   = f"V{i+1:02d}"
            eid   = random.choice(edge_ids)
            edge  = self.graph.edges[eid]
            t     = random.uniform(0.1, 0.9)
            direc = random.choice([-1, 1])
            spd   = self._sample_speed(edge.road_type)
            v = Vehicle(
                vehicle_id=vid,
                edge_id=eid,
                t=t,
                direction=direc,
                speed_kmh=spd,
            )
            v.kalman = KalmanFilter2D(dt=DT, meas_noise=GPS_NOISE_STD)
            self.vehicles[vid] = v

    @staticmethod
    def _sample_speed(road_type: str) -> float:
        """Sample a speed (km/h) from a Gaussian centred on the speed limit."""
        mu  = SPEED_LIMITS[road_type]
        std = mu * 0.15           # 15% variation
        return max(5.0, np.random.normal(mu, std))

    # ------------------------------------------------------------------
    # Physics: move vehicle along edges
    # ------------------------------------------------------------------

    def _move_vehicle(self, v: Vehicle):
        edge = self.graph.edges[v.edge_id]

        # Convert speed to parametric step (grid units → fraction of edge length)
        dist_per_tick = v.speed_kmh * SPEED_SCALE   # grid units / tick
        dt_param      = dist_per_tick / max(edge.length, 1.0)

        v.t += v.direction * dt_param

        if v.t >= 1.0:
            # Arrived at node_b – choose next edge from node_b
            v.t = 0.0
            next_eid   = self._next_edge(edge, edge.node_b)
            v.edge_id  = next_eid
            # Check whether to reverse direction on new edge
            next_edge  = self.graph.edges[next_eid]
            if next_edge.node_a == edge.node_b:
                v.direction = 1
            else:
                v.direction = -1
            v.speed_kmh = self._sample_speed(next_edge.road_type)

        elif v.t <= 0.0:
            # Arrived at node_a – choose next edge from node_a
            v.t = 1.0
            next_eid   = self._next_edge(edge, edge.node_a)
            v.edge_id  = next_eid
            next_edge  = self.graph.edges[next_eid]
            if next_edge.node_b == edge.node_a:
                v.direction = -1
            else:
                v.direction = 1
            v.speed_kmh = self._sample_speed(next_edge.road_type)

    def _next_edge(self, current_edge: Edge, junction_node: int) -> int:
        """Pick a random adjacent edge at a junction node (not the same edge)."""
        candidates = [
            eid for eid in self.graph.adjacent[current_edge.edge_id]
            if junction_node in (
                self.graph.edges[eid].node_a,
                self.graph.edges[eid].node_b,
            )
        ]
        if not candidates:
            candidates = list(self.graph.adjacent[current_edge.edge_id])
        if not candidates:
            return current_edge.edge_id   # no options, stay
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # True position from parametric coords
    # ------------------------------------------------------------------

    @staticmethod
    def _true_position(edge: Edge, t: float) -> Tuple[float, float]:
        return (
            edge.ax + t * (edge.bx - edge.ax),
            edge.ay + t * (edge.by - edge.ay),
        )

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self):
        """Advance simulation by one DT step."""
        for v in self.vehicles.values():
            # 1. Move
            self._move_vehicle(v)

            edge = self.graph.edges[v.edge_id]
            tx, ty = self._true_position(edge, v.t)

            # 2. Inject GPS noise
            v.raw_x = tx + np.random.normal(0, GPS_NOISE_STD)
            v.raw_y = ty + np.random.normal(0, GPS_NOISE_STD)

            # 3. Kalman filter smoothing
            sx, sy, vx, vy = v.kalman.update(v.raw_x, v.raw_y)
            v.smooth_x = sx
            v.smooth_y = sy
            v.vx       = vx
            v.vy       = vy

            # Heading: derived from the TRUE edge direction + travel direction.
            # Kalman velocity is too noisy for heading (15-unit GPS noise vs
            # ~6-12 unit/tick movement creates large angular error).
            # Edge direction is ground-truth and always correct.
            dx = edge.bx - edge.ax
            dy = edge.by - edge.ay
            if v.direction == -1:
                dx, dy = -dx, -dy
            v.heading_deg = math.degrees(math.atan2(dy, dx))

            # 4. Append to HMM observation window and match
            v.obs_window.append((v.smooth_x, v.smooth_y))
            v.matched_edge_id = self.matcher.match_latest(list(v.obs_window))

            # 5. Record speed on *matched* edge (or true edge as fallback)
            record_eid = v.matched_edge_id if v.matched_edge_id != OFF_ROAD_ID else v.edge_id
            if record_eid in self.speed_history:
                self.speed_history[record_eid].append(v.speed_kmh)

    # ------------------------------------------------------------------
    # Aggregated traffic stats
    # ------------------------------------------------------------------

    def get_traffic(self) -> List[Dict]:
        results = []
        for eid, edge in self.graph.edges.items():
            hist = self.speed_history[eid]
            if hist:
                avg_spd = float(np.mean(hist))
            else:
                avg_spd = SPEED_LIMITS[edge.road_type]   # default = speed limit

            if avg_spd >= GREEN_THRESH:
                color = "green"
            elif avg_spd > YELLOW_THRESH:
                color = "yellow"
            else:
                color = "red"

            count = sum(
                1 for v in self.vehicles.values()
                if v.matched_edge_id == eid or (v.matched_edge_id == OFF_ROAD_ID and v.edge_id == eid)
            )
            results.append({
                "edge_id":       eid,
                "road_name":     edge.road_name,
                "road_type":     edge.road_type,
                "avg_speed_kmh": round(avg_spd, 1),
                "color":         color,
                "vehicle_count": count,
            })
        return results

    # ------------------------------------------------------------------
    # Snapshot for /vehicles
    # ------------------------------------------------------------------

    def get_vehicles(self) -> List[Dict]:
        out = []
        for v in self.vehicles.values():
            matched_name = None
            if v.matched_edge_id != OFF_ROAD_ID and v.matched_edge_id in self.graph.edges:
                matched_name = self.graph.edges[v.matched_edge_id].road_name
            out.append({
                "vehicle_id":       v.vehicle_id,
                "raw_x":            round(v.raw_x, 2),
                "raw_y":            round(v.raw_y, 2),
                "smooth_x":         round(v.smooth_x, 2),
                "smooth_y":         round(v.smooth_y, 2),
                "heading_deg":      round(v.heading_deg, 1),
                "matched_edge_id":  v.matched_edge_id,
                "matched_road_name": matched_name,
                "speed_kmh":        round(v.speed_kmh, 1),
            })
        return out
