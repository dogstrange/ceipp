"""
main.py - FastAPI application for the synthetic map-matching simulation.

Endpoints:
  GET /          → serves index.html
  GET /map       → static map structure (nodes + edges)
  GET /vehicles  → current vehicle positions / states
  GET /traffic   → per-road traffic states

A background thread advances the simulation by one tick per second.
"""

import time
import threading
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import (
    MapResponse, NodeSchema, EdgeSchema,
    VehicleResponse, TrafficSegment, TrafficResponse
)
from simulation import SimulationEngine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Map-Matching Simulation", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Simulation singleton (shared state, lock-protected)
# ---------------------------------------------------------------------------

engine = SimulationEngine()
_lock  = threading.Lock()


def _sim_loop():
    """Background thread: advance simulation every second."""
    while True:
        with _lock:
            engine.tick()
        time.sleep(1.0)


_thread = threading.Thread(target=_sim_loop, daemon=True)
_thread.start()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path, media_type="text/html")


@app.get("/map", response_model=MapResponse, tags=["Map"])
def get_map():
    """Return the static road network (nodes and edges)."""
    with _lock:
        nodes = [
            NodeSchema(node_id=n.node_id, x=n.x, y=n.y)
            for n in engine.graph.nodes.values()
        ]
        edges = [
            EdgeSchema(
                edge_id=e.edge_id,
                node_a=e.node_a,
                node_b=e.node_b,
                road_type=e.road_type,
                road_name=e.road_name,
                ax=e.ax, ay=e.ay,
                bx=e.bx, by=e.by,
                length=round(e.length, 2),
            )
            for e in engine.graph.edges.values()
        ]
    return MapResponse(nodes=nodes, edges=edges)


@app.get("/vehicles", response_model=list[VehicleResponse], tags=["Vehicles"])
def get_vehicles():
    """Return current state for all vehicles."""
    with _lock:
        data = engine.get_vehicles()
    return [VehicleResponse(**d) for d in data]


@app.get("/traffic", response_model=TrafficResponse, tags=["Traffic"])
def get_traffic():
    """Return per-road traffic state (speed + colour category)."""
    with _lock:
        data    = engine.get_traffic()
        ts      = time.time()
    segments = [TrafficSegment(**d) for d in data]
    return TrafficResponse(segments=segments, timestamp=ts)


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
