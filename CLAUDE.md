# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Server

```bash
python3 main.py
# Serves on http://localhost:8000
```

Dependencies: `pip3 install -r requirements.txt` (fastapi, uvicorn, numpy, scipy, pydantic)

## Architecture

This is a **synthetic map-matching simulation** — not a routing engine. It generates noisy GPS data, filters it, snaps it to a road graph, and exposes traffic analytics via a REST API with a live canvas frontend.

### Data Flow (per tick)

```
Vehicle moves along edge (parametric t)
  → GPS noise injected → raw_x, raw_y
  → KalmanFilter2D.update() → smooth_x, smooth_y, vx, vy
  → HMMMapMatcher.match_latest(obs_window) → matched_edge_id
  → speed recorded to per-edge moving average buffer
```

### File Responsibilities

- **`models.py`** — Pure data layer. `MapGraph` builds the 9-node / 10-edge road network and pre-computes the adjacency set used by the HMM. Pydantic schemas at the bottom are only for FastAPI serialisation.
- **`algorithms.py`** — Stateless math. `KalmanFilter2D` is per-vehicle (holds covariance matrix P and state x_hat). `HMMMapMatcher` is shared across all vehicles — it pre-computes the full log-transition matrix at construction time; `decode()` runs Viterbi over a sliding window of smoothed observations.
- **`simulation.py`** — `SimulationEngine` owns all runtime state: vehicle structs, per-vehicle Kalman instances, per-edge speed-history deques. `tick()` advances one second of simulation. `get_vehicles()` / `get_traffic()` snapshot state for the API.
- **`main.py`** — FastAPI app. A daemon thread calls `engine.tick()` every second behind a `threading.Lock`. Three endpoints: `GET /map`, `GET /vehicles`, `GET /traffic`. Serves `index.html` at `/`.
- **`index.html`** — Self-contained vanilla JS / Canvas 2D frontend. Polls all three endpoints every 1.5 s. Coordinate transform: world Y is flipped (`wy = canvas.height - (offY + y * scale)`) so higher Y values appear visually higher.

### Map Network

9 nodes on a 1000×1000 Cartesian grid. Centre node 4 (500, 500) is shared by 6 of the 10 road edges, creating the HMM ambiguity. Road types: `tollway` (120 km/h limit), `main` (60 km/h), `sub` (30 km/h). Vehicle speed is sampled from a Gaussian centred on the limit (±15%).

### Key Constants (in `simulation.py`)

| Name | Default | Effect |
|---|---|---|
| `NUM_VEHICLES` | 8 | Vehicles spawned at startup |
| `GPS_NOISE_STD` | 15.0 | Grid units of Gaussian noise |
| `SPEED_SCALE` | 0.1 | Grid-units/s per km/h (120 km/h → 12 units/s) |
| `MOVING_AVG_WINDOW` | 10 | Samples for per-road speed average |
| `OBS_WINDOW` | 8 | Viterbi decoding window length |

Traffic colours: green ≥ 60 km/h, yellow 20–60 km/h, red ≤ 20 km/h.
