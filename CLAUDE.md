# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Setup

**First time:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Every subsequent run:**
```bash
source .venv/bin/activate && python3 main.py
```

Then open **http://localhost:8000** in a browser.
The frontend is served by FastAPI itself — do not open `index.html` via Live Server or file://; it makes API calls relative to its own origin and will break.

**Dependencies:** `fastapi`, `uvicorn`, `numpy`, `scipy`, `pydantic` (see `requirements.txt`).

---

## Architecture Overview

This is a **synthetic map-matching simulation**. It is not a routing engine. The purpose is to generate noisy GPS data, filter it, probabilistically snap it to a road graph, and serve live traffic analytics.

### Full Pipeline (one tick, per vehicle)

```
[SimulationEngine.tick()]
         │
         ▼
1. Move vehicle
   _move_vehicle(v)
   → advance parametric position t along current edge
   → if t overflows (0 or 1), pick a random connected edge at the junction
         │
         ▼
2. True position
   _true_position(edge, t)
   → linear interpolation: (ax + t*(bx-ax), ay + t*(by-ay))
         │
         ▼
3. Inject GPS noise
   raw_x = true_x + Normal(0, 15)
   raw_y = true_y + Normal(0, 15)
         │
         ▼
4. Kalman Filter  [KalmanFilter2D.update()]
   → smooths raw_x/raw_y → smooth_x, smooth_y, vx, vy
         │
         ▼
5. Compute heading
   → from TRUE edge direction (not Kalman velocity)
   → atan2(dy, dx) with direction sign applied
         │
         ▼
6. HMM Map Matching  [HMMMapMatcher.match_latest()]
   → append (smooth_x, smooth_y) to vehicle's 12-obs sliding window
   → run Viterbi over the window → matched_edge_id
         │
         ▼
7. Record speed
   → append vehicle.speed_kmh to speed_history[matched_edge_id]
   → speed_history is a circular deque of last 10 readings per road
```

Every second, `main.py`'s background thread calls `engine.tick()`. The three API endpoints snapshot engine state on demand.

---

## File Responsibilities

### `models.py`

Pure data layer. No simulation logic.

| Class / Symbol | Responsibility |
|---|---|
| `RoadType` | Constants: `"tollway"`, `"main"`, `"sub"` |
| `SPEED_LIMITS` | Dict mapping road type → km/h limit (120 / 60 / 30) |
| `Node` | Dataclass: `node_id`, `x`, `y` |
| `Edge` | Dataclass: endpoints, road type, name, precomputed `ax/ay/bx/by/length`. Method `closest_point_and_dist(px, py)` uses parametric projection `t = dot(P-A, B-A) / |B-A|²` to find the nearest point on the segment |
| `MapGraph.__init__()` | Hardcodes 9 nodes and 10 edges on a 1000×1000 grid. Centre node 4 (500,500) is shared by 6 edges — this is intentional to create HMM ambiguity |
| `MapGraph._build_adjacency()` | Computes `adjacent[edge_id] = set of edge_ids` that share a node endpoint. Used by the HMM transition matrix |
| `NodeSchema`, `EdgeSchema`, `MapResponse`, `VehicleResponse`, `TrafficSegment`, `TrafficResponse` | Pydantic models for FastAPI JSON serialisation only |

**Map topology:**
```
NW(0)--N(1)--NE(2)
 |      |       |
 W(3)--C(4)--E(5)
 |              |
SW(6)  S(7)  SE(8)
```
Edges: T1-West (3→4), T2-East (4→5), T3-North (1→4), M1-South (4→7), M2-NW-Diagonal (0→4), M3-SE-Diagonal (4→8), M4-NE-Vertical (2→5), S1-Left (0→3), S2-Top (1→2), S3-SW-Spur (6→3).

---

### `algorithms.py`

Stateless math. Both classes are instantiated once and reused.

#### `KalmanFilter2D`

One instance per vehicle. Maintains its own covariance matrix `P` and state estimate `x_hat`.

| Method | What it does |
|---|---|
| `__init__(dt, process_noise, meas_noise)` | Builds matrices `F` (state transition), `H` (observation), `Q` (process noise covariance), `R` (measurement noise covariance) |
| `reset(obs_x, obs_y)` | Seeds the filter from a first observation. Called automatically on first `update()` |
| `update(obs_x, obs_y)` | Runs one predict→update cycle. Returns `(smooth_x, smooth_y, vx, vy)` |

**Math — two steps per tick:**
- **Predict:** `x̂⁻ = F·x̂`,  `P⁻ = F·P·Fᵀ + Q`
- **Update:** `K = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹`,  `x̂ = x̂⁻ + K·(z − H·x̂⁻)`,  `P = (I − K·H)·P⁻`

State vector is `[x, y, vx, vy]`. Only `x, y` are observed (GPS), velocity is inferred from the motion model.

#### `HMMMapMatcher`

One instance shared by all vehicles. Stateless per-call (no per-vehicle memory).

| Method | What it does |
|---|---|
| `__init__(graph, emission_sigma, p_stay, p_adjacent, p_offroad, p_return, window)` | Builds the `(n_states × n_states)` log-transition matrix once at construction. States = 10 edges + 1 off-road sentinel (`-1`) |
| `_build_log_transition()` | Fills transition probs: `p_stay=0.88` for same road, `p_adjacent=0.10` split across graph neighbours, tiny leak to off-road. Normalises rows and takes log to avoid underflow |
| `_log_emission(obs_x, obs_y)` | For each state, computes Gaussian log-prob of the observation given distance to that segment: `log p = -0.5·log(2πσ²) - d²/(2σ²)`. Off-road gets a flat penalty of `log(1e-5)` |
| `decode(observations)` | Full Viterbi over a list of `(x,y)` observations. Returns the most-likely state sequence |
| `match_latest(obs_window)` | Slices the last `window` observations and returns only the final decoded state — the current matched road |

**Viterbi recursion:**
```
delta[t, j] = max_i [ delta[t-1, i] + log T[i,j] ] + log_emission[t, j]
psi[t, j]   = argmax_i above   ← backpointer
```
Backtrack from `argmax(delta[-1])` to recover the full path, return the last element.

---

### `simulation.py`

Owns all mutable runtime state.

| Class / Function | Responsibility |
|---|---|
| `Vehicle` (dataclass) | Holds all per-vehicle state: `edge_id`, `t` (parametric 0–1), `direction` (±1), `speed_kmh`, raw/smooth positions, heading, matched edge, its own `KalmanFilter2D` instance, and a `deque(maxlen=12)` observation window |
| `SimulationEngine.__init__()` | Creates `MapGraph`, `HMMMapMatcher`, spawns 8 vehicles, initialises per-edge `speed_history` deques |
| `_spawn_vehicles(n)` | Places vehicles at random edges/positions with random direction and speed sampled from `_sample_speed()` |
| `_sample_speed(road_type)` | `Normal(speed_limit, speed_limit × 0.15)`, clamped to min 5 km/h |
| `_move_vehicle(v)` | Advances `t` by `speed_kmh × SPEED_SCALE / edge.length`. On overflow/underflow: calls `_next_edge()` to pick a turn, resets `t`, updates direction, resamples speed |
| `_next_edge(current_edge, junction_node)` | Finds edges adjacent to `current_edge` that also touch `junction_node`. Returns a random choice. Falls back to any adjacent edge if none connect at that node |
| `_true_position(edge, t)` | Linear interpolation along edge: `(ax + t·(bx−ax), ay + t·(by−ay))` |
| `tick()` | Main per-second loop: move → get true pos → add noise → Kalman update → compute heading from edge direction → HMM match → record speed |
| `get_traffic()` | Averages `speed_history` deque per edge, applies colour thresholds (green ≥ 60, yellow 20–60, red ≤ 20 km/h), counts vehicles on each edge |
| `get_vehicles()` | Snapshots all vehicle fields into a list of dicts for the API |

**Key constants:**
| Name | Value | Effect |
|---|---|---|
| `GPS_NOISE_STD` | 15.0 | Grid units of Gaussian GPS noise |
| `SPEED_SCALE` | 0.1 | Grid-units/s per km/h (120 km/h → 12 units/s) |
| `MOVING_AVG_WINDOW` | 10 | Per-road speed history depth |
| `OBS_WINDOW` | 12 | Viterbi sliding window length |

---

### `main.py`

FastAPI app. Minimal logic — just wiring.

| Symbol | Responsibility |
|---|---|
| `engine` | Singleton `SimulationEngine`, created at module load |
| `_lock` | `threading.Lock` — all reads and writes to `engine` go through this |
| `_sim_loop()` | Background daemon thread: calls `engine.tick()` then sleeps 1 s, forever |
| `serve_index()` `GET /` | Returns `index.html` as a `FileResponse` |
| `get_map()` `GET /map` | Serialises `engine.graph` nodes and edges. Static — never changes |
| `get_vehicles()` `GET /vehicles` | Calls `engine.get_vehicles()` under lock, returns list of `VehicleResponse` |
| `get_traffic()` `GET /traffic` | Calls `engine.get_traffic()` under lock, returns `TrafficResponse` with timestamp |

---

### `index.html`

Self-contained. No build step, no dependencies — vanilla JS with Canvas 2D.

#### Coordinate system
World coordinates have Y increasing upward (north). Canvas Y increases downward. All world→canvas conversions flip Y:
```js
function wx(x) { return offX + x * scale; }
function wy(y) { return cssH() - (offY + y * scale); }
```
`cssH()` returns `canvas.height / dpr` (logical pixels, not physical) so the flip math is consistent on Retina displays.

#### HiDPI sharpness
`resize()` sets `canvas.width/height` to `clientWidth/Height × devicePixelRatio`, then calls `ctx.setTransform(dpr, 0, 0, dpr, 0, 0)`. All drawing uses logical CSS pixels; the browser composites at native resolution.

#### Rendering layers (drawn in order)
1. **Background fill** — `#f7fafc` light grey
2. **Road segments** — coloured by traffic state (green/yellow/red from `/traffic`); line width scales with road type (tollway=8, main=5, sub=3)
3. **Speed labels** — white pill background, text coloured to match road state, drawn at edge midpoint
4. **Intersection nodes** — white circle with dark stroke
5. **Snap lines** (toggleable) — dashed amber line from smooth position to its projected point on the matched edge
6. **Raw GPS dots** (toggleable) — translucent red circles at `raw_x, raw_y`
7. **Vehicle markers** (toggleable) — blue circle at `smooth_x, smooth_y` + open line arrow rotated to `heading_deg`

#### Heading on canvas
Heading from the backend is `atan2(dy, dx)` in world space (Y-up). To draw on canvas (Y-down), negate:
```js
const heading = -v.heading_deg * Math.PI / 180;
```
This is equivalent to `atan2(-dy, dx)` — correct Y-flip for canvas rotation.

#### Poll loop
`fetchAll()` runs every 1500 ms via `setInterval`. It fires two parallel fetches (`/vehicles` and `/traffic`), updates the `vehicles` array and `traffic` dict, then calls `render()` and `updateSidebar()`. `/map` is fetched once on load only (it never changes).

#### Toggle buttons
Each button calls `toggleLayer(key)` which flips `layers[key]` boolean and adds/removes the `.active` CSS class. `render()` checks `layers.raw`, `layers.smooth`, `layers.snap`, `layers.labels` before drawing each layer.

#### Tooltip
`mousemove` on the canvas checks Euclidean distance from cursor to each `wx(smooth_x), wy(smooth_y)`. If within 12px, shows a positioned `<div>` with vehicle ID, matched road, speed, and heading.
