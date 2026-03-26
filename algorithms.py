"""
algorithms.py - Kalman Filter (2-D) and Hidden Markov Model / Viterbi map-matching.

Math notes are inline so the algorithms are self-documenting.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from models import MapGraph, Edge


# ===========================================================================
# 2-D Kalman Filter
# ===========================================================================
#
# State vector:  x_k = [x, y, vx, vy]^T   (position + velocity)
# Observation:   z_k = [x_obs, y_obs]^T   (raw GPS reading)
#
# State transition (constant-velocity model):
#   x_k = F * x_{k-1} + w,   w ~ N(0, Q)
#
#   F = | 1  0  dt  0  |
#       | 0  1   0 dt  |
#       | 0  0   1  0  |
#       | 0  0   0  1  |
#
# Observation model:
#   z_k = H * x_k + v,   v ~ N(0, R)
#
#   H = | 1  0  0  0 |
#       | 0  1  0  0 |
#
# Prediction:
#   x̂⁻_k  = F * x̂_{k-1}
#   P⁻_k   = F * P_{k-1} * F^T + Q
#
# Update (with Kalman gain K):
#   K     = P⁻_k * H^T * (H * P⁻_k * H^T + R)^{-1}
#   x̂_k   = x̂⁻_k + K * (z_k - H * x̂⁻_k)
#   P_k   = (I - K*H) * P⁻_k
# ===========================================================================

class KalmanFilter2D:
    """
    Per-vehicle 2-D Kalman filter that tracks position and velocity.
    dt: time step in seconds between observations.
    """

    def __init__(self, dt: float = 1.0, process_noise: float = 2.0, meas_noise: float = 15.0):
        self.dt = dt
        n = 4  # state dimension

        # -- State transition matrix F (constant velocity) -------------------
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ],
        ], dtype=float)

        # -- Observation matrix H (we only observe x, y) --------------------
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # -- Process noise covariance Q (position gets small noise, velocity bigger) --
        q = process_noise
        self.Q = np.diag([q*0.25, q*0.25, q, q])

        # -- Measurement noise covariance R (GPS error variance) -------------
        r = meas_noise ** 2
        self.R = np.diag([r, r])

        # -- Initial state & covariance (uninitialised) ----------------------
        self.x_hat = np.zeros(n)           # state estimate
        self.P     = np.eye(n) * 500.0     # large initial uncertainty
        self._initialised = False

    def reset(self, obs_x: float, obs_y: float):
        """Seed the filter with the first observation."""
        self.x_hat = np.array([obs_x, obs_y, 0.0, 0.0])
        self.P     = np.eye(4) * 500.0
        self._initialised = True

    def update(self, obs_x: float, obs_y: float) -> Tuple[float, float, float, float]:
        """
        Feed one GPS observation; returns (smooth_x, smooth_y, vx, vy).
        Auto-initialises on first call.
        """
        if not self._initialised:
            self.reset(obs_x, obs_y)
            return obs_x, obs_y, 0.0, 0.0

        z = np.array([obs_x, obs_y])

        # -- Prediction step -------------------------------------------------
        x_pred = self.F @ self.x_hat
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # -- Innovation (residual) -------------------------------------------
        innovation = z - self.H @ x_pred

        # -- Kalman gain: K = P⁻ H^T (H P⁻ H^T + R)^{-1} -------------------
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # -- Update step -----------------------------------------------------
        self.x_hat = x_pred + K @ innovation
        I = np.eye(4)
        self.P = (I - K @ self.H) @ P_pred

        sx, sy, vx, vy = self.x_hat
        return float(sx), float(sy), float(vx), float(vy)


# ===========================================================================
# Hidden Markov Model – Map Matching (Viterbi)
# ===========================================================================
#
# States:    Each road segment = one HMM state; plus one "off-road" state.
# Obs:       Smoothed (x, y) from Kalman filter.
#
# Emission probability p(z | s):
#   We model distance d from z to segment s as Gaussian:
#     p(z|s) = (1/sqrt(2π σ²)) * exp(-d² / 2σ²)
#   where σ = emission_sigma (expected GPS error after smoothing, in grid units).
#   For "off-road" state:  flat low probability ε.
#
# Transition probability p(s_t | s_{t-1}):
#   - Same segment:     p_stay     (high, e.g. 0.7)
#   - Adjacent segment: p_adjacent (medium, shared equally among neighbours)
#   - Other segments:   p_far      (very low)
#   - Any→ off-road:    p_offroad  (small leak)
#   - Off-road→any:     p_return   (small return probability)
#
# Viterbi decodes the most-likely sequence of road segments over a window
# of T observations, using log probabilities to avoid underflow.
# ===========================================================================

OFF_ROAD_ID = -1   # special sentinel state

class HMMMapMatcher:
    """
    Window-based Viterbi decoder.
    We re-run Viterbi over a sliding window of the last `window` observations
    so the answer stays real-time and doesn't grow unbounded.
    """

    def __init__(
        self,
        graph: MapGraph,
        emission_sigma: float = 30.0,   # expected smoothed-GPS error (grid units)
        p_stay: float       = 0.88,     # high stay-prob prevents road jumping
        p_adjacent: float   = 0.10,     # total, split across neighbours
        p_offroad: float    = 0.005,
        p_return: float     = 0.05,
        window: int         = 12,       # longer window = more stable decoding
    ):
        self.graph = graph
        self.sigma = emission_sigma
        self.p_stay = p_stay
        self.p_adjacent = p_adjacent
        self.p_offroad = p_offroad
        self.p_return  = p_return
        self.window = window

        # All HMM states: road edges + off-road
        self.edge_ids: List[int] = list(graph.edges.keys())
        self.states: List[int]   = self.edge_ids + [OFF_ROAD_ID]
        self.n_states = len(self.states)
        self.state_idx: Dict[int, int] = {s: i for i, s in enumerate(self.states)}

        # Pre-compute log transition matrix
        self._log_T = self._build_log_transition()

    # ------------------------------------------------------------------
    def _build_log_transition(self) -> np.ndarray:
        """
        Build (n_states × n_states) log transition matrix A,
        where A[i, j] = log p(state_j | state_i).
        """
        n  = self.n_states
        T  = np.zeros((n, n))

        off_idx = self.state_idx[OFF_ROAD_ID]

        for i, s in enumerate(self.states):
            if s == OFF_ROAD_ID:
                # From off-road: small prob of returning to any real road
                for j, t in enumerate(self.states):
                    if t == OFF_ROAD_ID:
                        T[i, j] = 1.0 - self.p_return
                    else:
                        T[i, j] = self.p_return / len(self.edge_ids)
            else:
                # From real road segment s
                neighbours = self.graph.adjacent[s]
                n_nbr       = max(len(neighbours), 1)

                # Distribute p_adjacent equally among actual neighbours
                p_nbr_each  = self.p_adjacent / n_nbr

                # Remaining probability for "far" segments
                n_far  = n - 1 - len(neighbours) - 1  # excl. self, nbrs, off-road
                n_far  = max(n_far, 1)
                p_far  = max(1.0 - self.p_stay - self.p_adjacent - self.p_offroad, 1e-8)
                p_far_each = p_far / n_far

                for j, t in enumerate(self.states):
                    if t == s:
                        T[i, j] = self.p_stay
                    elif t == OFF_ROAD_ID:
                        T[i, j] = self.p_offroad
                    elif t in neighbours:
                        T[i, j] = p_nbr_each
                    else:
                        T[i, j] = p_far_each

        # Normalise rows to sum to 1, then take log
        T = T / T.sum(axis=1, keepdims=True)
        return np.log(T + 1e-300)

    # ------------------------------------------------------------------
    def _log_emission(self, obs_x: float, obs_y: float) -> np.ndarray:
        """
        Returns length-n_states array of log emission probabilities
        for a single observation (obs_x, obs_y).
        """
        log_em = np.zeros(self.n_states)
        two_sig2 = 2.0 * self.sigma ** 2
        log_norm  = -0.5 * math.log(2 * math.pi * self.sigma ** 2)

        for i, s in enumerate(self.states):
            if s == OFF_ROAD_ID:
                # flat very-low probability
                log_em[i] = math.log(1e-5)
            else:
                edge = self.graph.edges[s]
                _, _, d = edge.closest_point_and_dist(obs_x, obs_y)
                log_em[i] = log_norm - (d * d) / two_sig2

        return log_em

    # ------------------------------------------------------------------
    def decode(self, observations: List[Tuple[float, float]]) -> List[int]:
        """
        Run Viterbi on a sequence of (x, y) observations.
        Returns list of state IDs (edge_ids or OFF_ROAD_ID) of same length.
        """
        T_obs = len(observations)
        if T_obs == 0:
            return []

        n = self.n_states
        # delta[t, i] = max log-prob of best path ending at state i at time t
        delta   = np.full((T_obs, n), -np.inf)
        psi     = np.zeros((T_obs, n), dtype=int)   # backpointer

        # Initialisation: uniform prior over real roads
        log_prior = math.log(1.0 / len(self.edge_ids))
        log_em_0  = self._log_emission(*observations[0])
        for i, s in enumerate(self.states):
            if s == OFF_ROAD_ID:
                delta[0, i] = math.log(1e-4) + log_em_0[i]
            else:
                delta[0, i] = log_prior + log_em_0[i]

        # Recursion
        for t in range(1, T_obs):
            log_em_t = self._log_emission(*observations[t])
            # delta[t, j] = max_i [ delta[t-1, i] + log T[i,j] ] + log_em[j]
            scores = delta[t-1, :, np.newaxis] + self._log_T   # (n × n)
            best_prev = np.argmax(scores, axis=0)               # (n,)
            delta[t]  = scores[best_prev, np.arange(n)] + log_em_t
            psi[t]    = best_prev

        # Backtrack
        path = np.zeros(T_obs, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T_obs - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return [self.states[i] for i in path]

    # ------------------------------------------------------------------
    def match_latest(self, obs_window: List[Tuple[float, float]]) -> int:
        """
        Convenience: decode a short window, return only the last state.
        obs_window should be at most self.window long.
        """
        seq = obs_window[-self.window:]
        decoded = self.decode(seq)
        return decoded[-1] if decoded else OFF_ROAD_ID
