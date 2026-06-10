"""Small constant-velocity EKF used by the route synchronization pipeline."""

from __future__ import annotations

import numpy as np


class ExtendedKalmanFilter:
    """Track planar position, scalar speed, and heading."""

    def __init__(self, state_dim: int = 4, meas_dim_gps: int = 3) -> None:
        if state_dim != 4:
            raise ValueError("ExtendedKalmanFilter currently supports a 4D state")
        self.state_dim = state_dim
        self.meas_dim_gps = meas_dim_gps
        self.x = np.zeros(state_dim, dtype=float)
        self.P = np.eye(state_dim, dtype=float) * 100.0
        self.Q = np.eye(state_dim, dtype=float) * 0.01

    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def state_transition(self, accel: float, _ay: float, yaw_rate: float, dt: float) -> np.ndarray:
        heading = self.x[3]
        speed = self.x[2]
        return np.array(
            [
                [1.0, 0.0, np.cos(heading) * dt, -speed * np.sin(heading) * dt],
                [0.0, 1.0, np.sin(heading) * dt, speed * np.cos(heading) * dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def predict(self, accel_x: float, accel_y: float, yaw_rate: float, dt: float) -> None:
        dt = float(dt)
        heading = self.x[3]
        accel = float(accel_x)
        self.x[0] += self.x[2] * np.cos(heading) * dt + 0.5 * accel * np.cos(heading) * dt * dt
        self.x[1] += self.x[2] * np.sin(heading) * dt + 0.5 * float(accel_y) * np.sin(heading) * dt * dt
        self.x[2] += accel * dt
        self.x[3] += float(yaw_rate) * dt

        F = self.state_transition(accel_x, accel_y, yaw_rate, dt)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, x_utm: float, y_utm: float, speed: float, hdop: float = 1.0) -> None:
        z = np.array([x_utm, y_utm, speed], dtype=float)
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        variance = max(float(hdop), 0.1) ** 2
        R = np.diag([variance, variance, variance])
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
