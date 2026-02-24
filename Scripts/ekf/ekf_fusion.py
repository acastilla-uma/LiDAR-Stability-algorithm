"""
Extended Kalman Filter for GPS + IMU Fusion

Fuses low-frequency GPS (1 Hz) with high-frequency IMU (10 Hz) to produce
a smooth, continuous trajectory estimate in UTM coordinates.

State vector: x = [x_utm, y_utm, v, psi]
- x_utm, y_utm: Position in UTM (meters)
- v: Velocity (m/s)
- psi: Yaw angle (radians)
"""

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import euclidean
import logging

logger = logging.getLogger(__name__)


class ExtendedKalmanFilter:
    """
    EKF for GPS + IMU fusion with bicycle model kinematics.
    """
    
    def __init__(self, state_dim=4, meas_dim_gps=3, wheelbase=3.5):
        """
        Initialize EKF.
        
        Args:
            state_dim: Dimension of state vector (typically 4: x, y, v, psi)
            meas_dim_gps: Dimension of GPS measurement (3: x, y, v)
            wheelbase: Vehicle wheelbase for kinematic model (meters)
        """
        self.n = state_dim
        self.m = meas_dim_gps
        self.L = wheelbase  # wheelbase
        
        # State vector
        self.x = np.zeros(self.n)
        
        # Covariance matrices
        self.P = np.eye(self.n) * 1.0  # Initial state covariance
        self.Q = np.eye(self.n) * 0.1  # Process noise covariance
        self.R = np.eye(self.m) * 0.5  # Measurement noise covariance
        
        logger.info(f"EKF initialized: state_dim={state_dim}, meas_dim={meas_dim_gps}")
    
    def set_process_noise(self, q_diag):
        """Set process noise covariance (diagonal)."""
        self.Q = np.diag(q_diag)
    
    def set_measurement_noise(self, r_diag):
        """Set measurement noise covariance (diagonal)."""
        self.R = np.diag(r_diag)
    
    def state_transition(self, ax, ay, psi_dot, dt):
        """
        Compute state transition Jacobian F for the kinematic model.
        
        Simplified bicycle model:
        x_dot = v * cos(psi)
        y_dot = v * sin(psi)
        v_dot = ax  (direct acceleration input)
        psi_dot = psi_dot  (angular velocity from gyro)
        
        Args:
            ax: Longitudinal acceleration (m/s²)
            ay: Lateral acceleration (m/s²) - used for estimation
            psi_dot: Yaw rate (rad/s)
            dt: Time step (seconds)
            
        Returns:
            Jacobian matrix F
        """
        x, y, v, psi = self.x
        
        # Jacobian of the discrete-time kinematic model
        # ∂x/∂[x,y,v,psi]
        F = np.array([
            [1, 0, np.cos(psi) * dt, -v * np.sin(psi) * dt],
            [0, 1, np.sin(psi) * dt,  v * np.cos(psi) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        return F
    
    def predict(self, ax, ay, psi_dot, dt):
        """
        Prediction step: update state estimate using IMU data.
        
        Args:
            ax: Longitudinal acceleration (m/s²)
            ay: Lateral acceleration (m/s²)
            psi_dot: Yaw rate (rad/s)
            dt: Time step (seconds)
        """
        # State transition function (nonlinear)
        x, y, v, psi = self.x
        
        # Simple kinematic model
        self.x[0] = x + v * np.cos(psi) * dt
        self.x[1] = y + v * np.sin(psi) * dt
        self.x[2] = v + ax * dt
        self.x[3] = psi + psi_dot * dt
        
        # Normalize psi to [-pi, pi]
        self.x[3] = np.arctan2(np.sin(self.x[3]), np.cos(self.x[3]))
        
        # Compute Jacobian
        F = self.state_transition(ax, ay, psi_dot, dt)
        
        # Update covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
    
    def measurement_jacobian(self):
        """
        Compute measurement Jacobian H.
        
        Measurement model: we observe [x_utm, y_utm, v] directly from GPS
        h(x) = [x, y, v]
        
        Returns:
            Jacobian matrix H (3x4)
        """
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        return H
    
    def update(self, z_x, z_y, z_v, hdop=1.0):
        """
        Update step: correct state estimate using GPS measurement.
        
        Args:
            z_x: Measured X position (UTM meters)
            z_y: Measured Y position (UTM meters)
            z_v: Measured velocity (m/s)
            hdop: Horizontal dilution of precision (scales measurement noise)
        """
        # Measurement vector
        z = np.array([z_x, z_y, z_v])
        
        # Measurement Jacobian
        H = self.measurement_jacobian()
        
        # Innovation (measurement residual)
        y = z - (H @ self.x)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R * hdop
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(self.n) - K @ H) @ self.P
    
    def get_state(self):
        """Return current state [x_utm, y_utm, v, psi]."""
        return self.x.copy()
    
    def get_position(self):
        """Return (x, y) position in UTM."""
        return self.x[:2].copy()
    
    def get_velocity(self):
        """Return velocity."""
        return self.x[2]
    
    def get_yaw(self):
        """Return yaw angle."""
        return self.x[3]
    
    def get_covariance(self):
        """Return state covariance matrix."""
        return self.P.copy()
