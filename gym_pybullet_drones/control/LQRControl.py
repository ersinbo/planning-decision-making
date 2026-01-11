import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_discrete_are

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel


class LQRPositionControl(BaseControl):
    """
    Outer-loop LQR on translation (pos/vel) + inner-loop attitude PID.
    Outputs motor RPMs.
    """

    def __init__(self, drone_model: DroneModel, g: float = 9.8):
        super().__init__(drone_model=drone_model, g=g)

        # In THIS codebase BaseControl sets:
        #   self.GRAVITY = g * m   (a force, not an acceleration)
        # so store both:
        self.g = float(g)                       # acceleration [m/s^2]
        self.mass = float(self.GRAVITY / g)     # mass [kg]

        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] LQRPositionControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # --- Attitude PID gains (copied from DSLPIDControl) ---
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535

        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-.5, -.5, -1],
                [-.5,  .5,  1],
                [ .5,  .5, -1],
                [ .5, -.5,  1]
            ])
        else:  # CF2P
            self.MIXER_MATRIX = np.array([
                [0, -1, -1],
                [+1, 0,  1],
                [0,  1, -1],
                [-1, 0,  1]
            ])

        self._dt = None
        self._K = None

        self.reset()

    def reset(self):
        super().reset()
        self.last_rpy = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    def _build_lqr(self, dt: float):
        self._dt = float(dt)

        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        A = np.block([
            [I3, dt * I3],
            [Z3, I3]
        ])
        B = np.block([
            [0.5 * dt * dt * I3],
            [dt * I3]
        ])

        Q = np.diag([300, 300, 400,   60, 60, 80])   # pos/vel weights
        R = np.diag([1.0, 1.0, 1.5])           # accel penalty

        P = solve_discrete_are(A, B, Q, R)
        self._K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):

        self.control_counter += 1

        if self._K is None or (self._dt is None) or abs(control_timestep - self._dt) > 1e-9:
            self._build_lqr(control_timestep)

        pos_e = (target_pos - cur_pos).reshape(3,)
        vel_e = (target_vel - cur_vel).reshape(3,)
        x_e = np.hstack([pos_e, vel_e])

        # a_cmd is desired acceleration [m/s^2]
        a_cmd = (self._K @ x_e).reshape(3,)

        # limit aggressiveness
        a_max = 4.0
        a_cmd = np.clip(a_cmd, -a_max, a_max)

        # ---- IMPORTANT FIX (your BaseControl uses GRAVITY as FORCE) ----
        # Desired force vector in world frame:
        #   F = m*a_cmd + [0,0,m*g]
        thrust_vec = self.mass * a_cmd + np.array([0.0, 0.0, self.GRAVITY])

        thrust, target_euler = self._thrust_vector_to_attitude(cur_quat, thrust_vec, target_rpy[2])
        rpm = self._attitude_pid(control_timestep, thrust, cur_quat, target_euler, target_rpy_rates)

        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        yaw_e = float(target_euler[2] - cur_rpy[2])

        return rpm, pos_e, yaw_e

    def _thrust_vector_to_attitude(self, cur_quat, thrust_vec_world, target_yaw):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)

        scalar_thrust = max(0.0, float(np.dot(thrust_vec_world, cur_rotation[:, 2])))

        thrust = (math.sqrt(scalar_thrust / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE

        norm = np.linalg.norm(thrust_vec_world)
        target_z_ax = np.array([0.0, 0.0, 1.0]) if norm < 1e-9 else thrust_vec_world / norm

        target_x_c = np.array([math.cos(target_yaw), math.sin(target_yaw), 0.0])
        cross = np.cross(target_z_ax, target_x_c)
        cross_norm = np.linalg.norm(cross)
        target_y_ax = np.array([0.0, 1.0, 0.0]) if cross_norm < 1e-9 else cross / cross_norm
        target_x_ax = np.cross(target_y_ax, target_z_ax)

        target_rotation = np.vstack([target_x_ax, target_y_ax, target_z_ax]).T
        target_euler = Rotation.from_matrix(target_rotation).as_euler('XYZ', degrees=False)

        return float(thrust), target_euler

    def _attitude_pid(self, control_timestep, thrust, cur_quat, target_euler, target_rpy_rates):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))

        target_quat = Rotation.from_euler('XYZ', target_euler, degrees=False).as_quat()
        w, x, y, z = target_quat
        target_rotation = Rotation.from_quat([w, x, y, z]).as_matrix()

        rot_matrix_e = target_rotation.T @ cur_rotation - cur_rotation.T @ target_rotation
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])

        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy) / control_timestep
        self.last_rpy = cur_rpy

        self.integral_rpy_e = self.integral_rpy_e - rot_e * control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)

        target_torques = (
            - self.P_COEFF_TOR * rot_e
            + self.D_COEFF_TOR * rpy_rates_e
            + self.I_COEFF_TOR * self.integral_rpy_e
        )
        target_torques = np.clip(target_torques, -3200, 3200)

        pwm = thrust + self.MIXER_MATRIX @ target_torques
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)

        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        return rpm
