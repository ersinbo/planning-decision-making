print("=== RUNNING PID_RRT_star.py (edited version) ===")
print("=== entered run() ===")

from kino_rrt_star import kino_RRTStar_GRAPH, draw_rrt_tree_3d, draw_rrt_path_3d

import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from kino_double_integrator import cost_optimal
from gym_pybullet_drones.utils.enums import DroneModel, Physics


from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary # CtrlAviary is a custom environment for controlling multiple drones in a PyBullet simulation.

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool 



DEFAULT_DRONES = DroneModel("cf2x") # CF2X and CF2P are different drone models available in the gym_pybullet_drones library. CF2X is a more advanced model with better performance and stability, while CF2P is a simpler model that is easier to control.
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True 
DEFAULT_SIMULATION_FREQ_HZ = 240 # simulation frequency for the PyBullet physics engine 
'''
why are the simulation and control frequencies different?
The simulation frequency (SIMULATION_FREQ_HZ) refers to how often the physics engine updates the state of the simulation, while the control frequency (CONTROL_FREQ_HZ) refers to how often the control inputs are computed and applied to the drones.'''
DEFAULT_CONTROL_FREQ_HZ = 48 # control frequency for the PID controller
DEFAULT_DURATION_SEC = 12  # duration of the simulation in seconds
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
import numpy as np
from kino_double_integrator import cost_optimal, xbar, gramian_di_3

class KinoTrajectoryController:
    """
    Tracks a kinodynamic RRT* path for the 3D double integrator.
    Path is a list of states [x,y,z,vx,vy,vz].
    Produces acceleration command a(t) to be applied as force = m*a.
    """
    def __init__(self, path, dt: float, tmin=0.05, tmax=3.0, n=30,
                 pos_switch_tol=0.08, vel_switch_tol=0.4):
        ''' 
        Parameters:
        path : list of np.ndarray
            List of waypoints (states) to follow, each of shape (6,).
        dt : float
            Time step for the controller.
        '''
        self.path = None if path is None else [np.asarray(p, dtype=float).reshape(6,) for p in path]
        self.dt = float(dt)
        self.tmin, self.tmax, self.n = float(tmin), float(tmax), int(n)
        self.pos_switch_tol = float(pos_switch_tol)
        self.vel_switch_tol = float(vel_switch_tol)

        self.seg = 0
        self.t_in_seg = 0.0
        self._prep_segment()

    def _prep_segment(self):
        '''
        Prepares the current segment for tracking by computing optimal time and co-states.
        what is a segment?
        A segment is the portion of the trajectory between two consecutive waypoints in the path.
        '''
        self.t_in_seg = 0.0
        self.tau = None
        self.dp = None
        self.dv = None

        if self.path is None or len(self.path) < 2:
            return
        if self.seg >= len(self.path) - 1:
            return

        x0 = self.path[self.seg]
        x1 = self.path[self.seg + 1]

        _, tau = cost_optimal(x0, x1, tmin=self.tmin, tmax=self.tmax, n=self.n)
        self.tau = float(tau)

        e = (x1 - xbar(x0, self.tau)).reshape(6,)
        d = np.linalg.solve(gramian_di_3(self.tau), e)
        self.dp = d[:3]
        self.dv = d[3:]

    def _segment_done(self, state):
        '''
        Checks if the current segment is completed based on position and velocity tolerances or time.
        '''
        if self.path is None or len(self.path) < 2:
            return True
        if self.seg >= len(self.path) - 1:
            return True

        x1 = self.path[self.seg + 1]
        pos_err = np.linalg.norm(state[:3] - x1[:3])
        vel_err = np.linalg.norm(state[3:] - x1[3:])
        time_done = (self.tau is not None) and (self.t_in_seg >= self.tau)

        return time_done or (pos_err < self.pos_switch_tol and vel_err < self.vel_switch_tol)

    def get_control(self, state):
        '''
        Computes the acceleration command for the current state.
        Parameters:
        state : np.ndarray
            Current state of the drone, shape (6,).
        Returns:
        np.ndarray
            Acceleration command, shape (3,).
        '''
        state = np.asarray(state, dtype=float).reshape(6,)

        if self.path is None or len(self.path) < 2:
            return np.zeros(3)

        # advance segment if reached
        if self._segment_done(state):
            if self.seg < len(self.path) - 2:
                self.seg += 1
                self._prep_segment()
            else:
                return np.zeros(3)

        if self.tau is None or self.dp is None or self.dv is None:
            return np.zeros(3)

        t = min(self.t_in_seg, self.tau)
        a = (self.tau - t) * self.dp + self.dv  # optimal DI control along segment
        self.t_in_seg += self.dt
        return a


def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1 # initial height
    R = .3 # radius of the circle
    INIT_XYZS = np.array([[0, 0, 0.1]])  # Initial positions of the drones
    INIT_RPYS = np.array([[0, 0,  0]]) # Initial orientations of the drones (roll, pitch, yaw)

    #### Create the environment ################################
    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient() 

    # ---------- build and draw RRT tree here ----------
    start = np.array([INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2], 0,0,0], dtype=float)  # start at the first drone's initial position
    goal  = np.array([0.5, 0.8, 0.6,0,0,0], dtype=float) # choose any goal in your workspace

    rrt = kino_RRTStar_GRAPH(
        start=start,
        goal=goal,
        n_iterations=500,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 1.0),
        z_limits=(0.1, 1.0),
        vx_limits=(-1.0, 1.0),
        vy_limits=(-1.0, 1.0),
        vz_limits=(-1.0, 1.0),
        goal_sample_rate=0.1,
        neighbor_radius=2
    )

    print("neighbor_radius =", rrt.neighbor_radius, "n_iterations =", rrt.n_iterations)

    success = rrt.plan() # build RRT graph
    print("RRT* success:", success)
    print("num nodes:", len(rrt.nodes))
    print("goal index:", rrt.goal_index)

    path = rrt.extract_path() # extract path from RRT graph

    print("path length:", None if path is None else len(path))

    if not success or path is None:
        raise RuntimeError("RRT did not reach the goal (try more iterations / different limits / different goal)")


    kino_ctrl = KinoTrajectoryController(path, dt=env.CTRL_TIMESTEP)


    draw_rrt_tree_3d(rrt.nodes, rrt.parents, PYB_CLIENT, life_time=0.0) # draw RRT TREE in PyBullet
    draw_rrt_path_3d(path, PYB_CLIENT, life_time=0.0)

    # logger = Logger(logging_freq_hz=control_freq_hz,
    #                 output_folder=output_folder,
    #                 colab=colab
    #                 ) # logger instance to record simulation data for analysis and plotting
    


    #### Run the simulation ####################################
    START = time.time() #used for sim-time to real-time sync 

    drone_id = env.DRONE_IDS[0]
    # Mass from PyBullet
    mass = p.getDynamicsInfo(drone_id, -1)[0]
    g = 9.81

    # Thrust coefficient kf (try env.KF first, fallback to the URDF printout value)
    kf = getattr(env, "KF", 3.16e-10)

    rpm_hover = float(np.sqrt((mass * g) / (4.0 * kf)))

    hover_action = np.array([[rpm_hover, rpm_hover, rpm_hover, rpm_hover]], dtype=float)

    print("mass=", mass, "kf=", kf, "rpm_hover=", rpm_hover)

    for i in range(int(duration_sec * env.CTRL_FREQ)):

        obs, reward, terminated, truncated, info = env.step(hover_action)
        pos = obs[0][0:3]
        vel = obs[0][3:6]          # linear velocity
        state = np.hstack([pos, vel])


        acc_cmd = kino_ctrl.get_control(state)

        # DEBUG: print diagnostics for controller and path following
        if kino_ctrl.path is None:
            target_wp = None
        else:
            seg = kino_ctrl.seg
            target_wp = kino_ctrl.path[seg+1] if (seg+1) < len(kino_ctrl.path) else kino_ctrl.path[-1]
        pos_err = np.linalg.norm(state[:3] - target_wp[:3]) if target_wp is not None else float("nan")
        vel_err = np.linalg.norm(state[3:] - target_wp[3:]) if target_wp is not None else float("nan")
        print(f"step={i}, seg={kino_ctrl.seg}, pos={state[:3]}, target={None if target_wp is None else target_wp[:3]}, pos_err={pos_err:.3f}, vel_err={vel_err:.3f}, acc_norm={np.linalg.norm(acc_cmd):.3f}")

        # simple mass model (works)
        force = mass * np.asarray(acc_cmd)

        # safety clamp (optional): avoid insane forces
        max_force = 50.0 * mass
        force = np.clip(force, -max_force, max_force)

        p.applyExternalForce(
            objectUniqueId=drone_id,
            linkIndex=-1,
            forceObj=force.tolist(),
            posObj=obs[0][:3].tolist(),
            flags=p.WORLD_FRAME
        )

        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)


    # env.close()

    # logger.save() # save the logged data to disk

    # if plot:
    #     logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kino-RRT* flight script using CtrlAviary and KinoTrajectoryController')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES, type=int,           metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,    type=Physics,       metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,        type=str2bool,      metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION, type=str2bool,   metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI, type=str2bool,  metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,  type=str2bool,      metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,   metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,  type=int,     metavar='') 
    ARGS = parser.parse_args()
    run(**vars(ARGS))