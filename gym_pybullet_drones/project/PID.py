
from RRT_new import RRT_GRAPH, draw_rrt_tree_3d, draw_rrt_path_3d

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

from gym_pybullet_drones.utils.enums import DroneModel, Physics


from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary # CtrlAviary is a custom environment for controlling multiple drones in a PyBullet simulation.

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl 
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool 
from obstacles import walls

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
    env = walls(drone_model=drone,
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
    start = np.array([INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2]], dtype=float)  # start at the first drone's initial position
    goal  = np.array([0.5, 0.8, 0.6], dtype=float) # choose any goal in your workspace

    rrt = RRT_GRAPH( 
        start=start,
        goal=goal,
        n_iterations=1500,
        step_size=0.15,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 1.0),
        z_limits=(0.1, 1.0),        
        goal_sample_rate=0.1,
        goal_threshold=0.08, 
        rebuild_kdtree_every=50
    )

    success = rrt.build()
    path = rrt.extract_path()

    if not success or path is None:
        raise RuntimeError("RRT did not reach the goal (try more iterations / bigger step_size / different goal)")
    TARGET_POS = np.array(path, dtype=float)
    NUM_WP = len(TARGET_POS)

    # for k, pt in enumerate(path):
    #     TARGET_POS[k, :] = pt

    # ---------- convert RRT path to PID waypoints ----------
    # NUM_WP = len(path)
    # TARGET_POS = np.zeros((NUM_WP, 3), dtype=float)

    # z_const = INIT_XYZS[0, 2]  # keep altitude constant
    # for k, pt in enumerate(path):
    #     TARGET_POS[k, 0] = pt[0]
    #     TARGET_POS[k, 1] = pt[1]
    #     TARGET_POS[k, 2] = z_const

    # ---------- DRAW RRT TREE + PATH ----------
    draw_rrt_tree_3d(rrt.nodes, rrt.parents, PYB_CLIENT, life_time=0.0)
    draw_rrt_path_3d(path, PYB_CLIENT, life_time=0.0)

    # ------------------------------------------------------

    wp_counters = np.zeros(num_drones, dtype=int)

    # 2D RRT in (x, y) using the first drone's start position
        #     start = [INIT_XYZS[0, 0], INIT_XYZS[0, 1]]
        #     goal  = [0.5, 0.8]   # choose any goal in your workspace

        #     nodes, goal_idx = rrt_build_tree(
        #         start=start,
        #         goal=goal,
        #         n_iterations=500,
        #         step_size=0.1,
        #         x_limits=(-1.0, 1.0),
        #         y_limits=(-1.0, 1.0)
        # )
        #     spawn_tiny_sphere(goal[0], goal[1], z=0.1)
        #     draw_rrt_tree(nodes, PYB_CLIENT, line_width=1.0, life_time=0.0)
        #     path = extract_path(nodes, goal_index=goal_idx)
        #     draw_rrt_path(path, PYB_CLIENT, line_width=2.0, life_time=0.0)

        #     end = path[-1]
    
        #     NUM_WP = len(path)
        #     TARGET_POS = np.zeros((NUM_WP,3)) # Target_POS stores the waypoints for the circular trajectory.

        #     for k, pt in enumerate(path):
        #         TARGET_POS[k, 0] = pt[0]                 # x
        #         TARGET_POS[k, 1] = pt[1]                 # y
        #         TARGET_POS[k, 2] = INIT_XYZS[0, 2]       # z (keep constant altitude)

    # one waypoint index per drone
    # wp_counters = np.array([0 for _ in range(num_drones)])    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    output_folder=output_folder,
                    colab=colab
                    ) # logger instance to record simulation data for analysis and plotting
    

    wp_counter = 0 # waypoint counter
    ctrl = DSLPIDControl(drone_model=drone) # create a list of PID controllers, one per drone


    #### Run the simulation ####################################
    action = np.zeros((1, 4))   # (num_drones, 4)
    START = time.time() #used for sim-time to real-time sync 

    for i in range(0, int(duration_sec*env.CTRL_FREQ)): 

        obs, reward, terminated, truncated, info = env.step(action) # step the environment with the current control inputs. Obs contains the current state of the drone.

        pos = obs[0][0:3]  # current position of the drone

        action[0,:], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=TARGET_POS[wp_counter],
            target_rpy=INIT_RPYS[0, :]
        )
        if np.linalg.norm(pos - TARGET_POS[wp_counter]) < 0.05 and wp_counter < NUM_WP - 1:  # check if the drone is close enough to the current waypoint
            wp_counter += 1

        logger.log(
            drone=0,
            timestamp=i/env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([TARGET_POS[wp_counter, 0:3], INIT_RPYS[0, :], np.zeros(6)])
            )
        
        env.render() # render the environment

        if gui:
            sync(i, START, env.CTRL_TIMESTEP) # sync sim-time to real-time if GUI is enabled

    env.close()

    logger.save() # save the logged data to disk
    logger.save_as_csv("pid") # Optional CSV save

    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
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
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,     type=int,     metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,    type=str,     metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,            type=bool,    metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

