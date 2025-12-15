"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
# from RRT import RRTNode, rrt_build_tree, draw_rrt_tree, draw_rrt_path, extract_path, spawn_tiny_sphere
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
'''
Is there a similar library for MPC?
DSLPIDControl is a class that implements a PID controller for drones.
For MPC (Model Predictive Control), you might want to look into libraries like do-mpc or CasADi, which provide tools for implementing MPC algorithms.
'''
from gym_pybullet_drones.utils.Logger import Logger
'''
Logger is a utility class for logging
'''
from gym_pybullet_drones.utils.utils import sync, str2bool 
'''
sync is a utility function to synchronize the simulation time with real time.
'''

DEFAULT_DRONES = DroneModel("cf2x")
'''
CF2X and CF2P are different drone models available in the gym_pybullet_drones library.
CF2X is a more advanced model with better performance and stability, while CF2P is
a simpler model that is easier to control.
'''

DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
'''
PYB refers to the PyBullet physics engine, which is a popular open-source physics engine for simulating rigid body dynamics.

'''
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True 
'''
How to add obstacles?
Set obstacles=True to add predefined obstacles in the environment.
To customize obstacles, you would need to modify the CtrlAviary environment code to include your
own obstacle definitions.
How can i add individual obstacles?
'''
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
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
    H_STEP = .05 
    '''
    Initial positions and orientations of the drones are defined here.
    Drones are arranged in a circular pattern with increasing altitudes.

    '''
    R = .3 # radius of the circle
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    #INIT_XYZS = np.array([[0.2 * i, 0, H] for i in range(num_drones)])



    '''
    Drones are positioned in a circular formation around the point (0, -R) in the X-Y plane.
    Now we want to set different initial position in a line formation
    INIT_XYZS = np.array([[0.2 * i, 0, H] for i in range(num_drones)])
    '''
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])
    '''
    Each drone is given a different yaw angle, evenly distributed over 90 degrees.
    '''

    #### Initialize a circular trajectory ######################
    PERIOD = 10 # seconds to complete a lap
    NUM_WP = control_freq_hz*PERIOD  #Total number of way points
    '''
    Define a circular trajectory for the drones to follow.
    The trajectory consists of waypoints that form a circle in the X-Y plane.
    waypoints are calculated using cosine and sine functions to create a circular path.
    The drones will follow these waypoints at their respective altitudes.
    They are useful for testing the control algorithms and observing the drones' behavior in a controlled environment.

    '''
    TARGET_POS = np.zeros((NUM_WP,3)) # Target_POS stores the waypoints for the circular trajectory.
    '''
    what does the 3rd dimension represent in TARGET_POS?
    The third dimension in TARGET_POS represents the Z-coordinate (altitude) of the waypoints.
    In this case, the Z-coordinate is set to 0 for all waypoints, meaning that the circular trajectory lies in the X-Y plane at ground level.
    
    '''
    # for i in range(NUM_WP):
    #     TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])
    '''
    Is wpcounters the points each drone is heading to?
    Yes, wp_counters keeps track of the current waypoint index for each drone.
    Each drone uses its corresponding index in wp_counters to determine which waypoint in TARGET_POS it should head to next.
    '''
    
    #make wp counters follow the RRT path?
    wp_counters = np.array([0 for i in range(num_drones)])

    # for i in range(num_drones):
    #     if wp_counters[i] < len(TARGET_POS):
    #         TARGET_POS[i, :2] = TARGET_POS[wp_counters[i], :2]
    '''
    what does this code do?   
    This code initializes the target positions for each drone based on their respective waypoint counters.
    For each drone, if its waypoint counter (wp_counters[i]) is less than the'''

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

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
    
    '''
    can i add the individual obstacles here?
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
    How to add obstacles on a fixed position?
    Set obstacles=True to add predefined obstacles in the environment.
    To customize obstacles, you would need to modify the CtrlAviary environment code to include your
    own obstacle definitions.
    '''

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient() 
    
    '''
    pybullet client ID is needed to load additional objects in the environment    
    It also allows for direct interaction with the PyBullet simulation. 
    '''

    # ---------- build and draw RRT tree here ----------
# ---------- build RRT path (2D) ----------
    start = np.array([INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2]], dtype=float)
    goal  = np.array([0.5, 0.8, 0.6], dtype=float) 

    rrt = RRT_GRAPH(
        start=start,
        goal=goal,
        n_iterations=1500,
        step_size=0.15,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 1.0),
        z_limits=(0.1, 1.0),        
        goal_sample_rate=0.1,
        goal_threshold=0.08
    )

    success = rrt.build()
    path = rrt.extract_path()
    NUM_WP = len(path)
    TARGET_POS = np.zeros((NUM_WP, 3), dtype=float)

    for k, pt in enumerate(path):
        TARGET_POS[k, :] = pt


    if not success or path is None:
        raise RuntimeError("RRT did not reach the goal (try more iterations / bigger step_size / different goal)")

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
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    



    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    '''
    controller instances for each drone are created using the DSLPIDControl class.
    Each controller is configured based on the specified drone model.
    what does an controller instance do?
    Each controller instance is responsible for computing the control inputs (e.g., motor speeds, thrust, torques)
    required to achieve the desired state (position, orientation) of its corresponding drone.
    '''
    #### Run the simulation ####################################
    action = np.zeros((num_drones,4)) #action will store the control inputs for each drone.
    START = time.time() #used for sim-time to real-time sync
    for i in range(0, int(duration_sec*env.CTRL_FREQ)): # what changes for RRT path following?
        # Update target positions based on RRT waypoints


        #### Make it rain rubber ducks #############################
        #if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                    state=obs[j],
                                                                    target_pos=TARGET_POS[wp_counters[j]],
                                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                    target_rpy=INIT_RPYS[j, :]
                                                                    )

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = min(wp_counters[j] + 1, NUM_WP - 1)


        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:3], INIT_RPYS[j, :], np.zeros(6)])

                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
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

