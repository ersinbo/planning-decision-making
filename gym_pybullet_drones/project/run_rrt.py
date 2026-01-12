from RRT_new import RRT_GRAPH
from kino_rrt_star import KinoRRTStar, draw_rrt_tree_3d_curved, draw_rrt_path_3d,  draw_fast_begin, draw_fast_end, path_length_xyz
from RRT_star import RRTStar_GRAPH, draw_rrt_tree_3d, draw_rrt_path_3d

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
import csv
from pathlib import Path
from datetime import datetime


from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.LQRControl import LQRPositionControl
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary # CtrlAviary is a custom environment for controlling multiple drones in a PyBullet simulation.

# from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl 
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool 


DEFAULT_DRONES = DroneModel("cf2x") # CF2X and CF2P are different drone models available in the gym_pybullet_drones library. CF2X is a more advanced model with better performance and stability, while CF2P is a simpler model that is easier to control.
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True 
DEFAULT_SIMULATION_FREQ_HZ = 240 # simulation frequency for the PyBullet physics engine 
'''
why are the simulation and control frequencies different?
The simulation frequency (SIMULATION_FREQ_HZ) refers to how often the physics engine updates the state of the simulation, while the control frequency (CONTROL_FREQ_HZ) refers to how often the control inputs are computed and applied to the drones.'''
DEFAULT_CONTROL_FREQ_HZ = 48 # control frequency for the PID controller
DEFAULT_DURATION_SEC = 50  # duration of the simulation in seconds
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

TIMINGS_CSV = Path("results") / "timings.csv"

def append_timings(planner_type: str,
                   build_wall_time_s: float,
                   flight_time_s: float,
                   csv_path: Path = TIMINGS_CSV):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    total_time_s = build_wall_time_s + flight_time_s
    diff_flight_minus_build_s = flight_time_s - build_wall_time_s

    header = [
        "timestamp",
        "planner_type",
        "build_wall_time_s",
        "flight_time_s",
        "total_time_s",
        "diff_flight_minus_build_s",
    ]

    row = [
        datetime.now().isoformat(timespec="seconds"),
        planner_type,
        f"{build_wall_time_s:.6f}",
        f"{flight_time_s:.6f}",
        f"{total_time_s:.6f}",
        f"{diff_flight_minus_build_s:.6f}",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

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
    INIT_XYZS = np.array([[-1, -1, 0.1]])  # Initial positions of the drones
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
    
    OBSTACLE_IDS = env.getObstacleIds() # get the unique IDs assigned to each obstacle in the simulation
    DRONE_IDS = env.getDroneIds() # get the unique IDs assigned to each drone in the simulation  

    # ---------- build and draw RRT tree here ----------
    start = np.array([INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2]], dtype=float)  # start at the first drone's initial position
    goal  = np.array([-1.0, 0.6, 0.2], dtype=float) # choose any goal in your workspace

    PLANNER_TYPE = "RRT*" # set to "RRT", "RRT*", or "Kinodynamic RRT*"

    rrt = RRT_GRAPH( 
        start=start,
        goal=goal,
        n_iterations=7000,
        step_size=0.15,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 2.0),
        z_limits=(0.01, 1.5),        
        goal_sample_rate=0.1,
        goal_threshold=0.08, 
        rebuild_kdtree_every=50,
        pyb_client=PYB_CLIENT,
        obstacle_ids=OBSTACLE_IDS,
        
    ) # create RRT graph instance

    import time

    t0 = time.perf_counter() # start timer

    #draw_fast_begin(PYB_CLIENT) # TURN THESE OFF FORE TOTAL CALCULATION TIME
    success = rrt.build()

    t1 = time.perf_counter()
    build_wall_time_s = t1 - t0
    print(f"rrt.build() success={success} wall_time={t1 - t0:.3f}s")

    path = rrt.extract_path()


    #draw_fast_end(PYB_CLIENT) # TURN THESE OFF FORE TOTAL CALCULATION TIME

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
    #draw_fast_begin(PYB_CLIENT)
    #draw_rrt_tree_3d_curved(rrt.nodes, rrt.parents, rrt.edge, PYB_CLIENT, life_time=0.0)

    draw_rrt_path_3d(TARGET_POS, PYB_CLIENT, life_time=0.0)
    #draw_fast_end(PYB_CLIENT)

    START = time.time()

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
    # ctrl = DSLPIDControl(drone_model=drone) # create a list of PID controllers, one per drone
    ctrl = LQRPositionControl(drone_model=drone)
    logger = Logger(logging_freq_hz=control_freq_hz, output_folder=output_folder, colab=colab) # create logger instance

    action = np.zeros((1, 4))  # motor commands (e.g., RPM) expected by CtrlAviary
    wp_counter = 0

    # --- after TARGET_POS is created ---
    goal_pos = TARGET_POS[-1]

    # terminal behavior + smoothing params
    capture_r = 0.04     # enter goal-hold (m)
    release_r = 0.20     # exit hold if drift out (m)
    holding = False

    # optional setpoint low-pass (helps remove index jitter)
    use_filter = True
    alpha = 0.3
    target_pos_f = TARGET_POS[0].copy()
    goal_tol = 0.08                 # meters

    
    t_flight0 = time.perf_counter()

    for i in range(int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        pos = obs[0][0:3]
        now_t = i / env.CTRL_FREQ

        dist_goal = np.linalg.norm(pos - goal_pos)

        dist_goal = np.linalg.norm(pos - goal_pos)
        if dist_goal < goal_tol:
            break
      

        # --- goal capture + hold (hysteresis) ---
        if (not holding) and dist_goal < capture_r:
            holding = True
        elif holding and dist_goal > release_r:
            holding = False

        if holding:
            target_pos = goal_pos
        else:
            window = 10
            i0 = wp_counter
            i1 = min(wp_counter + window, len(TARGET_POS))
            dists = np.linalg.norm(TARGET_POS[i0:i1] - pos, axis=1)
            closest = i0 + int(np.argmin(dists))

            wp_counter = max(wp_counter, closest)

            lookahead_m = 0.25  # slower, smaller = less aggressive
            j = wp_counter
            while j < len(TARGET_POS) - 1 and np.linalg.norm(TARGET_POS[j] - pos) < lookahead_m:
                j += 1
            target_pos = TARGET_POS[j]


        # if holding:
        #     target_pos = goal_pos
        # else:
        #     # --- lookahead that shrinks near the goal ---
        #     lookahead_dist = 0.8
        #     max_step = 1

        #     if dist_goal < 0.6:
        #         lookahead_dist = 0.4
        #         max_step = 2
        #     if dist_goal < 0.2:
        #         lookahead_dist = 0.3
        #         max_step = 1

        #     # advance wp_counter to the closest point ahead (monotone)
        #     while wp_counter < len(TARGET_POS) - 1 and np.linalg.norm(pos - TARGET_POS[wp_counter]) < 0.15:
        #         wp_counter += 1

        #     # pick lookahead index ahead of wp_counter
        #     j = wp_counter
        #     for _ in range(max_step):
        #         if j < len(TARGET_POS) - 1 and np.linalg.norm(pos - TARGET_POS[j]) < lookahead_dist:
        #             j += 1
        #     target_pos = TARGET_POS[j]

        # --- optional setpoint filtering (disable while holding if you want exact stop) ---
        if use_filter and (not holding):
            target_pos_f = (1 - alpha) * target_pos_f + alpha * target_pos
            target_cmd = target_pos_f
        else:
            target_cmd = target_pos

        # --- controller uses the chosen target ---
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_cmd,
            target_rpy=INIT_RPYS[0, :],
        )

        logger.log(
            drone=0,
            timestamp=now_t,
            state=obs[0],
            control=np.hstack([target_cmd, INIT_RPYS[0, :], np.zeros(6)]),
        )

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    t_flight1 = time.perf_counter()
    flight_time_s = t_flight1 - t_flight0
    print(f"flight_time wall_time={flight_time_s:.3f}s")
    print("ABOUT TO WRITE CSV:", PLANNER_TYPE, build_wall_time_s, flight_time_s)

    total_length = path_length_xyz(TARGET_POS)
    print(total_length)

    append_timings(
        planner_type=PLANNER_TYPE,
        build_wall_time_s=build_wall_time_s,
        flight_time_s=flight_time_s,
    )
    print(f"Wrote timings to {TIMINGS_CSV}")
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

