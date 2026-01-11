# kino_controller.py
print("=== RUNNING kino_controller.py (Kinodynamic RRT* + PID tracking) ===")

import time
import argparse
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from kino_rrt_star import KinoRRTStar, draw_rrt_tree_3d_curved, draw_rrt_path_3d,  draw_fast_begin, draw_fast_end

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = True
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False # Whether to use user debug GUI, this allows for more detailed visualization and debugging
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240 # Simulation frequency in Hz
DEFAULT_CONTROL_FREQ_HZ = 48 # Control frequency in Hz
DEFAULT_DURATION_SEC = 20 # Duration of the simulation in seconds
DEFAULT_OUTPUT_FOLDER = "results" # Folder to save results
DEFAULT_COLAB = False # Whether to use Google Colab

def run(
    drone=DEFAULT_DRONES, 
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
    colab=DEFAULT_COLAB,
):
    INIT_XYZS = np.array([[-1.0, -1.0, 0.1]]) # Initial positions (x, y, z)
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]]) # Initial orientations (roll, pitch, yaw)

    env = CtrlAviary(
        drone_model=drone,
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=1,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui,
    )
    PYB_CLIENT = env.getPyBulletClient()

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=PYB_CLIENT) # disable the default PyBullet GUI

    OBSTACLE_IDS = getattr(env, "OBSTACLE_IDS", []) # get obstacle IDs
    start = np.array([INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2], 0, 0, 0], dtype=float)
    goal  = np.array([0.9, 0.5, 0.6, 0, 0, 0], dtype=float)

    rrt = KinoRRTStar(
        start=start,
        goal=goal,
        n_iterations=4000,
        x_limits=(-1.0, 1.0),
        y_limits=(-1.0, 1.0),
        z_limits=(0.1, 1.0),
        vx_limits=(-1.0, 1.0),
        vy_limits=(-1.0, 1.0),
        vz_limits=(-1.0, 1.0),
        goal_sample_rate=0.01, # probability of sampling the goal
        neighbor_radius=2,     # cost-space neighbor radius, NOT IN METERS, but cost
        goal_radius=5,          # goal region radius, NOT IN METERS, but cost
        tmin=0.1,               # minimum time duration for a trajectory segment
        tmax=2.0,               # maximum time duration for a trajectory segment
        n_grid=15,              # number of time steps for trajectory optimization
        pyb_client=PYB_CLIENT, obstacle_ids=OBSTACLE_IDS, collision_radius=0.06
    )




    import cProfile
    import pstats
    import io
    import time

    pr = cProfile.Profile() # create profiler instance for performance measurement of function calls
    t0 = time.perf_counter() # start timer

    pr.enable() # start tracking functions
    draw_fast_begin(PYB_CLIENT) # start drawing
    success = rrt.build(
    max_iterations=rrt.n_iterations,   # hard cap
    patience=2000,          # stop if no improvements
    min_delta=0.1,         # must improve by at least this
    warmup=2000,            # don't early-stop too early
    verbose=True
)
    draw_fast_end(PYB_CLIENT)

    
    pr.disable()

    t1 = time.perf_counter()
    print(f"rrt.build() success={success} wall_time={t1 - t0:.3f}s")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)

    # Sort by cumulative time (time in function + its callees)
    ps.sort_stats("cumtime")


    # Print ALL functions (can be very long). Change to print_stats(200) to limit.
    ps.print_stats(30)


    traj = rrt.extract_trajectory_samples(samples_per_edge=10) # extract trajectory samples for the drone to follow

    goal_tol = 0.08                 # meters
    if (not success) or (traj is None) or (len(traj) < 1):
        env.close()
        raise RuntimeError("No goal connection found; increase iterations/radius/time bounds")

    TARGET_POS = traj[:, 0:3].astype(float)

    draw_fast_begin(PYB_CLIENT)
    #draw_rrt_tree_3d_curved(rrt.nodes, rrt.parents, rrt.edge, PYB_CLIENT, life_time=0.0)

    draw_rrt_path_3d(TARGET_POS, PYB_CLIENT, life_time=0.0)
    draw_fast_end(PYB_CLIENT)


    # --- PID controller that produces motor commands for CtrlAviary ---
    
    ctrl = DSLPIDControl(drone_model=drone)
    logger = Logger(logging_freq_hz=control_freq_hz, output_folder=output_folder, colab=colab) # create logger instance

    action = np.zeros((1, 4))  # motor commands (e.g., RPM) expected by CtrlAviary
    wp_counter = 0
    START = time.time()

    # --- after TARGET_POS is created ---
    goal_pos = TARGET_POS[-1]

    # terminal behavior + smoothing params
    capture_r = 0.08     # enter goal-hold (m)
    release_r = 0.12     # exit hold if drift out (m)
    holding = False

    # optional setpoint low-pass (helps remove index jitter)
    use_filter = True
    alpha = 0.15
    target_pos_f = TARGET_POS[0].copy()


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
            # --- lookahead that shrinks near the goal ---
            lookahead_dist = 0.35
            max_step = 7

            if dist_goal < 0.6:
                lookahead_dist = 0.15
                max_step = 3
            if dist_goal < 0.2:
                lookahead_dist = 0.04
                max_step = 2

            # advance wp_counter to the closest point ahead (monotone)
            while wp_counter < len(TARGET_POS) - 1 and np.linalg.norm(pos - TARGET_POS[wp_counter]) < 0.15:
                wp_counter += 1

            # pick lookahead index ahead of wp_counter
            j = wp_counter
            for _ in range(max_step):
                if j < len(TARGET_POS) - 1 and np.linalg.norm(pos - TARGET_POS[j]) < lookahead_dist:
                    j += 1
            target_pos = TARGET_POS[j]

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



    env.close()
    logger.save()
    logger.save_as_csv("kino_pid")
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinodynamic RRT* + PID tracking in CtrlAviary")
    parser.add_argument("--drone", default=DEFAULT_DRONES, type=DroneModel, metavar="", choices=DroneModel)
    parser.add_argument("--physics", default=DEFAULT_PHYSICS, type=Physics, metavar="", choices=Physics)
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, metavar="")
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VISION, type=str2bool, metavar="")
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool, metavar="")
    parser.add_argument("--user_debug_gui", default=DEFAULT_USER_DEBUG_GUI, type=str2bool, metavar="")
    parser.add_argument("--obstacles", default=DEFAULT_OBSTACLES, type=str2bool, metavar="")
    parser.add_argument("--simulation_freq_hz", default=DEFAULT_SIMULATION_FREQ_HZ, type=int, metavar="")
    parser.add_argument("--control_freq_hz", default=DEFAULT_CONTROL_FREQ_HZ, type=int, metavar="")
    parser.add_argument("--duration_sec", default=DEFAULT_DURATION_SEC, type=int, metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, metavar="")
    parser.add_argument("--colab", default=DEFAULT_COLAB, type=bool, metavar="")
    ARGS = parser.parse_args()
    run(**vars(ARGS))
