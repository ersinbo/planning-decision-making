# kino_controller.py
print("=== RUNNING kino_controller.py (Kinodynamic RRT* + PID tracking) ===")

import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

from kino_rrt_star import KinoRRTStar, draw_rrt_tree_3d, draw_rrt_path_3d

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = "results"
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
    colab=DEFAULT_COLAB,
):
    INIT_XYZS = np.array([[0.0, 0.0, 0.1]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

    env = CtrlAviary(
        drone_model=drone,
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
        user_debug_gui=user_debug_gui,
    )
    PYB_CLIENT = env.getPyBulletClient()

    # --- Plan in double-integrator state space ---
    start = np.array([INIT_XYZS[0, 0], INIT_XYZS[0, 1], INIT_XYZS[0, 2], 0, 0, 0], dtype=float)
    goal  = np.array([5.0, 4.0, 0.6, 0, 0, 0], dtype=float)

    rrt = KinoRRTStar(
        start=start,
        goal=goal,
        n_iterations=3000,
        x_limits=(-10.0, 10.0),
        y_limits=(-10.0, 10.0),
        z_limits=(0.1, 1.0),
        vx_limits=(-1.0, 1.0),
        vy_limits=(-1.0, 1.0),
        vz_limits=(-1.0, 1.0),
        goal_sample_rate=0.20,
        neighbor_radius=50.0,     # cost-space neighbor radius
        goal_radius=2,
        tmin=0.1,
        tmax=2.0,
        n_grid=25, pyb_client=PYB_CLIENT, obstacle_ids=OBSTACLE_IDS, collision_radius=0.08
    )

    success = rrt.build()
    traj = rrt.extract_trajectory_samples(samples_per_edge=35)

    if (not success) or (traj is None) or (len(traj) < 2):
        env.close()
        raise RuntimeError("No goal connection found; increase iterations/radius/time bounds")

    TARGET_POS = traj[:, 0:3].astype(float)

    draw_rrt_tree_3d(rrt.nodes, rrt.parents, PYB_CLIENT, life_time=0.0)
    draw_rrt_path_3d(TARGET_POS, PYB_CLIENT, life_time=0.0)

    # --- PID controller that produces motor commands for CtrlAviary ---
    ctrl = DSLPIDControl(drone_model=drone)
    logger = Logger(logging_freq_hz=control_freq_hz, output_folder=output_folder, colab=colab)

    action = np.zeros((num_drones, 4))  # motor commands (e.g., RPM) expected by CtrlAviary
    wp_counter = 0
    START = time.time()

    # Optional: slow down waypoint switching; otherwise it can “race” through dense points
    min_wp_dt = 1.0 / float(control_freq_hz)   # at most 1 waypoint per control tick
    last_wp_switch_t = 0.0

    for i in range(int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        pos = obs[0][0:3]
        now_t = i / env.CTRL_FREQ

        target = TARGET_POS[wp_counter]

        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target,
            target_rpy=INIT_RPYS[0, :],
        )

        # Waypoint switching
        if wp_counter < len(TARGET_POS) - 1:
            if (np.linalg.norm(pos - target) < 0.06) and ((now_t - last_wp_switch_t) >= min_wp_dt):
                wp_counter += 1
                last_wp_switch_t = now_t

        logger.log(
            drone=0,
            timestamp=now_t,
            state=obs[0],
            control=np.hstack([TARGET_POS[wp_counter, 0:3], INIT_RPYS[0, :], np.zeros(6)]),
        )

        env.render()
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
    parser.add_argument("--num_drones", default=DEFAULT_NUM_DRONES, type=int, metavar="")
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
