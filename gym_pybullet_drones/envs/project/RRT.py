import numpy as np
import random
import pybullet as p


class RRTNode:
    def __init__(self, position, parent_index=None):
        """
        position: np.ndarray of shape (2,) or (3,)
        parent_index: index of parent node in the nodes list (int or None)
        """
        self.position = np.array(position, dtype=float)
        self.parent_index = parent_index


def rrt_build_tree(start, goal,
                   n_iterations=500,
                   step_size=0.1,
                   x_limits=(-1.0, 1.0),
                   y_limits=(-1.0, 1.0),
                   goal_sample_rate=0.1,
                   goal_threshold=0.05):

    def distance(a, b):
        return np.linalg.norm(a - b)

    def sample():
        if random.random() < goal_sample_rate:
            return np.array(goal, dtype=float)
        else:
            return np.array([
                random.uniform(x_limits[0], x_limits[1]),
                random.uniform(y_limits[0], y_limits[1])
            ], dtype=float)

    def nearest_node_index(nodes, point):
        dists = [distance(node.position, point) for node in nodes]
        return int(np.argmin(dists))

    def steer(from_pos, to_pos, max_step):
        v = to_pos - from_pos
        d = np.linalg.norm(v)
        if d <= max_step:
            return to_pos
        else:
            return from_pos + v / d * max_step

    nodes = [RRTNode(position=start, parent_index=None)]
    goal_node_index = None  # index where we first reach the goal

    for _ in range(n_iterations):
        q_rand = sample()
        idx_near = nearest_node_index(nodes, q_rand)
        q_near = nodes[idx_near].position
        q_new = steer(q_near, q_rand, step_size)

        nodes.append(RRTNode(position=q_new, parent_index=idx_near))
        new_index = len(nodes) - 1

        # check if we reached the goal region
        if distance(q_new, goal) <= goal_threshold:
            goal_node_index = new_index
            break

    return nodes, goal_node_index

def extract_path(nodes, goal_index=None):
    """
    If goal_index is None, backtracks from the last node.
    Otherwise backtracks from nodes[goal_index].
    """
    if goal_index is None:
        current_index = len(nodes) - 1
    else:
        current_index = goal_index

    path = []
    while current_index is not None:
        node = nodes[current_index]
        path.append(node.position)
        current_index = node.parent_index

    path.reverse()
    return path

def draw_rrt_tree(nodes, pyb_client, line_width=1.0, life_time=0.0):
    """
    Draws the RRT tree in PyBullet using debug lines.

    Parameters
    ----------
    nodes : list of RRTNode
        RRTNode must have .position (len 2 or 3) and .parent_index.
    pyb_client : int
        PyBullet client ID (from env.getPyBulletClient()).
    line_width : float
        Width of debug lines.
    life_time : float
        Lifetime of lines in seconds. 0 means persist until manually removed.
    """
    for i, node in enumerate(nodes):
        if node.parent_index is None:
            # root (start) node -> no edge to draw
            continue

        parent = nodes[node.parent_index]
        p1 = parent.position
        p2 = node.position

        # Ensure 3D coords for PyBullet
        if p1.shape[0] == 2:
            p1_3d = [p1[0], p1[1], 0.0]
            p2_3d = [p2[0], p2[1], 0.0]
        else:
            p1_3d = p1.tolist()
            p2_3d = p2.tolist()

        # Slightly raise z so it doesn't collide visually with the ground
        p1_3d[2] += 0.1
        p2_3d[2] += 0.1

        p.addUserDebugLine(
            p1_3d,
            p2_3d,
            lineColorRGB=[0, 1, 0],   # green
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=pyb_client
        )


def draw_rrt_path(path, pyb_client, line_width=2.0, life_time=0.0): 
    """
    Draws the RRT path in PyBullet using debug lines.

    Parameters
    ----------
    path : list of np.ndarray
        List of positions (len 2 or 3).
    pyb_client : int
        PyBullet client ID (from env.getPyBulletClient()).
    line_width : float
        Width of debug lines.
    life_time : float
        Lifetime of lines in seconds. 0 means persist until manually removed.
    """
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]

        # Ensure 3D coords for PyBullet
        if p1.shape[0] == 2:
            p1_3d = [p1[0], p1[1], 0.0]
            p2_3d = [p2[0], p2[1], 0.0]
        else:
            p1_3d = p1.tolist()
            p2_3d = p2.tolist()

        # Slightly raise z so it doesn't collide visually with the ground
        p1_3d[2] += 0.1
        p2_3d[2] += 0.1

        p.addUserDebugLine(
            p1_3d,
            p2_3d,
            lineColorRGB=[1, 0, 0],   # red
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=pyb_client
        )

    
'''
add a function drawing the goal point as a red sphere
'''

import pybullet as p

def spawn_tiny_sphere(x, y, z=0.1, radius=0.02, pyb_client=0):
    col = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=radius,
        physicsClientId=pyb_client
    )
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1, 0, 0, 1],  # red
        physicsClientId=pyb_client
    )
    p.createMultiBody(
        baseMass=0,  # static
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[x, y, z],
        physicsClientId=pyb_client
    )


