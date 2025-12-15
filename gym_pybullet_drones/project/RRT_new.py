import numpy as np
import random
import pybullet as p

class RRT_GRAPH:
    def __init__(
        self,
        start,
        goal,
        n_iterations,
        step_size,
        x_limits,
        y_limits,
        z_limits,
        goal_sample_rate=0.1,
        goal_threshold=0.05
    ):
        self.start = start # np.array consisting of x, y, z coordinates
        self.goal = goal # np.array consisting of x, y, z coordinates
        self.goal_index = None # index of goal node in the tree if reached

        self.n_iterations = n_iterations # maximum number of iterations
        self.step_size = step_size # step size for each extension in the direction of the sampled point
        self.x_limits = x_limits 
        self.y_limits = y_limits
        self.z_limits = z_limits
        # self.z_limits = (-1.0, 1.0)
        self.goal_sample_rate = goal_sample_rate # probability of sampling the goal
        self.goal_threshold = goal_threshold # distance threshold to consider goal reached

        self.nodes = [np.array(start)] # list to store nodes
        self.parents = [None] # list to store parent indices     

    def sample(self):
        """Generate random sample point """
        if random.random() < self.goal_sample_rate:
            return np.array(self.goal)
        else:
            x = random.uniform(self.x_limits[0], self.x_limits[1])
            y = random.uniform(self.y_limits[0], self.y_limits[1])
            z = random.uniform(self.z_limits[0], self.z_limits[1])
            return np.array([x, y, z])

    def nearest(self, q_rand):
        """Return index of the nearest node, q_near, in the tree to the sampled point, q_rand"""
        dists = [np.linalg.norm(node - q_rand) for node in self.nodes]
        return np.argmin(dists)
    
    def steer_step_size(self, q_near, q_rand):
        """Steer from nearest node , q_near, towards sampled point, q_rand,  by step size"""
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return q_rand
        else:
            return q_near + (direction / distance) * self.step_size

    def collision_check(self):
        pass

    def add_node_edge(self, q_new, parent_idx):
        """
        Add new node, q_new,  and parent index to the tree
        q_new is the new node position (in the direction of q_rand) if q_rand is farther than step size from q_near
        """
        self.nodes.append(q_new)
        self.parents.append(parent_idx)
        return len(self.nodes) - 1  # return index of new node

    def stop_condition(self, q_new):
        """Check if the goal has been reached"""
        return np.linalg.norm(q_new - self.goal) < self.goal_threshold

    def extract_path(self, goal_index=None):
        """ 
        Extract path from start to goal by backtracking 
        from goal node to start node by going through the parents.
        """
        if goal_index is None:
            goal_index = self.goal_index
        if goal_index is None:
            return None

        path = [] # list to store path nodes from goal to start
        idx = goal_index
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parents[idx]
        path.reverse() # reverse to get path from start to goal 
        return path
    
    def build(self):
        """Build RRT graph"""
        for _ in range(self.n_iterations):
            q_rand = self.sample()
            idx_near = self.nearest(q_rand)
            q_near = self.nodes[idx_near]
            q_new = self.steer_step_size(q_near, q_rand)

            # optional collision check
            # if not self.collision_check(q_near, q_new):
            #     continue

            self.nodes.append(q_new)
            self.parents.append(idx_near)

            if np.linalg.norm(q_new - self.goal) <= self.goal_threshold:
                self.goal_index = len(self.nodes) - 1
                return True
        self.goal_index = None
        return False


def draw_rrt_tree_3d(nodes, parents, pyb_client, line_width=1.0, life_time=0.0):
    for i in range(1, len(nodes)):
        pi = parents[i]
        if pi is None:
            continue
        p1 = nodes[pi]
        p2 = nodes[i]
        p.addUserDebugLine(
            [float(p1[0]), float(p1[1]), float(p1[2])],
            [float(p2[0]), float(p2[1]), float(p2[2])],
            lineColorRGB=[0, 1, 0],
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=pyb_client
        )

def draw_rrt_path_3d(path, pyb_client, line_width=2.0, life_time=0.0):
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        p.addUserDebugLine(
            [float(p1[0]), float(p1[1]), float(p1[2])],
            [float(p2[0]), float(p2[1]), float(p2[2])],
            lineColorRGB=[1, 0, 0],
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=pyb_client
        )
