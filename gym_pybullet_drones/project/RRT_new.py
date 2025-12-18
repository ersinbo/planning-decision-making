import numpy as np
import random
import pybullet as p
from scipy.spatial import cKDTree

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
        goal_threshold=0.05,
        rebuild_kdtree_every=50
    ):
        self.start = start # np.array consisting of x, y, z coordinates
        self.goal = goal # np.array consisting of x, y, z coordinates
        self.goal_index = None # index of goal node in the tree if reached

        self.n_iterations = n_iterations # maximum number of iterations
        self.step_size = step_size # step size for each extension in the direction of the sampled point
        self.x_limits = x_limits 
        self.y_limits = y_limits
        self.z_limits = z_limits

        self.goal_sample_rate = goal_sample_rate # probability of sampling the goal
        self.goal_threshold = goal_threshold # distance threshold to consider goal reached

        self.nodes = [start] # list to store nodes
        self.parents = [None] # list to store parent indices     

        self._rebuild_kdtree_every = int(rebuild_kdtree_every)
        self._kdtree = None

        self._recent_indices = []
        self._rebuild_kdtree()
    
    def _rebuild_kdtree(self):
        """Rebuild the KDTree for nearest neighbor search"""
        self._kdtree = cKDTree(np.vstack(self.nodes))
        self._recent_indices = []
    
    def _rebuild_kdtree_if_needed(self):
        """Rebuild KDTree if enough new nodes have been added since last rebuild"""
        if len(self._recent_indices) >= self._rebuild_kdtree_every:
            self._rebuild_kdtree()

    def sample(self):
        """Generate random sample point """
        if random.random() < self.goal_sample_rate:
            return np.array(self.goal)
        else:
            x = random.uniform(self.x_limits[0], self.x_limits[1])
            y = random.uniform(self.y_limits[0], self.y_limits[1])
            z = random.uniform(self.z_limits[0], self.z_limits[1])
            return np.array([x, y, z])

    def nearest_kdtree(self, q_rand):
        """Return index of the nearest node, q_near, in the tree to the sampled point, q_rand using KDTree"""

        distance, index = self._kdtree.query(q_rand)
        
        for i in self._recent_indices:
            dist_recent = np.linalg.norm(self.nodes[i] - q_rand)
            if dist_recent < distance:
                distance = dist_recent
                index = i
        return int(index)
        
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

    def add_node_edge(self, q_new, parent_index):
        """
        Add new node, q_new,  and parent index to the tree
        q_new is the new node position (in the direction of q_rand) if q_rand is farther than step size from q_near
        """
        self.nodes.append(q_new)
        self.parents.append(parent_index)
        new_index = len(self.nodes) - 1  # return index of new node

        self._recent_indices.append(new_index) # track recently added nodes for KDTree updates
        return new_index

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
        index = goal_index
        while index is not None:
            path.append(self.nodes[index])
            index = self.parents[index]
        path.reverse() # reverse to get path from start to goal 
        return path
    
    def build(self):
        """Build RRT graph"""
        for _ in range(self.n_iterations):
            q_rand = self.sample()
            index_near = self.nearest_kdtree(q_rand)
            q_near = self.nodes[index_near]
            q_new = self.steer_step_size(q_near, q_rand)

            # optional collision check
            # if not self.collision_check(q_near, q_new):
            #     continue

            new_index = self.add_node_edge(q_new, index_near)
            self._rebuild_kdtree_if_needed()

            if self.stop_condition(q_new):
                self.goal_index = new_index
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
