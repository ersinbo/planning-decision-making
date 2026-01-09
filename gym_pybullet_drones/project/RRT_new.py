from tracemalloc import start
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
        rebuild_kdtree_every=50,
        pyb_client=None,
        obstacle_ids=None
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

        self.nodes = [np.array(start)] # list to store nodes
        self.parents = [None] # list to store parent indices

        self._rebuild_kdtree_every = int(rebuild_kdtree_every)
        self._kdtree = None

        self._recent_indices = []
        self._rebuild_kdtree()
        self._pyb_client = pyb_client
        self._obstacle_ids = obstacle_ids if obstacle_ids is not None else []
    
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

    def collision_check(self, q_near, q_new, r=0.08):
        if self._pyb_client is None or len(self._obstacle_ids) == 0:
            print("No collision checking possible")
            return True  # no collision checking possible

        # start = [float(q_near[0]), float(q_near[1]), float(q_near[2])]
        # end = [float(q_new[0]), float(q_new[1]), float(q_new[2])]

        """for obs_id in self._obstacle_ids:
            pts = p.getClosestPoints(bodyA=obs_id, bodyB=-1, distance=0.0, 
                                     physicsClientId=self._pyb_client)"""
        
            

        # hit_object_id, hit_link, hit_fraction, hit_pos, hit_normal = p.rayTest(
        #     start, end, physicsClientId=self._pyb_client
        # )[0]

        # # hit_object_id == -1 means no hit
        # if hit_object_id in self._obstacle_ids:
        #     return False
        # return True

        """ batched ray test - more accurate, but WIP"""

        start0 = np.array(q_near, dtype=float)
        end0   = np.array(q_new, dtype=float)

        offsets = [
            np.array([0, 0, 0]),
            np.array([ r, 0, 0]),
            np.array([-r, 0, 0]),
            np.array([0,  r, 0]),
            np.array([0, -r, 0]),
            ]
        for off in offsets:
            start = (start0 + off).tolist()
            end   = (end0 + off).tolist()
            hit_object_id = p.rayTest(start, end, physicsClientId=self._pyb_client)[0][0]
            
            if hit_object_id in self._obstacle_ids:
                return False

        if hit_object_id != -1 and hit_object_id not in self._obstacle_ids:
            print("Ray hit non obstacle:", hit_object_id)
        
        return True

    def add_node_edge(self, q_new, parent_idx):
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
            idx_near = self.nearest_kdtree(q_rand)
            q_near = self.nodes[idx_near]
            q_new = self.steer_step_size(q_near, q_rand)

            #collision check
            if not self.collision_check(q_near, q_new):
                continue

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
