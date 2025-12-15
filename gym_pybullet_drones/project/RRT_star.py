import numpy as np
import random
import pybullet as p

from RRT_new import RRT_GRAPH

class RRTStar_GRAPH(RRT_GRAPH):
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
        neighbor_radius=0.2
    ):
        super().__init__(
            start,
            goal,
            n_iterations,
            step_size,
            x_limits,
            y_limits,
            z_limits,
            goal_sample_rate,
            goal_threshold
        )
        self.neighbor_radius = neighbor_radius # radius to search for nearby nodes for rewiring
        self.costs = [0.0] # list to store cost to reach each node

    def add_node_edge(self, q_new, parent_index):
        """Add new node to the tree with given parent index and compute its cost
            and return the index of the new node."""
        self.nodes.append(q_new) 
        self.parents.append(parent_index)

        #compute cost from start to new node 
        cost_to_new = (self.costs[parent_index] + np.linalg.norm(q_new - self.nodes[parent_index]))
        self.costs.append(cost_to_new)
        return len(self.nodes) - 1

    def find_nearby_nodes(self, q_new):
        """Find indices of nodes within neighbor_radius of q_new"""
        indexes = []
        for i, node in enumerate(self.nodes):
            if np.linalg.norm(node - q_new) <= self.neighbor_radius:
                indexes.append(i)
        return indexes
    
    def best_parent_search(self, q_new, near_indexes, default_parent):
        """Choose the best parent for q_new from near_indexes based on cost"""
        best_parent = default_parent
        best_cost = self.costs[default_parent] + np.linalg.norm(self.nodes[default_parent] - q_new)

        # Iterate through nearby nodes to find the one that gives the lowest cost to q_new wit respect to the start node
        for i in near_indexes:
            new_cost = self.costs[i] + np.linalg.norm(self.nodes[i] - q_new)
            if new_cost < best_cost:
                # if self.collision_check(self.nodes[i], q_new):
                #     best_cost = new_cost
                #     best_parent = i
                best_cost = new_cost
                best_parent = i
        return best_parent, best_cost

    def rewire(self, new_index, near_indexes):
        q_new = self.nodes[new_index]
        for i in near_indexes:
            if i == new_index:
                continue
            new_cost = self.costs[new_index] + np.linalg.norm(self.nodes[i] - q_new)
            if new_cost < self.costs[i]:
                # if self.collision_check(q_new, self.nodes[i]):
                self.parents[i] = new_index
                self.costs[i] = new_cost


    def build(self):
        for _ in range(self.n_iterations):
            q_rand = self.sample()

            index_near = self.nearest(q_rand)
            q_near = self.nodes[index_near]
            q_new = self.steer_step_size(q_near, q_rand)

            # if not self.collision_check(q_near, q_new):
            #     continue

            # compute neighbors BEFORE adding (standard)
            near_indexes = self.find_nearby_nodes(q_new)

            # choose best parent
            best_parent, _ = self.best_parent_search(q_new, near_indexes, index_near)

            # add node with best parent
            new_index = self.add_node_edge(q_new, best_parent)

            # rewire neighbors through new node
            self.rewire(new_index, near_indexes)

            if self.stop_condition(q_new):
                self.goal_index = new_index
                return True
        self.goal_index = None
        return False
                

# ---------- PyBullet drawing helpers ----------
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
            physicsClientId=pyb_client,
        )


def draw_rrt_path_3d(path, pyb_client, line_width=2.0, life_time=0.0):
    if path is None or len(path) < 2:
        return
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        p.addUserDebugLine(
            [float(p1[0]), float(p1[1]), float(p1[2])],
            [float(p2[0]), float(p2[1]), float(p2[2])],
            lineColorRGB=[1, 0, 0],
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=pyb_client,
        )