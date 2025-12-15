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
        neighbor_radius=0.5
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
        self.neighbor_radius = neighbor_radius
        self.costs = [0.0]          # cost-to-come for each node
        self.children = [set()]     # adjacency: children[i] = set of children indices of node i

    def add_node_edge(self, q_new, parent_index):
        """Add new node, set parent, and compute its cost-to-come. Return new node index."""
        new_index = len(self.nodes)
        self.nodes.append(q_new)
        self.parents.append(parent_index)

        # keep children list aligned
        self.children.append(set())

        # attach to parent in adjacency
        if parent_index is not None:
            self.children[parent_index].add(new_index)
            cost_to_new = self.costs[parent_index] + np.linalg.norm(q_new - self.nodes[parent_index])
        else:
            cost_to_new = 0.0

        self.costs.append(float(cost_to_new))
        return new_index

    def find_nearby_nodes(self, q_new):
        """Find indices of nodes within fixed neighbor_radius of q_new (O(n))."""
        idxs = []
        for i, node in enumerate(self.nodes):
            if np.linalg.norm(node - q_new) <= self.neighbor_radius:
                idxs.append(i)
        return idxs

    def best_parent_search(self, q_new, near_indexes, default_parent):
        """Choose the best parent for q_new from near_indexes based on cost-to-come + edge length."""
        best_parent = default_parent
        best_cost = self.costs[default_parent] + np.linalg.norm(self.nodes[default_parent] - q_new)

        for i in near_indexes:
            # (optional) skip self, though q_new is not in the tree yet
            new_cost = self.costs[i] + np.linalg.norm(self.nodes[i] - q_new)
            if new_cost < best_cost:
                # if self.collision_check(self.nodes[i], q_new):  # later
                best_cost = new_cost
                best_parent = i

        return best_parent, float(best_cost)

    def update_subtree_costs(self, root_index):
        """
        Update costs for the subtree rooted at root_index using children adjacency.
        This is O(size of subtree), not O(n²).
        """
        stack = [root_index]
        while stack:
            u = stack.pop()
            for v in self.children[u]:
                self.costs[v] = self.costs[u] + np.linalg.norm(self.nodes[v] - self.nodes[u])
                stack.append(v)

    def rewire(self, new_index, near_indexes):
        """Try to rewire nearby nodes through new_index if that lowers their cost."""
        q_new = self.nodes[new_index]

        for i in near_indexes:
            if i == new_index:
                continue

            # cost if we go start -> ... -> new_index -> i
            candidate_cost = self.costs[new_index] + np.linalg.norm(self.nodes[i] - q_new)

            if candidate_cost < self.costs[i]:
                # if self.collision_check(q_new, self.nodes[i]):  # later
                old_parent = self.parents[i]

                # detach i from old parent's children set
                if old_parent is not None:
                    self.children[old_parent].discard(i)

                # re-parent i under new_index
                self.parents[i] = new_index
                self.children[new_index].add(i)

                # update cost for i, then propagate to its descendants
                self.costs[i] = float(candidate_cost)
                self.update_subtree_costs(i)

    def build(self):
        """Build the RRT* graph (fixed radius)."""
        best_goal = None
        best_goal_cost = float("inf")

        for _ in range(self.n_iterations):
            q_rand = self.sample()

            index_near = self.nearest(q_rand)
            q_near = self.nodes[index_near]
            q_new = self.steer_step_size(q_near, q_rand)

            # if not self.collision_check(q_near, q_new):
            #     continue

            near_indexes = self.find_nearby_nodes(q_new)

            best_parent, _ = self.best_parent_search(q_new, near_indexes, index_near)

            new_index = self.add_node_edge(q_new, best_parent)

            # Important: near set for rewiring should include nodes near q_new in the existing tree;
            # using near_indexes computed before adding is fine.
            self.rewire(new_index, near_indexes)

            if self.stop_condition(self.nodes[new_index]):
                if self.costs[new_index] < best_goal_cost:
                    best_goal_cost = self.costs[new_index]
                    best_goal = new_index

        self.goal_index = best_goal
        return best_goal is not None


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
