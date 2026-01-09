# kino_rrt_star.py
import numpy as np
import random
import pybullet as p

from kino_double_integrator import connect_star_fast, cost_optimal_fast 

class KinoRRTStar:
    def __init__(self, start, goal, n_iterations,
                 x_limits, y_limits, z_limits,
                 vx_limits, vy_limits, vz_limits,
                 goal_sample_rate=0.05,
                 neighbor_radius=0.5,
                 goal_radius=0.5,
                 tmin=0.05, tmax=3.0, n_grid=30,pyb_client=None, obstacle_ids=None, collision_radius=0.08):

        self.start = np.asarray(start, dtype=float).reshape(6,)
        self.goal  = np.asarray(goal,  dtype=float).reshape(6,)

        self.n_iterations = int(n_iterations)
        self.goal_sample_rate = float(goal_sample_rate)

        self.x_limits, self.y_limits, self.z_limits = x_limits, y_limits, z_limits
        self.vx_limits, self.vy_limits, self.vz_limits = vx_limits, vy_limits, vz_limits

        self.r = float(neighbor_radius)
        self.goal_r = float(goal_radius)

        self.tmin, self.tmax, self.n_grid = float(tmin), float(tmax), int(n_grid)

        # Tree storage
        self.nodes   = [self.start.copy()]
        self.parents = [None]
        self.costs   = [0.0]
        self.edge    = [None]   # edge[i] stores dict for parent->i

        self.goal_parent = None
        self.goal_cost = float("inf")
        self.goal_edge = None

        self.pyb_client = pyb_client
        self.obstacle_ids = set(obstacle_ids or [])
        self.collision_radius = float(collision_radius)

    def sample(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.copy()
        x  = random.uniform(*self.x_limits)
        y  = random.uniform(*self.y_limits)
        z  = random.uniform(*self.z_limits)
        vx = random.uniform(*self.vx_limits)
        vy = random.uniform(*self.vy_limits)
        vz = random.uniform(*self.vz_limits)
        return np.array([x, y, z, vx, vy, vz], dtype=float)

    def c_star(self, a, b):
        c, tau = cost_optimal_fast(a, b, tmin=self.tmin, tmax=self.tmax, n=self.n_grid)
        return c, tau

    def near_set(self, x_new):
        p_new = x_new[0:3]
        dpos_max = 0.5  # same idea as in rewire_from; tune

        idxs = []
        for i, x in enumerate(self.nodes):
            if np.linalg.norm(x[0:3] - p_new) > dpos_max:
                continue
            c, _ = self.c_star(x, x_new)
            if c < self.r:
                idxs.append(i)
        return idxs

    def choose_parent(self, x_i):
        near = self.near_set(x_i)
        if not near:
            return None, None

        best_parent = None
        best_total = float("inf")
        best_edge = None

        for i in near:
            edge = connect_star_fast(self.nodes[i], x_i,
                                tmin=self.tmin, tmax=self.tmax, n_grid=self.n_grid,
                                n_samples=10)
            if not self.edge_collision_free(edge):
                continue
            total = self.costs[i] + edge["cost"]
            if total < best_total:
                best_total = total
                best_parent = i
                best_edge = edge

        return best_parent, best_edge

    def rewire_from(self, new_index):
        x_new = self.nodes[new_index]
        p_new = x_new[0:3]

        dpos_max = 0.5   # tune

        for i, x in enumerate(self.nodes):
            if i == new_index:
                continue

            # cheap prefilter
            if np.linalg.norm(x[0:3] - p_new) > dpos_max:
                continue


            # expensive kinodynamic test only on survivors
            c, _ = self.c_star(x_new, x)
            if c >= self.r:
                continue

            edge = connect_star_fast(x_new, x, tmin=self.tmin, tmax=self.tmax, n_grid=self.n_grid, n_samples=10)
            if not self.edge_collision_free(edge):
                continue

            candidate = self.costs[new_index] + edge["cost"]
            if candidate < self.costs[i]:
                self.parents[i] = new_index
                self.costs[i] = candidate
                self.edge[i] = edge


    def try_update_goal(self, new_index):
        x_new = self.nodes[new_index]
        c, _ = self.c_star(x_new, self.goal)
        if c >= self.goal_r:
            return

        edge = connect_star_fast(x_new, self.goal,
                            tmin=self.tmin, tmax=self.tmax, n_grid=self.n_grid,
                            n_samples=20)
        if not self.edge_collision_free(edge):
            return
        total = self.costs[new_index] + edge["cost"]
        if total < self.goal_cost:
            self.goal_cost = total
            self.goal_parent = new_index
            self.goal_edge = edge

    def edge_collision_free(self, edge):
        if (self.pyb_client is None) or (not self.obstacle_ids):
            return True

        xs = edge["xs"]
        r = self.collision_radius

        offsets = np.array([
            [0, 0, 0],
            [ r, 0, 0],
            [-r, 0, 0],
            [0,  r, 0],
            [0, -r, 0],
        ], dtype=float)

        # Build all rays at once
        p0 = xs[:-1, 0:3]  # (M,3)
        p1 = xs[ 1:, 0:3]  # (M,3)

        starts = (p0[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        ends   = (p1[:, None, :] + offsets[None, :, :]).reshape(-1, 3)

        results = p.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=self.pyb_client)

        # results is list of tuples; [0] is hit body unique id
        for hit in results:
            if hit[0] in self.obstacle_ids:
                return False
        return True


            

    def build(self):
        for _ in range(self.n_iterations):
            x_i = self.sample()
            parent, edge = self.choose_parent(x_i)
            if parent is None:
                continue

            new_index = len(self.nodes)
            self.nodes.append(x_i)
            self.parents.append(parent)
            self.edge.append(edge)
            self.costs.append(self.costs[parent] + edge["cost"])

            self.rewire_from(new_index)
            self.try_update_goal(new_index)

        return self.goal_parent is not None

    def extract_trajectory_samples(self, samples_per_edge=25):
        if self.goal_parent is None:
            return None

        # backtrack node indices (tree nodes only)
        idxs = []
        i = self.goal_parent
        while i is not None:
            idxs.append(i)
            i = self.parents[i]
        idxs.reverse()

        xs_all = []
        # edges along tree (use stored edges but resample with requested density)
        for k in range(1, len(idxs)):
            child = idxs[k]
            parent = idxs[k - 1]
            edge = connect_star_fast(self.nodes[parent], self.nodes[child],
                                tmin=self.tmin, tmax=self.tmax, n_grid=self.n_grid,
                                n_samples=samples_per_edge)
            xs_all.extend(edge["xs"] if not xs_all else edge["xs"][1:])

        # final edge to goal
        edge_g = connect_star_fast(self.nodes[self.goal_parent], self.goal,
                              tmin=self.tmin, tmax=self.tmax, n_grid=self.n_grid,
                              n_samples=samples_per_edge)
        xs_all.extend(edge_g["xs"][1:] if xs_all else edge_g["xs"])

        return np.array(xs_all, dtype=float)
    

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

def draw_rrt_path_3d(path_xyz, pyb_client, line_width=2.0, life_time=0.0):
    if path_xyz is None or len(path_xyz) < 2:
        return
    for i in range(len(path_xyz) - 1):
        p1 = path_xyz[i]
        p2 = path_xyz[i + 1]
        p.addUserDebugLine(
            [float(p1[0]), float(p1[1]), float(p1[2])],
            [float(p2[0]), float(p2[1]), float(p2[2])],
            lineColorRGB=[1, 0, 0],
            lineWidth=line_width,
            lifeTime=life_time,
            physicsClientId=pyb_client,
        )

def draw_rrt_tree_3d_curved(nodes, parents, edges, pyb_client,
                            line_width=1.0, life_time=0.0,
                            line_color=(0, 1, 0),
                            max_edges=None, stride=1):
    """
    Draws the tree using the sampled trajectory stored in edges[i]["xs"].
    - edges: list where edges[i] is dict for parent->i (same as KinoRRTStar.edge)
    - max_edges: optionally limit how many edges to draw (speed)
    - stride: draw every k-th sample point along an edge (speed)
    """
    drawn = 0
    for i in range(1, len(nodes)):
        if max_edges is not None and drawn >= max_edges:
            break

        pi = parents[i]
        if pi is None:
            continue

        edge = edges[i]
        if edge is None or "xs" not in edge:
            # fallback to straight line
            p1 = nodes[pi][0:3]
            p2 = nodes[i][0:3]
            p.addUserDebugLine(p1.tolist(), p2.tolist(),
                               lineColorRGB=list(line_color),
                               lineWidth=line_width,
                               lifeTime=life_time,
                               physicsClientId=pyb_client)
            drawn += 1
            continue

        xs = edge["xs"]
        # draw polyline along sampled points
        for k in range(0, len(xs) - 1, stride):
            p1 = xs[k, 0:3]
            p2 = xs[k + 1, 0:3]
            p.addUserDebugLine(p1.tolist(), p2.tolist(),
                               lineColorRGB=list(line_color),
                               lineWidth=line_width,
                               lifeTime=life_time,
                               physicsClientId=pyb_client)
        drawn += 1

def draw_fast_begin(pyb_client):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=pyb_client)

def draw_fast_end(pyb_client):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=pyb_client)