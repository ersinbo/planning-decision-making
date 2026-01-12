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
                 tmin=0.05, tmax=3.0, n_grid=30,pyb_client=None, obstacle_ids=None, collision_radius=0.1):

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
        self.path_bias_start_iter = 2000      # when to start biasing
        self.path_bias_prob_max   = 0.7       # max probability to sample near path
        self.path_sigma_pos       = 0.15      # meters
        self.path_sigma_vel       = 0.05      # m/s
        self._cached_path = None
        self._cached_path_iter = -1
        self._path_cache_period = 200  # update every 200 iterations


    def get_cached_best_path(self, it):
        '''
        save the best path states so it can be used to find closer points
        '''
        if self.goal_parent is None:
            return None
        if (self._cached_path is None) or (it - self._cached_path_iter >= self._path_cache_period):
            self._cached_path = self.best_path_states()
            self._cached_path_iter = it
        return self._cached_path

    def best_path_states(self):
        '''
        Get the best path states from the start to the goal.
        '''

        if self.goal_parent is None:
            return None
        idxs = []
        i = self.goal_parent
        while i is not None:
            idxs.append(i)
            i = self.parents[i]
        idxs.reverse()
        states = [self.nodes[j] for j in idxs]
        states.append(self.goal)
        return np.array(states, dtype=float)
    


    def sample_near_path(self, path_states):
        '''
        Sample a state near the given path states.
        state consists of (x, y, z, vx, vy, vz)
        '''
        x = path_states[random.randrange(len(path_states))].copy()

        x[0:3] += np.random.normal(0.0, self.path_sigma_pos, size=3) # position noise
        x[3:6] += np.random.normal(0.0, self.path_sigma_vel, size=3) # velocity noise

        # clamp to bounds
        x[0] = np.clip(x[0], *self.x_limits) 
        x[1] = np.clip(x[1], *self.y_limits)
        x[2] = np.clip(x[2], *self.z_limits)
        x[3] = np.clip(x[3], *self.vx_limits)
        x[4] = np.clip(x[4], *self.vy_limits)
        x[5] = np.clip(x[5], *self.vz_limits)
        return x


    def sample(self, it=0):
        '''
        Sample a state from the RRT* tree.

        '''
        # Goal bias
        if random.random() < self.goal_sample_rate:
            return self.goal.copy()

        # If a solution exists bias towards path
        if (self.goal_parent is not None) and (it >= self.path_bias_start_iter): # If a solution exists bias towards path and after number of iterations
            denom = max(1, self.n_iterations - self.path_bias_start_iter)
            p_bias = self.path_bias_prob_max * (it - self.path_bias_start_iter) / denom
            p_bias = float(np.clip(p_bias, 0.0, self.path_bias_prob_max))

            if random.random() < p_bias:
                path_states = self.get_cached_best_path(it)
                if path_states is not None and len(path_states) >= 2:
                    return self.sample_near_path(path_states)

        # Otherwise uniform sampling
        x  = random.uniform(*self.x_limits)
        y  = random.uniform(*self.y_limits)
        z  = random.uniform(*self.z_limits)
        vx = random.uniform(*self.vx_limits)
        vy = random.uniform(*self.vy_limits)
        vz = random.uniform(*self.vz_limits)
        return np.array([x, y, z, vx, vy, vz], dtype=float)



    def c_star(self, a, b):
        # Compute the cost and time to go from state a to state b
        c, tau = cost_optimal_fast(a, b, tmin=self.tmin, tmax=self.tmax, n=self.n_grid)
        return c, tau

    def near_set(self, x_new):
        '''
        Find all nodes near the new node.
        '''
        p_new = x_new[0:3]

        dpos_max = 0.7
        d2_max = dpos_max * dpos_max

        idxs = []
        for i, x in enumerate(self.nodes):
            if np.sum((x[0:3] - p_new)**2) > d2_max: 
                continue

            c, _ = self.c_star(x, x_new)
            if c < self.r:
                idxs.append(i)
        return idxs


    def choose_parent(self, x_i):
        '''
        Choose the best parent for the new node.
        '''
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
        '''
        Rewire the tree from the new node.
        '''
        x_new = self.nodes[new_index]
        p_new = x_new[0:3]

        dpos_max = 0.8   # tune

        for i, x in enumerate(self.nodes):
            if i == new_index:
                continue

            # cheap prefilter for certain distance
            if np.linalg.norm(x[0:3] - p_new) > dpos_max:
                continue

            # expensive kinodynamic test
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
        '''
        Try to update the goal node.
        '''
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
        '''
        Check if the edge is collision-free.
        '''
        if (self.pyb_client is None) or (not self.obstacle_ids):
            return True

        xs = edge["xs"]
        r = self.collision_radius

        offsets = np.array([
            [0, 0, 0],
            [ r, 0, 0],
            [-r, 0, 0],
            [0,  0.2*r, 0],
            [0, -r, 0],
            [0, 0, 0.1*r],
            [0, 0, -1*r],
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


            

    def build(self,
            max_iterations=None,
            patience=3000,
            min_delta=1e-3,
            warmup=1000,
            verbose=False):
        """
        Run until:
        - max_iterations reached, OR
        - goal found and best goal cost hasn't improved by >= min_delta
            for 'patience' iterations (after warmup).

        patience: number of iterations with no significant improvement before stopping
        min_delta: required improvement to reset patience
        warmup: ignore early-stopping before this many iterations (lets tree settle)
        """

        if max_iterations is None:
            max_iterations = self.n_iterations
        max_iterations = int(max_iterations)

        best_cost = float("inf")
        last_improve_iter = 0

        for it in range(max_iterations):
            x_i = self.sample(it)
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

            # ---- early stopping logic ----
            if self.goal_parent is not None:
                if self.goal_cost + min_delta < best_cost:
                    best_cost = self.goal_cost
                    last_improve_iter = it
                    if verbose:
                        print(f"[it={it}] goal_cost improved -> {best_cost:.6f}")

                # only allow stopping after warmup
                if it >= warmup and (it - last_improve_iter) >= patience:
                    if verbose:
                        print(f"[it={it}] early stop: no improvement in {patience} iters. best={best_cost:.6f}")
                    break

        return self.goal_parent is not None


    def extract_trajectory_samples(self, samples_per_edge=25):
        '''
        Extract trajectory samples from the RRT tree for visualization.
        '''
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
    '''
    Draw the RRT tree in 3D.
    '''
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
    '''
    Draw the RRT path in 3D.
    '''
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
                            line_width=0.2, life_time=0.0,
                            line_color=(0, 1, 0),
                            max_edges=None, stride=1):
    '''
    Draw the RRT tree in 3D with curved edges.
    '''

    EDGE_STRIDE = 10
    drawn = 0

    for i in range(1, len(nodes)):
        if i % EDGE_STRIDE != 0:
            continue

        if max_edges is not None and drawn >= max_edges:
            break

        pi = parents[i]
        if pi is None:
            continue

        edge = edges[i]
        if edge is None or "xs" not in edge:
            p1 = nodes[pi][0:3]
            p2 = nodes[i][0:3]
            p.addUserDebugLine(
                p1.tolist(), p2.tolist(),
                lineColorRGB=list(line_color),
                lineWidth=line_width,
                lifeTime=life_time,
                physicsClientId=pyb_client
            )
            drawn += 1
            continue

        xs = edge["xs"]
        for k in range(0, len(xs) - 1, stride):
            p1 = xs[k, 0:3]
            p2 = xs[k + 1, 0:3]
            p.addUserDebugLine(
                p1.tolist(), p2.tolist(),
                lineColorRGB=list(line_color),
                lineWidth=line_width,
                lifeTime=life_time,
                physicsClientId=pyb_client
            )

        drawn += 1


def draw_fast_begin(pyb_client):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=pyb_client)

def draw_fast_end(pyb_client):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=pyb_client)

def path_length_xyz(points: np.ndarray) -> float:
    """
    Total Euclidean length of a 3D polyline.
    points: (N,3) array
    """
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))