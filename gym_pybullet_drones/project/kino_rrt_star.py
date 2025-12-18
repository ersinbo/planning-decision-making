import numpy as np
import random
import pybullet as p
from kino_double_integrator import cost_optimal, xbar, gramian_di_3

class kino_RRTStar_GRAPH:
    def __init__(
        self,
        start,
        goal,
        n_iterations,
        x_limits,
        y_limits,
        z_limits,
        vx_limits,
        vy_limits,
        vz_limits,
        goal_sample_rate=0.1,
        goal_threshold=0.05,
        neighbor_radius=5,

        
    ):
        self.start = start # np.array consisting  of x, y, z, vx, vy, vz coordinates
        self.goal = goal # np.array consisting of x, y, z, vx, vy, vz coordinates

        self.n_iterations = n_iterations # maximum number of iterations
        self.x_limits = x_limits 
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.vx_limits = vx_limits
        self.vy_limits = vy_limits
        self.vz_limits = vz_limits
        self.goal_sample_rate = goal_sample_rate # probability of sampling the goal
        self.goal_threshold = goal_threshold # distance threshold to consider goal reached
        self.neighbor_radius = neighbor_radius
        self.costs = [0.0]          # cost-to-come for each node
        self.children = [set()]     # adjacency: children[i] = set of children indices of node i
        self.nodes = [np.array(start)] # list to store nodes
        self.parents = [None] # list to store parent indices  

        self.best_goal_parent = None
        self.best_goal_cost = float("inf")

    def sample(self):
        """Generate random sample point """
        if random.random() < self.goal_sample_rate:
            return np.array(self.goal)
        else:
            x = random.uniform(self.x_limits[0], self.x_limits[1])
            y = random.uniform(self.y_limits[0], self.y_limits[1])
            z = random.uniform(self.z_limits[0], self.z_limits[1])
            vx = random.uniform(self.vx_limits[0], self.vx_limits[1])
            vy = random.uniform(self.vy_limits[0], self.vy_limits[1])
            vz = random.uniform(self.vz_limits[0], self.vz_limits[1])
            return np.array([x, y, z, vx, vy, vz])
        
    def near_indexes(self, q_rand, r):
        """Find indices of nodes within fixed radius r of q_rand (O(n)).
        Parameters:
        ----------
        q_rand : np.array
            The random sample point.
        r : float
            The radius within which to search for nearby nodes.
        Returns:
        -------
        near : list of int
            Indices of nodes within radius r of q_rand.
        near_edge_costs : list of float
            Costs of edges from nearby nodes to q_rand.
        """
        near = []
        near_edge_costs = []
        for i, node in enumerate(self.nodes):
            c, tau = cost_optimal(node, q_rand)
            if c <= r:
                near.append(i)
                near_edge_costs.append((c))
        return near, near_edge_costs

    def best_parent_search(self, near, near_edge_costs):

        best_parent = None
        best_edge = None
        best_cost = float("inf")

        for i, c in zip(near, near_edge_costs):
            new_cost = self.costs[i] + c
            if new_cost < best_cost:
                best_cost = new_cost
                best_parent = i
                best_edge = c

        return best_parent, best_edge, float(best_cost)

    def add_node(self, xi, xi_parent, xi_edge_cost):
        new_id = len(self.nodes)

        self.nodes.append(xi)
        self.parents.append(xi_parent)
        self.costs.append(self.costs[xi_parent] + xi_edge_cost)
        self.children.append(set())

        self.children[xi_parent].add(new_id)

        return new_id
        

    def update_best_goal(self, q_new_index):
        c_to_goal, _ = cost_optimal(self.nodes[q_new_index], self.goal)

        
        total_cost = self.costs[q_new_index] + c_to_goal

        if total_cost < self.best_goal_cost:
            self.best_goal_cost = total_cost
            self.best_goal_parent = q_new_index

    def rewire(self,q_new_index):
        """
        Rewire the RRT tree by checking if the new node can be connected to any nearby nodes
        with a lower cost path. 

        Parameters:
        ----------
        q_new_index : int
            Index of the newly added node.
        Returns:
        -------
        None
        """
        q_new = self.nodes[q_new_index] 
        r = self.neighbor_radius

        for i in range(len(self.nodes)):
            if i == q_new_index:
                continue
            c = cost_optimal(q_new, self.nodes[i])[0] # cost from new node until i node

            if  c >= r:
                continue

            new_cost = self.costs[q_new_index] + c 

            if new_cost < self.costs[i]: # if the new connection to i is shorter than before
                # rewire
                old_parent = self.parents[i]  #the old parents
                self.parents[i] = q_new_index
                self.costs[i] = new_cost

                if old_parent is not None:
                    self.children[old_parent].discard(i) # old parent is discarded:(

                self.children[q_new_index].add(i) # new parents added :)

                self.propagate_costs_from(i)
    def finalize_goal_node(self):
        """
        Insert the goal as an explicit node connected to the best parent.
        Returns:
        -------
        success : bool
        """
        if self.best_goal_parent is None:
            self.goal_index = None
            return False

        c_edge, _ = cost_optimal(
            self.nodes[self.best_goal_parent],
            self.goal
        )

        self.goal_index = len(self.nodes)
        self.nodes.append(self.goal.copy())
        self.parents.append(self.best_goal_parent)
        self.costs.append(self.costs[self.best_goal_parent] + c_edge)
        self.children.append(set())
        self.children[self.best_goal_parent].add(self.goal_index)

        return True
    
    def plan(self):
        '''
        Build the RRT* graph.
        Returns:
        -------
        success : bool
        '''
        for i in range(self.n_iterations):
            q_rand = self.sample()
            c0, tau0 = cost_optimal(self.nodes[0], q_rand)
            print("c(start->sample)=", c0, "tau=", tau0, "q_rand=", q_rand)
            near_indexes, near_edge_costs = self.near_indexes(q_rand, self.neighbor_radius)
            best_parent, best_edge, best_cost = self.best_parent_search(near_indexes, near_edge_costs)

            if best_parent is None:
                continue

            q_new_index = self.add_node(q_rand, best_parent, best_edge)
            self.update_best_goal(q_new_index)

            self.rewire(q_new_index)
        return self.finalize_goal_node()
    

    def propagate_costs_from(self, root_idx):
        """
        After rewiring root_idx, update costs of all descendants to keep
        cost-to-come consistent.
        """
        stack = [root_idx]
        while stack:
            u = stack.pop()
            for child in list(self.children[u]):
                c_edge, _ = cost_optimal(self.nodes[u], self.nodes[child])
                self.costs[child] = float(self.costs[u] + c_edge)
                stack.append(child)


    def extract_path(self, goal_index=None):
        if goal_index is None:
            goal_index = self.goal_index

        if goal_index is None:
            return None

        path = []
        idx = goal_index
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.parents[idx]

        path.reverse()
        return path

# ---------- PyBullet drawing helpers ----------
def sample_kino_edge_positions(x0, x1, n=25):
    """
    Returns an (n+1, 3) array of positions along the optimal DI trajectory
    connecting x0 -> x1.
    x0, x1: [x,y,z,vx,vy,vz]
    """
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    x1 = np.asarray(x1, dtype=float).reshape(6,)

    c, tau = cost_optimal(x0, x1)                 # optimal time
    e = x1 - xbar(x0, tau)
    d = np.linalg.solve(gramian_di_3(tau), e)     # co-state
    dp = d[:3]
    dv = d[3:]

    # control: u(t) = (tau - t)*dp + dv
    # dynamics: p' = v, v' = u
    # integrate closed form with constant dp,dv:
    # v(t) = v0 + 0.5*dp*(2*tau*t - t^2) + dv*t
    # p(t) = p0 + v0*t + dp*(tau*t^2/2 - t^3/6) + dv*(t^2/2)

    p0 = x0[:3]
    v0 = x0[3:]

    ts = np.linspace(0.0, tau, n+1)
    pts = []
    for t in ts:
        p_t = (p0
               + v0 * t
               + dp * (tau * t**2 / 2.0 - t**3 / 6.0)
               + dv * (t**2 / 2.0))
        pts.append(p_t)

    return np.vstack(pts)



def draw_rrt_tree_3d(nodes, parents, pyb_client, line_width=1.0, life_time=0.0, samples_per_edge=20):
    for i in range(1, len(nodes)):
        pi = parents[i]
        if pi is None:
            continue

        x0 = nodes[pi]
        x1 = nodes[i]

        pts = sample_kino_edge_positions(x0, x1, n=samples_per_edge)
        for k in range(len(pts)-1):
            p.addUserDebugLine(
                pts[k].tolist(),
                pts[k+1].tolist(),
                lineColorRGB=[0, 1, 0],
                lineWidth=line_width,
                lifeTime=life_time,
                physicsClientId=pyb_client,
            )
def draw_rrt_path_3d(path, pyb_client, line_width=2.0, life_time=0.0, samples_per_edge=30):
    if path is None or len(path) < 2:
        return

    for i in range(len(path) - 1):
        x0 = path[i]
        x1 = path[i + 1]

        pts = sample_kino_edge_positions(x0, x1, n=samples_per_edge)
        for k in range(len(pts)-1):
            p.addUserDebugLine(
                pts[k].tolist(),
                pts[k+1].tolist(),
                lineColorRGB=[1, 0, 0],
                lineWidth=line_width,
                lifeTime=life_time,
                physicsClientId=pyb_client,
            )