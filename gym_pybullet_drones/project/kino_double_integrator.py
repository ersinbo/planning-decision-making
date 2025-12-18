import numpy as np


def gramian_di_3(t):
    I = np.eye(3)
    return np.block([
        [(t**3)/3.0 * I, (t**2)/2.0 * I],
        [(t**2)/2.0 * I,  t        * I],
    ])

def xbar(x0, t):
    """Uncontrolled dynamics for double integrator in 3D."""
    A = np.block([
        [np.eye(3), t * np.eye(3)],
        [np.zeros((3, 3)), np.eye(3)],
    ])
    return A @ x0

def cost_fixed_tau(x0, x1, t): # Compute optimal control cost for fixed time t
    e = (x1 - xbar(x0, t)).reshape(6,1)
    G = gramian_di_3(t)
    d = np.linalg.solve(G, e)
    return float(t + (e.T @ d).squeeze())

def cost_optimal(x0, x1, tmin=0.05, tmax=3.0, n=30): # Optimal control cost, tmin and tmax are in seconds of flight time, n is number of time samples.
    taus = np.linspace(tmin, tmax, n) 
    costs = [cost_fixed_tau(x0, x1, t) for t in taus]
    k = int(np.argmin(costs))
    return float(costs[k]), float(taus[k])

# if __name__ == "__main__":
#     x0 = np.array([0,0,0, 0,0,0], dtype=float)
#     x1 = np.array([0.5,0.8,0.6, 0,0,0], dtype=float)
#     c, tau = cost_optimal(x0, x1)
#     print("c* =", c, "tau* =", tau)
