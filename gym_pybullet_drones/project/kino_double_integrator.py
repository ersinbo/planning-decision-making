import numpy as np

EPS = 1e-9

def gramian_di_3d(t: float) -> np.ndarray:
    t = float(max(t, EPS))
    I = np.eye(3)
    return np.block([
        [(t**3)/3.0 * I, (t**2)/2.0 * I],
        [(t**2)/2.0 * I,  t        * I],
    ])

def xbar_di_3d(x0: np.ndarray, t: float) -> np.ndarray:
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    p0, v0 = x0[0:3], x0[3:6]
    return np.hstack([p0 + v0*t, v0])

def cost_fixed_tau(x0: np.ndarray, x1: np.ndarray, t: float) -> float:
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    x1 = np.asarray(x1, dtype=float).reshape(6,)
    t = float(max(t, EPS))
    e = (x1 - xbar_di_3d(x0, t)).reshape(6, 1)
    G = gramian_di_3d(t)
    d = np.linalg.solve(G, e)
    return float(t + (e.T @ d).squeeze())

def cost_optimal(x0: np.ndarray, x1: np.ndarray,
                 tmin: float = 0.05, tmax: float = 3.0, n: int = 30) -> tuple[float, float]:
    taus = np.linspace(tmin, tmax, n)
    costs = np.array([cost_fixed_tau(x0, x1, t) for t in taus], dtype=float)
    k = int(np.argmin(costs))
    return float(costs[k]), float(taus[k])

def d_at_tau(x0: np.ndarray, x1: np.ndarray, tau: float) -> np.ndarray:
    """d[tau] = G(tau)^{-1} (x1 - xbar(tau))"""
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    x1 = np.asarray(x1, dtype=float).reshape(6,)
    tau = float(max(tau, EPS))
    e = (x1 - xbar_di_3d(x0, tau)).reshape(6, 1)
    d = np.linalg.solve(gramian_di_3d(tau), e).reshape(6,)
    return d

def u_star_di_3d(t: float, tau: float, d: np.ndarray) -> np.ndarray:
    """
    From the paper: u(t) = R^{-1} B^T exp(A^T(tau - t)) d
    For 3D double integrator with R=I, B^T selects the last 3 components.
    exp(A^T s) = [[I,0],[sI,I]]
    If d = [d_p; d_v], then u(t) = (tau - t) d_p + d_v
    """
    d = np.asarray(d, dtype=float).reshape(6,) # the difference between np.array and npasarray is
    dp, dv = d[0:3], d[3:6]
    s = float(tau - t)
    return s * dp + dv

def x_star_di_3d(x0: np.ndarray, t: float, tau: float, d: np.ndarray) -> np.ndarray:
    """
    Integrate xdot = A x + B u with u(t) = (tau-t)dp + dv (closed form).
    v(t) = v0 + (tau t - t^2/2) dp + t dv
    p(t) = p0 + v0 t + (tau t^2/2 - t^3/6) dp + (t^2/2) dv
    """
    x0 = np.asarray(x0, dtype=float).reshape(6,)
    p0, v0 = x0[0:3], x0[3:6]
    d = np.asarray(d, dtype=float).reshape(6,)
    dp, dv = d[0:3], d[3:6]
    t = float(max(t, 0.0))

    v = v0 + (tau*t - 0.5*t*t)*dp + t*dv
    p = p0 + v0*t + (0.5*tau*t*t - (t**3)/6.0)*dp + 0.5*t*t*dv
    return np.hstack([p, v])

def connect_star(x0: np.ndarray, x1: np.ndarray,
                 tmin: float = 0.05, tmax: float = 3.0, n_grid: int = 30,
                 n_samples: int = 20) -> dict:
    """
    Returns optimal connection data (c*, tau*, d) and sampled trajectory.
    """
    cstar, tau = cost_optimal(x0, x1, tmin=tmin, tmax=tmax, n=n_grid)
    d = d_at_tau(x0, x1, tau)

    ts = np.linspace(0.0, tau, n_samples)
    xs = np.array([x_star_di_3d(x0, t, tau, d) for t in ts])
    us = np.array([u_star_di_3d(t, tau, d) for t in ts])

    return {"cost": cstar, "tau": tau, "d": d, "ts": ts, "xs": xs, "us": us}
