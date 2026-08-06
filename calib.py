"""
calib.py -- Levenberg-Marquardt pose/FOV refinement from floor landmarks.
"""
import math

import numpy as np

PARAM_NAMES = ["yaw", "pitch", "roll", "x", "y", "z", "f"]
PARAM_STEPS = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-2]


def free_params(n_obs, solve_fov):
    fr = ["yaw", "pitch"]
    if n_obs >= 4:
        fr += ["roll", "z"]
    if n_obs >= 6:
        fr += ["x", "y"]
    if n_obs >= 7 and solve_fov:
        fr += ["f"]
    return fr


def refine_pose(geom, obs, free, iters=120):
    """obs = [((u,v), (X,Y)), ...]  -> (params, rms_px, per_landmark_px)"""
    p = np.array([geom.yaw, geom.pitch, geom.roll,
                  geom.C[0], geom.C[1], geom.C[2], geom.f], float)
    sel = np.array([PARAM_NAMES.index(f) for f in free], int)

    def res(pv):
        gg = geom.clone_params(pv)
        out = []
        for (u, v), (X, Y) in obs:
            pr = gg.project((X, Y, 0.0))
            out += [1e3, 1e3] if pr is None else [pr[0] - u, pr[1] - v]
        return np.array(out, float)

    r = res(p)
    cost, lam = float(r @ r), 1e-3
    for _ in range(iters):
        J = np.zeros((len(r), len(sel)))
        for k, i in enumerate(sel):
            q = p.copy()
            q[i] += PARAM_STEPS[i]
            J[:, k] = (res(q) - r) / PARAM_STEPS[i]
        try:
            dx = np.linalg.solve(J.T @ J + lam * np.eye(len(sel)), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        q = p.copy()
        q[sel] += dx
        r2 = res(q)
        c2 = float(r2 @ r2)
        if c2 < cost:
            p, r, cost, lam = q, r2, c2, max(lam * 0.4, 1e-9)
        else:
            lam *= 5.0
            if lam > 1e7:
                break
    per = [math.hypot(r[2 * i], r[2 * i + 1]) for i in range(len(obs))]
    return p, math.sqrt(cost / max(1, len(r))), per
