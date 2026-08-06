"""
geometry.py -- pinhole + 2-term radial + roll camera model, and 2-ray triangulation.
"""
import math

import numpy as np


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CamGeom:
    """pixel <-> world bearing/elevation for one camera."""

    def __init__(self, c, proc_size):
        self.C = np.array(c["pos_m"], float)
        self.yaw = math.radians(c["yaw_deg"])
        self.pitch = math.radians(c["pitch_deg"] + c.get("pitch_trim_deg", 0.0))
        self.roll = math.radians(c.get("roll_deg", 0.0))
        self.k1, self.k2 = float(c.get("k1", 0.0)), float(c.get("k2", 0.0))
        self.W, self.H = int(proc_size[0]), int(proc_size[1])
        self.cx = (self.W - 1) / 2.0 + c.get("ppx", 0.0)
        self.cy = (self.H - 1) / 2.0 + c.get("ppy", 0.0)
        self.set_hfov(math.radians(c["hfov_deg"]))
        self._axes()

    # ---- intrinsics / extrinsics ----
    def set_hfov(self, hfov):
        self.hfov = hfov
        self.f = (self.W / 2.0) / math.tan(hfov / 2.0)

    def _axes(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        F = np.array([cy * cp, sy * cp, -sp])
        R0 = np.array([sy, -cy, 0.0])
        U0 = np.cross(R0, F)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        R = R0 * cr - U0 * sr                     # +roll = camera rolls clockwise
        U = U0 * cr + R0 * sr
        self.M = np.vstack([R, U, F])             # rows R,U,F (world <- cam)

    def set_pose(self, yaw, pitch, roll, C):
        self.yaw, self.pitch, self.roll = float(yaw), float(pitch), float(roll)
        self.C = np.asarray(C, float).copy()
        self._axes()

    def clone_params(self, p):
        """p = [yaw, pitch, roll, x, y, z, f]"""
        g = object.__new__(CamGeom)
        g.__dict__.update(self.__dict__)
        g.yaw, g.pitch, g.roll = float(p[0]), float(p[1]), float(p[2])
        g.C = np.array(p[3:6], float)
        g.f = max(50.0, float(p[6]))
        g.hfov = 2.0 * math.atan((g.W / 2.0) / g.f)
        g._axes()
        return g

    # ---- pixel -> world direction (vectorised) ----
    def dirs(self, u, v):
        xn = (np.asarray(u, float) - self.cx) / self.f
        yn = (np.asarray(v, float) - self.cy) / self.f
        if self.k1 or self.k2:
            xu, yu = xn.copy(), yn.copy()
            for _ in range(4):                            # invert radial model
                r2 = xu * xu + yu * yu
                d = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
                xu, yu = xn / d, yn / d
            xn, yn = xu, yu
        cam = np.stack([xn, -yn, np.ones_like(xn)], axis=-1)   # right, up, fwd
        return cam @ self.M                                    # -> world xyz

    def az_el(self, u, v):
        d = np.asarray(self.dirs(float(u), float(v)), float).reshape(3)
        return (math.atan2(d[1], d[0]),
                math.atan2(d[2], math.hypot(d[0], d[1])))

    def plane_hit(self, d, z=0.0):
        """Intersect direction array with horizontal plane z. Returns (P, valid)."""
        d = np.asarray(d, float)
        dz = d[..., 2]
        valid = dz < -1e-9
        s = np.where(valid, (z - self.C[2]) / np.where(valid, dz, -1.0), 0.0)
        return self.C + s[..., None] * d, valid & (s > 0)

    # ---- world -> pixel ----
    def project(self, P):
        d = self.M @ (np.asarray(P, float) - self.C)     # right, up, fwd
        if d[2] <= 1e-6:
            return None
        xn, yn = d[0] / d[2], -d[1] / d[2]
        r2 = xn * xn + yn * yn
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return self.cx + self.f * xn * s, self.cy + self.f * yn * s

    def project_many(self, P):
        d = (np.asarray(P, float) - self.C) @ self.M.T
        fwd = d[..., 2]
        ok = fwd > 1e-6
        den = np.where(ok, fwd, 1.0)
        xn = np.where(ok, d[..., 0] / den, 0.0)
        yn = np.where(ok, -d[..., 1] / den, 0.0)
        r2 = xn * xn + yn * yn
        s = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        u = self.cx + self.f * xn * s
        v = self.cy + self.f * yn * s
        vis = ok & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, vis


def triangulate(g0, az0, g1, az1):
    """Intersect two floor-plane bearings. -> (P_xy, t0, t1) or None."""
    d0 = np.array([math.cos(az0), math.sin(az0)])
    d1 = np.array([math.cos(az1), math.sin(az1)])
    A = np.column_stack([d0, -d1])
    if abs(np.linalg.det(A)) < 1e-7:
        return None
    t = np.linalg.solve(A, (g1.C[:2] - g0.C[:2]))
    if t[0] <= 0.02 or t[1] <= 0.02:
        return None
    return g0.C[:2] + t[0] * d0, float(t[0]), float(t[1])


def apply_geom(cfg, geoms):
    """Config -> CamGeom.  Returns True if anything actually moved."""
    import math as _m
    changed = False
    for i, nm in enumerate(("cam0", "cam1")):
        c, g = cfg["cameras"][nm], geoms[i]
        yaw = _m.radians(c["yaw_deg"])
        pitch = _m.radians(c["pitch_deg"] + c.get("pitch_trim_deg", 0.0))
        roll = _m.radians(c.get("roll_deg", 0.0))
        hf = _m.radians(c["hfov_deg"])
        C = np.array(c["pos_m"], float)
        if (abs(g.yaw - yaw) > 1e-9 or abs(g.pitch - pitch) > 1e-9
                or abs(g.roll - roll) > 1e-9 or abs(g.hfov - hf) > 1e-9
                or not np.allclose(g.C, C, atol=1e-9)):
            g.set_hfov(hf)
            g.set_pose(yaw, pitch, roll, C)
            changed = True
    return changed
