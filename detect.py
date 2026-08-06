"""
detect.py -- background subtraction inside a world-derived ROI, stereo pairing,
mono fallback and the floor coverage map.
"""
import itertools
import math

import cv2
import numpy as np

from boardmap import play_bounds
from geometry import triangulate


def _odd(v, lo=1):
    v = int(max(lo, v))
    return v | 1


def _contours(img):
    res = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res[0] if len(res) == 2 else res[1]      # OpenCV 3/4 safe


class Detector:
    """Bright-only background subtraction inside a world-derived ROI."""

    def __init__(self, geom):
        self.g = geom
        self.bg = None                                  # full-frame blurred bg
        self.mask = None
        self.mask_c = None
        self.bbox = (0, 0, geom.W, geom.H)
        self._sig = None

    # ------------------------------------------------------------ background
    def set_background(self, frames, blur_k):
        frames = [np.ascontiguousarray(f) for f in frames if f is not None]
        if not frames:
            print("[bg] no frames captured -- background unchanged")
            return None
        shp = frames[0].shape
        frames = [f for f in frames if f.shape == shp]
        if len(frames) < 2:
            print("[bg] not enough consistent frames -- background unchanged")
            return None
        stack = np.stack(frames).astype(np.float32)
        bg = np.median(stack, axis=0)
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        k = _odd(blur_k)
        self.bg = cv2.GaussianBlur(bg, (k, k), 0)
        return self.bg

    def load_background(self, path):
        try:
            bg = np.load(path)
        except Exception as e:
            print(f"[bg] {path} unreadable ({e})")
            return False
        if bg.ndim != 2 or bg.shape != (self.g.H, self.g.W):
            print(f"[bg] {path} shape {bg.shape} != {(self.g.H, self.g.W)}; ignored")
            return False
        self.bg = np.ascontiguousarray(bg, dtype=np.uint8)
        return True

    # ------------------------------------------------------------------- ROI
    def rebuild_roi(self, cfg):
        """Keep pixels whose ray pierces the play volume. Cached by signature."""
        d = cfg["detect"]
        bx0, bx1, by0, by1 = play_bounds(cfg)
        zmax = d["roi_zmax_cm"] / 100.0
        sig = (round(bx0, 4), round(bx1, 4), round(by0, 4), round(by1, 4),
               round(zmax, 4), round(self.g.yaw, 6), round(self.g.pitch, 6),
               round(self.g.roll, 6), round(self.g.f, 4),
               tuple(np.round(self.g.C, 6)))
        if sig == self._sig:
            return
        self._sig = sig
        vv, uu = np.mgrid[0:self.g.H, 0:self.g.W]
        dirs = self.g.dirs(uu.ravel(), vv.ravel())
        keep = np.zeros(dirs.shape[0], bool)
        for z in (0.0, zmax):
            P, ok = self.g.plane_hit(dirs, z)
            keep |= (ok & (P[:, 0] > bx0) & (P[:, 0] < bx1)
                     & (P[:, 1] > by0) & (P[:, 1] < by1))
        mask = keep.reshape(self.g.H, self.g.W).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(xs) < 100:
            self.mask, self.mask_c = None, None
            self.bbox = (0, 0, self.g.W, self.g.H)
            print("[roi] EMPTY -- check camera pose / cutoff side")
            return
        self.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self.mask = np.ascontiguousarray(mask)
        x0, y0, x1, y1 = self.bbox
        self.mask_c = np.ascontiguousarray(mask[y0:y1, x0:x1])
        cov = 100.0 * len(xs) / (self.g.W * self.g.H)
        print(f"[roi] {cov:.0f}% of frame, bbox={self.bbox}")

    # ---------------------------------------------------------------- detect
    def detect(self, frame, cfg):
        d = cfg["detect"]
        if self.bg is None or self.mask_c is None or frame is None:
            return [], None
        if frame.shape != self.bg.shape:
            return [], None
        x0, y0, x1, y1 = self.bbox
        sub = frame[y0:y1, x0:x1]
        bgc = self.bg[y0:y1, x0:x1]
        k = _odd(d["blur_k"])
        blur = cv2.GaussianBlur(sub, (k, k), 0)
        diff = cv2.subtract(blur, bgc).astype(np.int16)   # shadows/dark legs -> 0
        # per-pixel threshold: absolute + relative to local background brightness
        lim = (bgc.astype(np.int16) * int(d["rel_thresh_pct"])) // 100
        lim += int(d["diff_thresh"])
        th = np.zeros(diff.shape, np.uint8)
        th[diff > lim] = 255
        th = cv2.bitwise_and(th, self.mask_c)
        if int(d["open_k"]) > 1:
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(d["open_k"]),) * 2)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, ko)
        if int(d["close_k"]) > 1:
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(d["close_k"]),) * 2)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kc)
        blobs = []
        for c in _contours(th):
            a = float(cv2.contourArea(c))
            if a < d["min_area"] or a > d["max_area"]:
                continue
            p = c.reshape(-1, 2)
            ymax = int(p[:, 1].max())
            band = p[p[:, 1] >= ymax - max(1, int(d["band_px"]))]
            # LOWEST POINT = floor contact point; median x is robust to spikes
            lu = float(np.median(band[:, 0])) + x0
            lv = float(ymax) + y0
            blobs.append({"c": c, "off": (x0, y0), "area": a,
                          "low": (lu, lv), "w": float(np.ptp(p[:, 0]))})
        blobs.sort(key=lambda b: -b["area"])
        return blobs[:int(d["max_blobs"])], th


# ------------------------------------------------------------------- stereo
def stereo_feet(b0, b1, g0, g1, cfg, prev):
    """Match blobs across cameras; returns (feet, used0, used1)."""
    d = cfg["detect"]
    bx0, bx1, by0, by1 = play_bounds(cfg)
    n0, n1 = len(b0), len(b1)
    if not n0 or not n1:
        return [], set(), set()
    r0 = [g0.az_el(*b["low"]) for b in b0]
    r1 = [g1.az_el(*b["low"]) for b in b1]
    BIG = 1e6
    cost = np.full((n0, n1), BIG)
    cand = {}
    for i in range(n0):
        for j in range(n1):
            tri = triangulate(g0, r0[i][0], g1, r1[j][0])
            if tri is None:
                continue
            P, t0, t1 = tri
            if not (bx0 < P[0] < bx1 and by0 < P[1] < by1):
                continue
            z0 = g0.C[2] + t0 * math.tan(r0[i][1])
            z1 = g1.C[2] + t1 * math.tan(r1[j][1])
            c = abs(z0 - z1) * 1000.0
            if prev:                                   # temporal continuity
                c += d["w_track"] * 1000.0 * min(float(np.linalg.norm(P - p))
                                                 for p in prev)
            cost[i, j] = c
            cand[(i, j)] = {"pos": P, "z_mm": (z0 + z1) * 500.0,
                            "z0_mm": z0 * 1000.0, "z1_mm": z1 * 1000.0, "mono": False}
    tol = d["z_pair_tol_mm"] + (200.0 if prev else 0.0)
    best = []
    for k in range(min(n0, n1, 2), 0, -1):
        pool = []
        for ii in itertools.combinations(range(n0), k):
            for jj in itertools.permutations(range(n1), k):
                pr = list(zip(ii, jj))
                if any(cost[a, b] >= tol for a, b in pr):
                    continue
                pool.append((sum(cost[a, b] for a, b in pr), pr))
        if pool:
            best = min(pool)[1]
            break
    feet = [cand[p] for p in best]
    return feet, {a for a, _ in best}, {b for _, b in best}


def mono_feet(blobs, used, g, cfg, cov_mono):
    """Unmatched blob -> assume contact, intersect lowest-point ray with the floor."""
    d = cfg["detect"]
    if not d["mono_enable"]:
        return []
    bx0, bx1, by0, by1 = play_bounds(cfg, margin_cm=0)
    out = []
    for i, b in enumerate(blobs):
        if i in used:
            continue
        P, ok = g.plane_hit(np.asarray(g.dirs(*b["low"])).reshape(3))
        if not ok:
            continue
        x, y = float(P[0]), float(P[1])
        if not (bx0 <= x <= bx1 and by0 <= y <= by1):
            continue                                   # hovering feet land far outside
        if cov_mono is not None and not cov_mono(x, y):
            continue                                   # stereo region: trust stereo only
        out.append({"pos": np.array([x, y]), "z_mm": 0.0,
                    "z0_mm": 0.0, "z1_mm": 0.0, "mono": True})
    return out


# ----------------------------------------------------------------- coverage
class Coverage:
    """Floor grid -> 0 dead / 1 mono / 2 stereo. Also gates the mono fallback."""

    def __init__(self, g0, g1, cfg, step=0.01, quiet=False):
        w, dp = cfg["board"]["width_m"], cfg["board"]["depth_m"]
        self.step, self.w, self.dp = step, w, dp
        xs = np.arange(0, w + step, step)
        ys = np.arange(0, dp + step, step)
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X, Y, np.zeros_like(X)], axis=-1)
        v0 = g0.project_many(P)[2]
        v1 = g1.project_many(P)[2]
        self.map = v0.astype(np.uint8) + v1.astype(np.uint8)
        self.nx, self.ny = len(xs), len(ys)
        tot = self.map.size
        self.pct = (100.0 * np.count_nonzero(self.map == 0) / tot,
                    100.0 * np.count_nonzero(self.map == 1) / tot,
                    100.0 * np.count_nonzero(self.map == 2) / tot)
        if not quiet:
            print(f"[coverage] dead {self.pct[0]:.0f}%  mono {self.pct[1]:.0f}%  "
                  f"stereo {self.pct[2]:.0f}%")

    def is_mono(self, x, y):
        iy = int(round(y / self.step))
        ix = int(round(x / self.step))
        if not (0 <= iy < self.ny and 0 <= ix < self.nx):
            return False
        return self.map[iy, ix] <= 1

    def image(self, w, h):
        col = np.zeros((self.ny, self.nx, 3), np.uint8)
        col[self.map == 0] = (0, 0, 90)
        col[self.map == 1] = (0, 70, 90)
        col[self.map == 2] = (0, 70, 0)
        return cv2.resize(col, (max(1, w), max(1, h)),
                          interpolation=cv2.INTER_NEAREST)
