"""
detect.py -- YUV background subtraction with chroma shadow rejection,
bottom-profile contact splitting, stereo pairing and the coverage map.

detect() returns (points, comps, fg_mask):
  points -- floor contact candidates, one per foot even if the blobs merged
  comps  -- connected components (drawing / diagnostics only)
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


# ------------------------------------------------------------ contact points
def _refine(prof, i, band):
    lo, hi = max(0, i - band), min(prof.shape[0], i + band + 1)
    win = prof[lo:hi]
    ok = win >= 0
    if not ok.any():
        return float(i), float(max(prof[i], 0))
    vmax = int(win[ok].max())
    cols = np.nonzero(ok & (win >= vmax - 1))[0] + lo
    return float(np.median(cols)), float(vmax)


def _contacts(prof, d):
    """
    prof[col] = lowest image row of the blob in that column (-1 = empty).
    Returns up to `max_contacts` (col, row) points: the *prominent* local
    minima in world height, so two merged feet still give two contacts.
    """
    n = int(prof.shape[0])
    valid = prof >= 0
    nv = int(valid.sum())
    if nv < 3:
        return []
    band = max(1, int(d.get("band_px", 3)))
    p = prof.astype(np.float32)
    if nv < n:                                   # bridge gaps so peaks stay real
        xs = np.nonzero(valid)[0]
        ms = np.nonzero(~valid)[0]
        p[ms] = np.interp(ms, xs, p[xs])
    k = _odd(d.get("prof_smooth_px", 7))
    ps = cv2.blur(p.reshape(1, -1), (k, 1)).ravel() if (k > 1 and n > k) else p

    maxc = max(1, int(d.get("max_contacts", 2)))
    sep = max(2, int(d.get("split_min_sep_px", 22)))
    prom = float(d.get("split_min_prom_px", 6))
    if maxc == 1 or n <= sep + 2:
        return [_refine(prof, int(np.argmax(ps)), band)]

    dil = cv2.dilate(ps.reshape(1, -1), np.ones((1, 2 * sep + 1), np.uint8)).ravel()
    isp = ps >= dil - 1e-3
    runs, i = [], 0
    while i < n:
        if isp[i]:
            j = i
            while j + 1 < n and isp[j + 1]:
                j += 1
            runs.append(i + int(np.argmax(ps[i:j + 1])))
            i = j + 1
        else:
            i += 1
    if not runs:
        runs = [int(np.argmax(ps))]
    runs.sort(key=lambda t: -ps[t])

    chosen = [runs[0]]
    for r in runs[1:]:
        if len(chosen) >= maxc:
            break
        ok = True
        for c in chosen:
            a, b = (r, c) if r < c else (c, r)
            if b - a < sep:
                ok = False
                break
            valley = float(ps[a:b + 1].min())
            if min(float(ps[a]), float(ps[b])) - valley < prom:
                ok = False
                break
        if ok:
            chosen.append(r)
    chosen.sort()
    return [_refine(prof, i, band) for i in chosen]


# -------------------------------------------------------------------- detector
class Detector:
    def __init__(self, geom):
        self.g = geom
        self.bg_y = self.bg_u = self.bg_v = None
        self.mask = None
        self.mask_c = None
        self.bbox = (0, 0, geom.W, geom.H)
        self._roi_sig = None
        self._lim_sig = None
        self._lim = self._ymin = self._bgc = None
        self._k_sig = None
        self._ko = self._kc = None
        self._bg_stamp = 0
        self.dbg_cd = None                     # chroma-distance map (view mode 3)

    # ---------------- background ----------------
    def set_background(self, frames, cfg):
        ys = [f[0] for f in frames if f is not None and f[0] is not None]
        if len(ys) < 2:
            print("[bg] not enough frames -- background unchanged")
            return False
        shp = ys[0].shape
        ys = [f for f in ys if f.shape == shp]
        med = np.median(np.stack(ys), axis=0)
        k = _odd(cfg["detect"]["blur_k"])
        bg = np.clip(med, 0, 255).astype(np.uint8)
        self.bg_y = cv2.blur(bg, (k, k)) if k > 1 else bg
        us = [f[1] for f in frames if f is not None and f[1] is not None]
        vs = [f[2] for f in frames if f is not None and f[2] is not None]
        cshape = (shp[0] // 2, shp[1] // 2)
        us = [a for a in us if a.shape == cshape]
        vs = [a for a in vs if a.shape == cshape]
        if len(us) >= 2 and len(vs) >= 2:
            self.bg_u = cv2.blur(np.clip(np.median(np.stack(us), axis=0), 0, 255)
                                 .astype(np.uint8), (3, 3))
            self.bg_v = cv2.blur(np.clip(np.median(np.stack(vs), axis=0), 0, 255)
                                 .astype(np.uint8), (3, 3))
        else:
            self.bg_u = self.bg_v = None
            print("[bg] no chroma planes -- shadow rejection disabled")
        self._bg_stamp += 1
        self._lim_sig = None
        return True

    def save_background(self, path):
        d = {"y": self.bg_y}
        if self.bg_u is not None:
            d["u"], d["v"] = self.bg_u, self.bg_v
        np.savez(path, **d)

    def load_background(self, path):
        try:
            if path.endswith(".npz"):
                z = np.load(path)
                y = z["y"]
                u = z["u"] if "u" in z.files else None
                v = z["v"] if "v" in z.files else None
            else:
                y, u, v = np.load(path), None, None
        except Exception as e:
            print(f"[bg] {path} unreadable ({e})")
            return False
        if y.ndim != 2 or y.shape != (self.g.H, self.g.W):
            print(f"[bg] {path} shape {y.shape} != {(self.g.H, self.g.W)}; ignored")
            return False
        self.bg_y = np.ascontiguousarray(y, dtype=np.uint8)
        cshape = (self.g.H // 2, self.g.W // 2)
        self.bg_u = np.ascontiguousarray(u, np.uint8) if (u is not None and u.shape == cshape) else None
        self.bg_v = np.ascontiguousarray(v, np.uint8) if (v is not None and v.shape == cshape) else None
        self._bg_stamp += 1
        self._lim_sig = None
        return True

    # ---------------- ROI ----------------
    def rebuild_roi(self, cfg):
        d = cfg["detect"]
        bx0, bx1, by0, by1 = play_bounds(cfg)
        zmax = d["roi_zmax_cm"] / 100.0
        sig = (round(bx0, 4), round(bx1, 4), round(by0, 4), round(by1, 4),
               round(zmax, 4), round(self.g.yaw, 6), round(self.g.pitch, 6),
               round(self.g.roll, 6), round(self.g.f, 4),
               tuple(np.round(self.g.C, 6)))
        if sig == self._roi_sig:
            return
        self._roi_sig = sig
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
            self.mask = self.mask_c = None
            self.bbox = (0, 0, self.g.W, self.g.H)
            print("[roi] EMPTY -- check camera pose / cutoff side")
            return
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        x0 -= x0 % 2                                  # keep chroma sub-sampling exact
        y0 -= y0 % 2
        x1 = min(self.g.W, x1 + (x1 % 2))
        y1 = min(self.g.H, y1 + (y1 % 2))
        self.bbox = (x0, y0, x1, y1)
        self.mask = np.ascontiguousarray(mask)
        self.mask_c = np.ascontiguousarray(mask[y0:y1, x0:x1])
        self._lim_sig = None
        cov = 100.0 * len(xs) / (self.g.W * self.g.H)
        print(f"[roi] {cov:.0f}% of frame, bbox={self.bbox}")

    # ---------------- cached thresholds / kernels ----------------
    def _update_limits(self, d):
        sig = (int(d["diff_thresh"]), int(d["rel_thresh_pct"]),
               int(d["shadow_ymin_pct"]), self.bbox, self._bg_stamp)
        if sig == self._lim_sig:
            return
        self._lim_sig = sig
        x0, y0, x1, y1 = self.bbox
        bgc = np.ascontiguousarray(self.bg_y[y0:y1, x0:x1])
        self._bgc = bgc
        b16 = bgc.astype(np.uint16)
        self._lim = np.clip(b16 * int(d["rel_thresh_pct"]) // 100
                            + int(d["diff_thresh"]), 0, 255).astype(np.uint8)
        self._ymin = np.clip(b16 * int(d["shadow_ymin_pct"]) // 100,
                             0, 255).astype(np.uint8)

    def _kernels(self, d):
        sig = (int(d["open_k"]), int(d["close_k"]))
        if sig == self._k_sig:
            return
        self._k_sig = sig
        self._ko = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(d["open_k"]),) * 2)
                    if int(d["open_k"]) > 1 else None)
        self._kc = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(d["close_k"]),) * 2)
                    if int(d["close_k"]) > 1 else None)

    # ---------------- detection ----------------
    def detect(self, fr, cfg):
        d = cfg["detect"]
        if fr is None or self.bg_y is None or self.mask_c is None:
            return [], [], None
        y, u, v = fr
        if y is None or y.shape != self.bg_y.shape:
            return [], [], None
        x0, y0, x1, y1 = self.bbox
        cur = y[y0:y1, x0:x1]
        k = _odd(d["blur_k"])
        if k > 1:
            cur = cv2.blur(cur, (k, k))
        self._update_limits(d)
        self._kernels(d)

        fg = cv2.compare(cur, self._lim, cv2.CMP_GT)          # brighter than bg
        use_c = (int(d.get("use_chroma", 1)) and u is not None
                 and self.bg_u is not None)
        if use_c:
            hx0, hy0, hx1, hy1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
            du = cv2.absdiff(u[hy0:hy1, hx0:hx1], self.bg_u[hy0:hy1, hx0:hx1])
            dv = cv2.absdiff(v[hy0:hy1, hx0:hx1], self.bg_v[hy0:hy1, hx0:hx1])
            cd = cv2.resize(cv2.add(du, dv), (x1 - x0, y1 - y0),
                            interpolation=cv2.INTER_NEAREST)
            self.dbg_cd = cd
            drop = cv2.subtract(self._bgc, cur)               # >0 where darker
            dark = cv2.threshold(drop, float(d["dark_obj_thresh"]), 255,
                                 cv2.THRESH_BINARY)[1]
            chrom = cv2.threshold(cd, float(d["chroma_thresh"]), 255,
                                  cv2.THRESH_BINARY)[1]
            fg = cv2.bitwise_or(fg, chrom)                    # sock over wood
            fg = cv2.bitwise_or(fg, dark)                     # dark objects
            # shadow := darker, same hue as the floor, and not pitch black
            flat = cv2.threshold(cd, float(d["shadow_chroma_tol"]), 255,
                                 cv2.THRESH_BINARY_INV)[1]
            lit = cv2.compare(cur, self._ymin, cv2.CMP_GT)
            shadow = cv2.bitwise_and(dark, cv2.bitwise_and(flat, lit))
            fg = cv2.bitwise_and(fg, cv2.bitwise_not(shadow))
        else:
            self.dbg_cd = None

        fg = cv2.bitwise_and(fg, self.mask_c)
        if self._ko is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._ko)
        if self._kc is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kc)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8, cv2.CV_32S)
        min_a, max_a = int(d["min_area"]), int(d["max_area"])
        cands = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                 if min_a <= int(stats[i, cv2.CC_STAT_AREA]) <= max_a]
        cands.sort(reverse=True)
        cands = cands[:max(1, int(d["max_blobs"]))]

        comps, pts = [], []
        for a, i in cands:
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            sl = lab[by:by + bh, bx:bx + bw] == i
            has = sl.any(axis=0)
            prof = (bh - 1 - np.argmax(sl[::-1, :], axis=0)).astype(np.int32)
            prof[~has] = -1
            cs = _contacts(prof, d)
            if not cs:
                continue
            ci = len(comps)
            comps.append({"org": (bx + x0, by + y0), "wh": (bw, bh),
                          "area": a, "prof": prof, "n": len(cs)})
            for (cu, cvv) in cs:
                pts.append({"low": (bx + x0 + cu, by + y0 + cvv),
                            "area": float(a) / len(cs), "comp": ci,
                            "w": float(bw)})
        pts.sort(key=lambda b: -b["area"])
        return pts[:max(1, int(d.get("max_points", 4)))], comps, fg


# ------------------------------------------------------------------- stereo
def stereo_feet(b0, b1, g0, g1, cfg, prev):
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
            if prev:
                c += d["w_track"] * 1000.0 * min(float(np.linalg.norm(P - p)) for p in prev)
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


def mono_feet(points, used, g, cfg, cov_mono):
    d = cfg["detect"]
    if not d["mono_enable"]:
        return []
    bx0, bx1, by0, by1 = play_bounds(cfg, margin_cm=0)
    out = []
    for i, b in enumerate(points):
        if i in used:
            continue
        P, ok = g.plane_hit(np.asarray(g.dirs(*b["low"])).reshape(3))
        if not ok:
            continue
        x, yy = float(P[0]), float(P[1])
        if not (bx0 <= x <= bx1 and by0 <= yy <= by1):
            continue
        if cov_mono is not None and not cov_mono(x, yy):
            continue
        out.append({"pos": np.array([x, yy]), "z_mm": 0.0,
                    "z0_mm": 0.0, "z1_mm": 0.0, "mono": True})
    return out


# ----------------------------------------------------------------- coverage
class Coverage:
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
        iy, ix = int(round(y / self.step)), int(round(x / self.step))
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