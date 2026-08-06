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

NORM_Y = 160.0
CLS_BG, CLS_BRIGHT, CLS_SHADOW, CLS_CHROMA, CLS_OPAQUE, CLS_GROW = range(6)
CLS_COLOR = np.array([[40, 40, 40],      # bg          grey
                      [0, 220, 0],       # bright      green
                      [200, 60, 0],      # shadow      blue
                      [0, 230, 200],     # chroma obj  yellow-green
                      [220, 0, 220],     # opaque obj  magenta
                      [0, 0, 255]],      # grown       red
                     np.uint8)
def _odd(v, lo=1):
    v = int(max(lo, v))
    return v | 1

def _half(a):
    return cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2),
                      interpolation=cv2.INTER_AREA)
def _norm_chroma(u, v, yh, yfloor):
    """Illumination-invariant chromaticity: a shadow leaves this unchanged."""
    s = NORM_Y / np.maximum(yh.astype(np.float32), float(yfloor))
    return ((u.astype(np.float32) - 128.0) * s,
            (v.astype(np.float32) - 128.0) * s)
# ------------------------------------------------------------ contact points
# def _refine(prof, i, band):
#     lo, hi = max(0, i - band), min(prof.shape[0], i + band + 1)
#     win = prof[lo:hi]
#     ok = win >= 0
#     if not ok.any():
#         return float(i), float(max(prof[i], 0))
#     vmax = int(win[ok].max())
#     cols = np.nonzero(ok & (win >= vmax - 1))[0] + lo
#     return float(np.median(cols)), float(vmax)
def _refine(rows, cols_valid, i, band, n):
    lo, hi = max(0, i - band), min(n - 1, i + band)
    w = np.arange(lo, hi + 1)
    w = w[cols_valid[w]]
    if w.size == 0:
        return None
    r = rows[w]
    keep = w[r >= r.max() - 1]
    return float(keep.mean()), float(rows[keep].mean())

def _contacts_from_profile(rows, t, X, Y, d):
    """
    rows[c] = lowest object row in column c (-1 = empty)
    t[c]    = horizontal distance (m) from camera to the floor point that
              column's lowest pixel would land on (inf = ray misses floor)
    Contacts are prominent local *minima of t*: nearest-to-camera points.
    Prominence/separation are in centimetres, so they are resolution- and
    distance-independent (unlike pixel rows).
    """
    n = rows.shape[0]
    valid = (rows >= 0) & np.isfinite(t)
    nv = int(valid.sum())
    if nv < 2:
        return []
    idx = np.nonzero(valid)[0]
    s = (-t).astype(np.float32)                       # higher = nearer = "lower"
    if nv < n:                                        # bridge gaps so peaks stay real
        miss = np.nonzero(~valid)[0]
        s[miss] = np.interp(miss, idx, s[idx])
    k = _odd(d["prof_smooth_px"])
    ss = (cv2.blur(s.reshape(1, -1), (k, 1)).ravel()
          if (k > 1 and n > k) else s)
    band = max(1, int(d["band_px"]))
    maxc = max(1, int(d["max_contacts"]))
    if maxc == 1 or n < 5:
        p = _refine(rows, valid, int(np.argmax(ss)), band, n)
        return [] if p is None else [p]
    win = _odd(d["split_win_px"])
    dil = cv2.dilate(ss.reshape(1, -1),
                     np.ones((1, win), np.uint8)).ravel()
    ispeak = ss >= dil - 1e-6
    peaks, i = [], 0
    while i < n:
        if ispeak[i]:
            j = i
            while j + 1 < n and ispeak[j + 1]:
                j += 1
            peaks.append(i + int(np.argmax(ss[i:j + 1])))
            i = j + 1
        else:
            i += 1
    if not peaks:
        peaks = [int(np.argmax(ss))]
    peaks.sort(key=lambda c: -ss[c])
    sep = float(d["split_min_sep_cm"]) / 100.0
    prom = float(d["split_min_prom_cm"]) / 100.0
    chosen = [peaks[0]]
    for p in peaks[1:]:
        if len(chosen) >= maxc:
            break
        ok = True
        for c in chosen:
            a, b = (p, c) if p < c else (c, p)
            if np.isfinite(X[a]) and np.isfinite(X[b]):
                if np.hypot(X[a] - X[b], Y[a] - Y[b]) < sep:   # world separation
                    ok = False
                    break
            if min(ss[a], ss[b]) - float(ss[a:b + 1].min()) < prom:
                ok = False
                break
        if ok:
            chosen.append(p)
    chosen.sort()
    out = [_refine(rows, valid, c, band, n) for c in chosen]
    return [o for o in out if o is not None]
# ----------------------------------------------------------------- detector ---
class Detector:
    def __init__(self, geom):
        self.g = geom
        self.bg_y = self.bg_u = self.bg_v = None
        self.bg_nu = self.bg_nv = None          # half-res normalised chroma
        self.bg_ystd = self.bg_cstd = None      # measured per-pixel noise
        self.mask = self.mask_c = None
        self.bbox = (0, 0, geom.W, geom.H)
        self.f_t = self.f_x = self.f_y = None   # floor-intersection LUTs
        self._roi_sig = self._lim_sig = self._k_sig = None
        self._lim = self._darklim = self._tol = self._bgc = self._bgc_s = None
        self._ko = self._kc = None
        self._bg_stamp = 0
        self.dbg_cd = None                      # chroma distance (view 3)
        self.dbg_cls = None                     # classification  (view 4)
        self.want_cls = False
        self.chroma_state = "not run yet"
        self._probe = None
    # -------------------------------------------------------- background ----
    def _derive_chroma_bg(self):
        if self.bg_u is None or self.bg_v is None or self.bg_y is None:
            self.bg_nu = self.bg_nv = None
            return
        yh = _half(self.bg_y)
        if yh.shape != self.bg_u.shape:
            self.bg_nu = self.bg_nv = None
            return
        self.bg_nu, self.bg_nv = _norm_chroma(self.bg_u, self.bg_v, yh, 40.0)
    def set_background(self, frames, cfg):
        d = cfg["detect"]
        ys = [f[0] for f in frames if f is not None and f[0] is not None]
        if len(ys) < 2:
            print("[bg] not enough frames -- background unchanged")
            return False
        shp = ys[0].shape
        ys = [a for a in ys if a.shape == shp]
        stack = np.stack(ys)
        med = np.median(stack, axis=0)
        k = _odd(d["blur_k"])
        bg = np.clip(med, 0, 255).astype(np.uint8)
        self.bg_y = cv2.blur(bg, (k, k)) if k > 1 else bg
        n = float(len(ys))                              # per-pixel temporal noise
        acc = np.zeros(shp, np.float32)
        acc2 = np.zeros(shp, np.float32)
        for a in ys:
            f = a.astype(np.float32)
            acc += f
            acc2 += f * f
        mean = acc / n
        self.bg_ystd = np.sqrt(np.maximum(acc2 / n - mean * mean, 0.0)) \
                         .clip(0, 255).astype(np.uint8)
        cs = (shp[0] // 2, shp[1] // 2)
        us = [f[1] for f in frames if f is not None and f[1] is not None
              and f[1].shape == cs]
        vs = [f[2] for f in frames if f is not None and f[2] is not None
              and f[2].shape == cs]
        if len(us) >= 2 and len(vs) >= 2:
            self.bg_u = np.clip(np.median(np.stack(us), 0), 0, 255).astype(np.uint8)
            self.bg_v = np.clip(np.median(np.stack(vs), 0), 0, 255).astype(np.uint8)
            self._derive_chroma_bg()
            # noise of the *normalised* chroma, measured frame by frame
            yfl = float(d["chroma_y_floor"])
            a1 = np.zeros(cs, np.float32)
            a2 = np.zeros(cs, np.float32)
            m = 0
            for f in frames:
                if f is None or f[1] is None or f[1].shape != cs:
                    continue
                nu, nv = _norm_chroma(f[1], f[2], _half(f[0]), yfl)
                dd = np.abs(nu - self.bg_nu) + np.abs(nv - self.bg_nv)
                a1 += dd
                a2 += dd * dd
                m += 1
            if m >= 2:
                mu = a1 / m
                self.bg_cstd = np.sqrt(np.maximum(a2 / m - mu * mu, 0)) \
                                 .clip(0, 255).astype(np.uint8)
            else:
                self.bg_cstd = np.zeros(cs, np.uint8)
        else:
            self.bg_u = self.bg_v = self.bg_nu = self.bg_nv = self.bg_cstd = None
            print("[bg] no chroma planes -- shadow rejection disabled")
        self._bg_stamp += 1
        self._lim_sig = None
        self.report_noise()
        return True
    def report_noise(self):
        if self.bg_ystd is None:
            return
        x0, y0, x1, y1 = self.bbox
        ys = self.bg_ystd[y0:y1, x0:x1]
        p = float(np.percentile(ys, 99.5))
        msg = f"[bg] luma noise p99.5 = {p:.1f} LSB"
        if self.bg_cstd is not None:
            cs = self.bg_cstd[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
            msg += f" | chroma noise p99.5 = {float(np.percentile(cs, 99.5)):.1f}"
        print(msg)
    def suggest(self, cfg):
        """Print threshold suggestions from the empty-scene statistics."""
        if self.bg_ystd is None:
            print("[auto] snap a background first (b)")
            return
        x0, y0, x1, y1 = self.bbox
        ys = self.bg_ystd[y0:y1, x0:x1]
        print(f"[auto] diff_thresh >= {int(np.ceil(np.percentile(ys, 99.9) * 4)) + 2}")
        if self.bg_cstd is not None:
            cs = self.bg_cstd[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
            print(f"[auto] chroma_thresh >= {int(np.ceil(np.percentile(cs, 99.9) * 3)) + 3}")
    def save_background(self, path):
        d = {"y": self.bg_y}
        for k in ("u", "v", "ystd", "cstd"):
            a = getattr(self, "bg_" + k)
            if a is not None:
                d[k] = a
        np.savez(path, **d)
    def load_background(self, path):
        try:
            z = np.load(path) if path.endswith(".npz") else {"y": np.load(path)}
            y = z["y"]
        except Exception as e:
            print(f"[bg] {path} unreadable ({e})")
            return False
        if y.ndim != 2 or y.shape != (self.g.H, self.g.W):
            print(f"[bg] {path} shape {y.shape} != {(self.g.H, self.g.W)}; ignored")
            return False
        files = getattr(z, "files", list(z.keys()))
        cs = (self.g.H // 2, self.g.W // 2)
        get = lambda k, s: (np.ascontiguousarray(z[k], np.uint8)
                            if k in files and z[k].shape == s else None)
        self.bg_y = np.ascontiguousarray(y, np.uint8)
        self.bg_u = get("u", cs)
        self.bg_v = get("v", cs)
        self.bg_ystd = get("ystd", y.shape)
        self.bg_cstd = get("cstd", cs)
        self._derive_chroma_bg()
        self._bg_stamp += 1
        self._lim_sig = None
        return True
    # --------------------------------------------------------------- ROI ----
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
        H, W = self.g.H, self.g.W
        vv, uu = np.mgrid[0:H, 0:W]
        dirs = self.g.dirs(uu.ravel(), vv.ravel())
        P0, ok0 = self.g.plane_hit(dirs, 0.0)            # floor LUT (also used
        fx = np.where(ok0, P0[:, 0], np.nan)             # for contact splitting)
        fy = np.where(ok0, P0[:, 1], np.nan)
        self.f_x = fx.reshape(H, W).astype(np.float32)
        self.f_y = fy.reshape(H, W).astype(np.float32)
        self.f_t = np.hypot(self.f_x - self.g.C[0],
                            self.f_y - self.g.C[1]).astype(np.float32)
        keep = np.zeros(dirs.shape[0], bool)
        for z in (0.0, zmax):
            P, ok = self.g.plane_hit(dirs, z)
            keep |= (ok & (P[:, 0] > bx0) & (P[:, 0] < bx1)
                     & (P[:, 1] > by0) & (P[:, 1] < by1))
        mask = keep.reshape(H, W).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(xs) < 100:
            self.mask = self.mask_c = None
            self.bbox = (0, 0, W, H)
            print("[roi] EMPTY -- check camera pose / cutoff side")
            return
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
        x0 -= x0 % 2                                     # keep chroma alignment
        y0 -= y0 % 2
        x1 = min(W, x1 + (x1 % 2))
        y1 = min(H, y1 + (y1 % 2))
        self.bbox = (x0, y0, x1, y1)
        self.mask = np.ascontiguousarray(mask)
        self.mask_c = np.ascontiguousarray(mask[y0:y1, x0:x1])
        self._lim_sig = None
        print(f"[roi] {100.0*len(xs)/(W*H):.0f}% of frame, bbox={self.bbox}")
    # ------------------------------------------------- cached limit images ---
    def _update_limits(self, d):
        sig = (int(d["diff_thresh"]), int(d["rel_thresh_pct"]), int(d["noise_k"]),
               int(d["dark_min"]), int(d["chroma_thresh"]), self.bbox, self._bg_stamp)
        if sig == self._lim_sig:
            return
        self._lim_sig = sig
        x0, y0, x1, y1 = self.bbox
        bgc = np.ascontiguousarray(self.bg_y[y0:y1, x0:x1])
        self._bgc = bgc
        self._bgc_s = np.maximum(bgc, 1)
        b16 = bgc.astype(np.uint16)
        nk = int(d["noise_k"])
        ns = (np.ascontiguousarray(self.bg_ystd[y0:y1, x0:x1]).astype(np.uint16) * nk
              if self.bg_ystd is not None else np.zeros_like(b16))
        floor = np.maximum(ns, int(d["diff_thresh"]))          # noise-aware
        self._lim = np.clip(b16 + np.maximum(floor,
                            b16 * int(d["rel_thresh_pct"]) // 100), 0, 255).astype(np.uint8)
        self._darklim = np.clip(np.maximum(ns, int(d["dark_min"])), 0, 255).astype(np.uint8)
        if self.bg_cstd is not None:
            cs = self.bg_cstd[y0 // 2:y1 // 2, x0 // 2:x1 // 2].astype(np.uint16) * nk
            tol = np.maximum(cs, int(d["chroma_thresh"])).astype(np.float32)
            self._tol = cv2.resize(tol, (x1 - x0, y1 - y0),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            self._tol = np.full((y1 - y0, x1 - x0), float(d["chroma_thresh"]), np.float32)
    def _kernels(self, d):
        sig = (int(d["open_k"]), int(d["close_k"]))
        if sig == self._k_sig:
            return
        self._k_sig = sig
        self._ko = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(d["open_k"]),) * 2)
                    if int(d["open_k"]) > 1 else None)
        self._kc = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(d["close_k"]),) * 2)
                    if int(d["close_k"]) > 1 else None)
    # ------------------------------------------------------------ detect ----
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
        bright = cv2.compare(cur, self._lim, cv2.CMP_GT)
        drop = cv2.subtract(self._bgc, cur)                    # >0 where darker
        darker = cv2.compare(drop, self._darklim, cv2.CMP_GT)
        ratio = cv2.divide(cur, self._bgc_s, scale=100.0)      # Y/Ybg in percent
        want_c = int(d.get("use_chroma", 1))
        have_c = (u is not None and self.bg_nu is not None)
        if not want_c:
            self.chroma_state = "disabled (press u)"
        elif u is None:
            self.chroma_state = "camera sends no U/V"
        elif self.bg_nu is None:
            self.chroma_state = "background has no U/V (press b)"
        else:
            self.chroma_state = "ok"
        if want_c and have_c:
            hx0, hy0, hx1, hy1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
            nu, nv = _norm_chroma(u[hy0:hy1, hx0:hx1], v[hy0:hy1, hx0:hx1],
                                  _half(cur), d["chroma_y_floor"])
            cdh = (np.abs(nu - self.bg_nu[hy0:hy1, hx0:hx1]) +
                   np.abs(nv - self.bg_nv[hy0:hy1, hx0:hx1]))
            cd = cv2.resize(cdh, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            self.dbg_cd = np.clip(cd, 0, 255).astype(np.uint8)
            chrom = ((cd > self._tol).astype(np.uint8) * 255)
            # chroma is meaningless in near-black pixels: don't trust it there
            chrom = cv2.bitwise_and(chrom, cv2.compare(cur, int(d["chroma_min_y"]),
                                                       cv2.CMP_GT))
            opaque = cv2.bitwise_and(
                cv2.compare(ratio, int(d["shadow_lo_pct"]), cv2.CMP_LT),
                cv2.compare(drop, int(d["dark_obj_thresh"]), cv2.CMP_GT))
            obj = cv2.bitwise_or(cv2.bitwise_or(bright, chrom), opaque)
            shadow = cv2.bitwise_and(darker, cv2.bitwise_not(obj))
        else:
            self.dbg_cd = None
            chrom = opaque = None
            obj = cv2.bitwise_or(bright, cv2.bitwise_and(
                darker, cv2.compare(drop, int(d["dark_obj_thresh"]), cv2.CMP_GT)))
            shadow = cv2.bitwise_and(darker, cv2.bitwise_not(obj))
        obj = cv2.bitwise_and(obj, self.mask_c)
        # --- contact recovery: geodesic growth DOWNWARD into attached dark pixels
        grow_n = int(np.clip(d["contact_grow_px"], 0, 8))
        fg = obj
        grown = None
        if grow_n:
            kern = np.ones((2, 1), np.uint8)               # anchor bottom -> grows down
            dark_ok = cv2.bitwise_and(darker, self.mask_c)
            fg = obj.copy()
            for _ in range(grow_n):
                nxt = cv2.dilate(fg, kern, anchor=(0, 1))
                nxt = cv2.bitwise_and(nxt, cv2.bitwise_or(dark_ok, fg))
                if cv2.countNonZero(cv2.absdiff(nxt, fg)) == 0:
                    break
                fg = nxt
            grown = cv2.bitwise_and(fg, cv2.bitwise_not(obj))
        if self.want_cls:                                   # debug view 4
            cls = np.zeros(fg.shape, np.uint8)
            cls[shadow > 0] = CLS_SHADOW
            if opaque is not None:
                cls[opaque > 0] = CLS_OPAQUE
            if chrom is not None:
                cls[chrom > 0] = CLS_CHROMA
            cls[bright > 0] = CLS_BRIGHT
            if grown is not None:
                cls[grown > 0] = CLS_GROW
            cls[self.mask_c == 0] = CLS_BG
            self.dbg_cls = cls
        else:
            self.dbg_cls = None
        self._probe = (cur, self._bgc, ratio, drop, self.dbg_cd, self._tol)
        if self._ko is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._ko)
        if self._kc is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kc)
        # --------------------------------------------------------- blobs ----
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8, cv2.CV_32S)
        min_a, max_a = int(d["min_area"]), int(d["max_area"])
        cands = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                 if min_a <= int(stats[i, cv2.CC_STAT_AREA]) <= max_a]
        cands.sort(reverse=True)
        cands = cands[:max(1, int(d["max_blobs"]))]
        rej = int(d.get("reject_edge_px", 0))
        comps, pts = [], []
        for a, i in cands:
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            sl = lab[by:by + bh, bx:bx + bw] == i
            has = sl.any(axis=0)
            rows = (bh - 1 - np.argmax(sl[::-1, :], axis=0)).astype(np.int32)
            rows[~has] = -1
            gy, gx = by + y0, bx + x0                     # -> processing frame
            rr = np.clip(rows, 0, None) + gy
            cc = np.arange(bw) + gx
            tt = np.where(has, self.f_t[rr, cc], np.inf).astype(np.float32)
            XX = np.where(has, self.f_x[rr, cc], np.nan).astype(np.float32)
            YY = np.where(has, self.f_y[rr, cc], np.nan).astype(np.float32)
            cs = _contacts_from_profile(rows, tt, XX, YY, d)
            if not cs:
                continue
            ci = len(comps)
            comps.append({"org": (gx, gy), "wh": (bw, bh), "area": a,
                          "prof": rows, "n": len(cs)})
            for (cu, cvv) in cs:
                pu, pv = gx + cu, gy + cvv
                if rej and (pv >= y1 - 1 - rej or pu <= x0 + rej or pu >= x1 - 1 - rej):
                    continue                              # silhouette clipped by ROI
                iu, iv = int(round(pu)), int(round(pv))
                pts.append({"low": (pu, pv), "area": float(a) / len(cs),
                            "comp": ci, "w": float(bw),
                            "t": float(self.f_t[iv, iu]),
                            "xy": (float(self.f_x[iv, iu]), float(self.f_y[iv, iu]))})
        pts.sort(key=lambda b: -b["area"])
        return pts[:max(1, int(d.get("max_points", 4)))], comps, fg
    # --------------------------------------------------------------- probe --
    def probe(self, u, v, cfg):
        """Pixel interrogation for tuning: what is this pixel and why?"""
        if self._probe is None or self.mask_c is None:
            return "no data"
        x0, y0, x1, y1 = self.bbox
        cu, cv_ = int(u) - x0, int(v) - y0
        if not (0 <= cu < x1 - x0 and 0 <= cv_ < y1 - y0):
            return "outside ROI crop"
        cur, bgc, ratio, drop, cd, tol = self._probe
        d = cfg["detect"]
        s = (f"Y={cur[cv_,cu]:3d} bg={bgc[cv_,cu]:3d} "
             f"ratio={ratio[cv_,cu]:3d}% drop={drop[cv_,cu]:3d} "
             f"lim={self._lim[cv_,cu]:3d} darklim={self._darklim[cv_,cu]:3d}")
        if cd is not None:
            s += f" | cd={cd[cv_,cu]:3d} tol={tol[cv_,cu]:.0f}"
        if self.dbg_cls is not None:
            s += " | " + ["bg", "BRIGHT", "shadow", "CHROMA", "OPAQUE",
                          "GROWN"][int(self.dbg_cls[cv_, cu])]
        if self.f_t is not None and np.isfinite(self.f_t[int(v), int(u)]):
            s += (f" | floor {self.f_x[int(v),int(u)]*100:.1f},"
                  f"{self.f_y[int(v),int(u)]*100:.1f} cm  t={self.f_t[int(v),int(u)]:.2f} m")
        return s

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