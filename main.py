"""
pedalvision.py -- camera-vision virtual organ pedalboard (Raspberry Pi 5, 2x IMX219-77)

World frame (metres, matches boxes_*.json):
  X : along the pedalboard, 0 = low C end, +X = high notes
  Y : away from the camera baseline (cameras at Y=0), + into the board
  Z : up, 0 = floor

Keys work over SSH (raw stdin) *and* in the GUI window.  See help_text().
"""
import os, sys, json, math, time, select, itertools, threading, argparse
import numpy as np
import cv2

try:
    import termios, tty
    HAVE_TTY = True
except ImportError:
    HAVE_TTY = False

try:
    import rtmidi
    HAVE_MIDI = True
except ImportError:
    HAVE_MIDI = False

CFG_PATH = "pedalcfg.json"
BG_PATH = "bg_cam{}.npy"

# --------------------------------------------------------------------------- config
DEFAULT_CFG = {
    "capture": {
        "cam_index": [0, 1],
        "sensor_size": [1640, 1232],     # forces full-FOV 2x2 binned IMX219 mode
        "process_size": [820, 616],      # ISP-scaled; all detection maths happen here
        "display_size": [512, 384],      # what we draw overlays on (perf)
        "canvas_size": [1280, 720],      # composited window (GUI upscales to screen)
        "fps": 40,                       # 1640x1232 mode tops out ~41 fps
        "rot180": [True, True],
        "flicker_hz": 50                 # 0=off; rounds locked exposure to n*10ms
    },
    "board": {"width_m": 1.2981, "depth_m": 0.6501},   # = 129.81 x 65.01 cm
    "cameras": {
        "cam0": {"pos_m": [0.03444, 0.0, 0.070], "yaw_deg": 45.0,
                 "pitch_deg": 26.0, "pitch_trim_deg": 0.0, "roll_deg": 0.0,
                 "hfov_deg": 62.2, "k1": 0.0, "k2": 0.0, "ppx": 0.0, "ppy": 0.0},
        "cam1": {"pos_m": [1.26366, 0.0, 0.070], "yaw_deg": 135.0,
                 "pitch_deg": 26.0, "pitch_trim_deg": 0.0, "roll_deg": 0.0,
                 "hfov_deg": 62.2, "k1": 0.0, "k2": 0.0, "ppx": 0.0, "ppy": 0.0}
    },
    "view": {"fov_link": 1, "calib_solve_fov": 1},
    "detect": {
        "diff_thresh": 40, "rel_thresh_pct": 8, "blur_k": 5,
        "open_k": 3, "close_k": 5, "min_area": 120, "max_area": 30000,
        "band_px": 3, "z_contact_mm": 20, "z_pair_tol_mm": 60,
        "on_frames": 2, "off_frames": 4, "mono_on_frames": 4, "snap_mm": 15,
        "roi_margin_cm": 6, "roi_zmax_cm": 30, "max_blobs": 3,
        "mono_enable": 1, "w_track": 0.15
    },
    "limits": {"x_cutoff_cm": 130},
    "zones": {"black": "boxes_black.json", "white": "boxes_white.json",
              "zone_dx_mm": 0, "zone_dy_mm": 0},
    "midi": {"channel": 0, "vel_min": 45, "vel_max": 120, "vel_gain": 250.0},
    "layout": {"hsplit": 0.60, "vsplit": 0.50, "bsplit": 0.62},
    "landmarks_m": [[0.04, 0.05], [1.26, 0.05], [1.26, 0.61], [0.04, 0.61],
                    [0.65, 0.33], [0.65, 0.60]]
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_cfg(path=CFG_PATH):
    user = {}
    if os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
    return deep_merge(DEFAULT_CFG, user)


def save_cfg(cfg, path=CFG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[cfg] saved {path}")


# --------------------------------------------------------------------------- notes
SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(name):
    name = name.strip().upper()
    s = SEMI[name[0]]
    i = 1
    while i < len(name) and name[i] in "#B":
        s += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + s


def load_zones(cfg):
    """Returns list of zone dicts, black keys first (hit-test priority)."""
    dx = cfg["zones"]["zone_dx_mm"] / 1000.0
    dy = cfg["zones"]["zone_dy_mm"] / 1000.0
    zones = []
    for key, black in (("black", True), ("white", False)):
        path = cfg["zones"][key]
        if not os.path.exists(path):
            print(f"[zones] missing {path}")
            continue
        for item in json.load(open(path)):
            v = np.asarray(item["vertices"], np.float32) + np.float32([dx, dy])
            zones.append({"note": item["note"], "midi": note_to_midi(item["note"]),
                          "black": black, "poly": v.reshape(-1, 1, 2),
                          "cx": float(v[:, 0].mean()), "cy": float(v[:, 1].mean())})
    zones.sort(key=lambda z: (not z["black"], z["cx"]))     # blacks first
    print(f"[zones] {len(zones)} keys, "
          f"{sum(z['black'] for z in zones)} sharps, "
          f"midi {min(z['midi'] for z in zones)}..{max(z['midi'] for z in zones)}")
    return zones


# --------------------------------------------------------------------------- optics
def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CamGeom:
    """Pinhole + 2-term radial + roll.  pixel <-> world bearing/elevation."""

    def __init__(self, c, proc_size):
        self.C = np.array(c["pos_m"], float)
        self.yaw = math.radians(c["yaw_deg"])
        self.pitch = math.radians(c["pitch_deg"] + c.get("pitch_trim_deg", 0.0))
        self.roll = math.radians(c.get("roll_deg", 0.0))
        self.k1, self.k2 = float(c["k1"]), float(c["k2"])
        self.W, self.H = proc_size
        self.cx = (self.W - 1) / 2.0 + c["ppx"]
        self.cy = (self.H - 1) / 2.0 + c["ppy"]
        self.set_hfov(math.radians(c["hfov_deg"]))
        self._axes()

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
        # world-DOWN expressed in image axes -> used to find the true lowest pixel
        self.down_img = (sr, cr)

    def set_pose(self, yaw, pitch, roll, C):
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.C = np.asarray(C, float)
        self._axes()

    def clone_params(self, p):
        """p = [yaw, pitch, roll, x, y, z, f]"""
        g = object.__new__(CamGeom)
        g.__dict__.update(self.__dict__)
        g.yaw, g.pitch, g.roll = p[0], p[1], p[2]
        g.C = np.array(p[3:6], float)
        g.f = max(50.0, p[6])
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
        d = self.dirs(u, v)
        return (math.atan2(d[1], d[0]),
                math.atan2(d[2], math.hypot(d[0], d[1])))

    def plane_hit(self, d, z=0.0):
        """Intersect direction array with horizontal plane z. Returns (P, valid)."""
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
        xn = np.where(ok, d[..., 0] / np.where(ok, fwd, 1), 0)
        yn = np.where(ok, -d[..., 1] / np.where(ok, fwd, 1), 0)
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


# --------------------------------------------------------------------------- camera
class Cam:
    """Threaded grabber: always hands back the newest Y plane + timestamp."""

    def __init__(self, slot, cfg):
        from picamera2 import Picamera2
        idx = cfg["capture"]["cam_index"][slot]
        self.pw, self.ph = cfg["capture"]["process_size"]
        sw, sh = cfg["capture"]["sensor_size"]
        fd = int(round(1e6 / cfg["capture"]["fps"]))
        self.picam = Picamera2(idx)
        self.rot_cpu = False
        kw = {}
        if cfg["capture"]["rot180"][slot]:
            try:
                from libcamera import Transform
                kw["transform"] = Transform(hflip=1, vflip=1)
            except Exception:
                self.rot_cpu = True
        for attempt in (0, 1):
            try:
                conf = self.picam.create_video_configuration(
                    main={"size": (self.pw, self.ph), "format": "YUV420"},
                    raw={"size": (sw, sh)},
                    controls={"FrameDurationLimits": (fd, fd)},
                    buffer_count=4, queue=False, **kw)
                self.picam.configure(conf)
                break
            except Exception as e:
                if attempt or not kw:
                    raise
                print(f"[cam{slot}] sensor flip rejected ({e}); using CPU rotate")
                kw = {}
                self.rot_cpu = cfg["capture"]["rot180"][slot]
        self.picam.start()
        self.lock = threading.Lock()
        self.frame, self.ts, self.n = None, 0, 0
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                req = self.picam.capture_request()
            except Exception:
                break
            try:
                y = req.make_array("main")[:self.ph, :self.pw].copy()
                ts = req.get_metadata().get("SensorTimestamp", time.monotonic_ns())
            finally:
                req.release()
            if self.rot_cpu:
                y = cv2.rotate(y, cv2.ROTATE_180)
            with self.lock:
                self.frame, self.ts, self.n = y, ts, self.n + 1

    def read(self):
        with self.lock:
            return (None, 0) if self.frame is None else (self.frame, self.ts)

    def lock_ae(self, flicker_hz):
        md = self.picam.capture_metadata()
        exp, gain = md["ExposureTime"], md["AnalogueGain"]
        if flicker_hz:
            q = 1e6 / (2.0 * flicker_hz)                  # 10000 us @ 50 Hz
            new = max(q, round(exp / q) * q)
            gain = float(np.clip(gain * exp / new, 1.0, 8.0))
            exp = int(new)
        ctrl = {"AeEnable": False, "AwbEnable": False,
                "ExposureTime": int(exp), "AnalogueGain": gain}
        if "ColourGains" in md:
            ctrl["ColourGains"] = md["ColourGains"]
        self.picam.set_controls(ctrl)
        return exp, gain

    def stop(self):
        self.run = False
        time.sleep(0.05)
        try:
            self.picam.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- detect
class Detector:
    """Bright-only background subtraction inside a world-derived ROI."""

    def __init__(self, geom, cfg):
        self.g = geom
        self.bg = None                 # full-frame blurred background
        self.mask = None
        self.bbox = (0, 0, geom.W, geom.H)
        self._sig = None

    def set_background(self, frames, blur_k):
        k = max(1, blur_k | 1)
        bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
        self.bg = cv2.GaussianBlur(bg, (k, k), 0)
        self._sig = None
        return self.bg

    def rebuild_roi(self, cfg):
        """Keep pixels whose ray pierces the play volume. Cached by param signature."""
        d, lim, brd = cfg["detect"], cfg["limits"], cfg["board"]
        sig = (d["roi_margin_cm"], d["roi_zmax_cm"], lim["x_cutoff_cm"],
               round(self.g.yaw, 6), round(self.g.pitch, 6), round(self.g.roll, 6),
               round(self.g.f, 4), tuple(np.round(self.g.C, 6)))
        if sig == self._sig:
            return
        self._sig = sig
        m = d["roi_margin_cm"] / 100.0
        zmax = d["roi_zmax_cm"] / 100.0
        x0, x1 = -m, lim["x_cutoff_cm"] / 100.0 + m
        y0, y1 = -m, brd["depth_m"] + m
        vv, uu = np.mgrid[0:self.g.H, 0:self.g.W]
        dirs = self.g.dirs(uu.ravel(), vv.ravel())
        keep = np.zeros(dirs.shape[0], bool)
        for z in (0.0, zmax):
            P, ok = self.g.plane_hit(dirs, z)
            keep |= ok & (P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1)
        mask = (keep.reshape(self.g.H, self.g.W).astype(np.uint8)) * 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        ys, xs = np.nonzero(mask)
        if len(xs) < 100:
            self.mask, self.bbox = None, (0, 0, self.g.W, self.g.H)
            print("[roi] EMPTY -- check camera pose/cutoff")
            return
        self.bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self.mask = mask
        x0b, y0b, x1b, y1b = self.bbox
        self.mask_c = mask[y0b:y1b, x0b:x1b]
        cov = 100.0 * len(xs) / (self.g.W * self.g.H)
        print(f"[roi] {cov:.0f}% of frame, bbox={self.bbox}")

    def detect(self, frame, cfg):
        d = cfg["detect"]
        if self.bg is None or self.mask is None:
            return [], None
        x0, y0, x1, y1 = self.bbox
        sub = frame[y0:y1, x0:x1]
        bgc = self.bg[y0:y1, x0:x1]
        k = max(1, d["blur_k"] | 1)
        g = cv2.GaussianBlur(sub, (k, k), 0)
        diff = cv2.subtract(g, bgc)                        # shadows/dark legs -> 0
        # per-pixel threshold: absolute + relative to local background brightness
        lim = cv2.add(bgc.astype(np.uint16) * d["rel_thresh_pct"] // 100,
                      np.uint16(d["diff_thresh"]))
        th = ((diff.astype(np.uint16) > lim).astype(np.uint8)) * 255
        th = cv2.bitwise_and(th, self.mask_c)
        if d["open_k"] > 1:
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d["open_k"] | 1,) * 2)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, ko)
        if d["close_k"] > 1:
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d["close_k"] | 1,) * 2)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kc)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < d["min_area"] or a > d["max_area"]:
                continue
            p = c.reshape(-1, 2)
            ymax = int(p[:, 1].max())
            band = p[p[:, 1] >= ymax - max(1, d["band_px"])]
            # LOWEST POINT = floor contact point; median x is robust to spikes
            lu = float(np.median(band[:, 0])) + x0
            lv = float(ymax) + y0
            blobs.append({"c": c, "off": (x0, y0), "area": float(a),
                          "low": (lu, lv), "w": float(p[:, 0].ptp())})
        blobs.sort(key=lambda b: -b["area"])
        return blobs[:cfg["detect"]["max_blobs"]], th


# --------------------------------------------------------------------------- stereo
def stereo_feet(b0, b1, g0, g1, cfg, prev):
    """Match blobs across cameras; returns (feet, used0, used1)."""
    d, brd, lim = cfg["detect"], cfg["board"], cfg["limits"]
    m = d["roi_margin_cm"] / 100.0
    xmax = lim["x_cutoff_cm"] / 100.0 + m
    r0 = [g0.az_el(*b["low"]) for b in b0]
    r1 = [g1.az_el(*b["low"]) for b in b1]
    n0, n1 = len(b0), len(b1)
    if not n0 or not n1:
        return [], set(), set()
    BIG = 1e6
    cost = np.full((n0, n1), BIG)
    cand = {}
    for i in range(n0):
        for j in range(n1):
            tri = triangulate(g0, r0[i][0], g1, r1[j][0])
            if tri is None:
                continue
            P, t0, t1 = tri
            if not (-m < P[0] < xmax and -m < P[1] < brd["depth_m"] + m):
                continue
            z0 = g0.C[2] + t0 * math.tan(r0[i][1])
            z1 = g1.C[2] + t1 * math.tan(r1[j][1])
            c = abs(z0 - z1) * 1000.0
            if prev:                                   # temporal continuity
                c += d["w_track"] * 1000.0 * min(np.linalg.norm(P - p) for p in prev)
            cost[i, j] = c
            cand[(i, j)] = {"pos": P, "z_mm": (z0 + z1) * 500.0,
                            "z0_mm": z0 * 1000, "z1_mm": z1 * 1000, "mono": False}
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
    d, brd, lim = cfg["detect"], cfg["board"], cfg["limits"]
    if not d["mono_enable"]:
        return []
    out = []
    for i, b in enumerate(blobs):
        if i in used:
            continue
        P, ok = g.plane_hit(g.dirs(*b["low"]))
        if not ok:
            continue
        x, y = float(P[0]), float(P[1])
        if not (0 <= x <= lim["x_cutoff_cm"] / 100.0 and 0 <= y <= brd["depth_m"]):
            continue                                    # hovering feet land far outside
        if cov_mono is not None and not cov_mono(x, y):
            continue                                    # stereo region: trust stereo only
        out.append({"pos": np.array([x, y]), "z_mm": 0.0,
                    "z0_mm": 0.0, "z1_mm": 0.0, "mono": True})
    return out


# --------------------------------------------------------------------------- coverage
class Coverage:
    """Floor grid -> 0 dead / 1 mono / 2 stereo. Also used to gate mono fallback."""

    def __init__(self, g0, g1, cfg, step=0.01):
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
        print(f"[coverage] dead {self.pct[0]:.0f}%  mono {self.pct[1]:.0f}%  "
              f"stereo {self.pct[2]:.0f}%")

    def is_mono(self, x, y):
        i = int(round(y / self.step)), int(round(x / self.step))
        if not (0 <= i[0] < self.ny and 0 <= i[1] < self.nx):
            return False
        return self.map[i] <= 1

    def image(self, w, h):
        col = np.zeros((self.ny, self.nx, 3), np.uint8)
        col[self.map == 0] = (0, 0, 90)
        col[self.map == 1] = (0, 70, 90)
        col[self.map == 2] = (0, 70, 0)
        return cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- engine
class FootTracker:
    def __init__(self):
        self.tracks = []

    def update(self, feet, dt):
        for f in feet:
            best, bd = None, 0.10
            for t in self.tracks:
                dd = float(np.linalg.norm(f["pos"] - t["pos"]))
                if dd < bd:
                    best, bd = t, dd
            f["vz"] = ((f["z_mm"] - best["z_mm"]) / 1000.0 / max(dt, 1e-3)
                       if best else 0.0)
        self.tracks = [{"pos": f["pos"], "z_mm": f["z_mm"]} for f in feet]
        return [f["pos"] for f in feet]


class NoteEngine:
    def __init__(self, cfg, zones):
        self.cfg, self.zones = cfg, zones
        self.st = {z["midi"]: {"on": 0, "off": 0, "play": False} for z in zones}
        self.active = {}
        self.out = None
        self.enabled = True
        if HAVE_MIDI:
            self.out = rtmidi.MidiOut()
            self.out.open_virtual_port("PedalVision")
            print("[midi] virtual port 'PedalVision'")
        else:
            print("[midi] python-rtmidi missing -> console only")

    def send(self, midi, on, vel=100):
        if self.out and self.enabled:
            self.out.send_message([(0x90 if on else 0x80) | self.cfg["midi"]["channel"],
                                   midi, vel if on else 0])
        print(("NOTE ON  " if on else "NOTE OFF ") + str(midi), flush=True)

    def hit(self, p, cutoff_m):
        snap = self.cfg["detect"]["snap_mm"] / 1000.0
        near, nd = None, 1e9
        for z in self.zones:                            # blacks first
            if z["cx"] > cutoff_m:
                continue
            dist = cv2.pointPolygonTest(z["poly"], (float(p[0]), float(p[1])), True)
            if dist >= 0:
                return z
            if -dist < nd:
                near, nd = z, -dist
        return near if (near and nd <= snap) else None

    def update(self, feet, cutoff_m):
        d = self.cfg["detect"]
        hits = {}
        for f in feet:
            if not f["mono"] and f["z_mm"] > d["z_contact_mm"]:
                continue
            z = self.hit(f["pos"], cutoff_m)
            if z:
                need = d["mono_on_frames"] if f["mono"] else d["on_frames"]
                v = int(np.clip(self.cfg["midi"]["vel_min"] +
                                self.cfg["midi"]["vel_gain"] * max(0.0, -f.get("vz", 0)),
                                self.cfg["midi"]["vel_min"], self.cfg["midi"]["vel_max"]))
                prev = hits.get(z["midi"])
                if prev is None or need < prev[0]:
                    hits[z["midi"]] = (need, v)
        for m, s in self.st.items():
            if m in hits:
                s["on"] += 1
                s["off"] = 0
            else:
                s["off"] += 1
                s["on"] = 0
            if not s["play"] and m in hits and s["on"] >= hits[m][0]:
                s["play"] = True
                self.send(m, True, hits[m][1])
            elif s["play"] and s["off"] >= d["off_frames"]:
                s["play"] = False
                self.send(m, False)
        self.active = {m for m, s in self.st.items() if s["play"]}

    def panic(self):
        for m, s in self.st.items():
            if s["play"]:
                s["play"] = False
                self.send(m, False)


# --------------------------------------------------------------------------- U


def cfg_get(cfg, path):
    node = cfg
    for p in path:
        node = node[p]
    return node
def cfg_set(cfg, path, v):
    node = cfg
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = v
    
def _H(t):
    return (t, None, 0, 0, 0, 0, 1.0)

SLIDERS = [
    _H("- DETECT -"),
    ("Diff thresh",    ("detect", "diff_thresh"),      1, 120, 1, 0, 1.0),
    ("Rel thresh %",   ("detect", "rel_thresh_pct"),   0, 40, 1, 0, 1.0),
    ("Blur k",         ("detect", "blur_k"),           1, 15, 2, 0, 1.0),
    ("Open k",         ("detect", "open_k"),           1, 15, 2, 0, 1.0),
    ("Close k",        ("detect", "close_k"),          1, 25, 2, 0, 1.0),
    ("Min area px",    ("detect", "min_area"),        20, 4000, 10, 0, 1.0),
    ("Max area px",    ("detect", "max_area"),      1000, 60000, 500, 0, 1.0),
    ("Band px",        ("detect", "band_px"),          1, 15, 1, 0, 1.0),
    _H("- GATES / TIMING -"),
    ("Contact mm",     ("detect", "z_contact_mm"),     0, 80, 1, 0, 1.0),
    ("Pair tol mm",    ("detect", "z_pair_tol_mm"),    5, 200, 5, 0, 1.0),
    ("Snap mm",        ("detect", "snap_mm"),          0, 40, 1, 0, 1.0),
    ("On frames",      ("detect", "on_frames"),        1, 8, 1, 0, 1.0),
    ("Off frames",     ("detect", "off_frames"),       1, 15, 1, 0, 1.0),
    ("Mono on frm",    ("detect", "mono_on_frames"),   1, 12, 1, 0, 1.0),
    ("Mono enable",    ("detect", "mono_enable"),      0, 1, 1, 0, 1.0),
    _H("- AREA / ZONES -"),
    ("X cutoff cm",    ("limits", "x_cutoff_cm"),     20, 135, 1, 0, 1.0),
    ("ROI margin cm",  ("detect", "roi_margin_cm"),    0, 30, 1, 0, 1.0),
    ("ROI z max cm",   ("detect", "roi_zmax_cm"),      5, 60, 1, 0, 1.0),
    ("Zone dx mm",     ("zones", "zone_dx_mm"),     -150, 150, 1, 0, 1.0),
    ("Zone dy mm",     ("zones", "zone_dy_mm"),     -150, 150, 1, 0, 1.0),
    _H("- OPTICS / POSE -"),
    ("FOV c0 deg",     ("cameras", "cam0", "hfov_deg"),       45.0, 130.0, 0.1, 1, 1.0),
    ("FOV c1 deg",     ("cameras", "cam1", "hfov_deg"),       45.0, 130.0, 0.1, 1, 1.0),
    ("FOV link",       ("view", "fov_link"),                     0, 1, 1, 0, 1.0),
    ("Pitch trim c0",  ("cameras", "cam0", "pitch_trim_deg"), -6.0, 6.0, 0.05, 2, 1.0),
    ("Pitch trim c1",  ("cameras", "cam1", "pitch_trim_deg"), -6.0, 6.0, 0.05, 2, 1.0),
    ("Roll c0 deg",    ("cameras", "cam0", "roll_deg"),       -8.0, 8.0, 0.05, 2, 1.0),
    ("Roll c1 deg",    ("cameras", "cam1", "roll_deg"),       -8.0, 8.0, 0.05, 2, 1.0),
    ("Cam0 height mm", ("cameras", "cam0", "pos_m", 2),       30.0, 150.0, 0.5, 1, 0.001),
    # ("Cam1 height mm", ("cameras","cam1","pos_m",2), 30.0,150.0,0.5,1,0.001),
]
FOV_LINKS = {"FOV c0 deg": ("cameras", "cam1", "hfov_deg"),
             "FOV c1 deg": ("cameras", "cam0", "hfov_deg")}

def sl_get(cfg, spec):
    return cfg_get(cfg, spec[1]) / spec[6]
def sl_set(cfg, spec, v):
    _, path, lo, hi, step, dec, unit = spec
    v = float(np.clip(round(float(v) / step) * step, lo, hi))
    cfg_set(cfg, path, int(round(v)) if (dec == 0 and unit == 1.0) else round(v * unit, 6))
    if spec[0] in FOV_LINKS and cfg["view"]["fov_link"]:
        cfg_set(cfg, FOV_LINKS[spec[0]], round(v, 3))

class Panel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.live = [i for i, s in enumerate(SLIDERS) if s[1] is not None]
        self.sel = self.live[0]
        self.rows, self.drag = [], None

    def draw(self, w, h):
        img = np.full((h, w, 3), 26, np.uint8)
        rh, top = 24, 4
        cap = max(1, (h - top) // rh)
        ncol = max(1, int(math.ceil(len(SLIDERS) / cap)))
        cw = w // ncol
        self.rows = []
        for i, spec in enumerate(SLIDERS):
            c, r = i // cap, i % cap
            x, y = c * cw + 4, top + r * rh
            if y + rh > h:
                continue
            name, path, lo, hi, step, dec, unit = spec
            if path is None:
                cv2.putText(img, name, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, (120, 160, 255), 1, cv2.LINE_AA)
                continue
            bw = cw - 12
            val = sl_get(self.cfg, spec)
            fr = (val - lo) / float(hi - lo)
            sel = (i == self.sel)
            cv2.rectangle(img, (x, y + 13), (x + bw, y + 20), (55, 55, 55), -1)
            if lo < 0 < hi:                                   # bipolar: draw centre tick
                zx = x + int(bw * (-lo) / (hi - lo))
                cv2.rectangle(img, (min(zx, x + int(bw * fr)), y + 13),
                              (max(zx, x + int(bw * fr)), y + 20),
                              (0, 190, 255) if sel else (0, 120, 170), -1)
                cv2.line(img, (zx, y + 11), (zx, y + 22), (150, 150, 150), 1)
            else:
                cv2.rectangle(img, (x, y + 13), (x + int(bw * fr), y + 20),
                              (0, 190, 255) if sel else (0, 120, 170), -1)
            cv2.putText(img, f"{name}: {val:.{dec}f}", (x, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (255, 255, 120) if sel else (200, 200, 200), 1, cv2.LINE_AA)
            self.rows.append((i, x, y, bw, rh))
        return img

    def click(self, mx, my, down):
        if down:
            for i, x, y, bw, rh in self.rows:
                if x <= mx <= x + bw and y <= my <= y + rh:
                    self.sel, self.drag = i, (i, x, bw)
                    self._set(mx)
                    return True
        elif self.drag:
            self._set(mx)
            return True
        return False

    def _set(self, mx):
        i, x, bw = self.drag
        spec = SLIDERS[i]
        fr = float(np.clip((mx - x) / max(1, bw), 0, 1))
        sl_set(self.cfg, spec, spec[2] + fr * (spec[3] - spec[2]))

    def step_sel(self, d):
        k = self.live.index(self.sel)
        self.sel = self.live[(k + d) % len(self.live)]
        print("[slider]", SLIDERS[self.sel][0], sl_get(self.cfg, SLIDERS[self.sel]))

    def nudge(self, mult):
        spec = SLIDERS[self.sel]
        sl_set(self.cfg, spec, sl_get(self.cfg, spec) + spec[4] * mult)
        print(f"[slider] {spec[0]} = {sl_get(self.cfg, spec):.{spec[5]}f}")

class Layout:
    """4 panes / 3 draggable boundaries, single fullscreen canvas."""
    GRAB = 7

    def __init__(self, cfg, panel):
        self.f = cfg["layout"]
        self.panel = panel
        self.drag = None
        self.rects = {}

    def compute(self, W, H):
        hy = int(H * self.f["hsplit"])
        vx = int(W * self.f["vsplit"])
        bx = int(W * self.f["bsplit"])
        self.rects = {"cam0": (0, 0, vx, hy), "cam1": (vx, 0, W, hy),
                      "top": (0, hy, bx, H), "panel": (bx, hy, W, H)}
        self.lines = {"h": hy, "v": vx, "b": bx}
        return self.rects

    def on_mouse(self, ev, x, y, flags, _):
        hy, vx, bx = self.lines["h"], self.lines["v"], self.lines["b"]
        if ev == cv2.EVENT_LBUTTONDOWN:
            if abs(y - hy) < self.GRAB:
                self.drag = "hsplit"
            elif y < hy and abs(x - vx) < self.GRAB:
                self.drag = "vsplit"
            elif y > hy and abs(x - bx) < self.GRAB:
                self.drag = "bsplit"
            else:
                px, py, _, _ = self.rects["panel"]
                if x >= px and y >= py:
                    self.panel.click(x - px, y - py, True)
        elif ev == cv2.EVENT_MOUSEMOVE:
            if self.drag:
                W, H = self.canvas
                v = (y / H) if self.drag == "hsplit" else (x / W)
                self.f[self.drag] = float(np.clip(v, 0.15, 0.9))
            elif flags & cv2.EVENT_FLAG_LBUTTON:
                px, py, _, _ = self.rects["panel"]
                self.panel.click(x - px, y - py, False)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drag = None
            self.panel.drag = None

def apply_geom(cfg, geoms):
    """Config -> CamGeom. Returns True if anything actually moved."""
    changed = False
    for i, nm in enumerate(("cam0", "cam1")):
        c, g = cfg["cameras"][nm], geoms[i]
        yaw = math.radians(c["yaw_deg"])
        pitch = math.radians(c["pitch_deg"] + c["pitch_trim_deg"])
        roll = math.radians(c["roll_deg"])
        hf = math.radians(c["hfov_deg"])
        C = np.array(c["pos_m"], float)
        if (abs(g.yaw - yaw) > 1e-9 or abs(g.pitch - pitch) > 1e-9
                or abs(g.roll - roll) > 1e-9 or abs(g.hfov - hf) > 1e-9
                or not np.allclose(g.C, C, atol=1e-9)):
            g.set_hfov(hf)
            g.set_pose(yaw, pitch, roll, C)
            changed = True
    return changed
def blit(dst, rect, img, pad=2):
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ox, oy = x0 + (w - nw) // 2, y0 + (h - nh) // 2
    dst[oy:oy + nh, ox:ox + nw] = r


def txt(img, s, org, scale_ref=None, rel=0.032, col=(0, 255, 255), thick=None):
    """Font size proportional to image height -> readable at any pane size."""
    h = (scale_ref or img).shape[0]
    fs = max(0.32, rel * h / 12.0)
    th = thick or max(1, int(round(fs * 1.6)))
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)


# --------------------------------------------------------------------------- keys
class KeyReader:
    def __init__(self):
        self.fd = None
        if HAVE_TTY and sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)          # cbreak keeps Ctrl-C alive

    def get(self):
        if self.fd is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def help_text():
    return """
 q quit        b snapshot background (area EMPTY)   s save config
 l landmark calibration    SPACE grab landmark      r reload note JSONs
 v cycle camera view (gray / diff / mask)           o overlay note boxes
 c toggle coverage map     x cutoff := cam1 x       m MIDI mute/unmute
 [ ] select slider     - = nudge     _ + nudge x10     ? this help
"""


# --------------------------------------------------------------------------- calib
def refine_pose(geom, obs, free, iters=120):
    names = ["yaw", "pitch", "roll", "x", "y", "z", "f"]
    steps = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-2]
    p = np.array([geom.yaw, geom.pitch, geom.roll,
                  geom.C[0], geom.C[1], geom.C[2], geom.f])
    sel = np.array([names.index(f) for f in free])

    def res(pv):
        gg = geom.clone_params(pv)
        out = []
        for (u, v), (X, Y) in obs:
            pr = gg.project((X, Y, 0.0))
            out += [1e3, 1e3] if pr is None else [pr[0] - u, pr[1] - v]
        return np.array(out)

    r, lam = res(p), 1e-3
    cost = r @ r
    for _ in range(iters):
        J = np.zeros((len(r), len(sel)))
        for k, i in enumerate(sel):
            q = p.copy()
            q[i] += steps[i]
            J[:, k] = (res(q) - r) / steps[i]
        try:
            dx = np.linalg.solve(J.T @ J + lam * np.eye(len(sel)), -J.T @ r)
        except np.linalg.LinAlgError:
            break
        q = p.copy()
        q[sel] += dx
        r2 = res(q)
        c2 = r2 @ r2
        if c2 < cost:
            p, r, cost, lam = q, r2, c2, max(lam * 0.4, 1e-9)
        else:
            lam *= 5
            if lam > 1e7:
                break
    per = [math.hypot(r[2 * i], r[2 * i + 1]) for i in range(len(obs))]
    return p, math.sqrt(cost / max(1, len(r))), per


def free_params(n_obs, solve_fov):
    fr = ["yaw", "pitch"]
    if n_obs >= 4:
        fr += ["roll", "z"]
    if n_obs >= 6:
        fr += ["x", "y"]
    if n_obs >= 7 and solve_fov:
        fr += ["f"]
    return fr
# --------------------------------------------------------------------------- topdown
def draw_topdown(cfg, zones, engine, feet, cov, show_cov, px=900):
    w, dp = cfg["board"]["width_m"], cfg["board"]["depth_m"]
    sc = px / w
    W, H = int(w * sc), int(dp * sc) + 40
    img = np.full((H, W, 3), 22, np.uint8)
    if show_cov and cov is not None:
        img[:int(dp * sc), :] = cov.image(W, int(dp * sc))
    cut = cfg["limits"]["x_cutoff_cm"] / 100.0
    for z in zones:
        pts = (z["poly"].reshape(-1, 2) * sc).astype(np.int32)
        off = z["cx"] > cut
        if z["midi"] in engine.active:
            cv2.fillPoly(img, [pts], (0, 200, 0))
        elif z["black"]:
            cv2.fillPoly(img, [pts], (45, 45, 45) if not off else (30, 20, 20))
        cv2.polylines(img, [pts], True,
                      (40, 40, 60) if off else ((120, 120, 120) if not z["black"] else (90, 90, 160)), 1)
    cv2.line(img, (int(cut * sc), 0), (int(cut * sc), int(dp * sc)), (0, 0, 220), 2)
    for name, col in (("cam0", (255, 180, 0)), ("cam1", (255, 120, 255))):
        c = cfg["cameras"][name]
        p = (int(c["pos_m"][0] * sc), int(c["pos_m"][1] * sc))
        cv2.circle(img, p, 6, col, -1)
        yaw, hf = math.radians(c["yaw_deg"]), math.radians(c["hfov_deg"]) / 2
        for a in (yaw - hf, yaw + hf):
            cv2.line(img, p, (int(p[0] + 2.0 * sc * math.cos(a)),
                              int(p[1] + 2.0 * sc * math.sin(a))), col, 1)
    for f in feet:
        x, y = int(f["pos"][0] * sc), int(f["pos"][1] * sc)
        con = f["mono"] or f["z_mm"] < cfg["detect"]["z_contact_mm"]
        cv2.circle(img, (x, y), 10, (0, 0, 255) if con else (0, 220, 255), -1 if con else 2)
        txt(img, f"{'M' if f['mono'] else ''}{f['z_mm']:.0f}mm", (x + 12, y + 4),
            col=(255, 255, 255), rel=0.022)
    names = ",".join(sorted(z["note"] for z in zones if z["midi"] in engine.active))
    txt(img, f"ACTIVE: {names}", (8, H - 12), rel=0.028, col=(0, 255, 120))
    if len(feet) == 1 and not feet[0]["mono"]:
        f = feet[0]
        txt(img, f"z0 {f['z0_mm']:+6.1f}  z1 {f['z1_mm']:+6.1f}  "
                 f"d {f['z0_mm']-f['z1_mm']:+6.1f} mm", (8, H - 34),
            rel=0.028, col=(255, 200, 0))
    return img


# --------------------------------------------------------------------------- main
def coverage_only(cfg):
    ps = cfg["capture"]["process_size"]
    g0 = CamGeom(cfg["cameras"]["cam0"], ps)
    g1 = CamGeom(cfg["cameras"]["cam1"], ps)
    cov = Coverage(g0, g1, cfg)
    zones = load_zones(cfg)
    eng = type("E", (), {"active": set()})()
    img = draw_topdown(cfg, zones, eng, [], cov, True)
    print(help_text())
    if os.environ.get("DISPLAY"):
        cv2.imshow("coverage", img)
        cv2.waitKey(0)
    cv2.imwrite("coverage.png", img)
    print("[coverage] wrote coverage.png  (green=stereo, teal=1 cam, red=dead)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.coverage:
        return coverage_only(cfg)

    gui = (not args.headless) and bool(os.environ.get("DISPLAY"))
    ps = tuple(cfg["capture"]["process_size"])
    ds = tuple(cfg["capture"]["display_size"])
    g = [CamGeom(cfg["cameras"]["cam0"], ps), CamGeom(cfg["cameras"]["cam1"], ps)]
    cams = [Cam(0, cfg), Cam(1, cfg)]
    print("[ae] settling on EMPTY scene...")
    time.sleep(2.0)
    for i, c in enumerate(cams):
        e, gn = c.lock_ae(cfg["capture"]["flicker_hz"])
        print(f"[ae] cam{i} locked {e}us gain {gn:.2f}")

    det = [Detector(g[0], cfg), Detector(g[1], cfg)]
    for i in range(2):
        if os.path.exists(BG_PATH.format(i)):
            det[i].bg = np.load(BG_PATH.format(i))
            print(f"[bg] loaded cam{i}")
        det[i].rebuild_roi(cfg)
    zones = load_zones(cfg)
    engine = NoteEngine(cfg, zones)
    tracker = FootTracker()
    cov = Coverage(g[0], g[1], cfg)
    panel = Panel(cfg)
    layout = Layout(cfg, panel)
    keys = KeyReader()

    win = "PedalVision"
    if gui:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(win, layout.on_mouse)
    print(help_text())

    view, overlay, showcov = 0, True, False
    lm_mode, lm_i, lm_obs = False, 0, [[], []]
    lm = cfg["landmarks_m"]
    fps, prev, zsig = 0.0, time.time(), None
    prev_pos = []
    scale = ds[0] / float(ps[0])
    try:
        while True:
            # -- zone offsets / cutoff changed? rebuild
            sig = (cfg["zones"]["zone_dx_mm"], cfg["zones"]["zone_dy_mm"])
            if sig != zsig:
                zsig = sig
                zones = load_zones(cfg)
                engine.zones = zones
                engine.st = {z["midi"]: engine.st.get(z["midi"],
                             {"on": 0, "off": 0, "play": False}) for z in zones}
            if apply_geom(cfg, g):
                geom_dirty, t_dirty = True, time.time()
            if geom_dirty and time.time() - t_dirty > 0.15:
                for d_ in det:
                    d_.rebuild_roi(cfg)
                cov = Coverage(g[0], g[1], cfg, quiet=True)
                geom_dirty = False
            elif not geom_dirty:
                for d_ in det:
                    d_.rebuild_roi(cfg)      # cheap no-op via sig cache
            f0, t0 = cams[0].read()
            f1, t1 = cams[1].read()
            if f0 is None or f1 is None:
                time.sleep(0.005)
                continue
            skew_ms = abs(t0 - t1) / 1e6

            now = time.time()
            dt = now - prev
            fps = 0.9 * fps + 0.1 / max(dt, 1e-6)
            prev = now

            feet, ths, blobs = [], [None, None], [[], []]
            if det[0].bg is not None and det[1].bg is not None:
                blobs[0], ths[0] = det[0].detect(f0, cfg)
                blobs[1], ths[1] = det[1].detect(f1, cfg)
                feet, u0, u1 = stereo_feet(blobs[0], blobs[1], g[0], g[1], cfg, prev_pos)
                feet += mono_feet(blobs[0], u0, g[0], cfg, cov.is_mono)
                feet += mono_feet(blobs[1], u1, g[1], cfg, cov.is_mono)
                feet = feet[:2]                                   # 2-note polyphony
                prev_pos = tracker.update(feet, dt)
                if not lm_mode:
                    engine.update(feet, cfg["limits"]["x_cutoff_cm"] / 100.0)

            # ---------------- render
            if gui:
                CW, CH = cfg["capture"]["canvas_size"]
                layout.canvas = (CW, CH)
                canvas = np.full((CH, CW, 3), 12, np.uint8)
                rects = layout.compute(CW, CH)
                for i, (fr, th) in enumerate(((f0, ths[0]), (f1, ths[1]))):
                    small = cv2.resize(fr, ds, interpolation=cv2.INTER_AREA)
                    if view == 1 and det[i].bg is not None:
                        small = cv2.resize(cv2.subtract(fr, det[i].bg), ds)
                    vis = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                    if det[i].mask is not None:
                        mm = cv2.resize(det[i].mask, ds, interpolation=cv2.INTER_NEAREST)
                        vis[mm == 0] = vis[mm == 0] // 3
                    if th is not None and view == 2:
                        full = np.zeros((ps[1], ps[0]), np.uint8)
                        x0, y0, x1, y1 = det[i].bbox
                        full[y0:y1, x0:x1] = th
                        vis[cv2.resize(full, ds, interpolation=cv2.INTER_NEAREST) > 0] = (0, 90, 0)
                    if overlay:
                        for z in zones:
                            if z["cx"] > cfg["limits"]["x_cutoff_cm"] / 100.0:
                                continue
                            pp = [g[i].project((float(p[0]), float(p[1]), 0.0))
                                  for p in z["poly"].reshape(-1, 2)]
                            if any(p is None for p in pp):
                                continue
                            pts = (np.array(pp) * scale).astype(np.int32)
                            c = (0, 200, 0) if z["midi"] in engine.active else \
                                ((90, 90, 160) if z["black"] else (70, 70, 70))
                            cv2.polylines(vis, [pts], True, c, 1)
                    for b in blobs[i]:
                        cv2.drawContours(vis, [(b["c"] * scale).astype(np.int32)], -1,
                                         (0, 255, 0), 1, offset=(int(b["off"][0] * scale),
                                                                 int(b["off"][1] * scale)))
                        cv2.circle(vis, (int(b["low"][0] * scale), int(b["low"][1] * scale)),
                                   4, (0, 0, 255), -1)
                    txt(vis, f"cam{i} {'gray diff mask'.split()[view]}", (6, 16), rel=0.030)
                    blit(canvas, rects[f"cam{i}"], vis)
                blit(canvas, rects["top"], draw_topdown(cfg, zones, engine, feet, cov, showcov))
                pr = rects["panel"]
                blit(canvas, pr, panel.draw(pr[2] - pr[0] - 4, pr[3] - pr[1] - 4), pad=2)
                for k, cl in (("h", 0), ("v", 1), ("b", 1)):
                    if k == "h":
                        cv2.line(canvas, (0, layout.lines["h"]), (CW, layout.lines["h"]),
                                 (70, 70, 70), 2)
                    elif k == "v":
                        cv2.line(canvas, (layout.lines["v"], 0),
                                 (layout.lines["v"], layout.lines["h"]), (70, 70, 70), 2)
                    else:
                        cv2.line(canvas, (layout.lines["b"], layout.lines["h"]),
                                 (layout.lines["b"], CH), (70, 70, 70), 2)
                txt(canvas, f"FPS {fps:4.1f}  skew {skew_ms:4.1f}ms  "
                            f"blobs {len(blobs[0])}/{len(blobs[1])}  "
                            f"{'MIDI' if engine.enabled else 'MUTE'}",
                    (10, CH - 12), rel=0.030)
                if lm_mode:
                    txt(canvas, f"LANDMARK {lm_i+1}/{len(lm)} at {lm[lm_i]} m -- "
                                "place marker, SPACE", (10, 24), rel=0.040, col=(0, 150, 255))
                if det[0].bg is None or det[1].bg is None:
                    txt(canvas, "PRESS 'b' WITH THE AREA EMPTY", (10, 60),
                        rel=0.045, col=(0, 0, 255))
                cv2.imshow(win, canvas)

            # ---------------- keys (stdin first, then GUI)
            k = keys.get()
            if k is None and gui:
                kk = cv2.waitKey(1) & 0xFF
                k = chr(kk) if kk != 255 else None
            elif gui:
                cv2.waitKey(1)

            if k is None:
                continue
            if k == 'q':
                break
            elif k == 'b':
                print("[bg] capturing, keep area empty...")
                st = [[], []]
                for _ in range(25):
                    for i in range(2):
                        fr, _ = cams[i].read()
                        if fr is not None:
                            st[i].append(fr.copy())
                    time.sleep(1.0 / cfg["capture"]["fps"])
                for i in range(2):
                    np.save(BG_PATH.format(i), det[i].set_background(st[i], cfg["detect"]["blur_k"]))
                print("[bg] done")
            elif k == 'l':
                lm_mode, lm_i, lm_obs = True, 0, [[], []]
                print(f"[calib] place marker at {lm[0]} m, press SPACE")
            elif k == ' ' and lm_mode:
                if len(blobs[0]) == 1 and len(blobs[1]) == 1:
                    for i in range(2):
                        lm_obs[i].append((blobs[i][0]["low"], tuple(lm[lm_i])))
                    lm_i += 1
                    print(f"[calib] grabbed {lm_i}/{len(lm)}")
                    if lm_i >= len(lm):
                        for i, nm in enumerate(("cam0", "cam1")):
                            fr = free_params(len(lm), cfg["view"]["calib_solve_fov"])
                            p, rms, per = refine_pose(g[i], lm_obs[i], fr)
                            g[i].set_hfov(2 * math.atan((g[i].W / 2.0) / p[6]))
                            g[i].set_pose(p[0], p[1], p[2], p[3:6])
                            cfg["cameras"][nm].update({
                                "yaw_deg": math.degrees(p[0]),
                                "pitch_deg": math.degrees(p[1]),
                                "pitch_trim_deg": 0.0,          # folded into the base
                                "roll_deg": math.degrees(p[2]),
                                "pos_m": [float(p[3]), float(p[4]), float(p[5])],
                                "hfov_deg": math.degrees(g[i].hfov)})
                            print(f"[calib] {nm}: yaw {math.degrees(p[0]):.2f} "
                                  f"pitch {math.degrees(p[1]):.2f} roll {math.degrees(p[2]):.2f} "
                                  f"h {p[5]*1000:.1f}mm fov {math.degrees(g[i].hfov):.2f} "
                                  f"rms {rms:.2f}px  solved={fr}")
                            print("[calib]  per-landmark px: " +
                                  " ".join(f"{e:.1f}" for e in per))
                        if abs(cfg["cameras"]["cam0"]["hfov_deg"]
                               - cfg["cameras"]["cam1"]["hfov_deg"]) > 0.5:
                            cfg["view"]["fov_link"] = 0
                            print("[calib] FOVs differ -> link off")
                        geom_dirty, t_dirty = True, 0.0
                        save_cfg(cfg, args.config)
                        lm_mode = False
                    else:
                        print(f"[calib] next: {lm[lm_i]} m")
                else:
                    print(f"[calib] need exactly 1 blob per camera "
                          f"(saw {len(blobs[0])}/{len(blobs[1])})")
            elif k == 'v':
                view = (view + 1) % 3
            elif k == 'o':
                overlay = not overlay
            elif k == 'c':
                showcov = not showcov
            elif k == 'x':
                cfg["limits"]["x_cutoff_cm"] = int(cfg["cameras"]["cam1"]["pos_m"][0] * 100)
                print("[cutoff]", cfg["limits"]["x_cutoff_cm"], "cm")
            elif k == 'm':
                engine.panic()
                engine.enabled = not engine.enabled
            elif k == 'r':
                zsig = None
            elif k == '[':
                panel.sel = (panel.sel - 1) % len(SLIDERS)
                print("[slider]", SLIDERS[panel.sel][0])
            elif k == ']':
                panel.sel = (panel.sel + 1) % len(SLIDERS)
                print("[slider]", SLIDERS[panel.sel][0])
            elif k == '-':
                panel.nudge(-1)
            elif k == '=':
                panel.nudge(1)
            elif k == '_':
                panel.nudge(-10)
            elif k == '+':
                panel.nudge(10)
            elif k == 's':
                save_cfg(cfg, args.config)
            elif k == '?':
                print(help_text())
    finally:
        engine.panic()
        keys.restore()
        for c in cams:
            c.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
