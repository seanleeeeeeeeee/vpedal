"""
ui.py -- sliders, draggable 4-pane layout, top-down map, raw-stdin key reader.
"""
import math
import select
import sys

import cv2
import numpy as np

from boardmap import cutoff_m, zone_enabled

try:
    import termios
    import tty
    HAVE_TTY = True
except ImportError:
    HAVE_TTY = False


# ------------------------------------------------------------------ helpers
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


# name, cfg path, lo, hi, step, decimals, unit (cfg value = slider value * unit)
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
    ("Cut low side",   ("limits", "cutoff_low_side"),  0, 1, 1, 0, 1.0),
    ("Mirror zones X", ("zones", "mirror_x"),          0, 1, 1, 0, 1.0),
    ("Mirror zones Y", ("zones", "mirror_y"),          0, 1, 1, 0, 1.0),
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
    ("Cam1 height mm", ("cameras", "cam1", "pos_m", 2),       30.0, 150.0, 0.5, 1, 0.001),
    _H("- MIDI -"),
    ("Vel min",        ("midi", "vel_min"),            1, 127, 1, 0, 1.0),
    ("Vel max",        ("midi", "vel_max"),            1, 127, 1, 0, 1.0),
]

FOV_LINKS = {"FOV c0 deg": ("cameras", "cam1", "hfov_deg"),
             "FOV c1 deg": ("cameras", "cam0", "hfov_deg")}


def sl_get(cfg, spec):
    return float(cfg_get(cfg, spec[1])) / spec[6]


def sl_set(cfg, spec, v):
    _, path, lo, hi, step, dec, unit = spec
    v = float(np.clip(round(float(v) / step) * step, lo, hi))
    cfg_set(cfg, path, int(round(v)) if (dec == 0 and unit == 1.0) else round(v * unit, 6))
    if spec[0] in FOV_LINKS and cfg["view"]["fov_link"]:
        cfg_set(cfg, FOV_LINKS[spec[0]], round(v, 3))


# -------------------------------------------------------------------- panel
class Panel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.live = [i for i, s in enumerate(SLIDERS) if s[1] is not None]
        self.sel = self.live[0]
        self.rows, self.drag = [], None

    def draw(self, w, h):
        w, h = max(40, w), max(40, h)
        img = np.full((h, w, 3), 26, np.uint8)
        rh, top = 24, 4
        cap = max(1, (h - top) // rh)
        ncol = max(1, int(math.ceil(len(SLIDERS) / cap)))
        cw = w // ncol
        self.rows = []
        for i, spec in enumerate(SLIDERS):
            c, r = i // cap, i % cap
            x, y = c * cw + 4, top + r * rh
            if y + rh > h or x + 20 > w:
                continue
            name, path, lo, hi, step, dec, unit = spec
            if path is None:
                cv2.putText(img, name, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, (120, 160, 255), 1, cv2.LINE_AA)
                continue
            bw = max(10, cw - 12)
            val = sl_get(self.cfg, spec)
            fr = (val - lo) / float(hi - lo)
            sel = (i == self.sel)
            cv2.rectangle(img, (x, y + 13), (x + bw, y + 20), (55, 55, 55), -1)
            if lo < 0 < hi:                                # bipolar: centre tick
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
        k = self.live.index(self.sel) if self.sel in self.live else 0
        self.sel = self.live[(k + d) % len(self.live)]
        spec = SLIDERS[self.sel]
        print(f"[slider] {spec[0]} = {sl_get(self.cfg, spec):.{spec[5]}f}")

    def nudge(self, mult):
        spec = SLIDERS[self.sel]
        if spec[1] is None:
            return
        sl_set(self.cfg, spec, sl_get(self.cfg, spec) + spec[4] * mult)
        print(f"[slider] {spec[0]} = {sl_get(self.cfg, spec):.{spec[5]}f}")


# ------------------------------------------------------------------- layout
class Layout:
    """4 panes / 3 draggable boundaries, single fullscreen canvas."""
    GRAB = 7

    def __init__(self, cfg, panel):
        self.f = cfg["layout"]
        self.panel = panel
        self.drag = None
        self.rects, self.lines = {}, {"h": 1, "v": 1, "b": 1}
        self.canvas = (1280, 720)

    def compute(self, W, H):
        self.canvas = (W, H)
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


# -------------------------------------------------------------------- paint
def blit(dst, rect, img, pad=2):
    x0, y0, x1, y1 = rect
    x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8 or img is None:
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


def draw_topdown(cfg, zones, active, feet, cov, show_cov, px=900):
    w, dp = cfg["board"]["width_m"], cfg["board"]["depth_m"]
    sc = px / w
    W, H = int(w * sc), int(dp * sc) + 40
    img = np.full((H, W, 3), 22, np.uint8)
    if show_cov and cov is not None:
        img[:int(dp * sc), :] = cov.image(W, int(dp * sc))
    for z in zones:
        pts = (z["poly"].reshape(-1, 2) * sc).astype(np.int32)
        on = zone_enabled(z, cfg)
        if z["midi"] in active:
            cv2.fillPoly(img, [pts], (0, 200, 0))
        elif z["black"]:
            cv2.fillPoly(img, [pts], (45, 45, 45) if on else (30, 20, 20))
        edge = ((120, 120, 120) if not z["black"] else (90, 90, 160)) if on else (40, 40, 60)
        cv2.polylines(img, [pts], True, edge, 1)
    cut = int(cutoff_m(cfg) * sc)
    cv2.line(img, (cut, 0), (cut, int(dp * sc)), (0, 0, 220), 2)
    arrow = -30 if cfg["limits"].get("cutoff_low_side") else 30
    cv2.arrowedLine(img, (cut, 14), (cut + arrow, 14), (0, 0, 220), 2, tipLength=0.4)
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
    names = ",".join(sorted(z["note"] for z in zones if z["midi"] in active))
    txt(img, f"ACTIVE: {names}", (8, H - 12), rel=0.028, col=(0, 255, 120))
    if len(feet) == 1 and not feet[0]["mono"]:
        f = feet[0]
        txt(img, f"z0 {f['z0_mm']:+6.1f}  z1 {f['z1_mm']:+6.1f}  "
                 f"d {f['z0_mm'] - f['z1_mm']:+6.1f} mm", (8, H - 34),
            rel=0.028, col=(255, 200, 0))
    return img


# ---------------------------------------------------------------------- keys
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
 q quit (config saved)   b snapshot background (AREA EMPTY)   s save config now
 l landmark calibration  SPACE grab landmark                  r reload note JSONs
 v cycle camera view (gray / diff / mask)      o overlay note boxes
 c toggle coverage map   x cutoff := cam1 x    k flip cut-off side
 i mirror note boxes in X                      m MIDI mute/unmute
 [ ] select slider   - = nudge   _ + nudge x10  ? this help
"""
