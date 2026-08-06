"""
config.py -- defaults, disk persistence, debounced autosave.
"""
import json
import os
import tempfile

CFG_PATH = "pedalcfg.json"
BG_PATH = "bg_cam{}.npz"
BG_PATH_OLD = "bg_cam{}.npy"

DEFAULT_CFG = {
    "app": {"autosave": 1, "autosave_delay_s": 1.5},
    "capture": {
        "cam_index": [0, 1],
        "sensor_size": [1640, 1232],     # full-FOV 2x2 binned IMX219 mode
        "process_size": [820, 616],      # ISP-scaled; all detection maths happen here
        "display_size": [512, 384],
        "canvas_size": [1280, 720],
        "fps": 40,
        "rot180": [True, True],
        "flicker_hz": 50,
        "chroma": 1                      # keep U/V planes (needed for shadow reject)
    },
    "board": {"width_m": 1.2981, "depth_m": 0.6501},
    "cameras": {
        "cam0": {"pos_m": [0.3266556, 0.0, 0.070], "yaw_deg": 45.0,
                 "pitch_deg": 26.0, "pitch_trim_deg": 0.0, "roll_deg": 0.0,
                 "hfov_deg": 62.2, "k1": 0.0, "k2": 0.0, "ppx": 0.0, "ppy": 0.0},
        "cam1": {"pos_m": [1.33254, 0.0, 0.070], "yaw_deg": 135.0,
                 "pitch_deg": 26.0, "pitch_trim_deg": 0.0, "roll_deg": 0.0,
                 "hfov_deg": 62.2, "k1": 0.0, "k2": 0.0, "ppx": 0.0, "ppy": 0.0}
    },
    "view": {"fov_link": 1, "calib_solve_fov": 1, "topdown_mirror": 1},
    "ui": {"draw_every": 1, "topdown_px": 720, "cv_threads": 2},
    "detect": {
        # luma
        "diff_thresh": 40, "rel_thresh_pct": 8, "blur_k": 5,
        # chroma / shadow model
        "use_chroma": 1, "chroma_thresh": 14, "shadow_chroma_tol": 7,
        "shadow_ymin_pct": 30, "dark_obj_thresh": 25,
        # morphology / blobs
        "open_k": 3, "close_k": 5, "min_area": 120, "max_area": 60000,
        "max_blobs": 3, "max_points": 4,
        # bottom-profile contact splitting
        "band_px": 3, "prof_smooth_px": 7, "split_min_prom_px": 6,
        "split_min_sep_px": 22, "max_contacts": 2,
        # geometry gates / timing
        "z_contact_mm": 20, "z_pair_tol_mm": 60,
        "on_frames": 2, "off_frames": 4, "mono_on_frames": 4, "snap_mm": 15,
        "roi_margin_cm": 6, "roi_zmax_cm": 30,
        "mono_enable": 1, "w_track": 0.15
    },
    "limits": {"x_cutoff_cm": 130, "cutoff_low_side": 0},
    "zones": {"black": "boxes_black.json", "white": "boxes_white.json",
              "zone_dx_mm": 0, "zone_dy_mm": 0,
              "mirror_x": 1, "mirror_y": 0},
    "midi": {"channel": 0, "vel_min": 45, "vel_max": 120, "vel_gain": 250.0},
    "layout": {"hsplit": 0.60, "vsplit": 0.50, "bsplit": 0.62},
    "ui": {"draw_every": 1, "topdown_px": 720, "cv_threads": 2, "chroma_gain": 6},
    "landmarks_m": [[0.04, 0.05], [1.26, 0.05], [1.26, 0.61], [0.04, 0.61],
                    [0.65, 0.33], [0.65, 0.60]]
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = (deep_merge(base[k], v)
                  if isinstance(v, dict) and isinstance(base.get(k), dict) else v)
    return out


def load_cfg(path=CFG_PATH):
    user = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                user = json.load(f)
            print(f"[cfg] loaded {path}")
        except Exception as e:
            print(f"[cfg] {path} unreadable ({e}); using defaults")
    else:
        print(f"[cfg] {path} not found; using defaults (will be created)")
    return deep_merge(DEFAULT_CFG, user)


def _jsonable(o):
    try:
        return float(o)
    except Exception:
        return str(o)


def save_cfg(cfg, path=CFG_PATH):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".pedalcfg", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cfg, f, indent=2, default=_jsonable)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class ConfigStore:
    def __init__(self, path=CFG_PATH):
        self.path = path
        self.cfg = load_cfg(path)
        self._saved = self._snap()
        self._changed_at = None
        self._next_check = 0.0

    def _snap(self):
        return json.dumps(self.cfg, sort_keys=True, default=_jsonable)

    def save(self, verbose=True):
        try:
            save_cfg(self.cfg, self.path)
            self._saved = self._snap()
            self._changed_at = None
            if verbose:
                print(f"[cfg] saved {self.path}")
        except Exception as e:
            print(f"[cfg] SAVE FAILED: {e}")

    def tick(self, now):
        if not self.cfg["app"].get("autosave", 1):
            return
        if now < self._next_check:
            return
        self._next_check = now + 0.25
        cur = self._snap()
        if cur == self._saved:
            self._changed_at = None
            return
        if self._changed_at is None:
            self._changed_at = now
        elif now - self._changed_at >= float(self.cfg["app"]["autosave_delay_s"]):
            self.save(verbose=False)
            print("[cfg] autosaved")