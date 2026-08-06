"""
boardmap.py -- note names, key-zone loading (with mirroring) and play-area limits.

World frame (metres):
  X : along the pedalboard, 0 = low C end, +X = high notes
  Y : away from the camera baseline (cameras at Y=0), + into the board
  Z : up, 0 = floor
"""
import json
import os

import numpy as np

SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(name):
    name = name.strip().upper()
    s = SEMI[name[0]]
    i = 1
    while i < len(name) and name[i] in "#B":
        s += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + s


# ---------------------------------------------------------------- play area
def cutoff_m(cfg):
    return cfg["limits"]["x_cutoff_cm"] / 100.0


def zone_enabled(z, cfg):
    """One single place decides which side of the cut-off is dead."""
    c = cutoff_m(cfg)
    return (z["cx"] >= c) if cfg["limits"].get("cutoff_low_side") else (z["cx"] <= c)


def play_bounds(cfg, margin_cm=None):
    """(x0, x1, y0, y1) world rectangle that detection is allowed to look at."""
    m = (cfg["detect"]["roi_margin_cm"] if margin_cm is None else margin_cm) / 100.0
    w = cfg["board"]["width_m"]
    dp = cfg["board"]["depth_m"]
    c = min(max(cutoff_m(cfg), 0.0), w)
    if cfg["limits"].get("cutoff_low_side"):
        x0, x1 = c - m, w + m
    else:
        x0, x1 = -m, c + m
    return x0, x1, -m, dp + m


# ---------------------------------------------------------------- zones
def load_zones(cfg, quiet=False):
    """
    Returns list of zone dicts, black keys first (hit-test priority).

    The JSON boxes are transformed into world coordinates:
        mirror_x : x -> board_width - x   (fixes left/right-flipped exports)
        mirror_y : y -> board_depth - y
        + zone_dx_mm / zone_dy_mm nudge
    """
    zc = cfg["zones"]
    dx = zc["zone_dx_mm"] / 1000.0
    dy = zc["zone_dy_mm"] / 1000.0
    mx = bool(zc.get("mirror_x", 0))
    my = bool(zc.get("mirror_y", 0))
    W = float(cfg["board"]["width_m"])
    D = float(cfg["board"]["depth_m"])

    zones = []
    for key, black in (("black", True), ("white", False)):
        path = zc[key]
        if not os.path.exists(path):
            print(f"[zones] missing {path}")
            continue
        try:
            items = json.load(open(path))
        except Exception as e:
            print(f"[zones] {path} unreadable ({e})")
            continue
        for item in items:
            v = np.asarray(item["vertices"], np.float64).reshape(-1, 2).copy()
            if mx:
                v[:, 0] = W - v[:, 0]
            if my:
                v[:, 1] = D - v[:, 1]
            v[:, 0] += dx
            v[:, 1] += dy
            vf = v.astype(np.float32)
            zones.append({"note": item["note"], "midi": note_to_midi(item["note"]),
                          "black": black, "poly": vf.reshape(-1, 1, 2),
                          "cx": float(vf[:, 0].mean()), "cy": float(vf[:, 1].mean())})
    zones.sort(key=lambda z: (not z["black"], z["cx"]))     # blacks first
    if zones and not quiet:
        print(f"[zones] {len(zones)} keys, {sum(z['black'] for z in zones)} sharps, "
              f"midi {min(z['midi'] for z in zones)}..{max(z['midi'] for z in zones)}, "
              f"mirror_x={int(mx)} mirror_y={int(my)}")
    return zones


def zone_signature(cfg):
    """Anything that requires reloading/retransforming the zone polygons."""
    zc = cfg["zones"]
    return (zc["zone_dx_mm"], zc["zone_dy_mm"], int(zc.get("mirror_x", 0)),
            int(zc.get("mirror_y", 0)), zc["black"], zc["white"],
            round(cfg["board"]["width_m"], 6), round(cfg["board"]["depth_m"], 6))
