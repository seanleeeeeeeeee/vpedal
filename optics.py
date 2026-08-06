# optics.py
import numpy as np

FULL_RES = (3280, 2464)
PIXEL_SIZE_MM = 1.12e-3
FOCAL_LEN_MM = 2.96  # from datasheet; recompute if you calibrate

def get_intrinsics(capture_res):
    cw, ch = capture_res
    fw, fh = FULL_RES
    scale_x = cw / fw
    scale_y = ch / fh

    fx_full = FOCAL_LEN_MM / PIXEL_SIZE_MM   # px, at full res
    fy_full = fx_full  # square pixels

    fx = fx_full * scale_x
    fy = fy_full * scale_y
    cx = cw / 2
    cy = ch / 2
    return fx, fy, cx, cy

def pixel_to_ray(u, v, capture_res):
    """Returns a unit direction vector in camera coordinates (x-right, y-down, z-forward)."""
    fx, fy, cx, cy = get_intrinsics(capture_res)
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    vec = np.array([x, y, z])
    return vec / np.linalg.norm(vec)
