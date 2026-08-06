# flatlines.py
import numpy as np

def find_flat_segments(contour, min_len=5):
    pts = contour.reshape(-1, 2)
    y = pts[:, 1]
    dy = np.diff(y)
    flat = (dy == 0).astype(np.int8)

    # find run starts/ends via edge detection on the boolean mask
    padded = np.concatenate(([0], flat, [0]))
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends   = np.where(edges == -1)[0]  # exclusive end index

    segments = []
    for s, e in zip(starts, ends):
        seg_pts = pts[s:e+1]
        length = seg_pts[:, 0].max() - seg_pts[:, 0].min()
        if length >= min_len:
            segments.append({
                "y": int(seg_pts[0, 1]),
                "x_start": int(seg_pts[:, 0].min()),
                "x_end": int(seg_pts[:, 0].max()),
                "length": int(length),
                "midpoint": (int(seg_pts[:, 0].mean()), int(seg_pts[0, 1]))
            })

    segments.sort(key=lambda s: s["length"], reverse=True)
    return segments[:2]  # two most likely (longest) flat lines
def find_flat_segments_robust(contour, epsilon=2.0, slope_thresh=0.05, min_len=5):
    approx = cv2.approxPolyDP(contour, epsilon, closed=False).reshape(-1, 2)
    segments = []
    for i in range(len(approx) - 1):
        (x1, y1), (x2, y2) = approx[i], approx[i+1]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        slope = abs(dy / dx)
        length = abs(dx)
        if slope <= slope_thresh and length >= min_len:
            segments.append({
                "y": int((y1+y2)/2), "x_start": min(x1,x2), "x_end": max(x1,x2),
                "length": length, "slope": slope,
                "midpoint": (int((x1+x2)/2), int((y1+y2)/2))
            })
    segments.sort(key=lambda s: s["length"], reverse=True)
    return segments[:2]
