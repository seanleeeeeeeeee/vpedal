import numpy as np
from picamera2 import Picamera2

try:
    import rtmidi
    MIDI_OK = True
except ImportError:
    MIDI_OK = False
    print("python-rtmidi not found -> printing notes instead of MIDI")

CONFIG_PATH = "pedalcfg.json"
BG_PATH     = "background_{}.npy"

DEFAULT_CFG = {
    "resolution": [640, 360],
    "fps": 50,
    "board": {"width_cm": 131.0, "depth_cm": 67.0},
    "cameras": {
        "cam0": {"pos_cm": [0.0,   0.0], "height_cm": 6.0, "yaw_deg": 45.0,
                 "pitch_deg": 0.0, "hfov_deg": 66.0, "k1": 0.0, "az_sign": 1},
        "cam1": {"pos_cm": [131.0, 0.0], "height_cm": 6.0, "yaw_deg": 135.0,
                 "pitch_deg": 0.0, "hfov_deg": 66.0, "k1": 0.0, "az_sign": 1}
    },
    "detect": {"diff_thresh": 45, "blur_ksize": 5, "min_area": 150,
               "morph_k": 3, "z_contact_mm": 22, "z_pair_tol_mm": 60,
               "on_frames": 2, "off_frames": 4, "oob_margin_cm": 4.0},
    "notes": {"base_midi": 36, "count": 13, "front_y_cm": 64.0,
              "back_y_cm": 22.0, "fan_focus_y_cm": 190.0, "gap_cm": 0.5},
    "midi": {"velocity": 100, "channel": 0},
    "landmarks_cm": [[12,12],[119,12],[119,55],[12,55],[65.5,33.5]],
    "trapezoid": {"cam0": [[0,80],[640,80],[640,360],[0,360]],
                  "cam1": [[0,80],[640,80],[640,360],[0,360]]}
}

# ---------------------------------------------------------------- config
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = DEFAULT_CFG | json.load(f)
    else:
        cfg = DEFAULT_CFG
    return cfg

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved config.")
	
# ---------------------------------------------------------------- camera
def setup_camera(idx, res, fps):
    cam = Picamera2(idx)
    fd = int(1e6 / fps)
    conf = cam.create_video_configuration(
        main={"size": tuple(res), "format": "YUV420"},
        controls={"FrameDurationLimits": (fd, fd)})
    cam.configure(conf)
    cam.start()
    return cam

def lock_exposure(cams):
    """Let AE settle on the empty scene, then freeze everything."""
    time.sleep(2.0)
    for cam in cams:
        md = cam.capture_metadata()
        cam.set_controls({
            "AeEnable": False, "AwbEnable": False,
            "ExposureTime": md["ExposureTime"],
            "AnalogueGain": md["AnalogueGain"],
            "ColourGains": md["ColourGains"]})
    print("Exposure/AWB locked:", md["ExposureTime"], "us, gain",
          round(md["AnalogueGain"], 2))

def get_gray(cam, res):
    a = cam.capture_array()
    return a[:res[1], :res[0]]           # Y plane of YUV420

# ---------------------------------------------------------------- optics
def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class CamGeom:
    """Pixel -> world azimuth + elevation. All lengths in cm."""
    def __init__(self, c, res):
        self.C      = np.array(c["pos_cm"], float)
        self.h      = float(c["height_cm"])
        self.yaw    = math.radians(c["yaw_deg"])
        self.pitch  = math.radians(c["pitch_deg"])
        self.k1     = float(c.get("k1", 0.0))
        self.sign   = int(c.get("az_sign", 1))
        W, H = res
        self.cx, self.cy = W / 2.0, H / 2.0
        self.f = (W / 2.0) / math.tan(math.radians(c["hfov_deg"]) / 2.0)

    def raw_angles(self, u, v):
        """Camera-frame azimuth/elevation, before yaw/pitch (radial k1 undistort)."""
        xn = (u - self.cx) / self.f
        yn = (v - self.cy) / self.f
        s  = 1.0 + self.k1 * (xn * xn + yn * yn)   # simple 1-term undistort
        xn *= s; yn *= s
        fwd, right, up = 1.0, xn, -yn
        return right, fwd, up

    def pixel_to_world(self, u, v):
        right, fwd, up = self.raw_angles(u, v)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)   # pitch-down positive
        fh = fwd * cp + up * sp
        uw = up * cp - fwd * sp
        az = wrap_pi(self.yaw + self.sign * math.atan2(right, fh))
        el = math.atan2(uw, math.hypot(right, fh))
        return az, el

def triangulate(g0, az0, g1, az1):
    d0 = np.array([math.cos(az0), math.sin(az0)])
    d1 = np.array([math.cos(az1), math.sin(az1)])
    A = np.column_stack([d0, -d1])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, g1.C - g0.C)
    if t[0] <= 1.0 or t[1] <= 1.0:       # behind / at camera
        return None
    return g0.C + t[0] * d0, t[0], t[1]

# ---------------------------------------------------------------- detection
def make_trapezoid_mask(w, h, pts):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
    return m

def detect_blobs(gray, bg, mask, p):
    """Bright-only background subtraction. Returns list of blob dicts."""
    k = p["blur_ksize"] | 1
    g = cv2.GaussianBlur(gray, (k, k), 0)
    diff = cv2.subtract(g, bg)                     # <-- shadows & dark legs die here
    _, th = cv2.threshold(diff, p["diff_thresh"], 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_and(th, mask)
    mk = max(1, p["morph_k"])
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kern)
    th = cv2.dilate(th, kern)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < p["min_area"]:
            continue
        pts = c.reshape(-1, 2)
        ymax = pts[:, 1].max()
        band = pts[pts[:, 1] >= ymax - 2]           # bottom 3-px band -> stable
        low = (float(band[:, 0].mean()), float(ymax))   # LOWEST POINT = contact point
        blobs.append({"contour": c, "area": area, "low": low})
    blobs.sort(key=lambda b: -b["area"])
    return blobs[:2], th                            # keep 2 biggest (feet)

# ---------------------------------------------------------------- stereo pairing
def pair_feet(blobs0, blobs1, g0, g1, cfg):
    d = cfg["detect"]; brd = cfg["board"]
    m = d["oob_margin_cm"]
    rays0 = [(b, *g0.pixel_to_world(*b["low"])) for b in blobs0]
    rays1 = [(b, *g1.pixel_to_world(*b["low"])) for b in blobs1]
    n = min(len(rays0), len(rays1))
    if n == 0:
        return []
    best, best_cost = [], 1e9
    idx1_perms = itertools.permutations(range(len(rays1)), n)
    for i0s in itertools.permutations(range(len(rays0)), n):
        for i1s in itertools.permutations(range(len(rays1)), n):
            feet, cost, ok = [], 0.0, True
            for a, b in zip(i0s, i1s):
                _, az0, el0 = rays0[a]
                _, az1, el1 = rays1[b]
                tri = triangulate(g0, az0, g1, az1)
                if tri is None:
                    ok = False; break
                P, t0, t1 = tri
                z0 = g0.h + t0 * math.tan(el0)      # cm above floor, per camera
                z1 = g1.h + t1 * math.tan(el1)
                z = (z0 + z1) / 2.0
                cost += abs(z0 - z1) * 10.0         # mm-ish consistency cost
                if not (-m <= P[0] <= brd["width_cm"] + m and
                        -m <= P[1] <= brd["depth_cm"] + m):
                    cost += 500.0
                if abs(z0 - z1) * 10.0 > d["z_pair_tol_mm"]:
                    cost += 500.0
                feet.append({"pos": P, "z_mm": z * 10.0})
            if ok and cost < best_cost:
                best, best_cost = feet, cost
    return [f for f in best if best_cost < 400.0]

# ---------------------------------------------------------------- note zones
def build_zones(cfg):
    n = cfg["notes"]; W = cfg["board"]["width_cm"]
    F = np.array([W / 2.0, n["fan_focus_y_cm"]])        # fan focus behind player
    yf, yb, gap = n["front_y_cm"], n["back_y_cm"], n["gap_cm"]
    zones = []
    kw = W / n["count"]
    for i in range(n["count"]):
        x0, x1 = i * kw + gap / 2, (i + 1) * kw - gap / 2
        poly = []
        for x in (x0, x1):
            A = np.array([x, yf])
            d = A - F
            tb = (yb - F[1]) / d[1]                      # ray focus->front hits back line
            B = F + tb * d
            poly.append((A, B))
        (A0, B0), (A1, B1) = poly
        zones.append(np.array([A0, A1, B1, B0], np.float32))
    return zones

class NoteEngine:
    def __init__(self, cfg):
        self.zones = build_zones(cfg)
        self.base  = cfg["notes"]["base_midi"]
        self.on_f  = cfg["detect"]["on_frames"]
        self.off_f = cfg["detect"]["off_frames"]
        self.vel   = cfg["midi"]["velocity"]
        self.ch    = cfg["midi"]["channel"]
        self.state = [{"on": 0, "off": 0, "playing": False}
                      for _ in self.zones]
        self.midi = None
        if MIDI_OK:
            self.midi = rtmidi.MidiOut()
            self.midi.open_virtual_port("PedalVision")

    def send(self, note, on):
        if self.midi:
            status = (0x90 if on else 0x80) | self.ch
            self.midi.send_message([status, note, self.vel if on else 0])
        print(("ON " if on else "OFF"), note)

    def update(self, contacts):
        hit = set()
        for p in contacts:
            for i, z in enumerate(self.zones):
                if cv2.pointPolygonTest(z, tuple(p), False) >= 0:
                    hit.add(i); break
        for i, st in enumerate(self.state):
            if i in hit:
                st["on"] += 1; st["off"] = 0
            else:
                st["off"] += 1; st["on"] = 0
            if not st["playing"] and st["on"] >= self.on_f:
                st["playing"] = True;  self.send(self.base + i, True)
            elif st["playing"] and st["off"] >= self.off_f:
                st["playing"] = False; self.send(self.base + i, False)

    def all_off(self):
        for i, st in enumerate(self.state):
            if st["playing"]:
                self.send(self.base + i, False)

# ---------------------------------------------------------------- calibration
def solve_extrinsics(geom, obs):
    """obs = [((u,v),(X,Y)), ...]  Solve yaw + pitch, auto-detect azimuth sign."""
    best = None
    for sign in (1, -1):
        yaws, pitches = [], []
        for (u, v), (X, Y) in obs:
            right, fwd, up = geom.raw_angles(u, v)
            az_raw = sign * math.atan2(right, fwd)
            el_raw = math.atan2(up, math.hypot(right, fwd))
            vec = np.array([X, Y]) - geom.C
            d = np.linalg.norm(vec)
            yaws.append(wrap_pi(math.atan2(vec[1], vec[0]) - az_raw))
            pitches.append(el_raw - math.atan2(-geom.h, d))
        mean_yaw = math.atan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws)))
        resid = float(np.std([wrap_pi(y - mean_yaw) for y in yaws]))
        if best is None or resid < best[0]:
            best = (resid, sign, mean_yaw, float(np.mean(pitches)))
    resid, sign, yaw, pitch = best
    print(f"  solved: yaw={math.degrees(yaw):.2f} pitch={math.degrees(pitch):.2f} "
          f"sign={sign} resid={math.degrees(resid):.2f} deg")
    return sign, yaw, pitch

# ---------------------------------------------------------------- UI
TRACKBARS = [("DiffThresh", "diff_thresh", 255), ("Blur k", "blur_ksize", 15),
             ("MinArea", "min_area", 2000), ("Morph k", "morph_k", 15),
             ("Zcontact mm", "z_contact_mm", 80), ("OnFrames", "on_frames", 8),
             ("OffFrames", "off_frames", 12)]

def build_trackbars(cfg):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    for name, key, mx in TRACKBARS:
        cv2.createTrackbar(name, "Controls", cfg["detect"][key], mx, lambda x: None)
 
def read_trackbars(cfg):
    for name, key, _ in TRACKBARS:
        cfg["detect"][key] = cv2.getTrackbarPos(name, "Controls")

def draw_topdown(cfg, zones, feet, engine, scale=4):
    W = int(cfg["board"]["width_cm"] * scale)
    H = int(cfg["board"]["depth_cm"] * scale)
    img = np.full((H, W, 3), 30, np.uint8)
    for i, z in enumerate(zones):
        pts = (z * scale).astype(np.int32)
        col = (0, 200, 0) if engine.state[i]["playing"] else (90, 90, 90)
        cv2.polylines(img, [pts], True, col, 2)
    for f in feet:
        x, y = (f["pos"] * scale).astype(int)
        contact = f["z_mm"] < cfg["detect"]["z_contact_mm"]
        cv2.circle(img, (x, y), 8, (0, 0, 255) if contact else (0, 255, 255), -1)
        cv2.putText(img, f'{f["z_mm"]:.0f}mm', (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    res = tuple(cfg["resolution"])
    cam0 = setup_camera(0, res, cfg["fps"])
    cam1 = setup_camera(1, res, cfg["fps"])
    lock_exposure([cam0, cam1])

    g0 = CamGeom(cfg["cameras"]["cam0"], res)
    g1 = CamGeom(cfg["cameras"]["cam1"], res)
    mask0 = make_trapezoid_mask(*res, cfg["trapezoid"]["cam0"])
    mask1 = make_trapezoid_mask(*res, cfg["trapezoid"]["cam1"])

    bgs = [None, None]
    for i in range(2):
        if os.path.exists(BG_PATH.format(i)):
            bgs[i] = np.load(BG_PATH.format(i))
            print(f"Loaded background {i}")

    engine = NoteEngine(cfg)
    zones = engine.zones
    build_trackbars(cfg)
    cv2.namedWindow("Dual Contour Stream", cv2.WINDOW_NORMAL)

    lm = cfg["landmarks_cm"]
    lm_mode, lm_idx = False, 0
    lm_obs = [[], []]

    prev_time, fps = time.time(), 0.0
    try:
        while True:
            read_trackbars(cfg)
            f0 = get_gray(cam0, res)
            f1 = get_gray(cam1, res)

            feet = []
            views = []
            if bgs[0] is not None and bgs[1] is not None:
                b0, th0 = detect_blobs(f0, bgs[0], mask0, cfg["detect"])
                b1, th1 = detect_blobs(f1, bgs[1], mask1, cfg["detect"])
                feet = pair_feet(b0, b1, g0, g1, cfg)
                contacts = [f["pos"] for f in feet
                            if f["z_mm"] < cfg["detect"]["z_contact_mm"]]
                engine.update(contacts)
                for frame, blobs, th in ((f0, b0, th0), (f1, b1, th1)):
                    v = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    v[th > 0] = (0, 80, 0)
                    for b in blobs:
                        cv2.drawContours(v, [b["contour"]], -1, (0, 255, 0), 2)
                        lp = tuple(map(int, b["low"]))
                        cv2.circle(v, lp, 5, (0, 0, 255), -1)
                    views.append(v)
            else:
                views = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in (f0, f1)]
                cv2.putText(views[0], "PRESS 'b' ON EMPTY SCENE", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            combined = np.hstack(views)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(combined, f"FPS {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if lm_mode:
                cv2.putText(combined,
                            f"LANDMARK {lm_idx+1}/{len(lm)} at {lm[lm_idx]} cm "
                            "- place marker, SPACE to grab", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            cv2.imshow("Dual Contour Stream", combined)
            cv2.imshow("Top-down", draw_topdown(cfg, zones, feet, engine))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):                       # snapshot empty scene
                print("Capturing background (keep area empty)...")
                stacks = [[], []]
                for _ in range(25):
                    stacks[0].append(get_gray(cam0, res).copy())
                    stacks[1].append(get_gray(cam1, res).copy())
                for i in range(2):
                    k = cfg["detect"]["blur_ksize"] | 1
                    bg = np.median(np.stack(stacks[i]), axis=0).astype(np.uint8)
                    bgs[i] = cv2.GaussianBlur(bg, (k, k), 0)
                    np.save(BG_PATH.format(i), bgs[i])
                print("Background stored.")
            elif key == ord('l'):                       # landmark calibration
                lm_mode, lm_idx = True, 0
                lm_obs = [[], []]
            elif key == ord(' ') and lm_mode and bgs[0] is not None:
                b0, _ = detect_blobs(f0, bgs[0], mask0, cfg["detect"])
                b1, _ = detect_blobs(f1, bgs[1], mask1, cfg["detect"])
                if b0 and b1:
                    lm_obs[0].append((b0[0]["low"], tuple(lm[lm_idx])))
                    lm_obs[1].append((b1[0]["low"], tuple(lm[lm_idx])))
                    print(f"Landmark {lm_idx+1} grabbed.")
                    lm_idx += 1
                    if lm_idx >= len(lm):
                        for gi, g, name in ((0, g0, "cam0"), (1, g1, "cam1")):
                            sign, yaw, pitch = solve_extrinsics(g, lm_obs[gi])
                            g.sign, g.yaw, g.pitch = sign, yaw, pitch
                            cfg["cameras"][name]["az_sign"]   = sign
                            cfg["cameras"][name]["yaw_deg"]   = math.degrees(yaw)
                            cfg["cameras"][name]["pitch_deg"] = math.degrees(pitch)
                        save_config(cfg)
                        lm_mode = False
                else:
                    print("Marker not seen by both cameras!")
            elif key == ord('s'):
                save_config(cfg)
    finally:
        engine.all_off()
        cam0.stop(); cam1.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
