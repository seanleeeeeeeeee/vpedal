"""
pedalvision -- camera-vision virtual organ pedalboard (Raspberry Pi 5, 2x IMX219-77).

  config.py    persistent, autosaving config
  boardmap.py  note names, key zones (mirroring), play-area limits
  geometry.py  camera model + triangulation
  camera.py    picamera2 YUV grabbers (+ fake camera)
  detect.py    YUV background subtraction, contact splitting, coverage
  engine.py    foot tracking, hit-test, note debouncing, MIDI
  calib.py     landmark pose solver
  ui.py        sliders, layout, cached overlays, top-down view
"""
import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import ui
from boardmap import load_zones, zone_signature
from calib import free_params, refine_pose
from camera import open_cameras
from config import BG_PATH, BG_PATH_OLD, CFG_PATH, ConfigStore
from detect import Coverage, Detector, mono_feet, stereo_feet
from engine import FootTracker, NoteEngine
from geometry import CamGeom, apply_geom


class App:
    def __init__(self, store, gui, fake=False):
        self.store = store
        self.cfg = cfg = store.cfg
        self.gui = gui
        cv2.setUseOptimized(True)
        try:
            cv2.setNumThreads(int(cfg["ui"].get("cv_threads", 2)))
        except Exception:
            pass

        self.ps = tuple(cfg["capture"]["process_size"])
        self.ds = tuple(cfg["capture"]["display_size"])
        self.scale = self.ds[0] / float(self.ps[0])

        self.g = [CamGeom(cfg["cameras"]["cam0"], self.ps),
                  CamGeom(cfg["cameras"]["cam1"], self.ps)]
        self.cams = open_cameras(cfg, fake)
        print("[ae] settling on EMPTY scene...")
        time.sleep(2.0)
        for i, c in enumerate(self.cams):
            e, gn = c.lock_ae(cfg["capture"]["flicker_hz"])
            print(f"[ae] cam{i} locked {e}us gain {gn:.2f}")

        self.det = [Detector(self.g[0]), Detector(self.g[1])]
        for i in range(2):
            for p in (BG_PATH.format(i), BG_PATH_OLD.format(i)):
                if os.path.exists(p) and self.det[i].load_background(p):
                    print(f"[bg] loaded {p}")
                    break
            self.det[i].rebuild_roi(cfg)

        self.zones = load_zones(cfg)
        self.zsig = zone_signature(cfg)
        self.engine = NoteEngine(cfg, self.zones)
        self.tracker = FootTracker()
        self.cov = Coverage(self.g[0], self.g[1], cfg)
        self.panel = ui.Panel(cfg)
        self.layout = ui.Layout(cfg, self.panel)
        self.keys = ui.KeyReader()
        self.ov = [ui.ZoneOverlay(), ui.ZoneOverlay()]
        self.td = ui.TopDown()
        self.pool = ThreadPoolExecutor(max_workers=2)

        self.win = "PedalVision"
        if gui:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(self.win, self.layout.on_mouse)

        self.view, self.overlay, self.showcov = 0, True, False
        self.lm_mode, self.lm_i, self.lm_obs = False, 0, [[], []]
        self.fps, self.t_prev = 0.0, time.time()
        self.prev_pos = []
        self.geom_dirty, self.t_dirty = True, 0.0
        self.points = [[], []]
        self.comps = [[], []]
        self.masks = [None, None]
        self.feet = []
        self.frames = [None, None]
        self.last_n = [-1, -1]
        self.skew_ms = 0.0
        self.frame_i = 0
        self.t_vis = 0.0

    # --------------------------------------------------------- housekeeping
    def sync_config(self):
        cfg = self.cfg
        sig = zone_signature(cfg)
        if sig != self.zsig:
            self.zsig = sig
            self.zones = load_zones(cfg)
            self.engine.set_zones(self.zones)
        if apply_geom(cfg, self.g):
            self.geom_dirty, self.t_dirty = True, time.time()
        if self.geom_dirty and time.time() - self.t_dirty > 0.15:
            for d in self.det:
                d.rebuild_roi(cfg)
            self.cov = Coverage(self.g[0], self.g[1], cfg, quiet=True)
            self.geom_dirty = False
        elif not self.geom_dirty:
            for d in self.det:
                d.rebuild_roi(cfg)          # no-op via signature cache

    # ---------------------------------------------------------------- vision
    def process(self, fr0, fr1, dt):
        cfg = self.cfg
        if self.det[0].bg_y is None or self.det[1].bg_y is None:
            self.points = [[], []]
            self.comps = [[], []]
            self.masks = [None, None]
            self.feet = []
            return
        t0 = time.perf_counter()
        a = self.pool.submit(self.det[0].detect, fr0, cfg)
        b = self.pool.submit(self.det[1].detect, fr1, cfg)
        p0, c0, m0 = a.result()
        p1, c1, m1 = b.result()
        self.points, self.comps, self.masks = [p0, p1], [c0, c1], [m0, m1]
        feet, u0, u1 = stereo_feet(p0, p1, self.g[0], self.g[1], cfg, self.prev_pos)
        feet += mono_feet(p0, u0, self.g[0], cfg, self.cov.is_mono)
        feet += mono_feet(p1, u1, self.g[1], cfg, self.cov.is_mono)
        self.feet = feet[:2]                                # 2-note polyphony
        self.prev_pos = self.tracker.update(self.feet, dt)
        if not self.lm_mode:
            self.engine.update(self.feet)
        self.t_vis = 0.9 * self.t_vis + 0.1 * (time.perf_counter() - t0) * 1000.0

    # ---------------------------------------------------------------- render
    def cam_pane(self, i):
        cfg = self.cfg
        det, geom = self.det[i], self.g[i]
        y = self.frames[i][0]
        if self.view == 1 and det.bg_y is not None and det.bg_y.shape == y.shape:
            src = cv2.subtract(y, det.bg_y)
        else:
            src = y
        small = cv2.resize(src, self.ds, interpolation=cv2.INTER_NEAREST)
        vis = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        if det.mask is not None:
            mm = cv2.resize(det.mask, self.ds, interpolation=cv2.INTER_NEAREST)
            vis[mm == 0] = vis[mm == 0] // 3
        if self.view == 3:                                   # chroma distance map
            cd = det.dbg_cd
            x0, y0, x1, y1 = det.bbox
            if cd is None or cd.shape != (y1 - y0, x1 - x0):
                vis = vis // 3
                ui.txt(vis, "NO CHROMA: " + det.chroma_state, (6, 40),
                       rel=0.034, col=(0, 120, 255))
                ui.txt(vis, f"cam{i} {ui.VIEW_NAMES[self.view]}", (6, 16), rel=0.030)
                return vis
            full = np.zeros((self.ps[1], self.ps[0]), np.uint8)
            full[y0:y1, x0:x1] = cd
            gain = float(cfg["ui"].get("chroma_gain", 6))
            amp = cv2.resize(cv2.convertScaleAbs(full, alpha=gain), self.ds,
                             interpolation=cv2.INTER_NEAREST)
            vis = cv2.applyColorMap(amp, cv2.COLORMAP_INFERNO)
            over = cv2.resize(cv2.threshold(full, float(cfg["detect"]["chroma_thresh"]),
                                            255, cv2.THRESH_BINARY)[1],
                              self.ds, interpolation=cv2.INTER_NEAREST)
            vis[over > 0] = (0, 255, 0)                      # what counts as "object"
            if det.mask is not None:
                mm = cv2.resize(det.mask, self.ds, interpolation=cv2.INTER_NEAREST)
                vis[mm == 0] = vis[mm == 0] // 3
            ui.txt(vis, f"cam{i} chroma  gain x{gain:.0f}  thr "
                        f"{cfg['detect']['chroma_thresh']}", (6, 16), rel=0.030)
            return vis
        if self.view == 2:                                   # binary foreground
            layer = self.masks[i]
            if layer is not None:
                full = np.zeros((self.ps[1], self.ps[0]), np.uint8)
                x0, y0, x1, y1 = det.bbox
                if layer.shape == (y1 - y0, x1 - x0):
                    full[y0:y1, x0:x1] = layer
                    lr = cv2.resize(full, self.ds, interpolation=cv2.INTER_NEAREST)
                    vis[lr > 0] = (0, 200, 0)
        if self.overlay:
            self.ov[i].update(cfg, self.zones, geom, self.scale)
            self.ov[i].draw(vis, self.engine.active)
        s = self.scale
        for c in self.comps[i]:
            ox, oy = c["org"]
            w, h = c["wh"]
            cv2.rectangle(vis, (int(ox * s), int(oy * s)),
                          (int((ox + w) * s), int((oy + h) * s)),
                          (0, 255, 0) if c["n"] < 2 else (0, 220, 255), 1)
            prof = c["prof"]
            cols = np.nonzero(prof >= 0)[0]
            if len(cols) > 3:
                step = max(1, len(cols) // 64)
                pts = np.stack([(ox + cols[::step]) * s,
                                (oy + prof[cols[::step]]) * s], axis=1).astype(np.int32)
                cv2.polylines(vis, [pts], False, (255, 160, 0), 1)
        for p in self.points[i]:
            cv2.circle(vis, (int(p["low"][0] * s), int(p["low"][1] * s)), 4, (0, 0, 255), -1)
        ui.txt(vis, f"cam{i} {ui.VIEW_NAMES[self.view]}", (6, 16), rel=0.030)
        return vis

    def render(self):
        cfg = self.cfg
        CW, CH = cfg["capture"]["canvas_size"]
        canvas = np.full((CH, CW, 3), 12, np.uint8)
        rects = self.layout.compute(CW, CH)
        ui.blit(canvas, rects["cam0"], self.cam_pane(0))
        ui.blit(canvas, rects["cam1"], self.cam_pane(1))
        ui.blit(canvas, rects["top"],
                self.td.render(cfg, self.zones, self.engine.active, self.feet,
                               self.cov, self.showcov, int(cfg["ui"]["topdown_px"])))
        pr = rects["panel"]
        ui.blit(canvas, pr, self.panel.draw(pr[2] - pr[0] - 4, pr[3] - pr[1] - 4), pad=2)
        L = self.layout.lines
        cv2.line(canvas, (0, L["h"]), (CW, L["h"]), (70, 70, 70), 2)
        cv2.line(canvas, (L["v"], 0), (L["v"], L["h"]), (70, 70, 70), 2)
        cv2.line(canvas, (L["b"], L["h"]), (L["b"], CH), (70, 70, 70), 2)
        ui.txt(canvas, f"FPS {self.fps:4.1f}  vis {self.t_vis:4.1f}ms  "
                       f"skew {self.skew_ms:4.1f}ms  "
                       f"pts {len(self.points[0])}/{len(self.points[1])}  "
                       f"chroma {'on' if cfg['detect']['use_chroma'] else 'off'}  "
                       f"{'MIDI' if self.engine.enabled else 'MUTE'}",
               (10, 12), rel=0.030)
        if self.lm_mode:
            lm = cfg["landmarks_m"]
            ui.txt(canvas, f"LANDMARK {self.lm_i + 1}/{len(lm)} at {lm[self.lm_i]} m -- "
                           "place marker, SPACE", (10, 24), rel=0.040, col=(0, 150, 255))
        if self.det[0].bg_y is None or self.det[1].bg_y is None:
            ui.txt(canvas, "PRESS 'b' WITH THE AREA EMPTY", (10, 60),
                   rel=0.045, col=(0, 0, 255))
        cv2.imshow(self.win, canvas)

    # ------------------------------------------------------------------ keys
    def grab_background(self):
        print("[bg] capturing, keep area empty...")
        st, last = [[], []], [-1, -1]
        t_end = time.time() + 2.0
        while time.time() < t_end and min(len(st[0]), len(st[1])) < 20:
            for i in range(2):
                fr, ts, n = self.cams[i].read()
                if fr is not None and n != last[i]:
                    last[i] = n
                    st[i].append(fr)
            time.sleep(0.004)
        ok = True
        for i in range(2):
            if self.det[i].set_background(st[i], self.cfg):
                self.det[i].save_background(BG_PATH.format(i))
            else:
                ok = False
        print("[bg] done" if ok else "[bg] FAILED -- cameras delivering frames?")

    def solve_landmarks(self):
        cfg = self.cfg
        lm = cfg["landmarks_m"]
        for i, nm in enumerate(("cam0", "cam1")):
            fr = free_params(len(lm), cfg["view"]["calib_solve_fov"])
            p, rms, per = refine_pose(self.g[i], self.lm_obs[i], fr)
            self.g[i].set_hfov(2 * math.atan((self.g[i].W / 2.0) / p[6]))
            self.g[i].set_pose(p[0], p[1], p[2], p[3:6])
            cfg["cameras"][nm].update({
                "yaw_deg": math.degrees(p[0]), "pitch_deg": math.degrees(p[1]),
                "pitch_trim_deg": 0.0, "roll_deg": math.degrees(p[2]),
                "pos_m": [float(p[3]), float(p[4]), float(p[5])],
                "hfov_deg": math.degrees(self.g[i].hfov)})
            print(f"[calib] {nm}: yaw {math.degrees(p[0]):.2f} "
                  f"pitch {math.degrees(p[1]):.2f} roll {math.degrees(p[2]):.2f} "
                  f"h {p[5] * 1000:.1f}mm fov {math.degrees(self.g[i].hfov):.2f} "
                  f"rms {rms:.2f}px  solved={fr}")
            print("[calib]  per-landmark px: " + " ".join(f"{e:.1f}" for e in per))
        if abs(cfg["cameras"]["cam0"]["hfov_deg"] - cfg["cameras"]["cam1"]["hfov_deg"]) > 0.5:
            cfg["view"]["fov_link"] = 0
            print("[calib] FOVs differ -> link off")
        self.geom_dirty, self.t_dirty = True, 0.0
self.store.save()
        self.lm_mode = False

    def handle_key(self, k):
        cfg = self.cfg
        lm = cfg["landmarks_m"]
        if k == 'q':
            return False
        if k == 'b':
            self.grab_background()
        elif k == 'l':
            self.lm_mode, self.lm_i, self.lm_obs = True, 0, [[], []]
            print(f"[calib] place marker at {lm[0]} m, press SPACE")
        elif k == ' ' and self.lm_mode:
            if len(self.points[0]) == 1 and len(self.points[1]) == 1:
                for i in range(2):
                    self.lm_obs[i].append((self.points[i][0]["low"], tuple(lm[self.lm_i])))
                self.lm_i += 1
                print(f"[calib] grabbed {self.lm_i}/{len(lm)}")
                if self.lm_i >= len(lm):
                    self.solve_landmarks()
                else:
                    print(f"[calib] next: {lm[self.lm_i]} m")
            else:
                print(f"[calib] need exactly 1 contact point per camera "
                      f"(saw {len(self.points[0])}/{len(self.points[1])})")
        elif k == 'v':
            self.view = (self.view + 1) % len(ui.VIEW_NAMES)
            print("[view]", ui.VIEW_NAMES[self.view])
        elif k == 'o':
            self.overlay = not self.overlay
        elif k == 'c':
            self.showcov = not self.showcov
        elif k == 't':
            cfg["view"]["topdown_mirror"] = 0 if cfg["view"].get("topdown_mirror", 1) else 1
            print("[view] topdown mirror =", cfg["view"]["topdown_mirror"])
        elif k == 'u':
            cfg["detect"]["use_chroma"] = 0 if cfg["detect"]["use_chroma"] else 1
            print("[detect] chroma =", cfg["detect"]["use_chroma"])
        elif k == 'x':
            cfg["limits"]["x_cutoff_cm"] = int(cfg["cameras"]["cam1"]["pos_m"][0] * 100)
            print("[cutoff]", cfg["limits"]["x_cutoff_cm"], "cm")
        elif k == 'k':
            cfg["limits"]["cutoff_low_side"] = 0 if cfg["limits"]["cutoff_low_side"] else 1
            self.engine.panic()
            print("[cutoff] dead side =",
                  "low x" if cfg["limits"]["cutoff_low_side"] else "high x")
        elif k == 'i':
            cfg["zones"]["mirror_x"] = 0 if cfg["zones"]["mirror_x"] else 1
            self.engine.panic()
            print("[zones] mirror_x =", cfg["zones"]["mirror_x"])
        elif k == 'm':
            self.engine.panic()
            self.engine.enabled = not self.engine.enabled
            print("[midi]", "on" if self.engine.enabled else "muted")
        elif k == 'r':
            self.zsig = None
        elif k == '[':
            self.panel.step_sel(-1)
        elif k == ']':
            self.panel.step_sel(1)
        elif k == '-':
            self.panel.nudge(-1)
        elif k == '=':
            self.panel.nudge(1)
        elif k == '_':
            self.panel.nudge(-10)
        elif k == '+':
            self.panel.nudge(10)
        elif k == 's':
            self.store.save()
        elif k == '?':
            print(ui.help_text())
        return True

    def read_key(self):
        """stdin first (works over SSH), then the GUI window."""
        k = self.keys.get()
        if self.gui:
            kk = cv2.waitKey(1) & 0xFF        # also pumps the window / mouse events
            if k is None and kk != 255:
                k = chr(kk)
        return k

    # -------------------------------------------------------------------- loop
    def run(self):
        print(ui.help_text())
        draw_every = 1
        while True:
            self.sync_config()
            fr0, t0, n0 = self.cams[0].read()
            fr1, t1, n1 = self.cams[1].read()

            fresh = (fr0 is not None and fr1 is not None
                     and (n0 != self.last_n[0] or n1 != self.last_n[1]))
            if not fresh:
                # nothing new from either sensor: don't burn a core re-detecting
                k = self.read_key()
                if k is not None and not self.handle_key(k):
                    break
                if not self.gui:
                    time.sleep(0.001)
                continue
            self.last_n = [n0, n1]
            self.frames = [fr0, fr1]
            self.skew_ms = abs(t0 - t1) / 1e6

            now = time.time()
            dt = now - self.t_prev
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-6)
            self.t_prev = now

            self.process(fr0, fr1, dt)

            self.frame_i += 1
            draw_every = max(1, int(self.cfg["ui"].get("draw_every", 1)))
            if self.gui and (self.frame_i % draw_every == 0):
                self.render()

            self.store.tick(now)
            k = self.read_key()
            if k is not None and not self.handle_key(k):
                break

    def close(self):
        try:
            self.pool.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.engine.panic()
        finally:
            self.keys.restore()
            for c in self.cams:
                c.stop()
            cv2.destroyAllWindows()
            self.store.save()


# --------------------------------------------------------------------- extras
def coverage_only(cfg):
    ps = cfg["capture"]["process_size"]
    g0 = CamGeom(cfg["cameras"]["cam0"], ps)
    g1 = CamGeom(cfg["cameras"]["cam1"], ps)
    cov = Coverage(g0, g1, cfg)
    zones = load_zones(cfg)
    img = ui.TopDown().render(cfg, zones, set(), [], cov, True,
                              int(cfg["ui"]["topdown_px"]))
    print(ui.help_text())
    if os.environ.get("DISPLAY"):
        cv2.imshow("coverage", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite("coverage.png", img)
    print("[coverage] wrote coverage.png  (green=stereo, teal=1 cam, red=dead)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CFG_PATH)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--fake", action="store_true",
                    help="synthetic YUV cameras (desktop development)")
    args = ap.parse_args()

    store = ConfigStore(args.config)
    if args.coverage:
        return coverage_only(store.cfg)

    gui = (not args.headless) and bool(os.environ.get("DISPLAY"))
    app = App(store, gui, fake=args.fake)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        app.close()


if __name__ == "__main__":
    main()