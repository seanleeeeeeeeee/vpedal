"""
pedalvision -- camera-vision virtual organ pedalboard (Raspberry Pi 5, 2x IMX219-77).

Entry point / main loop.  Modules:
  config.py    defaults + persistent, autosaving config
  boardmap.py  note names, key zones (mirroring), play-area limits
  geometry.py  camera model + triangulation
  camera.py    picamera2 grabbers (and a fake camera for desktop dev)
  detect.py    background subtraction, stereo/mono feet, coverage map
  engine.py    foot tracking, hit-test, note debouncing, MIDI
  calib.py     landmark pose solver
  ui.py        sliders, layout, top-down view, key reader

Keys work over SSH (raw stdin) *and* in the GUI window -- see ui.help_text().
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

import ui
from boardmap import cutoff_m, load_zones, zone_enabled, zone_signature
from calib import free_params, refine_pose
from camera import open_cameras
from config import BG_PATH, CFG_PATH, ConfigStore
from detect import Coverage, Detector, mono_feet, stereo_feet
from engine import FootTracker, NoteEngine
from geometry import CamGeom, apply_geom

VIEW_NAMES = ("gray", "diff", "mask")


class App:
    def __init__(self, store, gui, fake=False):
        self.store = store
        self.cfg = cfg = store.cfg
        self.gui = gui
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
            p = BG_PATH.format(i)
            if os.path.exists(p) and self.det[i].load_background(p):
                print(f"[bg] loaded cam{i}")
            self.det[i].rebuild_roi(cfg)

        self.zones = load_zones(cfg)
        self.zsig = zone_signature(cfg)
        self.engine = NoteEngine(cfg, self.zones)
        self.tracker = FootTracker()
        self.cov = Coverage(self.g[0], self.g[1], cfg)
        self.panel = ui.Panel(cfg)
        self.layout = ui.Layout(cfg, self.panel)
        self.keys = ui.KeyReader()

        self.win = "PedalVision"
        if gui:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(self.win, self.layout.on_mouse)

        self.view, self.overlay, self.showcov = 0, True, False
        self.lm_mode, self.lm_i, self.lm_obs = False, 0, [[], []]
        self.fps, self.t_prev = 0.0, time.time()
        self.prev_pos = []
        self.geom_dirty, self.t_dirty = True, 0.0
        self.blobs = [[], []]
        self.ths = [None, None]
        self.feet = []
        self.skew_ms = 0.0

    # ------------------------------------------------------------- housekeeping
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
                d.rebuild_roi(cfg)          # cheap no-op via signature cache

    # ------------------------------------------------------------------ vision
    def process(self, f0, f1, dt):
        cfg = self.cfg
        self.blobs = [[], []]
        self.ths = [None, None]
        self.feet = []
        if self.det[0].bg is None or self.det[1].bg is None:
            return
        self.blobs[0], self.ths[0] = self.det[0].detect(f0, cfg)
        self.blobs[1], self.ths[1] = self.det[1].detect(f1, cfg)
        feet, u0, u1 = stereo_feet(self.blobs[0], self.blobs[1],
                                   self.g[0], self.g[1], cfg, self.prev_pos)
        feet += mono_feet(self.blobs[0], u0, self.g[0], cfg, self.cov.is_mono)
        feet += mono_feet(self.blobs[1], u1, self.g[1], cfg, self.cov.is_mono)
        self.feet = feet[:2]                                # 2-note polyphony
        self.prev_pos = self.tracker.update(self.feet, dt)
        if not self.lm_mode:
            self.engine.update(self.feet)

    # ------------------------------------------------------------------ render
    def cam_pane(self, i, frame):
        cfg = self.cfg
        det, geom = self.det[i], self.g[i]
        small = cv2.resize(frame, self.ds, interpolation=cv2.INTER_AREA)
        if self.view == 1 and det.bg is not None and det.bg.shape == frame.shape:
            small = cv2.resize(cv2.subtract(frame, det.bg), self.ds)
        vis = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        if det.mask is not None:
            mm = cv2.resize(det.mask, self.ds, interpolation=cv2.INTER_NEAREST)
            vis[mm == 0] = vis[mm == 0] // 3
        if self.ths[i] is not None and self.view == 2:
            full = np.zeros((self.ps[1], self.ps[0]), np.uint8)
            x0, y0, x1, y1 = det.bbox
            full[y0:y1, x0:x1] = self.ths[i]
            vis[cv2.resize(full, self.ds, interpolation=cv2.INTER_NEAREST) > 0] = (0, 90, 0)
        if self.overlay:
            for z in self.zones:
                if not zone_enabled(z, cfg):
                    continue
                pp = [geom.project((float(p[0]), float(p[1]), 0.0))
                      for p in z["poly"].reshape(-1, 2)]
                if any(p is None for p in pp):
                    continue
                pts = (np.array(pp) * self.scale).astype(np.int32)
                c = (0, 200, 0) if z["midi"] in self.engine.active else \
                    ((90, 90, 160) if z["black"] else (70, 70, 70))
                cv2.polylines(vis, [pts], True, c, 1)
        for b in self.blobs[i]:
            cv2.drawContours(vis, [(b["c"] * self.scale).astype(np.int32)], -1,
                             (0, 255, 0), 1,
                             offset=(int(b["off"][0] * self.scale),
                                     int(b["off"][1] * self.scale)))
            cv2.circle(vis, (int(b["low"][0] * self.scale),
                             int(b["low"][1] * self.scale)), 4, (0, 0, 255), -1)
        ui.txt(vis, f"cam{i} {VIEW_NAMES[self.view]}", (6, 16), rel=0.030)
        return vis

    def render(self, f0, f1):
        cfg = self.cfg
        CW, CH = cfg["capture"]["canvas_size"]
        canvas = np.full((CH, CW, 3), 12, np.uint8)
        rects = self.layout.compute(CW, CH)
        ui.blit(canvas, rects["cam0"], self.cam_pane(0, f0))
        ui.blit(canvas, rects["cam1"], self.cam_pane(1, f1))
        ui.blit(canvas, rects["top"],
                ui.draw_topdown(cfg, self.zones, self.engine.active,
                                self.feet, self.cov, self.showcov))
        pr = rects["panel"]
        ui.blit(canvas, pr, self.panel.draw(pr[2] - pr[0] - 4, pr[3] - pr[1] - 4), pad=2)
        L = self.layout.lines
        cv2.line(canvas, (0, L["h"]), (CW, L["h"]), (70, 70, 70), 2)
        cv2.line(canvas, (L["v"], 0), (L["v"], L["h"]), (70, 70, 70), 2)
        cv2.line(canvas, (L["b"], L["h"]), (L["b"], CH), (70, 70, 70), 2)
        ui.txt(canvas, f"FPS {self.fps:4.1f}  skew {self.skew_ms:4.1f}ms  "
                       f"blobs {len(self.blobs[0])}/{len(self.blobs[1])}  "
                       f"cut {cfg['limits']['x_cutoff_cm']}cm"
                       f"{'(low)' if cfg['limits']['cutoff_low_side'] else '(high)'}  "
                       f"mirX {int(bool(cfg['zones']['mirror_x']))}  "
                       f"{'MIDI' if self.engine.enabled else 'MUTE'}",
               (10, CH - 12), rel=0.030)
        if self.lm_mode:
            lm = cfg["landmarks_m"]
            ui.txt(canvas, f"LANDMARK {self.lm_i + 1}/{len(lm)} at {lm[self.lm_i]} m -- "
                           "place marker, SPACE", (10, 24), rel=0.040, col=(0, 150, 255))
        if self.det[0].bg is None or self.det[1].bg is None:
            ui.txt(canvas, "PRESS 'b' WITH THE AREA EMPTY", (10, 60),
                   rel=0.045, col=(0, 0, 255))
        cv2.imshow(self.win, canvas)

    # -------------------------------------------------------------------- keys
    def grab_background(self):
        cfg = self.cfg
        print("[bg] capturing, keep area empty...")
        st = [[], []]
        for _ in range(25):
            for i in range(2):
                fr, _ = self.cams[i].read()
                if fr is not None:
                    st[i].append(fr.copy())
            time.sleep(1.0 / max(1, cfg["capture"]["fps"]))
        for i in range(2):
            bg = self.det[i].set_background(st[i], cfg["detect"]["blur_k"])
            if bg is not None:
                np.save(BG_PATH.format(i), bg)
        print("[bg] done" if all(d.bg is not None for d in self.det)
              else "[bg] FAILED -- cameras delivering frames?")

    def solve_landmarks(self):
        cfg = self.cfg
        lm = cfg["landmarks_m"]
        for i, nm in enumerate(("cam0", "cam1")):
            fr = free_params(len(lm), cfg["view"]["calib_solve_fov"])
            p, rms, per = refine_pose(self.g[i], self.lm_obs[i], fr)
            self.g[i].set_hfov(2 * math.atan((self.g[i].W / 2.0) / p[6]))
            self.g[i].set_pose(p[0], p[1], p[2], p[3:6])
            cfg["cameras"][nm].update({
                "yaw_deg": math.degrees(p[0]),
                "pitch_deg": math.degrees(p[1]),
                "pitch_trim_deg": 0.0,                 # folded into the base
                "roll_deg": math.degrees(p[2]),
                "pos_m": [float(p[3]), float(p[4]), float(p[5])],
                "hfov_deg": math.degrees(self.g[i].hfov)})
            print(f"[calib] {nm}: yaw {math.degrees(p[0]):.2f} "
                  f"pitch {math.degrees(p[1]):.2f} roll {math.degrees(p[2]):.2f} "
                  f"h {p[5] * 1000:.1f}mm fov {math.degrees(self.g[i].hfov):.2f} "
                  f"rms {rms:.2f}px  solved={fr}")
            print("[calib]  per-landmark px: " + " ".join(f"{e:.1f}" for e in per))
        if abs(cfg["cameras"]["cam0"]["hfov_deg"]
               - cfg["cameras"]["cam1"]["hfov_deg"]) > 0.5:
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
            if len(self.blobs[0]) == 1 and len(self.blobs[1]) == 1:
                for i in range(2):
                    self.lm_obs[i].append((self.blobs[i][0]["low"], tuple(lm[self.lm_i])))
                self.lm_i += 1
                print(f"[calib] grabbed {self.lm_i}/{len(lm)}")
                if self.lm_i >= len(lm):
                    self.solve_landmarks()
                else:
                    print(f"[calib] next: {lm[self.lm_i]} m")
            else:
                print(f"[calib] need exactly 1 blob per camera "
                      f"(saw {len(self.blobs[0])}/{len(self.blobs[1])})")
        elif k == 'v':
            self.view = (self.view + 1) % 3
        elif k == 'o':
            self.overlay = not self.overlay
        elif k == 'c':
            self.showcov = not self.showcov
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
        k = self.keys.get()
        if self.gui:
            kk = cv2.waitKey(1) & 0xFF
            if k is None and kk != 255:
                k = chr(kk)
        return k

    # -------------------------------------------------------------------- loop
    def run(self):
        print(ui.help_text())
        while True:
            self.sync_config()
            f0, t0 = self.cams[0].read()
            f1, t1 = self.cams[1].read()
            if f0 is None or f1 is None:
                time.sleep(0.005)
                if self.gui:
                    cv2.waitKey(1)
                continue
            self.skew_ms = abs(t0 - t1) / 1e6

            now = time.time()
            dt = now - self.t_prev
            self.fps = 0.9 * self.fps + 0.1 / max(dt, 1e-6)
            self.t_prev = now

            self.process(f0, f1, dt)
            if self.gui:
                self.render(f0, f1)

            self.store.tick(now)
            k = self.read_key()
            if k is not None and not self.handle_key(k):
                break

    def close(self):
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
    img = ui.draw_topdown(cfg, zones, set(), [], cov, True)
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
                    help="synthetic cameras (desktop development)")
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
