"""
camera.py -- threaded picamera2 grabbers (always the newest Y plane) + a fake
camera so the rest of the pipeline can be developed on a desktop.
"""
import math
import threading
import time

import cv2
import numpy as np


class Cam:
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
                self.rot_cpu = bool(cfg["capture"]["rot180"][slot])
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
            y = np.ascontiguousarray(y, dtype=np.uint8)
            with self.lock:
                self.frame, self.ts, self.n = y, ts, self.n + 1

    def read(self):
        with self.lock:
            return (None, 0) if self.frame is None else (self.frame, self.ts)

    def lock_ae(self, flicker_hz):
        md = self.picam.capture_metadata()
        exp = md.get("ExposureTime", 10000)
        gain = md.get("AnalogueGain", 1.0)
        if flicker_hz:
            q = 1e6 / (2.0 * flicker_hz)                  # 10000 us @ 50 Hz
            new = max(q, round(exp / q) * q)
            gain = float(np.clip(gain * exp / new, 1.0, 8.0))
            exp = int(new)
        ctrl = {"AeEnable": False, "AwbEnable": False,
                "ExposureTime": int(exp), "AnalogueGain": float(gain)}
        if "ColourGains" in md:
            ctrl["ColourGains"] = md["ColourGains"]
        self.picam.set_controls(ctrl)
        return int(exp), float(gain)

    def stop(self):
        self.run = False
        time.sleep(0.05)
        try:
            self.picam.stop()
        except Exception:
            pass


class FakeCam:
    """Synthetic scene: static gradient + one sweeping bright blob."""

    def __init__(self, slot, cfg):
        self.pw, self.ph = cfg["capture"]["process_size"]
        rng = np.random.default_rng(slot)
        base = (np.linspace(60, 140, self.pw, dtype=np.float32)[None, :]
                * np.ones((self.ph, 1), np.float32))
        self.base = np.clip(base + rng.normal(0, 2.0, (self.ph, self.pw)),
                            0, 255).astype(np.uint8)
        self.slot, self.t0 = slot, time.time()

    def read(self):
        f = self.base.copy()
        t = time.time() - self.t0
        x = int(self.pw * (0.5 + 0.28 * math.sin(t * 0.6 + self.slot)))
        y = int(self.ph * 0.78)
        cv2.circle(f, (x, y), 26, 235, -1)
        return f, time.monotonic_ns()

    def lock_ae(self, flicker_hz):
        return 10000, 1.0

    def stop(self):
        pass


def open_cameras(cfg, fake=False):
    klass = FakeCam if fake else Cam
    return [klass(0, cfg), klass(1, cfg)]
