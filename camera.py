"""
camera.py -- threaded picamera2 grabbers delivering (Y, U, V) planes.

YUV420 from the ISP is free chroma: Y full res, U/V at half res in both axes.
read() -> ((y, u, v), sensor_timestamp_ns, frame_number)
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
        self.slot = slot
        self.pw, self.ph = cfg["capture"]["process_size"]
        self.pw -= self.pw % 2
        self.ph -= self.ph % 2
        self.chroma = bool(cfg["capture"].get("chroma", 1))
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
        self._warned = False
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- YUV420 (h*3/2, stride) -> Y, U, V ----
    def _split(self, a):
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        h3, wst = a.shape
        ah = (h3 * 2) // 3
        y = a[:self.ph, :self.pw].copy()
        if not self.chroma or ah % 2 or wst % 2 or ah < self.ph:
            return y, None, None
        try:
            cw, chh = wst // 2, ah // 2
            u = a[ah:ah + ah // 4].reshape(chh, cw)[:self.ph // 2, :self.pw // 2].copy()
            v = a[ah + ah // 4:ah + ah // 2].reshape(chh, cw)[:self.ph // 2, :self.pw // 2].copy()
            return y, u, v
        except Exception as e:
            if not self._warned:
                print(f"[cam{self.slot}] chroma split failed ({e}); luma only")
                self._warned = True
            return y, None, None

    def _loop(self):
        while self.run:
            try:
                req = self.picam.capture_request()
            except Exception:
                break
            try:
                arr = req.make_array("main")
                ts = req.get_metadata().get("SensorTimestamp", time.monotonic_ns())
                y, u, v = self._split(arr)
            finally:
                req.release()
            if self.rot_cpu:
                y = cv2.rotate(y, cv2.ROTATE_180)
                if u is not None:
                    u = cv2.rotate(u, cv2.ROTATE_180)
                    v = cv2.rotate(v, cv2.ROTATE_180)
            with self.lock:
                self.frame, self.ts, self.n = (y, u, v), ts, self.n + 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0, -1
            return self.frame, self.ts, self.n

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
    """Warm 'wood' background + a neutral 'sock' pair, for desktop development."""

    def __init__(self, slot, cfg):
        self.pw, self.ph = cfg["capture"]["process_size"]
        rng = np.random.default_rng(slot)
        base = (np.linspace(90, 165, self.pw, dtype=np.float32)[None, :]
                * np.ones((self.ph, 1), np.float32))
        self.y0 = np.clip(base + rng.normal(0, 2.0, (self.ph, self.pw)), 0, 255).astype(np.uint8)
        self.u0 = np.full((self.ph // 2, self.pw // 2), 108, np.uint8)   # warm wood
        self.v0 = np.full((self.ph // 2, self.pw // 2), 146, np.uint8)
        self.slot, self.t0, self.n = slot, time.time(), 0

    def read(self):
        t = time.time() - self.t0
        y = self.y0.copy()
        u = self.u0.copy()
        v = self.v0.copy()
        cx = int(self.pw * (0.5 + 0.22 * math.sin(t * 0.6 + self.slot)))
        cy = int(self.ph * 0.78)
        gap = int(18 + 16 * (1 + math.sin(t * 0.35)))
        for s in (-1, 1):
            x = cx + s * gap
            cv2.ellipse(y, (x, cy), (22, 34), 0, 0, 360, 205, -1)     # bright sock
            cv2.ellipse(y, (x, cy + 30), (22, 8), 0, 0, 360, 70, -1)  # dark underside
            cv2.ellipse(u, (x // 2, cy // 2), (11, 19), 0, 0, 360, 128, -1)
            cv2.ellipse(v, (x // 2, cy // 2), (11, 19), 0, 0, 360, 128, -1)
        cv2.ellipse(y, (cx + 70, cy + 20), (60, 26), 0, 0, 360, 70, -1)   # shadow
        self.n += 1
        return (y, u, v), time.monotonic_ns(), self.n

    def lock_ae(self, flicker_hz):
        return 10000, 1.0

    def stop(self):
        pass


def open_cameras(cfg, fake=False):
    klass = FakeCam if fake else Cam
    return [klass(0, cfg), klass(1, cfg)]