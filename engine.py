"""
engine.py -- foot tracking, zone hit-testing, note on/off debouncing, MIDI out.
"""
import cv2
import numpy as np

from boardmap import zone_enabled

try:
    import rtmidi
    HAVE_MIDI = True
except ImportError:
    HAVE_MIDI = False


class FootTracker:
    def __init__(self):
        self.tracks = []

    def update(self, feet, dt):
        for f in feet:
            best, bd = None, 0.10
            for t in self.tracks:
                dd = float(np.linalg.norm(f["pos"] - t["pos"]))
                if dd < bd:
                    best, bd = t, dd
            f["vz"] = ((f["z_mm"] - best["z_mm"]) / 1000.0 / max(dt, 1e-3)
                       if best else 0.0)
        self.tracks = [{"pos": f["pos"], "z_mm": f["z_mm"]} for f in feet]
        return [f["pos"] for f in feet]


class NoteEngine:
    def __init__(self, cfg, zones):
        self.cfg = cfg
        self.zones = []
        self.st = {}
        self.active = set()
        self.enabled = True
        self.out = None
        if HAVE_MIDI:
            try:
                self.out = rtmidi.MidiOut()
                self.out.open_virtual_port("PedalVision")
                print("[midi] virtual port 'PedalVision'")
            except Exception as e:
                self.out = None
                print(f"[midi] port failed ({e}) -> console only")
        else:
            print("[midi] python-rtmidi missing -> console only")
        self.set_zones(zones)

    def set_zones(self, zones):
        self.zones = zones
        old = self.st
        self.st = {z["midi"]: old.get(z["midi"], {"on": 0, "off": 0, "play": False})
                   for z in zones}
        for m, s in old.items():                    # kill notes whose zone vanished
            if s["play"] and m not in self.st:
                self.send(m, False)

    def send(self, midi, on, vel=100):
        if self.out and self.enabled:
            try:
                self.out.send_message([(0x90 if on else 0x80)
                                       | int(self.cfg["midi"]["channel"]),
                                       int(midi), int(vel) if on else 0])
            except Exception as e:
                print(f"[midi] send failed: {e}")
        print(("NOTE ON  " if on else "NOTE OFF ") + str(midi), flush=True)

    def hit(self, p):
        snap = self.cfg["detect"]["snap_mm"] / 1000.0
        near, nd = None, 1e9
        for z in self.zones:                        # blacks first
            if not zone_enabled(z, self.cfg):
                continue
            dist = cv2.pointPolygonTest(z["poly"], (float(p[0]), float(p[1])), True)
            if dist >= 0:
                return z
            if -dist < nd:
                near, nd = z, -dist
        return near if (near and nd <= snap) else None

    def update(self, feet):
        d = self.cfg["detect"]
        mi = self.cfg["midi"]
        hits = {}
        for f in feet:
            if not f["mono"] and f["z_mm"] > d["z_contact_mm"]:
                continue
            z = self.hit(f["pos"])
            if not z:
                continue
            need = d["mono_on_frames"] if f["mono"] else d["on_frames"]
            v = int(np.clip(mi["vel_min"] + mi["vel_gain"] * max(0.0, -f.get("vz", 0.0)),
                            mi["vel_min"], mi["vel_max"]))
            prev = hits.get(z["midi"])
            if prev is None or need < prev[0]:
                hits[z["midi"]] = (need, v)
        for m, s in self.st.items():
            if m in hits:
                s["on"] += 1
                s["off"] = 0
            else:
                s["off"] += 1
                s["on"] = 0
            if not s["play"] and m in hits and s["on"] >= hits[m][0]:
                s["play"] = True
                self.send(m, True, hits[m][1])
            elif s["play"] and s["off"] >= d["off_frames"]:
                s["play"] = False
                self.send(m, False)
        self.active = {m for m, s in self.st.items() if s["play"]}

    def panic(self):
        for m, s in self.st.items():
            if s["play"]:
                s["play"] = False
                self.send(m, False)
        self.active = set()
