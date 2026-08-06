# draggable.py
import numpy as np
import cv2

class DraggableTrapezoid:
    def __init__(self, width, height, norm_pts, hit_radius=10):
        self.w, self.h = width, height
        self.pts = [[int(x*width), int(y*height)] for x, y in norm_pts]
        self.hit_radius = hit_radius
        self.dragging_idx = None
        self.dirty = True
        self._update_mask()

    def _update_mask(self):
        poly = np.array(self.pts, dtype=np.int32)
        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.fillPoly(self.mask, [poly], 255)
        self.bbox = cv2.boundingRect(poly)
        self.dirty = False

    def get_mask_bbox(self):
        if self.dirty:
            self._update_mask()
        return self.mask, self.bbox

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (px, py) in enumerate(self.pts):
                if (px-x)**2 + (py-y)**2 < self.hit_radius**2:
                    self.dragging_idx = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_idx is not None:
            x = np.clip(x, 0, self.w-1)
            y = np.clip(y, 0, self.h-1)
            self.pts[self.dragging_idx] = [x, y]
            self.dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = None

    def draw_handles(self, img):
        for i, (px, py) in enumerate(self.pts):
            cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
        pts = np.array(self.pts, np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 255), 1)

    def to_normalized(self):
        return [[x/self.w, y/self.h] for x, y in self.pts]
