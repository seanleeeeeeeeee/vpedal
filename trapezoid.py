import numpy as np
import cv2

def make_trapezoid_mask(width, height, pts):
    """
    pts: list of 4 (x,y) points defining trapezoid, normalized 0-1 
    e.g. [(0.1,1.0),(0.4,0.3),(0.6,0.3),(0.9,1.0)]
    """
    poly = np.array([[int(x*width), int(y*height)] for x,y in pts], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    x, y, w, h = cv2.boundingRect(poly)
    return mask, (x, y, w, h)
