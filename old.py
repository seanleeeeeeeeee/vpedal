import cv2
import numpy as np
import json
import time
from picamera2 import Picamera2
from libcamera import controls
from trapezoid import make_trapezoid_mask

CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def setup_camera(cam_num, resolution):
    picam = Picamera2(cam_num)
    # Request Y-only / grayscale via YUV420, take luma plane -> avoids cv2.cvtColor
    config = picam.create_video_configuration(
        main={"size": resolution, "format": "YUV420"},
        controls={"FrameDurationLimits": (8333, 33333)}  # ~30-120fps range hint
    )
    picam.configure(config)
    picam.start()
    return picam

def get_gray_frame(picam, resolution):
    yuv = picam.capture_array("main")
    w, h = resolution
    # Y plane is the first w*h bytes of YUV420
    y_plane = yuv[:h, :w]
    return y_plane

def process_frame(gray, mask, bbox, canny_lo, canny_hi, blur_k, min_area):
    x, y, w, h = bbox
    roi = gray[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]

    if blur_k > 1:
        roi = cv2.GaussianBlur(roi, (blur_k | 1, blur_k | 1), 0)

    masked = cv2.bitwise_and(roi, roi, mask=roi_mask)
    edges = cv2.Canny(masked, canny_lo, canny_hi)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # only for display draw
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c_shifted = c + [x, y]
        cv2.drawContours(out, [c_shifted], -1, (0, 255, 0), 1)
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 0, 0), 1)
    return out

def nothing(x): pass

def build_trackbars(cfg):
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Canny Lo", "Controls", cfg["canny_lo"], 255, nothing)
    cv2.createTrackbar("Canny Hi", "Controls", cfg["canny_hi"], 255, nothing)
    cv2.createTrackbar("Blur k",   "Controls", cfg["blur_ksize"], 21, nothing)
    cv2.createTrackbar("MinArea",  "Controls", cfg["min_contour_area"], 5000, nothing)

def main():
    cfg = load_config()
    res = tuple(cfg["resolution"])

    cam0 = setup_camera(0, res)
    cam1 = setup_camera(1, res)

    mask0, bbox0 = make_trapezoid_mask(res[0], res[1], cfg["trapezoid"]["cam0"])
    mask1, bbox1 = make_trapezoid_mask(res[0], res[1], cfg["trapezoid"]["cam1"])

    build_trackbars(cfg)

    cv2.namedWindow("Dual Contour Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dual Contour Stream", 1280, 480)  # NOT fullscreen property
    cv2.moveWindow("Dual Contour Stream", 0, 0)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 200)
    cv2.moveWindow("Controls", 1290, 0)  # place beside, not behind
    prev_time = time.time()
    fps = 0

    try:
        while True:
            canny_lo = cv2.getTrackbarPos("Canny Lo", "Controls")
            canny_hi = cv2.getTrackbarPos("Canny Hi", "Controls")
            blur_k   = cv2.getTrackbarPos("Blur k", "Controls")
            min_area = cv2.getTrackbarPos("MinArea", "Controls")

            g0 = get_gray_frame(cam0, res)
            g1 = get_gray_frame(cam1, res)

            out0 = process_frame(g0, mask0, bbox0, canny_lo, canny_hi, blur_k, min_area)
            out1 = process_frame(g1, mask1, bbox1, canny_lo, canny_hi, blur_k, min_area)

            combined = np.hstack((out0, out1))

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time))
            prev_time = now
            cv2.putText(combined, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Dual Contour Stream", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cfg["canny_lo"] = canny_lo
                cfg["canny_hi"] = canny_hi
                cfg["blur_ksize"] = blur_k
                cfg["min_contour_area"] = min_area
                with open(CONFIG_PATH, "w") as f:
                    json.dump(cfg, f, indent=2)
                print("Saved config.")

    finally:
        cam0.stop()
        cam1.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
