import os
import json
import time
import math
import platform
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pyautogui
import mediapipe as mp

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

Point = Tuple[float, float]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def l2(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ----------------------------
# $1 Recognizer (unistroke)
# ----------------------------
def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_length(pts: List[Point]) -> float:
    return sum(_dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def _centroid(pts: List[Point]) -> Point:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _bounding_box(pts: List[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _resample(pts: List[Point], n: int) -> List[Point]:
    if len(pts) < 2:
        return pts[:]
    I = _path_length(pts) / max(n - 1, 1)
    D = 0.0
    new_pts = [pts[0]]
    i = 1
    pts2 = pts[:]  # don't mutate original
    while i < len(pts2):
        d = _dist(pts2[i - 1], pts2[i])
        if (D + d) >= I:
            t = (I - D) / max(d, 1e-9)
            q = (pts2[i - 1][0] + t * (pts2[i][0] - pts2[i - 1][0]),
                 pts2[i - 1][1] + t * (pts2[i][1] - pts2[i - 1][1]))
            new_pts.append(q)
            pts2.insert(i, q)
            D = 0.0
        else:
            D += d
            i += 1
    if len(new_pts) < n:
        new_pts.append(pts2[-1])
    return new_pts[:n]


def _rotate_by(pts: List[Point], theta: float) -> List[Point]:
    c = _centroid(pts)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for x, y in pts:
        dx, dy = x - c[0], y - c[1]
        out.append((dx * cos_t - dy * sin_t + c[0],
                    dx * sin_t + dy * cos_t + c[1]))
    return out


def _rotate_to_zero(pts: List[Point]) -> List[Point]:
    c = _centroid(pts)
    theta = math.atan2(c[1] - pts[0][1], c[0] - pts[0][0])
    return _rotate_by(pts, -theta)


def _scale_to_square(pts: List[Point], size: float = 250.0) -> List[Point]:
    minx, miny, maxx, maxy = _bounding_box(pts)
    w = max(maxx - minx, 1e-9)
    h = max(maxy - miny, 1e-9)
    out = []
    for x, y in pts:
        out.append(((x - minx) * (size / w), (y - miny) * (size / h)))
    return out


def _translate_to_origin(pts: List[Point]) -> List[Point]:
    c = _centroid(pts)
    return [(x - c[0], y - c[1]) for x, y in pts]


def _path_distance(a: List[Point], b: List[Point]) -> float:
    return sum(_dist(a[i], b[i]) for i in range(len(a))) / len(a)


def normalize_gesture(pts: List[Point], n: int = 64) -> List[Point]:
    pts2 = _resample(pts, n)
    pts2 = _rotate_to_zero(pts2)
    pts2 = _scale_to_square(pts2, 250.0)
    pts2 = _translate_to_origin(pts2)
    return pts2


@dataclass
class Template:
    name: str
    points: List[Point]


class DollarOneRecognizer:
    def __init__(self):
        self.templates: List[Template] = []

    def add_template_raw(self, name: str, raw_points: List[Point]):
        self.templates.append(Template(name, normalize_gesture(raw_points)))

    def recognize(self, raw_points: List[Point]) -> Tuple[Optional[str], float]:
        if not self.templates or len(raw_points) < 10:
            return None, float("inf")
        pts = normalize_gesture(raw_points)
        best_name, best_dist = None, float("inf")
        for t in self.templates:
            d = _path_distance(pts, t.points)
            if d < best_dist:
                best_dist = d
                best_name = t.name
        return best_name, best_dist


TEMPLATE_FILE = "gesture_templates.json"


def load_templates(rec: DollarOneRecognizer):
    if not os.path.exists(TEMPLATE_FILE):
        return
    try:
        data = json.load(open(TEMPLATE_FILE, "r", encoding="utf-8"))
        for item in data.get("templates", []):
            name = item["name"]
            points = [(float(p[0]), float(p[1])) for p in item["points"]]
            # points stored normalized already
            rec.templates.append(Template(name, points))
    except Exception as e:
        print("[WARN] Template load failed:", e)


def save_templates(rec: DollarOneRecognizer):
    data = {"templates": [{"name": t.name, "points": [[p[0], p[1]] for p in t.points]} for t in rec.templates]}
    json.dump(data, open(TEMPLATE_FILE, "w", encoding="utf-8"), indent=2)


ACTIONS = ["OPEN_SPOTIFY", "SCREENSHOT", "PLAY_PAUSE", "VOLUME_UP", "VOLUME_DOWN"]


def run_action(action: str):
    sysname = platform.system().lower()
    try:
        if action == "OPEN_SPOTIFY":
            if "windows" in sysname:
                subprocess.Popen(["cmd", "/c", "start", "spotify:"], shell=False)
            elif "darwin" in sysname:
                subprocess.Popen(["open", "spotify:"])
            else:
                subprocess.Popen(["xdg-open", "spotify:"])
            return

        if action == "SCREENSHOT":
            if "windows" in sysname:
                pyautogui.hotkey("win", "shift", "s")
            elif "darwin" in sysname:
                pyautogui.hotkey("command", "shift", "4")
            else:
                pyautogui.press("printscreen")
            return

        if action == "PLAY_PAUSE":
            pyautogui.press("playpause")
            return
        if action == "VOLUME_UP":
            pyautogui.press("volumeup")
            return
        if action == "VOLUME_DOWN":
            pyautogui.press("volumedown")
            return

    except pyautogui.FailSafeException:
        raise
    except Exception as e:
        print(f"[WARN] Action failed {action}: {e}")


def hand_scale(lm) -> float:
    w = (lm[0].x, lm[0].y)
    m = (lm[9].x, lm[9].y)  # middle_mcp
    return max(l2(w, m), 1e-6)


def is_fist(lm, scale: float) -> bool:
    wrist = (lm[0].x, lm[0].y)
    tips = [4, 8, 12, 16, 20]
    close = 0
    for idx in tips:
        d = l2((lm[idx].x, lm[idx].y), wrist) / scale
        if d < 1.2:
            close += 1
    return close >= 4


def draw_ui(frame, lines, y0=24):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, text in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(frame, text, (14, y), font, 0.6, (240, 240, 240), 2, cv2.LINE_AA)


def find_working_camera_index(max_index=5) -> int:
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                return i
    return 0


def main():
    print("[INFO] Starting Air Mouse... (press Q or ESC to quit)")

    screen_w, screen_h = pyautogui.size()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cam_index = find_working_camera_index()
    print(f"[INFO] Using camera index: {cam_index}")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Close Teams/Zoom/OBS/Camera app and retry.")

    smooth = 0.25
    cx, cy = screen_w / 2, screen_h / 2

    active_min_x, active_max_x = 0.15, 0.85
    active_min_y, active_max_y = 0.15, 0.85

    # pinch click/drag
    pinch_on = 0.35
    pinch_off = 0.45
    pinch_down = False
    pinch_start_t = 0.0
    dragging = False
    click_max_hold = 0.20

    # scroll
    scrolling = False
    last_scroll_y = None

    # draw gestures mode: thumb+pinky pinch
    drawing = False
    drawn_points: List[Point] = []
    draw_pinch_on = 0.45
    draw_pinch_off = 0.60

    recognizer = DollarOneRecognizer()
    load_templates(recognizer)
    train_next = False
    selected_action = ACTIONS[0]

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Camera frame read failed.")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            ui_lines = [
                "Air Mouse | Q/ESC quit | 1-5 choose action | T train | C clear templates",
                "Index tip moves cursor | Thumb+Index pinch: click/drag | Fist: scroll",
                "Thumb+Pinky pinch: DRAW gesture shortcut",
                f"Selected: {selected_action} | Templates: {len(recognizer.templates)} | Train armed: {train_next}"
            ]

            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                lm = hand.landmark
                scale = hand_scale(lm)

                idx_tip = (lm[8].x, lm[8].y)
                thumb_tip = (lm[4].x, lm[4].y)
                pinky_tip = (lm[20].x, lm[20].y)

                # Drawing mode toggle: thumb+pinky pinch
                draw_ratio = l2(thumb_tip, pinky_tip) / scale
                if (not drawing) and draw_ratio < draw_pinch_on:
                    drawing = True
                    drawn_points = []
                elif drawing and draw_ratio > draw_pinch_off:
                    drawing = False
                    if len(drawn_points) >= 12:
                        if train_next:
                            recognizer.add_template_raw(selected_action, drawn_points)
                            save_templates(recognizer)
                            train_next = False
                            print(f"[TRAINED] {selected_action} (templates={len(recognizer.templates)})")
                        else:
                            name, d = recognizer.recognize(drawn_points)
                            if name and d < 35.0:
                                print(f"[GESTURE] {name} (dist={d:.2f})")
                                run_action(name)
                            else:
                                print(f"[GESTURE] no match (best={name}, dist={d:.2f})")

                if drawing:
                    drawn_points.append(idx_tip)
                    for i in range(1, len(drawn_points)):
                        x1, y1 = int(drawn_points[i - 1][0] * w), int(drawn_points[i - 1][1] * h)
                        x2, y2 = int(drawn_points[i][0] * w), int(drawn_points[i][1] * h)
                        cv2.line(frame, (x1, y1), (x2, y2), (80, 255, 80), 3)

                # Scroll mode: fist
                fist = is_fist(lm, scale)
                if fist and not scrolling:
                    scrolling = True
                    last_scroll_y = idx_tip[1]
                elif (not fist) and scrolling:
                    scrolling = False
                    last_scroll_y = None

                if scrolling and last_scroll_y is not None:
                    dy = (idx_tip[1] - last_scroll_y)
                    scroll_amount = int(clamp(-dy * 900, -80, 80))
                    if abs(scroll_amount) > 2:
                        pyautogui.scroll(scroll_amount)
                    last_scroll_y = idx_tip[1]

                # Cursor move (avoid during drawing/scrolling)
                if (not drawing) and (not scrolling):
                    nx = (idx_tip[0] - active_min_x) / (active_max_x - active_min_x)
                    ny = (idx_tip[1] - active_min_y) / (active_max_y - active_min_y)
                    nx = clamp(nx, 0.0, 1.0)
                    ny = clamp(ny, 0.0, 1.0)

                    tx = nx * screen_w
                    ty = ny * screen_h
                    cx = (1 - smooth) * cx + smooth * tx
                    cy = (1 - smooth) * cy + smooth * ty
                    pyautogui.moveTo(cx, cy)

                # Click/drag: thumb+index pinch
                pinch = l2(thumb_tip, idx_tip) / scale

                if (not pinch_down) and pinch < pinch_on and (not drawing) and (not scrolling):
                    pinch_down = True
                    pinch_start_t = time.time()

                if pinch_down and pinch > pinch_off:
                    hold = time.time() - pinch_start_t
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    else:
                        if hold <= click_max_hold:
                            pyautogui.click()
                    pinch_down = False

                if pinch_down and (not dragging) and (time.time() - pinch_start_t > click_max_hold):
                    dragging = True
                    pyautogui.mouseDown()

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            draw_ui(frame, ui_lines)
            cv2.imshow("Air Mouse (MediaPipe + PyAutoGUI)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('t'):
                train_next = True
                print("[INFO] Training armed: thumb+pinky pinch -> draw -> release.")
            if key == ord('c'):
                recognizer.templates = []
                save_templates(recognizer)
                print("[INFO] Cleared all templates.")
            if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                selected_action = ACTIONS[int(chr(key)) - 1]
                print(f"[INFO] Selected action: {selected_action}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except pyautogui.FailSafeException:
        print("\n[ABORTED] PyAutoGUI failsafe triggered (moved cursor to top-left).")
