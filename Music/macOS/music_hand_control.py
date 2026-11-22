import math
import os
import subprocess
import time

import cv2
import mediapipe as mp

try:
    from AppKit import NSEvent  # NSEvent lives in AppKit, not Quartz
    from Quartz import CGEventPost, kCGHIDEventTap
    _HAS_QUARTZ = True
except ImportError:
    _HAS_QUARTZ = False

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Pairs of landmark indices to connect (thumb, index, middle, ring, pinky, palm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge
]

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0
MIDDLE_MCP = 9

VOLUME_MIN = 0.0
VOLUME_MAX = 100.0

# Tighter contact thresholds (relative to wrist-to-middle-MCP scale) to resist rotation overlap
CONTACT_RATIO_THRESHOLD = 0.22
THUMB_FINGER_RATIO_THRESHOLD = 0.21

# Require several consecutive matching frames before accepting a contact/gesture
CONTACT_CONFIRM_FRAMES = 3

# Adaptive travel learning & reduced sensitivity defaults
INITIAL_MAX_TRAVEL_PX = 160.0   # Starting learned max vertical pinch travel distance
FULL_RANGE_VOLUME_SPAN = 45.0   # Lowered span for less sensitivity
ACCEL_EXPONENT = 1.40           # > 1.0 makes fast moves jump/drop more than slow moves
VELOCITY_GAIN = 0.015           # Toned down speed boost
MOVE_DEADZONE_PX = 3.5          # Ignores small movements and rotation jitter

# Gesture debounce (seconds) so holding fingers together doesn't repeat rapidly
GESTURE_COOLDOWN_SEC = 1.0

# macOS NX_KEYTYPE constants (from hidsystem/ev_keymap.h)
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18
NX_SUBTYPE_AUX_CONTROL = 8

latest_result = None  # updated by the callback, read by the main loop


def on_result(result, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


def dist_px(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# --- macOS System Volume Helpers ---
def get_macos_volume(default: float = 50.0) -> float:
    try:
        res = subprocess.run(
            ["osascript", "-e", "output volume of (get volume settings)"],
            capture_output=True,
            text=True,
            check=False,
        )
        val = res.stdout.strip()
        if val.isdigit():
            return float(max(0, min(100, int(val))))
    except Exception:
        pass
    return default


def set_macos_volume(vol_0_to_100: float) -> None:
    pct = int(round(max(VOLUME_MIN, min(VOLUME_MAX, vol_0_to_100))))
    subprocess.Popen(
        ["osascript", "-e", f"set volume output volume {pct}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --- macOS media helpers (Music / Spotify directly, everything else via media keys) ---
def _is_running(process_name: str) -> bool:
    res = subprocess.run(["pgrep", "-x", process_name], capture_output=True, check=False)
    return res.returncode == 0


def _osascript(script: str) -> str:
    res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
    if res.stderr.strip():
        print(f"[Debug] osascript stderr: {res.stderr.strip()}")
    return res.stdout.strip().lower()


def _post_macos_aux_key(nx_key_type: int) -> bool:
    """Send a system media key (what the keyboard's F7/F8/F9 do). macOS routes it to
    whichever app currently owns 'Now Playing' (Chrome, Safari, Firefox, ...)."""
    if not _HAS_QUARTZ:
        print("[Debug] pyobjc missing: pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa")
        return False
    try:
        for is_down in (True, False):
            key_state = 0xA if is_down else 0xB
            data1 = (nx_key_type << 16) | (key_state << 8)
            ns_ev = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                14,  # NSSystemDefined
                (0, 0),
                key_state << 8,
                0,
                0,
                None,
                NX_SUBTYPE_AUX_CONTROL,
                data1,
                -1,
            )
            CGEventPost(kCGHIDEventTap, ns_ev.CGEvent())
            time.sleep(0.02)
        return True
    except Exception as exc:
        print(f"[Debug] Media key error: {exc}")
        return False


def _control_active_desktop_player(command: str, label: str = "") -> bool:
    """Only acts on Spotify/Music if it is running AND currently playing."""
    for app in ("Spotify", "Music"):
        if not _is_running(app):  # avoids osascript trying to launch/locate the app
            continue
        state = _osascript(f'tell application "{app}" to return (player state as string)')
        if state == "playing":
            _osascript(f'tell application "{app}" to {command}')
            print(f"[Debug] ({label}) targeted {app}")
            return True
    return False


def _trigger_active_audio_source(app_command: str, nx_key_type: int, label: str) -> None:
    if _control_active_desktop_player(app_command, label):
        return
    # Browser (or anything else): use the system media key, once.
    if _post_macos_aux_key(nx_key_type):
        print(f"[Debug] ({label}) sent media key")


def on_next_song(state: dict):
    state["last_action"] = "NEXT SONG"
    state["action_time"] = time.monotonic()
    print("[Media] Next song")
    _trigger_active_audio_source("next track", NX_KEYTYPE_NEXT, "NEXT")


def on_prev_song(state: dict):
    state["last_action"] = "PREV SONG"
    state["action_time"] = time.monotonic()
    print("[Media] Previous song")
    _trigger_active_audio_source("previous track", NX_KEYTYPE_PREVIOUS, "PREV")


def on_toggle_play_pause(state: dict):
    state["is_playing"] = not state["is_playing"]
    state["last_action"] = "PLAY" if state["is_playing"] else "PAUSE"
    state["action_time"] = time.monotonic()
    print(f"[Media] Play/Pause -> {'PLAY' if state['is_playing'] else 'PAUSE'}")
    _trigger_active_audio_source("playpause", NX_KEYTYPE_PLAY, "PLAY_PAUSE")



options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

start = time.monotonic()

volume = get_macos_volume(50.0)
last_applied_volume = int(round(volume))

max_travel_px = INITIAL_MAX_TRAVEL_PX
pinch_anchor_y = None
prev_contact_y = None
prev_contact_time = None
is_pinching = False

candidate_name = None
candidate_frames = 0

media_state = {
    "is_playing": True,
    "last_action": "READY",
    "action_time": 0.0,
    "last_gesture_time": 0.0,
    "active_pair": None,
}

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        now = time.monotonic()
        frame = cv2.flip(frame, 1)  # mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int((now - start) * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result is not None:
            h, w, _ = frame.shape
            hands = latest_result.hand_landmarks

            if hands:
                # Track on one hand only (the first detected hand)
                hand = hands[0]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                # Lines between connected joints
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, points[a], points[b], (255, 255, 255), 2)

                # Dots on each joint
                for p in points:
                    cv2.circle(frame, p, 4, (0, 255, 0), -1)

                hand_scale = max(dist_px(points[WRIST], points[MIDDLE_MCP]), 1.0)
                thumb_pt = points[THUMB_TIP]

                # Find whichever finger is closest to the thumb
                finger_checks = [
                    ("VOLUME", INDEX_TIP, CONTACT_RATIO_THRESHOLD, None),
                    ("NEXT", MIDDLE_TIP, THUMB_FINGER_RATIO_THRESHOLD, on_next_song),
                    ("PREV", RING_TIP, THUMB_FINGER_RATIO_THRESHOLD, on_prev_song),
                    ("PLAY_PAUSE", PINKY_TIP, THUMB_FINGER_RATIO_THRESHOLD, on_toggle_play_pause),
                ]

                closest_thumb_pair = None
                for name, tip_idx, threshold, callback in finger_checks:
                    p_finger = points[tip_idx]
                    ratio = dist_px(thumb_pt, p_finger) / hand_scale
                    if ratio < threshold:
                        if closest_thumb_pair is None or ratio < closest_thumb_pair["ratio"]:
                            closest_thumb_pair = {
                                "name": name,
                                "ratio": ratio,
                                "callback": callback,
                                "mx": (thumb_pt[0] + p_finger[0]) // 2,
                                "my": (thumb_pt[1] + p_finger[1]) // 2,
                            }

                # Confirm contact across multiple frames to reduce false triggers during rotation
                if closest_thumb_pair is not None:
                    if candidate_name == closest_thumb_pair["name"]:
                        candidate_frames += 1
                    else:
                        candidate_name = closest_thumb_pair["name"]
                        candidate_frames = 1
                else:
                    candidate_name = None
                    candidate_frames = 0

                confirmed_pair = (
                    closest_thumb_pair
                    if closest_thumb_pair is not None and candidate_frames >= CONTACT_CONFIRM_FRAMES
                    else None
                )

                # --- 1. Thumb + Index vertical pinch volume control (adaptive) ---
                contact_now = confirmed_pair is not None and confirmed_pair["name"] == "VOLUME"

                if contact_now:
                    contact_x = confirmed_pair["mx"]
                    contact_y = confirmed_pair["my"]
                    cv2.circle(frame, (contact_x, contact_y), 8, (0, 255, 255), -1)

                    if not is_pinching:
                        pinch_anchor_y = contact_y

                    if is_pinching and prev_contact_y is not None and prev_contact_time is not None and pinch_anchor_y is not None:
                        current_travel = abs(contact_y - pinch_anchor_y)
                        if current_travel > max_travel_px:
                            max_travel_px = current_travel

                        dt = max(now - prev_contact_time, 1e-3)
                        dy = prev_contact_y - contact_y

                        if abs(dy) > MOVE_DEADZONE_PX:
                            travel_factor = max_travel_px / max(INITIAL_MAX_TRAVEL_PX, 1.0)
                            normalized_step = (abs(dy) / max_travel_px) * FULL_RANGE_VOLUME_SPAN

                            speed_px_per_sec = abs(dy) / dt
                            speed_boost = 1.0 + VELOCITY_GAIN * (speed_px_per_sec ** (ACCEL_EXPONENT - 1.0)) * travel_factor

                            direction = 1.0 if dy > 0 else -1.0
                            delta = direction * normalized_step * speed_boost
                            volume = max(VOLUME_MIN, min(VOLUME_MAX, volume + delta))

                            new_vol_int = int(round(volume))
                            if new_vol_int != last_applied_volume:
                                set_macos_volume(new_vol_int)
                                last_applied_volume = new_vol_int

                    prev_contact_y = contact_y
                    prev_contact_time = now
                    is_pinching = True
                else:
                    is_pinching = False
                    pinch_anchor_y = None
                    prev_contact_y = None
                    prev_contact_time = None

                # --- 2. Thumb + Middle / Ring / Pinky contact gestures ---
                if confirmed_pair is not None and confirmed_pair["name"] != "VOLUME":
                    name = confirmed_pair["name"]
                    cv2.circle(frame, (confirmed_pair["mx"], confirmed_pair["my"]), 8, (255, 0, 255), -1)

                    cooldown_ok = (now - media_state["last_gesture_time"]) >= GESTURE_COOLDOWN_SEC
                    if media_state["active_pair"] != name and cooldown_ok:
                        media_state["active_pair"] = name
                        media_state["last_gesture_time"] = now
                        confirmed_pair["callback"](media_state)
                elif confirmed_pair is None:
                    media_state["active_pair"] = None

                # --- 3. Display volume & gesture status near the hand skeleton ---
                label_x = min(max(points[WRIST][0] - 55, 10), w - 260)
                label_y = min(max(points[WRIST][1] + 28, 30), h - 45)

                cv2.putText(
                    frame,
                    f"Vol: {volume:.0f}% (Max:{max_travel_px:.0f}px)",
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255) if contact_now else (0, 255, 0),
                    2,
                )

                show_recent = (now - media_state["action_time"]) < 1.2
                status_text = media_state["last_action"] if show_recent else ("PLAYING" if media_state["is_playing"] else "PAUSED")
                cv2.putText(
                    frame,
                    f"Media: {status_text}",
                    (label_x, min(label_y + 26, h - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255) if show_recent else (200, 200, 200),
                    2,
                )
            else:
                is_pinching = False
                pinch_anchor_y = None
                prev_contact_y = None
                prev_contact_time = None
                candidate_name = None
                candidate_frames = 0
                media_state["active_pair"] = None

        cv2.imshow("Hand tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
