"""
EcoFlow AI — Ambulance / Emergency-Vehicle Detector
====================================================
Self-contained module:  import and call ``check_track(frame, track_id,
cls_id, x1, y1, x2, y2, state)`` every frame.  Returns True the first
time a vehicle accumulates enough evidence to be confirmed as an
emergency vehicle.

Detection uses THREE complementary paths so that India's diverse
ambulance fleet (red-cross vans, force Travellers with blue/red sirens,
white government ambulances, etc.) are all caught:

  Path 1 – Body-colour (bus / truck only)
      • Strong red body / red-cross marking
      • Blue siren light + yellow-green reflective stripe together
      • White-body vehicle with visible red cross

  Path 2 – Top-strip siren (any vehicle class)
      Scans only the TOP 22 % of the bounding box (where the siren bar
      lives) for simultaneous red + blue LEDs.  Naturally excludes
      tail-lights (bottom of box) and orange auto tops.

  Path 3 – Temporal blink history (any vehicle class)
      Accumulates (red_frac, blue_frac) across 20 frames; confirms if
      ≥ 2 frames show significant red AND ≥ 2 show significant blue
      (alternating flash pattern).

All three paths feed a vote counter: confirmation requires
VOTE_THRESHOLD (3) votes within any VOTE_WINDOW (8)-frame rolling
window, preventing single-frame false positives.

Additional guards applied before any colour check:
  • Minimum bounding-box size   (80 × 60 px)  — excludes autos / bikes
  • Maximum aspect ratio        (3.2)          — excludes wide flat cars

Usage
-----
from ambulance_detection import AmbulanceState, check_track

state = AmbulanceState()         # one instance per camera / session
frame_idx = 0

while True:
    frame = camera.read()
    for track_id, cls_id, x1, y1, x2, y2 in detections:
        if check_track(frame, track_id, cls_id, x1, y1, x2, y2, state):
            # first time this vehicle is confirmed as ambulance
            trigger_green_light()

    # confirmed set persists across frames
    ambulances = state.confirmed

Tunable constants
-----------------
All caps constants near the top of this file.
"""

from __future__ import annotations

import collections
from typing import Dict

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────────────────────────────────

# ── HSV ambulance colour thresholds ──────────────────────────────────────────

# Red  (hue wraps 0/180)
RED_LOWER1 = np.array([0,   80, 60],  dtype=np.uint8)
RED_UPPER1 = np.array([10,  255, 255], dtype=np.uint8)
RED_LOWER2 = np.array([165, 80, 60],  dtype=np.uint8)
RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

# Blue (siren lights)
BLUE_LOWER = np.array([100, 80, 80],  dtype=np.uint8)
BLUE_UPPER = np.array([130, 255, 255], dtype=np.uint8)

# Yellow-green (reflective stripe on sides — excludes auto orange-yellow)
YG_LOWER = np.array([40,  60, 100], dtype=np.uint8)
YG_UPPER = np.array([80, 255, 255], dtype=np.uint8)

# White body
WHITE_LOWER = np.array([0,   0,  190], dtype=np.uint8)
WHITE_UPPER = np.array([180, 40, 255], dtype=np.uint8)

# ── Detection thresholds ──────────────────────────────────────────────────────
RED_THRESHOLD   = 0.06    # ≥ 6 % vivid-red  → confirms (body-colour path)
BLUE_THRESHOLD  = 0.04    # ≥ 4 % blue siren  (used with YG)
YG_THRESHOLD    = 0.07    # ≥ 7 % YG stripe   (used with blue)
WHITE_THRESHOLD = 0.30    # ≥ 30 % white body
WHITE_RED_COMBO = 0.04    # red fraction required when body is white (general)
WHITE_RED_BUS   = WHITE_RED_COMBO * 0.6   # lower threshold for bus/truck class

# Siren / blink detection (top-strip scan)
BLINK_RED_THRESH  = 0.018
BLINK_BLUE_THRESH = 0.018
BLINK_MIN_HITS    = 2           # frames that must exceed threshold
COLOUR_HISTORY_LEN = 20         # rolling window length in frames

# ── Guards ────────────────────────────────────────────────────────────────────
MIN_AMBULANCE_W  = 40    # pixels (boosted sensitivity)
MIN_AMBULANCE_H  = 30    # pixels (boosted sensitivity)
MAX_ASPECT_RATIO = 3.2   # max width/height

# ── Vote-based confirmation ───────────────────────────────────────────────────
VOTE_WINDOW    = 8    # rolling window (frames)
VOTE_THRESHOLD = 2    # lowered from 3 for faster trigger

# Fraction of bounding-box height used for the top-strip siren scan
SIREN_STRIP = 0.22

# COCO class IDs eligible for body-colour check (bus=5, truck=7)
AMBULANCE_CANDIDATE_CLASSES: frozenset[int] = frozenset({5, 7})


# ─────────────────────────────────────────────────────────────────────────────
# Session state  (one instance per camera)
# ─────────────────────────────────────────────────────────────────────────────

class AmbulanceState:
    """
    Holds all per-track mutable state needed by the detector.
    Create one instance and pass it to every ``check_track`` call.
    """

    def __init__(self):
        # Set of track IDs confirmed as ambulances
        self.confirmed: set[int] = set()

        # Per-track rolling siren colour history: deque of (red_frac, blue_frac)
        self._colour_hist: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=COLOUR_HISTORY_LEN)
        )

        # Per-track rolling vote window: deque of 0/1
        self._votes: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=VOTE_WINDOW)
        )

    def reset(self):
        """Clear all state (call between unrelated video clips)."""
        self.confirmed.clear()
        self._colour_hist.clear()
        self._votes.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _frac(hsv: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of pixels in *hsv* that fall within the HSV range [lo, hi]."""
    mask = cv2.inRange(hsv, lo, hi)
    return float(np.count_nonzero(mask)) / max(hsv.size // 3, 1)


def _body_colour_check(frame: np.ndarray,
                       x1: int, y1: int, x2: int, y2: int,
                       cls_id: int) -> bool:
    """
    Path 1: full-body HSV colour analysis.

    Returns True if the vehicle crop matches any of:
      • Strong red body / red cross
      • Blue siren + yellow-green reflective stripe
      • White body + red cross  (lower threshold for bus/truck)
    """
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red   = _frac(hsv, RED_LOWER1, RED_UPPER1) + _frac(hsv, RED_LOWER2, RED_UPPER2)
    blue  = _frac(hsv, BLUE_LOWER, BLUE_UPPER)
    yg    = _frac(hsv, YG_LOWER,   YG_UPPER)
    white = _frac(hsv, WHITE_LOWER, WHITE_UPPER)

    if red >= RED_THRESHOLD:
        return True
    if blue >= BLUE_THRESHOLD and yg >= YG_THRESHOLD:
        return True
    wrc = WHITE_RED_BUS if cls_id in {5, 7} else WHITE_RED_COMBO
    if white >= WHITE_THRESHOLD and red >= wrc:
        return True
    return False


def _siren_fracs(frame: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    """
    Return (red_frac, blue_frac) scanning ONLY the top SIREN_STRIP of the
    bounding box.  Siren lights are always at the roof; this excludes
    tail-lights, floor reflections, and auto tops.
    """
    strip_h  = max(int((y2 - y1) * SIREN_STRIP), 10)
    strip    = frame[y1: y1 + strip_h, x1:x2]
    if strip.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    r   = _frac(hsv, RED_LOWER1, RED_UPPER1) + _frac(hsv, RED_LOWER2, RED_UPPER2)
    b   = _frac(hsv, BLUE_LOWER, BLUE_UPPER)
    return r, b


def _has_blink_pattern(hist: list[tuple[float, float]]) -> bool:
    """
    Return True if the colour history shows an alternating red/blue blink
    pattern (≥ BLINK_MIN_HITS frames with red AND ≥ BLINK_MIN_HITS with blue).
    """
    if len(hist) < max(4, BLINK_MIN_HITS * 2):
        return False
    red_hits  = sum(1 for r, _ in hist if r >= BLINK_RED_THRESH)
    blue_hits = sum(1 for _, b in hist if b >= BLINK_BLUE_THRESH)
    return red_hits >= BLINK_MIN_HITS and blue_hits >= BLINK_MIN_HITS


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def check_track(
    frame: np.ndarray,
    track_id: int,
    cls_id: int,
    x1: int, y1: int, x2: int, y2: int,
    state: AmbulanceState,
) -> bool:
    """
    Analyse one bounding box for this frame and update *state*.

    Returns
    -------
    bool
        True **only on the frame when confirmation first happens**.
        Use ``state.confirmed`` to check confirmed status on later frames.
    """
    # Already confirmed — skip expensive checks
    if track_id in state.confirmed:
        return False

    box_w, box_h = x2 - x1, y2 - y1

    # ── Size & aspect guards ─────────────────────────────────────────────────
    if box_w < MIN_AMBULANCE_W or box_h < MIN_AMBULANCE_H:
        state._votes[track_id].append(0)
        return False
    if (box_w / max(box_h, 1)) > MAX_ASPECT_RATIO:
        state._votes[track_id].append(0)
        return False

    # ── Path 1: body-colour check (bus / truck only) ─────────────────────────
    colour_hit = (cls_id in AMBULANCE_CANDIDATE_CLASSES and
                  _body_colour_check(frame, x1, y1, x2, y2, cls_id))

    # ── Paths 2 & 3: siren top-strip ─────────────────────────────────────────
    siren_hit = False
    r_f, b_f  = _siren_fracs(frame, x1, y1, x2, y2)

    # Path 2: instant — both colours simultaneously (e.g. red+blue siren bar)
    if r_f >= BLINK_RED_THRESH and b_f >= BLINK_BLUE_THRESH:
        siren_hit = True
    else:
        # Path 3: temporal — accumulate history and look for alternating flash
        state._colour_hist[track_id].append((r_f, b_f))
        if _has_blink_pattern(list(state._colour_hist[track_id])):
            siren_hit = True

    # ── Vote accumulation ────────────────────────────────────────────────────
    state._votes[track_id].append(1 if (colour_hit or siren_hit) else 0)

    if sum(state._votes[track_id]) >= VOTE_THRESHOLD:
        state.confirmed.add(track_id)
        return True   # ← first confirmation event

    return False


def build_tracks_list(ids, xyxy, clsids, confs, confirmed: set[int]) -> list[dict]:
    """
    Helper: build the ``tracks`` list expected by signal_controller and
    eco_risk from raw YOLO output tensors.

    Returns a list of dicts:
        {"track_id", "label", "cx", "cy", "conf"}
    """
    from signal_controller import VEHICLE_CLASSES  # avoid circular import at module level

    result = []
    for tid, box, cls_id, conf in zip(ids, xyxy, clsids, confs):
        x1, y1, x2, y2 = (int(v) for v in box)
        label = "Ambulance" if tid in confirmed else VEHICLE_CLASSES.get(cls_id, "vehicle")
        result.append({
            "track_id": tid,
            "label":    label,
            "cx":       (x1 + x2) / 2,
            "cy":       (y1 + y2) / 2,
            "conf":     conf,
        })
    return result
