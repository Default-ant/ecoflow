from __future__ import annotations
"""
EcoFlow AI — Adaptive Traffic Signal Controller + GPIO Driver
=============================================================
Two responsibilities kept in one file:

  A) Physical LED control (GPIO)
     ─────────────────────────────
     TrafficLight  — wraps gpiozero LEDs; call .set_red() / .set_green() etc.
     Works on Raspberry Pi 5 with the gpiozero + lgpio backend.
     Degrades gracefully to a mock when GPIO is not available (desktop testing).

  B) Signal-phase logic
     ──────────────────
     get_lane_density   — count vehicles per ROI lane
     signal_logic       — proportional green-time allocation
     get_signal_phase   — full pipeline, returns SignalPhase (incl. emergency)
     draw_rois          — overlay lane polygons on a BGR frame

Hardware wiring (BCM GPIO numbering):
  Red    LED  →  GPIO 17  (Physical Pin 11)
  Yellow LED  →  GPIO 27  (Physical Pin 13)
  Green  LED  →  GPIO 22  (Physical Pin 15)
  Common GND  →  Physical Pin 9 or 14

Change PIN_RED / PIN_YELLOW / PIN_GREEN below if your wiring differs.
"""  # noqa: E501

# ─────────────────────────────────────────────────────────────────────────────
# Section A — GPIO Traffic Light
# ─────────────────────────────────────────────────────────────────────────────

# BCM pin numbers
PIN_RED    = 17
PIN_YELLOW = 27
PIN_GREEN  = 22

# COCO vehicle classes used throughout EcoFlow AI
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def _init_leds(no_gpio: bool = False):
    """Return (red, yellow, green) gpiozero LED objects, or (None, None, None)."""
    if no_gpio:
        return None, None, None
    try:
        from gpiozero import LED
        return LED(PIN_RED), LED(PIN_YELLOW), LED(PIN_GREEN)
    except Exception as exc:
        print(f"[GPIO] Not available ({exc}). Running without physical LEDs.")
        return None, None, None


class TrafficLight:
    """
    Simple traffic-light abstraction over three gpiozero LEDs.

    Parameters
    ----------
    no_gpio : bool
        Pass True on a non-Pi machine to run without physical GPIO.

    Example
    -------
    light = TrafficLight()
    light.set_green()   # → green LED on, others off
    light.set_red()     # → red LED on
    light.cleanup()     # call on shutdown
    """

    def __init__(self, no_gpio: bool = False):
        self._r, self._y, self._g = _init_leds(no_gpio)
        self._state = "off"
        self.set_red()            # safe power-on default

    # ── Private ──────────────────────────────────────────────────────────────
    def _all_off(self):
        for led in (self._r, self._y, self._g):
            if led:
                led.off()

    # ── Public ───────────────────────────────────────────────────────────────
    def set_red(self):
        self._all_off()
        if self._r:
            self._r.on()
        self._state = "red"
        print("[Light] 🔴  RED")

    def set_yellow(self):
        self._all_off()
        if self._y:
            self._y.on()
        self._state = "yellow"
        print("[Light] 🟡  YELLOW")

    def set_green(self):
        self._all_off()
        if self._g:
            self._g.on()
        self._state = "green"
        print("[Light] 🟢  GREEN")

    @property
    def state(self) -> str:
        return self._state

    def cleanup(self):
        """Turn all LEDs off and release GPIO resources."""
        self._all_off()
        try:
            from gpiozero import Device
            Device.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Section B — Lane density + signal logic (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

"""
EcoFlow AI — Adaptive Traffic Signal Controller
================================================
Integrates with YOLOv11 tracker output to:

  • get_lane_density(tracks)   – count vehicles inside each lane's ROI polygon
  • signal_logic(densities)    – allocate green time proportionally to density
  • Emergency Override         – immediately grants green if an Ambulance or
                                  Fire Truck track is detected in any lane

Lane layout (default, override via LANE_ROIS):
  Lane 0 – North approach
  Lane 1 – East  approach
  Lane 2 – South approach
  Lane 3 – West  approach

tracks format (list of dicts, one per active track):
  {
      "track_id" : int,
      "label"    : str,   # e.g. "car", "truck", "Ambulance", "Fire Truck"
      "cx"       : float, # centre-x in frame pixels
      "cy"       : float, # centre-y in frame pixels
  }
"""



import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────────────

# A polygon region-of-interest: list of (x, y) vertex tuples (pixel space)
Polygon = List[Tuple[int, int]]
Track   = Dict  # {"track_id": int, "label": str, "cx": float, "cy": float}


# ──────────────────────────────────────────────────────────────────────────────
# Default ROI polygons  (edit to match your camera/intersection geometry)
#
#  Assumes a 1920×1080 frame split into four rectangular approach zones.
#  Replace with real perspective-correct polygons for deployment.
# ──────────────────────────────────────────────────────────────────────────────

LANE_ROIS: List[Polygon] = [
    # Lane 0 – North approach  (top strip)
    [(400, 0),   (1520, 0),   (1520, 400),  (400, 400)],

    # Lane 1 – East approach   (right strip)
    [(1520, 200), (1920, 200), (1920, 880),  (1520, 880)],

    # Lane 2 – South approach  (bottom strip)
    [(400, 680),  (1520, 680), (1520, 1080), (400, 1080)],

    # Lane 3 – West approach   (left strip)
    [(0, 200),    (400, 200),  (400, 880),   (0, 880)],
]

# Labels that trigger an Emergency Override (case-insensitive comparison)
EMERGENCY_LABELS: frozenset[str] = frozenset({"ambulance", "fire truck"})

# Green-time bounds (seconds)
MIN_GREEN: float = 20.0
MAX_GREEN: float = 60.0

# Fallback equal green time when all lanes are empty
DEFAULT_GREEN: float = 30.0

# Number of lanes
NUM_LANES: int = len(LANE_ROIS)

# Lane names for display / logging
LANE_NAMES: List[str] = ["North", "East", "South", "West"]


# ──────────────────────────────────────────────────────────────────────────────
# Signal phase dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalPhase:
    """Describes the current signal decision for all lanes."""

    green_lane: int                        # which lane gets green
    green_time: float                      # seconds of green
    all_green_times: List[float]           # calculated green time per lane
    densities: List[int]                   # vehicle count per lane
    emergency_override: bool = False       # True if triggered by emergency vehicle
    emergency_lane: Optional[int] = None  # lane that triggered override
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        base = (
            f"[Signal] Green → Lane {self.green_lane} "
            f"({LANE_NAMES[self.green_lane]}) "
            f"for {self.green_time:.1f}s"
        )
        if self.emergency_override:
            base = (
                f"[Signal] 🚨 EMERGENCY OVERRIDE → Lane {self.emergency_lane} "
                f"({LANE_NAMES[self.emergency_lane]}) "
                f"GREEN immediately"
            )
        density_str = "  |  Densities: " + ", ".join(
            f"{LANE_NAMES[i]}={self.densities[i]}" for i in range(NUM_LANES)
        )
        return base + density_str


# ──────────────────────────────────────────────────────────────────────────────
# Helper: point-in-polygon test
# ──────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(x: float, y: float, polygon: Polygon) -> bool:
    """
    Ray-casting algorithm.
    Returns True if (x, y) is inside *polygon*.
    """
    n     = len(polygon)
    inside = False
    px, py = x, y
    xi, yi = polygon[0]
    for j in range(1, n + 1):
        xj, yj = polygon[j % n]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        xi, yi = xj, yj
    return inside


# ──────────────────────────────────────────────────────────────────────────────
# Core API
# ──────────────────────────────────────────────────────────────────────────────

def get_lane_density(
    tracks: List[Track],
    lane_rois: Optional[List[Polygon]] = None,
) -> List[int]:
    """
    Count the number of vehicles (tracks) whose centre point falls inside
    each lane's ROI polygon.

    Parameters
    ----------
    tracks : list of track dicts
        Each dict must have at least:
            "cx"  (float) – centre-x in frame pixels
            "cy"  (float) – centre-y in frame pixels
        Optional keys used elsewhere:
            "track_id" (int), "label" (str)
    lane_rois : list of Polygon, optional
        Defaults to the module-level ``LANE_ROIS``.

    Returns
    -------
    List[int]
        Vehicle count for each lane, in lane-index order.

    Example
    -------
    >>> tracks = [
    ...     {"track_id": 1, "label": "car",    "cx": 800, "cy": 200},
    ...     {"track_id": 2, "label": "truck",  "cx": 800, "cy": 750},
    ...     {"track_id": 3, "label": "Ambulance", "cx": 50, "cy": 500},
    ... ]
    >>> get_lane_density(tracks)
    [1, 0, 1, 1]
    """
    rois = lane_rois or LANE_ROIS
    densities: List[int] = [0] * len(rois)

    for track in tracks:
        cx = float(track.get("cx", -1))
        cy = float(track.get("cy", -1))
        if cx < 0 or cy < 0:
            continue
        for lane_idx, polygon in enumerate(rois):
            if _point_in_polygon(cx, cy, polygon):
                densities[lane_idx] += 1
                break  # a vehicle belongs to at most one lane

    return densities


def signal_logic(
    densities: List[int],
    min_green: float = MIN_GREEN,
    max_green: float = MAX_GREEN,
) -> List[float]:
    """
    Allocate green-light durations proportionally to lane vehicle density.

    • The lane with the **highest** density receives *max_green* seconds.
    • Other lanes receive time scaled linearly between *min_green* and
      *max_green*.
    • If all lanes are empty, every lane gets ``DEFAULT_GREEN``.

    Parameters
    ----------
    densities : list of int
        Vehicle count per lane (from ``get_lane_density``).
    min_green : float
        Minimum green time in seconds (default 20 s).
    max_green : float
        Maximum green time in seconds (default 60 s).

    Returns
    -------
    List[float]
        Suggested green-light duration (seconds) for each lane.

    Example
    -------
    >>> signal_logic([5, 2, 0, 3])
    [60.0, 36.0, 20.0, 44.0]
    """
    if not densities:
        return []

    total = sum(densities)
    if total == 0:
        return [DEFAULT_GREEN] * len(densities)

    max_d   = max(densities)
    span    = max_green - min_green

    green_times = []
    for d in densities:
        # Linear interpolation: 0 density → min_green, max density → max_green
        gt = min_green + span * (d / max_d)
        green_times.append(round(gt, 1))

    return green_times


def get_signal_phase(
    tracks: List[Track],
    lane_rois: Optional[List[Polygon]] = None,
    min_green: float = MIN_GREEN,
    max_green: float = MAX_GREEN,
) -> SignalPhase:
    """
    Full pipeline: density → signal logic → emergency check → SignalPhase.

    This is the primary entry-point for the tracking loop.

    Parameters
    ----------
    tracks : list of track dicts
    lane_rois : list of Polygon, optional
    min_green, max_green : float
        Green-time bounds (seconds).

    Returns
    -------
    SignalPhase
        Contains green lane, green time, per-lane times, densities,
        and emergency override state.

    Emergency Override
    ------------------
    If *any* track in *any* lane carries a label that matches
    ``EMERGENCY_LABELS`` (case-insensitive), the signal for **that lane**
    is flipped to green immediately, bypassing normal proportional logic.
    If multiple lanes have emergency vehicles, the first one found wins.

    Notes
    -----
    Call this function once per decision cycle (e.g., every 5–10 seconds
    in a real controller, or every frame for simulation).
    """
    rois       = lane_rois or LANE_ROIS
    n_lanes    = len(rois)

    # ── Step 1: Compute densities ─────────────────────────────────────────
    densities = get_lane_density(tracks, rois)

    # ── Step 2: Compute proportional green times ──────────────────────────
    green_times = signal_logic(densities, min_green, max_green)

    # ── Step 3: Emergency Override scan ──────────────────────────────────
    #   Map each track to a lane, then check its label.
    emergency_lane: Optional[int] = None
    for track in tracks:
        label = str(track.get("label", "")).strip().lower()
        if label not in EMERGENCY_LABELS:
            continue
        # Find which lane this emergency vehicle is in
        cx = float(track.get("cx", -1))
        cy = float(track.get("cy", -1))
        for lane_idx, polygon in enumerate(rois):
            if _point_in_polygon(cx, cy, polygon):
                emergency_lane = lane_idx
                break
        if emergency_lane is not None:
            break  # first emergency vehicle wins

    if emergency_lane is not None:
        return SignalPhase(
            green_lane=emergency_lane,
            green_time=max_green,          # give maximum green immediately
            all_green_times=green_times,
            densities=densities,
            emergency_override=True,
            emergency_lane=emergency_lane,
        )

    # ── Step 4: Normal phase – highest density wins ───────────────────────
    best_lane = int(np.argmax(green_times))   # ties go to lower lane index
    return SignalPhase(
        green_lane=best_lane,
        green_time=green_times[best_lane],
        all_green_times=green_times,
        densities=densities,
        emergency_override=False,
        emergency_lane=None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Optional: visualise ROIs on a frame
# ──────────────────────────────────────────────────────────────────────────────

def draw_rois(
    frame,   # np.ndarray (BGR)
    phase: Optional[SignalPhase] = None,
    lane_rois: Optional[List[Polygon]] = None,
    alpha: float = 0.25,
):
    """
    Overlay ROI polygons on *frame* (in-place).
    Active green lane is highlighted green; others are grey.
    Emergency override lane is highlighted red.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate.
    phase : SignalPhase, optional
        If provided, colours active lane accordingly.
    lane_rois : list of Polygon, optional
    alpha : float
        Blend opacity for the filled polygon (0–1).
    """
    import cv2  # local import – keeps module importable without OpenCV

    rois   = lane_rois or LANE_ROIS
    overlay = frame.copy()

    for lane_idx, polygon in enumerate(rois):
        pts = np.array(polygon, dtype=np.int32)

        if phase is None:
            color = (100, 100, 100)
        elif phase.emergency_override and lane_idx == phase.emergency_lane:
            color = (0, 0, 220)          # red  – emergency
        elif not phase.emergency_override and lane_idx == phase.green_lane:
            color = (0, 200, 60)         # green – active lane
        else:
            color = (80, 80, 80)         # grey  – waiting

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # Lane label
        cx_l = int(np.mean([p[0] for p in polygon]))
        cy_l = int(np.mean([p[1] for p in polygon]))
        density_txt = f"{LANE_NAMES[lane_idx]}"
        if phase:
            density_txt += f" D={phase.densities[lane_idx]}"
            density_txt += f" G={phase.all_green_times[lane_idx]:.0f}s"
        cv2.putText(
            frame, density_txt,
            (cx_l - 60, cy_l),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_tracks = [
        # Lane 0 – North  (top strip)
        {"track_id": 1, "label": "car",   "cx": 800,  "cy": 200},
        {"track_id": 2, "label": "truck", "cx": 900,  "cy": 300},
        {"track_id": 3, "label": "car",   "cx": 1000, "cy": 150},

        # Lane 1 – East   (right strip)
        {"track_id": 4, "label": "motorcycle", "cx": 1700, "cy": 500},

        # Lane 2 – South  (bottom strip)  ← empty in this sample

        # Lane 3 – West   (left strip)
        {"track_id": 5, "label": "car",   "cx": 200, "cy": 500},
        {"track_id": 6, "label": "truck", "cx": 100, "cy": 600},
    ]

    print("=== Normal operation ===")
    densities = get_lane_density(sample_tracks)
    print(f"Densities  : {densities}")
    print(f"Green times: {signal_logic(densities)}")
    phase = get_signal_phase(sample_tracks)
    print(phase)
    print()

    print("=== Emergency Override (Ambulance in West lane) ===")
    emergency_tracks = sample_tracks + [
        {"track_id": 99, "label": "Ambulance", "cx": 50, "cy": 400},
    ]
    phase_e = get_signal_phase(emergency_tracks)
    print(phase_e)
    print()

    print("=== Emergency Override (Fire Truck in North lane) ===")
    fire_tracks = sample_tracks + [
        {"track_id": 77, "label": "Fire Truck", "cx": 700, "cy": 100},
    ]
    phase_f = get_signal_phase(fire_tracks)
    print(phase_f)
