from __future__ import annotations
"""
EcoFlow AI — 4-Direction Adaptive Traffic Signal Controller + GPIO Driver
========================================================================

Hardware wiring (BCM GPIO numbering) for 4 Signals (12 LEDs):
  Signal 0 (North): R:17, Y:27, G:22  (Pins 11, 13, 15)
  Signal 1 (East) : R:10, Y:9,  G:11  (Pins 19, 21, 23)
  Signal 2 (South): R:5,  Y:6,  G:13  (Pins 29, 31, 33)
  Signal 3 (West) : R:19, Y:26, G:21  (Pins 35, 37, 40)

Logic:
  1. Default: 10s green per signal (Round-robin 0 -> 1 -> 2 -> 3).
  2. High Traffic Priority: If Lane X is heavy (>5) and others are empty (<2 sum), jump to Lane X.
  3. Ambulance Priority: Give immediate green to the lane with an ambulance.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Section A — GPIO Pin Mapping
# ─────────────────────────────────────────────────────────────────────────────

# BCM pin numbers (R, Y, G)
SIGNAL_PINS = [
    (17, 27, 22),  # Signal 0 - North
    (10, 9,  11),  # Signal 1 - East
    (5,  6,  13),  # Signal 2 - South
    (19, 26, 21),  # Signal 3 - West
]

# COCO vehicle classes
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def _init_signal_leds(no_gpio: bool = False):
    """Return list of (red, yellow, green) LED objects for 4 signals."""
    if no_gpio:
        return [(None, None, None)] * 4
    try:
        from gpiozero import LED
        led_signals = []
        for r, y, g in SIGNAL_PINS:
            led_signals.append((LED(r), LED(y), LED(g)))
        return led_signals
    except Exception as exc:
        print(f"[GPIO] Not available ({exc}). Running without physical LEDs.")
        return [(None, None, None)] * 4

class TrafficLight:
    """Manages 4 separate traffic signals (12 LEDs total)."""

    def __init__(self, no_gpio: bool = False):
        self._signals = _init_signal_leds(no_gpio)
        self.states = ["red"] * 4
        self.all_red()  # Safe default

    def _all_off(self, signal_idx: int):
        for led in self._signals[signal_idx]:
            if led:
                led.off()

    def set_signal(self, signal_idx: int, state: str):
        """Set a specific signal to red, yellow, or green."""
        if signal_idx < 0 or signal_idx >= 4:
            return
        
        self._all_off(signal_idx)
        r, y, g = self._signals[signal_idx]
        
        if state == "red":
            if r: r.on()
            self.states[signal_idx] = "red"
        elif state == "yellow":
            if y: y.on()
            self.states[signal_idx] = "yellow"
        elif state == "green":
            if g: g.on()
            self.states[signal_idx] = "green"
            
        # print(f"[Signal {signal_idx}] -> {state.upper()}")

    def all_red(self):
        for i in range(4):
            self.set_signal(i, "red")

    def update_4way(self, green_lane: int):
        """Turn specified lane GREEN, all others RED."""
        for i in range(4):
            if i == green_lane:
                self.set_signal(i, "green")
            else:
                self.set_signal(i, "red")

    def cleanup(self):
        for i in range(4):
            self._all_off(i)
        try:
            from gpiozero import Device
            Device.close()
        except:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# Section B — Logic State Machine
# ─────────────────────────────────────────────────────────────────────────────

LANE_ROIS: List[List[Tuple[int, int]]] = [
    [(400, 0),   (1520, 0),   (1520, 400),  (400, 400)],    # North
    [(1520, 200), (1920, 200), (1920, 880),  (1520, 880)],  # East
    [(400, 680),  (1520, 680), (1520, 1080), (400, 1080)], # South
    [(0, 200),    (400, 200),  (400, 880),   (0, 880)],    # West
]

LANE_NAMES = ["North", "East", "South", "West"]
EMERGENCY_LABELS = {"ambulance", "fire truck"}

# Thresholds
DEFAULT_GREEN_TIME = 10.0   # seconds
HIGH_TRAFFIC_THRESH = 5     # vehicles
EMPTY_OTHERS_THRESH = 2     # total vehicles in all other lanes

@dataclass
class SignalState:
    current_lane: int = 0
    last_switch_time: float = field(default_factory=time.time)
    override_reason: Optional[str] = None
    remaining_time: float = DEFAULT_GREEN_TIME

class AdaptiveController:
    """Main logic core for 4-direction traffic management."""

    def __init__(self, green_time: float = DEFAULT_GREEN_TIME):
        self.state = SignalState()
        self.green_time_default = green_time

    def get_decision(self, tracks: List[dict]) -> Tuple[int, str]:
        """
        Determines which lane should be green based on current frame data.
        Returns: (lane_index, reason)
        """
        now = time.time()
        densities = self._get_densities(tracks)
        emergency_lane = self._get_emergency_lane(tracks)
        elapsed = now - self.state.last_switch_time

        # 1. Ambulance Priority (Highest)
        if emergency_lane is not None:
            if self.state.current_lane != emergency_lane:
                self.state.current_lane = emergency_lane
                self.state.last_switch_time = now
            self.state.override_reason = "AMBULANCE"
            self.state.remaining_time = self.green_time_default
            return emergency_lane, "AMBULANCE"

        # 2. Dynamic High Traffic Priority
        max_d = max(densities)
        if max_d >= HIGH_TRAFFIC_THRESH:
            # If another lane has MORE traffic than the current lane, jump to it
            if max_d > densities[self.state.current_lane]:
                best_lane = densities.index(max_d)
                self.state.current_lane = best_lane
                self.state.last_switch_time = now
            
            # Keep priority (hold green) until traffic is controlled
            self.state.override_reason = "HIGH TRAFFIC"
            self.state.remaining_time = self.green_time_default 
            return self.state.current_lane, "HIGH TRAFFIC"

        # 3. Round-Robin Cycle (Default)
        if elapsed >= self.green_time_default:
            self.state.current_lane = (self.state.current_lane + 1) % 4
            self.state.last_switch_time = now
            elapsed = 0
        
        self.state.override_reason = "NORMAL CYCLE"
        self.state.remaining_time = max(0.0, self.green_time_default - elapsed)
        return self.state.current_lane, "NORMAL CYCLE"

    def _get_densities(self, tracks: List[dict]) -> List[int]:
        densities = [0] * 4
        for track in tracks:
            cx, cy = track.get("cx", -1), track.get("cy", -1)
            for i, roi in enumerate(LANE_ROIS):
                if self._point_in_roi(cx, cy, roi):
                    densities[i] += 1
                    break
        return densities

    def _get_emergency_lane(self, tracks: List[dict]) -> Optional[int]:
        for track in tracks:
            label = str(track.get("label", "")).lower()
            if label in EMERGENCY_LABELS:
                cx, cy = track.get("cx", -1), track.get("cy", -1)
                for i, roi in enumerate(LANE_ROIS):
                    if self._point_in_roi(cx, cy, roi):
                        return i
        return None

    def _point_in_roi(self, x, y, polygon):
        if x < 0 or y < 0: return False
        n = len(polygon)
        inside = False
        px, py = x, y
        for i in range(n):
            j = (i + 1) % n
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
        return inside

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation Support (Compatibility with original draw_rois)
# ─────────────────────────────────────────────────────────────────────────────

def draw_rois(frame, lane_idx: int, densities: List[int], reason: str, remaining: float):
    import cv2
    overlay = frame.copy()
    for i, roi in enumerate(LANE_ROIS):
        pts = np.array(roi, dtype=np.int32)
        color = (0, 255, 0) if i == lane_idx else (80, 80, 80)
        if "AMBULANCE" in reason and i == lane_idx:
            color = (0, 0, 255) # Red for emergency
        
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)

        cx = int(np.mean([p[0] for p in roi]))
        cy = int(np.mean([p[1] for p in roi]))
        txt = f"{LANE_NAMES[i]}: {densities[i]}"
        if i == lane_idx:
            txt += f" ({reason} - {remaining:.1f}s)"
        
        cv2.putText(frame, txt, (cx - 80, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
