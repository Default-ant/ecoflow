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
        
        if not no_gpio:
            self.hardware_test()
            
        self.update_4way(0)  # Start with North GREEN

    def hardware_test(self):
        """Diagnostic: Cycle every LED in the system for 0.2s each."""
        print("[GPIO] 🔍 Running Hardware Diagnostics (12 LEDs)...")
        for i in range(4):
            for state in ["red", "yellow", "green"]:
                self.set_signal(i, state)
                time.sleep(0.2)
            self._all_off(i)
        print("[GPIO] ✅ Diagnostics complete. Starting AI cycle.\n")

    @property
    def status_bar(self) -> str:
        """Returns a compact status string like: [ N:🟢 | E:🔴 | S:🔴 | W:🔴 ]"""
        parts = []
        for i, name in enumerate(["N", "E", "S", "W"]):
            st = self.states[i]
            icon = "🟢" if st == "green" else ("🟡" if st == "yellow" else "🔴")
            parts.append(f"{name}:{icon}")
        return f"[ {' | '.join(parts)} ]"

    @property
    def state(self) -> str:
        """Compatibility/Simple state property."""
        for i, s in enumerate(self.states):
            if s == "green": return f"Lane {i}"
        return "ALL RED"

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
        changed = self.states[green_lane] != "green"
        for i in range(4):
            if i == green_lane:
                self.set_signal(i, "green")
            else:
                self.set_signal(i, "red")
        if changed:
            print(f"\n[Signal] 🚥 Global State Update: {self.status_bar}")

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

# Normalized ROIs (0.0 - 1.0) — auto-scaled to any resolution
_LANE_ROIS_NORM = [
    [(0.21, 0.0),  (0.79, 0.0),  (0.79, 0.37), (0.21, 0.37)],  # North (top)
    [(0.79, 0.19), (1.0, 0.19),  (1.0, 0.81),  (0.79, 0.81)],  # East  (right)
    [(0.21, 0.63), (0.79, 0.63), (0.79, 1.0),  (0.21, 1.0)],   # South (bottom)
    [(0.0, 0.19),  (0.21, 0.19), (0.21, 0.81), (0.0, 0.81)],   # West  (left)
]

def get_lane_rois(w: int = 320, h: int = 240) -> List[List[Tuple[int, int]]]:
    """Return LANE_ROIS scaled to the given frame dimensions."""
    return [
        [(int(x * w), int(y * h)) for x, y in roi]
        for roi in _LANE_ROIS_NORM
    ]

# Default for backward compatibility
LANE_ROIS: List[List[Tuple[int, int]]] = get_lane_rois(320, 240)

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

    def get_decision(self, tracks: List[dict], frame_w: int = 320, frame_h: int = 240) -> Tuple[int, str]:
        """
        Determines which lane should be green based on current frame data.
        Returns: (lane_index, reason)
        """
        now = time.time()
        densities = self._get_densities(tracks, frame_w, frame_h)
        emergency_lane = self._get_emergency_lane(tracks, frame_w, frame_h)
        elapsed = now - self.state.last_switch_time

        # 1. Ambulance Priority (Highest)
        if emergency_lane is not None:
            if self.state.current_lane != emergency_lane:
                self.state.current_lane = emergency_lane
                self.state.last_switch_time = now
            self.state.override_reason = "AMBULANCE"
            self.state.remaining_time = self.green_time_default
            
            # --- MASTER LOCK (v8.0) ---
            # Force the controller state so its internal timer staying synchronized
            self.state.current_lane = emergency_lane
            self.state.last_switch_time = time.time() # Reset timer to prevent rapid flip
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

    def _get_densities(self, tracks: List[dict], frame_w: int = 320, frame_h: int = 240) -> List[int]:
        densities = [0] * 4
        rois = get_lane_rois(frame_w, frame_h)
        for track in tracks:
            cx, cy = track.get("cx", -1), track.get("cy", -1)
            for i, roi in enumerate(rois):
                if self._point_in_roi(cx, cy, roi):
                    densities[i] += 1
                    break
        return densities

    def _get_emergency_lane(self, tracks: List[dict], frame_w: int = 320, frame_h: int = 240) -> Optional[int]:
        rois = get_lane_rois(frame_w, frame_h)
        for track in tracks:
            label = str(track.get("label", "")).lower()
            if label in EMERGENCY_LABELS:
                cx, cy = track.get("cx", -1), track.get("cy", -1)
                for i, roi in enumerate(rois):
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

def draw_rois(frame, lane_idx: int, densities: List[int], reason: str, remaining: float, focus_lane: int = None):
    import cv2
    h, w = frame.shape[:2]
    
    # Grid System Removed (v8.0)
    # We only draw a status board and detection boxes.
    
    overlay = frame.copy()
    
    # 1. Clean Status Display (Bottom)
    status_msg = f"LANE: {LANE_NAMES[lane_idx]} ({reason}) | COUNT: {densities[lane_idx]} | {remaining:.1f}s"
    color = (0, 255, 0)
    if "AMBULANCE" in reason: color = (0, 0, 255)
    
    cv2.rectangle(frame, (0, h-60), (w, h), (30, 30, 30), -1) # Background
    cv2.putText(frame, status_msg, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 2. Focus Mode Indicator
    if focus_lane is not None:
        cv2.putText(frame, "LOCKED ON: " + LANE_NAMES[focus_lane], (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
