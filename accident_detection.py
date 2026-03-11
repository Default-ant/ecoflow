"""
EcoFlow AI — Accident Detection Module
=======================================
Detects potential traffic accidents or stalled vehicles by analyzing 
trajectory history (velocity, duration of stop, and proximity).

Heuristics:
1.  SUDDEN STOP: Velocity drops from 'High' to 0 abruptly.
2.  PROLONGED STOP: Vehicle is stationary for > N seconds in a flow-zone.
3.  STALLED: Vehicle has 0 movement while in the middle of a lane ROI.
"""

import time
import collections
from typing import Dict, List, Optional, Tuple

# ── Tunable constants ────────────────────────────────────────────────────────
VELOCITY_WINDOW   = 10      # frames to calculate moving average velocity
STOP_THRESHOLD    = 1.5     # pixels per frame considered "stopped"
STALL_TIME_LIMIT  = 5.0     # seconds stationary before triggering alert
SUDDEN_DROP_PCT   = 0.70    # 70% drop in velocity within 5 frames = sudden stop
MAX_HISTORY_LEN   = 60      # 2 seconds at 30fps

@dataclass
class TrackPoint:
    x: float
    y: float
    timestamp: float

class AccidentDetector:
    """
    Maintains track history and analyzes motion for anomalies.
    """
    def __init__(self):
        # track_id -> deque of TrackPoint
        self.history: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=MAX_HISTORY_LEN)
        )
        # track_id -> timestamp when first stopped
        self.stop_start_times: Dict[int, float] = {}
        # track_id -> whether an accident is currently flagged
        self.confirmed_accidents: set[int] = set()

    def update(self, tracks: List[dict]) -> set[int]:
        """
        Update history with current tracks and return set of accident-flagged track IDs.
        
        'tracks' format: [{"track_id": int, "cx": float, "cy": float, ...}]
        """
        now = time.time()
        current_ids = {t["track_id"] for t in tracks}
        
        # Cleanup old data for lost tracks
        lost_ids = set(self.history.keys()) - current_ids
        for lid in lost_ids:
            # We keep confirmed accidents for a bit unless they are gone for seconds
            # For simplicity, just pop them
            self.history.pop(lid, None)
            self.stop_start_times.pop(lid, None)
            self.confirmed_accidents.discard(lid)

        new_alerts = set()

        for t in tracks:
            tid = t["track_id"]
            cx, cy = t["cx"], t["cy"]
            
            # 1. Update history
            self.history[tid].append(TrackPoint(cx, cy, now))
            
            # 2. Analyze Motion
            if len(self.history[tid]) < 5:
                continue
                
            # Calculate instantaneous velocity (px/sec)
            hist = self.history[tid]
            p_curr = hist[-1]
            p_prev = hist[-2]
            dt = p_curr.timestamp - p_prev.timestamp
            dist = ((p_curr.x - p_prev.x)**2 + (p_curr.y - p_prev.y)**2)**0.5
            velocity = dist / dt if dt > 0 else 0
            
            # 3. Detect "Stalled" (Prolonged stop)
            if velocity < STOP_THRESHOLD:
                if tid not in self.stop_start_times:
                    self.stop_start_times[tid] = now
                elif now - self.stop_start_times[tid] > STALL_TIME_LIMIT:
                    self.confirmed_accidents.add(tid)
                    new_alerts.add(tid)
            else:
                # Reset stop timer if moving
                self.stop_start_times.pop(tid, None)
                # Keep accident flag for a few frames even if slight movement
                if velocity > STOP_THRESHOLD * 3:
                     self.confirmed_accidents.discard(tid)

            # 4. Sudden Stop Detection (Advanced)
            if len(hist) >= 15:
                # Average velocity 1 second ago vs now
                # Not implemented in v1 for stability, but placeholder for refinement
                pass

        return self.confirmed_accidents

# Helper for integration
from dataclasses import dataclass
