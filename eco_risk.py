"""
EcoFlow AI — Eco Risk Assessor
==============================
Assesses urban pollution and green-cover balance using:

  • estimate_vegetation_coverage(frame)
        HSV green-mask analysis on the camera frame, with an optional
        fallback to YOLO's 'tree' / 'potted plant' class detections.

  • compute_pollution_index(tracks)
        Weighted vehicle count:  car = 1 pt, truck = 3 pts, bus = 5 pts.

  • eco_status_check(pollution_index, veg_pct, ...)
        Compares the two metrics, returns an EcoStatus result, prints a
        "CRITICAL ACTION REQUIRED" alert when triggered, and appends a
        row to an eco_log.csv for urban-planning analytics.

  • draw_eco_overlay(frame, status)
        Renders a side-bar "Pollution vs. Nature" dual progress bar
        with colour-coded risk badges directly on a BGR frame.

Integration with tracker.py / signal_controller.py
----------------------------------------------------
tracks format expected by this module:
    {"track_id": int, "label": str, "cx": float, "cy": float}

Labels are matched case-insensitively.  The module is standalone; OpenCV
and NumPy are the only hard dependencies (YOLO tree detection is optional).
"""

from __future__ import annotations

import csv
import datetime
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ──────────────────────────────────────────────────────────────────────────────

# ── Vegetation HSV thresholds ─────────────────────────────────────────────────
# Covers lush grass, tree canopy, shrubs; excludes yellow-green road markings.
VEG_LOWER_1 = np.array([35,  50,  40], dtype=np.uint8)   # spring / light green
VEG_UPPER_1 = np.array([85, 255, 255], dtype=np.uint8)
VEG_LOWER_2 = np.array([86,  40,  30], dtype=np.uint8)   # dark forest green
VEG_UPPER_2 = np.array([100, 255, 200], dtype=np.uint8)

# ── Pollution scoring weights ─────────────────────────────────────────────────
POLLUTION_WEIGHTS: Dict[str, float] = {
    "car":        1.0,
    "motorcycle": 0.5,
    "bus":        5.0,
    "truck":      3.0,
}

# ── Risk thresholds ───────────────────────────────────────────────────────────
CRITICAL_VEG_THRESHOLD    = 15.0   # % canopy below which risk escalates
WARNING_VEG_THRESHOLD     = 30.0   # moderate warning band

CRITICAL_POLLUTION_INDEX  = 20     # above this → HIGH pollution
WARNING_POLLUTION_INDEX   = 10     # moderate warning band

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(os.path.dirname(__file__), "eco_log.csv")
LOG_FIELDS = [
    "timestamp", "frame_idx", "vegetation_pct",
    "pollution_index", "risk_level", "alert",
]

# ── Overlay geometry ──────────────────────────────────────────────────────────
OVERLAY_WIDTH  = 200   # pixels wide
OVERLAY_MARGIN = 14    # gap from right edge
BAR_WIDTH      = 28    # thickness of each progress bar
BAR_HEIGHT     = 220   # max bar height in pixels

# ── YOLO tree detection (optional) ───────────────────────────────────────────
YOLO_TREE_MODEL_PATH: Optional[str] = None   # set e.g. "yolo11n.pt" to enable
YOLO_TREE_CLASSES: List[str] = ["tree", "potted plant", "plant"]  # label names


# ──────────────────────────────────────────────────────────────────────────────
# Risk level enum
# ──────────────────────────────────────────────────────────────────────────────

class RiskLevel:
    SAFE     = "SAFE"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EcoStatus:
    """Snapshot of one eco-risk assessment cycle."""
    vegetation_pct:    float
    pollution_index:   float
    risk_level:        str
    alert:             bool
    frame_idx:         int         = 0
    timestamp:         str         = field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )

    # ── Derived convenience ───────────────────────────────────────────────────
    @property
    def veg_bar_fraction(self) -> float:
        """0–1, clamped. Higher = more vegetation."""
        return min(1.0, max(0.0, self.vegetation_pct / 100.0))

    @property
    def pollution_bar_fraction(self) -> float:
        """0–1, clamped. Higher = worse pollution."""
        return min(1.0, max(0.0, self.pollution_index / max(CRITICAL_POLLUTION_INDEX * 1.5, 1)))

    def __str__(self) -> str:
        alert_str = " ⚠ CRITICAL ACTION REQUIRED" if self.alert else ""
        return (
            f"[EcoRisk] {self.timestamp} | "
            f"Veg: {self.vegetation_pct:.1f}%  "
            f"Pollution index: {self.pollution_index:.1f}  "
            f"Risk: {self.risk_level}{alert_str}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Vegetation coverage estimator
# ──────────────────────────────────────────────────────────────────────────────

def estimate_vegetation_coverage(
    frame: np.ndarray,
    use_yolo: bool = False,
    yolo_model=None,          # ultralytics YOLO instance (optional)
) -> float:
    """
    Estimate the percentage of tree / vegetation canopy visible in *frame*.

    Strategy
    --------
    1. **HSV colour mask** (always applied):
       Two green hue bands cover spring green through dark forest.
       The combined mask is morphologically cleaned and the fraction of
       non-zero pixels is returned.

    2. **YOLO tree class** (optional, only if ``use_yolo=True`` and a
       ``yolo_model`` is supplied):
       Detections whose class label is in ``YOLO_TREE_CLASSES`` have their
       bounding-box areas summed.  The union of HSV mask and bbox pixels
       forms the final coverage estimate.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (from cv2.VideoCapture).
    use_yolo : bool
        Enable YOLO-based tree detection as a supplement.
    yolo_model : ultralytics.YOLO, optional
        A pre-loaded YOLO model.  If None and ``use_yolo`` is True the
        function falls back to HSV-only.

    Returns
    -------
    float
        Vegetation coverage as a percentage (0.0 – 100.0).

    Examples
    --------
    >>> import numpy as np
    >>> green_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> green_frame[:, :] = (34, 180, 80)   # BGR  ≈ olive green
    >>> pct = estimate_vegetation_coverage(green_frame)
    >>> pct > 50
    True
    """
    h, w = frame.shape[:2]
    total_pixels = h * w

    # ── HSV colour mask ───────────────────────────────────────────────────────
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, VEG_LOWER_1, VEG_UPPER_1) |
        cv2.inRange(hsv, VEG_LOWER_2, VEG_UPPER_2)
    )

    # Morphological clean-up: remove tiny noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Optional YOLO supplement ──────────────────────────────────────────────
    if use_yolo and yolo_model is not None:
        try:
            results = yolo_model.predict(frame, verbose=False)
            if results and results[0].boxes is not None:
                boxes  = results[0].boxes
                names  = results[0].names          # {class_id: name}
                clsids = boxes.cls.int().cpu().tolist()
                xyxys  = boxes.xyxy.cpu().tolist()
                for cls_id, xyxy in zip(clsids, xyxys):
                    if names.get(cls_id, "").lower() in YOLO_TREE_CLASSES:
                        x1, y1, x2, y2 = (int(v) for v in xyxy)
                        # Paint the bounding-box area white in the mask
                        mask[y1:y2, x1:x2] = 255
        except Exception:
            pass   # gracefully degrade if model call fails

    green_pixels = int(np.count_nonzero(mask))
    return round(green_pixels / max(total_pixels, 1) * 100.0, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pollution index
# ──────────────────────────────────────────────────────────────────────────────

def compute_pollution_index(tracks: List[Dict]) -> float:
    """
    Calculate a weighted pollution score from active vehicle tracks.

    Weights
    -------
    car / sedan  →  1 pt
    motorcycle   →  0.5 pt
    truck        →  3 pts
    bus          →  5 pts
    (all others) →  1 pt  (default)

    Parameters
    ----------
    tracks : list of dicts
        Each dict must contain at least ``"label"`` (str).

    Returns
    -------
    float
        Cumulative pollution index for the current frame / sample.

    Example
    -------
    >>> tracks = [
    ...     {"label": "car"},
    ...     {"label": "truck"},
    ...     {"label": "bus"},
    ... ]
    >>> compute_pollution_index(tracks)
    9.0
    """
    score = 0.0
    for track in tracks:
        label = str(track.get("label", "")).strip().lower()
        # Ambulances / fire trucks are vehicles too – default weight 1
        score += POLLUTION_WEIGHTS.get(label, 1.0)
    return round(score, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 3. CSV logger  (internal)
# ──────────────────────────────────────────────────────────────────────────────

def _log_eco_status(status: EcoStatus) -> None:
    """Append one row to eco_log.csv (creates file + header if needed)."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp":       status.timestamp,
            "frame_idx":       status.frame_idx,
            "vegetation_pct":  f"{status.vegetation_pct:.2f}",
            "pollution_index": f"{status.pollution_index:.1f}",
            "risk_level":      status.risk_level,
            "alert":           int(status.alert),
        })


# ──────────────────────────────────────────────────────────────────────────────
# 4. Eco status check
# ──────────────────────────────────────────────────────────────────────────────

def eco_status_check(
    pollution_index: float,
    vegetation_pct:  float,
    frame_idx:       int  = 0,
    log:             bool = True,
    verbose:         bool = True,
) -> EcoStatus:
    """
    Compare pollution load against vegetation coverage and return an
    :class:`EcoStatus` with an appropriate risk level.

    Risk Matrix
    -----------
    +-------------------------+-------------------+------------------+
    | Condition               | Vegetation        | Risk             |
    +=========================+===================+==================+
    | High pollution + low veg| < 15 %            | **CRITICAL**     |
    | High pollution          | 15–30 %           | HIGH             |
    | Moderate pollution      | < 30 %            | MODERATE         |
    | Otherwise               | ≥ 30 %            | SAFE             |
    +-------------------------+-------------------+------------------+

    A ``CRITICAL`` status additionally:
      • Prints a prominent "CRITICAL ACTION REQUIRED" banner to stdout.
      • Appends a row to ``eco_log.csv`` (unless ``log=False``).

    Parameters
    ----------
    pollution_index : float
        Returned by :func:`compute_pollution_index`.
    vegetation_pct : float
        Returned by :func:`estimate_vegetation_coverage`.
    frame_idx : int
        Current frame number (for logging).
    log : bool
        Write to CSV log when status is CRITICAL or HIGH (default True).
    verbose : bool
        Print all status messages (default True).

    Returns
    -------
    EcoStatus
    """
    high_pollution  = pollution_index >= CRITICAL_POLLUTION_INDEX
    mod_pollution   = pollution_index >= WARNING_POLLUTION_INDEX
    low_veg         = vegetation_pct  <  CRITICAL_VEG_THRESHOLD
    moderate_veg    = vegetation_pct  <  WARNING_VEG_THRESHOLD

    # ── Risk classification ────────────────────────────────────────────────
    if high_pollution and low_veg:
        risk  = RiskLevel.CRITICAL
        alert = True
    elif high_pollution and moderate_veg:
        risk  = RiskLevel.HIGH
        alert = False
    elif mod_pollution and low_veg:
        risk  = RiskLevel.HIGH
        alert = False
    elif mod_pollution or moderate_veg:
        risk  = RiskLevel.MODERATE
        alert = False
    else:
        risk  = RiskLevel.SAFE
        alert = False

    status = EcoStatus(
        vegetation_pct=vegetation_pct,
        pollution_index=pollution_index,
        risk_level=risk,
        alert=alert,
        frame_idx=frame_idx,
    )

    # ── Terminal output ────────────────────────────────────────────────────
    if verbose:
        if alert:
            banner = (
                "\n" + "!" * 65 + "\n"
                "!  🌍  CRITICAL ACTION REQUIRED  🌍"
                "                            !\n"
                f"!  Pollution Index : {pollution_index:<6.1f}  "
                f"Vegetation : {vegetation_pct:.1f}%"
                f"{'':>18}!\n"
                "!  Immediate urban-planning intervention recommended.      !\n"
                "!" * 65 + "\n"
            )
            print(banner)
        else:
            print(status)

    # ── Logging ────────────────────────────────────────────────────────────
    if log and risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        _log_eco_status(status)

    return status


# ──────────────────────────────────────────────────────────────────────────────
# 5. Visual overlay
# ──────────────────────────────────────────────────────────────────────────────

# Risk-level colours (BGR)
_RISK_COLORS = {
    RiskLevel.SAFE:     (60,  200,  60),    # green
    RiskLevel.MODERATE: (0,   180, 220),    # amber
    RiskLevel.HIGH:     (0,   100, 240),    # orange
    RiskLevel.CRITICAL: (30,   30, 210),    # red
}

_VEG_COLOR    = (50, 200, 80)     # green bar
_POLL_COLOR   = (40, 100, 230)    # orange-red bar
_TRACK_COLOR  = (40,  40,  40)    # dark track background
_TEXT_COLOR   = (230, 230, 230)


def draw_eco_overlay(
    frame: np.ndarray,
    status: EcoStatus,
    alpha: float = 0.72,
) -> None:
    """
    Draw a "Pollution vs. Nature" vertical dual-bar overlay on the
    right edge of *frame* (in-place).

    Layout (right side, top-aligned)
    ---------------------------------
    ┌──────────────────────┐
    │  EcoFlow Risk        │  ← header
    │  ██ Nature  45 %     │  ← green vegetation bar (fills upward)
    │  ██ Pollut  18 pts   │  ← orange pollution bar
    │  [ SAFE ]            │  ← risk badge
    └──────────────────────┘

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate (modified in-place).
    status : EcoStatus
        Result from :func:`eco_status_check`.
    alpha : float
        Panel opacity (0 = fully transparent, 1 = fully opaque).
    """
    h, w = frame.shape[:2]

    # ── Panel geometry ────────────────────────────────────────────────────
    panel_w  = OVERLAY_WIDTH
    panel_h  = BAR_HEIGHT + 120         # bars + header + badge
    px       = w - panel_w - OVERLAY_MARGIN
    py       = 10
    px2, py2 = px + panel_w, py + panel_h

    # ── Semi-transparent background panel ─────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px2, py2), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Helper: draw one vertical bar ─────────────────────────────────────
    def _bar(
        label: str,
        fraction: float,
        bar_color: Tuple,
        value_str: str,
        col_x: int,
        base_y: int,
    ):
        """Draw a single labelled vertical progress bar."""
        track_top  = base_y
        track_bot  = base_y + BAR_HEIGHT
        bx1        = col_x
        bx2        = col_x + BAR_WIDTH
        filled_top = track_bot - int(fraction * BAR_HEIGHT)

        # Track (dark background)
        cv2.rectangle(frame, (bx1, track_top), (bx2, track_bot),
                      _TRACK_COLOR, cv2.FILLED)

        # Gradient-ish fill (two-tone)
        if filled_top < track_bot:
            mid = (filled_top + track_bot) // 2
            dark_color = tuple(max(0, c - 40) for c in bar_color)
            cv2.rectangle(frame, (bx1, filled_top), (bx2, mid),
                          bar_color, cv2.FILLED)
            cv2.rectangle(frame, (bx1, mid), (bx2, track_bot),
                          dark_color, cv2.FILLED)

        # Border
        cv2.rectangle(frame, (bx1, track_top), (bx2, track_bot),
                      (80, 80, 80), 1)

        # Percentage line
        cv2.line(frame,
                 (bx1, filled_top), (bx2, filled_top),
                 (255, 255, 255), 1)

        # Label below bar
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        cv2.putText(frame, label,
                    (bx1, track_bot + 14),
                    font, scale, _TEXT_COLOR, thick, cv2.LINE_AA)
        cv2.putText(frame, value_str,
                    (bx1, track_bot + 28),
                    font, scale, _TEXT_COLOR, thick, cv2.LINE_AA)

    # ── Header ────────────────────────────────────────────────────────────
    cv2.putText(frame, "Pollution vs Nature",
                (px + 6, py + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.line(frame, (px + 4, py + 22), (px2 - 4, py + 22),
             (80, 80, 80), 1)

    base_y   = py + 30
    col_veg  = px + 22
    col_poll = px + panel_w - BAR_WIDTH - 22

    # ── Vegetation bar (green, high = good) ───────────────────────────────
    _bar(
        label     = "Nature",
        fraction  = status.veg_bar_fraction,
        bar_color = _VEG_COLOR,
        value_str = f"{status.vegetation_pct:.1f}%",
        col_x     = col_veg,
        base_y    = base_y,
    )

    # ── Pollution bar (orange, high = bad) ────────────────────────────────
    poll_color = _RISK_COLORS.get(status.risk_level, _POLL_COLOR)
    _bar(
        label     = "Pollution",
        fraction  = status.pollution_bar_fraction,
        bar_color = poll_color,
        value_str = f"{status.pollution_index:.0f} pts",
        col_x     = col_poll,
        base_y    = base_y,
    )

    # ── Risk badge ────────────────────────────────────────────────────────
    badge_y   = py + panel_h - 26
    badge_col = _RISK_COLORS.get(status.risk_level, (80, 80, 80))
    cv2.rectangle(frame,
                  (px + 10, badge_y - 16),
                  (px2 - 10, badge_y + 6),
                  badge_col, cv2.FILLED)
    risk_text = ("⚠ CRITICAL" if status.alert else status.risk_level)
    cv2.putText(frame, risk_text,
                (px + 16, badge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: full-pipeline single call
# ──────────────────────────────────────────────────────────────────────────────

def assess_eco_risk(
    frame: np.ndarray,
    tracks: List[Dict],
    frame_idx: int        = 0,
    draw_overlay: bool    = True,
    log: bool             = True,
    verbose: bool         = True,
    use_yolo: bool        = False,
    yolo_model            = None,
) -> EcoStatus:
    """
    One-call pipeline: frame + tracks → eco risk assessment + optional overlay.

    Calls, in order:
      1. ``estimate_vegetation_coverage(frame)``
      2. ``compute_pollution_index(tracks)``
      3. ``eco_status_check(...)``
      4. ``draw_eco_overlay(frame, status)``  (if ``draw_overlay=True``)

    Parameters
    ----------
    frame : np.ndarray
        Current BGR camera frame.
    tracks : list of dicts
        Active vehicle tracks from the YOLO tracker.
    frame_idx : int
        Frame counter (used in logs).
    draw_overlay : bool
        If True, annotate *frame* in-place with the side-bar overlay.
    log, verbose, use_yolo, yolo_model
        Forwarded to the sub-functions.

    Returns
    -------
    EcoStatus
    """
    veg_pct  = estimate_vegetation_coverage(frame, use_yolo=use_yolo,
                                             yolo_model=yolo_model)
    poll_idx = compute_pollution_index(tracks)
    status   = eco_status_check(poll_idx, veg_pct,
                                  frame_idx=frame_idx,
                                  log=log, verbose=verbose)
    if draw_overlay:
        draw_eco_overlay(frame, status)
    return status


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test (run as script)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  EcoFlow AI — Eco Risk Assessor  |  self-test")
    print("=" * 60)

    # ── Test 1: healthy image, moderate traffic ───────────────────────────
    print("\n[Test 1] Healthy green scene, light traffic")
    green_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    green_frame[100:900, 100:1800] = (34, 160, 60)   # BGR forest green
    tracks_1 = [
        {"track_id": 1, "label": "car"},
        {"track_id": 2, "label": "car"},
        {"track_id": 3, "label": "motorcycle"},
    ]
    s1 = assess_eco_risk(green_frame, tracks_1, frame_idx=1,
                          draw_overlay=False, log=False)
    assert s1.risk_level in (RiskLevel.SAFE, RiskLevel.MODERATE), s1
    print(f"  ✓ Risk={s1.risk_level}  Veg={s1.vegetation_pct:.1f}%  Poll={s1.pollution_index}")

    # ── Test 2: barren frame + heavy traffic → CRITICAL ───────────────────
    print("\n[Test 2] Barren concrete scene, heavy traffic → CRITICAL expected")
    grey_frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    tracks_2 = [
        {"track_id": i, "label": "bus"}   for i in range(3)
    ] + [
        {"track_id": i + 10, "label": "truck"} for i in range(4)
    ]
    s2 = assess_eco_risk(grey_frame, tracks_2, frame_idx=2,
                          draw_overlay=False, log=False)
    assert s2.alert, f"Expected CRITICAL, got {s2.risk_level}"
    print(f"  ✓ Risk={s2.risk_level}  Veg={s2.vegetation_pct:.1f}%  Poll={s2.pollution_index}")

    # ── Test 3: medium scene ──────────────────────────────────────────────
    print("\n[Test 3] Partial vegetation, moderate traffic → HIGH / MODERATE")
    mixed_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mixed_frame[0:400, 0:1920] = (34, 160, 60)        # top 37 % green
    tracks_3 = [
        {"track_id": 1, "label": "truck"},
        {"track_id": 2, "label": "bus"},
        {"track_id": 3, "label": "car"},
        {"track_id": 4, "label": "car"},
    ]
    s3 = assess_eco_risk(mixed_frame, tracks_3, frame_idx=3,
                          draw_overlay=False, log=False)
    print(f"  ✓ Risk={s3.risk_level}  Veg={s3.vegetation_pct:.1f}%  Poll={s3.pollution_index}")

    # ── Test 4: overlay draw (no assertion, just check no exception) ──────
    print("\n[Test 4] Overlay draw on a frame (visual check skipped in CI)")
    sample_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
    status_4 = EcoStatus(vegetation_pct=12.5, pollution_index=27.0,
                          risk_level=RiskLevel.CRITICAL, alert=True, frame_idx=4)
    draw_eco_overlay(sample_frame, status_4)
    print(f"  ✓ Overlay drawn without exception ({sample_frame.shape})")

    print("\n" + "=" * 60)
    print("  All tests passed ✅")
    print(f"  CSV log location: {LOG_FILE}")
    print("=" * 60)
