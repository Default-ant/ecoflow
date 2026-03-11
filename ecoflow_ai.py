"""
EcoFlow AI — Main Brain  (Raspberry Pi 5)
==========================================
The top-level orchestrator.  Run this file to start the system.

What it does
─────────────
1. Opens the IP Webcam live stream from your phone.
2. Runs YOLOv11n vehicle tracking on every frame.
3. Delegates ambulance detection  →  ambulance_detection.py
4. Delegates traffic-light GPIO   →  signal_controller.py
5. Delegates eco risk assessment  →  eco_risk.py
6. Prints status and optionally shows a preview window.

Usage (on the Pi over SSH)
──────────────────────────
  python ecoflow_ai.py --url http://192.168.x.x:8080/video --no-preview

Optional flags
──────────────
  --url          IP Webcam stream  (required)
  --no-preview   Headless: skip cv2.imshow  (use when running over SSH)
  --no-gpio      Disable GPIO (desktop / CI testing)
  --conf         YOLO detection confidence threshold   (default 0.35)
  --width        Resize frame width before inference   (default 640)
  --height       Resize frame height before inference  (default 360)
  --green-hold   Seconds to hold GREEN after ambulance clears (default 10)
  --eco-every    Run eco-risk check every N frames      (default 30)

Install on RPi 5
───────────────
  pip install ultralytics opencv-python-headless gpiozero lgpio
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── EcoFlow AI modules ────────────────────────────────────────────────────────
from ambulance_detection import AmbulanceState, check_track, build_tracks_list
from signal_controller   import TrafficLight, VEHICLE_CLASSES, get_signal_phase
from eco_risk            import assess_eco_risk, draw_eco_overlay
from accident_detection  import AccidentDetector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH          = "models/yolo11n.onnx"   # TFLite — Best for RPi5
VEHICLE_CLASS_IDS   = list(VEHICLE_CLASSES.keys())   # [2, 3, 5, 7]
GREEN_HOLD_DEFAULT  = 10.0   # seconds to keep green after ambulance last seen


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOR_VEHICLE   = (0, 200, 0)    # BGR green  – normal vehicle box
COLOR_AMBULANCE = (0, 50, 255)   # BGR red    – ambulance box


def _draw_boxes(frame: np.ndarray,
                ids: list, xyxy: list, clsids: list, confs: list,
                confirmed: set[int]) -> None:
    """Draw bounding boxes + labels directly on *frame* (in-place)."""
    for track_id, box, cls_id, conf in zip(ids, xyxy, clsids, confs):
        x1, y1, x2, y2 = (int(v) for v in box)
        is_amb = track_id in confirmed
        color  = COLOR_AMBULANCE if is_amb else COLOR_VEHICLE
        label  = (f"AMBULANCE#{track_id}" if is_amb
                  else f"{VEHICLE_CLASSES.get(cls_id,'veh')}#{track_id}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_status_bar(frame: np.ndarray, light_state: str,
                     n_ambulances: int, eco_risk: str | None, frame_idx: int) -> None:
    """Render a thin status banner at the top of the frame."""
    h, w = frame.shape[:2]
    banner_h = 36
    is_emergency = (light_state == "green" and n_ambulances > 0)
    banner_color = (0, 80, 200) if is_emergency else (20, 20, 20)
    cv2.rectangle(frame, (0, 0), (w, banner_h), banner_color, -1)

    light_emoji = {"red": "🔴", "yellow": "🟡", "green": "🟢"}.get(light_state, "⬜")
    text = (f"  {light_emoji} {light_state.upper()}"
            f"   |  Ambulances: {n_ambulances}"
            f"   |  Eco: {eco_risk or '—'}"
            f"   |  Frame: {frame_idx}")
    cv2.putText(frame, text, (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[EcoFlow] Loading  : {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")
    print("[EcoFlow] Model ready.\n")

    # ── Stream ────────────────────────────────────────────────────────────────
    print(f"[EcoFlow] Connecting: {args.url}")
    cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        sys.exit(
            f"\n[ERROR] Cannot open stream: {args.url}\n"
            "  • Make sure 'IP Webcam' app is running on your phone.\n"
            "  • Try:  http://<phone-ip>:8080/video\n"
            "  • Or:   rtsp://<phone-ip>:8080/h264_ulaw.sdp\n"
        )
    print("[EcoFlow] Stream open.\n")

    # ── GPIO Traffic Light ────────────────────────────────────────────────────
    light = TrafficLight(no_gpio=args.no_gpio)

    # ── Per-session state ─────────────────────────────────────────────────────
    amb_state           = AmbulanceState()
    accident_detector   = AccidentDetector()
    last_ambulance_time = 0.0
    last_eco_risk_label = None
    frame_idx           = 0

    print("[EcoFlow] Live detection started — Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN]  Frame grab failed — retrying …")
                time.sleep(0.5)
                continue

            frame_idx += 1

            # Resize to inference size (reduces CPU load on the Pi)
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            # ── 1. YOLO tracking ──────────────────────────────────────────────
            results = model.track(
                frame,
                tracker   = "bytetrack.yaml",
                persist   = True,
                imgsz     = 640,
                classes   = VEHICLE_CLASS_IDS,
                conf      = args.conf,
                verbose   = False,
            )

            ambulance_in_frame = False
            tracks             = []

            if results and results[0].boxes is not None:
                boxes  = results[0].boxes
                ids    = (boxes.id.int().cpu().tolist()
                          if boxes.id is not None else [])
                xyxy   = boxes.xyxy.cpu().tolist()
                clsids = boxes.cls.int().cpu().tolist()
                confs  = boxes.conf.cpu().tolist()

                # ── 2. Ambulance detection (per track) ────────────────────────
                for track_id, box, cls_id in zip(ids, xyxy, clsids):
                    x1, y1, x2, y2 = (int(v) for v in box)
                    newly_confirmed = check_track(
                        frame, track_id, cls_id,
                        x1, y1, x2, y2, amb_state)
                    if newly_confirmed:
                        print(f"\n[EcoFlow] 🚑 Ambulance confirmed:"
                              f" ID={track_id}  frame={frame_idx}")

                if amb_state.confirmed & set(ids):   # any confirmed amb visible now
                    ambulance_in_frame = True

                # Build tracks list for signal + eco modules
                tracks = build_tracks_list(ids, xyxy, clsids, confs,
                                           amb_state.confirmed)

            # ── 3. Accident Detection ────────────────────────────────────────
            accidents = accident_detector.update(tracks)
            if accidents:
                print(f"\n[EcoFlow] 🚨 ACCIDENT DETECTED: IDs={list(accidents)}")

            # ── 3. Traffic-light GPIO control ────────────────────────────────
            now = time.time()
            if ambulance_in_frame:
                last_ambulance_time = now
                if light.state != "green":
                    light.set_green()
            elif now - last_ambulance_time < args.green_hold:
                pass   # hold green; ambulance just left frame
            else:
                if light.state != "red":
                    light.set_red()

            # ── 4. Eco risk assessment (every N frames) ───────────────────────
            if frame_idx % args.eco_every == 0 and tracks:
                eco_status = assess_eco_risk(
                    frame, tracks, frame_idx=frame_idx,
                    draw_overlay=not args.no_preview,
                    log=True, verbose=False)
                last_eco_risk_label = eco_status.risk_level
                if eco_status.alert:
                    print(f"\n[EcoFlow] ⚠  ECO CRITICAL — "
                          f"Pollution={eco_status.pollution_index:.1f}  "
                          f"Veg={eco_status.vegetation_pct:.1f}%")

            # ── 5. Optional preview ───────────────────────────────────────────
            if not args.no_preview:
                if results and results[0].boxes is not None:
                    _draw_boxes(frame, ids, xyxy, clsids, confs,
                                amb_state.confirmed)
                
                # Draw accident overlay
                for acc_id in accidents:
                    # Find box for this ID
                    if accidents and results and results[0].boxes is not None:
                         # Simple text for now, could be improved
                         cv2.putText(frame, "!!! ACCIDENT !!!", (10, 100), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                _draw_status_bar(frame, light.state,
                                 len(amb_state.confirmed),
                                 last_eco_risk_label, frame_idx)
                try:
                    cv2.imshow("EcoFlow AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[EcoFlow] 'q' pressed — exiting.")
                        break
                except Exception:
                    pass   # headless fallback

            # Progress heartbeat
            if frame_idx % 60 == 0:
                print(f"\r[EcoFlow] frame={frame_idx:6d}  "
                      f"ambulances={len(amb_state.confirmed)}  "
                      f"light={light.state}  "
                      f"eco={last_eco_risk_label or '—'}",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n[EcoFlow] Interrupted.")
    finally:
        print("\n[EcoFlow] Shutting down …")
        cap.release()
        light.set_red()
        light.cleanup()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[EcoFlow] Stopped cleanly. Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EcoFlow AI — Raspberry Pi 5 live traffic monitor")
    p.add_argument("--url", default="http://172.20.38.70:8080/video",
                   help="IP Webcam stream URL (default: http://172.20.38.70:8080/video)")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--conf",   type=float, default=0.35,
                   help="YOLO detection confidence (default 0.35)")
    p.add_argument("--green-hold", type=float, default=GREEN_HOLD_DEFAULT,
                   dest="green_hold",
                   help=f"Seconds to hold green after ambulance clears "
                        f"(default {GREEN_HOLD_DEFAULT})")
    p.add_argument("--eco-every", type=int, default=30,
                   dest="eco_every",
                   help="Run eco-risk check every N frames (default 30)")
    p.add_argument("--no-gpio",    action="store_true",
                   help="Disable GPIO (for desktop testing)")
    p.add_argument("--no-preview", action="store_true",
                   help="Disable cv2.imshow (headless SSH mode)")
    return p.parse_args()


if __name__ == "__main__":
    run(_args())
