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
  --url          IP Webcam stream URL
  --cam          Physical webcam index (e.g. 0)
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
import os

# Signal to libraries (OpenCV/Qt) that we are running headless
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
from ultralytics import YOLO

# ── EcoFlow AI modules ────────────────────────────────────────────────────────
from ambulance_detection import AmbulanceState, check_track, build_tracks_list
from signal_controller   import TrafficLight, VEHICLE_CLASSES, AdaptiveController, EMERGENCY_LABELS, LANE_NAMES
from eco_risk            import assess_eco_risk, draw_eco_overlay
from accident_detection  import AccidentDetector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH          = "models/yolo11n.pt"   # Standard PT — Most stable across RPi OS versions
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


class FreshFrameReader:
    """A background thread that grabs the absolute latest frame by draining the buffer."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Force small internal buffer
        self.ret = False
        self.frame = None
        self.stopped = False
        from threading import Thread
        self.thread = Thread(target=self._update, args=(), daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            
            # Drain the buffer: grab everything currently in the pipe
            # and only save the very last one.
            tmp_ret, tmp_frame = self.cap.read()
            if tmp_ret:
                self.ret, self.frame = tmp_ret, tmp_frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.stopped = True
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()

def run(args: argparse.Namespace, light: TrafficLight) -> None:
    # ── Focus Reset Timer ─────────────────────────────────────────────────────
    focus_idle_start = None  # tracks when current focus lane became empty
    
    # ── 1. Warm up ────────────────────────────────────────────────────────────

    # ── Source Selection ──────────────────────────────────────────────────────
    if args.cam is not None:
        source_name = f"Webcam {args.cam}"
        source_id = args.cam
    elif args.url is not None:
        source_name = args.url
        source_id = args.url
    else:
        # Final fallback: physical webcam 0
        print("[EcoFlow] No source specified, falling back to physical webcam 0.")
        source_name = "Webcam 0 (Fallback)"
        source_id = 0

    print(f"[EcoFlow] Connecting to: {source_name}")
    reader = FreshFrameReader(source_id)

    if not reader.is_opened():
        sys.exit(
            f"\n[ERROR] Cannot open stream: {source_name}\n"
            "  • Check connections (USB/Network).\n"
        )
    reader.start()
    print(f"[EcoFlow] {source_name} open and buffered.")
    print("[EcoFlow] Warming up stream (2s delay)...")
    time.sleep(2.0)
    print("")

    # ── Web Streamer ──────────────────────────────────────────────────────────
    if args.stream:
        from threading import Thread
        from web_stream import streamer, start_server
        print(f"[EcoFlow] Starting web streamer on http://0.0.0.0:5000")
        Thread(target=start_server, daemon=True).start()

    # ── Per-session state ─────────────────────────────────────────────────────
    from signal_controller import DEFAULT_GREEN_TIME
    amb_state           = AmbulanceState()
    accident_detector   = AccidentDetector()
    controller          = AdaptiveController(green_time=DEFAULT_GREEN_TIME)
    last_eco_risk_label = None
    print("="*60)
    print("   [EcoFlow v3.1] REALTIME PERFORMANCE MODE")
    print("="*60)
    print(f"[EcoFlow] Loading AI: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        sys.exit(f"\n[CRITICAL] YOLO loading failed: {e}")
    print("[EcoFlow] AI Brain ready.\n")
    frame_idx           = 0

    print("[EcoFlow] Live detection started — Ctrl+C to stop.\n")

    try:
        while True:
            # ── 1. YOLO tracking ──────────────────────────────────────────────
            ret, frame = reader.read()
            if not ret or frame is None:
                continue

            frame_idx += 1

            # YOLO inference at optimized resolution
            results = model.track(
                frame,
                tracker   = "bytetrack.yaml",
                persist   = True,
                imgsz     = 320,
                classes   = VEHICLE_CLASS_IDS,
                conf      = args.conf,
                verbose   = False,
            )

            ids, xyxy, clsids, confs = [], [], [], []
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
                        cx_tmp, cy_tmp = (x1 + x2) / 2, (y1 + y2) / 2
                        print(f"\n[EcoFlow] 🚑 Ambulance confirmed:"
                              f" ID={track_id} at ({int(cx_tmp)}, {int(cy_tmp)})")

                if amb_state.confirmed & set(ids):   # any confirmed amb visible now
                    ambulance_in_frame = True

                # Build tracks list for signal + eco modules
                tracks = build_tracks_list(ids, xyxy, clsids, confs,
                                           amb_state.confirmed)

            # ── 3. Accident Detection ────────────────────────────────────────
            accidents = accident_detector.update(tracks)
            if accidents:
                print(f"\n[EcoFlow] 🚨 ACCIDENT DETECTED: IDs={list(accidents)}")

            # ── 3. Adaptive Traffic-light Logic ─────────────────────────────
            h, w = frame.shape[:2]
            
            # --- DYNAMIC FOCUS CONTROL (v5.0) ---
            # Check if there is a web-controlled focus OR a CLI focus
            from web_stream import streamer
            effective_lane = streamer.active_lane if streamer.active_lane is not None else args.lane
            
            # --- AUTO-RESET LOGIC (v6.0) ---
            # If we are in focus mode but see 0 vehicles, start a timer
            if effective_lane is not None and len(tracks) == 0:
                if focus_idle_start is None:
                    focus_idle_start = time.time()
                elif time.time() - focus_idle_start > 5.0: # 5 second timeout
                    print(f"\n[EcoFlow] Lane {effective_lane} clear — auto-resetting to Normal Cycle.")
                    streamer.active_lane = None
                    effective_lane = None
                    focus_idle_start = None
            else:
                focus_idle_start = None # Reset timer if anyone is seen
            
            # --- SINGLE LANE FOCUS OVERRIDE ---
            if effective_lane is not None:
                # Count ALL detections as belonging to the chosen lane
                raw_densities = [0] * 4
                raw_densities[effective_lane] = len(tracks)
                
                # Check for ambulance anywhere in the frame
                amb_in_lane = any(t.get("label", "").lower() in EMERGENCY_LABELS for t in tracks)
                force_emergency_lane = effective_lane if amb_in_lane else None
                
                densities = raw_densities
                green_lane, reason = controller.get_decision(tracks, w, h) # keep machinery running
                
                # Override the machinery's decision for the focused lane
                if force_emergency_lane is not None:
                    green_lane, reason = force_emergency_lane, "AMBULANCE (FOCUS)"
                elif raw_densities[effective_lane] >= 5: # simple threshold for focus
                    green_lane, reason = effective_lane, "TRAFFIC (FOCUS)"
            else:
                # Normal 4-way ROI Logic
                densities = controller._get_densities(tracks, w, h)
                green_lane, reason = controller.get_decision(tracks, w, h)
            
            # --- CONSOLIDATED VISUALIZATION (v4.1+) ---
            if args.calibrate or args.stream or not args.no_preview:
                # 1. Create Annotated Frame (off-screen safe)
                vis_frame = frame.copy()
                
                # 2. Draw YOLO Boxes manually 
                for tid, box, cls_id, conf in zip(ids, xyxy, clsids, confs):
                    x1, y1, x2, y2 = (int(v) for v in box)
                    color = (0, 255, 0)
                    if str(tid) in amb_state.confirmed: color = (0, 0, 255)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{tid} {VEHICLE_CLASSES.get(cls_id, 'obj')}"
                    cv2.putText(vis_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 3. Draw Lane Zone overlays (Dynamic cleanup built-in)
                from signal_controller import draw_rois
                draw_rois(vis_frame, green_lane, densities, reason, 
                          controller.state.remaining_time, focus_lane=effective_lane)

                # 4. Final Status Text
                st_txt = f"{LANE_NAMES[green_lane]} ({reason}) | AMBS: {len(amb_state.confirmed)}"
                if effective_lane is not None: st_txt = f"[FOCUS: {LANE_NAMES[effective_lane]}] " + st_txt
                cv2.putText(vis_frame, st_txt, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if args.calibrate:
                    cv2.imwrite("roi_calibration.jpg", vis_frame)
                    sys.exit(0)

                if args.stream:
                    from web_stream import streamer
                    streamer.update_frame(vis_frame)

                if not args.no_preview:
                    try:
                        cv2.imshow("EcoFlow AI", vis_frame)
                        cv2.waitKey(1)
                    except: pass

            # Update physical LEDs
            light.update_4way(green_lane)
            

            # Progress heartbeat
            if frame_idx % 30 == 0:
                cur_focus = streamer.active_lane if streamer.active_lane is not None else args.lane
                focus_prefix = f"Focus={cur_focus} | " if cur_focus is not None else ""
                print(f"\r[EcoFlow] f={frame_idx:6d} | {focus_prefix}Ambs={len(amb_state.confirmed)} | {light.status_bar}",
                      end="", flush=True)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Core loop failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass



# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EcoFlow AI — Raspberry Pi 5 live traffic monitor")
    p.add_argument("--url", default=None,
                   help="IP Webcam stream URL (e.g. http://192.168.1.50:8080/video)")
    p.add_argument("--cam", type=int, default=None,
                   help="Physical webcam index (e.g. 0)")
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--conf",   type=float, default=0.25,
                   help="YOLO detection confidence (default 0.25)")
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
    p.add_argument("--calibrate",  action="store_true",
                   help="Save 'roi_calibration.jpg' with Lane Zones and exit")
    p.add_argument("--lane",       type=int, choices=[0, 1, 2, 3],
                   help="Focus camera on a single lane (0:N, 1:E, 2:S, 3:W)")
    p.add_argument("--stream",     action="store_true",
                   help="Enable MJPEG web stream on port 5000")
    return p.parse_args()


if __name__ == "__main__":
    args = _args()
    # ── Global Cleanup Protection ─────────────────────────────────────────────
    # We initialize light here so we can guarentee cleanup even if run() crashes
    from signal_controller import TrafficLight
    light = TrafficLight(no_gpio=args.no_gpio)
    
    try:
        run(args, light)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Application crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[EcoFlow] Emergency cleanup: Turning off LEDs...")
        light.cleanup()
        print("[EcoFlow] Cleanup complete. System stopped.")

