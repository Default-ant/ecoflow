import sys
import time

print("--- Testing Library Imports ---")
try:
    import numpy as np
    print(f"[OK] NumPy version: {np.__version__}")
except Exception as e:
    print(f"[FAIL] NumPy import failed: {e}")

try:
    import cv2
    print(f"[OK] OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"[FAIL] OpenCV import failed: {e}")

try:
    from ultralytics import YOLO
    print(f"[OK] Ultralytics/YOLO imported")
except Exception as e:
    print(f"[FAIL] Ultralytics import failed: {e}")

try:
    import onnxruntime
    print(f"[OK] ONNX Runtime version: {onnxruntime.__version__}")
except Exception as e:
    print(f"[FAIL] ONNX Runtime import failed: {e}")

print("\n--- Testing Model Load ---")
try:
    # Attempt to load the model (modify path if needed)
    model = YOLO("models/yolo11n.onnx", task="detect")
    print("[OK] YOLO Model loaded successfully")
except Exception as e:
    print(f"[FAIL] Model loading failed: {e}")

print("\nIf you see this, basic imports and loading are NOT the cause of the Segfault.")
