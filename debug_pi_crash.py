import sys
import os

# Force unbuffered output so we see text even during a crash
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

def test_step(name, func):
    print(f"Testing {name:.<30}", end="", flush=True)
    try:
        func()
        print("[OK]", flush=True)
    except Exception as e:
        print(f"[FAIL] - {e}", flush=True)

print("=== Raspberry Pi 5 Dependency Diagnostic ===\n", flush=True)

# 1. Imports - One by one with immediate flushing
print("Checking imports...", flush=True)
test_step("Import NumPy",   lambda: __import__("numpy"))
test_step("Import CV2",     lambda: __import__("cv2"))
test_step("Import Torch",   lambda: __import__("torch"))
test_step("Import NCNN",    lambda: __import__("ncnn"))
test_step("Import Flask",   lambda: __import__("flask"))
test_step("Import Ultralytics", lambda: __import__("ultralytics"))

# 2. Hardware Checks
import numpy as np
print(f"\nNumPy Info: {np.__config__.show() if hasattr(np, '__config__') else 'N/A'}")

# 3. Model Loading (The likely spot for Illegal Instruction if binaries are bad)
from ultralytics import YOLO

def load_model():
    model = YOLO("models/yolo11n_ncnn_model")
    return model

test_step("Load NCNN Model", load_model)

# 4. Dummy Inference
def run_inference():
    m = load_model()
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    m.predict(dummy, verbose=False)

test_step("Run Dummy Inference", run_inference)

print("\nIf the script stopped abruptly with 'Illegal instruction' before a line finished, that's the culprit.")
