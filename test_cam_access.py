import cv2
import sys

def test_cam(index):
    print(f"Testing webcam index {index}...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {index}")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"Success: Captured frame from webcam {index} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        print(f"Error: Could not read frame from webcam {index}")
    
    cap.release()
    return ret

if __name__ == "__main__":
    idx = 0
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    test_cam(idx)
