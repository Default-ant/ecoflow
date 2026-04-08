"""
EcoFlow AI — Web Streamer
=========================
A lightweight MJPEG streamer using Flask.
Provides a simple route to view the live processed frame in a browser.
"""

import time
from threading import Lock
from flask import Flask, Response, render_template_string

import cv2

app = Flask(__name__)

class WebStreamer:
    def __init__(self):
        self.frame = None
        self.lock = Lock()
        self.active_lane = None # None means 4-Way Auto Mode

    def update_frame(self, frame):
        """Update the current frame to be streamed."""
        with self.lock:
            self.frame = frame.copy()

    def get_frame(self):
        """Encode the current frame as JPEG."""
        with self.lock:
            if self.frame is None:
                return None
            # Encode as JPEG with performance-tuned quality (45)
            ret, buffer = cv2.imencode('.jpg', self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
            return buffer.tobytes()

streamer = WebStreamer()

def generate():
    """Video streaming generator function."""
    while True:
        frame_bytes = streamer.get_frame()
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS max

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_lane/<int:lane_id>')
def set_lane(lane_id):
    """API to switch camera focus. lane_id -1 means Auto Mode."""
    if lane_id == -1:
        streamer.active_lane = None
        msg = "Switched to Auto (4-Way) Mode"
    else:
        streamer.active_lane = lane_id
        names = ["North", "East", "South", "West"]
        msg = f"Focused on {names[lane_id]} Lane"
    
    print(f"[Web] {msg}")
    return {"status": "success", "message": msg}

@app.route('/')
def index():
    return render_template_string("""
        <html>
          <head>
            <title>EcoFlow AI - Remote Control</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body { font-family: sans-serif; background: #121212; color: white; text-align: center; margin: 0; padding: 10px; }
              .stream-container { max-width: 800px; margin: auto; }
              img { width: 100%; border: 2px solid #333; border-radius: 8px; }
              .controls { margin-top: 20px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; max-width: 400px; margin-inline: auto; }
              button { padding: 15px; font-size: 1em; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; transition: 0.2s; }
              .btn-n { background: #ff5252; color: white; }
              .btn-e { background: #448aff; color: white; }
              .btn-s { background: #ffab40; color: white; }
              .btn-w { background: #7c4dff; color: white; }
              .btn-auto { background: #4caf50; color: white; grid-column: span 2; margin-top: 5px; }
              button:active { transform: scale(0.95); opacity: 0.8; }
              #status { margin-top: 15px; color: #4caf50; font-weight: bold; height: 20px; }
            </style>
          </head>
          <body>
            <h1>EcoFlow AI — Live Focus</h1>
            <div class="stream-container">
                <img src="{{ url_for('video_feed') }}">
            </div>
            
            <div id="status">Ready</div>

            <div class="controls">
                <button class="btn-n" onclick="setLane(0)">North ↑</button>
                <button class="btn-e" onclick="setLane(1)">East →</button>
                <button class="btn-w" onclick="setLane(3)">← West</button>
                <button class="btn-s" onclick="setLane(2)">South ↓</button>
                <button class="btn-auto" onclick="setLane(-1)">Reset All (4-Way Zoom Out)</button>
            </div>

            <script>
                function setLane(id) {
                    document.getElementById('status').innerText = "Switching...";
                    fetch('/set_lane/' + id)
                        .then(r => r.json())
                        .then(d => {
                            document.getElementById('status').innerText = d.message;
                        });
                }
            </script>
          </body>
        </html>
    """)

def start_server(host='0.0.0.0', port=5000):
    """Start the Flask server."""
    # disable logging to keep console clean
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

if __name__ == '__main__':
    start_server()
