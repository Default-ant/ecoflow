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

    def update_frame(self, frame):
        """Update the current frame to be streamed."""
        with self.lock:
            self.frame = frame.copy()

    def get_frame(self):
        """Encode the current frame as JPEG."""
        with self.lock:
            if self.frame is None:
                return None
            ret, buffer = cv2.imencode('.jpg', self.frame)
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
        # Limit frame rate for the web view to save bandwidth
        time.sleep(0.04)  # ~25 FPS max

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string("""
        <html>
          <head>
            <title>EcoFlow AI - Live Feed</title>
            <style>
              body { font-family: sans-serif; background: #121212; color: white; text-align: center; margin: 0; padding: 20px; }
              img { max-width: 100%; border: 4px solid #333; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
              h1 { margin-bottom: 20px; color: #4CAF50; }
              .status { margin-top: 10px; font-size: 0.9em; opacity: 0.7; }
            </style>
          </head>
          <body>
            <h1>EcoFlow AI - Live Traffic Monitor</h1>
            <img src="{{ url_for('video_feed') }}">
            <div class="status">Streaming live from Raspberry Pi 5</div>
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
