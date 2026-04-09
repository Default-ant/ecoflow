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
        
        # --- ECO STATUS DATA (v10.0) ---
        self.veg_pct = 0.0
        self.pollution_idx = 0.0
        self.risk_level = "WAITING"
        self.total_cars = 0
        self.last_update = time.time()

    def update_status(self, veg_pct, pollution_idx, risk_level, total_cars):
        """Update system status data for the web UI."""
        with self.lock:
            self.veg_pct = veg_pct
            self.pollution_idx = pollution_idx
            self.risk_level = risk_level
            self.total_cars = total_cars
            self.last_update = time.time()

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

@app.route('/status')
def get_status():
    """Returns current system status as JSON for AJAX polling."""
    with streamer.lock:
        return {
            "active_lane": streamer.active_lane,
            "lane_name": ["North", "East", "South", "West"][streamer.active_lane] if streamer.active_lane is not None else "Auto-Cycle",
            "veg_pct": round(streamer.veg_pct, 1),
            "pollution_idx": round(streamer.pollution_idx, 1),
            "risk_level": streamer.risk_level,
            "total_cars": streamer.total_cars,
            "timestamp": streamer.last_update
        }

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
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>EcoFlow AI - Premium Dashboard</title>
            <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
            <style>
                :root {
                    --bg: #0a0a0c;
                    --glass: rgba(255, 255, 255, 0.03);
                    --glass-border: rgba(255, 255, 255, 0.08);
                    --accent-n: #ff5252;
                    --accent-e: #448aff;
                    --accent-s: #ffab40;
                    --accent-w: #7c4dff;
                    --accent-eco: #00e676;
                    --accent-poll: #ffea00;
                    --text: #ffffff;
                }

                * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
                body { 
                    font-family: 'Outfit', sans-serif; 
                    background: var(--bg); 
                    background-image: radial-gradient(circle at 50% 10%, #1a1a2e 0%, #0a0a0c 100%);
                    color: var(--text); 
                    margin: 0; 
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                }

                .header { margin-bottom: 30px; text-align: center; }
                .header h1 { 
                    font-weight: 800; 
                    font-size: 2.5rem; 
                    margin: 0; 
                    background: linear-gradient(to right, #fff, #888);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    letter-spacing: -1px;
                }
                .header .badge {
                    background: var(--accent-eco);
                    color: #000;
                    font-weight: 800;
                    font-size: 0.7rem;
                    padding: 3px 10px;
                    border-radius: 20px;
                    text-transform: uppercase;
                    vertical-align: middle;
                    margin-left: 10px;
                }

                .main-grid {
                    display: grid;
                    grid-template-columns: 1fr 340px;
                    gap: 20px;
                    width: 100%;
                    max-width: 1200px;
                }

                @media (max-width: 1000px) {
                    .main-grid { grid-template-columns: 1fr; }
                }

                .card {
                    background: var(--glass);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border: 1px solid var(--glass-border);
                    border-radius: 24px;
                    padding: 24px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
                }

                .stream-card { position: relative; overflow: hidden; }
                .stream-card img {
                    width: 100%;
                    border-radius: 16px;
                    display: block;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                }
                .stream-overlay {
                    position: absolute;
                    top: 34px;
                    left: 34px;
                    display: flex;
                    gap: 10px;
                }
                .live-dot {
                    width: 10px;
                    height: 10px;
                    background: #ff5252;
                    border-radius: 50%;
                    animation: pulse 1.5s infinite;
                }

                @keyframes pulse {
                    0% { transform: scale(0.9); opacity: 0.7; }
                    50% { transform: scale(1.2); opacity: 1; }
                    100% { transform: scale(0.9); opacity: 0.7; }
                }

                .stats-sidebar {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }

                .stat-group { margin-bottom: 20px; }
                .stat-label { 
                    font-size: 0.8rem; 
                    color: #888; 
                    text-transform: uppercase; 
                    font-weight: 600; 
                    letter-spacing: 1px;
                    margin-bottom: 8px;
                    display: block;
                }
                .stat-value { font-size: 1.8rem; font-weight: 700; }

                .eco-bars { margin-top: 10px; }
                .bar-container {
                    background: rgba(255,255,255,0.05);
                    height: 8px;
                    border-radius: 4px;
                    overflow: hidden;
                    margin-top: 6px;
                    margin-bottom: 15px;
                }
                .bar-fill {
                    height: 100%;
                    border-radius: 4px;
                    transition: width 1s ease-out;
                }

                .controls-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 12px;
                }
                
                button {
                    background: var(--glass);
                    border: 1px solid var(--glass-border);
                    color: #fff;
                    padding: 16px;
                    border-radius: 16px;
                    font-family: inherit;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    gap: 4px;
                }
                
                button.active { border-color: rgba(255,255,255,0.4); background: rgba(255,255,255,0.1); }
                button:hover { background: rgba(255,255,255,0.08); transform: translateY(-2px); }
                button:active { transform: translateY(0); }

                .btn-n { border-bottom: 3px solid var(--accent-n); }
                .btn-e { border-bottom: 3px solid var(--accent-e); }
                .btn-s { border-bottom: 3px solid var(--accent-s); }
                .btn-w { border-bottom: 3px solid var(--accent-w); }
                .btn-auto { 
                    grid-column: span 2; 
                    border-bottom: 3px solid var(--accent-eco);
                    background: rgba(0, 230, 118, 0.05);
                }

                .risk-badge {
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    font-weight: 700;
                    margin-top: 8px;
                }
                .risk-safe { background: rgba(0, 230, 118, 0.2); color: #00e676; }
                .risk-warn { background: rgba(255, 234, 0, 0.2); color: #ffea00; }
                .risk-crit { background: rgba(255, 82, 82, 0.2); color: #ff5252; }

            </style>
        </head>
        <body>
            <div class="header">
                <h1>EcoFlow AI<span class="badge">V10.0</span></h1>
            </div>

            <div class="main-grid">
                <div class="card stream-card">
                    <div class="stream-overlay">
                        <div class="live-dot"></div>
                        <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #ff5252;">Live Stream</span>
                    </div>
                    <img id="main-feed" src="{{ url_for('video_feed') }}">
                </div>

                <div class="stats-sidebar">
                    <div class="card">
                        <div class="stat-group">
                            <span class="stat-label">System Focus</span>
                            <div class="stat-value" id="lane-name">Auto-Cycle</div>
                        </div>
                        <div class="stat-group" style="margin-bottom: 0;">
                            <span class="stat-label">Active Monitoring</span>
                            <div class="stat-value" id="car-count">0 Cars</div>
                        </div>
                    </div>

                    <div class="card">
                        <span class="stat-label">Eco-Risk Analysis</span>
                        <div id="risk-display" class="risk-badge risk-safe">WAITING</div>
                        
                        <div class="eco-bars">
                            <div style="display:flex; justify-content:space-between; font-size: 0.8rem; margin-top:15px;">
                                <span>Nature Balance</span>
                                <span id="veg-val">0%</span>
                            </div>
                            <div class="bar-container">
                                <div id="veg-bar" class="bar-fill" style="width: 0%; background: var(--accent-eco);"></div>
                            </div>

                            <div style="display:flex; justify-content:space-between; font-size: 0.8rem;">
                                <span>Air Quality</span>
                                <span id="poll-val">0.0</span>
                            </div>
                            <div class="bar-container">
                                <div id="poll-bar" class="bar-fill" style="width: 0%; background: var(--accent-poll);"></div>
                            </div>
                        </div>
                    </div>

                    <div class="card controls-grid">
                        <button class="btn-n" onclick="setLane(0)" id="btn-0"><span>North</span><span style="font-size: 0.6rem; opacity: 0.6;">UPWARD TRAFFIC</span></button>
                        <button class="btn-e" onclick="setLane(1)" id="btn-1"><span>East</span><span style="font-size: 0.6rem; opacity: 0.6;">RIGHT LANE</span></button>
                        <button class="btn-w" onclick="setLane(3)" id="btn-3"><span>West</span><span style="font-size: 0.6rem; opacity: 0.6;">LEFT LANE</span></button>
                        <button class="btn-s" onclick="setLane(2)" id="btn-2"><span>South</span><span style="font-size: 0.6rem; opacity: 0.6;">DOWNWARD</span></button>
                        <button class="btn-auto" onclick="setLane(-1)" id="btn-auto">MASTER RESET (AUTO-CYCLE)</button>
                    </div>
                </div>
            </div>

            <script>
                let currentLane = null;

                function updateStatus() {
                    fetch('/status')
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById('lane-name').innerText = data.lane_name;
                            document.getElementById('car-count').innerText = data.total_cars + " Cars Detected";
                            
                            // Update Risk Badge
                            const riskBar = document.getElementById('risk-display');
                            riskBar.innerText = data.risk_level;
                            riskBar.className = 'risk-badge ' + (
                                data.risk_level === 'SAFE' ? 'risk-safe' : 
                                data.risk_level === 'CRITICAL' ? 'risk-crit' : 'risk-warn'
                            );

                            // Update Bars
                            document.getElementById('veg-val').innerText = data.veg_pct + "%";
                            document.getElementById('veg-bar').style.width = data.veg_pct + "%";
                            
                            // Scale pollution (assuming 0-20 scale for visualization)
                            let pWidth = Math.min(data.pollution_idx * 5, 100);
                            document.getElementById('poll-val').innerText = data.pollution_idx;
                            document.getElementById('poll-bar').style.width = pWidth + "%";

                            // Update Active Buttons
                            document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                            let btnId = data.active_lane === null ? 'btn-auto' : 'btn-' + data.active_lane;
                            document.getElementById(btnId).classList.add('active');
                        });
                }

                function setLane(id) {
                    fetch('/set_lane/' + id).then(() => updateStatus());
                }

                // Poll status every 2 seconds
                setInterval(updateStatus, 2000);
                updateStatus();
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
