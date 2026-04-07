# EcoFlow AI: Smart Adaptive Traffic Control

EcoFlow AI is a high-performance, edge-AI traffic management system optimized for the **Raspberry Pi 5**. It leverages **YOLOv11** and **Computer Vision** to dynamically adjust traffic signal timings, prioritize emergency vehicles, and monitor environmental risks in real-time.

## 🚀 Features

- **Adaptive Signal Control**: Dynamically adjusts green light duration based on vehicle density per lane.
- **Ambulance Priority**: Immediate "Emergency Green" phase detection for confirmed emergency vehicles.
- **Accident Detection**: Real-time trajectory analysis to identify potential collisions or stalled vehicles.
- **Eco-Risk Monitoring**: Analyzes traffic flow vs. vegetation data to estimate local pollution levels.
- **Live MJPEG Stream**: Low-latency web dashboard for remote traffic monitoring.
- **GPIO Integration**: Direct control of physical LED traffic lights via the RPi 5.

## 🛠️ Hardware Requirements

- **Raspberry Pi 5** (8GB recommended for tracking performance).
- **RPi Camera Module** or any compatible USB/IP Webcam.
- **Traffic Light LEDs** (Red, Yellow, Green) connected via GPIO.
- **Connecting Wires** and a Breadboard.

## 💻 Tech Stack

- **Language**: Python 3.11+
- **AI Model**: Ultralytics YOLOv11 (NCNN Optimized for ARM).
- **Computer Vision**: OpenCV, ByteTrack.
- **Hardware Control**: `gpiozero`, `lgpio`.
- **Package Manager**: `uv` (Fast Python dependency management).

## 🚦 Quick Start

### 1. Installation
Ensure you have `uv` installed, then sync the environment:
```bash
uv sync
```

### 2. Run the System
Start the main orchestrator with an IP webcam stream:
```bash
uv run python ecoflow_ai.py --url http://192.168.x.x:8080/video --stream
```

### 3. Generate Presentation
To re-generate the project presentation slides:
```bash
uv run python gen_presentation.py
```

## 📁 Project Structure

- `ecoflow_ai.py`: Main brain and system orchestrator.
- `ambulance_detection.py`: Specific logic for identifying emergency vehicles.
- `signal_controller.py`: Dynamic timing logic and GPIO hardware bridge.
- `eco_risk.py`: Environmental analysis and pollution logging.
- `accident_detection.py`: Safety monitoring and collision identification.
- `web_stream.py`: MJPEG server for the live dashboard.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
