# 🎭 Neutral Emotion Detector & Fall Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated real-time vision system that combines multi-person tracking, advanced facial emotion recognition, and automated fall detection. Built with a robust Flask-SocketIO backend and a dynamic web dashboard.

## ✨ Key Features

- **🚀 Real-time Processing**: Low-latency video analysis (~25 FPS) using optimized vision engines.
- **👥 Multi-Person Tracking**: Simultaneously tracks up to 4 individuals with unique IDs and color coding.
- **🎭 Advanced Emotion Recognition**: 
  - Uses MediaPipe's 52 ARKit blendshape coefficients for precise expression classification.
  - Multi-backend support: ONNX (MobileFaceNet), PyTorch (AffectNet), and HDF5 (MobileNetV2).
  - Temporal smoothing to prevent rapid "flickering" between emotion labels.
- **⚠️ Fall Detection**: Intelligent pose analysis using spine angle, aspect ratio, and velocity to detect and alert on falls.
- **📊 Interactive Dashboard**: Web-based UI with live video streaming via WebSockets and real-time statistics.
- **🌐 REST API**: Programmatic access to telemetry, facial expressions, and model management.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Flask-SocketIO (Eventlet)
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: PyTorch, TensorFlow, ONNX Runtime
- **Frontend**: HTML5, Vanilla CSS, JavaScript (WebSockets)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A webcam connected to your system

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/shivamgithubok/Neutral-Emotion-Detector.git
   cd Neutral-Emotion-Detector
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the modles here**
   - [ ] https://www.kaggle.com/code/tangthanhvuik18ct/dbm-project-deepface/output
   - [ ] https://github.com/liminze/Real-time-Facial-Expression-Recognition-and-Fast-Face-Detection/tree/master/models/best_model
   - [] the modilefacenet download the model from the github repo

5. **Run the Convert_model_in_onnx if running on cpu make it #x faster**
   ```bash
   python convert_model_in_onnx.py  
   ```

### Running the Application

Start the Flask server:
```bash
python app.py
```
By default, the application will be available at `http://localhost:5000`.

## 📖 Usage

### Web Dashboard
Access the main interface at `http://localhost:5000`. The dashboard provides:
- Live annotated video feed.
- Real-time HUD showing detected emotions and pose status.
- System metrics and event logs.

### API Endpoints

- `GET /api/stats`: Retrieve current system-wide detection statistics.
- `GET /api/expressions`: Get list of active persons and their current detected expressions.
- `GET /api/start` / `GET /api/stop`: Control the vision engine state.
- `GET /api/set_model/<model_type>`: Switch between prediction models (`pth`, `h5`, or `onnx`).

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the Neutral Emotion Detector, please:
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed by the Elevatics AI team.*