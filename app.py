import eventlet
eventlet.monkey_patch()

import os
import time
import threading
import base64
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
from vision_engine import VisionEngine, PersonStatus

# Initialize Flask and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global Vision Engine Instance
engine = VisionEngine()
frame_lock = threading.Lock()
output_frame = None

class VisionManager:
    def __init__(self):
        self.camera = None
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        # Reverting to CAP_DSHOW as confirmed by diagnostic script
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.camera.isOpened():
            print("[Error] Could not open camera.")
            self.running = False
            return
        
        # Verify first frame isn't black
        success, test_frame = self.camera.read()
        if success:
            mean_b = np.mean(test_frame)
            print(f"[Vision] Camera initialized. Test frame brightness: {mean_b:.2f}")
            if mean_b < 5:
                print("[Warning] Camera initialized but producing dark frames.")
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        if self.thread:
            self.thread.join()

    def _run(self):
        global output_frame
        while self.running:
            success, frame = self.camera.read()
            if not success:
                time.sleep(0.1)
                continue
            
            # Process frame through Vision Engine
            annotated_frame, _ = engine.process_frame(frame)
            
            with frame_lock:
                output_frame = annotated_frame.copy()
            
            # Encode frame as Base64 for WebSocket streaming
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})
            
            # Broadcast stats via WebSocket
            stats = engine.get_stats()
            socketio.emit('stats_update', stats)
            
            # Optional: Emit events from event_log
            if stats.get('event_log'):
                socketio.emit('event_update', stats['event_log'][-1])
            
            time.sleep(0.04) # ~25 FPS

manager = VisionManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # MJPEG stream is disabled in favor of WebSocket streaming
    return "Please use / for WebSocket-based dashboard.", 404

@app.route('/api/start')
def start_engine():
    manager.start()
    return jsonify({"status": "started"})

@app.route('/api/stop')
def stop_engine():
    manager.stop()
    return jsonify({"status": "stopped"})

@app.route('/api/stats')
def get_stats():
    return jsonify(engine.get_stats())

@app.route('/api/expressions')
def get_expressions():
    stats = engine.get_stats()
    expressions = [
        {"id": p["id"], "expression": p["expression"], "confidence": p["expr_conf"]}
        for p in stats.get("persons", [])
    ]
    return jsonify(expressions)

@app.route('/api/set_model/<model_type>')
def set_model(model_type):
    if model_type.lower() == "pth":
        filename = "Affectnet_model.pth"
    elif model_type.lower() == "h5":
        filename = "MUL_KSIZE_MobileNet_v2_best.hdf5"
    else:
        filename = "facial_expression_recognition_mobilefacenet.onnx"
    
    success = engine.set_expression_model(filename)
    return jsonify({"status": "success" if success else "failed", "model": filename})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'data': 'Connected'})

if __name__ == '__main__':
    # Start the engine by default
    manager.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)
