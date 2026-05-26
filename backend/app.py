import os
import time
import asyncio
import logging
import threading
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import (
    HOST, PORT, WS_PATH, WINDOW_SIZE_SEC, MIN_PACKETS_IN_WINDOW, DEFAULT_INTERFACE
)
from backend.parser import parse_raw_bfi_payload
from backend.sniffer import BFISniffer
from backend.model import get_trained_classifier, extract_features_from_window

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bfi_app")

app = FastAPI(title="Zero-Hack Wi-Fi Presence & Motion Tracker")

# Global instances
sniffer = BFISniffer()
classifier = get_trained_classifier()

# Sliding window state
packet_window: List[Dict[str, Any]] = []
prediction_history: List[str] = []
window_lock = threading.Lock()

# Custom Training Buffers
is_recording = False
recording_label = ""
recording_buffer: List[Dict[str, Any]] = []
recording_lock = threading.Lock()

# Dataset for custom training (features, labels)
# Pre-populate with empty lists, we'll append to them
custom_training_X: List[List[float]] = []
custom_training_y: List[str] = []

# Main event loop reference for scheduling ws broadcasts from sniffer thread
main_loop: Optional[asyncio.AbstractEventLoop] = None

# Pydantic models for API
class ConfigRequest(BaseModel):
    mode: str  # "simulation" or "live"
    interface: Optional[str] = DEFAULT_INTERFACE
    sim_state: Optional[str] = "EMPTY"  # EMPTY, PRESENCE, WALKING, FALLING
    layout_distance: Optional[float] = 4.0
    layout_azimuth: Optional[float] = 0.0
    layout_height: Optional[float] = 1.5
    bf_format: Optional[str] = "vht"
    shield_active: Optional[bool] = False
    shield_seed: Optional[int] = 42
    shield_authorized: Optional[bool] = True

class StateRequest(BaseModel):
    state: str  # EMPTY, PRESENCE, WALKING, FALLING

class StartTrainRequest(BaseModel):
    label: str  # Custom label e.g., "SITTING_IN_CHAIR"

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, data: Dict[str, Any]):
        # Make a copy of connections list to avoid modifying during iteration
        with self.lock:
            connections = list(self.active_connections)
            
        for connection in connections:
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

# Packet Callback
def on_packet_captured(parsed_pkt: Dict[str, Any]):
    global is_recording, recording_label, recording_buffer, main_loop, classifier
    
    # 1. Decrypt if shield is active and authorized
    if sniffer.shield_active and sniffer.shield_authorized:
        from backend.parser import get_obfuscation_noise
        token = parsed_pkt["sounding_dialog_token"]
        seed = sniffer.shield_seed
        num_subcarriers = parsed_pkt["num_subcarriers"]
        phi_noise, psi_noise = get_obfuscation_noise(token, seed, num_subcarriers)
        
        # Descramble the angles in-place
        for i, subcarrier in enumerate(parsed_pkt["angles"]):
            if i < num_subcarriers:
                if subcarrier["phi"]:
                    phi_val = subcarrier["phi"][0]
                    phi_desc = (phi_val - phi_noise[i]) % (2 * np.pi)
                    subcarrier["phi"] = [phi_desc]
                if subcarrier["psi"]:
                    psi_val = subcarrier["psi"][0]
                    psi_desc = (psi_val - psi_noise[i]) % (np.pi / 2)
                    subcarrier["psi"] = [psi_desc]

    # 2. Update sliding window (thread-safe)
    with window_lock:
        packet_window.append(parsed_pkt)
        now = time.time()
        # Keep only packets within the sliding window size (sec)
        while packet_window and (now - packet_window[0]["timestamp"] > WINDOW_SIZE_SEC):
            packet_window.pop(0)
            
        # Perform ML inference if we have enough samples
        prediction = "UNKNOWN"
        probabilities = {}
        cir_data = {"avg_profile": [], "dynamic_tap": 0}
        if len(packet_window) >= MIN_PACKETS_IN_WINDOW:
            try:
                # If shield is active and we are unauthorized, prediction is stably UNKNOWN
                if sniffer.shield_active and not sniffer.shield_authorized:
                    prediction = "UNKNOWN"
                    probabilities = {c: 0.25 for c in classifier.classes}
                else:
                    # Pass layout parameters to extract features
                    layout = (sniffer.layout_distance, sniffer.layout_azimuth, sniffer.layout_height)
                    features, cir_data = extract_features_from_window(packet_window, layout=layout, return_cir=True)
                    raw_prediction = classifier.predict(features)
                    probabilities = classifier.predict_proba(features)
                    
                    # Temporal Smoothing / Hysteresis majority vote filter (size = 5)
                    prediction_history.append(raw_prediction)
                    if len(prediction_history) > 5:
                        prediction_history.pop(0)
                        
                    from collections import Counter
                    prediction = Counter(prediction_history).most_common(1)[0][0]
            except Exception as e:
                logger.error(f"Inference error: {e}")
                
    # 3. Update custom training recording buffer if active
    with recording_lock:
        if is_recording:
            recording_buffer.append(parsed_pkt)

    # 4. Extract visualization vectors (phi, psi)
    # First spatial stream (index 0) of each subcarrier
    phi_vals = [s["phi"][0] for s in parsed_pkt["angles"] if s["phi"]]
    psi_vals = [s["psi"][0] for s in parsed_pkt["angles"] if s["psi"]]
    
    # 5. Formulate WebSocket message payload
    payload = {
        "timestamp": parsed_pkt["timestamp"],
        "src": parsed_pkt["src"],
        "dst": parsed_pkt["dst"],
        "bssid": parsed_pkt["bssid"],
        "snr": parsed_pkt["snr"],
        "num_subcarriers": parsed_pkt["num_subcarriers"],
        "angles": {
            "phi": phi_vals,
            "psi": psi_vals
        },
        "cir_profile": cir_data["avg_profile"],
        "cir_dynamic_tap": cir_data["dynamic_tap"],
        "prediction": prediction,
        "probabilities": probabilities,
        "mode": sniffer.mode,
        "sim_state": sniffer.simulator.state,
        "is_recording": is_recording,
        "recording_label": recording_label,
        "recording_count": len(recording_buffer),
        
        # Calibration & Shield status
        "layout_distance": sniffer.layout_distance,
        "layout_azimuth": sniffer.layout_azimuth,
        "layout_height": sniffer.layout_height,
        "bf_format": sniffer.bf_format,
        "shield_active": sniffer.shield_active,
        "shield_seed": sniffer.shield_seed,
        "shield_authorized": sniffer.shield_authorized
    }
    
    # 5. Broadcast to all clients (via main thread event loop)
    if main_loop:
        asyncio.run_coroutine_threadsafe(manager.broadcast(payload), main_loop)

# API Endpoints
@app.get("/api/config")
def get_config():
    return {
        "mode": sniffer.mode,
        "interface": sniffer.interface,
        "sim_state": sniffer.simulator.state,
        "is_recording": is_recording,
        "recording_label": recording_label,
        "recording_count": len(recording_buffer),
        "is_classifier_trained": classifier.is_trained,
        
        # Configuration priors & shielding
        "layout_distance": sniffer.layout_distance,
        "layout_azimuth": sniffer.layout_azimuth,
        "layout_height": sniffer.layout_height,
        "bf_format": sniffer.bf_format,
        "shield_active": sniffer.shield_active,
        "shield_seed": sniffer.shield_seed,
        "shield_authorized": sniffer.shield_authorized
    }

@app.post("/api/config")
def update_config(req: ConfigRequest):
    sniffer.stop()
    sniffer.set_mode(req.mode)
    if req.interface:
        sniffer.interface = req.interface
    if req.mode == "simulation" and req.sim_state:
        sniffer.set_simulator_state(req.sim_state)
        
    # Update layout configuration
    if req.layout_distance is not None:
        sniffer.layout_distance = req.layout_distance
    if req.layout_azimuth is not None:
        sniffer.layout_azimuth = req.layout_azimuth
    if req.layout_height is not None:
        sniffer.layout_height = req.layout_height
        
    # Update standard alignment format & shield settings
    if req.bf_format is not None:
        sniffer.bf_format = req.bf_format
    if req.shield_active is not None:
        sniffer.shield_active = req.shield_active
    if req.shield_seed is not None:
        sniffer.shield_seed = req.shield_seed
    if req.shield_authorized is not None:
        sniffer.shield_authorized = req.shield_authorized
        
    # Clear sliding window to prevent cross-contamination of layout/shield transition states
    with window_lock:
        packet_window.clear()
        prediction_history.clear()
        
    sniffer.start(on_packet_captured)
    return get_config()

@app.post("/api/simulate/state")
def update_simulate_state(req: StateRequest):
    if sniffer.mode != "simulation":
        raise HTTPException(status_code=400, detail="Sniffer is not in simulation mode")
    sniffer.set_simulator_state(req.state)
    return {"status": "ok", "state": req.state}

@app.post("/api/training/start")
def start_training(req: StartTrainRequest):
    global is_recording, recording_label, recording_buffer
    with recording_lock:
        is_recording = True
        recording_label = req.label
        recording_buffer = []
    logger.info(f"Custom training recording started for label: {req.label}")
    return {"status": "ok", "label": req.label}

@app.post("/api/training/stop")
def stop_training():
    global is_recording, recording_label, recording_buffer, custom_training_X, custom_training_y, classifier
    
    with recording_lock:
        if not is_recording:
            raise HTTPException(status_code=400, detail="Recording was not active")
        
        is_recording = False
        captured_packets = list(recording_buffer)
        label = recording_label
        recording_buffer = []
        recording_label = ""
        
    # Extract features from sliding windows of the captured data
    window_length = int(WINDOW_SIZE_SEC * 10)  # 2 seconds at 10Hz = 20 packets
    new_samples_count = 0
    
    if len(captured_packets) >= window_length:
        # Extract features by sliding over the captured logs
        for i in range(len(captured_packets) - window_length + 1):
            sub_window = captured_packets[i:i + window_length]
            layout = (sniffer.layout_distance, sniffer.layout_azimuth, sniffer.layout_height)
            feat = extract_features_from_window(sub_window, layout=layout)
            custom_training_X.append(feat.tolist())
            custom_training_y.append(label)
            new_samples_count += 1
            
        # Re-train model incorporating custom data + initial synthetic data
        # To avoid forgetting, let's rebuild synthetic dataset and add custom data on top
        from backend.model import generate_synthetic_training_data
        X_syn, y_syn = generate_synthetic_training_data()
        
        # Combine
        X_combined = np.vstack([X_syn, np.array(custom_training_X)])
        y_combined = y_syn + custom_training_y
        
        classifier.train(X_combined, y_combined)
        logger.info(f"Classifier retrained with {new_samples_count} new samples for label '{label}'.")
        
    return {
        "status": "ok",
        "new_samples": new_samples_count,
        "total_custom_samples": len(custom_training_y),
        "is_classifier_trained": classifier.is_trained
    }

# WebSockets endpoint
@app.websocket(WS_PATH)
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for any client events if needed
            data = await websocket.receive_text()
            # Ignore client text messages for now
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve Frontend static files
# Make sure frontend path exists
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def read_root():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}

# Server startup and shutdown lifecycle
@app.on_event("startup")
def startup_event():
    global main_loop
    main_loop = asyncio.get_event_loop()
    # Start sniffer (simulation mode by default)
    sniffer.start(on_packet_captured)
    logger.info("FastAPI backend startup complete. Sniffer running.")

@app.on_event("shutdown")
def shutdown_event():
    sniffer.stop()
    logger.info("FastAPI backend shutdown complete. Sniffer stopped.")
