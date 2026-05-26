# Aletheia — Wi-Fi Sensing for Passive Presence & Motion Detection

<p align="center">
  <strong>Zero-hardware, privacy-preserving indoor sensing using standard Wi-Fi Beamforming Feedback (BFI)</strong>
</p>

---

## Overview

Aletheia is a deep-tech Wi-Fi sensing system that detects **human presence, movement, and fall events** by analyzing the Channel State Information (CSI) embedded in standard 802.11ac/ax/bf Beamforming Feedback frames — without any cameras, wearables, or additional hardware.

### Key Capabilities

| Feature | Description |
|---|---|
| **Passive Sensing** | Detects presence, walking, and falls using existing Wi-Fi signals |
| **Privacy-First** | No cameras or wearables — only radio wave analysis |
| **Real-Time Dashboard** | Glassmorphic web UI with live φ/ψ waveforms, CIR profiles, and ML predictions |
| **IEEE 802.11bf Ready** | Supports VHT (Cat 21), HE (Cat 26), and SENS (Cat 33) frame parsing |
| **Aletheia-Shield** | Active defense system that scrambles BFI to prevent unauthorized sensing |
| **Edge ML** | Lightweight Random Forest classifier with layout-aware feature engineering |

---

## Architecture

```
┌─────────────┐    802.11 Action Frames     ┌──────────────┐
│  Wi-Fi AP   │ ──────────────────────────▶  │   Sniffer    │
│  (Router)   │   Beamforming Feedback       │  (Monitor)   │
└─────────────┘                              └──────┬───────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   BFI Parser          │
                                        │   • LSB Bit Reader    │
                                        │   • Givens Angles     │
                                        │   • Hampel Filter     │
                                        │   • CFO Detrending    │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   DSP Pipeline        │
                                        │   • Phase Unwrap      │
                                        │   • IFFT → CIR        │
                                        │   • Dynamic Tap ID    │
                                        │   • Feature Extract   │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   ML Classifier       │
                                        │   • Random Forest     │
                                        │   • Layout Priors     │
                                        │   • Majority Vote     │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   WebSocket Server    │
                                        │   • FastAPI Backend   │
                                        │   • Real-time Push    │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   Frontend Dashboard  │
                                        │   • Live Waveforms    │
                                        │   • CIR Visualization │
                                        │   • Alert System      │
                                        └───────────────────────┘
```

---

## Project Structure

```
Aletheai/
├── backend/
│   ├── __init__.py
│   ├── config.py          # System constants & configuration
│   ├── parser.py          # BFI frame parser (LSB bit reader, Givens angles)
│   ├── simulator.py       # Physics-based MIMO channel simulator
│   ├── sniffer.py         # Packet capture (live monitor mode + simulation)
│   └── model.py           # ML pipeline (feature extraction, classifier)
├── frontend/
│   ├── index.html         # Dashboard UI
│   ├── app.js             # Real-time WebSocket client & visualization
│   └── styles.css         # Glassmorphic design system
├── run.py                 # Application entry point
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- A Wi-Fi adapter supporting monitor mode (for live capture)
- Or use **simulation mode** (no hardware needed)

### Installation

```bash
# Clone the repository
git clone https://github.com/SarthakDhatrak/Aletheai.git
cd Aletheai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

### Access the Dashboard

Open your browser and navigate to:

```
http://localhost:8000
```

The system starts in **simulation mode** by default — no Wi-Fi hardware needed.

---

## How It Works

### 1. Signal Capture
Wi-Fi Access Points periodically send **Beamforming Feedback Information (BFI)** frames containing compressed steering matrices. These matrices encode the **Givens rotation angles** (φ and ψ) that describe the wireless channel between transmitter and receiver.

### 2. Channel Decomposition
The parser extracts φ (azimuthal) and ψ (elevation) angles using an LSB-first bit reader, then:
- **Hampel filtering** removes impulse outliers
- **Phase unwrapping + linear detrending** eliminates Carrier Frequency Offset (CFO) drift
- **IFFT projection** converts frequency-domain angles to delay-domain **Channel Impulse Response (CIR)**

### 3. Dynamic Path Identification
The **Dylign** algorithm identifies the CIR tap with highest temporal variance (excluding DC leakage), isolating the multipath component most affected by human movement.

### 4. Feature Engineering
A 23-dimensional feature vector is extracted per window:
- Temporal variance, MAD, and range of φ and ψ
- Adjacent subcarrier correlation (phase coherence)
- CIR dynamic tap statistics
- Layout priors (distance, azimuth, height)

### 5. Classification
A Random Forest classifier maps features to states: **EMPTY**, **PRESENCE**, **WALKING**, **FALLING** — with a 5-frame majority vote filter for temporal smoothing.

---

## Aletheia-Shield (Privacy Defense)

The Shield module protects against unauthorized Wi-Fi sensing by:
- Generating deterministic pseudo-random noise from a shared seed + sounding token
- Adding noise to φ/ψ angles before transmission
- Only authorized receivers with the correct seed can descramble the BFI

---

## Configuration

### Layout Calibration (PerceptAlign)

| Parameter | Description | Range |
|---|---|---|
| Distance (d) | AP-to-sensor distance in meters | 1–10m |
| Azimuth (θ) | Angular offset from boresight | -90° to +90° |
| Height (z) | Sensor mounting height | 0.5–4.0m |

### Simulation States

| State | Description |
|---|---|
| `EMPTY` | No occupant — static ambient channel |
| `PRESENCE` | Stationary person — subtle breathing micro-Doppler |
| `WALKING` | Person walking — periodic multipath oscillation |
| `FALLING` | Fall event — rapid acceleration → impact → lying down |

---

## Tech Stack

- **Backend**: Python, FastAPI, NumPy, SciPy, scikit-learn, Scapy
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Canvas API
- **Protocol**: IEEE 802.11ac/ax/bf (VHT/HE/SENS Action Frames)
- **ML**: Random Forest with synthetic + custom training pipeline

---

## License

This project is for research and educational purposes.

---

## Author

**Sarthak Dhatrak**

- GitHub: [@SarthakDhatrak](https://github.com/SarthakDhatrak)
