# Aletheia 👁️📡

**Aletheia** is an advanced, privacy-preserving Wi-Fi sensing and smart-home integration platform. It transforms standard commercial off-the-shelf (COTS) Wi-Fi routers and devices into a high-precision, non-invasive radar system capable of detecting human presence, movement, and critical events (like falls) entirely through walls and without cameras.

![Aletheia Dashboard](https://img.shields.io/badge/Aletheia-Dashboard-00f2fe?style=for-the-badge)

## 🌟 Core Features

### 📡 Passive Wi-Fi Radar
Aletheia eavesdrops on standard 802.11ac/ax/be Compressed Beamforming (BFI) Action frames (VHT Category 21 / HE Category 26). By intercepting these sounding packets, it continuously extracts the Channel State Information (CSI) matrix containing fine-grained physical environment reflections.

### 🧠 Advanced Digital Signal Processing (DSP)
- **Domino (RF Distortion Compensation)**: Neutralizes Carrier Frequency Offset (CFO) and Sampling Clock Offset (SCO) by locking onto the direct Line-of-Sight (LoS) path as a static phase reference, cleaning the signal for sub-millimeter precision.
- **Dylign (Dynamic Path Alignment)**: Automatically scans the Delay-Domain Channel Impulse Response (CIR) and extracts the specific multipath reflection tap exhibiting the highest temporal variance, isolating human micro-movements from static wall clutter.
- **PerceptAlign**: Integrates physical room dimensions (Distance, Azimuth, Height) as geometric priors, projecting the delay paths into absolute space for environment-agnostic accuracy.

### 🛡️ Unbreakable Privacy & Security
- **BFLD Anonymization**: All intercepted physical MAC addresses (for both the router and local devices) are cryptographically hashed at the edge before leaving the backend, ensuring complete identity protection.
- **Aletheia-Shield (Secured Sensing Shield)**: Defends against malicious Maximum Likelihood Estimation (MLE) eavesdropping attacks by actively scrambling the transmitted Givens Phase Coordinates (phi, psi) on the PHY layer. Unauthorized listeners only see high-entropy noise, while Aletheia decrypts the radar map.

### 🤖 Edge Machine Learning & ADBlock
- **Random Forest Classifier**: A highly optimized edge ML model that translates spatial phase variance into human states: `EMPTY`, `PRESENCE`, `WALKING`, and `FALLING`.
- **ADBlock (Out-of-Distribution Guardrail)**: A custom-built, lightweight 3-layer Autoencoder that evaluates real-time feature reconstruction error. If a non-human anomaly occurs (e.g., a pet running, or a vacuum cleaner), ADBlock triggers an `OUT_OF_DISTRIBUTION` override, preventing false positive alarms.

### 🏠 Smart Home Integration & MQTT
Aletheia natively integrates with **Home Assistant** via MQTT Auto-Discovery.
- Real-time classification states are published instantly.
- Binary sensor triggers are exposed for emergency `FALLING` events, allowing you to instantly trigger sirens, turn on lights, or send SMS alerts via Home Assistant automations.

### 🔍 Local Network Scanner
Built-in lightweight Ping-Sweep and ARP cache scanner to discover all connected IoT devices, phones, and laptops on your local subnet. The results are securely displayed via the dashboard using BFLD anonymized MACs.

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SarthakDhatrak/Aletheai.git
   cd Aletheai
   ```

2. **Set up the virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the Aletheia Edge Server**:
   ```bash
   sudo ./venv/bin/python run.py
   ```
   *Note: `sudo` is required to enable Monitor Mode on physical Wi-Fi adapters and to perform active OTA validation.*

4. **Access the Dashboard**:
   Open your browser and navigate to: `http://localhost:8000`

## 🛠️ Over-The-Air (OTA) Validation Campaign
To test Aletheia in a real-world physical environment:
1. Plug in a Monitor-Mode capable USB Wi-Fi adapter.
2. Select **OTA Hardware Validation** on the Web Dashboard.
3. Aletheia's `NDPTrigger` ping flooding engine will aggressively wake sleeping smart devices on your network to enforce a steady 10Hz+ MU-MIMO sounding rate.
4. Watch the Space Radar visualize the phase coordinates of the physical environment in real-time!

## 📜 License
Private & Proprietary. All rights reserved.
