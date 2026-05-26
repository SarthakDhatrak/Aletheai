// Frontend controller for Aletheai BFI Wi-Fi Sensing Radar

let ws = null;
let lastState = null;
let prevAngles = null;

// Persistent alarm states
let fallAlarmActive = false;
let alarmInterval = null;

// Heatmap History Buffer
const maxHeatmapCols = 100;
let heatmapData = []; // Array of arrays: [time_slice][subcarrier]

// Web Audio API context for synthesized alerts
let audioCtx = null;

// Initialize elements and event listeners
document.addEventListener("DOMContentLoaded", () => {
    initVisualizers();
    connectWebSocket();
    setupEventListeners();
    fetchCurrentConfig();
});

// Fetch current backend configuration on startup
async function fetchCurrentConfig() {
    try {
        const res = await fetch("/api/config");
        const data = await res.json();
        updateConfigUI(data);
    } catch (err) {
        console.error("Error fetching current configuration:", err);
    }
}

// Setup WebSocket connection with automatic reconnect
function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    appendLog("System", "Establishing socket connection to backend...", "info");
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        document.getElementById("connBadge").className = "connection-badge connected";
        document.getElementById("connBadge").querySelector(".badge-text").innerText = "Sensor Active";
        appendLog("System", "WebSocket connection active. Receiving live telemetry.", "success");
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            processTelemetry(data);
        } catch (err) {
            console.error("Error parsing WebSocket message:", err);
        }
    };
    
    ws.onclose = () => {
        document.getElementById("connBadge").className = "connection-badge";
        document.getElementById("connBadge").querySelector(".badge-text").innerText = "Sensor Disconnected";
        appendLog("System", "WebSocket connection lost. Reconnecting in 3 seconds...", "warn");
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
    };
}

// Update the configuration form based on backend state
function updateConfigUI(config) {
    const modeSelect = document.getElementById("modeSelect");
    const ifaceInput = document.getElementById("ifaceInput");
    const ifaceGroup = document.getElementById("ifaceGroup");
    const statusMode = document.getElementById("statusMode");
    
    modeSelect.value = config.mode;
    ifaceInput.value = config.interface;
    
    if (config.mode === "live") {
        ifaceGroup.style.display = "flex";
        statusMode.innerText = `LIVE: ${config.interface}`;
        document.getElementById("simControllerCard").style.opacity = "0.5";
        document.getElementById("simControllerCard").style.pointerEvents = "none";
    } else {
        ifaceGroup.style.display = "none";
        statusMode.innerText = "SIMULATED";
        document.getElementById("simControllerCard").style.opacity = "1";
        document.getElementById("simControllerCard").style.pointerEvents = "auto";
        
        // Highlight active simulation state button
        updateSimStateBtnHighlight(config.sim_state);
    }

    // Populate layout calibration inputs
    if (config.layout_distance !== undefined) {
        document.getElementById("layoutDistanceInput").value = config.layout_distance;
    }
    if (config.layout_azimuth !== undefined) {
        document.getElementById("layoutAzimuthInput").value = config.layout_azimuth;
    }
    if (config.layout_height !== undefined) {
        document.getElementById("layoutHeightInput").value = config.layout_height;
    }
    
    // Populate standard format selection
    if (config.bf_format !== undefined) {
        document.getElementById("formatSelect").value = config.bf_format;
        
        // Update standard telemetry label
        const stdLabel = {
            "vht": "VHT (802.11ac)",
            "he": "HE Sensing (802.11ax)",
            "11bf": "IEEE 802.11bf SENS"
        }[config.bf_format] || config.bf_format.toUpperCase();
        document.getElementById("statusStandard").innerText = stdLabel;
    }
    
    // Populate security / shield inputs
    if (config.shield_active !== undefined) {
        const toggle = document.getElementById("shieldToggle");
        toggle.checked = config.shield_active;
        
        const detailsGroup = document.getElementById("shieldDetailsGroup");
        const statusRow = document.getElementById("shieldStatusIcon").parentElement;
        const statusIcon = document.getElementById("shieldStatusIcon");
        const toggleDesc = document.getElementById("shieldToggleDesc");
        const statusShield = document.getElementById("statusShield");
        
        if (config.shield_active) {
            detailsGroup.style.display = "block";
            statusRow.classList.add("shield-active");
            statusIcon.innerText = "🔒";
            toggleDesc.innerText = "BFI Scrambling Enabled";
            
            if (config.shield_authorized) {
                statusShield.innerText = "ACTIVE (AUTH)";
                statusShield.style.color = "var(--emerald-glow)";
            } else {
                statusShield.innerText = "ACTIVE (SCRAMBLED)";
                statusShield.style.color = "var(--red-glow)";
            }
        } else {
            detailsGroup.style.display = "none";
            statusRow.classList.remove("shield-active");
            statusIcon.innerText = "🔓";
            toggleDesc.innerText = "Plaintext BFI Transmission";
            statusShield.innerText = "INACTIVE";
            statusShield.style.color = "var(--text-secondary)";
        }
    }
    
    if (config.shield_seed !== undefined) {
        document.getElementById("shieldSeedInput").value = config.shield_seed;
    }
    if (config.shield_authorized !== undefined) {
        document.getElementById("authSelect").value = config.shield_authorized ? "authorized" : "unauthorized";
    }
}

// Highlight the active simulation state button
function updateSimStateBtnHighlight(state) {
    const buttons = document.querySelectorAll(".sim-btn");
    buttons.forEach(btn => {
        if (btn.getAttribute("data-state") === state) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });
}

// Bind interactive event listeners
function setupEventListeners() {
    // Hardware configuration form
    const configForm = document.getElementById("configForm");
    configForm.addEventListener("submit", async () => {
        const mode = document.getElementById("modeSelect").value;
        const interfaceName = document.getElementById("ifaceInput").value;
        
        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode, interface: interfaceName })
            });
            const data = await res.json();
            updateConfigUI(data);
            appendLog("Config", `Configuration updated. Mode: ${mode.toUpperCase()}`, "success");
        } catch (err) {
            appendLog("Config", "Failed to update hardware configuration.", "warn");
        }
    });
    
    // Sniff mode selector change (show/hide interface input)
    const modeSelect = document.getElementById("modeSelect");
    const ifaceGroup = document.getElementById("ifaceGroup");
    modeSelect.addEventListener("change", () => {
        if (modeSelect.value === "live") {
            ifaceGroup.style.display = "flex";
        } else {
            ifaceGroup.style.display = "none";
        }
    });
    
    // Simulation scenario injector buttons
    const simButtons = document.querySelectorAll(".sim-btn");
    simButtons.forEach(btn => {
        btn.addEventListener("click", async () => {
            const state = btn.getAttribute("data-state");
            try {
                const res = await fetch("/api/simulate/state", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ state })
                });
                const data = await res.json();
                if (data.status === "ok") {
                    updateSimStateBtnHighlight(state);
                    appendLog("Simulator", `Injected environment scenario: ${state}`, "info");
                }
            } catch (err) {
                console.error("Simulation command failed", err);
            }
        });
    });

    // Layout calibration form
    const calibrationForm = document.getElementById("calibrationForm");
    calibrationForm.addEventListener("submit", async () => {
        const distance = parseFloat(document.getElementById("layoutDistanceInput").value);
        const azimuth = parseFloat(document.getElementById("layoutAzimuthInput").value);
        const height = parseFloat(document.getElementById("layoutHeightInput").value);
        
        try {
            const currentRes = await fetch("/api/config");
            const currentConfig = await currentRes.json();
            
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: currentConfig.mode,
                    interface: currentConfig.interface,
                    layout_distance: distance,
                    layout_azimuth: azimuth,
                    layout_height: height
                })
            });
            const data = await res.json();
            updateConfigUI(data);
            appendLog("Calibration", `Layout coordinates applied: ${distance}m, ${azimuth}°, ${height}m`, "success");
        } catch (err) {
            appendLog("Calibration", "Failed to update layout coordinates.", "warn");
        }
    });

    // Toggle switch handler for shield (immediate show/hide details)
    const shieldToggle = document.getElementById("shieldToggle");
    shieldToggle.addEventListener("change", () => {
        const detailsGroup = document.getElementById("shieldDetailsGroup");
        const statusRow = document.getElementById("shieldStatusIcon").parentElement;
        const statusIcon = document.getElementById("shieldStatusIcon");
        const toggleDesc = document.getElementById("shieldToggleDesc");
        
        if (shieldToggle.checked) {
            detailsGroup.style.display = "block";
            statusRow.classList.add("shield-active");
            statusIcon.innerText = "🔒";
            toggleDesc.innerText = "BFI Scrambling Enabled";
        } else {
            detailsGroup.style.display = "none";
            statusRow.classList.remove("shield-active");
            statusIcon.innerText = "🔓";
            toggleDesc.innerText = "Plaintext BFI Transmission";
        }
    });

    // Security form submission
    const securityForm = document.getElementById("securityForm");
    securityForm.addEventListener("submit", async () => {
        const format = document.getElementById("formatSelect").value;
        const active = document.getElementById("shieldToggle").checked;
        const seed = parseInt(document.getElementById("shieldSeedInput").value, 10);
        const auth = document.getElementById("authSelect").value === "authorized";
        
        try {
            const currentRes = await fetch("/api/config");
            const currentConfig = await currentRes.json();
            
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: currentConfig.mode,
                    interface: currentConfig.interface,
                    bf_format: format,
                    shield_active: active,
                    shield_seed: seed,
                    shield_authorized: auth
                })
            });
            const data = await res.json();
            updateConfigUI(data);
            appendLog("Security", `Settings applied. Standard: ${format.toUpperCase()}, Shield: ${active ? "ACTIVE" : "INACTIVE"} (${auth ? "Auth" : "Unauth"})`, "success");
        } catch (err) {
            appendLog("Security", "Failed to apply security settings.", "warn");
        }
    });

    // Custom Classifier Training Buttons
    const recordBtn = document.getElementById("recordBtn");
    const trainStopBtn = document.getElementById("trainStopBtn");
    const trainLabelInput = document.getElementById("trainLabel");

    recordBtn.addEventListener("click", async () => {
        const label = trainLabelInput.value.trim().toUpperCase();
        if (!label) {
            alert("Please provide a valid activity label first.");
            return;
        }

        if (recordBtn.classList.contains("active")) {
            // Act as cancel
            stopRecordingSession(false);
            return;
        }

        try {
            const res = await fetch("/api/training/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ label })
            });
            const data = await res.json();
            if (data.status === "ok") {
                recordBtn.innerText = "🛑 Stop Recording";
                recordBtn.classList.add("active");
                trainStopBtn.disabled = true;
                trainLabelInput.disabled = true;
                document.getElementById("recordCounter").style.display = "block";
                appendLog("ML Record", `Started capturing data for activity: ${label}`, "warn");
            }
        } catch (err) {
            console.error("Failed to start training", err);
        }
    });

    trainStopBtn.addEventListener("click", async () => {
        trainStopBtn.disabled = true;
        trainStopBtn.innerText = "Training...";
        try {
            const res = await fetch("/api/training/stop", { method: "POST" });
            const data = await res.json();
            if (data.status === "ok") {
                appendLog("ML Pipeline", `Model retrained successfully. Added ${data.new_samples} samples.`, "success");
            }
        } catch (err) {
            appendLog("ML Pipeline", "Failed to compile custom model training data.", "warn");
        } finally {
            trainStopBtn.innerText = "⚡ Train Model";
            trainLabelInput.disabled = false;
            trainLabelInput.value = "";
        }
    });
}

function stopRecordingSession(hasEnoughData) {
    const recordBtn = document.getElementById("recordBtn");
    const trainStopBtn = document.getElementById("trainStopBtn");
    
    recordBtn.innerText = "🔴 Start Recording";
    recordBtn.classList.remove("active");
    document.getElementById("recordCounter").style.display = "none";
    
    if (hasEnoughData) {
        trainStopBtn.disabled = false;
    } else {
        document.getElementById("trainLabel").disabled = false;
        appendLog("ML Record", "Recording canceled (insufficient frames).", "info");
    }
}

// Play synthesized warning tone for fall alarm
function playFallAlarm() {
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        
        // Synthesize an alarm tone
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc.type = "sawtooth";
        osc.frequency.setValueAtTime(880, audioCtx.currentTime); // high tone
        osc.frequency.linearRampToValueAtTime(440, audioCtx.currentTime + 0.4); // sweep down
        
        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
        
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.start();
        osc.stop(audioCtx.currentTime + 0.5);
    } catch (e) {
        console.error("Failed to play web audio alarm:", e);
    }
}

// Persistent / Latched Alarm Handlers
function triggerFallAlarm() {
    if (fallAlarmActive) return;
    fallAlarmActive = true;
    
    // Show latched emergency overlay
    const overlay = document.getElementById("fallAlarmOverlay");
    if (overlay) {
        overlay.style.display = "flex";
    }
    
    // Play immediately
    playFallAlarm();
    
    // Sound alarm periodically every 2 seconds
    alarmInterval = setInterval(() => {
        if (fallAlarmActive) {
            playFallAlarm();
        }
    }, 2000);
}

function dismissFallAlarm() {
    fallAlarmActive = false;
    if (alarmInterval) {
        clearInterval(alarmInterval);
        alarmInterval = null;
    }
    
    // Hide emergency overlay
    const overlay = document.getElementById("fallAlarmOverlay");
    if (overlay) {
        overlay.style.display = "none";
    }
    
    // Recalibrate/Reset simulation scenario to EMPTY to clear the fallen physics state
    fetch("/api/simulate/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: "EMPTY" })
    }).then(res => res.json())
      .then(data => {
          if (data.status === "ok") {
              updateSimStateBtnHighlight("EMPTY");
              appendLog("System", "Emergency alarm dismissed. Sensor recalibrated to EMPTY.", "success");
          }
      }).catch(err => {
          console.error("Failed to reset simulator state", err);
      });
}

// Process incoming WebSocket packet data
function processTelemetry(data) {
    // 1. Update Telemetry Summary
    document.getElementById("statusSrc").innerText = data.src;
    document.getElementById("statusSnr").innerText = `${data.snr} dBm`;
    document.getElementById("statusCarriers").innerText = data.num_subcarriers;
    
    // SNR progress bar filling
    const snrPercent = Math.min(100, Math.max(0, (data.snr + 100) * 1.5));
    document.getElementById("snrFill").style.width = `${snrPercent}%`;
    
    // Update Standard and Shield telemetry labels dynamically
    if (data.bf_format !== undefined) {
        const stdLabel = {
            "vht": "VHT (802.11ac)",
            "he": "HE Sensing (802.11ax)",
            "11bf": "IEEE 802.11bf SENS"
        }[data.bf_format] || data.bf_format.toUpperCase();
        document.getElementById("statusStandard").innerText = stdLabel;
    }
    if (data.shield_active !== undefined) {
        const statusShield = document.getElementById("statusShield");
        if (data.shield_active) {
            if (data.shield_authorized) {
                statusShield.innerText = "ACTIVE (AUTH)";
                statusShield.style.color = "var(--emerald-glow)";
            } else {
                statusShield.innerText = "ACTIVE (SCRAMBLED)";
                statusShield.style.color = "var(--red-glow)";
            }
        } else {
            statusShield.innerText = "INACTIVE";
            statusShield.style.color = "var(--text-secondary)";
        }
    }

    // 2. Update ML Classification Prediction Display
    const currentPrediction = data.prediction;
    const isRecordingActive = data.is_recording;
    
    if (isRecordingActive) {
        document.getElementById("recordedFramesCount").innerText = data.recording_count;
        if (data.recording_count >= 20) {
            const trainStopBtn = document.getElementById("trainStopBtn");
            if (trainStopBtn.disabled && document.getElementById("recordBtn").classList.contains("active")) {
                // Enable button once minimum threshold of 20 samples is met
                trainStopBtn.disabled = false;
            }
        }
    }
    
    // If prediction state changes, update indicator and append logs
    if (currentPrediction !== lastState) {
        updateStateDisplay(currentPrediction);
        lastState = currentPrediction;
    }
    
    // 3. Update Probabilities list
    if (data.probabilities) {
        for (const [cls, prob] of Object.entries(data.probabilities)) {
            const bar = document.getElementById(`prob_${cls}`);
            const val = document.getElementById(`val_${cls}`);
            if (bar && val) {
                const percent = Math.round(prob * 100);
                bar.style.width = `${percent}%`;
                val.innerText = `${percent}%`;
            }
        }
    }
    
    // 4. Update Space Radar Visualizer
    drawSpaceRadar(data.angles);
    
    // 5. Update Heatmap Buffer
    computeHeatmapSlice(data.angles);
    
    // 6. Update Delay-Domain CIR Visualizer
    drawCIRProfile(data.cir_profile, data.cir_dynamic_tap);
}

// Update the main Activity Card & Indicator Orb styles
function updateStateDisplay(state) {
    const orb = document.getElementById("stateOrb");
    const predText = document.getElementById("statePrediction");
    
    orb.className = "state-indicator-orb"; // reset
    
    if (state === "EMPTY") {
        orb.classList.add("state-empty");
        predText.innerText = "EMPTY ROOM";
        appendLog("Sensing", "Room is unoccupied (static ambient channel).", "success");
    } else if (state === "PRESENCE") {
        orb.classList.add("state-presence");
        predText.innerText = "PRESENCE";
        appendLog("Sensing", "Static occupant presence detected.", "info");
    } else if (state === "WALKING") {
        orb.classList.add("state-walking");
        predText.innerText = "MOVEMENT";
        appendLog("Sensing", "Active human walking motion detected.", "warn");
    } else if (state === "FALLING") {
        orb.classList.add("state-falling");
        predText.innerText = "FALL DETECTED";
        appendLog("ALERT", "🚨 CRITICAL: High-impact fall event signature matched!", "danger");
        triggerFallAlarm();
    } else if (state === "UNKNOWN") {
        orb.classList.add("state-unknown");
        predText.innerText = "SCRAMBLED / UNKNOWN";
        appendLog("Shield", "Obfuscation active: signal scrambled for unauthorized interceptors.", "warn");
    } else {
        predText.innerText = "CALIBRATING";
    }
}

// Visualizer Setup and Drawing Logic
let radarCanvas, radarCtx;
let heatmapCanvas, heatmapCtx;
let cirCanvas, cirCtx;

function initVisualizers() {
    radarCanvas = document.getElementById("radarCanvas");
    radarCtx = radarCanvas.getContext("2d");
    
    heatmapCanvas = document.getElementById("heatmapCanvas");
    heatmapCtx = heatmapCanvas.getContext("2d");
    
    cirCanvas = document.getElementById("cirCanvas");
    cirCtx = cirCanvas.getContext("2d");
    
    // Set internal canvas resolution to match client bounding boxes
    resizeCanvases();
    window.addEventListener("resize", resizeCanvases);
}

function resizeCanvases() {
    const rRect = radarCanvas.parentElement.getBoundingClientRect();
    radarCanvas.width = rRect.width;
    radarCanvas.height = rRect.height;
    
    const hRect = heatmapCanvas.parentElement.getBoundingClientRect();
    heatmapCanvas.width = hRect.width;
    heatmapCanvas.height = hRect.height;
    
    const cRect = cirCanvas.parentElement.getBoundingClientRect();
    cirCanvas.width = cRect.width;
    cirCanvas.height = cRect.height;
    
    drawSpaceRadarGrid();
}

// Draw static background grids on Radar Canvas
function drawSpaceRadarGrid() {
    const W = radarCanvas.width;
    const H = radarCanvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = Math.min(W, H) / 2 * 0.85;
    
    radarCtx.clearRect(0, 0, W, H);
    
    // Polar grids
    radarCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    radarCtx.lineWidth = 1;
    
    // Concentric circles
    for (let rFactor = 0.25; rFactor <= 1.0; rFactor += 0.25) {
        radarCtx.beginPath();
        radarCtx.arc(cx, cy, R * rFactor, 0, 2 * Math.PI);
        radarCtx.stroke();
    }
    
    // Radial axis lines
    for (let angle = 0; angle < 180; angle += 30) {
        const rad = angle * Math.PI / 180;
        const dx = R * Math.cos(rad);
        const dy = R * Math.sin(rad);
        
        radarCtx.beginPath();
        radarCtx.moveTo(cx - dx, cy - dy);
        radarCtx.lineTo(cx + dx, cy + dy);
        radarCtx.stroke();
    }
}

// Draw live coordinates on Radar Canvas
function drawSpaceRadar(angles) {
    if (!angles || !angles.phi || angles.phi.length === 0) return;
    
    drawSpaceRadarGrid();
    
    const W = radarCanvas.width;
    const H = radarCanvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = Math.min(W, H) / 2 * 0.85;
    
    const count = angles.phi.length;
    
    // Draw the spatial signal signature loop by connecting vectors
    radarCtx.beginPath();
    
    for (let i = 0; i < count; i++) {
        const phi = angles.phi[i];
        const psi = angles.psi[i];
        
        // Scale psi (0 to pi/2) into radius (0 to R)
        const radius = R * (psi / (Math.PI / 2));
        
        // Convert polar coordinates to Cartesian
        const x = cx + radius * Math.cos(phi);
        const y = cy - radius * Math.sin(phi);
        
        if (i === 0) {
            radarCtx.moveTo(x, y);
        } else {
            radarCtx.lineTo(x, y);
        }
    }
    
    // Close path and stroke with a glowing neon cyan-to-purple gradient
    const gradient = radarCtx.createRadialGradient(cx, cy, 5, cx, cy, R);
    gradient.addColorStop(0, "#00f2fe");
    gradient.addColorStop(0.5, "#4facfe");
    gradient.addColorStop(1, "#b100ff");
    
    radarCtx.strokeStyle = gradient;
    radarCtx.lineWidth = 2.5;
    radarCtx.shadowBlur = 12;
    radarCtx.shadowColor = "rgba(0, 242, 254, 0.4)";
    radarCtx.stroke();
    
    // Reset shadow for subsequent drawings
    radarCtx.shadowBlur = 0;
    
    // Draw discrete subcarrier particles on the loop
    for (let i = 0; i < count; i++) {
        const phi = angles.phi[i];
        const psi = angles.psi[i];
        const radius = R * (psi / (Math.PI / 2));
        const x = cx + radius * Math.cos(phi);
        const y = cy - radius * Math.sin(phi);
        
        radarCtx.beginPath();
        radarCtx.arc(x, y, 4, 0, 2 * Math.PI);
        radarCtx.fillStyle = i % 2 === 0 ? "#00f2fe" : "#ff9f43";
        radarCtx.fill();
    }
}

// Compute BFI variance/phase differential for the Heatmap waterfall
function computeHeatmapSlice(angles) {
    if (!angles || !angles.phi || angles.phi.length === 0) return;
    
    const count = angles.phi.length;
    let slice = [];
    
    if (!prevAngles) {
        // Base case: initialize differences to 0
        slice = new Array(count).fill(0.0);
    } else {
        for (let i = 0; i < count; i++) {
            // Compute the absolute difference between consecutive frames
            // accounting for phi phase wrapping (0 to 2pi)
            let diffPhi = Math.abs(angles.phi[i] - prevAngles.phi[i]);
            if (diffPhi > Math.PI) {
                diffPhi = 2 * Math.PI - diffPhi;
            }
            
            let diffPsi = Math.abs(angles.psi[i] - prevAngles.psi[i]);
            
            // Combined weighted metric of variance/fluctuation
            const totalDiff = (diffPhi * 0.7) + (diffPsi * 0.3);
            slice.push(totalDiff);
        }
    }
    
    prevAngles = angles;
    
    // Append slice to history
    heatmapData.push(slice);
    if (heatmapData.length > maxHeatmapCols) {
        heatmapData.shift();
    }
    
    drawHeatmap();
}

// Draw the scrolling spectrogram-style Heatmap Canvas
function drawHeatmap() {
    const W = heatmapCanvas.width;
    const H = heatmapCanvas.height;
    
    heatmapCtx.clearRect(0, 0, W, H);
    
    const totalCols = heatmapData.length;
    if (totalCols === 0) return;
    
    const colWidth = W / maxHeatmapCols;
    const rowHeight = H / heatmapData[0].length;
    
    // Draw columns from left to right (scrolling right-to-left effect)
    // The most recent column is drawn at the right edge
    const startX = W - (totalCols * colWidth);
    
    for (let c = 0; c < totalCols; c++) {
        const slice = heatmapData[c];
        const x = startX + c * colWidth;
        
        for (let r = 0; r < slice.length; r++) {
            const val = slice[r]; // Magnitude of differential fluctuation
            const y = H - (r + 1) * rowHeight; // Low subcarriers at bottom
            
            // Map magnitude to color palette:
            // 0 -> black/dark slate
            // 0.2 -> cyan
            // 0.6 -> amber
            // >1.2 -> white/red (high impact)
            let color = "rgba(8, 12, 24, 0.8)"; // fallback
            
            if (val > 1.2) {
                // High impact, falls
                const alpha = Math.min(1.0, val / 2.0);
                color = `rgba(255, 63, 63, ${alpha})`;
            } else if (val > 0.4) {
                // Movement, walking
                const alpha = Math.min(0.9, (val - 0.4) / 0.8 + 0.3);
                color = `rgba(255, 159, 67, ${alpha})`;
            } else if (val > 0.05) {
                // Slow drift / presence breathing
                const alpha = Math.min(0.5, (val - 0.05) / 0.35 + 0.1);
                color = `rgba(0, 242, 254, ${alpha})`;
            } else {
                // Completely static
                color = "rgba(16, 22, 42, 0.1)";
            }
            
            heatmapCtx.fillStyle = color;
            // Draw slightly larger rect to prevent scaling pixel gaps
            heatmapCtx.fillRect(Math.floor(x), Math.floor(y), Math.ceil(colWidth), Math.ceil(rowHeight));
        }
    }
}

// Draw the delay-domain Channel Impulse Response (CIR)
function drawCIRProfile(profile, dynamicTap) {
    if (!profile || profile.length === 0) return;
    
    const W = cirCanvas.width;
    const H = cirCanvas.height;
    
    cirCtx.clearRect(0, 0, W, H);
    
    const count = profile.length;
    const barWidth = W / count;
    
    // Draw background horizontal grid lines
    cirCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    cirCtx.lineWidth = 1;
    for (let y = 0.25; y <= 1.0; y += 0.25) {
        cirCtx.beginPath();
        cirCtx.moveTo(0, H * y);
        cirCtx.lineTo(W, H * y);
        cirCtx.stroke();
    }
    
    // Find the max value for scaling
    const maxVal = Math.max(...profile, 0.1);
    
    // Draw bars
    for (let i = 0; i < count; i++) {
        const val = profile[i];
        const barHeight = H * (val / maxVal) * 0.82; // leave some headroom for label
        const x = i * barWidth;
        const y = H - barHeight;
        
        cirCtx.beginPath();
        if (i === dynamicTap) {
            // Highlight dynamic tap (Dylign human path) with a pulsing Amber/Orange glow
            cirCtx.fillStyle = "#ff9f43";
            cirCtx.shadowBlur = 12;
            cirCtx.shadowColor = "#ff9f43";
        } else {
            cirCtx.fillStyle = "rgba(0, 242, 254, 0.45)";
            cirCtx.shadowBlur = 0;
        }
        
        cirCtx.fillRect(x + 1, y, Math.max(1, barWidth - 2), barHeight);
        cirCtx.shadowBlur = 0;
    }
    
    // Draw text label on the dynamic tap
    if (dynamicTap >= 0 && dynamicTap < count) {
        cirCtx.fillStyle = "#ff9f43";
        cirCtx.font = "600 10px var(--font-main)";
        const labelX = Math.max(5, Math.min(W - 85, dynamicTap * barWidth - 20));
        const labelY = H - H * (profile[dynamicTap] / maxVal) * 0.82 - 8;
        cirCtx.fillText(`Dylign Peak (Tap ${dynamicTap})`, labelX, Math.max(15, labelY));
    }
}

// Log and event list helper
function appendLog(category, text, type = "info") {
    const container = document.getElementById("eventLogs");
    if (!container) return;
    
    const row = document.createElement("div");
    row.className = `log-row ${type}`;
    
    const timeStr = new Date().toLocaleTimeString();
    
    row.innerHTML = `
        <span class="log-time">[${timeStr}]</span>
        <span class="log-text"><strong>${category}:</strong> ${text}</span>
    `;
    
    container.appendChild(row);
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function clearEventLogs() {
    const container = document.getElementById("eventLogs");
    if (container) {
        container.innerHTML = `
            <div class="log-row info">
                <span class="log-time">[${new Date().toLocaleTimeString()}]</span>
                <span class="log-text"><strong>System:</strong> Log cleared. Monitoring channel state.</span>
            </div>
        `;
    }
}
