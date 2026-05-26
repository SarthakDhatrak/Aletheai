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
    const otaGroup = document.getElementById("otaGroup");
    const statusMode = document.getElementById("statusMode");
    
    modeSelect.value = config.mode;
    ifaceInput.value = config.interface;
    
    if (config.mode === "live") {
        ifaceGroup.style.display = "flex";
        otaGroup.style.display = "none";
        statusMode.innerText = `LIVE: ${config.interface}`;
        document.getElementById("simControllerCard").style.opacity = "0.5";
        document.getElementById("simControllerCard").style.pointerEvents = "none";
    } else if (config.mode === "ota") {
        ifaceGroup.style.display = "flex";
        otaGroup.style.display = "block";
        statusMode.innerText = `OTA: ${config.interface} (Ch ${config.ota_channel || 36})`;
        document.getElementById("simControllerCard").style.opacity = "0.5";
        document.getElementById("simControllerCard").style.pointerEvents = "none";
    } else {
        ifaceGroup.style.display = "none";
        otaGroup.style.display = "none";
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
            
            const verStr = config.shield_version === 2 ? `v2 FIR (${config.shield_num_taps} Taps)` : "v1 Phase";
            if (config.shield_authorized) {
                statusShield.innerText = `ACTIVE (${verStr} AUTH)`;
                statusShield.style.color = "var(--emerald-glow)";
            } else {
                statusShield.innerText = `ACTIVE (${verStr} SCRAMBLED)`;
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
    
    if (config.shield_version !== undefined) {
        document.getElementById("shieldVersionSelect").value = config.shield_version;
        const tapsGroup = document.getElementById("shieldTapsGroup");
        if (tapsGroup) {
            tapsGroup.style.display = parseInt(config.shield_version) === 2 ? "block" : "none";
        }
    }
    if (config.shield_num_taps !== undefined) {
        document.getElementById("shieldTapsSlider").value = config.shield_num_taps;
        document.getElementById("shieldTapsValue").innerText = config.shield_num_taps;
    }
    if (config.shield_seed !== undefined) {
        document.getElementById("shieldSeedInput").value = config.shield_seed;
    }
    if (config.shield_authorized !== undefined) {
        document.getElementById("authSelect").value = config.shield_authorized ? "authorized" : "unauthorized";
    }
    
    // OTA configurations
    if (config.ota_channel !== undefined) {
        document.getElementById("otaChannelInput").value = config.ota_channel;
    }
    if (config.ota_trigger_ip !== undefined) {
        document.getElementById("otaTriggerIp").value = config.ota_trigger_ip || "";
    }
    
    // Model engine badge
    const modelBadge = document.getElementById("modelBadge");
    if (modelBadge) {
        if (config.foundation_enabled) {
            modelBadge.innerText = "AM-FM Foundation Encoder";
            modelBadge.style.borderColor = "var(--emerald-glow-dark)";
            modelBadge.style.color = "var(--emerald-glow)";
            document.getElementById("statusEngine").innerText = "Foundation Model";
        } else {
            modelBadge.innerText = "RF Inference Active";
            modelBadge.style.borderColor = "var(--cyan-glow-dark)";
            modelBadge.style.color = "var(--cyan-glow)";
            document.getElementById("statusEngine").innerText = "Random Forest";
        }
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
        const ota_channel = parseInt(document.getElementById("otaChannelInput").value, 10) || 36;
        const ota_trigger_ip = document.getElementById("otaTriggerIp").value.trim() || null;
        
        try {
            const res = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    mode, 
                    interface: interfaceName,
                    ota_channel,
                    ota_trigger_ip
                })
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
    const otaGroup = document.getElementById("otaGroup");
    modeSelect.addEventListener("change", () => {
        if (modeSelect.value === "live") {
            ifaceGroup.style.display = "flex";
            otaGroup.style.display = "none";
        } else if (modeSelect.value === "ota") {
            ifaceGroup.style.display = "flex";
            otaGroup.style.display = "block";
        } else {
            ifaceGroup.style.display = "none";
            otaGroup.style.display = "none";
        }
    });

    // Detect Wi-Fi adapters
    const otaDetectBtn = document.getElementById("otaDetectBtn");
    const otaAdaptersList = document.getElementById("otaAdaptersList");
    if (otaDetectBtn && otaAdaptersList) {
        otaDetectBtn.addEventListener("click", async () => {
            otaDetectBtn.disabled = true;
            otaDetectBtn.innerText = "Searching...";
            try {
                const res = await fetch("/api/ota/adapters");
                const data = await res.json();
                otaAdaptersList.innerHTML = "";
                
                if (!data.adapters || data.adapters.length === 0) {
                    otaAdaptersList.innerHTML = `<div class="adapter-item" style="color: var(--red-glow); font-size: 0.8rem; padding: 0.5rem;">No Wi-Fi adapters detected. Make sure iw/netsh is available.</div>`;
                } else {
                    data.adapters.forEach(adapter => {
                        const item = document.createElement("div");
                        item.className = "adapter-item";
                        item.style.display = "flex";
                        item.style.justifyContent = "space-between";
                        item.style.alignItems = "center";
                        item.style.margin = "0.5rem 0";
                        item.style.padding = "0.5rem";
                        item.style.background = "rgba(255,255,255,0.03)";
                        item.style.borderRadius = "4px";
                        
                        item.innerHTML = `
                            <div>
                                <span style="font-weight:600; color: var(--cyan-glow); font-size: 0.8rem;">${adapter.interface}</span> 
                                <span style="font-size:0.75rem; color: var(--text-secondary);">(${adapter.type || 'unknown'}, ${adapter.mac || 'no mac'})</span>
                            </div>
                            <button type="button" class="action-btn select-iface-btn" data-iface="${adapter.interface}" style="width: auto; padding: 0.25rem 0.5rem; font-size: 0.75rem; background: var(--card-border);">Use</button>
                        `;
                        otaAdaptersList.appendChild(item);
                    });
                    
                    // Add listeners to select button
                    otaAdaptersList.querySelectorAll(".select-iface-btn").forEach(btn => {
                        btn.addEventListener("click", () => {
                            document.getElementById("ifaceInput").value = btn.getAttribute("data-iface");
                            appendLog("OTA", `Selected adapter: ${btn.getAttribute("data-iface")}`, "info");
                        });
                    });
                }
                otaAdaptersList.style.display = "block";
            } catch (err) {
                console.error("Detecting adapters failed", err);
            } finally {
                otaDetectBtn.disabled = false;
                otaDetectBtn.innerText = "🔍 Detect Wi-Fi Adapters";
            }
        });
    }

    // Run OTA validation suite
    const otaValidateBtn = document.getElementById("otaValidateBtn");
    const otaValidationResult = document.getElementById("otaValidationResult");
    if (otaValidateBtn && otaValidationResult) {
        otaValidateBtn.addEventListener("click", async () => {
            const interfaceName = document.getElementById("ifaceInput").value;
            const channel = parseInt(document.getElementById("otaChannelInput").value, 10) || 36;
            const trigger_ip = document.getElementById("otaTriggerIp").value.trim() || null;
            
            if (!interfaceName) {
                alert("Please select or enter a monitor interface first.");
                return;
            }
            
            otaValidateBtn.disabled = true;
            otaValidateBtn.innerText = "Running Validation (10s)...";
            otaValidationResult.innerHTML = `<span style="color: var(--amber-glow); font-family: var(--font-mono); font-size: 0.8rem;">Starting over-the-air capture & validation...</span>`;
            otaValidationResult.style.display = "block";
            
            try {
                const res = await fetch("/api/ota/validate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        interface: interfaceName,
                        duration: 10.0,
                        channel,
                        trigger_ip
                    })
                });
                const data = await res.json();
                
                if (data.status === "completed") {
                    otaValidationResult.innerHTML = `
                        <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.4; color: var(--emerald-glow);">
                            <strong>OTA validation success!</strong><br>
                            • Total packets: ${data.frames_captured}<br>
                            • BFI frames parsed: ${data.bfi_frames_parsed}<br>
                            • Parse success: ${(data.parse_success_rate * 100).toFixed(1)}%<br>
                            • Clients found: ${data.unique_clients.join(", ") || 'none'}
                        </div>
                    `;
                    appendLog("OTA Validation", `Validation completed successfully. Parsed ${data.bfi_frames_parsed} BFI frames.`, "success");
                } else {
                    const errors = data.errors ? data.errors.join("; ") : "Unknown error";
                    otaValidationResult.innerHTML = `
                        <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--red-glow); line-height: 1.4;">
                            <strong>Validation failed:</strong><br>
                            ${errors}
                        </div>
                    `;
                    appendLog("OTA Validation", `Validation failed: ${errors}`, "warn");
                }
            } catch (err) {
                otaValidationResult.innerHTML = `<span style="color: var(--red-glow); font-size: 0.8rem;">Error calling validation API.</span>`;
                appendLog("OTA Validation", "Failed to contact validation API endpoint.", "warn");
            } finally {
                otaValidateBtn.disabled = false;
                otaValidateBtn.innerText = "🧪 Run OTA Validation";
            }
        });
    }
    
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

    // Shield version selector change
    const shieldVersionSelect = document.getElementById("shieldVersionSelect");
    const shieldTapsGroup = document.getElementById("shieldTapsGroup");
    if (shieldVersionSelect && shieldTapsGroup) {
        shieldVersionSelect.addEventListener("change", () => {
            shieldTapsGroup.style.display = parseInt(shieldVersionSelect.value) === 2 ? "block" : "none";
        });
    }

    // Shield taps slider change
    const shieldTapsSlider = document.getElementById("shieldTapsSlider");
    const shieldTapsValue = document.getElementById("shieldTapsValue");
    if (shieldTapsSlider && shieldTapsValue) {
        shieldTapsSlider.addEventListener("input", () => {
            shieldTapsValue.innerText = shieldTapsSlider.value;
        });
    }

    // Security form submission
    const securityForm = document.getElementById("securityForm");
    securityForm.addEventListener("submit", async () => {
        const format = document.getElementById("formatSelect").value;
        const active = document.getElementById("shieldToggle").checked;
        const version = parseInt(document.getElementById("shieldVersionSelect").value, 10) || 2;
        const num_taps = parseInt(document.getElementById("shieldTapsSlider").value, 10) || 5;
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
                    shield_authorized: auth,
                    shield_version: version,
                    shield_num_taps: num_taps
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

window.dismissOodAlert = function() {
    const overlay = document.getElementById("oodAlertOverlay");
    if (overlay) {
        overlay.style.display = "none";
    }
};

// Process incoming WebSocket packet data
function processTelemetry(data) {
    // 1. Update Telemetry Summary
    document.getElementById("statusSrc").innerText = data.src;
    document.getElementById("statusSnr").innerText = `${data.snr} dBm`;
    document.getElementById("statusCarriers").innerText = data.num_subcarriers;
    
    // MQTT Status Update
    const statusMqtt = document.getElementById("statusMqtt");
    if (data.mqtt_connected !== undefined) {
        if (data.mqtt_connected) {
            statusMqtt.innerText = "ACTIVE (Home Assistant)";
            statusMqtt.style.color = "var(--emerald-glow)";
        } else {
            statusMqtt.innerText = "NOT CONNECTED";
            statusMqtt.style.color = "var(--danger)";
        }
    }

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
            const verStr = data.shield_version === 2 ? `v2 (${data.shield_num_taps} Taps)` : "v1";
            if (data.shield_authorized) {
                statusShield.innerText = `ACTIVE (${verStr}-AUTH)`;
                statusShield.style.color = "var(--emerald-glow)";
            } else {
                statusShield.innerText = `ACTIVE (${verStr}-SCRAMBLED)`;
                statusShield.style.color = "var(--red-glow)";
            }
        } else {
            statusShield.innerText = "INACTIVE";
            statusShield.style.color = "var(--text-secondary)";
        }
    }

    // Update Domino SSNR
    const statusSsnr = document.getElementById("statusSsnr");
    if (statusSsnr) {
        statusSsnr.innerText = `${data.domino_ssnr.toFixed(2)} dB`;
        if (data.domino_ssnr > 3.0) {
            statusSsnr.style.color = "var(--emerald-glow)";
        } else if (data.domino_ssnr > 0.0) {
            statusSsnr.style.color = "var(--cyan-glow)";
        } else {
            statusSsnr.style.color = "var(--text-secondary)";
        }
    }

    // Update ADBlock telemetry
    const statusAdblock = document.getElementById("statusAdblock");
    if (statusAdblock) {
        if (data.is_ood) {
            statusAdblock.innerText = "⚠️ ANOMALY DETECTED";
            statusAdblock.style.color = "var(--red-glow)";
            // Show OOD Alert Overlay
            const overlay = document.getElementById("oodAlertOverlay");
            if (overlay) {
                overlay.style.display = "flex";
            }
        } else {
            statusAdblock.innerText = "ACTIVE";
            statusAdblock.style.color = "var(--emerald-glow)";
        }
    }

    // Update OOD probability bar
    const oodBar = document.getElementById("prob_OOD");
    const oodVal = document.getElementById("val_OOD");
    if (oodBar && oodVal) {
        const ratio = data.ood_threshold > 0 ? (data.ood_score / data.ood_threshold) : 0;
        const percent = Math.min(100, Math.round(ratio * 100));
        oodBar.style.width = `${percent}%`;
        oodVal.innerText = `${percent}%`;
        
        if (data.is_ood) {
            oodBar.style.background = "var(--red-glow)";
        } else {
            oodBar.style.background = "var(--amber-glow)";
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
    
    // Request Notification Permission
    if ("Notification" in window) {
        if (Notification.permission !== "granted" && Notification.permission !== "denied") {
            Notification.requestPermission();
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
    drawSpaceRadar(data.angles, data.shield_active && !data.shield_authorized);
    
    // 5. Update Heatmap Buffer
    computeHeatmapSlice(data.angles);
    
    // 6. Update Delay-Domain CIR Visualizer
    drawCIRProfile(data.cir_profile, data.cir_profile_pre, data.cir_dynamic_tap);
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
    } else if (state === "OUT_OF_DISTRIBUTION") {
        orb.classList.add("state-ood");
        predText.innerText = "ANOMALY DETECTED";
        appendLog("ADBlock", "⚠️ ADBlock: Out-of-Distribution anomaly detected (untrained physical event).", "warn");
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
    const ctx = cirCanvas.getContext("2d");

    // Elements for Network Scanner Modal
    const btnScanNetwork = document.getElementById("btnScanNetwork");
    const scannerModal = document.getElementById("scannerModal");
    const btnCloseScanner = document.getElementById("btnCloseScanner");
    const scannerContent = document.getElementById("scannerContent");
    
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
function drawSpaceRadar(angles, isScrambled = false) {
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
        let phi = angles.phi[i];
        let psi = angles.psi[i];
        
        // If scrambled, add random jitter to emphasize scrambled state
        if (isScrambled) {
            phi += (Math.random() - 0.5) * 0.3;
            psi += (Math.random() - 0.5) * 0.1;
        }
        
        // Scale psi. Givens angles (psi) are typically small (0.0 to 0.5 rads).
        // Scaling by 0.5 instead of PI/2 pushes the shapes out from the center so they are clearly visible.
        const psiBounded = Math.max(0, Math.min(0.6, psi));
        const radius = R * (psiBounded / 0.6);
        
        // Convert polar coordinates to Cartesian
        const x = cx + radius * Math.cos(phi);
        const y = cy - radius * Math.sin(phi);
        
        if (i === 0) {
            radarCtx.moveTo(x, y);
        } else {
            radarCtx.lineTo(x, y);
        }
    }
    
    // Close path and stroke with a glowing neon cyan-to-purple gradient (or red/orange if scrambled)
    radarCtx.closePath();
    const gradient = radarCtx.createRadialGradient(cx, cy, 5, cx, cy, R);
    
    if (isScrambled) {
        gradient.addColorStop(0, "rgba(255, 63, 63, 0.9)");
        gradient.addColorStop(1, "rgba(179, 22, 22, 0.4)");
        radarCtx.fillStyle = "rgba(255, 63, 63, 0.2)";
        radarCtx.strokeStyle = gradient;
        radarCtx.lineWidth = 3;
        radarCtx.shadowBlur = 20;
        radarCtx.shadowColor = "rgba(255, 63, 63, 0.8)";
    } else {
        gradient.addColorStop(0, "rgba(0, 242, 254, 0.9)");
        gradient.addColorStop(1, "rgba(131, 58, 180, 0.4)");
        radarCtx.fillStyle = "rgba(0, 242, 254, 0.15)";
        radarCtx.strokeStyle = gradient;
        radarCtx.lineWidth = 3;
        radarCtx.shadowBlur = 20;
        radarCtx.shadowColor = "rgba(0, 242, 254, 0.8)";
    }
    
    radarCtx.fill();
    radarCtx.stroke();
    
    // Reset shadow for subsequent drawings
    radarCtx.shadowBlur = 0;
    
    // Draw discrete subcarrier particles on the loop
    for (let i = 0; i < count; i++) {
        let phi = angles.phi[i];
        let psi = angles.psi[i];
        
        if (isScrambled) {
            phi += (Math.random() - 0.5) * 0.3;
            psi += (Math.random() - 0.5) * 0.1;
        }
        
        const radius = R * (Math.max(0, Math.min(Math.PI / 2, psi)) / (Math.PI / 2));
        const x = cx + radius * Math.cos(phi);
        const y = cy - radius * Math.sin(phi);
        
        radarCtx.beginPath();
        radarCtx.arc(x, y, isScrambled ? 3 : 4, 0, 2 * Math.PI);
        if (isScrambled) {
            radarCtx.fillStyle = Math.random() > 0.5 ? "var(--red-glow)" : "rgba(255, 255, 255, 0.25)";
        } else {
            radarCtx.fillStyle = i % 2 === 0 ? "#00f2fe" : "#ff9f43";
        }
        radarCtx.fill();
    }
    
    // Add extra scanning static overlay if scrambled
    if (isScrambled) {
        radarCtx.strokeStyle = "rgba(255, 63, 63, 0.1)";
        radarCtx.lineWidth = 1;
        for (let j = 0; j < 4; j++) {
            const rx = cx + (Math.random() - 0.5) * R * 2;
            radarCtx.beginPath();
            radarCtx.moveTo(rx, cy - R);
            radarCtx.lineTo(rx, cy + R);
            radarCtx.stroke();
        }
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
function drawCIRProfile(profile, profilePre, dynamicTap) {
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
    
    // Find the max value for scaling across both profiles
    const maxVal = Math.max(...profile, ...(profilePre || []), 0.1);
    
    // 1. Draw "pre-Domino" ghost profile (faded) if available
    if (profilePre && profilePre.length === count) {
        for (let i = 0; i < count; i++) {
            const val = profilePre[i];
            const barHeight = H * (val / maxVal) * 0.82;
            const x = i * barWidth;
            const y = H - barHeight;
            
            cirCtx.fillStyle = "rgba(142, 155, 180, 0.18)";
            cirCtx.fillRect(x + 1, y, Math.max(1, barWidth - 2), barHeight);
        }
    }
    
    // 2. Draw corrected "post-Domino" bars
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
