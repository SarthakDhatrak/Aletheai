import os

# Server Configurations
HOST = "0.0.0.0"
PORT = 8000
WS_PATH = "/ws"

# Sniffer Configurations
DEFAULT_INTERFACE = "wlan0mon"
TARGET_MAC = None  # Set to a specific client MAC to filter, or None for all
SNIFF_TIMEOUT = 1.0  # Timeout in seconds for live sniff loop

# ML Pipeline Configurations
WINDOW_SIZE_SEC = 2.0      # Sliding window length in seconds
MIN_PACKETS_IN_WINDOW = 5   # Minimum BFI frames required to perform inference
SAMPLING_RATE_HZ = 10.0     # Target rate for BFI sampling (for interpolation if needed)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "bfi_classifier.joblib")

# BFI Protocol Parameters
NUM_SUBCARRIERS = 52       # Standard number of subcarriers for 20MHz VHT
NC = 2                     # Number of Tx Antennas (columns in V matrix)
NR = 2                     # Number of Rx Antennas (rows in V matrix)

# Simulator Configuration
SIMULATOR_TICK_RATE_HZ = 10.0 # Frequency of simulated BFI packet arrivals
SIM_NOISE_FLOOR = 0.05       # Standard deviation of thermal noise in simulation
SIM_PACKET_LOSS_RATE = 0.145  # Standard-like 14.5% packet loss rate

# ---------------------------------------------------------------------------
#  Frontier 1: ADBlock — Out-of-Distribution Anomaly Detection
# ---------------------------------------------------------------------------
ADBLOCK_ENABLED = True
ADBLOCK_THRESHOLD_K = 3.0          # Number of standard deviations for static OOD threshold
ADBLOCK_ADAPTIVE_ALPHA = 0.02      # EMA smoothing factor for adaptive threshold
ADBLOCK_HIDDEN_DIM = 12            # Autoencoder hidden layer width
ADBLOCK_LATENT_DIM = 6             # Autoencoder bottleneck dimension
ADBLOCK_LEARNING_RATE = 0.005      # Training learning rate
ADBLOCK_TRAIN_EPOCHS = 200         # Number of training epochs
ADBLOCK_SAVE_PATH = os.path.join(os.path.dirname(__file__), "adblock_weights.npz")

# ---------------------------------------------------------------------------
#  Frontier 2: Domino — Fractional Delay & Hardware Impairment Compensation
# ---------------------------------------------------------------------------
DOMINO_ENABLED = True
DOMINO_REF_TAP_STABILITY_WINDOW = 5  # Consecutive windows to confirm stable LoS tap
CHANNEL_BANDWIDTH_HZ = 20e6          # Channel bandwidth (20MHz for standard VHT)

# ---------------------------------------------------------------------------
#  Frontier 3: Shield v2 — Multi-Tap Convolution Obfuscation
# ---------------------------------------------------------------------------
SHIELD_VERSION = 2                 # 1 = legacy phase offset, 2 = multi-tap convolution
SHIELD_NUM_TAPS = 5                # Number of FIR filter taps (configurable 3-7)
SHIELD_TAP_SPACING = 1.0 / 20e6   # Inter-tap delay spacing (one sample period at 20MHz)

# ---------------------------------------------------------------------------
#  Frontier 4: AM-FM Foundation Encoder (Optional, CPU-only)
# ---------------------------------------------------------------------------
FOUNDATION_ENABLED = os.environ.get("USE_FOUNDATION_MODEL", "false").lower() == "true"
FOUNDATION_EMBED_DIM = 64          # Transformer embedding dimension
FOUNDATION_NUM_HEADS = 4           # Multi-head attention heads
FOUNDATION_DEPTH = 3               # Number of Transformer layers in encoder
FOUNDATION_MASK_RATIO = 0.75       # Fraction of patches to mask during pre-training
FOUNDATION_PRETRAIN_EPOCHS = 50    # Number of pre-training epochs
FOUNDATION_SAVE_PATH = os.path.join(os.path.dirname(__file__), "foundation_encoder.pt")

# ---------------------------------------------------------------------------
#  Frontier 5: OTA Hardware Validation
# ---------------------------------------------------------------------------
OTA_NDP_TRIGGER_INTERVAL = 0.1    # Seconds between ping flood packets
OTA_CHANNEL = 36                   # Default 5GHz channel for monitor mode
OTA_AUTO_CHANNEL_HOP = False       # Whether to cycle through channels automatically
OTA_VALIDATION_DURATION = 10       # Seconds to run OTA validation capture

# ---------------------------------------------------------------------------
#  Feature Vector Dimension (updated with Domino SSNR metric)
# ---------------------------------------------------------------------------
FEATURE_DIM = 24                   # Was 23, now includes Domino SSNR
