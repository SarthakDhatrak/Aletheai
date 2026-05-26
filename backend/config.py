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
