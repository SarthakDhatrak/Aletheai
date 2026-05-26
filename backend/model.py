import os
import joblib
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from backend.config import MODEL_SAVE_PATH, WINDOW_SIZE_SEC, SAMPLING_RATE_HZ, MIN_PACKETS_IN_WINDOW

def interpolate_packets_to_grid(packets: List[Dict[str, Any]], target_len: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates a sequence of packets with missing packets/irregular timestamps
    onto a regular grid of target_len (representing 2 seconds at 10Hz).
    """
    from scipy.interpolate import interp1d
    T = len(packets)
    F = packets[0]["num_subcarriers"]
    
    # 1. Extract timestamps and values
    t_vals = np.array([p["timestamp"] for p in packets])
    
    phi_vals = np.zeros((T, F))
    psi_vals = np.zeros((T, F))
    
    for t_idx, pkt in enumerate(packets):
        for f_idx, subcarrier in enumerate(pkt["angles"]):
            if f_idx < F:
                phi_vals[t_idx, f_idx] = subcarrier["phi"][0] if subcarrier["phi"] else 0.0
                psi_vals[t_idx, f_idx] = subcarrier["psi"][0] if subcarrier["psi"] else 0.0
                
    # 2. Handle unwrapping on phi before interpolating to avoid edge jump interpolation issues
    phi_unwrapped = np.unwrap(phi_vals, axis=0)
    
    # 3. Create regular time grid from start to end timestamp
    t_grid = np.linspace(t_vals[0], t_vals[-1], target_len)
    
    # 4. Interpolate
    phi_grid = np.zeros((target_len, F))
    psi_grid = np.zeros((target_len, F))
    
    # Use linear interpolation to avoid spline overshoot and Runge's phenomenon artifacts
    kind = 'linear'
    
    for f in range(F):
        # Interpolate phi (unwrapped)
        f_phi = interp1d(t_vals, phi_unwrapped[:, f], kind=kind, fill_value="extrapolate")
        phi_grid[:, f] = f_phi(t_grid)
        
        # Interpolate psi
        f_psi = interp1d(t_vals, psi_vals[:, f], kind=kind, fill_value="extrapolate")
        # Keep psi within standard [0, pi/2] bounds
        psi_grid[:, f] = np.clip(f_psi(t_grid), 0.0, np.pi/2)
        
    return phi_grid, psi_grid


def compute_cir_and_dynamic_tap(phi_matrix: np.ndarray, psi_matrix: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Computes the Channel Impulse Response (CIR) using IFFT on reconstructed steering vectors,
    identifies the dynamic path index (Dylign) with highest variance,
    and returns the 2D CIR magnitude matrix, the dynamic tap index, and the average CIR profile.
    """
    T, F = phi_matrix.shape
    
    # 1. Reconstruct steering coefficient v21 = sin(psi) * exp(j * phi)
    v21 = np.sin(psi_matrix) * np.exp(1j * phi_matrix)
    
    # 2. Compute IFFT along the subcarrier axis (axis 1) to get the delay domain CIR
    cir = np.fft.ifft(v21, n=F, axis=1)
    cir_abs = np.abs(cir)
    
    # 3. Dynamic Path Alignment (Dylign): find the tap with the highest temporal variance
    tap_variances = np.var(cir_abs, axis=0)
    
    # Exclude DC (tap 0) and very close reflections (taps 1-2) which are dominated by direct-path leakage
    if F > 4:
        dynamic_tap = int(3 + np.argmax(tap_variances[3:]))
    else:
        dynamic_tap = int(np.argmax(tap_variances))
        
    # 4. Average CIR amplitude profile over the window
    avg_cir_profile = np.mean(cir_abs, axis=0)
    
    return cir_abs, dynamic_tap, avg_cir_profile


def extract_features_from_window(packets: List[Dict[str, Any]], layout: Tuple[float, float, float] = (4.0, 0.0, 1.5), return_cir: bool = False) -> Any:
    """
    Extracts statistical and temporal features from a sliding window of parsed BFI packets.
    First interpolates the packets onto a regular grid to mitigate packet loss,
    applies Hampel filter and phase-unwrapping detrending, projects to delay domain (CIR),
    identifies dynamic peak paths, and extracts features.
    """
    T = len(packets)
    if T < 2:
        empty_feat = np.zeros(23)
        return (empty_feat, {"avg_profile": [], "dynamic_tap": 0}) if return_cir else empty_feat
        
    # Get subcarrier count from first packet
    F = packets[0]["num_subcarriers"]
    if F == 0:
        empty_feat = np.zeros(23)
        return (empty_feat, {"avg_profile": [], "dynamic_tap": 0}) if return_cir else empty_feat
        
    # 1. Interpolate packets onto regular 20-sample grid (Cubic/Linear spline)
    phi_grid, psi_grid = interpolate_packets_to_grid(packets, target_len=20)
    
    # 2. Apply Hampel Filter (rolling outlier removal)
    from backend.parser import apply_hampel_filter_2d, unwrap_and_detrend_phases
    phi_filtered = apply_hampel_filter_2d(phi_grid, k=3, n_sigmas=3.0)
    psi_filtered = apply_hampel_filter_2d(psi_grid, k=3, n_sigmas=3.0)
    
    # 3. Phase Unwrap and Detrend (CFO removal)
    phi_detrended = unwrap_and_detrend_phases(phi_filtered)
    
    # 4. Compute CIR Delay Domain Profiles
    cir_abs, dynamic_tap, avg_cir_profile = compute_cir_and_dynamic_tap(phi_detrended, psi_filtered)
    
    # 5. Temporal Variance
    phi_var = np.var(phi_detrended, axis=0)
    psi_var = np.var(psi_filtered, axis=0)
    
    mean_phi_var = np.mean(phi_var)
    max_phi_var = np.max(phi_var)
    std_phi_var = np.std(phi_var)
    
    mean_psi_var = np.mean(psi_var)
    max_psi_var = np.max(psi_var)
    std_psi_var = np.std(psi_var)
    
    # 6. Temporal Mean Absolute Differences (MAD)
    phi_diff = np.abs(np.diff(phi_detrended, axis=0))
    psi_diff = np.abs(np.diff(psi_filtered, axis=0))
    
    mean_phi_diff = np.mean(phi_diff)
    max_phi_diff = np.max(phi_diff)
    
    mean_psi_diff = np.mean(psi_diff)
    max_psi_diff = np.max(psi_diff)
    
    # 7. Overall Range
    phi_range = np.mean(np.max(phi_detrended, axis=0) - np.min(phi_detrended, axis=0))
    psi_range = np.mean(np.max(psi_filtered, axis=0) - np.min(psi_filtered, axis=0))
    
    # 8. Phase coherence across adjacent subcarriers (Vectorized for 100x speedup)
    phi_corr = 0.0
    psi_corr = 0.0
    if F > 1:
        # Subtract mean along time axis (axis 0)
        phi_diff_mean = phi_detrended - np.mean(phi_detrended, axis=0)
        psi_diff_mean = psi_filtered - np.mean(psi_filtered, axis=0)
        
        # Compute covariance of adjacent subcarriers
        cov_phi = np.sum(phi_diff_mean[:, :-1] * phi_diff_mean[:, 1:], axis=0)
        cov_psi = np.sum(psi_diff_mean[:, :-1] * psi_diff_mean[:, 1:], axis=0)
        
        # Compute sum of squares (variance numerator) of each subcarrier
        var_phi = np.sum(phi_diff_mean ** 2, axis=0)
        var_psi = np.sum(psi_diff_mean ** 2, axis=0)
        
        # Standard deviation product denominator
        std_prod_phi = np.sqrt(var_phi[:-1] * var_phi[1:])
        std_prod_psi = np.sqrt(var_psi[:-1] * var_psi[1:])
        
        # Avoid division by zero, compute correlation coefficient
        phi_corrs = np.divide(cov_phi, std_prod_phi, out=np.zeros_like(cov_phi), where=std_prod_phi > 1e-8)
        psi_corrs = np.divide(cov_psi, std_prod_psi, out=np.zeros_like(cov_psi), where=std_prod_psi > 1e-8)
        
        phi_corr = np.mean(phi_corrs)
        psi_corr = np.mean(psi_corrs)
        
    # 9. CIR / Dylign Features
    mean_cir_var = np.mean(np.var(cir_abs, axis=0))
    dyn_tap_var = np.var(cir_abs[:, dynamic_tap])
    dyn_tap_mad = np.mean(np.abs(np.diff(cir_abs[:, dynamic_tap], axis=0)))
    dyn_tap_range = np.max(cir_abs[:, dynamic_tap]) - np.min(cir_abs[:, dynamic_tap])
        
    # Normalize layout parameters
    d, az, h = layout
    d_norm = np.clip(d / 10.0, 0.0, 1.0)
    az_norm = np.clip(az / 180.0, -1.0, 1.0)
    h_norm = np.clip(h / 4.0, 0.0, 1.0)
        
    features = np.array([
        mean_phi_var, max_phi_var, std_phi_var,
        mean_psi_var, max_psi_var, std_psi_var,
        mean_phi_diff, max_phi_diff,
        mean_psi_diff, max_psi_diff,
        phi_range, psi_range,
        phi_corr, psi_corr,
        mean_cir_var, dyn_tap_var, dyn_tap_mad, dyn_tap_range,
        float(20.0), float(F),
        float(d_norm), float(az_norm), float(h_norm)
    ])
    
    # Clean features of NaN or Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    if return_cir:
        cir_data = {
            "avg_profile": avg_cir_profile.tolist(),
            "dynamic_tap": dynamic_tap
        }
        return features, cir_data
    return features


class BFIClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        self.is_trained = False
        self.classes = ["EMPTY", "PRESENCE", "WALKING", "FALLING"]

    def train(self, X: np.ndarray, y: List[str]):
        """
        Trains the random forest classifier.
        """
        self.model.fit(X, y)
        self.is_trained = True
        self.save()

    def predict(self, features: np.ndarray) -> str:
        if not self.is_trained:
            return "UNKNOWN"
        pred = self.model.predict(features.reshape(1, -1))
        return pred[0]

    def predict_proba(self, features: np.ndarray) -> Dict[str, float]:
        if not self.is_trained:
            return {c: 0.0 for c in self.classes}
        probas = self.model.predict_proba(features.reshape(1, -1))[0]
        # model.classes_ might be in a different order than self.classes, map correctly
        class_probas = {}
        for idx, cls in enumerate(self.model.classes_):
            class_probas[cls] = float(probas[idx])
        return class_probas

    def save(self):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump((self.model, self.is_trained), MODEL_SAVE_PATH)

    def load(self) -> bool:
        if os.path.exists(MODEL_SAVE_PATH):
            try:
                self.model, self.is_trained = joblib.load(MODEL_SAVE_PATH)
                return self.is_trained
            except Exception:
                self.is_trained = False
        return False


def generate_synthetic_training_data() -> Tuple[np.ndarray, List[str]]:
    """
    Generates synthetic BFI packet windows using BFISimulator for training purposes,
    randomizing environmental layout parameters to prevent overfitting.
    """
    from backend.simulator import BFISimulator
    from backend.parser import parse_raw_bfi_payload
    import random
    
    sim = BFISimulator()
    states = ["EMPTY", "PRESENCE", "WALKING", "FALLING"]
    X = []
    y = []
    
    # For each state, generate multiple overlapping sliding windows
    packets_per_sec = int(SAMPLING_RATE_HZ)
    window_length = int(WINDOW_SIZE_SEC * SAMPLING_RATE_HZ)
    
    print("Generating synthetic training data for BFI ML classifier...")
    
    for state in states:
        # Simulate multiple segments with randomized layouts
        num_segments = 8 if state == "FALLING" else 4
        segment_duration = 10 if state == "FALLING" else 15
        
        for seg in range(num_segments):
            # Randomize layout parameters
            dist = random.uniform(2.0, 8.0)
            az = random.uniform(-45.0, 45.0)
            height = random.uniform(0.5, 2.5)
            
            sim.layout_distance = dist
            sim.layout_azimuth = az
            sim.layout_height = height
            sim.set_state(state)
            
            # Reset simulator time offsets
            sim.time_offset = 0.0
            
            # Buffer to hold recent packets
            packet_buffer = []
            
            num_steps = int(segment_duration * SAMPLING_RATE_HZ)
            dt = 1.0 / SAMPLING_RATE_HZ
            
            for step in range(num_steps):
                # If state is FALLING, reset fall at start of segment
                if state == "FALLING" and step == 0:
                    sim.set_state("FALLING")
                    sim.time_offset = 0.0
                    
                sim.update_physics(dt)
                payload = sim.generate_packet_payload()
                parsed = parse_raw_bfi_payload(payload)
                
                if parsed:
                    parsed["timestamp"] = step * dt
                    # Simulate packet loss during training (e.g. 15% packet loss)
                    if random.random() >= 0.15:
                        packet_buffer.append(parsed)
                        if len(packet_buffer) > window_length:
                            packet_buffer.pop(0)
                        
                    if len(packet_buffer) == window_length:
                        # Extract features using this segment's layout parameters
                        feat = extract_features_from_window(packet_buffer, layout=(dist, az, height))
                        X.append(feat)
                        y.append(state)
                        
    return np.array(X), y


def get_trained_classifier() -> BFIClassifier:
    """
    Loads a saved classifier or trains a new one using synthetic data.
    """
    clf = BFIClassifier()
    if clf.load():
        # Validate model shape matches layout priors feature size
        if hasattr(clf.model, "n_features_in_") and clf.model.n_features_in_ == 23:
            print("Successfully loaded pre-trained BFI classifier model.")
            return clf
        else:
            print("Pre-trained model feature dimension mismatch. Retraining...")
            
    print("Pre-trained classifier not found. Training model on synthetic data...")
    X, y = generate_synthetic_training_data()
    clf.train(X, y)
    print("BFI ML model training complete and saved.")
    return clf
