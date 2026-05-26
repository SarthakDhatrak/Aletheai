import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scapy.all import Packet
from scapy.layers.dot11 import Dot11
from backend.config import NUM_SUBCARRIERS, NC, NR, SHIELD_NUM_TAPS, SHIELD_TAP_SPACING, CHANNEL_BANDWIDTH_HZ

class LsbBitReader:
    """
    Reads bits from a byte stream starting from the LSB of the first byte (LSB-first order),
    as specified by the 802.11 standard.
    """
    def __init__(self, data: bytes):
        self.data = data
        self.byte_idx = 0
        self.bit_idx = 0
        self.total_bits = len(data) * 8

    def read_bits(self, num_bits: int) -> int:
        val = 0
        bits_read = 0
        while bits_read < num_bits:
            if self.byte_idx >= len(self.data):
                break
            bits_left_in_byte = 8 - self.bit_idx
            bits_to_read = min(num_bits - bits_read, bits_left_in_byte)
            
            # Extract those bits
            mask = (1 << bits_to_read) - 1
            byte_val = (self.data[self.byte_idx] >> self.bit_idx) & mask
            val |= (byte_val << bits_read)
            
            self.bit_idx += bits_to_read
            bits_read += bits_to_read
            
            if self.bit_idx == 8:
                self.bit_idx = 0
                self.byte_idx += 1
        return val

    def has_bits(self, num_bits: int) -> bool:
        current_bit_pos = self.byte_idx * 8 + self.bit_idx
        return (self.total_bits - current_bit_pos) >= num_bits


def parse_vht_mimo_control(mimo_bytes: bytes) -> Dict[str, Any]:
    """
    Parses the 3-byte VHT MIMO Control field.
    """
    if len(mimo_bytes) < 3:
        return {}
        
    val = int.from_bytes(mimo_bytes[:3], byteorder='little')
    nc = (val & 0x03) + 1                 # Bits 0-1
    nr = ((val >> 2) & 0x03) + 1           # Bits 2-3
    chan_width = (val >> 4) & 0x03         # Bits 4-5 (0: 20MHz, 1: 40MHz, 2: 80MHz, 3: 160MHz)
    grouping = (val >> 6) & 0x03           # Bits 6-7 (0: Ng=1, 1: Ng=2, 2: Ng=4)
    codebook_info = (val >> 8) & 0x01      # Bit 8 (0: 7/5 bits, 1: 9/7 bits)
    feedback_type = (val >> 9) & 0x01      # Bit 9 (0: SU, 1: MU-MIMO)
    
    # Map constants to human readable values
    chan_width_mhz = {0: 20, 1: 40, 2: 80, 3: 160}.get(chan_width, 20)
    grouping_val = {0: 1, 1: 2, 2: 4}.get(grouping, 1)
    
    return {
        "nc": nc,
        "nr": nr,
        "chan_width_mhz": chan_width_mhz,
        "grouping": grouping_val,
        "codebook_info": codebook_info,
        "feedback_type": "MU-MIMO" if feedback_type else "SU-MIMO"
    }


def get_num_subcarriers(chan_width_mhz: int, grouping: int) -> int:
    """
    Returns the expected number of subcarriers based on channel width and grouping Ng.
    """
    # Standard VHT subcarrier configurations
    base_carriers = {20: 52, 40: 114, 80: 242, 160: 484}.get(chan_width_mhz, 52)
    # Grouping reduces subcarrier count
    if grouping == 2:
        return math.ceil(base_carriers / 2.0) + 4  # Approximation of standard Ng=2 subcarriers
    elif grouping == 4:
        return math.ceil(base_carriers / 4.0) + 2  # Approximation of standard Ng=4 subcarriers
    return base_carriers


def unpack_bfi_angles(report_bytes: bytes, nc: int, nr: int, num_subcarriers: int, codebook_info: int) -> List[Dict[str, List[float]]]:
    """
    Unpacks the Givens rotation angles (phi and psi) from the Compressed Beamforming Report payload.
    For Nc, Nr dimensions, the number of phi and psi angles per subcarrier is:
    N_angles = Nc * (2*Nr - Nc - 1) / 2
    For Nc=2, Nr=2: N_angles = 1 (1 phi, 1 psi)
    For Nc=1, Nr=2: N_angles = 1 (1 phi, 1 psi)
    For Nc=2, Nr=3: N_angles = 2 (2 phi, 2 psi)
    """
    phi_bits = 9 if codebook_info else 7
    psi_bits = 7 if codebook_info else 5
    
    phi_step = (2 * math.pi) / (1 << phi_bits)
    psi_step = (math.pi / 2) / (1 << psi_bits)
    
    # Calculate number of phi and psi angles per subcarrier
    # Under standard Givens rotations for Nc columns and Nr rows:
    # We have angles:
    # - phi_ij (for column j from j to Nr-1) -> Nc * (2*Nr - Nc - 1) / 2 angles
    # - psi_ij (for column j from j+1 to Nr) -> Nc * (2*Nr - Nc - 1) / 2 angles
    num_angles = int(nc * (2 * nr - nc - 1) / 2)
    if num_angles <= 0:
        return []
        
    reader = LsbBitReader(report_bytes)
    subcarrier_angles = []
    
    for _ in range(num_subcarriers):
        if not reader.has_bits(num_angles * (phi_bits + psi_bits)):
            break
            
        phis = []
        psis = []
        for _ in range(num_angles):
            phi_val = reader.read_bits(phi_bits)
            psi_val = reader.read_bits(psi_bits)
            
            # Map back to radians
            phi_rad = (phi_val + 0.5) * phi_step
            psi_rad = (psi_val + 0.5) * psi_step
            
            phis.append(phi_rad)
            psis.append(psi_rad)
            
        subcarrier_angles.append({
            "phi": phis,
            "psi": psis
        })
        
    return subcarrier_angles


def parse_raw_bfi_payload(payload: bytes) -> Optional[Dict[str, Any]]:
    """
    Parses a raw BFI action payload directly.
    Supports Category 21 (VHT), 26 (HE), and 33 (IEEE 802.11bf Sensing Category).
    """
    if len(payload) < 5:
        return None
        
    category = payload[0]
    action = payload[1]
    
    # Action Category 21: VHT, Action 0: Compressed Beamforming
    # Category 26 is HE Action (802.11ax), Action 0: Compressed Beamforming
    # Category 33 is SENS Action (802.11bf), Action 0: SENS Sounding Feedback
    if category not in (21, 26, 33) or action != 0:
        return None
        
    mimo_control = payload[2:5]
    sounding_token = payload[5]
    report_payload = payload[6:]
    
    mimo_info = parse_vht_mimo_control(mimo_control)
    if not mimo_info:
        return None
        
    num_subcarriers = get_num_subcarriers(mimo_info["chan_width_mhz"], mimo_info["grouping"])
    angles = unpack_bfi_angles(
        report_payload, 
        mimo_info["nc"], 
        mimo_info["nr"], 
        num_subcarriers, 
        mimo_info["codebook_info"]
    )
    
    return {
        "category": category,
        "mimo_control": mimo_info,
        "sounding_dialog_token": sounding_token,
        "angles": angles,
        "num_subcarriers": len(angles)
    }


def parse_bfi_packet(pkt: Packet) -> Optional[Dict[str, Any]]:
    """
    Parses a Scapy packet. Checks if it is an 802.11 Action frame containing BFI.
    """
    if not pkt.haslayer(Dot11):
        return None
        
    # We want Action frames: type=0 (Management), subtype=13 or 14
    dot11 = pkt[Dot11]
    if dot11.type != 0 or dot11.subtype not in (13, 14):
        return None
        
    # Retrieve management payload directly from the 802.11 layer
    payload = bytes(dot11.payload)
        
    parsed = parse_raw_bfi_payload(payload)
    if parsed:
        # Enrich with packet metadata
        parsed["src"] = dot11.addr2
        parsed["dst"] = dot11.addr1
        parsed["bssid"] = dot11.addr3
        parsed["snr"] = getattr(pkt, "dBm_AntSignal", -60)  # Default if RadioTap isn't present
        return parsed
        
    return None


def apply_hampel_filter_2d(data: np.ndarray, k: int = 3, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Applies Hampel filter to a 2D array along the time axis (axis 0).
    Replaces outliers with rolling median.
    """
    T, F = data.shape
    filtered = data.copy()
    for i in range(T):
        start = max(0, i - k)
        end = min(T, i + k + 1)
        window = data[start:end, :]
        median = np.median(window, axis=0)
        mad = np.median(np.abs(window - median), axis=0)
        
        # Threshold
        threshold = n_sigmas * 1.4826 * mad
        diff = np.abs(data[i, :] - median)
        
        # Replace outliers (where diff > threshold and mad > 1e-6 to avoid dividing by 0)
        outliers = (diff > threshold) & (mad > 1e-6)
        filtered[i, outliers] = median[outliers]
    return filtered


def unwrap_and_detrend_phases(phi_matrix: np.ndarray) -> np.ndarray:
    """
    Unwraps phase angles along the time axis (axis 0) to remove 2pi jumps,
    then applies a linear detrend (CFO removal) to remove carrier drift.
    phi_matrix is T x F.
    """
    T, F = phi_matrix.shape
    # 1. Unwrap phases along time axis
    unwrapped = np.unwrap(phi_matrix, axis=0)
    # 2. Linear fit and detrend (remove slope but preserve intercept offset)
    detrended = np.zeros_like(unwrapped)
    t = np.arange(T)
    for f in range(F):
        # Fit line: y = slope * t + intercept
        slope, intercept = np.polyfit(t, unwrapped[:, f], 1)
        # Detrend: subtract slope * t to remove drift slope but keep relative offsets
        detrended[:, f] = unwrapped[:, f] - slope * t
    return detrended


# ==========================================================================
#  Shield v1 — Legacy Phase Offset Obfuscation (backward compatible)
# ==========================================================================

def get_v1_obfuscation_noise(token: int, seed: int, num_subcarriers: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shield v1: Generates pseudo-random phi and psi noise vectors for each subcarrier
    using the sounding dialog token and the shared seed.
    """
    rng = np.random.RandomState((seed * 256 + token) & 0xFFFFFFFF)
    phi_noise = rng.uniform(0.0, 2.0 * np.pi, num_subcarriers)
    psi_noise = rng.uniform(0.0, np.pi / 2.0, num_subcarriers)
    return phi_noise, psi_noise


# Backward compatibility alias
get_obfuscation_noise = get_v1_obfuscation_noise


# ==========================================================================
#  Shield v2 — Multi-Tap Convolution Obfuscation
# ==========================================================================

def generate_multitap_filter(
    token: int,
    seed: int,
    num_subcarriers: int,
    num_taps: int = SHIELD_NUM_TAPS,
    tap_spacing: float = SHIELD_TAP_SPACING,
    bandwidth: float = CHANNEL_BANDWIDTH_HZ
) -> np.ndarray:
    """
    Shield v2: Generates a deterministic, time-varying, frequency-selective
    multi-tap FIR convolution filter for BFI obfuscation.

    The filter G[f] at subcarrier index f is:
        G[f] = Σ_{k=0}^{K-1} g_k · exp(-j·2π·f·k·Δτ·BW/N)

    where g_k are complex tap coefficients derived from PRNG(seed, token).

    This creates an artificial, dynamic multipath environment that destroys
    the mathematical consistency multi-antenna MLE solvers rely on.

    Args:
        token: Sounding dialog token (time-varying seed component)
        seed: Shared secret encryption seed
        num_subcarriers: Number of subcarriers
        num_taps: Number of FIR filter taps (K)
        tap_spacing: Inter-tap delay spacing in seconds
        bandwidth: Channel bandwidth in Hz

    Returns:
        Complex filter array of shape (num_subcarriers,)
    """
    # Deterministic PRNG from combined seed + token
    combined_seed = (seed * 65537 + token * 257 + 0xDEADBEEF) & 0xFFFFFFFF
    rng = np.random.RandomState(combined_seed)

    # Generate complex tap coefficients with random amplitude and phase
    # Amplitude: uniform [0.3, 1.0] to ensure each tap contributes meaningfully
    # Phase: uniform [0, 2π] for full phase randomization
    tap_amplitudes = rng.uniform(0.3, 1.0, num_taps)
    tap_phases = rng.uniform(0.0, 2.0 * np.pi, num_taps)
    g_taps = tap_amplitudes * np.exp(1j * tap_phases)

    # Normalize the filter so it doesn't change overall signal power dramatically
    g_taps = g_taps / np.sqrt(np.sum(np.abs(g_taps) ** 2))

    # Compute frequency response G[f] for each subcarrier
    f_indices = np.arange(num_subcarriers)
    # Normalized delay per tap: k * Δτ * BW expressed in samples
    delay_per_tap = tap_spacing * bandwidth  # In sample units

    G = np.zeros(num_subcarriers, dtype=complex)
    for k in range(num_taps):
        phase_shift = -2.0 * np.pi * f_indices * k * delay_per_tap / num_subcarriers
        G += g_taps[k] * np.exp(1j * phase_shift)

    # Project response to the unit circle to ensure it behaves as an all-pass filter (phase-only scramble),
    # which preserves the magnitude of the steering vectors and enables 100% perfect reconstruction.
    G_unit = G / (np.abs(G) + 1e-12)
    return G_unit


def apply_shield_v2_obfuscation(
    phi_array: np.ndarray,
    psi_array: np.ndarray,
    filter_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shield v2: Applies multi-tap convolution obfuscation to BFI angles.

    Converts (φ, ψ) → complex steering vector v21 = sin(ψ)·exp(jφ),
    multiplies by filter G[f], then re-extracts obfuscated (φ', ψ').

    Args:
        phi_array: φ angles per subcarrier, shape (N,)
        psi_array: ψ angles per subcarrier, shape (N,)
        filter_coeffs: Complex filter G[f], shape (N,)

    Returns:
        Tuple of (phi_obfuscated, psi_obfuscated)
    """
    # Reconstruct complex steering vector
    v21 = np.sin(psi_array) * np.exp(1j * phi_array)

    # Apply multi-tap convolution filter
    v21_obf = v21 * filter_coeffs

    # Re-extract angles
    phi_obf = np.angle(v21_obf) % (2.0 * np.pi)
    psi_obf = np.arcsin(np.clip(np.abs(v21_obf), 0.0, 1.0))

    return phi_obf, psi_obf


def invert_shield_v2_obfuscation(
    phi_obf: np.ndarray,
    psi_obf: np.ndarray,
    filter_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shield v2: Inverts the multi-tap convolution for authorized receivers.

    Divides by the filter: v21_recovered = v21_obfuscated / G[f]

    Args:
        phi_obf: Obfuscated φ angles, shape (N,)
        psi_obf: Obfuscated ψ angles, shape (N,)
        filter_coeffs: Complex filter G[f], shape (N,)

    Returns:
        Tuple of (phi_recovered, psi_recovered)
    """
    # Reconstruct obfuscated steering vector
    v21_obf = np.sin(psi_obf) * np.exp(1j * phi_obf)

    # Invert filter (divide)
    # Guard against division by zero
    safe_filter = np.where(np.abs(filter_coeffs) > 1e-10, filter_coeffs, 1e-10)
    v21_recovered = v21_obf / safe_filter

    # Re-extract angles
    phi_rec = np.angle(v21_recovered) % (2.0 * np.pi)
    psi_rec = np.arcsin(np.clip(np.abs(v21_recovered), 0.0, 1.0))

    return phi_rec, psi_rec


def get_shield_version(shield_active: bool, shield_version: int = 2) -> int:
    """Returns the active shield version (0 = off, 1 = legacy, 2 = multi-tap)."""
    if not shield_active:
        return 0
    return shield_version


def test_shield_v2():
    """
    Self-validation: verifies that authorized descrambling recovers original
    angles within quantization error, and unauthorized receiver sees noise.
    """
    from scipy.stats import kstest
    print("Running Shield v2 self-test...")

    seed = 42
    token = 128
    N = 52
    num_taps = 5

    # Generate original angles
    rng = np.random.RandomState(99)
    phi_orig = rng.uniform(0.0, 2.0 * np.pi, N)
    psi_orig = rng.uniform(0.0, np.pi / 2.0, N)

    # Generate filter
    G = generate_multitap_filter(token, seed, N, num_taps)

    # Obfuscate
    phi_obf, psi_obf = apply_shield_v2_obfuscation(phi_orig, psi_orig, G)

    # Authorized recovery
    phi_rec, psi_rec = invert_shield_v2_obfuscation(phi_obf, psi_obf, G)

    # Check recovery accuracy
    phi_error = np.max(np.abs(phi_rec - phi_orig))
    psi_error = np.max(np.abs(psi_rec - psi_orig))

    print(f"  Max phi recovery error: {phi_error:.8f} rad")
    print(f"  Max psi recovery error: {psi_error:.8f} rad")

    # Check unauthorized view (should look like noise)
    # Test if obfuscated phi is uniformly distributed (KS test vs uniform [0, 2pi])
    ks_stat, ks_pval = kstest(phi_obf / (2.0 * np.pi), 'uniform')
    print(f"  KS test (phi_obf vs uniform): stat={ks_stat:.4f}, p={ks_pval:.4f}")

    passed = phi_error < 1e-6 and psi_error < 1e-6
    print(f"  Recovery test: {'PASSED' if passed else 'FAILED'}")
    print(f"  Noise quality: {'GOOD' if ks_pval > 0.05 else 'WARN (not sufficiently uniform)'}")
    print("Shield v2 self-test complete.")


if __name__ == "__main__":
    test_shield_v2()
