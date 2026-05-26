import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scapy.all import Packet
from scapy.layers.dot11 import Dot11
from backend.config import NUM_SUBCARRIERS, NC, NR

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


def get_obfuscation_noise(token: int, seed: int, num_subcarriers: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pseudo-random phi and psi noise vectors for each subcarrier
    using the sounding dialog token and the shared seed.
    """
    rng = np.random.RandomState((seed * 256 + token) & 0xFFFFFFFF)
    phi_noise = rng.uniform(0.0, 2.0 * np.pi, num_subcarriers)
    psi_noise = rng.uniform(0.0, np.pi / 2.0, num_subcarriers)
    return phi_noise, psi_noise
