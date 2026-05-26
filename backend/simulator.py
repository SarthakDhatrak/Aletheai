import time
import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple
from backend.config import (
    NUM_SUBCARRIERS, NC, NR, SIM_NOISE_FLOOR,
    SHIELD_VERSION, SHIELD_NUM_TAPS, SHIELD_TAP_SPACING, CHANNEL_BANDWIDTH_HZ
)

class LsbBitWriter:
    """
    Writes bits into a bytearray in LSB-first order.
    """
    def __init__(self):
        self.data = bytearray()
        self.current_byte = 0
        self.bit_idx = 0

    def write_bits(self, val: int, num_bits: int):
        for i in range(num_bits):
            bit = (val >> i) & 1
            self.current_byte |= (bit << self.bit_idx)
            self.bit_idx += 1
            if self.bit_idx == 8:
                self.data.append(self.current_byte)
                self.current_byte = 0
                self.bit_idx = 0

    def flush(self) -> bytes:
        if self.bit_idx > 0:
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bit_idx = 0
        return bytes(self.data)


class BFISimulator:
    def __init__(self):
        self.state = "EMPTY"  # EMPTY, PRESENCE, WALKING, FALLING
        self.state_start_time = time.time()
        self.time_offset = 0.0
        
        # Layout parameters (PerceptAlign)
        self.layout_distance = 4.0
        self.layout_azimuth = 0.0
        self.layout_height = 1.5
        
        # Subcarrier frequencies (20MHz VHT at 5.18 GHz, 52 data carriers)
        self.f_c = 5.18e9
        self.df = 312.5e3
        self.carrier_indices = np.array([i for i in range(-28, 29) if i != 0][:NUM_SUBCARRIERS])
        self.frequencies = self.f_c + self.carrier_indices * self.df
        self.speed_of_light = 3.0e8
        
        # Initialize static environment paths (multipath)
        self.num_static_paths = 6
        # Distances of reflections (in meters)
        self.static_distances = [4.0, 6.2, 8.5, 10.1, 12.4, 15.0]
        # Path losses
        self.static_amplitudes = [0.8, 0.4, 0.25, 0.18, 0.12, 0.08]
        # Angles of Arrival (AoA) and Angles of Departure (AoD) (in radians)
        self.static_aoa = [0.1, -0.4, 0.6, -0.8, 0.3, -0.5]
        self.static_aod = [-0.1, 0.3, -0.5, 0.7, -0.2, 0.4]
        
        # Dynamic state variables for occupant
        self.fall_progress = 0.0  # 0.0 to 1.0 during a fall event
        self.human_position = 5.0 # Distance of human reflection
        self.human_speed = 0.0
        self.human_aoa = 0.2
        self.human_aod = -0.2
        self.human_reflection_coeff = 0.0 # 0 when empty

    def set_state(self, state: str):
        if state not in ("EMPTY", "PRESENCE", "WALKING", "FALLING"):
            return
        self.state = state
        self.state_start_time = self.time_offset
        self.fall_progress = 0.0
        
        if state == "EMPTY":
            self.human_reflection_coeff = 0.0
            self.human_speed = 0.0
        elif state == "PRESENCE":
            self.human_reflection_coeff = 0.25
            self.human_speed = 0.0
            self.human_position = self.layout_distance + 0.5
        elif state == "WALKING":
            self.human_reflection_coeff = 0.3
            self.human_speed = 1.1  # Walking speed in m/s
            self.human_position = self.layout_distance * 0.9
        elif state == "FALLING":
            self.human_reflection_coeff = 0.35
            self.human_speed = 0.5  # Starts slow, increases rapidly
            self.human_position = self.layout_distance * 0.8

    def update_physics(self, dt: float):
        self.time_offset += dt
        
        if self.state == "EMPTY":
            pass
        elif self.state == "PRESENCE":
            # Subtle breathing motion (micro-movements of 1-2 cm at 0.25 Hz)
            breathing_amplitude = 0.015
            breathing_freq = 0.25
            base_pos = self.layout_distance + 0.5
            self.human_position = base_pos + breathing_amplitude * math.sin(2 * math.pi * breathing_freq * self.time_offset)
        elif self.state == "WALKING":
            # Walk back and forth in a range relative to layout distance
            min_pos = max(1.5, self.layout_distance * 0.7)
            max_pos = self.layout_distance * 1.5
            self.human_position += self.human_speed * dt
            if self.human_position > max_pos:
                self.human_position = max_pos
                self.human_speed = -1.1
            elif self.human_position < min_pos:
                self.human_position = min_pos
                self.human_speed = 1.1
            self.human_aoa += 0.05 * dt * math.sin(self.time_offset)
            self.human_aod -= 0.05 * dt * math.sin(self.time_offset)
        elif self.state == "FALLING":
            # Fall is structured into:
            # 1. Acceleration: 0.0 to 0.7s (speed goes from 0.5 to 3.5 m/s)
            # 2. Impact: 0.7 to 0.9s (speed drops to 0, high perturbation)
            # 3. Lying down: > 0.9s (stable presence on the floor)
            elapsed = self.time_offset - self.state_start_time
            if elapsed < 0.7:
                # Acceleration phase
                self.human_speed = 0.5 + 4.0 * elapsed # acceleration
                self.human_position += self.human_speed * dt
                self.human_reflection_coeff = 0.35 + 0.1 * math.sin(elapsed * 10)
            elif elapsed < 1.0:
                # Impact/Crash phase
                self.human_speed = max(0.0, self.human_speed - 15.0 * dt)
                self.human_position += self.human_speed * dt
                # Rapid channel fluctuation due to impact
                self.human_reflection_coeff = 0.45 * (1.0 - (elapsed - 0.7)/0.3)
            else:
                # Lying down on the floor (Static presence at a new, lower reflection path)
                self.human_speed = 0.0
                self.human_position = self.layout_distance * 1.4
                self.human_reflection_coeff = 0.15 + 0.01 * math.sin(self.time_offset * 0.1) # low height, smaller reflection
                # Once fall is completed, keep it in static presence but with fallen flag
                pass

    def compute_channel_matrix(self, f: float) -> np.ndarray:
        """
        Computes the NR x NC channel matrix H for a specific subcarrier frequency.
        """
        # H is NR x NC (Rx antennas x Tx antennas). For standard naming, NC is columns, NR is rows.
        H = np.zeros((NC, NR), dtype=complex) # 2x2 matrix
        
        # 1. Direct Path (LoS)
        direct_dist = self.layout_distance
        direct_phase = -2 * math.pi * f * direct_dist / self.speed_of_light
        # Simple steering vector representation for Tx and Rx shifted by azimuth
        az_rad = math.radians(self.layout_azimuth)
        tx_steer = np.array([1.0, np.exp(1j * math.pi * math.sin(az_rad))], dtype=complex)
        rx_steer = np.array([1.0, np.exp(1j * math.pi * math.sin(az_rad))], dtype=complex)
        H += 1.0 * np.exp(1j * direct_phase) * np.outer(rx_steer, tx_steer.conj())
        
        # 2. Static Multipath
        for i in range(self.num_static_paths):
            # Scale static distances by layout distance relative to baseline 4.0m
            dist = self.static_distances[i] * (self.layout_distance / 4.0)
            phase = -2 * math.pi * f * dist / self.speed_of_light
            tx_steer = np.array([1.0, np.exp(1j * math.pi * math.sin(self.static_aod[i]))], dtype=complex)
            rx_steer = np.array([1.0, np.exp(1j * math.pi * math.sin(self.static_aoa[i]))], dtype=complex)
            H += self.static_amplitudes[i] * np.exp(1j * phase) * np.outer(rx_steer, tx_steer.conj())
            
        # 3. Dynamic Human Path
        if self.human_reflection_coeff > 0.0:
            phase = -2 * math.pi * f * self.human_position / self.speed_of_light
            tx_steer = np.array([1.0, np.exp(1j * math.pi * math.sin(self.human_aod))], dtype=complex)
            rx_steer = np.array([1.0, np.exp(1j * math.pi * math.sin(self.human_aoa))], dtype=complex)
            H += self.human_reflection_coeff * np.exp(1j * phase) * np.outer(rx_steer, tx_steer.conj())
            
        # Add thermal noise
        noise = (np.random.normal(0, SIM_NOISE_FLOOR, (NC, NR)) + 
                 1j * np.random.normal(0, SIM_NOISE_FLOOR, (NC, NR)))
        H += noise
        
        return H

    def get_bfi_angles(self) -> List[Tuple[float, float]]:
        """
        Computes the Givens angles (phi, psi) for all subcarriers.
        For Nc=2, Nr=2: SVD H = U S V^H
        The first column of steering matrix V is V[:, 0] = [v11, v21]^T
        phi = angle(v21) - angle(v11)
        psi = arctan2(|v21|, |v11|)
        """
        angles = []
        for f in self.frequencies:
            H = self.compute_channel_matrix(f)
            # Perform SVD
            try:
                U, S, Vh = np.linalg.svd(H)
                # Steering matrix V is conjugate transpose of Vh
                V = Vh.conj().T
                
                v11 = V[0, 0]
                v21 = V[1, 0]
                
                # Extract angles
                psi = math.atan2(abs(v21), abs(v11))
                # Phase difference
                phi = math.angle(v21) if hasattr(math, "angle") else np.angle(v21)
                phi -= math.angle(v11) if hasattr(math, "angle") else np.angle(v11)
                
                # Wrap phi to [0, 2*pi]
                phi = phi % (2 * math.pi)
                # Wrap psi to [0, pi/2]
                psi = max(0.0, min(math.pi/2, psi))
                
                angles.append((phi, psi))
            except np.linalg.LinAlgError:
                angles.append((0.0, 0.0))
                
        return angles

    def generate_packet_payload(
        self,
        bf_format: str = "vht",
        shield_active: bool = False,
        shield_seed: int = 42,
        shield_version: int = SHIELD_VERSION,
        shield_num_taps: int = SHIELD_NUM_TAPS
    ) -> bytes:
        """
        Generates a standard-compliant VHT (21), HE (26), or 802.11bf SENS (33) Action frame payload
        containing the simulated BFI data. Applies Aletheia-Shield obfuscation if active.

        Supports both Shield v1 (legacy phase offset) and Shield v2 (multi-tap convolution).
        """
        # 1. Action frame headers
        # Category: VHT (21), HE (26), or SENS (33)
        if bf_format == "11bf":
            category = 33
        elif bf_format == "he":
            category = 26
        else:
            category = 21
            
        payload = bytearray([category, 0])
        
        # 2. MIMO Control: 3 bytes
        # Nc = 2 (index = 1), Nr = 2 (index = 1) -> First byte = 1 | (1 << 2) = 5
        # Channel Width = 20MHz (0), Grouping = Ng=1 (0), Codebook Info = 7/5 bits (0), Feedback Type = SU (0)
        # Remaining bits of MIMO control are 0
        mimo_control = bytearray([5, 0, 0])
        payload.extend(mimo_control)
        
        # 3. Sounding Dialog Token: 1 byte
        token = random.randint(1, 254)
        payload.append(token)
        
        # 4. Report Payload (Givens angles)
        angles = self.get_bfi_angles()
        
        # Apply obfuscation if active
        if shield_active:
            if shield_version >= 2:
                # Shield v2: Multi-tap convolution obfuscation
                from backend.parser import generate_multitap_filter, apply_shield_v2_obfuscation
                G = generate_multitap_filter(
                    token, shield_seed, len(angles),
                    num_taps=shield_num_taps,
                    tap_spacing=SHIELD_TAP_SPACING,
                    bandwidth=CHANNEL_BANDWIDTH_HZ
                )
                phi_arr = np.array([a[0] for a in angles])
                psi_arr = np.array([a[1] for a in angles])
                phi_obf, psi_obf = apply_shield_v2_obfuscation(phi_arr, psi_arr, G)
                angles = list(zip(phi_obf.tolist(), psi_obf.tolist()))
            else:
                # Shield v1: Legacy phase offset
                from backend.parser import get_v1_obfuscation_noise
                phi_noise, psi_noise = get_v1_obfuscation_noise(token, shield_seed, len(angles))
                obfuscated_angles = []
                for i, (phi, psi) in enumerate(angles):
                    phi_obf = (phi + phi_noise[i]) % (2 * math.pi)
                    psi_obf = (psi + psi_noise[i]) % (math.pi / 2)
                    obfuscated_angles.append((phi_obf, psi_obf))
                angles = obfuscated_angles
        
        writer = LsbBitWriter()
        for phi, psi in angles:
            # Quantize phi: 7 bits (0 to 127) -> phi * 128 / (2*pi)
            phi_q = int(round(phi * 128 / (2 * math.pi))) % 128
            # Quantize psi: 5 bits (0 to 31) -> psi * 32 / (pi/2)
            psi_q = int(round(psi * 32 / (math.pi / 2))) % 32
            
            writer.write_bits(phi_q, 7)
            writer.write_bits(psi_q, 5)
            
        payload.extend(writer.flush())
        return bytes(payload)
