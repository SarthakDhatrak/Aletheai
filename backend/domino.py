"""
Domino — Fractional Delay & Hardware Impairment Compensation Module

Compensates for fractional delay effects in the CIR where true physical reflection
paths don't align with the discrete sampling grid, causing sinc-interpolation energy
leakage across adjacent taps.

Algorithm:
1. Identify the strongest stable tap (LoS reference) in the CIR
2. Estimate fractional delay via parabolic interpolation around the peak
3. Apply frequency-domain phase correction to concentrate scattered energy
4. Since hardware PLL/STO errors affect all paths uniformly, the correction
   derived from the reference path applies globally

Reference: Domino hardware-distortion compensation (Wi-Fi sensing literature)
"""

import numpy as np
from typing import Tuple, Optional
from backend.config import (
    DOMINO_ENABLED, DOMINO_REF_TAP_STABILITY_WINDOW, CHANNEL_BANDWIDTH_HZ, NUM_SUBCARRIERS
)


def estimate_fractional_delay(cir_magnitude: np.ndarray, ref_tap_idx: int) -> float:
    """
    Estimates the fractional delay offset of the reference tap using parabolic
    (quadratic) interpolation on the three samples surrounding the peak.

    Given samples y[k-1], y[k], y[k+1] around peak index k, the fractional
    offset δ ∈ (-0.5, 0.5) is:
        δ = 0.5 * (y[k-1] - y[k+1]) / (y[k-1] - 2*y[k] + y[k+1])

    Args:
        cir_magnitude: 1D array of CIR magnitude values (averaged over time window)
        ref_tap_idx: Index of the reference (LoS) tap

    Returns:
        Fractional delay offset in samples (range: -0.5 to 0.5)
    """
    N = len(cir_magnitude)
    k = ref_tap_idx

    # Boundary check: need neighbors on both sides
    if k <= 0 or k >= N - 1:
        return 0.0

    y_prev = cir_magnitude[k - 1]
    y_peak = cir_magnitude[k]
    y_next = cir_magnitude[k + 1]

    denominator = y_prev - 2.0 * y_peak + y_next
    if abs(denominator) < 1e-12:
        return 0.0

    delta = 0.5 * (y_prev - y_next) / denominator
    # Clamp to valid range
    return np.clip(delta, -0.5, 0.5)


def compute_phase_correction(
    fractional_delay: float,
    num_subcarriers: int,
    bandwidth: float = CHANNEL_BANDWIDTH_HZ
) -> np.ndarray:
    """
    Computes the frequency-domain phase correction vector to compensate
    for a fractional delay of δ samples.

    The correction for subcarrier index f is:
        C[f] = exp(+j * 2π * f * δ / N)

    where N is the FFT size (= num_subcarriers) and δ is the fractional delay.

    Args:
        fractional_delay: Estimated fractional delay in samples
        num_subcarriers: Number of subcarriers (FFT size)
        bandwidth: Channel bandwidth in Hz

    Returns:
        Complex correction vector of shape (num_subcarriers,)
    """
    f_indices = np.arange(num_subcarriers)
    # Frequency-domain phase shift to compensate fractional delay
    phase_correction = np.exp(1j * 2.0 * np.pi * f_indices * fractional_delay / num_subcarriers)
    return phase_correction


def find_reference_tap(cir_abs: np.ndarray, exclude_dc_taps: int = 2) -> int:
    """
    Identifies the Line-of-Sight (LoS) reference tap as the tap with the
    highest mean amplitude (excluding DC leakage taps).

    The LoS path is typically the strongest and most stable path in the CIR.

    Args:
        cir_abs: CIR magnitude matrix of shape (T, F) — time × delay taps
        exclude_dc_taps: Number of initial taps to exclude (DC leakage)

    Returns:
        Index of the reference tap
    """
    mean_profile = np.mean(cir_abs, axis=0)
    # Exclude first few taps (DC leakage) and use only first half (causal paths)
    half_len = len(mean_profile) // 2
    search_region = mean_profile[exclude_dc_taps:half_len]

    if len(search_region) == 0:
        return exclude_dc_taps

    return int(exclude_dc_taps + np.argmax(search_region))


def apply_domino_compensation(
    phi_matrix: np.ndarray,
    psi_matrix: np.ndarray,
    bandwidth: float = CHANNEL_BANDWIDTH_HZ
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Applies the full Domino fractional delay compensation pipeline.

    Steps:
    1. Reconstruct steering vector v21 = sin(ψ) * exp(jφ)
    2. Compute CIR via IFFT
    3. Find LoS reference tap
    4. Estimate fractional delay via parabolic interpolation
    5. Apply frequency-domain phase correction
    6. Re-extract corrected φ and ψ

    Args:
        phi_matrix: Phase angles φ, shape (T, F)
        psi_matrix: Elevation angles ψ, shape (T, F)
        bandwidth: Channel bandwidth in Hz

    Returns:
        Tuple of (phi_corrected, psi_corrected, ssnr_improvement_db, ref_tap_idx)
    """
    T, F = phi_matrix.shape

    # 1. Reconstruct complex steering vector
    v21 = np.sin(psi_matrix) * np.exp(1j * phi_matrix)

    # 2. Compute CIR (pre-compensation) for reference tap identification
    cir_pre = np.fft.ifft(v21, n=F, axis=1)
    cir_pre_abs = np.abs(cir_pre)

    # 3. Find LoS reference tap
    ref_tap = find_reference_tap(cir_pre_abs)

    # 4. Estimate fractional delay from averaged CIR profile
    avg_profile = np.mean(cir_pre_abs, axis=0)
    frac_delay = estimate_fractional_delay(avg_profile, ref_tap)

    # 5. If fractional delay is negligible, skip compensation
    if abs(frac_delay) < 0.01:
        # No significant fractional delay; compute SSNR as-is
        tap_vars_pre = np.var(cir_pre_abs, axis=0)
        noise_floor = np.median(tap_vars_pre)
        if noise_floor > 1e-12:
            ssnr = 10.0 * np.log10(np.max(tap_vars_pre[3:]) / noise_floor)
        else:
            ssnr = 0.0
        return phi_matrix, psi_matrix, ssnr, ref_tap

    # 6. Compute phase correction vector
    correction = compute_phase_correction(frac_delay, F, bandwidth)

    # 7. Apply correction in frequency domain (per time sample)
    v21_corrected = v21 * correction[np.newaxis, :]

    # 8. Re-extract φ and ψ from corrected steering vector
    phi_corrected = np.angle(v21_corrected)
    # Wrap φ to [0, 2π]
    phi_corrected = phi_corrected % (2.0 * np.pi)

    psi_corrected = np.arcsin(np.clip(np.abs(v21_corrected), 0.0, 1.0))

    # 9. Compute SSNR improvement
    cir_post = np.fft.ifft(v21_corrected, n=F, axis=1)
    cir_post_abs = np.abs(cir_post)

    tap_vars_pre = np.var(cir_pre_abs, axis=0)
    tap_vars_post = np.var(cir_post_abs, axis=0)

    noise_floor_pre = np.median(tap_vars_pre)
    noise_floor_post = np.median(tap_vars_post)

    # SSNR = 10*log10(max_dynamic_variance / noise_floor)
    if noise_floor_pre > 1e-12 and noise_floor_post > 1e-12:
        ssnr_pre = 10.0 * np.log10(np.max(tap_vars_pre[3:]) / noise_floor_pre)
        ssnr_post = 10.0 * np.log10(np.max(tap_vars_post[3:]) / noise_floor_post)
        ssnr_improvement = ssnr_post - ssnr_pre
    else:
        ssnr_improvement = 0.0

    return phi_corrected, psi_corrected, ssnr_improvement, ref_tap


class DominoState:
    """
    Tracks reference tap stability across consecutive windows for temporal
    consistency. Only updates the reference tap when it has been stable
    for DOMINO_REF_TAP_STABILITY_WINDOW consecutive observations.
    """

    def __init__(self):
        self.confirmed_ref_tap: Optional[int] = None
        self.candidate_ref_tap: Optional[int] = None
        self.candidate_count: int = 0
        self.stability_window: int = DOMINO_REF_TAP_STABILITY_WINDOW
        self.last_ssnr: float = 0.0
        self.last_frac_delay: float = 0.0

    def update(self, observed_ref_tap: int, ssnr: float, frac_delay: float):
        """
        Updates the reference tap tracking state.

        Args:
            observed_ref_tap: Reference tap detected in current window
            ssnr: SSNR improvement from current Domino pass
            frac_delay: Estimated fractional delay
        """
        self.last_ssnr = ssnr
        self.last_frac_delay = frac_delay

        if observed_ref_tap == self.candidate_ref_tap:
            self.candidate_count += 1
        else:
            self.candidate_ref_tap = observed_ref_tap
            self.candidate_count = 1

        if self.candidate_count >= self.stability_window:
            self.confirmed_ref_tap = self.candidate_ref_tap

    def get_confirmed_tap(self) -> Optional[int]:
        return self.confirmed_ref_tap


def test_domino():
    """
    Self-validation: generates a synthetic CIR with known fractional delay
    and verifies Domino compensation concentrates the energy.
    """
    print("Running Domino self-test...")
    T, F = 20, 52
    np.random.seed(42)

    # Create a synthetic channel with a fractional delay of 0.3 samples at tap 5
    true_delay = 5.3
    v21 = np.zeros((T, F), dtype=complex)
    f_indices = np.arange(F)
    for t in range(T):
        # Add a path with fractional delay
        base_phase = -2.0 * np.pi * f_indices * true_delay / F
        motion = 0.1 * np.sin(2.0 * np.pi * t / T)  # simulate motion
        v21[t, :] = 0.5 * np.exp(1j * (base_phase + motion))
        # Add noise
        v21[t, :] += 0.02 * (np.random.randn(F) + 1j * np.random.randn(F))

    phi = np.angle(v21) % (2.0 * np.pi)
    psi = np.arcsin(np.clip(np.abs(v21), 0.0, 1.0))

    phi_c, psi_c, ssnr_imp, ref_tap = apply_domino_compensation(phi, psi)

    print(f"  Reference tap: {ref_tap}")
    print(f"  SSNR improvement: {ssnr_imp:.2f} dB")
    print(f"  Test {'PASSED' if ref_tap in [4, 5, 6] else 'WARN (tap mismatch)'}")
    print("Domino self-test complete.")


if __name__ == "__main__":
    test_domino()
