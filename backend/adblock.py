"""
ADBlock — Out-of-Distribution (OOD) Anomaly Detection Guardrails

Prevents false alarms from untrained physical dynamics (pets, vacuum cleaners,
furniture rearrangement) by detecting when incoming BFI features fall outside
the learned distribution.

Architecture:
- Lightweight 3-layer autoencoder (input→12→6→12→input) using pure numpy
- Trained on "normal" class features (EMPTY + PRESENCE)
- Reconstruction error L(x) = ||x - Decode(Encode(x))||²
- OOD threshold: adaptive EMA — τ_t = α·L(x_t) + (1-α)·τ_{t-1}
- If L(x) > τ → flag as OUT_OF_DISTRIBUTION

No PyTorch dependency — pure numpy forward/backward passes.
"""

import os
import numpy as np
from typing import Tuple, Optional, List
from backend.config import (
    ADBLOCK_ENABLED, ADBLOCK_THRESHOLD_K, ADBLOCK_ADAPTIVE_ALPHA,
    ADBLOCK_HIDDEN_DIM, ADBLOCK_LATENT_DIM, ADBLOCK_LEARNING_RATE,
    ADBLOCK_TRAIN_EPOCHS, ADBLOCK_SAVE_PATH, FEATURE_DIM
)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def xavier_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, (fan_in, fan_out))


class FeatureAutoencoder:
    """
    3-layer autoencoder for anomaly detection on BFI feature vectors.
    Architecture: input_dim → hidden_dim → latent_dim → hidden_dim → input_dim

    Pure numpy implementation with mini-batch SGD training.
    """

    def __init__(self, input_dim: int = FEATURE_DIM,
                 hidden_dim: int = ADBLOCK_HIDDEN_DIM,
                 latent_dim: int = ADBLOCK_LATENT_DIM,
                 seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        rng = np.random.RandomState(seed)

        # Encoder weights
        self.W1 = xavier_init(input_dim, hidden_dim, rng)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier_init(hidden_dim, latent_dim, rng)
        self.b2 = np.zeros(latent_dim)

        # Decoder weights
        self.W3 = xavier_init(latent_dim, hidden_dim, rng)
        self.b3 = np.zeros(hidden_dim)
        self.W4 = xavier_init(hidden_dim, input_dim, rng)
        self.b4 = np.zeros(input_dim)

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through encoder. Returns (latent, hidden_pre, hidden_post)."""
        z1_pre = x @ self.W1 + self.b1
        z1 = relu(z1_pre)
        z2_pre = z1 @ self.W2 + self.b2
        z2 = relu(z2_pre)
        return z2, z1_pre, z1

    def decode(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through decoder. Returns (output, hidden_pre, hidden_post)."""
        z3_pre = z @ self.W3 + self.b3
        z3 = relu(z3_pre)
        z4 = z3 @ self.W4 + self.b4  # Linear output (no activation)
        return z4, z3_pre, z3

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass: encode then decode."""
        z, _, _ = self.encode(x)
        out, _, _ = self.decode(z)
        return out

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error."""
        x_hat = self.forward(x)
        return np.mean((x - x_hat) ** 2, axis=-1)

    def train_step(self, x_batch: np.ndarray, lr: float) -> float:
        """
        Single mini-batch training step with backpropagation.
        Returns the mean loss for the batch.
        """
        batch_size = x_batch.shape[0]

        # --- Forward Pass ---
        # Encoder
        z1_pre = x_batch @ self.W1 + self.b1
        z1 = relu(z1_pre)
        z2_pre = z1 @ self.W2 + self.b2
        z2 = relu(z2_pre)

        # Decoder
        z3_pre = z2 @ self.W3 + self.b3
        z3 = relu(z3_pre)
        x_hat = z3 @ self.W4 + self.b4

        # Loss: MSE
        error = x_hat - x_batch
        loss = np.mean(error ** 2)

        # --- Backward Pass ---
        # d_loss/d_x_hat = 2 * error / (batch_size * input_dim)
        d_xhat = 2.0 * error / (batch_size * self.input_dim)

        # Layer 4 (decoder output, linear)
        d_W4 = z3.T @ d_xhat
        d_b4 = np.sum(d_xhat, axis=0)
        d_z3 = d_xhat @ self.W4.T

        # Layer 3 (decoder hidden, ReLU)
        d_z3_pre = d_z3 * relu_derivative(z3_pre)
        d_W3 = z2.T @ d_z3_pre
        d_b3 = np.sum(d_z3_pre, axis=0)
        d_z2 = d_z3_pre @ self.W3.T

        # Layer 2 (encoder latent, ReLU)
        d_z2_pre = d_z2 * relu_derivative(z2_pre)
        d_W2 = z1.T @ d_z2_pre
        d_b2 = np.sum(d_z2_pre, axis=0)
        d_z1 = d_z2_pre @ self.W2.T

        # Layer 1 (encoder hidden, ReLU)
        d_z1_pre = d_z1 * relu_derivative(z1_pre)
        d_W1 = x_batch.T @ d_z1_pre
        d_b1 = np.sum(d_z1_pre, axis=0)

        # --- Gradient Descent Update ---
        self.W4 -= lr * d_W4
        self.b4 -= lr * d_b4
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

        return loss

    def save(self, path: str = ADBLOCK_SAVE_PATH):
        """Save weights to .npz file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

    def load(self, path: str = ADBLOCK_SAVE_PATH) -> bool:
        """Load weights from .npz file. Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.W3 = data["W3"]
            self.b3 = data["b3"]
            self.W4 = data["W4"]
            self.b4 = data["b4"]
            return True
        except Exception:
            return False


class ADBlock:
    """
    Out-of-Distribution guardrail wrapper around the FeatureAutoencoder.

    Maintains an adaptive threshold using Exponential Moving Average (EMA)
    of reconstruction errors from recent "normal" predictions.
    """

    def __init__(self, input_dim: int = FEATURE_DIM):
        self.autoencoder = FeatureAutoencoder(input_dim=input_dim)
        self.is_trained = False

        # Static threshold (calibrated during training)
        self.static_threshold: float = 1.0
        self.train_mean_error: float = 0.0
        self.train_std_error: float = 1.0

        # Adaptive threshold (EMA updated at inference time)
        self.adaptive_threshold: float = 1.0
        self.alpha = ADBLOCK_ADAPTIVE_ALPHA
        self.k = ADBLOCK_THRESHOLD_K

        # Counters for diagnostics
        self.total_predictions: int = 0
        self.ood_predictions: int = 0

    def train(self, X_normal: np.ndarray, epochs: int = ADBLOCK_TRAIN_EPOCHS,
              lr: float = ADBLOCK_LEARNING_RATE, batch_size: int = 32):
        """
        Trains the autoencoder on normal-class (EMPTY + PRESENCE) feature vectors.

        Args:
            X_normal: Feature matrix of shape (N, input_dim) — only normal samples
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Mini-batch size
        """
        N = X_normal.shape[0]
        if N < 10:
            print("[ADBlock] Insufficient normal samples for training. Skipping.")
            return

        # Normalize features (z-score) for stable training
        self.feature_mean = np.mean(X_normal, axis=0)
        self.feature_std = np.std(X_normal, axis=0) + 1e-8
        X_norm = (X_normal - self.feature_mean) / self.feature_std

        print(f"[ADBlock] Training autoencoder on {N} normal samples for {epochs} epochs...")

        rng = np.random.RandomState(42)
        for epoch in range(epochs):
            # Shuffle
            indices = rng.permutation(N)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = X_norm[indices[start:end]]
                loss = self.autoencoder.train_step(batch, lr)
                epoch_loss += loss
                n_batches += 1

            if (epoch + 1) % 50 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(1, n_batches)
                print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")

        # Calibrate threshold from training reconstruction errors
        train_errors = self.autoencoder.reconstruction_error(X_norm)
        self.train_mean_error = float(np.mean(train_errors))
        self.train_std_error = float(np.std(train_errors))
        self.static_threshold = self.train_mean_error + self.k * self.train_std_error
        self.adaptive_threshold = self.static_threshold

        self.is_trained = True
        print(f"[ADBlock] Training complete. Threshold: {self.static_threshold:.6f} "
              f"(mean={self.train_mean_error:.6f}, std={self.train_std_error:.6f})")

    def is_ood(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Checks whether a feature vector is Out-of-Distribution.

        Args:
            features: 1D feature vector of shape (input_dim,)

        Returns:
            Tuple of (is_ood_flag, reconstruction_error)
        """
        if not self.is_trained:
            return False, 0.0

        # Normalize using training statistics
        x_norm = (features - self.feature_mean) / self.feature_std
        x_norm = x_norm.reshape(1, -1)

        error = float(self.autoencoder.reconstruction_error(x_norm)[0])

        # Check against adaptive threshold
        is_anomalous = error > self.adaptive_threshold

        # Update adaptive threshold (EMA) only for non-OOD samples
        # to prevent threshold drift from persistent anomalies
        if not is_anomalous:
            self.adaptive_threshold = (
                self.alpha * (self.train_mean_error + self.k * max(self.train_std_error, abs(error - self.train_mean_error)))
                + (1.0 - self.alpha) * self.adaptive_threshold
            )

        # Update counters
        self.total_predictions += 1
        if is_anomalous:
            self.ood_predictions += 1

        return is_anomalous, error

    def get_threshold(self) -> float:
        """Returns the current adaptive OOD threshold."""
        return self.adaptive_threshold

    def get_ood_rate(self) -> float:
        """Returns the fraction of recent predictions flagged as OOD."""
        if self.total_predictions == 0:
            return 0.0
        return self.ood_predictions / self.total_predictions

    def save(self, path: str = ADBLOCK_SAVE_PATH):
        """Save autoencoder weights and calibration stats."""
        self.autoencoder.save(path)
        # Save calibration alongside
        cal_path = path.replace(".npz", "_calibration.npz")
        np.savez(cal_path,
                 feature_mean=self.feature_mean,
                 feature_std=self.feature_std,
                 train_mean_error=np.array([self.train_mean_error]),
                 train_std_error=np.array([self.train_std_error]),
                 static_threshold=np.array([self.static_threshold]))

    def load(self, path: str = ADBLOCK_SAVE_PATH) -> bool:
        """Load autoencoder weights and calibration stats. Returns True on success."""
        if not self.autoencoder.load(path):
            return False

        cal_path = path.replace(".npz", "_calibration.npz")
        if not os.path.exists(cal_path):
            return False
        try:
            cal = np.load(cal_path)
            self.feature_mean = cal["feature_mean"]
            self.feature_std = cal["feature_std"]
            self.train_mean_error = float(cal["train_mean_error"][0])
            self.train_std_error = float(cal["train_std_error"][0])
            self.static_threshold = float(cal["static_threshold"][0])
            self.adaptive_threshold = self.static_threshold
            self.is_trained = True
            return True
        except Exception:
            return False


def train_adblock_from_data(X: np.ndarray, y: List[str], input_dim: int = FEATURE_DIM) -> ADBlock:
    """
    Helper: extracts normal-class (EMPTY, PRESENCE) samples from labeled training
    data and trains an ADBlock instance.

    Args:
        X: Full feature matrix of shape (N, input_dim)
        y: Labels list of length N
        input_dim: Feature dimension

    Returns:
        Trained ADBlock instance
    """
    adblock = ADBlock(input_dim=input_dim)

    # Filter for normal classes only
    normal_mask = np.array([label in ("EMPTY", "PRESENCE") for label in y])
    X_normal = X[normal_mask]

    if len(X_normal) < 10:
        print("[ADBlock] Not enough normal samples to train. ADBlock will be inactive.")
        return adblock

    adblock.train(X_normal)
    adblock.save()
    return adblock


def test_adblock():
    """
    Self-validation: trains on synthetic normal data, then tests OOD detection
    on extreme/random inputs.
    """
    print("Running ADBlock self-test...")
    rng = np.random.RandomState(42)

    # Generate "normal" features (small variance, centered)
    N_normal = 200
    X_normal = rng.randn(N_normal, FEATURE_DIM) * 0.1 + 0.5

    # Train
    adblock = ADBlock(input_dim=FEATURE_DIM)
    adblock.train(X_normal, epochs=100, lr=0.01)

    # Test with normal samples
    n_normal_ood = 0
    for i in range(20):
        x = rng.randn(FEATURE_DIM) * 0.1 + 0.5
        is_ood, error = adblock.is_ood(x)
        if is_ood:
            n_normal_ood += 1

    # Test with anomalous samples (extreme values)
    n_anomaly_detected = 0
    for i in range(20):
        x = rng.randn(FEATURE_DIM) * 5.0 + 10.0  # Far from training distribution
        is_ood, error = adblock.is_ood(x)
        if is_ood:
            n_anomaly_detected += 1

    print(f"  Normal samples flagged OOD: {n_normal_ood}/20 (expect ~0)")
    print(f"  Anomalous samples detected: {n_anomaly_detected}/20 (expect ~20)")
    passed = n_normal_ood <= 3 and n_anomaly_detected >= 15
    print(f"  Test {'PASSED' if passed else 'FAILED'}")
    print("ADBlock self-test complete.")


if __name__ == "__main__":
    test_adblock()
