"""
AM-FM Foundation Encoder — Self-Supervised BFI Representations

Optional module (requires PyTorch, disabled by default).
Implements a lightweight Masked Autoencoder (MAE) operating on BFI spectrograms
for learning universal channel representations that generalize across rooms,
devices, and sampling rates.

Architecture:
- Input: BFI spectrogram X ∈ R^{T×F} (time × subcarrier phi angles)
- Patch embedding: split into non-overlapping patches
- Mask 75% of patches randomly
- Encoder: 3-layer Transformer (dim=64, heads=4)
- Decoder: 2-layer Transformer for reconstruction
- Pre-training: masked reconstruction MSE loss
- Fine-tuning: frozen encoder + MLP head → classification

CPU-only design: small architecture, no GPU required.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from backend.config import (
    FOUNDATION_ENABLED, FOUNDATION_EMBED_DIM, FOUNDATION_NUM_HEADS,
    FOUNDATION_DEPTH, FOUNDATION_MASK_RATIO, FOUNDATION_PRETRAIN_EPOCHS,
    FOUNDATION_SAVE_PATH, FEATURE_DIM
)

# Guard PyTorch import — this module is completely optional
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def is_foundation_available() -> bool:
    """Check if the foundation encoder can be used (PyTorch installed + enabled)."""
    return _TORCH_AVAILABLE and FOUNDATION_ENABLED


if _TORCH_AVAILABLE:

    class PatchEmbedding(nn.Module):
        """Converts a T×F BFI matrix into a sequence of patch tokens."""

        def __init__(self, input_t: int = 20, input_f: int = 52,
                     patch_t: int = 4, patch_f: int = 4,
                     embed_dim: int = FOUNDATION_EMBED_DIM):
            super().__init__()
            self.patch_t = patch_t
            self.patch_f = patch_f
            self.num_patches_t = input_t // patch_t
            self.num_patches_f = input_f // patch_f
            self.num_patches = self.num_patches_t * self.num_patches_f
            self.patch_dim = patch_t * patch_f

            self.proj = nn.Linear(self.patch_dim, embed_dim)
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches, embed_dim) * 0.02
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (B, T, F) input spectrogram

            Returns:
                (B, num_patches, embed_dim) patch tokens with positional embeddings
            """
            B, T, F_dim = x.shape
            # Reshape into patches
            x = x[:, :self.num_patches_t * self.patch_t, :self.num_patches_f * self.patch_f]
            x = x.reshape(B, self.num_patches_t, self.patch_t, self.num_patches_f, self.patch_f)
            x = x.permute(0, 1, 3, 2, 4).reshape(B, self.num_patches, self.patch_dim)
            # Project and add positional embedding
            x = self.proj(x) + self.pos_embed
            return x

    class TransformerBlock(nn.Module):
        """Standard multi-head self-attention + FFN block."""

        def __init__(self, dim: int = FOUNDATION_EMBED_DIM,
                     num_heads: int = FOUNDATION_NUM_HEADS,
                     mlp_ratio: float = 2.0, dropout: float = 0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            mlp_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Pre-norm architecture
            x_norm = self.norm1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x

    class BFIEncoder(nn.Module):
        """Transformer encoder producing fixed-dimensional representations."""

        def __init__(self, embed_dim: int = FOUNDATION_EMBED_DIM,
                     depth: int = FOUNDATION_DEPTH,
                     num_heads: int = FOUNDATION_NUM_HEADS):
            super().__init__()
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            # Global average pool over patches → single vector per sample
            return x.mean(dim=1)

    class BFIDecoder(nn.Module):
        """Lightweight decoder for masked reconstruction."""

        def __init__(self, embed_dim: int = FOUNDATION_EMBED_DIM,
                     num_patches: int = 65, patch_dim: int = 16,
                     depth: int = 2, num_heads: int = FOUNDATION_NUM_HEADS):
            super().__init__()
            self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.blocks = nn.ModuleList([
                TransformerBlock(dim=embed_dim, num_heads=num_heads)
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, patch_dim)

        def forward(self, x_visible: torch.Tensor,
                    mask_indices: torch.Tensor,
                    num_patches: int) -> torch.Tensor:
            B = x_visible.shape[0]
            # Create full sequence with mask tokens
            full_seq = self.mask_token.expand(B, num_patches, -1).clone()
            # Fill in visible tokens
            visible_indices = (~mask_indices).nonzero(as_tuple=True)
            if len(visible_indices) > 1:
                full_seq[visible_indices[0], visible_indices[1]] = x_visible.reshape(-1, x_visible.shape[-1])[
                    :full_seq[visible_indices[0], visible_indices[1]].shape[0]
                ]

            for block in self.blocks:
                full_seq = block(full_seq)
            full_seq = self.norm(full_seq)
            return self.head(full_seq)

    class BFIMaskedAutoencoder(nn.Module):
        """Full MAE: patch embedding → random masking → encode → decode → reconstruct."""

        def __init__(self, input_t: int = 20, input_f: int = 52,
                     patch_t: int = 4, patch_f: int = 4,
                     embed_dim: int = FOUNDATION_EMBED_DIM):
            super().__init__()
            self.patch_embed = PatchEmbedding(input_t, input_f, patch_t, patch_f, embed_dim)
            self.encoder = BFIEncoder(embed_dim=embed_dim)
            patch_dim = patch_t * patch_f
            num_patches = self.patch_embed.num_patches
            self.decoder = BFIDecoder(embed_dim, num_patches, patch_dim)
            self.mask_ratio = FOUNDATION_MASK_RATIO

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # 1. Patch embedding
            patches = self.patch_embed(x)
            B, N, D = patches.shape

            # 2. Random masking
            num_masked = int(N * self.mask_ratio)
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = noise.argsort(dim=1)
            mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
            for b in range(B):
                mask[b, ids_shuffle[b, :num_masked]] = True

            # 3. Encode visible patches only
            visible = patches[~mask].reshape(B, N - num_masked, D)
            encoded = self.encoder(visible)

            # 4. Decode (reconstruct all patches)
            reconstructed = self.decoder(visible, mask, N)

            # 5. Compute loss on masked patches only
            target = self.patch_embed(x)
            return reconstructed, target

    class FoundationClassifier(nn.Module):
        """Fine-tuning head: frozen encoder + MLP → classification."""

        def __init__(self, encoder: BFIEncoder, num_classes: int = 4,
                     embed_dim: int = FOUNDATION_EMBED_DIM):
            super().__init__()
            self.encoder = encoder
            # Freeze encoder weights
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, num_classes)
            )

        def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                features = self.encoder(x_patches)
            return self.head(features)


def extract_foundation_features(
    encoder,
    phi_matrix: np.ndarray,
    psi_matrix: np.ndarray
) -> np.ndarray:
    """
    Extract foundation encoder features from a BFI window.

    Args:
        encoder: Trained BFIEncoder instance
        phi_matrix: (T, F) phi angle matrix
        psi_matrix: (T, F) psi angle matrix

    Returns:
        1D feature vector of shape (FOUNDATION_EMBED_DIM,)
    """
    if not _TORCH_AVAILABLE:
        return np.zeros(FOUNDATION_EMBED_DIM)

    # Use phi matrix as input spectrogram
    x = torch.FloatTensor(phi_matrix).unsqueeze(0)  # (1, T, F)

    # Create patch embeddings
    patch_embed = PatchEmbedding(
        input_t=phi_matrix.shape[0],
        input_f=phi_matrix.shape[1]
    )

    with torch.no_grad():
        patches = patch_embed(x)
        features = encoder(patches)

    return features.squeeze(0).numpy()


def pretrain_foundation(num_epochs: int = FOUNDATION_PRETRAIN_EPOCHS) -> Optional[Any]:
    """
    Pre-trains the foundation MAE on synthetic BFI data.
    Returns trained encoder or None if PyTorch unavailable.
    """
    if not _TORCH_AVAILABLE:
        print("[Foundation] PyTorch not available. Skipping pre-training.")
        return None

    print(f"[Foundation] Pre-training MAE on synthetic data for {num_epochs} epochs (CPU)...")

    from backend.simulator import BFISimulator
    from backend.parser import parse_raw_bfi_payload

    # Generate synthetic spectrograms
    sim = BFISimulator()
    spectrograms = []

    for state in ["EMPTY", "PRESENCE", "WALKING", "FALLING"]:
        for seg in range(5):
            sim.layout_distance = np.random.uniform(2.0, 8.0)
            sim.set_state(state)
            sim.time_offset = 0.0

            phi_window = []
            for step in range(20):
                sim.update_physics(0.1)
                payload = sim.generate_packet_payload()
                parsed = parse_raw_bfi_payload(payload)
                if parsed:
                    phis = [s["phi"][0] for s in parsed["angles"] if s["phi"]]
                    phi_window.append(phis[:52])

            if len(phi_window) == 20:
                spectrograms.append(np.array(phi_window))

    if len(spectrograms) < 5:
        print("[Foundation] Insufficient data for pre-training.")
        return None

    X = torch.FloatTensor(np.array(spectrograms))  # (N, 20, 52)

    mae = BFIMaskedAutoencoder(input_t=20, input_f=52)
    optimizer = torch.optim.Adam(mae.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recon, target = mae(X)
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} — Loss: {loss.item():.6f}")

    # Save encoder
    torch.save(mae.encoder.state_dict(), FOUNDATION_SAVE_PATH)
    print(f"[Foundation] Pre-training complete. Encoder saved to {FOUNDATION_SAVE_PATH}")
    return mae.encoder
