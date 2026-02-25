"""Tokenization pipeline for AR model training.

This module provides a tokenizer that wraps a trained multi-scale VQ-VAE model
to prepare data for autoregressive model training. It supports both streaming
(live) and file-saving modes.

Key features:
- Load trained VQ-VAE checkpoint with matching config
- Encode data to discrete indices and codebook vectors (z_q)
- Remap sparse codebook usage to consecutive indices
- Unified per-scale codebook with contiguous index ranges per scale
- Stream tokenized data or save to file for reuse
"""

import argparse
import ast
import json
import os
from typing import Optional

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np

from .models import VQVAE2d


@eqx.filter_jit
def _vmap_encode(model, batch):
    """JIT-compiled batched encode — XLA reuses buffers for dead intermediates."""
    model = eqx.nn.inference_mode(model)
    return jax.vmap(model.encode)(batch)


def load_config(config_path: str) -> dict:
    """Load config from a key-value text file.

    Each line has the format ``key: value`` where values are Python literals
    (parsed with ``ast.literal_eval``).

    Args:
        config_path: Path to config.txt file

    Returns:
        Dictionary of config values
    """
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            try:
                config[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                config[key] = value
    return config


def load_data_from_hdf5(data_path, field="omega", sample_start=0, sample_stop=None):
    """Load field data from a gust-format HDF5 file.

    Args:
        data_path: Path to .h5 file with ``/fields/{name}`` datasets
        field: Field name to load (default ``"omega"``)
        sample_start: First sample index (inclusive)
        sample_stop: Last sample index (exclusive); ``None`` means all

    Returns:
        numpy array of shape ``(N, 1, H, W)`` float32
    """
    with h5py.File(data_path, "r") as f:
        ds = f["fields"][field]
        if sample_stop is None:
            sample_stop = ds.shape[0]
        data = ds[sample_start:sample_stop].astype(np.float32)  # (N, H, W)
    return data[:, np.newaxis, :, :]  # (N, 1, H, W)


def load_vqvae_checkpoint(checkpoint_path: str, config: dict, key) -> VQVAE2d:
    """Load a multi-scale VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to the .eqx checkpoint file
        config: Dictionary with model configuration:
            - hidden_dim, codebook_dim, vocab_size, decay
            - base_channels, channel_mult, num_res_blocks
            - use_attention, use_norm, attention_heads
            - scales: tuple of (h, w) tuples
        key: JAX random key for model initialization

    Returns:
        Loaded VQVAE2d model with checkpoint weights
    """
    model = VQVAE2d(
        hidden_dim=config.get("hidden_dim", 512),
        codebook_dim=config.get("codebook_dim", 64),
        vocab_size=config.get("vocab_size", 4096),
        scales=config.get("scales", ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16))),
        in_channels=config.get("in_channels", 1),
        decay=config.get("decay", 0.99),
        base_channels=config.get("base_channels", 128),
        channel_mult=config.get("channel_mult", (1, 2, 4, 4)),
        num_res_blocks=config.get("num_res_blocks", 2),
        use_attention=config.get("use_attention", True),
        use_norm=config.get("use_norm", True),
        attention_heads=config.get("attention_heads", 8),
        key=key,
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    return model


def flatten_multiscale_indices(indices_list):
    """Flatten multi-scale indices to a single 1D array.

    Args:
        indices_list: List of arrays with shapes [sh, sw] for each scale (sh, sw)

    Returns:
        Flattened 1D array of all indices concatenated
    """
    return jnp.concatenate([idx.flatten() for idx in indices_list])


def unflatten_to_scales(flat_indices, scales):
    """Unflatten 1D indices back to multi-scale list.

    Args:
        flat_indices: 1D array of concatenated indices
        scales: Tuple of (h, w) tuples (e.g., ((1,1), (2,2), (4,4), (8,8), (16,16)))

    Returns:
        List of arrays with shapes [sh, sw] for each scale (sh, sw)
    """
    indices_list = []
    offset = 0
    for (sh, sw) in scales:
        size = sh * sw
        idx = flat_indices[offset : offset + size].reshape(sh, sw)
        indices_list.append(idx)
        offset += size
    return indices_list


class VQVAETokenizer:
    """Tokenizer wrapping a trained multi-scale VQ-VAE for AR model data preparation.

    This tokenizer:
    1. Encodes images to discrete codebook indices at multiple scales
    2. Remaps sparse codebook usage to consecutive indices
    3. Builds a unified per-scale codebook where each scale gets its own
       contiguous index range
    4. Provides access to codebook vectors for AR input embeddings

    Attributes:
        model: The trained VQVAE2d model (always multi-scale)
        scales: Tuple of (h, w) scales from the quantizer

        Unified mapping (set by fit() or set_mapping()):
            scale_old_to_unified: List of [K] arrays, original -> unified index per scale
            unified_to_scale: [unified_vocab] scale index for each unified entry
            unified_to_original: [unified_vocab] original codebook index per entry
            unified_codebook: [unified_vocab, D] concatenated codebook
            scale_offsets: [n_scales] start of each scale's range in unified vocab
            scale_vocab_sizes: [n_scales] number of unique tokens per scale
    """

    def __init__(
        self,
        model: VQVAE2d,
        first_trainable_scale: Optional[int] = None,
    ):
        self.model = model
        self.scales = model.quantizer.scales

        # Deterministic vs trainable scale tracking
        self.first_trainable_scale = first_trainable_scale
        self.deterministic_scales = None

        # Unified per-scale codebook mapping (set by fit() or set_mapping())
        self.scale_old_to_unified = None
        self.unified_to_scale = None
        self.unified_to_original = None
        self.unified_codebook = None
        self.scale_offsets = None
        self.scale_vocab_sizes = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: dict, key):
        """Load tokenizer from VQ-VAE checkpoint.

        Args:
            checkpoint_path: Path to .eqx checkpoint
            config: Model configuration dict
            key: JAX random key

        Returns:
            VQVAETokenizer instance (unfitted)
        """
        model = load_vqvae_checkpoint(checkpoint_path, config, key)
        return cls(
            model,
            first_trainable_scale=config.get("first_trainable_scale"),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the tokenizer has been fitted with index mappings."""
        return self.scale_old_to_unified is not None

    @property
    def codebook(self) -> jnp.ndarray:
        """First scale's original codebook [vocab_size, codebook_dim]."""
        return self.model.quantizer.codebooks[0]

    @property
    def codebooks(self):
        """Per-scale codebooks. Tuple of [vocab_size, codebook_dim] arrays."""
        return self.model.quantizer.codebooks

    @property
    def codebook_dim(self) -> int:
        """Dimension of codebook vectors."""
        return self.model.quantizer.D

    @property
    def vocab_size(self) -> int:
        """Original vocabulary size (per-scale codebook size)."""
        return self.model.quantizer.K

    @property
    def effective_vocab_size(self) -> int:
        """Unified vocabulary size (sum of per-scale uniques) after fit()."""
        if self.unified_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return len(self.unified_codebook)

    @property
    def remapped_codebook(self) -> jnp.ndarray:
        """Unified codebook (all scales concatenated) [effective_vocab, codebook_dim]."""
        if self.unified_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return self.unified_codebook

    @property
    def remapped_codebooks(self):
        """Per-scale slices of the unified codebook.

        Returns:
            Tuple of [scale_vocab_size, codebook_dim] arrays, one per scale.
        """
        if self.unified_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        result = []
        for k in range(len(self.scales)):
            start = int(self.scale_offsets[k])
            end = start + int(self.scale_vocab_sizes[k])
            result.append(self.unified_codebook[start:end])
        return tuple(result)

    @property
    def tokens_per_sample(self) -> int:
        """Total number of tokens per sample."""
        return sum(sh * sw for sh, sw in self.scales)

    def fit(self, batches):
        """First pass: collect unique indices per scale and build unified mapping.

        Builds a unified per-scale codebook where each scale gets its own
        contiguous index range in the unified vocabulary.

        Args:
            batches: Iterable yielding (B, 1, H, W) arrays (e.g. HDF5Dataset)

        Returns:
            self (for chaining)
        """
        print("Fitting tokenizer: collecting unique codebook indices...")

        n_scales = len(self.scales)
        per_scale_unique = [[] for _ in range(n_scales)]
        n_processed = 0
        batch_count = 0

        for batch in batches:
            batch = jnp.asarray(batch)
            n_processed += batch.shape[0]
            batch_count += 1

            _, indices_list = _vmap_encode(self.model, batch)
            # indices_list is list of [B, s, s] arrays
            for k in range(n_scales):
                unique_k = jnp.unique(indices_list[k].reshape(-1))
                per_scale_unique[k].append(np.array(unique_k))

            if batch_count % 50 == 0:
                print(f"  Processed {n_processed} samples")

        print(f"  Processed {n_processed} samples (done)")

        # Build unified mapping
        unified_offset = 0
        self.scale_old_to_unified = []
        unified_to_scale_parts = []
        unified_to_original_parts = []
        unified_codebook_parts = []
        scale_offsets = []
        scale_vocab_sizes = []

        for k in range(n_scales):
            combined = np.unique(np.concatenate(per_scale_unique[k]))
            unique_sorted = jnp.array(combined)
            n_unique = len(unique_sorted)

            scale_offsets.append(unified_offset)
            scale_vocab_sizes.append(n_unique)

            # Per-scale mapping: original index -> unified index
            old_to_unified_k = jnp.full(self.vocab_size, -1, dtype=jnp.int32)
            old_to_unified_k = old_to_unified_k.at[unique_sorted].set(
                jnp.arange(n_unique, dtype=jnp.int32) + unified_offset
            )
            self.scale_old_to_unified.append(old_to_unified_k)

            # Reverse mappings
            unified_to_scale_parts.append(jnp.full(n_unique, k, dtype=jnp.int32))
            unified_to_original_parts.append(unique_sorted.astype(jnp.int32))

            # Unified codebook part from this scale's codebook
            unified_codebook_parts.append(self.codebooks[k][unique_sorted])

            sh, sw = self.scales[k]
            print(f"  Scale {sh}x{sw}: {n_unique:4d} unique codes "
                  f"(unified [{unified_offset}, {unified_offset + n_unique}))")

            unified_offset += n_unique

        self.scale_offsets = np.array(scale_offsets, dtype=np.int64)
        self.scale_vocab_sizes = np.array(scale_vocab_sizes, dtype=np.int64)
        self.unified_to_scale = jnp.concatenate(unified_to_scale_parts)
        self.unified_to_original = jnp.concatenate(unified_to_original_parts)
        self.unified_codebook = jnp.concatenate(unified_codebook_parts, axis=0)

        # Auto-detect deterministic scales (scales where only 1 unique code is used)
        if self.first_trainable_scale is None:
            last_deterministic = -1
            for k in range(n_scales):
                if scale_vocab_sizes[k] == 1:
                    last_deterministic = k
                else:
                    break
            self.first_trainable_scale = last_deterministic + 1
        self.deterministic_scales = list(range(self.first_trainable_scale))

        if self.first_trainable_scale > 0:
            det_names = [f"{sh}x{sw}" for sh, sw in self.scales[:self.first_trainable_scale]]
            print(f"Deterministic scales: {', '.join(det_names)} (indices 0-{self.first_trainable_scale - 1})")
            sh, sw = self.scales[self.first_trainable_scale]
            print(f"First trainable scale: {sh}x{sw} (index {self.first_trainable_scale})")
        else:
            print("No deterministic scales detected")

        print(f"Fit complete: {self.effective_vocab_size} unified vocab "
              f"(from {self.vocab_size} per-scale codebook, {n_scales} scales)")

        return self

    def set_mapping(self, **kwargs):
        """Set index mapping manually (e.g., from loaded tokenized data).

        Keyword Args:
            scale_old_to_unified: list of [K] arrays per scale
            unified_to_scale: [unified_vocab] int array
            unified_to_original: [unified_vocab] int array
            unified_codebook: [unified_vocab, D] array
            scale_offsets: [n_scales] int array
            scale_vocab_sizes: [n_scales] int array
        """
        for attr in ('scale_old_to_unified', 'unified_to_scale', 'unified_to_original',
                     'unified_codebook', 'scale_offsets', 'scale_vocab_sizes'):
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

    def encode(self, x: jnp.ndarray):
        """Encode a single sample.

        Args:
            x: Single input [1, H, W]

        Returns:
            remapped_indices: List of [s, s] arrays (unified indices per scale)
            z_q: Quantized vectors [D, H', W']
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        z_q, indices = self.model.encode(x)
        remapped = [self.scale_old_to_unified[k][idx] for k, idx in enumerate(indices)]
        return remapped, z_q

    def encode_batch(self, batch: jnp.ndarray):
        """Encode a batch of samples.

        Args:
            batch: Batch of inputs [B, 1, H, W]

        Returns:
            remapped_indices: List of [B, s, s] arrays (unified indices per scale)
            z_q: Quantized vectors [B, D, H', W']
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        z_q, indices = _vmap_encode(self.model, batch)
        remapped = [self.scale_old_to_unified[k][idx] for k, idx in enumerate(indices)]
        return remapped, z_q

    def encode_batch_flat(self, batch: jnp.ndarray):
        """Encode a batch and return flattened indices and vectors.

        Args:
            batch: Batch of inputs [B, 1, H, W]

        Returns:
            indices_flat: [B, total_tokens] remapped discrete indices (unified)
            vectors_flat: [B, total_tokens, codebook_dim] corresponding vectors
        """
        remapped, z_q = self.encode_batch(batch)

        B = batch.shape[0]

        # Flatten indices: [B, total_tokens]
        indices_flat = jnp.concatenate(
            [idx.reshape(B, -1) for idx in remapped], axis=1
        )

        # Vectors from unified codebook directly
        vectors_flat = self.unified_codebook[indices_flat]  # [B, total_tokens, D]

        return indices_flat, vectors_flat

    def decode_indices(self, remapped_indices):
        """Convert remapped indices back to original and decode.

        Args:
            remapped_indices: List of [s, s] arrays (unified indices per scale)

        Returns:
            Reconstruction [1, H, W]
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        original_indices = [self.unified_to_original[idx] for idx in remapped_indices]
        return self.model.decode_indices(original_indices)

    def decode_flat_indices(self, flat_remapped_indices: jnp.ndarray):
        """Decode from flattened remapped indices.

        Args:
            flat_remapped_indices: [total_tokens] flattened remapped indices

        Returns:
            Reconstruction [1, H, W]
        """
        indices_list = unflatten_to_scales(flat_remapped_indices, self.scales)
        return self.decode_indices(indices_list)


def create_tokenized_dataloader(
    tokenizer: VQVAETokenizer,
    batches,
):
    """Generator that tokenizes on-the-fly during iteration.

    Args:
        tokenizer: Fitted VQVAETokenizer
        batches: Iterable yielding (B, 1, H, W) arrays (e.g. HDF5Dataset).
            Caller controls shuffle/seed via the dataset.

    Yields:
        indices: [B, total_tokens] remapped discrete indices
        vectors: [B, total_tokens, codebook_dim] corresponding codebook vectors
    """
    for batch in batches:
        batch = jnp.asarray(batch)
        indices, vectors = tokenizer.encode_batch_flat(batch)
        yield indices, vectors


def save_tokenized_data(
    path: str,
    tokenizer: VQVAETokenizer,
    batches,
    config: dict,
):
    """Tokenize and save entire dataset to NPZ.

    Args:
        path: Output path for .npz file
        tokenizer: Fitted VQVAETokenizer
        batches: Iterable yielding (B, 1, H, W) arrays (e.g. HDF5Dataset)
        config: Model configuration dict
    """
    if not tokenizer.is_fitted:
        raise ValueError("Tokenizer not fitted. Call fit() first.")

    print(f"Tokenizing and saving to {path}...")

    # Collect tokenized indices (vectors are reconstructed from codebook on load)
    all_indices = []

    # Per-scale indices (keyed by scale index)
    per_scale_indices = {k: [] for k in range(len(tokenizer.scales))}

    n_processed = 0
    batch_count = 0

    for batch in batches:
        batch = jnp.asarray(batch)
        n_processed += batch.shape[0]
        batch_count += 1

        # Only need indices — skip vectors to avoid ~5.6 GB accumulation
        remapped, _ = tokenizer.encode_batch(batch)
        B = batch.shape[0]
        indices_flat = jnp.concatenate(
            [idx.reshape(B, -1) for idx in remapped], axis=1
        )
        all_indices.append(np.array(indices_flat))

        for scale_idx in range(len(tokenizer.scales)):
            per_scale_indices[scale_idx].append(np.array(remapped[scale_idx]))

        if batch_count % 50 == 0:
            print(f"  Processed {n_processed} samples")

    print(f"  Processed {n_processed} samples (done)")

    # Concatenate all batches
    all_indices = np.concatenate(all_indices, axis=0)

    # Prepare save dict
    save_dict = {
        "codebook": np.array(tokenizer.remapped_codebook),
        "effective_vocab_size": np.array(tokenizer.effective_vocab_size),
        "vocab_size": np.array(tokenizer.vocab_size),
        "codebook_dim": np.array(tokenizer.codebook_dim),
        "indices_flat": all_indices,
        "config_json": np.array(json.dumps(config)),
    }

    # Unified per-scale codebook mapping
    save_dict["scales"] = np.array(tokenizer.scales)
    save_dict["scale_offsets"] = np.array(tokenizer.scale_offsets)
    save_dict["scale_vocab_sizes"] = np.array(tokenizer.scale_vocab_sizes)
    save_dict["unified_to_scale"] = np.array(tokenizer.unified_to_scale)
    save_dict["unified_to_original"] = np.array(tokenizer.unified_to_original)
    if tokenizer.first_trainable_scale is not None:
        save_dict["first_trainable_scale"] = np.array(tokenizer.first_trainable_scale)

    for k, (sh, sw) in enumerate(tokenizer.scales):
        save_dict[f"indices_scale_{sh}x{sw}"] = np.concatenate(per_scale_indices[k], axis=0)
        save_dict[f"original_codebook_scale_{k}"] = np.array(tokenizer.codebooks[k])
        save_dict[f"scale_old_to_unified_{k}"] = np.array(tokenizer.scale_old_to_unified[k])

    np.savez_compressed(path, **save_dict)
    print(f"Saved tokenized data: {all_indices.shape[0]} samples, "
          f"{tokenizer.effective_vocab_size} effective vocab, "
          f"{all_indices.shape[1]} tokens per sample")


def load_tokenized_data(path: str) -> dict:
    """Load tokenized data for AR training.

    Args:
        path: Path to .npz file

    Returns:
        Dictionary with:
            - indices_flat: [N, total_tokens] discrete targets
            - vectors_flat: [N, total_tokens, D] input embeddings (reconstructed from codebook)
            - codebook: [effective_vocab, D] unified codebook
            - effective_vocab_size: int
            - vocab_size: int
            - codebook_dim: int
            - config: dict (parsed from JSON)
            - scales: tuple of (h, w) tuples
            - scale_offsets: [n_scales] start of each scale's unified range
            - scale_vocab_sizes: [n_scales] unique codes per scale
            - unified_to_scale: [unified_vocab] scale index per entry
            - unified_to_original: [unified_vocab] original codebook index per entry
            - scale_old_to_unified: list of [K] per-scale mapping arrays
            - indices_scale_{sh}x{sw}: [N, sh, sw] per-scale indices
            - original_codebooks: list of [vocab_size, D] per-scale original codebooks
            - first_trainable_scale: int (if available)
    """
    data = dict(np.load(path, allow_pickle=True))

    codebook = data["codebook"]
    indices_flat = data["indices_flat"]

    # Reconstruct vectors from codebook lookup (backward compat with old files)
    if "vectors_flat" in data:
        vectors_flat = data["vectors_flat"]
    else:
        vectors_flat = codebook[indices_flat]

    result = {
        "indices_flat": indices_flat,
        "vectors_flat": vectors_flat,
        "codebook": codebook,
        "effective_vocab_size": int(data["effective_vocab_size"]),
        "vocab_size": int(data["vocab_size"]),
        "codebook_dim": int(data["codebook_dim"]),
        "config": json.loads(str(data["config_json"])),
    }

    # Unified per-scale codebook fields
    result["scales"] = tuple(tuple(s) for s in data["scales"].tolist())
    n_scales = len(result["scales"])

    result["scale_offsets"] = data["scale_offsets"]
    result["scale_vocab_sizes"] = data["scale_vocab_sizes"]
    result["unified_to_scale"] = data["unified_to_scale"]
    result["unified_to_original"] = data["unified_to_original"]

    if "first_trainable_scale" in data:
        result["first_trainable_scale"] = int(data["first_trainable_scale"])

    scale_old_to_unified = []
    original_codebooks = []
    for k, (sh, sw) in enumerate(result["scales"]):
        key = f"indices_scale_{sh}x{sw}"
        if key in data:
            result[key] = data[key]
        mapping_key = f"scale_old_to_unified_{k}"
        if mapping_key in data:
            scale_old_to_unified.append(data[mapping_key])
        ocb_key = f"original_codebook_scale_{k}"
        if ocb_key in data:
            original_codebooks.append(data[ocb_key])
    result["scale_old_to_unified"] = scale_old_to_unified
    if original_codebooks:
        result["original_codebooks"] = original_codebooks

    return result


def print_tokenizer_info(tokenizer: VQVAETokenizer, n_samples: int):
    """Print statistics about the tokenizer and data.

    Args:
        tokenizer: Fitted VQVAETokenizer
        n_samples: Number of samples in the dataset
    """
    print("\n" + "=" * 60)
    print("TOKENIZER INFO")
    print("=" * 60)

    print(f"\nModel type: VQVAE2d (multi-scale)")
    print(f"Original vocab size (per-scale): {tokenizer.vocab_size}")
    print(f"Effective vocab size: {tokenizer.effective_vocab_size}")
    print(f"Codebook dimension: {tokenizer.codebook_dim}")

    scale_strs = [f"{sh}x{sw}" for sh, sw in tokenizer.scales]
    print(f"\nScales: {', '.join(scale_strs)}")
    print(f"Tokens per scale: {[sh*sw for sh, sw in tokenizer.scales]}")
    print(f"Total tokens per sample: {tokenizer.tokens_per_sample}")

    if tokenizer.first_trainable_scale is not None:
        print(f"First trainable scale: index {tokenizer.first_trainable_scale}")
        if tokenizer.deterministic_scales:
            det_names = [f"{sh}x{sw}" for sh, sw in tokenizer.scales[:tokenizer.first_trainable_scale]]
            print(f"Deterministic scales: {', '.join(det_names)}")

    # Unified codebook breakdown
    print(f"\nUnified codebook breakdown:")
    for k, (sh, sw) in enumerate(tokenizer.scales):
        offset = int(tokenizer.scale_offsets[k])
        n = int(tokenizer.scale_vocab_sizes[k])
        print(f"  Scale {sh}x{sw}: {n:4d} unique codes "
              f"(unified [{offset}, {offset + n}), "
              f"{sh*sw} positions/frame)")

    print(f"\nDataset size: {n_samples} samples")
    print(f"Total tokens in dataset: {n_samples * tokenizer.tokens_per_sample}")

    # Codebook usage histogram
    for k, (sh, sw) in enumerate(tokenizer.scales):
        print(f"\nCodebook usage distribution (scale {sh}x{sw}):")
        usage = np.array(tokenizer.model.quantizer.cluster_sizes[k])
        total_usage = usage.sum()
        if total_usage > 0:
            nonzero_mask = usage > 0
            print(f"  Codes with usage > 0: {nonzero_mask.sum()}")
            print(f"  Max usage: {usage.max():.0f} ({100*usage.max()/total_usage:.2f}%)")
            print(f"  Min nonzero usage: {usage[nonzero_mask].min():.0f} ({100*usage[nonzero_mask].min()/total_usage:.4f}%)")
            print(f"  Mean usage: {usage.mean():.1f}")

    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize turbulence data using trained VQ-VAE"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint"
    )
    common.add_argument(
        "--config_path", type=str, default=None,
        help="Path to config.txt file (if not given, model hyperparams must be specified on CLI)",
    )
    common.add_argument(
        "--data_path", type=str, required=True, help="Path to HDF5 data file"
    )
    common.add_argument("--field", type=str, default="omega", help="HDF5 field name")
    common.add_argument("--sample_start", type=int, default=0, help="Start sample index")
    common.add_argument("--sample_stop", type=int, default=None, help="Stop sample index (exclusive)")
    common.add_argument("--batch_size", type=int, default=128, help="Batch size (inference only, no grad graph)")
    common.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model config arguments (overridden by --config_path if given)
    common.add_argument("--hidden_dim", type=int, default=None)
    common.add_argument("--codebook_dim", type=int, default=None)
    common.add_argument("--vocab_size", type=int, default=None)
    common.add_argument("--decay", type=float, default=None)
    common.add_argument("--base_channels", type=int, default=None)
    common.add_argument(
        "--channel_mult", type=str, default=None, help="Comma-separated"
    )
    common.add_argument("--num_res_blocks", type=int, default=None)
    common.add_argument("--use_attention", action="store_true", default=None)
    common.add_argument("--no_attention", action="store_true")
    common.add_argument("--use_norm", action="store_true", default=None)
    common.add_argument("--no_norm", action="store_true")
    common.add_argument("--attention_heads", type=int, default=None)
    common.add_argument(
        "--scales", type=str, default=None,
        help="Comma-separated HxW scales (e.g. 1x1,2x2,4x4,8x8,16x16)",
    )
    common.add_argument(
        "--first_trainable_scale", type=int, default=None,
        help="Index of first trainable scale (auto-detected if not set)",
    )

    # Save command
    save_parser = subparsers.add_parser(
        "save", parents=[common], help="Tokenize and save to file"
    )
    save_parser.add_argument(
        "--output", type=str, required=True, help="Output .npz file path"
    )

    # Info command
    subparsers.add_parser(
        "info", parents=[common], help="Show tokenizer stats without saving"
    )

    return parser.parse_args()


def build_config(args) -> dict:
    """Build config dict from config file and/or CLI overrides."""
    # Start from config file if provided
    if args.config_path is not None:
        config = load_config(args.config_path)
    else:
        config = {}

    # CLI overrides (only set if explicitly provided)
    if args.hidden_dim is not None:
        config["hidden_dim"] = args.hidden_dim
    if args.codebook_dim is not None:
        config["codebook_dim"] = args.codebook_dim
    if args.vocab_size is not None:
        config["vocab_size"] = args.vocab_size
    if args.decay is not None:
        config["decay"] = args.decay
    if args.base_channels is not None:
        config["base_channels"] = args.base_channels
    if args.channel_mult is not None:
        config["channel_mult"] = tuple(int(m) for m in args.channel_mult.split(","))
    if args.num_res_blocks is not None:
        config["num_res_blocks"] = args.num_res_blocks
    if args.use_attention is not None:
        config["use_attention"] = args.use_attention and not args.no_attention
    elif args.no_attention:
        config["use_attention"] = False
    if args.use_norm is not None:
        config["use_norm"] = args.use_norm and not args.no_norm
    elif args.no_norm:
        config["use_norm"] = False
    if args.attention_heads is not None:
        config["attention_heads"] = args.attention_heads
    if args.scales is not None:
        config["scales"] = tuple(
            tuple(int(d) for d in s.split("x")) for s in args.scales.split(",")
        )
    if args.first_trainable_scale is not None:
        config["first_trainable_scale"] = args.first_trainable_scale

    return config


def main():
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use 'save' or 'info'.")
        return

    # Build config
    config = build_config(args)

    from dataloaders.hdf5_dataset import HDF5Dataset

    def make_dataset():
        return HDF5Dataset(
            data_path=args.data_path,
            fields=[args.field],
            batch_size=args.batch_size,
            shuffle=False,
            prefetch=0,
            sample_start=args.sample_start,
            sample_stop=args.sample_stop,
        )

    dataset = make_dataset()
    print(f"Dataset: {dataset.n_samples} samples from {args.data_path} (field={args.field})")

    # Load tokenizer
    print(f"Loading VQ-VAE from {args.checkpoint}...")
    key = jax.random.PRNGKey(args.seed)
    tokenizer = VQVAETokenizer.from_checkpoint(args.checkpoint, config, key)

    # Fit tokenizer (iterates dataset once)
    tokenizer.fit(dataset)

    if args.command == "info":
        print_tokenizer_info(tokenizer, dataset.n_samples)

        # Verify round-trip with a single sample
        print("Verifying round-trip encoding/decoding...")
        sample = load_data_from_hdf5(
            args.data_path,
            field=args.field,
            sample_start=args.sample_start,
            sample_stop=args.sample_start + 1,
        )
        sample = jnp.array(sample[0])  # [1, H, W]

        remapped, z_q = tokenizer.encode(sample)
        recon = tokenizer.decode_indices(remapped)

        mse = float(jnp.mean((sample - recon) ** 2))
        print(f"Round-trip MSE: {mse:.6f}")

    elif args.command == "save":
        # Fresh iterator for save pass (fit exhausted the first one)
        save_tokenized_data(
            args.output, tokenizer, make_dataset(), config
        )

        # Print info after saving
        print_tokenizer_info(tokenizer, dataset.n_samples)


if __name__ == "__main__":
    main()
