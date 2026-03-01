"""
Next-Scale Prediction (NSP) model for turbulence generation.

Predicts all tokens in a scale simultaneously at t1, conditioned on:
1. All scales at t0 (Full Context)
2. All coarser scales at t1 (Block-Causal)
"""

import math
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NextScalePredConfig:
    tokens_per_frame: int = 341
    scales: tuple = ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16))
    scale_vocab_sizes: tuple = (1, 1, 1, 35, 361)
    scale_offsets: tuple = (0, 1, 2, 3, 38)

    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True

    codebook_dim: int = 32
    unified_vocab_size: int = 399
    first_trainable_scale: int = 0

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def scale_boundaries(self) -> list:
        """Cumulative token positions per scale."""
        boundaries = [0]
        for (h, w) in self.scales:
            boundaries.append(boundaries[-1] + h * w)
        return boundaries

    @property
    def padded_seq_len(self) -> int:
        """Round tokens_per_frame up to next multiple of 128 for Flash Attention."""
        return ((self.tokens_per_frame + 127) // 128) * 128

    @property
    def total_seq_len(self) -> int:
        """Total sequence length for t0 + t1."""
        return 2 * self.padded_seq_len

    @property
    def trainable_scale_indices(self) -> list:
        """Range of trainable scale indices."""
        return list(range(self.first_trainable_scale, self.n_scales))

    @property
    def n_trainable_scales(self) -> int:
        return self.n_scales - self.first_trainable_scale


# =============================================================================
# Block-Causal Mask
# =============================================================================

def build_temporal_mask(scales: tuple, padded_len: int) -> jnp.ndarray:
    """Build attention mask for [t0, t1] autoregressive generation.

    Shape: [2*padded_len, 2*padded_len]

    Quadrants:
    1. Top-Left (t0 -> t0): Full Attention (t0 is fully visible context)
    2. Top-Right (t0 -> t1): Masked (Future cannot influence past)
    3. Bottom-Left (t1 -> t0): Full Attention (t1 sees all of t0)
    4. Bottom-Right (t1 -> t1): Block-Causal (NSP logic for generation)
    """
    total_len = 2 * padded_len

    # 1. Base Block-Causal Mask for t1 -> t1
    single_frame_mask = _build_single_frame_mask(scales, padded_len)

    # Initialize with masking (large negative)
    # Using -1e9 instead of -inf for bfloat16 stability
    full_mask = jnp.full((total_len, total_len), -1e9, dtype=jnp.float32)

    # --- Q: t0 (0:L), K: t0 (0:L) ---
    full_mask = full_mask.at[:padded_len, :padded_len].set(0.0)

    # --- Q: t0 (0:L), K: t1 (L:2L) ---
    # Remains -1e9 (Masked)

    # --- Q: t1 (L:2L), K: t0 (0:L) ---
    full_mask = full_mask.at[padded_len:, :padded_len].set(0.0)

    # --- Q: t1 (L:2L), K: t1 (L:2L) ---
    full_mask = full_mask.at[padded_len:, padded_len:].set(single_frame_mask)

    # --- Fix Padding ---
    total_tokens = sum(h * w for h, w in scales)

    is_padding = jnp.arange(total_len) % padded_len >= total_tokens
    diag_mask = jnp.eye(total_len, dtype=bool)

    # Apply identity to padding rows
    full_mask = jnp.where(is_padding[:, None] & diag_mask, 0.0, full_mask)

    # Mask attention FROM real tokens TO padding tokens
    is_padding_col = jnp.arange(total_len) % padded_len >= total_tokens
    full_mask = jnp.where(is_padding_col[None, :], -1e9, full_mask)

    # Re-open diagonal for padding (safety)
    full_mask = jnp.where(is_padding[:, None] & diag_mask, 0.0, full_mask)

    return full_mask


def _build_single_frame_mask(scales: tuple, padded_len: int) -> jnp.ndarray:
    """Helper: Standard NSP block-causal mask."""
    total = sum(h * w for h, w in scales)
    n_scales = len(scales)

    scale_ids = []
    for k, (h, w) in enumerate(scales):
        scale_ids.extend([k] * (h * w))
    scale_ids = jnp.array(scale_ids, dtype=jnp.int32)
    scale_ids = jnp.pad(scale_ids, (0, padded_len - total), constant_values=n_scales)

    target_scale = scale_ids[:, None]
    source_scale = scale_ids[None, :]

    # Rule 1: Scale K < Scale Target
    mask = (source_scale < target_scale)
    # Rule 2: Scale 0 sees Scale 0
    mask = mask | ((target_scale == 0) & (source_scale == 0))

    bias = jnp.where(mask, 0.0, -1e9)
    return bias


def get_scale_ids(scales: tuple, padded_len: int) -> jnp.ndarray:
    n_scales = len(scales)
    scale_ids = []
    for k, (h, w) in enumerate(scales):
        scale_ids.extend([k] * (h * w))
    scale_ids = jnp.array(scale_ids, dtype=jnp.int32)
    scale_ids = jnp.pad(scale_ids, (0, padded_len - len(scale_ids)),
                        constant_values=n_scales)
    return scale_ids


# =============================================================================
# Embedding Module
# =============================================================================

class NextScaleEmbedding(eqx.Module):
    """Embedding layer for t0 -> t1 NSP model.

    Embeds [t0, t1] sequence with:
    1. Codebook vectors
    2. Positional embeddings (repeated for t0 and t1)
    3. Scale embeddings (repeated for t0 and t1)
    4. Frame embeddings (0 for t0, 1 for t1)
    """
    codebook: jax.Array               # [unified_vocab_size, codebook_dim]
    codebook_proj: eqx.nn.Linear      # codebook_dim -> n_embd
    pos_embed: eqx.nn.Embedding       # [tokens_per_frame, n_embd]
    scale_embed: eqx.nn.Embedding     # [n_scales, n_embd]
    frame_embed: eqx.nn.Embedding     # [2, n_embd]
    mask_token: jax.Array             # [n_embd]

    _config: NextScalePredConfig = eqx.field(static=True)

    def __init__(self, config: NextScalePredConfig, codebook: jax.Array, key):
        self._config = config
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.codebook = codebook
        self.codebook_proj = eqx.nn.Linear(
            config.codebook_dim, config.n_embd, use_bias=config.bias, key=k1
        )
        self.pos_embed = eqx.nn.Embedding(
            config.tokens_per_frame, config.n_embd, key=k2
        )
        self.scale_embed = eqx.nn.Embedding(
            config.n_scales, config.n_embd, key=k3
        )
        self.frame_embed = eqx.nn.Embedding(
            2, config.n_embd, key=k4
        )
        self.mask_token = jax.random.normal(k5, (config.n_embd,)) * 0.02

    def __call__(self, tokens: jax.Array, mask_positions: jax.Array,
                 token_vectors: Optional[jax.Array] = None) -> jax.Array:
        """
        Args:
            tokens: [2 * padded_len] indices for t0 and t1
            mask_positions: [2 * padded_len] boolean mask (True = replace with mask token)
            token_vectors: Optional [2 * padded_len, codebook_dim] pre-looked-up
                codebook vectors.  When provided the codebook gather is skipped
                (avoids sharding ambiguity when called inside vmap with sharded
                batches).
        """
        total_len = tokens.shape[0]
        padded_len = total_len // 2
        tpf = self._config.tokens_per_frame

        # 1. Token Embeddings
        if token_vectors is not None:
            vectors = jax.lax.stop_gradient(token_vectors)
        else:
            vectors = jax.lax.stop_gradient(self.codebook[tokens])
        tok_emb = jax.vmap(self.codebook_proj)(vectors)

        # Masking (only affects t1 targets)
        tok_emb = jnp.where(mask_positions[:, None], self.mask_token[None, :], tok_emb)

        # 2. Positional Embedding (Repeat [0..tpf-1] for t0 and t1)
        raw_positions = jnp.arange(padded_len) % tpf
        positions = jnp.concatenate([raw_positions, raw_positions])
        pos_emb = jax.vmap(self.pos_embed)(positions)

        # 3. Scale Embedding (Repeat)
        scale_ids_single = get_scale_ids(self._config.scales, padded_len)
        scale_ids = jnp.concatenate([scale_ids_single, scale_ids_single])
        scale_ids_clamped = jnp.minimum(scale_ids, self._config.n_scales - 1)
        scale_emb = jax.vmap(self.scale_embed)(scale_ids_clamped)

        # 4. Frame Embedding (0 for t0, 1 for t1)
        frame_ids = jnp.concatenate([
            jnp.zeros(padded_len, dtype=jnp.int32),
            jnp.ones(padded_len, dtype=jnp.int32)
        ])
        frame_emb = jax.vmap(self.frame_embed)(frame_ids)

        return tok_emb + pos_emb + scale_emb + frame_emb


# =============================================================================
# Attention and MLP (Standard)
# =============================================================================

class BlockCausalAttention(eqx.Module):
    """Multi-head attention with block-causal masking via additive bias."""
    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    attn_dropout: eqx.nn.Dropout
    resid_dropout: eqx.nn.Dropout

    _config: NextScalePredConfig = eqx.field(static=True)

    def __init__(self, config: NextScalePredConfig, key):
        assert config.n_embd % config.n_head == 0
        k1, k2 = jax.random.split(key)

        self.c_attn = eqx.nn.Linear(
            config.n_embd, 3 * config.n_embd, use_bias=config.bias, key=k1
        )
        self.c_proj = eqx.nn.Linear(
            config.n_embd, config.n_embd, use_bias=config.bias, key=k2
        )
        self.attn_dropout = eqx.nn.Dropout(config.dropout)
        self.resid_dropout = eqx.nn.Dropout(config.dropout)
        self._config = config

    def __call__(self, x, attn_bias, *, key=None):
        T, C = x.shape
        n_head = self._config.n_head
        head_dim = C // n_head

        qkv = jax.vmap(self.c_attn)(x)
        q, k, v = jnp.split(qkv, 3, axis=1)

        q = q.reshape(T, n_head, head_dim)
        k = k.reshape(T, n_head, head_dim)
        v = v.reshape(T, n_head, head_dim)

        q_bf16 = q.astype(jnp.bfloat16)
        k_bf16 = k.astype(jnp.bfloat16)
        v_bf16 = v.astype(jnp.bfloat16)

        # Broadcast bias to heads: [1, 1, T, T]
        bias = attn_bias[None, None, :, :].astype(jnp.bfloat16)

        y = jax.nn.dot_product_attention(
            q_bf16, k_bf16, v_bf16,
            bias=bias,
            implementation='cudnn'
        )
        y = y.reshape(T, C)
        y = jax.vmap(self.c_proj)(y)

        if key is not None:
            y = self.resid_dropout(y, key=key)
        return y


class MLP(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(self, config: NextScalePredConfig, key):
        k1, k2, k3 = jax.random.split(key, 3)
        hidden_dim = (4 * config.n_embd * 2) // 3
        # Round up to multiple of 64 for hardware efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.gate_proj = eqx.nn.Linear(config.n_embd, hidden_dim, use_bias=False, key=k1)
        self.up_proj   = eqx.nn.Linear(config.n_embd, hidden_dim, use_bias=False, key=k2)
        self.down_proj = eqx.nn.Linear(hidden_dim, config.n_embd, use_bias=False, key=k3)
        self.dropout   = eqx.nn.Dropout(config.dropout)

    def __call__(self, x, *, key=None):
        gate = jax.nn.silu(jax.vmap(self.gate_proj)(x))
        up   = jax.vmap(self.up_proj)(x)
        x = jax.vmap(self.down_proj)(gate * up)
        if key is not None:
            x = self.dropout(x, key=key)
        return x


class Block(eqx.Module):
    ln_1: eqx.nn.LayerNorm
    attn: BlockCausalAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, config: NextScalePredConfig, key):
        k1, k2 = jax.random.split(key)
        self.ln_1 = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = BlockCausalAttention(config, k1)
        self.ln_2 = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = MLP(config, k2)

    def __call__(self, x, attn_bias, *, key=None):
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None
        x = x + self.attn(jax.vmap(self.ln_1)(x), attn_bias, key=k1)
        x = x + self.mlp(jax.vmap(self.ln_2)(x), key=k2)
        return x


# =============================================================================
# Full Model
# =============================================================================

class NextScalePredictor(eqx.Module):
    _config: NextScalePredConfig = eqx.field(static=True)
    embedding: NextScaleEmbedding
    drop: eqx.nn.Dropout
    blocks: list
    ln_f: eqx.nn.LayerNorm
    scale_heads: list

    def __init__(self, config: NextScalePredConfig, codebook: jax.Array, key):
        self._config = config
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.embedding = NextScaleEmbedding(config, codebook, k1)
        self.drop = eqx.nn.Dropout(config.dropout)

        block_keys = jax.random.split(k2, config.n_layer)
        self.blocks = [Block(config, bk) for bk in block_keys]

        self.ln_f = eqx.nn.LayerNorm(config.n_embd, use_bias=config.bias)

        head_keys = jax.random.split(k3, config.n_trainable_scales)
        self.scale_heads = []
        for i, scale_idx in enumerate(config.trainable_scale_indices):
            vocab_k = config.scale_vocab_sizes[scale_idx]
            self.scale_heads.append(
                eqx.nn.Linear(config.n_embd, vocab_k, use_bias=False, key=head_keys[i])
            )
        print(f"NextScalePredictor: {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )
        if non_embedding:
            n_params -= self.embedding.pos_embed.weight.size
        return n_params

    def __call__(self, tokens: jax.Array, mask_positions: jax.Array,
                 attn_bias: jax.Array, token_vectors: Optional[jax.Array] = None,
                 *, key=None) -> jax.Array:
        """
        Args:
            tokens: [2 * padded_len] indices (t0 concatenated with t1)
            mask_positions: [2 * padded_len] boolean mask (t1 target positions True)
            attn_bias: [2 * padded_len, 2 * padded_len] temporal mask
            token_vectors: Optional [2 * padded_len, codebook_dim] pre-looked-up
                vectors (pass to avoid gather inside vmap with sharded batches)
        """
        x = self.embedding(tokens, mask_positions, token_vectors=token_vectors)
        if key is not None:
            key, drop_key = jax.random.split(key)
            x = self.drop(x, key=drop_key)

        for block in self.blocks:
            if key is not None:
                key, block_key = jax.random.split(key)
            else:
                block_key = None
            x = eqx.filter_checkpoint(block)(x, attn_bias, key=block_key)

        x = jax.vmap(self.ln_f)(x)
        return x

    def predict_scale(self, hidden_states: jax.Array,
                      target_scale_idx: int) -> jax.Array:
        """
        Args:
            hidden_states: [2 * padded_len, n_embd]
            target_scale_idx: absolute scale index
        """
        t1_start_idx = self._config.padded_seq_len

        boundaries = self._config.scale_boundaries
        start = t1_start_idx + boundaries[target_scale_idx]
        end = t1_start_idx + boundaries[target_scale_idx + 1]

        h = hidden_states[start:end]
        head_idx = target_scale_idx - self._config.first_trainable_scale
        logits = jax.vmap(self.scale_heads[head_idx])(h)
        return logits

    def generate(self, codebook: jax.Array, context_frame: jax.Array,
                 seed_tokens: jax.Array, attn_bias: jax.Array, *, key,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> jax.Array:
        """Generate t1 frame conditioned on t0 context frame.

        Args:
            codebook: [unified_vocab, D]
            context_frame: [tokens_per_frame] t0 indices (fully known)
            seed_tokens: [n_det_tokens] t1 deterministic prefix (deterministic scales)
            attn_bias: [2*padded, 2*padded] mask

        Returns:
            [tokens_per_frame] generated t1 tokens
        """
        config = self._config
        boundaries = config.scale_boundaries
        padded_len = config.padded_seq_len

        # Setup t0 (Context)
        t0_padded = jnp.pad(context_frame, (0, padded_len - len(context_frame)), constant_values=0)

        # Setup t1 (Target)
        n_det = boundaries[config.first_trainable_scale]
        t1_tokens = jnp.zeros(config.tokens_per_frame, dtype=jnp.int32)
        t1_tokens = t1_tokens.at[:n_det].set(seed_tokens[:n_det])

        # Iterative generation over scales
        for scale_idx in config.trainable_scale_indices:
            key, sample_key = jax.random.split(key)
            start = boundaries[scale_idx]
            end = boundaries[scale_idx + 1]

            # 1. Build input [t0, t1]
            t1_padded = jnp.pad(t1_tokens, (0, padded_len - config.tokens_per_frame), constant_values=0)
            full_input = jnp.concatenate([t0_padded, t1_padded])

            # 2. Build Mask (Mask this scale and future scales in t1)
            mask_positions = jnp.zeros(2 * padded_len, dtype=jnp.bool_)
            t1_offset = padded_len
            mask_positions = mask_positions.at[t1_offset + start : t1_offset + config.tokens_per_frame].set(True)

            # 3. Forward
            hidden = self(full_input, mask_positions, attn_bias)

            # 4. Predict
            logits = self.predict_scale(hidden, scale_idx)
            logits = logits / temperature

            if top_k is not None:
                top_vals, _ = lax.top_k(logits, min(top_k, logits.shape[-1]))
                threshold = top_vals[:, -1:]
                logits = jnp.where(logits < threshold, float('-inf'), logits)

            sampled = jax.random.categorical(sample_key, logits)
            unified_indices = sampled + config.scale_offsets[scale_idx]

            t1_tokens = t1_tokens.at[start:end].set(unified_indices)

        return t1_tokens


def create_nsp_model_from_tokenized_data(
    token_data: dict, config: NextScalePredConfig, key
) -> NextScalePredictor:
    """Create an NSP model initialized from tokenized data.

    Args:
        token_data: Dict from ``tokenizer.load_tokenized_data()``
        config: NextScalePredConfig (model hyperparams; data-derived fields
                are overwritten from token_data)
        key: JAX random key

    Returns:
        Initialized NextScalePredictor
    """
    codebook = jnp.array(token_data['codebook'])

    config.unified_vocab_size = int(token_data['effective_vocab_size'])
    config.codebook_dim = int(token_data['codebook_dim'])
    config.scales = tuple(tuple(s) for s in token_data['scales'])
    config.tokens_per_frame = sum(h * w for h, w in config.scales)
    config.scale_offsets = tuple(int(x) for x in token_data['scale_offsets'])
    config.scale_vocab_sizes = tuple(int(x) for x in token_data['scale_vocab_sizes'])
    config.first_trainable_scale = int(token_data.get('first_trainable_scale', 0))

    return NextScalePredictor(config, codebook, key)
