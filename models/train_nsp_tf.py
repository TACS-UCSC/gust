"""
Teacher-Forced Next-Scale Prediction (NSP) training.

VAR-style approach: predicts each scale k from scale k-1's hidden states
(bilinear upsampled), with teacher forcing at all scales.  Trains all
trainable scales in a single forward pass.

Key differences from train_nsp.py:
- No masking — teacher-forced (all tokens visible)
- t1→t1 attention: source_scale <= target_scale (within-scale self-attn)
- Sequence: full t0 + truncated t1 (last scale excluded from input)
- Prediction: coarser-scale hidden states bilinear-upsampled → expansion head
- All trainable scales per step, single optimizer chain

Usage:
    python -m models.train_nsp_tf --tokens_path tokens.npz
"""

import argparse
import json
import os
import socket

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")

from .nsp_model import (
    NextScalePredictor, NextScalePredConfig,
    create_nsp_model_from_tokenized_data,
    get_scale_ids, build_rope_coords,
)
from .tokenizer import load_tokenized_data


# =============================================================================
# Distributed init (same as train_nsp.py)
# =============================================================================

def _maybe_init_distributed():
    """Initialize JAX distributed runtime for multi-node training."""
    if "PBS_NODEFILE" not in os.environ:
        return
    nodefile = os.environ["PBS_NODEFILE"]
    nodes = open(nodefile).read().strip().splitlines()
    if len(nodes) <= 1:
        return

    try:
        jax.distributed.initialize(cluster_detection_method="mpi4py")
        return
    except Exception as e:
        print(f"mpi4py init failed ({e}), falling back to manual init")

    coordinator_ip = socket.gethostbyname(nodes[0])
    port = 29500
    num_processes = len(set(nodes))
    hostname = socket.gethostname()
    my_node = socket.gethostbyname(hostname)
    process_id = sorted(set(nodes)).index(
        next(n for n in set(nodes) if socket.gethostbyname(n) == my_node)
    )
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:{port}",
        num_processes=num_processes,
        process_id=process_id,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train AR-NSP (teacher-forced)")
    # Data
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Path to tokenized .npz file from models.tokenizer")
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: 0.02 for muon, 1e-4 for others)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="muon",
                        choices=["muon", "adamw", "lion", "adafactor"],
                        help="Optimizer to use (default: muon)")
    parser.add_argument("--muon_ns_steps", type=int, default=5,
                        help="Newton-Schulz iterations for Muon optimizer")
    # Model
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope_theta", type=float, default=16.0,
                        help="Base frequency for 2D axial RoPE (default: 16.0)")
    # Logging
    parser.add_argument("--wandb_project", type=str, default="gust-nsp")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="Wandb run ID for resuming a run across jobs")
    parser.add_argument("--log_every", type=int, default=10)
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./checkpoints_nsp_tf")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use only the first N frames (default: all)")
    return parser.parse_args()


def create_paired_dataloader(data, batch_size, shuffle=True, seed=0):
    """Yield batches of [B, 2 * tokens_per_frame] by pairing consecutive frames."""
    n_samples = len(data) - 1
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for i in range(0, n_samples - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        t0 = data[batch_indices]
        t1 = data[batch_indices + 1]
        yield jnp.array(np.concatenate([t0, t1], axis=1))


# =============================================================================
# Teacher-forced attention mask
# =============================================================================

def build_teacher_forced_mask(scales_t0, padded_len_t0,
                              scales_t1, padded_len_t1):
    """Build asymmetric attention mask for full t0 + truncated t1.

    Shape: [padded_len_t0 + padded_len_t1, padded_len_t0 + padded_len_t1]

    Quadrants:
    - t0→t0: full attention (0.0)
    - t0→t1: blocked (-1e9)
    - t1→t0: full attention (0.0)
    - t1→t1: source_scale <= target_scale (within-scale self-attn allowed)
    """
    L0 = padded_len_t0
    L1 = padded_len_t1
    total_len = L0 + L1

    full_mask = jnp.full((total_len, total_len), -1e9, dtype=jnp.float32)

    # t0→t0: full attention
    full_mask = full_mask.at[:L0, :L0].set(0.0)

    # t0→t1: remains blocked

    # t1→t0: full attention
    full_mask = full_mask.at[L0:, :L0].set(0.0)

    # t1→t1: source_scale <= target_scale
    t1_mask = _build_tf_t1_mask(scales_t1, padded_len_t1)
    full_mask = full_mask.at[L0:, L0:].set(t1_mask)

    # --- Fix padding ---
    total_tokens_t0 = sum(h * w for h, w in scales_t0)
    total_tokens_t1 = sum(h * w for h, w in scales_t1)

    is_padding = jnp.concatenate([
        jnp.arange(L0) >= total_tokens_t0,
        jnp.arange(L1) >= total_tokens_t1,
    ])

    diag_mask = jnp.eye(total_len, dtype=bool)

    # Padding rows: identity
    full_mask = jnp.where(is_padding[:, None] & diag_mask, 0.0, full_mask)

    # Block attention to padding columns
    full_mask = jnp.where(is_padding[None, :], -1e9, full_mask)

    # Re-open diagonal for padding
    full_mask = jnp.where(is_padding[:, None] & diag_mask, 0.0, full_mask)

    return full_mask


def _build_tf_t1_mask(scales, padded_len):
    """t1→t1 mask: source_scale <= target_scale (within-scale self-attn)."""
    total = sum(h * w for h, w in scales)
    n_scales = len(scales)

    scale_ids = []
    for k, (h, w) in enumerate(scales):
        scale_ids.extend([k] * (h * w))
    scale_ids = jnp.array(scale_ids, dtype=jnp.int32)
    scale_ids = jnp.pad(scale_ids, (0, padded_len - total),
                        constant_values=n_scales)

    target_scale = scale_ids[:, None]
    source_scale = scale_ids[None, :]

    # source_scale <= target_scale
    mask = (source_scale <= target_scale)

    bias = jnp.where(mask, 0.0, -1e9)
    return bias


# =============================================================================
# Expansion Heads
# =============================================================================

class ExpansionHeads(eqx.Module):
    """Per-scale prediction heads + shared target position encoding.

    Each head maps n_embd → vocab_k for its scale.
    pos_proj maps 2D normalized coordinates → n_embd, added to upsampled
    hidden states to distinguish target positions sharing the same source.
    """
    heads: list
    pos_proj: eqx.nn.Linear

    def __init__(self, config: NextScalePredConfig, trainable_indices, key):
        k1, k2 = jax.random.split(key)
        head_keys = jax.random.split(k1, len(trainable_indices))

        self.heads = []
        for i, scale_idx in enumerate(trainable_indices):
            vocab_k = config.scale_vocab_sizes[scale_idx]
            self.heads.append(
                eqx.nn.Linear(config.n_embd, vocab_k, use_bias=False,
                              key=head_keys[i])
            )

        self.pos_proj = eqx.nn.Linear(2, config.n_embd, use_bias=True, key=k2)


# =============================================================================
# Custom forward pass (asymmetric t0/t1)
# =============================================================================

def forward_teacher_forced(model, tokens_full, config,
                           scales_t0, padded_len_t0,
                           scales_t1, padded_len_t1,
                           attn_bias, token_vectors=None):
    """Forward pass supporting asymmetric t0 (full) / t1 (truncated) lengths.

    Calls model internals directly, bypassing model.__call__ which assumes
    symmetric padded_len = total_len // 2.
    """
    emb = model.embedding

    # 1. Token embeddings (no masking — teacher forced)
    if token_vectors is not None:
        vectors = jax.lax.stop_gradient(token_vectors)
    else:
        vectors = jax.lax.stop_gradient(emb.codebook[tokens_full])
    tok_emb = jax.vmap(emb.codebook_proj)(vectors)

    # 2. Scale embeddings — asymmetric
    scale_ids_t0 = get_scale_ids(scales_t0, padded_len_t0)
    scale_ids_t1 = get_scale_ids(scales_t1, padded_len_t1)
    scale_ids = jnp.concatenate([scale_ids_t0, scale_ids_t1])
    scale_ids_clamped = jnp.minimum(scale_ids, config.n_scales - 1)
    scale_emb = jax.vmap(emb.scale_embed)(scale_ids_clamped)

    # 3. Frame embeddings
    frame_ids = jnp.concatenate([
        jnp.zeros(padded_len_t0, dtype=jnp.int32),
        jnp.ones(padded_len_t1, dtype=jnp.int32),
    ])
    frame_emb = jax.vmap(emb.frame_embed)(frame_ids)

    x = tok_emb + scale_emb + frame_emb

    # 4. RoPE coords — asymmetric
    coords_t0 = build_rope_coords(scales_t0, padded_len_t0)
    coords_t1 = build_rope_coords(scales_t1, padded_len_t1)
    rope_coords = jnp.concatenate([coords_t0, coords_t1], axis=0)

    # 5. Transformer blocks (with gradient checkpointing)
    for block in model.blocks:
        x = eqx.filter_checkpoint(block)(x, attn_bias, rope_coords)

    x = jax.vmap(model.ln_f)(x)
    return x


# =============================================================================
# Loss computation — all trainable scales in one pass
# =============================================================================

def make_compute_loss(config, scales_t0, padded_len_t0,
                      scales_t1, padded_len_t1,
                      attn_bias, scale_weights, trainable_indices):
    """Build the loss function capturing static config.

    Returns a function: (model, exp_heads, batch_tokens) -> (loss, metrics)
    where metrics is a dict of per-scale losses and accuracies.
    """
    tokens_per_frame = config.tokens_per_frame

    # Boundaries for t0 (full frame) — used to source scale 0 from t0
    boundaries_t0 = [0]
    for h, w in scales_t0:
        boundaries_t0.append(boundaries_t0[-1] + h * w)

    # Boundaries for t1 (truncated) — used to find source hidden states
    boundaries_t1 = [0]
    for h, w in scales_t1:
        boundaries_t1.append(boundaries_t1[-1] + h * w)

    # Boundaries for full frame — used to find targets
    boundaries_full = config.scale_boundaries

    # Total tokens in truncated t1
    tokens_t1_trunc = sum(h * w for h, w in scales_t1)

    def compute_loss(model, exp_heads, batch_tokens):
        B = batch_tokens.shape[0]

        # Split into t0 and t1 (full, for targets)
        t0_full = batch_tokens[:, :tokens_per_frame]
        t1_full = batch_tokens[:, tokens_per_frame:]

        # Build truncated t1 input (exclude last scale's tokens)
        t1_trunc = t1_full[:, :tokens_t1_trunc]

        # Pad
        t0_pad = jnp.pad(t0_full, ((0, 0), (0, padded_len_t0 - tokens_per_frame)))
        t1_pad = jnp.pad(t1_trunc, ((0, 0), (0, padded_len_t1 - tokens_t1_trunc)))
        tokens_in = jnp.concatenate([t0_pad, t1_pad], axis=1)

        # Codebook lookup at batch level (outside vmap)
        codebook = jax.lax.stop_gradient(model.embedding.codebook)
        token_vecs = codebook[tokens_in]  # [B, L0+L1, D]

        # Forward pass via vmap
        hidden = jax.vmap(
            lambda t, v: forward_teacher_forced(
                model, t, config,
                scales_t0, padded_len_t0,
                scales_t1, padded_len_t1,
                attn_bias, token_vectors=v,
            )
        )(tokens_in, token_vecs)
        # hidden: [B, L0+L1, n_embd]

        # t0 and t1 portions of hidden states
        h_t0 = hidden[:, :padded_len_t0, :]   # [B, L0, n_embd]
        h_t1 = hidden[:, padded_len_t0:, :]   # [B, L1, n_embd]

        total_loss = jnp.float32(0.0)
        per_scale_losses = []
        per_scale_accs = []

        for i, scale_idx in enumerate(trainable_indices):
            h_k, w_k = config.scales[scale_idx]
            n_tokens_k = h_k * w_k

            if scale_idx == 0:
                # Scale 0 has no coarser scale at t1; source from t0's scale 0
                src_start = boundaries_t0[0]
                src_end = boundaries_t0[1]
                h_src, w_src = config.scales[0]
                h_source = h_t0[:, src_start:src_end, :]
            else:
                # Source: scale k-1 hidden states from t1 portion
                src_scale_idx_in_t1 = scale_idx - 1
                src_start = boundaries_t1[src_scale_idx_in_t1]
                src_end = boundaries_t1[src_scale_idx_in_t1 + 1]
                h_src, w_src = scales_t1[src_scale_idx_in_t1]
                h_source = h_t1[:, src_start:src_end, :]
            h_source_2d = h_source.reshape(B, h_src, w_src, config.n_embd)

            # Bilinear upsample to target resolution
            h_upsampled = jax.vmap(
                lambda x: jax.image.resize(
                    x, (h_k, w_k, config.n_embd), method='bilinear'
                )
            )(h_source_2d)  # [B, h_k, w_k, n_embd]

            # Target position encoding
            rows = jnp.arange(h_k, dtype=jnp.float32) / max(h_k - 1, 1)
            cols = jnp.arange(w_k, dtype=jnp.float32) / max(w_k - 1, 1)
            grid_r, grid_c = jnp.meshgrid(rows, cols, indexing='ij')
            coords = jnp.stack([grid_r, grid_c], axis=-1)  # [h_k, w_k, 2]

            pos_emb = jax.vmap(jax.vmap(exp_heads.pos_proj))(coords)  # [h_k, w_k, n_embd]
            h_positioned = h_upsampled + pos_emb[None]  # [B, h_k, w_k, n_embd]

            # Apply expansion head → logits
            h_flat = h_positioned.reshape(B, n_tokens_k, config.n_embd)
            head_idx = i  # i-th trainable scale
            logits = jax.vmap(jax.vmap(exp_heads.heads[head_idx]))(h_flat)
            # [B, n_tokens_k, vocab_k]

            # Targets from full t1 (including last scale)
            tgt_start = boundaries_full[scale_idx]
            tgt_end = boundaries_full[scale_idx + 1]
            offset = config.scale_offsets[scale_idx]
            targets = t1_full[:, tgt_start:tgt_end] - offset

            # Cross-entropy
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs, targets[:, :, None], axis=-1
            ).squeeze(-1)
            raw_loss = -jnp.mean(target_log_probs)
            weighted_loss = raw_loss * scale_weights[scale_idx]

            total_loss = total_loss + weighted_loss

            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(preds == targets)

            per_scale_losses.append(raw_loss)
            per_scale_accs.append(accuracy)

        metrics = jnp.stack(per_scale_losses + per_scale_accs)
        return total_loss, metrics

    return compute_loss


# =============================================================================
# Train step
# =============================================================================

def make_train_step(compute_loss_fn, total_devices=1, n_trainable=1):
    """Build a compiled train step function.

    Returns: step(model, exp_heads, opt_state, batch_tokens, optimizer)
             -> (model, exp_heads, opt_state, loss, metrics)
    """

    if total_devices > 1:
        @eqx.filter_pmap(in_axes=(0, 0, 0, 0, None), axis_name="batch")
        def step(model, exp_heads, opt_state, batch_tokens, optimizer):
            # Differentiate w.r.t. single tuple to keep tree structure aligned
            (loss, metrics), grads = eqx.filter_value_and_grad(
                lambda m_eh: compute_loss_fn(m_eh[0], m_eh[1], batch_tokens),
                has_aux=True,
            )((model, exp_heads))

            params = eqx.filter((model, exp_heads), eqx.is_inexact_array)

            safe_grads = jax.tree.map(
                lambda g, p: jnp.zeros_like(p) if (g is None and p is not None) else g,
                grads, params, is_leaf=lambda x: x is None,
            )

            safe_grads = jax.lax.pmean(safe_grads, "batch")
            loss = jax.lax.pmean(loss, "batch")
            metrics = jax.lax.pmean(metrics, "batch")

            updates, opt_state = optimizer.update(safe_grads, opt_state, params)

            updates = jax.tree.map(
                lambda g, u: None if g is None else u,
                grads, updates, is_leaf=lambda x: x is None,
            )

            model_updates, exp_updates = updates
            model = eqx.apply_updates(model, model_updates)
            exp_heads = eqx.apply_updates(exp_heads, exp_updates)
            return model, exp_heads, opt_state, loss, metrics
    else:
        @eqx.filter_jit
        def step(model, exp_heads, opt_state, batch_tokens, optimizer):
            # Differentiate w.r.t. single tuple to keep tree structure aligned
            (loss, metrics), grads = eqx.filter_value_and_grad(
                lambda m_eh: compute_loss_fn(m_eh[0], m_eh[1], batch_tokens),
                has_aux=True,
            )((model, exp_heads))

            params = eqx.filter((model, exp_heads), eqx.is_inexact_array)

            safe_grads = jax.tree.map(
                lambda g, p: jnp.zeros_like(p) if (g is None and p is not None) else g,
                grads, params, is_leaf=lambda x: x is None,
            )

            updates, opt_state = optimizer.update(safe_grads, opt_state, params)

            updates = jax.tree.map(
                lambda g, u: None if g is None else u,
                grads, updates, is_leaf=lambda x: x is None,
            )

            model_updates, exp_updates = updates
            model = eqx.apply_updates(model, model_updates)
            exp_heads = eqx.apply_updates(exp_heads, exp_updates)
            return model, exp_heads, opt_state, loss, metrics

    return step


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, exp_heads, opt_state, epoch, global_step,
                    checkpoint_dir, arch_config=None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    eqx.tree_serialise_leaves(
        os.path.join(checkpoint_dir, "checkpoint.eqx"), model)
    eqx.tree_serialise_leaves(
        os.path.join(checkpoint_dir, "exp_heads.eqx"), exp_heads)
    eqx.tree_serialise_leaves(
        os.path.join(checkpoint_dir, "opt_state.eqx"), opt_state)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "training_mode": "teacher_forced",
    }
    if arch_config is not None:
        state["arch_config"] = arch_config
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(state, f)

    print(f"Saved checkpoint (epoch {epoch}) to {checkpoint_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    _maybe_init_distributed()
    key = jax.random.PRNGKey(args.seed)

    # Device topology
    num_local = jax.local_device_count()
    num_processes = jax.process_count()
    total_devices = num_local * num_processes
    is_main = jax.process_index() == 0
    multi_device = total_devices > 1

    if is_main:
        print(f"Using {total_devices} device(s)"
              + (f" ({num_processes} process(es) x {num_local} local)"
                 if num_processes > 1 else ""))

    if args.batch_size % total_devices != 0:
        raise ValueError(
            f"Batch size ({args.batch_size}) must be divisible by "
            f"number of devices ({total_devices})"
        )

    # Load tokenized data
    if is_main:
        print(f"Loading data from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    scales = token_data["scales"]
    tokens_per_frame = sum(h * w for h, w in scales)

    if is_main:
        print(f"Loaded {len(indices)} frames, {tokens_per_frame} tokens/frame")
        print(f"Scales: {scales}")

    # Setup Model Config
    config = NextScalePredConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
    )

    key, model_key = jax.random.split(key)
    model = create_nsp_model_from_tokenized_data(token_data, config, model_key)

    trainable_indices = config.trainable_scale_indices
    if is_main:
        print(f"First trainable scale: {config.first_trainable_scale}")
        print(f"Trainable scales: {[config.scales[i] for i in trainable_indices]}")

    # Create expansion heads
    key, heads_key = jax.random.split(key)
    exp_heads = ExpansionHeads(config, trainable_indices, heads_key)

    if is_main:
        n_exp_params = sum(
            x.size for x in jax.tree_util.tree_leaves(
                eqx.filter(exp_heads, eqx.is_array)
            )
        )
        print(f"ExpansionHeads: {n_exp_params/1e6:.2f}M parameters")

    # Architecture config for checkpoint validation
    arch_config = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "scales": [list(s) for s in config.scales],
        "tokens_per_frame": config.tokens_per_frame,
        "unified_vocab_size": config.unified_vocab_size,
        "codebook_dim": config.codebook_dim,
        "first_trainable_scale": config.first_trainable_scale,
        "scale_vocab_sizes": list(config.scale_vocab_sizes),
        "scale_offsets": list(config.scale_offsets),
        "rope_theta": config.rope_theta,
        "training_mode": "teacher_forced",
    }

    # Compute sequence lengths
    scales_t0 = config.scales  # all scales
    scales_t1 = config.scales[:-1]  # exclude last scale (truncated)

    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)

    padded_len_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_len_t1 = ((tokens_t1 + 127) // 128) * 128

    if is_main:
        print(f"\nSequence layout:")
        print(f"  t0: {tokens_t0} tokens -> padded {padded_len_t0}")
        print(f"  t1 (truncated): {tokens_t1} tokens -> padded {padded_len_t1}")
        print(f"  Total: {padded_len_t0 + padded_len_t1} "
              f"(vs symmetric {2 * config.padded_seq_len})")

    # Build attention mask
    attn_bias = build_teacher_forced_mask(
        scales_t0, padded_len_t0, scales_t1, padded_len_t1)
    if is_main:
        print(f"Attention mask shape: {attn_bias.shape}")

    # Per-scale loss weights: 1/sqrt(token_count), normalized
    token_counts = [config.scales[i][0] * config.scales[i][1]
                    for i in trainable_indices]
    raw_weights = [1.0 / c ** 0.5 for c in token_counts]
    mean_w = sum(raw_weights) / len(raw_weights)
    scale_weights = {idx: w / mean_w
                     for idx, w in zip(trainable_indices, raw_weights)}
    if is_main:
        for idx, w in scale_weights.items():
            h, w_s = config.scales[idx]
            print(f"  Scale {h}x{w_s} ({h*w_s} tokens): loss weight = {w:.3f}")

    # Build loss function
    compute_loss_fn = make_compute_loss(
        config, scales_t0, padded_len_t0,
        scales_t1, padded_len_t1,
        attn_bias, scale_weights, trainable_indices,
    )

    # Build train step
    train_step = make_train_step(
        compute_loss_fn, total_devices=total_devices,
        n_trainable=len(trainable_indices),
    )

    # Optimizer — single chain, no multi_transform
    n_samples = len(indices) - 1
    steps_per_epoch = n_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    if args.lr is None:
        args.lr = 0.02 if args.optimizer == "muon" else 1e-4

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, args.warmup_steps, total_steps, args.lr * 0.01
    )

    if args.optimizer == "muon":
        if is_main:
            print(f"Using Muon optimizer (lr={args.lr}, "
                  f"ns_steps={args.muon_ns_steps})")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.contrib.muon(
                learning_rate=schedule,
                weight_decay=args.weight_decay,
                ns_steps=args.muon_ns_steps,
            )
        )
    elif args.optimizer == "lion":
        if is_main:
            print(f"Using Lion optimizer (lr={args.lr})")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.lion(learning_rate=schedule, weight_decay=args.weight_decay)
        )
    elif args.optimizer == "adamw":
        if is_main:
            print(f"Using AdamW optimizer (lr={args.lr})")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(schedule, weight_decay=args.weight_decay)
        )
    elif args.optimizer == "adafactor":
        if is_main:
            print(f"Using Adafactor optimizer")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adafactor(learning_rate=schedule)
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    params = eqx.filter((model, exp_heads), eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        state_path = os.path.join(args.checkpoint_dir, "training_state.json")
        model_path = os.path.join(args.checkpoint_dir, "checkpoint.eqx")
        heads_path = os.path.join(args.checkpoint_dir, "exp_heads.eqx")
        opt_path = os.path.join(args.checkpoint_dir, "opt_state.eqx")

        required = [state_path, model_path, heads_path, opt_path]
        if not all(os.path.exists(p) for p in required):
            raise FileNotFoundError(
                f"Cannot resume: missing checkpoint files in "
                f"{args.checkpoint_dir}"
            )

        with open(state_path) as f:
            training_state = json.load(f)

        # Validate architecture matches
        saved_arch = training_state.get("arch_config")
        if saved_arch is not None:
            mismatches = []
            for k, current_val in arch_config.items():
                saved_val = saved_arch.get(k)
                if saved_val is not None and saved_val != current_val:
                    mismatches.append(
                        f"  {k}: checkpoint={saved_val}, current={current_val}"
                    )
            if mismatches:
                raise ValueError(
                    "Cannot resume: model architecture mismatch:\n"
                    + "\n".join(mismatches)
                    + f"\nUse matching args or clear {args.checkpoint_dir}."
                )
        else:
            if is_main:
                print("Warning: checkpoint has no saved arch config; "
                      "skipping architecture validation")

        start_epoch = training_state["epoch"]
        global_step = training_state["global_step"]
        model = eqx.tree_deserialise_leaves(model_path, model)
        exp_heads = eqx.tree_deserialise_leaves(heads_path, exp_heads)
        opt_state = eqx.tree_deserialise_leaves(opt_path, opt_state)
        if is_main:
            print(f"Resumed from epoch {start_epoch}, "
                  f"global step {global_step}")

    # Replicate for pmap
    if multi_device:
        model = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_local,) + x.shape), model)
        exp_heads = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_local,) + x.shape), exp_heads)
        opt_state = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_local,) + x.shape), opt_state)

    # Initialize wandb
    if WANDB_AVAILABLE and is_main:
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config={**vars(args), **arch_config},
        )
        if args.wandb_id is not None:
            wandb_kwargs["id"] = args.wandb_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)

    # --- Training Loop ---
    n_trainable = len(trainable_indices)
    if is_main:
        print(f"\nStarting training: {args.epochs} epochs, "
              f"{steps_per_epoch} steps/epoch, "
              f"{n_trainable} trainable scales (all per step)")

    for epoch in range(start_epoch, args.epochs):
        if is_main:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        key, loader_key = jax.random.split(key)

        epoch_losses = []
        # Accumulate per-scale metrics across epoch
        epoch_scale_losses = {idx: [] for idx in trainable_indices}
        epoch_scale_accs = {idx: [] for idx in trainable_indices}

        dataloader = create_paired_dataloader(
            indices, args.batch_size, seed=int(loader_key[0]),
        )

        for batch_idx, batch in enumerate(dataloader):
            if multi_device:
                per_process = args.batch_size // num_processes
                start = jax.process_index() * per_process
                batch = batch[start : start + per_process]
                batch = batch.reshape(num_local, -1, batch.shape[-1])

            model, exp_heads, opt_state, loss, metrics = train_step(
                model, exp_heads, opt_state, batch, optimizer
            )

            if multi_device:
                loss = loss[0]
                metrics = metrics[0]

            # Unpack metrics: [per_scale_losses..., per_scale_accs...]
            loss_val = float(loss)
            epoch_losses.append(loss_val)

            scale_losses = metrics[:n_trainable]
            scale_accs = metrics[n_trainable:]

            for i, idx in enumerate(trainable_indices):
                epoch_scale_losses[idx].append(float(scale_losses[i]))
                epoch_scale_accs[idx].append(float(scale_accs[i]))

            if is_main and batch_idx % args.log_every == 0:
                parts = []
                for i, idx in enumerate(trainable_indices):
                    h, w = config.scales[idx]
                    parts.append(f"{h}x{w}={scale_accs[i]:.3f}")
                acc_str = " ".join(parts)
                print(f"  Step {batch_idx}: loss={loss_val:.4f} acc=[{acc_str}]")

                if WANDB_AVAILABLE:
                    log_dict = {
                        "loss": loss_val,
                        "step": global_step,
                    }
                    for i, idx in enumerate(trainable_indices):
                        h, w = config.scales[idx]
                        log_dict[f"loss/scale_{h}x{w}"] = float(scale_losses[i])
                        log_dict[f"acc/scale_{h}x{w}"] = float(scale_accs[i])
                    wandb.log(log_dict)

            global_step += 1

        # End of Epoch Summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        epoch_log = {"epoch": epoch + 1, "epoch/loss": avg_loss}

        if is_main:
            print(f"--- Epoch {epoch + 1} Summary (avg loss: {avg_loss:.4f}) ---")
        for idx in trainable_indices:
            h, w = config.scales[idx]
            if epoch_scale_accs[idx]:
                avg_acc = np.mean(epoch_scale_accs[idx])
                avg_sloss = np.mean(epoch_scale_losses[idx])
                if is_main:
                    print(f"  Scale {h}x{w}: acc={avg_acc:.4f} "
                          f"loss={avg_sloss:.4f}")
                epoch_log[f"acc/scale_{h}x{w}"] = avg_acc
                epoch_log[f"loss_epoch/scale_{h}x{w}"] = avg_sloss

        if WANDB_AVAILABLE and is_main:
            wandb.log(epoch_log)

        if is_main and (epoch + 1) % args.save_every == 0:
            save_model = (jax.tree.map(lambda x: x[0], model)
                          if multi_device else model)
            save_exp = (jax.tree.map(lambda x: x[0], exp_heads)
                        if multi_device else exp_heads)
            save_opt = (jax.tree.map(lambda x: x[0], opt_state)
                        if multi_device else opt_state)
            save_checkpoint(save_model, save_exp, save_opt,
                            epoch + 1, global_step,
                            args.checkpoint_dir, arch_config=arch_config)

    # Final Save
    if is_main:
        save_model = (jax.tree.map(lambda x: x[0], model)
                      if multi_device else model)
        save_exp = (jax.tree.map(lambda x: x[0], exp_heads)
                    if multi_device else exp_heads)
        save_opt = (jax.tree.map(lambda x: x[0], opt_state)
                    if multi_device else opt_state)
        save_checkpoint(save_model, save_exp, save_opt,
                        args.epochs, global_step,
                        args.checkpoint_dir, arch_config=arch_config)

    if WANDB_AVAILABLE and is_main and wandb.run is not None:
        wandb.finish()
    if is_main:
        print("Training complete.")


if __name__ == "__main__":
    main()
