"""Autoregressive rollout for teacher-forced NSP model.

Generates token and field .npz files. No plots — use plot_rollout.py
and video_rollout.py for visualization.

Outputs:
  - rollout_tokens.npz: predicted tokens (tokenizer-compatible format)
  - gt_tokens.npz: ground truth tokens (tokenizer-compatible format)
  - rollout_fields.npz: decoded fields (if --vqvae_checkpoint provided)
  - rollout_metrics.json: per-step token accuracy

Usage:
    # Tokens only (fast):
    python -m models.rollout_nsp_tf \
        --checkpoint_dir checkpoints_nsp_tf_25m \
        --tokens_path tokens.npz \
        --start_frame 0 --n_steps 1000 \
        --output_dir rollout_1000

    # Tokens + decoded fields:
    python -m models.rollout_nsp_tf \
        --checkpoint_dir checkpoints_nsp_tf_25m \
        --tokens_path tokens.npz \
        --start_frame 0 --n_steps 1000 \
        --output_dir rollout_1000 \
        --vqvae_checkpoint scales-B-current/vqvae_epoch_90.eqx \
        --vqvae_config scales-B-current/config.txt
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from .nsp_model import (
    NextScalePredictor, NextScalePredConfig,
    create_nsp_model_from_tokenized_data,
    get_scale_ids, build_rope_coords,
)
from .tokenizer import load_tokenized_data, unflatten_to_scales
from .train_nsp_tf import (
    ExpansionHeads,
    forward_teacher_forced,
    build_teacher_forced_mask,
)


def parse_args():
    parser = argparse.ArgumentParser(description="NSP-TF autoregressive rollout")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory with checkpoint.eqx, exp_heads.eqx, training_state.json")
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Path to tokenized .npz from models.tokenizer")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Index of the starting frame (t0)")
    parser.add_argument("--n_steps", type=int, default=10,
                        help="Number of autoregressive rollout steps")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 = argmax)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (None = no filtering)")
    parser.add_argument("--output_dir", type=str, default="./rollout_output",
                        help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42)
    # VQ-VAE decoding (optional)
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                        help="Path to VQ-VAE .eqx checkpoint for field decoding")
    parser.add_argument("--vqvae_config", type=str, default=None,
                        help="Path to VQ-VAE config.txt")
    return parser.parse_args()


def generate_t1_frame(model, exp_heads, config, t0_tokens,
                      scales_t0, padded_len_t0,
                      scales_t1, padded_len_t1,
                      attn_bias, trainable_indices,
                      *, key, temperature=0.0, top_k=None):
    """Generate a full t1 frame from t0, scale by scale."""
    boundaries = config.scale_boundaries
    tokens_per_frame = config.tokens_per_frame
    tokens_t1_trunc = sum(h * w for h, w in scales_t1)

    boundaries_t0 = [0]
    for h, w in scales_t0:
        boundaries_t0.append(boundaries_t0[-1] + h * w)

    boundaries_t1 = [0]
    for h, w in scales_t1:
        boundaries_t1.append(boundaries_t1[-1] + h * w)

    t1_tokens = jnp.zeros(tokens_per_frame, dtype=jnp.int32)

    for i, scale_idx in enumerate(trainable_indices):
        key, sample_key = jax.random.split(key)
        h_k, w_k = config.scales[scale_idx]
        n_tokens_k = h_k * w_k

        t1_trunc = t1_tokens[:tokens_t1_trunc]

        t0_pad = jnp.pad(t0_tokens, (0, padded_len_t0 - tokens_per_frame))
        t1_pad = jnp.pad(t1_trunc, (0, padded_len_t1 - tokens_t1_trunc))
        tokens_in = jnp.concatenate([t0_pad, t1_pad])

        codebook = model.embedding.codebook
        token_vecs = codebook[tokens_in]

        hidden = forward_teacher_forced(
            model, tokens_in, config,
            scales_t0, padded_len_t0,
            scales_t1, padded_len_t1,
            attn_bias, token_vectors=token_vecs,
        )

        h_t0 = hidden[:padded_len_t0, :]
        h_t1 = hidden[padded_len_t0:, :]

        if scale_idx == 0:
            src_start = boundaries_t0[0]
            src_end = boundaries_t0[1]
            h_src, w_src = config.scales[0]
            h_source = h_t0[src_start:src_end, :]
        else:
            src_scale_idx_in_t1 = scale_idx - 1
            src_start = boundaries_t1[src_scale_idx_in_t1]
            src_end = boundaries_t1[src_scale_idx_in_t1 + 1]
            h_src, w_src = scales_t1[src_scale_idx_in_t1]
            h_source = h_t1[src_start:src_end, :]

        h_source_2d = h_source.reshape(h_src, w_src, config.n_embd)

        h_upsampled = jax.image.resize(
            h_source_2d, (h_k, w_k, config.n_embd), method='bilinear'
        )

        rows = jnp.arange(h_k, dtype=jnp.float32) / max(h_k - 1, 1)
        cols = jnp.arange(w_k, dtype=jnp.float32) / max(w_k - 1, 1)
        grid_r, grid_c = jnp.meshgrid(rows, cols, indexing='ij')
        coords = jnp.stack([grid_r, grid_c], axis=-1)

        pos_emb = jax.vmap(jax.vmap(exp_heads.pos_proj))(coords)
        h_positioned = h_upsampled + pos_emb

        h_flat = h_positioned.reshape(n_tokens_k, config.n_embd)
        logits = jax.vmap(exp_heads.heads[i])(h_flat)

        if temperature <= 0:
            predicted = jnp.argmax(logits, axis=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                top_vals = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0]
                threshold = top_vals[:, -1:]
                logits = jnp.where(logits < threshold, -1e9, logits)
            predicted = jax.random.categorical(sample_key, logits)

        unified_indices = predicted + config.scale_offsets[scale_idx]

        tgt_start = boundaries[scale_idx]
        tgt_end = boundaries[scale_idx + 1]
        t1_tokens = t1_tokens.at[tgt_start:tgt_end].set(unified_indices)

    return t1_tokens


def compute_token_accuracy(pred_tokens, gt_tokens, config):
    """Compute per-scale and overall token accuracy."""
    boundaries = config.scale_boundaries
    results = {}
    total_correct = 0
    total_tokens = 0

    for scale_idx in config.trainable_scale_indices:
        start = boundaries[scale_idx]
        end = boundaries[scale_idx + 1]
        pred_k = pred_tokens[start:end]
        gt_k = gt_tokens[start:end]
        correct = int(jnp.sum(pred_k == gt_k))
        n = end - start
        h, w = config.scales[scale_idx]
        results[f"scale_{h}x{w}"] = correct / n
        total_correct += correct
        total_tokens += n

    results["overall"] = total_correct / total_tokens
    return results


def _save_tokenizer_npz(path, src_data, token_array, scales, extra_meta=None):
    """Save tokens in tokenizer-compatible .npz format."""
    save_dict = {}
    metadata_keys = [
        "codebook", "effective_vocab_size", "vocab_size", "codebook_dim",
        "config_json", "scales", "scale_offsets", "scale_vocab_sizes",
        "unified_to_scale", "unified_to_original", "first_trainable_scale",
    ]
    for k in metadata_keys:
        if k in src_data:
            save_dict[k] = src_data[k]

    for k in src_data:
        if k.startswith("original_codebook_scale_") or k.startswith("scale_old_to_unified_"):
            save_dict[k] = src_data[k]

    save_dict["indices_flat"] = token_array

    for si, (sh, sw) in enumerate(scales):
        per_scale = []
        for frame in token_array:
            idx_list = unflatten_to_scales(frame, scales)
            per_scale.append(np.array(idx_list[si]))
        save_dict[f"indices_scale_{sh}x{sw}"] = np.stack(per_scale)

    if extra_meta:
        for k, v in extra_meta.items():
            save_dict[k] = np.array(v)

    np.savez_compressed(path, **save_dict)


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenized data
    print(f"Loading tokenized data from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    scales = token_data["scales"]
    tokens_per_frame = sum(h * w for h, w in scales)
    print(f"  {len(indices)} frames, {tokens_per_frame} tokens/frame, {len(scales)} scales")

    # Load training state for architecture config
    state_path = os.path.join(args.checkpoint_dir, "training_state.json")
    with open(state_path) as f:
        training_state = json.load(f)
    arch = training_state["arch_config"]
    print(f"  Architecture: {arch['n_layer']}L {arch['n_head']}H {arch['n_embd']}D")

    # Setup model config
    config = NextScalePredConfig(
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        n_embd=arch["n_embd"],
        dropout=0.0,
        rope_theta=arch.get("rope_theta", 16.0),
    )

    key, model_key = jax.random.split(key)
    model = create_nsp_model_from_tokenized_data(token_data, config, model_key)

    model_path = os.path.join(args.checkpoint_dir, "checkpoint.eqx")
    model = eqx.tree_deserialise_leaves(model_path, model)
    model = eqx.nn.inference_mode(model)
    print(f"  Loaded model from {model_path}")

    trainable_indices = config.trainable_scale_indices
    key, heads_key = jax.random.split(key)
    exp_heads = ExpansionHeads(config, trainable_indices, heads_key)
    heads_path = os.path.join(args.checkpoint_dir, "exp_heads.eqx")
    exp_heads = eqx.tree_deserialise_leaves(heads_path, exp_heads)
    exp_heads = eqx.nn.inference_mode(exp_heads)
    print(f"  Loaded expansion heads from {heads_path}")
    print(f"  Trainable scales: {[config.scales[i] for i in trainable_indices]}")

    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)
    padded_len_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_len_t1 = ((tokens_t1 + 127) // 128) * 128
    print(f"  Sequence: t0={tokens_t0}->{padded_len_t0}, "
          f"t1_trunc={tokens_t1}->{padded_len_t1}")

    attn_bias = build_teacher_forced_mask(
        scales_t0, padded_len_t0, scales_t1, padded_len_t1)

    # Optionally load VQ-VAE for decoding
    vqvae_model = None
    if args.vqvae_checkpoint and args.vqvae_config:
        from .tokenizer import load_config, load_vqvae_checkpoint
        vqvae_config = load_config(args.vqvae_config)
        key, vqvae_key = jax.random.split(key)
        vqvae_model = load_vqvae_checkpoint(
            args.vqvae_checkpoint, vqvae_config, vqvae_key)
        vqvae_model = eqx.nn.inference_mode(vqvae_model)
        print(f"  Loaded VQ-VAE from {args.vqvae_checkpoint}")

    # Validate start frame
    max_start = len(indices) - args.n_steps - 1
    if args.start_frame > max_start:
        print(f"Warning: start_frame {args.start_frame} too large, "
              f"clamping to {max(0, max_start)}.")
        args.start_frame = max(0, max_start)

    # --- Rollout ---
    print(f"\nRolling out {args.n_steps} steps from frame {args.start_frame}...")
    t0_tokens = jnp.array(indices[args.start_frame])

    rollout_tokens = [np.array(t0_tokens)]
    gt_tokens = [np.array(indices[args.start_frame])]
    all_accuracies = []

    if args.n_steps <= 20:
        log_every = 1
    elif args.n_steps <= 200:
        log_every = 10
    else:
        log_every = 50

    t_start = time.time()

    for step in range(args.n_steps):
        key, gen_key = jax.random.split(key)

        t1_pred = generate_t1_frame(
            model, exp_heads, config, t0_tokens,
            scales_t0, padded_len_t0,
            scales_t1, padded_len_t1,
            attn_bias, trainable_indices,
            key=gen_key,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        gt_frame_idx = args.start_frame + step + 1
        gt_t1 = jnp.array(indices[gt_frame_idx])
        acc = compute_token_accuracy(t1_pred, gt_t1, config)
        all_accuracies.append(acc)

        if (step + 1) % log_every == 0 or step == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t_start
            steps_done = step + 1
            sec_per_step = elapsed / steps_done
            eta = sec_per_step * (args.n_steps - steps_done)

            scale_parts = []
            for scale_idx in trainable_indices:
                h, w = config.scales[scale_idx]
                scale_parts.append(f"{h}x{w}={acc[f'scale_{h}x{w}']:.3f}")
            print(f"  Step {steps_done}/{args.n_steps}: "
                  f"acc={acc['overall']:.4f}  "
                  f"[{' '.join(scale_parts)}]  "
                  f"({sec_per_step:.1f}s/step, ETA {eta/60:.1f}min)")

        rollout_tokens.append(np.array(t1_pred))
        gt_tokens.append(np.array(indices[gt_frame_idx]))
        t0_tokens = t1_pred

    elapsed_total = time.time() - t_start
    print(f"\nRollout done: {args.n_steps} steps in {elapsed_total/60:.1f} min "
          f"({elapsed_total/args.n_steps:.1f}s/step)")

    rollout_tokens = np.stack(rollout_tokens)
    gt_tokens = np.stack(gt_tokens)

    # --- Save tokens ---
    print("Saving tokens...")
    src_data = dict(np.load(args.tokens_path, allow_pickle=True))
    extra = {
        "rollout_start_frame": args.start_frame,
        "rollout_n_steps": args.n_steps,
        "rollout_temperature": args.temperature,
    }

    rollout_npz = os.path.join(args.output_dir, "rollout_tokens.npz")
    _save_tokenizer_npz(rollout_npz, src_data, rollout_tokens, scales, extra)
    print(f"  {rollout_npz} ({rollout_tokens.shape[0]} frames)")

    gt_npz = os.path.join(args.output_dir, "gt_tokens.npz")
    _save_tokenizer_npz(gt_npz, src_data, gt_tokens, scales, extra)
    print(f"  {gt_npz} ({gt_tokens.shape[0]} frames)")

    # --- Save metrics ---
    summary = {
        "start_frame": args.start_frame,
        "n_steps": args.n_steps,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "elapsed_seconds": elapsed_total,
        "per_step": all_accuracies,
    }
    with open(os.path.join(args.output_dir, "rollout_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Decode fields ---
    if vqvae_model is not None:
        print("\nDecoding tokens to fields...")
        unified_to_original = jnp.array(token_data["unified_to_original"])

        def decode_frame(flat_tokens):
            indices_list = unflatten_to_scales(flat_tokens, scales)
            original_indices = [unified_to_original[idx] for idx in indices_list]
            return vqvae_model.decode_indices(original_indices)

        rollout_fields = []
        gt_fields = []
        n_total = len(rollout_tokens)
        for i in range(n_total):
            rollout_fields.append(np.array(decode_frame(jnp.array(rollout_tokens[i]))))
            gt_fields.append(np.array(decode_frame(jnp.array(gt_tokens[i]))))
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Decoded {i + 1}/{n_total}")

        rollout_fields = np.stack(rollout_fields)
        gt_fields = np.stack(gt_fields)

        fields_path = os.path.join(args.output_dir, "rollout_fields.npz")
        np.savez(fields_path, rollout=rollout_fields, ground_truth=gt_fields)
        print(f"  Saved {fields_path}")

    print("\nRollout complete.")


if __name__ == "__main__":
    main()
