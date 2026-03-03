"""
Training script for Autoregressive Next-Scale Prediction (NSP).

Trains the model to predict t1 given t0 (context) and coarser scales of t1.
Data is loaded as pairs [t0, t1].

Optimizer uses optax.multi_transform for per-head state isolation:
- Trunk (backbone) params get a standard optimizer chain
- Each scale head gets a skip_on_zero_grads wrapper that freezes optimizer
  state when the head is inactive (not the target scale for a given step)

Optimizer choices: muon (default), lion, adamw, adafactor.
Muon (Momentum + Newton-Schulz orthogonalization) applies NS iterations to
orthogonalize momentum for 2D weight matrices and falls back to AdamW for
non-matrix params (embeddings, biases, layer norms).

Single-GPU: standard filter_jit.
Multi-GPU:  filter_pmap with lax.pmean for data-parallel gradient averaging.

Usage:
    python -m models.train_nsp --tokens_path tokens.npz
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
    build_temporal_mask, create_nsp_model_from_tokenized_data,
)
from .tokenizer import load_tokenized_data


def _maybe_init_distributed():
    """Initialize JAX distributed runtime for multi-node training.

    Tries mpi4py auto-detection (works with cray-mpich on Derecho),
    falls back to manual PBS_NODEFILE parsing.
    No-op when not running under MPI / PBS multi-node.
    """
    if "PBS_NODEFILE" not in os.environ:
        return
    nodefile = os.environ["PBS_NODEFILE"]
    nodes = open(nodefile).read().strip().splitlines()
    if len(nodes) <= 1:
        return  # single-node, no distributed init needed

    # Try mpi4py first (auto-detects coordinator, rank, world size)
    try:
        jax.distributed.initialize(cluster_detection_method="mpi4py")
        return
    except Exception as e:
        print(f"mpi4py init failed ({e}), falling back to manual init")

    # Manual fallback: parse PBS_NODEFILE
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
    parser = argparse.ArgumentParser(description="Train AR-NSP model")
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
    # Logging
    parser.add_argument("--wandb_project", type=str, default="gust-nsp")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="Wandb run ID for resuming a run across jobs")
    parser.add_argument("--log_every", type=int, default=10)
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_nsp")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in checkpoint_dir")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_paired_dataloader(data: np.ndarray, batch_size: int,
                             shuffle: bool = True, seed: int = 0):
    """Yield batches of [B, 2 * tokens_per_frame] by pairing consecutive frames.

    Args:
        data: [N, tokens_per_frame] array of tokenized frames
        batch_size: Number of pairs per batch
        shuffle: Whether to shuffle pair indices
        seed: Random seed for shuffling
    """
    n_samples = len(data) - 1  # Need pairs
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
# Per-head optimizer isolation
# =============================================================================

def build_param_labels(model):
    """Label params: 'trunk' for backbone, 'head_i' for each scale head."""
    params = eqx.filter(model, eqx.is_inexact_array)

    def _label(path, _leaf):
        for j, key in enumerate(path):
            if (isinstance(key, jax.tree_util.GetAttrKey)
                    and key.name == 'scale_heads'
                    and j + 1 < len(path)):
                next_key = path[j + 1]
                return f"head_{next_key.idx}"
        return "trunk"

    return jax.tree_util.tree_map_with_path(_label, params)


def skip_on_zero_grads(inner):
    """Skip optimizer state update when grads are all zeros (inactive head).

    When a scale head is not the target for a training step, its gradients
    are None (filled to zeros for optax compatibility). This wrapper detects
    exact-zero gradient norm and returns zero updates with unchanged state,
    preventing momentum/EMA corruption from zero-grad steps.
    """
    def init_fn(params):
        return inner.init(params)

    def update_fn(updates, state, params=None):
        grad_norm = optax.global_norm(updates)
        return jax.lax.cond(
            grad_norm > 0,
            lambda: inner.update(updates, state, params),
            lambda: (jax.tree.map(jnp.zeros_like, updates), state),
        )

    return optax.GradientTransformation(init_fn, update_fn)


# =============================================================================
# Train step
# =============================================================================

def make_train_step(config: NextScalePredConfig, target_scale_idx: int,
                    attn_bias: jax.Array, total_devices: int = 1):
    """Create train step for t1 prediction at a specific scale.

    Returns a filter_jit'd function for single GPU, or a filter_pmap'd
    function (with lax.pmean gradient averaging) for multi-GPU.
    """

    boundaries = config.scale_boundaries
    padded_len = config.padded_seq_len

    scale_start = boundaries[target_scale_idx]
    scale_end = boundaries[target_scale_idx + 1]

    offset = config.scale_offsets[target_scale_idx]
    head_idx = target_scale_idx - config.first_trainable_scale

    # Precompute mask: Mask t1 target scale and beyond
    mask_positions = jnp.zeros(2 * padded_len, dtype=jnp.bool_)
    t1_offset = padded_len
    mask_positions = mask_positions.at[t1_offset + scale_start : t1_offset + config.tokens_per_frame].set(True)

    def compute_loss(model, batch_tokens):
        """Forward pass — shared between single-GPU and multi-GPU paths."""
        t0 = batch_tokens[:, :config.tokens_per_frame]
        t1 = batch_tokens[:, config.tokens_per_frame:]

        t0_pad = jnp.pad(t0, ((0, 0), (0, padded_len - config.tokens_per_frame)))
        t1_pad = jnp.pad(t1, ((0, 0), (0, padded_len - config.tokens_per_frame)))

        tokens_in = jnp.concatenate([t0_pad, t1_pad], axis=1)

        # Codebook lookup at batch level (outside vmap)
        codebook = jax.lax.stop_gradient(model.embedding.codebook)
        token_vecs = codebook[tokens_in]  # [B, 2*padded_len, D]

        hidden = jax.vmap(
            lambda t, v: model(t, mask_positions, attn_bias, token_vectors=v)
        )(tokens_in, token_vecs)

        h_scale = hidden[:, t1_offset + scale_start : t1_offset + scale_end, :]

        logits = jax.vmap(jax.vmap(model.scale_heads[head_idx]))(h_scale)

        targets = t1[:, scale_start:scale_end] - offset

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(
            log_probs, targets[:, :, None], axis=-1
        ).squeeze(-1)

        loss = -jnp.mean(target_log_probs)

        preds = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(preds == targets)

        return loss, accuracy

    if total_devices > 1:
        # Multi-GPU: pmap over devices, pmean to sync gradients
        @eqx.filter_pmap(in_axes=(0, 0, 0, None), axis_name="batch")
        def step(model, opt_state, batch_tokens, optimizer):
            (loss, accuracy), grads = eqx.filter_value_and_grad(
                lambda m: compute_loss(m, batch_tokens), has_aux=True
            )(model)

            params = eqx.filter(model, eqx.is_inexact_array)

            # Fill None grads with zeros (required for optax tree traversal)
            safe_grads = jax.tree.map(
                lambda g, p: jnp.zeros_like(p) if (g is None and p is not None) else g,
                grads, params, is_leaf=lambda x: x is None,
            )

            safe_grads = jax.lax.pmean(safe_grads, "batch")
            loss = jax.lax.pmean(loss, "batch")
            accuracy = jax.lax.pmean(accuracy, "batch")

            updates, opt_state = optimizer.update(safe_grads, opt_state, params)

            # Mask updates to None where original grads were None
            updates = jax.tree.map(
                lambda g, u: None if g is None else u,
                grads, updates, is_leaf=lambda x: x is None,
            )

            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, accuracy
    else:
        # Single GPU: standard JIT
        @eqx.filter_jit
        def step(model, opt_state, batch_tokens, optimizer):
            (loss, accuracy), grads = eqx.filter_value_and_grad(
                lambda m: compute_loss(m, batch_tokens), has_aux=True
            )(model)

            params = eqx.filter(model, eqx.is_inexact_array)

            # Fill None grads with zeros (required for optax tree traversal)
            # FIX: Only call zeros_like if p is an actual array (not None)
            safe_grads = jax.tree.map(
                lambda g, p: jnp.zeros_like(p) if (g is None and p is not None) else g,
                grads, params, is_leaf=lambda x: x is None,
            )

            updates, opt_state = optimizer.update(safe_grads, opt_state, params)

            # Mask updates to None where original grads were None
            updates = jax.tree.map(
                lambda g, u: None if g is None else u,
                grads, updates, is_leaf=lambda x: x is None,
            )

            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, accuracy

    return step


def save_checkpoint(model, opt_state, epoch, global_step, checkpoint_dir,
                    arch_config=None):
    """Save model, optimizer state, and training state to checkpoint_dir."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = os.path.join(checkpoint_dir, "checkpoint.eqx")
    eqx.tree_serialise_leaves(model_path, model)

    opt_path = os.path.join(checkpoint_dir, "opt_state.eqx")
    eqx.tree_serialise_leaves(opt_path, opt_state)

    state = {"epoch": epoch, "global_step": global_step}
    if arch_config is not None:
        state["arch_config"] = arch_config
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    print(f"Saved checkpoint (epoch {epoch}) to {checkpoint_dir}")


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
    )

    # Populate data-derived fields
    key, model_key = jax.random.split(key)
    model = create_nsp_model_from_tokenized_data(token_data, config, model_key)

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
    }

    # Build attention mask
    attn_bias = build_temporal_mask(config.scales, config.padded_seq_len)
    if is_main:
        print(f"Temporal mask shape: {attn_bias.shape}")

    # Compile train steps (one per trainable scale)
    trainable_indices = config.trainable_scale_indices
    if is_main:
        print(f"Trainable scales: {[config.scales[i] for i in trainable_indices]}")
    train_steps = {}
    for scale_idx in trainable_indices:
        train_steps[scale_idx] = make_train_step(
            config, scale_idx, attn_bias, total_devices)

    # Optimizer — multi_transform with per-head isolation
    n_samples = len(indices) - 1
    steps_per_epoch = n_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    # Per-optimizer default learning rate
    if args.lr is None:
        args.lr = 0.02 if args.optimizer == "muon" else 1e-4

    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, args.warmup_steps, total_steps, args.lr * 0.01
    )

    if args.optimizer == "muon":
        if is_main:
            print(f"Using Muon optimizer (lr={args.lr}, ns_steps={args.muon_ns_steps})")
        base_opt = optax.chain(
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
        base_opt = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.lion(learning_rate=schedule, weight_decay=args.weight_decay)
        )
    elif args.optimizer == "adamw":
        if is_main:
            print(f"Using AdamW optimizer (lr={args.lr})")
        base_opt = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(schedule, weight_decay=args.weight_decay)
        )
    elif args.optimizer == "adafactor":
        if is_main:
            print(f"Using Adafactor optimizer")
        base_opt = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adafactor(learning_rate=schedule)
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Build multi_transform: trunk gets base_opt, each head gets skip_on_zero_grads
    transforms = {"trunk": base_opt}
    for i in range(config.n_trainable_scales):
        transforms[f"head_{i}"] = skip_on_zero_grads(base_opt)

    labels = build_param_labels(model)
    optimizer = optax.multi_transform(transforms, build_param_labels)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        state_path = os.path.join(args.checkpoint_dir, "training_state.json")
        model_path = os.path.join(args.checkpoint_dir, "checkpoint.eqx")
        opt_path = os.path.join(args.checkpoint_dir, "opt_state.eqx")
        if not all(os.path.exists(p) for p in [state_path, model_path, opt_path]):
            raise FileNotFoundError(
                f"Cannot resume: missing checkpoint files in {args.checkpoint_dir}"
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
                    "Cannot resume: model architecture in checkpoint does not "
                    "match current args:\n" + "\n".join(mismatches) +
                    f"\nUse matching args or clear {args.checkpoint_dir} to "
                    "start fresh."
                )
        else:
            if is_main:
                print("Warning: checkpoint has no saved arch config; "
                      "skipping architecture validation")

        start_epoch = training_state["epoch"]
        global_step = training_state["global_step"]
        model = eqx.tree_deserialise_leaves(model_path, model)
        opt_state = eqx.tree_deserialise_leaves(opt_path, opt_state)
        if is_main:
            print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # Replicate model and optimizer state across local devices for pmap
    if multi_device:
        # Explicitly add the num_local dimension for pmap
        model = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_local,) + x.shape), 
            model
        )
        opt_state = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_local,) + x.shape), 
            opt_state
        )

    # Initialize wandb (main process only)
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
    if is_main:
        print(f"\nStarting training: {args.epochs} epochs, "
              f"{steps_per_epoch} steps/epoch, "
              f"{len(trainable_indices)} trainable scales")

    for epoch in range(start_epoch, args.epochs):
        if is_main:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        key, loader_key = jax.random.split(key)

        per_scale_accs = {idx: [] for idx in trainable_indices}
        epoch_losses = []

        dataloader = create_paired_dataloader(
            indices, args.batch_size, seed=int(loader_key[0]),
        )

        for batch_idx, batch in enumerate(dataloader):
            # For multi-device: each process takes its shard, then
            # reshape for pmap across local devices
            if multi_device:
                per_process = args.batch_size // num_processes
                start = jax.process_index() * per_process
                batch = batch[start : start + per_process]
                batch = batch.reshape(num_local, -1, batch.shape[-1])

            # Sample a random trainable scale
            key, sk = jax.random.split(key)
            target_idx = trainable_indices[int(jax.random.randint(sk, (), 0, len(trainable_indices)))]

            model, opt_state, loss, acc = train_steps[target_idx](
                model, opt_state, batch, optimizer
            )

            # pmap returns per-device values; take [0] (all identical after pmean)
            if multi_device:
                loss, acc = loss[0], acc[0]

            per_scale_accs[target_idx].append(float(acc))
            epoch_losses.append(float(loss))

            if is_main and batch_idx % args.log_every == 0:
                h, w = config.scales[target_idx]
                print(f"  Step {batch_idx}: loss={loss:.4f} acc={acc:.4f} "
                      f"(Scale {h}x{w})")
                if WANDB_AVAILABLE:
                    wandb.log({
                        "loss": float(loss),
                        "acc": float(acc),
                        "scale_idx": target_idx,
                        "step": global_step,
                    })

            global_step += 1

        # End of Epoch Summary
        epoch_log = {"epoch": epoch + 1}
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        if is_main:
            print(f"--- Epoch {epoch + 1} Summary (avg loss: {avg_loss:.4f}) ---")
        for idx in trainable_indices:
            h, w = config.scales[idx]
            if len(per_scale_accs[idx]) > 0:
                avg_acc = np.mean(per_scale_accs[idx])
                if is_main:
                    print(f"  Scale {h}x{w}: acc={avg_acc:.4f} ({len(per_scale_accs[idx])} steps)")
                epoch_log[f"acc/scale_{h}x{w}"] = avg_acc
            else:
                if is_main:
                    print(f"  Scale {h}x{w}: no steps sampled")

        epoch_log["epoch/loss"] = avg_loss
        if WANDB_AVAILABLE and is_main:
            wandb.log(epoch_log)

        if is_main and (epoch + 1) % args.save_every == 0:
            # Extract single-device copy for serialization
            save_model = jax.tree.map(lambda x: x[0], model) if multi_device else model
            save_opt = jax.tree.map(lambda x: x[0], opt_state) if multi_device else opt_state
            save_checkpoint(save_model, save_opt, epoch + 1, global_step,
                            args.checkpoint_dir, arch_config=arch_config)

    # Final Save
    if is_main:
        save_model = jax.tree.map(lambda x: x[0], model) if multi_device else model
        save_opt = jax.tree.map(lambda x: x[0], opt_state) if multi_device else opt_state
        save_checkpoint(save_model, save_opt, args.epochs, global_step,
                        args.checkpoint_dir, arch_config=arch_config)

    if WANDB_AVAILABLE and is_main and wandb.run is not None:
        wandb.finish()
    if is_main:
        print("Training complete.")


if __name__ == "__main__":
    main()