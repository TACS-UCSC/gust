import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import optax
import equinox as eqx
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for logging.")

from .models import VQVAE2d
from .trainer import make_step
from dataloaders import HDF5Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train multi-scale VQ-VAE on 2D turbulence data")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .h5 file in gust common format")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--codebook_dim", type=int, default=64, help="Codebook embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=512, help="Number of codebook vectors")
    parser.add_argument("--commitment_weight", type=float, default=0.25, help="Commitment loss weight (beta)")
    parser.add_argument("--decay", type=float, default=0.99, help="EMA decay for codebook")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--normalize", action="store_true", help="Normalize data")
    parser.add_argument("--wandb_project", type=str, default="vqvae-turbulence", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    # Scales
    parser.add_argument("--scales", type=str, default="1x1,2x2,4x4,8x8,16x16",
                        help="Comma-separated (h)x(w) scales (e.g. 1x1,2x2,4x4,8x8,16x16)")
    # Encoder/decoder capacity arguments
    parser.add_argument("--base_channels", type=int, default=128,
                        help="Base channel count (multiplied by channel_mult)")
    parser.add_argument("--channel_mult", type=str, default="1,2,4,4",
                        help="Comma-separated channel multipliers per stage")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of ResBlocks per resolution stage")
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use self-attention at bottleneck")
    parser.add_argument("--no_attention", action="store_true",
                        help="Disable self-attention at bottleneck")
    parser.add_argument("--use_norm", action="store_true", default=True,
                        help="Use GroupNorm in ResBlocks")
    parser.add_argument("--no_norm", action="store_true",
                        help="Disable GroupNorm in ResBlocks")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="Number of attention heads")
    return parser.parse_args()


def plot_reconstruction(inputs, outputs):
    """Plot input vs reconstruction comparison and return figure."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i in range(min(4, len(inputs))):
        # Input
        axes[0, i].imshow(inputs[i, 0], cmap='RdBu_r', vmin=-10, vmax=10)
        axes[0, i].set_title(f"Input {i}")
        axes[0, i].axis('off')

        # Reconstruction
        axes[1, i].imshow(outputs[i, 0], cmap='RdBu_r', vmin=-10, vmax=10)
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis('off')

    plt.tight_layout()
    return fig


def plot_codebook_usage(indices, vocab_size):
    """Plot histogram of codebook usage and return figure."""
    flat_indices = indices.flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.array(flat_indices), bins=vocab_size, range=(0, vocab_size), density=True)
    ax.axhline(y=1/vocab_size, color='r', linestyle='--', label='Uniform')
    ax.set_xlabel("Codebook Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Codebook Usage Distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def save_checkpoint(model, opt_state, epoch, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch}.eqx")
    eqx.tree_serialise_leaves(checkpoint_path, model)
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    args = parse_args()

    # Set random seed
    key = jax.random.PRNGKey(args.seed)

    # Set up device mesh for data parallelism
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("batch",))
    jax.sharding.set_mesh(mesh)
    print(f"Using {num_devices} device(s)")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({args.batch_size}) must be divisible by "
            f"number of devices ({num_devices})"
        )

    data_sharding = NamedSharding(mesh, P("batch", None, None, None))

    # Load data
    print("Loading turbulence data...")
    dataset = HDF5Dataset(
        data_path=args.data_path,
        fields=["omega"],
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        normalize=args.normalize,
        prefetch=2,
        sharding=data_sharding,
        drop_last=True,
    )
    C, H, W = dataset.sample_shape
    print(f"Dataset: {dataset.n_samples} samples, shape ({C}, {H}, {W})")

    # Parse scales as (h, w) tuples
    scales = tuple(
        tuple(int(d) for d in s.split("x")) for s in args.scales.split(",")
    )

    # Parse channel multipliers
    channel_mult = tuple(int(m) for m in args.channel_mult.split(","))

    # Handle attention/norm flags
    use_attention = args.use_attention and not args.no_attention
    use_norm = args.use_norm and not args.no_norm

    # Initialize model
    key, model_key = jax.random.split(key)
    print(f"Using multi-scale VQ-VAE with scales: {scales}")
    print(f"  base_channels={args.base_channels}, channel_mult={channel_mult}")
    print(f"  num_res_blocks={args.num_res_blocks}, attention={use_attention}, norm={use_norm}")
    model = VQVAE2d(
        hidden_dim=args.hidden_dim,
        codebook_dim=args.codebook_dim,
        vocab_size=args.vocab_size,
        scales=scales,
        decay=args.decay,
        base_channels=args.base_channels,
        channel_mult=channel_mult,
        num_res_blocks=args.num_res_blocks,
        use_attention=use_attention,
        use_norm=use_norm,
        attention_heads=args.attention_heads,
        in_channels=1,
        key=model_key,
    )

    # Test forward pass
    test_input = jnp.zeros((1, 1, H, W))
    z_e, z_q, _, indices_list, _, y = jax.vmap(model)(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape (z_e): {z_e.shape}")
    print(f"Indices shapes: {[idx.shape for idx in indices_list]}")
    total_tokens = sum(sh * sw for sh, sw in scales)
    print(f"Total tokens per sample: {total_tokens}")
    print(f"Output shape: {y.shape}")

    # Calculate total training steps for LR schedule
    steps_per_epoch = len(dataset)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(1000, total_steps // 10)

    print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

    # Initialize optimizer with warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Initialize wandb
    if WANDB_AVAILABLE:
        config = {
            "hidden_dim": args.hidden_dim,
            "codebook_dim": args.codebook_dim,
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "commitment_weight": args.commitment_weight,
            "decay": args.decay,
            "epochs": args.epochs,
            "seed": args.seed,
            "normalize": args.normalize,
            "scales": scales,
            "total_tokens": total_tokens,
            # Capacity hyperparameters
            "base_channels": args.base_channels,
            "channel_mult": channel_mult,
            "num_res_blocks": args.num_res_blocks,
            "use_attention": use_attention,
            "use_norm": use_norm,
            "attention_heads": args.attention_heads,
            "num_devices": num_devices,
        }
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_losses = []
        epoch_recon_losses = []
        epoch_commit_losses = []

        for batch_idx, inputs in enumerate(dataset):
            key, step_key = jax.random.split(key)

            model, opt_state, total_loss, recon_loss, commit_loss, indices_out, outputs = make_step(
                model, optimizer, opt_state, inputs, step_key, args.commitment_weight
            )

            epoch_losses.append(float(total_loss))
            epoch_recon_losses.append(float(recon_loss))
            epoch_commit_losses.append(float(commit_loss))

            # Count unique codes used per scale
            per_scale_unique = []
            for idx in indices_out:
                per_scale_unique.append(len(np.unique(np.array(idx).flatten())))
            unique_codes = sum(per_scale_unique)

            # Log to wandb
            if WANDB_AVAILABLE:
                log_dict = {
                    "loss/total": float(total_loss),
                    "loss/reconstruction": float(recon_loss),
                    "loss/commitment": float(commit_loss),
                    "codebook/unique_codes": unique_codes,
                    "step": global_step,
                }
                for si, (sh, sw) in enumerate(scales):
                    log_dict[f"codebook/unique_codes_scale_{sh}x{sw}"] = per_scale_unique[si]
                    log_dict[f"codebook/utilization_scale_{sh}x{sw}"] = per_scale_unique[si] / args.vocab_size
                wandb.log(log_dict)

            if batch_idx % 50 == 0:
                scale_str = " ".join(f"{sh}x{sw}:{n}" for (sh, sw), n in zip(scales, per_scale_unique))
                print(f"  Batch {batch_idx}: Loss={total_loss:.4f}, Recon={recon_loss:.4f}, Commit={commit_loss:.4f}, Codes={unique_codes} [{scale_str}]")

            global_step += 1

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_recon = np.mean(epoch_recon_losses)
        avg_commit = np.mean(epoch_commit_losses)
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Commit={avg_commit:.4f}")

        if WANDB_AVAILABLE:
            epoch_log = {
                "epoch/loss": avg_loss,
                "epoch/reconstruction": avg_recon,
                "epoch/commitment": avg_commit,
                "epoch": epoch + 1,
            }
            recon_fig = plot_reconstruction(np.array(inputs), np.array(outputs))
            all_idx = np.concatenate([np.array(idx).flatten() for idx in indices_out])
            usage_fig = plot_codebook_usage(all_idx, args.vocab_size)
            epoch_log["reconstructions"] = wandb.Image(recon_fig)
            epoch_log["codebook_usage"] = wandb.Image(usage_fig)
            wandb.log(epoch_log)
            plt.close(recon_fig)
            plt.close(usage_fig)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, opt_state, epoch + 1, args.checkpoint_dir)

    # Save final checkpoint
    save_checkpoint(model, opt_state, args.epochs, args.checkpoint_dir)

    # Also save final weights to wandb run directory
    if WANDB_AVAILABLE and wandb.run is not None:
        os.makedirs(wandb.run.dir, exist_ok=True)
        wandb_checkpoint_path = os.path.join(wandb.run.dir, "vqvae_final.eqx")
        eqx.tree_serialise_leaves(wandb_checkpoint_path, model)
        print(f"Saved final weights to {wandb_checkpoint_path}")
        wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
