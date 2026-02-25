"""Spectral and histogram analysis of VQ-VAE reconstruction quality.

Compares ground truth vorticity fields against their VQ-VAE reconstructions
using TKE spectrum E(k), enstrophy spectrum Z(k), and pixel value histograms.

Usage:
    python -m models.analyze_reconstruction \
        --checkpoint path/to/model.eqx \
        --config_path path/to/config.txt \
        --data_path data.h5 \
        --output_dir ./analysis_output
"""

import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .models import VQVAE2d
from .tokenizer import load_config, load_vqvae_checkpoint


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------


def setup_spectral_analysis(H, W):
    """Set up wavenumber grids and precompute radial bin masks."""
    kx = np.fft.fftfreq(W, d=1.0) * 2 * np.pi
    ky = np.fft.fftfreq(H, d=1.0) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky)
    Ksq = Kx**2 + Ky**2

    k_mag = np.sqrt(Ksq)
    k_max = np.max(k_mag)
    n_bins = min(H // 2, W // 2)
    k_bins = np.linspace(0, k_max, n_bins)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    bin_masks = []
    for i in range(len(k_centers)):
        if i == 0:
            bin_masks.append(k_mag <= k_bins[1])
        else:
            bin_masks.append((k_mag > k_bins[i]) & (k_mag <= k_bins[i + 1]))

    return Kx, Ky, Ksq, k_centers, bin_masks


def radial_average(density_2d, bin_masks):
    """Radially average a 2D spectral density field using precomputed bin masks."""
    spectrum = np.zeros(len(bin_masks))
    for i, mask in enumerate(bin_masks):
        if np.any(mask):
            spectrum[i] = np.mean(density_2d[mask])
    return spectrum


def compute_tke_spectrum(omega, Kx, Ky, Ksq, bin_masks):
    """Compute radially-averaged TKE spectrum E(k) from a vorticity field."""
    omega_hat = np.fft.fft2(omega)
    psi_hat = np.zeros_like(omega_hat, dtype=complex)
    nonzero = Ksq > 0
    psi_hat[nonzero] = omega_hat[nonzero] / Ksq[nonzero]

    u_hat = 1j * Ky * psi_hat
    v_hat = 1j * Kx * psi_hat
    KE_density = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)

    return radial_average(KE_density, bin_masks)


def compute_enstrophy_spectrum(omega, bin_masks):
    """Compute radially-averaged enstrophy spectrum Z(k) from a vorticity field."""
    omega_hat = np.fft.fft2(omega)
    enstrophy_density = 0.5 * np.abs(omega_hat) ** 2
    return radial_average(enstrophy_density, bin_masks)


# ---------------------------------------------------------------------------
# Pixel histogram analysis
# ---------------------------------------------------------------------------


def compute_pixel_histograms(gt_pixels, recon_pixels, n_bins=100):
    """Compute density-normalized histograms for GT and reconstruction pixels.

    Args:
        gt_pixels: Flattened array of ground truth pixel values.
        recon_pixels: Flattened array of reconstruction pixel values.
        n_bins: Number of histogram bins.

    Returns:
        dict with bins, bin_centers, gt_hist, recon_hist.
    """
    bin_min = np.min(gt_pixels)
    bin_max = np.max(gt_pixels)
    margin = (bin_max - bin_min) * 0.01
    bins = np.linspace(bin_min - margin, bin_max + margin, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    gt_hist, _ = np.histogram(gt_pixels, bins=bins, density=True)
    recon_hist, _ = np.histogram(recon_pixels, bins=bins, density=True)

    return {
        "bins": bins,
        "bin_centers": bin_centers,
        "gt_hist": gt_hist,
        "recon_hist": recon_hist,
    }


def compute_pixel_metrics(hist_data):
    """Compute JS divergence and TV distance between GT and reconstruction."""
    eps = 1e-10

    gt = hist_data["gt_hist"] + eps
    recon = hist_data["recon_hist"] + eps

    gt = gt / gt.sum()
    recon = recon / recon.sum()

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    def tv_distance(p, q):
        return float(0.5 * np.sum(np.abs(p - q)))

    return {
        "js_divergence": js_divergence(gt, recon),
        "tv_distance": tv_distance(gt, recon),
    }


def plot_pixel_histograms(hist_data, output_path):
    """Plot overlaid step histograms for GT vs Reconstruction."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = hist_data["bin_centers"]

    ax.step(
        bin_centers, hist_data["gt_hist"], where="mid",
        label="Ground Truth", color="blue", linewidth=2, alpha=0.8,
    )
    ax.step(
        bin_centers, hist_data["recon_hist"], where="mid",
        label="Reconstruction", color="green", linewidth=2, alpha=0.8,
    )

    ax.set_xlabel("Pixel Value (Vorticity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Value Distribution Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved pixel histogram to {output_path}")


# ---------------------------------------------------------------------------
# Spectral plotting
# ---------------------------------------------------------------------------


def plot_spectral_comparison(k_centers, gt_spec, recon_spec, spectrum_type, ylabel, output_path):
    """Plot sample-averaged spectral comparison (loglog)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_k = k_centers > 0

    for spec, label, color, ls in [
        (gt_spec, "Ground Truth", "blue", "-"),
        (recon_spec, "Reconstruction", "green", "--"),
    ]:
        mask = valid_k & (spec > 0)
        if np.any(mask):
            ax.loglog(
                k_centers[mask], spec[mask],
                label=label, color=color, linestyle=ls, alpha=0.8, linewidth=2,
            )

    ax.set_title(f"Sample-Averaged {spectrum_type} Spectrum", fontsize=14)
    ax.set_xlabel("Wavenumber |k|", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {spectrum_type} spectrum to {output_path}")


# ---------------------------------------------------------------------------
# Batched VQ-VAE forward pass
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _vmap_forward(model, batch):
    """JIT-compiled batched forward pass in inference mode.

    Args:
        model: VQVAE2d model.
        batch: [B, 1, H, W] input batch.

    Returns:
        y: [B, 1, H, W] reconstructions.
    """
    model = eqx.nn.inference_mode(model)
    _, _, _, _, _, y = jax.vmap(model)(batch)
    return y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze VQ-VAE reconstruction quality (spectra + histograms)"
    )

    # Data / checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint")
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Path to config.txt file (if not given, model hyperparams must be specified on CLI)",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 data file")
    parser.add_argument("--field", type=str, default="omega", help="HDF5 field name")
    parser.add_argument("--sample_start", type=int, default=0, help="Start sample index")
    parser.add_argument("--sample_stop", type=int, default=None, help="Stop sample index (exclusive)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output plots and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_bins", type=int, default=100, help="Number of histogram bins")

    # Model config overrides (same as tokenizer.py)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--codebook_dim", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--channel_mult", type=str, default=None, help="Comma-separated")
    parser.add_argument("--num_res_blocks", type=int, default=None)
    parser.add_argument("--use_attention", action="store_true", default=None)
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--use_norm", action="store_true", default=None)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=None)
    parser.add_argument(
        "--scales", type=str, default=None,
        help="Comma-separated HxW scales (e.g. 1x1,2x2,4x4,8x8,16x16)",
    )

    return parser.parse_args()


def build_config(args) -> dict:
    """Build config dict from config file and/or CLI overrides."""
    if args.config_path is not None:
        config = load_config(args.config_path)
    else:
        config = {}

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

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    config = build_config(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading VQ-VAE from {args.checkpoint}...")
    key = jax.random.PRNGKey(args.seed)
    model = load_vqvae_checkpoint(args.checkpoint, config, key)

    # Load dataset
    from dataloaders.hdf5_dataset import HDF5Dataset

    dataset = HDF5Dataset(
        data_path=args.data_path,
        fields=[args.field],
        batch_size=args.batch_size,
        shuffle=False,
        prefetch=0,
        sample_start=args.sample_start,
        sample_stop=args.sample_stop,
    )
    n_samples = dataset.n_samples
    H, W = dataset.sample_shape[1], dataset.sample_shape[2]
    print(f"Dataset: {n_samples} samples, field={args.field}, shape=({H}, {W})")

    # Setup spectral analysis
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    n_spectral_bins = len(k_centers)

    # Accumulators
    gt_tke_sum = np.zeros(n_spectral_bins)
    gt_ens_sum = np.zeros(n_spectral_bins)
    recon_tke_sum = np.zeros(n_spectral_bins)
    recon_ens_sum = np.zeros(n_spectral_bins)
    all_gt_pixels = []
    all_recon_pixels = []
    mse_list = []
    sample_count = 0

    # Main loop
    print("Running inference and computing spectra...")
    for batch_idx, batch in enumerate(dataset):
        batch = jnp.asarray(batch)  # [B, 1, H, W]
        recon = _vmap_forward(model, batch)  # [B, 1, H, W]

        batch_np = np.array(batch)
        recon_np = np.array(recon)
        B = batch_np.shape[0]

        # Per-sample MSE
        for i in range(B):
            mse_list.append(float(np.mean((batch_np[i] - recon_np[i]) ** 2)))

        # Per-sample spectra
        for i in range(B):
            gt_field = batch_np[i, 0]    # [H, W]
            rc_field = recon_np[i, 0]    # [H, W]

            gt_tke_sum += compute_tke_spectrum(gt_field, Kx, Ky, Ksq, bin_masks)
            gt_ens_sum += compute_enstrophy_spectrum(gt_field, bin_masks)
            recon_tke_sum += compute_tke_spectrum(rc_field, Kx, Ky, Ksq, bin_masks)
            recon_ens_sum += compute_enstrophy_spectrum(rc_field, bin_masks)

        # Collect pixel values
        all_gt_pixels.append(batch_np[:, 0].ravel())
        all_recon_pixels.append(recon_np[:, 0].ravel())

        sample_count += B
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {sample_count}/{n_samples} samples")

    print(f"  Processed {sample_count}/{n_samples} samples (done)")

    # Average spectra
    gt_tke_avg = gt_tke_sum / sample_count
    gt_ens_avg = gt_ens_sum / sample_count
    recon_tke_avg = recon_tke_sum / sample_count
    recon_ens_avg = recon_ens_sum / sample_count

    # Pixel histograms
    all_gt_pixels = np.concatenate(all_gt_pixels)
    all_recon_pixels = np.concatenate(all_recon_pixels)
    hist_data = compute_pixel_histograms(all_gt_pixels, all_recon_pixels, n_bins=args.n_bins)
    pixel_metrics = compute_pixel_metrics(hist_data)

    # MSE stats
    mse_arr = np.array(mse_list)
    mse_mean = float(mse_arr.mean())
    mse_std = float(mse_arr.std())

    # Plot
    print("Saving plots...")
    plot_spectral_comparison(
        k_centers, gt_tke_avg, recon_tke_avg,
        "TKE", "E(k)", os.path.join(args.output_dir, "tke_spectrum.png"),
    )
    plot_spectral_comparison(
        k_centers, gt_ens_avg, recon_ens_avg,
        "Enstrophy", "Z(k)", os.path.join(args.output_dir, "enstrophy_spectrum.png"),
    )
    plot_pixel_histograms(hist_data, os.path.join(args.output_dir, "pixel_histogram.png"))

    # Save metrics
    metrics = {
        "n_samples": sample_count,
        "mse": {"mean": mse_mean, "std": mse_std},
        "pixel": {
            "js_divergence": pixel_metrics["js_divergence"],
            "tv_distance": pixel_metrics["tv_distance"],
        },
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Summary
    print(f"\nResults ({sample_count} samples):")
    print(f"  MSE: {mse_mean:.6f} +/- {mse_std:.6f}")
    print(f"  Pixel JS divergence: {pixel_metrics['js_divergence']:.6f}")
    print(f"  Pixel TV distance:   {pixel_metrics['tv_distance']:.6f}")


if __name__ == "__main__":
    main()
