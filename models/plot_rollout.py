"""Plot rollout analysis: snapshots, spectra, histograms, metrics.

Three-way comparison of:
  1. Raw ground truth field (pre-tokenization, from HDF5)
  2. Ground truth tokens decoded through VQ-VAE ("tokenized GT")
  3. Rollout prediction tokens decoded through VQ-VAE

Outputs:
  - snapshots.png: side-by-side comparison at selected timesteps
  - spectrum_tke.png: TKE spectrum E(k)
  - spectrum_enstrophy.png: enstrophy spectrum Z(k)
  - histogram_pixels.png: pixel value distributions
  - viz_metrics.json: JS divergence and TV distance

Usage:
    # From pre-saved rollout_fields.npz:
    python -m models.plot_rollout \
        --fields_path rollout_1000/rollout_fields.npz \
        --data_path output/output.h5 \
        --start_frame 0 \
        --output_dir rollout_1000

    # From tokens (decodes on the fly):
    python -m models.plot_rollout \
        --tokens_path rollout_1000/rollout_tokens.npz \
        --gt_tokens_path rollout_1000/gt_tokens.npz \
        --data_path output/output.h5 \
        --start_frame 0 \
        --output_dir rollout_1000 \
        --vqvae_checkpoint scales-B-current/vqvae_epoch_90.eqx \
        --vqvae_config scales-B-current/config.txt
"""

import argparse
import json
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze_reconstruction import (
    setup_spectral_analysis,
    compute_tke_spectrum,
    compute_enstrophy_spectrum,
    compute_pixel_histograms,
    compute_pixel_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot rollout analysis: snapshots + spectra + histograms")

    # Input sources
    parser.add_argument("--fields_path", type=str, default=None,
                        help="Path to rollout_fields.npz (pre-decoded)")
    parser.add_argument("--tokens_path", type=str, default=None,
                        help="Path to rollout_tokens.npz (tokenizer-compatible)")
    parser.add_argument("--gt_tokens_path", type=str, default=None,
                        help="Path to gt_tokens.npz (tokenizer-compatible)")

    # Raw HDF5 ground truth
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to HDF5 data file for raw GT fields")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name")
    parser.add_argument("--start_frame", type=int, default=None,
                        help="Start frame index (auto-detected from tokens if available)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Only use the first N frames")

    # VQ-VAE (needed if using tokens_path)
    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    parser.add_argument("--vqvae_config", type=str, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default="./rollout_output")
    parser.add_argument("--snapshots", type=str, default="1,5,10,50,100",
                        help="Snapshot frames: a number for evenly-spaced, "
                             "or comma-separated indices (e.g. '1,5,10,50,100')")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_raw_fields(data_path, field, start_frame, n_frames):
    """Load raw (pre-tokenization) fields from HDF5."""
    import h5py
    with h5py.File(data_path, "r") as f:
        ds = f["fields"][field]
        end = min(start_frame + n_frames, ds.shape[0])
        data = ds[start_frame:end].astype(np.float32)
    return data  # [N, H, W]


def decode_tokens_to_fields(tokens_npz_path, vqvae_checkpoint, vqvae_config_path):
    """Decode a tokenizer-compatible .npz to fields via VQ-VAE."""
    from .tokenizer import (
        load_tokenized_data, load_config, load_vqvae_checkpoint,
        unflatten_to_scales,
    )

    token_data = load_tokenized_data(tokens_npz_path)
    indices = token_data["indices_flat"]
    scales = token_data["scales"]
    unified_to_original = jnp.array(token_data["unified_to_original"])

    config = load_config(vqvae_config_path)
    key = jax.random.PRNGKey(42)
    vqvae = load_vqvae_checkpoint(vqvae_checkpoint, config, key)
    vqvae = eqx.nn.inference_mode(vqvae)

    fields = []
    for i in range(len(indices)):
        flat = jnp.array(indices[i])
        idx_list = unflatten_to_scales(flat, scales)
        orig_idx = [unified_to_original[idx] for idx in idx_list]
        field = vqvae.decode_indices(orig_idx)
        fields.append(np.array(field))
        if (i + 1) % 100 == 0:
            print(f"    Decoded {i + 1}/{len(indices)}")

    return np.stack(fields)  # [N, 1, H, W]


def plot_three_way_spectrum(k_centers, specs, spectrum_type, ylabel, output_path):
    """Plot up to 3 spectra on one loglog plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_k = k_centers > 0

    colors = {"Ground Truth (raw)": "black",
              "GT (tokenized)": "blue",
              "Rollout Prediction": "green"}
    styles = {"Ground Truth (raw)": "-",
              "GT (tokenized)": "--",
              "Rollout Prediction": ":"}

    for label, spec in specs.items():
        mask = valid_k & (spec > 0)
        if np.any(mask):
            ax.loglog(k_centers[mask], spec[mask],
                      label=label,
                      color=colors.get(label, "red"),
                      linestyle=styles.get(label, "-"),
                      linewidth=2, alpha=0.85)

    ax.set_title(f"Sample-Averaged {spectrum_type} Spectrum", fontsize=14)
    ax.set_xlabel("Wavenumber |k|", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {spectrum_type} spectrum to {output_path}")


def plot_three_way_histogram(all_pixels, output_path, n_bins=100):
    """Plot overlaid histograms for up to 3 sources."""
    colors = {"Ground Truth (raw)": "black",
              "GT (tokenized)": "blue",
              "Rollout Prediction": "green"}

    ref_key = "Ground Truth (raw)" if "Ground Truth (raw)" in all_pixels else list(all_pixels.keys())[0]
    ref_vals = all_pixels[ref_key]
    margin = (ref_vals.max() - ref_vals.min()) * 0.01
    bins = np.linspace(ref_vals.min() - margin, ref_vals.max() + margin, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, pixels in all_pixels.items():
        hist, _ = np.histogram(pixels, bins=bins, density=True)
        ax.step(centers, hist, where="mid",
                label=label, color=colors.get(label, "red"),
                linewidth=2, alpha=0.8)

    ax.set_xlabel("Pixel Value (Vorticity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Value Distribution Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved histogram to {output_path}")


def _load_fields(args):
    """Load rollout and GT tokenized fields from either fields_path or tokens_path."""
    if args.fields_path:
        print(f"Loading decoded fields from {args.fields_path}...")
        data = np.load(args.fields_path)
        rollout_fields = data["rollout"]
        gt_tok_fields = data["ground_truth"]
        n_frames = len(rollout_fields)
        print(f"  {n_frames} frames, shape {rollout_fields.shape}")
        return rollout_fields, gt_tok_fields, n_frames

    if args.tokens_path:
        if not (args.vqvae_checkpoint and args.vqvae_config):
            raise ValueError("--vqvae_checkpoint and --vqvae_config required "
                             "when using --tokens_path")
        print(f"Decoding rollout tokens from {args.tokens_path}...")
        rollout_fields = decode_tokens_to_fields(
            args.tokens_path, args.vqvae_checkpoint, args.vqvae_config)
        n_frames = len(rollout_fields)

        gt_tok_fields = None
        if args.gt_tokens_path:
            print(f"Decoding GT tokens from {args.gt_tokens_path}...")
            gt_tok_fields = decode_tokens_to_fields(
                args.gt_tokens_path, args.vqvae_checkpoint, args.vqvae_config)
        return rollout_fields, gt_tok_fields, n_frames

    raise ValueError("Provide either --fields_path or --tokens_path")


def _detect_start_frame(args):
    """Auto-detect start_frame from npz metadata."""
    if args.start_frame is not None:
        return args.start_frame
    for path in [args.tokens_path, args.fields_path]:
        if path is None:
            continue
        try:
            d = np.load(path, allow_pickle=True)
            if "rollout_start_frame" in d:
                return int(d["rollout_start_frame"])
            if "start_frame" in d:
                return int(d["start_frame"])
        except Exception:
            pass
    print("  Warning: could not detect start_frame, defaulting to 0")
    return 0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rollout_fields, gt_tok_fields, n_frames = _load_fields(args)
    start_frame = _detect_start_frame(args)
    print(f"  Start frame: {start_frame}, n_frames: {n_frames}")

    if args.max_frames is not None and args.max_frames < n_frames:
        n_frames = args.max_frames
        rollout_fields = rollout_fields[:n_frames]
        if gt_tok_fields is not None:
            gt_tok_fields = gt_tok_fields[:n_frames]
        print(f"  Truncated to {n_frames} frames (--max_frames)")

    # Load raw GT fields
    gt_raw_fields = None
    if args.data_path:
        print(f"Loading raw GT fields from {args.data_path}...")
        gt_raw_fields = load_raw_fields(
            args.data_path, args.field, start_frame, n_frames)
        print(f"  Loaded {len(gt_raw_fields)} frames, shape {gt_raw_fields.shape}")
        n_frames = min(n_frames, len(gt_raw_fields))
        gt_raw_fields = gt_raw_fields[:n_frames]
        rollout_fields = rollout_fields[:n_frames]
        if gt_tok_fields is not None:
            gt_tok_fields = gt_tok_fields[:n_frames]

    # ---- Snapshot comparison ----
    if args.snapshots and args.snapshots != "0":
        if "," in args.snapshots:
            snap_idx = np.array([int(x) for x in args.snapshots.split(",")], dtype=int)
            snap_idx = snap_idx[snap_idx < n_frames]
        else:
            n_show = min(int(args.snapshots), n_frames)
            snap_idx = np.linspace(0, n_frames - 1, n_show, dtype=int)
        n_show = len(snap_idx)

        def _get2d(arr, i):
            return arr[i, 0] if arr.ndim == 4 else arr[i]

        has_raw = gt_raw_fields is not None
        has_tok = gt_tok_fields is not None
        ref_fields = gt_raw_fields if has_raw else gt_tok_fields

        row_labels = []
        if has_raw:
            row_labels.append("Ground Truth")
        if has_tok:
            row_labels.append("Ground Truth (tokenized)")
        row_labels.append("Prediction")
        n_rows = len(row_labels)

        vmin = float(ref_fields.min())
        vmax = float(ref_fields.max())

        fig, axes = plt.subplots(
            n_rows, n_show,
            figsize=(2.5 * n_show + 1.2, 2.5 * n_rows + 0.5),
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_show == 1:
            axes = axes[:, np.newaxis]

        for col, si in enumerate(snap_idx):
            row = 0
            pred_2d = _get2d(rollout_fields, si)

            if has_raw:
                raw_2d = gt_raw_fields[si] if gt_raw_fields.ndim == 3 else _get2d(gt_raw_fields, si)
                axes[row, col].imshow(raw_2d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axes[row, col].axis("off")
                row += 1

            if has_tok:
                tok_2d = _get2d(gt_tok_fields, si)
                axes[row, col].imshow(tok_2d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
                axes[row, col].axis("off")
                row += 1

            axes[row, col].imshow(pred_2d, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            axes[row, col].axis("off")

        for col, si in enumerate(snap_idx):
            axes[0, col].set_title(f"t = {start_frame + si}", fontsize=11,
                                   fontweight="bold")

        for r, label in enumerate(row_labels):
            axes[r, 0].text(
                -0.05, 0.5, label,
                transform=axes[r, 0].transAxes,
                fontsize=12, fontweight="bold",
                ha="right", va="center", rotation=90,
            )

        fig.subplots_adjust(left=0.08, wspace=0.05, hspace=0.1)
        snap_path = os.path.join(args.output_dir, "snapshots.png")
        fig.savefig(snap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved snapshot comparison to {snap_path}")

    # ---- Spectral analysis ----
    def to_2d(arr):
        if arr.ndim == 4:
            return arr[:, 0]
        return arr

    pred_2d = to_2d(rollout_fields)
    H, W = pred_2d.shape[1], pred_2d.shape[2]
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    n_bins = len(k_centers)

    print("\nComputing spectra...")
    sources = {"Rollout Prediction": pred_2d}
    if gt_tok_fields is not None:
        sources["GT (tokenized)"] = to_2d(gt_tok_fields)
    if gt_raw_fields is not None:
        sources["Ground Truth (raw)"] = gt_raw_fields

    tke_spectra = {}
    ens_spectra = {}
    all_pixels = {}

    for label, fields_2d in sources.items():
        tke_sum = np.zeros(n_bins)
        ens_sum = np.zeros(n_bins)
        for i in range(len(fields_2d)):
            tke_sum += compute_tke_spectrum(fields_2d[i], Kx, Ky, Ksq, bin_masks)
            ens_sum += compute_enstrophy_spectrum(fields_2d[i], bin_masks)
        tke_spectra[label] = tke_sum / len(fields_2d)
        ens_spectra[label] = ens_sum / len(fields_2d)
        all_pixels[label] = fields_2d.ravel()
        print(f"  Computed spectra for {label} ({len(fields_2d)} frames)")

    plot_three_way_spectrum(
        k_centers, tke_spectra, "TKE", "E(k)",
        os.path.join(args.output_dir, "spectrum_tke.png"),
    )
    plot_three_way_spectrum(
        k_centers, ens_spectra, "Enstrophy", "Z(k)",
        os.path.join(args.output_dir, "spectrum_enstrophy.png"),
    )
    plot_three_way_histogram(
        all_pixels,
        os.path.join(args.output_dir, "histogram_pixels.png"),
    )

    # ---- Metrics ----
    metrics = {"start_frame": start_frame, "n_frames": n_frames}

    ref_key = "Ground Truth (raw)" if "Ground Truth (raw)" in all_pixels else "GT (tokenized)"
    if ref_key in all_pixels:
        ref_pixels = all_pixels[ref_key]
        for label, pixels in all_pixels.items():
            if label == ref_key:
                continue
            hist = compute_pixel_histograms(ref_pixels, pixels)
            pm = compute_pixel_metrics(hist)
            tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
            metrics[f"js_{tag}_vs_raw"] = pm["js_divergence"]
            metrics[f"tv_{tag}_vs_raw"] = pm["tv_distance"]
            print(f"  {label} vs {ref_key}: JS={pm['js_divergence']:.6f} "
                  f"TV={pm['tv_distance']:.6f}")

    metrics_path = os.path.join(args.output_dir, "viz_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    print("\nPlotting complete.")


if __name__ == "__main__":
    main()
